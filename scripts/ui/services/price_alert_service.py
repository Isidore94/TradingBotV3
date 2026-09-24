"""Background monitor for the position price-level alerts (Evening mode's
wake-up channel, and an always-on safety net the rest of the time).

Runs whenever the GUI is open - no inbound ports, no extra process: quotes
come from outbound yfinance polls and notifications go out through ntfy
(push_notify). All network work happens on one-shot daemon threads; the
QObject only orchestrates, mirroring AutopilotService.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

import price_alerts
import push_notify

_POLL_INTERVAL_MS = 60_000
# Extended-hours coverage on a Pacific clock: ET premarket opens 04:00 ET =
# 01:00 local; the post-market close 20:00 ET = 17:00 local.
_POLL_START_HOUR = 1
_POLL_END_HOUR = 17

ALWAYS_ON_SETTING = "price_alerts_always_on"

#: How many announced watch ids to remember (see `notify_armed_watch`).
_ANNOUNCED_WATCH_ID_LIMIT = 2_000

#: How long `shutdown()` waits, in TOTAL, for armed-watch deliveries that are
#: still in flight. `push_notify`'s HTTP timeout is 10 s, so a dead endpoint
#: would otherwise stall the desk's close; the threads are daemons, so what is
#: still running when the budget expires can never hold the process.
ARMED_PUSH_SHUTDOWN_WAIT_SECONDS = 2.0

#: How long a read of the Auto mode file is trusted. A flip reaches the
#: service at once through `set_auto_mode`; this only bounds a missed wire.
_AUTO_MODE_CACHE_SECONDS = 5.0

#: DESK sends nothing to the phone (trader, 2026-09-23).
PHONE_QUIET_MODE = "DESK"


class PriceAlertService(QObject):
    """Polls last prices for armed alert entries and fires push notifications.

    Monitoring runs while any side is armed, every weekday 01:00-17:00 local
    (full ET extended hours). With the always-on setting disabled it only
    watches while Auto mode is EVENING. Only the designated shared-store
    writer machine monitors, so two machines never double-push one cross.
    """

    triggered = Signal(str)
    alertTriggered = Signal(dict)
    entriesChanged = Signal()
    statusChanged = Signal(dict)

    def __init__(self, parent=None, *, engine_enabled: bool = True) -> None:
        super().__init__(parent)
        self.engine_enabled = bool(engine_enabled)
        self._checking = False
        self._last_check_at: datetime | None = None
        self._last_check_note = (
            "not checked yet"
            if self.engine_enabled
            else "not the engine machine - monitoring and phone push are off here"
        )
        self._last_push_error = ""
        self._writer_refusal_logged = False
        #: Armed-watch ids already announced to the phone this session.
        self._announced_watch_ids: set[str] = set()
        #: Deliveries the service has handed to a worker and not yet joined.
        #: The lock guards this list AND the announced set, because the Qt
        #: thread adds to both while a worker may still be finishing.
        self._armed_push_threads: list[threading.Thread] = []
        self._armed_push_lock = threading.Lock()
        #: (monotonic stamp, mode) of the last Auto mode reading.
        self._auto_mode_cache: tuple[float, str] | None = None
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self.check_now)
        if self.engine_enabled:
            self._timer.start()

    # ------------------------------------------------------------------
    # Store passthrough for the panel
    # ------------------------------------------------------------------
    def entries(self) -> list[dict[str, Any]]:
        return price_alerts.load_price_alerts()

    def save_entries(self, entries: list[dict[str, Any]]) -> bool:
        if not self.engine_enabled:
            self._last_check_note = "read-only here - price alerts are edited on the main desk"
            self.statusChanged.emit(self.status_snapshot())
            return False
        saved = price_alerts.save_price_alerts(entries)
        if saved:
            self.entriesChanged.emit()
        return saved

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "checking": self._checking,
            "engine_enabled": self.engine_enabled,
            "last_check_at": (
                self._last_check_at.strftime("%H:%M:%S") if self._last_check_at else ""
            ),
            "note": self._last_check_note,
            "push_configured": push_notify.push_configured(),
            "push_error": self._last_push_error,
        }

    #: What the urgent test says on the phone. It has to be self-describing:
    #: the trader reads it half asleep, and the whole point of the test is
    #: that they can tell the difference between "it woke me" and "it did
    #: not" without going back to the desk to check what was sent.
    WAKE_TEST_TITLE = "TradingBotV3 WAKE TEST"
    WAKE_TEST_MESSAGE = (
        "This should have sounded through Sleep Focus. If it did not: add ntfy "
        "to iOS Settings > Focus > Sleep > Allowed Apps, and make sure this "
        "topic is not set to Deliver Quietly. Your price alerts and the SPY "
        "wake alarm push at exactly this priority."
    )

    def test_push(self, *, urgent: bool = False) -> dict[str, Any]:
        """Panel button: verify the phone actually buzzes before relying on it.

        ``urgent`` is the overnight question. Both EVENING-permitted senders -
        the Focus/Research price alerts (``_notify`` below) and the SPY +/-1%
        wake alarm in ``AutopilotService`` - already push at ntfy's maximum
        priority, but nothing on the desk could produce one on demand, so
        "will this actually wake me through Sleep Focus" had never been
        answered. This is a TEST of the channel those two already use, not a
        new sender: nothing schedules it and nothing but the panel button
        calls it, so the phone-push policy is untouched.

        Same fail-quiet contract as the ordinary test either way: the dict
        says what happened, ``send_push`` never raises, and an unconfigured
        topic is reported rather than logged as a delivery.
        """
        if not self.engine_enabled:
            result = {"ok": False, "error": "Phone pushes originate from the main desk only."}
            self._last_push_error = str(result["error"])
            self.statusChanged.emit(self.status_snapshot())
            return result
        if urgent:
            title, message = self.WAKE_TEST_TITLE, self.WAKE_TEST_MESSAGE
            priority, tags = "urgent", "rotating_light"
        else:
            title = "TradingBotV3 test"
            message = "Price alert channel is working. Sleep well."
            priority, tags = "high", "white_check_mark"
        result = push_notify.send_push(
            title, message, priority=priority, tags=tags
        )
        if not result.get("ok") and not result.get("error"):
            result["error"] = "No ntfy topic configured yet."
        self._last_push_error = str(result.get("error") or "")
        self.statusChanged.emit(self.status_snapshot())
        return result

    # ------------------------------------------------------------------
    # Auto mode
    # ------------------------------------------------------------------
    def set_auto_mode(self, mode: str) -> None:
        """The Auto mode just changed; take it now rather than at the next read."""
        text = str(mode or "").strip().upper() or "OFF"
        self._auto_mode_cache = (time.monotonic(), text)

    def on_auto_mode_changed(self, _previous: str, current: str) -> None:
        """Slot for `AutopilotService.autoModeChanged`."""
        self.set_auto_mode(current)

    def _current_auto_mode(self) -> str:
        """OFF/DESK/AWAY/EVENING, re-read from the Auto Pilot state file at most
        every few seconds. An unreadable mode reads OFF, which still pushes."""
        cached = self._auto_mode_cache
        now = time.monotonic()
        if cached is not None and now - cached[0] < _AUTO_MODE_CACHE_SECONDS:
            return cached[1]
        try:
            import autopilot_core as core

            mode = str(core.read_auto_pilot_mode() or "OFF").upper()
        except Exception:
            mode = "OFF"
        self._auto_mode_cache = (now, mode)
        return mode

    def _phone_quiet(self) -> bool:
        return self._current_auto_mode() == PHONE_QUIET_MODE

    def notify_armed_watch(
        self,
        *,
        watch_id: str,
        title: str,
        message: str,
        event_key: str | None = None,
    ) -> dict[str, Any]:
        """Push one TRADER-ARMED watch hit, once, in AWAY, EVENING and OFF.

        DESK sends nothing to the phone (trader, 2026-09-23): the hit is still
        on the feed, only the push is skipped, and the result says
        ``skipped: "DESK"``. The armed
        Research/Focus price alerts are the standing exception - the trader
        asked for that exact condition and is waiting on it - and an armed
        chart watch is the same request made from the chart instead of the
        Focus board, so it rides the SAME sender rather than growing a second
        door to the phone (`docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md`).

        De-duplicated by `event_key`, which DEFAULTS to `watch_id`: a one-shot
        arm is one episode, so a poll that somehow sees the same fire twice
        buzzes once, and every caller written before PCT-1 keeps exactly that
        behaviour by saying nothing.

        **A STANDING arm needs a finer key** (PCT-1 review blocker 1,
        2026-09-15). The Pullback alert does not disarm when it fires: one
        watch legitimately speaks again on the next trigger, the next
        timeframe and the next episode, and keying the refusal on `watch_id`
        meant only the FIRST fire of a multi-day arm ever reached the phone.
        That caller passes
        ``f"{watch_id}:{trigger}:{timeframe}:{bar_dt.isoformat()}"``, so the
        same bar still buzzes once and a new bar buzzes again.

        **Dispatch is synchronous, delivery is not.** The caller is the GUI
        poll (`_poll_h1_bounce_watches` -> `_push_armed_watch`) and
        `push_notify`'s HTTP timeout is 10 s, so the transport may not run
        here: this method makes only the cheap decisions - the engine check
        and the watch-id de-duplication - and hands the send to a one-shot
        daemon worker the service owns, mirroring `check_now`. ``ok`` now
        means "accepted for delivery by the one armed sender", not "a push
        left the desk"; the outcome arrives later on `_last_push_error`, the
        ``ARMED WATCH ...`` log line and `statusChanged`, exactly the way
        `_notify` already reports one.

        The key joins `_announced_watch_ids` BEFORE the dispatch, so a second
        call in the same tick is refused without waiting for the first send.
        Refusals are unchanged: engine disabled ``{"ok": False, "error": ...}``,
        duplicate ``{"ok": False, "deduplicated": True, "watch_id": ...}``.
        """
        watch_id = str(watch_id or "").strip()
        key = str(event_key if event_key is not None else watch_id or "").strip()
        if not self.engine_enabled:
            return {
                "ok": False,
                "error": "Phone pushes originate from the main desk only.",
            }
        if self._phone_quiet():
            logging.info("ARMED WATCH %s (DESK - phone quiet)", message)
            return {"ok": False, "skipped": PHONE_QUIET_MODE, "watch_id": watch_id}
        if key:
            with self._armed_push_lock:
                if key in self._announced_watch_ids:
                    return {"ok": False, "deduplicated": True, "watch_id": watch_id}
                self._announced_watch_ids.add(key)
                # A one-shot arm adds one key per fire; a STANDING arm adds one
                # per (trigger, timeframe, bar), which is still a handful a
                # day. The cap is belt and braces for a desk open for weeks.
                while len(self._announced_watch_ids) > _ANNOUNCED_WATCH_ID_LIMIT:
                    self._announced_watch_ids.pop()
        title_text = str(title or "Armed watch")
        message_text = str(message or "")
        thread = threading.Thread(
            target=self._deliver_armed_watch,
            args=(title_text, message_text),
            name="armed-watch-push",
            daemon=True,
        )
        with self._armed_push_lock:
            # One thread per fire is the `check_now` pattern and the right
            # shape here: an armed watch fires once and then disarms, so a
            # standing consumer thread would idle for days to serve a handful
            # of sends. The list is what lets `shutdown()` know what is still
            # in flight.
            self._armed_push_threads = [
                existing for existing in self._armed_push_threads if existing.is_alive()
            ]
            self._armed_push_threads.append(thread)
        thread.start()
        return {"ok": True, "queued": True, "watch_id": watch_id}

    def _deliver_armed_watch(self, title: str, message: str) -> None:
        """Send one armed-watch push, off the Qt thread, and report it.

        `send_push` never raises, but a transport that does must not lose the
        event or kill the worker silently - so the call is wrapped and the
        failure is logged with its traceback.
        """
        try:
            result = dict(
                push_notify.send_push(
                    title, message, priority="urgent", tags="bell"
                )
                or {}
            )
        except Exception as exc:  # pragma: no cover - send_push does not raise
            self._last_push_error = f"push failed: {exc}"
            logging.exception("ARMED WATCH %s (push failed)", message)
            self.statusChanged.emit(self.status_snapshot())
            return
        self._last_push_error = str(result.get("error") or "")
        logging.info(
            "ARMED WATCH %s (push %s)",
            message,
            "sent" if result.get("ok") else (self._last_push_error or "not configured"),
        )
        self.statusChanged.emit(self.status_snapshot())

    def shutdown(self) -> None:
        self._timer.stop()
        self._join_armed_pushes()

    def _join_armed_pushes(self) -> None:
        """Wait for in-flight armed deliveries, with ONE total budget.

        The desk closing should not lose a push that is a second from landing,
        and it should not wait on a dead endpoint either; the workers are
        daemons, so whatever outlives the budget cannot hold the process.

        A worker's own `statusChanged` is QUEUED to the GUI thread (Qt's
        auto-connection from a non-GUI thread), so an outcome that lands while
        the desk is closing would never be delivered - the event loop it is
        waiting on is the one that just stopped. Having WAITED for that
        outcome, the joining thread reports it once itself; a status refresh
        is idempotent, so a later queued copy costs nothing.
        """
        with self._armed_push_lock:
            pending = [
                thread for thread in self._armed_push_threads if thread.is_alive()
            ]
        deadline = time.monotonic() + ARMED_PUSH_SHUTDOWN_WAIT_SECONDS
        for thread in pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            thread.join(remaining)
        with self._armed_push_lock:
            self._armed_push_threads = [
                thread for thread in self._armed_push_threads if thread.is_alive()
            ]
        if pending:
            self.statusChanged.emit(self.status_snapshot())

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------
    def _monitoring_wanted(self, now: datetime) -> tuple[bool, str]:
        if now.weekday() >= 5:
            return False, "weekend - markets closed"
        if not (_POLL_START_HOUR <= now.hour < _POLL_END_HOUR):
            return False, "outside extended trading hours"
        try:
            from project_paths import get_local_setting

            always_on = bool(get_local_setting(ALWAYS_ON_SETTING, True))
        except Exception:
            always_on = True
        if not always_on:
            import autopilot_core as core

            if core.read_auto_pilot_mode() != "EVENING":
                return False, "always-on disabled and Auto mode is not EVENING"
        return True, ""

    def check_now(self) -> None:
        if not self.engine_enabled:
            self._last_check_note = "not the engine machine - monitoring and phone push are off here"
            self.statusChanged.emit(self.status_snapshot())
            return
        if self._checking:
            return
        now = datetime.now()
        wanted, why_not = self._monitoring_wanted(now)
        if not wanted:
            self._last_check_note = why_not
            return
        entries = price_alerts.load_price_alerts()
        symbols = price_alerts.armed_symbols(entries)
        if not symbols:
            self._last_check_note = "no armed alert levels"
            return
        try:
            import autopilot_core as core
            from project_paths import PRICE_ALERTS_FILE

            refusal = core.shared_write_refusal(PRICE_ALERTS_FILE)
        except Exception:
            refusal = ""
        if refusal:
            # A second machine watching the same store would double-push every
            # cross and race the disarm write; layer 1 already knows which
            # machine owns shared state - defer to it.
            self._last_check_note = "not the designated writer machine - monitoring is off here"
            if not self._writer_refusal_logged:
                self._writer_refusal_logged = True
                logging.info("Price alerts idle on this machine: %s", refusal)
            return
        # A2 (2026-09-01): an alert that has sat armed for 10 trading days
        # without firing is disarmed, so the Armed board keeps meaning "what I
        # am waiting on". It runs HERE - after the writer check, on the timer
        # this service already owns - so no second component writes the store
        # and no new timer appears. Nothing is deleted; see `price_alerts`.
        surviving = self._expire_stale(entries)
        if surviving is not None:
            # An empty list is a real answer - every armed level just expired -
            # so `or` would be wrong here and would poll the old symbol set.
            symbols = surviving
        if not symbols:
            self._last_check_note = "no armed alert levels"
            return
        self._checking = True

        def worker() -> None:
            try:
                self._check(symbols)
            except Exception:
                logging.exception("Price alert check failed")
            finally:
                self._checking = False

        threading.Thread(target=worker, name="price-alerts", daemon=True).start()

    def _expire_stale(self, entries: list[dict[str, Any]]) -> list[str] | None:
        """Disarm what has run out of sessions. Returns the surviving armed
        symbols, or ``None`` when nothing changed.

        Never raises into the poll: an expiry pass that fails costs the
        cleanup, never the alerting behind it.
        """
        try:
            updated, rows = price_alerts.expire_stale_alerts(entries)
        except Exception:
            logging.debug("Price alert expiry pass failed", exc_info=True)
            return None
        if not rows:
            return None
        try:
            import armed_alert_expiry

            armed_alert_expiry.record_expiries(rows)
        except Exception:
            logging.debug("Price alert expiry rows were not written", exc_info=True)
        price_alerts.save_price_alerts(updated)
        self.entriesChanged.emit()
        names = ", ".join(sorted({str(row.get("symbol") or "") for row in rows}))
        logging.info("Price alerts disarmed after their session window: %s", names)
        return price_alerts.armed_symbols(updated)

    def _check(self, symbols: list[str]) -> None:
        quotes = price_alerts.fetch_last_quotes(symbols, log=logging.info)
        self._last_check_at = datetime.now()
        if not quotes:
            self._last_check_note = f"no quotes returned for {len(symbols)} symbols"
            self.statusChanged.emit(self.status_snapshot())
            return
        # Re-read at evaluation time so an edit made while the fetch was in
        # flight (say, the trader re-arming a level) is never overwritten.
        entries = price_alerts.load_price_alerts()
        updated, triggers = price_alerts.evaluate_price_alerts(entries, quotes)
        if triggers:
            price_alerts.save_price_alerts(updated)
            price_alerts.append_trigger_log(triggers)
            self.entriesChanged.emit()
            self._notify(triggers)
        self._last_check_note = (
            f"checked {len(quotes)}/{len(symbols)} symbols"
            + (f"; {len(triggers)} alert(s) fired" if triggers else "")
        )
        self.statusChanged.emit(self.status_snapshot())

    def _notify(self, triggers: list[dict[str, Any]]) -> None:
        # Trader decision: every price crossing is urgent, including rows made
        # from the advanced Research view. The store has no origin marker.
        priority = "urgent"
        # DESK keeps the phone quiet; the desk still shows every crossing.
        quiet = self._phone_quiet()
        for trigger in triggers:
            message = price_alerts.format_trigger_message(trigger)
            tags = "chart_with_upwards_trend" if trigger.get("side") == "above" else "chart_with_downwards_trend"
            payload = dict(trigger)
            if quiet:
                logging.info("PRICE ALERT %s (DESK - phone quiet)", message)
                payload.update(
                    {
                        "message": message,
                        "priority": priority,
                        "push_ok": None,
                        "push_error": "",
                        "push_skipped": PHONE_QUIET_MODE,
                    }
                )
                self.triggered.emit(message)
                self.alertTriggered.emit(payload)
                continue
            result = push_notify.send_push(
                "Price alert", message, priority=priority, tags=tags
            )
            self._last_push_error = str(result.get("error") or "")
            logging.info(
                "PRICE ALERT %s (push %s)",
                message,
                "sent" if result.get("ok") else (self._last_push_error or "not configured"),
            )
            payload.update(
                {
                    "message": message,
                    "priority": priority,
                    "push_ok": bool(result.get("ok")),
                    "push_error": self._last_push_error,
                }
            )
            # Push deliberately happens before either local presentation or
            # A broken display path cannot suppress the phone.
            self.triggered.emit(message)
            self.alertTriggered.emit(payload)
