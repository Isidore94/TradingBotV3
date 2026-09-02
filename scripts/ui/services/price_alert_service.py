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

    def shutdown(self) -> None:
        self._timer.stop()

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
        for trigger in triggers:
            message = price_alerts.format_trigger_message(trigger)
            tags = "chart_with_upwards_trend" if trigger.get("side") == "above" else "chart_with_downwards_trend"
            result = push_notify.send_push(
                "Price alert", message, priority=priority, tags=tags
            )
            self._last_push_error = str(result.get("error") or "")
            logging.info(
                "PRICE ALERT %s (push %s)",
                message,
                "sent" if result.get("ok") else (self._last_push_error or "not configured"),
            )
            payload = dict(trigger)
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
