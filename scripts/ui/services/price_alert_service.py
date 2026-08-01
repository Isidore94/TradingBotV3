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
    statusChanged = Signal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._checking = False
        self._last_check_at: datetime | None = None
        self._last_check_note = "not checked yet"
        self._last_push_error = ""
        self._writer_refusal_logged = False
        self._timer = QTimer(self)
        self._timer.setInterval(_POLL_INTERVAL_MS)
        self._timer.timeout.connect(self.check_now)
        self._timer.start()

    # ------------------------------------------------------------------
    # Store passthrough for the panel
    # ------------------------------------------------------------------
    def entries(self) -> list[dict[str, Any]]:
        return price_alerts.load_price_alerts()

    def save_entries(self, entries: list[dict[str, Any]]) -> bool:
        return price_alerts.save_price_alerts(entries)

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "checking": self._checking,
            "last_check_at": (
                self._last_check_at.strftime("%H:%M:%S") if self._last_check_at else ""
            ),
            "note": self._last_check_note,
            "push_configured": push_notify.push_configured(),
            "push_error": self._last_push_error,
        }

    def test_push(self) -> dict[str, Any]:
        """Panel button: verify the phone actually buzzes before relying on it."""
        result = push_notify.send_push(
            "TradingBotV3 test",
            "Price alert channel is working. Sleep well.",
            priority="high",
            tags="white_check_mark",
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
        self._checking = True

        def worker() -> None:
            try:
                self._check(symbols)
            except Exception:
                logging.exception("Price alert check failed")
            finally:
                self._checking = False

        threading.Thread(target=worker, name="price-alerts", daemon=True).start()

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
            self._notify(triggers)
        self._last_check_note = (
            f"checked {len(quotes)}/{len(symbols)} symbols"
            + (f"; {len(triggers)} alert(s) fired" if triggers else "")
        )
        self.statusChanged.emit(self.status_snapshot())

    def _notify(self, triggers: list[dict[str, Any]]) -> None:
        import autopilot_core as core

        # Evening mode is the wake-the-trader case: urgent breaks through the
        # iPhone's sleep focus (with critical alerting enabled on the topic).
        priority = "urgent" if core.read_auto_pilot_mode() == "EVENING" else "high"
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
            self.triggered.emit(message)
