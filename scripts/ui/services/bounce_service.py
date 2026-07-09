from __future__ import annotations

import threading
from typing import Any, Callable

from PySide6.QtCore import QObject, QTimer, Signal, Slot

from ui.models.bounce import BounceAlert


def load_bounce_config() -> dict[str, Any]:
    from bounce_bot import BOUNCE_TYPE_DEFAULTS, BOUNCE_TYPE_LABELS, MARKET_ENVIRONMENTS, RRS_TIMEFRAMES

    return {
        "bounce_type_defaults": dict(BOUNCE_TYPE_DEFAULTS),
        "bounce_type_labels": dict(BOUNCE_TYPE_LABELS),
        "market_environments": dict(MARKET_ENVIRONMENTS),
        "rrs_timeframes": dict(RRS_TIMEFRAMES),
    }


class BounceService(QObject):
    alertReceived = Signal(object)
    rrsStatusChanged = Signal(str)
    rrsSnapshotChanged = Signal(object)
    statusChanged = Signal(str)
    connectionChanged = Signal(str)
    activeBouncesChanged = Signal(int)
    scanningChanged = Signal(bool)
    autoRegimeChanged = Signal(object)  # reading dict from get_auto_regime_reading(), or {}
    entryAssistChanged = Signal(object)  # state dict from entry_assist_state(), or {}
    started = Signal()
    stopped = Signal()
    failed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        config = load_bounce_config()
        self.bounce_type_settings: dict[str, bool] = dict(config["bounce_type_defaults"])
        self.rrs_threshold = 2.0
        self.rrs_timeframe_key = "5m"
        self.market_environment = "bullish_strong"
        self.scanning_enabled = False
        self.include_approaching = False

        self._bot = None
        self._lock = threading.Lock()
        self._starting = False

        self._health_timer = QTimer(self)
        self._health_timer.setInterval(3000)
        self._health_timer.timeout.connect(self.refresh_health)
        self.started.connect(self._start_health_timer)

        # Always-on auto-regime readout: what auto tracking thinks right now
        # (even under a manual override), refreshed from cached SPY bars.
        self._regime_timer = QTimer(self)
        self._regime_timer.setInterval(30_000)
        self._regime_timer.timeout.connect(self.refresh_auto_regime)
        self.started.connect(self._start_regime_timer)

    @property
    def running(self) -> bool:
        with self._lock:
            return self._bot is not None or self._starting

    def start(self) -> None:
        if self.running:
            return
        self._starting = True
        self.connectionChanged.emit("IB: connecting")
        self.statusChanged.emit("connecting")
        threading.Thread(target=self._start_worker, name="qt-bouncebot-start", daemon=True).start()

    def restart(self) -> None:
        self.stop()
        self.start()

    def stop(self) -> None:
        with self._lock:
            bot = self._bot
            self._bot = None
            self._starting = False
        if bot is not None:
            try:
                bot.disconnect()
            except Exception:
                pass
        self._health_timer.stop()
        self._regime_timer.stop()
        self.autoRegimeChanged.emit({})
        self.connectionChanged.emit("IB: disconnected")
        self.activeBouncesChanged.emit(0)
        self.statusChanged.emit("stopped")
        self.stopped.emit()

    def start_scanning(self) -> None:
        self.set_scanning_enabled(True)

    def stop_scanning(self) -> None:
        self.set_scanning_enabled(False)

    def set_scanning_enabled(self, enabled: bool) -> None:
        self.scanning_enabled = bool(enabled)
        self._with_bot(lambda bot: bot.set_scanning_enabled(self.scanning_enabled))
        self.scanningChanged.emit(self.scanning_enabled)
        self.statusChanged.emit("scanning enabled" if self.scanning_enabled else "scanning paused")

    def set_rrs_threshold(self, value: float) -> None:
        self.rrs_threshold = float(value)
        self._with_bot(lambda bot: bot.set_rrs_threshold(self.rrs_threshold))

    def set_rrs_timeframe(self, key: str) -> None:
        config = load_bounce_config()
        if key not in config["rrs_timeframes"]:
            return
        self.rrs_timeframe_key = key
        self._with_bot(lambda bot: bot.set_rrs_timeframe(key))

    def set_market_environment(self, env_key: str) -> None:
        config = load_bounce_config()
        if env_key not in config["market_environments"]:
            return
        self.market_environment = env_key
        self._with_bot(lambda bot: bot.set_market_environment(env_key))

    def clear_market_environment_override(self) -> None:
        """Return regime control to the bot's SPY-based auto tracking."""
        self._with_bot(lambda bot: bot.clear_market_environment_override())
        self.statusChanged.emit("Market regime back on auto (SPY vs yesterday's close).")

    def set_bounce_type_enabled(self, bounce_type: str, enabled: bool) -> None:
        if bounce_type not in self.bounce_type_settings:
            return
        self.bounce_type_settings[bounce_type] = bool(enabled)
        self._with_bot(lambda bot: bot.set_bounce_type_enabled(bounce_type, bool(enabled)))

    @Slot()
    def _start_health_timer(self) -> None:
        self._health_timer.start()
        self.refresh_health()

    @Slot()
    def _start_regime_timer(self) -> None:
        self._regime_timer.start()
        self.refresh_auto_regime()

    @Slot()
    def refresh_auto_regime(self) -> None:
        """Emit the bot's read-only auto-regime reading + entry-assist state."""
        bot = self._current_bot()
        reading = None
        assist = None
        if bot is not None:
            try:
                reading = bot.get_auto_regime_reading()
            except Exception:
                reading = None
            try:
                assist = bot.entry_assist_state()
            except Exception:
                assist = None
        self.autoRegimeChanged.emit(reading or {})
        self.entryAssistChanged.emit(assist or {})

    def entry_assist(self) -> dict | None:
        """The strip button: regime-tailored window toggle / movers output."""
        result = self._with_bot(lambda bot: bot.entry_assist_action())
        if isinstance(result, dict) and result.get("note"):
            self.statusChanged.emit(f"Entry assist: {result['note']}")
        self.refresh_auto_regime()
        return result

    @Slot()
    def refresh_health(self) -> None:
        bot = self._current_bot()
        if bot is None:
            self.connectionChanged.emit("IB: disconnected")
            self.activeBouncesChanged.emit(0)
            return

        connected = bool(getattr(bot, "connection_status", False))
        self.connectionChanged.emit("IB: connected" if connected else "IB: retrying")
        try:
            count = len(bot.find_active_master_avwap_bounces())
        except Exception:
            count = 0
        self.activeBouncesChanged.emit(count)

    def _start_worker(self) -> None:
        try:
            from bounce_bot import run_bot_with_gui

            bot = run_bot_with_gui(self._make_callback(), start_scanning_enabled=self.scanning_enabled)
            self._apply_saved_state(bot)
            self._sync_state_from_bot(bot)
            with self._lock:
                self._bot = bot
                self._starting = False
            self.connectionChanged.emit("IB: connected" if bool(getattr(bot, "connection_status", False)) else "IB: retrying")
            self.statusChanged.emit("connected")
            self.started.emit()
        except Exception as exc:
            with self._lock:
                self._bot = None
                self._starting = False
            message = str(exc)
            self.connectionChanged.emit("IB: disconnected")
            self.statusChanged.emit(f"start failed: {message}")
            self.failed.emit(message)

    def _make_callback(self) -> Callable[[Any, str], None]:
        def gui_callback(message: Any, tag: str) -> None:
            tag_text = str(tag or "")
            if tag_text == "rrs_status":
                self.rrsStatusChanged.emit(str(message))
                return
            if tag_text == "rrs_snapshot":
                self.rrsSnapshotChanged.emit(message)
                return
            if tag_text == "blue" and "removed from" in str(message):
                return
            if not self.include_approaching and (
                tag_text == "approaching" or tag_text.startswith("approaching_")
            ):
                return
            self.alertReceived.emit(BounceAlert.from_callback(message, tag_text))

        return gui_callback

    def _apply_saved_state(self, bot) -> None:
        bot.set_rrs_threshold(self.rrs_threshold)
        bot.set_rrs_timeframe(self.rrs_timeframe_key)
        bot.set_market_environment(self.market_environment)
        bot.set_scanning_enabled(self.scanning_enabled)
        for bounce_key, enabled in self.bounce_type_settings.items():
            bot.set_bounce_type_enabled(bounce_key, enabled)

    def _sync_state_from_bot(self, bot) -> None:
        self.rrs_threshold = float(getattr(bot, "rrs_threshold", self.rrs_threshold))
        self.rrs_timeframe_key = str(getattr(bot, "rrs_timeframe_key", self.rrs_timeframe_key))
        try:
            self.market_environment = str(bot.get_market_environment())
        except Exception:
            pass
        try:
            self.scanning_enabled = bool(bot.is_scanning_enabled())
        except Exception:
            pass
        for bounce_key in self.bounce_type_settings:
            try:
                self.bounce_type_settings[bounce_key] = bool(bot.is_bounce_type_enabled(bounce_key))
            except Exception:
                pass
        self.scanningChanged.emit(self.scanning_enabled)

    def current_bot(self):
        """The live BounceBot instance, or None (used by Auto Pilot)."""
        return self._current_bot()

    def _current_bot(self):
        with self._lock:
            return self._bot

    def _with_bot(self, callback: Callable[[Any], Any]) -> Any:
        bot = self._current_bot()
        if bot is None:
            return None
        try:
            return callback(bot)
        except Exception as exc:
            self.statusChanged.emit(f"command failed: {exc}")
            return None
