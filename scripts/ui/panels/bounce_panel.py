from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
)

from ui.services.bounce_service import BounceService, load_bounce_config

def entry_assist_button_state(state) -> tuple[str, str, bool]:
    """(label, tooltip, enabled) for the regime-tailored entry-assist button.

    Strong regimes toggle a pullback/bounce window; weak regimes emit the
    strongest/weakest trailing-30m list; neutral/chop emits both.
    """
    state = state if isinstance(state, dict) else {}
    env = str(state.get("env_key") or "")
    if not state:
        return ("Entry Assist", "Waiting for the bot to connect.", False)
    if state.get("window_active"):
        started = state.get("window_started") or "?"
        if env == "bearish_strong":
            return (
                f"Bounce over → weakest (since {started})",
                "SPY's bounce is done: click to rank which shorts stayed weakest through it. "
                "Auto mode also closes auto-opened windows when the tape resumes.",
                True,
            )
        return (
            f"Pullback over → strongest (since {started})",
            "SPY's pullback is done: click to rank which longs held up best through it. "
            "Auto mode also closes auto-opened windows when the tape resumes.",
            True,
        )
    if env == "bullish_strong":
        return (
            "⏱ Pullback started",
            "SPY starting to pull back? Click to open a window tracking which longs hold up / stay RS. "
            "Click again when the pullback ends to get the ranked list. Auto mode does both on its own "
            "via SPY pause detection.",
            True,
        )
    if env == "bearish_strong":
        return (
            "⏱ Bounce started",
            "SPY starting to bounce? Click to open a window tracking which shorts stay weak / RW. "
            "Click again when the bounce ends to get the ranked list. Auto mode does both on its own.",
            True,
        )
    if env == "bullish_weak":
        return ("Strongest 30m", "Emit the strongest longs vs SPY over the last 30 minutes. Auto mode emits this every 30m.", True)
    if env == "bearish_weak":
        return ("Weakest 30m", "Emit the weakest shorts vs SPY over the last 30 minutes. Auto mode emits this every 30m.", True)
    return (
        "Movers 30m",
        "Neutral/chop: emit BOTH the strongest longs and weakest shorts of the last 30 minutes. "
        "Auto mode emits both every 30m.",
        True,
    )


def format_auto_regime_reading(reading) -> tuple[str, str]:
    """(chip_text, tooltip) for the always-on auto-regime readout.

    The chip shows what auto tracking thinks right now (and the manual
    override next to it when one is active); the tooltip carries the
    measurements behind the read and how close each other regime is.
    """
    reading = reading if isinstance(reading, dict) else {}
    if not reading:
        return (
            "Auto regime: n/a",
            "No SPY session read yet - the auto regime appears once the bot has SPY 5m bars "
            "(kept fresh even while scanning is paused).",
        )
    label = str(reading.get("label") or reading.get("env_key") or "?")
    if reading.get("override_active"):
        chip = f"Manual: {reading.get('active_label')} (auto sees {label})"
    else:
        chip = f"Auto: {label}"

    day_pct = float(reading.get("day_pct") or 0.0)
    lines = [
        f"Auto regime read @ {reading.get('bar_time', '?')} (SPY 5m): {label}",
        f"SPY {float(reading.get('last_close') or 0.0):,.2f} · {day_pct:+.2f}% vs yday "
        f"{float(reading.get('prev_close') or 0.0):,.2f}",
    ]
    vwap = reading.get("vwap")
    stdev = float(reading.get("stdev") or 0.0)
    needed = float(reading.get("band_fraction_needed") or 0.6)
    if isinstance(vwap, (int, float)):
        last_close = float(reading.get("last_close") or 0.0)
        sigma_text = f" ({(last_close - vwap) / stdev:+.1f} sigma)" if stdev > 0 else ""
        above = float(reading.get("above_band_frac") or 0.0)
        below = float(reading.get("below_band_frac") or 0.0)
        lines.append(f"Session VWAP {vwap:,.2f} · 1 sigma {stdev:,.2f} · SPY vs VWAP{sigma_text}")
        lines.append("Possibilities right now:")
        lines.append(f"- Bullish Strong: {above:.0%} of bars >= VWAP+1 sigma (needs >= {needed:.0%})")
        lines.append(f"- Bearish Strong: {below:.0%} of bars <= VWAP-1 sigma (needs >= {needed:.0%})")
        lines.append(
            f"- Bullish Weak: SPY above VWAP and green on the day - "
            f"{'YES' if last_close > float(vwap) and day_pct > 0 else 'no'}"
        )
        lines.append(
            f"- Bearish Weak: SPY below VWAP and red on the day - "
            f"{'YES' if last_close < float(vwap) and day_pct < 0 else 'no'}"
        )
    else:
        strong_pct = float(reading.get("strong_abs_pct") or 0.5)
        lines.append(
            f"Session too young for the VWAP read - day% rule applies "
            f"(+/-{strong_pct:.1f}% on the day = strong)."
        )
    if reading.get("override_active"):
        lines.append(
            f"Manual override active ({reading.get('active_label')}); auto keeps measuring - "
            "click Auto to hand control back."
        )
    return chip, "\n".join(lines)


# Kept for the Settings page, which renders the bounce-type toggle grid.
BOUNCE_TOGGLE_ORDER = [
    "10_candle",
    "vwap",
    "dynamic_vwap",
    "eod_vwap",
    "vwap_eod_confluence",
    "impulse_retest_vwap_eod",
    "ema_8",
    "ema_15",
    "ema_21",
    "vwap_upper_band",
    "vwap_lower_band",
    "dynamic_vwap_upper_band",
    "dynamic_vwap_lower_band",
    "eod_vwap_upper_band",
    "eod_vwap_lower_band",
    "prev_day_high",
    "prev_day_low",
]


class BouncePanel(QFrame):
    """One-line BounceBot status strip for the Trading Desk.

    Owns the live service (auto-connects on startup) and shows only what
    matters while trading, in a single row: health chips, the market-regime
    chip with its manual override, RRS status, and Start/Stop scanning.
    Connection management, RRS tuning, and bounce-type toggles live on the
    Settings page; alert feeds live in the Alert Center.
    """

    statusChanged = Signal(str)

    def __init__(self, focus_service=None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.focus_service = focus_service
        self.service = BounceService(self)
        self.config = load_bounce_config()
        self._syncing_environment = False

        self.connection_label = QLabel("IB: disconnected")
        self.connection_label.setObjectName("MutedLabel")
        self.status_label = QLabel("stopped")
        self.status_label.setObjectName("MutedLabel")
        self.active_label = QLabel("active bounces: 0")
        self.active_label.setObjectName("MutedLabel")
        self.rrs_status_label = QLabel("RRS ready")
        self.rrs_status_label.setObjectName("MutedLabel")

        self.start_scanning_button = QPushButton("Start Scanning")
        self.start_scanning_button.setObjectName("PrimaryButton")
        self.stop_scanning_button = QPushButton("Stop Scanning")
        self.entry_assist_button = QPushButton("Entry Assist")
        self.entry_assist_button.setEnabled(False)

        self.regime_label = QLabel("Auto regime: n/a")
        self.regime_label.setObjectName("MutedLabel")
        self.environment_input = QComboBox()
        for key, item in self.config["market_environments"].items():
            self.environment_input.addItem(str(item.get("label", key)), key)
        self.environment_input.setCurrentIndex(
            max(0, self.environment_input.findData(self.service.market_environment))
        )
        self.environment_auto_button = QPushButton("Auto")
        self.environment_auto_button.setToolTip(
            "Return regime control to the bot (SPY green/red vs yesterday's close)."
        )

        self._build_layout()
        self._wire_service()
        self._sync_scanning_buttons(False)
        QTimer.singleShot(0, self.start)

    def _build_layout(self) -> None:
        title = QLabel("BounceBot")
        title.setStyleSheet("font-weight: 700;")
        title.setToolTip(
            "Auto-connects on launch. Connection, RRS tuning, and bounce types live in Settings; "
            "alerts land in the Alert Center."
        )
        # The RRS status shares the row; let it shrink/clip instead of forcing width.
        self.rrs_status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

        strip = QHBoxLayout(self)
        strip.setContentsMargins(12, 6, 12, 6)
        strip.setSpacing(12)
        strip.addWidget(title)
        strip.addWidget(self.connection_label)
        strip.addWidget(self.status_label)
        strip.addWidget(self.active_label)
        strip.addWidget(self.regime_label)
        strip.addWidget(self.environment_input)
        strip.addWidget(self.environment_auto_button)
        strip.addWidget(self.rrs_status_label, 1)
        strip.addWidget(self.entry_assist_button)
        strip.addWidget(self.start_scanning_button)
        strip.addWidget(self.stop_scanning_button)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def _wire_service(self) -> None:
        self.start_scanning_button.clicked.connect(self.service.start_scanning)
        self.stop_scanning_button.clicked.connect(self.service.stop_scanning)
        self.environment_input.currentIndexChanged.connect(self._on_environment_changed)
        self.environment_auto_button.clicked.connect(self._on_environment_auto)

        self.service.rrsStatusChanged.connect(self._set_rrs_status)
        self.service.statusChanged.connect(self._set_status)
        self.service.connectionChanged.connect(self._set_connection)
        self.service.autoRegimeChanged.connect(self._set_auto_regime)
        self.service.entryAssistChanged.connect(self._set_entry_assist)
        self.entry_assist_button.clicked.connect(self.service.entry_assist)
        self.service.activeBouncesChanged.connect(
            lambda count: self.active_label.setText(f"active bounces: {count}")
        )
        self.service.scanningChanged.connect(self._sync_scanning_buttons)
        self.service.failed.connect(
            lambda message: QMessageBox.critical(self, "BounceBot Start Failed", message)
        )

    def start(self) -> None:
        self.service.start()

    def stop(self) -> None:
        self.service.stop()

    def start_scanning(self) -> None:
        self.service.start_scanning()

    def stop_scanning(self) -> None:
        self.service.stop_scanning()

    def on_close(self) -> None:
        self.service.stop()

    def _on_environment_changed(self) -> None:
        if self._syncing_environment:
            return
        key = self.environment_input.currentData()
        if key:
            self.service.set_market_environment(str(key))
            self.service.refresh_auto_regime()  # re-render the chip with the override

    def _on_environment_auto(self) -> None:
        self.service.clear_market_environment_override()
        self.service.refresh_auto_regime()

    def _set_auto_regime(self, reading) -> None:
        chip, tooltip = format_auto_regime_reading(reading)
        self.regime_label.setText(chip)
        self.regime_label.setToolTip(tooltip)

    def _set_entry_assist(self, state) -> None:
        label, tooltip, enabled = entry_assist_button_state(state)
        self.entry_assist_button.setText(label)
        self.entry_assist_button.setToolTip(tooltip)
        self.entry_assist_button.setEnabled(enabled)

    def _set_rrs_status(self, message: str) -> None:
        self.rrs_status_label.setText(message)
        self.rrs_status_label.setToolTip(message)  # row label may clip; hover shows all
        self.statusChanged.emit(message)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.statusChanged.emit(f"BounceBot: {message}")
        # Mirror the bot's auto-regime changes into the dropdown + readout.
        text = str(message or "")
        if text.startswith("Auto market regime"):
            self.service.refresh_auto_regime()
            for key, item in self.config["market_environments"].items():
                if str(item.get("label", "")) in text or key in text:
                    self._syncing_environment = True
                    try:
                        self.environment_input.setCurrentIndex(
                            max(0, self.environment_input.findData(key))
                        )
                    finally:
                        self._syncing_environment = False
                    break

    def _set_connection(self, message: str) -> None:
        self.connection_label.setText(message)

    def _sync_scanning_buttons(self, scanning_enabled: bool) -> None:
        self.start_scanning_button.setEnabled(not scanning_enabled)
        self.stop_scanning_button.setEnabled(scanning_enabled)
