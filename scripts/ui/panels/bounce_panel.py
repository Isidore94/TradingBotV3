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
    QVBoxLayout,
)

from technical_integrity import format_technical_integrity_snapshot
from ui.services.bounce_service import BounceService, load_bounce_config

def entry_assist_button_specs(state) -> list[dict]:
    """One spec per entry-assist action.

    Pullback/bounce windows are automatic in normal operation and therefore
    marked as advanced manual diagnostics. The useful on-demand views remain
    strongest, weakest, and both-side movers over the trailing 30 minutes.
    """
    state = state if isinstance(state, dict) else {}
    connected = bool(state)
    env = str(state.get("env_key") or "")
    window_active = bool(state.get("window_active"))
    window_sides = {str(side or "").lower() for side in state.get("window_sides") or []}
    started = state.get("window_started") or "?"
    long_window = window_active and "long" in window_sides
    short_window = window_active and "short" in window_sides

    if long_window:
        pullback_label = f"Pullback over → strongest (since {started})"
        pullback_tip = (
            "SPY's pullback is done: click to rank which longs held up best / stayed RS through it. "
            "The ranked list lands in the Alert Center. Auto mode also closes auto-opened windows "
            "when the tape resumes."
        )
    else:
        pullback_label = "⏱ Pullback started"
        pullback_tip = (
            "SPY starting to pull back? Click to open a window tracking which longs hold up / stay RS. "
            "Click again when the pullback ends to get the ranked list in the Alert Center. "
            "Auto mode does both on its own via SPY pause detection."
        )
    if short_window:
        bounce_label = f"Bounce over → weakest (since {started})"
        bounce_tip = (
            "SPY's bounce is done: click to rank which shorts stayed weakest through it. "
            "The ranked list lands in the Alert Center. Auto mode also closes auto-opened windows "
            "when the tape resumes."
        )
    else:
        bounce_label = "⏱ Bounce started"
        bounce_tip = (
            "SPY starting to bounce? Click to open a window tracking which shorts stay weak / RW. "
            "Click again when the bounce ends to get the ranked list in the Alert Center. "
            "Auto mode does both on its own."
        )

    not_connected_tip = "Waiting for the bot to connect."
    return [
        {
            "command": "pullback_window",
            "label": pullback_label,
            "tooltip": pullback_tip if connected else not_connected_tip,
            "enabled": connected,
            "recommended": env == "bullish_strong" or long_window,
            "advanced": True,
        },
        {
            "command": "bounce_window",
            "label": bounce_label,
            "tooltip": bounce_tip if connected else not_connected_tip,
            "enabled": connected,
            "recommended": env == "bearish_strong" or short_window,
            "advanced": True,
        },
        {
            "command": "strongest_30m",
            "label": "Strongest 30m",
            "tooltip": (
                "Emit the strongest longs vs SPY over the last 30 minutes into the Alert Center. "
                "Auto mode emits this every 30m in bullish-weak."
                if connected
                else not_connected_tip
            ),
            "enabled": connected,
            "recommended": env == "bullish_weak",
            "advanced": False,
        },
        {
            "command": "weakest_30m",
            "label": "Weakest 30m",
            "tooltip": (
                "Emit the weakest shorts vs SPY over the last 30 minutes into the Alert Center. "
                "Auto mode emits this every 30m in bearish-weak."
                if connected
                else not_connected_tip
            ),
            "enabled": connected,
            "recommended": env == "bearish_weak",
            "advanced": False,
        },
        {
            "command": "movers_30m",
            "label": "Movers 30m",
            "tooltip": (
                "Emit BOTH the strongest longs and weakest shorts of the last 30 minutes into the "
                "Alert Center. Auto mode emits both every 30m in neutral/chop."
                if connected
                else not_connected_tip
            ),
            "enabled": connected,
            "recommended": env == "neutral_chop",
            "advanced": False,
        },
    ]


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
            "select User mode: N/A to hand control back."
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
    """Compact BounceBot status strip for the Trading Desk.

    Owns the live service (auto-connects on startup) and shows only what
    matters while trading: health chips, the market-regime chip with its
    user annotation/override, RRS status, and Start/Stop scanning on the top
    row. Automatic pullback/bounce tracking is the default; the normal entry
    row only exposes strongest/weakest/movers 30m. Manual window controls stay
    available behind an advanced toggle for diagnostics.
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

        # Auto owns pullback/bounce windows. Keep those manual controls hidden
        # unless explicitly opened for diagnostics.
        self.entry_assist_buttons: dict[str, QPushButton] = {}
        for spec in entry_assist_button_specs({}):
            button = QPushButton(spec["label"])
            button.setToolTip(spec["tooltip"])
            button.setEnabled(spec["enabled"])
            button.setVisible(not spec["advanced"])
            self.entry_assist_buttons[spec["command"]] = button

        self.entry_assist_auto_label = QLabel("Auto entry monitoring: waiting for connection")
        self.entry_assist_auto_label.setObjectName("MutedLabel")
        self.entry_assist_advanced_button = QPushButton("Manual window tools")
        self.entry_assist_advanced_button.setCheckable(True)
        self.entry_assist_advanced_button.setToolTip(
            "Advanced diagnostics only. Auto already opens and closes pullback/bounce windows "
            "from completed SPY bars."
        )

        self.regime_label = QLabel("Auto regime: n/a")
        self.regime_label.setObjectName("MutedLabel")
        self.technical_integrity_label = QLabel("Technicals: building")
        self.technical_integrity_label.setObjectName("MutedLabel")
        self.technical_integrity_label.setToolTip(
            "Technical Integrity appears after completed-M5 level tests resolve. Advisory only."
        )
        self.environment_input = QComboBox()
        self.environment_input.addItem("User mode: N/A (follow Auto)", "")
        for key, item in self.config["market_environments"].items():
            self.environment_input.addItem(f"User mode: {item.get('label', key)}", key)
        self.environment_input.setCurrentIndex(
            max(0, self.environment_input.findData(self.service.market_environment or ""))
        )
        self.environment_input.setToolTip(
            "N/A is the default: Auto controls the active market regime. Choosing a value creates "
            "a session-only user override and logs the bot's simultaneous Auto read for later review."
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

        strip = QHBoxLayout()
        strip.setContentsMargins(0, 0, 0, 0)
        strip.setSpacing(12)
        strip.addWidget(title)
        strip.addWidget(self.connection_label)
        strip.addWidget(self.status_label)
        strip.addWidget(self.active_label)
        strip.addWidget(self.regime_label)
        strip.addWidget(self.technical_integrity_label)
        strip.addWidget(self.environment_input)
        strip.addWidget(self.rrs_status_label, 1)
        strip.addWidget(self.start_scanning_button)
        strip.addWidget(self.stop_scanning_button)

        assist_label = QLabel("Entry assist:")
        assist_label.setObjectName("MutedLabel")
        assist_label.setToolTip(
            "Auto handles pullback/bounce windows from completed SPY bars. These buttons request "
            "plain trailing-30m strongest, weakest, or both-side mover lists."
        )
        assist_row = QHBoxLayout()
        assist_row.setContentsMargins(0, 0, 0, 0)
        assist_row.setSpacing(8)
        assist_row.addWidget(assist_label)
        assist_row.addWidget(self.entry_assist_auto_label)
        for button in self.entry_assist_buttons.values():
            assist_row.addWidget(button)
        assist_row.addWidget(self.entry_assist_advanced_button)
        assist_row.addStretch(1)

        rows = QVBoxLayout(self)
        rows.setContentsMargins(12, 6, 12, 6)
        rows.setSpacing(4)
        rows.addLayout(strip)
        rows.addLayout(assist_row)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

    def _wire_service(self) -> None:
        self.start_scanning_button.clicked.connect(self.service.start_scanning)
        self.stop_scanning_button.clicked.connect(self.service.stop_scanning)
        self.environment_input.currentIndexChanged.connect(self._on_environment_changed)
        self.entry_assist_advanced_button.toggled.connect(self._toggle_manual_entry_tools)

        self.service.rrsStatusChanged.connect(self._set_rrs_status)
        self.service.statusChanged.connect(self._set_status)
        self.service.connectionChanged.connect(self._set_connection)
        self.service.autoRegimeChanged.connect(self._set_auto_regime)
        self.service.technicalIntegrityChanged.connect(self._set_technical_integrity)
        self.service.entryAssistChanged.connect(self._set_entry_assist)
        for command, button in self.entry_assist_buttons.items():
            button.clicked.connect(
                lambda _checked=False, cmd=command: self.service.entry_assist_command(cmd)
            )
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
        else:
            self.service.clear_market_environment_override()
        self.service.refresh_auto_regime()

    def _toggle_manual_entry_tools(self, visible: bool) -> None:
        for spec in entry_assist_button_specs({}):
            if spec["advanced"]:
                self.entry_assist_buttons[spec["command"]].setVisible(bool(visible))

    def _set_auto_regime(self, reading) -> None:
        chip, tooltip = format_auto_regime_reading(reading)
        self.regime_label.setText(chip)
        self.regime_label.setToolTip(tooltip)

    def _set_technical_integrity(self, snapshot) -> None:
        chip, tooltip, color = format_technical_integrity_snapshot(snapshot)
        self.technical_integrity_label.setText(chip)
        self.technical_integrity_label.setToolTip(tooltip)
        self.technical_integrity_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    def _set_entry_assist(self, state) -> None:
        state = state if isinstance(state, dict) else {}
        if not state:
            auto_text = "Auto entry monitoring: waiting for connection"
        elif state.get("window_active"):
            sides = "/".join(str(side) for side in state.get("window_sides") or []) or "market"
            source = str(state.get("window_source") or "auto")
            auto_text = f"Auto entry monitoring: {source} {sides} window active"
        else:
            auto_text = "Auto entry monitoring: ON (completed-bar SPY pauses)"
        self.entry_assist_auto_label.setText(auto_text)
        for spec in entry_assist_button_specs(state):
            button = self.entry_assist_buttons.get(spec["command"])
            if button is None:
                continue
            button.setText(spec["label"])
            button.setToolTip(spec["tooltip"])
            button.setEnabled(spec["enabled"])
            wanted_name = "PrimaryButton" if spec["recommended"] else ""
            if button.objectName() != wanted_name:
                button.setObjectName(wanted_name)
                button.style().unpolish(button)
                button.style().polish(button)

    def _set_rrs_status(self, message: str) -> None:
        self.rrs_status_label.setText(message)
        self.rrs_status_label.setToolTip(message)  # row label may clip; hover shows all
        self.statusChanged.emit(message)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.statusChanged.emit(f"BounceBot: {message}")
        # Auto is a read-only chip. Never mirror it into the user's selector:
        # N/A stays N/A until the trader deliberately chooses an override.
        text = str(message or "")
        if text.startswith("Auto market regime"):
            self.service.refresh_auto_regime()

    def _set_connection(self, message: str) -> None:
        self.connection_label.setText(message)

    def _sync_scanning_buttons(self, scanning_enabled: bool) -> None:
        self.start_scanning_button.setEnabled(not scanning_enabled)
        self.stop_scanning_button.setEnabled(scanning_enabled)
