from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from ui.services.bounce_service import BounceService, load_bounce_config
from ui.widgets.section_header import SectionHeader

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
    """Slim BounceBot status strip for the Trading Desk.

    Owns the live service (auto-connects on startup) and shows only what
    matters while trading: health chips, the market-regime chip with its
    manual override, and Start/Stop scanning. Connection management, RRS
    tuning, and bounce-type toggles live on the Settings page; alert feeds
    live in the Alert Center.
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

        self.regime_label = QLabel("Regime: auto")
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
        header = SectionHeader(
            "BounceBot",
            "Auto-connects on launch. Connection, RRS tuning, and bounce types live in Settings.",
        )

        strip = QHBoxLayout()
        strip.setContentsMargins(0, 0, 0, 0)
        strip.setSpacing(12)
        strip.addWidget(self.connection_label)
        strip.addWidget(self.status_label)
        strip.addWidget(self.active_label)
        strip.addWidget(self.regime_label)
        strip.addWidget(self.environment_input)
        strip.addWidget(self.environment_auto_button)
        strip.addStretch(1)
        strip.addWidget(self.start_scanning_button)
        strip.addWidget(self.stop_scanning_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)
        layout.addWidget(header)
        layout.addLayout(strip)
        layout.addWidget(self.rrs_status_label)

    def _wire_service(self) -> None:
        self.start_scanning_button.clicked.connect(self.service.start_scanning)
        self.stop_scanning_button.clicked.connect(self.service.stop_scanning)
        self.environment_input.currentIndexChanged.connect(self._on_environment_changed)
        self.environment_auto_button.clicked.connect(self._on_environment_auto)

        self.service.rrsStatusChanged.connect(self._set_rrs_status)
        self.service.statusChanged.connect(self._set_status)
        self.service.connectionChanged.connect(self._set_connection)
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
            self.regime_label.setText("Regime: manual")

    def _on_environment_auto(self) -> None:
        self.service.clear_market_environment_override()
        self.regime_label.setText("Regime: auto")

    def _set_rrs_status(self, message: str) -> None:
        self.rrs_status_label.setText(message)
        self.statusChanged.emit(message)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.statusChanged.emit(f"BounceBot: {message}")
        # Mirror the bot's auto-regime changes into the chip + dropdown.
        text = str(message or "")
        if text.startswith("Auto market regime"):
            self.regime_label.setText("Regime: auto")
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
