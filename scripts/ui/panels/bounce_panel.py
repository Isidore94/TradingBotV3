from __future__ import annotations

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ui.models.bounce import BounceAlert
from ui.services.bounce_service import BounceService, load_bounce_config
from ui.widgets.alert_feed_item import AlertFeedItem
from ui.widgets.empty_state import EmptyState
from ui.widgets.rrs_snapshot import RrsSnapshotWidget
from ui.widgets.section_header import SectionHeader


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
    statusChanged = Signal(str)

    def __init__(self, focus_service=None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.focus_service = focus_service
        self.service = BounceService(self)
        self.config = load_bounce_config()
        self._alert_count = 0
        self._d1_count = 0
        self._max_feed_items = 250
        self._syncing_controls = False
        self.toggle_boxes: dict[str, QCheckBox] = {}

        self.connection_label = QLabel("IB: disconnected")
        self.connection_label.setObjectName("MutedLabel")
        self.status_label = QLabel("stopped")
        self.status_label.setObjectName("MutedLabel")
        self.active_label = QLabel("active bounces: 0")
        self.active_label.setObjectName("MutedLabel")
        self.rrs_status_label = QLabel("RRS ready")
        self.rrs_status_label.setObjectName("MutedLabel")

        self.start_button = QPushButton("Connect")
        self.start_button.setObjectName("PrimaryButton")
        self.disconnect_button = QPushButton("Disconnect")
        self.restart_button = QPushButton("Reconnect")
        self.start_scanning_button = QPushButton("Start Scanning")
        self.stop_scanning_button = QPushButton("Stop Scanning")
        self.clear_button = QPushButton("Clear Feed")
        self.settings_button = QPushButton("Bounce Types")
        self.settings_button.setCheckable(True)

        self.rrs_threshold_input = QDoubleSpinBox()
        self.rrs_threshold_input.setRange(0.0, 5.0)
        self.rrs_threshold_input.setDecimals(1)
        self.rrs_threshold_input.setSingleStep(0.1)
        self.rrs_threshold_input.setValue(self.service.rrs_threshold)

        self.timeframe_input = QComboBox()
        for key, item in self.config["rrs_timeframes"].items():
            self.timeframe_input.addItem(str(item.get("label", key)), key)
        self.timeframe_input.setCurrentIndex(max(0, self.timeframe_input.findData(self.service.rrs_timeframe_key)))

        self.environment_input = QComboBox()
        for key, item in self.config["market_environments"].items():
            self.environment_input.addItem(str(item.get("label", key)), key)
        self.environment_input.setCurrentIndex(max(0, self.environment_input.findData(self.service.market_environment)))

        self.feed_container = QWidget()
        self.feed_layout = QVBoxLayout(self.feed_container)
        self.feed_layout.setContentsMargins(0, 0, 0, 0)
        self.feed_layout.setSpacing(8)
        self.feed_layout.addStretch(1)

        self.d1_container = QWidget()
        self.d1_layout = QVBoxLayout(self.d1_container)
        self.d1_layout.setContentsMargins(0, 0, 0, 0)
        self.d1_layout.setSpacing(8)
        self.d1_layout.addStretch(1)

        self.rrs_snapshot = RrsSnapshotWidget()
        if self.focus_service is not None:
            self.rrs_snapshot.set_focus_service(self.focus_service)

        self._build_layout()
        self._wire_service()
        self._sync_scanning_buttons(False)
        QTimer.singleShot(0, self.start)

    def _build_layout(self) -> None:
        header = SectionHeader(
            "BounceBot Live",
            "Auto-connects to BounceBot; live scan controls and RRS context stay on the desk.",
        )

        health_row = QHBoxLayout()
        health_row.setContentsMargins(0, 0, 0, 0)
        health_row.setSpacing(12)
        health_row.addWidget(self.connection_label)
        health_row.addWidget(self.status_label)
        health_row.addWidget(self.active_label)
        health_row.addStretch(1)

        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(8)
        for button in (
            self.disconnect_button,
            self.restart_button,
            self.start_scanning_button,
            self.stop_scanning_button,
            self.clear_button,
            self.settings_button,
        ):
            button_row.addWidget(button)
        button_row.addStretch(1)

        controls = QFrame()
        controls.setObjectName("StatusStrip")
        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(10, 8, 10, 8)
        controls_layout.setHorizontalSpacing(10)
        controls_layout.setVerticalSpacing(8)
        controls_layout.addWidget(QLabel("RRS sensitivity"), 0, 0)
        controls_layout.addWidget(self.rrs_threshold_input, 0, 1)
        controls_layout.addWidget(QLabel("Timeframe"), 0, 2)
        controls_layout.addWidget(self.timeframe_input, 0, 3)
        controls_layout.addWidget(QLabel("Market environment"), 0, 4)
        controls_layout.addWidget(self.environment_input, 0, 5)
        controls_layout.addWidget(self.rrs_status_label, 1, 0, 1, 6)

        feed_scroll = QScrollArea()
        feed_scroll.setWidgetResizable(True)
        feed_scroll.setWidget(self.feed_container)
        d1_scroll = QScrollArea()
        d1_scroll.setWidgetResizable(True)
        d1_scroll.setWidget(self.d1_container)

        split = QSplitter()
        split.addWidget(_framed_panel("Recent Bounce Alerts", feed_scroll))
        split.addWidget(_framed_panel("D1 Focus Alerts", d1_scroll))
        split.addWidget(_framed_panel("Relative Strength Board (RRS)", self.rrs_snapshot))
        split.setSizes([560, 300, 440])

        self.filter_panel = self._build_filter_panel()
        self.filter_panel.setVisible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(header)
        layout.addLayout(health_row)
        layout.addLayout(button_row)
        layout.addWidget(controls)
        layout.addWidget(split, 1)
        layout.addWidget(self.filter_panel)

    def _build_filter_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("StatusStrip")
        layout = QGridLayout(panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(4)
        labels = self.config["bounce_type_labels"]
        defaults = self.service.bounce_type_settings
        for index, key in enumerate(BOUNCE_TOGGLE_ORDER):
            if key not in defaults:
                continue
            checkbox = QCheckBox(str(labels.get(key, key)))
            checkbox.setChecked(bool(defaults.get(key)))
            checkbox.toggled.connect(lambda checked, bounce_key=key: self.service.set_bounce_type_enabled(bounce_key, checked))
            self.toggle_boxes[key] = checkbox
            layout.addWidget(checkbox, index // 5, index % 5)
        return panel

    def _wire_service(self) -> None:
        self.start_button.clicked.connect(self.start)
        self.disconnect_button.clicked.connect(self.stop)
        self.restart_button.clicked.connect(self.service.restart)
        self.start_scanning_button.clicked.connect(self.service.start_scanning)
        self.stop_scanning_button.clicked.connect(self.service.stop_scanning)
        self.clear_button.clicked.connect(self.clear_feed)
        self.settings_button.toggled.connect(self._toggle_bounce_settings)

        self.rrs_threshold_input.valueChanged.connect(self.service.set_rrs_threshold)
        self.timeframe_input.currentIndexChanged.connect(self._on_timeframe_changed)
        self.environment_input.currentIndexChanged.connect(self._on_environment_changed)

        self.service.alertReceived.connect(self._add_alert)
        self.service.rrsStatusChanged.connect(self._set_rrs_status)
        self.service.rrsSnapshotChanged.connect(self._set_rrs_snapshot)
        self.service.statusChanged.connect(self._set_status)
        self.service.connectionChanged.connect(self._set_connection)
        self.service.activeBouncesChanged.connect(lambda count: self.active_label.setText(f"active bounces: {count}"))
        self.service.scanningChanged.connect(self._sync_scanning_buttons)
        self.service.failed.connect(lambda message: QMessageBox.critical(self, "BounceBot Start Failed", message))

    def start(self) -> None:
        self.service.start()

    def stop(self) -> None:
        self.service.stop()

    def start_scanning(self) -> None:
        self.service.start_scanning()

    def stop_scanning(self) -> None:
        self.service.stop_scanning()

    def clear_feed(self) -> None:
        _clear_feed(self.feed_layout)
        _clear_feed(self.d1_layout)
        self._alert_count = 0
        self._d1_count = 0
        self.statusChanged.emit("Bounce feed cleared.")

    def on_close(self) -> None:
        self.service.stop()

    def _on_timeframe_changed(self) -> None:
        key = self.timeframe_input.currentData()
        if key:
            self.service.set_rrs_timeframe(str(key))

    def _on_environment_changed(self) -> None:
        key = self.environment_input.currentData()
        if key:
            self.service.set_market_environment(str(key))

    def _add_alert(self, alert: BounceAlert) -> None:
        if _is_feed_noise_alert(alert):
            return
        if alert.is_d1 and not _is_actionable_d1_alert(alert):
            return
        target_layout = self.d1_layout if alert.is_d1 else self.feed_layout
        is_focus = bool(self.focus_service and alert.symbol and self.focus_service.is_focus(alert.symbol))
        target_layout.insertWidget(0, AlertFeedItem(alert, is_focus=is_focus))
        if alert.is_d1:
            self._d1_count += 1
            _trim_feed(target_layout, self._max_feed_items)
        else:
            self._alert_count += 1
            _trim_feed(target_layout, self._max_feed_items)
        label = f"Bounce alerts: {self._alert_count} | D1 events: {self._d1_count}"
        self.statusChanged.emit(label)

    def _set_rrs_status(self, message: str) -> None:
        self.rrs_status_label.setText(message)
        self.statusChanged.emit(message)

    def _set_rrs_snapshot(self, payload) -> None:
        self.rrs_snapshot.update_snapshot(payload)

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.statusChanged.emit(f"BounceBot: {message}")

    def _set_connection(self, message: str) -> None:
        self.connection_label.setText(message)

    def _sync_scanning_buttons(self, scanning_enabled: bool) -> None:
        self.start_scanning_button.setEnabled(not scanning_enabled)
        self.stop_scanning_button.setEnabled(scanning_enabled)

    def _toggle_bounce_settings(self, visible: bool) -> None:
        self.filter_panel.setVisible(visible)
        self.settings_button.setText("Hide Bounce Types" if visible else "Bounce Types")


def _framed_panel(title: str, child: QWidget) -> QFrame:
    frame = QFrame()
    frame.setObjectName("Panel")
    label = QLabel(title)
    label.setObjectName("SectionTitle")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(8)
    layout.addWidget(label)
    layout.addWidget(child, 1)
    return frame


def _clear_feed(layout: QVBoxLayout) -> None:
    while layout.count() > 1:
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def _trim_feed(layout: QVBoxLayout, max_items: int) -> None:
    while layout.count() > max_items + 1:
        item = layout.takeAt(layout.count() - 2)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def _is_feed_noise_alert(alert: BounceAlert) -> bool:
    text = f"{alert.raw_text} {alert.trigger}".strip().lower()
    return not alert.is_d1 and alert.side == "WATCH" and "candle has closed" in text


def _is_actionable_d1_alert(alert: BounceAlert) -> bool:
    text = f"{alert.raw_text} {alert.trigger} {alert.tag}".upper()
    actionable_tokens = (
        "UPGRADE",
        "FAVORITE",
        "HIGH_CONVICTION",
        "NEAR_FAVORITE",
        "RETEST FOLLOW",
        "15EMA",
        "EMA15",
        "1ST-DEV D1 BOUNCE",
        "FIRST_DEV_RETEST",
        "AVWAPE D1 BOUNCE",
        "AVWAP_BOUNCE",
        "TRENDLINE",
        "PREVIOUS-DAY",
        "5D BREAKOUT",
        "THETA_RECOMMENDED",
        "PUT PREMIUM VIABLE",
    )
    if any(token in text for token in actionable_tokens):
        return True

    noisy_tokens = (
        "2ND-DEV",
        "SECOND_DEV",
        "UPPER_2",
        "LOWER_2",
        "UPPER_3",
        "LOWER_3",
        "CROSS_UP_UPPER_2",
        "CROSS_DOWN_LOWER_2",
        "PREV_CROSS_UP_UPPER_2",
        "PREV_CROSS_DOWN_LOWER_2",
        "BOUNCE_UPPER_2",
        "BOUNCE_LOWER_2",
        "PREV_BOUNCE_UPPER_2",
        "PREV_BOUNCE_LOWER_2",
    )
    if any(token in text for token in noisy_tokens):
        return False

    return False
