from __future__ import annotations

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from project_paths import get_local_setting, save_local_setting
from ui.models.bounce import BounceAlert
from ui.widgets.alert_feed_item import AlertFeedItem
from ui.widgets.rrs_snapshot import RrsSnapshotWidget
from ui.widgets.section_header import SectionHeader
from ui.widgets.setup_detail_view import SetupDetailView

_TIER_RE = re.compile(r"\[([SABCD])-TIER\]", re.IGNORECASE)

MIN_TIER_CHOICES = (
    ("All alerts", "all"),
    ("B tier and above", "B"),
    ("A tier and above", "A"),
    ("S tier / bangers only", "S"),
)
_TIER_RANK = {"S": 4, "A": 3, "B": 2, "C": 1, "D": 0}
MAX_FEED_ITEMS = 250


def _is_feed_noise_alert(alert: BounceAlert) -> bool:
    text = f"{alert.raw_text} {alert.trigger}".strip().lower()
    return not alert.is_d1 and alert.side == "WATCH" and "candle has closed" in text


def _is_actionable_d1_alert(alert: BounceAlert) -> bool:
    prefix = str(alert.raw_text or "").split(":", 1)[0].strip().upper()
    return prefix == "MASTER_AVWAP_D1_BUCKET_UPGRADE"


def extract_alert_tier(alert: BounceAlert) -> str:
    match = _TIER_RE.search(str(alert.raw_text or ""))
    return match.group(1).upper() if match else ""


def is_banger_alert(alert: BounceAlert) -> bool:
    return "BANGER" in str(alert.raw_text or "").upper()


def alert_passes_min_tier(alert: BounceAlert, mode: str) -> bool:
    """Filter policy for the live feed.

    Bangers always pass (they are the sit-back-and-wait trades). Untiered
    alerts (D1 focus upgrades, regime notes) pass everything except the
    S-only mode, where only bangers/S-tier remain.
    """
    if mode in ("", "all"):
        return True
    if is_banger_alert(alert):
        return True
    tier = extract_alert_tier(alert)
    if not tier:
        return mode != "S"
    return _TIER_RANK.get(tier, 0) >= _TIER_RANK.get(mode, 0)


def alert_is_loud(alert: BounceAlert) -> bool:
    """Alerts worth a sound: bangers and S/A tiers."""
    return is_banger_alert(alert) or extract_alert_tier(alert) in {"S", "A"}


class _ClickableItem(QFrame):
    clicked = Signal(object)

    def __init__(self, alert: BounceAlert, *, is_focus: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.alert = alert
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(AlertFeedItem(alert, is_focus=is_focus))
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.clicked.emit(self.alert)
        super().mousePressEvent(event)


class AlertCenterPanel(QFrame):
    """The sit-back-and-wait surface: one loud, tier-filtered alert stream.

    Bounce alerts, RW/RS bangers, D1 focus upgrades, and regime changes land
    in a single feed with a minimum-tier gate and an optional sound for
    S/A-tier and banger alerts. Clicking an alert opens the symbol's setup
    docs and trade plan below the feed.
    """

    statusChanged = Signal(str)

    def __init__(self, focus_service=None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.focus_service = focus_service
        self._alerts: list[BounceAlert] = []

        self.min_tier_input = QComboBox()
        for label, mode in MIN_TIER_CHOICES:
            self.min_tier_input.addItem(label, mode)
        saved_mode = str(get_local_setting("qt_alert_min_tier", "all") or "all")
        self.min_tier_input.setCurrentIndex(max(0, self.min_tier_input.findData(saved_mode)))
        self.min_tier_input.currentIndexChanged.connect(self._on_prefs_changed)

        self.sound_input = QCheckBox("Sound on S/A + bangers")
        self.sound_input.setChecked(bool(get_local_setting("qt_alert_sound", True)))
        self.sound_input.toggled.connect(self._on_prefs_changed)

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_feed)

        self.feed_container = QWidget()
        self.feed_layout = QVBoxLayout(self.feed_container)
        self.feed_layout.setContentsMargins(0, 0, 0, 0)
        self.feed_layout.setSpacing(8)
        self.feed_layout.addStretch(1)

        feed_scroll = QScrollArea()
        feed_scroll.setWidgetResizable(True)
        feed_scroll.setWidget(self.feed_container)

        self.rrs_snapshot = RrsSnapshotWidget()
        if self.focus_service is not None:
            self.rrs_snapshot.set_focus_service(self.focus_service)

        self.tabs = QTabWidget()
        self.tabs.addTab(feed_scroll, "Alerts")
        self.tabs.addTab(self.rrs_snapshot, "RRS Board")

        self.detail_view = SetupDetailView(self)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.detail_view)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        header = SectionHeader(
            "Alert Center",
            "Every live signal in one stream: bounces, bangers, D1 focus upgrades, regime changes.",
        )
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        controls.addWidget(QLabel("Show"))
        controls.addWidget(self.min_tier_input)
        controls.addWidget(self.sound_input)
        controls.addStretch(1)
        controls.addWidget(clear_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(header)
        layout.addLayout(controls)
        layout.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    def attach_service(self, service) -> None:
        service.alertReceived.connect(self.add_alert)
        service.rrsSnapshotChanged.connect(self.rrs_snapshot.update_snapshot)
        service.statusChanged.connect(self._maybe_add_status_alert)

    def add_alert(self, alert: BounceAlert) -> None:
        if _is_feed_noise_alert(alert):
            return
        if alert.is_d1 and not _is_actionable_d1_alert(alert):
            return
        self._alerts.insert(0, alert)
        del self._alerts[MAX_FEED_ITEMS * 2 :]
        if alert_passes_min_tier(alert, self._min_tier_mode()):
            self._insert_item(alert)
            if self.sound_input.isChecked() and alert_is_loud(alert):
                QApplication.beep()
        loud = sum(1 for item in self._alerts if alert_is_loud(item))
        self.statusChanged.emit(f"Alert center: {len(self._alerts)} alert(s), {loud} loud.")

    def clear_feed(self) -> None:
        self._alerts.clear()
        self._rebuild_feed()
        self.statusChanged.emit("Alert feed cleared.")

    # ------------------------------------------------------------------
    def _min_tier_mode(self) -> str:
        return str(self.min_tier_input.currentData() or "all")

    def _on_prefs_changed(self, *_args) -> None:
        save_local_setting("qt_alert_min_tier", self._min_tier_mode())
        save_local_setting("qt_alert_sound", bool(self.sound_input.isChecked()))
        self._rebuild_feed()

    def _insert_item(self, alert: BounceAlert) -> None:
        is_focus = bool(self.focus_service and alert.symbol and self.focus_service.is_focus(alert.symbol))
        item = _ClickableItem(alert, is_focus=is_focus)
        item.clicked.connect(self._show_alert_detail)
        self.feed_layout.insertWidget(0, item)
        while self.feed_layout.count() > MAX_FEED_ITEMS + 1:
            taken = self.feed_layout.takeAt(self.feed_layout.count() - 2)
            widget = taken.widget()
            if widget is not None:
                widget.deleteLater()

    def _rebuild_feed(self) -> None:
        while self.feed_layout.count() > 1:
            taken = self.feed_layout.takeAt(0)
            widget = taken.widget()
            if widget is not None:
                widget.deleteLater()
        mode = self._min_tier_mode()
        for alert in reversed([a for a in self._alerts if alert_passes_min_tier(a, mode)][:MAX_FEED_ITEMS]):
            self._insert_item(alert)

    def _show_alert_detail(self, alert: BounceAlert) -> None:
        if not alert.symbol:
            return
        feedback = alert.payload.get("feedback") if isinstance(alert.payload, dict) else {}
        feedback = feedback if isinstance(feedback, dict) else {}
        self.detail_view.show_setup(
            symbol=alert.symbol,
            side=alert.side if alert.side in {"LONG", "SHORT"} else "LONG",
            setup_family=str(feedback.get("master_avwap_setup_family") or ""),
            favorite_signals=[],
        )

    def _maybe_add_status_alert(self, message: str) -> None:
        text = str(message or "")
        if text.startswith("Auto market regime") or text.startswith("Market environment set"):
            self.add_alert(
                BounceAlert.from_callback(text, "regime")
            )
