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
MAX_D1_FEED_ITEMS = 100

# D1 focus alerts that mark a "ready now" moment: price crossed a key armed
# S/R level toward an A/S upgrade, or the scan confirmed a genuine bucket
# upgrade. UPGRADE_WATCH / generic flags are context, not triggers.
_D1_READY_PREFIXES = {
    "MASTER_AVWAP_D1_BUCKET_UPGRADE",
    "MASTER_AVWAP_D1_UPGRADE_TRIGGER",
}


def _is_feed_noise_alert(alert: BounceAlert) -> bool:
    text = f"{alert.raw_text} {alert.trigger}".strip().lower()
    return not alert.is_d1 and alert.side == "WATCH" and "candle has closed" in text


def _d1_alert_prefix(alert: BounceAlert) -> str:
    return str(alert.raw_text or "").split(":", 1)[0].strip().upper()


def is_ready_d1_alert(alert: BounceAlert) -> bool:
    return _d1_alert_prefix(alert) in _D1_READY_PREFIXES


def extract_alert_tier(alert: BounceAlert) -> str:
    match = _TIER_RE.search(str(alert.raw_text or ""))
    return match.group(1).upper() if match else ""


def is_banger_alert(alert: BounceAlert) -> bool:
    return "BANGER" in str(alert.raw_text or "").upper()


def alert_passes_min_tier(alert: BounceAlert, mode: str) -> bool:
    """Filter policy for the live feed (D1 alerts route to their own feed).

    Bangers always pass (they are the sit-back-and-wait trades). Untiered
    alerts (regime notes, pause-watch summaries) pass everything except the
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
    """Alerts worth a sound: bangers, S/A tiers, and ready D1 focus alerts."""
    return (
        is_banger_alert(alert)
        or is_ready_d1_alert(alert)
        or extract_alert_tier(alert) in {"S", "A"}
    )


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
    """The sit-back-and-wait surface, split into two stacked feeds.

    Top: the live intraday stream (bounce alerts, RW/RS bangers, regime
    notes) behind the minimum-tier gate with an optional sound. Bottom: the
    D1 Focus feed - key-level crossings and bucket upgrades where a
    low-ranked swing name earns its way toward A/S. Clicking an alert in
    either feed opens the symbol's setup docs and trade plan below.
    """

    statusChanged = Signal(str)

    def __init__(self, focus_service=None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.focus_service = focus_service
        self._alerts: list[BounceAlert] = []
        self._d1_alerts: list[BounceAlert] = []

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

        self.d1_feed_container = QWidget()
        self.d1_feed_layout = QVBoxLayout(self.d1_feed_container)
        self.d1_feed_layout.setContentsMargins(0, 0, 0, 0)
        self.d1_feed_layout.setSpacing(8)
        self.d1_feed_layout.addStretch(1)

        d1_scroll = QScrollArea()
        d1_scroll.setWidgetResizable(True)
        d1_scroll.setWidget(self.d1_feed_container)

        d1_section = QWidget()
        d1_section_layout = QVBoxLayout(d1_section)
        d1_section_layout.setContentsMargins(0, 0, 0, 0)
        d1_section_layout.setSpacing(4)
        d1_section_layout.addWidget(
            SectionHeader(
                "D1 Focus",
                "Key S/R crossings + bucket upgrades: low-ranked swing names earning their way to A/S.",
            )
        )
        d1_section_layout.addWidget(d1_scroll, 1)

        self.detail_view = SetupDetailView(self)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.tabs)
        splitter.addWidget(d1_section)
        splitter.addWidget(self.detail_view)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 2)

        header = SectionHeader(
            "Alert Center",
            "Live intraday stream on top; D1 focus level-crossings and upgrades below.",
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
        if alert.is_d1:
            self._add_d1_alert(alert)
            return
        self._alerts.insert(0, alert)
        del self._alerts[MAX_FEED_ITEMS * 2 :]
        if alert_passes_min_tier(alert, self._min_tier_mode()):
            self._insert_item_into(self.feed_layout, alert, MAX_FEED_ITEMS)
            if self.sound_input.isChecked() and alert_is_loud(alert):
                QApplication.beep()
        self._emit_feed_status()

    def _add_d1_alert(self, alert: BounceAlert) -> None:
        self._d1_alerts.insert(0, alert)
        del self._d1_alerts[MAX_D1_FEED_ITEMS * 2 :]
        self._insert_item_into(self.d1_feed_layout, alert, MAX_D1_FEED_ITEMS)
        if self.sound_input.isChecked() and is_ready_d1_alert(alert):
            QApplication.beep()
        self._emit_feed_status()

    def _emit_feed_status(self) -> None:
        loud = sum(1 for item in self._alerts if alert_is_loud(item))
        ready = sum(1 for item in self._d1_alerts if is_ready_d1_alert(item))
        self.statusChanged.emit(
            f"Alert center: {len(self._alerts)} live alert(s), {loud} loud; "
            f"{len(self._d1_alerts)} D1 focus, {ready} ready."
        )

    def clear_feed(self) -> None:
        self._alerts.clear()
        self._d1_alerts.clear()
        self._rebuild_feed()
        self.statusChanged.emit("Alert feeds cleared.")

    # ------------------------------------------------------------------
    def _min_tier_mode(self) -> str:
        return str(self.min_tier_input.currentData() or "all")

    def _on_prefs_changed(self, *_args) -> None:
        save_local_setting("qt_alert_min_tier", self._min_tier_mode())
        save_local_setting("qt_alert_sound", bool(self.sound_input.isChecked()))
        self._rebuild_feed()

    def _insert_item_into(self, layout, alert: BounceAlert, max_items: int) -> None:
        is_focus = bool(self.focus_service and alert.symbol and self.focus_service.is_focus(alert.symbol))
        item = _ClickableItem(alert, is_focus=is_focus)
        item.clicked.connect(self._show_alert_detail)
        layout.insertWidget(0, item)
        while layout.count() > max_items + 1:
            taken = layout.takeAt(layout.count() - 2)
            widget = taken.widget()
            if widget is not None:
                widget.deleteLater()

    @staticmethod
    def _clear_feed_layout(layout) -> None:
        while layout.count() > 1:
            taken = layout.takeAt(0)
            widget = taken.widget()
            if widget is not None:
                widget.deleteLater()

    def _rebuild_feed(self) -> None:
        self._clear_feed_layout(self.feed_layout)
        mode = self._min_tier_mode()
        for alert in reversed([a for a in self._alerts if alert_passes_min_tier(a, mode)][:MAX_FEED_ITEMS]):
            self._insert_item_into(self.feed_layout, alert, MAX_FEED_ITEMS)
        self._clear_feed_layout(self.d1_feed_layout)
        for alert in reversed(self._d1_alerts[:MAX_D1_FEED_ITEMS]):
            self._insert_item_into(self.d1_feed_layout, alert, MAX_D1_FEED_ITEMS)

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
