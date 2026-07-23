from __future__ import annotations

import re

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from project_paths import get_local_setting, save_local_setting
from ui.models.bounce import BounceAlert, is_entry_assist_text
from ui.widgets.alert_feed_item import AlertFeedItem
from ui.widgets.entry_assist_board import EntryAssistBoard
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

# D1 focus alerts that mark a stock TURNING INTO a favorite / high-conviction
# name: the scan confirmed a genuine bucket upgrade. An armed-level crossing
# is still only developing evidence and stays out of both actionable feeds. A
# final Favorite / High Conviction bucket result belongs in the D1 Focus feed
# (user rule 2026-07-09: "only things that turn a stock into a favourite or
# high conviction bucket stock"). Generic champion D1 flags retain their live
# routing under the normal tier gate.
_D1_READY_PREFIXES = {
    # The D1 Focus feed is the M5 band-zone rubric: a scanned name bouncing off
    # AVWAPE / 1st-dev / 15-21EMA or breaking the next band, confirmed on two
    # completed bars. A fresh Favorite / High Conviction bucket upgrade still
    # surfaces here too.
    "MASTER_AVWAP_D1_ZONE",
    "MASTER_AVWAP_D1_BUCKET_UPGRADE",
    # Pre-armed tier flip: a non-S/A name closed through the A/S upgrade-target
    # level the scan armed one small move away - the headline D1 Focus event
    # (few per day, rvol/context gated, predicted pending next-scan confirm).
    "MASTER_AVWAP_D1_TIER_FLIP",
}
_D1_DEVELOPING_PREFIXES = {
    "MASTER_AVWAP_D1_RESEARCH",
    # Compatibility with messages already queued by an older bot process.
    "MASTER_AVWAP_D1_UPGRADE_TRIGGER",
    "MASTER_AVWAP_D1_UPGRADE_WATCH",
}


def _d1_alert_prefix(alert: BounceAlert) -> str:
    return str(alert.raw_text or "").split(":", 1)[0].strip().upper()


def is_developing_d1_alert(alert: BounceAlert) -> bool:
    return _d1_alert_prefix(alert) in _D1_DEVELOPING_PREFIXES


def _is_feed_noise_alert(alert: BounceAlert) -> bool:
    if is_developing_d1_alert(alert):
        return True
    text = f"{alert.raw_text} {alert.trigger}".strip().lower()
    return not alert.is_d1 and alert.side == "WATCH" and "candle has closed" in text


def is_ready_d1_alert(alert: BounceAlert) -> bool:
    return _d1_alert_prefix(alert) in _D1_READY_PREFIXES


def extract_alert_tier(alert: BounceAlert) -> str:
    match = _TIER_RE.search(str(alert.raw_text or ""))
    return match.group(1).upper() if match else ""


def is_banger_alert(alert: BounceAlert) -> bool:
    return "BANGER" in str(alert.raw_text or "").upper()


# Learning-loop PROVEN stamp: this exact bounce configuration (type/combo/
# swing trait/family/focus) has a measured winning record (n>=12, avg>=+0.45R,
# median>=0). These are the "see it live, take it" alerts.
_PROVEN_RE = re.compile(r"\bPROVEN\b")


def is_proven_alert(alert: BounceAlert) -> bool:
    return bool(_PROVEN_RE.search(str(alert.raw_text or "")))


def is_entry_assist_alert(alert: BounceAlert) -> bool:
    return str(alert.tag or "") == "entry_assist" or is_entry_assist_text(alert.raw_text)


def alert_passes_min_tier(alert: BounceAlert, mode: str) -> bool:
    """Filter policy for the live feed (D1 alerts route to their own feed).

    Bangers always pass (they are the sit-back-and-wait trades), and so does
    entry-assist output — the trader clicked a button asking for it, so it
    must never be swallowed by the tier gate. Untiered alerts (regime notes,
    pause-watch summaries) pass everything except the S-only mode, where only
    bangers/S-tier remain.
    """
    if mode in ("", "all"):
        return True
    if is_banger_alert(alert) or is_proven_alert(alert) or is_entry_assist_alert(alert):
        return True
    tier = extract_alert_tier(alert)
    if not tier:
        return mode != "S"
    return _TIER_RANK.get(tier, 0) >= _TIER_RANK.get(mode, 0)


def alert_is_loud(alert: BounceAlert) -> bool:
    """Alerts worth a sound: bangers, proven configs, S/A tiers, ready D1."""
    return (
        is_banger_alert(alert)
        or is_proven_alert(alert)
        or is_ready_d1_alert(alert)
        or extract_alert_tier(alert) in {"S", "A"}
    )


def alert_passes_feed_gate(alert: BounceAlert, mode: str, *, is_focus: bool = False) -> bool:
    """Liked (focus) picks always surface; everything else obeys the tier gate."""
    return is_focus or alert_passes_min_tier(alert, mode)


def alert_should_sound(alert: BounceAlert, *, is_focus: bool = False) -> bool:
    """Liked (focus) picks always sound; everything else needs to be loud."""
    return is_focus or alert_is_loud(alert)


def favorite_category_for_alert(alert: BounceAlert) -> str:
    """Where the ★ files a pick: D1/H1 alerts are swing material, the rest M5.

    Matches the trader's split: longs/shorts.txt alerts are M5 day-trade
    based, while bot-generated D1/H1 output is multi-day swing evidence.
    """
    if alert.is_d1 or str(alert.timeframe or "").strip().lower() in {"d1", "h1", "1h"}:
        return "swing"
    return "m5"


def favorite_origin_for_alert(alert: BounceAlert) -> str:
    """Which alert flavor a verdict came from - logged so the tracker can grade
    H1-sourced picks separately from D1-sourced ones (and M5 likewise)."""
    if alert.is_d1:
        return "d1"
    if str(alert.timeframe or "").strip().lower() in {"h1", "1h"}:
        return "h1"
    return "m5"


class _ClickableItem(QFrame):
    clicked = Signal(object)
    favoriteToggled = Signal(object)  # alert
    dislikeRequested = Signal(object)  # alert
    symbolClicked = Signal(object)  # alert - ticker name click -> chart snapshot

    def __init__(
        self,
        alert: BounceAlert,
        *,
        focus_category: str = "",
        show_favorite_button: bool = False,
        favorite_hint: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.alert = alert
        feed_item = AlertFeedItem(
            alert,
            focus_category=focus_category,
            show_favorite_button=show_favorite_button,
            favorite_hint=favorite_hint,
        )
        feed_item.favoriteToggled.connect(lambda: self.favoriteToggled.emit(self.alert))
        feed_item.dislikeRequested.connect(lambda: self.dislikeRequested.emit(self.alert))
        feed_item.symbolClicked.connect(lambda: self.symbolClicked.emit(self.alert))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(feed_item)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.clicked.emit(self.alert)
        super().mousePressEvent(event)


class AlertCenterPanel(QFrame):
    """The sit-back-and-wait surface, split into two stacked feeds.

    Top: the live intraday stream (bounce alerts, RW/RS bangers, regime
    notes, and generic champion D1 flags) behind the minimum-tier gate with an
    optional sound. Bottom: the D1 Focus feed - ONLY the moments a completed
    scan confirms that a stock turned into a favorite/high-conviction name.
    Developing armed-level observations remain in research logs. Clicking an
    alert opens the symbol's setup docs and trade plan - in the embedded pane
    below by default, or routed out through `setupRequested` when the desk
    disables the embedded pane (workspace mode shows the plan once, in the
    setups workspace's detail pane, instead of twice). Every alert carries a
    ★ at its right edge: click to favorite the pick into Focus Picks (D1/H1
    alerts file as Swing, intraday as M5), click a lit star to unfavorite.
    Favorited names come back gold-framed with a category badge, skip the tier
    gate, and always sound. ✕ logs a dislike with a typed reason.
    """

    statusChanged = Signal(str)
    setupRequested = Signal(dict)  # show_setup kwargs, when the embedded pane is off

    def __init__(self, focus_service=None, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.focus_service = focus_service
        self._bounce_service = None
        self._alerts: list[BounceAlert] = []
        self._d1_alerts: list[BounceAlert] = []
        self._embedded_detail_enabled = True
        if self.focus_service is not None:
            # Liking a pick (here or on the setups table) re-renders both feeds
            # so every alert for that name immediately shows the gold flag.
            self.focus_service.focusChanged.connect(self._rebuild_feed)

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

        # RS/RW Board tab: the automatic entry-assist board on top (regime +
        # pause detection + live window / preview rankings + 30m movers, no
        # clicks) over the RRS sweep snapshot.
        self.entry_board = EntryAssistBoard()
        board_tab = QWidget()
        board_layout = QVBoxLayout(board_tab)
        board_layout.setContentsMargins(0, 0, 0, 0)
        board_layout.setSpacing(8)
        board_layout.addWidget(self.entry_board, 3)
        board_layout.addWidget(self.rrs_snapshot, 2)

        self.tabs = QTabWidget()
        self.tabs.addTab(feed_scroll, "Alerts")
        self.tabs.addTab(board_tab, "RS/RW Board")

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
                "M5 band-zone rubric: a scanned name bouncing off AVWAPE / 1st-dev / "
                "15-21EMA or breaking the next band (two-bar confirm), plus fresh "
                "Favorite / High Conviction promotions.",
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
            "Live actionable alerts on top; confirmed Favorite / High Conviction D1 promotions below.",
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
        self._bounce_service = service
        service.alertReceived.connect(self.add_alert)
        service.rrsSnapshotChanged.connect(self.rrs_snapshot.update_snapshot)
        service.statusChanged.connect(self._maybe_add_status_alert)
        board_signal = getattr(service, "entryBoardChanged", None)
        if board_signal is not None:
            board_signal.connect(self.entry_board.update_board)

    def add_alert(self, alert: BounceAlert) -> None:
        if _is_feed_noise_alert(alert):
            return
        # D1 Focus is reserved for favorite/high-conviction transitions
        # (final bucket upgrades only). Developing trigger/watch observations
        # are research evidence and are excluded from both actionable feeds.
        if alert.is_d1 and is_ready_d1_alert(alert):
            self._add_d1_alert(alert)
            return
        self._alerts.insert(0, alert)
        del self._alerts[MAX_FEED_ITEMS * 2 :]
        is_focus = self._alert_is_focus(alert)
        if alert_passes_feed_gate(alert, self._min_tier_mode(), is_focus=is_focus):
            self._insert_item_into(self.feed_layout, alert, MAX_FEED_ITEMS)
            if self.sound_input.isChecked() and alert_should_sound(alert, is_focus=is_focus):
                QApplication.beep()
        self._emit_feed_status()

    def _add_d1_alert(self, alert: BounceAlert) -> None:
        self._d1_alerts.insert(0, alert)
        del self._d1_alerts[MAX_D1_FEED_ITEMS * 2 :]
        self._insert_item_into(self.d1_feed_layout, alert, MAX_D1_FEED_ITEMS)
        if self.sound_input.isChecked() and (is_ready_d1_alert(alert) or self._alert_is_focus(alert)):
            QApplication.beep()
        self._emit_feed_status()

    def _emit_feed_status(self) -> None:
        loud = sum(1 for item in self._alerts if alert_should_sound(item, is_focus=self._alert_is_focus(item)))
        self.statusChanged.emit(
            f"Alert center: {len(self._alerts)} live alert(s), {loud} loud; "
            f"{len(self._d1_alerts)} favorite-bucket transition(s) in D1 Focus."
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

    def _alert_is_focus(self, alert: BounceAlert) -> bool:
        return bool(self.focus_service and alert.symbol and self.focus_service.is_focus(alert.symbol))

    def _toggle_favorite(self, alert: BounceAlert) -> None:
        """The ★ on a feed item: favorite the pick, or unfavorite a lit one."""
        if self.focus_service is None or not alert.symbol:
            return
        origin = favorite_origin_for_alert(alert)
        if self.focus_service.is_focus(alert.symbol):
            self.focus_service.remove_everywhere(alert.symbol, origin=origin, context=alert.raw_text)
            message = f"Unfavorited {alert.symbol}: removed from focus picks."
        else:
            category = favorite_category_for_alert(alert)
            side = "short" if alert.side == "SHORT" else "long"
            self.focus_service.add(alert.symbol, side, category, origin=origin, context=alert.raw_text)
            bucket = "Swing" if category == "swing" else "M5"
            message = (
                f"★ {alert.symbol}: added to {bucket} Focus {side}s - its alerts now flag gold, "
                "skip the tier gate, and sound."
            )
        self.statusChanged.emit(message)

    def _dislike_alert(self, alert: BounceAlert) -> None:
        """The ✕ on a feed item: ask why, then log the dislike for AI review."""
        if self.focus_service is None or not alert.symbol:
            return
        reason, accepted = QInputDialog.getMultiLineText(
            self,
            f"Dislike {alert.symbol}",
            "Why is this a bad pick? Saved to pick_feedback.jsonl so an AI can\n"
            "review your dislikes and suggest scan/scoring changes.",
        )
        if not accepted:
            return
        self._record_dislike(alert, reason)

    def _record_dislike(self, alert: BounceAlert, reason: str) -> None:
        self.focus_service.record_feedback(
            alert.symbol,
            alert.side,
            "dislike",
            category=self.focus_service.focus_category(alert.symbol) or favorite_category_for_alert(alert),
            origin=favorite_origin_for_alert(alert),
            reason=reason,
            context=alert.raw_text,
        )
        message = f"✕ {alert.symbol}: dislike logged for AI review."
        if self.focus_service.is_focus(alert.symbol):
            self.focus_service.remove_everywhere(alert.symbol)
            message = f"✕ {alert.symbol}: dislike logged and removed from focus picks."
        self.statusChanged.emit(message)

    def _insert_item_into(self, layout, alert: BounceAlert, max_items: int) -> None:
        focus_category = ""
        if self.focus_service and alert.symbol:
            focus_category = self.focus_service.focus_category(alert.symbol) or ""
        bucket = "Swing Focus" if favorite_category_for_alert(alert) == "swing" else "M5 Focus"
        item = _ClickableItem(
            alert,
            focus_category=focus_category,
            show_favorite_button=self.focus_service is not None,
            favorite_hint=bucket,
        )
        item.clicked.connect(self._show_alert_detail)
        item.favoriteToggled.connect(self._toggle_favorite)
        item.dislikeRequested.connect(self._dislike_alert)
        item.symbolClicked.connect(self._show_symbol_snapshot)
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
        for alert in reversed(
            [
                a
                for a in self._alerts
                if alert_passes_feed_gate(a, mode, is_focus=self._alert_is_focus(a))
            ][:MAX_FEED_ITEMS]
        ):
            self._insert_item_into(self.feed_layout, alert, MAX_FEED_ITEMS)
        self._clear_feed_layout(self.d1_feed_layout)
        for alert in reversed(self._d1_alerts[:MAX_D1_FEED_ITEMS]):
            self._insert_item_into(self.d1_feed_layout, alert, MAX_D1_FEED_ITEMS)

    def set_embedded_detail_enabled(self, enabled: bool) -> None:
        """Workspace mode turns the embedded plan pane off so the setup is
        described in one place (the setups workspace's detail pane)."""
        self._embedded_detail_enabled = bool(enabled)
        if not self._embedded_detail_enabled:
            self.detail_view.setVisible(False)

    def _show_symbol_snapshot(self, alert: BounceAlert) -> None:
        """Ticker-name click: the D1+M5 candle quick look."""
        if not alert.symbol:
            return
        from ui.widgets.symbol_snapshot_dialog import show_symbol_snapshot

        bot = None
        if self._bounce_service is not None:
            try:
                bot = self._bounce_service.current_bot()
            except Exception:
                bot = None
        show_symbol_snapshot(self, alert.symbol, bot=bot, side=alert.side)

    def _show_alert_detail(self, alert: BounceAlert) -> None:
        if not alert.symbol:
            return
        feedback = alert.payload.get("feedback") if isinstance(alert.payload, dict) else {}
        feedback = feedback if isinstance(feedback, dict) else {}
        payload = {
            "symbol": alert.symbol,
            "side": alert.side if alert.side in {"LONG", "SHORT"} else "LONG",
            "setup_family": str(feedback.get("master_avwap_setup_family") or ""),
            "favorite_signals": [],
        }
        if self._embedded_detail_enabled:
            self.detail_view.show_setup(**payload)
        else:
            self.setupRequested.emit(payload)

    def _maybe_add_status_alert(self, message: str) -> None:
        text = str(message or "")
        if text.startswith("Auto market regime") or text.startswith("Market environment set"):
            self.add_alert(
                BounceAlert.from_callback(text, "regime")
            )
