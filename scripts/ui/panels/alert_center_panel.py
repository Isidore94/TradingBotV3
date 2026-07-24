from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
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

from alert_review_state import load_ignored_alert_symbols, save_ignored_alert_symbols
from chart_watch import (
    ChartWatch,
    WATCH_KINDS,
    arm_chart_watch,
    evaluate_chart_watch,
    watch_is_stale,
)
from project_paths import (
    ALERT_CENTER_IGNORED_SYMBOLS_FILE,
    get_local_setting,
    save_local_setting,
)
from ui.models.bounce import (
    BounceAlert,
    CHART_WATCH_TAG,
    is_chart_watch_alert,
    is_entry_assist_text,
)
from ui.widgets.alert_chart_review import AlertChartReview
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
    must never be swallowed by the tier gate. Chart-watch hits pass for the
    same reason: the trader armed that exact condition from the M5 chart.
    Untiered alerts (regime notes, pause-watch summaries) pass everything
    except the S-only mode, where only bangers/S-tier remain.
    """
    if mode in ("", "all"):
        return True
    if (
        is_banger_alert(alert)
        or is_proven_alert(alert)
        or is_entry_assist_alert(alert)
        or is_chart_watch_alert(alert)
    ):
        return True
    tier = extract_alert_tier(alert)
    if not tier:
        return mode != "S"
    return _TIER_RANK.get(tier, 0) >= _TIER_RANK.get(mode, 0)


def alert_is_loud(alert: BounceAlert) -> bool:
    """Alerts worth a sound: bangers, proven configs, S/A tiers, ready D1,
    and chart-watch hits (the trader armed the exact condition and is
    waiting on it)."""
    return (
        is_banger_alert(alert)
        or is_proven_alert(alert)
        or is_ready_d1_alert(alert)
        or is_chart_watch_alert(alert)
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

    def __init__(self, focus_service=None, parent=None, *, ignored_symbols_path=None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.focus_service = focus_service
        self._bounce_service = None
        self._alerts: list[BounceAlert] = []
        self._d1_alerts: list[BounceAlert] = []
        self._review_queue: list[BounceAlert] = []
        self._current_review_alert: BounceAlert | None = None
        # One-shot chart watches armed from the visual M5 review. Session-
        # scoped in-memory state, like the review queue itself.
        self._chart_watches: list[ChartWatch] = []
        self._embedded_detail_enabled = True
        focus_store = getattr(self.focus_service, "store", None)
        default_store = bool(
            self.focus_service is not None
            and getattr(focus_store, "uses_default_paths", lambda: False)()
        )
        persist_ignored = ignored_symbols_path is not None or default_store
        self._ignored_symbols_path = (
            Path(ignored_symbols_path or ALERT_CENTER_IGNORED_SYMBOLS_FILE)
            if persist_ignored
            else None
        )
        self._ignored_market_date = date.today().isoformat()
        self._ignored_symbols = (
            load_ignored_alert_symbols(
                self._ignored_symbols_path,
                market_date=self._ignored_market_date,
            )
            if self._ignored_symbols_path is not None
            else set()
        )
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
        self.ignored_button = QPushButton()
        self.ignored_button.clicked.connect(self._restore_ignored_symbol_dialog)
        self._refresh_ignored_button()

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
        self.rrs_snapshot.symbolActivated.connect(self._show_board_symbol_snapshot)

        # RS/RW Board tab: the automatic entry-assist board on top (regime +
        # pause detection + live window / preview rankings + 30m movers, no
        # clicks) over the RRS sweep snapshot.
        self.entry_board = EntryAssistBoard()
        self.entry_board.symbolActivated.connect(self._show_board_symbol_snapshot)
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
        self.chart_review = AlertChartReview(self)
        self.chart_review.removeTodayRequested.connect(
            self._remove_review_alert_for_today
        )
        self.chart_review.focusRequested.connect(self._add_review_alert_to_focus)
        self.chart_review.skipRequested.connect(self._skip_review_alert)
        self.chart_review.crossFocusToggled.connect(self._toggle_review_cross_focus)
        self.chart_review.watchToggled.connect(self._toggle_chart_watch)

        # Armed chart watches are re-checked against the bot's cached M5 bars
        # every 30s (bars complete on 5-minute boundaries; this bounds the
        # trigger latency the same way the integrity/regime timers do).
        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(30_000)
        self._watch_timer.timeout.connect(self._poll_chart_watches)
        self._watch_timer.start()

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.chart_review)
        splitter.addWidget(self.tabs)
        splitter.addWidget(d1_section)
        splitter.addWidget(self.detail_view)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        splitter.setStretchFactor(3, 1)

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
        controls.addWidget(self.ignored_button)
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
        self._refresh_ignored_market_date()
        if _is_feed_noise_alert(alert):
            return
        if alert.symbol and alert.symbol in self._ignored_symbols:
            return
        # D1 Focus is reserved for favorite/high-conviction transitions
        # (final bucket upgrades only). Developing trigger/watch observations
        # are research evidence and are excluded from both actionable feeds.
        if alert.is_d1 and is_ready_d1_alert(alert):
            self._enqueue_review_alert(alert)
            self._add_d1_alert(alert)
            return
        self._alerts.insert(0, alert)
        del self._alerts[MAX_FEED_ITEMS * 2 :]
        is_focus = self._alert_is_focus(alert)
        if alert_passes_feed_gate(alert, self._min_tier_mode(), is_focus=is_focus):
            self._enqueue_review_alert(alert)
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
        armed = len(self._chart_watches)
        self._alerts.clear()
        self._d1_alerts.clear()
        self._review_queue.clear()
        self._chart_watches.clear()
        self._current_review_alert = None
        self.chart_review.clear()
        self._rebuild_feed()
        suffix = f" {armed} armed chart watch(es) disarmed." if armed else ""
        self.statusChanged.emit(f"Alert feeds cleared.{suffix}")

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
        message = f"✕ {alert.symbol}: disliked and removed from today's Alert Center review."
        if self.focus_service.is_focus(alert.symbol):
            self.focus_service.remove_everywhere(alert.symbol)
            message += " Removed from focus picks."
        self._ignore_alert_symbol(alert.symbol)
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
                if a.symbol not in self._ignored_symbols
                and alert_passes_feed_gate(a, mode, is_focus=self._alert_is_focus(a))
            ][:MAX_FEED_ITEMS]
        ):
            self._insert_item_into(self.feed_layout, alert, MAX_FEED_ITEMS)
        self._clear_feed_layout(self.d1_feed_layout)
        for alert in reversed(
            [
                alert
                for alert in self._d1_alerts
                if alert.symbol not in self._ignored_symbols
            ][:MAX_D1_FEED_ITEMS]
        ):
            self._insert_item_into(self.d1_feed_layout, alert, MAX_D1_FEED_ITEMS)

    def _enqueue_review_alert(self, alert: BounceAlert) -> None:
        """Queue one visual review per symbol; refresh the active symbol live."""
        if not alert.symbol:
            return
        if (
            self._current_review_alert is not None
            and self._current_review_alert.symbol == alert.symbol
        ):
            self._current_review_alert = alert
            self._render_current_review()
            return
        self._review_queue = [
            queued for queued in self._review_queue if queued.symbol != alert.symbol
        ]
        self._review_queue.append(alert)
        if self._current_review_alert is None:
            self._advance_review_queue()
        else:
            self.chart_review.set_queued_count(len(self._review_queue))

    def _select_review_alert(self, alert: BounceAlert) -> None:
        """A feed-row click makes that alert the active visual review."""
        if not alert.symbol or alert.symbol in self._ignored_symbols:
            return
        current = self._current_review_alert
        if current is not None and current.symbol != alert.symbol:
            self._review_queue = [
                queued
                for queued in self._review_queue
                if queued.symbol not in {current.symbol, alert.symbol}
            ]
            self._review_queue.insert(0, current)
        else:
            self._review_queue = [
                queued for queued in self._review_queue if queued.symbol != alert.symbol
            ]
        self._current_review_alert = alert
        self._render_current_review()

    def _advance_review_queue(self) -> None:
        self._current_review_alert = (
            self._review_queue.pop(0) if self._review_queue else None
        )
        self._render_current_review()

    def _render_current_review(self) -> None:
        alert = self._current_review_alert
        if alert is None:
            self.chart_review.clear()
            return
        bot = None
        if self._bounce_service is not None:
            try:
                bot = self._bounce_service.current_bot()
            except Exception:
                bot = None
        self.chart_review.set_alert(
            alert,
            bot=bot,
            focus_category=favorite_category_for_alert(alert),
            queued=len(self._review_queue),
            armed_kinds=self.armed_watch_kinds(alert.symbol),
            cross_active=self._review_cross_active(alert),
        )

    def _skip_review_alert(self, alert: BounceAlert) -> None:
        if (
            self._current_review_alert is None
            or self._current_review_alert.symbol != alert.symbol
        ):
            return
        self.statusChanged.emit(
            f"Skipped {alert.symbol} for now; its feed item remains available."
        )
        self._advance_review_queue()

    def _add_review_alert_to_focus(self, alert: BounceAlert) -> None:
        if self.focus_service is None or not alert.symbol:
            return
        category = favorite_category_for_alert(alert)
        side = "short" if alert.side == "SHORT" else "long"
        added = self.focus_service.add(
            alert.symbol,
            side,
            category,
            origin=favorite_origin_for_alert(alert),
            context=alert.raw_text,
        )
        bucket = "Swing" if category == "swing" else "M5"
        message = (
            f"★ {alert.symbol}: added to {bucket} Focus {side}s."
            if added
            else f"★ {alert.symbol}: already in Focus Picks."
        )
        self.statusChanged.emit(message)
        self._advance_review_queue()

    def _toggle_review_cross_focus(self, alert: BounceAlert) -> None:
        """The chart's cross-promote toggle. Never advances the queue.

        M5 pick: toggle Swing Focus (the Focus Picks tab's D1/swing bucket)
        plus a pin in the D1 Focus feed. Swing pick: toggle the M5 Focus
        day-trade list."""
        if not alert.symbol:
            return
        if favorite_category_for_alert(alert) == "swing":
            self.toggle_m5_focus(
                alert.symbol,
                alert.side,
                origin=favorite_origin_for_alert(alert),
                context=alert.raw_text,
            )
        else:
            self.toggle_d1_focus(
                alert.symbol,
                alert.side,
                origin=favorite_origin_for_alert(alert),
                context=alert.raw_text,
            )
        self._refresh_review_cross_state()

    def is_d1_focus_pinned(self, symbol: str) -> bool:
        symbol = str(symbol or "").strip().upper()
        return any(
            alert.symbol == symbol and alert.tag == "d1_focus_pin"
            for alert in self._d1_alerts
        )

    def is_d1_focus_active(self, symbol: str, side: str = "") -> bool:
        """On = the pick sits in Swing Focus (Focus Picks tab) or is pinned."""
        symbol = str(symbol or "").strip().upper()
        if self.is_d1_focus_pinned(symbol):
            return True
        if self.focus_service is None:
            return False
        focus_side = "short" if side == "SHORT" else "long"
        return bool(self.focus_service.is_focus(symbol, focus_side, "swing"))

    def toggle_d1_focus(
        self, symbol: str, side: str = "", *, origin: str = "chart", context: str = ""
    ) -> bool:
        """'Add to D1 Focus' toggle: files the pick into SWING Focus (so it
        lands on the Focus Picks tab and the swing watchlists) AND pins it in
        the D1 Focus feed. Off removes both. Never touches the review queue.
        Returns the new state."""
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return False
        focus_side = "short" if side == "SHORT" else "long"
        if self.is_d1_focus_active(symbol, side):
            if self.focus_service is not None and self.focus_service.is_focus(
                symbol, focus_side, "swing"
            ):
                self.focus_service.remove(symbol, focus_side, "swing")
            self._unpin_d1_focus(symbol)
            self.statusChanged.emit(
                f"{symbol}: removed from Swing Focus and unpinned from the D1 Focus feed."
            )
            self._refresh_review_cross_state()
            return False
        if self.focus_service is not None:
            self.focus_service.add(symbol, focus_side, "swing", origin=origin, context=context)
        pinned = BounceAlert(
            time_text=datetime.now().strftime("%H:%M:%S"),
            symbol=symbol,
            side=side if side in ("LONG", "SHORT") else "WATCH",
            trigger=context or "Pinned to D1 Focus from a chart",
            tag="d1_focus_pin",
            raw_text=f"D1 FOCUS PIN {symbol}" + (f": {context}" if context else ""),
            payload={"d1_focus_pin": True},
        )
        self._add_d1_alert(pinned)
        self.statusChanged.emit(
            f"{symbol}: added to Swing Focus {focus_side}s (Focus Picks tab) "
            "and pinned to the D1 Focus feed."
        )
        self._refresh_review_cross_state()
        return True

    def _unpin_d1_focus(self, symbol: str) -> None:
        self._d1_alerts = [
            alert
            for alert in self._d1_alerts
            if not (alert.symbol == symbol and alert.tag == "d1_focus_pin")
        ]
        self._rebuild_feed()

    def is_m5_focus(self, symbol: str, side: str = "") -> bool:
        if self.focus_service is None:
            return False
        focus_side = "short" if side == "SHORT" else "long"
        return bool(
            self.focus_service.is_focus(str(symbol or "").strip().upper(), focus_side, "m5")
        )

    def toggle_m5_focus(
        self, symbol: str, side: str = "", *, origin: str = "chart", context: str = ""
    ) -> bool:
        """Toggle a name on the M5 Focus day-trade list; returns new state."""
        symbol = str(symbol or "").strip().upper()
        if self.focus_service is None or not symbol:
            return False
        focus_side = "short" if side == "SHORT" else "long"
        if self.focus_service.is_focus(symbol, focus_side, "m5"):
            self.focus_service.remove(symbol, focus_side, "m5")
            self.statusChanged.emit(f"{symbol}: removed from M5 Focus {focus_side}s.")
            self._refresh_review_cross_state()
            return False
        self.focus_service.add(symbol, focus_side, "m5", origin=origin, context=context)
        self.statusChanged.emit(
            f"{symbol}: added to M5 Focus {focus_side}s - BounceBot M5-scans it now."
        )
        self._refresh_review_cross_state()
        return True

    def _review_cross_active(self, alert: BounceAlert) -> bool:
        if not alert.symbol:
            return False
        if favorite_category_for_alert(alert) == "swing":
            return self.is_m5_focus(alert.symbol, alert.side)
        return self.is_d1_focus_active(alert.symbol, alert.side)

    def _refresh_review_cross_state(self) -> None:
        current = self._current_review_alert
        if current is not None:
            self.chart_review.set_cross_active(self._review_cross_active(current))

    # ------------------------------------------------------------------
    # Chart watches: armed only from visual charts (the review pane here, or
    # a snapshot popup passing this panel as its watch_host); a hit fires a
    # red Alert Center alert (tier-gate bypass + sound) and retires itself.
    def armed_watch_kinds(self, symbol: str) -> set[str]:
        symbol = str(symbol or "").strip().upper()
        return {watch.kind for watch in self._chart_watches if watch.symbol == symbol}

    def _m5_bars_for(self, symbol: str) -> list:
        bot = None
        if self._bounce_service is not None:
            try:
                bot = self._bounce_service.current_bot()
            except Exception:
                bot = None
        if bot is None:
            return []
        try:
            return bot.m5_chart_bars(symbol, max_sessions=1) or []
        except Exception:
            return []

    def arm_chart_watch_for(
        self, symbol: str, side: str, kind: str, *, source_text: str = ""
    ) -> bool:
        """Public arming surface for any visual chart. Returns True on arm."""
        symbol = str(symbol or "").strip().upper()
        if not symbol or kind not in WATCH_KINDS:
            return False
        label = WATCH_KINDS[kind]
        if kind in self.armed_watch_kinds(symbol):
            self.statusChanged.emit(f"{symbol}: {label} watch already armed.")
            return False
        watch = arm_chart_watch(
            kind,
            symbol,
            side,
            self._m5_bars_for(symbol),
            source_text=source_text,
        )
        self._chart_watches.append(watch)
        self._refresh_review_armed_kinds()
        level = f" against {watch.baseline:.2f}" if watch.baseline is not None else ""
        self.statusChanged.emit(
            f"{symbol}: {label} watch armed{level} - the first completed "
            "M5 bar that meets it flags red in the Alert Center."
        )
        return True

    def disarm_chart_watch_for(self, symbol: str, kind: str) -> bool:
        """Public disarm surface (the toggles' off-click). True if removed."""
        symbol = str(symbol or "").strip().upper()
        if kind not in self.armed_watch_kinds(symbol):
            return False
        self._chart_watches = [
            watch
            for watch in self._chart_watches
            if not (watch.symbol == symbol and watch.kind == kind)
        ]
        self._refresh_review_armed_kinds()
        self.statusChanged.emit(
            f"{symbol}: {WATCH_KINDS.get(kind, kind)} watch disarmed."
        )
        return True

    def _toggle_chart_watch(self, alert: BounceAlert, kind: str) -> None:
        if not alert.symbol:
            return
        if kind in self.armed_watch_kinds(alert.symbol):
            self.disarm_chart_watch_for(alert.symbol, kind)
        else:
            self.arm_chart_watch_for(
                alert.symbol, alert.side, kind, source_text=alert.raw_text
            )

    def _poll_chart_watches(self, now: datetime | None = None) -> None:
        if not self._chart_watches:
            return
        moment = now or datetime.now()
        before = len(self._chart_watches)
        live = [
            watch
            for watch in self._chart_watches
            if not watch_is_stale(watch, now=moment)
        ]
        remaining: list[ChartWatch] = []
        triggered = []
        for watch in live:
            hit = None
            bars = self._m5_bars_for(watch.symbol)
            if bars:
                try:
                    hit = evaluate_chart_watch(watch, bars, now=moment)
                except Exception:
                    hit = None
            if hit is None:
                remaining.append(watch)
            else:
                triggered.append(hit)
        self._chart_watches = remaining
        for hit in triggered:
            self.add_alert(self._chart_watch_alert(hit, moment))
        if len(remaining) != before:
            self._refresh_review_armed_kinds()

    def _chart_watch_alert(self, hit, moment: datetime) -> BounceAlert:
        watch = hit.watch
        return BounceAlert(
            time_text=moment.strftime("%H:%M:%S"),
            symbol=watch.symbol,
            side=watch.side,
            trigger=hit.message,
            timeframe="M5",
            tag=CHART_WATCH_TAG,
            raw_text=f"CHART WATCH {watch.symbol} ({watch.side}): {hit.message}",
            payload={
                "chart_watch_kind": watch.kind,
                "armed_at": watch.armed_at.isoformat(),
                "source_text": watch.source_text,
            },
        )

    def _refresh_review_armed_kinds(self) -> None:
        current = self._current_review_alert
        if current is not None:
            self.chart_review.set_armed_kinds(self.armed_watch_kinds(current.symbol))

    def _remove_review_alert_for_today(self, alert: BounceAlert) -> None:
        """Drop a name from today's visual processing without changing scans."""
        if not alert.symbol:
            return
        self._ignore_alert_symbol(alert.symbol)
        self.statusChanged.emit(
            f"{alert.symbol}: removed from Alert Center processing for today. "
            "BounceBot scanning and watchlists are unchanged."
        )

    def _ignore_alert_symbol(self, symbol: str) -> None:
        self._refresh_ignored_market_date()
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return
        self._ignored_symbols.add(symbol)
        if self._ignored_symbols_path is not None:
            try:
                self._ignored_symbols = save_ignored_alert_symbols(
                    self._ignored_symbols,
                    self._ignored_symbols_path,
                    market_date=self._ignored_market_date,
                )
            except OSError:
                pass
        self._alerts = [alert for alert in self._alerts if alert.symbol != symbol]
        self._d1_alerts = [
            alert for alert in self._d1_alerts if alert.symbol != symbol
        ]
        self._review_queue = [
            alert for alert in self._review_queue if alert.symbol != symbol
        ]
        self._chart_watches = [
            watch for watch in self._chart_watches if watch.symbol != symbol
        ]
        if (
            self._current_review_alert is not None
            and self._current_review_alert.symbol == symbol
        ):
            self._current_review_alert = None
        self._rebuild_feed()
        self._refresh_ignored_button()
        if self._current_review_alert is None:
            self._advance_review_queue()

    def _restore_ignored_symbol_dialog(self) -> None:
        if not self._ignored_symbols:
            return
        symbol, accepted = QInputDialog.getItem(
            self,
            "Restore Alert Center symbol",
            "Return this symbol to today's Alert Center processing:",
            sorted(self._ignored_symbols),
            0,
            False,
        )
        if accepted and symbol:
            self._restore_ignored_symbol(symbol)

    def _restore_ignored_symbol(self, symbol: str) -> None:
        self._refresh_ignored_market_date()
        symbol = str(symbol or "").strip().upper()
        if symbol not in self._ignored_symbols:
            return
        self._ignored_symbols.remove(symbol)
        if self._ignored_symbols_path is not None:
            try:
                self._ignored_symbols = save_ignored_alert_symbols(
                    self._ignored_symbols,
                    self._ignored_symbols_path,
                    market_date=self._ignored_market_date,
                )
            except OSError:
                pass
        self._refresh_ignored_button()
        self.statusChanged.emit(
            f"{symbol}: restored to today's Alert Center processing."
        )

    def _refresh_ignored_button(self) -> None:
        count = len(self._ignored_symbols)
        self.ignored_button.setText(f"Removed today ({count})")
        self.ignored_button.setEnabled(count > 0)
        self.ignored_button.setToolTip(
            "Restore a symbol removed from today's Alert Center processing."
        )

    def _refresh_ignored_market_date(self) -> None:
        current = date.today().isoformat()
        if current == self._ignored_market_date:
            return
        self._ignored_market_date = current
        self._ignored_symbols = (
            load_ignored_alert_symbols(
                self._ignored_symbols_path,
                market_date=current,
            )
            if self._ignored_symbols_path is not None
            else set()
        )
        self._refresh_ignored_button()

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
        self._show_board_symbol_snapshot(alert.symbol, alert.side)

    def _show_board_symbol_snapshot(self, symbol: str, side: str = "") -> None:
        """RS/RW-board ticker click: use the same cache-only quick look."""
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return
        from ui.widgets.symbol_snapshot_dialog import show_symbol_snapshot

        bot = None
        if self._bounce_service is not None:
            try:
                bot = self._bounce_service.current_bot()
            except Exception:
                bot = None
        # The popup is a visual chart, so it carries the chart-only actions
        # (D1 Focus pin + armed watches) with this panel as their host.
        show_symbol_snapshot(self, symbol, bot=bot, side=side, watch_host=self)

    def _show_alert_detail(self, alert: BounceAlert) -> None:
        if not alert.symbol:
            return
        self._select_review_alert(alert)
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
