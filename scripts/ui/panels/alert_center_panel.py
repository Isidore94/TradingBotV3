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
    QSizePolicy,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from alert_review_state import load_ignored_alert_symbols, save_ignored_alert_symbols
from chart_watch import (
    BAND_BOUNCE_PRIME_BUCKETS,
    BAND_BOUNCE_TRACKER_TYPES,
    ChartWatch,
    D1EventWatch,
    D1LevelWatch,
    D1_EVENT_KINDS,
    D1_LEVEL_KINDS,
    WATCH_KINDS,
    arm_chart_watch,
    evaluate_chart_watch,
    evaluate_d1_event_watch,
    evaluate_d1_level_watch,
    load_chart_watches,
    load_d1_event_watches,
    load_d1_level_watches,
    save_chart_watches,
    save_d1_event_watches,
    save_d1_level_watches,
    watch_is_stale,
)
from project_paths import (
    ALERT_CENTER_IGNORED_SYMBOLS_FILE,
    ALERT_CHART_WATCHES_FILE,
    ALERT_REVIEW_EVENTS_FILE,
    ALERT_REVIEW_PARKED_SYMBOLS_FILE,
    AUTO_POPULATE_PENDING_FILE,
    D1_EVENT_WATCHES_FILE,
    D1_LEVEL_WATCHES_FILE,
    get_local_setting,
    save_local_setting,
)
from review_events import record_review_event
from review_guidance import ORDERING_ANNOTATION_ONLY, AlertGuidance, ReviewGuide
from ui.panels import desk_layout
from ui.models.bounce import (
    AUTO_PICK_TAG,
    BounceAlert,
    CHART_WATCH_TAG,
    MANUAL_CHART_TAG,
    SYMBOL_RE,
    is_auto_pick_alert,
    is_chart_watch_alert,
    is_entry_assist_text,
)
from ui.widgets.alert_chart_review import AlertChartReview
from ui.widgets.alert_feed_item import AlertFeedItem
from ui.widgets.armed_watch_list import ArmedWatchList
from ui.widgets.entry_assist_board import EntryAssistBoard
from ui.widgets.focus_strength_board import FocusStrengthBoard
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

ALERT_SPLIT_KEY = "qt_alert_center_split_sizes_v2"
# The lower row of the alert column: tab stack | Focus strength board.
ALERT_TABS_SPLIT_KEY = "qt_alert_tabs_row_split_sizes_v1"

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
    armedWatchesChanged = Signal()  # any arm/disarm, so the inventory can redraw

    def __init__(
        self,
        focus_service=None,
        parent=None,
        *,
        ignored_symbols_path=None,
        parked_symbols_path=None,
        chart_watches_path=None,
        d1_level_watches_path=None,
        d1_event_watches_path=None,
        review_events_path=None,
        review_guide=None,
        auto_pick_pending_path=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.focus_service = focus_service
        self._bounce_service = None
        self._alerts: list[BounceAlert] = []
        self._d1_alerts: list[BounceAlert] = []
        self._review_queue: list[BounceAlert] = []
        self._current_review_alert: BounceAlert | None = None
        self._embedded_detail_enabled = True
        # Decision-log dwell tracking: which symbol the review pane is showing
        # and since when, so actions can report "considered for N ms".
        self._review_shown_symbol = ""
        self._review_shown_at: datetime | None = None
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
        # Day-scoped "parked" set: the trader armed a D1 alert on the chart
        # and then hit Skip - decision made, the armed alert does the
        # watching, so the chart queue stops re-showing the name. The FEED
        # still records its alerts; Focus names and armed-watch hits still
        # occupy the chart. Same file format/day scoping as ignored symbols.
        self._parked_symbols_path = (
            Path(parked_symbols_path)
            if parked_symbols_path is not None
            else (ALERT_REVIEW_PARKED_SYMBOLS_FILE if persist_ignored else None)
        )
        self._parked_symbols = (
            load_ignored_alert_symbols(
                self._parked_symbols_path,
                market_date=self._ignored_market_date,
            )
            if self._parked_symbols_path is not None
            else set()
        )
        # One-shot chart watches armed from the visual charts. Persisted to a
        # trading-day-scoped file so a GUI restart keeps them armed; only a
        # new session starts clean.
        self._chart_watches_path = (
            Path(chart_watches_path)
            if chart_watches_path is not None
            else (ALERT_CHART_WATCHES_FILE if persist_ignored else None)
        )
        self._chart_watches: list[ChartWatch] = (
            load_chart_watches(self._chart_watches_path)
            if self._chart_watches_path is not None
            else []
        )
        # Persistent D1 candle-level alerts: kept ACROSS sessions until they
        # flag, even for symbols outside the current scan set.
        self._d1_level_watches_path = (
            Path(d1_level_watches_path)
            if d1_level_watches_path is not None
            else (D1_LEVEL_WATCHES_FILE if persist_ignored else None)
        )
        self._d1_level_watches: list[D1LevelWatch] = (
            load_d1_level_watches(self._d1_level_watches_path)
            if self._d1_level_watches_path is not None
            else []
        )
        # Persistent D1 event watches (15EMA reject / 5d-20d extremes / SMA
        # break): same lifecycle as level watches, but the reference level is
        # re-derived from the daily store on every poll.
        self._d1_event_watches_path = (
            Path(d1_event_watches_path)
            if d1_event_watches_path is not None
            else (D1_EVENT_WATCHES_FILE if persist_ignored else None)
        )
        self._d1_event_watches: list[D1EventWatch] = (
            load_d1_event_watches(self._d1_event_watches_path)
            if self._d1_event_watches_path is not None
            else []
        )
        # Append-only decision log (review_events.py): every shown/skip/focus/
        # arm decision with its structured alert context. Gated exactly like
        # the other persistence paths so bare test panels never write it.
        self._review_events_path = (
            Path(review_events_path)
            if review_events_path is not None
            else (ALERT_REVIEW_EVENTS_FILE if persist_ignored else None)
        )
        # DESK-mode auto-populate approval queue: the engine stages its picks
        # in this file instead of writing the watchlists; the panel charts
        # each one with Approve/Pass verbs. Gated like the other persistence
        # paths so bare test panels never read or write the real queue.
        self._auto_pick_pending_path = (
            Path(auto_pick_pending_path)
            if auto_pick_pending_path is not None
            else (AUTO_POPULATE_PENDING_FILE if persist_ignored else None)
        )
        # (date, side, symbol) triples already turned into a review chart, so
        # a pick the trader skipped is not re-queued on every poll tick.
        self._auto_picks_enqueued: set[tuple[str, str, str]] = set()
        # Phase 2 guidance: scoreboard + AI policy -> queue ordering and
        # chart annotations (review_guidance.py). Advisory only; with no
        # documents on disk every score is 0 and the queue stays FIFO.
        self._review_guide = (
            review_guide
            if review_guide is not None
            else (ReviewGuide() if persist_ignored else ReviewGuide(None, None))
        )
        self._review_guidance: dict[str, AlertGuidance] = {}
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

        # The armed-watch inventory. Built before the tab bar that hosts it.
        self.armed_list = ArmedWatchList(self)
        self.armed_list.disarmWatchRequested.connect(self.disarm_chart_watch_for)
        self.armed_list.disarmLevelRequested.connect(self.disarm_d1_level_watch)
        self.armed_list.disarmEventRequested.connect(self.disarm_d1_event_watch)
        self.armed_list.symbolActivated.connect(self.chart_symbol)
        self.armedWatchesChanged.connect(self._refresh_armed_list)

        self.tabs = QTabWidget()
        self.tabs.addTab(feed_scroll, "Alerts")

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
        # D1 Focus used to hold a permanent full-width splitter section. It is
        # occasionally-useful rather than continuously-useful, so it becomes a
        # badged tab: still one click away with its unread count in peripheral
        # vision, but no longer spending 104-119px of chart height all day.
        # Deliberately NOT merged into the Alerts feed - D1 rows are untiered,
        # so alert_passes_min_tier would silently drop every tier flip the
        # moment the trader selects an S/A gate, and the separate 100-item
        # retention would be lost.
        self._d1_tab_index = self.tabs.addTab(d1_section, "D1 Focus")
        self.tabs.addTab(board_tab, "RS/RW Board")
        self._armed_tab_index = self.tabs.addTab(self.armed_list, "Armed")
        self._refresh_armed_list()
        self._d1_unread = 0
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._refresh_d1_tab_label()

        # The alert feed is a narrow list in a column-wide tab stack, so the
        # right half of that row was empty. A compact strongest/weakest board
        # with the trader's Focus names pinned on top fills it, and stays
        # visible on every tab - the RS/RW tab keeps the full entry-assist
        # board for the deep read. Same rrsSnapshotChanged payload the RRS
        # snapshot already consumes: no new service, thread, timer, or request.
        self.focus_strength = FocusStrengthBoard()
        if self.focus_service is not None:
            self.focus_strength.set_focus_service(self.focus_service)
        self.focus_strength.symbolActivated.connect(self._show_board_symbol_snapshot)

        self.tabs_row = QSplitter(Qt.Orientation.Horizontal)
        self.tabs_row.addWidget(self.tabs)
        self.tabs_row.addWidget(self.focus_strength)
        self.tabs_row.setStretchFactor(0, 3)
        self.tabs_row.setStretchFactor(1, 2)
        self.tabs_row.setChildrenCollapsible(False)
        # The tab stack hints wide enough to squeeze the board out entirely;
        # an explicit minimum takes precedence over minimumSizeHint and hands
        # the split back to the preset (same fix the desk columns needed).
        # 170 + the board's 170 stays inside the alert column's 360px floor, so
        # adding the board cannot force the whole desk column wider.
        self.tabs.setMinimumWidth(170)
        desk_layout.apply_saved_sizes(
            self.tabs_row, ALERT_TABS_SPLIT_KEY, desk_layout.ALERT_TABS_ROW_WEIGHTS
        )
        desk_layout.track_preset(
            self,
            self.tabs_row,
            ALERT_TABS_SPLIT_KEY,
            lambda _extent: desk_layout.ALERT_TABS_ROW_WEIGHTS,
        )
        desk_layout.persist_sizes(self, self.tabs_row, ALERT_TABS_SPLIT_KEY)

        self.detail_view = SetupDetailView(self)
        self.chart_review = AlertChartReview(self)
        self.chart_review.removeTodayRequested.connect(
            self._remove_review_alert_for_today
        )
        self.chart_review.focusRequested.connect(self._add_review_alert_to_focus)
        self.chart_review.skipRequested.connect(self._skip_review_alert)
        self.chart_review.crossFocusToggled.connect(self._toggle_review_cross_focus)
        self.chart_review.watchToggled.connect(self._toggle_chart_watch)
        self.chart_review.d1EventToggled.connect(self._toggle_d1_event_watch)
        self.chart_review.d1LevelAlertRequested.connect(self._arm_d1_level_from_chart)
        self.chart_review.symbolRequested.connect(self.chart_symbol)
        self.chart_review.levelArmRequested.connect(self._arm_level_from_dock)
        self.chart_review.levelDisarmRequested.connect(self._disarm_level_from_dock)

        # Armed chart watches are re-checked against the bot's cached M5 bars
        # every 30s (bars complete on 5-minute boundaries; this bounds the
        # trigger latency the same way the integrity/regime timers do).
        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(30_000)
        self._watch_timer.timeout.connect(self._poll_chart_watches)
        # The review chart rides the same tick: it renders when an alert
        # LANDS, and the trader often reaches it minutes later - without a
        # refresh the M5 pane is missing every bar since and the D1 preview
        # candle never moves. Cheap local reads; re-renders only on change.
        self._watch_timer.timeout.connect(self._refresh_review_chart)
        # DESK-mode auto picks ride the same 30s tick: the staging file is a
        # cheap local read and a new pick is not latency-critical.
        self._watch_timer.timeout.connect(self._poll_auto_pick_pending)
        self._watch_timer.start()
        # Persistent D1 level alerts poll less often: the daily-store reads
        # are mtime-cached and the evidence changes at most once per M5 bar.
        # The D1 event watches (derived-level alerts) ride the same tick.
        self._d1_watch_timer = QTimer(self)
        self._d1_watch_timer.setInterval(60_000)
        self._d1_watch_timer.timeout.connect(self._poll_d1_level_watches)
        self._d1_watch_timer.timeout.connect(self._poll_d1_event_watches)
        self._d1_watch_timer.start()

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(self.chart_review)
        splitter.addWidget(self.tabs_row)
        splitter.addWidget(self.detail_view)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)
        # The 5:2 above is not enough on its own. QSplitter honours a child's
        # size POLICY ahead of its stretch factor, and the tab stack is
        # Expanding while the chart pane is only Preferred - which measured as
        # an inverted [232, 455] at 1640x980, i.e. the charts got 1/3 of the
        # column the code intends them to own 5/7 of. Making the chart pane
        # Expanding lets the stretch factor actually apply, and an explicit
        # setSizes (there was none) pins the opening split.
        self.chart_review.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        splitter.setChildrenCollapsible(False)
        self.splitter = splitter
        # Versioned key: the child count changed 4 -> 3 when D1 Focus became a
        # tab, so a split saved by an older build must not be restored.
        desk_layout.apply_saved_sizes(
            splitter, ALERT_SPLIT_KEY, desk_layout.ALERT_COLUMN_WEIGHTS
        )
        desk_layout.track_preset(
            self,
            splitter,
            ALERT_SPLIT_KEY,
            lambda _extent: desk_layout.ALERT_COLUMN_WEIGHTS,
        )
        desk_layout.persist_sizes(self, splitter, ALERT_SPLIT_KEY)

        # One control row, no section header. The header's subtitle described a
        # D1 Focus feed sitting "below", which is now a tab, and this column's
        # scarcest resource is the vertical space the charts read in.
        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        show_label = QLabel("Show")
        show_label.setObjectName("MutedLabel")
        controls.addWidget(show_label)
        controls.addWidget(self.min_tier_input)
        controls.addWidget(self.sound_input)
        controls.addStretch(1)
        controls.addWidget(self.ignored_button)
        controls.addWidget(clear_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        layout.addLayout(controls)
        layout.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    def attach_service(self, service) -> None:
        self._bounce_service = service
        service.alertReceived.connect(self.add_alert)
        service.rrsSnapshotChanged.connect(self.rrs_snapshot.update_snapshot)
        service.rrsSnapshotChanged.connect(self.focus_strength.update_snapshot)
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
        # Count only genuine scan events toward the badge. `_d1_alerts` doubles
        # as the D1-Focus pin registry, so counting its length would make the
        # badge grow every time the trader pins a name themselves.
        if alert.tag != "d1_focus_pin" and not self._d1_tab_is_current():
            self._d1_unread += 1
            self._refresh_d1_tab_label()
        if self.sound_input.isChecked() and (is_ready_d1_alert(alert) or self._alert_is_focus(alert)):
            QApplication.beep()
        self._emit_feed_status()

    def _d1_tab_is_current(self) -> bool:
        return self.tabs.currentIndex() == self._d1_tab_index

    def _on_tab_changed(self, index: int) -> None:
        if index == self._d1_tab_index and self._d1_unread:
            self._d1_unread = 0
            self._refresh_d1_tab_label()

    def _refresh_d1_tab_label(self) -> None:
        label = f"D1 Focus ({self._d1_unread})" if self._d1_unread else "D1 Focus"
        self.tabs.setTabText(self._d1_tab_index, label)

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
        self._save_chart_watches()
        # Persistent D1 level alerts survive a feed clear by design.
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
            self._record_review_event(
                "favorite", alert=alert, detail={"on": False, "origin": origin}
            )
            message = f"Unfavorited {alert.symbol}: removed from focus picks."
        else:
            category = favorite_category_for_alert(alert)
            side = "short" if alert.side == "SHORT" else "long"
            self.focus_service.add(alert.symbol, side, category, origin=origin, context=alert.raw_text)
            self._record_review_event(
                "favorite",
                alert=alert,
                detail={"on": True, "origin": origin, "category": category},
            )
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
        self._record_review_event(
            "dislike",
            alert=alert,
            dwell_ms=self._review_dwell_ms(alert.symbol),
            detail={"reason": str(reason or "").strip()},
        )
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
        """Queue one visual review per symbol; refresh the active symbol live.

        Only real tickers get a chart. Summary/list messages can carry junk
        pseudo-symbols extracted from their prefix (e.g. "(BULLISH_STRONG)"
        from an old AUTO WATCHLIST line) - those must never occupy the
        review pane."""
        if not alert.symbol or not SYMBOL_RE.fullmatch(alert.symbol):
            return
        # Parked = the trader armed a D1 alert on this chart and skipped:
        # decision made for the day, so ordinary alerts stop re-occupying the
        # chart. The armed watch firing (chart_watch) is exactly what they
        # asked to see, and a Focus name is theirs - both still show.
        if (
            alert.symbol in self._parked_symbols
            and not is_chart_watch_alert(alert)
            and not self._alert_is_focus(alert)
        ):
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
        if is_chart_watch_alert(alert):
            # The trader armed this exact condition and is waiting on it, so it
            # goes to the FRONT. Appending sent the one chart they asked for to
            # the back of a queue that can be dozens deep.
            self._review_queue.insert(0, alert)
        else:
            # Guidance-ordered insertion: higher scores review sooner. Armed
            # chart-watch hits always stay ahead regardless of score, and
            # equal scores keep arrival order, so with no guidance documents
            # this degrades to the old FIFO exactly. While the Phase 0 gate
            # holds every queue_score is 0.0 and the order is that same FIFO
            # even when the scoreboard and policy are populated.
            score = self._queue_score(alert)
            index = len(self._review_queue)
            for position, queued in enumerate(self._review_queue):
                if is_chart_watch_alert(queued):
                    continue
                if self._queue_score(queued) < score:
                    index = position
                    break
            self._review_queue.insert(index, alert)
        if self._current_review_alert is None:
            self._advance_review_queue()
        else:
            self.chart_review.set_queued_count(len(self._review_queue))

    def _guidance_for(self, alert: BounceAlert) -> AlertGuidance:
        """Cached per-symbol guidance; a failed lookup is neutral, never fatal."""
        guidance = self._review_guidance.get(alert.symbol)
        if guidance is None:
            try:
                guidance = self._review_guide.guidance_for(alert)
            except Exception:
                guidance = AlertGuidance()
            self._review_guidance[alert.symbol] = guidance
        return guidance

    def _queue_score(self, alert: BounceAlert) -> float:
        """The only value allowed to influence review-queue position."""
        try:
            return self._review_guide.queue_score(self._guidance_for(alert))
        except Exception:
            return 0.0

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

    # ------------------------------------------------------------------
    # Decision logging: the training data for learning the trader's revealed
    # preferences. Best-effort by design - a logging failure must never cost
    # a click - and disabled whenever the panel runs on non-default stores.
    def _record_review_event(self, action: str, **kwargs) -> None:
        if self._review_events_path is None:
            return
        try:
            record_review_event(action, path=self._review_events_path, **kwargs)
        except Exception:
            pass

    def _review_dwell_ms(self, symbol: str) -> int | None:
        """How long the review pane showed this symbol before the action."""
        if self._review_shown_at is None or self._review_shown_symbol != symbol:
            return None
        return int((datetime.now() - self._review_shown_at).total_seconds() * 1000)

    def _current_bot(self):
        """The bounce service's live bot, or None - never raises."""
        if self._bounce_service is None:
            return None
        try:
            return self._bounce_service.current_bot()
        except Exception:
            return None

    def _refresh_review_chart(self) -> None:
        """30s tick: keep the visible review chart on current bars.

        Passes a fresh bot handle each tick (the service may have restarted
        since the alert rendered). The chart widget itself skips the re-render
        when nothing changed, so a quiet chart keeps its pan/zoom.
        """
        if self._current_review_alert is None:
            return
        try:
            self.chart_review.refresh_chart(bot=self._current_bot())
        except Exception:
            # Display refresh only - it must never break the watch tick that
            # shares this timer.
            pass

    def _render_current_review(self) -> None:
        alert = self._current_review_alert
        if alert is None:
            self._review_shown_symbol = ""
            self._review_shown_at = None
            self.chart_review.clear()
            return
        guidance = self._guidance_for(alert)
        if alert.symbol != self._review_shown_symbol:
            # The impression: a chart for this symbol was put in front of the
            # trader. Same-symbol refreshes keep the original dwell clock.
            self._review_shown_symbol = alert.symbol
            self._review_shown_at = datetime.now()
            detail = None
            if guidance.score or guidance.take_prob is not None:
                # Stamp what the guidance claimed at impression time, so a
                # later pass can measure whether the ordering/annotations
                # actually changed behavior (Phase 3 material). The ordering
                # mode rides along: an episode collected under the FIFO gate
                # is not evidence about preference-ordered delivery.
                detail = {
                    "guidance_score": guidance.score,
                    "take_prob": guidance.take_prob,
                    "queue_ordering": getattr(
                        self._review_guide, "ordering_mode", ORDERING_ANNOTATION_ONLY
                    ),
                }
            self._record_review_event(
                "shown", alert=alert, queue_len=len(self._review_queue), detail=detail
            )
        bot = self._current_bot()
        self.chart_review.set_alert(
            alert,
            bot=bot,
            focus_category=favorite_category_for_alert(alert),
            queued=len(self._review_queue),
            armed_kinds=self.armed_watch_kinds(alert.symbol),
            cross_active=self._review_cross_active(alert),
            armed_levels=self.armed_levels_for(alert.symbol),
            armed_d1_events=self.armed_d1_event_kinds(alert.symbol),
            guidance_text=guidance.summary_text(),
        )

    def _skip_review_alert(self, alert: BounceAlert) -> None:
        if (
            self._current_review_alert is None
            or self._current_review_alert.symbol != alert.symbol
        ):
            return
        # Skip after arming a D1 alert = "the alert does the watching now":
        # park the chart for the rest of the day (user rule 2026-07-29).
        parked = self._has_armed_d1_alerts(alert.symbol) and not self._alert_is_focus(
            alert
        )
        if parked:
            self._park_review_symbol(alert.symbol)
        self._record_review_event(
            "skip",
            alert=alert,
            dwell_ms=self._review_dwell_ms(alert.symbol),
            queue_len=len(self._review_queue),
            detail={"parked": True} if parked else None,
        )
        if parked:
            self.statusChanged.emit(
                f"Skipped {alert.symbol}: chart parked for today - its armed D1 "
                "alert still fires red, and adding it to Focus un-parks it."
            )
        else:
            self.statusChanged.emit(
                f"Skipped {alert.symbol} for now; its feed item remains available."
            )
        self._advance_review_queue()

    def _add_review_alert_to_focus(self, alert: BounceAlert) -> None:
        # Unified verb row (2026-07-31): the add button's "yes" for a DESK
        # auto pick is the watchlist, not Focus.
        if is_auto_pick_alert(alert):
            self._resolve_auto_pick(alert, True)
            return
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
        self._record_review_event(
            "add_focus",
            alert=alert,
            dwell_ms=self._review_dwell_ms(alert.symbol),
            queue_len=len(self._review_queue),
            detail={"category": category, "added": bool(added)},
        )
        bucket = "Swing" if category == "swing" else "M5"
        message = (
            f"★ {alert.symbol}: added to {bucket} Focus {side}s."
            if added
            else f"★ {alert.symbol}: already in Focus Picks."
        )
        self.statusChanged.emit(message)
        self._advance_review_queue()

    # ------------------------------------------------------------------
    # DESK-mode auto-populate picks: chart first, watchlist only on approval.
    def _poll_auto_pick_pending(self) -> None:
        """Turn newly staged auto-populate picks into review-queue charts."""
        if self._auto_pick_pending_path is None:
            return
        try:
            from autopilot_core import load_auto_populate_pending_picks

            payload = load_auto_populate_pending_picks(self._auto_pick_pending_path)
        except Exception:
            return
        day = str(payload.get("date") or "")
        for side_key, side_label in (("long", "LONG"), ("short", "SHORT")):
            entries = payload.get("pending", {}).get(side_key) or {}
            for symbol, entry in entries.items():
                symbol = str(symbol or "").strip().upper()
                if not symbol or not SYMBOL_RE.fullmatch(symbol):
                    continue
                key = (day, side_key, symbol)
                if key in self._auto_picks_enqueued:
                    continue
                self._auto_picks_enqueued.add(key)
                entry = entry if isinstance(entry, dict) else {}
                reason = str(entry.get("reason") or "auto-populate pick")
                score = entry.get("score")
                trigger = f"Auto pick ({side_label.lower()}): {reason}"
                if score:
                    trigger += f" · score {float(score):.2f}"
                self._enqueue_review_alert(
                    BounceAlert(
                        time_text=str(entry.get("staged_at") or "")
                        or datetime.now().strftime("%H:%M:%S"),
                        symbol=symbol,
                        side=side_label,
                        trigger=trigger,
                        timeframe="M5",
                        tag=AUTO_PICK_TAG,
                        raw_text=f"AUTO PICK {side_label} {symbol}: {reason}",
                        payload={"auto_pick": dict(entry), "auto_pick_side": side_key},
                    )
                )

    def _resolve_auto_pick(self, alert: BounceAlert, approved: bool) -> None:
        if (
            self._current_review_alert is None
            or self._current_review_alert.symbol != alert.symbol
        ):
            return
        result = self._record_auto_pick_verdict(alert, approved)
        symbol = alert.symbol
        if approved:
            if result.get("written"):
                side_word = "shorts" if alert.side == "SHORT" else "longs"
                self.statusChanged.emit(
                    f"✓ {symbol}: approved auto pick - added to the {side_word} "
                    "watchlist (BounceBot picks it up on its next M5 cycle)."
                )
            elif result.get("already_listed"):
                self.statusChanged.emit(
                    f"✓ {symbol}: approved auto pick - already on a watchlist, nothing to add."
                )
            else:
                self.statusChanged.emit(
                    f"{symbol}: approval recorded, but the watchlist write was "
                    "refused on this machine - check the log."
                )
        else:
            self.statusChanged.emit(
                f"✕ {symbol}: not today - this auto pick will not be proposed "
                "again this session; watchlists untouched."
            )
        self._advance_review_queue()

    def _record_auto_pick_verdict(self, alert: BounceAlert, approved: bool) -> dict:
        """File the verdict in the staging store + the decision log."""
        result: dict = {}
        if self._auto_pick_pending_path is not None:
            try:
                from autopilot_core import resolve_auto_populate_pick

                result = resolve_auto_populate_pick(
                    alert.symbol,
                    str(alert.payload.get("auto_pick_side") or alert.side),
                    approved,
                    pending_path=self._auto_pick_pending_path,
                )
            except Exception:
                result = {}
        self._record_review_event(
            "auto_pick_approve" if approved else "auto_pick_pass",
            alert=alert,
            dwell_ms=self._review_dwell_ms(alert.symbol),
            queue_len=len(self._review_queue),
            detail={
                "written": bool(result.get("written")),
                "already_listed": bool(result.get("already_listed")),
                "auto_pick": alert.payload.get("auto_pick") or None,
            },
        )
        return result

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
            self._record_review_event(
                "toggle_d1_focus",
                symbol=symbol,
                side=side,
                detail={"on": False, "origin": origin},
            )
            self.statusChanged.emit(
                f"{symbol}: removed from Swing Focus and unpinned from the D1 Focus feed."
            )
            self._refresh_review_cross_state()
            return False
        if self.focus_service is not None:
            self.focus_service.add(symbol, focus_side, "swing", origin=origin, context=context)
        self._record_review_event(
            "toggle_d1_focus",
            symbol=symbol,
            side=side,
            detail={"on": True, "origin": origin},
        )
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
            self._record_review_event(
                "toggle_m5_focus",
                symbol=symbol,
                side=side,
                detail={"on": False, "origin": origin},
            )
            self.statusChanged.emit(f"{symbol}: removed from M5 Focus {focus_side}s.")
            self._refresh_review_cross_state()
            return False
        self.focus_service.add(symbol, focus_side, "m5", origin=origin, context=context)
        self._record_review_event(
            "toggle_m5_focus",
            symbol=symbol,
            side=side,
            detail={"on": True, "origin": origin},
        )
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
        self._save_chart_watches()
        self._refresh_review_armed_kinds()
        self.armedWatchesChanged.emit()
        current = self._current_review_alert
        self._record_review_event(
            "arm_watch",
            alert=current if current is not None and current.symbol == symbol else None,
            symbol=symbol,
            side=side,
            dwell_ms=self._review_dwell_ms(symbol),
            detail={"kind": kind, "baseline": watch.baseline},
        )
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
        self._save_chart_watches()
        self._refresh_review_armed_kinds()
        self.armedWatchesChanged.emit()
        self._record_review_event(
            "disarm_watch", symbol=symbol, detail={"kind": kind}
        )
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
        live = []
        for watch in self._chart_watches:
            if watch_is_stale(watch, now=moment):
                # The third way an armed watch ends (besides firing and an
                # explicit disarm) - without this the decision log could not
                # tell them apart.
                self._record_review_event(
                    "watch_expired",
                    symbol=watch.symbol,
                    side=watch.side,
                    detail={"kind": watch.kind},
                )
            else:
                live.append(watch)
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
            self._record_review_event(
                "watch_fired",
                symbol=hit.watch.symbol,
                side=getattr(hit, "resolved_side", "") or hit.watch.side,
                detail={"kind": hit.watch.kind, "message": str(hit.message or "")},
            )
            self.add_alert(self._chart_watch_alert(hit, moment))
        if len(remaining) != before:
            self._save_chart_watches()
            self._refresh_review_armed_kinds()

    def _chart_watch_alert(self, hit, moment: datetime) -> BounceAlert:
        watch = hit.watch
        resolved = str(getattr(hit, "resolved_side", "") or "").upper()
        side = str(getattr(watch, "side", "") or "")
        if side not in ("LONG", "SHORT"):
            side = resolved if resolved in ("LONG", "SHORT") else "WATCH"
        kind = watch.kind
        trigger = hit.message
        note = self._tracker_note_for(watch, hit, moment)
        if note:
            trigger = f"{trigger} | {note}"
        return BounceAlert(
            time_text=moment.strftime("%H:%M:%S"),
            symbol=watch.symbol,
            side=side,
            trigger=trigger,
            timeframe="D1" if (kind in D1_LEVEL_KINDS or kind in D1_EVENT_KINDS) else "M5",
            tag=CHART_WATCH_TAG,
            raw_text=f"CHART WATCH {watch.symbol} ({side}): {trigger}",
            payload={
                "chart_watch_kind": kind,
                "armed_at": watch.armed_at.isoformat(),
                "source_text": getattr(watch, "source_text", "")
                or getattr(watch, "candle_date", ""),
            },
        )

    def _tracker_note_for(self, watch, hit, moment: datetime) -> str:
        """Day-trade-tracker context stamped onto σ-band triggers: the
        measured segment stats plus whether we're inside the family's prime
        production window. Read-only decision support - never changes tiering."""
        if getattr(watch, "kind", "") != "band_bounce":
            return ""
        resolved = str(getattr(hit, "resolved_side", "") or "").lower()
        segment_type = BAND_BOUNCE_TRACKER_TYPES.get(resolved)
        if not segment_type:
            return ""
        bucket = ""
        stats = ""
        try:
            from bounce_bot_lib.learning import load_bounce_learning_state, time_bucket_for

            bucket = str(time_bucket_for(moment) or "")
            state = load_bounce_learning_state() or {}
            entry = ((state.get("segments") or {}).get("bounce_type") or {}).get(
                f"{resolved}|{segment_type}"
            )
            if entry:
                stats = f"{entry['avg_close_r']:+.2f}R n={entry['sample_count']}"
        except Exception:
            pass
        window = ""
        if bucket:
            window = (
                "prime window"
                if bucket in BAND_BOUNCE_PRIME_BUCKETS
                else f"off-window ({bucket})"
            )
        parts = [part for part in (f"tracker {segment_type} {stats}" if stats else "", window) if part]
        return "; ".join(parts)

    def _save_chart_watches(self) -> None:
        if self._chart_watches_path is None:
            return
        try:
            save_chart_watches(self._chart_watches, self._chart_watches_path)
        except OSError:
            pass

    def _save_d1_level_watches(self) -> None:
        if self._d1_level_watches_path is None:
            return
        try:
            save_d1_level_watches(self._d1_level_watches, self._d1_level_watches_path)
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Persistent D1 candle-level alerts: armed by clicking a D1 chart candle,
    # kept across sessions until they flag (symbol need not be scanned).
    def _arm_d1_level_from_chart(
        self, symbol: str, direction: str, level: float, candle_date: str
    ) -> None:
        self.arm_d1_level_watch(
            symbol, direction, level, candle_date=candle_date, fill_source="candle"
        )

    def arm_d1_level_watch(
        self,
        symbol: str,
        direction: str,
        level: float,
        *,
        candle_date: str = "",
        fill_source: str = "",
    ) -> bool:
        symbol = str(symbol or "").strip().upper()
        try:
            level = float(level)
        except (TypeError, ValueError):
            return False
        if not symbol or direction not in ("above", "below") or not level > 0:
            return False
        for watch in self._d1_level_watches:
            if (
                watch.symbol == symbol
                and watch.direction == direction
                and abs(watch.level - level) < 1e-6
            ):
                self.statusChanged.emit(
                    f"{symbol}: D1 level alert break {direction} {level:.2f} already armed."
                )
                return False
        self._d1_level_watches.append(
            D1LevelWatch(
                symbol=symbol,
                direction=direction,
                level=level,
                armed_at=datetime.now(),
                candle_date=str(candle_date or ""),
            )
        )
        self._save_d1_level_watches()
        self.armedWatchesChanged.emit()
        current = self._current_review_alert
        self._record_review_event(
            "arm_level",
            alert=current if current is not None and current.symbol == symbol else None,
            symbol=symbol,
            dwell_ms=self._review_dwell_ms(symbol),
            detail={
                "direction": direction,
                "level": level,
                "candle_date": str(candle_date or ""),
                "fill_source": str(fill_source or ""),
            },
        )
        origin = f" (from the {candle_date} candle)" if candle_date else ""
        self.statusChanged.emit(
            f"{symbol}: D1 level alert armed - break {direction} {level:.2f}{origin}. "
            "It stays on across sessions until it flags, even while the symbol "
            "is not being scanned."
        )
        return True

    def chart_symbol(self, symbol: str) -> bool:
        """Put any symbol on the big chart on demand.

        The review pane previously only ever showed what the alert queue handed
        it, so on a quiet tape it sat on "Waiting for the next ticker alert"
        with no way to look at a name. A typed symbol is charted immediately,
        even if it has never alerted and is not in the scan set.

        Typing a symbol also un-ignores it: "Remove for today" would otherwise
        make it silently un-chartable for the rest of the session, which reads
        as the box being broken.
        """
        symbol = str(symbol or "").strip().upper()
        if not symbol or not SYMBOL_RE.fullmatch(symbol):
            self.statusChanged.emit(f"{symbol or 'That'} is not a valid ticker.")
            return False
        if symbol in self._ignored_symbols:
            self._restore_ignored_symbol(symbol)
        # Typing a parked symbol is re-engaging with it: un-park so its
        # alerts can occupy the chart again.
        self._unpark_review_symbol(symbol)
        alert = BounceAlert(
            time_text=datetime.now().strftime("%H:%M:%S"),
            symbol=symbol,
            side="WATCH",
            trigger="Charted on demand",
            tag=MANUAL_CHART_TAG,
            raw_text=f"MANUAL CHART {symbol}",
        )
        # Straight to the review pane; never into the alert feed, which is a
        # record of what the scanner said, not of what was looked at.
        self._select_review_alert(alert)
        self.statusChanged.emit(f"{symbol}: charted on demand.")
        return True

    def _arm_level_from_dock(self, symbol: str, direction: str, level: float) -> None:
        # The arm bar remembers which quick-fill button (vwap/+1σ/hod/...) or
        # chart click produced the price - the "what do I arm levels off"
        # half of the decision log.
        fill_source = ""
        try:
            fill_source = self.chart_review.arm_bar.last_fill_source()
        except Exception:
            pass
        self.arm_d1_level_watch(symbol, direction, level, fill_source=fill_source)

    def _disarm_level_from_dock(self, symbol: str, direction: str, level: float) -> None:
        self.disarm_d1_level_watch(symbol, direction, level)

    def armed_levels_for(self, symbol: str) -> list:
        symbol = str(symbol or "").strip().upper()
        return [watch for watch in self._d1_level_watches if watch.symbol == symbol]

    def _refresh_armed_list(self) -> None:
        self.armed_list.set_watches(
            self._chart_watches,
            self._d1_level_watches,
            d1_events=self._d1_event_watches,
            has_m5_bars=lambda symbol: bool(self._m5_bars_for(symbol)),
        )
        current = self._current_review_alert
        if current is not None:
            self.chart_review.set_armed_levels(self.armed_levels_for(current.symbol))

    def disarm_d1_level_watch(self, symbol: str, direction: str, level: float) -> bool:
        """Cancel a persistent D1 level alert. Returns True if one was removed.

        D1 level watches survive across sessions and are only otherwise removed
        by firing, so without this the only way to cancel one was to hand-edit
        d1_level_watches.json.
        """
        symbol = str(symbol or "").strip().upper()
        try:
            level = float(level)
        except (TypeError, ValueError):
            return False
        remaining = [
            watch
            for watch in self._d1_level_watches
            if not (
                watch.symbol == symbol
                and watch.direction == direction
                and abs(watch.level - level) < 1e-6
            )
        ]
        if len(remaining) == len(self._d1_level_watches):
            return False
        self._d1_level_watches = remaining
        self._save_d1_level_watches()
        self.armedWatchesChanged.emit()
        self._record_review_event(
            "disarm_level",
            symbol=symbol,
            detail={"direction": direction, "level": level},
        )
        self.statusChanged.emit(
            f"{symbol}: D1 level alert break {direction} {level:.2f} disarmed."
        )
        return True

    def armed_watches(self) -> list[ChartWatch]:
        """Every armed session watch, for the armed-watch inventory UI."""
        return list(self._chart_watches)

    def armed_d1_levels(self) -> list[D1LevelWatch]:
        """Every armed persistent level alert, for the inventory UI."""
        return list(self._d1_level_watches)

    def _d1_bars_for(self, symbol: str) -> list:
        try:
            import chart_snapshot

            return chart_snapshot.load_d1_bars(symbol) or []
        except Exception:
            return []

    def _poll_d1_level_watches(self, now: datetime | None = None) -> None:
        if not self._d1_level_watches:
            return
        moment = now or datetime.now()
        remaining: list[D1LevelWatch] = []
        triggered = []
        for watch in self._d1_level_watches:
            if watch.symbol in self._ignored_symbols:
                # Removed-for-today symbols defer; the watch survives the day.
                remaining.append(watch)
                continue
            hit = None
            m5_bars = self._m5_bars_for(watch.symbol)
            d1_bars = self._d1_bars_for(watch.symbol)
            if m5_bars or d1_bars:
                try:
                    hit = evaluate_d1_level_watch(watch, m5_bars, d1_bars, now=moment)
                except Exception:
                    hit = None
            if hit is None:
                remaining.append(watch)
            else:
                triggered.append(hit)
        self._d1_level_watches = remaining
        if triggered:
            self._save_d1_level_watches()
        for hit in triggered:
            self._record_review_event(
                "level_fired",
                symbol=hit.watch.symbol,
                side=getattr(hit, "resolved_side", "") or "",
                detail={
                    "direction": hit.watch.direction,
                    "level": hit.watch.level,
                    "message": str(hit.message or ""),
                },
            )
            self.add_alert(self._chart_watch_alert(hit, moment))

    # ------------------------------------------------------------------
    # Persistent D1 event watches: derived-level alerts (15EMA reject, new
    # 5d/20d extreme, SMA break) armed from the dock's D1 row. Same rails as
    # the level watches - 60s poll, red chart-watch alert, one-shot retire.
    def armed_d1_event_kinds(self, symbol: str) -> set[str]:
        symbol = str(symbol or "").strip().upper()
        return {watch.kind for watch in self._d1_event_watches if watch.symbol == symbol}

    def _save_d1_event_watches(self) -> None:
        if self._d1_event_watches_path is not None:
            try:
                save_d1_event_watches(self._d1_event_watches, self._d1_event_watches_path)
            except Exception:
                pass

    def _toggle_d1_event_watch(self, alert: BounceAlert, kind: str) -> None:
        if alert is None or not alert.symbol:
            return
        if kind in self.armed_d1_event_kinds(alert.symbol):
            self.disarm_d1_event_watch(alert.symbol, kind)
        else:
            self.arm_d1_event_watch(alert.symbol, kind)

    def arm_d1_event_watch(self, symbol: str, kind: str) -> bool:
        symbol = str(symbol or "").strip().upper()
        if not symbol or kind not in D1_EVENT_KINDS:
            return False
        label = D1_EVENT_KINDS[kind]
        if kind in self.armed_d1_event_kinds(symbol):
            self.statusChanged.emit(f"{symbol}: {label} alert already armed.")
            return False
        self._d1_event_watches.append(
            D1EventWatch(symbol=symbol, kind=kind, armed_at=datetime.now())
        )
        self._save_d1_event_watches()
        self._refresh_review_armed_kinds()
        self.armedWatchesChanged.emit()
        current = self._current_review_alert
        self._record_review_event(
            "arm_d1_event",
            alert=current if current is not None and current.symbol == symbol else None,
            symbol=symbol,
            dwell_ms=self._review_dwell_ms(symbol),
            detail={"kind": kind},
        )
        self.statusChanged.emit(
            f"{symbol}: {label} alert armed - it stays on across sessions "
            "until it fires."
        )
        return True

    def disarm_d1_event_watch(self, symbol: str, kind: str) -> bool:
        symbol = str(symbol or "").strip().upper()
        remaining = [
            watch
            for watch in self._d1_event_watches
            if not (watch.symbol == symbol and watch.kind == kind)
        ]
        if len(remaining) == len(self._d1_event_watches):
            return False
        self._d1_event_watches = remaining
        self._save_d1_event_watches()
        self._refresh_review_armed_kinds()
        self.armedWatchesChanged.emit()
        self._record_review_event(
            "disarm_d1_event", symbol=symbol, detail={"kind": kind}
        )
        self.statusChanged.emit(
            f"{symbol}: {D1_EVENT_KINDS.get(kind, kind)} alert disarmed."
        )
        return True

    def _poll_d1_event_watches(self, now: datetime | None = None) -> None:
        if not self._d1_event_watches:
            return
        moment = now or datetime.now()
        remaining: list[D1EventWatch] = []
        triggered = []
        for watch in self._d1_event_watches:
            if watch.symbol in self._ignored_symbols:
                # Removed-for-today symbols defer; the watch survives the day.
                remaining.append(watch)
                continue
            hit = None
            m5_bars = self._m5_bars_for(watch.symbol)
            d1_bars = self._d1_bars_for(watch.symbol)
            if d1_bars:
                # Unlike a frozen price level, every event kind needs the
                # daily store for its reference; without it there is nothing
                # to measure against yet and the watch just waits.
                avwape_anchor = None
                if watch.kind.startswith("avwape_"):
                    try:
                        import chart_snapshot

                        avwape_anchor = chart_snapshot.earnings_anchor_date(watch.symbol)
                    except Exception:
                        avwape_anchor = None
                try:
                    hit = evaluate_d1_event_watch(
                        watch, m5_bars, d1_bars, now=moment, avwape_anchor=avwape_anchor
                    )
                except Exception:
                    hit = None
            if hit is None:
                remaining.append(watch)
            else:
                triggered.append(hit)
        self._d1_event_watches = remaining
        if triggered:
            self._save_d1_event_watches()
            self._refresh_review_armed_kinds()
            self.armedWatchesChanged.emit()
        for hit in triggered:
            self._record_review_event(
                "d1_event_fired",
                symbol=hit.watch.symbol,
                side=getattr(hit, "resolved_side", "") or "",
                detail={
                    "kind": hit.watch.kind,
                    "message": str(hit.message or ""),
                },
            )
            self.add_alert(self._chart_watch_alert(hit, moment))

    def _refresh_review_armed_kinds(self) -> None:
        current = self._current_review_alert
        if current is not None:
            self.chart_review.set_armed_kinds(self.armed_watch_kinds(current.symbol))
            self.chart_review.set_armed_d1_events(
                self.armed_d1_event_kinds(current.symbol)
            )

    def _remove_review_alert_for_today(self, alert: BounceAlert) -> None:
        """Drop a name from today's visual processing without changing scans."""
        if not alert.symbol:
            return
        # Unified verb row (2026-07-31): "✕ Not today" on an auto pick is the
        # decline verdict - retire the proposal for the day, advance the
        # queue, and leave the symbol's ordinary alerting untouched.
        if is_auto_pick_alert(alert):
            self._resolve_auto_pick(alert, False)
            return
        self._record_review_event(
            "remove_today",
            alert=alert,
            dwell_ms=self._review_dwell_ms(alert.symbol),
            queue_len=len(self._review_queue),
        )
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
        self._review_guidance.pop(symbol, None)
        self._chart_watches = [
            watch for watch in self._chart_watches if watch.symbol != symbol
        ]
        self._save_chart_watches()
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
        self._record_review_event("restore_today", symbol=symbol)
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
        # Parked symbols are day-scoped exactly like ignored ones: the file's
        # stale market_date loads as an empty set on the new day.
        self._parked_symbols = (
            load_ignored_alert_symbols(
                self._parked_symbols_path,
                market_date=current,
            )
            if self._parked_symbols_path is not None
            else set()
        )
        self._refresh_ignored_button()

    def _park_review_symbol(self, symbol: str) -> None:
        """Keep a symbol's chart out of the review queue for the day."""
        self._refresh_ignored_market_date()
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return
        self._parked_symbols.add(symbol)
        if self._parked_symbols_path is not None:
            try:
                self._parked_symbols = save_ignored_alert_symbols(
                    self._parked_symbols,
                    self._parked_symbols_path,
                    market_date=self._ignored_market_date,
                )
            except Exception:
                pass

    def _unpark_review_symbol(self, symbol: str) -> None:
        symbol = str(symbol or "").strip().upper()
        if symbol not in self._parked_symbols:
            return
        self._parked_symbols.discard(symbol)
        if self._parked_symbols_path is not None:
            try:
                self._parked_symbols = save_ignored_alert_symbols(
                    self._parked_symbols,
                    self._parked_symbols_path,
                    market_date=self._ignored_market_date,
                )
            except Exception:
                pass

    def _has_armed_d1_alerts(self, symbol: str) -> bool:
        symbol = str(symbol or "").strip().upper()
        return any(watch.symbol == symbol for watch in self._d1_level_watches) or any(
            watch.symbol == symbol for watch in self._d1_event_watches
        )

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
