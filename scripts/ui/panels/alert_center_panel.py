from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from collections.abc import Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut
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

from alert_review_state import (
    load_day_scoped_flags,
    load_ignored_alert_symbols,
    save_day_scoped_flags,
    save_ignored_alert_symbols,
)
from chart_watch import (
    ANY_BOUNCE_KINDS,
    AnyBounceWatch,
    BAND_BOUNCE_PRIME_BUCKETS,
    BAND_BOUNCE_TRACKER_TYPES,
    ChartWatch,
    D1EventWatch,
    D1LevelWatch,
    D1_EVENT_KINDS,
    D1_LEVEL_KINDS,
    D1_PULLBACK_KINDS,
    WATCH_KINDS,
    any_bounce_levels,
    arm_chart_watch,
    d1_event_levels,
    evaluate_any_bounce_watch,
    evaluate_chart_watch,
    evaluate_d1_event_watch,
    evaluate_d1_level_watch,
    completed_session_bars,
    load_any_bounce_watches,
    load_chart_watches,
    load_d1_event_watches,
    load_d1_level_watches,
    save_any_bounce_watches,
    save_chart_watches,
    save_d1_event_watches,
    save_d1_level_watches,
    watch_is_stale,
)
import focus_adoption_gate
import regime_pause_hold
from regime_pause_focus import day_bias, focus_side_for
import sma_trend_gate
from prev_day_gate import (
    CLOSED as PREV_DAY_CLOSED,
    OPEN as PREV_DAY_BREAK_OPEN,
    UNKNOWN as PREV_DAY_UNKNOWN,
    prev_day_break_state,
    prev_session_extremes,
)
from project_paths import (
    ALERT_CENTER_IGNORED_SYMBOLS_FILE,
    ALERT_CHART_WATCHES_FILE,
    ALERT_REVIEW_EVENTS_FILE,
    ALERT_REVIEW_PARKED_SYMBOLS_FILE,
    AUTO_POPULATE_PENDING_FILE,
    FOCUS_D1_FLAGS_FILE,
    ANY_BOUNCE_WATCHES_FILE,
    D1_EVENT_WATCHES_FILE,
    D1_LEVEL_WATCHES_FILE,
    get_local_setting,
    save_local_settings,
)
from alert_repetition import ACTION_DIGEST, ACTION_FOLD
from review_events import record_review_event
from review_guidance import ORDERING_ANNOTATION_ONLY, AlertGuidance, ReviewGuide
from ui import theme
from ui.panels import desk_layout
from ui.timer_utils import SignalCoalescer, start_staggered
from ui.models.bounce import (
    AUTO_PICK_TAG,
    BounceAlert,
    CHART_WATCH_TAG,
    FOCUS_D1_EVENT_TAG,
    FOCUS_FADED_TAG,
    FOCUS_REVIEW_TAG,
    MANUAL_CHART_TAG,
    SYMBOL_RE,
    is_auto_pick_alert,
    is_chart_watch_alert,
    is_entry_assist_text,
    is_regime_pause_alert,
    REGIME_PAUSE_TRIGGER_PREFIX,
)
from ui.widgets.alert_chart_review import AlertChartReview
from ui.widgets.alert_feed_item import AlertFeedItem
from ui.widgets.armed_watch_list import ArmedWatchList
from ui.widgets.collapsible_section import CollapsibleSection
from ui.widgets.entry_assist_board import EntryAssistBoard
from ui.widgets.focus_strength_board import FocusStrengthBoard
from ui.widgets.rrs_snapshot import RrsSnapshotWidget
from ui.widgets.section_header import SectionHeader
from ui.widgets.setup_detail_view import SetupDetailView

if TYPE_CHECKING:  # pragma: no cover - annotation only, never imported at runtime
    # `attach_strength_board` imports the real class inside the method, so the
    # board's module is loaded only when a host actually installs one. The
    # annotation on `self.strength_board` still has to name it, and with
    # `from __future__ import annotations` that name is never evaluated - which
    # is exactly why it went unnoticed as an undefined name until ruff was first
    # run against this tree on 2026-08-31.
    from ui.panels.strength_board_panel import StrengthBoardPanel

#: How many (symbol, sessions) M5 materializations to keep. The poll set is
#: the Focus list plus whatever is armed - ~105 symbols on 2026-08-31 - and
#: each entry is one already-built list, so this is comfortably inside it.
#: Mirrors `chart_data_service._MAX_MATERIALIZED_SYMBOLS`.
M5_BAR_DICT_CACHE_LIMIT = 240

_TIER_RE = re.compile(r"\[([SABCD])-TIER\]", re.IGNORECASE)

MIN_TIER_CHOICES = (
    ("All alerts", "all"),
    ("B tier and above", "B"),
    ("A tier and above", "A"),
    ("S tier / PROVEN only", "S"),
)
_TIER_RANK = {"S": 4, "A": 3, "B": 2, "C": 1, "D": 0}
MAX_FEED_ITEMS = 250
MAX_D1_FEED_ITEMS = 100

ALERT_SPLIT_KEY = "qt_alert_center_split_sizes_v2"
# The lower row of the alert column: tab stack | Focus strength board.
ALERT_TABS_SPLIT_KEY = "qt_alert_tabs_row_split_sizes_v1"

#: How long a FAILED flip re-verification waits before trying again (R2.2).
#: The poll itself runs every 30 s; retrying on every one of those would hammer
#: a feed that has already failed, and the trader gains nothing from a second
#: attempt a few seconds after the first.
FLIP_REVERIFY_RETRY_SECONDS = 60
#: How many consecutive failures the fast path attempts before giving up. It is
#: bounded rather than endless because giving up is SAFE: the flip barrier keeps
#: refusing every pre-flip verdict, and the ordinary 30-minute staging refresh
#: re-stamps the queue with post-flip verdicts that drain normally. Five
#: attempts is five minutes of trying before falling back to that slower path.
FLIP_REVERIFY_MAX_ATTEMPTS = 5

#: How many staged picks one DESK drain cycle may adopt (trader-approved
#: 2026-08-31: "cap the auto-adopt batch and slow the redraws").
#:
#: PACING, never policy. Nothing here decides differently about a pick - the
#: freshness gate, the flip barrier, the ownership marker and AWAY/EVENING's
#: refusal are all upstream of it and untouched. What is left over stays
#: STAGED and is adopted by the next cycle of the same 30-second timer, so a
#: 45-pick morning finishes inside ~2.5 minutes instead of freezing the desk
#: for 13 seconds. **No pick is ever dropped**; a cap that withheld one would
#: be the suppression field this chain deliberately does not have.
#:
#: The measurement behind the number (2026-08-31 ui_stalls.jsonl): 45 adoptions
#: at ~300 ms apart cost 13.5 s of solid GUI-thread work, and the 15.2 s stall
#: charged to the Focus board landed at the end of it. Ten is what fits inside
#: one 30-second tick with the coalesced redraws and room to spare.
AUTO_ADOPT_BATCH_LIMIT = 10

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


def _bar_close(bar: object) -> float | None:
    try:
        return float(bar["close"])  # type: ignore[index]
    except (KeyError, TypeError, ValueError):
        return None


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


# Short labels for the scanner's own D1 focus alerts, so the hourly phone push
# reads "NVDA bucket upgrade" rather than a 200-character raw alert line.
_D1_PUSH_LABELS = {
    "MASTER_AVWAP_D1_ZONE": "D1 zone",
    "MASTER_AVWAP_D1_BUCKET_UPGRADE": "bucket upgrade",
    "MASTER_AVWAP_D1_TIER_FLIP": "tier flip",
}


def d1_push_event(alert: BounceAlert) -> dict[str, str] | None:
    """What the hourly D1 phone push should carry for this alert, if anything.

    One classifier, here rather than in the Auto Pilot service, because this
    module already owns which D1 alerts are actionable and which are developing
    research. The phone therefore names exactly the events the D1 Focus feed
    shows, and the two can never drift apart.
    """
    symbol = str(getattr(alert, "symbol", "") or "").strip().upper()
    if not symbol:
        return None
    kind = ""
    payload = getattr(alert, "payload", None)
    if isinstance(payload, dict):
        kind = str(payload.get("chart_watch_kind") or payload.get("focus_d1_kind") or "")
    label = ""
    if kind:
        # Armed D1 levels and D1 event watches: the trader asked for exactly
        # this condition, so it belongs on the phone by definition.
        label = D1_LEVEL_KINDS.get(kind) or D1_EVENT_KINDS.get(kind) or ""
        if not label:
            return None
    elif getattr(alert, "is_d1", False) and is_ready_d1_alert(alert):
        label = _D1_PUSH_LABELS.get(_d1_alert_prefix(alert), "D1 event")
    else:
        return None
    return {
        "symbol": symbol,
        "label": label,
        "time_text": str(getattr(alert, "time_text", "") or ""),
    }


def extract_alert_tier(alert: BounceAlert) -> str:
    match = _TIER_RE.search(str(alert.raw_text or ""))
    return match.group(1).upper() if match else ""


# "BANGER" was retired 2026-09-01 (trader: "We can probably remove this because
# idk what it is"). It was only ever a literal token match against alert text,
# and nothing in the tree ever emitted the token: 0 of 8,818 recorded review
# rows carried banger=True. PROVEN is the top alert class and is untouched.


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

    PROVEN alerts always pass (they are the sit-back-and-wait trades), and so
    does entry-assist output — the trader clicked a button asking for it, so it
    must never be swallowed by the tier gate. Chart-watch hits pass for the
    same reason: the trader armed that exact condition from the M5 chart.
    Untiered alerts (regime notes, pause-watch summaries) pass everything
    except the S-only mode, where only PROVEN/S-tier remain.
    """
    if mode in ("", "all"):
        return True
    if (
        is_proven_alert(alert)
        or is_entry_assist_alert(alert)
        or is_chart_watch_alert(alert)
    ):
        return True
    tier = extract_alert_tier(alert)
    if not tier:
        return mode != "S"
    return _TIER_RANK.get(tier, 0) >= _TIER_RANK.get(mode, 0)


def alert_is_loud(alert: BounceAlert) -> bool:
    """Alerts worth a sound: proven configs, S/A tiers, ready D1, and
    chart-watch hits (the trader armed the exact condition and is
    waiting on it)."""
    return (
        is_proven_alert(alert)
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
        self.feed_item = feed_item
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(feed_item)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_repeat_count(self, count: int, *, latest_trigger: str = "") -> None:
        """Forward R4 section 6.3's fold to the row this wrapper contains.

        This class wraps an ``AlertFeedItem`` rather than subclassing it, so
        the repeat badge has to be forwarded explicitly - the feed only ever
        holds wrappers, so without this the fold silently fails over to a new
        row and the whole control does nothing.
        """
        self.feed_item.set_repeat_count(count, latest_trigger=latest_trigger)

    @property
    def repeat_badge(self):
        return self.feed_item.repeat_badge

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt override)
        self.clicked.emit(self.alert)
        super().mousePressEvent(event)


class AlertCenterPanel(QFrame):
    """The sit-back-and-wait surface, split into two stacked feeds.

    Top: the live intraday stream (bounce alerts, RW/RS movers, regime
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
    #: Emitted after the 60-second D1 poll re-measures every Focus name
    #: against yesterday's range, so surfaces showing the "moving" flag
    #: repaint on the cadence that already exists rather than polling
    #: themselves (trader rule 2026-08-19).
    focusBreakStatesChanged = Signal()
    # Trader, 2026-08-27: intraday alerts are a LIST beside the chart, not a
    # queue in front of it. Every M5 alert that would have queued a chart is
    # posted here instead; the desk hangs the M5 alert bar on it.
    m5AlertPosted = Signal(object)  # BounceAlert
    m5AlertsDayRolled = Signal()  # the bar is day-scoped like the queue
    # One D1 level/event alert worth the hourly Away phone push, as the
    # {symbol, label, time_text} dict d1_push_event builds. Emitted for every
    # qualifying alert in every mode; Auto Pilot owns the AWAY-only gate.
    d1EventRecorded = Signal(object)
    #: The faded list changed (a fade, a restore, a discard), so the button
    #: count can repaint. The active Focus lists change too, so `focusChanged`
    #: already fires - this exists so a surface that only shows the FADED
    #: count does not have to listen to every Focus mutation.
    focusFadedChanged = Signal()

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
        focus_d1_flags_path=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.focus_service = focus_service
        self._bounce_service = None
        self._alerts: list[BounceAlert] = []
        self._d1_alerts: list[BounceAlert] = []
        self._review_queue: list[BounceAlert] = []
        # Whether the chart in front belongs to the waiting list (dequeued, or
        # a clicked D1 row) or was merely clicked off the M5 alert bar. Decides
        # what a click elsewhere does with it: re-queue, or skip for now.
        self._current_review_holds_place: bool = True
        #: Trader rule 2026-08-19 (evening): "a long inside yesterday's range is
        #: probably chop. Chart review should only show me longs above the
        #: previous day's high and shorts below the previous day's low."
        #: DEFAULT-ON, and it HIDES - nothing is removed from any store, feed or
        #: history, no alert is muted, and no watchlist entry is touched.
        self._review_movers_only = True
        #: What the filter withheld, newest per symbol, so the count is honest
        #: and one click can show exactly those names.
        self._hidden_inside_range: dict[str, BounceAlert] = {}
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
        # R5 section 4: the any-bounce watch rides the same persistence
        # gate, so a bare test panel never writes it either.
        self._any_bounce_watches_path = (
            ANY_BOUNCE_WATCHES_FILE if persist_ignored else None
        )
        self._any_bounce_watches: list[AnyBounceWatch] = (
            load_any_bounce_watches(self._any_bounce_watches_path)
            if self._any_bounce_watches_path is not None
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
        #: How the last adoption attempt ended: "adopted", "already_auto",
        #: "already_trader_owned", or "failed". Only "adopted" means this desk
        #: took ownership of the entry.
        self._last_adoption_outcome = "adopted"
        #: Auto mode as of the previous poll, so the flip back to DESK is
        #: detectable. None until the first poll - a desk that starts in DESK
        #: has not flipped and drains on the ordinary stored verdicts.
        self._last_seen_auto_mode: str | None = None
        #: Single flight for the flip-triggered re-measurement.
        self._reverify_running = False
        #: When AWAY/EVENING last flipped back to DESK, floored to the second.
        #: The drain adopts only verdicts stamped at or after this moment, so an
        #: unattended stretch's verdicts can never be adopted by any path (R2.2).
        #: None on a desk that has not flipped - it drains on the ordinary
        #: stored verdicts, which is what DESK has always done.
        self._desk_flip_at: datetime | None = None
        #: Which DESK return the owed re-verification answers. Incremented on
        #: every flip back to DESK, and it - not `_desk_flip_at` - is the
        #: identity a finishing worker compares against. The timestamp cannot
        #: be the identity: it is floored to the second (that is the resolution
        #: the verdict barrier needs), so two DESK returns inside one second
        #: would share it, and an in-flight run's success would clear the
        #: newer flip's debt (external review, 2026-08-15). A counter has no
        #: such collision.
        self._desk_flip_generation = 0
        #: Set when a flip re-verification is OWED: the drain adopts nothing
        #: until it succeeds. Carries the earliest moment the next attempt may
        #: start, so a failure retries on a later poll instead of falling
        #: through. None means nothing is owed.
        self._reverify_retry_at: datetime | None = None
        #: Per-poll-cycle bookkeeping for gate-check FAILURES (2026-08-19). On
        #: the morning the gate raised, the wrapper logged one traceback per
        #: pick: 121 tracebacks every 30 seconds rotated `trading_bot.log` and
        #: nearly took the evidence with it. One traceback and one summary per
        #: cycle keeps the fault just as loud and the record survivable.
        self._gate_check_errors = 0
        self._gate_check_error_reason = ""
        #: Consecutive failures of the owed re-verification, capped by
        #: FLIP_REVERIFY_MAX_ATTEMPTS.
        self._reverify_failures = 0
        # Focus-pick D1 interest flags: every Focus name is auto-watched for
        # the whole D1 event set (15EMA reject, 5d/20d extremes, SMA breaks,
        # AVWAPE touches). "SYM|kind" fires at most once per session; the
        # registry is day-scoped like the ignored/parked stores.
        self._focus_d1_flags_path = (
            Path(focus_d1_flags_path)
            if focus_d1_flags_path is not None
            else (FOCUS_D1_FLAGS_FILE if persist_ignored else None)
        )
        self._focus_d1_flags: set[str] = (
            load_day_scoped_flags(
                self._focus_d1_flags_path, market_date=self._ignored_market_date
            )
            if self._focus_d1_flags_path is not None
            else set()
        )
        # Previous-day extreme gate on Focus flagging (trader rule 2026-08-05:
        # "I don't want focus picks to flag if they are below the previous day
        # high for longs, or above the previous day low for shorts - otherwise
        # it's just noise"). A Focus name earns its Focus PRIVILEGES - the
        # automatic D1 interest flags, the tier-gate bypass, the always-sound -
        # only once it trades beyond yesterday's extreme in its own direction.
        # Below that it is not silenced: it simply falls back to the ordinary
        # tier gate, so a genuinely strong bounce (S/A, PROVEN) still
        # comes through. "SYM|long" -> prev_day_gate state; the companion map
        # stamps when the break was first seen so the D1 event window opens
        # THERE and never replays what the name did while still inside
        # yesterday's range. Both are day-scoped.
        self._focus_break_state: dict[str, str] = {}
        #: (symbol, side) -> (bar-identity stamp, state). One entry per pair,
        #: replaced when its bars change - see `_measure_mover_state`.
        self._mover_measure_cache: dict[tuple[str, str], tuple[tuple, str]] = {}
        #: Same shape for the session-VWAP leg (trader rule 2026-08-27).
        self._vwap_measure_cache: dict[tuple[str, str], tuple[tuple, str]] = {}
        #: And for the D1 trend leg (trader rule 3, 2026-08-27).
        self._sma_measure_cache: dict[tuple[str, str], tuple[tuple, str]] = {}
        self._focus_break_open_at: dict[str, datetime] = {}
        self._focus_gate_held = 0
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
            #
            # COALESCED (2026-08-31, trader-approved under the file-scoped
            # ask-first rule). `_rebuild_feed` destroys and reconstructs every
            # row widget in both feeds - up to MAX_FEED_ITEMS + MAX_D1_FEED_ITEMS
            # = 350 widget trees, each with its own stylesheet - and the DESK
            # drain that morning fired it 45 times in 13 seconds. Only the
            # TRIGGER is coalesced: which alerts pass the feed gate, their
            # order, the repetition fold and the digest are all decided inside
            # `_rebuild_feed` and are untouched. Nothing is withheld - the
            # rebuild still happens, once, within 200 ms of the last change.
            # Late-bound so the coalescer calls whatever `_rebuild_feed` is at
            # fire time - the seam a test spies on is the one that runs.
            self._focus_feed_coalescer = SignalCoalescer(
                lambda: self._rebuild_feed(), parent=self
            )
            self.focus_service.focusChanged.connect(
                self._focus_feed_coalescer.request
            )

        self.min_tier_input = QComboBox()
        for label, mode in MIN_TIER_CHOICES:
            self.min_tier_input.addItem(label, mode)
        saved_mode = str(get_local_setting("qt_alert_min_tier", "all") or "all")
        self.min_tier_input.setCurrentIndex(max(0, self.min_tier_input.findData(saved_mode)))
        self.min_tier_input.currentIndexChanged.connect(self._on_prefs_changed)

        self.sound_input = QCheckBox("Sound on S/A + PROVEN")
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

        # Built BEFORE the tab strip, because the tab strip hosts one of its
        # widgets. Trader, 2026-08-20, on the desk column: "I cannot see the
        # charts at all". The pane used to stack title -> setup text -> charts
        # -> two arm rows -> a ~600px capture rail -> the verb row, so the
        # charts - the whole point of the surface - got whatever was left.
        #
        # Only the capture rail goes to a tab. Measured at this column's 420px
        # the rail is 697px and the arm bar 131px, so the rail was 84% of the
        # problem, and the arm bar carries the controls the trader reaches for
        # per-chart: the M5 and D1 alert hotbuttons and the type-a-ticker box
        # ("I also need my m5 and D1 alert hotbuttons back on the bottom of
        # the visual chart... I also need the ability to input a ticker
        # manually as well", same day). It stays welded under the chart.
        self.chart_review = AlertChartReview(
            self, dock_arm_bar=True, dock_capture_rail=False
        )
        self.chart_review.removeTodayRequested.connect(
            self._remove_review_alert_for_today
        )
        self.chart_review.likeAdvanceRequested.connect(self._advance_after_like)
        self.chart_review.focusRequested.connect(self._add_review_alert_to_focus)
        self.chart_review.skipRequested.connect(self._skip_review_alert)
        self.chart_review.crossFocusToggled.connect(self._toggle_review_cross_focus)
        self.chart_review.watchToggled.connect(self._toggle_chart_watch)
        self.chart_review.d1EventToggled.connect(self._toggle_d1_event_watch)
        self.chart_review.anyBounceToggled.connect(self._toggle_any_bounce_watch)
        self.chart_review.externalChartRequested.connect(self._open_external_chart)
        self.chart_review.revealHiddenRequested.connect(self.reveal_hidden_reviews)
        self.chart_review.d1LevelAlertRequested.connect(self._arm_d1_level_from_chart)
        self.chart_review.symbolRequested.connect(self.chart_symbol)
        self.chart_review.levelArmRequested.connect(self._arm_level_from_dock)
        self.chart_review.levelDisarmRequested.connect(self._disarm_level_from_dock)
        self.chart_review.levelAlertRequested.connect(self._arm_price_alert_from_level)
        self.chart_review.vetoDayTradeRequested.connect(self._veto_but_day_trade)

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

        # The Armed tab is the INVENTORY across every symbol. The controls that
        # fill it live under the chart, on the arm bar, where the symbol they
        # act on is the one being looked at.
        self._armed_tab_index = self.tabs.addTab(self.armed_list, "Armed")

        # The capture rail. Scrolled, because its four sections are taller than
        # this column's tab body and a rail whose Note field is below the fold
        # is a rail that does not get used. Its contract is unchanged by moving
        # it: it records, and it has never muted, suppressed, scored, gated,
        # alerted or written a watchlist.
        capture_scroll = QScrollArea()
        capture_scroll.setWidgetResizable(True)
        capture_scroll.setWidget(self.chart_review.capture_rail)
        self._capture_tab_index = self.tabs.addTab(capture_scroll, "Capture")

        # R10.H: the market-journal tab, AFTER Capture. A note written while
        # the tape is still moving, M5 by default because that is what the
        # trader is watching when they reach for it. It writes through the same
        # MarketJournalService the left-nav page uses - one store, one writer -
        # so an entry means the same thing whichever surface produced it.
        #
        # File-scoped ask-first note: this file houses alert code, so the rule
        # fires on any edit here. The trader authorized this packet explicitly
        # on 2026-08-24 ("go ahead and do R10E R10F R10G R10H"), which is that
        # answer. The edit itself is presentation only: a tab, a text box and a
        # save button. No alert, tier, fold, digest or queue behaviour is
        # touched.
        self._journal_tab_index = self.tabs.addTab(
            self._build_journal_tab(), "Journal  Ctrl+J"
        )

        self._refresh_armed_list()
        self.chart_review.armedSummaryChanged.connect(self._refresh_armed_tab_label)
        self._refresh_armed_tab_label(self.chart_review.armed_count())
        self._bind_capture_shortcuts()
        self._bind_journal_shortcut()
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
        self.focus_strength.reviewAllRequested.connect(self.review_focus_picks)
        self.focus_strength.fadedReviewRequested.connect(self.review_faded_picks)
        # A fade/restore/discard changes a count the board paints. It rides
        # the board's OWN coalescer (`set_focus_service`), so a burst is one
        # render - the coalescing lives at the listener.
        self.focusFadedChanged.connect(self.focus_strength.request_refresh)

        # The M5 Strength Board moved in under it (trader, 2026-08-31: "it
        # really should be modified to fit in the 'strength' window in the
        # trading desk - either integrated directly or be positioned below
        # it"). Positioned below, in a section that starts CLOSED: the alert
        # column has a 360 px floor and everything left of it is chart, so a
        # board that claimed space at startup would take it from the chart the
        # trader is reading. Closed it is one header row.
        #
        # This panel only HOSTS. `MainWindow` still owns the one
        # `StrengthBoardService`, its one timer and its one fetch, and hands
        # it here through `attach_strength_board` - the board changed address,
        # not owner, and nothing here refreshes, schedules or caches.
        self.strength_board: "StrengthBoardPanel | None" = None
        self.strength_board_section = CollapsibleSection("M5 Strength Board")
        self.strength_column = QWidget()
        strength_layout = QVBoxLayout(self.strength_column)
        strength_layout.setContentsMargins(0, 0, 0, 0)
        strength_layout.setSpacing(theme.px(4))
        strength_layout.addWidget(self.focus_strength, 1)
        strength_layout.addWidget(self.strength_board_section, 0)
        # A closed section asks for nothing; an open one earns the larger half
        # of the column, and closing hands every pixel straight back.
        self.strength_board_section.toggled.connect(
            lambda expanded: strength_layout.setStretch(1, 2 if expanded else 0)
        )

        self.tabs_row = QSplitter(Qt.Orientation.Horizontal)
        self.tabs_row.addWidget(self.tabs)
        self.tabs_row.addWidget(self.strength_column)
        self.tabs_row.setStretchFactor(0, 3)
        self.tabs_row.setStretchFactor(1, 2)
        self.tabs_row.setChildrenCollapsible(False)
        # The tab stack hints wide enough to squeeze the board out entirely;
        # an explicit minimum takes precedence over minimumSizeHint and hands
        # the split back to the preset (same fix the desk columns needed).
        # 170 + the board's 170 stays inside the alert column's 360px floor, so
        # adding the board cannot force the whole desk column wider.
        self.tabs.setMinimumWidth(theme.px(170))
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
        # D1 watches consume only the shared chart service's in-memory series.
        # A throttled prefetch request performs every stat/read/parse on the
        # chart worker pool; the first poll may honestly be UNKNOWN rather than
        # blocking the whole desk on one symbol's durable store.
        self._d1_prefetch_last: dict[str, float] = {}
        # Snappiness packet 2, item 1a. `bot.m5_chart_bars` rebuilds ~150 dicts
        # with six float() coercions each, and eight timer-driven sites ask for
        # the same symbol's bars on the 30s and 60s ticks. Memoized here rather
        # than in ChartDataService because the source series belongs to
        # BounceBot, which the service cannot see. Same shape as the D1 twin
        # (`ChartDataService.cached_bar_dicts`, 2026-08-21): a strong reference
        # to the source list is held so the identity check cannot be fooled by
        # a recycled id, and length + last stamp catch an in-place append.
        self._m5_bar_dicts: "OrderedDict[tuple[str, int], tuple[list, tuple, list]]" = OrderedDict()
        # Item 1c: one prefetch per tick instead of ~105 single-element tasks
        # queued ahead of the snapshot for the chart the trader just clicked.
        # Flushed on the next event-loop turn, so every caller batches without
        # each poll having to know it is the last one.
        self._d1_prefetch_pending: list[str] = []
        self._d1_prefetch_flush_armed = False

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
        # The refetch is connected FIRST so bars that landed since the last
        # tick are already in hand when the chart rebuilds immediately below;
        # Qt runs same-signal slots in connection order.
        self._watch_timer.timeout.connect(self._refresh_stale_queue_bars)
        # A "holding highs" row that stopped holding is deleted from the queue
        # (trader rule 2026-08-21). Connected AFTER the refetch so it judges
        # bars that just landed, and BEFORE the re-render so the chart never
        # repaints a row that is about to go.
        self._watch_timer.timeout.connect(self._expire_stale_hold_alerts)
        self._watch_timer.timeout.connect(self._refresh_review_chart)
        # DESK-mode auto picks ride the same 30s tick: the staging file is a
        # cheap local read and a new pick is not latency-critical.
        self._watch_timer.timeout.connect(self._poll_auto_pick_pending)
        # Rides the same timer rather than owning one: both drain a file the
        # engine wrote, and one owner per timer (plan.md sec 5). Unlike
        # adoption this runs in EVERY mode - a Focus entry whose scan line was
        # cut is wrong on the board whether the trader is at the desk or not.
        self._watch_timer.timeout.connect(self._drain_focus_desync_requests)
        start_staggered(self._watch_timer, 39_000)
        # A refetch finishes off-thread; repaint the moment it lands rather
        # than waiting up to 30s for the next tick. Qt queues this across the
        # thread boundary because both ends are QObjects.
        try:
            from ui.services.chart_bar_refresh import shared_refresh_service

            shared_refresh_service().barsRefreshed.connect(self._on_bars_refreshed)
        except Exception:
            logging.debug("Chart bar refresh signal not connected.", exc_info=True)
        # Persistent D1 level alerts poll less often: the daily-store reads
        # are mtime-cached and the evidence changes at most once per M5 bar.
        # The D1 event watches (derived-level alerts) ride the same tick.
        self._d1_watch_timer = QTimer(self)
        self._d1_watch_timer.setInterval(60_000)
        self._d1_watch_timer.timeout.connect(self._poll_d1_level_watches)
        self._d1_watch_timer.timeout.connect(self._poll_d1_event_watches)
        self._d1_watch_timer.timeout.connect(self._poll_any_bounce_watches)
        # Focus picks are auto-watched for every D1 event kind - no arming
        # needed. Rides the same 60s cadence as the armed D1 watches.
        self._d1_watch_timer.timeout.connect(self._poll_focus_d1_interest)
        start_staggered(self._d1_watch_timer, 77_000)
        # A3: the fade check. Deliberately NOT on the 60s tick above - it walks
        # every Focus entry and asks a calendar, which has no business inside a
        # per-symbol poll loop. Half-hourly is far finer than a clock measured
        # in trading DAYS needs; the day roll runs it too, so a desk left open
        # over a session boundary does not wait for the next tick.
        self._focus_fade_timer = QTimer(self)
        self._focus_fade_timer.setInterval(1_800_000)
        self._focus_fade_timer.timeout.connect(self.run_focus_fade_check)
        start_staggered(self._focus_fade_timer, 300_000)

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

    #: The Auto mode is re-read from disk at most this often. Alerts arrive in
    #: bursts, and each one asks; a JSON read per alert is cheap but pointless.
    _AUTO_MODE_CACHE_SECONDS = 5.0

    def _auto_mode_now(self) -> str:
        """Machine-local Auto mode (OFF/DESK/AWAY/EVENING), briefly cached.

        Read from the Auto Pilot state file rather than through a service
        handle: that file is already the one shared truth every off-thread
        reader uses, and this panel is constructed in contexts where no
        AutopilotService exists (tests, the legacy path).
        """
        now = time.monotonic()
        cached = getattr(self, "_auto_mode_cached", None)
        if cached is not None and now - cached[0] < self._AUTO_MODE_CACHE_SECONDS:
            return cached[1]
        try:
            from autopilot_core import read_auto_pilot_mode

            mode = read_auto_pilot_mode()
        except Exception:
            # Fail LOUD: an unreadable mode must never be the reason the desk
            # goes silent on the trader.
            mode = "OFF"
        self._auto_mode_cached = (now, mode)
        return mode

    def _alerts_may_sound(self) -> bool:
        """The sound checkbox, and then the away-from-the-desk rule.

        Trader rule 2026-08-14: AWAY and EVENING both queue alerts silently -
        away, or asleep, there is nobody the sound could reach, and EVENING has
        its own deliberate wake channel in the SPY alarm. Only the sound is
        suppressed: the feed, the history and the D1 unread badge all keep
        filling, so the size of what accrued is the first thing visible on
        sitting back down.
        """
        if not self.sound_input.isChecked():
            return False
        return self._auto_mode_now() not in ("AWAY", "EVENING")

    def add_alert(self, alert: BounceAlert) -> None:
        self._refresh_ignored_market_date()
        if _is_feed_noise_alert(alert):
            return
        if (
            alert.symbol
            and alert.symbol in self._ignored_symbols
            and alert.tag != CHART_WATCH_TAG
        ):
            return
        # Announced before any routing below, so a D1 event reaches the phone
        # whether it lands in the D1 Focus feed or the main feed, and whichever
        # tier gate the trader has set.
        d1_event = d1_push_event(alert)
        if d1_event is not None:
            self.d1EventRecorded.emit(d1_event)
        # A Focus pick's automatic D1 interest flag belongs in the D1 Focus
        # feed (the name is already the trader's) plus the chart queue.
        if alert.tag == FOCUS_D1_EVENT_TAG:
            self._enqueue_review_alert(alert)
            self._add_d1_alert(alert)
            return
        # D1 Focus is reserved for favorite/high-conviction transitions
        # (final bucket upgrades only). Developing trigger/watch observations
        # are research evidence and are excluded from both actionable feeds.
        if alert.is_d1 and is_ready_d1_alert(alert):
            self._enqueue_review_alert(alert)
            self._add_d1_alert(alert)
            return
        # The backing list is written BEFORE any repetition decision, and is
        # never consulted by one. History, the evidence streams and the AWAY
        # push all read from here, so folding a row can never cost a record.
        self._alerts.insert(0, alert)
        del self._alerts[MAX_FEED_ITEMS * 2 :]
        is_focus = self._alert_has_focus_privilege(alert)
        # Trader rule 2026-08-27: a with-trend regime-pause row ("holding
        # highs" on a bullish day, "pressing lows" on a bearish day) goes
        # straight to M5 Focus and never occupies the review chart - the
        # decision is made. Measured AFTER `is_focus` on purpose: the feed row
        # is presented exactly as it was before the placement, so the rule
        # changes where the name goes and not how the row looks or sounds.
        auto_focused = self._auto_focus_regime_pause(alert)
        if alert_passes_feed_gate(alert, self._min_tier_mode(), is_focus=is_focus):
            # The chart review queue is likewise decided before, and
            # independently of, how the row is presented.
            if not auto_focused:
                self._enqueue_review_alert(alert)
            decision = self._repetition_decision(alert, is_focus=is_focus)
            if decision.action == ACTION_FOLD and self._fold_into_existing_row(alert, decision):
                pass
            elif decision.action == ACTION_DIGEST and self._add_to_open_digest(alert):
                pass
            else:
                self._insert_item_into(
                    self.feed_layout, alert, MAX_FEED_ITEMS, repeat=decision
                )
            if (
                decision.sounds
                and self._alerts_may_sound()
                and alert_should_sound(alert, is_focus=is_focus)
            ):
                QApplication.beep()
        self._emit_feed_status()

    # -- R4 section 6.3: display-only repetition control -----------------
    def _repetition_decision(self, alert: BounceAlert, *, is_focus: bool):
        """How this alert should be presented in the feed.

        Fails OPEN on any error: an exception here yields a plain new row,
        which is exactly today's behaviour. A presentation control must never
        be able to cost the trader an alert.
        """
        from alert_repetition import ACTION_NEW, RepeatDecision

        try:
            ledger = self._repetition_ledger()
            privileged = bool(
                is_focus
                or is_chart_watch_alert(alert)
                or is_entry_assist_alert(alert)
                or is_ready_d1_alert(alert)
            )
            return ledger.consider(
                symbol=alert.symbol,
                side=alert.side,
                tier=extract_alert_tier(alert),
                is_proven=is_proven_alert(alert),
                privileged=privileged,
            )
        except Exception:
            logging.debug("Alert repetition decision failed.", exc_info=True)
            return RepeatDecision(ACTION_NEW, 1)

    def _repetition_ledger(self):
        from alert_repetition import RepetitionLedger, configured_digest_minutes

        ledger = getattr(self, "_repeat_ledger", None)
        if ledger is None:
            ledger = RepetitionLedger(digest_minutes=configured_digest_minutes())
            self._repeat_ledger = ledger
        market_date, session_open = self._current_session_bounds()
        ledger.set_market_date(market_date, session_open=session_open)
        if ledger.session_open is None and session_open is not None:
            ledger.session_open = session_open
        return ledger

    @staticmethod
    def _current_session_bounds():
        """(market date, regular open) or (today, None) if unknowable.

        A None open disables the digest rather than digesting all day - the
        fail-open direction, because this control is presentation and must
        never become accidental suppression.
        """
        try:
            from market_session import get_market_session_window

            window = get_market_session_window()
            return (
                window.market_date.isoformat(),
                window.open_local.replace(tzinfo=None),
            )
        except Exception:
            from datetime import datetime as _dt

            return _dt.now().date().isoformat(), None

    def _fold_into_existing_row(self, alert: BounceAlert, decision) -> bool:
        """Update the live row in place. False if there is no row to update.

        A row can legitimately be gone - trimmed off the bottom by
        MAX_FEED_ITEMS, or destroyed by a feed rebuild - and in that case the
        honest answer is a fresh row rather than a silently dropped alert.
        """
        key = (str(alert.symbol or "").upper(), str(alert.side or "").upper())
        item = getattr(self, "_feed_rows", {}).get(key)
        if item is None:
            return False
        try:
            item.set_repeat_count(
                decision.repeat_count,
                latest_trigger=alert.trigger or alert.raw_text,
            )
        except RuntimeError:
            # The C++ side was deleted (trimmed or rebuilt).
            self._feed_rows.pop(key, None)
            return False
        return True

    def _add_to_open_digest(self, alert: BounceAlert) -> bool:
        """Fold an ordinary open-burst alert into one ranked digest row.

        Nothing is discarded: every digested alert is in the backing list, in
        History, in the chart review queue, and named on the digest row itself.
        """
        row = getattr(self, "_digest_row", None)
        try:
            if row is None or row.parent() is None:
                row = QLabel()
                row.setObjectName("Panel")
                row.setWordWrap(True)
                row.setStyleSheet(
                    f"QLabel#Panel {{ color: {theme.color('text_secondary')}; "
                    "padding: 8px 10px; }"
                )
                self.feed_layout.insertWidget(0, row)
                self._digest_row = row
            symbols = self._repetition_ledger().digest_symbols()
            row.setText(
                f"Open burst · {len(symbols)} name(s) grouped: "
                + ", ".join(symbols)
            )
            row.setToolTip(
                "Ordinary alerts in the first minutes after the open are "
                "grouped here so the burst does not bury the feed. Every one "
                "of them is still in History, in the chart review queue, and "
                "in the evidence log - nothing was dropped. PROVEN "
                "configs, Focus names and anything you armed yourself bypass "
                "this entirely."
            )
        except Exception:
            logging.debug("Open-burst digest row failed.", exc_info=True)
            return False
        return True

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
        if self._alerts_may_sound() and (
            is_ready_d1_alert(alert) or self._alert_has_focus_privilege(alert)
        ):
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

    def _refresh_armed_tab_label(self, count: int = 0) -> None:
        """Armed state stays readable with the Armed tab closed.

        Two places carry it, deliberately: this count in the tab title, in
        peripheral vision, and the always-visible line on the review pane's
        verb row. The arm bar's own "Nothing armed" text went onto the tab
        with the bar, and a state the trader has to go looking for is a state
        that gets forgotten while a watch is live.
        """
        count = max(0, int(count or 0))
        self.tabs.setTabText(
            self._armed_tab_index, f"Armed ({count})" if count else "Armed"
        )

    # ------------------------------------------------------------------
    # R4 section 2.3's founding contract: every capture under five seconds,
    # no mouse.
    # ------------------------------------------------------------------
    def _bind_capture_shortcuts(self) -> None:
        """Own the rail's verb keys at PANEL scope - whatever they are.

        The list comes from `action_shortcuts()` and is never restated here, so
        P9's Alt+L was bound the day it was added without this method changing.

        The rail is on a tab page now, and a QShortcut bound inside a page the
        trader is not looking at never fires - so the keys would have silently
        stopped working the moment the rail moved. They are bound here, on the
        panel, with WidgetWithChildrenShortcut: focus anywhere in the Alert
        Center reaches them, including the charts and the verb row.

        The rail's own copies are switched off for this host
        (``bind_action_shortcuts=False``); two live bindings for one sequence
        is an ambiguous shortcut in Qt, and Qt fires NEITHER. The handlers come
        from the rail itself, so this is a rebinding, not a second list.
        """
        self._capture_shortcuts: dict[str, QShortcut] = {}
        for sequence, handler in self.chart_review.capture_rail.action_shortcuts():
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(
                lambda bound=handler: self._focus_capture_action(bound)
            )
            self._capture_shortcuts[sequence] = shortcut

    def _focus_capture_action(self, handler) -> None:
        """Raise the Capture tab, then arm/focus the rail exactly as before."""
        self.tabs.setCurrentIndex(self._capture_tab_index)
        handler()

    def _bind_journal_shortcut(self) -> None:
        """Ctrl+J: select the Journal tab and focus the composer.

        §5.3 option (a), decision 10. The trader could not find this tab on
        2026-08-26; it is the sixth of the lower strip and reachable only by
        clicking it. A keyboard route costs no row, so the 2026-08-20 rule -
        at most ONE slim row between the charts and the tab strip - is intact.
        No verb-row verb: that is a mouse route and needs its own ask.

        Panel scope with WidgetWithChildrenShortcut, exactly like the capture
        keys: a QShortcut bound inside a hidden tab page never fires. Ctrl+J is
        unbound everywhere else in scripts/ui (Ctrl+R, Ctrl+F, F9, Ctrl+Return
        and the rail's Alt+V/K/N/P/L are the whole inventory) - two live bindings for one
        sequence is an ambiguous shortcut and Qt fires NEITHER.
        """
        shortcut = QShortcut(QKeySequence("Ctrl+J"), self)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self._focus_journal_composer)
        self._journal_route_shortcut = shortcut

    def _focus_journal_composer(self) -> None:
        self.tabs.setCurrentIndex(self._journal_tab_index)
        self._journal_text.setFocus()

    def _emit_feed_status(self) -> None:
        loud = sum(
            1
            for item in self._alerts
            if alert_should_sound(item, is_focus=self._alert_has_focus_privilege(item))
        )
        # The held count makes the prev-day gate visible: silence should never
        # be indistinguishable from a dead feed.
        held = (
            f" {self._focus_gate_held} Focus name(s) waiting on yesterday's high/low."
            if self._focus_gate_held
            else ""
        )
        self.statusChanged.emit(
            f"Alert center: {len(self._alerts)} live alert(s), {loud} loud; "
            f"{len(self._d1_alerts)} favorite-bucket transition(s) in D1 Focus.{held}"
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
        # ONE read-modify-write of the settings file, not two. Two separate
        # saves are two full cycles over the same JSON and a window between
        # them in which another process can drop whatever was written first
        # (the reason `save_local_settings` exists - 2026-08-25).
        save_local_settings(
            {
                "qt_alert_min_tier": self._min_tier_mode(),
                "qt_alert_sound": bool(self.sound_input.isChecked()),
            }
        )
        self._rebuild_feed()

    def _alert_is_focus(self, alert: BounceAlert) -> bool:
        """Membership only: is this symbol one of the trader's Focus picks."""
        return bool(self.focus_service and alert.symbol and self.focus_service.is_focus(alert.symbol))

    # ------------------------------------------------------ prev-day gate
    @staticmethod
    def _focus_gate_key(symbol: str, side: str) -> str:
        return f"{str(symbol or '').strip().upper()}|{side}"

    def focus_break_state(self, symbol: str, side: str) -> str:
        """Cached prev-day-extreme state for one Focus name/side.

        Refreshed by the 60s D1 poll, which is the only place that already
        holds both bar sets. A symbol the poll has not reached yet reads
        UNKNOWN, which does not grant Focus privileges - missing data is
        uncertainty, never confirmation (plan.md sec 5).
        """
        return self._focus_break_state.get(self._focus_gate_key(symbol, side), "unknown")

    def _update_focus_break_state(
        self,
        symbol: str,
        side: str,
        m5_bars: list,
        d1_bars: list,
        moment: datetime,
    ) -> datetime | None:
        """Re-measure one Focus name against yesterday's range.

        Returns the START of the M5 bar that FIRST broke the level today (the
        D1 event window's opening edge), or None while the latest completed
        close is not beyond it. Anchoring to the bar rather than to the poll
        tick matters twice: the breakout bar's own D1 events count, and a
        60s poll that arrives late - or a desk started at 11:00 - opens the
        same window as one watching from the first print. The first stamp is
        kept even if price dips back inside the range, so an event that
        printed while the name was genuinely beyond yesterday's extreme stays
        eligible and a re-break does not restart the clock.
        """
        key = self._focus_gate_key(symbol, side)
        prev_high, prev_low = prev_session_extremes(d1_bars, session=moment.date())
        completed = completed_session_bars(m5_bars, now=moment)
        state = prev_day_break_state(
            side, _bar_close(completed[-1]) if completed else None, prev_high, prev_low
        )
        self._focus_break_state[key] = state
        if state != PREV_DAY_BREAK_OPEN:
            return None
        stamped = self._focus_break_open_at.get(key)
        if stamped is None:
            stamped = moment
            for bar in completed:
                if (
                    prev_day_break_state(side, _bar_close(bar), prev_high, prev_low)
                    == PREV_DAY_BREAK_OPEN
                ):
                    stamp = bar.get("dt")
                    stamped = stamp if isinstance(stamp, datetime) else moment
                    break
            self._focus_break_open_at[key] = stamped
        return stamped

    def mover_state(self, symbol: str, side: str) -> str:
        """OPEN / CLOSED / UNKNOWN for "beyond yesterday's extreme".

        ONE definition, shared with the adoption gate: this reads
        `focus_adoption_gate.mover_state`, which is the same
        `prev_day_break_state` call the gate makes for its extreme leg. A
        display filter with its own copy of the rule would eventually hide a
        name the machine had just adopted.

        The 60-second D1 poll already measures every Focus name and caches the
        answer, so those cost nothing here. Anything else is measured on
        demand from bars the desk already holds - `_m5_bars_for` and
        `_d1_bars_for` read the running bot's in-memory series and the local
        daily store, so this adds no fetch and no IB traffic.
        """
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return PREV_DAY_UNKNOWN
        side_key = str(side or "").strip().lower()
        sides = (side_key,) if side_key in ("long", "short") else ("long", "short")
        for item in sides:
            cached = self._focus_break_state.get(self._focus_gate_key(symbol, item))
            if cached == PREV_DAY_BREAK_OPEN:
                return PREV_DAY_BREAK_OPEN
        measured = [self._measure_mover_state(symbol, item) for item in sides]
        if PREV_DAY_BREAK_OPEN in measured:
            return PREV_DAY_BREAK_OPEN
        # A name whose sides disagree between "measured, inside" and "could not
        # measure" is not verified inside anything: uncertainty wins, and the
        # display shows it rather than hiding it.
        if PREV_DAY_UNKNOWN in measured:
            return PREV_DAY_UNKNOWN
        return PREV_DAY_CLOSED

    def vwap_state(self, symbol: str, side: str) -> str:
        """OPEN / CLOSED / UNKNOWN for "on the right side of session VWAP".

        Trader rule 2026-08-27: "it's below VWAP trending lower on the M5 -
        what a waste of my time." The predicate is the adoption gate's own
        VWAP leg, `focus_adoption_gate.session_vwap_state`, fed by
        `regime_pause_hold.session_levels` over the cached M5 series - session
        VWAP from `chart_snapshot.session_vwap_series` on completed bars, never
        BounceBot's dynamic/EOD VWAP (CLAUDE.md, packet R2). A sideless row has
        no right side and is UNKNOWN, which shows.
        """
        symbol = str(symbol or "").strip().upper()
        side_key = str(side or "").strip().lower()
        if not symbol or side_key not in ("long", "short"):
            return PREV_DAY_UNKNOWN
        try:
            moment = datetime.now()
            m5_bars = self._m5_bars_for(symbol)
            stamp = (moment.date(), self._series_stamp(m5_bars))
            remembered = self._vwap_measure_cache.get((symbol, side_key))
            if remembered is not None and remembered[0] == stamp:
                return remembered[1]
            levels = regime_pause_hold.session_levels(m5_bars, now=moment)
            state = focus_adoption_gate.session_vwap_state(
                side_key, levels.price, levels.vwap
            )
            self._vwap_measure_cache[(symbol, side_key)] = (stamp, state)
            return state
        except Exception:
            logging.debug("Session VWAP state unavailable for %s.", symbol, exc_info=True)
            return PREV_DAY_UNKNOWN

    @staticmethod
    def _is_d1_review(alert: BounceAlert) -> bool:
        """A chart the D1 side of the desk recommended - the swing scanner's
        D1 rows and the Focus D1 interest flags. The trend leg applies to
        these and to nothing intraday."""
        return bool(alert.is_d1) or str(alert.tag or "") == FOCUS_D1_EVENT_TAG

    def sma_trend_state(self, symbol: str, side: str) -> str:
        """OPEN / CLOSED / UNKNOWN for "a long above the SMA200, a short below the SMA50".

        Trader rule 3, 2026-08-27, from MUFG: a D1 short recommended above
        every SMA in a clean uptrend. The rule is `sma_trend_gate`; this only
        feeds it numbers the desk already holds - the averages off completed
        bars of the local daily store, the price off the last completed M5
        bar when the bot has one and off the last daily bar otherwise. No
        fetch, no IB traffic. Memoized on the identity of both series.
        """
        symbol = str(symbol or "").strip().upper()
        side_key = str(side or "").strip().lower()
        if not symbol or side_key not in ("long", "short"):
            return PREV_DAY_UNKNOWN
        try:
            moment = datetime.now()
            d1_bars = self._d1_bars_for(symbol)
            m5_bars = self._m5_bars_for(symbol)
            stamp = (
                moment.date(),
                self._series_stamp(d1_bars),
                self._series_stamp(m5_bars),
            )
            remembered = self._sma_measure_cache.get((symbol, side_key))
            if remembered is not None and remembered[0] == stamp:
                return remembered[1]
            completed = completed_session_bars(m5_bars, now=moment)
            price = _bar_close(completed[-1]) if completed else None
            if price is None and d1_bars:
                price = _bar_close(d1_bars[-1])
            sma50, sma200 = sma_trend_gate.trend_levels(d1_bars, today=moment.date())
            state, _reason = sma_trend_gate.sma_trend_state(side_key, price, sma50, sma200)
            self._sma_measure_cache[(symbol, side_key)] = (stamp, state)
            return state
        except Exception:
            logging.debug("SMA trend state unavailable for %s.", symbol, exc_info=True)
            return PREV_DAY_UNKNOWN

    def _review_chart_state(self, alert: BounceAlert) -> str:
        """Should this chart show? Every leg, one answer.

        CLOSED when ANY leg is verified against the name - inside yesterday's
        range, on the wrong side of session VWAP, or (a D1 recommendation
        only) a long under its SMA200 / a short over its SMA50. One measured
        reason to hide is enough; that is deliberately not the adoption
        gate's ordering, which reports "could not measure" before "measured
        and failed" because it is explaining an eviction, not deciding a
        display. UNKNOWN (nothing verified against it, something unmeasurable)
        SHOWS, tagged; OPEN is a verified pass on every leg asked.
        """
        legs = [
            self.mover_state(alert.symbol, alert.side),
            self.vwap_state(alert.symbol, alert.side),
        ]
        if self._is_d1_review(alert):
            legs.append(self.sma_trend_state(alert.symbol, alert.side))
        if PREV_DAY_CLOSED in legs:
            return PREV_DAY_CLOSED
        if PREV_DAY_UNKNOWN in legs:
            return PREV_DAY_UNKNOWN
        return PREV_DAY_BREAK_OPEN

    def _review_badge_state(self, alert: BounceAlert) -> str:
        """What the review chart's badge says about the name in front of you.

        `open` (MOVING) needs the extreme leg verified and no later leg
        verified against it; a name revealed after the VWAP leg hid it says
        `wrong_side_vwap`, after the trend leg `wrong_side_sma`; the extreme
        leg's own answers are unchanged.
        """
        mover = self.mover_state(alert.symbol, alert.side)
        if mover != PREV_DAY_BREAK_OPEN:
            return mover
        if self.vwap_state(alert.symbol, alert.side) == PREV_DAY_CLOSED:
            return "wrong_side_vwap"
        if (
            self._is_d1_review(alert)
            and self.sma_trend_state(alert.symbol, alert.side) == PREV_DAY_CLOSED
        ):
            return "wrong_side_sma"
        return mover

    @staticmethod
    def _series_stamp(bars) -> tuple:
        """Cheap identity for a bar series: how many, and when the last one is.

        Enough to decide "these are the same bars I measured last time", and
        O(1) - the point is to avoid re-deriving from them, so the check must
        not cost what it saves.
        """
        if not bars:
            return (0, None)
        try:
            return (len(bars), bars[-1].get("dt"))
        except Exception:
            return (len(bars), None)

    def _measure_mover_state(self, symbol: str, side: str) -> str:
        """One side, measured now from cached bars. Never raises.

        Memoized per (symbol, side) on the IDENTITY of the bars the answer came
        from - session date plus the length and last timestamp of both series -
        so a reused answer is one that provably could not have changed. It is a
        memo, deliberately not a cache with an expiry: `mover_state` feeds the
        movers-only review filter, which decides what the trader SEES, and a
        time-based cache would let a name that has just broken yesterday's high
        stay hidden until it lapsed. A new bar is a new key.

        Only the newest stamp per (symbol, side) is kept, so this cannot grow a
        row per five-minute bucket across a session.

        Measured before it was written (synthetic series at realistic sizes):
        0.234 ms per (symbol, side), of which 79% is what this skips - the
        materialization above it is paid either way.
        """
        try:
            moment = datetime.now()
            d1_bars = self._d1_bars_for(symbol)
            m5_bars = self._m5_bars_for(symbol)
            stamp = (
                moment.date(),
                self._series_stamp(d1_bars),
                self._series_stamp(m5_bars),
            )
            remembered = self._mover_measure_cache.get((symbol, side))
            if remembered is not None and remembered[0] == stamp:
                return remembered[1]
            prev_high, prev_low = prev_session_extremes(
                d1_bars, session=moment.date()
            )
            completed = completed_session_bars(m5_bars, now=moment)
            price = _bar_close(completed[-1]) if completed else None
            state, _reason = focus_adoption_gate.mover_state(
                side, price, prev_high, prev_low
            )
            # A failure never reaches here, so UNKNOWN-from-a-broken-read is
            # never remembered: it is the absence of an answer, not one.
            self._mover_measure_cache[(symbol, side)] = (stamp, state)
            return state
        except Exception:
            # An unreadable measurement is UNKNOWN, which SHOWS. A filter that
            # failed closed would blank the review the moment a data source
            # hiccuped - the opposite of what a trader needs mid-session.
            logging.debug("Mover state unavailable for %s.", symbol, exc_info=True)
            return PREV_DAY_UNKNOWN

    def _review_shows_regardless(self, alert: BounceAlert) -> bool:
        """Entry points the movers-only filter must never touch.

        - A deliberate Focus review shows EVERYTHING: the trader asked for
          their own list, and answering with a filtered subset of names they
          chose themselves is the surface lying about what it holds.
        - An armed chart-watch hit is the exact condition the trader armed and
          is waiting on.
        """
        return (
            str(alert.tag or "") in (FOCUS_REVIEW_TAG, FOCUS_FADED_TAG)
            or is_chart_watch_alert(alert)
        )

    def hidden_inside_range_count(self) -> int:
        return len(self._hidden_inside_range)

    def reveal_hidden_reviews(self) -> int:
        """Show the withheld names, and stop filtering for this session.

        "For that session" is literal: the flag resets with the market date,
        beside the other day-scoped state, so tomorrow opens filtered again.
        """
        withheld = list(self._hidden_inside_range.values())
        self._hidden_inside_range.clear()
        self._review_movers_only = False
        self.chart_review.set_hidden_count(0)
        for alert in withheld:
            self._enqueue_review_alert(alert)
        if withheld:
            self.statusChanged.emit(
                f"Showing {len(withheld)} name(s) inside yesterday's range - "
                "the movers-only review filter is off for the rest of today."
            )
        else:
            self.statusChanged.emit(
                "Movers-only review filter is off for the rest of today."
            )
        return len(withheld)

    def _alert_has_focus_privilege(self, alert: BounceAlert) -> bool:
        """Focus membership AND the prev-day break on the alert's own side.

        This is what the feed gate and the beep ask - NOT plain membership. A Focus long still inside yesterday's range is
        ordinary: it competes on tier like any other name.
        """
        if not self._alert_is_focus(alert):
            return False
        side = str(alert.side or "").strip().lower()
        sides = (side,) if side in ("long", "short") else ("long", "short")
        return any(
            self.focus_break_state(alert.symbol, item) == PREV_DAY_BREAK_OPEN
            for item in sides
        )

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

    def _insert_item_into(
        self, layout, alert: BounceAlert, max_items: int, *, repeat=None
    ) -> None:
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
        # R4 section 6.3: an escalation re-floats the row and carries the count
        # with it, so "third time, now S-tier" reads as one story rather than
        # as an unrelated new alert.
        if repeat is not None and getattr(repeat, "repeat_count", 1) > 1:
            try:
                item.set_repeat_count(repeat.repeat_count)
            except Exception:
                logging.debug("Repeat badge failed.", exc_info=True)
        layout.insertWidget(0, item)
        if layout is self.feed_layout and alert.symbol:
            rows = getattr(self, "_feed_rows", None)
            if rows is None:
                rows = {}
                self._feed_rows = rows
            rows[(str(alert.symbol).upper(), str(alert.side or "").upper())] = item
        while layout.count() > max_items + 1:
            taken = layout.takeAt(layout.count() - 2)
            widget = taken.widget()
            if widget is not None:
                self._forget_feed_row(widget)
                widget.deleteLater()

    def _forget_feed_row(self, widget) -> None:
        """Drop a trimmed row from the fold registry.

        Without this the registry would hold a deleted C++ object and every
        later repeat of that name would try to update a row that is no longer
        on screen - which fails safe (a fresh row) but only by accident.
        """
        rows = getattr(self, "_feed_rows", None)
        if not rows:
            return
        for key, item in list(rows.items()):
            if item is widget:
                rows.pop(key, None)

    @staticmethod
    def _clear_feed_layout(layout) -> None:
        while layout.count() > 1:
            taken = layout.takeAt(0)
            widget = taken.widget()
            if widget is not None:
                widget.deleteLater()

    def flush_pending_focus_refresh(self) -> None:
        """Run an owed coalesced feed rebuild now. The seam the tests drive."""
        coalescer = getattr(self, "_focus_feed_coalescer", None)
        if coalescer is not None:
            coalescer.flush()

    def _rebuild_feed(self) -> None:
        # Every row widget is about to be destroyed, so the fold registry and
        # the digest row must go with them - a registry pointing at deleted
        # widgets would make the next repeat of each name silently fail over
        # to a new row instead of folding.
        self._feed_rows = {}
        self._digest_row = None
        self._clear_feed_layout(self.feed_layout)
        mode = self._min_tier_mode()
        for alert in reversed(
            [
                a
                for a in self._alerts
                if a.symbol not in self._ignored_symbols
                and alert_passes_feed_gate(a, mode, is_focus=self._alert_has_focus_privilege(a))
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

    def _note_away_recap_alert(self, alert: BounceAlert) -> None:
        """Count an alert diverted from the queue into the AWAY recap.

        A COUNT, not a store: the alerts themselves are already in
        `self._alerts`, History and the evidence streams, so a second copy here
        would be a second writer for data that already has one (ground rule 8).

        The count exists because "nothing accumulated" and "nothing happened"
        must not look the same on the return. It is session-scoped, so a day
        roll starts it at zero rather than reporting yesterday's total as
        today's.
        """
        from datetime import date as _date

        today = _date.today().isoformat()
        if getattr(self, "_away_recap_session", None) != today:
            self._away_recap_session = today
            self._away_recap_diverted = 0
        self._away_recap_diverted = getattr(self, "_away_recap_diverted", 0) + 1

    def away_recap_count(self) -> int:
        """How many alerts this session routed to the recap instead of the queue."""
        from datetime import date as _date

        if getattr(self, "_away_recap_session", None) != _date.today().isoformat():
            return 0
        return int(getattr(self, "_away_recap_diverted", 0) or 0)

    def _enqueue_review_alert(self, alert: BounceAlert) -> None:
        """Queue one visual review per symbol; refresh the active symbol live.

        Only real tickers get a chart. Summary/list messages can carry junk
        pseudo-symbols extracted from their prefix (e.g. "(BULLISH_STRONG)"
        from an old AUTO WATCHLIST line) - those must never occupy the
        review pane."""
        if not alert.symbol or not SYMBOL_RE.fullmatch(alert.symbol):
            return
        # R1 amendment 2026-08-24: an AWAY day ends in a RECAP, not a queue.
        #
        # The trader returned from one AWAY day to 317 pending review items.
        # The chart-review queue is a return surface for someone sitting here;
        # in AWAY nobody is, so it stops accumulating and the day is assembled
        # into the EOD recap instead.
        #
        # This is the ONE door into the queue - the auto-pick drain, the D1
        # feed and the ordinary feed all arrive here - which is why the routing
        # belongs at this line and nowhere else. Everything upstream is
        # untouched by design: `self._alerts`, the D1 feed and badge, History
        # and every evidence stream are written BEFORE this call and never read
        # from the queue. That is the repetition-control precedent holding -
        # a display decision withholds nothing from evidence.
        #
        # EVENING deliberately keeps its queue: it is for sleeping through the
        # morning, and the queue is what the trader wakes up to.
        if self._auto_mode_now() == "AWAY":
            self._note_away_recap_alert(alert)
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
        # Trader rule 2026-08-27: an intraday alert is a LINE in the M5 alert
        # bar, never a chart in the waiting list - "purge M5 alerts from the
        # waiting list and keep those for D1 alerts". Posted here, at the one
        # door into the queue, so everything upstream (the backing list, the
        # feed, History, the evidence streams, the AWAY recap above) is
        # untouched. A click on the bar charts it through `chart_alert`.
        is_m5 = self._is_m5_review_alert(alert)
        if is_m5:
            self._attach_cached_take_prob(alert)
            self.m5AlertPosted.emit(alert)
        if (
            self._current_review_alert is not None
            and self._current_review_alert.symbol == alert.symbol
        ):
            self._current_review_alert = alert
            self._render_current_review()
            return
        if is_m5:
            return
        # Movers only (trader rule 2026-08-19). Applied HERE because this is
        # the single door into the review queue - every caller, including the
        # auto-pick drain and the D1 feed, arrives through it. It hides and
        # counts; it deletes nothing, mutes nothing and records nothing to the
        # review-learning stream.
        if self._review_movers_only and not self._review_shows_regardless(alert):
            if self._review_chart_state(alert) == PREV_DAY_CLOSED:
                self._hidden_inside_range[alert.symbol] = alert
                self.chart_review.set_hidden_count(len(self._hidden_inside_range))
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
            self._prefetch_review_queue()

    @staticmethod
    def _is_m5_review_alert(alert: BounceAlert) -> bool:
        """An ordinary intraday alert - the kind the M5 bar lists instead of the queue.

        Not one of these, which keep their chart: a D1 row, a Focus D1 flag, a
        chart-watch hit or a price alert the trader armed themselves, the
        auto-pick proposals, a typed symbol and a deliberate Focus review.
        """
        if alert.is_d1:
            return False
        if str(alert.tag or "") in (
            CHART_WATCH_TAG,
            AUTO_PICK_TAG,
            MANUAL_CHART_TAG,
            FOCUS_REVIEW_TAG,
            FOCUS_FADED_TAG,
            FOCUS_D1_EVENT_TAG,
        ):
            return False
        if str(alert.raw_text or "").lstrip().upper().startswith("PRICE ALERT"):
            return False
        return True

    def chart_alert(self, alert: BounceAlert) -> None:
        """Public: chart this alert now (the M5 bar's click). Same path as a
        feed-row click. A D1 chart in front keeps its place at the head of
        the queue; an M5 chart in front is skipped (trader rule 2026-08-27,
        second pass - see `_select_review_alert`)."""
        self._select_review_alert(alert)

    def _attach_cached_take_prob(self, alert: BounceAlert) -> None:
        """Hand the M5 bar the take probability, IF one is already cached.

        Deliberately `_review_guidance.get`, never `_guidance_for`. The cached
        lookup is a dict read; `_guidance_for` on a miss calls
        `ReviewGuide.guidance_for`, whose `_refresh()` stats two files and can
        re-read a 34 KB JSON - per alert, on the Qt thread, in the alert path.
        That is precisely the drip the snappiness packets spent three rounds
        removing, and a take-rate suffix is not worth reintroducing it.

        **This differs from the packet's premise, which assumed guidance is
        computed before the M5 emit.** It is not: `m5AlertPosted` fires here
        and `_enqueue_review_alert` returns immediately afterwards for an M5
        alert, before `_queue_score` is ever reached. The cache is filled by
        `_render_current_review`, so the suffix appears for a symbol the desk
        has already charted this session and is silent otherwise - which is the
        honest rendering of "not measured". A missing suffix says nothing; a
        0% would be a claim.

        Nothing is computed, nothing is fetched, and the alert is not otherwise
        touched: one float is attached for the bar to read.
        """
        try:
            guidance = self._review_guidance.get(alert.symbol)
            if guidance is not None and guidance.take_prob is not None:
                alert.review_take_prob = float(guidance.take_prob)
        except Exception:  # noqa: BLE001 - a row suffix never costs an alert
            logging.debug("Take-rate suffix skipped for %s.", alert.symbol, exc_info=True)

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
        """A feed-row or M5-bar click makes that alert the active visual review.

        What happens to the chart it replaces depends on where that chart
        came from. A chart that HOLDS A PLACE in the waiting list (it was
        dequeued, or it is a D1 row / armed hit the trader clicked) goes back
        to the head of the queue, so a look-elsewhere never loses it. An M5
        chart clicked off the alert bar holds no place - the bar is a list,
        not a queue (trader rule 2026-08-27) - so clicking away from it is a
        "skip for now": a `skip` review event is written and it is NOT put in
        the waiting list (trader, same day, second pass: "it shouldn't queue
        the old m5 alert in the waiting list"). Its line already left the bar
        when it was clicked; the feed and History keep it.
        """
        if not alert.symbol or alert.symbol in self._ignored_symbols:
            return
        current = self._current_review_alert
        if current is not None and current.symbol != alert.symbol:
            self._review_queue = [
                queued
                for queued in self._review_queue
                if queued.symbol not in {current.symbol, alert.symbol}
            ]
            if self._current_review_holds_place:
                self._review_queue.insert(0, current)
            else:
                # A click away IS a pass, and that is the intended meaning -
                # trader decision 2026-09-01: "clicking away = a pass". See
                # docs/DESK_INTERNALS.md, the M5 alert bar entry. Do not
                # "fix" this into a take or into silence, and do not rename
                # the reason string: review_learning keys on it.
                self._record_review_event(
                    "skip",
                    alert=current,
                    dwell_ms=self._review_dwell_ms(current.symbol),
                    queue_len=len(self._review_queue),
                    detail={"reason": "clicked_away_from_m5_alert"},
                )
        else:
            self._review_queue = [
                queued for queued in self._review_queue if queued.symbol != alert.symbol
            ]
        self._current_review_alert = alert
        self._current_review_holds_place = not self._is_m5_review_alert(alert)
        self._render_current_review()

    def _advance_review_queue(self) -> None:
        """Show the next chart - measured NOW, not when it was queued.

        Trader rule 2026-08-27: EPD was flagged on the 06:30 bar and reached
        the pane at 07:30, by which time it sat under VWAP and was fading -
        the queue-time answer was an hour stale. So the filter is asked again
        at the moment a chart is about to show, and a name that has since
        fallen inside yesterday's range or onto the wrong side of session VWAP
        is withheld (counted, one click reveals) instead of shown. Same
        exemptions as at queue time: a deliberate Focus review and an armed
        chart-watch hit always show, and once the trader has revealed the
        hidden names for the session nothing is re-checked.
        """
        hidden_before = len(self._hidden_inside_range)
        next_alert = None
        while self._review_queue:
            candidate = self._review_queue.pop(0)
            if (
                self._review_movers_only
                and not self._review_shows_regardless(candidate)
                and self._review_chart_state(candidate) == PREV_DAY_CLOSED
            ):
                self._hidden_inside_range[candidate.symbol] = candidate
                continue
            next_alert = candidate
            break
        self._current_review_alert = next_alert
        # Popped from the waiting list, so it keeps a place there if the
        # trader clicks elsewhere for a moment (see `_select_review_alert`).
        self._current_review_holds_place = True
        if len(self._hidden_inside_range) != hidden_before:
            self.chart_review.set_hidden_count(len(self._hidden_inside_range))
        self._render_current_review()
        self._prefetch_review_queue()

    def _prefetch_review_queue(self, limit: int = 24) -> None:
        """Warm the bar cache for the charts coming up next (Part C rule D4).

        Read-only and off-thread: this only populates the chart's bar cache,
        so the NEXT advance paints from memory instead of parsing parquet out
        of the Drive-backed store. It touches no alert, score, or watchlist,
        and a failure here costs nothing but a slower first paint.
        """
        try:
            from ui.services.chart_data_service import shared_service

            symbols = [
                queued.symbol for queued in self._review_queue[:limit] if queued.symbol
            ]
            if symbols:
                shared_service().prefetch(symbols)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Decision logging: the training data for learning the trader's revealed
    # preferences. Best-effort by design - a logging failure must never cost
    # a click - and disabled whenever the panel runs on non-default stores.
    @staticmethod
    def _alert_moment(alert: BounceAlert) -> datetime | None:
        """When this alert landed, as a datetime on today's date.

        ``time_text`` is all the alert carries (``%H:%M:%S``), and the review
        queue is an intraday structure cleared on the day roll, so today is the
        only date it can mean. Unparseable returns None, and the freshness rule
        then falls back to the extreme's own timestamp rather than inventing a
        moment.
        """
        text = str(getattr(alert, "time_text", "") or "").strip()
        if not text:
            return None
        for shape in ("%H:%M:%S", "%H:%M"):
            try:
                parsed = datetime.strptime(text, shape)
            except ValueError:
                continue
            return datetime.now().replace(
                hour=parsed.hour,
                minute=parsed.minute,
                second=parsed.second,
                microsecond=0,
            )
        return None

    def _hold_verdict_for(self, alert: BounceAlert):
        """Live "is it still holding?" for one regime-pause row, or None.

        None means the question could not be answered - no bot, no bars, no
        ATR - and every caller treats that as KEEP. Uncertainty may not delete
        a row (see regime_pause_hold.queue_verdict).
        """
        try:
            bars = self._m5_bars_for(alert.symbol, sessions=2)
            if not bars:
                return None
            return regime_pause_hold.queue_verdict(
                bars,
                alert.side,
                alert_time=self._alert_moment(alert),
                now=datetime.now(),
            )
        except Exception:
            logging.debug("Hold verdict failed for %s.", alert.symbol, exc_info=True)
            return None

    @staticmethod
    def _apply_hold_caption(alert: BounceAlert, verdict) -> None:
        """Re-caption a kept row with what is true NOW, not at alert time.

        The feed row keeps the words it was born with - it is a record of what
        was said - while the review header, rebuilt on every render, stops
        asserting "holding highs" about a name that is merely still inside its
        fifteen minutes.
        """
        if verdict.hold.reason == regime_pause_hold.UNMEASURABLE:
            return
        alert.trigger = f"{REGIME_PAUSE_TRIGGER_PREFIX} \u00b7 {verdict.hold.describe()}"

    def _expire_stale_hold_alerts(self) -> None:
        """Drop regime-pause rows whose claim has gone stale.

        Trader rule, 2026-08-21: a "holding highs" row is good for fifteen
        minutes and is then deleted UNLESS the name keeps making new highs. The
        claim MRK carried that morning was over an hour past true by the time
        it was read.

        Deletion is from the QUEUE only. The alert list, the review-event
        stream and the tracker's outcome rows already hold the row and are not
        consulted here - the trader's explicit call, so the forward record of
        whether stale calls were any good stays measurable. A ``hold_expired``
        event is written for the same reason.

        Rides the 30s chart tick rather than owning a timer, and is connected
        after the bar refresh so it measures bars that just landed.
        """
        expired: list[BounceAlert] = []
        # EXACTLY ONE evaluation per alert per tick. `survives` has side
        # effects - it rewrites the caption and writes a `hold_expired` review
        # event - and the current alert used to be run through it a second time
        # after the queue filter, so an alert that was both queued and on screen
        # produced two events and two caption mutations on the tick it expired.
        # The verdicts are computed first, into a dict keyed by identity, and
        # every consumer below reads from that dict.
        verdicts: dict[int, bool] = {}

        def survives(alert: BounceAlert) -> bool:
            key = id(alert)
            if key in verdicts:
                return verdicts[key]
            verdicts[key] = True  # provisional, so a re-entrant read is honest
            if not is_regime_pause_alert(alert):
                return True
            verdict = self._hold_verdict_for(alert)
            if verdict is None:
                return True
            self._apply_hold_caption(alert, verdict)
            if verdict.keep:
                return True
            expired.append(alert)
            self._record_review_event(
                "hold_expired",
                alert=alert,
                queue_len=len(self._review_queue),
                detail={
                    "reason": verdict.reason,
                    "distance_atr": verdict.hold.distance_atr,
                    "bars_since_extreme": verdict.hold.bars_since_extreme,
                },
            )
            verdicts[key] = False
            return False

        queue_before = len(self._review_queue)
        self._review_queue = [alert for alert in self._review_queue if survives(alert)]
        hidden_before = len(self._hidden_inside_range)
        self._hidden_inside_range = {
            symbol: alert
            for symbol, alert in self._hidden_inside_range.items()
            if survives(alert)
        }
        current = self._current_review_alert
        current_expired = current is not None and not survives(current)
        if hidden_before != len(self._hidden_inside_range):
            self.chart_review.set_hidden_count(len(self._hidden_inside_range))
        if current_expired:
            # The chart in front of the trader just stopped being true. Move on
            # to the next one exactly as a retire does.
            self._advance_review_queue()
        elif queue_before != len(self._review_queue):
            self.chart_review.set_queued_count(len(self._review_queue))
        if expired:
            logging.info(
                "Regime-pause rows expired (stale hold): %s",
                ", ".join(alert.symbol for alert in expired),
            )

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

    def _on_bars_refreshed(self, symbol: str) -> None:
        """Repaint when a refetch lands for the alert currently on the chart."""
        alert = self._current_review_alert
        if alert is None:
            return
        if str(symbol or "").strip().upper() != str(alert.symbol or "").strip().upper():
            return
        self._refresh_review_chart()

    def _refresh_stale_queue_bars(self) -> None:
        """30s tick: refetch M5 for the chart on screen and the next few queued.

        The bot's cache is only rewritten when the scan loop reaches a symbol,
        so an alert opened twenty minutes after it fired otherwise charts its
        scan-time bars. Bounded on purpose: IB allows ~60 historical requests
        per 10 minutes and the champion scan needs that budget, so this covers
        the displayed alert plus a short lookahead, behind a per-symbol
        cooldown - never the whole queue.

        Display-only. The refetched bars go to the chart, never into the bot's
        detector-facing cache (plan.md sec 5).
        """
        try:
            from ui.services.chart_bar_refresh import (
                DEFAULT_LOOKAHEAD,
                shared_refresh_service,
            )

            symbols: list[str] = []
            if self._current_review_alert is not None:
                symbols.append(self._current_review_alert.symbol)
            symbols.extend(
                queued.symbol
                for queued in self._review_queue[:DEFAULT_LOOKAHEAD]
                if queued.symbol
            )
            if not symbols:
                return
            bot = self._current_bot()
            if bot is None:
                return
            shared_refresh_service().refresh_if_stale(
                symbols, lambda sym: bot.m5_chart_bars(sym, max_sessions=2), bot
            )
        except Exception:
            # Display refresh only - it must never break the watch tick that
            # shares this timer.
            pass

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
            any_bounce_armed=self.any_bounce_armed_for(alert.symbol),
            mover_state=self._review_badge_state(alert),
            guidance_text=guidance.summary_text(),
            in_focus=self._alert_is_focus(alert),
            auto_adopted=self._alert_is_auto_adopted(alert),
        )

    def _alert_is_auto_adopted(self, alert: BounceAlert) -> bool:
        """Whether this chart's name is an M5 Focus entry the machine adopted.

        Absence of a marker - including no focus service at all - reads as
        user-entered, which is what keeps the scoped removal off the trader's
        own names.
        """
        service = self.focus_service
        if service is None or not alert.symbol:
            return False
        checker = getattr(service, "is_auto_adopted", None)
        if not callable(checker):
            return False
        try:
            side = str(alert.side or "").strip().lower()
            sides = ("long", "short") if side not in ("long", "short") else (side,)
            return any(checker(alert.symbol, one, "m5") for one in sides)
        except Exception:
            return False

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

    def _advance_after_like(self, alert: BounceAlert) -> None:
        """A LIKE moves to the next chart and does nothing else (R9.2).

        Everything this function deliberately does NOT do was previously done
        to every liked symbol, because a like was routed through
        ``_remove_review_alert_for_today``: it does not touch
        ``_ignored_symbols``, so the name keeps alerting and keeps reaching the
        hourly D1 phone push; it does not drop an auto-adopted Focus pick; it
        does not sweep the symbol's other queued alerts. It also does not
        place anything - the capture rail is analysis-only and the explicit
        Focus verb remains the one thing that places.

        What it does record is ``like_advance``, which
        ``review_learning.TAKE_ACTIONS`` reads as positive engagement. The old
        route wrote ``remove_today``, and ``REJECT_ACTIONS`` scored 40 of the
        window's 52 likes as dismissals.
        """
        if alert is None or not alert.symbol:
            return
        self._record_review_event(
            "like_advance",
            alert=alert,
            dwell_ms=self._review_dwell_ms(alert.symbol),
            queue_len=len(self._review_queue),
        )
        self.statusChanged.emit(
            f"♥ {alert.symbol}: liked and claimed; it keeps alerting as normal."
        )
        self._advance_review_queue()

    def _add_review_alert_to_focus(self, alert: BounceAlert) -> None:
        # Unified verb row (2026-07-31): the add button's "yes" for a DESK
        # auto pick is the watchlist, not Focus.
        if is_auto_pick_alert(alert):
            self._resolve_auto_pick(alert, True)
            return
        # Faded walkthrough: the primary verb RESTORES the pick, with a fresh
        # ten-session clock. A restore is not a fade-proof.
        if alert.tag == FOCUS_FADED_TAG:
            self._restore_faded_review_alert(alert)
            return
        # Focus walkthrough: the pick is already in Focus - "keep" just
        # records the verdict and walks on. It also RESETS the fade clock:
        # the trader looking at the chart and saying "keep" is the strongest
        # statement of interest the desk ever gets (A3).
        if alert.tag == FOCUS_REVIEW_TAG:
            self._note_focus_activity(alert.symbol, reason="kept_in_focus")
            self._record_review_event(
                "focus_review_keep",
                alert=alert,
                dwell_ms=self._review_dwell_ms(alert.symbol),
                queue_len=len(self._review_queue),
            )
            self.statusChanged.emit(f"★ {alert.symbol}: kept in Focus.")
            self._advance_review_queue()
            return
        if self.focus_service is None or not alert.symbol:
            return
        # On a pick that is already the trader's, the primary slot is the
        # removal verb (see AlertChartReview.set_alert): drop it everywhere,
        # exactly as the Focus walkthrough's dismiss does.
        if self._alert_is_focus(alert):
            self._remove_alert_from_focus(alert, origin="d1_focus_chart")
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

    def _veto_but_day_trade(self, alert: BounceAlert) -> None:
        """Trader vetoed the D1 chart and still wants the name for the day.

        Trader, 2026-08-20: "it may be a shit D1 chart but its a good
        daytrade." The veto row is already on disk - the rail wrote it before
        emitting - so this only does the two things the rail is not allowed to
        do: place the name on M5 Focus (this panel owns that store; the rail
        has never written a list and still does not), then retire the chart
        from today's queue exactly as "Not today" does.

        Order matters and is not incidental: retiring the chart is what drops
        the alert object the placement needs, so the placement goes first. A
        failed placement still retires the chart, because the veto has already
        been recorded and leaving the name on screen would invite a second one.

        The Focus entry carries NO auto-pick marker, so it is the trader's own
        - "Not today" and the desync repair cannot reach it (packet R2
        provenance rule).
        """
        if alert is None:
            return
        added = False
        if self.focus_service is not None and alert.symbol:
            side = "short" if alert.side == "SHORT" else "long"
            try:
                added = bool(
                    self.focus_service.add(
                        alert.symbol,
                        side,
                        "m5",
                        origin="veto_day_trade",
                        context=alert.raw_text,
                    )
                )
            except Exception:
                logging.warning(
                    "Veto day-trade: M5 Focus add failed for %s.",
                    alert.symbol,
                    exc_info=True,
                )
                added = False
        self._record_review_event(
            "veto_day_trade",
            alert=alert,
            dwell_ms=self._review_dwell_ms(alert.symbol),
            queue_len=len(self._review_queue),
            detail={"category": "m5", "added": added},
        )
        self.statusChanged.emit(
            f"✕ {alert.symbol}: D1 vetoed, added to M5 Focus for today."
            if added
            else f"✕ {alert.symbol}: D1 vetoed - M5 Focus unchanged (already there or unavailable)."
        )
        self._remove_review_alert_for_today(alert)

    def _remove_alert_from_focus(self, alert: BounceAlert, *, origin: str) -> None:
        """Delete the charted name from Focus Picks and walk on."""
        removed = 0
        if self.focus_service is not None:
            try:
                removed = int(
                    self.focus_service.remove_everywhere(
                        alert.symbol, origin=origin, context=alert.raw_text
                    )
                )
            except Exception:
                removed = 0
        self._record_review_event(
            "focus_remove",
            alert=alert,
            dwell_ms=self._review_dwell_ms(alert.symbol),
            queue_len=len(self._review_queue),
            detail={"entries_removed": removed, "origin": origin},
        )
        self.statusChanged.emit(
            f"✕ {alert.symbol}: removed from Focus Picks "
            f"({removed} entr{'y' if removed == 1 else 'ies'}; "
            "focus-injected watchlist lines went with it)."
            if removed
            else f"{alert.symbol}: was not in Focus Picks anymore."
        )
        self._advance_review_queue()

    # ------------------------------------------------------------------
    # DESK-mode auto-populate picks: chart first, watchlist only on approval.
    def _poll_auto_pick_pending(self) -> None:
        """Land newly staged auto-populate picks straight in M5 Focus for today.

        Trader rule 2026-08-05, replacing the 2026-07-31 chart-approval queue:
        "just add the auto picks into the M5 focus for today and then I will
        prune them out manually - it's quicker than adding them in and then
        seeing their alerts." Approving one at a time meant a pick produced no
        alerts until it had been reviewed, which is backwards: the picks are
        already gated (PDH/PDL break, daily trend, score >= 1.25), so the
        cheaper direction is to take them all and cull.

        M5 Focus is the right home rather than the bare watchlist because it
        is already day-scoped - tomorrow's first store load clears the list AND
        un-injects it from longs/shorts.txt, so "for today" needs no new
        expiry. Pruning a name from Focus removes the watchlist line with it,
        so a pruned pick stops alerting entirely.

        With no Focus service (tests) this falls back to the old approval
        queue - the picks must not silently vanish.

        AWAY and EVENING both refuse adoption outright (trader rule
        2026-08-14). Nobody is at the desk to prune - away, or asleep - so a
        name adopted at 09:00 would alert unwatched all day. Nothing is marked
        seen on a refusal, so the whole day's picks are still pending when the
        trader flips back to DESK and the next poll adopts them together -
        after packet R2's freshness gate has re-checked them, so stale picks
        get dropped rather than adopted.

        DESK keeps immediate adoption (2026-08-05 directive): the trader is
        sitting there and culling is quicker than approving one at a time.
        """
        if self._auto_pick_pending_path is None:
            return
        mode = self._auto_mode_now()
        previous = getattr(self, "_last_seen_auto_mode", None)
        self._last_seen_auto_mode = mode
        if mode in ("AWAY", "EVENING"):
            return
        # The flip back to the desk. Two things are recorded here, and they are
        # deliberately independent (R2.2 - the drain must be explicitly locked,
        # not incidentally so):
        #
        # 1. THE BARRIER. `_desk_flip_at` is the moment the trader came back;
        #    from here on the drain adopts only verdicts stamped at or after it.
        #    Everything measured during the unattended stretch is therefore
        #    unusable no matter which path reaches the drain. Floored to the
        #    second because that is the resolution `gate_checked_at` carries -
        #    a re-measurement finishing inside the same second as the flip
        #    stamps that same second and must count as being after it.
        # 2. THE OWED RE-VERIFICATION. The queue may have been measured half an
        #    hour ago, so re-measure just those symbols before adopting anything
        #    (R2.1). Until that succeeds the drain adopts nothing.
        #
        # The barrier is the lock and the re-verification is how it is cleared.
        # The 2-bar lag bound in `pending_pick_gate_ok` still applies underneath
        # both - defense in depth, no longer the only thing standing between a
        # stalled feed and an adoption.
        if previous in ("AWAY", "EVENING"):
            self._desk_flip_at = datetime.now().replace(microsecond=0)
            # The generation is the flip's identity; the floored timestamp
            # above is only the verdict barrier. Kept separate deliberately -
            # two flips inside one second share a timestamp but never a
            # generation, so an older in-flight run can never answer for the
            # newer return.
            self._desk_flip_generation += 1
            self._reverify_failures = 0
            self._reverify_retry_at = datetime.now()
        if self._reverify_running:
            return
        if self._reverify_retry_at is not None:
            # A re-verification is owed. A failed one waits out its retry delay
            # here rather than falling through to the ordinary stored-verdict
            # drain: those verdicts predate the flip, and "the barrier would
            # have refused them anyway" is not a reason to try.
            if datetime.now() < self._reverify_retry_at:
                return
            self._reverify_retry_at = None
            self._start_pending_reverify()
            return
        try:
            from autopilot_core import load_auto_populate_pending_picks

            payload = load_auto_populate_pending_picks(self._auto_pick_pending_path)
        except Exception:
            return
        day = str(payload.get("date") or "")
        adopted: list[str] = []
        refused: list[str] = []
        #: Picks this cycle left staged because the batch filled up. They are
        #: still pending, so the next tick adopts them.
        deferred = 0
        # One cycle, one traceback (see `_pending_pick_gate_ok`).
        self._gate_check_errors = 0
        self._gate_check_error_reason = ""
        for side_key, side_label in (("long", "LONG"), ("short", "SHORT")):
            entries = payload.get("pending", {}).get(side_key) or {}
            for symbol, entry in entries.items():
                symbol = str(symbol or "").strip().upper()
                if not symbol or not SYMBOL_RE.fullmatch(symbol):
                    continue
                key = (day, side_key, symbol)
                if key in self._auto_picks_enqueued:
                    continue
                entry = entry if isinstance(entry, dict) else {}
                # The adoption-time re-check (packet R2). A pick can sit in the
                # queue for a whole AWAY day, so what qualified when it was
                # staged may not qualify now. The verdict is stored by the
                # 30-minute staging refresh rather than measured here: this runs
                # on the GUI thread, and a staged pick is on no watchlist yet,
                # so BounceBot holds no bars for it.
                #
                # A refusal deliberately does NOT mark the pick seen. The next
                # refresh either re-stamps it (it qualifies again) or evicts it,
                # so a stale verdict costs one cycle rather than the pick.
                ok, gate_reason = self._pending_pick_gate_ok(
                    entry, not_before=self._desk_flip_at
                )
                if not ok:
                    refused.append(f"{symbol} ({gate_reason})")
                    continue
                if len(adopted) >= AUTO_ADOPT_BATCH_LIMIT:
                    # The batch is full. Leave this pick STAGED and unseen so
                    # the next cycle finds it exactly as this one did - it is
                    # deferred by a few seconds, never refused and never lost.
                    # Counted against adoptions rather than iterations, so a
                    # day the gate refuses most of the queue still adopts a
                    # full batch of the ones that qualify.
                    deferred += 1
                    continue
                self._auto_picks_enqueued.add(key)
                reason = str(entry.get("reason") or "auto-populate pick")
                score = entry.get("score")
                if self._adopt_auto_pick_into_focus(symbol, side_key, entry, reason):
                    # Only a real add counts as adopted. A name already in Focus
                    # is resolved but was not taken over - saying "added" would
                    # claim ownership of the trader's own pick in the status
                    # line as well as in the sidecar.
                    if getattr(self, "_last_adoption_outcome", "adopted") == "adopted":
                        adopted.append(symbol)
                    continue
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
        if self._gate_check_errors:
            # The summary the flood used to bury. WARNING, not INFO: every pick
            # in this cycle was refused for a reason that is a fault in the
            # desk, not a judgement about the picks.
            logging.warning(
                "Focus gate check unavailable for %d staged pick(s) this cycle; "
                "all refused (%s). One traceback logged above.",
                self._gate_check_errors,
                self._gate_check_error_reason,
            )
        if refused:
            # Logged, not surfaced: the trader asked for eviction to be silent,
            # and a refusal is the same event one step later. It has to be
            # reconstructable afterwards, which is what the log is for.
            logging.info(
                "Focus gate refused %d staged pick(s) at adoption: %s",
                len(refused),
                ", ".join(refused[:8]),
            )
        if deferred:
            logging.info(
                "Focus adoption batch full (%d); %d staged pick(s) deferred to a "
                "later cycle. Nothing was dropped.",
                AUTO_ADOPT_BATCH_LIMIT,
                deferred,
            )
        if adopted:
            more = f" ({deferred} more still queued)" if deferred else ""
            self.statusChanged.emit(
                f"{len(adopted)} auto pick(s) added to M5 Focus for today "
                f"({', '.join(adopted[:8])}{'...' if len(adopted) > 8 else ''}){more} - "
                "prune with Review ▶ on the Focus board."
            )

    def _start_pending_reverify(self) -> None:
        """Re-measure the queued picks on a worker, then drain from that.

        Off the GUI thread because it fetches bars. Single-flight: a second
        flip while one is running is ignored rather than stacking fetches.

        A failure leaves every pick staged and adopts nothing, then RETRIES on a
        later poll (R2.2). It deliberately does not hand back to the ordinary
        drain: the flip barrier would refuse those verdicts anyway, and an
        attempt that silently stops trying looks exactly like one that
        succeeded. After FLIP_REVERIFY_MAX_ATTEMPTS the fast path gives up and
        the ordinary 30-minute staging refresh becomes the recovery - it stamps
        post-flip verdicts, which the barrier accepts.

        The cost of a bad fetch is therefore a delay, never a stale adoption:
        one cycle is cheaper than a breakout that stopped being one twenty
        minutes ago.
        """
        import threading

        self._reverify_running = True
        # Which flip this attempt is answering. A DESK -> AWAY -> DESK round
        # trip while it runs owes a NEW measurement, and this run's success
        # must not clear that debt: its bars predate the second flip, so the
        # barrier would refuse everything it stamped and the queue would sit
        # unadopted until the next 30-minute refresh with the trader watching.
        # The generation counter is the identity, never the flip timestamp -
        # two returns inside one second share the (second-floored) timestamp,
        # and comparing it let the older run clear the newer debt.
        started_for = self._desk_flip_generation

        def worker() -> None:
            outcome = "ok"
            try:
                from autopilot_core import reverify_pending_picks

                reverify_pending_picks(pending_path=self._auto_pick_pending_path)
            except Exception as exc:
                outcome = str(exc) or exc.__class__.__name__
                logging.warning(
                    "Pending-pick re-verification failed; picks stay staged.",
                    exc_info=True,
                )
            finally:
                # Bookkeeping BEFORE the single-flight flag drops: a poll that
                # sees `_reverify_running` False has to already see whether
                # another attempt is owed, or it would drain in that gap.
                stale_run = self._desk_flip_generation != started_for
                if outcome == "ok":
                    self._reverify_failures = 0
                    if stale_run:
                        # A newer flip landed mid-flight: owe it an attempt now.
                        self._reverify_retry_at = datetime.now()
                    else:
                        self._reverify_retry_at = None
                elif stale_run:
                    # This failure belongs to a superseded flip. The newer
                    # return owes its own attempt with its own full budget -
                    # the flip handler already reset the failure count, and
                    # spending it here would shorten a debt this run was
                    # never answering.
                    self._reverify_retry_at = datetime.now()
                else:
                    self._reverify_failures += 1
                    self._reverify_retry_at = (
                        None
                        if self._reverify_failures >= FLIP_REVERIFY_MAX_ATTEMPTS
                        else datetime.now()
                        + timedelta(seconds=FLIP_REVERIFY_RETRY_SECONDS)
                    )
                self._reverify_running = False
            if outcome == "ok":
                # Re-enter the poll now that the verdicts are current. Queued
                # onto the GUI thread: everything downstream touches widgets.
                QTimer.singleShot(0, self._poll_auto_pick_pending)
                return
            retrying = self._reverify_retry_at is not None
            message = (
                "Auto picks left staged - could not re-check them against the "
                f"current tape ({outcome}). "
                + (
                    f"Retrying in {FLIP_REVERIFY_RETRY_SECONDS}s."
                    if retrying
                    else "They adopt after the next staging refresh re-measures them."
                )
            )
            QTimer.singleShot(0, lambda: self.statusChanged.emit(message))

        threading.Thread(target=worker, name="focus-pick-reverify", daemon=True).start()

    def _drain_focus_desync_requests(self) -> None:
        """Reconcile Focus with watchlist lines BounceBot's VWAP rule cut.

        The triple-VWAP invalidation deletes a raw watchlist line without
        telling `FocusPickStore`, so a Focus-listed name could sit on the board
        looking healthy while nothing scanned it (packet R2 A.3.4).

        Two branches, and the difference is the whole point:

        - the machine's own pick is removed from Focus, scoped to that one M5
          entry, so the board stops showing a pick that has been invalidated;
        - a name the TRADER typed is left exactly where it is and the mismatch
          is surfaced instead. Silently deleting it would be the automatic
          removal of a user-entered name that plan.md sec 5 forbids, and
          silently keeping it would leave them trusting a dead entry.
        """
        service = self.focus_service
        if service is None:
            return
        try:
            from autopilot_core import take_focus_desync_requests

            requests = take_focus_desync_requests()
        except Exception:
            return
        if not requests:
            return
        dropped: list[str] = []
        stranded: list[str] = []
        for row in requests:
            symbol = str(row.get("symbol") or "").strip().upper()
            side = str(row.get("side") or "").strip().lower()
            if not symbol or side not in ("long", "short"):
                continue
            try:
                if not service.is_focus(symbol, side, "m5"):
                    continue  # not a Focus name; the cut needs no reconciling
                if service.remove_if_auto_adopted(
                    symbol, side, "m5", reason="triple-VWAP invalidation", origin="auto_pick"
                ):
                    dropped.append(symbol)
                    logging.info(
                        "Focus desync: auto pick %s (%s) removed - %s",
                        symbol, side, row.get("reason") or "watchlist line cut",
                    )
                else:
                    stranded.append(f"{symbol} ({side})")
                    logging.warning(
                        "Focus desync: %s (%s) is YOUR Focus pick and its watchlist "
                        "line was cut by %s - it is no longer being scanned. Left in "
                        "Focus; re-add it to the watchlist to resume scanning.",
                        symbol, side, row.get("reason") or "the VWAP rule",
                    )
            except Exception:
                logging.warning("Focus desync handling failed for %s.", symbol, exc_info=True)
        if stranded:
            self.statusChanged.emit(
                f"⚠ {', '.join(stranded[:6])}{'...' if len(stranded) > 6 else ''}: "
                "your Focus pick(s) lost their watchlist line to the VWAP rule and "
                "are no longer being scanned. Still in Focus - re-add to resume."
            )
        elif dropped:
            self.statusChanged.emit(
                f"{len(dropped)} invalidated auto pick(s) removed from M5 Focus "
                f"({', '.join(dropped[:8])}{'...' if len(dropped) > 8 else ''})."
            )

    def _pending_pick_gate_ok(
        self, entry: dict, *, not_before: datetime | None = None
    ) -> tuple[bool, str]:
        """Thin wrapper so the import stays local to the poll (headless paths
        construct this panel without `autopilot_core` on the path).

        `not_before` is the flip barrier: after a return to the desk, only a
        verdict stamped at or after the flip may be adopted.

        Failure stays FAIL-CLOSED and stays loud; what is bounded is the
        VOLUME. Before 2026-08-19 this logged a full traceback per pick, so a
        single systematic fault (the naive/aware gate crash) wrote 121
        tracebacks every 30 seconds, rotated the log, and nearly destroyed the
        evidence needed to diagnose it. The first failure of each poll cycle
        carries the traceback; the rest are counted and reported once by the
        caller.
        """
        try:
            from autopilot_core import pending_pick_gate_ok

            return pending_pick_gate_ok(entry, not_before=not_before)
        except Exception as exc:
            # Fail CLOSED: an unverifiable pick is not an approved pick.
            self._gate_check_errors += 1
            self._gate_check_error_reason = f"{type(exc).__name__}: {exc}"
            if self._gate_check_errors == 1:
                logging.warning(
                    "Focus gate check unavailable; refusing adoption.", exc_info=True
                )
            return False, "gate check unavailable"

    def _adopt_auto_pick_into_focus(
        self, symbol: str, side: str, entry: dict, reason: str
    ) -> bool:
        """Add one staged pick to M5 Focus and retire its proposal.

        Returns True when the proposal is RESOLVED - adopted, or found to be
        the trader's already - so the caller knows not to queue a review alert.
        `self._last_adoption_outcome` distinguishes the two for the status line.

        Writes through the STORE, not `FocusService.add`: the service logs every
        add to the trader-verdict feedback JSONL as a "like", and a machine
        adding 30 names is not the trader liking 30 names. The store's listener
        still fires focusChanged, so every surface refreshes, and the action is
        logged to the review-decision ledger instead.

        **The marker is written only when `add()` actually added something.**
        `add()` returns False for a name already on the list, and marking that
        entry would relabel the TRADER's pick as machine-owned - after which
        "Not today" and the desync repair could both remove it. The sequence is
        real: AWAY stages SYM, the trader adds SYM by hand, the DESK flip
        drains, and their entry silently changes owner. Absence of a marker is
        what makes a name untouchable, so it is never written speculatively.
        """
        self._last_adoption_outcome = "failed"
        store = getattr(self.focus_service, "store", None)
        if store is None:
            return False
        try:
            added = bool(store.add(symbol, side, "m5"))
            marker_writer = getattr(store, "mark_auto_adopted", None)
            if added:
                # Provenance (packet R2): this marker is the ONLY thing that
                # makes the entry removable by "Not today" or by the desync
                # repair. An entry without one is the trader's, untouchable by
                # both.
                if callable(marker_writer):
                    marker_writer(
                        symbol,
                        side,
                        "m5",
                        staged_at=str(entry.get("staged_at") or ""),
                        reason=reason,
                    )
                self._last_adoption_outcome = "adopted"
            else:
                # Already on the list. If a marker exists it is a previous
                # adoption of ours and stays as it is; if none exists the entry
                # is the trader's and must not acquire one. Either way the
                # proposal is finished - the name is already in Focus.
                reader = getattr(store, "is_auto_adopted", None)
                already_ours = bool(reader(symbol, side, "m5")) if callable(reader) else False
                self._last_adoption_outcome = (
                    "already_auto" if already_ours else "already_trader_owned"
                )
                if not already_ours:
                    logging.info(
                        "Auto pick %s (%s) is already YOUR Focus entry - proposal "
                        "retired without claiming ownership.",
                        symbol,
                        side,
                    )
        except Exception:
            logging.warning("Auto pick %s could not be added to Focus.", symbol, exc_info=True)
            self._last_adoption_outcome = "failed"
            return False
        if self._auto_pick_pending_path is not None:
            try:
                from autopilot_core import resolve_auto_populate_pick

                # Accepted, but Focus owns the watchlist line it just injected -
                # a second owner here would let one side delete the other's entry.
                resolve_auto_populate_pick(
                    symbol,
                    side,
                    True,
                    # The ledger records WHICH outcome retired the proposal, so
                    # "the machine adopted it" and "it was already the trader's"
                    # are never confused when reading back a session.
                    decision_label=(
                        "auto_focus"
                        if self._last_adoption_outcome == "adopted"
                        else f"auto_focus_{self._last_adoption_outcome}"
                    ),
                    write_watchlist=False,
                    pending_path=self._auto_pick_pending_path,
                )
            except Exception:
                logging.warning(
                    "Auto pick %s resolved into Focus but its proposal was not retired.",
                    symbol,
                    exc_info=True,
                )
        self._record_review_event(
            "auto_pick_auto_focus",
            symbol=symbol,
            side=side.upper(),
            detail={
                "auto_pick": dict(entry),
                "reason": reason,
                "outcome": self._last_adoption_outcome,
            },
        )
        return True

    def _regime_pause_day_env(self) -> str:
        """The day's directional label, as discovery sees it.

        `resolve_discovery_env` is the ONE definition of "which way is the
        day": BounceBot's live label while it is directional, else the opening
        read recorded at the auto-populate slot (first directional write wins
        for the day). Blank when neither can answer - and blank admits
        nothing, so a row seen before any read exists stays on the queue.
        """
        current = ""
        if self._bounce_service is not None:
            try:
                bot = self._bounce_service.current_bot()
                if bot is not None:
                    current = str(bot.get_market_environment() or "")
            except Exception:
                current = ""
        try:
            from autopilot_core import load_opening_environment, resolve_discovery_env

            return str(resolve_discovery_env(current, load_opening_environment()) or "")
        except Exception:
            return ""

    def _auto_focus_regime_pause(self, alert: BounceAlert) -> bool:
        """Place a with-trend regime-pause row on M5 Focus (trader rule 2026-08-27).

        On 2026-08-27 the trader reviewed 21 "holding highs" charts in nine
        minutes on a bullish open and put twelve on M5 Focus by hand while 74
        more charts waited. The rule: a swing LONG holding highs on a bullish
        day, or a swing SHORT pressing lows on a bearish day, is added to M5
        Focus by the machine and skips the review chart. The mirror cases and
        a non-directional day are untouched (`regime_pause_focus`).

        Returns True when the row is RESOLVED - placed, or already the
        machine's own entry - so `add_alert` knows not to queue it. False for
        everything else, including a Focus name the TRADER owns (their chart
        shows as it always did) and any failure: this must never be the
        reason a chart went missing, so it fails open onto the old path.

        DESK only, like auto-pick adoption (R1 matrix): AWAY and EVENING have
        nobody present to prune what the machine adopted, and OFF adopts
        nothing. Writes through the STORE, not `FocusService.add`, for the
        same reason `_adopt_auto_pick_into_focus` does - a machine adding a
        name is not the trader liking it - and stamps the auto-pick marker so
        "Not today" and the desync repair can reach the entry. The marker is
        written only when `add()` actually added: an existing unmarked entry
        is the trader's and must not change owner.
        """
        if not is_regime_pause_alert(alert):
            return False
        if not alert.symbol or not SYMBOL_RE.fullmatch(alert.symbol):
            return False
        if self._auto_mode_now() != "DESK":
            return False
        store = getattr(self.focus_service, "store", None)
        if store is None:
            return False
        env = self._regime_pause_day_env()
        side = focus_side_for(env, alert.side)
        if side is None:
            return False
        try:
            added = bool(store.add(alert.symbol, side, "m5"))
            if added:
                marker_writer = getattr(store, "mark_auto_adopted", None)
                if callable(marker_writer):
                    marker_writer(
                        alert.symbol,
                        side,
                        "m5",
                        staged_at=str(alert.time_text or ""),
                        reason=f"{alert.trigger} on a {day_bias(env)} day ({env})",
                    )
                outcome = "adopted"
            else:
                reader = getattr(store, "is_auto_adopted", None)
                already_ours = bool(reader(alert.symbol, side, "m5")) if callable(reader) else False
                outcome = "already_auto" if already_ours else "already_trader_owned"
        except Exception:
            logging.warning(
                "Regime-pause row %s could not be placed on M5 Focus; queued instead.",
                alert.symbol,
                exc_info=True,
            )
            return False
        self._record_review_event(
            "regime_pause_auto_focus",
            alert=alert,
            queue_len=len(self._review_queue),
            detail={"env": env, "focus_side": side, "outcome": outcome},
        )
        if outcome == "adopted":
            self.statusChanged.emit(
                f"★ {alert.symbol}: {alert.trigger} on a {day_bias(env)} day - "
                f"added to M5 Focus {side}s, no chart to review."
            )
        return outcome in ("adopted", "already_auto")

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

    # ------------------------------------------------------------------
    # Focus picks: desk-side chart walkthrough + automatic D1 interest flags
    # (2026-07-31 user request).
    def review_focus_picks(self) -> None:
        """Queue every current Focus pick onto the review chart.

        Fired by the strength board's "Review ▶" button. Swing picks first
        (the headline bucket), then M5; one chart per symbol; walk them with
        the ordinary verb row. Ignored-for-today names stay out.
        """
        if self.focus_service is None:
            self.statusChanged.emit("Focus review: no Focus store attached.")
            return
        try:
            by_category = self.focus_service.all_focus_by_category()
        except Exception:
            by_category = {}
        queued = 0
        seen: set[str] = set()
        now_text = datetime.now().strftime("%H:%M:%S")
        for category, bucket in (("swing", "Swing"), ("m5", "M5")):
            sides = by_category.get(category) or {}
            for side_key, side_label in (("long", "LONG"), ("short", "SHORT")):
                for symbol in sides.get(side_key) or []:
                    symbol = str(symbol or "").strip().upper()
                    if (
                        not symbol
                        or symbol in seen
                        or symbol in self._ignored_symbols
                        or not SYMBOL_RE.fullmatch(symbol)
                    ):
                        continue
                    seen.add(symbol)
                    self._enqueue_review_alert(
                        BounceAlert(
                            time_text=now_text,
                            symbol=symbol,
                            side=side_label,
                            trigger=f"Focus review · {bucket} {side_key}",
                            timeframe="D1" if category == "swing" else "M5",
                            tag=FOCUS_REVIEW_TAG,
                            raw_text=f"FOCUS REVIEW {symbol} ({bucket} {side_key})",
                        )
                    )
                    queued += 1
        if queued:
            self._record_review_event("focus_review_started", detail={"count": queued})
            self.statusChanged.emit(
                f"Reviewing {queued} Focus pick(s) on the chart - Skip walks to the next."
            )
        else:
            self.statusChanged.emit("Focus review: no Focus picks to show.")

    def _note_focus_activity(self, symbol: str, *, reason: str = "") -> None:
        """Restart one Focus pick's fade clock. Never raises into a poll."""
        if self.focus_service is None:
            return
        try:
            self.focus_service.note_focus_activity(symbol, reason=reason)
        except Exception:
            logging.debug("Focus fade clock not reset for %s", symbol, exc_info=True)

    def run_focus_fade_check(self) -> list:
        """Move Focus picks silent past their window to the faded list.

        Runs on the half-hourly timer and on the day roll. The store owns the
        decision and the writes; this is the caller. A failure here costs the
        housekeeping, never the picks.
        """
        if self.focus_service is None:
            return []
        try:
            faded = self.focus_service.fade_stale_picks()
        except Exception:
            logging.debug("Focus fade check failed", exc_info=True)
            return []
        if faded:
            names = ", ".join(sorted({str(row.get("symbol") or "") for row in faded}))
            self._record_review_event(
                "focus_picks_faded",
                detail={"count": len(faded), "symbols": names},
            )
            self.statusChanged.emit(
                f"{len(faded)} quiet Focus pick(s) faded: {names}. "
                "Open “Faded review” to restore or discard them."
            )
            self.focusFadedChanged.emit()
        return faded

    def _restore_faded_review_alert(self, alert: BounceAlert) -> None:
        side = str(alert.payload.get("faded_side") or "long")
        category = str(alert.payload.get("faded_category") or "m5")
        restored = False
        if self.focus_service is not None:
            try:
                restored = bool(
                    self.focus_service.restore_faded(alert.symbol, side, category)
                )
            except Exception:
                restored = False
        self._record_review_event(
            "faded_review_restore",
            alert=alert,
            dwell_ms=self._review_dwell_ms(alert.symbol),
            queue_len=len(self._review_queue),
            detail={"restored": restored, "side": side, "category": category},
        )
        self.statusChanged.emit(
            f"★ {alert.symbol}: back in {category} Focus with a fresh clock."
            if restored
            else f"{alert.symbol}: it was no longer on the faded list."
        )
        self.focusFadedChanged.emit()
        self._advance_review_queue()

    def _discard_faded_review_alert(self, alert: BounceAlert) -> None:
        side = str(alert.payload.get("faded_side") or "long")
        category = str(alert.payload.get("faded_category") or "m5")
        discarded = False
        if self.focus_service is not None:
            try:
                discarded = bool(
                    self.focus_service.discard_faded(alert.symbol, side, category)
                )
            except Exception:
                discarded = False
        self._record_review_event(
            "faded_review_discard",
            alert=alert,
            dwell_ms=self._review_dwell_ms(alert.symbol),
            queue_len=len(self._review_queue),
            detail={"discarded": discarded, "side": side, "category": category},
        )
        self.statusChanged.emit(f"✕ {alert.symbol}: cleared off the faded list.")
        self.focusFadedChanged.emit()
        self._advance_review_queue()

    def review_faded_picks(self) -> None:
        """Walk the faded list onto the review chart.

        Through `_enqueue_review_alert` - the one door - with its own tag, so
        the restore/discard verbs know which list they are acting on and the
        movers-only filter leaves it alone (a faded pick is by definition one
        that has not been moving).
        """
        if self.focus_service is None:
            self.statusChanged.emit("Faded review: no Focus store attached.")
            return
        try:
            faded = self.focus_service.faded_picks()
        except Exception:
            faded = []
        queued = 0
        now_text = datetime.now().strftime("%H:%M:%S")
        for row in faded:
            symbol = str(row.get("symbol") or "").strip().upper()
            if not symbol or not SYMBOL_RE.fullmatch(symbol):
                continue
            side = str(row.get("side") or "long")
            category = str(row.get("category") or "m5")
            self._enqueue_review_alert(
                BounceAlert(
                    time_text=now_text,
                    symbol=symbol,
                    side="LONG" if side == "long" else "SHORT",
                    trigger=(
                        f"Faded {category} {side} - quiet since "
                        f"{row.get('clock_from') or 'unknown'}"
                    ),
                    timeframe="D1" if category == "swing" else "M5",
                    tag=FOCUS_FADED_TAG,
                    raw_text=f"FADED {symbol} ({category} {side})",
                    payload={"faded_side": side, "faded_category": category},
                )
            )
            queued += 1
        if queued:
            self._record_review_event("faded_review_started", detail={"count": queued})
            self.statusChanged.emit(
                f"Reviewing {queued} faded pick(s) - ★ restores, ✕ discards."
            )
        else:
            self.statusChanged.emit("Faded review: nothing has faded.")

    def _poll_focus_d1_interest(self, now=None) -> None:
        """Flag Focus picks on a D1 PULLBACK - once they have taken out
        yesterday's extreme in their own direction.

        **Pullbacks only** (trader, 2026-09-01, Phase 0.12 A1). Every Focus
        name is implicitly watched for the pullback set - a 15EMA reject, an
        AVWAPE or 1σ bounce - and nothing else. The EXTENSION set (a new 5d or
        20d extreme, a close through an SMA, through the AVWAPE line or
        through 1σ) no longer fires automatically at all: those are the alerts
        that filled the Focus feed with "still going" news about names the
        trader had already seen, and the trader now arms the ones they want
        per symbol.

        That gate is at the flag-GENERATION seam, not a filter downstream. An
        extension kind is never evaluated here, so no extension flag is
        written and nothing has to be suppressed later.

        An armed extension watch is the ONE surviving route, and it is a
        different poll: `_poll_d1_event_watches` evaluates
        `d1_event_watches.json` and is untouched by this rule. Keeping the two
        lanes disjoint is what makes double-firing structurally impossible.

        The 2026-08-05 "one extension event per name per day" rule (FRPT) was
        a ration on this lane; with the lane closed it has nothing left to
        ration and is gone.

        Each (symbol, event) still flags at most once per session; hits land in
        the D1 Focus feed and the chart queue.

        The prev-day gate (trader rule 2026-08-05) is what keeps that set from
        emptying itself into the open: a long inside yesterday's range flags
        nothing, and when it does break out the event window starts THERE, so
        the 09:35 15EMA reject it printed while still below yesterday's high
        never fires. Held names keep their pending kinds - nothing is consumed
        while the gate is shut.
        """
        if self.focus_service is None or self._focus_d1_flags_path is None:
            return
        self._refresh_ignored_market_date()
        try:
            focus = self.focus_service.all_focus()
        except Exception:
            return
        moment = now or datetime.now()
        day_start = datetime(moment.year, moment.month, moment.day)
        hits: list[tuple[str, str, str, object]] = []
        held = 0
        for side_key, side_label in (("long", "LONG"), ("short", "SHORT")):
            for symbol in focus.get(side_key) or []:
                symbol = str(symbol or "").strip().upper()
                if not symbol or symbol in self._ignored_symbols:
                    continue
                d1_bars = self._d1_bars_for(symbol)
                m5_bars = self._m5_bars_for(symbol)
                # Measured every tick even when nothing is pending: the feed
                # gate and the beep read this state for every alert on the
                # name, not just the automatic D1 flags.
                break_open_at = self._update_focus_break_state(
                    symbol, side_key, m5_bars, d1_bars, moment
                )
                if break_open_at is None:
                    held += 1
                    continue
                # A1: the automatic lane is the pullback set. Iterating
                # `D1_PULLBACK_KINDS` rather than filtering `D1_EVENT_KINDS`
                # is the point - an extension kind is never constructed, so it
                # cannot be evaluated, flagged, or suppressed.
                pending_kinds = [
                    kind
                    for kind in D1_EVENT_KINDS
                    if kind in D1_PULLBACK_KINDS
                    and f"{symbol}|{kind}" not in self._focus_d1_flags
                ]
                if not pending_kinds or not d1_bars:
                    continue
                avwape_anchor = None
                if any(kind.startswith("avwape_") for kind in pending_kinds):
                    try:
                        import chart_snapshot

                        avwape_anchor = chart_snapshot.earnings_anchor_date(symbol)
                    except Exception:
                        avwape_anchor = None
                # The window opens at the break, not at midnight: everything
                # the name did while inside yesterday's range stays unflagged.
                armed_at = max(day_start, break_open_at)
                # Item 1b: ten kinds per symbol used to re-enter
                # `d1_event_levels` ten times with identical arguments - a sort
                # of ~490 bars, 5d/20d extremes, three SMAs, an EMA15 recursion
                # and the AVWAP bands, each time. One cache per symbol, built
                # here and dropped when the symbol's loop ends.
                levels_cache: dict = {}
                for kind in pending_kinds:
                    watch = D1EventWatch(symbol=symbol, kind=kind, armed_at=armed_at)
                    try:
                        hit = evaluate_d1_event_watch(
                            watch,
                            m5_bars,
                            d1_bars,
                            now=moment,
                            avwape_anchor=avwape_anchor if kind.startswith("avwape_") else None,
                            levels_cache=levels_cache,
                        )
                    except Exception:
                        hit = None
                    if hit is None:
                        continue
                    self._focus_d1_flags.add(f"{symbol}|{kind}")
                    hits.append((symbol, side_label, kind, hit))
                    # A3: the pick just said something, so its ten-session
                    # fade clock restarts here.
                    self._note_focus_activity(symbol, reason="focus_d1_flag")
        self._focus_gate_held = held
        # Every Focus name has just been re-measured against yesterday's range.
        # Surfaces that show the "moving" flag repaint from here rather than
        # owning a timer of their own (trader rule 2026-08-19).
        self.focusBreakStatesChanged.emit()
        if not hits:
            self._emit_feed_status()
            return
        self._save_focus_d1_flags()
        for symbol, side_label, kind, hit in hits:
            self._record_review_event(
                "focus_d1_flag",
                symbol=symbol,
                side=side_label,
                detail={"kind": kind, "message": hit.message},
            )
            self.add_alert(
                BounceAlert(
                    time_text=datetime.now().strftime("%H:%M:%S"),
                    symbol=symbol,
                    side=side_label,
                    trigger=f"Focus D1 · {hit.message}",
                    timeframe="D1",
                    tag=FOCUS_D1_EVENT_TAG,
                    raw_text=f"FOCUS D1 {symbol} ({side_label}): {hit.message}",
                    is_d1=True,
                    payload={"focus_d1_kind": kind},
                )
            )

    def _save_focus_d1_flags(self) -> None:
        if self._focus_d1_flags_path is None:
            return
        try:
            self._focus_d1_flags = save_day_scoped_flags(
                self._focus_d1_flags,
                self._focus_d1_flags_path,
                market_date=self._ignored_market_date,
            )
        except OSError:
            pass

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

    @staticmethod
    def _m5_source_bars(bot, symbol: str) -> list:
        """The raw series `m5_chart_bars` will read, for cache-keying only.

        Deliberately the same two-key lookup `BounceBot.m5_chart_bars` does, in
        the same order - RRS stores under the qualified key, the confirmation
        fetch under the plain one. It is never the VALUE: the dicts always come
        from `m5_chart_bars` itself, so if this lookup ever diverged the cost
        would be a missed cache hit, not a wrong bar.
        """
        latest = getattr(bot, "latest_bars", None)
        if not isinstance(latest, dict):
            return []
        key = str(symbol or "").strip().upper()
        return latest.get(f"{key}|5 D|5 mins") or latest.get(key) or []

    @staticmethod
    def _m5_source_stamp(source: list) -> tuple:
        """(length, last bar time) - what changes when a bar arrives.

        The series is sometimes replaced (new object, identity catches it) and
        sometimes appended to in place (same object, this catches it).
        """
        if not source:
            return (0, None)
        return (len(source), getattr(source[-1], "dt", None))

    def _m5_bars_for(self, symbol: str, *, sessions: int = 1) -> list:
        """Cached M5 bars for a symbol. Reads memory only; never fetches.

        ``sessions`` is 1 for anything that asks about today. Pass 2 when a
        measure needs warm-up bars that today cannot supply - an ATR(14) needs
        fifteen bars, and forty minutes after the open there are nine.

        Materialized once per (symbol, sessions) per source series. Eight
        timer-driven sites ask for the same bars on the 30s and 60s ticks -
        chart watches, hold expiry, D1 level and event watches, any-bounce
        twice per watch, Focus D1 interest - and each call was rebuilding ~150
        dicts with six float() coercions apiece, on the Qt thread, for ~105
        symbols. Nothing about WHICH bars come back changes: the value is
        always `m5_chart_bars`'s own output.
        """
        bot = None
        if self._bounce_service is not None:
            try:
                bot = self._bounce_service.current_bot()
            except Exception:
                bot = None
        if bot is None:
            return []
        key = (str(symbol or "").strip().upper(), max(1, int(sessions)))
        try:
            source = self._m5_source_bars(bot, symbol)
        except Exception:
            source = []
        stamp = self._m5_source_stamp(source)
        cached = self._m5_bar_dicts.get(key)
        if cached is not None and cached[0] is source and cached[1] == stamp:
            self._m5_bar_dicts.move_to_end(key)
            return cached[2]
        try:
            bars = bot.m5_chart_bars(symbol, max_sessions=key[1]) or []
        except Exception:
            return []
        self._m5_bar_dicts[key] = (source, stamp, bars)
        self._m5_bar_dicts.move_to_end(key)
        while len(self._m5_bar_dicts) > M5_BAR_DICT_CACHE_LIMIT:
            self._m5_bar_dicts.popitem(last=False)
        return bars

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
        # A3: an armed watch firing on a Focus name is the name speaking. Every
        # armed poll builds its alert here, so one call covers all three.
        self._note_focus_activity(watch.symbol, reason="armed_watch_hit")
        resolved = str(getattr(hit, "resolved_side", "") or "").upper()
        side = str(getattr(watch, "side", "") or "")
        if side not in ("LONG", "SHORT"):
            side = resolved if resolved in ("LONG", "SHORT") else "WATCH"
        # An any-bounce hit names the level that actually held; every other
        # watch carries exactly one kind of its own.
        kind = str(getattr(hit, "kind", "") or "") or watch.kind
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

    def chart_symbol(self, symbol: str, *, side: str = "", origin: str = "") -> bool:
        """Put any symbol on the big chart on demand.

        The review pane previously only ever showed what the alert queue handed
        it, so on a quiet tape it sat on "Waiting for the next ticker alert"
        with no way to look at a name. A typed symbol is charted immediately,
        even if it has never alerted and is not in the scan set.

        Typing a symbol also un-ignores it: "Remove for today" would otherwise
        make it silently un-chartable for the rest of the session, which reads
        as the box being broken.

        `side` and `origin` are for callers that know more than the lookup box
        does - the M5 Strength Board knows which of its two tables the row came
        from, and a short charted as a plain WATCH reads as the wrong thesis.
        They are display and provenance only: the chart stays a MANUAL_CHART,
        so it is muted rather than red (nothing fired - the trader was
        looking), it never enters the alert feed, and it is not an alert of any
        kind. Defaults reproduce the lookup box exactly.
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
        source = str(origin or "").strip()
        alert = BounceAlert(
            time_text=datetime.now().strftime("%H:%M:%S"),
            symbol=symbol,
            side=str(side or "WATCH").strip().upper(),
            trigger=f"Charted from {source}" if source else "Charted on demand",
            tag=MANUAL_CHART_TAG,
            raw_text=f"MANUAL CHART {symbol}",
        )
        # Straight to the review pane; never into the alert feed, which is a
        # record of what the scanner said, not of what was looked at.
        self._select_review_alert(alert)
        self.statusChanged.emit(
            f"{symbol}: charted from {source}." if source
            else f"{symbol}: charted on demand."
        )
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

    def _arm_price_alert_from_level(
        self, symbol: str, direction: str, level: float
    ) -> None:
        """Arm a PHONE price alert at a painted D1 level the trader clicked.

        Trader decision, 2026-08-09: arming always routes through the panel
        that owns the store. The chart requests; this panel performs the
        caller-only merge against the desk's single ``PriceAlertService``
        (injected by TradingDeskPanel), so ``price_alerts.json`` keeps exactly
        one writer (plan.md sec 5). The merge is the Focus tab board's, key
        for key - including its deliberate rule that an unchanged level does
        NOT re-arm a side that has already fired; re-arming stays explicit.
        The opposite side, the note and the history are never touched.
        """
        symbol = str(symbol or "").strip().upper()
        try:
            level = float(level)
        except (TypeError, ValueError):
            return
        if not symbol or direction not in ("above", "below") or not level > 0:
            return
        service = getattr(self, "price_alert_service", None)
        if service is None:
            self.statusChanged.emit(
                f"{symbol}: no price-alert service on this desk - arm the "
                "cross on the Focus tab instead."
            )
            return
        entries = service.entries()
        entry = next((row for row in entries if row.get("symbol") == symbol), None)
        if entry is None:
            entry = {
                "symbol": symbol,
                "above": level if direction == "above" else None,
                "below": level if direction == "below" else None,
                "armed_above": direction == "above",
                "armed_below": direction == "below",
                "note": "",
                "history": [],
            }
            entries.append(entry)
        else:
            old_level = entry.get(direction)
            entry[direction] = level
            if old_level != level:
                entry[f"armed_{direction}"] = True
                # A2: arming restarts the trading-day expiry clock - the
                # board's merge stamps this too, and without it a level
                # re-armed from the chart would still carry the date that
                # expired it and be disarmed again on the next poll. The
                # unchanged-level branch deliberately leaves the stamp alone.
                import price_alerts

                price_alerts.mark_armed_now(entry)
        if not service.save_entries(entries):
            self.statusChanged.emit(
                f"{symbol}: phone price alert NOT saved - the price-alert "
                "store refused the write on this machine."
            )
            return
        if entry.get(f"armed_{direction}"):
            self.statusChanged.emit(
                f"{symbol}: phone price alert armed - cross {direction} "
                f"{level:.2f}. It fires once, pushes to your phone, then "
                "stays off until you re-arm it."
            )
        else:
            self.statusChanged.emit(
                f"{symbol}: cross {direction} {level:.2f} kept, still "
                "disarmed - it already fired at this level. Re-arm it on the "
                "Focus tab."
            )

    def armed_levels_for(self, symbol: str) -> list:
        symbol = str(symbol or "").strip().upper()
        return [watch for watch in self._d1_level_watches if watch.symbol == symbol]

    def _build_journal_tab(self):
        """The in-session market-journal note (R10.H).

        Deliberately small: a timeframe, a box, a button and a status line.
        The sit-down review lives on the left-nav Market Journal page; this is
        for the thought you have at 10:40 and would otherwise lose.
        """
        from PySide6.QtWidgets import QComboBox, QPlainTextEdit

        import market_journal
        from ui.services.market_journal_service import shared_journal_service

        # The SHARED service, not a second instance. Both were writing the same
        # file correctly, but a note typed here never told the left-nav Market
        # Journal page to refresh - its `entryWritten` came from an object that
        # page had never heard of. One writer is what the R10.H docstring
        # always claimed; this is what makes it true.
        self.market_journal_service = shared_journal_service()
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)

        self._journal_timeframe = QComboBox()
        self._journal_timeframe.addItems(list(market_journal.TIMEFRAMES))
        self._journal_timeframe.setCurrentText(market_journal.TIMEFRAME_M5)
        self._journal_text = QPlainTextEdit()
        self._journal_text.setPlaceholderText(
            "What the tape is doing, and what you make of it. Ctrl+Enter saves."
        )
        self._journal_status = QLabel("")
        self._journal_status.setWordWrap(True)
        save = QPushButton("Save entry (Ctrl+Enter)")
        save.clicked.connect(self._commit_journal_entry)

        row = QHBoxLayout()
        row.addWidget(QLabel("Timeframe"))
        row.addWidget(self._journal_timeframe)
        row.addStretch(1)
        row.addWidget(save)

        layout.addWidget(QLabel("Market journal - this session"))
        layout.addWidget(self._journal_text, 1)
        layout.addLayout(row)
        layout.addWidget(self._journal_status)

        self.market_journal_service.statusChanged.connect(self._journal_status.setText)
        shortcut = QShortcut(QKeySequence("Ctrl+Return"), container)
        shortcut.activated.connect(self._commit_journal_entry)
        self._journal_shortcut = shortcut
        return container

    def _commit_journal_entry(self) -> None:
        from datetime import date

        import market_journal

        # The chart in front of the trader, when there is one. A stale symbol
        # would be worse than none: it would assert a link they never made.
        current = getattr(self, "_current_review_alert", None)
        symbol = str(getattr(current, "symbol", "") or "").strip().upper()

        result = self.market_journal_service.write_entry(
            text=self._journal_text.toPlainText(),
            session_date=date.today().isoformat(),
            timeframe=self._journal_timeframe.currentText(),
            symbols=[symbol] if symbol else [],
            origin=market_journal.ORIGIN_DESK_TAB,
        )
        if result.get("ok"):
            self._journal_text.clear()
            self._capture_journal_charts(result.get("entry") or {}, symbol)

    def journal_chart_bars(self, symbol: str) -> tuple[list, list]:
        """(M5, D1) cached bars for one symbol, for a Market Journal capture.

        Public because the auto-mode flip capture lives in `ui.app` and must
        not reach into this panel's private accessors. Cache reads only - the
        same two the D1 watch poll already makes - so it never fetches and is
        safe from the Qt thread.
        """
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return [], []
        return self._m5_bars_for(symbol, sessions=2), self._d1_bars_for(symbol)

    def _capture_journal_charts(self, entry: dict, symbol: str) -> None:
        """Store the tape this note was written against.

        AFTER the entry is on disk, never before: a note must not wait on a
        chart, and a capture that fails leaves an entry that is honestly
        chartless rather than a thought that was lost.

        Every bar list here is a CACHE read - `_m5_bars_for` reads
        `latest_bars` and `_d1_bars_for` reads the chart service's memoized
        dicts - so nothing fetches and nothing blocks. The trimming, the digest
        and both file writes happen on the service's worker.

        Capture-side only. No alert, tier, fold, digest, queue, score or
        detector behaviour is touched by this method or its caller.
        """
        import market_journal_capture

        entry_id = str(entry.get("entry_id") or "")
        if not entry_id:
            return
        benchmark = market_journal_capture.BENCHMARK_SYMBOL
        symbol_m5, symbol_d1 = self.journal_chart_bars(symbol)
        benchmark_m5, benchmark_d1 = self.journal_chart_bars(benchmark)
        try:
            self.market_journal_service.capture_charts(
                entry_id=entry_id,
                symbol=symbol,
                reason=market_journal_capture.REASON_ENTRY,
                m5_bars=symbol_m5,
                d1_bars=symbol_d1,
                benchmark_m5=benchmark_m5,
                benchmark_d1=benchmark_d1,
            )
        except Exception:
            # The note is saved; the picture beside it is best-effort.
            logging.exception("Market journal chart capture could not be started.")

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
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return []
        try:
            from ui.services.chart_data_service import shared_service

            service = shared_service()
            series = service.cached_series(symbol)
            now = time.monotonic()
            last = self._d1_prefetch_last.get(symbol, 0.0)
            retry_seconds = 60.0 if series is not None else 15.0
            if now - last >= retry_seconds:
                self._d1_prefetch_last[symbol] = now
                # Queued, not issued: ~105 single-element prefetch tasks per
                # minute queue ahead of the snapshot task for the chart the
                # trader just clicked in the 2-thread chart pool. One batched
                # call per event-loop turn instead (item 1c).
                self._queue_d1_prefetch(symbol)
            # cached_bar_dicts, not series.as_bar_dicts(): this runs on the Qt
            # thread for every armed and every Focus symbol on a 60s timer, and
            # materializing ~490 dicts per symbol per poll is what the service
            # now memoizes against the series object.
            return service.cached_bar_dicts(symbol) if series is not None else []
        except Exception:
            return []

    def _queue_d1_prefetch(self, symbol: str) -> None:
        """Add a symbol to this turn's prefetch batch and arm the flush."""
        symbol = str(symbol or "").strip().upper()
        if not symbol:
            return
        if symbol not in self._d1_prefetch_pending:
            self._d1_prefetch_pending.append(symbol)
        if self._d1_prefetch_flush_armed:
            return
        self._d1_prefetch_flush_armed = True
        try:
            QTimer.singleShot(0, self._flush_d1_prefetch)
        except Exception:
            # No event loop to defer into (tests, teardown): send it now rather
            # than losing the warm-up entirely.
            self._d1_prefetch_flush_armed = False
            self._flush_d1_prefetch()

    def _flush_d1_prefetch(self) -> None:
        """Issue this turn's queued prefetch as ONE task."""
        self._d1_prefetch_flush_armed = False
        symbols = list(self._d1_prefetch_pending)
        self._d1_prefetch_pending.clear()
        if not symbols:
            return
        try:
            from ui.services.chart_data_service import shared_service

            shared_service().prefetch(symbols)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # A2 (2026-09-01): an arm has a life, measured in SESSIONS.
    #
    # The trader's Armed inventory is supposed to read as "the exact conditions
    # I am waiting on". A watch armed six weeks ago on a thesis that has since
    # gone stale is noise in that list, so a 5-day extreme watch is given 5
    # trading days, a 20-day one 10, and everything else 10.
    #
    # Policy lives in `armed_alert_expiry`; this is the seam that applies it to
    # the panel's own stores. It runs at the head of each poll that already
    # owns one, so no new timer appears. Uncertainty never deletes: an entry
    # the calendar cannot date is kept, and every removal writes an
    # append-only row naming symbol, kind, armed_at and expired_at.
    def _expire_armed_watches(
        self,
        store: str,
        watches: list,
        *,
        kind_of=None,
        now: datetime | None = None,
    ) -> tuple[list, list[dict]]:
        try:
            import armed_alert_expiry

            moment = now or datetime.now()
            kept, rows = armed_alert_expiry.partition(
                watches, store=store, today=moment.date(), kind_of=kind_of
            )
        except Exception:
            # A broken expiry pass must never cost the poll behind it.
            logging.debug("Armed-alert expiry pass failed for %s", store, exc_info=True)
            return list(watches), []
        if not rows:
            return kept, []
        armed_alert_expiry.record_expiries(rows)
        for row in rows:
            self._record_review_event(
                "armed_alert_expired",
                symbol=str(row.get("symbol") or ""),
                detail={
                    "store": row.get("store"),
                    "kind": row.get("kind"),
                    "armed_at": row.get("armed_at"),
                    "expired_at": row.get("expired_at"),
                    "trading_days": row.get("trading_days"),
                },
            )
        names = ", ".join(sorted({str(row.get("symbol") or "") for row in rows}))
        self.statusChanged.emit(
            f"{len(rows)} armed alert(s) expired and were retired: {names}."
        )
        return kept, rows

    def _poll_d1_level_watches(self, now: datetime | None = None) -> None:
        if not self._d1_level_watches:
            return
        moment = now or datetime.now()
        kept, expired = self._expire_armed_watches(
            "d1_level_watches", self._d1_level_watches, now=moment
        )
        if expired:
            self._d1_level_watches = kept
            self._save_d1_level_watches()
            self._refresh_review_armed_kinds()
            self.armedWatchesChanged.emit()
            if not self._d1_level_watches:
                return
        remaining: list[D1LevelWatch] = []
        triggered = []
        for watch in self._d1_level_watches:
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
        kept, expired = self._expire_armed_watches(
            "d1_event_watches", self._d1_event_watches, now=moment
        )
        if expired:
            self._d1_event_watches = kept
            self._save_d1_event_watches()
            self._refresh_review_armed_kinds()
            self.armedWatchesChanged.emit()
            if not self._d1_event_watches:
                return
        remaining: list[D1EventWatch] = []
        triggered = []
        # One reference-level build per symbol per tick, shared across every
        # watch on it (item 1b). Scoped to this tick and discarded with it.
        levels_caches: dict[str, dict] = {}
        for watch in self._d1_event_watches:
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
                        from ui.services.chart_data_service import shared_service

                        avwape_anchor = shared_service().cached_earnings_anchor(
                            watch.symbol
                        )
                    except Exception:
                        avwape_anchor = None
                try:
                    hit = evaluate_d1_event_watch(
                        watch,
                        m5_bars,
                        d1_bars,
                        now=moment,
                        avwape_anchor=avwape_anchor,
                        levels_cache=levels_caches.setdefault(watch.symbol, {}),
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

    # ------------------------------------------------------------------
    # Persistent any-bounce watches (R5 section 4): one armed request per
    # symbol+side covering the whole level set. Same rails as the D1 event
    # watches - same 60s poll, same red chart-watch alert, same one-shot
    # retire - and the same single owner, because a second writer to a watch
    # store is how two components start disagreeing about what is armed.
    def any_bounce_armed_for(self, symbol: str) -> bool:
        symbol = str(symbol or "").strip().upper()
        return any(watch.symbol == symbol for watch in self._any_bounce_watches)

    def _save_any_bounce_watches(self) -> None:
        if self._any_bounce_watches_path is not None:
            try:
                save_any_bounce_watches(
                    self._any_bounce_watches, self._any_bounce_watches_path
                )
            except Exception:
                pass

    def _toggle_any_bounce_watch(self, alert: BounceAlert) -> None:
        if alert is None or not alert.symbol:
            return
        if self.any_bounce_armed_for(alert.symbol):
            self.disarm_any_bounce_watch(alert.symbol)
        else:
            self.arm_any_bounce_watch(alert.symbol, alert.side or "long")

    def arm_any_bounce_watch(self, symbol: str, side: str = "long") -> bool:
        symbol = str(symbol or "").strip().upper()
        side = "short" if str(side or "").strip().lower().startswith("short") else "long"
        if not symbol:
            return False
        if self.any_bounce_armed_for(symbol):
            self.statusChanged.emit(f"{symbol}: any-bounce alert already armed.")
            return False
        self._any_bounce_watches.append(
            AnyBounceWatch(
                symbol=symbol,
                side=side,
                kinds=tuple(ANY_BOUNCE_KINDS),
                armed_at=datetime.now(),
            )
        )
        self._save_any_bounce_watches()
        self._refresh_review_armed_kinds()
        self.armedWatchesChanged.emit()
        current = self._current_review_alert
        self._record_review_event(
            "arm_any_bounce",
            alert=current if current is not None and current.symbol == symbol else None,
            symbol=symbol,
            side=side,
            dwell_ms=self._review_dwell_ms(symbol),
            detail={"kinds": list(ANY_BOUNCE_KINDS)},
        )
        self.statusChanged.emit(
            f"{symbol}: any-bounce alert armed - it fires once, on whichever "
            "of your levels holds, and then disarms."
        )
        return True

    def disarm_any_bounce_watch(self, symbol: str) -> bool:
        symbol = str(symbol or "").strip().upper()
        remaining = [
            watch for watch in self._any_bounce_watches if watch.symbol != symbol
        ]
        if len(remaining) == len(self._any_bounce_watches):
            return False
        self._any_bounce_watches = remaining
        self._save_any_bounce_watches()
        self._refresh_review_armed_kinds()
        self.armedWatchesChanged.emit()
        self._record_review_event("disarm_any_bounce", symbol=symbol)
        self.statusChanged.emit(f"{symbol}: any-bounce alert disarmed.")
        return True

    def _any_bounce_levels_for(
        self, symbol: str, moment: datetime, *, m5_bars: list | None = None
    ) -> dict:
        """The armed level set from whatever the desk already has cached.

        The D1 side comes from the scan's zone-arms file (which is where the
        prior-anchor AVWAP now rides, R5 section 8.3); the session and hourly
        EMAs are aggregated from the cached M5 bars. Nothing here fetches.
        """
        entry = None
        try:
            bot = self._current_bot()
            arms = getattr(bot, "d1_zone_arms", None) or {}
            candidate = arms.get(symbol)
            if isinstance(candidate, Mapping):
                entry = candidate
        except Exception:
            entry = None
        d1_levels = None
        try:
            d1_bars = self._d1_bars_for(symbol)
            if d1_bars:
                d1_levels = d1_event_levels(d1_bars, session=moment.date())
        except Exception:
            d1_levels = None
        return any_bounce_levels(
            zone_arm_entry=entry,
            m5_bars=self._m5_bars_for(symbol) if m5_bars is None else m5_bars,
            d1_levels=d1_levels,
            now=moment,
        )

    def _poll_any_bounce_watches(self, now: datetime | None = None) -> None:
        if not self._any_bounce_watches:
            return
        moment = now or datetime.now()
        # An any-bounce watch covers a SET of levels, so it has no single
        # `kind`; it files under the default 10-session window by name.
        kept, expired = self._expire_armed_watches(
            "any_bounce_watches",
            self._any_bounce_watches,
            kind_of=lambda watch: "any_bounce",
            now=moment,
        )
        if expired:
            self._any_bounce_watches = kept
            self._save_any_bounce_watches()
            self._refresh_review_armed_kinds()
            self.armedWatchesChanged.emit()
            if not self._any_bounce_watches:
                return
        remaining: list[AnyBounceWatch] = []
        triggered = []
        for watch in self._any_bounce_watches:
            hit = None
            try:
                # Once per watch, not twice: the levels builder and the
                # evaluation both need today's M5 bars.
                m5_bars = self._m5_bars_for(watch.symbol)
                levels = self._any_bounce_levels_for(watch.symbol, moment, m5_bars=m5_bars)
                if levels:
                    hit = evaluate_any_bounce_watch(watch, m5_bars, levels, now=moment)
            except Exception:
                hit = None
            if hit is None:
                remaining.append(watch)
            else:
                triggered.append(hit)
        self._any_bounce_watches = remaining
        if triggered:
            self._save_any_bounce_watches()
            self._refresh_review_armed_kinds()
            self.armedWatchesChanged.emit()
        for hit in triggered:
            self._record_review_event(
                "any_bounce_fired",
                symbol=hit.watch.symbol,
                side=hit.resolved_side,
                detail={"kind": hit.kind, "level": hit.level, "message": hit.message},
            )
            self.add_alert(self._chart_watch_alert(hit, moment))

    def _open_external_chart(self, symbol: str) -> None:
        """Deep-link the charted name into the trader's external tool.

        Read-only in both directions: it opens a URL and reads nothing back, so
        no second source of truth about a symbol enters the system. A refused
        open is REPORTED - silence would read as "it worked, look elsewhere".
        """
        from external_chart_links import open_chart

        _opened, message = open_chart(symbol, "D1")
        self.statusChanged.emit(message)

    def _refresh_review_armed_kinds(self) -> None:
        current = self._current_review_alert
        if current is not None:
            self.chart_review.set_armed_kinds(self.armed_watch_kinds(current.symbol))
            self.chart_review.set_armed_d1_events(
                self.armed_d1_event_kinds(current.symbol)
            )
            self.chart_review.set_any_bounce_armed(
                self.any_bounce_armed_for(current.symbol)
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
        # Faded walkthrough: the dismiss verb DISCARDS - it clears the entry
        # off the faded list. The pick is already out of Focus, so nothing is
        # removed here; the append-only evidence row stays either way.
        if alert.tag == FOCUS_FADED_TAG:
            self._discard_faded_review_alert(alert)
            return
        # Focus walkthrough: the dismiss verb DELETES the pick from Focus
        # (every bucket/side; un-injects its watchlist entries; logs the
        # unfavorite to pick feedback). The symbol itself is not muted.
        if alert.tag == FOCUS_REVIEW_TAG:
            removed = 0
            if self.focus_service is not None:
                try:
                    removed = int(
                        self.focus_service.remove_everywhere(
                            alert.symbol, origin="focus_review", context=alert.raw_text
                        )
                    )
                except Exception:
                    removed = 0
            self._record_review_event(
                "focus_review_remove",
                alert=alert,
                dwell_ms=self._review_dwell_ms(alert.symbol),
                queue_len=len(self._review_queue),
                detail={"entries_removed": removed},
            )
            self.statusChanged.emit(
                f"✕ {alert.symbol}: removed from Focus Picks."
                if removed
                else f"{alert.symbol}: was not in Focus Picks anymore."
            )
            self._advance_review_queue()
            return
        # An M5 Focus entry the machine adopted can be thrown back (packet R2).
        # A name the trader typed falls through to the quiet feed-only verb
        # below - `remove_if_auto_adopted` refuses it, and that refusal is the
        # never-auto-remove-user-names invariant doing its job.
        dropped = self._drop_auto_adopted_pick(alert)
        self._record_review_event(
            "remove_today",
            alert=alert,
            dwell_ms=self._review_dwell_ms(alert.symbol),
            queue_len=len(self._review_queue),
            detail={"auto_pick_dropped": dropped} if dropped else None,
        )
        self._ignore_alert_symbol(alert.symbol)
        if dropped:
            self.statusChanged.emit(
                f"✕ {alert.symbol}: auto pick dropped from M5 Focus for today "
                "(your own picks and the swing list are untouched)."
            )
            return
        self.statusChanged.emit(
            f"{alert.symbol}: removed from Alert Center processing for today. "
            "BounceBot scanning and watchlists are unchanged."
        )

    def _drop_auto_adopted_pick(self, alert: BounceAlert) -> str:
        """Scoped removal of an auto-adopted M5 entry. Returns the side, or "".

        Only the side the chart is about; with no side on the alert, only a
        side that actually carries a marker. Never both blindly - "Not today"
        on a long chart must not silently drop a short entry the trader is
        still holding.
        """
        service = self.focus_service
        if service is None or not alert.symbol:
            return ""
        remover = getattr(service, "remove_if_auto_adopted", None)
        if not callable(remover):
            return ""
        side = str(alert.side or "").strip().lower()
        sides = (side,) if side in ("long", "short") else ("long", "short")
        for one in sides:
            try:
                if remover(
                    alert.symbol,
                    one,
                    "m5",
                    reason="not today",
                    origin="auto_pick",
                ):
                    return one
            except Exception:
                logging.warning(
                    "Scoped removal of %s failed; Focus is unchanged.",
                    alert.symbol,
                    exc_info=True,
                )
                return ""
        return ""

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
        # Focus D1 interest flags reset with the session too - yesterday's
        # "already flagged" must not mute today's events.
        self._focus_d1_flags = (
            load_day_scoped_flags(self._focus_d1_flags_path, market_date=current)
            if self._focus_d1_flags_path is not None
            else set()
        )
        # So must the previous-day extreme gate: yesterday's breakout says
        # nothing about today's range, and a stale open stamp would let the
        # first poll of a new session replay the whole morning.
        self._focus_break_state.clear()
        self._focus_break_open_at.clear()
        self._focus_gate_held = 0
        # The movers-only filter is day-scoped for the same reason: a reveal is
        # "show me the chop for the rest of today", not a preference change.
        self._review_movers_only = True
        self._hidden_inside_range.clear()
        self.chart_review.set_hidden_count(0)
        self._refresh_ignored_button()
        # The M5 alert bar is day-scoped like the queue it replaced.
        self.m5AlertsDayRolled.emit()
        # A3: the fade clock is measured in sessions, so the day roll is
        # exactly when a pick can come due.
        self.run_focus_fade_check()

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

    def apply_scaled_metrics(self) -> None:
        """Re-apply the column's scale-dependent floors after a scale change."""
        self.tabs.setMinimumWidth(theme.px(170))
        self.focus_strength.apply_scaled_metrics()
        self.chart_review.arm_bar.apply_scaled_metrics()

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

    def show_board_symbol(self, symbol: str, side: str = "") -> None:
        """Public entry for boards that live on OTHER pages.

        The RS/RW, entry, Focus-strength and (since 2026-08-31) M5 strength
        boards are all children of this panel and call the private opener
        directly. This stays as the named door for anything that is not -
        the AWAY Recap page uses it - so a board on another page never has to
        reach into a private method. Same popup, same owner, same capture rail
        and painted levels (R4 unification, 2026-08-19).
        """
        self._show_board_symbol_snapshot(symbol, side)

    def _chart_strength_board_symbol(self, symbol: str, side: str = "") -> None:
        """A strength-board row click charts that name in the review pane.

        Trader, 2026-08-31: *"when I click on a stock in this M5 strength board
        it should come up on the Visual chart review in the trading desk."* It
        used to open the snapshot popup, which was the right answer while the
        board was a page of its own and the review pane was somewhere else;
        now that the board sits in the same column as the pane, a popup over
        the top of it is a window in the way.

        This goes through `chart_symbol` - the SAME door the lookup box uses -
        and deliberately not through `_enqueue_review_alert`, which is the door
        for things the SCANNER said. That one would have been wrong four ways
        for a click: it drops everything in AWAY, it drops parked symbols, it
        diverts M5 alerts to the alert bar instead of the chart, and the
        movers-only filter can hide a row. A name the trader clicked must
        appear.
        """
        self.chart_symbol(symbol, side=side, origin="the M5 Strength Board")

    def attach_strength_board(self, service, focus_service=None) -> None:
        """Host the M5 Strength Board under the Strength window.

        `MainWindow` builds and owns the one `StrengthBoardService`; this
        panel is given it. Called once at startup - a second call would
        replace the section's body, not add a second board, but nothing does.

        Deliberately NOT here: any refresh, timer, thread or fetch. The
        service's single-flight owner and its 15-minute clock are unchanged by
        the move, and the board is still batched yfinance over
        `universe_all.txt` with **zero IB traffic**. The only thing this panel
        adds is a parent and a snapshot popup for a row click - the same popup
        every other board on this panel opens.
        """
        from ui.panels.strength_board_panel import StrengthBoardPanel

        board = StrengthBoardPanel(
            service=service,
            focus_service=self.focus_service if focus_service is None else focus_service,
        )
        board.symbolActivated.connect(self._chart_strength_board_symbol)
        self.strength_board = board

        # The alert column's floor is 360 px and the tab stack already claims
        # 170 of it. The board asks for 270 (two side tables, each with a
        # heading row and an "Add all" button), and a widget's minimum reaches
        # the splitter, so hosting it bare would have raised the floor the
        # charts are sized against - the one thing this move must not do.
        # Inside a scroll area the board's minimum stops here: at a normal
        # column width nothing scrolls, and a trader who drags the column
        # narrower gets a scrollbar instead of narrower charts.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(board)
        scroll.setMinimumWidth(theme.px(170))
        self.strength_board_section.set_content(scroll)

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
