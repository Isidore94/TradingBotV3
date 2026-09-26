from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict, defaultdict
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
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
    AnyBounceWatch,
    BAND_BOUNCE_PRIME_BUCKETS,
    BAND_BOUNCE_TRACKER_TYPES,
    ChartWatch,
    D1EventWatch,
    D1LevelWatch,
    D1_EVENT_KINDS,
    D1_LEVEL_KINDS,
    D1_PULLBACK_KINDS,
    d1_event_fired_detail,
    D1_SIDED_KINDS,
    WATCH_KINDS,
    arm_chart_watch,
    d1_kind_needs_avwape,
    incoming_trendline_type,
    evaluate_chart_watch,
    evaluate_d1_event_watch,
    evaluate_d1_level_watch,
    completed_session_bars,
    load_any_bounce_watches,
    load_chart_watches,
    load_d1_event_watches,
    load_d1_level_watches,
    save_chart_watches,
    save_d1_event_watches,
    save_d1_level_watches,
    watch_is_stale,
    PERSISTENT_WATCH_KINDS,
    PULLBACK_KIND,
    PULLBACK_TRIGGERS,
)
import focus_adoption_gate
import alert_show_filter
import regime_pause_hold
import sector_exclusion
from regime_pause_focus import day_bias, focus_side_for
import sma_trend_gate
import wall_gate
import wall_gate_arms
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
    CLAIMED_PICKS_FILE,
    FOCUS_D1_FLAGS_FILE,
    MASTER_AVWAP_AI_STATE_FILE,
    MASTER_AVWAP_D1_UPGRADE_ALERTS_FILE,
    ANY_BOUNCE_WATCHES_FILE,
    D1_EVENT_WATCHES_FILE,
    D1_LEVEL_WATCHES_FILE,
    WALL_GATE_ARMS_FILE,
    get_local_setting,
    save_local_settings,
)
from alert_repetition import ACTION_DIGEST, ACTION_FOLD
from review_events import record_review_event
from review_guidance import ORDERING_ANNOTATION_ONLY, AlertGuidance, ReviewGuide
from ui import theme
from ui.panels import desk_layout
from ui.timer_utils import SignalCoalescer, start_staggered
from ui.services.m5_bar_cache import is_process_proxy, shared_m5_cache
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
    is_regime_pause_alert,
    REGIME_PAUSE_TRIGGER_PREFIX,
)
from ui.widgets.alert_chart_review import AlertChartReview
from ui.widgets.armed_watch_list import ArmedWatchList
from ui.widgets.entry_assist_board import EntryAssistBoard
from ui.widgets.focus_strength_board import FocusStrengthBoard
from ui.widgets.movers_board import MoversBoard
from ui.widgets.rrs_snapshot import RrsSnapshotWidget
from ui.widgets.section_header import SectionHeader
from ui.widgets.strength_page import StrengthPage
from ui.widgets.tab_drawer import TabDrawer
from ui.widgets.setup_detail_view import SetupDetailView
from swallowed import note_swallowed
from ui.panels.alert_center.items import _ClickableItem
from ui.panels.alert_center.strength_board import StrengthBoardAdoptionMixin
from ui.panels.alert_center.pullback import PullbackWatchMixin
from ui.panels.alert_center.h1 import H1RetesterMixin
from ui.panels.alert_center.any_bounce import AnyBounceWatchMixin
from ui.panels.alert_center.wall import WallGateMixin
from ui.panels.alert_center import gates
from ui.panels.alert_center.gates import (  # noqa: F401 - re-exported for callers and tests
    _D1_DEVELOPING_PREFIXES,
    _D1_PUSH_LABELS,
    _D1_READY_PREFIXES,
    _PROVEN_RE,
    _TIER_RANK,
    _TIER_RE,
    _bar_close,
    _d1_alert_prefix,
    _is_feed_noise_alert,
    alert_is_loud,
    alert_passes_feed_gate,
    alert_passes_min_tier,
    alert_should_sound,
    d1_push_event,
    extract_alert_tier,
    favorite_category_for_alert,
    favorite_origin_for_alert,
    intraday_last_bucket_end,
    is_developing_d1_alert,
    is_entry_assist_alert,
    is_proven_alert,
    is_ready_d1_alert,
)

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

#: "Not passed": `arm_d1_event_watch` reads the trendline report itself.
_UNREAD = object()

MIN_TIER_CHOICES = (
    ("All alerts", "all"),
    ("B tier and above", "B"),
    ("A tier and above", "A"),
    ("S tier / PROVEN only", "S"),
)
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


#: R4 A10: `held_run_score`'s segment index and the D1 setups by session, built
#: once per process PER TRADING DAY. `None` means "not built yet"; a dict - even
#: an empty one - means "built, and this is the answer". A failed read is an
#: answer too. The payload carries `built_for`, the desk-local date it was built
#: on, and that is what makes it expire.
#:
#: **The day key is the whole point** (R4 fix round 1). Without it the memo was
#: set once and never invalidated, which broke A9's own fix after one day of
#: uptime: `d1_setups_by_session` is keyed by `trade_date`, so on day 2 there was
#: no key for today and every alert read `d1_setup_present=False` again - exactly
#: the state A9 was built to end. The index is also a 20-TRADING-SESSION window
#: (`held_run_score.ROLLING_SESSIONS`), so a memo that never rolls stops being
#: "lately" while still calling itself that, against CLAUDE.md's rule that
#: "lately" is ONE number counted in trading sessions. The desk is the always-on
#: mini-PC and the checkpoint records multi-day uptimes, so "once per process"
#: was never the same thing as "once per session".
_HELD_RUN_INDEX_MEMO: dict | None = None

#: Packet D1C-A: "the claims file has never been read", which is a different
#: answer from "there is no claims file" (a real, cacheable stamp of None).
_CLAIM_KEYS_UNREAD = object()


class AlertCenterPanel(
    StrengthBoardAdoptionMixin,
    WallGateMixin,
    AnyBounceWatchMixin,
    H1RetesterMixin,
    PullbackWatchMixin,
    QFrame,
):
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
    #: The Pullback alert's off-thread SMA evaluation, coming home. Emitted
    #: FROM the worker, so the connection is auto/QUEUED and the slot runs on
    #: the Qt thread - the only thread that may record, push or draw.
    pullbackFiresReady = Signal(object)
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
    #: P9: the Show filter's answer may have changed (choice, grades, Best list).
    showFilterChanged = Signal()
    #: The faded list changed (a fade, a restore, a discard), so the button
    #: count can repaint. The active Focus lists change too, so `focusChanged`
    #: already fires - this exists so a surface that only shows the FADED
    #: count does not have to listen to every Focus mutation.
    focusFadedChanged = Signal()
    #: Packet D1C-A: a claim was placed or dropped, so the surfaces that read
    #: `claimed_picks.jsonl` re-read it. The setups table is the one that shows
    #: the pick; this panel's own queue gate listens too, because a drop has to
    #: take effect on the very next alert.
    claimsChanged = Signal()
    #: Trader, 2026-09-15: a veto or a claim on the centre chart is a decision
    #: the setups table must see at once (its hide filter and its ✕ mark).
    reviewDecisionRecorded = Signal()
    #: R4 A10: `held_run_score`'s segment index, built once per session on a
    #: worker. `object` because the payload is a plain dict Qt must not marshal.
    _heldRunIndexLoaded = Signal(object)

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
        claimed_picks_path=None,
        wall_gate_arms_path=None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self.focus_service = focus_service
        self._bounce_service = None
        # Queued arms (built on first use). While a queued arm commits, its
        # review rows are deferred to the queue's worker and use the click's context.
        self._arm_queue_obj = None
        self._force_async_arms = False
        # B6: one hidden_by_show row per (day, symbol, side); the write and the
        # store read of today's keys run on one evidence worker, never here.
        self._show_hidden_seen: set[tuple[str, str, str]] = set()
        self._show_hidden_executor = None
        self._show_hidden_last = None
        self._show_hidden_store_keys: dict[str, set[tuple[str, str]]] = {}
        self._deferred_review_events: list | None = None
        self._arm_review_override: tuple | None = None
        self._alerts: list[BounceAlert] = []
        self._d1_alerts: list[BounceAlert] = []
        self._review_queue: list[BounceAlert] = []
        # AR-2B: ordinary D1 scan ideas remain recorded but start outside the
        # visual-review display.  One newest row per symbol is enough for the
        # session switch; it is a presentation cache, never a verdict or a
        # second alert store.
        self._held_d1_scan_reviews: dict[str, BounceAlert] = {}
        self._show_all_d1_scan_reviews = False
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
        #: ST6.5. The day-trade verdict order off the desk's shared snapshot,
        #: `[(bounce_type, SIDE)]` best first. Empty until the Working-lately
        #: service hands one over, and honoured only while the switch is ON.
        self._working_lately_order: list[tuple[str, str]] = []
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
        # Wall gate (trader, 2026-09-23): which follow-up watches the desk
        # armed for a name hidden at a wall, and which the trader declined.
        self._wall_arms_path = (
            Path(wall_gate_arms_path)
            if wall_gate_arms_path is not None
            else (WALL_GATE_ARMS_FILE if persist_ignored else None)
        )
        self._wall_arms: dict[str, dict] = wall_gate_arms.load(self._wall_arms_path)
        # Register the trendline projection now, so the first shared ai_state
        # parse (BounceBot's worker) already carries it. No parse here.
        self._wall_trendlines_for("")
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
        # PCT-1. The Pullback alert's SMA legs are evaluated on a worker and
        # come back here; the connection is made once, in the constructor, so
        # a fire can never arrive before there is a slot to take it.
        self._pullback_eval_busy = False
        self._pullback_episodes: dict = {}
        self._pullback_judged: dict = {}
        self.pullbackFiresReady.connect(self._on_pullback_fires)
        # Packet D1C-A. The trader's claimed D1 picks. Gated exactly like the
        # stores above - a bare test panel neither reads nor writes the live
        # file - and this panel is only ever a READER of it plus the writer of
        # the claim the chart in front of it produces.
        self._claimed_picks_path = (
            Path(claimed_picks_path)
            if claimed_picks_path is not None
            else (CLAIMED_PICKS_FILE if persist_ignored else None)
        )
        #: mtime+size+day keyed cache of the ACTIVE `(symbol, side)` keys. The
        #: gate below runs on every alert and an alert burst is exactly where a
        #: per-alert file read would be paid for, so the file is read once when
        #: it changed and not again (nothing expensive on the Qt thread).
        self._claim_keys_cache: set[tuple[str, str]] = set()
        self._claim_keys_stamp: object = _CLAIM_KEYS_UNREAD
        self._claim_keys_day: str = ""
        #: How many repeat D1 charts the claim gate skipped, per symbol. A
        #: COUNT, the way the movers-only filter's hidden count is: it hides
        #: and states a number, it deletes nothing, mutes nothing and writes
        #: nothing to `review_policy.json`.
        self._claimed_d1_skipped: dict[str, int] = defaultdict(int)
        self.claimsChanged.connect(self._on_claims_changed)
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
        #: And for the wall gate (trader, 2026-09-23): (stamp, WallVerdict).
        self._wall_measure_cache: dict[tuple[str, str], tuple[tuple, object]] = {}
        #: symbol -> why an at-wall chart still shows (nothing could follow it up).
        self._wall_uncovered: dict[str, str] = {}
        #: (date, symbol, side, action, wall) already written to the decision log.
        self._wall_logged: set[tuple] = set()
        self._focus_break_open_at: dict[str, datetime] = {}
        #: Break-state map last sent on `focusBreakStatesChanged`; None = never sent.
        self._focus_break_emitted: dict[str, str] | None = None
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
        # R4 A10 - decision 0016 answer 4's day-trade headline on the M5 row.
        # Built ONCE per session on a worker (the outcome log is ~90 MB) and
        # then read as a dict, exactly like the take-rate cache above it: a row
        # suffix must never put a file read in the alert path.
        self._held_run_index: dict = {}
        self._held_run_d1_symbols: dict = {}
        self._held_run_thread = None
        #: The desk-local date this panel's copy was built for. Empty means it
        #: has none yet; a date that is not today's means it has expired.
        self._held_run_built_for = ""
        self._heldRunIndexLoaded.connect(self._on_held_run_index_loaded)
        if self.focus_service is not None:
            # Liking a pick (here or on the setups table) re-renders both feeds
            # so every alert for that name immediately shows the gold flag.
            #
            # COALESCED (2026-08-31, trader-approved under the file-scoped
            # ask-first rule), and since SN4 (2026-09-12) a DIFF rather than a
            # rebuild. `_rebuild_feed` destroys and reconstructs every row
            # widget in both feeds - up to MAX_FEED_ITEMS + MAX_D1_FEED_ITEMS
            # = 350 widget trees, each with its own stylesheet - and the DESK
            # drain that morning fired it 45 times in 13 seconds; one coalesced
            # change still cost 24.2 s on 2026-09-08. `_sync_feed` restyles the
            # star on the rows the change touches and leaves the rest alone.
            # Only the TRIGGER is coalesced: which alerts pass the feed gate,
            # their order, the repetition fold and the digest are all decided
            # by `_feed_target_rows` and are untouched. Nothing is withheld -
            # the refresh still happens, once, within 200 ms of the last
            # change. Late-bound so the coalescer calls whatever `_sync_feed`
            # is at fire time - the seam a test spies on is the one that runs.
            self._focus_feed_coalescer = SignalCoalescer(
                lambda: self._on_focus_feed_coalesced(), parent=self
            )
            self.focus_service.focusChanged.connect(
                self._focus_feed_coalescer.request
            )
            # A held ordinary scan that becomes a real Focus name is no
            # longer an ordinary scan.  Move that current-session display row
            # through the normal queue door immediately, rather than leaving
            # a stale Show all count or waiting for a second scanner alert.
            self.focus_service.focusChanged.connect(self._on_focus_membership_changed)

        self.min_tier_input = QComboBox()
        for label, mode in MIN_TIER_CHOICES:
            self.min_tier_input.addItem(label, mode)
        saved_mode = str(get_local_setting("qt_alert_min_tier", "all") or "all")
        self.min_tier_input.setCurrentIndex(max(0, self.min_tier_input.findData(saved_mode)))
        self.min_tier_input.currentIndexChanged.connect(self._on_prefs_changed)

        self.sound_input = QCheckBox("Sound on S/A + PROVEN")
        self.sound_input.setChecked(bool(get_local_setting("qt_alert_sound", True)))
        self.sound_input.toggled.connect(self._on_prefs_changed)

        # Trader, 2026-09-23: one shared switch hides Oil & Gas / Real Estate
        # names from the feeds (display only; the alert is still recorded).
        self.hide_sector_input = QCheckBox(sector_exclusion.HIDE_LABEL)
        self.hide_sector_input.setToolTip(
            "Hides Oil & Gas and Real Estate alerts from these feeds: no row, no chart, "
            "no sound. They are still recorded. Focus names and armed watches always show."
        )
        self.hide_sector_input.setChecked(sector_exclusion.hide_enabled())
        self.hide_sector_input.toggled.connect(self._on_hide_sector_toggled)

        # P9 (trader, 2026-09-25): which M5 rows show. Display only; every
        # alert is still recorded and still reaches the review-queue door.
        self._show_grades: dict = {}
        self._show_best_keys: frozenset | None = None
        #: `id(alert) -> (alert, (hidden, is_new))`; cleared when any input changes.
        self._show_verdicts: dict = {}
        self._show_typed_seen: frozenset | None = None
        self.show_filter_input = QComboBox()
        self.show_filter_input.setObjectName("AlertShowFilter")
        for value, label in alert_show_filter.MODES:
            self.show_filter_input.addItem(label, value)
        self.show_filter_input.setCurrentIndex(
            max(0, self.show_filter_input.findData(alert_show_filter.mode()))
        )
        self.show_filter_input.setToolTip(
            "Which M5 alert rows show. Grade B and up = PROVEN, A, B (C, D and New hide). "
            "Best right now = only names on the Best-right-now list. Hidden rows make no "
            "sound and open no chart, but are still recorded. Your typed names, Focus "
            "names, armed watches, price alerts and regime-pause rows always show."
        )
        self.show_filter_input.currentIndexChanged.connect(self._on_show_filter_changed)
        # S2: hide 09:30-10:00 ET M5 rows (PROVEN, Focus, typed, watches still show).
        self.first30_input = QCheckBox(alert_show_filter.FIRST30_LABEL)
        self.first30_input.setObjectName("AlertShowFirst30")
        self.first30_input.setToolTip(
            "Hide M5 alerts from 9:30 to 10:00 ET (they win less and end red on "
            "average). PROVEN, Focus names, your typed names and armed watches "
            "always show. Hidden rows are still recorded."
        )
        self.first30_input.setChecked(alert_show_filter.first30_enabled())
        self.first30_input.toggled.connect(self._on_show_filter_changed)

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
        self.rrs_snapshot.symbolActivated.connect(
            lambda symbol, side: self._chart_board_symbol(symbol, side, "the RS/RW board")
        )

        # The automatic entry-assist board (regime + pause detection + live
        # window / preview rankings + 30m movers, no clicks). It and the RRS
        # sweep snapshot above are two reads on the Strength page built below
        # the tab strip; nothing here is a tab any more.
        self.entry_board = EntryAssistBoard()
        self.entry_board.symbolActivated.connect(
            lambda symbol, side: self._chart_board_symbol(symbol, side, "the entry board")
        )

        # The armed-watch inventory. Built before the tab bar that hosts it.
        self.armed_list = ArmedWatchList(self)
        self.armed_list.disarmWatchRequested.connect(self.disarm_chart_watch_for)
        self.armed_list.disarmLevelRequested.connect(self.disarm_d1_level_watch)
        self.armed_list.disarmEventRequested.connect(self.disarm_d1_event_watch)
        self.armed_list.symbolActivated.connect(self.chart_symbol)
        self.armed_list.pendingCancelRequested.connect(self._cancel_or_dismiss_arm)
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
            self,
            dock_arm_bar=True,
            dock_capture_rail=False,
            # Packet D1C-A. The pane owns the ROUTE; this panel owns the STORE,
            # exactly as it owns the review-events and parked-symbols files, so
            # the writer is bound to the panel's path here rather than resolved
            # inside the widget. A panel with NO claims path - a bare test
            # panel - hands over NOTHING, and the pane keeps its pre-packet
            # route: it cannot place a pick, and a desk that was never given a
            # store has lost nothing by not writing one.
            claim_writer=(
                self._write_claim if self._claimed_picks_path is not None else None
            ),
        )
        self.chart_review.removeTodayRequested.connect(
            self._remove_review_alert_for_today
        )
        self.chart_review.vetoRetireRequested.connect(self._retire_after_veto)
        self.chart_review.savedVeto.connect(self._arm_saved_extended_veto)
        self.chart_review.likeRecorded.connect(self._after_like)
        self.chart_review.likeAdvanceRequested.connect(self._advance_after_like)
        self.chart_review.claimPlaced.connect(self._place_claimed_d1)
        self.chart_review.focusRequested.connect(self._add_review_alert_to_focus)
        self.chart_review.skipRequested.connect(self._skip_review_alert)
        self.chart_review.crossFocusToggled.connect(self._toggle_review_cross_focus)
        self.chart_review.watchToggled.connect(self._toggle_chart_watch)
        self.chart_review.d1EventToggled.connect(self._toggle_d1_event_watch)
        self.chart_review.anyBounceToggled.connect(self._toggle_any_bounce_watch)
        self.chart_review.externalChartRequested.connect(self._open_external_chart)
        self.chart_review.revealHiddenRequested.connect(self.reveal_hidden_reviews)
        self.chart_review.scanReviewViewToggled.connect(self._toggle_d1_scan_review_view)
        self.chart_review.d1LevelAlertRequested.connect(self._arm_d1_level_from_chart)
        self.chart_review.symbolRequested.connect(self.chart_symbol)
        self.chart_review.levelArmRequested.connect(self._arm_level_from_dock)
        self.chart_review.levelDisarmRequested.connect(self._disarm_level_from_dock)
        self.chart_review.levelAlertRequested.connect(self._request_price_alert_from_level)
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
        # V1 (decision 0016 answer 7) moved the RS/RW board OUT of this tab
        # stack (trader: *"The Strength tab loses to the trader's own TC2000
        # scan; the RS/RW board should sit where Strength is."*); since
        # 2026-09-07 it is a block on the Strength page beside this stack -
        # same widgets, same signals, same owner.

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
        self.focus_strength.symbolActivated.connect(
            lambda symbol, side: self._chart_board_symbol(
                symbol, side, "the Focus strength board"
            )
        )
        self.focus_strength.reviewAllRequested.connect(self.review_focus_picks)
        self.focus_strength.fadedReviewRequested.connect(self.review_faded_picks)
        # A fade/restore/discard changes a count the board paints. It rides
        # the board's OWN coalescer (`set_focus_service`), so a burst is one
        # render - the coalescing lives at the listener.
        self.focusFadedChanged.connect(self.focus_strength.request_refresh)

        # THE STRENGTH PAGE (trader, 2026-09-07: *"the strength tab is unusable
        # there's like 2 tabs and they get no space each. Create a solution
        # that removes the tabs and just collates all the data to be more
        # easily readable"*). Until then this half of the row was a column:
        # the Focus board over two collapsible sections - RS/RW Board (open,
        # V1 / decision 0016 answer 7) and the M5 Strength Board (closed,
        # trader 2026-08-31) - sharing the column's height by stretch factor,
        # so each open section was a small window with its own scrollbar onto
        # a document several times taller. Now it is ONE scrolling page with
        # the four reads one under another, each sized to its own content
        # (`ui.widgets.strength_page`). Same widgets, same signals, same owner
        # - only the hosting changed.
        #
        # This panel only HOSTS. `MainWindow` still owns the one
        # `StrengthBoardService`, its one timer and its one fetch, and hands
        # it here through `attach_strength_board`, which puts the board at the
        # foot of the page; nothing here refreshes, schedules or caches.
        self.strength_board: "StrengthBoardPanel | None" = None
        self.strength_page = StrengthPage(
            focus_strength=self.focus_strength,
            entry_board=self.entry_board,
            rrs_snapshot=self.rrs_snapshot,
        )

        # THE MOVERS BOARD (trader, 2026-09-23: "what's the strongest thing
        # moving right now ... and what's strong during a SPY pullback"). It
        # tops the column; the Strength page above sits behind its "Deep read"
        # toggle, unchanged. The two review doors moved into its "Review" menu.
        # Display only; `MainWindow` owns the one MoversService.
        self.movers_board = MoversBoard()
        self.movers_board.symbolActivated.connect(
            lambda symbol, side: self._chart_board_symbol(symbol, side, "the Movers board")
        )
        self.movers_board.focusAddRequested.connect(self._add_movers_row_to_focus)
        self.movers_board.reviewAllRequested.connect(self.review_focus_picks)
        self.movers_board.fadedReviewRequested.connect(self.review_faded_picks)
        if self.focus_service is not None:
            self.movers_board.set_focus_service(self.focus_service)
        self.focusFadedChanged.connect(self.movers_board.request_counts_refresh)
        self.focus_strength.review_button.setVisible(False)
        self.focus_strength.faded_button.setVisible(False)
        self.movers_column = QWidget()
        movers_layout = QVBoxLayout(self.movers_column)
        movers_layout.setContentsMargins(0, 0, 0, 0)
        movers_layout.setSpacing(theme.px(4))
        movers_layout.addWidget(self.movers_board, 1)
        movers_layout.addWidget(self.strength_page, 2)
        self.movers_board.deepReadToggled.connect(self._set_deep_read)
        self._set_deep_read(self.movers_board.deep_read_button.isChecked())

        self.tabs_row = QSplitter(Qt.Orientation.Horizontal)
        self.tabs_row.addWidget(self.tabs)
        self.tabs_row.addWidget(self.movers_column)
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
        self._controls_layout = controls
        self._control_widgets = (
            show_label,
            self.min_tier_input,
            self.sound_input,
            self.hide_sector_input,
            self.show_filter_input,
            self.first30_input,
            None,  # the stretch
            self.ignored_button,
            clear_button,
        )
        self._fill_controls(controls)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)
        layout.addLayout(controls)
        layout.addWidget(splitter, 1)

        # The compact desk (Settings > Desk layout): the chart pane over the
        # tab stack as a collapsible drawer, and the Movers column handed to
        # the desk. Presentation only - the same widgets, re-hosted.
        self._compact_layout = False
        self._classic_policies: dict[str, object] = {}
        self._drawer = TabDrawer(self, self.tabs)
        self._root_layout = layout
        layout.addWidget(self._drawer.splitter, 1)

    # ------------------------------------------------------------------
    def movers_hosted_outside(self) -> bool:
        """True while the compact desk holds the Movers column in its own column."""
        return self._compact_layout

    def is_compact_layout(self) -> bool:
        return self._compact_layout

    def set_compact_layout(self, compact: bool) -> None:
        """Compact: tabs become a drawer under the chart, Movers leave the row.

        Classic puts every widget back where it was, with its size policy and
        the classic splitters' saved sizes. No alert, queue or watch changes.
        """
        compact = bool(compact)
        if compact == self._compact_layout:
            return
        self._compact_layout = compact
        # Compact trims the column's own margins; the charts get the pixels.
        self._root_layout.setContentsMargins(*((4, 2, 4, 2) if compact else (8, 6, 8, 6)))
        self._root_layout.setSpacing(2 if compact else 6)
        if compact:
            self._classic_policies = {
                "chart_review": self.chart_review.sizePolicy(),
                "tabs_row": self.tabs_row.sizePolicy(),
                "movers_column": self.movers_column.sizePolicy(),
            }
            # The desk takes the column next; until then it has no parent.
            self.movers_column.setParent(None)
            self.splitter.setVisible(False)
            self._set_controls_in_tab_corner(True)
            self._drawer.activate(self.chart_review, self.tabs_row)
        else:
            self._drawer.deactivate()
            self._set_controls_in_tab_corner(False)
            self.splitter.insertWidget(0, self.chart_review)
            self.splitter.insertWidget(1, self.tabs_row)
            if self.movers_column.parent() is not self.tabs_row:
                self.tabs_row.insertWidget(1, self.movers_column)
            self.movers_column.setMinimumWidth(0)
            policies = self._classic_policies
            if "chart_review" in policies:
                self.chart_review.setSizePolicy(policies["chart_review"])
                self.tabs_row.setSizePolicy(policies["tabs_row"])
                self.movers_column.setSizePolicy(policies["movers_column"])
            self.splitter.setVisible(True)
            desk_layout.apply_saved_sizes(
                self.splitter, ALERT_SPLIT_KEY, desk_layout.ALERT_COLUMN_WEIGHTS
            )
            desk_layout.apply_saved_sizes(
                self.tabs_row, ALERT_TABS_SPLIT_KEY, desk_layout.ALERT_TABS_ROW_WEIGHTS
            )
        self.chart_review.set_compact_controls(compact)

    def _fill_controls(self, row) -> None:
        """(Re)fill the Show/sound/filter control row in its canonical order."""
        while row.count():
            row.takeAt(0)
        for widget in self._control_widgets:
            if widget is None:
                row.addStretch(1)
            else:
                row.addWidget(widget)

    def _set_controls_in_tab_corner(self, corner: bool) -> None:
        """Compact: the control row rides the drawer's tab bar, not a row of its own."""
        if corner:
            host = getattr(self, "_controls_corner", None)
            if host is None:
                host = QWidget()
                host.setObjectName("AlertControlsCorner")
                row = QHBoxLayout(host)
                row.setContentsMargins(0, 0, theme.px(4), 0)
                row.setSpacing(6)
                self._controls_corner = host
            while self._controls_layout.count():
                self._controls_layout.takeAt(0)
            self._fill_controls(host.layout())
            self.tabs.setCornerWidget(host, Qt.Corner.TopRightCorner)
            host.setVisible(True)
        else:
            host = getattr(self, "_controls_corner", None)
            if host is None:
                return
            self.tabs.setCornerWidget(None, Qt.Corner.TopRightCorner)
            host.setParent(self)
            host.setVisible(False)
            while host.layout().count():
                host.layout().takeAt(0)
            self._fill_controls(self._controls_layout)

    def _reveal_drawer(self) -> None:
        """A hotkey that raises a tab opens the compact drawer too."""
        self._drawer.expand()

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
        self.sync_sector_switch()
        self._check_typed_symbols()
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
        sector_hidden = self._sector_hidden(alert)
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
            if sector_hidden:
                # Recorded in the D1 list, but no row, chart or sound.
                self._d1_alerts.insert(0, alert)
                del self._d1_alerts[MAX_D1_FEED_ITEMS * 2 :]
                self._emit_feed_status()
                return
            self._enqueue_review_alert(alert)
            self._add_d1_alert(alert)
            return
        # The backing list is written BEFORE any repetition decision, and is
        # never consulted by one. History, the evidence streams and the AWAY
        # push all read from here, so folding a row can never cost a record.
        self._alerts.insert(0, alert)
        del self._alerts[MAX_FEED_ITEMS * 2 :]
        if sector_hidden:
            # Recorded above; no row, no chart, no sound.
            self._emit_feed_status()
            return
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
            if self.show_filter_hides(alert):
                # P9: posted to the M5 bar's backing list above; no row, no sound.
                self._record_show_hidden(alert)
                self._emit_feed_status()
                return
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
        if getattr(self, "_digest_market_date", None) != ledger.market_date:
            # The ledger just rolled to a new market day and cleared its own
            # digest; the feed's registry of digested rows rolls with it, or
            # yesterday's burst would keep those names off today's feed.
            self._digest_market_date = ledger.market_date
            self._digested_keys().clear()
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
                latest_alert=alert,
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

        The key is registered BEFORE the row is drawn and taken back off if
        the row could not be drawn, so a failure here falls through to an
        ordinary row (fail-open) rather than leaving a name with no row and
        nothing standing in for it.
        """
        key = self._feed_row_key(alert)
        self._digested_keys().add(key)
        if self._refresh_open_digest_row():
            return True
        self._digested_keys().discard(key)
        return False

    def _refresh_open_digest_row(self) -> bool:
        """Create, update or retire the ONE open-burst row. True if on screen.

        SN4 made this a function of the digested-key registry rather than a
        side effect of one alert arriving, so a veto or a Focus change redraws
        the row in place instead of exploding the burst into forty rows.
        """
        keys = self._digested_keys()
        row = getattr(self, "_digest_row", None)
        try:
            if row is not None and row.parent() is None:
                row = None
                self._digest_row = None
            if not keys:
                if row is not None:
                    self.feed_layout.removeWidget(row)
                    row.setParent(None)
                    row.deleteLater()
                    self._digest_row = None
                return False
            live = {symbol for symbol, _side in keys}
            symbols = [
                symbol
                for symbol in self._repetition_ledger().digest_symbols()
                if symbol in live
            ] or sorted(live)
            if row is None:
                row = QLabel()
                row.setObjectName("Panel")
                row.setWordWrap(True)
                row.setStyleSheet(
                    f"QLabel#Panel {{ color: {theme.color('text_secondary')}; "
                    "padding: 8px 10px; }"
                )
                self.feed_layout.insertWidget(0, row)
                self._digest_row = row
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
        self._reveal_drawer()
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
        self._reveal_drawer()
        self.tabs.setCurrentIndex(self._journal_tab_index)
        self._journal_text.setFocus()

    def _alert_is_loud_cached(self, alert: BounceAlert) -> bool:
        """`alert_is_loud` for one alert, remembered: it reads only the alert's own text."""
        cache = self.__dict__.setdefault("_loud_cache", {})
        cached = cache.get(id(alert))
        if cached is not None and cached[0] is alert:
            return cached[1]
        loud = gates.alert_is_loud(alert)
        if len(cache) > MAX_FEED_ITEMS * 8:
            cache.clear()
        cache[id(alert)] = (alert, loud)
        return loud

    def _feed_status_counts(self) -> tuple[int, int, int, int]:
        """`(loud, hidden rows, hidden New, hidden first30)` in ONE pass over the backing list."""
        loud = 0
        active = self.show_filter_active()
        first30_on = active and self.first30_input.isChecked()
        mode = self._min_tier_mode()
        hide_sectors = sector_exclusion.hide_enabled()
        sector_memo: dict = {}
        first30 = alert_show_filter.REASON_FIRST30
        verdicts: list = []
        first30_verdicts: list = []
        for alert in self._alerts:
            is_focus = self._alert_has_focus_privilege(alert)
            if is_focus or self._alert_is_loud_cached(alert):
                loud += 1
            if not active or alert.symbol in self._ignored_symbols:
                continue
            if is_chart_watch_alert(alert):
                # A chart-watch row is exempt per alert, so it never reads or feeds the memo.
                sector_hidden = self._sector_hidden(alert, hide_sectors)
            else:
                sector_hidden = sector_memo.get(alert.symbol)
                if sector_hidden is None:
                    sector_hidden = self._sector_hidden(alert, hide_sectors)
                    sector_memo[alert.symbol] = sector_hidden
            if sector_hidden or not alert_passes_feed_gate(alert, mode, is_focus=is_focus):
                continue
            key = self._feed_row_key(alert)
            hidden, is_new = self.show_filter_verdict(alert)
            verdicts.append((key, hidden, is_new))
            if first30_on:
                first30_verdicts.append(
                    (key, hidden and self.show_filter_reason(alert) == first30, False)
                )
        rows, new = alert_show_filter.count_hidden(verdicts) if active else (0, 0)
        first30_rows = alert_show_filter.count_hidden(first30_verdicts)[0] if first30_on else 0
        return loud, rows, new, first30_rows

    def _emit_feed_status(self) -> None:
        loud, hidden_rows, hidden_new, first30_rows = self._feed_status_counts()
        # The held count makes the prev-day gate visible: silence should never
        # be indistinguishable from a dead feed.
        held = (
            f" {self._focus_gate_held} Focus name(s) waiting on yesterday's high/low."
            if self._focus_gate_held
            else ""
        )
        shown_hidden = alert_show_filter.hidden_text(
            hidden_rows, hidden_new, first30=first30_rows
        )
        shown_hidden = f" {shown_hidden}." if shown_hidden else ""
        self.statusChanged.emit(
            f"Alert center: {len(self._alerts)} live alert(s), {loud} loud; "
            f"{len(self._d1_alerts)} favorite-bucket transition(s) in D1 Focus.{held}"
            f"{shown_hidden}"
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
        self._manual_next_pick = None
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

    def _sector_hidden(self, alert: BounceAlert, enabled: bool | None = None) -> bool:
        """Oil & Gas / Real Estate under the shared switch. Focus names and armed watches always show."""
        if enabled is None:
            enabled = sector_exclusion.hide_enabled()
        if not enabled or not alert.symbol:
            return False
        if is_chart_watch_alert(alert) or self._alert_is_focus(alert):
            return False
        return sector_exclusion.symbol_is_excluded(alert.symbol)

    # -- P9: the Show filter (display only) -------------------------------
    def show_filter_mode(self) -> str:
        return str(self.show_filter_input.currentData() or alert_show_filter.DEFAULT_MODE)

    def show_filter_active(self) -> bool:
        """True when the Show mode or the first-30 switch can hide a row."""
        return self.show_filter_mode() != alert_show_filter.ALL or self.first30_input.isChecked()

    def _on_show_filter_changed(self, *_args) -> None:
        try:
            alert_show_filter.set_mode(self.show_filter_mode())
            alert_show_filter.set_first30_enabled(self.first30_input.isChecked())
        except Exception:  # noqa: BLE001 - a preference never costs the feed
            logging.debug("Show filter setting not saved.", exc_info=True)
        self._show_verdicts.clear()
        self._rebuild_feed()
        self.showFilterChanged.emit()
        self._emit_feed_status()

    def set_setup_grades(self, payload) -> None:
        """The day-trade grades the M5 bar shows; the Show filter reads the same."""
        import setup_grades

        self._show_grades = setup_grades.daytrade_lookup(payload)
        self._show_filter_inputs_changed(alert_show_filter.GRADE_B_UP)

    def set_best_now_entries(self, entries) -> None:
        """The Best-right-now strip's rows (P1-5); the Best filter shows only these."""
        keys = alert_show_filter.best_keys(entries)
        if keys == self._show_best_keys:
            return
        self._show_best_keys = keys
        self._show_filter_inputs_changed(alert_show_filter.BEST_NOW)

    def _show_filter_inputs_changed(self, affects: str | None = None) -> None:
        """An input the verdicts read changed: drop them and redraw by diff if it matters."""
        self._show_verdicts.clear()
        show_mode = self.show_filter_mode()
        if not self.show_filter_active():
            return
        if affects is not None and show_mode != affects and not self.first30_input.isChecked():
            return
        self._sync_feed()
        self.showFilterChanged.emit()
        self._emit_feed_status()

    def _on_focus_feed_coalesced(self) -> None:
        """The one reaction to a burst of Focus changes: a diff, never a rebuild.

        Focus membership is a Show-filter input, so the cached verdicts are
        dropped first; the diff then shows/hides the existing rows and the M5
        bar redraws from the same answer.
        """
        self._show_verdicts.clear()
        self._sync_feed()
        if self.show_filter_active():
            self.showFilterChanged.emit()
            self._emit_feed_status()

    def _on_show_filter_membership_changed(self, *_args) -> None:
        """Membership changed outside a Focus burst (typed lists): same diff."""
        self._show_filter_inputs_changed()

    def _check_typed_symbols(self) -> None:
        """longs.txt / shorts.txt changed (throttled stat): privileged rows may have changed."""
        try:
            typed = alert_show_filter.typed_symbols()
        except Exception:  # noqa: BLE001 - unknown shows
            return
        if self._show_typed_seen is None:
            self._show_typed_seen = typed
            return
        if typed != self._show_typed_seen:
            self._show_typed_seen = typed
            self._show_filter_inputs_changed()

    def show_filter_grade(self, alert: BounceAlert) -> str | None:
        return alert_show_filter.daytrade_grade(self._show_grades, alert)

    def _show_filter_privileged(self, alert: BounceAlert) -> bool:
        """Rows the Show filter never hides."""
        if is_chart_watch_alert(alert) or is_regime_pause_alert(alert) or is_entry_assist_alert(alert):
            return True
        if str(alert.raw_text or "").lstrip().upper().startswith("PRICE ALERT"):
            return True
        if self._alert_is_focus(alert):
            return True
        symbol = str(alert.symbol or "").strip().upper()
        try:
            return bool(symbol) and symbol in alert_show_filter.typed_symbols()
        except Exception:  # noqa: BLE001 - unknown shows
            return True

    def show_filter_verdict(self, alert: BounceAlert) -> tuple[bool, bool]:
        """`(hidden, is_new)` for one alert, cached until an input changes."""
        if not self.show_filter_active():
            return (False, False)
        cached = self._show_verdicts.get(id(alert))
        if cached is not None and cached[0] is alert:
            return cached[1]
        verdict = self._compute_show_verdict(alert)
        if len(self._show_verdicts) > MAX_FEED_ITEMS * 8:
            self._show_verdicts.clear()
        self._show_verdicts[id(alert)] = (alert, verdict)
        return verdict

    def _compute_show_verdict(self, alert: BounceAlert) -> tuple[bool, bool]:
        """Only ordinary M5 rows on a real symbol can hide."""
        if not str(alert.symbol or "").strip() or not self._is_m5_review_alert(alert):
            return (False, False)
        grade = self.show_filter_grade(alert)
        hidden = bool(self.show_filter_reason(alert, grade))
        return (hidden, grade == alert_show_filter.setup_grades.NEW)

    def show_filter_reason(self, alert: BounceAlert, grade: str | None = None) -> str:
        """Why the row hides: `first30`, the Show mode, or "" (shows)."""
        if not self.show_filter_active():
            return ""
        if not str(alert.symbol or "").strip() or not self._is_m5_review_alert(alert):
            return ""
        if grade is None:
            grade = self.show_filter_grade(alert)
        return alert_show_filter.hide_reason(
            self.show_filter_mode(),
            grade=grade,
            best=self._show_best_keys,
            symbol=alert.symbol,
            side=alert.side,
            privileged=self._show_filter_privileged(alert),
            first30=self.first30_input.isChecked(),
            when=alert_show_filter.alert_time(alert),
        )

    def show_filter_hides(self, alert: BounceAlert) -> bool:
        return self.show_filter_verdict(alert)[0]

    def _record_show_hidden(self, alert: BounceAlert) -> None:
        """B6: the first hide per (day, symbol, side) queues one `hidden_by_show` row.

        The Qt thread only checks an in-memory set; the store read and the append
        run on the evidence worker. Best-effort: a failure loses the row, never the alert.
        """
        path = self._review_events_path
        if path is None:
            return
        symbol = str(alert.symbol or "").strip().upper()
        side = str(alert.side or "").strip().upper()
        key = (self._ignored_market_date or date.today().isoformat(), symbol, side)
        if key in self._show_hidden_seen:
            return
        self._show_hidden_seen.add(key)
        try:
            grade = self.show_filter_grade(alert)
        except Exception:  # noqa: BLE001 - evidence never costs the alert
            grade = None
        detail = {"grade": grade or "", "show_mode": self.show_filter_mode()}
        try:
            reason = self.show_filter_reason(alert, grade)
        except Exception:  # noqa: BLE001 - evidence never costs the alert
            reason = ""
        if reason:
            detail["reason"] = reason
        try:
            if self._show_hidden_executor is None:
                from concurrent.futures import ThreadPoolExecutor

                self._show_hidden_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="show-hidden-evidence"
                )
            self._show_hidden_last = self._show_hidden_executor.submit(
                self._write_show_hidden, alert, symbol, side, detail, path
            )
        except Exception as exc:  # noqa: BLE001
            note_swallowed("hidden_by_show evidence not queued", exc, quiet=True)

    def _write_show_hidden(self, alert, symbol: str, side: str, detail: dict, path) -> None:
        """Evidence worker: skip a key already in today's store, else append the row."""
        try:
            import review_events

            day = review_events._trade_date_text()
            keys = self._show_hidden_store_keys.get(day)
            if keys is None:
                keys = {
                    (str(row.get("symbol") or "").upper(), str(row.get("side") or "").upper())
                    for row in review_events.load_review_events(path)
                    if row.get("action") == "hidden_by_show"
                    and str(row.get("trade_date") or "") == day
                }
                self._show_hidden_store_keys = {day: keys}
            if (symbol, side) in keys:
                return
            keys.add((symbol, side))
            record_review_event("hidden_by_show", alert=alert, detail=detail, path=path)
        except Exception as exc:  # noqa: BLE001 - a failed evidence write loses the event
            note_swallowed("hidden_by_show review event write failed", exc, quiet=True)

    def flush_show_hidden_writes(self, timeout: float = 10.0) -> None:
        """Wait for queued hidden_by_show writes (tests, shutdown)."""
        last = self._show_hidden_last
        if last is not None:
            try:
                last.result(timeout=timeout)
            except Exception as exc:  # noqa: BLE001
                note_swallowed("hidden_by_show flush failed", exc, quiet=True)

    def show_filter_hidden_counts(self) -> tuple[int, int]:
        """`(rows, new)` the Show filter holds back from the feed, one per name+side."""
        if not self.show_filter_active():
            return (0, 0)

        def verdicts():
            for alert in self._show_filter_candidates():
                hidden, is_new = self.show_filter_verdict(alert)
                yield (self._feed_row_key(alert), hidden, is_new)

        return alert_show_filter.count_hidden(verdicts())

    def show_filter_first30_count(self) -> int:
        """Rows (one per name+side) the first-30 switch holds back."""
        if not self.first30_input.isChecked():
            return 0
        first30 = alert_show_filter.REASON_FIRST30

        def verdicts():
            for alert in self._show_filter_candidates():
                hidden = self.show_filter_verdict(alert)[0]
                yield (
                    self._feed_row_key(alert),
                    hidden and self.show_filter_reason(alert) == first30,
                    False,
                )

        return alert_show_filter.count_hidden(verdicts())[0]

    def _show_filter_candidates(self):
        """Alerts the feed would show but for the Show filter."""
        mode = self._min_tier_mode()
        hide_sectors = sector_exclusion.hide_enabled()
        for alert in self._alerts:
            if (
                alert.symbol in self._ignored_symbols
                or self._sector_hidden(alert, hide_sectors)
                or not alert_passes_feed_gate(
                    alert, mode, is_focus=self._alert_has_focus_privilege(alert)
                )
            ):
                continue
            yield alert

    def _on_hide_sector_toggled(self, checked: bool) -> None:
        try:
            sector_exclusion.set_hide_enabled(bool(checked))
        except Exception:  # noqa: BLE001 - a preference never costs the feed
            logging.debug("Sector hide setting not saved.", exc_info=True)
        self._rebuild_feed()

    def sync_sector_switch(self) -> None:
        """Follow the shared switch when another surface flipped it."""
        stored = sector_exclusion.hide_enabled()
        if stored == self.hide_sector_input.isChecked():
            return
        self.hide_sector_input.blockSignals(True)
        self.hide_sector_input.setChecked(stored)
        self.hide_sector_input.blockSignals(False)
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
            if self._m5_unknown(symbol):
                return PREV_DAY_UNKNOWN
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
            if self._m5_unknown(symbol):
                return PREV_DAY_UNKNOWN
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

        The wall leg (2026-09-23) is asked last and only when no other leg has
        already hidden the name, because asking it arms follow-up watches.
        """
        legs = [
            self.mover_state(alert.symbol, alert.side),
            self.vwap_state(alert.symbol, alert.side),
        ]
        if self._is_d1_review(alert):
            legs.append(self.sma_trend_state(alert.symbol, alert.side))
        if PREV_DAY_CLOSED in legs:
            return PREV_DAY_CLOSED
        legs.append(self.wall_state(alert))
        if PREV_DAY_CLOSED in legs:
            return PREV_DAY_CLOSED
        if PREV_DAY_UNKNOWN in legs:
            return PREV_DAY_UNKNOWN
        return PREV_DAY_BREAK_OPEN

    def _review_badge_state(self, alert: BounceAlert) -> str:
        """What the review chart's badge says about the name in front of you.

        `open` (MOVING) needs the extreme leg verified and no later leg
        verified against it; a name revealed after the VWAP leg hid it says
        `wrong_side_vwap`, after the trend leg `wrong_side_sma`, and one at a
        wall (hidden and revealed, or shown because nothing could follow it
        up) `at_wall`; the extreme leg's own answers are unchanged. Reads the
        memoized wall verdict only - a render never arms anything.
        """
        mover = self.mover_state(alert.symbol, alert.side)
        if mover == PREV_DAY_UNKNOWN:
            # An unmeasured name that is verified at a wall says the wall.
            if self.wall_verdict(alert.symbol, alert.side).state == PREV_DAY_CLOSED:
                return "at_wall"
            return mover
        if mover != PREV_DAY_BREAK_OPEN:
            return mover
        if self.vwap_state(alert.symbol, alert.side) == PREV_DAY_CLOSED:
            return "wrong_side_vwap"
        if (
            self._is_d1_review(alert)
            and self.sma_trend_state(alert.symbol, alert.side) == PREV_DAY_CLOSED
        ):
            return "wrong_side_sma"
        if self.wall_verdict(alert.symbol, alert.side).state == PREV_DAY_CLOSED:
            return "at_wall"
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
            if self._m5_unknown(symbol):
                return PREV_DAY_UNKNOWN
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

    # ------------------------------------------------ the feed's own rows
    #
    # SN4 (WISHLIST item 4, trader 2026-09-08). Measured live at 13:16 that
    # day: a veto cost 4.0-4.1 s and one coalesced `focusChanged` cost 24.2 s,
    # because both called `_rebuild_feed`, which destroys and reconstructs up
    # to MAX_FEED_ITEMS + MAX_D1_FEED_ITEMS = 350 row widget trees on the Qt
    # thread. Removing one row and restyling one star are both O(1) jobs.
    #
    # The shape of the fix: `_feed_target_rows` says what the feed SHOULD look
    # like, and both paths read it - `_rebuild_feed` builds every row from it,
    # `_sync_feed` reconciles the rows already on screen against it. One
    # definition, so a diffed feed and a rebuilt feed cannot disagree; that
    # equality is what `tests/test_ws_sn4_feed_diff.py` asserts.
    #
    # Nothing here gates, scores, folds or records anything. The backing
    # lists, the repetition ledger, the review queue, History and every
    # evidence stream are written before any of this and are untouched by it.

    @staticmethod
    def _feed_row_key(alert: BounceAlert) -> tuple[str, str]:
        """One live row per symbol + side - the feed's own row identity."""
        return (str(alert.symbol or "").upper(), str(alert.side or "").upper())

    def _feed_row_registry(self) -> dict:
        rows = getattr(self, "_feed_rows", None)
        if rows is None:
            rows = {}
            self._feed_rows = rows
        return rows

    def _digested_keys(self) -> set:
        """The (symbol, side) keys the open-burst digest row stands in for.

        A digested alert deliberately has NO row of its own, so the target
        below has to know which keys those are - otherwise a veto would
        redraw the burst as forty rows, which is what the digest exists to
        prevent. Day-scoped with the ledger (see `_repetition_ledger`).
        """
        keys = getattr(self, "_digest_keys", None)
        if keys is None:
            keys = set()
            self._digest_keys = keys
        return keys

    def _feed_repeat_counts(self) -> dict:
        """Today's ×N counts, read from the ledger and never re-decided.

        `RepetitionLedger.consider` is a DECISION and counts the alert it is
        handed; a row being redrawn is not a new alert, so it re-stamps its
        badge from this snapshot instead of asking again.
        """
        try:
            return self._repetition_ledger().repeat_counts()
        except Exception:
            logging.debug("Repeat counts unavailable.", exc_info=True)
            return {}

    def _feed_target_rows(self) -> list:
        """`(key, alert, repeat_count)` per M5 row, top row first.

        Three rules, in order:

        * the ignore list and the minimum-tier gate decide what QUALIFIES -
          the same two filters the feed has always applied;
        * one row per (symbol, side): a name that has alerted three times has
          three entries in `self._alerts` and ONE row, and that row sits where
          the OLDEST of them put it, because a fold keeps the row's
          first-seen time and its place - which is what the trader is looking
          at when they veto the row above it;
        * the ×N badge comes from the ledger, so a redrawn row carries the
          count it had.
        """
        mode = self._min_tier_mode()
        hide_sectors = sector_exclusion.hide_enabled()
        qualifying = [
            alert
            for alert in self._alerts
            if alert.symbol not in self._ignored_symbols
            and not self._sector_hidden(alert, hide_sectors)
            and alert_passes_feed_gate(
                alert, mode, is_focus=self._alert_has_focus_privilege(alert)
            )
            and not self.show_filter_hides(alert)
        ]
        digested = self._digested_keys()
        if digested:
            # A digested name that has since been vetoed, gated out or aged
            # off the backing list is no longer being stood in for.
            digested &= {self._feed_row_key(alert) for alert in qualifying}
        rows: list = []
        seen: set = set()
        for alert in reversed(qualifying):  # oldest first
            key = self._feed_row_key(alert)
            if key in seen:
                continue
            seen.add(key)
            if key in digested:
                continue
            rows.append((key, alert))
        rows.reverse()  # newest row first - the order the layout draws
        del rows[MAX_FEED_ITEMS:]
        counts = self._feed_repeat_counts()
        return [(key, alert, int(counts.get(key, 1) or 1)) for key, alert in rows]

    def _d1_target_rows(self) -> list:
        """The D1 Focus feed's rows, top first, keyed on the alert itself.

        No fold and no tier gate here: the D1 feed shows one row per ready
        transition or pin, and two of them on one symbol are two events.
        """
        hide_sectors = sector_exclusion.hide_enabled()
        return [
            (id(alert), alert, 1)
            for alert in [
                alert
                for alert in self._d1_alerts
                if alert.symbol not in self._ignored_symbols
                and not self._sector_hidden(alert, hide_sectors)
            ][:MAX_D1_FEED_ITEMS]
        ]

    def _build_feed_row(self, alert: BounceAlert, *, repeat_count: int = 1):
        """One row widget, wired. The only place a feed row is constructed."""
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
        if repeat_count > 1:
            self._stamp_repeat_count(item, repeat_count)
        return item

    @staticmethod
    def _place_feed_row(layout, index: int, item) -> None:
        """Put one row into a feed layout at `index`, shown.

        Qt shows a widget added to a visible layout when that layout next
        activates, which is one trip through the event loop away; saying it
        here costs nothing, makes the row visible the moment it is placed,
        and means a row's own state answers "is this on screen" without
        waiting for a paint.
        """
        layout.insertWidget(index, item)
        item.show()

    @staticmethod
    def _stamp_repeat_count(item, count: int) -> None:
        try:
            item.set_repeat_count(int(count))
        except Exception:
            logging.debug("Repeat badge failed.", exc_info=True)

    def _restyle_feed_row(self, item) -> None:
        """Re-dress one surviving row for the current Focus lists.

        SN4's Focus half: the star's `focusOn` property, the gold frame and
        the ★ badge are set on the rows that changed and nowhere else. A row
        whose category is unchanged does nothing at all, so this is cheap to
        call for every visible row.
        """
        if self.focus_service is None:
            return
        symbol = str(getattr(item.alert, "symbol", "") or "")
        try:
            category = (self.focus_service.focus_category(symbol) or "") if symbol else ""
            item.feed_item.apply_focus_state(category)
        except Exception:
            logging.debug("Feed row restyle failed.", exc_info=True)

    def _insert_item_into(
        self, layout, alert: BounceAlert, max_items: int, *, repeat=None
    ) -> None:
        count = int(getattr(repeat, "repeat_count", 1) or 1) if repeat is not None else 1
        item = self._build_feed_row(alert, repeat_count=count)
        self._place_feed_row(layout, 0, item)
        if layout is self.feed_layout and alert.symbol:
            key = self._feed_row_key(alert)
            self._feed_row_registry()[key] = item
            # This name has a row of its own again, so the open-burst digest
            # is no longer standing in for it.
            self._digested_keys().discard(key)
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
        """Run an owed coalesced feed refresh now. The seam the tests drive."""
        coalescer = getattr(self, "_focus_feed_coalescer", None)
        if coalescer is not None:
            coalescer.flush()

    def _sync_layout_rows(self, layout, targets: list, *, key_of, track_rows: bool) -> None:
        """Reconcile one layout's row widgets against its target list.

        Take every row out of the layout (cheap: no widget is constructed and
        none is reparented), destroy the ones the target no longer wants, then
        put the survivors back in target order and build only what is missing.
        The layout OBJECT is never replaced, so the scroll area, the trailing
        stretch and the open-burst digest row all stay where they are.

        A duplicate key keeps the BOTTOM row - the oldest, which is the one
        already sitting at the target's position.
        """
        wanted = {key: (alert, count) for key, alert, count in targets}
        kept: dict = {}
        taken_items: list = []
        for index in range(layout.count() - 1, -1, -1):
            item = layout.itemAt(index)
            widget = item.widget() if item is not None else None
            if not isinstance(widget, _ClickableItem):
                continue
            taken_items.append(layout.takeAt(index))
            key = key_of(widget)
            if key in wanted and key not in kept:
                kept[key] = widget
            else:
                widget.setParent(None)
                widget.deleteLater()
        if track_rows:
            self._feed_rows = {}
        offset = 0
        digest_row = getattr(self, "_digest_row", None)
        if digest_row is not None and layout.indexOf(digest_row) == 0:
            offset = 1
        for position, (key, alert, count) in enumerate(targets):
            item = kept.get(key)
            if item is None:
                item = self._build_feed_row(alert, repeat_count=count)
            else:
                self._stamp_repeat_count(item, count)
                self._restyle_feed_row(item)
            self._place_feed_row(layout, offset + position, item)
            if track_rows:
                self._feed_rows[key] = item
        taken_items.clear()

    def _sync_feed(self) -> None:
        """Bring both feeds to the target state by DIFF, not by rebuild.

        What a veto and a Focus change call. The rows the trader did not
        touch are the same widgets afterwards, at the same positions, with
        their ×N badges intact.
        """
        targets = self._feed_target_rows()
        self._refresh_open_digest_row()
        self._sync_layout_rows(
            self.feed_layout,
            targets,
            key_of=lambda item: self._feed_row_key(item.alert),
            track_rows=True,
        )
        self._sync_layout_rows(
            self.d1_feed_layout,
            self._d1_target_rows(),
            key_of=lambda item: id(item.alert),
            track_rows=False,
        )

    def _rebuild_feed(self) -> None:
        """Destroy every row and draw both feeds again - the whole-feed repaint.

        Kept for the decisions that are about EVERY row at once: the
        minimum-tier switch, the day's Clear, and unpinning a D1 Focus name.
        Everything else goes through `_sync_feed`.

        Every row widget is about to be destroyed, so the fold registry and
        the digest row go with them - a registry pointing at deleted widgets
        would make the next repeat of each name silently fail over to a new
        row instead of folding. It then draws from the SAME target the diff
        reads, so "the feed after a veto" and "the feed after a rebuild" are
        the same feed.
        """
        self._feed_rows = {}
        self._digest_row = None
        self._clear_feed_layout(self.feed_layout)
        self._clear_feed_layout(self.d1_feed_layout)
        self._sync_feed()

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

    #: Most diverted EVENING alerts held for the catch-up card (references).
    _EVENING_CATCHUP_CAP = 2_000

    def _note_evening_catchup_alert(self, alert: BounceAlert) -> None:
        """Hold one EVENING-diverted alert for the catch-up card.

        References to rows already in the backing lists, not a new store; the
        list is session-scoped and reset when EVENING starts.
        """
        from datetime import date as _date

        today = _date.today().isoformat()
        if getattr(self, "_evening_catchup_session", None) != today:
            self._evening_catchup_session = today
            self._evening_catchup = []
        held = self._evening_catchup
        held.append(alert)
        if len(held) > self._EVENING_CATCHUP_CAP:
            del held[: len(held) - self._EVENING_CATCHUP_CAP]

    def evening_catchup_alerts(self) -> list[BounceAlert]:
        """This session's alerts diverted from the queue while in EVENING."""
        from datetime import date as _date

        if getattr(self, "_evening_catchup_session", None) != _date.today().isoformat():
            return []
        return list(getattr(self, "_evening_catchup", []) or [])

    def evening_started_at(self) -> datetime | None:
        """When this process saw Auto mode enter EVENING, if it did."""
        return getattr(self, "_evening_started_at", None)

    def evening_catchup_snapshot(self) -> dict:
        """Cheap copy of what the catch-up card needs from this panel.

        Tier and cell are read through the Alert Center's own functions, as
        the AWAY recap does; nothing is ranked here.
        """
        import working_lately

        alerts = []
        for alert in self.evening_catchup_alerts():
            try:
                cell = " ".join(working_lately.alert_priority_key(alert)).strip()
            except Exception:  # noqa: BLE001 - a row without a cell still lists
                cell = ""
            alerts.append(
                {
                    "symbol": str(getattr(alert, "symbol", "") or ""),
                    "side": str(getattr(alert, "side", "") or ""),
                    "tier": extract_alert_tier(alert),
                    "trigger": str(getattr(alert, "trigger", "") or ""),
                    "time_text": str(getattr(alert, "time_text", "") or ""),
                    "is_d1": bool(getattr(alert, "is_d1", False)),
                    "cell": cell if not getattr(alert, "is_d1", False) else "",
                }
            )
        board = getattr(self, "movers_board", None)
        try:
            movers = board.board() if board is not None else {}
        except Exception:  # noqa: BLE001
            movers = {}
        return {"alerts": alerts, "movers_board": movers, "since": self.evening_started_at()}

    def on_auto_mode_changed(self, previous: str, current: str) -> None:
        """Take a mode flip at once (slot for `autoModeChanged`).

        Entering EVENING starts a fresh catch-up list. Leaving it empties the
        waiting queue, so the trader comes back to zero charts; the chart on
        screen and every backing list are left alone.
        """
        from datetime import date as _date

        previous = str(previous or "").strip().upper()
        current = str(current or "").strip().upper() or "OFF"
        self._auto_mode_cached = (time.monotonic(), current)
        if current == "EVENING" and previous != "EVENING":
            self._evening_started_at = datetime.now()
            self._evening_catchup_session = _date.today().isoformat()
            self._evening_catchup = []
        if previous == "EVENING" and current != "EVENING":
            self._review_queue = []
            self.chart_review.set_queued_count(0)

    def away_recap_count(self) -> int:
        """How many alerts this session routed to the recap instead of the queue."""
        from datetime import date as _date

        if getattr(self, "_away_recap_session", None) != _date.today().isoformat():
            return 0
        return int(getattr(self, "_away_recap_diverted", 0) or 0)

    def _is_ordinary_d1_scan_review(self, alert: BounceAlert) -> bool:
        """Whether this is the scanner's ordinary D1 idea family.

        This is deliberately the real, narrow taxonomy rather than a guess
        from text.  Focus membership comes from the existing Focus provider;
        a chart's destination category is not evidence of membership.
        Unknown future D1 families fail open into the personal view.
        """
        return (
            bool(getattr(alert, "is_d1", False))
            and str(getattr(alert, "tag", "") or "").casefold().startswith("d1_flag_")
            and not self._alert_is_focus(alert)
        )

    def _held_d1_scan_review_count(self) -> int:
        """How many ordinary scan rows the My-alerts view is actually hiding."""
        current = self._current_review_alert
        claim_keys = self._active_claim_keys()
        return sum(
            1
            for alert in self._held_d1_scan_reviews.values()
            if self._held_d1_scan_review_is_eligible(alert, claim_keys=claim_keys)
            if not (
                current is not None
                and current.symbol == alert.symbol
                and self._is_ordinary_d1_scan_review(current)
            )
        )

    def _held_d1_scan_review_is_eligible(
        self, alert: BounceAlert, *, claim_keys: set[tuple[str, str]] | None = None
    ) -> bool:
        """A held row may be shown only while its normal queue path allows it."""
        if not self._is_ordinary_d1_scan_review(alert):
            return False
        if alert.symbol in self._ignored_symbols or alert.symbol in self._parked_symbols:
            return False
        if claim_keys is None:
            claim_keys = self._active_claim_keys()
        return (alert.symbol, alert.side) not in claim_keys

    def _refresh_d1_scan_review_view(self) -> None:
        self.chart_review.set_scan_review_view(
            show_all=self._show_all_d1_scan_reviews,
            hidden_count=self._held_d1_scan_review_count(),
        )

    def _on_focus_membership_changed(self, *_args) -> None:
        """Promote held rows that are now actual Focus members into review."""
        promoted = []
        for symbol, alert in list(self._held_d1_scan_reviews.items()):
            if self._alert_is_focus(alert):
                self._held_d1_scan_reviews.pop(symbol, None)
                promoted.append(alert)
        for alert in promoted:
            self._enqueue_review_alert(alert)
        self._refresh_d1_scan_review_view()

    def _toggle_d1_scan_review_view(self) -> None:
        """Toggle only which queued D1 scan ideas are displayed this session."""
        self._show_all_d1_scan_reviews = not self._show_all_d1_scan_reviews
        if self._show_all_d1_scan_reviews:
            # The cache has the latest row for each symbol.  Enqueueing uses
            # the existing ordering and prefetch paths; a manual chart in
            # front remains a manual chart (the same-symbol guard below).
            claim_keys = self._active_claim_keys()
            for alert in self._held_d1_scan_reviews.values():
                if self._held_d1_scan_review_is_eligible(alert, claim_keys=claim_keys):
                    self._enqueue_review_alert(alert)
        else:
            self._review_queue = [
                alert
                for alert in self._review_queue
                if not self._is_ordinary_d1_scan_review(alert)
            ]
            self.chart_review.set_queued_count(len(self._review_queue))
            self._prefetch_review_queue()
        self._refresh_d1_scan_review_view()

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
        # EVENING diverts too (trader, 2026-09-23): the trader flips out of it
        # to an EMPTY queue and reads the catch-up card instead.
        mode = self._auto_mode_now()
        if mode == "AWAY":
            self._note_away_recap_alert(alert)
            return
        if mode == "EVENING":
            self._note_evening_catchup_alert(alert)
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
        # Packet D1C-A (trader, 2026-09-14): *"Once claimed, keep that same D1
        # setup out of repeat review while the claim remains active. ...
        # Preserve M5 entry review for that symbol. This is a D1 review-queue
        # change, not symbol-wide alert suppression."*
        #
        # So it sits HERE: after the parked check, and BEFORE the M5 routing
        # below, which means every M5 alert still reaches the M5 bar exactly as
        # it does today. It keys on (symbol, SIDE) - a claimed LONG says
        # nothing about a SHORT thesis - and on D1 SCAN alerts only: a
        # chart-watch hit is a condition the trader armed and is waiting on.
        # Everything upstream of this line is untouched: the feed, History, the
        # D1 badge, the evidence streams, the AWAY recap and the phone push are
        # all written before it. Detection, the alert row and the outcome
        # record are not this line's business at all.
        if (
            alert.is_d1
            and not is_chart_watch_alert(alert)
            and (alert.symbol, alert.side) in self._active_claim_keys()
        ):
            self._claimed_d1_skipped[alert.symbol] += 1
            self.chart_review.set_claimed_skipped_count(
                sum(self._claimed_d1_skipped.values())
            )
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
            self._attach_held_run_suffix(alert)
            self.m5AlertPosted.emit(alert)
        # Default My alerts holds only the scanner's recognized ordinary D1
        # taxonomy.  It happens before same-symbol refresh, so a routine scan
        # cannot replace a manual/personal reason on the chart already being
        # reviewed.  The alert was already written to its backing feed.
        ordinary_d1_scan = self._is_ordinary_d1_scan_review(alert)
        if ordinary_d1_scan:
            self._held_d1_scan_reviews[alert.symbol] = alert
            self._refresh_d1_scan_review_view()
            if not self._show_all_d1_scan_reviews:
                return
        if (
            self._current_review_alert is not None
            and self._current_review_alert.symbol == alert.symbol
        ):
            if ordinary_d1_scan:
                # Reclassifying a scan is never a chart selection.  A Show-all
                # click (or a later same-symbol scan while it is on) leaves
                # every current chart and its capture draft intact, whether it
                # came from a typed look, Focus, a price arm, a watch hit, or
                # another ordinary scan.
                pass
            else:
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

    @staticmethod
    def _is_manual_chart_look(alert: BounceAlert) -> bool:
        """A chart the TRADER opened - a board row click or the lookup box.

        `chart_symbol` is the one door that builds these and it stamps
        `MANUAL_CHART_TAG` on every one, so this is an exact test rather than
        a heuristic; nothing the scanner produces carries that tag.

        Deliberately NOT folded into `_is_m5_review_alert`, which already
        returns False for this tag and must keep doing so: that method decides
        whether an alert is a LINE IN THE M5 BAR, and a look is not. Two
        different questions, two different answers, one tag.
        """
        return str(getattr(alert, "tag", "") or "") == MANUAL_CHART_TAG

    def chart_alert(self, alert: BounceAlert) -> None:
        """Public: chart this alert now (the M5 bar's click). Same path as a
        feed-row click. A D1 chart in front keeps its place at the head of
        the queue; an M5 chart in front is skipped (trader rule 2026-08-27,
        second pass - see `_select_review_alert`)."""
        # An explicit click back onto an ignored chart is re-engagement, just
        # like typing that ticker in the lookup box.  It restores the day
        # state before charting; otherwise a direct chart request silently did
        # nothing and a following capture applied to whichever chart was old.
        if alert.symbol in self._ignored_symbols:
            self._restore_ignored_symbol(alert.symbol)
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

    def _attach_held_run_suffix(self, alert: BounceAlert) -> None:
        """Hand the M5 bar "held NN% / ran N.NR", IF the cell is already known.

        R4 A10, decision 0016 answer 4. `held_run_score.segment_index` said it
        existed "for a per-alert lookup" and nothing ever built the key, so the
        bar's rows carried no such suffix at all.

        A DICT READ, and never anything more. The index is built once per
        session on a worker started by the first M5 alert; until it lands there
        is no suffix, which is the honest rendering of "not measured". The same
        rule the take-rate suffix follows, and for the same reason: a file read
        in the alert path is the drip three snappiness packets removed.

        Blank below the evidence floor - `alert_suffix` enforces that, not this.
        A row reading "held 100% / ran 3.2R (n=2)" is read as a strong segment
        at a glance, and a glance is what the row is for.
        """
        try:
            self._ensure_held_run_index()
            if not self._held_run_index:
                return
            import held_run_score

            feedback = alert.payload.get("feedback") if isinstance(alert.payload, dict) else None
            feedback = feedback if isinstance(feedback, dict) else {}
            bounce_type = str(
                (feedback.get("bounce_types") or "").split(";")[0]
            ).strip() or str(alert.trigger or "").strip()
            environment = str(feedback.get("market_environment") or "").strip()
            trade_date = str(feedback.get("trade_date") or "").strip() or date.today().isoformat()
            entry_time = f"{trade_date}T{str(alert.time_text or '')[:8]}"
            cell = held_run_score.alert_cell(
                self._held_run_index,
                bounce_type=bounce_type,
                entry_time=entry_time,
                market_environment=environment,
                # Packet Q1: the SIDE travels with the join. A SHORT swing setup
                # on a long M5 alert is OPPOSED, not "a D1 setup"; no snapshot
                # is UNKNOWN, not False.
                d1_alignment=held_run_score.d1_alignment(
                    self._held_run_d1_symbols, trade_date, alert.symbol, alert.side
                ),
            )
            alert.held_run_suffix = held_run_score.alert_suffix(cell)
        except Exception:  # noqa: BLE001 - a row suffix never costs an alert
            logging.debug("Held/ran suffix skipped for %s.", alert.symbol, exc_info=True)

    def _held_run_day(self) -> str:
        """The desk-local date the held/ran index is about.

        A method so the day roll has one seam and a test can move the clock.
        The SAME string `_attach_held_run_suffix` falls back to when an alert
        carries no `trade_date`, so the memo and the lookup can never disagree
        about which day they mean.
        """
        return date.today().isoformat()

    def _ensure_held_run_index(self) -> None:
        """Start the one background build, once per process PER DAY. Never blocks.

        The memo is module-level rather than per panel because the read is ~90 MB
        of outcome log plus a 19 MB snapshot and the answer is the same for every
        panel in the process - a second Alert Center (a test, a second window)
        must not pay for it again. A panel that finds a memo built for TODAY
        takes it immediately and starts nothing.

        A memo built for an earlier day is discarded and rebuilt, on the worker,
        at the first M5 alert of the new day. See :data:`_HELD_RUN_INDEX_MEMO`
        for what a memo that never expired cost.
        """
        global _HELD_RUN_INDEX_MEMO
        today = self._held_run_day()
        if self._held_run_built_for == today or self._held_run_thread is not None:
            return
        memo = _HELD_RUN_INDEX_MEMO
        if isinstance(memo, dict) and memo.get("built_for") == today:
            self._on_held_run_index_loaded(memo)
            return
        import threading

        self._held_run_thread = threading.Thread(
            target=self._held_run_index_worker,
            name="alert-center-held-run",
            daemon=True,
        )
        self._held_run_thread.start()

    def _held_run_index_worker(self) -> None:
        global _HELD_RUN_INDEX_MEMO
        payload = {"index": {}, "d1": {}, "built_for": self._held_run_day()}
        try:
            import held_run_score

            from project_paths import MASTER_AVWAP_TRACKER_SCORING_SNAPSHOT_FILE

            episodes = held_run_score.load_episodes()
            payload["index"] = held_run_score.segment_index(
                held_run_score.build_segments(episodes)
            )
            payload["d1"] = held_run_score.d1_setups_by_session(
                held_run_score.d1_setup_rows(MASTER_AVWAP_TRACKER_SCORING_SNAPSHOT_FILE)
            )
        except Exception:  # noqa: BLE001 - an absent suffix is a real answer
            logging.debug("Held/ran index unavailable.", exc_info=True)
        # Memoised even when it came back empty: a failed read is an answer, and
        # retrying a 90 MB parse on every alert is exactly the drip this avoids.
        _HELD_RUN_INDEX_MEMO = payload
        try:
            self._heldRunIndexLoaded.emit(payload)
        except RuntimeError:
            # The panel went away mid-read. Nothing left to update; the memo is
            # still stamped, so the next panel in this process gets it free.
            self._held_run_thread = None

    def _on_held_run_index_loaded(self, payload) -> None:
        if not isinstance(payload, dict):
            return
        self._held_run_index = payload.get("index") or {}
        self._held_run_d1_symbols = payload.get("d1")  # None = no snapshot = UNKNOWN (packet Q1); never flattened to {}
        # Stamped LAST, so a payload that arrived for a day that has already
        # rolled leaves this panel asking for a fresh one rather than settling.
        self._held_run_built_for = str(payload.get("built_for") or "")
        self._held_run_thread = None

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

        A chart the trader opened from a BOARD or the lookup box holds no
        place either, and unlike the M5 bar it is not even a skip (packet T1,
        trader 2026-09-04: *"when I click on ANYTHING from the RS/RW board it
        should not make a queue of picks if I click on more nor should it add
        to the 'waiting' list. once i look and click off, its done."*). It was
        a LOOK, never a shown alert, so it belongs in no P(take | shown)
        denominator - as neither a take nor a skip. Nothing at all is written.
        """
        if not alert.symbol or alert.symbol in self._ignored_symbols:
            return
        if not self._is_manual_chart_look(alert):
            # Any other chart taking the pane ends the setups-table walk.
            self._manual_next_pick = None
        current = self._current_review_alert
        if current is not None and current.symbol != alert.symbol:
            self._review_queue = [
                queued
                for queued in self._review_queue
                if queued.symbol not in {current.symbol, alert.symbol}
            ]
            if self._current_review_holds_place:
                self._review_queue.insert(0, current)
            elif self._is_manual_chart_look(current):
                # A look the trader opened themselves. Write NOTHING - see the
                # docstring. The M5 branch below is a different population.
                pass
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
        self._current_review_holds_place = not (
            self._is_m5_review_alert(alert) or self._is_manual_chart_look(alert)
        )
        self._render_current_review()

    def set_working_lately_order(self, order) -> None:
        """`[(bounce_type, SIDE)]`, best first - the priority switch (ST6.5).

        Presentation only, and applied at the moment the panel decides the NEXT
        chart, never when a row is written. The backing list, every evidence
        write, the tier gate and the movers-only filter are untouched: the same
        names are shown either way and only the order differs.
        """
        self._working_lately_order = [
            (str(cell), str(side)) for cell, side in (order or ())
        ]

    def _next_review_index(self) -> int:
        """Which WAITING row to take next. The list itself is never reordered.

        **ST6 re-review, blocker 1.** This used to re-sort `_review_queue` in
        place, which made the switch irreversible: the stored order was gone, so
        turning the switch off left the queue in the order the switch had put it
        and the next chart was still the prioritised one. The backing list is the
        record of what the day produced and a display preference may not rewrite
        it - so the priority order is applied HERE, at the moment the panel picks
        the next chart, by choosing an index rather than by moving anything.
        Ties keep arrival order; with the switch off this is always 0, which is
        exactly today's behaviour.
        """
        import working_lately

        order = getattr(self, "_working_lately_order", None)
        if not order or len(self._review_queue) < 2:
            return 0
        if not working_lately.prioritise_enabled():
            return 0
        return min(
            range(len(self._review_queue)),
            key=lambda index: (
                working_lately.priority_rank(
                    order,
                    working_lately.alert_priority_key(self._review_queue[index]),
                ),
                index,
            ),
        )

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
        # Trader, 2026-09-15: a chart opened FROM the setups table advances to
        # that table's next row, not to the waiting list. The callback is
        # consumed here; the row it charts brings its own.
        advance = getattr(self, "_manual_next_pick", None)
        self._manual_next_pick = None
        if advance is not None:
            try:
                if advance():
                    return
            except Exception:
                logging.warning("Setups-table advance failed; falling back to the waiting list.", exc_info=True)
        hidden_before = len(self._hidden_inside_range)
        next_alert = None
        while self._review_queue:
            candidate = self._review_queue.pop(self._next_review_index())
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
        except Exception as exc:
            note_swallowed("review queue chart prefetch failed", exc, quiet=True)

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
        if self._deferred_review_events is not None:
            self._deferred_review_events.append((action, dict(kwargs)))
            return
        try:
            record_review_event(action, path=self._review_events_path, **kwargs)
        except Exception as exc:
            note_swallowed("alert review event write failed", exc)

    def _record_review_events(self, entries) -> None:
        """Many rows, ONE kernel lock and ONE open (PCT-1 review blocker 3a).

        The Pullback alert's auto-arm can produce dozens of machine rows in
        one tick - 95 on its first one against the live stores - and the
        per-row writer takes the lock each time, on the Qt thread. Same
        best-effort contract: a failure loses the rows, never the arm.
        """
        if self._review_events_path is None or not entries:
            return
        try:
            from review_events import record_review_events

            record_review_events(entries, path=self._review_events_path)
        except Exception as exc:
            note_swallowed("alert review events batch write failed", exc)

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
            # A proxy symbol not fetched yet is unknown, not empty: no refetch.
            symbols = [sym for sym in symbols if not self._m5_unknown(sym, sessions=2)]
            if not symbols:
                return
            bot = self._current_bot()
            if bot is None:
                return
            shared_refresh_service().refresh_if_stale(
                symbols, lambda sym: self._m5_bars_for(sym, sessions=2), bot
            )
        except Exception as exc:
            # Display refresh only - it must never break the watch tick that
            # shares this timer.
            note_swallowed("stale review-queue bar refresh failed", exc)

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
        except Exception as exc:
            # Display refresh only - it must never break the watch tick that
            # shares this timer.
            note_swallowed("review chart refresh failed", exc)

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
        pending = self.pending_arm_kinds(alert.symbol)
        self.chart_review.set_pending_arms(
            pending["watch"], pending["d1_event"], bool(pending["any_bounce"])
        )
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
            # Packet D1C-A item 2: the horizon of a claim made on this chart.
            # The panel owns `_is_m5_review_alert`; the pane never imports the
            # panel, so the answer travels with the alert.
            is_m5_review=self._is_m5_review_alert(alert),
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

    # ------------------------------------------------------------------
    # Packet D1C-A: a claimed D1 like is a pick, and the chart is done
    # ------------------------------------------------------------------
    def _active_claim_keys(self) -> set[tuple[str, str]]:
        """`(symbol, side)` for every active claim, from an mtime-keyed cache.

        One small read when the claims file CHANGED, never one per alert. The
        day is part of the key because the fade is a session clock: a desk left
        running past midnight has to ask again.
        """
        path = getattr(self, "_claimed_picks_path", None)
        if path is None:
            return set()
        try:
            stat = Path(path).stat()
            stamp: object = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            stamp = None
        today = self._ignored_market_date
        if stamp == self._claim_keys_stamp and today == self._claim_keys_day:
            return self._claim_keys_cache
        keys: set[tuple[str, str]] = set()
        if stamp is not None:
            try:
                import claimed_picks

                keys = claimed_picks.active_keys(Path(path))
            except Exception:  # noqa: BLE001 - an unreadable store gates nothing
                keys = set()
        self._claim_keys_cache = keys
        self._claim_keys_stamp = stamp
        self._claim_keys_day = today
        return keys

    def _write_claim(self, **fields) -> dict | None:
        """Write one claim into THIS panel's store. None when nothing was written.

        The seam between the pane (which knows a claim was made) and the store
        (which knows where the desk's claims live). It never raises: a failed
        placement is an answer the caller acts on, not an exception on a click.
        """
        path = getattr(self, "_claimed_picks_path", None)
        if path is None:
            return None
        try:
            import claimed_picks

            row = claimed_picks.record_claim(path=Path(path), **fields)
        except Exception:  # noqa: BLE001 - a failed placement keeps the chart
            return None
        if row is not None:
            self._claim_keys_stamp = _CLAIM_KEYS_UNREAD
        return row

    def _on_claims_changed(self) -> None:
        """A claim was placed or dropped: the gate re-reads on the next alert."""
        self._claim_keys_stamp = _CLAIM_KEYS_UNREAD

    def _sweep_expired_claims(self) -> None:
        """The day roll's half of the lifecycle: one `expire` row per fade.

        Here and nowhere else. The fade can only change when the session does,
        so a timer asking more often would buy nothing and cost a file read; a
        calendar that cannot answer expires nothing.
        """
        path = getattr(self, "_claimed_picks_path", None)
        if path is None:
            return
        try:
            import claimed_picks

            claimed_picks.sweep_expired(Path(path))
        except Exception as exc:  # noqa: BLE001 - a sweep never costs the day roll
            note_swallowed("expired claim sweep failed", exc)
        self._claim_keys_stamp = _CLAIM_KEYS_UNREAD

    def _place_claimed_d1(self, alert: BounceAlert, claim_row: object) -> None:
        """The pick was SAVED: record the like, say so, and finish the chart.

        Trader, 2026-09-14: *"Save and confirm the pick before removing its D1
        item from Visual Chart Review."* The save already happened - the pane
        only emits `claimPlaced` with a row in hand - so this is the confirm
        and the retire, in that order.

        `_record_like_advance` is the UNCHANGED writer, so the review
        scoreboard's take side sees exactly what it saw before: one
        `like_advance` row, under the historical name `review_learning` keys
        on. No second verdict is written (P5), and the retirement is
        `_retire_claimed_review`, never the parking verb.
        """
        if not self._record_like_advance(alert):
            return
        setup = ""
        if isinstance(claim_row, dict):
            setup = str(claim_row.get("claimed_setup_id") or "").strip()
        self.statusChanged.emit(
            f"♥ {alert.symbol}: claimed {setup} - placed in Setups."
            if setup
            else f"♥ {alert.symbol}: claimed - placed in Setups."
        )
        self._retire_claimed_review(alert)
        self.claimsChanged.emit()
        self.reviewDecisionRecorded.emit()

    def _retire_claimed_review(self, alert: BounceAlert) -> None:
        """Take a claimed chart out of today's review, and do NOTHING else.

        Deliberately a separate method from `_retire_review_alert`, not a flag
        on it. That body is the PARKING verb: it writes a `remove_today` review
        event, adds the symbol to `_parked_symbols`, drops an auto-adopted
        Focus pick and runs three early-return branches for auto picks, faded
        picks and Focus reviews. A claim is none of those things - the trader
        said YES to this name - and a flag threaded through that ladder would
        be one edit away from parking a name they just claimed.

        So: the alert leaves the current slot, the waiting list and the hidden
        set, and the next chart comes up. The symbol keeps alerting, keeps its
        place in the feed and the phone push, and its M5 alerts are untouched.
        The repeat-D1 gate in `_enqueue_review_alert` is what stops the same
        chart coming back, and only while the claim is active.
        """
        symbol = getattr(alert, "symbol", "")
        side = getattr(alert, "side", "")
        if not symbol:
            return
        held = self._held_d1_scan_reviews.get(symbol)
        if held is not None and held.side == side:
            # A claim answers this exact scan thesis.  Do not leave its
            # presentation cache behind to be reintroduced after a view flip.
            self._held_d1_scan_reviews.pop(symbol, None)
            self._refresh_d1_scan_review_view()
        self._review_queue = [
            queued
            for queued in self._review_queue
            if not (queued.symbol == symbol and queued.side == side)
        ]
        self._hidden_inside_range.pop(symbol, None)
        current = self._current_review_alert
        if current is not None and current.symbol == symbol:
            self._current_review_alert = None
        self._advance_review_queue()

    def _record_like_advance(self, alert: BounceAlert) -> bool:
        """The review event both like verbs write. ONE copy, deliberately.

        The action is ``like_advance``, and **the NAME IS HISTORICAL AND MUST
        NOT CHANGE** - ``review_learning.TAKE_ACTIONS`` keys on the exact
        string, and renaming it would drop every past like out of the take side
        of the scoreboard. What it MEANS depends on the mode the like was made
        in (quick: the chart stays; claimed: the chart advances), and that
        difference is the caller's, not this row's.

        Two handlers writing their own copy of this call is exactly how the
        quick and claimed paths would drift - one gaining a dwell field or a
        detail the other never got - so there is one.
        """
        if alert is None or not alert.symbol:
            return False
        self._record_review_event(
            "like_advance",
            alert=alert,
            dwell_ms=self._review_dwell_ms(alert.symbol),
            queue_len=len(self._review_queue),
        )
        return True

    def _advance_after_like(self, alert: BounceAlert) -> None:
        """A CLAIMED like: record it, say so, and show the next chart.

        Trader, 2026-09-04, second pass (packet T2): *"for the 'like and claim'
        part of the capture tab, a double click of any of the setups there
        should be sufficient. I shouldnt have to type anything below that box.
        and then double clicking that box should advance the chart."*

        The claimed like has named the setup, so the trader is done with the
        chart; the QUICK like has named nothing and stays (`_after_like`).

        An advance is NOT a retirement, and everything the "Not today" route
        does is still not done here: ``_ignored_symbols`` is untouched, so the
        name keeps alerting and keeps reaching the hourly D1 phone push; no
        auto-adopted Focus pick is dropped; the symbol's other queued alerts
        keep their places and simply come round again; and nothing is placed -
        a like carries zero privileges (plan.md P3.1).
        """
        if not self._record_like_advance(alert):
            return
        self.statusChanged.emit(
            f"♥ {alert.symbol}: liked and claimed - next chart."
        )
        self._advance_review_queue()

    def _after_like(self, alert: BounceAlert) -> None:
        """A QUICK LIKE is RECORDED and the chart STAYS (packet T1, 2026-09-04).

        Trader: *"the 'like' button in the visual chart review should NOT
        advance the char to the next page because i still need time to enter
        alerts etc."* So this records and says so, and moves nothing: the
        waiting list is untouched, the liked chart is still on screen, and the
        trader leaves it with Skip or "Not today" when they are done arming.

        Packet T2 (2026-09-04, second pass) narrowed this to the QUICK like -
        Alt+L, the rail's "♥ Quick like", the chart's "♥ Like" button. A
        CLAIMED like has named its setup and advances (`_advance_after_like`).

        Everything this function deliberately does NOT do was previously done
        to every liked symbol, because a like was routed through
        ``_remove_review_alert_for_today``: it does not touch
        ``_ignored_symbols``, so the name keeps alerting and keeps reaching the
        hourly D1 phone push; it does not drop an auto-adopted Focus pick; it
        does not sweep the symbol's other queued alerts. It also does not
        place anything - the capture rail is analysis-only and the explicit
        Focus verb remains the one thing that places.

        What it does record is ``like_advance``, which
        ``review_learning.TAKE_ACTIONS`` reads as positive engagement. **The
        NAME IS HISTORICAL AND MUST NOT CHANGE** - that module keys on the
        exact string, and renaming it would drop every past like out of the
        take side of the scoreboard. Since 2026-09-04 it means "liked; the
        symbol keeps alerting and the chart stays". The route it replaced wrote
        ``remove_today``, and ``REJECT_ACTIONS`` scored 40 of the window's 52
        likes as dismissals (R9.2).
        """
        if not self._record_like_advance(alert):
            return
        self.statusChanged.emit(
            f"♥ {alert.symbol}: liked. The chart stays - arm your alerts, then "
            "Skip or Not today."
        )

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

        The retire is `_retire_after_veto`, not the "Not today" button's verb
        (lead ruling 2026-09-04): this click came from the capture rail with a
        reason code already on disk, so the trader's "either veto or
        like+claim ... no pop up note box" covers it. The ORDER is unchanged
        and a failed placement still retires.
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
        self._retire_after_veto(alert)

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
                if is_process_proxy(bot):
                    # An RPC on the proxy: the cached label, blank until fetched.
                    current = str(
                        shared_m5_cache().peek_value(bot, "get_market_environment", call=True)
                        or ""
                    )
                elif bot is not None:
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
                if self._m5_unknown(symbol):
                    held += 1  # bars not fetched yet: the break state stays as it was
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
        # owning a timer of their own (trader rule 2026-08-19). Emit only when
        # a state changed: each emit repaints the whole Focus board.
        if self._focus_break_state != self._focus_break_emitted:
            self._focus_break_emitted = dict(self._focus_break_state)
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
        except OSError as exc:
            note_swallowed("focus D1 flags write failed", exc)

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
        """Which kinds are ARMED on this symbol - what the buttons read.

        A DECLINED row is remembered, not armed (PCT-1 item 5), so it is not
        in here: the button must read "not armed" or the trader is looking at
        a pressed toggle beside an empty Armed board and has no way back
        (review blocker 6). The row itself still exists, and
        `arm_chart_watch_for` is what clears it.
        """
        symbol = str(symbol or "").strip().upper()
        return {
            watch.kind
            for watch in self._chart_watches
            if watch.symbol == symbol and not bool(getattr(watch, "declined", False))
        }

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

        On the process proxy this reads `shared_m5_cache()` only - never an
        RPC on the Qt thread - and bars not fetched yet come back as [].
        Callers that must not treat "not fetched" as "no bars" ask
        `_m5_unknown` first.
        """
        bars = self._m5_cached(symbol, sessions=sessions)
        return bars if bars is not None else []

    def _m5_cached(self, symbol: str, *, sessions: int = 1) -> list | None:
        """Memory-only M5 bars; None means a proxy bot's bars are not fetched yet."""
        bot = None
        if self._bounce_service is not None:
            try:
                bot = self._bounce_service.current_bot()
            except Exception:
                bot = None
        if bot is None:
            return []
        if is_process_proxy(bot):
            return shared_m5_cache().peek(bot, symbol, sessions)
        return self._m5_local_bars(bot, symbol, sessions=sessions)

    def _m5_unknown(self, symbol: str, *, sessions: int = 1) -> bool:
        """True while a proxy bot's bars for this symbol have never been fetched."""
        return self._m5_cached(symbol, sessions=sessions) is None

    def _m5_local_bars(self, bot, symbol: str, *, sessions: int = 1) -> list:
        """An in-process bot's bars, memoized per source series (item 1a)."""
        key = (str(symbol or "").strip().upper(), max(1, int(sessions)))
        try:
            source = self._m5_source_bars(bot, symbol)
        except Exception:
            source = []
        stamp = self._m5_source_stamp(source)
        cached = self._m5_bar_dicts.get(key)
        same_source = cached is not None and cached[0] is source
        if cached is not None and same_source and cached[1] == stamp:
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
        self,
        symbol: str,
        side: str,
        kind: str,
        *,
        source_text: str = "",
        timeframes: tuple[str, ...] = (),
        m5_bars: list | None = None,
    ) -> bool:
        """Public arming surface for any visual chart. Returns True on arm.

        ``m5_bars`` is the baseline a queued arm fetched off the Qt thread;
        None reads the desk's cached bars as before.

        A DECLINED row for this (symbol, kind) is DROPPED first (review
        blocker 6): the trader turned an auto-armed watch off and is now
        turning it back on from the chart, which has to work. What comes back
        is a HAND-armed watch - this call's own `source_text`, a fresh
        `watch_id`, nothing fired - so it is a new episode and the auto-arm
        sweep no longer owns it.
        """
        symbol = str(symbol or "").strip().upper()
        if not symbol or kind not in WATCH_KINDS:
            return False
        label = WATCH_KINDS[kind]
        if kind in self.armed_watch_kinds(symbol):
            self.statusChanged.emit(f"{symbol}: {label} watch already armed.")
            return False
        candidate_watches = [
            existing
            for existing in self._chart_watches
            if not (existing.symbol == symbol and existing.kind == kind)
        ]
        watch = arm_chart_watch(
            kind,
            symbol,
            side,
            # A Pullback watch has no M5 baseline. Reading the chart cache here
            # can take its lock while a refresh owns it, so a button press must
            # never pay for bars this watch does not use.
            ()
            if kind == PULLBACK_KIND
            else (m5_bars if m5_bars is not None else self._m5_bars_for(symbol)),
            source_text=source_text,
            timeframes=timeframes,
        )
        candidate_watches.append(watch)
        # An arm is a trader-visible promise.  Keep the in-memory collection
        # untouched when its durable row cannot be saved, so a later poll can
        # never fire an alert the desk only pretended to arm.
        if not self._save_chart_watches(candidate_watches):
            self.statusChanged.emit(f"{symbol}: NOT ARMED - the watch file could not be saved.")
            return False
        self._chart_watches = candidate_watches
        self._refresh_review_armed_kinds()
        self.armedWatchesChanged.emit()
        self._record_review_event(
            "arm_watch",
            alert=self._arm_review_alert(symbol),
            symbol=symbol,
            side=side,
            dwell_ms=self._arm_dwell_ms(symbol),
            detail={"kind": kind, "baseline": watch.baseline},
        )
        if kind == PULLBACK_KIND:
            self.statusChanged.emit(
                f"{symbol}: {label} armed - {watch.reason}. "
                f"{self._h1_warmup_note(watch)}"
            )
            return True
        level = f" against {watch.baseline:.2f}" if watch.baseline is not None else ""
        self.statusChanged.emit(
            f"{symbol}: {label} watch armed{level} - the first completed "
            "M5 bar that meets it flags red in the Alert Center."
        )
        return True

    def disarm_chart_watch_for(self, symbol: str, kind: str) -> bool:
        """Public disarm surface (the toggles' off-click). True if removed.

        A watch the trader armed BY HAND is deleted, exactly as it always was.
        A `pullback` watch the desk armed for them off a claimed D1 pick or a
        swing Focus name is instead kept as `declined` (PCT-1 item 5, lead
        ruling 3): the auto-arm sweep runs every 60 seconds, so deleting it
        would simply put it back on the next tick and the trader could never
        turn one off. A declined row is hidden from the Armed board, never
        evaluated and never pushed, and it goes when its claim or pick does.
        """
        symbol = str(symbol or "").strip().upper()
        if kind not in self.armed_watch_kinds(symbol):
            return False
        kept: list = []
        declined: list = []
        for watch in self._chart_watches:
            if not (watch.symbol == symbol and watch.kind == kind):
                kept.append(watch)
                continue
            if self._is_auto_pullback_watch(watch) or self._is_wall_pullback_watch(watch):
                remembered = replace(watch, declined=True)
                declined.append(remembered)
                kept.append(remembered)
            # else: a hand-armed watch is simply dropped, as it always was.
        self._chart_watches = kept
        if declined:
            self._save_chart_watches()
            self._refresh_review_armed_kinds()
            self.armedWatchesChanged.emit()
            self._record_review_event(
                "disarm_watch",
                symbol=symbol,
                detail={"kind": kind, "declined": True},
            )
            self.statusChanged.emit(
                f"{symbol}: {WATCH_KINDS.get(kind, kind)} disarmed - it stays "
                "off until the pick that armed it is dropped."
            )
            return True
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
        self.toggle_chart_watch(alert.symbol, alert.side, kind, source_text=alert.raw_text)

    def toggle_chart_watch(self, symbol: str, side: str, kind: str, *, source_text: str = "") -> None:
        """A watch button click: cancel a queued arm, disarm, or arm (queued on a proxy)."""
        symbol = str(symbol or "").strip().upper()
        if not symbol or self._cancel_pending_arm(("watch", symbol, kind)):
            return
        if kind in self.armed_watch_kinds(symbol):
            self.disarm_chart_watch_for(symbol, kind)
            return
        if not self._arms_async() or kind not in WATCH_KINDS:
            self.arm_chart_watch_for(symbol, side, kind, source_text=source_text)
            return
        bot = self._current_bot()
        prepare = None
        if kind != PULLBACK_KIND and is_process_proxy(bot):

            def prepare():
                try:
                    return shared_m5_cache().fetch_now(bot, symbol, 1)
                except Exception:  # an unreadable child arms on the cached bars, as before
                    logging.debug("Arm baseline fetch failed for %s.", symbol, exc_info=True)
                    return None

        self._queue_arm(
            "watch",
            symbol,
            kind,
            WATCH_KINDS[kind],
            prepare,
            lambda bars: self.arm_chart_watch_for(
                symbol, side, kind, source_text=source_text, m5_bars=bars
            )
            or kind in self.armed_watch_kinds(symbol),
        )

    # -- queued arms ---------------------------------------------------------
    def _arm_queue(self):
        if self._arm_queue_obj is None:
            from ui.services.arm_queue import ArmQueue

            self._arm_queue_obj = ArmQueue(self)
            self._arm_queue_obj.jobChanged.connect(self._on_arm_job_changed)
        return self._arm_queue_obj

    def _arms_async(self) -> bool:
        """Queue arm clicks when the scanner is the process proxy (its reads can block)."""
        return bool(self._force_async_arms) or is_process_proxy(self._current_bot())

    def pending_arm_kinds(self, symbol: str) -> dict[str, set[str]]:
        """Kinds still QUEUED for a symbol, per lane (watch / d1_event / any_bounce / level)."""
        lanes: dict[str, set[str]] = {
            "watch": set(), "d1_event": set(), "any_bounce": set(), "level": set(), "phone": set()
        }
        queue = self._arm_queue_obj
        symbol = str(symbol or "").strip().upper()
        if queue is None or not symbol:
            return lanes
        for job in queue.jobs():
            if job.pending and job.symbol == symbol and job.key[0] in lanes:
                lanes[job.key[0]].add(job.key[2])
        return lanes

    def _queue_arm(self, lane: str, symbol: str, kind: str, label: str, prepare, commit) -> None:
        """Mark QUEUED now; `prepare` runs on the arm worker, `commit` back here."""
        alert = self._current_review_alert
        context = (
            symbol,
            alert if alert is not None and alert.symbol == symbol else None,
            self._review_dwell_ms(symbol),
        )
        self._arm_queue().submit(
            (lane, symbol, kind),
            symbol,
            label,
            prepare,
            lambda result: self._commit_queued_arm(context, lambda: commit(result)),
        )
        self.statusChanged.emit(f"{symbol}: arming {label}… (queued)")

    def _commit_queued_arm(self, context: tuple, commit) -> tuple[bool, str]:
        """Run the unchanged arm body with the click's review context; its review
        rows are appended by the arm worker, in order."""
        messages: list[str] = []

        def grab(text: str) -> None:
            messages.append(str(text))

        self._arm_review_override = context
        self._deferred_review_events = []
        self.statusChanged.connect(grab)
        try:
            ok = bool(commit())
        finally:
            self.statusChanged.disconnect(grab)
            deferred, self._deferred_review_events = self._deferred_review_events, None
            self._arm_review_override = None
        path = self._review_events_path
        if deferred and path is not None:

            def write() -> None:
                for action, kwargs in deferred:
                    try:
                        record_review_event(action, path=path, **kwargs)
                    except Exception:  # evidence loses the event, never the arm
                        logging.debug("Queued review event %s failed.", action, exc_info=True)

            self._arm_queue().post(write, f"review events for {context[0]}")
        return ok, (messages[-1] if messages else "")

    def _arm_review_alert(self, symbol: str):
        """The alert a review row names: the click's own when a queued arm commits."""
        override = self._arm_review_override
        if override is not None and override[0] == symbol:
            return override[1]
        current = self._current_review_alert
        return current if current is not None and current.symbol == symbol else None

    def _arm_dwell_ms(self, symbol: str) -> int | None:
        override = self._arm_review_override
        if override is not None and override[0] == symbol:
            return override[2]
        return self._review_dwell_ms(symbol)

    def _cancel_pending_arm(self, key: tuple) -> bool:
        """A second click on a QUEUED arm cancels it before it saves."""
        queue = self._arm_queue_obj
        job = queue.pending(key) if queue is not None else None
        if job is None or not queue.cancel(key):
            return False
        self.statusChanged.emit(f"{job.symbol}: {job.label} arm cancelled - it had not saved yet.")
        return True

    def _cancel_or_dismiss_arm(self, key) -> None:
        if not self._cancel_pending_arm(tuple(key)) and self._arm_queue_obj is not None:
            self._arm_queue_obj.dismiss(key)

    def _on_arm_job_changed(self, job) -> None:
        from ui.services.arm_queue import FAILED

        self._refresh_review_armed_kinds()
        self.armedWatchesChanged.emit()  # the armed list and any open snapshot chart
        if job.state == FAILED:
            self.statusChanged.emit(
                f"{job.symbol}: {job.label} FAILED - {job.reason or 'not armed'}"
            )

    def _pending_arm_rows(self) -> list[tuple]:
        queue = self._arm_queue_obj
        if queue is None:
            return []
        return [
            (job.symbol, job.label, job.state, job.reason, job.key)
            for job in queue.jobs()
        ]

    def shutdown(self) -> None:
        """Drain queued arms (bounded); anything not armed is logged loudly."""
        executor = self._show_hidden_executor
        if executor is not None:
            self.flush_show_hidden_writes(timeout=2.0)
            executor.shutdown(wait=False)
        queue = self._arm_queue_obj
        if queue is None:
            return
        dropped = queue.shutdown()
        if dropped:
            self.statusChanged.emit(
                "NOT ARMED at shutdown: "
                + ", ".join(f"{job.symbol} {job.label}" for job in dropped)
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
        payload = {
            "chart_watch_kind": kind,
            "armed_at": watch.armed_at.isoformat(),
            "source_text": getattr(watch, "source_text", "")
            or getattr(watch, "candle_date", ""),
        }
        watch_id = str(getattr(watch, "watch_id", "") or "")
        if watch_id:
            payload["watch_id"] = watch_id
        # Everything the evaluation measured, for the kinds that measure more
        # than their own message says (the H1 retester's two bar times and its
        # full reason list). Never allowed to overwrite the keys above.
        for key, value in dict(getattr(hit, "details", None) or {}).items():
            payload.setdefault(str(key), value)
        return BounceAlert(
            time_text=moment.strftime("%H:%M:%S"),
            symbol=watch.symbol,
            side=side,
            trigger=trigger,
            timeframe=(
                "D1"
                if (
                    kind in D1_LEVEL_KINDS
                    or kind in D1_EVENT_KINDS
                    # A persistent watch is a multi-day arm and belongs beside
                    # the other armed events on the D1 feed, never on the
                    # session M5 list (which is today's intraday tape).
                    or kind in PERSISTENT_WATCH_KINDS
                )
                else "M5"
            ),
            tag=CHART_WATCH_TAG,
            raw_text=f"CHART WATCH {watch.symbol} ({side}): {trigger}",
            payload=payload,
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
        except Exception as exc:
            note_swallowed("bounce learning stats unreadable for tracker note", exc, quiet=True)
        window = ""
        if bucket:
            window = (
                "prime window"
                if bucket in BAND_BOUNCE_PRIME_BUCKETS
                else f"off-window ({bucket})"
            )
        parts = [part for part in (f"tracker {segment_type} {stats}" if stats else "", window) if part]
        return "; ".join(parts)

    def _save_chart_watches(self, watches=None) -> bool:
        if self._chart_watches_path is None:
            return True
        try:
            save_chart_watches(
                self._chart_watches if watches is None else watches,
                self._chart_watches_path,
            )
            return True
        except OSError:
            return False

    def _save_d1_level_watches(self) -> None:
        if self._d1_level_watches_path is None:
            return
        try:
            save_d1_level_watches(self._d1_level_watches, self._d1_level_watches_path)
        except OSError as exc:
            note_swallowed("D1 level watches write failed", exc)

    # ------------------------------------------------------------------
    # Persistent D1 candle-level alerts: armed by clicking a D1 chart candle,
    # kept across sessions until they flag (symbol need not be scanned).
    def _arm_d1_level_from_chart(
        self, symbol: str, direction: str, level: float, candle_date: str
    ) -> None:
        self.request_d1_level_watch(
            symbol, direction, level, candle_date=candle_date, fill_source="candle"
        )

    def request_d1_level_watch(
        self, symbol: str, direction: str, level: float, *, candle_date: str = "", fill_source: str = ""
    ) -> None:
        """A level-arm click: queued on a proxy, else `arm_d1_level_watch` at once."""
        symbol = str(symbol or "").strip().upper()
        try:
            level = float(level)
        except (TypeError, ValueError):
            return
        if not self._arms_async() or not symbol:
            self.arm_d1_level_watch(
                symbol, direction, level, candle_date=candle_date, fill_source=fill_source
            )
            return

        def armed() -> bool:
            return any(
                watch.symbol == symbol and watch.direction == direction
                and abs(watch.level - level) < 1e-6
                for watch in self._d1_level_watches
            )

        self._queue_arm(
            "level",
            symbol,
            f"{direction}:{level:.4f}",
            f"D1 level {direction} {level:.2f}",
            None,
            lambda _result: self.arm_d1_level_watch(
                symbol, direction, level, candle_date=candle_date, fill_source=fill_source
            )
            or armed(),
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
        self._record_review_event(
            "arm_level",
            alert=self._arm_review_alert(symbol),
            symbol=symbol,
            dwell_ms=self._arm_dwell_ms(symbol),
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

    def chart_symbol(self, symbol: str, *, side: str = "", origin: str = "", next_pick=None) -> bool:
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

        `next_pick` (trader, 2026-09-15) is a caller's own "what comes after
        this chart": a callable returning True when it charted something. It
        is consulted INSTEAD of the waiting list when this chart is vetoed,
        claimed or stepped past, and it is dropped the moment any other chart
        takes the pane. The setups table passes one so a veto or a claim on a
        row it charted cycles to its next row; the lookup box passes none.
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
        self._manual_next_pick = next_pick if callable(next_pick) else None
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
        except Exception as exc:
            note_swallowed("arm bar fill source unreadable", exc, quiet=True)
        self.request_d1_level_watch(symbol, direction, level, fill_source=fill_source)

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
        self._announce_price_alert(
            symbol, direction, level, self._write_price_alert(service, symbol, direction, level)
        )

    def _request_price_alert_from_level(self, symbol: str, direction: str, level: float) -> None:
        """The phone-alert click: the store merge/save runs on the arm worker on a proxy."""
        symbol = str(symbol or "").strip().upper()
        try:
            level = float(level)
        except (TypeError, ValueError):
            return
        service = getattr(self, "price_alert_service", None)
        if (
            not self._arms_async()
            or service is None
            or not symbol
            or direction not in ("above", "below")
            or not level > 0
        ):
            self._arm_price_alert_from_level(symbol, direction, level)
            return
        self._queue_arm(
            "phone",
            symbol,
            direction,
            f"phone alert {direction} {level:.2f}",
            lambda: self._write_price_alert(service, symbol, direction, level),
            lambda outcome: self._announce_price_alert(symbol, direction, level, outcome),
        )

    @staticmethod
    def _write_price_alert(service, symbol: str, direction: str, level: float) -> str:
        """The Focus board's merge, key for key, then one save. Any thread.

        Returns "refused", "armed" or "kept" (an unchanged level that already fired).
        """
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
            return "refused"
        return "armed" if entry.get(f"armed_{direction}") else "kept"

    def _announce_price_alert(self, symbol: str, direction: str, level: float, outcome: str) -> bool:
        if outcome == "refused":
            self.statusChanged.emit(
                f"{symbol}: phone price alert NOT saved - the price-alert "
                "store refused the write on this machine."
            )
            return False
        if outcome == "armed":
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
        return True

    def armed_levels_for(self, symbol: str) -> list:
        symbol = str(symbol or "").strip().upper()
        return [watch for watch in self._d1_level_watches if watch.symbol == symbol]

    def _build_journal_tab(self):
        """ONE BOX, ONE ENTER (V2 item 4, decision 0016 answer 11).

        The trader's own description of what this should be: *"one box, one
        Enter, one or two thesis entries a day."* It had a timeframe picker, a
        box, a Save button and a status line - four decisions for a thought you
        have at 10:40 and would otherwise lose.

        Everything except the box is gone from the SURFACE. **Nothing is gone
        from the SCHEMA**: `timeframe` is still written, at the M5 default, and
        every other field is written empty rather than dropped, because a field
        that exists at v1 keeps its name and meaning forever - rows already on
        disk carry it, and the nightly `market_journal` scope reads it.

        The sit-down review still lives on the left-nav Market Journal page.
        """
        from PySide6.QtWidgets import QPlainTextEdit

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

        # The timeframe the entry is written with. NOT a picker any more - the
        # schema field stays and the surface loses the decision. M5 because that
        # is what the trader is looking at when the thought arrives, and it was
        # the default of the picker this replaces.
        self._journal_timeframe_value = market_journal.TIMEFRAME_M5

        self._journal_text = QPlainTextEdit()
        self._journal_text.setPlaceholderText(
            "What you make of the tape. Enter saves; Shift+Enter for a new line."
        )
        self._journal_text.installEventFilter(self)
        self._journal_status = QLabel("")
        self._journal_status.setWordWrap(True)

        layout.addWidget(self._journal_text, 1)
        layout.addWidget(self._journal_status)

        self.market_journal_service.statusChanged.connect(self._journal_status.setText)
        # Ctrl+Enter stays bound as well as plain Enter. The trader has been
        # typing it since R10.H, and removing a shortcut that already works is a
        # cost with no benefit.
        shortcut = QShortcut(QKeySequence("Ctrl+Return"), container)
        shortcut.activated.connect(self._commit_journal_entry)
        self._journal_shortcut = shortcut
        return container

    def eventFilter(self, watched, event):  # noqa: N802 - Qt's own spelling
        """Plain Enter saves the market-journal note; Shift+Enter is a newline.

        An event filter rather than a `QShortcut`: a shortcut on Return would
        fire for every widget in this panel's scope, and this key must mean
        "save" only while the cursor is in this one box.
        """
        try:
            from PySide6.QtCore import QEvent, Qt as _Qt

            if (
                watched is getattr(self, "_journal_text", None)
                and event.type() == QEvent.Type.KeyPress
                and event.key() in (_Qt.Key.Key_Return, _Qt.Key.Key_Enter)
                and not (event.modifiers() & _Qt.KeyboardModifier.ShiftModifier)
            ):
                self._commit_journal_entry()
                return True
        except Exception as exc:  # noqa: BLE001 - a key handler never breaks the panel
            note_swallowed("alert center journal key handler raised", exc)
        return super().eventFilter(watched, event)

    def _commit_journal_entry(self) -> None:
        import market_journal

        # The chart in front of the trader, when there is one. A stale symbol
        # would be worse than none: it would assert a link they never made.
        current = getattr(self, "_current_review_alert", None)
        symbol = str(getattr(current, "symbol", "") or "").strip().upper()

        result = self.market_journal_service.write_entry(
            text=self._journal_text.toPlainText(),
            # THE SESSION IT IS ABOUT, not the date it was typed. Today's until
            # the close, the last session after it - a thought written at 18:00
            # is about the day that just ended, and dating it tomorrow would file
            # it against a session that has not happened. `written_after_the_
            # session` is still COMPUTED by the store from `created_at`, so the
            # distinction is recorded rather than erased.
            session_date=market_journal.session_date_for(),
            timeframe=self._journal_timeframe_value,
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
        # A DECLINED row is remembered, not armed: the trader turned it off and
        # the auto-arm sweep is keeping the memory so it does not come back
        # (PCT-1 item 5). It has no place on a board headed "what am I waiting
        # on", and it is never evaluated or pushed either.
        self.armed_list.set_watches(
            [
                watch
                for watch in self._chart_watches
                if not bool(getattr(watch, "declined", False))
            ],
            self._d1_level_watches,
            d1_events=self._d1_event_watches,
            has_m5_bars=lambda symbol: bool(self._m5_bars_for(symbol)),
            watch_note=self._armed_watch_note,
            pending=self._pending_arm_rows(),
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
        except Exception as exc:
            note_swallowed("D1 chart prefetch failed", exc, quiet=True)

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
            if self._m5_unknown(watch.symbol):
                remaining.append(watch)  # bars not fetched yet: unknown, never judged
                continue
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
            except Exception as exc:
                note_swallowed("D1 event watches write failed", exc)

    def _toggle_d1_event_watch(self, alert: BounceAlert, kind: str) -> None:
        if alert is None or not alert.symbol:
            return
        self.toggle_d1_event_watch(alert.symbol, kind, side=alert.side)

    def toggle_d1_event_watch(self, symbol: str, kind: str, side: str = "") -> None:
        """A D1 event button click: cancel a queued arm, disarm, or arm (queued on a proxy)."""
        symbol = str(symbol or "").strip().upper()
        if not symbol or self._cancel_pending_arm(("d1_event", symbol, kind)):
            return
        if kind in self.armed_d1_event_kinds(symbol):
            self.disarm_d1_event_watch(symbol, kind)
            return
        if not self._arms_async() or kind not in D1_EVENT_KINDS:
            self.arm_d1_event_watch(symbol, kind, side=side)
            return
        reads_line = kind in {"trendline_break", "trendline_break_retest"}

        def prepare():
            return self._current_trendline_report_evidence(symbol, side)

        self._queue_arm(
            "d1_event",
            symbol,
            kind,
            D1_EVENT_KINDS[kind],
            prepare if reads_line else None,
            lambda evidence: self.arm_d1_event_watch(
                symbol,
                kind,
                side=side,
                trendline_evidence=evidence if reads_line else _UNREAD,
            )
            or kind in self.armed_d1_event_kinds(symbol),
        )

    def _current_trendline_candidate(self, symbol: str, side: str = "") -> dict | None:
        evidence = self._current_trendline_report_evidence(symbol, side)
        return evidence[0] if evidence is not None else None

    def _current_trendline_report_evidence(
        self, symbol: str, side: str = ""
    ) -> tuple[dict, datetime] | None:
        """The compact saved scan report is the arm-time source of the line.

        This is a small report read on the explicit arm click, not the 38 MB
        ai-state file and never part of the timer poll. A read failure is
        uncertainty: the button refuses rather than re-deriving a line. Its
        ``generated_at`` is the only honest time at which the scan knew the
        frozen geometry; an arm-click timestamp must never replace it.
        """
        symbol = str(symbol or "").strip().upper()
        requested_side = str(side or "").strip().upper()
        try:
            payload = json.loads(
                Path(MASTER_AVWAP_D1_UPGRADE_ALERTS_FILE).read_text(encoding="utf-8")
            )
        except (OSError, TypeError, ValueError):
            return None
        try:
            knowledge_at = datetime.fromisoformat(str(payload.get("generated_at") or ""))
        except (AttributeError, TypeError, ValueError):
            return None
        if knowledge_at.tzinfo is None or knowledge_at.utcoffset() is None:
            return None
        events = payload.get("alerts") if isinstance(payload, dict) else []
        for event in events or []:
            if not isinstance(event, dict):
                continue
            if str(event.get("event_type") or "").strip().lower() != "trendline_break":
                continue
            if str(event.get("symbol") or "").strip().upper() != symbol:
                continue
            event_side = str(event.get("side") or "").strip().upper()
            if requested_side and event_side and event_side != requested_side:
                continue
            candidate = event.get("trendline_candidate")
            if isinstance(candidate, dict):
                frozen = dict(candidate)
                frozen.setdefault("side", event_side)
                return frozen, knowledge_at
        return None

    def arm_d1_event_watch(
        self, symbol: str, kind: str, side: str = "", *, trendline_evidence=_UNREAD
    ) -> bool:
        symbol = str(symbol or "").strip().upper()
        if not symbol or kind not in D1_EVENT_KINDS:
            return False
        label = D1_EVENT_KINDS[kind]
        if kind in self.armed_d1_event_kinds(symbol):
            self.statusChanged.emit(f"{symbol}: {label} alert already armed.")
            return False
        moment = datetime.now()
        if kind in {"trendline_break", "trendline_break_retest"}:
            evidence = (
                self._current_trendline_report_evidence(symbol, side)
                if trendline_evidence is _UNREAD
                else trendline_evidence
            )
            candidate, knowledge_at = evidence if evidence is not None else (None, None)
            resolved_side = str(side or "").strip().upper()
            if not resolved_side and isinstance(candidate, dict):
                resolved_side = str(candidate.get("side") or "").strip().upper()
            if not isinstance(candidate, dict) or resolved_side not in {"LONG", "SHORT"}:
                self.statusChanged.emit(
                    f"{symbol}: {label} needs the saved scan line before it can arm."
                )
                return False
            self._d1_event_watches.append(
                D1EventWatch(
                    symbol=symbol,
                    kind=kind,
                    armed_at=moment,
                    side=resolved_side,
                    trendline_candidate=candidate,
                    trendline_knowledge_at=knowledge_at,
                )
            )
        elif kind == "sma_break_retest":
            resolved_side = str(side or "").strip().upper()
            if resolved_side not in {"LONG", "SHORT"}:
                self.statusChanged.emit(f"{symbol}: {label} needs a long or short chart to arm.")
                return False
            self._d1_event_watches.append(
                D1EventWatch(symbol=symbol, kind=kind, armed_at=moment, side=resolved_side)
            )
        else:
            self._d1_event_watches.append(
                D1EventWatch(symbol=symbol, kind=kind, armed_at=moment)
            )
        self._save_d1_event_watches()
        self._refresh_review_armed_kinds()
        self.armedWatchesChanged.emit()
        self._record_review_event(
            "arm_d1_event",
            alert=self._arm_review_alert(symbol),
            symbol=symbol,
            dwell_ms=self._arm_dwell_ms(symbol),
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
        self._decline_wall_d1_event(symbol, kind)
        self._refresh_review_armed_kinds()
        self.armedWatchesChanged.emit()
        self._record_review_event(
            "disarm_d1_event", symbol=symbol, detail={"kind": kind}
        )
        self.statusChanged.emit(
            f"{symbol}: {D1_EVENT_KINDS.get(kind, kind)} alert disarmed."
        )
        return True

    def _push_armed_watch(self, hit) -> None:
        """One phone event per armed-watch fire, through the ONE armed sender.

        AWAY is the only mode that pushes routine output; the armed
        Research/Focus price alerts are the recorded exception that pushes in
        every mode, and a watch the trader armed from the chart is the same
        request made from a different surface - so it rides that sender rather
        than opening a second door. De-duplication is by watch id, inside the
        service. Delivery is best-effort: an unconfigured topic or a dead
        network must never cost the alert behind it.

        This call DISPATCHES and returns at once (RV-H1-PHONE-WORKER,
        2026-09-13): `notify_armed_watch` makes the cheap decisions - the
        engine check and the watch-id de-duplication - on this thread and
        hands the send to a worker the service owns, so the `add_alert` right
        after it draws the row while the phone is still answering. ``ok``
        there means "accepted for delivery", and the outcome comes back on
        the service's `statusChanged`, never here.
        """
        service = getattr(self, "price_alert_service", None)
        if service is None:
            return
        watch = hit.watch
        label = WATCH_KINDS.get(watch.kind, watch.kind)
        measured = dict(getattr(hit, "details", None) or {})
        if "push" in measured and not measured.get("push"):
            # A retest on a watch the DESK armed: on the feed and in the
            # evidence, but not on the phone (review advisory 5). Nothing is
            # withheld - the row is written either way.
            return
        watch_id = str(getattr(watch, "watch_id", "") or "")
        # A STANDING arm speaks more than once, so the de-duplication key is
        # this EVENT, not the arm (review blocker 1). A one-shot watch passes
        # nothing and keeps the old watch-id behaviour.
        event_key = None
        trigger = str(measured.get("trigger") or "")
        if trigger:
            event_key = "%s:%s:%s:%s" % (
                watch_id,
                trigger,
                str(measured.get("timeframe") or ""),
                str(measured.get("bar_dt") or ""),
            )
        try:
            service.notify_armed_watch(
                watch_id=watch_id,
                title=f"{label}: {watch.symbol}",
                message=str(hit.message or ""),
                **({"event_key": event_key} if event_key else {}),
            )
        except TypeError:
            # A stand-in service written before PCT-1 takes three keywords.
            service.notify_armed_watch(
                watch_id=watch_id,
                title=f"{label}: {watch.symbol}",
                message=str(hit.message or ""),
            )
        except Exception:
            logging.debug(
                "Armed-watch push failed for %s", watch.symbol, exc_info=True
            )

    def _poll_d1_event_watches(self, now: datetime | None = None) -> None:
        # The Pullback alerts live in the chart-watch store but keep this
        # poll's clock, so they are evaluated BEFORE the early return below -
        # which asks only whether any D1 EVENT watch is armed. Their auto-arm
        # sweep rides the same call, so a pick claimed a minute ago is armed
        # even when nothing else on this desk is.
        self._poll_pullback_watches(now=now)
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
        fired_trendline_breaks: set[tuple[str, str, str]] = set()
        # One reference-level build per symbol per tick, shared across every
        # watch on it (item 1b). Scoped to this tick and discarded with it.
        levels_caches: dict[str, dict] = {}
        for watch in self._d1_event_watches:
            hit = None
            if self._m5_unknown(watch.symbol):
                remaining.append(watch)  # bars not fetched yet: unknown, never judged
                continue
            m5_bars = self._m5_bars_for(watch.symbol)
            d1_bars = self._d1_bars_for(watch.symbol)
            if d1_bars:
                # Unlike a frozen price level, every event kind needs the
                # daily store for its reference; without it there is nothing
                # to measure against yet and the watch just waits.
                avwape_anchor = None
                if d1_kind_needs_avwape(watch.kind):
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
                if watch.kind == "trendline_break":
                    candidate = watch.trendline_candidate or {}
                    key = (
                        watch.symbol,
                        str(watch.side or "").strip().upper(),
                        str(candidate.get("break_date") or "").strip(),
                    )
                    if key in fired_trendline_breaks:
                        continue
                    fired_trendline_breaks.add(key)
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
                detail=d1_event_fired_detail(hit),
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
            pending = self.pending_arm_kinds(current.symbol)
            self.chart_review.set_pending_arms(
                pending["watch"], pending["d1_event"], bool(pending["any_bounce"])
            )
            self.chart_review.set_armed_kinds(self.armed_watch_kinds(current.symbol))
            self.chart_review.set_armed_d1_events(
                self.armed_d1_event_kinds(current.symbol)
            )
            self.chart_review.set_any_bounce_armed(
                self.any_bounce_armed_for(current.symbol)
            )

    def _remove_review_alert_for_today(self, alert: BounceAlert) -> None:
        """The "✕ Not today" BUTTON: retire the name, and offer the note box.

        This verb has no reason picklist, so the uncoded row plus the optional
        box IS its why (P10 A1/A2, and the trader kept it in so many words on
        2026-09-04: *"not today can continue to go to the next chart with a pop
        up note box"*). A veto from the capture rail takes `_retire_after_veto`
        instead - same retirement, no second row, no box.
        """
        self._retire_review_alert(alert, write_not_today_annotation=True)

    def _retire_after_veto(self, alert: BounceAlert) -> None:
        """A CODED veto from the capture rail: retire, write nothing more.

        Trader, 2026-09-04: *"when i double tap something in the capture window
        (either veto or like+claim) i shouldnt get a pop up note box. the point
        of the capture window is to quickly enter 'WHY' I like or dislike
        something."* The rail has already written one veto row carrying the
        reason code and the trader's own words; a second, UNCODED row and a box
        asking for the why again are both noise.

        Everything else the "Not today" verb does still happens, through the
        SAME body: the auto-pick / faded / Focus-review branches, the
        auto-adopted Focus drop, the `remove_today` review event (the name is
        historical and `review_learning.REJECT_ACTIONS` keys on the string),
        parking the symbol for the day, and the advance that
        `_ignore_alert_symbol` performs.

        Also the retire half of the day-trade veto (lead ruling 2026-09-04):
        that verb ends here too, after its Focus placement.
        """
        self._retire_review_alert(alert, write_not_today_annotation=False)
        self.reviewDecisionRecorded.emit()

    #: A saved veto reason -> the follow-up alert it arms (trader, 2026-09-24).
    #: "pullback" is the chart-watch Pullback alert; the rest are D1 event kinds.
    #: Any reason not listed arms nothing. compressed -> range_breakout is the
    #: lead's choice; the trader can overrule it.
    VETO_FOLLOW_UPS = {
        "too_extended_from_base": PULLBACK_KIND,
        "incoming_trendline": "trendline_break_retest",
        "sma_incoming": "sma_break_retest",
        "compressed": "range_breakout",
    }

    def _arm_saved_extended_veto(self, alert: BounceAlert, row: dict) -> None:
        """Arm the follow-up alert a saved veto reason asks for.

        The widget forwards only a saved, identity-matching row; repeat the
        match at the store-owning host so a delayed capture can neither arm nor
        retire the chart now in front.  This runs before ``vetoRetireRequested``
        in the widget, so the normal retirement can advance immediately after
        the durable arm attempt, armed or not.
        """
        reason_code = str(row.get("reason_code") or "")
        symbol = str(row.get("symbol") or "").strip().upper()
        side = str(row.get("side") or "").strip().upper()
        current = self._current_review_alert
        follow_up = self.VETO_FOLLOW_UPS.get(reason_code)
        if (
            follow_up is None
            or current is None
            or current is not alert
            or symbol != str(current.symbol or "").strip().upper()
            or side != str(current.side or "").strip().upper()
            or side not in ("LONG", "SHORT")
        ):
            return
        if follow_up == PULLBACK_KIND:
            self._arm_veto_pullback(symbol, side)
        else:
            self._arm_veto_d1_event(reason_code, follow_up, symbol, side)

    def _arm_veto_d1_event(self, reason_code: str, kind: str, symbol: str, side: str) -> None:
        """Arm one D1 event follow-up for a saved veto; say why when it cannot."""
        label = D1_EVENT_KINDS.get(kind, kind)

        def not_armed(why: str) -> None:
            self.chart_review.capture_rail.set_capture_status(
                f"VETO {symbol} - {reason_code}; {label} not armed ({why})", ok=False
            )

        existing = next(
            (w for w in self._d1_event_watches if w.symbol == symbol and w.kind == kind),
            None,
        )
        if existing is not None:
            existing_side = str(existing.side or "").strip().upper()
            if kind in D1_SIDED_KINDS and existing_side and existing_side != side:
                not_armed(f"{existing_side} one already armed")
            return  # already armed for this name: never reset or duplicate it
        evidence = _UNREAD
        if kind == "trendline_break_retest":
            evidence, why = self._incoming_trendline_evidence(symbol, side)
            if evidence is None:
                not_armed(why)
                return
        armed = self.arm_d1_event_watch(
            symbol,
            kind,
            side=side if kind in D1_SIDED_KINDS else "",
            trendline_evidence=evidence,
        )
        if not armed:
            not_armed("arm failed")

    def _wall_trendline_knowledge_at(self) -> datetime | None:
        """When the scan's trendline records were written (aware), or None."""
        try:
            stamp = Path(MASTER_AVWAP_AI_STATE_FILE).stat().st_mtime
        except OSError:
            return None
        return datetime.fromtimestamp(stamp, tz=timezone.utc)

    def _incoming_trendline_evidence(self, symbol: str, side: str):
        """((candidate, knowledge_at), "") for the nearest in-path incoming line, else (None, why).

        Same records as the wall gate (`_wall_trendlines_for`), same projection
        (`wall_gate.trendline_value`): the not-yet-broken line of the side's
        type (H- above a long, L+ below a short) nearest to the last close.
        """
        wanted = incoming_trendline_type(side)
        moment = datetime.now()
        completed = wall_gate.completed_daily_bars(self._d1_bars_for(symbol), today=moment.date())
        if not completed:
            return None, "no daily bars"
        try:
            price = float(completed[-1]["close"])
        except (KeyError, TypeError, ValueError):
            return None, "no price"
        best = None
        for record in self._wall_trendlines_for(symbol):
            if not isinstance(record, dict) or str(record.get("type") or "") != wanted:
                continue
            if record.get("break_date"):
                continue
            value = wall_gate.trendline_value(record, completed, today=moment.date())
            if value is None:
                continue
            if (side == "LONG" and value < price) or (side == "SHORT" and value > price):
                continue
            distance = abs(value - price)
            if best is None or distance < best[0]:
                best = (distance, record)
        if best is None:
            return None, "no trendline"
        knowledge_at = self._wall_trendline_knowledge_at()
        if knowledge_at is None:
            return None, "no scan time for the trendline"
        return (dict(best[1], side=side), knowledge_at), ""

    def _arm_veto_pullback(self, symbol: str, side: str) -> None:
        """too_extended_from_base: the narrow M30/H1 Pullback alert (unchanged)."""
        source = "veto: too_extended_from_base"
        active_existing = next(
            (
                watch
                for watch in self._chart_watches
                if watch.symbol == symbol
                and watch.kind == PULLBACK_KIND
                and not bool(getattr(watch, "declined", False))
            ),
            None,
        )
        if active_existing is not None:
            existing_side = str(getattr(active_existing, "side", "") or "").upper()
            if existing_side == side:
                scope = {
                    str(timeframe).upper()
                    for timeframe in (getattr(active_existing, "timeframes", ()) or ())
                }
                triggers = set(getattr(active_existing, "triggers", ()) or ())
                if (
                    (not scope or {"M30", "H1"}.issubset(scope))
                    and set(PULLBACK_TRIGGERS).issubset(triggers)
                ):
                    return  # this arm already covers the requested follow-up
                self.chart_review.capture_rail.set_capture_status(
                    f"VETO {symbol} - too_extended_from_base; pullback not armed "
                    "(existing pullback does not cover the M30/H1 follow-up)",
                    ok=False,
                )
                return
            self.chart_review.capture_rail.set_capture_status(
                f"VETO {symbol} - too_extended_from_base; pullback not armed "
                f"({existing_side or 'WATCH'} pullback already armed)",
                ok=False,
            )
            return
        armed = self.arm_chart_watch_for(
            symbol,
            side,
            PULLBACK_KIND,
            source_text=source,
            timeframes=("M30", "H1"),
        )
        if armed:
            return
        # CaptureRail gives a saved-veto listener's status precedence for this
        # one commit, after its own cohort write completes.  The saved veto
        # still retires normally.
        self.chart_review.capture_rail.set_capture_status(
            f"VETO {symbol} - too_extended_from_base; pullback not armed", ok=False
        )

    def _retire_review_alert(
        self, alert: BounceAlert, *, write_not_today_annotation: bool
    ) -> None:
        """Drop a name from today's visual processing without changing scans.

        ONE body for both retirement verbs. The flag is the ONLY difference:
        whether the uncoded "Not today" annotation (and the note box behind it)
        is written. Two copies of the branch ladder below would drift at the
        auto-pick / faded / Focus-review branches first, and those three each
        return early - a copy that lost one of them would silently start
        parking symbols that must not be parked.
        """
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
        # P10 A1/A2. "Not today" has always written a `pick_feedback` verdict
        # with the hardcoded free-text reason "not today" - never a code, never
        # a word of the trader's own. It ALSO writes one uncoded veto
        # annotation and then offers a note box, which is the surface the trader
        # named: *"not-for-today in visual chart review I SHOULD get a little
        # pop-up that lets me write a note if I am not using the quick buttons."*
        #
        # Packet T1 (2026-09-04) made that the BUTTON's half alone. A veto from
        # the capture rail arrives here with `write_not_today_annotation=False`,
        # because its coded row and its why are already on disk and the trader
        # asked for no box on that path.
        #
        # The `pick_feedback` row and the Focus removal above are untouched: P5
        # grades them as `focus__m5_not_today` and several surfaces read them.
        if write_not_today_annotation:
            self._record_not_today_annotation(alert)
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

    def _record_not_today_annotation(self, alert: BounceAlert) -> None:
        """One uncoded veto row for a "Not today", then the optional note.

        The ROW GOES FIRST. Escape on the box leaves the click counted, which is
        the trader's own rule - the fact they clicked is the evidence, and the
        sentence is a bonus.

        No scanner row is attached: this click happens on a chart, and the alert
        does not carry the tracker row behind it. **A capture click never
        fetches** (the pass rule), so the fields are absent rather than guessed.

        Every failure is swallowed. The Focus removal and the review event have
        already happened, and an evidence store never costs the event it records.
        """
        try:
            from ui.annotations import verdicts

            side = str(getattr(alert, "side", "") or "").strip().upper()
            written = verdicts.record_not_today(
                symbol=alert.symbol,
                side="SHORT" if side.startswith("SHORT") else "LONG",
                timeframe=str(getattr(alert, "timeframe", "") or "M5"),
            )
        except Exception:
            return
        if written is None:
            return
        # LAST, and asynchronous - exactly as the Master AVWAP prompt is, and
        # for the same three reasons: a BLOCKING modal opened inside the handler
        # never returns in a headless test, the queue advance must finish first,
        # and A2 asks that the box not block the 60 s poll. (R4 A6: the word
        # was "DEFERRED", which claimed a later turn of the event loop this call
        # never takes.)
        self._prompt_for_not_today_note(written)

    def _prompt_for_not_today_note(self, written: dict) -> None:
        """The note box. Optional, WINDOW-MODAL, and it never blocks the queue.

        R4 A6 corrected "MODELESS" here: `QDialog.open()` shows the dialog
        window-modal. The property that matters is that it does not BLOCK.
        `QInputDialog.getMultiLineText` runs a nested event loop and does not
        return until the trader answers. That would sit between the "Not today"
        click and the review queue advancing - and in a headless test it never
        returns at all, so every existing test that clicks this button would HANG
        rather than fail.

        Enter saves and Shift+Enter makes a newline (R4 A6), through the one
        helper both note boxes on the desk now use.
        """
        try:
            from ui.widgets.note_prompt import open_note_prompt

            # Held until it closes: a dialog with no reference is garbage the
            # moment this method returns.
            self._not_today_note_dialog = open_note_prompt(
                self,
                title=f"Not today: {written.get('symbol', '')}",
                label=(
                    "Add a note if you want one - it is optional, and the click "
                    "is already saved either way. Enter saves; Shift+Enter "
                    "starts a new line."
                ),
                on_text=lambda text, row=written: self._save_not_today_note(row, text),
            )
        except Exception:
            return

    def _save_not_today_note(self, written: dict, note: str) -> None:
        """The note row. Swallowed on failure: the veto row is already on disk."""
        try:
            from ui.annotations import verdicts

            verdicts.record_note_on(written, note)
        except Exception:
            return

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
        # Ignoring is a day decision for the whole name.  The held ordinary
        # scan cache is presentation-only and must not resurrect that decision
        # if the trader flips Show all later in the day.
        self._held_d1_scan_reviews.pop(symbol, None)
        self._ignored_symbols.add(symbol)
        if self._ignored_symbols_path is not None:
            try:
                self._ignored_symbols = save_ignored_alert_symbols(
                    self._ignored_symbols,
                    self._ignored_symbols_path,
                    market_date=self._ignored_market_date,
                )
            except OSError as exc:
                note_swallowed("ignored alert symbols write failed", exc)
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
        # SN4: a veto removes THAT row. Every other row keeps its widget, its
        # position and its ×N badge - measured 4.0-4.1 s per veto on
        # 2026-09-08 when this was a 350-widget rebuild.
        self._sync_feed()
        self._refresh_d1_scan_review_view()
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
            except OSError as exc:
                note_swallowed("ignored alert symbols write failed on restore", exc)
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
        self._focus_break_emitted = None
        self._focus_gate_held = 0
        # The movers-only filter is day-scoped for the same reason: a reveal is
        # "show me the chop for the rest of today", not a preference change.
        self._review_movers_only = True
        self._hidden_inside_range.clear()
        self.chart_review.set_hidden_count(0)
        # AR-2B's held scan rows are a day-local display cache.  Yesterday's
        # candidates must never make today's Show all count look nonzero.
        self._held_d1_scan_reviews.clear()
        self._show_all_d1_scan_reviews = False
        self._refresh_d1_scan_review_view()
        self._refresh_ignored_button()
        # The M5 alert bar is day-scoped like the queue it replaced.
        self.m5AlertsDayRolled.emit()
        # A3: the fade clock is measured in sessions, so the day roll is
        # exactly when a pick can come due.
        self.run_focus_fade_check()
        # Packet D1C-A: a claim's fade is the SAME clock, so it comes due here
        # and only here. The skip counts are day-scoped like the queue - a
        # count of yesterday's skipped charts says nothing about today.
        self._claimed_d1_skipped.clear()
        self.chart_review.set_claimed_skipped_count(0)
        self._sweep_expired_claims()
        self.claimsChanged.emit()

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
            except Exception as exc:
                note_swallowed("parked review symbols write failed", exc)

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
            except Exception as exc:
                note_swallowed("parked review symbols write failed on unpark", exc)

    def _has_armed_d1_alerts(self, symbol: str) -> bool:
        symbol = str(symbol or "").strip().upper()
        return any(watch.symbol == symbol for watch in self._d1_level_watches) or any(
            watch.symbol == symbol for watch in self._d1_event_watches
        )

    def apply_scaled_metrics(self) -> None:
        """Re-apply the column's scale-dependent floors after a scale change."""
        self.tabs.setMinimumWidth(theme.px(170))
        self.focus_strength.apply_scaled_metrics()
        self.strength_page.apply_scaled_metrics()
        self.movers_board.apply_scaled_metrics()
        self.chart_review.arm_bar.apply_scaled_metrics()

    def set_embedded_detail_enabled(self, enabled: bool) -> None:
        """Workspace mode turns the embedded plan pane off so the setup is
        described in one place (the setups workspace's detail pane)."""
        self._embedded_detail_enabled = bool(enabled)
        if not self._embedded_detail_enabled:
            self.detail_view.setVisible(False)

    def _show_symbol_snapshot(self, alert: BounceAlert) -> None:
        """Ticker-name click on a feed row: the same as clicking the row.

        It used to open the snapshot popup. Trader, 2026-09-03: on the Trading
        Desk every ticker click lands on the centre chart, so the name click
        charts the alert itself in the review pane - the real alert, with its
        trigger, not a manual chart of the same name.
        """
        if not alert.symbol:
            return
        self._show_alert_detail(alert)

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

    def _chart_board_symbol(self, symbol: str, side: str, origin: str) -> None:
        """A click on any board inside this panel charts in the review pane.

        Trader, 2026-09-03: *"i click things in the auto RS/RW board ... and it
        does a pop up ... the main tab should always be centralized with the
        main chart."* The RS/RW, entry and Focus-strength boards live in the
        same column as the pane, so the rule the M5 Strength Board got on
        2026-08-31 now covers all of them. Same door (`chart_symbol`), same
        reasons for not using `_enqueue_review_alert`. The popup stays as
        `show_board_symbol`, the door for a board on ANOTHER page.
        """
        self.chart_symbol(symbol, side=side, origin=origin)

    #: The tabs the trader never opens (decision 0016 answer 7). HIDDEN, never
    #: removed: the Alerts feed is the review-alert door, the D1 Focus tab holds
    #: the flag list several polls write into, and the Armed tab is the
    #: inventory across every symbol. All three are load-bearing behind the
    #: scenes; what they are not is worth a tab the trader has to skip past.
    UNUSED_TAB_TITLES = ("Alerts", "D1 Focus", "Armed")

    def apply_unused_tab_visibility(self, show: bool) -> None:
        """Show or hide the three tabs the trader does not open.

        **HIDING IS NOT REMOVING**, and the difference is the whole design. The
        widgets are built, parented and connected exactly as before; every timer
        behind them stays visibility-gated as the snappiness work left it; every
        signal still arrives. A hidden tab costs one row of tab strip and
        nothing else.

        It is a `setTabVisible` on the existing index rather than a `removeTab`,
        so no index shifts and nothing that remembers one - `_d1_tab_index`,
        `_armed_tab_index`, `_capture_tab_index` - has to be recomputed.

        **Every shortcut those tabs hosted must still fire**, which is why the
        capture rail rebinds `action_shortcuts` at PANEL scope: a `QShortcut`
        owned by a widget inside a hidden tab never fires, and two bindings for
        one sequence fire NEITHER.
        """
        wanted = bool(show)
        for index in range(self.tabs.count()):
            if self.tabs.tabText(index) in self.UNUSED_TAB_TITLES:
                self.tabs.setTabVisible(index, wanted)
        if not wanted and self.tabs.tabText(self.tabs.currentIndex()) in self.UNUSED_TAB_TITLES:
            # Never leave the trader looking at a tab that just vanished.
            self.tabs.setCurrentIndex(self._capture_tab_index)

    def _set_deep_read(self, on: bool) -> None:
        """Show or hide the old Strength page under the Movers board."""
        self.strength_page.setVisible(bool(on))
        layout = self.movers_column.layout()
        layout.setStretchFactor(self.movers_board, 0 if on else 1)

    def _add_movers_row_to_focus(self, symbol: str, side: str) -> None:
        """The trader's explicit +F click on a Movers row: M5 Focus through the one
        adoption gate, then `FocusService.add` - the same manual path the Strength
        Board's Add uses (the store injects into longs.txt/shorts.txt)."""
        symbol = str(symbol or "").strip().upper()
        side = "short" if str(side or "").lower().startswith("short") else "long"
        board = self.movers_board.board()
        row = None
        for mode in ("pop", "dip", "rip", "mine"):
            for candidate in ((board.get(mode) or {}).get(side)) or []:
                if str(candidate.get("symbol") or "").upper() == symbol:
                    row = candidate
                    break
            if row is not None:
                break
        if self.focus_service is None:
            message = f"✕ {symbol} (no Focus service on this desk)"
        elif row is None:
            message = f"✕ {symbol} (no longer on the board)"
        else:
            # The same row gate as the automatic Movers feed (prior session checked).
            import movers_notify

            state, reason, _levels = movers_notify.row_level_gate(
                row, side, str(board.get("as_of") or "")[:10]
            )
            passes = state == focus_adoption_gate.OPEN
            if not passes:
                message = f"✕ {symbol} ({reason})"
            else:
                try:
                    added = self.focus_service.add(
                        symbol, side, "m5", origin="movers_board",
                        context=f"movers 15m {row.get('move15_pct')}",
                    )
                    message = (
                        f"★ {symbol} added to M5 Focus ({side})." if added
                        else f"{symbol} is already in M5 Focus ({side})."
                    )
                except Exception:
                    logging.warning("Movers board could not add %s.", symbol, exc_info=True)
                    message = f"✕ {symbol} (add failed)"
        self.movers_board.show_status(message)
        self.statusChanged.emit(message)

    def attach_movers_service(self, service) -> None:
        """Feed the Movers board from `MainWindow`'s one MoversService. Hosting only."""
        service.moversChanged.connect(self.movers_board.update_board)
        notice_signal = getattr(service, "moversNotice", None)
        if notice_signal is not None:
            notice_signal.connect(self.announce_movers)
        adopt_signal = getattr(service, "moversAdopt", None)
        if adopt_signal is not None:
            adopt_signal.connect(self.adopt_movers_picks)
        board = service.board()
        if board:
            self.movers_board.update_board(board)

    def adopt_movers_picks(self, payload: dict) -> None:
        """New Movers names onto the M5 Focus AUTO lane (trader 2026-09-25).

        Only a name whose level gate held on the bar (`passes`) is added: long
        above the previous session's high and session VWAP, short below both.
        "Not today" and a name taken off today are refused. Writes go through the
        store plus an auto-pick marker (never `FocusService.add`, which would log
        a trader "like"); an existing entry is never re-marked; nothing is removed.
        Refusals are one log line; the `kind: adopt` rows are written off-thread.
        """
        candidates = [dict(c) for c in (payload or {}).get("candidates") or []
                      if isinstance(c, dict)]
        if not candidates:
            return
        store = getattr(self.focus_service, "store", None)
        declined_today = getattr(store, "declined_today", None)
        wanted: dict[str, list[dict]] = {"long": [], "short": []}
        for cand in candidates:
            symbol, side = str(cand.get("symbol") or "").upper(), cand.get("side")
            taken_off = False
            if callable(declined_today):
                try:
                    taken_off = bool(declined_today(symbol, side, "m5"))
                except Exception:
                    taken_off = False
            if not cand.get("passes"):
                cand["result"] = f"gated: {cand.get('gate_reason') or cand.get('gate')}"
            elif not SYMBOL_RE.fullmatch(symbol):
                cand["result"] = "refused: not a ticker"
            elif store is None:
                cand["result"] = "refused: no Focus store"
            elif symbol in self._ignored_symbols:
                cand["result"] = "refused: you said not today"
            elif taken_off:
                cand["result"] = "refused: you took it off today"
            elif side in wanted and all(c["symbol"] != symbol for c in wanted[side]):
                wanted[side].append(cand)
            else:
                cand["result"] = "refused: duplicate"
        for side, group in wanted.items():
            if not group:
                continue
            try:
                added = set(store.add_many([c["symbol"] for c in group], side, "m5"))
            except Exception:
                logging.warning("Movers could not add names to M5 Focus.", exc_info=True)
                for cand in group:
                    cand["result"] = "refused: add failed"
                continue
            marker_writer = getattr(store, "mark_auto_adopted", None)
            for cand in group:
                if cand["symbol"] not in added:
                    cand["result"] = "already in M5 Focus"
                    continue
                cand["result"] = "adopted"
                if callable(marker_writer):
                    try:
                        marker_writer(cand["symbol"], side, "m5", staged_at=str(cand.get("bar") or ""),
                                      reason=f"Movers {cand.get('label')} {side}: {cand.get('gate_reason')}")
                    except Exception:
                        logging.warning("Movers could not mark %s auto-adopted.", cand["symbol"],
                                        exc_info=True)
        refused = [f"{c['symbol']} {c['side']} ({c['result']})" for c in candidates
                   if c.get("result") not in ("adopted", "already in M5 Focus")]
        if refused:
            logging.info("Movers M5 watch: not added: %s", "; ".join(refused))
        self._write_movers_adopt_rows(payload, candidates)

    def _write_movers_adopt_rows(self, payload: dict, candidates: list[dict]) -> None:
        """Append the `kind: adopt` rows on a daemon thread (never the Qt thread)."""
        import threading
        from pathlib import Path

        import movers_notify
        import movers_outcomes

        paths = (payload or {}).get("log_paths") or {}
        rows = movers_notify.adopt_records(candidates, at=str((payload or {}).get("at") or ""))
        by_path: dict[str, list[dict]] = {}
        for row in rows:
            path = paths.get(movers_notify.LOG_FOR_LIST.get(str(row.get("list")), "dip"))
            if path:
                by_path.setdefault(str(path), []).append(row)

        def write() -> None:
            for path, chunk in by_path.items():
                movers_outcomes.append_records(Path(path), chunk)

        thread = threading.Thread(target=write, name="movers-adopt-log", daemon=True)
        self._movers_adopt_log_thread = thread
        thread.start()

    def announce_movers(self, notice: dict) -> None:
        """A DESK Movers notice: the Alert Center's beep (same checkbox and mode
        rule as alerts) and the line in the status bar and under the board."""
        line = str((notice or {}).get("line") or "")
        if not line:
            return
        if self._alerts_may_sound():
            QApplication.beep()
        self.movers_board.show_status(line)
        self.statusChanged.emit(line)

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
