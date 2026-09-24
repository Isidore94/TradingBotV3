from __future__ import annotations

"""User-armed one-shot M5 chart watches for the visual alert review surface.

The trader arms a watch ("New HOD", "New LOD", "HOD/LOD AVWAP", "VWAP
bounce") while looking at a symbol's M5 chart in the Alert Center's visual
review pane. Each watch
is session-scoped and one-shot: the first COMPLETED M5 bar that meets the
condition produces a trigger (the hosting panel turns it into a red Alert
Center alert) and the watch is retired. A forming bar is preview only and
never triggers - plan.md section 5.

Pure module: plain datetimes and bar dicts ({"dt", "open", "high", "low",
"close", "volume"} as returned by ``BounceBot.m5_chart_bars``), no Qt, no
network. VWAP comes from ``chart_snapshot.session_vwap_series`` so the
bounce condition uses the exact running-deviation band math the desk is
calibrated to.
"""

import json
import math
import os
import uuid
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

from chart_snapshot import anchored_vwap_band_series, session_vwap_series

M5_BAR_SPAN = timedelta(minutes=5)

# kind -> button/badge label. Ordered as the buttons appear in the review pane.
WATCH_KINDS = {
    "new_hod": "New HOD",
    "new_lod": "New LOD",
    "hod_avwap": "HOD AVWAP",
    "lod_avwap": "LOD AVWAP",
    "vwap_bounce": "VWAP bounce",
    "band_bounce": "σ-band bounce",
    "pullback": "Pullback alert",
}

#: PCT-1 (trader, 2026-09-15: *"in the same vein as H1 retester we should
#: rename it to Pullback alert and include any of these phenomena in the alert
#: pattern"*). ONE button, ONE kind, named TRIGGERS: the hourly 15-EMA
#: retester WISHLIST 10C shipped is now one of four things this watch waits
#: for, and the other three are the trader's M15/M30 SMA pullback.
PULLBACK_KIND = "pullback"

#: The stored kind every watch armed before the rename carries. It is NOT a
#: button any more and not in `WATCH_KINDS`; `chart_watch_from_dict` loads such
#: a row as a `pullback` watch whose only trigger is the H1 bounce it was armed
#: for, so nothing the trader armed is lost and nothing they did not ask for is
#: added to it.
H1_EMA_BOUNCE_KIND = "h1_ema_bounce"

#: The four triggers a Pullback alert waits for. `h1_ema15_bounce` is the
#: WISHLIST 10C rule, unchanged and still `h1_ema_bounce_v1`; the other three
#: are `indicators.pullback_sma_reclaim`'s, on M15 and M30. Every fire names
#: its trigger and its timeframe, so one button never hides which phenomenon
#: spoke.
TRIGGER_H1_EMA15_BOUNCE = "h1_ema15_bounce"
TRIGGER_SMA_RECLAIM_LRSI = "sma_reclaim_lrsi"
TRIGGER_RECLAIM_THEN_LRSI = "reclaim_then_lrsi"
TRIGGER_SMA_RETEST = "sma_retest"
PULLBACK_TRIGGERS = (
    TRIGGER_H1_EMA15_BOUNCE,
    TRIGGER_SMA_RECLAIM_LRSI,
    TRIGGER_RECLAIM_THEN_LRSI,
    TRIGGER_SMA_RETEST,
)

#: Watch kinds that are NOT session-scoped. Every other kind on this surface
#: dies at midnight because it is a statement about today's tape ("a new high
#: for the session"); the Pullback alert is a statement about a multi-day
#: pattern and is given ten TRADING days by `armed_alert_expiry`, so it
#: survives a desk restart and tomorrow's date roll. One name, three readers:
#: `load_chart_watches` (which otherwise drops the whole file on a market-date
#: mismatch), `watch_is_stale` (which the panel's M5 poll uses to retire), and
#: the armed inventory's health column. A kind in here also belongs on the D1
#: armed-event feed rather than the session M5 list.
PERSISTENT_WATCH_KINDS = frozenset({PULLBACK_KIND})

# The σ-band button mirrors the day-trade tracker's measured M5 winners:
# long = dynamic_vwap_upper_band (ride above +1σ, dip-tag it, reclaim),
# short = dynamic_vwap_lower_band (the mirror below -1σ). The tracker's
# 2026-07-24 read puts the family's prime production in the late-morning and
# afternoon buckets; trigger alerts annotate accordingly.
BAND_BOUNCE_TRACKER_TYPES = {
    "long": "dynamic_vwap_upper_band",
    "short": "dynamic_vwap_lower_band",
}
BAND_BOUNCE_PRIME_BUCKETS = ("late_morning", "afternoon")

# Persistent D1 candle-level alerts (armed by clicking a D1 candle). Not in
# WATCH_KINDS: they are level breaks, not session-scoped chart watches.
D1_LEVEL_KINDS = {
    "d1_level_above": "D1 break above",
    "d1_level_below": "D1 break below",
}

# Persistent D1 EVENT watches (armed from the dock's D1 row). Unlike a level
# watch, the reference is derived fresh from the daily store on every
# evaluation - a 5-day extreme, an SMA, or the D1 15EMA moves each session,
# and freezing it at arm time would alert on yesterday's number.
D1_EVENT_KINDS = {
    "ema15_reject": "15EMA reject",
    "new_5d_high": "5d high",
    "new_5d_low": "5d low",
    "new_20d_high": "20d high",
    "new_20d_low": "20d low",
    "sma_break": "SMA break",
    # AVWAPE (current earnings anchored VWAP) levels. The trader trades the
    # line itself and the FIRST deviation band - nothing watches 2σ/3σ (user
    # rule 2026-07-29). Bounce = tag + close back on the approach side;
    # break = close THROUGH. The trigger message names the exact level.
    "avwape_bounce": "AVWAPE bounce",
    "avwape_break": "AVWAPE break",
    "avwape_dev1_bounce": "1σ bounce",
    "avwape_dev1_break": "1σ break",
    # PCT-2 is deliberately the exception to the derived-level rule below:
    # its scan line is frozen when the trader arms it, so a redraw cannot move
    # the alert that was requested.
    "trendline_break": "Trendline break",
    "trendline_break_retest": "Trendline break + retest",
    # Grouped menu kinds (2026-09-24): each fires on the FIRST of its parts.
    "d1_line_pullback": "Pullback to D1 line",
    "range_breakout": "Range breakout",
    "line_break": "Line break",
}

# The parts each grouped kind checks, in order; the fire names the part that hit.
D1_LINE_PULLBACK_PARTS = ("ema15_reject", "avwape_bounce", "avwape_dev1_bounce")
D1_LINE_BREAK_PARTS = ("sma_break", "avwape_break", "avwape_dev1_break")

# The D1 alert menu, grouped by principle. "pullback" is the M5-store Pullback
# alert (WATCH_KINDS); every other entry is a D1 event kind.
D1_MENU_GROUPS = (
    ("PULLBACK — it ran, let it calm down", ("pullback", "d1_line_pullback")),
    ("BREAKOUT — it was tight, let it go", ("range_breakout",)),
    (
        "LINE BREAK — it crossed a big line",
        ("line_break", "trendline_break", "trendline_break_retest"),
    ),
)
D1_MENU_LABELS = {"pullback": "Pullback (fast)"}
D1_MENU_KINDS = tuple(
    kind for _title, kinds in D1_MENU_GROUPS for kind in kinds if kind in D1_EVENT_KINDS
)
# Kinds off the menu that still load, evaluate and fire for saved rows.
D1_LEGACY_KINDS = tuple(kind for kind in D1_EVENT_KINDS if kind not in D1_MENU_KINDS)

# EXTENSION events say "the move is going": a new range high/low, or a close
# THROUGH a major line. PULLBACK events say "it came back to something": a
# bounce off a level, or a rejection at one. The split drives the Focus
# auto-watch's one-extension-per-day rule (trader rule 2026-08-05, on FRPT
# printing a new 20-day high and then simply staying extended: "it comes up as
# a new 20 day high alert but now it's extended and I'd only want to see it on
# an SMA bounce or something"). Coarse on purpose - an SMA break DOWN on a long
# is really invalidation, not extension - but "break = the move continues,
# bounce/reject = it came back" is the distinction the trader reads.
D1_EXTENSION_KINDS = frozenset(
    {
        "new_5d_high",
        "new_5d_low",
        "new_20d_high",
        "new_20d_low",
        "sma_break",
        "avwape_break",
        "avwape_dev1_break",
        "trendline_break",
        "range_breakout",
        "line_break",
    }
)
# Multi-bar thesis watches are armed only by the trader.  They are neither an
# automatic Focus pullback nor a one-bar extension. `d1_line_pullback` repeats
# the auto lane's own kinds, so it stays out of that lane (no double fire).
D1_TRADER_ONLY_KINDS = frozenset({"trendline_break_retest", "d1_line_pullback"})
D1_PULLBACK_KINDS = (
    frozenset(D1_EVENT_KINDS) - D1_EXTENSION_KINDS - D1_TRADER_ONLY_KINDS
)

TRENDLINE_BREAK_RETEST_RULE_VERSION = "trendline_break_retest_v1"
TRENDLINE_BREAK_RETEST_ATR_LENGTH = 14
TRENDLINE_RETEST_TOUCH_ATR = 0.25
TRENDLINE_RETEST_CONFIRM_ATR = 0.10
TRENDLINE_RETEST_MAX_BARS = 10

# Which of the derived AVWAPE levels each kind watches ("" = the line).
_AVWAPE_KIND_BANDS = {
    "avwape_bounce": ("",),
    "avwape_break": ("",),
    "avwape_dev1_bounce": ("+1σ", "-1σ"),
    "avwape_dev1_break": ("+1σ", "-1σ"),
}

# range_breakout: a new 20-day high/low only when the prior 20 sessions'
# high-low range is <= 4.0 x D1 ATR14 (about the tightest third of such events).
RANGE_BREAKOUT_RULE_VERSION = "range_breakout_v1"
RANGE_BREAKOUT_BASE_SESSIONS = 20
RANGE_BREAKOUT_TIGHT_ATR = 4.0
RANGE_BREAKOUT_ATR_LENGTH = 14
# ATR input tail: enough bars for Wilder smoothing to settle, cheap per poll.
RANGE_BREAKOUT_ATR_WINDOW = 60


def d1_kind_needs_avwape(kind: str) -> bool:
    """Whether a D1 event kind reads the AVWAPE levels (needs the earnings anchor)."""
    return kind in _AVWAPE_KIND_BANDS or kind in ("d1_line_pullback", "line_break")

# SMA periods the sma_break watch monitors ("anyone up or down"): the desk's
# three D1 majors, matching the snapshot chart's overlays.
D1_BREAK_SMA_PERIODS = (50, 100, 200)
# An EMA needs history to mean anything; below this many completed sessions
# the 15EMA is mostly seed value and the reject watch just waits.
D1_EMA15_MIN_SESSIONS = 15


@dataclass(frozen=True)
class ChartWatch:
    symbol: str
    kind: str
    armed_at: datetime
    side: str = "WATCH"
    baseline: float | None = None
    source_text: str = ""
    #: Stable identity for this ARM, so a fire can be de-duplicated on the
    #: phone and a disarm/re-arm is unambiguously a new episode rather than a
    #: second chance at the old one. Blank on every row written before
    #: WISHLIST 10C - absent is blank, never an error.
    watch_id: str = ""
    #: What the trader is waiting for, in their own words, for the armed
    #: inventory to print back at them.
    reason: str = ""
    #: PCT-1: which phenomena this watch waits for. Empty on every kind but
    #: `pullback` (whose condition IS its trigger list) and on every row
    #: written before the rename - absent is empty, never an error.
    triggers: tuple[str, ...] = ()
    #: An optional, deliberately narrow set of legs for a Pullback alert.
    #: Empty is the legacy/manual meaning: judge every leg the button has
    #: always armed.  The veto-created pullback uses ("M30", "H1") so its
    #: M15 cache may still be a companion input but can never speak alone.
    timeframes: tuple[str, ...] = ()
    #: trigger -> the bar time it last fired on, so one event speaks once and
    #: a NEW episode's event still speaks. Persisted, so a desk restart does
    #: not re-announce a move the trader was already told about.
    fired: Mapping[str, str] = field(default_factory=dict)
    #: The trader disarmed a watch the desk armed for them. The row is KEPT so
    #: the auto-arm sweep does not simply put it back while that claim or
    #: Focus pick lives; it is hidden from the Armed board, never evaluated
    #: and never pushed. A watch the trader armed by hand is deleted on
    #: disarm, exactly as before.
    declined: bool = False


@dataclass(frozen=True)
class ChartWatchTrigger:
    watch: ChartWatch
    price: float
    bar_dt: datetime
    message: str
    # The direction the trigger actually fired as ("long"/"short") for the
    # bounce kinds a WATCH-side watch can hit either way; "" when the watch's
    # own side already says it.
    resolved_side: str = ""
    # Measured facts the hosting panel copies onto the fired alert's payload.
    # Empty for every kind whose message already says everything it measured.
    details: Mapping[str, Any] = field(default_factory=dict)


def _naive(moment: datetime) -> datetime:
    # IB serves this desk's bars on the local clock (sometimes tz-stamped);
    # arm times come from the same clock, so comparisons drop tzinfo rather
    # than convert across zones.
    return moment.replace(tzinfo=None) if moment.tzinfo is not None else moment


def _parse_date(value: object) -> date | None:
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _session_bars(bars: Iterable[Mapping[str, Any]] | None, moment: datetime) -> list[dict[str, Any]]:
    session = _naive(moment).date()
    kept = []
    for bar in bars or []:
        stamp = bar.get("dt")
        if isinstance(stamp, datetime) and _naive(stamp).date() == session:
            kept.append(dict(bar))
    kept.sort(key=lambda bar: _naive(bar["dt"]))
    return kept


def _bar_end(bar: Mapping[str, Any]) -> datetime:
    return _naive(bar["dt"]) + M5_BAR_SPAN


def completed_session_bars(
    m5_bars: Iterable[Mapping[str, Any]] | None,
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Today's M5 bars that have finished printing, oldest first.

    The forming bar is a preview (plan.md sec 5), so a caller deciding a live
    state - "is this name trading above yesterday's high yet" - reads only
    what the tape actually printed.
    """
    moment = _naive(now or datetime.now())
    return [bar for bar in _session_bars(m5_bars, moment) if _bar_end(bar) <= moment]


def last_completed_session_close(
    m5_bars: Iterable[Mapping[str, Any]] | None,
    *,
    now: datetime | None = None,
) -> float | None:
    """Close of today's last COMPLETED M5 bar, or None before the first one."""
    for bar in reversed(completed_session_bars(m5_bars, now=now)):
        try:
            return float(bar["close"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def arm_chart_watch(
    kind: str,
    symbol: str,
    side: str,
    bars: Iterable[Mapping[str, Any]] | None,
    *,
    now: datetime | None = None,
    source_text: str = "",
    timeframes: tuple[str, ...] = (),
) -> ChartWatch:
    """Arm a watch against what the trader sees on the chart right now.

    The HOD/LOD baseline is today's extreme across ALL cached bars including
    a forming one - that is exactly the day high/low drawn on the chart at
    the moment the button is clicked. Triggering later still requires a
    completed bar.
    """
    if kind not in WATCH_KINDS:
        raise ValueError(f"unknown chart watch kind: {kind!r}")
    moment = _naive(now or datetime.now())
    session = _session_bars(bars, moment)
    baseline: float | None = None
    if kind == "new_hod" and session:
        baseline = max(float(bar["high"]) for bar in session)
    elif kind == "new_lod" and session:
        baseline = min(float(bar["low"]) for bar in session)
    resolved_side = side if side in ("LONG", "SHORT") else "WATCH"
    return ChartWatch(
        symbol=str(symbol or "").strip().upper(),
        kind=kind,
        armed_at=moment,
        side=resolved_side,
        baseline=baseline,
        source_text=str(source_text or ""),
        watch_id=uuid.uuid4().hex,
        reason=watch_reason(kind, resolved_side),
        triggers=PULLBACK_TRIGGERS if kind == PULLBACK_KIND else (),
        timeframes=tuple(str(item).upper() for item in timeframes if str(item).strip()),
    )


def watch_reason(kind: str, side: str) -> str:
    """What the trader is waiting for, for the armed inventory to print back.

    Only the kinds whose condition is not already obvious from their label and
    baseline carry one; everything else keeps the blank it has always had. The
    Pullback alert names all three FAMILIES of trigger, because one button now
    covers four phenomena and a health cell that said only "pullback" would
    leave the trader guessing which one they are waiting on.
    """
    if kind == PULLBACK_KIND:
        label = side if side in ("LONG", "SHORT") else "EITHER SIDE"
        return (
            f"waiting for a pullback entry ({label}): H1 15-EMA bounce, "
            "M15/M30 SMA reclaim + LRSI, SMA retest"
        )
    return ""


def watch_is_stale(watch: ChartWatch, *, now: datetime | None = None) -> bool:
    """A session watch never survives into the next session.

    A PERSISTENT kind does: it is a statement about a multi-day pattern, not
    about today's tape, and its life is counted in trading days by
    `armed_alert_expiry` instead.
    """
    if str(getattr(watch, "kind", "") or "") in PERSISTENT_WATCH_KINDS:
        return False
    moment = _naive(now or datetime.now())
    return _naive(watch.armed_at).date() != moment.date()


def evaluate_chart_watch(
    watch: ChartWatch,
    bars: Iterable[Mapping[str, Any]] | None,
    *,
    now: datetime | None = None,
) -> ChartWatchTrigger | None:
    """First completed post-arm bar meeting the condition, or None."""
    moment = _naive(now or datetime.now())
    completed = [
        bar for bar in _session_bars(bars, moment) if _bar_end(bar) <= moment
    ]
    if not completed:
        return None
    if watch.kind in ("new_hod", "new_lod"):
        return _evaluate_extreme(watch, completed)
    if watch.kind in ("hod_avwap", "lod_avwap"):
        return _evaluate_extreme_avwap(watch, completed)
    if watch.kind == "vwap_bounce":
        return _evaluate_vwap_bounce(watch, completed)
    if watch.kind == "band_bounce":
        return _evaluate_band_bounce(watch, completed)
    # The Pullback alert is deliberately absent: none of its four triggers is
    # a session-scoped M5 condition. The `h1_ema15_bounce` one is evaluated
    # once per COMPLETED H1 BAR by `evaluate_h1_bounce_watch` below, from the
    # same cached M5 bars; the three SMA ones are evaluated on their own M15
    # and M30 series by `indicators.pullback_sma_reclaim`.
    return None


#: The ATR the H1 rule measures its distances in (Wilder, on H1 bars).
H1_ATR_LENGTH = 14

#: Which history a verdict was measured on, for the armed inventory to print.
H1_SOURCE_CACHE = "cache"
H1_SOURCE_YFINANCE = "yfinance"


def h1_bars_for_watch(
    m5_bars: Iterable[Mapping[str, Any]] | None,
    *,
    fallback_h1_bars: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """(completed H1 bars, which source they came from).

    **The desk's own cache is PRIMARY.** The fallback - `h1_history`'s yfinance
    read for an armed symbol - is consulted only when the cached M5 window
    cannot reach the rule's warm-up, and then only if it actually carries more
    bars. A symbol whose cache is long enough never touches the network, which
    is the property the second test in `tests/test_ws_10c_h1_retester_builder.py`
    exists to hold.
    """
    from indicators.h1_ema_bounce import WARMUP_BARS, closed_h1_bars

    cached = closed_h1_bars(m5_bars)
    if len(cached) >= WARMUP_BARS or not fallback_h1_bars:
        return cached, H1_SOURCE_CACHE
    fallback = [dict(bar) for bar in fallback_h1_bars]
    fallback.sort(key=lambda bar: _naive(bar["dt"]))
    if len(fallback) <= len(cached):
        return cached, H1_SOURCE_CACHE
    return fallback, H1_SOURCE_YFINANCE


def evaluate_h1_bounce_watch(
    watch: ChartWatch,
    m5_bars: Iterable[Mapping[str, Any]] | None,
    *,
    now: datetime | None = None,
    fallback_h1_bars: Iterable[Mapping[str, Any]] | None = None,
):
    """Run `h1_ema_bounce_v1` against the desk's cached M5 bars.

    Returns the rule's own `H1Bounce` (or `None` when nothing is measurable):
    the CALLER decides what a verdict costs, because a confirmation and an
    invalidation both end the watch but only one of them is an event worth a
    phone buzz.

    Cost, since this runs on the Qt thread inside the 60 s armed poll: one
    O(bars) pass to bucket ~1,100 cached M5 dicts into H1, one O(bars) ATR and
    one O(bars) EMA over the ~55 resulting bars, per armed H1 watch. Nothing
    is fetched and nothing is written; the M5 dicts are already materialised
    by `_m5_bars_for`.

    A WATCH-side arm (the chart had no side) is evaluated BOTH ways and the
    first confirmation wins - the same courtesy the VWAP and σ-band kinds
    already extend. It is never invalidated, because a close a full ATR
    through the line is the other side's setup, not this one's failure.
    """
    h1_bars, _source = h1_bars_for_watch(m5_bars, fallback_h1_bars=fallback_h1_bars)
    return evaluate_h1_bars(watch, h1_bars, now=now)


#: A `chart_watch`-level verdict, NEVER an indicator reason: the frozen rule
#: measured a real confirmation or invalidation, but its event bar had already
#: finished printing when this watch was armed. `h1_ema_bounce_v1` is a
#: statement about a series and knows nothing about arm times; deciding whose
#: episode an event belongs to is this module's job, exactly as it is for the
#: M5 kinds (`_evaluate_extreme`).
H1_PRE_ARM_REASON = "pre_arm"


def _market_local_zone():
    """The desk's market-local zone, or None when there is no zone database."""
    try:
        from market_session import get_market_local_timezone

        zone, _name = get_market_local_timezone()
    except Exception:  # pragma: no cover - settings or tzdata unavailable
        return None
    return zone


def _comparable_moments(left: datetime, right: datetime) -> tuple[datetime, datetime]:
    """Two stamps that can be compared as instants - ATTACH, never strip.

    `autopilot_core._gate_moment`'s pattern, and the 2026-08-19 outage's
    lesson: a naive stamp here is market-local by this store's own convention
    (`armed_at` stays naive) so the desk's zone is ATTACHED to it, while an
    aware stamp is already an instant and is kept as the instant it is.
    Stripping instead would read an arm written three hours west of the desk
    as three hours EARLIER than it happened, turning a pre-arm arm into a
    post-arm one - the quiet version of the same bug.
    """
    if (left.tzinfo is None) == (right.tzinfo is None):
        return left, right
    zone = _market_local_zone()
    if zone is None:  # pragma: no cover - last resort, no zone to attach
        return _naive(left), _naive(right)
    if left.tzinfo is None:
        left = left.replace(tzinfo=zone)
    if right.tzinfo is None:
        right = right.replace(tzinfo=zone)
    return left, right


def h1_bar_end(bar_dt: datetime) -> datetime:
    """When the session-aligned H1 bar starting at `bar_dt` finished printing.

    `bar_dt + 60 min`, except the day's short closing bucket, which ends at the
    bell - the one definition, shared with the fetched history
    (`h1_history.h1_bucket_end`), so the two sources cannot disagree about when
    a bar became the past.
    """
    try:
        from h1_history import h1_bucket_end

        return h1_bucket_end(bar_dt)
    except Exception:  # pragma: no cover - market_session unavailable
        return bar_dt + timedelta(minutes=60)


def h1_event_is_post_arm(watch: ChartWatch, event_bar_dt: datetime | None) -> bool:
    """Is this event bar the armed trader's, rather than yesterday's news?

    Eligible when the bar's END is STRICTLY after `armed_at`, which is the
    armed-watch convention the M5 kinds already hold (`_evaluate_extreme`:
    `_bar_end(bar) <= armed_at` is a pre-arm bar), inclusive on the pre-arm
    side. A bar that was still FORMING when the button was pressed (started
    before, ends after) is therefore the trader's once it completes - the same
    courtesy the M5 kinds give.
    """
    armed_at = getattr(watch, "armed_at", None)
    if not isinstance(event_bar_dt, datetime) or not isinstance(armed_at, datetime):
        # An event that cannot be dated is NOT the trader's (lead ruling on
        # the arm-time reviewer's advisory, 2026-09-13): missing data is
        # uncertainty, never confirmation, so the fence fails CLOSED - the
        # watch stays armed and answers on the next bar it can date.
        # Unreachable today (`armed_at` is a required field and the frozen
        # rule stamps `confirm_bar_dt` on every fire and invalidation).
        return False
    end, armed = _comparable_moments(h1_bar_end(event_bar_dt), armed_at)
    return end > armed


def _fence_pre_arm(watch: ChartWatch, result):
    """A confirmation or invalidation that finished before the arm is not an event.

    The frozen rule is NOT asked a different question and the series is NOT
    trimmed: the EMA and the ATR still warm up over every bar, and while a
    pre-arm closing-through bar sits inside the rule's age window the rule
    keeps answering `invalidated` - the watch simply waits, exactly as it waits
    on `awaiting_reclaim`, until that bar ages out or a post-arm event lands.
    """
    from indicators.h1_ema_bounce import REASON_INVALIDATED

    if result is None:
        return None
    if not (result.fired or result.reason == REASON_INVALIDATED):
        return result
    event_bar_dt = getattr(result, "confirm_bar_dt", None)
    if h1_event_is_post_arm(watch, event_bar_dt):
        return result
    note = "the event bar had already closed when this watch was armed"
    if isinstance(event_bar_dt, datetime):
        note = (
            f"the {event_bar_dt.strftime('%m/%d %H:%M')} bar closed at "
            f"{h1_bar_end(event_bar_dt).strftime('%H:%M')}, before this watch "
            "was armed"
        )
    return replace(
        result,
        fired=False,
        reason=H1_PRE_ARM_REASON,
        reasons=tuple(result.reasons) + (note,),
    )


def evaluate_h1_bars(watch: ChartWatch, h1_bars, *, now: datetime | None = None):
    """The rule against a series the caller has already chosen (see above).

    **Only a POST-ARM event may finish the watch.** The rule anchors its
    verdict at the LAST completed bar, so a series that already holds a
    finished bounce would otherwise fire the instant the trader armed - on a
    move that was over before they pressed the button (review blocker B2,
    2026-09-13). Every bar is still kept for warm-up; what is fenced is the
    EVENT, whose bar must END strictly after `armed_at` (`h1_event_is_post_arm`).
    A pre-arm confirmation or invalidation comes back as `pre_arm`: not fired,
    not invalidated, so the caller leaves the watch armed and nothing is
    recorded, pushed or drawn.
    """
    from indicators.atr import wilder_atr
    from indicators.h1_ema_bounce import REASON_INVALIDATED, evaluate

    if not h1_bars:
        return None
    atr = wilder_atr(h1_bars, H1_ATR_LENGTH)
    sides = (
        (watch.side,) if watch.side in ("LONG", "SHORT") else ("LONG", "SHORT")
    )
    results = [
        fenced
        for fenced in (
            _fence_pre_arm(watch, evaluate(h1_bars, side, atr=atr, now=now))
            for side in sides
        )
        if fenced is not None
    ]
    if not results:
        return None
    for result in results:
        if result.fired:
            return result
    live = [result for result in results if result.reason != REASON_INVALIDATED]
    return live[0] if live else results[0]


def h1_bounce_message(watch: ChartWatch, result) -> str:
    """The one line the alert, the phone and the decision log all read."""
    side = str(getattr(result, "side", "") or "").upper() or watch.side
    touch = getattr(result, "touch_bar_dt", None)
    confirm = getattr(result, "confirm_bar_dt", None)
    when = ""
    if isinstance(touch, datetime) and isinstance(confirm, datetime):
        when = (
            f" - tagged {touch.strftime('%m/%d %H:%M')}, "
            f"reclaimed {confirm.strftime('%m/%d %H:%M')}"
        )
    distance = getattr(result, "distance_atr", None)
    how_close = f" ({distance:.2f} ATR off the line)" if distance is not None else ""
    return (
        f"{watch.symbol} {side}: H1 15-EMA retest confirmed{when}{how_close}"
    )


def _evaluate_extreme(
    watch: ChartWatch, completed: list[dict[str, Any]]
) -> ChartWatchTrigger | None:
    is_high = watch.kind == "new_hod"
    armed_at = _naive(watch.armed_at)
    baseline = watch.baseline
    for bar in completed:
        value = float(bar["high"] if is_high else bar["low"])
        if _bar_end(bar) <= armed_at:
            # Pre-arm bar: it can only tighten the reference level (covers a
            # watch armed before the bot had cached this symbol's bars).
            if baseline is None:
                baseline = value
            else:
                baseline = max(baseline, value) if is_high else min(baseline, value)
            continue
        if baseline is None:
            # No reference yet: the first tracked bar defines the day's
            # extreme instead of trivially "breaking" nothing.
            baseline = value
            continue
        if (is_high and value > baseline) or (not is_high and value < baseline):
            stamp = _naive(bar["dt"])
            if is_high:
                message = (
                    f"New HOD {value:.2f} > armed day high {baseline:.2f} "
                    f"(bar {stamp:%H:%M})"
                )
            else:
                message = (
                    f"New LOD {value:.2f} < armed day low {baseline:.2f} "
                    f"(bar {stamp:%H:%M})"
                )
            return ChartWatchTrigger(watch=watch, price=value, bar_dt=stamp, message=message)
    return None


def _evaluate_extreme_avwap(
    watch: ChartWatch, completed: list[dict[str, Any]]
) -> ChartWatchTrigger | None:
    """Cross-and-close through the AVWAP anchored on the session's extreme
    candle (trader request 2026-07-31; M5-only like every session watch).

    lod_avwap: the anchor is whichever completed bar printed the session LOW
    (the earliest when several share it; a fresh LOD re-anchors automatically
    on the next evaluation). Fires on the first completed post-arm bar that
    trades through that anchored VWAP from above (high >= AVWAP) and CLOSES
    below it. hod_avwap mirrors: anchor on the session-high candle, fire on a
    tag from below that closes above.

    The anchor candle itself never triggers - a one-bar AVWAP is just that
    bar's typical price, not a level anyone traded against.
    """
    is_low_anchor = watch.kind == "lod_avwap"
    armed_at = _naive(watch.armed_at)
    if is_low_anchor:
        extremes = [float(bar["low"]) for bar in completed]
        anchor_index = extremes.index(min(extremes))
    else:
        extremes = [float(bar["high"]) for bar in completed]
        anchor_index = extremes.index(max(extremes))
    avwap_values = anchored_vwap_band_series(completed, anchor_index)["avwap"]
    anchor_stamp = _naive(completed[anchor_index]["dt"])
    for index, bar in enumerate(completed):
        if index <= anchor_index:
            continue
        if _bar_end(bar) <= armed_at:
            continue
        avwap = avwap_values[index]
        if avwap is None:
            continue
        low = float(bar["low"])
        high = float(bar["high"])
        close = float(bar["close"])
        stamp = _naive(bar["dt"])
        if is_low_anchor and high >= avwap and close < avwap:
            return ChartWatchTrigger(
                watch=watch,
                price=close,
                bar_dt=stamp,
                message=(
                    f"LOD AVWAP break: closed {close:.2f} below AVWAP {avwap:.2f} "
                    f"anchored on the {anchor_stamp:%H:%M} LOD candle "
                    f"(bar {stamp:%H:%M})"
                ),
                resolved_side="short",
            )
        if not is_low_anchor and low <= avwap and close > avwap:
            return ChartWatchTrigger(
                watch=watch,
                price=close,
                bar_dt=stamp,
                message=(
                    f"HOD AVWAP reclaim: closed {close:.2f} above AVWAP {avwap:.2f} "
                    f"anchored on the {anchor_stamp:%H:%M} HOD candle "
                    f"(bar {stamp:%H:%M})"
                ),
                resolved_side="long",
            )
    return None


def _evaluate_vwap_bounce(
    watch: ChartWatch, completed: list[dict[str, Any]]
) -> ChartWatchTrigger | None:
    """Touch-and-reclaim off session VWAP on a completed bar.

    Long: the bar trades down to VWAP (low <= vwap) and closes back above.
    Short: the bar trades up to VWAP (high >= vwap) and closes back below.
    A WATCH-side watch accepts either direction.
    """
    armed_at = _naive(watch.armed_at)
    vwap_values = session_vwap_series(completed)["vwap"]
    want_long = watch.side in ("LONG", "WATCH")
    want_short = watch.side in ("SHORT", "WATCH")
    for index, bar in enumerate(completed):
        if _bar_end(bar) <= armed_at:
            continue
        vwap = vwap_values[index]
        if vwap is None:
            continue
        low = float(bar["low"])
        high = float(bar["high"])
        close = float(bar["close"])
        stamp = _naive(bar["dt"])
        if want_long and low <= vwap and close > vwap:
            return ChartWatchTrigger(
                watch=watch,
                price=close,
                bar_dt=stamp,
                message=(
                    f"VWAP bounce (long): tagged VWAP {vwap:.2f}, closed back "
                    f"above at {close:.2f} (bar {stamp:%H:%M})"
                ),
                resolved_side="long",
            )
        if want_short and high >= vwap and close < vwap:
            return ChartWatchTrigger(
                watch=watch,
                price=close,
                bar_dt=stamp,
                message=(
                    f"VWAP bounce (short): tagged VWAP {vwap:.2f}, closed back "
                    f"below at {close:.2f} (bar {stamp:%H:%M})"
                ),
                resolved_side="short",
            )
    return None


def _evaluate_band_bounce(
    watch: ChartWatch, completed: list[dict[str, Any]]
) -> ChartWatchTrigger | None:
    """Touch-and-reclaim off the session VWAP ±1σ band on a completed bar.

    Long: the bar tags the UPPER band from above (low <= +1σ) and closes back
    over it - the tracker's dynamic_vwap_upper_band continuation long.
    Short: the bar tags the LOWER band from below (high >= -1σ) and closes
    back under it. A WATCH-side watch accepts either direction.
    """
    armed_at = _naive(watch.armed_at)
    series = session_vwap_series(completed)
    upper_values = series["upper_1"]
    lower_values = series["lower_1"]
    want_long = watch.side in ("LONG", "WATCH")
    want_short = watch.side in ("SHORT", "WATCH")
    for index, bar in enumerate(completed):
        if _bar_end(bar) <= armed_at:
            continue
        upper = upper_values[index]
        lower = lower_values[index]
        low = float(bar["low"])
        high = float(bar["high"])
        close = float(bar["close"])
        stamp = _naive(bar["dt"])
        if want_long and upper is not None and low <= upper and close > upper:
            return ChartWatchTrigger(
                watch=watch,
                price=close,
                bar_dt=stamp,
                message=(
                    f"σ-band bounce (long): tagged +1σ {upper:.2f}, closed back "
                    f"above at {close:.2f} (bar {stamp:%H:%M})"
                ),
                resolved_side="long",
            )
        if want_short and lower is not None and high >= lower and close < lower:
            return ChartWatchTrigger(
                watch=watch,
                price=close,
                bar_dt=stamp,
                message=(
                    f"σ-band bounce (short): tagged -1σ {lower:.2f}, closed back "
                    f"below at {close:.2f} (bar {stamp:%H:%M})"
                ),
                resolved_side="short",
            )
    return None


# ---------------------------------------------------------------------------
# Persistence: intraday watches are trading-day scoped (a GUI restart keeps
# them armed; a new session drops them), mirroring alert_review_state.py.
# ---------------------------------------------------------------------------
def _market_date_text(value: date | str | None) -> str:
    return value.isoformat() if isinstance(value, date) else str(value or date.today().isoformat())


def _atomic_write_json(payload: dict, path: Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".tmp")
    try:
        staged.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(staged, target)
    finally:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass


def chart_watch_to_dict(watch: ChartWatch) -> dict:
    payload = {
        "symbol": watch.symbol,
        "kind": watch.kind,
        "armed_at": _naive(watch.armed_at).isoformat(),
        "side": watch.side,
        "baseline": watch.baseline,
        "source_text": watch.source_text,
        "watch_id": watch.watch_id,
        "reason": watch.reason,
        "triggers": list(watch.triggers or ()),
        "fired": dict(watch.fired or {}),
        "declined": bool(watch.declined),
    }
    # Do not rewrite an ordinary/manual arm merely to say that it has its
    # historical all-timeframe scope.  The field is only evidence for the
    # new narrow veto arm.
    if watch.timeframes:
        payload["timeframes"] = list(watch.timeframes)
    return payload


def chart_watch_from_dict(payload: Mapping[str, Any]) -> ChartWatch | None:
    """One stored row, or None when it cannot be read at all.

    A row stored as `h1_ema_bounce` before PCT-1 loads as a `pullback` watch
    whose ONLY trigger is `h1_ema15_bounce`: nothing the trader armed is lost
    by the rename, and nothing they did not ask for is added to it. A stored
    `pullback` row with no trigger list is a row written by a build that had
    only one list, so it gets all four.
    """
    try:
        armed_at = datetime.fromisoformat(str(payload["armed_at"]))
        kind = str(payload["kind"])
        symbol = str(payload["symbol"] or "").strip().upper()
    except (KeyError, TypeError, ValueError):
        return None
    stored_triggers = payload.get("triggers")
    if kind == H1_EMA_BOUNCE_KIND:
        kind = PULLBACK_KIND
        if stored_triggers is None:
            stored_triggers = [TRIGGER_H1_EMA15_BOUNCE]
    if not symbol or kind not in WATCH_KINDS:
        return None
    if kind == PULLBACK_KIND:
        triggers = tuple(
            str(name)
            for name in (
                stored_triggers if stored_triggers is not None else PULLBACK_TRIGGERS
            )
        )
    else:
        triggers = ()
    stored_timeframes = payload.get("timeframes")
    timeframes = (
        tuple(str(item).upper() for item in stored_timeframes if str(item).strip())
        if isinstance(stored_timeframes, (list, tuple))
        else ()
    )
    fired_payload = payload.get("fired")
    fired = (
        {str(key): str(value) for key, value in fired_payload.items()}
        if isinstance(fired_payload, Mapping)
        else {}
    )
    baseline = payload.get("baseline")
    try:
        baseline = float(baseline) if baseline is not None else None
    except (TypeError, ValueError):
        baseline = None
    side = str(payload.get("side") or "WATCH")
    return ChartWatch(
        symbol=symbol,
        kind=kind,
        armed_at=armed_at,
        side=side if side in ("LONG", "SHORT") else "WATCH",
        baseline=baseline,
        source_text=str(payload.get("source_text") or ""),
        # Absent on every row written before WISHLIST 10C: blank, never a raise.
        watch_id=str(payload.get("watch_id") or ""),
        reason=str(payload.get("reason") or ""),
        triggers=triggers,
        timeframes=timeframes,
        fired=fired,
        declined=bool(payload.get("declined") or False),
    )


def save_chart_watches(
    watches: Iterable[ChartWatch],
    path: Path,
    *,
    market_date: date | str | None = None,
) -> None:
    _atomic_write_json(
        {
            "market_date": _market_date_text(market_date),
            "watches": [chart_watch_to_dict(watch) for watch in watches],
        },
        path,
    )


def load_chart_watches(
    path: Path,
    *,
    market_date: date | str | None = None,
) -> list[ChartWatch]:
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8") if target.exists() else ""
    except OSError:
        return []
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    # Session watches never survive into a new session. The PERSISTENT kinds
    # do - an H1 retester is armed for ten TRADING days, so a desk restart (or
    # simply tomorrow) must not silently retire it while the session-scoped
    # kinds beside it in the same file still go.
    same_session = str(payload.get("market_date") or "") == _market_date_text(market_date)
    watches = []
    for item in payload.get("watches") or []:
        if isinstance(item, Mapping):
            watch = chart_watch_from_dict(item)
            if watch is None:
                continue
            if same_session or watch.kind in PERSISTENT_WATCH_KINDS:
                watches.append(watch)
    return watches


# ---------------------------------------------------------------------------
# Persistent D1 candle-level alerts: armed from a clicked D1 candle, kept
# ACROSS sessions until the level flags. The symbol need not be in any scan -
# evaluation uses whatever evidence exists (cached M5 bars while scanned, the
# durable daily store otherwise) and simply waits when there is none.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class D1LevelWatch:
    symbol: str
    direction: str  # "above" | "below"
    level: float
    armed_at: datetime
    candle_date: str = ""  # ISO date of the clicked candle

    @property
    def kind(self) -> str:
        return "d1_level_above" if self.direction == "above" else "d1_level_below"


def d1_level_watch_to_dict(watch: D1LevelWatch) -> dict:
    return {
        "symbol": watch.symbol,
        "direction": watch.direction,
        "level": watch.level,
        "armed_at": _naive(watch.armed_at).isoformat(),
        "candle_date": watch.candle_date,
    }


def d1_level_watch_from_dict(payload: Mapping[str, Any]) -> D1LevelWatch | None:
    try:
        symbol = str(payload["symbol"] or "").strip().upper()
        direction = str(payload["direction"])
        level = float(payload["level"])
        armed_at = datetime.fromisoformat(str(payload["armed_at"]))
    except (KeyError, TypeError, ValueError):
        return None
    if not symbol or direction not in ("above", "below") or not level > 0:
        return None
    return D1LevelWatch(
        symbol=symbol,
        direction=direction,
        level=level,
        armed_at=armed_at,
        candle_date=str(payload.get("candle_date") or ""),
    )


def save_d1_level_watches(watches: Iterable[D1LevelWatch], path: Path) -> None:
    _atomic_write_json(
        {"watches": [d1_level_watch_to_dict(watch) for watch in watches]},
        path,
    )


def load_d1_level_watches(path: Path) -> list[D1LevelWatch]:
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8") if target.exists() else ""
    except OSError:
        return []
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    watches = []
    for item in payload.get("watches") or []:
        if isinstance(item, Mapping):
            watch = d1_level_watch_from_dict(item)
            if watch is not None:
                watches.append(watch)
    return watches


# ---------------------------------------------------------------------------
# Persistent D1 event watches: condition alerts (new N-day extreme, SMA
# break, 15EMA rejection) whose reference levels are re-derived from the
# durable daily store on every poll. Kept across sessions until they fire.
# Triggers need a COMPLETED bar - M5 while the symbol is scanned (intraday
# latency), completed daily bars otherwise - plan.md section 5.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class D1EventWatch:
    symbol: str
    kind: str
    armed_at: datetime
    # The ordinary D1 event kinds derive a moving reference afresh. A
    # trendline is the opposite: preserve the scan geometry the trader saw at
    # arm time. Old rows keep their three original fields and therefore load
    # safely, but a trendline row without this evidence cannot confirm.
    side: str = ""
    trendline_candidate: dict[str, Any] | None = None
    trendline_knowledge_at: datetime | None = None

    @property
    def direction(self) -> str:
        """For chip/badge coloring only; sma_break/ema15_reject go either way."""
        return "below" if self.kind.endswith("_low") else "above"


def d1_event_watch_to_dict(watch: D1EventWatch) -> dict:
    payload = {
        "symbol": watch.symbol,
        "kind": watch.kind,
        "armed_at": _naive(watch.armed_at).isoformat(),
    }
    if watch.kind in {"trendline_break", "trendline_break_retest"}:
        payload.update(
            {
                "side": str(watch.side or "").strip().upper(),
                "trendline_candidate": dict(watch.trendline_candidate or {}),
                "trendline_knowledge_at": (
                    watch.trendline_knowledge_at.isoformat()
                    if watch.trendline_knowledge_at is not None
                    else ""
                ),
            }
        )
    return payload


def d1_event_watch_from_dict(payload: Mapping[str, Any]) -> D1EventWatch | None:
    try:
        symbol = str(payload["symbol"] or "").strip().upper()
        kind = str(payload["kind"])
        armed_at = datetime.fromisoformat(str(payload["armed_at"]))
    except (KeyError, TypeError, ValueError):
        return None
    if not symbol or kind not in D1_EVENT_KINDS:
        return None
    if kind not in {"trendline_break", "trendline_break_retest"}:
        return D1EventWatch(symbol=symbol, kind=kind, armed_at=armed_at)
    candidate = payload.get("trendline_candidate")
    knowledge_at = payload.get("trendline_knowledge_at")
    try:
        knowledge = datetime.fromisoformat(str(knowledge_at)) if knowledge_at else None
    except (TypeError, ValueError):
        knowledge = None
    return D1EventWatch(
        symbol=symbol,
        kind=kind,
        armed_at=armed_at,
        side=str(payload.get("side") or "").strip().upper(),
        trendline_candidate=dict(candidate) if isinstance(candidate, Mapping) else None,
        trendline_knowledge_at=knowledge,
    )


def save_d1_event_watches(watches: Iterable[D1EventWatch], path: Path) -> None:
    _atomic_write_json(
        {"watches": [d1_event_watch_to_dict(watch) for watch in watches]},
        path,
    )


def load_d1_event_watches(path: Path) -> list[D1EventWatch]:
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8") if target.exists() else ""
    except OSError:
        return []
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    watches = []
    for item in payload.get("watches") or []:
        if isinstance(item, Mapping):
            watch = d1_event_watch_from_dict(item)
            if watch is not None:
                watches.append(watch)
    return watches


def d1_event_levels(
    d1_bars: Iterable[Mapping[str, Any]] | None,
    *,
    session: date,
    avwape_anchor: date | None = None,
) -> dict[str, Any]:
    """Reference levels from COMPLETED daily sessions strictly before ``session``.

    Keys (present only when enough history exists): high_5d / low_5d /
    high_20d / low_20d (prior N-session extremes), sma50 / sma100 / sma200,
    ema15 (pandas ewm(span, adjust=False) recursion, matching the snapshot
    chart's drawn line), prev_close (the last completed session's close - the
    cross-detection anchor for the first bar of a new session), and - when
    ``avwape_anchor`` names a stored session - "avwape_levels": [(band label,
    price), ...] for AVWAPE and its ±1/2/3σ bands via the calc_anchored_vwap_
    bands running-deviation math (chart_snapshot.anchored_vwap_band_series).
    """
    completed = []
    for bar in d1_bars or []:
        stamp = bar.get("dt")
        if isinstance(stamp, datetime) and _naive(stamp).date() < session:
            completed.append(bar)
    completed.sort(key=lambda bar: _naive(bar["dt"]))
    if not completed:
        return {}
    levels: dict[str, Any] = {}
    closes = [float(bar["close"]) for bar in completed]
    levels["prev_close"] = closes[-1]
    for count in (5, 20):
        if len(completed) >= count:
            tail = completed[-count:]
            levels[f"high_{count}d"] = max(float(bar["high"]) for bar in tail)
            levels[f"low_{count}d"] = min(float(bar["low"]) for bar in tail)
    if len(completed) >= RANGE_BREAKOUT_BASE_SESSIONS:
        from indicators.atr import wilder_atr

        atr = wilder_atr(completed[-RANGE_BREAKOUT_ATR_WINDOW:], RANGE_BREAKOUT_ATR_LENGTH)
        if atr is not None and atr > 0:
            base = completed[-RANGE_BREAKOUT_BASE_SESSIONS:]
            base_range = max(float(bar["high"]) for bar in base) - min(
                float(bar["low"]) for bar in base
            )
            levels["atr14"] = float(atr)
            levels["base_range_20d"] = base_range
    for period in D1_BREAK_SMA_PERIODS:
        if len(closes) >= period:
            levels[f"sma{period}"] = sum(closes[-period:]) / float(period)
    if len(closes) >= D1_EMA15_MIN_SESSIONS:
        alpha = 2.0 / 16.0
        ema = closes[0]
        for value in closes[1:]:
            ema = alpha * value + (1.0 - alpha) * ema
        levels["ema15"] = ema
    if avwape_anchor is not None:
        anchor_index = None
        for index, bar in enumerate(completed):
            if _naive(bar["dt"]).date() == avwape_anchor:
                anchor_index = index
                break
        if anchor_index is not None:
            series = anchored_vwap_band_series(completed, anchor_index)
            if series["avwap"] and series["avwap"][-1] is not None:
                pairs = [("", series["avwap"][-1])]
                for k in (1, 2, 3):
                    pairs.append((f"+{k}σ", series[f"upper_{k}"][-1]))
                    pairs.append((f"-{k}σ", series[f"lower_{k}"][-1]))
                levels["avwape_levels"] = [
                    (label, float(value)) for label, value in pairs if value is not None
                ]
    return levels


def _d1_event_hit(
    kind: str,
    levels: Mapping[str, float],
    prev_close: float | None,
    high: float,
    low: float,
    close: float,
) -> tuple[str, str, float] | None:
    """(message core, resolved side, trigger price) for one evidence bar."""
    if kind in ("d1_line_pullback", "line_break"):
        parts = D1_LINE_PULLBACK_PARTS if kind == "d1_line_pullback" else D1_LINE_BREAK_PARTS
        title = "Pullback to D1 line" if kind == "d1_line_pullback" else "Line break"
        for part in parts:
            hit = _d1_event_hit(part, levels, prev_close, high, low, close)
            if hit is not None:
                message, side, price = hit
                return f"{title}: {message}", side, price
        return None
    if kind == "range_breakout":
        atr = levels.get("atr14")
        base_range = levels.get("base_range_20d")
        top = levels.get("high_20d")
        bottom = levels.get("low_20d")
        if not atr or base_range is None or top is None or bottom is None:
            return None
        ratio = base_range / atr
        if ratio > RANGE_BREAKOUT_TIGHT_ATR:
            return None
        tight = (
            f"tight base: 20-session range {base_range:.2f} = {ratio:.1f}x ATR14 "
            f"{atr:.2f}, needs <= {RANGE_BREAKOUT_TIGHT_ATR:.1f}x"
        )
        if high > top:
            return (
                f"Range breakout (long): {high:.2f} > {top:.2f} 20-day high ({tight})",
                "long",
                high,
            )
        if low < bottom:
            return (
                f"Range breakout (short): {low:.2f} < {bottom:.2f} 20-day low ({tight})",
                "short",
                low,
            )
        return None
    if kind in ("new_5d_high", "new_20d_high"):
        key = "high_5d" if kind == "new_5d_high" else "high_20d"
        level = levels.get(key)
        days = "5" if key == "high_5d" else "20"
        if level is not None and high > level:
            return (
                f"New {days}-day high: {high:.2f} > {level:.2f} (prior {days}-session high)",
                "long",
                high,
            )
        return None
    if kind in ("new_5d_low", "new_20d_low"):
        key = "low_5d" if kind == "new_5d_low" else "low_20d"
        level = levels.get(key)
        days = "5" if key == "low_5d" else "20"
        if level is not None and low < level:
            return (
                f"New {days}-day low: {low:.2f} < {level:.2f} (prior {days}-session low)",
                "short",
                low,
            )
        return None
    if kind == "sma_break":
        if prev_close is None:
            return None
        for period in D1_BREAK_SMA_PERIODS:
            sma = levels.get(f"sma{period}")
            if sma is None:
                continue
            if prev_close < sma and close > sma:
                return (
                    f"SMA{period} break up: closed {close:.2f} over {sma:.2f}",
                    "long",
                    close,
                )
            if prev_close > sma and close < sma:
                return (
                    f"SMA{period} break down: closed {close:.2f} under {sma:.2f}",
                    "short",
                    close,
                )
        return None
    if kind in ("avwape_bounce", "avwape_dev1_bounce"):
        allowed = _AVWAPE_KIND_BANDS[kind]
        pairs = [
            (label, level)
            for label, level in (levels.get("avwape_levels") or [])
            if label in allowed
        ]
        if prev_close is None or not pairs:
            return None
        # A bounce approaches the level from prev_close's side, tags it, and
        # closes back on that side. Picking the max (long) / min (short)
        # qualifying level keeps the label zone-honest: the close can never
        # sit beyond the next band out, or that band would qualify instead.
        long_hits = [
            (level, label)
            for label, level in pairs
            if prev_close > level and low <= level and close > level
        ]
        if long_hits:
            level, label = max(long_hits)
            name = f"AVWAPE {label}".rstrip()
            return (
                f"{name} bounce (long): tagged {level:.2f}, closed back above at {close:.2f}",
                "long",
                close,
            )
        short_hits = [
            (level, label)
            for label, level in pairs
            if prev_close < level and high >= level and close < level
        ]
        if short_hits:
            level, label = min(short_hits)
            name = f"AVWAPE {label}".rstrip()
            return (
                f"{name} bounce (short): tagged {level:.2f}, closed back below at {close:.2f}",
                "short",
                close,
            )
        return None
    if kind in ("avwape_break", "avwape_dev1_break"):
        allowed = _AVWAPE_KIND_BANDS[kind]
        pairs = [
            (label, level)
            for label, level in (levels.get("avwape_levels") or [])
            if label in allowed
        ]
        if prev_close is None or not pairs:
            return None
        crossed_up = [
            (level, label) for label, level in pairs if prev_close < level < close
        ]
        if crossed_up:
            # Name the furthest level the close carried through.
            level, label = max(crossed_up)
            name = f"AVWAPE {label}".rstrip()
            return (
                f"{name} break up: closed {close:.2f} over {level:.2f}",
                "long",
                close,
            )
        crossed_down = [
            (level, label) for label, level in pairs if prev_close > level > close
        ]
        if crossed_down:
            level, label = min(crossed_down)
            name = f"AVWAPE {label}".rstrip()
            return (
                f"{name} break down: closed {close:.2f} under {level:.2f}",
                "short",
                close,
            )
        return None
    if kind == "ema15_reject":
        ema = levels.get("ema15")
        if ema is None:
            return None
        # Touch-and-reclaim off the D1 15EMA, either way - the same shape as
        # the VWAP bounce, but against the daily line the desk trades off.
        if low <= ema and close > ema:
            return (
                f"D1 15EMA rejection (long): tagged {ema:.2f}, closed back above at {close:.2f}",
                "long",
                close,
            )
        if high >= ema and close < ema:
            return (
                f"D1 15EMA rejection (short): tagged {ema:.2f}, closed back below at {close:.2f}",
                "short",
                close,
            )
        return None
    return None


def _cached_d1_event_levels(
    cache: dict | None,
    d1_bars: Iterable[Mapping[str, Any]] | None,
    session: date,
    avwape_anchor: date | None,
) -> dict[str, Any]:
    """`d1_event_levels`, memoized inside one caller-supplied dict.

    `d1_event_levels` sorts ~490 bars and builds 5d/20d extremes, SMA
    50/100/200, an EMA15 recursion and the AVWAP band series. The Focus D1
    interest poll evaluates up to ten kinds per symbol per minute and every one
    of them re-entered it with the SAME arguments, ~105 symbols at a time.

    The cache is the CALLER's, and it must be scoped to one symbol and one
    `d1_bars` list for one tick - the key is only (session, anchor), because
    within that scope nothing else can vary. With ``cache=None`` this is
    exactly the call it replaced, which is what makes the fast path
    behaviour-identical by construction rather than by argument.
    """
    if cache is None:
        return d1_event_levels(d1_bars, session=session, avwape_anchor=avwape_anchor)
    key = (session, avwape_anchor)
    levels = cache.get(key)
    if levels is None:
        levels = d1_event_levels(d1_bars, session=session, avwape_anchor=avwape_anchor)
        cache[key] = levels
    return levels


def _trendline_candidate_is_frozen(candidate: Mapping[str, Any] | None) -> bool:
    """Whether an armed trendline carries one exact, scan-known line.

    The line id is derived by the chart's stable identity contract, while the
    dates *and prices* record the two pivots which drew it.  A partial legacy
    row remains readable, but no missing piece is safe to reconstruct in a
    later poll.
    """
    if not isinstance(candidate, Mapping):
        return False
    try:
        kind = str(candidate.get("type") or "").strip()
        line_id = str(candidate.get("line_id") or "").strip()
        start_date = _parse_date(candidate.get("start_date"))
        end_date = _parse_date(candidate.get("end_date"))
        lookback_end = _parse_date(candidate.get("lookback_end"))
        break_date = _parse_date(candidate.get("break_date"))
        start_price = float(candidate.get("start_price"))
        end_price = float(candidate.get("end_price"))
        price = float(candidate.get("current_line_price"))
        slope = float(candidate.get("slope_log_per_bar"))
    except (TypeError, ValueError):
        return False
    if (
        not kind
        or not line_id
        or start_date is None
        or end_date is None
        or lookback_end is None
        or break_date is None
        or start_date >= end_date
        or end_date > lookback_end
        or break_date < end_date
        or break_date > lookback_end
        or line_id != f"d1_trendline:{kind}:{start_date.isoformat()}_{end_date.isoformat()}"
    ):
        return False
    return all(
        math.isfinite(value) and value > 0
        for value in (start_price, end_price, price)
    ) and math.isfinite(slope)


def _trendline_anchor_index(daily: list[dict], candidate: Mapping[str, Any]) -> int | None:
    anchor_date = _parse_date(candidate.get("lookback_end"))
    if anchor_date is None:
        return None
    for index, bar in enumerate(daily):
        if _naive(bar["dt"]).date() == anchor_date:
            return index
    return None


def _weekday_offset(anchor: date, target: date) -> int:
    """Fallback only for a narrow post-arm daily slice without the anchor."""
    if target == anchor:
        return 0
    sign = 1 if target > anchor else -1
    cursor = anchor
    steps = 0
    while cursor != target:
        cursor += timedelta(days=sign)
        if cursor.weekday() < 5:
            steps += sign
    return steps


def _frozen_trendline_price(
    candidate: Mapping[str, Any], daily: list[dict], index: int
) -> float | None:
    """Project the arm-time line onto one completed D1 bar, never a redraw."""
    try:
        anchor_price = float(candidate["current_line_price"])
        slope = float(candidate["slope_log_per_bar"])
    except (KeyError, TypeError, ValueError):
        return None
    anchor_index = _trendline_anchor_index(daily, candidate)
    if anchor_index is None:
        anchor_date = _parse_date(candidate.get("lookback_end"))
        target_date = _naive(daily[index]["dt"]).date()
        if anchor_date is None:
            return None
        offset = _weekday_offset(anchor_date, target_date)
    else:
        offset = index - anchor_index
    exponent = slope * offset
    if abs(exponent) > 50:
        return None
    price = anchor_price * math.exp(exponent)
    return price if math.isfinite(price) and price > 0 else None


def _evaluate_frozen_trendline_break(
    watch: D1EventWatch, daily: list[dict], moment: datetime
) -> ChartWatchTrigger | None:
    """One close-through of the exact D1 line captured when the watch armed."""
    if not _trendline_candidate_is_frozen(watch.trendline_candidate):
        return None
    if not isinstance(watch.trendline_knowledge_at, datetime):
        return None
    side = str(watch.side or "").strip().upper()
    if side not in {"LONG", "SHORT"}:
        return None
    armed_at = _naive(watch.armed_at)
    candidate = watch.trendline_candidate
    for index, bar in enumerate(daily):
        stamp = _naive(bar["dt"])
        if stamp.date() <= armed_at.date() or stamp.date() >= moment.date() or index == 0:
            continue
        line = _frozen_trendline_price(candidate, daily, index)
        prior_line = _frozen_trendline_price(candidate, daily, index - 1)
        if line is None or prior_line is None:
            continue
        try:
            previous_close = float(daily[index - 1]["close"])
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            continue
        crossed = (
            previous_close <= prior_line and close > line
            if side == "LONG"
            else previous_close >= prior_line and close < line
        )
        if not crossed:
            continue
        break_date = stamp.date().isoformat()
        return ChartWatchTrigger(
            watch=watch,  # type: ignore[arg-type]
            price=close,
            bar_dt=stamp,
            message=(
                f"Trendline break ({side.lower()}): closed {close:.2f} through "
                f"frozen line {line:.2f} (D1 bar {stamp:%m/%d})"
            ),
            resolved_side=side.lower(),
            details={"break_date": break_date, "line_price": line},
        )
    return None


def _evaluate_frozen_trendline_break_retest(
    watch: D1EventWatch, daily: list[dict], moment: datetime
) -> ChartWatchTrigger | None:
    """Confirm a break, a later retest, then a later continuation close.

    All three observations use completed D1 bars and the exact line frozen at
    arm time.  ATR only sizes tolerance; missing ATR means unmeasured.
    """
    if not _trendline_candidate_is_frozen(watch.trendline_candidate):
        return None
    if not isinstance(watch.trendline_knowledge_at, datetime):
        return None
    side = str(watch.side or "").strip().upper()
    if side not in {"LONG", "SHORT"}:
        return None

    from indicators.atr import wilder_atr

    armed_at = _naive(watch.armed_at)
    candidate = watch.trendline_candidate
    break_index: int | None = None
    break_date: date | None = None
    retest_index: int | None = None
    retest_date: date | None = None

    for index, bar in enumerate(daily):
        stamp = _naive(bar["dt"])
        if stamp.date() <= armed_at.date() or stamp.date() >= moment.date() or index == 0:
            continue
        line = _frozen_trendline_price(candidate, daily, index)
        prior_line = _frozen_trendline_price(candidate, daily, index - 1)
        if line is None or prior_line is None:
            continue
        try:
            previous_close = float(daily[index - 1]["close"])
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            continue
        atr = wilder_atr(daily[: index + 1], TRENDLINE_BREAK_RETEST_ATR_LENGTH)
        if atr is None:
            continue

        if break_index is None:
            crossed = (
                previous_close <= prior_line and close > line
                if side == "LONG"
                else previous_close >= prior_line and close < line
            )
            if crossed:
                break_index = index
                break_date = stamp.date()
            continue

        if index - break_index > TRENDLINE_RETEST_MAX_BARS:
            break_index = None
            break_date = None
            retest_index = None
            retest_date = None
            continue

        wrong_side = (
            close < line - (TRENDLINE_RETEST_TOUCH_ATR * atr)
            if side == "LONG"
            else close > line + (TRENDLINE_RETEST_TOUCH_ATR * atr)
        )
        if wrong_side:
            break_index = None
            break_date = None
            retest_index = None
            retest_date = None
            continue

        if retest_index is None:
            touched = low <= line + (TRENDLINE_RETEST_TOUCH_ATR * atr) and high >= line - (
                TRENDLINE_RETEST_TOUCH_ATR * atr
            )
            held = close >= line if side == "LONG" else close <= line
            if touched and held:
                retest_index = index
                retest_date = stamp.date()
            continue

        if index <= retest_index:
            continue
        confirmed = (
            close >= line + (TRENDLINE_RETEST_CONFIRM_ATR * atr)
            if side == "LONG"
            else close <= line - (TRENDLINE_RETEST_CONFIRM_ATR * atr)
        )
        if not confirmed:
            continue
        return ChartWatchTrigger(
            watch=watch,  # type: ignore[arg-type]
            price=close,
            bar_dt=stamp,
            message=(
                f"Trendline break + retest ({side.lower()}): confirmed at {close:.2f} "
                f"over frozen line {line:.2f} (D1 bar {stamp:%m/%d})"
            ),
            resolved_side=side.lower(),
            details={
                "rule_version": TRENDLINE_BREAK_RETEST_RULE_VERSION,
                "break_date": break_date.isoformat() if break_date else "",
                "retest_date": retest_date.isoformat() if retest_date else "",
                "confirm_date": stamp.date().isoformat(),
                "line_id": str(candidate.get("line_id") or ""),
                "line_price": line,
                "atr": atr,
            },
        )
    return None


def evaluate_d1_event_watch(
    watch: D1EventWatch,
    m5_bars: Iterable[Mapping[str, Any]] | None,
    d1_bars: Iterable[Mapping[str, Any]] | None,
    *,
    now: datetime | None = None,
    avwape_anchor: date | None = None,
    levels_cache: dict | None = None,
) -> ChartWatchTrigger | None:
    """First post-arm completed bar meeting the condition, or None.

    Evidence mirrors the level watches: today's completed M5 bars against
    levels from sessions before today (intraday latency while scanned), then
    completed daily bars from sessions strictly after the arm date with
    per-session levels (covers unscanned symbols). SMA/AVWAPE crosses track
    the running previous close so a gap over a line counts exactly once.
    ``avwape_anchor`` (the current earnings anchor date) feeds the AVWAPE
    kinds; without it they simply wait.

    ``levels_cache`` is an optional dict the caller owns, scoped to one symbol
    and one ``d1_bars`` list for one tick. It changes nothing about what is
    evaluated - see `_cached_d1_event_levels` - only how many times the same
    reference levels get rebuilt when ten kinds are asked about one symbol.
    """
    moment = _naive(now or datetime.now())
    armed_at = _naive(watch.armed_at)

    daily = []
    for bar in d1_bars or []:
        stamp = bar.get("dt")
        if isinstance(stamp, datetime):
            daily.append(dict(bar))
    daily.sort(key=lambda bar: _naive(bar["dt"]))
    if watch.kind == "trendline_break":
        # This event has no intraday path: a wick or a forming D1 bar is not
        # confirmation, and the current scan is never consulted here.
        return _evaluate_frozen_trendline_break(watch, daily, moment)
    if watch.kind == "trendline_break_retest":
        # Like the direct break, this is completed-D1 evidence only.  The
        # break, retest and confirmation must be three distinct bars.
        return _evaluate_frozen_trendline_break_retest(watch, daily, moment)

    session_bars = _session_bars(m5_bars, moment)
    completed = [bar for bar in session_bars if _bar_end(bar) <= moment]
    if completed:
        levels = _cached_d1_event_levels(
            levels_cache,
            d1_bars,
            _naive(completed[0]["dt"]).date(),
            avwape_anchor,
        )
        prev_close = levels.get("prev_close")
        for bar in completed:
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            hit = None
            if _bar_end(bar) > armed_at:
                hit = _d1_event_hit(watch.kind, levels, prev_close, high, low, close)
            prev_close = close
            if hit is not None:
                message, side, price = hit
                stamp = _naive(bar["dt"])
                return ChartWatchTrigger(
                    watch=watch,  # type: ignore[arg-type] (duck-typed carrier)
                    price=price,
                    bar_dt=stamp,
                    message=f"{message} (M5 bar {stamp:%m/%d %H:%M})",
                    resolved_side=side,
                )

    for bar in daily:
        bar_date = _naive(bar["dt"]).date()
        # Completed sessions only, strictly after the arm date (the armed
        # day's own daily bar also contains pre-arm prices).
        if bar_date <= armed_at.date() or bar_date >= moment.date():
            continue
        levels = _cached_d1_event_levels(levels_cache, daily, bar_date, avwape_anchor)
        hit = _d1_event_hit(
            watch.kind,
            levels,
            levels.get("prev_close"),
            float(bar["high"]),
            float(bar["low"]),
            float(bar["close"]),
        )
        if hit is not None:
            message, side, price = hit
            return ChartWatchTrigger(
                watch=watch,  # type: ignore[arg-type]
                price=price,
                bar_dt=_naive(bar["dt"]),
                message=f"{message} (D1 bar {bar_date:%m/%d})",
                resolved_side=side,
            )
    return None


def evaluate_d1_level_watch(
    watch: D1LevelWatch,
    m5_bars: Iterable[Mapping[str, Any]] | None,
    d1_bars: Iterable[Mapping[str, Any]] | None,
    *,
    now: datetime | None = None,
) -> ChartWatchTrigger | None:
    """First post-arm evidence bar crossing the level, or None.

    Evidence: completed M5 bars ending after the arm (covers the armed day
    while the symbol is scanned), and COMPLETED daily bars from sessions
    strictly after the arm date (covers unscanned symbols; the armed day's
    own daily bar is excluded because it also contains pre-arm prices).
    """
    moment = _naive(now or datetime.now())
    armed_at = _naive(watch.armed_at)
    is_above = watch.direction == "above"
    level = float(watch.level)

    def _hit(value: float) -> bool:
        return value >= level if is_above else value <= level

    for bar in _session_bars(m5_bars, moment):
        if _bar_end(bar) <= armed_at or _bar_end(bar) > moment:
            continue
        value = float(bar["high"] if is_above else bar["low"])
        if _hit(value):
            stamp = _naive(bar["dt"])
            word = "above" if is_above else "below"
            return ChartWatchTrigger(
                watch=watch,  # type: ignore[arg-type] (duck-typed carrier)
                price=value,
                bar_dt=stamp,
                message=(
                    f"D1 level break {word} {level:.2f}: reached {value:.2f} "
                    f"(M5 bar {stamp:%m/%d %H:%M})"
                ),
                resolved_side="long" if is_above else "short",
            )

    for bar in d1_bars or []:
        stamp = bar.get("dt")
        if not isinstance(stamp, datetime):
            continue
        bar_date = _naive(stamp).date()
        # Completed sessions only, strictly after the arm date.
        if bar_date <= armed_at.date() or bar_date >= moment.date():
            continue
        value = float(bar["high"] if is_above else bar["low"])
        if _hit(value):
            word = "above" if is_above else "below"
            return ChartWatchTrigger(
                watch=watch,  # type: ignore[arg-type]
                price=value,
                bar_dt=_naive(stamp),
                message=(
                    f"D1 level break {word} {level:.2f}: reached {value:.2f} "
                    f"(D1 bar {bar_date:%m/%d})"
                ),
                resolved_side="long" if is_above else "short",
            )
    return None


# ---------------------------------------------------------------------------
# The any-bounce watch (R5 section 4).
#
# One armed request per symbol and side: "tell me when this name bounces off
# ANY of my levels." It is deliberately not a second price-alert system - it
# reuses the two-bar bounce idiom the D1 zone arms already use
# (``master_avwap_lib.d1_zone_arms.detect_zone_arm_triggers``), so a bounce
# means here exactly what it means there.
#
# It fires ONCE, naming the level that held, and disarms. The trader's own
# words for why that is right: "if I still dislike it when that alert fires
# then I can set it again."
#
# A level the data cannot supply is silently absent - never fabricated, never
# zero. A symbol with no zone-arms entry simply watches its session EMAs.
# ---------------------------------------------------------------------------

#: kind -> label. The trader's level set, 2026-08-14 and 2026-08-15.
ANY_BOUNCE_KINDS = {
    "d1_band_1": "D1 1st-dev band",
    "avwape": "current AVWAP",
    "prev_avwape": "previous AVWAP",
    "prev_band_1": "previous 1st-dev band",
    "d1_ema15": "D1 15 EMA",
    "d1_ema21": "D1 21 EMA",
    "m5_ema15": "session 5m 15 EMA",
    "m5_ema21": "session 5m 21 EMA",
    "h1_ema15": "H1 15 EMA",
}

#: The tolerance band for "dipped to it and held", as a fraction of the level.
#: The D1 zone arms carry a per-entry tolerance measured from the scan; a
#: watch armed from a chart has no such measurement, so this is the fallback
#: and it is deliberately small.
ANY_BOUNCE_TOLERANCE_FRACTION = 0.0015


@dataclass(frozen=True)
class AnyBounceWatch:
    """One armed 'bounce off any of these levels' request."""

    symbol: str
    side: str
    kinds: tuple[str, ...]
    armed_at: datetime

    @property
    def direction(self) -> str:
        """For chip/badge coloring: a long watches for a bounce UP."""
        return "below" if self.side == "short" else "above"


@dataclass(frozen=True)
class AnyBounceTrigger:
    watch: AnyBounceWatch
    kind: str
    level: float
    price: float
    triggered_at: datetime
    message: str

    @property
    def resolved_side(self) -> str:
        return self.watch.side


def any_bounce_watch_to_dict(watch: AnyBounceWatch) -> dict:
    return {
        "symbol": watch.symbol,
        "side": watch.side,
        "kinds": list(watch.kinds),
        "armed_at": _naive(watch.armed_at).isoformat(),
    }


def any_bounce_watch_from_dict(payload: Mapping[str, Any]) -> AnyBounceWatch | None:
    try:
        symbol = str(payload["symbol"] or "").strip().upper()
        side = str(payload["side"] or "").strip().lower()
        armed_at = datetime.fromisoformat(str(payload["armed_at"]))
        raw_kinds = payload["kinds"]
    except (KeyError, TypeError, ValueError):
        return None
    if not symbol or side not in ("long", "short"):
        return None
    kinds = tuple(
        kind for kind in (str(item) for item in raw_kinds or ()) if kind in ANY_BOUNCE_KINDS
    )
    if not kinds:
        # A watch with no recognised level watches nothing. Dropping it is
        # honest; keeping it would show an armed chip that can never fire.
        return None
    return AnyBounceWatch(symbol=symbol, side=side, kinds=kinds, armed_at=armed_at)


def save_any_bounce_watches(watches: Iterable[AnyBounceWatch], path: Path) -> None:
    _atomic_write_json(
        {"watches": [any_bounce_watch_to_dict(watch) for watch in watches]},
        path,
    )


def load_any_bounce_watches(path: Path) -> list[AnyBounceWatch]:
    target = Path(path)
    try:
        text = target.read_text(encoding="utf-8") if target.exists() else ""
    except OSError:
        return []
    if not text.strip():
        return []
    try:
        payload = json.loads(text)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    watches = []
    for item in payload.get("watches") or []:
        if isinstance(item, Mapping):
            watch = any_bounce_watch_from_dict(item)
            if watch is not None:
                watches.append(watch)
    return watches


def _ema_last(values: list[float], length: int) -> float | None:
    if len(values) < length:
        return None
    alpha = 2.0 / (float(length) + 1.0)
    ema = values[0]
    for value in values[1:]:
        ema = alpha * value + (1.0 - alpha) * ema
    return ema


def _hourly_closes(bars: Iterable[Mapping[str, Any]] | None, moment: datetime) -> list[float]:
    """Completed hourly closes aggregated from M5 bars.

    A forming hour is preview, so the bucket containing ``moment`` is dropped
    (plan.md sec 5). Aggregation is by clock hour, which is what the desk's
    hourly chart draws.
    """
    buckets: dict[datetime, float] = {}
    for bar in bars or []:
        stamp = bar.get("dt")
        if not isinstance(stamp, datetime):
            continue
        close = _finite_value(bar.get("close"))
        if close is None:
            continue
        hour = _naive(stamp).replace(minute=0, second=0, microsecond=0)
        buckets[hour] = close
    if not buckets:
        return []
    forming = _naive(moment).replace(minute=0, second=0, microsecond=0)
    return [close for hour, close in sorted(buckets.items()) if hour < forming]


def _finite_value(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def any_bounce_levels(
    *,
    zone_arm_entry: Mapping[str, Any] | None = None,
    m5_bars: Iterable[Mapping[str, Any]] | None = None,
    d1_levels: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, float]:
    """The armed level set for one symbol, from whatever data exists.

    D1 levels (the 1st-dev bands, the current and previous AVWAP, the daily
    15/21 EMAs) come from the scan's zone-arms entry and the daily store; the
    session 15/21 EMAs and the H1 15 EMA are aggregated from the cached M5
    bars. Anything missing is simply absent from the result - R5 section 6:
    "missing zone-arm data for a symbol means that level silently absent from
    the watch, never a fabricated level."
    """
    moment = _naive(now or datetime.now())
    levels: dict[str, float] = {}

    entry = zone_arm_entry if isinstance(zone_arm_entry, Mapping) else {}
    for source_key, kind in (
        ("avwape", "avwape"),
        ("prev_avwape", "prev_avwape"),
    ):
        value = _finite_value(entry.get(source_key))
        if value is not None:
            levels[kind] = value
    # The 1st-dev bands ride on the zone-arm trigger levels, whose names are
    # the established label vocabulary (UPPER_1/LOWER_1, PREV_UPPER_1/...).
    for arm in entry.get("trigger_levels") or []:
        if not isinstance(arm, Mapping):
            continue
        name = str(arm.get("name") or arm.get("label") or "").strip().upper()
        value = _finite_value(arm.get("level"))
        if value is None:
            continue
        if name in ("UPPER_1", "LOWER_1") and "d1_band_1" not in levels:
            levels["d1_band_1"] = value
        elif name in ("PREV_UPPER_1", "PREV_LOWER_1") and "prev_band_1" not in levels:
            levels["prev_band_1"] = value
        elif name in ("EMA_15", "D1_EMA_15") and "d1_ema15" not in levels:
            levels["d1_ema15"] = value
        elif name in ("EMA_21", "D1_EMA_21") and "d1_ema21" not in levels:
            levels["d1_ema21"] = value

    for source_key, kind in (("ema15", "d1_ema15"), ("ema21", "d1_ema21")):
        if kind in levels:
            continue
        value = _finite_value((d1_levels or {}).get(source_key))
        if value is not None:
            levels[kind] = value

    session = completed_session_bars(m5_bars, now=moment)
    closes = [
        value
        for value in (_finite_value(bar.get("close")) for bar in session)
        if value is not None
    ]
    for length, kind in ((15, "m5_ema15"), (21, "m5_ema21")):
        value = _ema_last(closes, length)
        if value is not None:
            levels[kind] = value

    hourly = _hourly_closes(m5_bars, moment)
    h1_ema15 = _ema_last(hourly, 15)
    if h1_ema15 is not None:
        levels["h1_ema15"] = h1_ema15

    return levels


def evaluate_any_bounce_watch(
    watch: AnyBounceWatch,
    m5_bars: Iterable[Mapping[str, Any]] | None,
    levels: Mapping[str, float],
    *,
    now: datetime | None = None,
    tolerance_fraction: float = ANY_BOUNCE_TOLERANCE_FRACTION,
) -> AnyBounceTrigger | None:
    """The first armed level that produced a two-bar bounce, or ``None``.

    The idiom is ``detect_zone_arm_triggers``' own: bar A dips to within the
    tolerance of the level and closes on the right side of it, then bar B
    closes better than bar A and clear of the level. Both bars must be
    COMPLETED - a forming bar is preview and can un-happen.
    """
    moment = _naive(now or datetime.now())
    bars = completed_session_bars(m5_bars, now=moment)
    if len(bars) < 2:
        return None
    previous, latest = bars[-2], bars[-1]
    a_high = _finite_value(previous.get("high"))
    a_low = _finite_value(previous.get("low"))
    a_close = _finite_value(previous.get("close"))
    b_close = _finite_value(latest.get("close"))
    if None in (a_high, a_low, a_close, b_close):
        return None

    for kind in watch.kinds:
        level = _finite_value(levels.get(kind))
        if level is None:
            continue
        tolerance = abs(level) * float(tolerance_fraction)
        if watch.side == "short":
            tagged = a_high >= level - tolerance and a_close <= level + tolerance
            hit = tagged and b_close < a_close and b_close < level
        else:
            tagged = a_low <= level + tolerance and a_close >= level - tolerance
            hit = tagged and b_close > a_close and b_close > level
        if not hit:
            continue
        label = ANY_BOUNCE_KINDS.get(kind, kind)
        verb = "rejected from" if watch.side == "short" else "bounced off"
        return AnyBounceTrigger(
            watch=watch,
            kind=kind,
            level=level,
            price=b_close,
            triggered_at=_bar_end(latest),
            message=(
                f"{watch.symbol} {verb} the {label} at {level:.2f} "
                f"(completed 5m close {b_close:.2f})"
            ),
        )
    return None
