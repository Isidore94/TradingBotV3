"""Pure M5 signal engines (R5 section 3), one step below the detector.

The three indicator modules under ``scripts/indicators`` are pure maths over a
list of numbers; ``bounce_bot_lib.legacy`` is an 11k-line live detector. This
module is the seam between them: it turns *bars* into *events*, with no clock,
no I/O, no alerting and no BounceBot import, so every rule below is testable
without standing up a scanner.

Three rules are enforced here rather than at the call site, because the call
site is the place that has historically got them wrong:

1. **Completed bars only.** Every engine filters through
   :func:`completed_bars.completed_m5_bars` before it computes anything. A
   forming bar is preview (``plan.md`` sec 5) and can un-happen; an engine that
   fires on one produces an alert the chart will not agree with five minutes
   later.
2. **The indicator warms up across sessions; the *event* belongs to one.**
   Indicator series are computed over every cached completed bar so the EMA is
   warm, then crossings are reported only for bars inside the requested
   session. This is the ``_evaluate_ema8_grind`` precedent (``legacy.py``
   computes ``_ema_series`` over all bars, then slices to today) and it matters:
   restarting the series at the open would make the first ~9 bars of every day
   unanswerable exactly when the trader is watching hardest.
3. **Shorts are the mirror, taken by negating price, not by inverting the
   test.** The efficiency oscillator is deliberately clamped at zero
   (``indicators/efficiency_lrsi.py``), so a downward-efficient window reads
   LOW, never negative -- there is no "cross down through 20" that means for a
   short what "cross up through 20" means for a long. Negating the closes makes
   the short-side series measure *downward* efficiency on the same 0..100
   scale, so one code path and one set of thresholds serve both sides.

Missing data is uncertainty, never confirmation: a bar whose timestamp cannot
be read is dropped by the completed-bars helper, and a symbol with too little
history simply produces no events.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping, Sequence

from completed_bars import bar_time, completed_m5_bars
from indicators.efficiency_lrsi import (
    CROSS_LEVELS,
    EfficiencyLrsiConfig,
    compute_efficiency_lrsi,
)
from indicators.heikin_ashi import compute_heikin_ashi
from indicators.smi import SmiConfig, compute_smi

#: R5 section 8.1: one new tag family for every M5 signal engine. `d1_flag` is
#: deliberately NOT reused - folding three unproven detectors into the champion
#: D1 family makes "is this engine noisy?" unanswerable exactly when section 7
#: requires it answered, and it would lend them champion privileges they have
#: not earned. Per-engine identity rides `bounce_type`, not this tag.
#:
#: It lives here rather than in `ui/models/bounce.py` because both the detector
#: (`bounce_bot_lib.legacy`) and the UI must agree on it, and the detector
#: cannot import from the UI. One definition, imported twice.
M5_SIGNAL_TAG = "m5_signal"

LONG = "long"
SHORT = "short"


def _sign(side: str) -> float:
    """+1 for a long, -1 for a short. Anything else is a long."""
    return -1.0 if str(side or "").strip().lower() == SHORT else 1.0


def _closes(bars: Sequence[Mapping[str, Any]], side: str) -> list[float]:
    """Close prices, negated for shorts so 'up through' reads on both sides."""
    sign = _sign(side)
    out: list[float] = []
    for bar in bars:
        try:
            out.append(sign * float(bar["close"]))
        except (KeyError, TypeError, ValueError):
            # A bar without a readable close cannot be measured. Appending a
            # placeholder would silently corrupt the EMA for every later bar,
            # so the whole series is refused instead.
            return []
    return out


def _session_of(bar: Mapping[str, Any]) -> date | None:
    stamp = bar_time(bar)
    return stamp.date() if stamp is not None else None


@dataclass(frozen=True)
class LrsiCrossEvent:
    """One completed M5 bar crossing up through an LRSI level."""

    symbol: str
    side: str
    level: float
    value: float
    previous: float
    bar_index: int
    bar_time: datetime | None
    close: float

    @property
    def is_strongest(self) -> bool:
        """The 20-level crossing -- a name coming out of pure churn."""
        return self.level == min(CROSS_LEVELS)


def lrsi_cross_events(
    bars: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    side: str,
    now: datetime,
    session: date | None = None,
    levels: Sequence[float] = CROSS_LEVELS,
    config: EfficiencyLrsiConfig | None = None,
) -> tuple[LrsiCrossEvent, ...]:
    """Every LRSI level crossing on a completed M5 bar of ``session``.

    ``session`` defaults to the session of the last completed bar, which is
    what a live scan wants. Events come back in bar order; a caller firing
    alerts wants the last one, and a test wants all of them.
    """
    completed = completed_m5_bars(bars, now=now)
    if len(completed) < 2:
        return ()

    closes = _closes(completed, side)
    if not closes:
        return ()

    if session is None:
        session = _session_of(completed[-1])

    result = compute_efficiency_lrsi(closes, config)
    events: list[LrsiCrossEvent] = []
    for level in levels:
        for index in result.cross_up_indices(float(level)):
            if session is not None and _session_of(completed[index]) != session:
                continue
            value = result.values[index]
            previous = result.values[index - 1]
            if value is None or previous is None:
                continue
            events.append(
                LrsiCrossEvent(
                    symbol=str(symbol or "").strip().upper(),
                    side=SHORT if _sign(side) < 0 else LONG,
                    level=float(level),
                    value=float(value),
                    previous=float(previous),
                    bar_index=index,
                    bar_time=bar_time(completed[index]),
                    close=float(completed[index]["close"]),
                )
            )
    events.sort(key=lambda event: (event.bar_index, event.level))
    return tuple(events)


def latest_lrsi_cross(
    bars: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    side: str,
    now: datetime,
    session: date | None = None,
    levels: Sequence[float] = CROSS_LEVELS,
    config: EfficiencyLrsiConfig | None = None,
) -> LrsiCrossEvent | None:
    """The crossing on the most recently completed bar, or ``None``.

    A live scan fires on what just happened, not on everything that happened
    today -- re-emitting an older crossing every cycle is precisely the
    repetition R4 section 6.3 was built to stop. When one bar crosses two
    levels at once the STRONGER (lower) level wins, because that is the one the
    trader asked to hear about.
    """
    events = lrsi_cross_events(
        bars,
        symbol=symbol,
        side=side,
        now=now,
        session=session,
        levels=levels,
        config=config,
    )
    if not events:
        return None
    last_index = max(event.bar_index for event in events)
    completed = completed_m5_bars(bars, now=now)
    if last_index != len(completed) - 1:
        return None
    on_last = [event for event in events if event.bar_index == last_index]
    return min(on_last, key=lambda event: event.level)


# ----------------------------------------------------------------------
# R5 section 3.2 -- the HA + SMI + LRSI confluence ("strongest").
#
# Built as a PURE, STATELESS function of the session's completed bars rather
# than as the "correlator object tracking each signal's most recent firing bar
# per symbol" the spec sketched. The two are equivalent -- the most recent
# firing bar of each signal IS derivable from the bars every scan already
# holds -- and statelessness is what section 8.2 was actually worried about:
# it named "a dormant state machine that springs alive on a toggle flip
# mid-session, carrying contents no session ever exercised" as the risk. A
# function that recomputes from bars has no contents to carry, so flipping its
# toggle mid-session gives exactly the same answer as having run all morning.
# ----------------------------------------------------------------------

#: How far apart the three signals may fire and still count as one confluence,
#: measured in completed M5 bars between the FIRST and the LAST of them. The
#: trader's framing was "within 3-4 candles of each other"; 4 is the tunable
#: default and section 8.1 defers the real tuning to the desk session.
CONFLUENCE_WINDOW_BARS = 4


def _mirrored_ohlc(
    bars: Sequence[Mapping[str, Any]], side: str
) -> tuple[list[float], list[float], list[float], list[float]] | None:
    """OHLC series, mirrored for shorts so one code path serves both sides.

    Negating a candle swaps its high and low -- ``-low`` is the larger number
    -- so the mirrored series is a genuine upside-down chart, not a sign flip
    with the extremes left crossed over. Any bar that cannot be read refuses
    the whole series, for the reason ``_closes`` gives.
    """
    sign = _sign(side)
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    for bar in bars:
        try:
            raw_open = float(bar["open"])
            raw_high = float(bar["high"])
            raw_low = float(bar["low"])
            raw_close = float(bar["close"])
        except (KeyError, TypeError, ValueError):
            return None
        if sign < 0:
            opens.append(-raw_open)
            highs.append(-raw_low)
            lows.append(-raw_high)
            closes.append(-raw_close)
        else:
            opens.append(raw_open)
            highs.append(raw_high)
            lows.append(raw_low)
            closes.append(raw_close)
    return opens, highs, lows, closes


@dataclass(frozen=True)
class ConfluenceEvent:
    """One HA reversal, one SMI turn and one LRSI crossing, close together."""

    symbol: str
    side: str
    bar_index: int
    bar_time: datetime | None
    close: float
    ha_index: int
    smi_index: int
    lrsi_index: int
    lrsi_level: float
    span_bars: int

    @property
    def parts(self) -> tuple[int, int, int]:
        """The three firing bars, in the order the spec names them."""
        return (self.ha_index, self.smi_index, self.lrsi_index)


def confluence_events(
    bars: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    side: str,
    now: datetime,
    session: date | None = None,
    window_bars: int = CONFLUENCE_WINDOW_BARS,
    levels: Sequence[float] = CROSS_LEVELS,
    config: EfficiencyLrsiConfig | None = None,
    smi_config: SmiConfig | None = None,
) -> tuple[ConfluenceEvent, ...]:
    """Every completed bar that closes a three-signal confluence.

    A confluence is reported on the bar carrying the LAST of the three signals,
    which is the first moment the trader could have known about it. The other
    two must have fired no more than ``window_bars`` completed bars earlier.
    Each distinct triple reports once; a later re-firing of one leg against the
    same two others is a new triple and reports again, because on the chart it
    is a second event and suppressing it would hide information the trader is
    the one entitled to judge.
    """
    completed = completed_m5_bars(bars, now=now)
    if len(completed) < 2:
        return ()

    series = _mirrored_ohlc(completed, side)
    if series is None:
        return ()
    opens, highs, lows, closes = series

    if session is None:
        session = _session_of(completed[-1])

    ha = compute_heikin_ashi(opens, highs, lows, closes)
    ha_bars = set(ha.bullish_reversal_indices())
    smi = compute_smi(highs, lows, closes, smi_config)
    smi_bars = set(smi.bullish_cross_indices())

    lrsi = compute_efficiency_lrsi(closes, config)
    lrsi_bars: dict[int, float] = {}
    for level in levels:
        for index in lrsi.cross_up_indices(float(level)):
            # A bar crossing two levels keeps the STRONGER (lower) one, the
            # same rule latest_lrsi_cross applies.
            if index not in lrsi_bars or float(level) < lrsi_bars[index]:
                lrsi_bars[index] = float(level)

    window = max(0, int(window_bars))
    events: list[ConfluenceEvent] = []
    seen: set[tuple[int, int, int]] = set()
    for index in range(len(completed)):
        if session is not None and _session_of(completed[index]) != session:
            continue
        if not (index in ha_bars or index in smi_bars or index in lrsi_bars):
            continue
        # This bar must carry the LAST leg, so every leg is at or before it.
        floor = index - window
        ha_index = max((i for i in ha_bars if floor <= i <= index), default=None)
        smi_index = max((i for i in smi_bars if floor <= i <= index), default=None)
        lrsi_index = max((i for i in lrsi_bars if floor <= i <= index), default=None)
        if ha_index is None or smi_index is None or lrsi_index is None:
            continue
        legs = (ha_index, smi_index, lrsi_index)
        if max(legs) != index:
            continue
        if legs in seen:
            continue
        seen.add(legs)
        events.append(
            ConfluenceEvent(
                symbol=str(symbol or "").strip().upper(),
                side=SHORT if _sign(side) < 0 else LONG,
                bar_index=index,
                bar_time=bar_time(completed[index]),
                close=float(completed[index]["close"]),
                ha_index=ha_index,
                smi_index=smi_index,
                lrsi_index=lrsi_index,
                lrsi_level=lrsi_bars[lrsi_index],
                span_bars=max(legs) - min(legs),
            )
        )
    return tuple(events)


def latest_confluence(
    bars: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    side: str,
    now: datetime,
    session: date | None = None,
    window_bars: int = CONFLUENCE_WINDOW_BARS,
    levels: Sequence[float] = CROSS_LEVELS,
    config: EfficiencyLrsiConfig | None = None,
    smi_config: SmiConfig | None = None,
) -> ConfluenceEvent | None:
    """The confluence completed by the most recently completed bar, if any."""
    events = confluence_events(
        bars,
        symbol=symbol,
        side=side,
        now=now,
        session=session,
        window_bars=window_bars,
        levels=levels,
        config=config,
        smi_config=smi_config,
    )
    if not events:
        return None
    completed = completed_m5_bars(bars, now=now)
    last = len(completed) - 1
    on_last = [event for event in events if event.bar_index == last]
    if not on_last:
        return None
    # One bar can close at most one NEW triple per leg-set; if several survive,
    # the tightest span is the one that actually clustered.
    return min(on_last, key=lambda event: event.span_bars)


# ----------------------------------------------------------------------
# R5 section 3.3 -- the first-candle ORB flow.
#
# Also pure and stateless, for section 8.2's reason. The whole flow -- gap,
# first-candle extreme, LRSI pullback, re-break, LRSI recross -- is a walk over
# the session's completed bars, so a scan that starts at 11:00 sees exactly
# what a scan running since the open would have seen.
#
# Shorts mirror by negation like everything else here, so "gap up / first
# candle prints the session HOD / new session HOD" reads as "gap down / first
# candle prints the session LOD / new session LOD" without a second code path.
# ----------------------------------------------------------------------

ORB_CANDIDATE = "candidate"
ORB_NEW_EXTREME = "new_extreme"
ORB_LRSI_RECROSS = "lrsi_recross"

#: The pullback that arms the follow-ups. The spec says "an LRSI pullback
#: (below 50/20)"; 50 is the arming level and a dip below 20 is recorded on the
#: event as `deep`, because "it went dead first" is the trader's own
#: distinction and it costs nothing to carry.
ORB_PULLBACK_LEVEL = 50.0
ORB_DEEP_PULLBACK_LEVEL = 20.0


@dataclass(frozen=True)
class OrbEvent:
    """One step of the first-candle ORB flow on a completed M5 bar."""

    symbol: str
    side: str
    kind: str
    bar_index: int
    bar_time: datetime | None
    close: float
    first_extreme: float
    gap_from: float
    level: float | None = None
    deep: bool = False

    @property
    def is_informational(self) -> bool:
        """The LRSI recross is information, not a break. Section 3.3."""
        return self.kind == ORB_LRSI_RECROSS


def orb_events(
    bars: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    side: str,
    now: datetime,
    session: date | None = None,
    prior_close: float | None = None,
    config: EfficiencyLrsiConfig | None = None,
) -> tuple[OrbEvent, ...]:
    """The first-candle ORB flow for one symbol and session, in bar order.

    Returns, at most and in this order: the ``candidate`` mark (a gap whose
    first completed candle sets the session extreme), then -- only after the
    LRSI has pulled back below 50 -- a ``new_extreme`` when a later completed
    bar takes out that first candle's extreme, and an informational
    ``lrsi_recross`` when the LRSI crosses back up through 50.

    ``prior_close`` defaults to the close of the last completed bar BEFORE the
    session, which is what a live scan has cached. With no prior close there is
    no gap to measure, so nothing is returned: missing data is uncertainty, not
    a gap of zero.
    """
    completed = completed_m5_bars(bars, now=now)
    if not completed:
        return ()

    series = _mirrored_ohlc(completed, side)
    if series is None:
        return ()
    opens, highs, lows, closes = series
    sign = _sign(side)

    if session is None:
        session = _session_of(completed[-1])
    if session is None:
        return ()

    session_indices = [
        index for index in range(len(completed)) if _session_of(completed[index]) == session
    ]
    if not session_indices:
        return ()
    first = session_indices[0]

    if prior_close is None:
        earlier = [index for index in range(first) if _session_of(completed[index]) is not None]
        if not earlier:
            return ()
        mirrored_prior = closes[earlier[-1]]
    else:
        mirrored_prior = sign * float(prior_close)

    # The gap itself. A flat or adverse open is not this setup.
    if not opens[first] > mirrored_prior:
        return ()

    first_extreme = highs[first]
    lrsi = compute_efficiency_lrsi(closes, config)

    events: list[OrbEvent] = []

    def _event(kind, index, level=None, deep=False):
        return OrbEvent(
            symbol=str(symbol or "").strip().upper(),
            side=SHORT if sign < 0 else LONG,
            kind=kind,
            bar_index=index,
            bar_time=bar_time(completed[index]),
            close=float(completed[index]["close"]),
            # Reported on the trader's chart scale, not the mirrored one.
            first_extreme=sign * first_extreme,
            gap_from=sign * mirrored_prior,
            level=level,
            deep=deep,
        )

    events.append(_event(ORB_CANDIDATE, first))

    armed_at: int | None = None
    deep = False
    running_extreme = first_extreme
    broke_out = False
    recrossed = False
    for index in session_indices[1:]:
        value = lrsi.values[index]
        if armed_at is None:
            # Arming needs a MEASURED pullback. An unmeasurable bar (warm-up,
            # or a gap in the series) arms nothing.
            if value is not None and value < ORB_PULLBACK_LEVEL:
                armed_at = index
                deep = value < ORB_DEEP_PULLBACK_LEVEL
            running_extreme = max(running_extreme, highs[index])
            continue
        if value is not None and value < ORB_DEEP_PULLBACK_LEVEL:
            deep = True
        if not broke_out and highs[index] > running_extreme:
            # A new session extreme, which after the pullback is the re-break
            # the trader armed for. Once only: the second higher bar of the
            # same push is the move, not a new signal.
            broke_out = True
            events.append(_event(ORB_NEW_EXTREME, index, level=sign * highs[index], deep=deep))
        if not recrossed and value is not None:
            previous = lrsi.values[index - 1]
            if previous is not None and previous <= ORB_PULLBACK_LEVEL < value:
                recrossed = True
                events.append(
                    _event(ORB_LRSI_RECROSS, index, level=ORB_PULLBACK_LEVEL, deep=deep)
                )
        running_extreme = max(running_extreme, highs[index])

    events.sort(key=lambda event: (event.bar_index, event.kind))
    return tuple(events)


def latest_orb_events(
    bars: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    side: str,
    now: datetime,
    session: date | None = None,
    prior_close: float | None = None,
    config: EfficiencyLrsiConfig | None = None,
) -> tuple[OrbEvent, ...]:
    """Only the flow steps that landed on the most recently completed bar."""
    events = orb_events(
        bars,
        symbol=symbol,
        side=side,
        now=now,
        session=session,
        prior_close=prior_close,
        config=config,
    )
    if not events:
        return ()
    last = len(completed_m5_bars(bars, now=now)) - 1
    return tuple(event for event in events if event.bar_index == last)
