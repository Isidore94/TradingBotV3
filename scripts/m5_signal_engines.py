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
from datetime import date, datetime, timedelta
from typing import Any, Callable, Mapping, Sequence

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


# ----------------------------------------------------------------------
# S7 (2026-09-26): four SHADOW setup engines for the setups the desk has no
# detector for. SHADOW ONLY: their events go to the `m5_shadow_setups` sidecar
# and nowhere else - no alert, no score, no Show, no phone - and nothing live
# reads them. Graduation only through the `docs/SETUPS_TEST.md` ladder.
#
# Same rules as above: completed bars only, shorts by mirroring, missing data
# is no event. They read the REGULAR session (09:30-16:00 ET) only; bars are
# naive market-local (``tz``) unless they carry a zone. Each event carries what
# the S3 bracket needs: event_id, side, level, entry, stop, bar time with tz.
# ----------------------------------------------------------------------

SHADOW_PD_BREAK_HOLD = "pd_level_break_hold"
SHADOW_VWAP_RECLAIM = "vwap_reclaim_after_flush"
SHADOW_COMPRESSION_BREAK = "m5_compression_break"
SHADOW_TRENDLINE_BREAK = "trendline_break"
SHADOW_ENGINES = (
    SHADOW_PD_BREAK_HOLD,
    SHADOW_VWAP_RECLAIM,
    SHADOW_COMPRESSION_BREAK,
    SHADOW_TRENDLINE_BREAK,
)

EXCHANGE_TIMEZONE = "America/New_York"
_M5_MINUTES = 5
_REGULAR_OPEN = 9 * 60 + 30
_REGULAR_CLOSE = 16 * 60
_FULL_SESSION_SLOTS = tuple(range(_REGULAR_OPEN, _REGULAR_CLOSE, _M5_MINUTES))

#: (a) The event bar must start at or after 10:00 ET, with session RVOL at least this.
PD_BREAK_EARLIEST = 10 * 60
PD_BREAK_RVOL_MIN = 1.5
#: RVOL = today's volume through the bar over the mean of the same span in up to this
#: many earlier sessions; fewer than ``RVOL_MIN_PRIOR_SESSIONS`` usable ones is unknown.
RVOL_LOOKBACK_SESSIONS = 5
RVOL_MIN_PRIOR_SESSIONS = 2
#: (b) The first 30 minutes (six M5 bars) and the environment longs need.
FLUSH_BARS = 6
VWAP_RECLAIM_ENVIRONMENT = "bullish_strong"
#: (c) The S6 squeeze: 12 bars no wider than 2.5 M5 ATR20 (`setup_permutations.M5_SQUEEZE_RANGE_ATR`).
SQUEEZE_BOX_BARS = 12
SQUEEZE_ATR_BARS = 20
SQUEEZE_RANGE_ATR = 2.5
#: (d) A pivot high is higher than this many bars on each side.
PIVOT_SPAN = 2


@dataclass(frozen=True)
class ShadowSetupEvent:
    """One shadow setup on a completed M5 bar, on the trader's chart scale."""

    engine: str
    symbol: str
    side: str
    bar_index: int
    bar_time: datetime  # the bar's START, zone-aware (exchange time)
    level: float
    entry: float
    stop: float
    details: tuple[tuple[str, Any], ...] = ()

    @property
    def bar_close(self) -> datetime:
        return self.bar_time + timedelta(minutes=_M5_MINUTES)

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def event_id(self) -> str:
        return f"s7:{self.engine}:{self.symbol}:{self.side}:{self.bar_time.isoformat()}"


@dataclass(frozen=True)
class _RegularBar:
    index: int  # into the completed bars
    start: datetime  # exchange time
    day: date
    minutes: int
    open: float  # OHLC mirrored for shorts
    high: float
    low: float
    close: float
    volume: float | None


def _exchange_tz():
    from zoneinfo import ZoneInfo

    return ZoneInfo(EXCHANGE_TIMEZONE)


def _default_local_tz():
    from market_session import get_market_local_timezone

    return get_market_local_timezone()[0]


def _bar_volume(bar: Any) -> float | None:
    raw = bar.get("volume") if isinstance(bar, Mapping) else getattr(bar, "volume", None)
    try:
        volume = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
    if volume is None or volume != volume or volume < 0:
        return None
    return volume


def _regular_bars(
    bars: Sequence[Mapping[str, Any]], side: str, *, now: datetime, tz
) -> list[_RegularBar] | None:
    """Completed regular-session bars, mirrored for shorts; None when any bar is unreadable."""
    completed = completed_m5_bars(bars, now=now)
    series = _mirrored_ohlc(completed, side)
    if series is None:
        return None
    opens, highs, lows, closes = series
    local = tz if tz is not None else _default_local_tz()
    exchange = _exchange_tz()
    out: list[_RegularBar] = []
    for index, bar in enumerate(completed):
        stamp = bar_time(bar)
        if stamp is None:
            continue
        start = (stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=local)).astimezone(exchange)
        minutes = start.hour * 60 + start.minute
        if not _REGULAR_OPEN <= minutes < _REGULAR_CLOSE:
            continue
        out.append(_RegularBar(index, start, start.date(), minutes, opens[index], highs[index],
                               lows[index], closes[index], _bar_volume(bar)))
    return out


def _split_session(regular: list[_RegularBar], session: date | None):
    """(session day, that day's bars) - the last regular day when ``session`` is None."""
    if not regular:
        return None, []
    day = session if session is not None else regular[-1].day
    return day, [bar for bar in regular if bar.day == day]


def _event(engine, symbol, side, bar: _RegularBar, *, level, entry, stop, **details):
    sign = _sign(side)
    return ShadowSetupEvent(
        engine=engine,
        symbol=str(symbol or "").strip().upper(),
        side=SHORT if sign < 0 else LONG,
        bar_index=bar.index,
        bar_time=bar.start,
        level=round(sign * level, 6),
        entry=round(sign * entry, 6),
        stop=round(sign * stop, 6),
        details=tuple(sorted(details.items())),
    )


def session_rvol(regular: list[_RegularBar], day: date, position: int) -> tuple[float, int] | None:
    """(RVOL, prior sessions used) through ``day``'s bar at ``position``; None when unknown.

    Today's volume from the open through that bar over the mean of the same span in
    the last ``RVOL_LOOKBACK_SESSIONS`` earlier sessions that hold every bar of it.
    """
    today = [bar for bar in regular if bar.day == day]
    through = today[: position + 1]
    slots = tuple(range(_REGULAR_OPEN, through[-1].minutes + _M5_MINUTES, _M5_MINUTES))
    if tuple(bar.minutes for bar in through) != slots or any(bar.volume is None for bar in through):
        return None
    earlier = sorted({bar.day for bar in regular if bar.day < day})
    totals: list[float] = []
    for prior in reversed(earlier):
        span = [bar for bar in regular if bar.day == prior and bar.minutes <= slots[-1]]
        if tuple(bar.minutes for bar in span) != slots or any(bar.volume is None for bar in span):
            continue
        totals.append(sum(bar.volume for bar in span))
        if len(totals) == RVOL_LOOKBACK_SESSIONS:
            break
    if len(totals) < RVOL_MIN_PRIOR_SESSIONS:
        return None
    baseline = sum(totals) / len(totals)
    if baseline <= 0:
        return None
    return sum(bar.volume for bar in through) / baseline, len(totals)


def pd_level_break_hold_events(
    bars: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    side: str,
    now: datetime,
    session: date | None = None,
    tz=None,
    rvol_min: float = PD_BREAK_RVOL_MIN,
) -> tuple[ShadowSetupEvent, ...]:
    """(a) Previous-day high (long) / low (short) break-and-hold, RVOL >= 1.5, after 10:00 ET.

    The break bar is the first close beyond the level after a close (or the day's
    open) at or inside it; the hold bar is the next bar, closing beyond it too. The
    event is on the hold bar, which must start at 10:00 ET or later with session RVOL
    at least ``rvol_min``. A failed hold re-arms. First event of the session only.
    The previous session must be a full 78-bar day. Stop: the lower low of the two bars.
    """
    regular = _regular_bars(bars, side, now=now, tz=tz)
    if regular is None:
        return ()
    day, today = _split_session(regular, session)
    if day is None or len(today) < 2:
        return ()
    earlier = sorted({bar.day for bar in regular if bar.day < day})
    if not earlier:
        return ()
    previous = [bar for bar in regular if bar.day == earlier[-1]]
    if tuple(bar.minutes for bar in previous) != _FULL_SESSION_SLOTS:
        return ()
    level = max(bar.high for bar in previous)
    for position in range(1, len(today)):
        hold, broke = today[position], today[position - 1]
        before = today[position - 2].close if position >= 2 else broke.open
        if not (before <= level < broke.close and hold.close > level):
            continue
        if hold.minutes < PD_BREAK_EARLIEST:
            continue
        rvol = session_rvol(regular, day, position)
        if rvol is None or rvol[0] < rvol_min:
            continue
        stop = min(broke.low, hold.low)
        if not stop < hold.close:
            continue
        return (
            _event(SHADOW_PD_BREAK_HOLD, symbol, side, hold, level=level, entry=hold.close, stop=stop,
                   rvol=round(rvol[0], 4), rvol_sessions=rvol[1], break_bar=broke.start.isoformat()),
        )
    return ()


def _running_vwap(today: list[_RegularBar]) -> list[float] | None:
    """Session VWAP (typical price) at each bar; None when any volume is missing.

    On a mirrored series the typical price is negated, so the VWAP mirrors with it.
    """
    cum_volume = cum_value = 0.0
    out: list[float] = []
    for bar in today:
        if bar.volume is None:
            return None
        cum_volume += bar.volume
        cum_value += (bar.high + bar.low + bar.close) / 3.0 * bar.volume
        if cum_volume <= 0:
            return None
        out.append(cum_value / cum_volume)
    return out


def vwap_reclaim_after_flush_events(
    bars: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    side: str,
    now: datetime,
    session: date | None = None,
    tz=None,
    environment_at: Callable[[datetime], str | None] | None = None,
) -> tuple[ShadowSetupEvent, ...]:
    """(b) VWAP reclaim after a first-30 flush; longs only, only in ``bullish_strong``.

    Flush: all six first-30 bars present, their low under the day's open, and the
    09:55 bar closing under session VWAP. Reclaim: the first bar from 10:00 ET that
    closes back over VWAP. ``environment_at(bar start)`` must say ``bullish_strong``
    at the reclaim bar; no reader or an unknown label is no event. Stop: the
    session low through the reclaim bar.
    """
    if _sign(side) < 0 or environment_at is None:
        return ()
    regular = _regular_bars(bars, side, now=now, tz=tz)
    if regular is None:
        return ()
    day, today = _split_session(regular, session)
    if day is None or len(today) <= FLUSH_BARS:
        return ()
    first30 = today[:FLUSH_BARS]
    if tuple(bar.minutes for bar in first30) != _FULL_SESSION_SLOTS[:FLUSH_BARS]:
        return ()
    vwap = _running_vwap(today)
    if vwap is None:
        return ()
    flush_low = min(bar.low for bar in first30)
    if not (flush_low < first30[0].open and first30[-1].close < vwap[FLUSH_BARS - 1]):
        return ()
    for position in range(FLUSH_BARS, len(today)):
        bar = today[position]
        if bar.close <= vwap[position]:
            continue
        try:
            label = environment_at(bar.start)
        except Exception:  # noqa: BLE001 - an unreadable environment is unknown
            label = None
        if str(label or "").strip().lower() != VWAP_RECLAIM_ENVIRONMENT:
            return ()
        stop = min(item.low for item in today[: position + 1])
        if not stop < bar.close:
            return ()
        return (
            _event(SHADOW_VWAP_RECLAIM, symbol, side, bar, level=vwap[position], entry=bar.close,
                   stop=stop, environment=VWAP_RECLAIM_ENVIRONMENT, flush_low=round(flush_low, 6)),
        )
    return ()


def compression_break_events(
    bars: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    side: str,
    now: datetime,
    session: date | None = None,
    tz=None,
) -> tuple[ShadowSetupEvent, ...]:
    """(c) M5 compression break: a close beyond a 12-bar squeeze box.

    The box is the 12 same-session bars before the bar, no wider than 2.5 M5 ATR20
    (ATR over the regular-session bars before it; 21 needed). One event per box: a
    break starts a 12-bar cooldown. Level: the box edge; stop: the far edge.
    """
    regular = _regular_bars(bars, side, now=now, tz=tz)
    if regular is None:
        return ()
    day, _today = _split_session(regular, session)
    if day is None:
        return ()
    # True range of bar i against bar i-1, computed once (index 0 is unused).
    true_range = [0.0] + [
        max(item.high - item.low, abs(item.high - regular[i - 1].close), abs(item.low - regular[i - 1].close))
        for i, item in enumerate(regular) if i > 0
    ]
    events: list[ShadowSetupEvent] = []
    last_event: int | None = None
    for k, bar in enumerate(regular):
        if bar.day != day or k < max(SQUEEZE_ATR_BARS + 1, SQUEEZE_BOX_BARS):
            continue
        if last_event is not None and k - last_event <= SQUEEZE_BOX_BARS:
            continue
        box = regular[k - SQUEEZE_BOX_BARS:k]
        if any(item.day != day for item in box):
            continue
        atr = sum(true_range[k - SQUEEZE_ATR_BARS:k]) / SQUEEZE_ATR_BARS
        box_high = max(item.high for item in box)
        box_low = min(item.low for item in box)
        if atr <= 0 or (box_high - box_low) / atr > SQUEEZE_RANGE_ATR:
            continue
        if bar.close <= box_high:
            continue
        last_event = k
        events.append(
            _event(SHADOW_COMPRESSION_BREAK, symbol, side, bar, level=box_high, entry=bar.close,
                   stop=box_low, range_atr=round((box_high - box_low) / atr, 4))
        )
    return tuple(events)


def trendline_break_events(
    bars: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    side: str,
    now: datetime,
    session: date | None = None,
    tz=None,
    pivot_span: int = PIVOT_SPAN,
) -> tuple[ShadowSetupEvent, ...]:
    """(d) Intraday trendline break from pivots (long: a falling line through pivot highs).

    A pivot high is higher than ``pivot_span`` bars on each side and is known only once
    those later bars have completed. The line runs through the last two known pivots
    of the session when the second is lower. The event is the first close over the
    line after a close at or under it; once per line. Stop: the lowest low since the
    second pivot. Shorts mirror: a rising line through pivot lows, broken down.
    """
    regular = _regular_bars(bars, side, now=now, tz=tz)
    if regular is None:
        return ()
    day, today = _split_session(regular, session)
    span = max(1, int(pivot_span))
    if day is None or len(today) < 2 * span + 3:
        return ()
    pivots = [
        p for p in range(span, len(today) - span)
        if all(today[p].high > today[q].high for q in range(p - span, p + span + 1) if q != p)
    ]
    sign = _sign(side)
    events: list[ShadowSetupEvent] = []
    used: set[tuple[int, int]] = set()
    for j in range(1, len(today)):
        known = [p for p in pivots if p + span <= j - 1]
        if len(known) < 2:
            continue
        p1, p2 = known[-2], known[-1]
        if not today[p2].high < today[p1].high or (p1, p2) in used:
            continue
        slope = (today[p2].high - today[p1].high) / (p2 - p1)
        line_now = today[p2].high + slope * (j - p2)
        line_before = today[p2].high + slope * (j - 1 - p2)
        if not (today[j].close > line_now and today[j - 1].close <= line_before):
            continue
        stop = min(item.low for item in today[p2: j + 1])
        if not stop < today[j].close:
            continue
        used.add((p1, p2))
        events.append(
            _event(SHADOW_TRENDLINE_BREAK, symbol, side, today[j], level=line_now, entry=today[j].close,
                   stop=stop, pivot_1=today[p1].start.isoformat(), pivot_1_price=round(sign * today[p1].high, 6),
                   pivot_2=today[p2].start.isoformat(), pivot_2_price=round(sign * today[p2].high, 6))
        )
    return tuple(events)


def shadow_setup_events(
    bars: Sequence[Mapping[str, Any]],
    *,
    symbol: str,
    now: datetime,
    session: date | None = None,
    tz=None,
    environment_at: Callable[[datetime], str | None] | None = None,
) -> tuple[ShadowSetupEvent, ...]:
    """All four shadow engines, both sides, in bar order. Shadow only (see above)."""
    events: list[ShadowSetupEvent] = []
    for side in (LONG, SHORT):
        common = {"symbol": symbol, "side": side, "now": now, "session": session, "tz": tz}
        events.extend(pd_level_break_hold_events(bars, **common))
        events.extend(vwap_reclaim_after_flush_events(bars, environment_at=environment_at, **common))
        events.extend(compression_break_events(bars, **common))
        events.extend(trendline_break_events(bars, **common))
    events.sort(key=lambda event: (event.bar_time, event.engine, event.side))
    return tuple(events)
