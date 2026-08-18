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
