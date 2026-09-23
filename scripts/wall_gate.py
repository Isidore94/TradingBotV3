"""The wall gate: do not show a chart that is sitting right under a wall.

Trader, 2026-09-23: "Don't show me stocks right at SMAs (within 1 ATR).
Instead auto-set an alert to see if it breaks the SMA or rejects it then
retests the 15 EMA. Once a stock is through the SMA it can show it to me.
Same with trendline breaks: I don't want to see it the day it breaks ... If
it's within 1 ATR (20-day) of the trendline it shouldn't show me."

This module is the decision only: no Qt, no I/O, no clock. The Alert Center
feeds it bars and trendline records it already holds and folds the answer
into its one display verdict. Answers are `focus_adoption_gate`'s three:

- ``CLOSED``  - a wall is verified in the way (the caller may hide);
- ``OPEN``    - measured, no wall;
- ``UNKNOWN`` - no price, no ATR or no level to measure. Never CLOSED.

Levels are the daily SMA 50/100/200 and the scan's D1 trendline. ATR is a
20-day Wilder ATR. Both come off COMPLETED daily bars only: a bar marked
``preview`` or dated today is the forming candle and is left out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping, Sequence

from focus_adoption_gate import CLOSED, OPEN, UNKNOWN
from indicators.atr import wilder_atr
from prev_day_gate import finite_float, is_short_side
from strength_scan import sma

#: Kill switch. False makes every answer OPEN.
WALL_GATE_ENABLED = True
#: How close (in 20-day ATRs) counts as "at the wall".
WALL_ATR_MULTIPLE = 1.0
ATR_LENGTH = 20
SMA_PERIODS = (50, 100, 200)
#: A projected trendline goes stale fast; same budget as `d1_level_feed`.
TRENDLINE_MAX_AGE_DAYS = 5

WALL_TRENDLINE = "trendline"
WALL_TRENDLINE_BREAK_DAY = "trendline_break_day"
FOLLOW_UP_SMA = "sma"
FOLLOW_UP_TRENDLINE = "trendline"


@dataclass(frozen=True)
class WallVerdict:
    state: str
    reason: str
    #: "SMA50" / "SMA100" / "SMA200" / "trendline" / "trendline_break_day", or "".
    wall: str = ""
    level: float | None = None
    distance_atr: float | None = None
    atr: float | None = None
    #: Which follow-up alerts cover this wall: "sma", "trendline" or "".
    follow_up: str = ""

    def source_text(self) -> str:
        """What an auto-armed follow-up watch says about why it exists."""
        return f"auto: {self.reason}" if self.wall else ""


def _bar_date(bar: Mapping[str, Any]) -> date | None:
    stamp = bar.get("dt") if isinstance(bar, Mapping) else None
    if isinstance(stamp, datetime):
        return stamp.date()
    if isinstance(stamp, date):
        return stamp
    return None


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()[:10]
    if not text:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def completed_daily_bars(
    d1_bars: Iterable[Mapping[str, Any]] | None, *, today: date | None = None
) -> list[Mapping[str, Any]]:
    """Daily bars with a date and a close, the forming one left out."""
    out: list[Mapping[str, Any]] = []
    for bar in d1_bars or ():
        if not isinstance(bar, Mapping) or bar.get("preview"):
            continue
        stamp = _bar_date(bar)
        if stamp is None or (today is not None and stamp >= today):
            continue
        if finite_float(bar.get("close")) is None:
            continue
        out.append(bar)
    return out


def atr20(completed: Sequence[Mapping[str, Any]]) -> float | None:
    """20-day Wilder ATR off completed daily bars, or None."""
    return wilder_atr(list(completed), ATR_LENGTH)


def sma_levels(completed: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """{"SMA50": v, ...} for every period with enough completed closes."""
    closes = [finite_float(bar.get("close")) for bar in completed]
    values = [value for value in closes if value is not None]
    out: dict[str, float] = {}
    for period in SMA_PERIODS:
        value = sma(values, period)
        if value is not None and value > 0:
            out[f"SMA{period}"] = value
    return out


def _next_weekday(day: date) -> date:
    step = day + timedelta(days=1)
    while step.weekday() >= 5:
        step += timedelta(days=1)
    return step


def trendline_value(
    candidate: Mapping[str, Any],
    completed: Sequence[Mapping[str, Any]],
    *,
    today: date,
) -> float | None:
    """The scan's trendline projected to today's session, or None.

    ``current_line_price`` is the line at ``lookback_end`` and
    ``slope_log_per_bar`` its slope in log price per daily bar, so a session
    ``i`` bars later sits at ``current_line_price * exp(slope * i)``. Today
    is one bar past the last completed bar once a new session has begun.
    None when the record cannot support a projection or is too old.
    """
    if not isinstance(candidate, Mapping) or not completed:
        return None
    slope = finite_float(candidate.get("slope_log_per_bar"))
    anchor_price = finite_float(candidate.get("current_line_price"))
    end_date = _parse_date(candidate.get("lookback_end"))
    if slope is None or anchor_price is None or anchor_price <= 0 or end_date is None:
        return None
    age = (today - end_date).days
    if age < 0 or age > TRENDLINE_MAX_AGE_DAYS:
        return None
    dates = [_bar_date(bar) for bar in completed]
    try:
        anchor_index = dates.index(end_date)
    except ValueError:
        return None
    last_index = len(completed) - 1
    target = last_index + 1 if today > dates[-1] else last_index
    exponent = slope * (target - anchor_index)
    if abs(exponent) > 50:
        return None
    value = anchor_price * math.exp(exponent)
    return value if math.isfinite(value) and value > 0 else None


def _is_break_day(
    candidate: Mapping[str, Any], completed: Sequence[Mapping[str, Any]], today: date
) -> bool:
    """The scan's break bar is today's session, or the latest completed bar
    while the next session has not started yet (the evening of the break)."""
    break_date = _parse_date(candidate.get("break_date"))
    if break_date is None:
        return False
    if break_date >= today:
        return True
    last = _bar_date(completed[-1]) if completed else None
    return last is not None and break_date == last and today < _next_weekday(last)


def _distance_atr(price: float, level: float, atr: float) -> float:
    return abs(price - level) / atr


def sma_wall(
    side: Any, price: float, levels: Mapping[str, float], atr: float, multiple: float
) -> WallVerdict | None:
    """The nearest SMA in the trade's path within ``multiple`` ATRs, or None.

    Long: an SMA at or above price. Short: at or below. An SMA behind the
    trade (a long just above its SMA) is support, not a wall.
    """
    short = is_short_side(side)
    best: WallVerdict | None = None
    for label, level in levels.items():
        in_path = level <= price if short else level >= price
        if not in_path:
            continue
        distance = _distance_atr(price, level, atr)
        if distance > multiple:
            continue
        if best is None or distance < (best.distance_atr or 0.0):
            best = WallVerdict(
                state=CLOSED,
                reason=f"{label} wall {distance:.1f} ATR",
                wall=label,
                level=level,
                distance_atr=distance,
                atr=atr,
                follow_up=FOLLOW_UP_SMA,
            )
    return best


def trendline_wall(
    price: float,
    candidates: Iterable[Mapping[str, Any]],
    completed: Sequence[Mapping[str, Any]],
    *,
    today: date,
    atr: float,
    multiple: float,
) -> WallVerdict | None:
    """A trendline within ``multiple`` ATRs on either side, or broken today."""
    best: WallVerdict | None = None
    last_close = finite_float(completed[-1].get("close")) if completed else None
    for candidate in candidates or ():
        if not isinstance(candidate, Mapping):
            continue
        value = trendline_value(candidate, completed, today=today)
        if value is None:
            continue
        distance = _distance_atr(price, value, atr)
        crossed_today = (
            last_close is not None and (last_close - value) * (price - value) < 0
        )
        if _is_break_day(candidate, completed, today) or crossed_today:
            return WallVerdict(
                state=CLOSED,
                reason="trendline broke today",
                wall=WALL_TRENDLINE_BREAK_DAY,
                level=value,
                distance_atr=distance,
                atr=atr,
                follow_up=FOLLOW_UP_TRENDLINE,
            )
        if distance > multiple:
            continue
        if best is None or distance < (best.distance_atr or 0.0):
            best = WallVerdict(
                state=CLOSED,
                reason=f"trendline wall {distance:.1f} ATR",
                wall=WALL_TRENDLINE,
                level=value,
                distance_atr=distance,
                atr=atr,
                follow_up=FOLLOW_UP_TRENDLINE,
            )
    return best


def wall_state(
    side: Any,
    price: Any,
    d1_bars: Iterable[Mapping[str, Any]] | None,
    *,
    today: date,
    trendlines: Iterable[Mapping[str, Any]] = (),
    enabled: bool | None = None,
    multiple: float | None = None,
) -> WallVerdict:
    """One verdict for one name on one side.

    A trendline break day wins; otherwise the nearest wall. With no wall:
    OPEN when at least one level was measured, UNKNOWN when none could be.
    """
    if not (WALL_GATE_ENABLED if enabled is None else enabled):
        return WallVerdict(state=OPEN, reason="wall gate off")
    side_key = str(side or "").strip().lower()
    if side_key not in ("long", "short"):
        return WallVerdict(state=UNKNOWN, reason="no side")
    last = finite_float(price)
    if last is None or last <= 0:
        return WallVerdict(state=UNKNOWN, reason="no price")
    completed = completed_daily_bars(d1_bars, today=today)
    atr = atr20(completed)
    if atr is None:
        return WallVerdict(state=UNKNOWN, reason="no 20-day ATR")
    limit = WALL_ATR_MULTIPLE if multiple is None else float(multiple)
    candidates = [item for item in trendlines or () if isinstance(item, Mapping)]
    line = trendline_wall(
        last, candidates, completed, today=today, atr=atr, multiple=limit
    )
    if line is not None and line.wall == WALL_TRENDLINE_BREAK_DAY:
        return line
    levels = sma_levels(completed)
    average = sma_wall(side_key, last, levels, atr, limit)
    walls = [item for item in (line, average) if item is not None]
    if walls:
        return min(walls, key=lambda item: item.distance_atr or 0.0)
    measured_line = any(
        trendline_value(item, completed, today=today) is not None for item in candidates
    )
    if not levels and not measured_line:
        return WallVerdict(state=UNKNOWN, reason="no SMA or trendline to measure", atr=atr)
    return WallVerdict(state=OPEN, reason="no wall within 1 ATR", atr=atr)
