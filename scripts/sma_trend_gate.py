"""Longs above the SMA200, shorts below the SMA50 - the D1 trend leg (trader rule 3, 2026-08-27).

The chart it came out of: MUFG, a D1 "short - zone-1 reject at the earnings
AVWAP" from the swing scanner, sitting above its SMA50, SMA100 and SMA200 in a
clean uptrend. "It's above all the SMAs and clearly up trending." The scanner
looked at one line and never at the trend. The trader's floor: **a long must
be above the 200-day SMA, a short below the 50-day SMA, at least.**

This module is the whole decision and nothing else. It reads nothing, knows no
clock, and returns the same three answers the other review legs return
(`focus_adoption_gate.OPEN / CLOSED / UNKNOWN`), so the Alert Center can fold
it into the one display verdict it already keeps. UNKNOWN - not enough
history, no price, an unmeasurable average - is never folded into CLOSED:
missing data is uncertainty, and the chart SHOWS, tagged.

`trend_levels` is the extraction: the two averages off COMPLETED daily bars.
A forming candle (a bar marked ``preview``, or one dated today while today is
still trading) is excluded, because an average that moves with every tick is
a preview of an average, and a preview must never be the thing that hides a
chart. A row with fewer than 200 completed closes has no SMA200 - not "as
many as we have"; `strength_scan.sma` already refuses that, for the same
reason it gives there.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping, Sequence

from prev_day_gate import CLOSED, OPEN, UNKNOWN, finite_float, is_short_side
from strength_scan import sma

#: What the trader asked for, and nothing tighter.
LONG_SMA_PERIOD = 200
SHORT_SMA_PERIOD = 50


def _bar_date(bar: Mapping[str, Any]) -> date | None:
    stamp = bar.get("dt") if isinstance(bar, Mapping) else None
    if isinstance(stamp, datetime):
        return stamp.date()
    if isinstance(stamp, date):
        return stamp
    return None


def completed_closes(
    d1_bars: Sequence[Mapping[str, Any]], *, today: date | None = None
) -> list[float]:
    """Closes of the completed daily bars, in order; the forming one is left out."""
    closes: list[float] = []
    for bar in d1_bars or ():
        if not isinstance(bar, Mapping) or bar.get("preview"):
            continue
        if today is not None and _bar_date(bar) == today:
            continue
        close = finite_float(bar.get("close"))
        if close is None:
            continue
        closes.append(close)
    return closes


def trend_levels(
    d1_bars: Sequence[Mapping[str, Any]], *, today: date | None = None
) -> tuple[float | None, float | None]:
    """(sma50, sma200) off completed daily closes, each None when short of history."""
    closes = completed_closes(d1_bars, today=today)
    return sma(closes, SHORT_SMA_PERIOD), sma(closes, LONG_SMA_PERIOD)


def sma_trend_state(side: Any, price: Any, sma50: Any, sma200: Any) -> tuple[str, str]:
    """(state, reason) for the trend leg alone.

    A long needs ``price > sma200``; a short needs ``price < sma50``. The
    other average is not consulted - the trader said "at least", and a
    tighter rule is a different decision. UNKNOWN when the price or the
    average that side needs is missing.
    """
    last = finite_float(price)
    if is_short_side(side):
        level = finite_float(sma50)
        if last is None or level is None:
            return UNKNOWN, "cannot verify the SMA50"
        if last < level:
            return OPEN, "below the SMA50"
        return CLOSED, "not below the SMA50"
    level = finite_float(sma200)
    if last is None or level is None:
        return UNKNOWN, "cannot verify the SMA200"
    if last > level:
        return OPEN, "above the SMA200"
    return CLOSED, "not above the SMA200"
