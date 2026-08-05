"""Previous-session extreme gate: longs above yesterday's high, shorts below
yesterday's low.

Trader rule 2026-07-31, first written for Auto Pilot's auto-populate picks
(`autopilot_core.passes_prev_day_extreme_gate`), extended to Focus-pick
flagging 2026-08-05: "I don't want focus picks to flag if they are below the
previous day high for longs, or above the previous day low for shorts -
otherwise it's just noise." Both callers share this one definition of the
break so the desk and the headless engine can never disagree about it.

Plain Python (no Qt, no pandas, no project imports) so the Qt Alert Center,
`autopilot_core`, and tests all import it cheaply.

Three answers, and callers must keep them distinct:

- ``OPEN``    - the break is verified against a completed prior session.
- ``CLOSED``  - verified NOT broken; the name is inside yesterday's range.
- ``UNKNOWN`` - no price, or no completed prior session to measure against.

plan.md sec 5: missing data is uncertainty, never confirmation. UNKNOWN is
therefore never reported as a break - `passes_prev_day_extreme_gate` answers
False for it - but a caller that wants to distinguish "I checked and it has
not broken out" from "I could not check" reads the state directly.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Iterable, Mapping


OPEN = "open"
CLOSED = "closed"
UNKNOWN = "unknown"


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def is_short_side(side: Any) -> bool:
    return str(side or "").strip().lower().startswith("short")


def _naive(moment: datetime) -> datetime:
    # Same convention as chart_watch: IB serves this desk's bars on the local
    # clock (sometimes tz-stamped), so comparisons drop tzinfo rather than
    # convert across zones.
    return moment.replace(tzinfo=None) if moment.tzinfo is not None else moment


def prev_day_break_state(
    side: Any,
    price: Any,
    prev_high: Any,
    prev_low: Any,
) -> str:
    """OPEN / CLOSED / UNKNOWN for one symbol on one side."""
    last = finite_float(price)
    if last is None:
        return UNKNOWN
    if is_short_side(side):
        level = finite_float(prev_low)
        if level is None:
            return UNKNOWN
        return OPEN if last < level else CLOSED
    level = finite_float(prev_high)
    if level is None:
        return UNKNOWN
    return OPEN if last > level else CLOSED


def passes_prev_day_extreme_gate(
    side: Any,
    price: Any,
    prev_high: Any,
    prev_low: Any,
) -> bool:
    """True only when the break is verified (UNKNOWN fails - see module docs)."""
    return prev_day_break_state(side, price, prev_high, prev_low) == OPEN


def prev_session_extremes(
    d1_bars: Iterable[Mapping[str, Any]] | None,
    *,
    session: date | None = None,
) -> tuple[float | None, float | None]:
    """(high, low) of the last COMPLETED daily session before ``session``.

    Daily stores routinely carry today's forming bar; measuring against it
    would compare the day to itself, so bars dated on or after ``session``
    are dropped. Returns (None, None) when there is no prior session.
    """
    cutoff = session or date.today()
    latest_stamp: datetime | None = None
    latest_bar: Mapping[str, Any] | None = None
    for bar in d1_bars or []:
        stamp = bar.get("dt") if isinstance(bar, Mapping) else None
        if not isinstance(stamp, datetime):
            continue
        stamp = _naive(stamp)
        if stamp.date() >= cutoff:
            continue
        if latest_stamp is None or stamp > latest_stamp:
            latest_stamp = stamp
            latest_bar = bar
    if latest_bar is None:
        return None, None
    return finite_float(latest_bar.get("high")), finite_float(latest_bar.get("low"))
