"""The combined gate an automatic M5 Focus pick must pass.

Trader rule 2026-08-14: an auto M5 Focus pick must be trading **above the
previous day's high AND above session VWAP** on the M5 for longs, and below
both for shorts. The same test runs at three points, and they must never
disagree, so it lives in one place:

1. candidate build, where it replaces the previous-day-only filter;
2. every staging refresh - a queued pick that has fallen back through either
   level is evicted;
3. adoption into M5 Focus, including the drain when AWAY or EVENING flips to
   DESK (packet R1).

Plain Python - no Qt, no pandas, no project imports beyond `prev_day_gate` -
so the headless engine, the Qt Alert Center and the tests all import it
cheaply. This mirrors `prev_day_gate`'s philosophy deliberately: the two
halves of one rule should be equally cheap to reach.

The VWAP *value* is computed by the caller, not here. Callers on the
auto-populate path get it from `autopilot_core.fetch_intraday_profiles`
(`completed_session_vwap`), which runs `chart_snapshot.session_vwap_series`
over the same completed bars. Do NOT substitute BounceBot's
`calculate_dynamic_vwap`/`calculate_eod_vwap`: those blend prior sessions and
answer a different question.

Three answers, kept distinct for the same reason `prev_day_gate` keeps them:

- ``OPEN``    - verified on the right side of both levels.
- ``CLOSED``  - verified NOT to qualify.
- ``UNKNOWN`` - a level or the price could not be measured.

plan.md sec 5: missing data is uncertainty, never confirmation. UNKNOWN
therefore fails, and every price fed in here is a COMPLETED bar's close - a
forming bar is a preview, and a pick admitted on a break the bar then closes
back inside is exactly the noise this gate exists to remove.
"""

from __future__ import annotations

from typing import Any

from prev_day_gate import (
    CLOSED,
    OPEN,
    UNKNOWN,
    finite_float,
    is_short_side,
    prev_day_break_state,
)

__all__ = [
    "CLOSED",
    "OPEN",
    "UNKNOWN",
    "focus_adoption_gate_state",
    "passes_focus_adoption_gate",
    "session_vwap_state",
]


def session_vwap_state(side: Any, price: Any, vwap: Any) -> str:
    """OPEN / CLOSED / UNKNOWN for one symbol against its session VWAP."""
    last = finite_float(price)
    if last is None:
        return UNKNOWN
    level = finite_float(vwap)
    if level is None:
        return UNKNOWN
    if is_short_side(side):
        return OPEN if last < level else CLOSED
    return OPEN if last > level else CLOSED


def focus_adoption_gate_state(
    side: Any,
    price: Any,
    prev_high: Any,
    prev_low: Any,
    vwap: Any,
) -> tuple[str, str]:
    """(state, reason) for one candidate on one side.

    The reason is written for the Auto Pilot log, which is where an evicted
    pick's disappearance has to be explainable after the fact. UNKNOWN is
    reported before CLOSED when both apply: "could not measure" and "measured,
    failed" are different operational problems.
    """
    extreme = prev_day_break_state(side, price, prev_high, prev_low)
    vwap_state = session_vwap_state(side, price, vwap)
    short = is_short_side(side)
    extreme_label = "yesterday's low" if short else "yesterday's high"
    side_label = "below" if short else "above"

    if extreme == UNKNOWN and vwap_state == UNKNOWN:
        return UNKNOWN, "no completed price, prior session or session VWAP to measure"
    if extreme == UNKNOWN:
        return UNKNOWN, f"cannot verify the break of {extreme_label}"
    if vwap_state == UNKNOWN:
        return UNKNOWN, "cannot verify session VWAP"
    if extreme == CLOSED and vwap_state == CLOSED:
        return CLOSED, f"not {side_label} {extreme_label} and not {side_label} session VWAP"
    if extreme == CLOSED:
        return CLOSED, f"not {side_label} {extreme_label}"
    if vwap_state == CLOSED:
        return CLOSED, f"not {side_label} session VWAP"
    return OPEN, f"{side_label} {extreme_label} and {side_label} session VWAP"


def passes_focus_adoption_gate(
    side: Any,
    price: Any,
    prev_high: Any,
    prev_low: Any,
    vwap: Any,
) -> tuple[bool, str]:
    """(passes, reason). Only a verified OPEN passes - UNKNOWN never does."""
    state, reason = focus_adoption_gate_state(side, price, prev_high, prev_low, vwap)
    return state == OPEN, reason
