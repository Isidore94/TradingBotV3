"""Packet ST3 - the versioned execution convention for the tracker replay.

The Setup Tracker replay books a fill at the LITERAL level whenever a daily bar
touches it. On a bar that gapped clean through the level, the "fill" is a price
the market never printed: entry 100, risk 5, hard stop 95, next bar
O80/H85/L79/C82 books the stop at 95 for -1.014R when the honest fill is the
open at 80 for -4.014R. That is not a rounding difference; it is the tail of the
distribution being deleted.

This module declares the convention as a NAMED, VERSIONED policy so the repair
is opt-in and the shipped behaviour keeps its own name:

``EXECUTION_LITERAL_LEVEL_V1`` (``"literal_level_v1"``, the DEFAULT)
    Today's behaviour exactly. A touched level fills at the level. Nothing in
    this module changes it; ``resolve_fill`` is simply not consulted.

``EXECUTION_GAP_AWARE_V2`` (``"gap_aware_v2"``)
    The repaired convention. Four rules, and no rule ever pretends that daily
    OHLC reveals the intrabar sequence:

    1. **Stop gap.** The open is known and is already through the stop (long:
       ``open < stop``; short: ``open > stop``) -> the fill is the OPEN, basis
       ``gap_open``. A protective stop is a market order once the level trades
       through it, and the first price available is the open.
    2. **Target gap.** The open is known and is already through the target
       (long: ``open > target``; short: ``open < target``) -> the fill is the
       OPEN, basis ``gap_open``. A resting limit order that opens through its
       price fills at the open, which is BETTER than the level; the convention
       is symmetric and is not a one-sided pessimism.
    3. **Missing open.** No ``open``, or a NaN one -> nothing can be said about
       a gap, so the level is CLAMPED into ``[low, high]`` and the basis is
       ``clamped_no_open``. A long stop above the bar's high fills at the high;
       a short stop below the low fills at the low. The clamp is what keeps the
       simulation inside prices that existed.
    4. **Invalid OHLC.** ``low <= open, close <= high`` broken, or a NaN among
       high/low/close -> the bar books NOTHING (``booked=False``, basis
       ``invalid_bar``). A candle whose own four prices contradict each other
       cannot answer "was the stop hit"; the honest answer is "unknown", not a
       fill. The HOLD CLOCK STILL ADVANCES, so the maximum-hold force close is
       preserved: an unusable bar delays the time stop to the next usable bar
       (basis ``deferred_invalid_bar``), it never cancels it.

    Under v2 the fill price is ALWAYS inside ``[low, high]``. The function
    asserts it, and clamps rather than trusting the assertion.

Level knowledge is a SEPARATE, independently versioned axis, declared here so
the two policies a replay runs under are named in one place:

``LEVEL_KNOWLEDGE_SAME_SESSION_V1`` (``"same_session_v1"``, the DEFAULT)
    Today's behaviour: a bar's own high/low are tested against
    ``band_history[<that same day>]``, whose anchored-VWAP bands were computed
    by folding that same day's bar into the cumulative sums. The level is known
    only at the day's CLOSE, so an intrabar test against it is look-ahead.

``LEVEL_KNOWLEDGE_PRIOR_SESSION_V2`` (``"prior_session_v2"``)
    The levels handed to the INTRABAR tests for bar D are the LAST SESSION's
    strictly before D. CLOSE-based decisions (the two-closes protective stop,
    the maximum-hold force close) keep day D's levels, because at the close
    day D's levels are known. A bar with no prior session level skips its
    intrabar test and the reason ``no_prior_session_level`` is counted on the
    scenario - never silently treated as "not hit" without a count.

**Nothing here is authorization.** ``gap_aware_v2`` / ``prior_session_v2`` are
shadow evidence, produced on copies by ``scripts/tracker_execution_compare.py``.
The champion's scoring convention stays ``literal_level_v1`` /
``same_session_v1`` until the trader decides otherwise, and
``calc_anchored_vwap_bands`` / ``calc_anchored_vwap_band_history`` are frozen
(decision 0008) - this module changes WHICH DAY's levels a bar is tested
against and WHAT PRICE a touch books, never one line of the formula.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Mapping

#: Today's convention: a touched level fills at the level. The default.
EXECUTION_LITERAL_LEVEL_V1 = "literal_level_v1"
#: The repaired convention: gaps, missing opens and invalid candles are honest.
EXECUTION_GAP_AWARE_V2 = "gap_aware_v2"
#: What every production caller gets when it says nothing.
DEFAULT_EXECUTION_CONVENTION = EXECUTION_LITERAL_LEVEL_V1

EXECUTION_CONVENTIONS = (EXECUTION_LITERAL_LEVEL_V1, EXECUTION_GAP_AWARE_V2)

#: Today's level knowledge: bar D is tested against day D's own bands.
LEVEL_KNOWLEDGE_SAME_SESSION_V1 = "same_session_v1"
#: The repaired level knowledge: intrabar tests read the prior session.
LEVEL_KNOWLEDGE_PRIOR_SESSION_V2 = "prior_session_v2"
#: What every production caller gets when it says nothing.
DEFAULT_LEVEL_KNOWLEDGE = LEVEL_KNOWLEDGE_SAME_SESSION_V1

LEVEL_KNOWLEDGE_POLICIES = (
    LEVEL_KNOWLEDGE_SAME_SESSION_V1,
    LEVEL_KNOWLEDGE_PRIOR_SESSION_V2,
)

#: The reason counted on a scenario when a prior-session level does not exist.
NO_PRIOR_SESSION_LEVEL = "no_prior_session_level"

# --- fill bases -------------------------------------------------------------
#: The level was inside the bar and the open did not gap through it.
FILL_BASIS_LEVEL = "level"
#: The bar opened already through the level; the open is the first fillable price.
FILL_BASIS_GAP_OPEN = "gap_open"
#: No usable open, so the level was clamped into the bar's own range.
FILL_BASIS_CLAMPED_NO_OPEN = "clamped_no_open"
#: Defensive: the open was known and did not gap, yet the level sat outside the
#: bar. Unreachable from the replay (the hit test guarantees otherwise), kept so
#: an out-of-range price can never be booked by a future caller.
FILL_BASIS_CLAMPED_LEVEL = "clamped_level"
#: The candle contradicts itself; nothing is booked from it.
FILL_BASIS_INVALID_BAR = "invalid_bar"
#: No level to test against; nothing is booked.
FILL_BASIS_NO_LEVEL = "no_level"
#: An end-of-bar decision (two-closes stop, maximum-hold force close).
FILL_BASIS_CLOSE = "close"
#: A maximum-hold force close that an invalid bar pushed to the next usable bar.
FILL_BASIS_DEFERRED_INVALID_BAR = "deferred_invalid_bar"

FILL_KIND_STOP = "stop"
FILL_KIND_TARGET = "target"


@dataclasses.dataclass(frozen=True)
class Fill:
    """One resolved exit price.

    ``booked`` False means the bar cannot answer the question; the caller books
    no event and leaves the scenario open. ``price`` is NaN in that case and is
    never read.
    """

    price: float
    basis: str
    booked: bool


def _finite(value: Any) -> float | None:
    """``value`` as a finite float, or None for anything unusable."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _bar_value(bar: Mapping[str, Any] | Any, key: str) -> float | None:
    """One finite OHLC field from a mapping or a pandas Series."""
    if bar is None:
        return None
    getter = getattr(bar, "get", None)
    if getter is None:
        return None
    try:
        raw = getter(key)
    except Exception:
        return None
    return _finite(raw)


def bar_is_valid(bar: Mapping[str, Any] | Any) -> bool:
    """True when the candle's four prices do not contradict each other.

    The invariant is the same one the charts draw against: ``low <= open,
    close <= high``. A missing/NaN OPEN is NOT invalid - it is unknown, and
    ``resolve_fill`` answers it by clamping. A missing/NaN high, low or close
    IS invalid: without them there is no range to reason about at all.
    """
    high = _bar_value(bar, "high")
    low = _bar_value(bar, "low")
    close = _bar_value(bar, "close")
    if high is None or low is None or close is None:
        return False
    if low > high:
        return False
    if not (low <= close <= high):
        return False
    open_value = _bar_value(bar, "open")
    if open_value is not None and not (low <= open_value <= high):
        return False
    return True


def _is_long(side: Any) -> bool:
    text = str(side or "").strip().upper()
    return text not in {"SHORT", "S", "SELL", "-1", "-1.0"}


def resolve_fill(
    side: Any,
    kind: str,
    level: Any,
    bar: Mapping[str, Any] | Any,
) -> Fill:
    """The ``gap_aware_v2`` fill for one level touch. Pure; no I/O, no state.

    ``kind`` is ``"stop"`` or ``"target"``. ``bar`` is any mapping with
    ``high`` / ``low`` / ``close`` and an OPTIONAL ``open``.

    This function is the v2 convention itself. ``literal_level_v1`` never calls
    it: v1 is "the level", and giving it a code path here would be a second
    description of behaviour that already ships.
    """
    kind_text = str(kind or "").strip().lower()
    if kind_text not in {FILL_KIND_STOP, FILL_KIND_TARGET}:
        raise ValueError(f"unknown fill kind {kind!r}; expected 'stop' or 'target'")

    level_value = _finite(level)
    if level_value is None:
        return Fill(price=float("nan"), basis=FILL_BASIS_NO_LEVEL, booked=False)

    if not bar_is_valid(bar):
        return Fill(price=float("nan"), basis=FILL_BASIS_INVALID_BAR, booked=False)

    high = _bar_value(bar, "high")
    low = _bar_value(bar, "low")
    open_value = _bar_value(bar, "open")
    long_side = _is_long(side)

    if open_value is None:
        # Nothing is known about where the bar started, so no gap can be
        # claimed. The only defensible price is the level pulled inside the bar.
        price = min(max(level_value, low), high)
        return _checked(Fill(price=price, basis=FILL_BASIS_CLAMPED_NO_OPEN, booked=True), low, high)

    if kind_text == FILL_KIND_STOP:
        gapped = open_value < level_value if long_side else open_value > level_value
    else:
        gapped = open_value > level_value if long_side else open_value < level_value
    if gapped:
        return _checked(Fill(price=open_value, basis=FILL_BASIS_GAP_OPEN, booked=True), low, high)

    price = min(max(level_value, low), high)
    basis = FILL_BASIS_LEVEL if price == level_value else FILL_BASIS_CLAMPED_LEVEL
    return _checked(Fill(price=price, basis=basis, booked=True), low, high)


def _checked(fill: Fill, low: float, high: float) -> Fill:
    """The v2 promise, enforced rather than documented: the price traded."""
    if fill.booked and not (low <= fill.price <= high):  # pragma: no cover - guard
        raise AssertionError(
            f"gap_aware_v2 produced {fill.price!r} outside the bar [{low!r}, {high!r}]"
        )
    return fill


def is_gap_aware(execution_convention: Any) -> bool:
    """True when the caller asked for ``gap_aware_v2``.

    Written as an equality on the NAME rather than "not the default" so a
    typo'd convention string can never silently opt a run into v2.
    """
    return str(execution_convention or "") == EXECUTION_GAP_AWARE_V2


def uses_prior_session_levels(level_knowledge: Any) -> bool:
    """True when intrabar tests must read the previous session's levels."""
    return str(level_knowledge or "") == LEVEL_KNOWLEDGE_PRIOR_SESSION_V2
