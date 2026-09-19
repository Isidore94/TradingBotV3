"""`REAL_MISS_V1` - when a name the trader turned down really did run (TJ-11).

Trader, 2026-09-19: *"if they said no to a bunch of stocks that went on to have
great moves that day or the next day, then I want to know about it."*

A glance calls any green candle a miss. This module is the RULE that replaces
the glance, and it is versioned so a later rule can never be mistaken for it::

    a real run  =  the move reached RUN_ATR x ATR in the decision's favour
                   BEFORE it went ADVERSE_ATR x ATR against it

Three things this rule refuses to do:

* **Guess.** A missing or unreadable ATR, and a stamp with no completed bar
  after it, are ``unmeasured:<reason>`` - never ``no_run``, never zero. Missing
  data is uncertainty, never confirmation (`plan.md` sec 5).
* **Favour itself inside a bar.** One bar's high and low carry no order, so the
  ADVERSE extreme is taken first. A bar that touches both thresholds is
  ``no_run``.
* **Read a forming bar.** With ``now`` and ``bar_minutes`` supplied, the ONE
  completed-bar rule (`scripts/completed_bars.py`) decides what counts.

Pure and import-light on purpose: TJ-15's nightly slot calls this same function
under `requirements-core.txt`, so nothing here may drag pandas, Qt or a broker
client in. It is REPORTED evidence and reaches no detector, score, alert,
watchlist, Focus, review queue or `review_policy.json`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Sequence

from completed_bars import align_to, bar_time, is_completed_bar

#: The rule's name. Stored beside any verdict this module produces, so a later
#: rule is a different name rather than a silent re-reading of an old row.
REAL_MISS_V1 = "real_miss_v1"

#: How far the move must go in the decision's favour, in ATR.
RUN_ATR = 1.0

#: How far against it may go first, in ATR. Reached first, the answer is
#: `no_run` whatever happened afterwards.
ADVERSE_ATR = 0.5

RUN = "run"
NO_RUN = "no_run"

#: Every reason this module can decline to answer for. Spelled once so a reader
#: can branch on them without matching free text.
UNMEASURED_ATR = "unmeasured:atr_unreadable"
UNMEASURED_NO_BARS = "unmeasured:no_completed_bar_after_the_stamp"
UNMEASURED_NO_REFERENCE = "unmeasured:no_reference_price"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # NaN fails this comparison with itself; an infinity is not a price.
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _price(bar: Mapping[str, Any] | Any, key: str) -> float | None:
    if isinstance(bar, Mapping):
        return _number(bar.get(key))
    return _number(getattr(bar, key, None))


def eligible_bars(
    bars: Sequence[Any],
    *,
    stamp: Any,
    now: datetime | None = None,
    bar_minutes: int | None = None,
) -> list[Any]:
    """The bars that START strictly after ``stamp``, oldest first.

    With ``now`` and ``bar_minutes`` a forming bar is excluded by the one
    completed-bar rule. Zones are converted with ``astimezone`` (through
    `completed_bars.align_to`) and never stripped.
    """
    moment = stamp if isinstance(stamp, datetime) else None
    if moment is None and stamp is not None:
        try:
            moment = datetime.fromisoformat(str(stamp))
        except (TypeError, ValueError):
            moment = None
    if moment is None and not (now is not None and bar_minutes):
        # Nothing to filter on. A caller that has already selected its window -
        # the D1 ruler does - pays no timestamp parse per bar for an answer
        # that cannot change.
        return list(bars or ())
    out: list[Any] = []
    for bar in bars or ():
        start = bar_time(bar)
        if start is None:
            continue
        if moment is not None and align_to(start, moment) <= moment:
            continue
        if now is not None and bar_minutes and not is_completed_bar(
            bar, int(bar_minutes), now=now
        ):
            continue
        out.append(bar)
    return out


def verdict(
    bars: Sequence[Any],
    *,
    stamp: Any,
    side: str,
    atr: Any,
    now: datetime | None = None,
    bar_minutes: int | None = None,
    reference: Any = None,
) -> str:
    """``"run"``, ``"no_run"`` or ``"unmeasured:<reason>"`` for one decision.

    ``reference`` overrides the price the excursion is measured from. It exists
    for the D1 ruler, which measures a swing call from the CLOSE of the session
    it was made in rather than from the next bar's open; left out, the rule is
    exactly the one `walkaway_day._after_move` already uses - the open of the
    first eligible bar.
    """
    size = _number(atr)
    if size is None or size <= 0:
        return UNMEASURED_ATR
    rows = eligible_bars(bars, stamp=stamp, now=now, bar_minutes=bar_minutes)
    if not rows:
        return UNMEASURED_NO_BARS
    start = _number(reference)
    if start is None:
        start = _price(rows[0], "open")
    if start is None or start <= 0:
        return UNMEASURED_NO_REFERENCE

    short = str(side or "").strip().upper() == "SHORT"
    run_at = RUN_ATR * size
    adverse_at = ADVERSE_ATR * size
    for bar in rows:
        high = _price(bar, "high")
        low = _price(bar, "low")
        if high is None or low is None:
            # A bar nobody can read is not evidence in either direction.
            continue
        adverse = (high - start) if short else (start - low)
        favourable = (start - low) if short else (high - start)
        # The adverse extreme first: inside one bar the order is unknown.
        if adverse >= adverse_at:
            return NO_RUN
        if favourable >= run_at:
            return RUN
    return NO_RUN


__all__ = [
    "ADVERSE_ATR",
    "NO_RUN",
    "REAL_MISS_V1",
    "RUN",
    "RUN_ATR",
    "UNMEASURED_ATR",
    "UNMEASURED_NO_BARS",
    "UNMEASURED_NO_REFERENCE",
    "eligible_bars",
    "verdict",
]
