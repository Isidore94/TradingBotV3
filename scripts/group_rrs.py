"""Pure Real Relative Strength maths for the group RS/RW tape.

plan.md Phase 0.5 item 11, packet T-1. Spec:
`docs/prompts/GROUP_TAPE_REBUILD_OPUS_PROMPT.md`.

Why this module exists at all
-----------------------------
The formula was never wrong. The old tape read
`bounce_bot_lib.legacy.real_relative_strength` over `RRS_LENGTH = 12` M5 bars
taken from a **5-day** IB fetch, refreshed only when a scan cycle's RRS pass
finished. Two consequences, both measured on 2026-08-27:

* it was 10-30 minutes stale (once 31 minutes late on a flip), and
* for the first hour of the session the 12-bar window reached back across the
  overnight gap, so 06:36 read XLK +10.5 / XLC -18.6 - the gap, not the
  morning.

So the maths is lifted out unchanged and given today's bars and three windows.
A parity test (`tests/test_group_rrs.py`) feeds identical bars to this module
and to `legacy.real_relative_strength` and asserts equality to 1e-9. Nothing
here imports `legacy`: the tape must keep working with BounceBot off, and the
service must not drag a 14k-line detector module onto a worker thread.

Purity contract: bars in, floats out. No I/O, no Qt, no clock of its own -
``now`` is always passed in. A bar may be a mapping (``autopilot_core.
_frame_rows`` emits dicts) or an object with attributes (BounceBot's ``IbBar``);
the shape of a bar is a producer detail, not a different rule.

UNKNOWN is never invented. A window without enough completed bars is ``None``,
not zero and not "as many as we have" - plan.md sec 5, missing data is
uncertainty, never confirmation.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Mapping, Sequence

from completed_bars import align_to, bar_time, completed_m5_bars

#: The 11 SPDR sector ETFs, keyed by Yahoo's sector key. A copy of
#: ``legacy.DEFAULT_SECTOR_ETF_MAP`` rather than an import: the tape must not
#: depend on BounceBot being importable, and the service must not drag a
#: 14k-line detector module onto a worker thread. A drift test pins the two
#: together, so a future edit to either one fails loudly.
SECTOR_ETFS: dict[str, str] = {
    "communication-services": "XLC",
    "consumer-cyclical": "XLY",
    "consumer-defensive": "XLP",
    "energy": "XLE",
    "financial-services": "XLF",
    "healthcare": "XLV",
    "industrials": "XLI",
    "basic-materials": "XLB",
    "real-estate": "XLRE",
    "technology": "XLK",
    "utilities": "XLU",
}

#: The benchmark every group is measured against.
BENCHMARK = "SPY"

#: Window label -> number of completed 5-minute bars. 6/12/18 bars = 30/60/90
#: minutes, which is the read the trader asked for ("what is actually strong
#: over the last 30-60-90 minutes").
RRS_WINDOWS: dict[str, int] = {"30": 6, "60": 12, "90": 18}

#: Ordered oldest-window-first so a sparkline reads left-to-right as "where it
#: has been" -> "where it is now".
WINDOW_ORDER: tuple[str, ...] = ("90", "60", "30")


def _value(bar: Any, key: str) -> float | None:
    """One OHLC field off a mapping-shaped or attribute-shaped bar."""
    raw = bar.get(key) if isinstance(bar, Mapping) else getattr(bar, key, None)
    try:
        number = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # NaN guard


def wilder_atr_last(bars: Sequence[Any], length: int) -> float | None:
    """Wilder ATR of the final bar, or None.

    Behaviourally identical to ``legacy._wilder_atr_last``, including the two
    details a re-derivation would get wrong: the true ranges are taken over
    **every** bar supplied (the first ``length`` seed a simple mean, the rest
    smooth it), so the answer depends on the whole series and not just its
    tail; and a non-positive ATR is ``None`` rather than 0, which is what keeps
    the division in ``real_relative_strength`` from producing an infinity.
    """
    bars = list(bars or ())
    if len(bars) < length + 1:
        return None
    true_ranges: list[float] = []
    for index in range(1, len(bars)):
        high = _value(bars[index], "high")
        low = _value(bars[index], "low")
        prev_close = _value(bars[index - 1], "close")
        if high is None or low is None or prev_close is None:
            return None
        true_ranges.append(
            max(high - low, abs(high - prev_close), abs(low - prev_close))
        )
    if len(true_ranges) < length:
        return None
    atr = sum(true_ranges[:length]) / float(length)
    for true_range in true_ranges[length:]:
        atr = ((atr * (length - 1)) + true_range) / float(length)
    return atr if atr > 0 else None


def real_relative_strength(
    symbol_bars: Sequence[Any], spy_bars: Sequence[Any], length: int
) -> tuple[float | None, float | None]:
    """(rrs, power_index) for one series against the benchmark, or (None, None).

    The parity target. ``power_index`` is how far SPY moved in its own ATR;
    ``rrs`` is how far the symbol moved beyond what that market push alone
    would explain, expressed in the symbol's own ATR - which is why a 14x
    spread in raw volatility across one batch does not distort the ranking.
    """
    symbol_bars = list(symbol_bars or ())
    spy_bars = list(spy_bars or ())
    if not symbol_bars or not spy_bars:
        return None, None
    min_bars = length + 2
    if len(symbol_bars) < min_bars or len(spy_bars) < min_bars:
        return None, None
    sym_last = _value(symbol_bars[-1], "close")
    sym_prior = _value(symbol_bars[-1 - length], "close")
    spy_last = _value(spy_bars[-1], "close")
    spy_prior = _value(spy_bars[-1 - length], "close")
    if sym_last is None or sym_prior is None or spy_last is None or spy_prior is None:
        return None, None
    sym_move = sym_last - sym_prior
    spy_move = spy_last - spy_prior
    sym_atr = wilder_atr_last(symbol_bars[:-1], length)
    spy_atr = wilder_atr_last(spy_bars[:-1], length)
    if not sym_atr or not spy_atr:
        return None, None
    power_index = spy_move / spy_atr
    return (sym_move - (power_index * sym_atr)) / sym_atr, power_index


def session_bars(
    bars: Sequence[Any], *, now: datetime, session_date: date | None = None
) -> list[Any]:
    """Today's COMPLETED 5-minute bars, in the order given.

    Two filters, and both are load-bearing:

    * completeness comes from `completed_bars.completed_m5_bars` - the one
      rule, inclusive at the boundary, converting timezones with ``astimezone``
      rather than stripping them; and
    * everything outside ``session_date`` is dropped, which is what stops a
      window reaching back over the overnight gap. ``period="1d"`` already
      returns one session, so on the live path this is a no-op; it is here
      because "the fetch happens to be shaped right" is not a guarantee, and a
      fixture that spans two days must answer the way the desk does.

    ``session_date`` defaults to ``now``'s date. US regular trading hours fall
    on one calendar date in ET and in every US local zone, so the desk clock
    answers this correctly; it stays a parameter so a desk configured east of
    UTC can pass the market's date instead of inheriting its own.
    """
    if session_date is None:
        session_date = now.date()
    out: list[Any] = []
    for bar in completed_m5_bars(bars or (), now=now):
        stamp = bar_time(bar)
        if stamp is None:
            continue
        if align_to(stamp, now).date() != session_date:
            continue
        out.append(bar)
    return out


def align_bars(
    symbol_bars: Sequence[Any], spy_bars: Sequence[Any], *, now: datetime
) -> tuple[list[Any], list[Any]]:
    """The two series on their common timestamps, ascending.

    A bar either side lacks is dropped from BOTH rather than compared against
    whatever sat next to it - an ETF that halted for one bar would otherwise
    have its move measured over a longer span than SPY's and read as strength.
    This is `legacy._align_bars_with_map`'s rule, restated for bars that may
    carry a timezone: the stamps are normalized before they are matched, so an
    ET-stamped ETF and a UTC-stamped SPY still intersect instead of silently
    sharing no keys at all.
    """
    spy_by_stamp: dict[datetime, Any] = {}
    for bar in spy_bars or ():
        stamp = bar_time(bar)
        if stamp is not None:
            spy_by_stamp[align_to(stamp, now)] = bar
    sym_by_stamp: dict[datetime, Any] = {}
    for bar in symbol_bars or ():
        stamp = bar_time(bar)
        if stamp is not None:
            sym_by_stamp[align_to(stamp, now)] = bar
    common = sorted(sym_by_stamp.keys() & spy_by_stamp.keys())
    if not common:
        return [], []
    return [sym_by_stamp[key] for key in common], [spy_by_stamp[key] for key in common]


def session_rrs(
    symbol_bars: Sequence[Any],
    spy_bars: Sequence[Any],
    *,
    now: datetime,
    length: int,
    session_date: date | None = None,
) -> float | None:
    """RRS over the last ``length`` completed bars of today's session, or None.

    None means "not measurable yet", and the caller must render it as a blank
    rather than a zero: on a tape 0.0 reads as "this group is exactly in line
    with SPY", which is a claim, while a blank reads as "no answer yet", which
    is the truth.
    """
    aligned_symbol, aligned_spy = _aligned_session(
        symbol_bars, spy_bars, now=now, session_date=session_date
    )
    rrs, _power = real_relative_strength(aligned_symbol, aligned_spy, length)
    return rrs


def rrs_windows(
    symbol_bars: Sequence[Any],
    spy_bars: Sequence[Any],
    *,
    now: datetime,
    session_date: date | None = None,
) -> dict[str, float | None]:
    """The 30 / 60 / 90-minute reads for one symbol against SPY.

    The session filter and the alignment are done once and the three windows
    share them, so the three numbers are guaranteed to describe the same bars -
    a chip whose 30 and 90 disagreed because they had been measured over
    different series would be worse than no chip.
    """
    aligned_symbol, aligned_spy = _aligned_session(
        symbol_bars, spy_bars, now=now, session_date=session_date
    )
    out: dict[str, float | None] = {}
    for label, length in RRS_WINDOWS.items():
        rrs, _power = real_relative_strength(aligned_symbol, aligned_spy, length)
        out[label] = rrs
    return out


def _aligned_session(
    symbol_bars: Sequence[Any],
    spy_bars: Sequence[Any],
    *,
    now: datetime,
    session_date: date | None,
) -> tuple[list[Any], list[Any]]:
    return align_bars(
        session_bars(symbol_bars, now=now, session_date=session_date),
        session_bars(spy_bars, now=now, session_date=session_date),
        now=now,
    )


def minimum_bars_for(label: str) -> int:
    """Completed bars a window needs before it can answer.

    ``length + 2``: the move spans ``length`` bars, and the ATR is taken over
    the series with its last bar removed, which needs one more on top. Stated
    here so a surface can say how far off an answer is instead of showing an
    empty tape with no explanation.
    """
    return RRS_WINDOWS[label] + 2
