"""Path capture and the frozen exit policies — R10.B.

The outcome store records what one exit rule produced. That is enough to say
"this trade made 0.4R under `eod_hold`" and not enough to answer the question
the trader actually asks — *would a different exit have done better, and by how
much?* Answering it later means refetching bars that were in memory at the
time, for thousands of trades, from a provider that rate-limits.

So every entry claim carries its **path**: what price did between entry and the
end of the session, compressed enough to store on every row and complete enough
to simulate an exit policy offline with no refetch.

What is captured, and why each piece:

* **MFE / MAE at declared bar marks** (1/3/6/12/24/36 and EOD). The excursion
  *so far*, not the close — a trade that ran 2R and gave it all back is a
  different trade from one that never moved, and a close-only record cannot
  tell them apart.
* **First-touch stamps.** The bar index where price first reached +1R and where
  it first hit the stop. Order matters: a trade that touched its stop before
  its target is not a winner under any honest policy, and only the ORDER can
  say so. Where a single bar contains both, the STOP is taken first
  (`stop_first_intrabar`) — the pessimistic reading, because a bar's OHLC does
  not carry the sequence within it and assuming the good one manufactures
  profits.
* **Giveback.** MFE minus the final close, in R. The number that says "you had
  it and lost it".
* **A compact excursion path.** Per-bar (high, low) in R units, rounded, so a
  future exit model replays offline.

The four **frozen** exit policies are frozen in the sense that matters: their
definitions do not change, so a number computed today and one computed in a
year are comparable. Each is reported on its OWN — never blended, never
best-of-presented-as-result.

`oracle_best_ex_post_r` is the best of them chosen with hindsight. It is an
**upper bound and never a result** (R10 ground rule 12), it is labelled that
way in the returned payload, and no policy is allowed to claim it. "Realizable
R" is not a term this repo uses.

Pure: bars in, numbers out. No IO, no clock, no state. Missing data is
uncertainty — an unmeasurable value is ``None`` and is never a zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

#: Schema NAME for the payload this module produces (ground rule 5).
PATH_SCHEMA = "outcome_path_v1"

#: Bar marks the excursion is reported at. 24 and 36 extend the existing
#: 1/3/6/12 milestones so a full session is covered at M5 (78 bars), without
#: storing a mark per bar.
PATH_BAR_MARKS: tuple[int, ...] = (1, 3, 6, 12, 24, 36)

#: The frozen exit policies, by name. Changing one of these definitions means a
#: NEW name - `trail_2bar_after_1r_v2`, never a redefinition - because a rollup
#: that mixes two meanings under one word is worse than no rollup.
POLICY_EOD_HOLD = "eod_hold"
POLICY_TRAIL_2BAR = "trail_2bar_after_1r"
POLICY_VWAP_CLOSE = "vwap_close_after_1r"
POLICY_ATR_TRAIL = "atr_1p5_trail"
FROZEN_EXIT_POLICIES = (POLICY_EOD_HOLD, POLICY_TRAIL_2BAR, POLICY_VWAP_CLOSE, POLICY_ATR_TRAIL)

#: The hindsight upper bound. Never a result (ground rule 12).
ORACLE_KEY = "oracle_best_ex_post_r"

_ROUND = 4


@dataclass(frozen=True)
class Bar:
    """One completed bar. `vwap` is optional and only `vwap_close_after_1r` reads it."""

    high: float
    low: float
    close: float
    open: float | None = None
    time: str = ""
    vwap: float | None = None


@dataclass(frozen=True)
class ExitResult:
    """One policy's answer, with the reason it stopped and where."""

    policy: str
    r: float | None
    exit_bar: int | None
    reason: str

    def under(self, policy: str) -> "ExitResult":
        """The same measurement, reported under another policy's name.

        Two policies fall back to holding when +1R never arrives, and they need
        that result under their OWN name: a row labelled `eod_hold` inside the
        `trail_2bar_after_1r` column mis-attributes the number to a policy that
        did not produce it.
        """
        return ExitResult(policy, self.r, self.exit_bar, self.reason)


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def to_bars(rows: Iterable[Mapping[str, Any]]) -> tuple[Bar, ...]:
    """Coerce dict-ish rows into bars, dropping any that cannot be measured.

    A row missing a high, a low or a close is not a bar with zeros in it - it
    is a row this module cannot measure, and it is dropped rather than
    fabricated. Callers that need to know how many were dropped compare
    lengths.
    """
    bars: list[Bar] = []
    for row in rows or ():
        high = _as_float(row.get("high"))
        low = _as_float(row.get("low"))
        close = _as_float(row.get("close"))
        if high is None or low is None or close is None:
            continue
        bars.append(
            Bar(
                high=high,
                low=low,
                close=close,
                open=_as_float(row.get("open")),
                time=str(row.get("time") or row.get("datetime") or ""),
                vwap=_as_float(row.get("vwap")),
            )
        )
    return tuple(bars)


def _r_of(price: float, entry: float, risk: float, side: str) -> float:
    """Price expressed in R. Long: above entry is positive. Short: mirrored."""
    if str(side).strip().lower().startswith("short"):
        return (entry - price) / risk
    return (price - entry) / risk


def _favourable_adverse(bar: Bar, entry: float, risk: float, side: str) -> tuple[float, float]:
    """(best, worst) this bar reached, in R.

    For a long the best is the high and the worst is the low; for a short they
    swap. Computing both through `_r_of` rather than by hand is what keeps the
    short case from being written twice and drifting.
    """
    a = _r_of(bar.high, entry, risk, side)
    b = _r_of(bar.low, entry, risk, side)
    return (max(a, b), min(a, b))


def capture_path(
    *,
    entry_price: Any,
    stop_price: Any,
    side: str,
    bars: Sequence[Bar] | Sequence[Mapping[str, Any]],
    atr: Any = None,
    marks: Sequence[int] = PATH_BAR_MARKS,
) -> dict[str, Any]:
    """Measure one entry claim's whole path. Returns a JSON-safe payload.

    Returns a payload whose `measurable` is False, with a reason, when the
    inputs cannot support a measurement - never a payload of zeros. A trade
    with no risk is not a trade that risked nothing.
    """
    entry = _as_float(entry_price)
    stop = _as_float(stop_price)
    side_key = "short" if str(side).strip().lower().startswith("short") else "long"
    series = bars if bars and isinstance(bars[0], Bar) else to_bars(bars)  # type: ignore[arg-type]

    if entry is None or stop is None:
        return _unmeasurable("entry or stop price is missing", side_key, len(series))
    risk = (entry - stop) if side_key == "long" else (stop - entry)
    if risk <= 0:
        return _unmeasurable(
            "risk per share is not positive, so R has no meaning for this row",
            side_key,
            len(series),
        )
    if not series:
        return _unmeasurable("no bars after entry, so nothing has been measured yet", side_key, 0)

    excursion: list[list[float]] = []
    running_mfe = None
    running_mae = None
    at_marks: dict[str, Any] = {}
    first_target_bar: int | None = None
    first_stop_bar: int | None = None
    stop_first_intrabar = False

    for index, bar in enumerate(series, start=1):
        best, worst = _favourable_adverse(bar, entry, risk, side_key)
        running_mfe = best if running_mfe is None else max(running_mfe, best)
        running_mae = worst if running_mae is None else min(running_mae, worst)
        excursion.append([round(best, 2), round(worst, 2)])

        touched_target = best >= 1.0
        touched_stop = worst <= -1.0
        if touched_stop and first_stop_bar is None:
            first_stop_bar = index
            # Both in the same bar: OHLC carries no intrabar sequence, so the
            # STOP is taken first. Assuming the favourable order here would
            # manufacture a profit out of an unknown, on every such bar, in one
            # direction only.
            if touched_target and first_target_bar is None:
                stop_first_intrabar = True
        if touched_target and first_target_bar is None and not (touched_stop and first_stop_bar == index):
            first_target_bar = index

        if index in marks:
            at_marks[str(index)] = {
                "mfe_r": round(running_mfe, _ROUND),
                "mae_r": round(running_mae, _ROUND),
                "close_r": round(_r_of(bar.close, entry, risk, side_key), _ROUND),
            }

    final_close_r = _r_of(series[-1].close, entry, risk, side_key)
    mfe = float(running_mfe if running_mfe is not None else 0.0)
    mae = float(running_mae if running_mae is not None else 0.0)

    policies = {
        result.policy: {
            "r": None if result.r is None else round(result.r, _ROUND),
            "exit_bar": result.exit_bar,
            "reason": result.reason,
        }
        for result in evaluate_exit_policies(
            entry=entry,
            risk=risk,
            side=side_key,
            bars=series,
            atr=_as_float(atr),
            first_target_bar=first_target_bar,
            first_stop_bar=first_stop_bar,
        )
    }
    scored = [value["r"] for value in policies.values() if value["r"] is not None]

    return {
        "schema": PATH_SCHEMA,
        "measurable": True,
        "side": side_key,
        "bars_measured": len(series),
        "risk_per_share": round(risk, 6),
        "mfe_r": round(mfe, _ROUND),
        "mae_r": round(mae, _ROUND),
        "close_r": round(final_close_r, _ROUND),
        # "You had it and lost it." Never negative: a trade whose close is its
        # own high gave nothing back.
        "giveback_r": round(max(0.0, mfe - final_close_r), _ROUND),
        "first_target_bar": first_target_bar,
        "first_stop_bar": first_stop_bar,
        "stop_first_intrabar": stop_first_intrabar,
        "at_marks": at_marks,
        "excursion_r": excursion,
        "exit_policies": policies,
        # Hindsight. An UPPER BOUND, never a result (ground rule 12) - no
        # policy achieved this, and nothing may report it as though one did.
        ORACLE_KEY: round(max(scored), _ROUND) if scored else None,
        "oracle_note": (
            "best of the frozen policies chosen with hindsight: an upper bound "
            "on what any of them could have produced, never a result and never "
            "attributable to a policy"
        ),
    }


def _unmeasurable(reason: str, side: str, bar_count: int) -> dict[str, Any]:
    return {
        "schema": PATH_SCHEMA,
        "measurable": False,
        "reason": reason,
        "side": side,
        "bars_measured": bar_count,
        "mfe_r": None,
        "mae_r": None,
        "close_r": None,
        "giveback_r": None,
        "first_target_bar": None,
        "first_stop_bar": None,
        "stop_first_intrabar": False,
        "at_marks": {},
        "excursion_r": [],
        "exit_policies": {},
        ORACLE_KEY: None,
    }


def evaluate_exit_policies(
    *,
    entry: float,
    risk: float,
    side: str,
    bars: Sequence[Bar],
    atr: float | None = None,
    first_target_bar: int | None = None,
    first_stop_bar: int | None = None,
) -> tuple[ExitResult, ...]:
    """Each frozen policy, on its own, over the same bars."""
    return (
        _eod_hold(entry, risk, side, bars, first_stop_bar),
        _trail_2bar_after_1r(entry, risk, side, bars, first_target_bar, first_stop_bar),
        _vwap_close_after_1r(entry, risk, side, bars, first_target_bar, first_stop_bar),
        _atr_trail(entry, risk, side, bars, atr, first_stop_bar),
    )


def _stopped_first(first_stop_bar: int | None, upto: int | None = None) -> bool:
    return first_stop_bar is not None and (upto is None or first_stop_bar <= upto)


def _eod_hold(entry, risk, side, bars, first_stop_bar) -> ExitResult:
    """Hold to the last measured bar unless the stop was hit first."""
    if first_stop_bar is not None:
        return ExitResult(POLICY_EOD_HOLD, -1.0, first_stop_bar, "stop hit before the close")
    return ExitResult(
        POLICY_EOD_HOLD,
        _r_of(bars[-1].close, entry, risk, side),
        len(bars),
        "held to the last measured bar",
    )


def _trail_2bar_after_1r(entry, risk, side, bars, first_target_bar, first_stop_bar) -> ExitResult:
    """After +1R, exit on a close below the lowest low of the prior 2 bars.

    Before +1R the initial stop still governs; a policy that starts trailing
    from entry is a different policy and would need its own name.
    """
    if _stopped_first(first_stop_bar, first_target_bar):
        return ExitResult(POLICY_TRAIL_2BAR, -1.0, first_stop_bar, "stop hit before +1R")
    if first_target_bar is None:
        return _eod_hold(entry, risk, side, bars, first_stop_bar).under(POLICY_TRAIL_2BAR)
    for index in range(first_target_bar + 1, len(bars) + 1):
        window = bars[max(0, index - 3): index - 1]
        if not window:
            continue
        if side == "long":
            trail = min(bar.low for bar in window)
            if bars[index - 1].close < trail:
                return ExitResult(
                    POLICY_TRAIL_2BAR,
                    _r_of(bars[index - 1].close, entry, risk, side),
                    index,
                    "closed below the 2-bar trail after +1R",
                )
        else:
            trail = max(bar.high for bar in window)
            if bars[index - 1].close > trail:
                return ExitResult(
                    POLICY_TRAIL_2BAR,
                    _r_of(bars[index - 1].close, entry, risk, side),
                    index,
                    "closed above the 2-bar trail after +1R",
                )
    return ExitResult(
        POLICY_TRAIL_2BAR,
        _r_of(bars[-1].close, entry, risk, side),
        len(bars),
        "trail never triggered; held to the last measured bar",
    )


def _vwap_close_after_1r(entry, risk, side, bars, first_target_bar, first_stop_bar) -> ExitResult:
    """After +1R, exit on the first close through session VWAP.

    Unmeasurable without VWAP on the bars, and says so: a policy that silently
    degrades into a different policy when its input is missing produces a
    number under the wrong name.
    """
    if not any(bar.vwap is not None for bar in bars):
        return ExitResult(
            POLICY_VWAP_CLOSE,
            None,
            None,
            "no session VWAP on these bars, so this policy is unmeasured (not zero)",
        )
    if _stopped_first(first_stop_bar, first_target_bar):
        return ExitResult(POLICY_VWAP_CLOSE, -1.0, first_stop_bar, "stop hit before +1R")
    if first_target_bar is None:
        return _eod_hold(entry, risk, side, bars, first_stop_bar).under(POLICY_VWAP_CLOSE)
    for index in range(first_target_bar + 1, len(bars) + 1):
        bar = bars[index - 1]
        if bar.vwap is None:
            continue
        through = bar.close < bar.vwap if side == "long" else bar.close > bar.vwap
        if through:
            return ExitResult(
                POLICY_VWAP_CLOSE,
                _r_of(bar.close, entry, risk, side),
                index,
                "closed through session VWAP after +1R",
            )
    return ExitResult(
        POLICY_VWAP_CLOSE,
        _r_of(bars[-1].close, entry, risk, side),
        len(bars),
        "never closed back through VWAP; held to the last measured bar",
    )


def _atr_trail(entry, risk, side, bars, atr, first_stop_bar) -> ExitResult:
    """Trail 1.5 ATR from the best price reached, from entry.

    Unmeasurable without an ATR, and says so rather than substituting risk.
    """
    if atr is None or atr <= 0:
        return ExitResult(
            POLICY_ATR_TRAIL,
            None,
            None,
            "no ATR supplied, so this policy is unmeasured (not zero)",
        )
    distance = 1.5 * float(atr)
    best: float | None = None
    for index, bar in enumerate(bars, start=1):
        if first_stop_bar is not None and index >= first_stop_bar and best is None:
            return ExitResult(POLICY_ATR_TRAIL, -1.0, first_stop_bar, "initial stop hit before the trail engaged")
        extreme = bar.high if side == "long" else bar.low
        best = extreme if best is None else (max(best, extreme) if side == "long" else min(best, extreme))
        trail = (best - distance) if side == "long" else (best + distance)
        through = bar.close < trail if side == "long" else bar.close > trail
        if through:
            return ExitResult(
                POLICY_ATR_TRAIL,
                _r_of(bar.close, entry, risk, side),
                index,
                "closed through the 1.5-ATR trail",
            )
    return ExitResult(
        POLICY_ATR_TRAIL,
        _r_of(bars[-1].close, entry, risk, side),
        len(bars),
        "trail never triggered; held to the last measured bar",
    )
