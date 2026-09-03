"""Ground rule 10, implemented once — R10.C.

Every evidence-facing summary in this repo answers the same questions, and
until now each surface answered a different subset of them in its own way. The
cohort rollup published a bare mean, a win rate and a profit factor; the setup
scoreboard published a trimmed mean and quantiles but no concentration and no
interval; the review report published counts. A reader comparing two of them
was comparing two different disciplines.

So the discipline lives here, once, and every surface named in ground rule 11
routes through it: the Daytrade Tracker, the Setup Tracker and Focus Picks
summaries, the cohort performance CSVs, `setup_scoreboard.py`, and the
`review_learning` report.

**What every summary must carry, and why each piece earns its place:**

* **Event, symbol and session counts.** n=200 from four symbols on two sessions
  is not n=200. Concentration is reported for the same reason: a family whose
  record is one name on one day has a sample size of roughly one.
* **Excluded and unresolved, by reason, beside n.** A number that quietly
  dropped 40% of its rows is not the number the reader thinks it is. Missing
  data is uncertainty, never confirmation (plan.md sec 5).
* **Raw and robust side by side.** A plain mean on a ratio with an unbounded
  numerator is the statistic that produced `regime_pause_rw`'s −1.82R. It is
  still printed — hiding it would be its own dishonesty — but never alone.
* **Uncapped, 4R-clipped and trimmed together.** The 4R clip is what the
  ranking views already use; showing all three makes the effect of the clip
  visible instead of baked in.
* **Profit factor with its convention stated.** A cohort with no losers has an
  undefined PF, and printing a large finite number there is a lie about a
  denominator of zero.
* **A session-block bootstrap interval.** Trades inside one session are not
  independent — they share the tape. Resampling whole sessions is the cheapest
  honest interval; resampling individual trades would report a precision the
  data does not have.
* **A `discovery` vs `confirmation` label.** Everything computed post hoc is
  discovery. Confirmation requires a window declared in advance, so a caller
  must pass evidence that one exists; it can never be inferred from n.

**n ≥ 30 is necessary, not sufficient.** `meets_n_floor` says only that the
floor was cleared. Nothing here ever returns `reportable: True` on the strength
of a count.

Pure: values in, numbers out. No IO, no clock, no global state, and every
result is deterministic across runs — the bootstrap seeds from the data itself
rather than from a system RNG, because a report that changes between two runs
over identical inputs cannot be checked by anyone.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Iterable, Mapping, Sequence

#: Schema NAME (ground rule 5). A changed meaning is a new name.
SUMMARY_SCHEMA = "evidence_summary_v1"

#: Necessary, never sufficient.
MIN_REPORTABLE_N = 30

#: "LATELY" IS ONE NUMBER, AND IT LIVES HERE (V3 item 3, decision 0016 answer 6:
#: *"'this market regime' needs no definition. 'Lately' is a rolling window
#: (about 20 sessions). No regime label."*).
#:
#: Every surface that says "lately" reads this: the Working-lately section, the
#: blind-spot and leak callouts, the per-family win rates, the priority switch and
#: `held_run_score`'s rolling segments. Before V3 those paths carried their own
#: literals - 20 in one module, 90 days in another - so two screens could
#: disagree about what the trader's own word meant.
#:
#: TRADING SESSIONS, never calendar days. Twenty calendar days is fourteen
#: sessions in a normal month and twelve across a holiday week, so a calendar
#: window silently shortens the sample exactly when the market was closed.
LATELY_SESSIONS = 20


def lately_start(end=None, *, sessions: int = LATELY_SESSIONS):
    """The first session of the "lately" window ending at `end` (inclusive).

    Walks the exchange calendar backwards, so the window is `sessions` SESSIONS
    long whatever holidays fall in it. `end` defaults to today's date.

    Falls back to a calendar-day estimate if the calendar refuses the date - the
    validated range has ends, and a window that cannot be computed must not stop
    a readout from rendering. The fallback is deliberately the CONSERVATIVE
    direction: `sessions * 7 / 5` calendar days is longer than the true window,
    so it can include an extra session but never silently drop one.
    """
    from datetime import date as _date, timedelta as _timedelta

    last = end or _date.today()
    if isinstance(last, str):
        try:
            last = _date.fromisoformat(last[:10])
        except ValueError:
            last = _date.today()
    try:
        from market_calendar import is_session, previous_session

        cursor = last if is_session(last) else previous_session(last)
        for _ in range(max(0, int(sessions) - 1)):
            cursor = previous_session(cursor)
        return cursor
    except Exception:  # noqa: BLE001 - a window is never worth a blank readout
        return last - _timedelta(days=int(round(int(sessions) * 7 / 5)))


def lately_window(end=None, *, sessions: int = LATELY_SESSIONS):
    """`(first, last)` ISO dates for the window. Inclusive at both ends."""
    from datetime import date as _date

    last = end or _date.today()
    if isinstance(last, str):
        try:
            last = _date.fromisoformat(last[:10])
        except ValueError:
            last = _date.today()
    return lately_start(last, sessions=sessions).isoformat(), last.isoformat()

#: The 10% trimmed mean this repo already uses.
TRIM_FRACTION = 0.10

#: The existing clip where a view feeds ranking (ground rule 10).
R_CLIP = 4.0

#: Bootstrap resamples. Enough to stabilise a percentile interval, few enough
#: that a rollup over hundreds of cells stays fast.
BOOTSTRAP_RESAMPLES = 400
BOOTSTRAP_LOW = 5.0
BOOTSTRAP_HIGH = 95.0

LABEL_DISCOVERY = "discovery"
LABEL_CONFIRMATION = "confirmation"

#: Stated wherever a profit factor is printed.
PROFIT_FACTOR_CONVENTION = (
    "profit factor = sum(gains) / abs(sum(losses)). With no losing rows the "
    "denominator is zero and PF is reported as null with all_wins=true, never "
    "as a large finite number; with no winning rows it is 0.0 with "
    "all_losses=true."
)


def _finite(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values or ():
        try:
            if value is None or value == "":
                continue
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(number) or math.isinf(number):
            continue
        out.append(number)
    return out


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated quantile over an already-sorted sequence."""
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = fraction * (len(sorted_values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(sorted_values[low])
    weight = position - low
    return float(sorted_values[low] * (1 - weight) + sorted_values[high] * weight)


def trimmed_mean(values: Sequence[float], fraction: float = TRIM_FRACTION) -> float | None:
    """Mean with `fraction` trimmed from each tail.

    Returns the plain mean when trimming would empty the sample - a trimmed
    mean of nothing is not zero, and refusing to answer here would lose the
    only number a tiny cell has.
    """
    ordered = sorted(values)
    if not ordered:
        return None
    cut = int(len(ordered) * fraction)
    kept = ordered[cut: len(ordered) - cut] if cut and len(ordered) - 2 * cut > 0 else ordered
    return float(sum(kept) / len(kept))


def profit_factor(values: Sequence[float]) -> dict[str, Any]:
    """PF with its convention carried alongside, never bare."""
    gains = sum(value for value in values if value > 0)
    losses = sum(value for value in values if value < 0)
    all_wins = bool(values) and losses == 0
    all_losses = bool(values) and gains == 0
    if not values:
        return {"value": None, "all_wins": False, "all_losses": False,
                "convention": PROFIT_FACTOR_CONVENTION}
    if all_wins:
        # No denominator. A large finite number here would be a claim about a
        # division nobody performed.
        return {"value": None, "all_wins": True, "all_losses": False,
                "convention": PROFIT_FACTOR_CONVENTION}
    return {
        "value": round(gains / abs(losses), 4) if losses else None,
        "all_wins": False,
        "all_losses": all_losses,
        "convention": PROFIT_FACTOR_CONVENTION,
    }


def _concentration(labels: Sequence[str]) -> dict[str, Any]:
    """How much of the sample one label supplies.

    `top_share` near 1.0 means the cell is one thing wearing a large n.
    """
    cleaned = [str(label) for label in labels if str(label or "").strip()]
    if not cleaned:
        return {"distinct": 0, "top": None, "top_share": None, "measured": False}
    counts: dict[str, int] = {}
    for label in cleaned:
        counts[label] = counts.get(label, 0) + 1
    top, top_count = max(counts.items(), key=lambda item: (item[1], item[0]))
    return {
        "distinct": len(counts),
        "top": top,
        "top_share": round(top_count / len(cleaned), 4),
        "measured": True,
    }


def _seed_from(values: Sequence[float], sessions: Sequence[str]) -> int:
    """A seed derived from the data, so two runs over identical inputs agree.

    A system RNG would make every report unreproducible, and a hard-coded seed
    would make every cell resample in the same order regardless of content.
    """
    payload = "|".join(f"{value:.6f}" for value in values) + "||" + "|".join(sessions)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def session_block_bootstrap(
    values: Sequence[float],
    sessions: Sequence[str],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> dict[str, Any]:
    """A percentile interval on the mean, resampling whole SESSIONS.

    Trades inside one session share the tape, so they are not independent
    draws; resampling them individually would report an interval narrower than
    the data supports. Blocks are sessions, drawn with replacement.

    Unmeasurable - and says so - when the sample spans fewer than two sessions.
    An interval over one block is a statement about one day, and printing it
    would give a single session the appearance of a range.
    """
    if not values:
        return {"measured": False, "reason": "no values"}
    labels = [str(label or "") for label in sessions]
    if len(labels) != len(values) or not all(labels):
        return {"measured": False, "reason": "no session identity on these rows"}
    blocks: dict[str, list[float]] = {}
    for value, label in zip(values, labels):
        blocks.setdefault(label, []).append(value)
    if len(blocks) < 2:
        return {
            "measured": False,
            "reason": f"only {len(blocks)} session in the sample; an interval would "
                      "describe one day as though it were a range",
            "sessions": len(blocks),
        }
    keys = sorted(blocks)
    rng = random.Random(_seed_from(list(values), keys))
    means: list[float] = []
    for _ in range(max(1, int(resamples))):
        drawn: list[float] = []
        for _ in range(len(keys)):
            drawn.extend(blocks[keys[rng.randrange(len(keys))]])
        if drawn:
            means.append(sum(drawn) / len(drawn))
    if not means:
        return {"measured": False, "reason": "no resample produced a value"}
    means.sort()
    return {
        "measured": True,
        "sessions": len(keys),
        "resamples": len(means),
        "low": round(_quantile(means, BOOTSTRAP_LOW / 100.0), 4),
        "high": round(_quantile(means, BOOTSTRAP_HIGH / 100.0), 4),
        "interval": f"{BOOTSTRAP_LOW:.0f}-{BOOTSTRAP_HIGH:.0f} percentile of a "
                    "session-block bootstrap on the mean",
    }


def _moments(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "median": None, "trimmed_mean": None, "p10": None, "p90": None}
    ordered = sorted(values)
    return {
        "mean": round(sum(values) / len(values), 4),
        "median": round(_quantile(ordered, 0.5), 4),
        "trimmed_mean": round(trimmed_mean(values), 4),
        "p10": round(_quantile(ordered, 0.10), 4),
        "p90": round(_quantile(ordered, 0.90), 4),
    }


def summarize(
    values: Iterable[Any],
    *,
    symbols: Sequence[str] | None = None,
    sessions: Sequence[str] | None = None,
    stop_flags: Sequence[Any] | None = None,
    excluded: Mapping[str, int] | None = None,
    unresolved: Mapping[str, int] | None = None,
    clip: float | None = R_CLIP,
    confirmation_window: str | None = None,
    min_n: int = MIN_REPORTABLE_N,
) -> dict[str, Any]:
    """One evidence-facing summary, carrying everything ground rule 10 demands.

    `confirmation_window` is the ONLY route to a `confirmation` label, and it
    must name the window that was declared in advance. It can never be inferred
    from n - a large post-hoc sample is a large discovery, not a confirmation.
    """
    numbers = _finite(values)
    n = len(numbers)
    excluded_map = {str(k): int(v) for k, v in (excluded or {}).items() if int(v or 0)}
    unresolved_map = {str(k): int(v) for k, v in (unresolved or {}).items() if int(v or 0)}

    stop_rate = None
    if stop_flags is not None:
        flags = []
        for flag in stop_flags:
            if isinstance(flag, bool):
                flags.append(1.0 if flag else 0.0)
            elif str(flag).strip().lower() in {"true", "1"}:
                flags.append(1.0)
            elif str(flag).strip().lower() in {"false", "0"}:
                flags.append(0.0)
        if flags:
            stop_rate = round(sum(flags) / len(flags), 4)

    clipped = None
    if clip is not None and numbers:
        bound = abs(float(clip))
        clipped = _moments([max(-bound, min(bound, value)) for value in numbers])

    session_labels = [str(label or "") for label in (sessions or [])]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "n": n,
        "counts": {
            "events": n,
            "symbols": _concentration(list(symbols or [])).get("distinct", 0),
            "sessions": len({label for label in session_labels if label}),
        },
        "excluded_by_reason": excluded_map,
        "excluded_total": sum(excluded_map.values()),
        "unresolved_by_reason": unresolved_map,
        "unresolved_total": sum(unresolved_map.values()),
        "raw": _moments(numbers),
        "clipped": clipped,
        "clip": None if clip is None else abs(float(clip)),
        "profit_factor": profit_factor(numbers),
        "stop_rate": stop_rate,
        "concentration": {
            "by_symbol": _concentration(list(symbols or [])),
            "by_session": _concentration(session_labels),
        },
        "bootstrap": session_block_bootstrap(numbers, session_labels),
        "meets_n_floor": n >= int(min_n),
        "n_floor": int(min_n),
        # Everything post hoc is discovery. A caller with a pre-declared window
        # says so explicitly and names it.
        "evidence_label": LABEL_CONFIRMATION if confirmation_window else LABEL_DISCOVERY,
        "confirmation_window": str(confirmation_window or ""),
        "n_floor_note": (
            f"n >= {int(min_n)} is NECESSARY, not sufficient: it clears the floor "
            "and says nothing about concentration, session coverage, or whether "
            "the window was declared in advance"
        ),
    }
    return summary


def format_note(summary: Mapping[str, Any]) -> str:
    """One line a report or a health tile can print verbatim."""
    n = summary.get("n", 0)
    raw = summary.get("raw") or {}
    parts = [
        f"n={n}",
        f"{summary.get('counts', {}).get('symbols', 0)} symbol(s)",
        f"{summary.get('counts', {}).get('sessions', 0)} session(s)",
        f"mean={raw.get('mean')}",
        f"trimmed={raw.get('trimmed_mean')}",
        f"median={raw.get('median')}",
    ]
    boot = summary.get("bootstrap") or {}
    if boot.get("measured"):
        parts.append(f"block CI [{boot.get('low')}, {boot.get('high')}]")
    else:
        parts.append(f"block CI unmeasured ({boot.get('reason')})")
    concentration = (summary.get("concentration") or {}).get("by_symbol") or {}
    if concentration.get("measured"):
        parts.append(f"top symbol {concentration.get('top')} {concentration.get('top_share')}")
    excluded = summary.get("excluded_total") or 0
    if excluded:
        reasons = ", ".join(
            f"{reason} {count}" for reason, count in sorted((summary.get("excluded_by_reason") or {}).items())
        )
        parts.append(f"excluded {excluded} ({reasons})")
    parts.append(str(summary.get("evidence_label")))
    if not summary.get("meets_n_floor"):
        parts.append(f"below the n>={summary.get('n_floor')} floor")
    return " | ".join(parts)
