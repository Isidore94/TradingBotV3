"""The ONE deterministic contrast between two groups (TJ-15 item 1).

`plan.md` §12.4 TJ-15 asks one question of the trader's own decisions: *what did
the misses have in common?* The method is the same method every time, so it is
written once here and reused - TJ-16 contrasts right predictions against wrong
ones through this same function.

What a contrast is, exactly:

* two groups of point-in-time feature MAPPINGS, never a population defined by
  its own outcome inside this module - the caller decides who is in which group;
* per feature: the two integer counts, the two medians, and **ONE rank
  statistic** - `compression_calibration.auc`, the precedent PCT-3 set. It is
  CALLED, not re-derived: a second rank statistic computed a second way is a
  second answer to the same question;
* a rank key of ``abs(auc - 0.5)`` descending, ties by feature NAME ascending.
  **No group size, no median magnitude and no R statistic may enter that key**
  (gate #43: a SIZE rule orders a bounded view, a RESULT never does);
* a statement that says `observational` and `top K of N` every time it is read,
  so "the top three" is never mistaken for "the only three".

And what it refuses to do:

* **Read a blank as a zero.** These rows arrive through `csv.DictReader` off
  `d1_features_history.csv`, so an old row has its key PRESENT and EMPTY.
  A cell that is not a number is left out of the median and out of ``n``.
* **Compare a feature only one side measured.** It is NAMED in
  ``unmeasured_features`` instead, never silently dropped and never compared.
* **Spell a z.** :func:`rate` is the ONE Wilson - `swing_headline.WILSON_Z`
  through `walkaway_day._wilson`, the desk's two-ended form - and it counts
  CLOSED horizons only, with the open ones printed as ``pending`` and in
  neither half of the fraction (TJ-11's blocker 2).

Pure: no store, no clock, no network, no model. Everything it returns is
REPORTED evidence and reaches no detector, score, alert, watchlist, Focus,
review queue or `review_policy.json`.
"""

from __future__ import annotations

import statistics
from typing import Any, Iterable, Mapping, Sequence

import compression_calibration
import walkaway_day
from evidence_stats import MIN_REPORTABLE_N

#: How many features a contrast SHOWS. The number compared is always stated
#: beside it, so this is a bounded view and never a filter on the finding.
DEFAULT_TOP = 3

#: TJ-15's two default group names: a name the trader turned down that ran, and
#: one that did not. TJ-16 and the likes half pass their own.
LABEL_A = "real_miss"
LABEL_B = "correct_rejection"

#: Text a CSV writes for "nothing was measured here". None of these is a value.
_NOT_A_NUMBER = frozenset({"", "n/a", "na", "none", "null", "nan", "-", "--"})


def measurement(value: Any) -> float | None:
    """One cell as a number, or ``None`` when it is not a measurement.

    Missing data is uncertainty, never confirmation (`plan.md` sec 5): a blank,
    an ``n/a``, a NaN and an infinity are all "not measured", and not one of
    them is zero.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if not text or text.lower() in _NOT_A_NUMBER:
            return None
        try:
            number = float(text)
        except (TypeError, ValueError):
            return None
    # NaN fails this comparison with itself; an infinity is not a measurement.
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _values(rows: Sequence[Mapping[str, Any]], name: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        number = measurement(row.get(name))
        if number is not None:
            out.append(number)
    return out


def _mappings(group: Iterable[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    return [row for row in (group or ()) if isinstance(row, Mapping)]


def contrast(
    group_a: Iterable[Mapping[str, Any]] | None,
    group_b: Iterable[Mapping[str, Any]] | None,
    *,
    label_a: str = LABEL_A,
    label_b: str = LABEL_B,
    features: Sequence[str] | None = None,
    top: int = DEFAULT_TOP,
) -> dict[str, Any]:
    """Contrast two groups of feature mappings. Pure; see the module docstring.

    ``features`` names the columns to consider; ``None`` means every column
    either group carries, in name order so the walk is deterministic.
    """
    rows_a = _mappings(group_a)
    rows_b = _mappings(group_b)
    if features is None:
        names: list[str] = sorted(
            {key for row in rows_a for key in row} | {key for row in rows_b for key in row}
        )
    else:
        names = [str(name) for name in features]

    measured: list[dict[str, Any]] = []
    unmeasured: list[str] = []
    for name in names:
        values_a = _values(rows_a, name)
        values_b = _values(rows_b, name)
        if not values_a or not values_b:
            # One side has no measurement at all, so there is nothing to
            # contrast. Named rather than dropped: a feature nobody can see was
            # skipped is a feature the reader assumes was looked at.
            unmeasured.append(name)
            continue
        measured.append(
            {
                "feature": name,
                "median_a": statistics.median(values_a),
                "median_b": statistics.median(values_b),
                "n_a": len(values_a),
                "n_b": len(values_b),
                # The desk's ONE rank statistic, CALLED (PCT-3's precedent).
                "auc": compression_calibration.auc(values_a, values_b),
            }
        )

    compared = len(measured)
    limit = max(0, int(top))
    ranked = sorted(
        measured,
        # Separation first, then the NAME. Nothing about how big a group is and
        # no R statistic may reach this key (gate #43).
        key=lambda row: (-abs(float(row["auc"] if row["auc"] is not None else 0.5) - 0.5), row["feature"]),
    )[:limit]

    statement = (
        f"observational, not causal: top {len(ranked)} of {compared} feature(s) "
        f"measured on both sides, {label_a} (n={len(rows_a)}) against "
        f"{label_b} (n={len(rows_b)}); ranked by |AUC-0.5| then feature name, "
        "never by group size and never by an R statistic"
    )
    return {
        "label_a": str(label_a),
        "label_b": str(label_b),
        "n_a": len(rows_a),
        "n_b": len(rows_b),
        "compared": compared,
        "top": limit,
        "statement": statement,
        "unmeasured_features": tuple(sorted(unmeasured)),
        "features": ranked,
    }


def rate(runs: Any, measured: Any, pending: Any = 0) -> dict[str, Any]:
    """One rate over CLOSED horizons, with the ONE Wilson interval.

    ``measured`` is the denominator: runs and no-runs alike, every one of them
    with its clock run out. ``pending`` is COUNTED and PRINTED and is in neither
    half of the fraction - pooling an open horizon censors the rate by its own
    outcome, which is how a shipped cell read 34% (30/87) where the closed-only
    truth was 26% (20/77) (reviewer, 2026-09-19).

    Nothing measured is not a rate of zero: it is ``None``, with no interval.
    """
    total = max(0, int(measured or 0))
    hits = max(0, min(int(runs or 0), total))
    low, high = walkaway_day._wilson(hits, total)
    return {
        "runs": int(runs or 0),
        "measured": total,
        "pending": max(0, int(pending or 0)),
        "rate": (hits / total) if total else None,
        "low": low,
        "high": high,
        "reportable": total >= MIN_REPORTABLE_N,
    }


__all__ = [
    "DEFAULT_TOP",
    "LABEL_A",
    "LABEL_B",
    "contrast",
    "measurement",
    "rate",
]
