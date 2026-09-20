"""TJ-15 item 1 - `scripts/evidence_contrast.py`, the ONE contrast method.

plan.md §12.4 TJ-15 and the packet: *"a deterministic contrast between two
groups over point-in-time fields - counts, medians, ONE rank statistic (AUC, as
`compression_calibration` does), the ONE Wilson interval for rates,
`observational, top 3 of K` stated every time, nothing named under
`MIN_REPORTABLE_N`"*. Built once here and reused by TJ-16.

The contract these tests pin (the builder may add keys, never remove one):

    evidence_contrast.contrast(
        group_a, group_b, *,            # sequences of feature MAPPINGS
        label_a="real_miss", label_b="correct_rejection",
        features=None,                  # None = every column both groups carry
        top=3,
    ) -> {
        # ... see below ...
    }
    evidence_contrast.rate(runs, measured, pending=0) -> {
        "runs": int, "measured": int, "pending": int,
        "rate": float|None, "low": float|None, "high": float|None,
        "reportable": bool,             # measured >= evidence_stats.MIN_REPORTABLE_N
    }

The contrast's own return:

    {
        "label_a": str, "label_b": str,
        "n_a": int, "n_b": int,         # the GROUP sizes, not the measured ones
        "compared": int,                # K: features with a measurement on BOTH sides
        "top": int,                     # how many are shown
        "statement": str,               # says "observational" and "top 3 of K"
        "unmeasured_features": (...),   # named, never silently dropped
        "features": [                   # at most `top`, ranked
            {"feature": str, "median_a": float|None, "median_b": float|None,
             "n_a": int, "n_b": int, "auc": float|None},
        ],
    }

Ranking key: ``abs(auc - 0.5)`` descending, ties by feature NAME ascending. No
group size, no median magnitude and no R statistic may enter that key.

Values arrive as TEXT: these rows come back through `csv.DictReader` off
`d1_features_history.csv`, so an old row has its key PRESENT and EMPTY.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _rows(**columns) -> list[dict]:
    """Column-wise -> row-wise, the way a CSV reader hands them over (text)."""
    length = len(next(iter(columns.values())))
    return [
        {name: ("" if values[i] is None else str(values[i])) for name, values in columns.items()}
        for i in range(length)
    ]


def test_a_feature_that_separates_the_groups_reports_both_counts_two_medians_and_the_precedent_auc():
    """The statistic is `compression_calibration.auc`'s, to the digit.

    PCT-3's report is the precedent the packet names; a second rank statistic
    computed a second way would be a second answer to the same question.
    """
    import compression_calibration
    import evidence_contrast

    misses = _rows(
        pct_from_current_vwap=[3.0, 3.5, 4.0],
        spy_five_day_return_pct=[1.0, 1.0, 1.0],
    )
    correct = _rows(
        pct_from_current_vwap=[1.0, 1.5, 2.0],
        spy_five_day_return_pct=[1.0, 1.0, 1.0],
    )

    # LEAD AMENDMENT 2026-09-19 (floor): the fix round's feature floor is 10 a
    # side and 30 across both; this 3 v 3 fixture pins the AUC, not the floor.
    out = evidence_contrast.contrast(
        misses, correct, label_a="real_miss", label_b="correct_rejection",
        min_side=3, min_total=6,
    )

    assert (out["n_a"], out["n_b"]) == (3, 3)
    assert out["compared"] == 2
    by_name = {row["feature"]: row for row in out["features"]}
    separated = by_name["pct_from_current_vwap"]
    assert separated["median_a"] == pytest.approx(3.5)
    assert separated["median_b"] == pytest.approx(1.5)
    assert (separated["n_a"], separated["n_b"]) == (3, 3)
    assert separated["auc"] == pytest.approx(
        compression_calibration.auc([3.0, 3.5, 4.0], [1.0, 1.5, 2.0])
    )
    assert separated["auc"] == pytest.approx(1.0)
    flat = by_name["spy_five_day_return_pct"]
    assert flat["auc"] == pytest.approx(0.5)


def test_only_the_top_three_features_are_shown_and_the_number_compared_is_stated():
    """`observational, top 3 of K` - K is the count, stated every time."""
    import evidence_contrast

    # Seven features. Only the separation matters, so each is built from two
    # constant columns whose AUC is exactly 1.0, 0.0 or 0.5.
    high = [9.0, 9.0, 9.0]
    low = [1.0, 1.0, 1.0]
    same = [5.0, 5.0, 5.0]
    misses = _rows(
        atr20=high, priority_score=low, compression_penalty=same,
        days_to_next_earnings=same, pct_from_current_vwap=same,
        recent_band_extension_days=same, spy_one_day_return_pct=[6.0, 5.0, 5.0],
    )
    correct = _rows(
        atr20=low, priority_score=high, compression_penalty=same,
        days_to_next_earnings=same, pct_from_current_vwap=same,
        recent_band_extension_days=same, spy_one_day_return_pct=same,
    )

    # LEAD AMENDMENT 2026-09-19 (floor): 3 v 3 pins top-K and the count, not the
    # fix round's 10-a-side floor.
    out = evidence_contrast.contrast(
        misses, correct, label_a="a", label_b="b", min_side=3, min_total=6
    )

    assert out["compared"] == 7
    assert out["top"] == 3
    assert [row["feature"] for row in out["features"]] == [
        "atr20", "priority_score", "spy_one_day_return_pct",
    ]
    assert "observational" in out["statement"], out["statement"]
    assert "top 3 of 7" in out["statement"], out["statement"]


def test_a_tie_breaks_by_feature_name_and_never_by_how_big_the_group_is():
    """A SIZE rule orders nothing here; the rank statistic and the name do."""
    import evidence_contrast

    high = [9.0, 9.0, 9.0, 9.0]
    low = [1.0, 1.0, 1.0, 1.0]
    strong_a = [9.0, 9.0, 9.0, 9.0]
    misses = _rows(z_strongest=strong_a, b_tied=high, a_tied=high, mid=[6.0, 5.0, 5.0, 5.0])
    correct = _rows(
        z_strongest=[0.0, 0.0, 0.0, 0.0],
        b_tied=[1.0, 1.0, 9.0, 9.0],
        a_tied=[1.0, 1.0, 9.0, 9.0],
        mid=[5.0, 5.0, 5.0, 5.0],
    )

    # LEAD AMENDMENT 2026-09-19 (floor): 4 v 4 pins the tie-break, not the floor.
    out = evidence_contrast.contrast(
        misses, correct, label_a="a", label_b="b", top=3, min_side=4, min_total=8
    )

    names = [row["feature"] for row in out["features"]]
    assert names == ["z_strongest", "a_tied", "b_tied"], names
    by_name = {row["feature"]: row for row in out["features"]}
    assert by_name["a_tied"]["auc"] == pytest.approx(by_name["b_tied"]["auc"])


def test_a_blank_cell_is_left_out_of_the_median_and_is_never_read_as_zero():
    """An old row has the key PRESENT and EMPTY. Missing data is not a value.

    `atr20` is measured on two of the four misses: the median is 3.5, which a
    zero-fill would turn into 1.75. `mid_earnings_zone_streak_days` was never
    written on the other side at all, so it is NAMED unmeasured and is not one
    of the K features compared.
    """
    import evidence_contrast

    misses = _rows(
        atr20=[3.0, None, 4.0, "n/a"],
        mid_earnings_zone_streak_days=[2.0, 2.0, 2.0, 2.0],
    )
    correct = _rows(
        atr20=[1.0, 1.0, 1.0, 1.0],
        mid_earnings_zone_streak_days=[None, None, "", ""],
    )

    # LEAD AMENDMENT 2026-09-19 (floor): `atr20` is measured 2 v 4 here and this
    # fixture pins the blank-cell rule, not the fix round's floor.
    out = evidence_contrast.contrast(
        misses, correct, label_a="a", label_b="b", min_side=2, min_total=6
    )

    assert (out["n_a"], out["n_b"]) == (4, 4)
    assert out["compared"] == 1
    assert "mid_earnings_zone_streak_days" in tuple(out["unmeasured_features"])
    row = out["features"][0]
    assert row["feature"] == "atr20"
    assert row["median_a"] == pytest.approx(3.5)
    assert (row["n_a"], row["n_b"]) == (2, 4)


def test_a_rate_is_the_one_wilson_over_closed_horizons_and_an_open_one_is_in_neither_half():
    """TJ-11 blocker 2, inherited: a rate counts CLOSED horizons only.

    Runs and no-runs alike enter it; an open horizon is counted and PRINTED as
    `pending` and is in neither half of the fraction. `pending` must therefore
    move `rate` not at all - the live lately-rejected cell read 34% (30/87)
    where the closed-only truth was 26% (20/77) when it did.

    The interval is the ONE Wilson - `swing_headline`'s z, 1.96 two-sided - and
    the floor under a named cell is `evidence_stats.MIN_REPORTABLE_N`. A second
    Wilson computed a second way is a second answer to the same question, so
    this compares against the desk's function rather than a literal.
    """
    import evidence_contrast
    from evidence_stats import MIN_REPORTABLE_N
    from swing_headline import wilson_lower_bound

    cell = evidence_contrast.rate(9, 40, pending=7)
    assert (cell["runs"], cell["measured"], cell["pending"]) == (9, 40, 7)
    assert cell["rate"] == pytest.approx(9 / 40)
    assert cell["low"] == pytest.approx(wilson_lower_bound(9, 40))
    assert cell["low"] < cell["rate"] < cell["high"]
    assert cell["reportable"] is True

    # The same closed evidence with a hundred more open horizons is the same
    # rate and the same interval. Only `pending` moves.
    with_open = evidence_contrast.rate(9, 40, pending=107)
    assert with_open["rate"] == pytest.approx(cell["rate"])
    assert with_open["low"] == pytest.approx(cell["low"])
    assert with_open["pending"] == 107

    thin = evidence_contrast.rate(3, MIN_REPORTABLE_N - 1)
    assert thin["reportable"] is False
    assert thin["rate"] == pytest.approx(3 / (MIN_REPORTABLE_N - 1))

    # Nothing measured is not a rate of zero.
    nothing = evidence_contrast.rate(0, 0, pending=4)
    assert nothing["rate"] is None
    assert nothing["low"] is None and nothing["high"] is None
    assert nothing["reportable"] is False
