"""Review round R1 - the three P4 blockers, each reproduced before it was fixed.

All three are the same shape of bug: a value is COMPUTED correctly and then lost
on the way out, so the file or the view ships something that reads as an answer
("no drop", "derived_from_bucket", "no edge") rather than as a gap.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


# ---------------------------------------------------------------------------
# (a) the stale-horizon coverage line reaches the exported CSV
# ---------------------------------------------------------------------------


def test_the_exported_leaderboard_carries_a_real_drop_count(tmp_path):
    """`pd.DataFrame(rows, columns=COLUMNS)` drops any key not in COLUMNS.

    The row dict carried the coverage line and the file never had it, so a
    reader could not tell "nothing was dropped" from "the count was thrown
    away".
    """
    from master_avwap_lib import legacy

    for column in (
        "observations_before_stale_filter",
        "stale_horizon_observations_dropped",
        "stale_horizon_drop_note",
    ):
        assert column in legacy.SCAN_FACTOR_LEADERBOARD_COLUMNS, column

    row = {
        column: "" for column in legacy.SCAN_FACTOR_LEADERBOARD_COLUMNS
    }
    row.update(
        {
            "observations_before_stale_filter": 120,
            "stale_horizon_observations_dropped": 7,
            "stale_horizon_drop_note": "7 of 120 observation(s) dropped",
        }
    )
    target = tmp_path / "leaderboard.csv"
    legacy._write_scan_factor_csv(target, [row], legacy.SCAN_FACTOR_LEADERBOARD_COLUMNS)

    written = pd.read_csv(target)
    assert int(written.loc[0, "stale_horizon_observations_dropped"]) == 7
    assert int(written.loc[0, "observations_before_stale_filter"]) == 120
    assert "dropped" in str(written.loc[0, "stale_horizon_drop_note"])


# ---------------------------------------------------------------------------
# (b) the tier that shipped survives to the grader
# ---------------------------------------------------------------------------


def test_the_feature_allowlist_carries_the_assigned_tier():
    """The grader reads the feature-history CSV, which this list builds."""
    source = (ROOT / "scripts" / "master_avwap_lib" / "runner.py").read_text(encoding="utf-8")
    assert "ASSIGNED_TIER_FIELD," in source, "the allowlist must name the field"
    assert "feature_row[ASSIGNED_TIER_FIELD]" in source, "and something must write it"


def test_a_row_with_the_assigned_tier_grades_as_assigned_and_an_old_row_does_not():
    """A mixed file must read honestly: B4 rows say `assigned`, older rows say
    which rule actually graded them."""
    from master_avwap_lib.legacy import ASSIGNED_TIER_FIELD, tier_for_tracker_row

    new_row = {ASSIGNED_TIER_FIELD: "A", "priority_bucket": "favorite_setup"}
    assert tier_for_tracker_row(new_row) == ("A", "assigned")

    old_row = {"priority_bucket": "favorite_setup"}
    tier, source = tier_for_tracker_row(old_row)
    assert source == "derived_from_bucket"
    assert tier


def test_both_tier_files_carry_the_source_column():
    """The pick rows built `tier_source` and the column list dropped it; the
    outcome rows unpacked it into a dead local."""
    from master_avwap_lib import legacy

    assert "tier_source" in legacy.TIER_LIST_COLUMNS
    assert "tier_source" in legacy.TIER_OUTCOME_COLUMNS


def test_the_outcome_row_actually_sets_the_source():
    source = (ROOT / "scripts" / "master_avwap_lib" / "legacy.py").read_text(encoding="utf-8")
    assert source.count('"tier_source": tier_source,') == 2, (
        "both the pick row and the outcome row must write it"
    )


# ---------------------------------------------------------------------------
# (c) the baseline is looked up by NAME
# ---------------------------------------------------------------------------


def test_the_baseline_lookup_survives_a_prepended_group_field():
    """`extra_group_fields` PREPENDS, so positions 0 and 1 stop being
    (side, priority_bucket) and every edge in the finer views ships blank.

    Scoped to `_build_tracker_attribute_leaderboard_rows`, which is the ONLY
    builder whose key can grow: the other three positional lookups in this file
    each build their own fixed tuple and are correct as they are. Banning the
    pattern file-wide would have made three fine call sites look like defects.
    """
    source = (ROOT / "scripts" / "master_avwap_lib" / "legacy.py").read_text(encoding="utf-8")
    start = source.index("def _build_tracker_attribute_leaderboard_rows(")
    end = source.index(chr(10) + "def ", start + 1)
    body = source[start:end]
    assert "baseline_map.get((group_key[0], group_key[1])" not in body, (
        "positional lookup is the defect"
    )
    assert "named = dict(zip(group_cols, group_key))" in body


def test_the_finer_views_report_a_populated_edge():
    """The reproduction, on the branch's OWN B1 fixture.

    With an extra field prepended, the positional lookup asked
    `baseline_map[(setup_family, side)]` for a map keyed `(side, bucket)`, so
    every baseline came back empty and every edge column in the by-family and
    by-regime views shipped blank. A blank edge reads as "no edge", which is a
    worse failure than a crash.
    """
    import copy
    import json

    from master_avwap_lib import legacy

    fixture = json.loads(
        (ROOT / "tests" / "fixtures" / "p4_attribute_leaderboard_v1.json").read_text(
            encoding="utf-8"
        )
    )
    rows = copy.deepcopy(fixture["attribute_rows"])
    for index, row in enumerate(rows):
        row["setup_family"] = "avwape_to_first_dev" if index < 30 else "post_earnings_candle_break"
        row["market_regime_label"] = "bullish_weak" if index % 2 else "bearish_weak"

    default = legacy._build_tracker_attribute_leaderboard_rows(copy.deepcopy(rows))
    assert any(
        row.get("baseline_avg_total_r") not in (None, "") for row in default
    ), "the default view must have baselines, or the fixture proves nothing"

    for field in ("setup_family", "market_regime_label"):
        view = legacy._build_tracker_attribute_leaderboard_rows(
            copy.deepcopy(rows), extra_group_fields=(field,)
        )
        assert view, field
        populated = [
            row for row in view if row.get("baseline_avg_total_r") not in (None, "")
        ]
        assert populated, f"{field}: every baseline blank - the positional lookup is back"


# ---------------------------------------------------------------------------
# The two "also" items
# ---------------------------------------------------------------------------


def test_the_panel_believes_the_files_own_floor_verdict():
    """Two floors that can disagree is one floor too many.

    B1 made the export state `meets_n_floor`; the panel recomputed it. A file
    written before B1 has no column and still falls back to the comparison.
    """
    from ui.panels.setup_tracker_panel import _rank_attribute_leaderboard

    stated_below = {
        "attribute_label": "trend", "value_label": "up",
        "closed_tradeable_setup_count": "500", "avg_closed_r_edge": "0.4",
        "meets_n_floor": "0",
    }
    stated_ok = {
        "attribute_label": "trend", "value_label": "down",
        "closed_tradeable_setup_count": "1", "avg_closed_r_edge": "0.9",
        "meets_n_floor": "1",
    }
    legacy_row = {
        "attribute_label": "trend", "value_label": "sideways",
        "closed_tradeable_setup_count": "2", "avg_closed_r_edge": "0.1",
    }

    ranked = {row["value_label"]: row for row in _rank_attribute_leaderboard(
        [stated_below, stated_ok, legacy_row]
    )}
    assert ranked["up"]["_meets_floor"] is False, "the file said below floor"
    assert ranked["down"]["_meets_floor"] is True, "the file said ok"
    assert ranked["sideways"]["_meets_floor"] is False, "no column: fall back"


def test_the_setup_tracker_panels_shutdown_is_actually_called():
    """It owns a reader thread and nothing joined it."""
    source = (ROOT / "scripts" / "ui" / "panels" / "research_panel.py").read_text(
        encoding="utf-8"
    )
    assert "self.setup_tracker_panel.shutdown()" in source
