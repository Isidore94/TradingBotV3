"""P4 Half B: grade what shipped. Each item ships behind a frozen fixture.

Half B changes MEASUREMENT ARTIFACTS - the files the desk and the offline tuner
read to decide what is working. None of them changes a live weight, and the
goldens here are what says so: every one was frozen from the code as it stood
BEFORE its item, with `scripts/` stashed, and each is replayed rather than
compared against rows a test built for itself.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from master_avwap_lib import legacy  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _clean(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _contract(name: str):
    """The repo's OWN contract loader - never a second hashing rule here."""
    from conftest import validate_fixture_contract

    return validate_fixture_contract(_fixture(name), name)


# ==========================================================================
# B1 - the sample floor reaches the attribute leaderboard
# ==========================================================================
B1 = "p4_attribute_leaderboard_v1"

#: The only two columns B1 adds. Named here so a third one cannot appear
#: without this test saying so.
B1_NEW_COLUMNS = {"meets_n_floor", "evidence_label"}


def test_b1_fixture_replays_the_inputs_it_was_frozen_with():
    """Validated through the repo's Milestone 3 contract, so an edited input is
    a changed hash rather than a silently different golden."""
    contract = _contract(B1)
    assert contract.raw_input_digest() == contract.data["raw_input_sha256"]


def test_b1_adds_only_the_floor_columns_and_changes_no_existing_value():
    """The contract. `analyze_master_avwap_scoring.py --apply` reads this file
    into live scoring weights, so an evidence column that moved an existing
    number would move a weight.

    Fail-before-fix: `meets_n_floor` is not a column at all.
    """
    fixture = _fixture(B1)
    rows = legacy._build_tracker_attribute_leaderboard_rows(
        copy.deepcopy(fixture["attribute_rows"])
    )
    frozen = fixture["leaderboard_rows"]

    assert len(rows) == len(frozen)
    for actual, expected in zip(_clean(rows), frozen):
        assert set(actual) - set(expected) == B1_NEW_COLUMNS
        for key, value in expected.items():
            assert actual[key] == value, f"{key} moved: {value!r} -> {actual[key]!r}"


def test_b1_marks_both_sides_of_the_floor():
    """Only numeric bucketing had a floor; categorical, bool and list rows were
    emitted at setup_count=1 with full averages and full edges. On the live
    export that is 37,049 of 38,617 groups."""
    fixture = _fixture(B1)
    rows = legacy._build_tracker_attribute_leaderboard_rows(
        copy.deepcopy(fixture["attribute_rows"])
    )
    by_value = {str(row["value_label"]): row for row in rows}

    assert by_value["lucky_zone"]["meets_n_floor"] == "0", "one setup is not a finding"
    assert by_value["up"]["meets_n_floor"] == "1", "40 closed setups clears the floor"
    assert by_value["down"]["meets_n_floor"] == "0", "20 closed setups does not"
    for row in rows:
        assert row["evidence_label"]


def test_b1_keeps_every_row():
    """Visibility, not suppression. This file is the tuner's input, and its own
    --min-setups / --min-closed-setups gates stay the thing that decides what
    may influence scoring."""
    fixture = _fixture(B1)
    rows = legacy._build_tracker_attribute_leaderboard_rows(
        copy.deepcopy(fixture["attribute_rows"])
    )
    assert len(rows) == len(fixture["leaderboard_rows"])
    assert any(row["meets_n_floor"] == "0" for row in rows), "the thin row is still here"


def test_b1_leaves_the_tuner_recommending_exactly_what_it_did():
    """The reason the fixture froze the TUNER's output and not just the file."""
    pd = pytest.importorskip("pandas")
    import analyze_master_avwap_scoring as tuner

    fixture = _fixture(B1)
    attribute_rows = copy.deepcopy(fixture["attribute_rows"])
    leaderboard = legacy._build_tracker_attribute_leaderboard_rows(attribute_rows)

    baseline = tuner._baseline_by_context(pd.DataFrame(attribute_rows))
    rules = tuner._recommend_attribute_rules(
        pd.DataFrame(leaderboard),
        baseline,
        min_setups=tuner.DEFAULT_MIN_SETUPS,
        min_closed_setups=tuner.DEFAULT_MIN_CLOSED_SETUPS,
    )
    signals = tuner._recommend_signal_changes(
        pd.DataFrame(leaderboard),
        baseline,
        {},
        min_setups=tuner.DEFAULT_MIN_SETUPS,
        min_closed_setups=tuner.DEFAULT_MIN_CLOSED_SETUPS,
    )

    assert _clean(rules) == fixture["tuner_attribute_rules"]
    assert _clean(signals) == fixture["tuner_signal_changes"]

    # The case worth pinning: the tuner writes a rule for a group that is
    # UNDER the reportable-n floor (20 closed setups against a floor of 30),
    # because its OWN gates are min 8 setups / 12 closed. B1 publishes the
    # disagreement and changes nothing about it. Deciding whether the tuner
    # should adopt this floor is a scoring change and its own packet.
    assert any(rule["value"] == "down" for rule in rules), (
        "the fixture must exercise a sub-floor group the tuner still acts on"
    )


def test_b1_floor_comes_from_evidence_stats_not_a_second_copy():
    from evidence_stats import MIN_REPORTABLE_N

    below = legacy._attribute_leaderboard_evidence(
        [0.1] * (MIN_REPORTABLE_N - 1), closed_setups=MIN_REPORTABLE_N - 1
    )
    at = legacy._attribute_leaderboard_evidence(
        [0.1] * MIN_REPORTABLE_N, closed_setups=MIN_REPORTABLE_N
    )
    assert below["meets_n_floor"] == "0"
    assert at["meets_n_floor"] == "1"


def test_b1_asks_the_floor_of_CLOSED_setups():
    """A group with 200 open setups and 2 closed ones has measured two things,
    and every average and edge on the row is computed over the closed ones."""
    from evidence_stats import MIN_REPORTABLE_N

    mismatched = legacy._attribute_leaderboard_evidence(
        [0.1] * MIN_REPORTABLE_N, closed_setups=2
    )
    assert mismatched["meets_n_floor"] == "0"


def test_b1_never_costs_the_export(monkeypatch):
    """A leaderboard row is evidence about setups; it must not be able to stop
    one being written."""
    import evidence_stats

    def boom(*args, **kwargs):
        raise RuntimeError("statistics module is gone")

    monkeypatch.setattr(evidence_stats, "summarize", boom)
    result = legacy._attribute_leaderboard_evidence([0.1] * 50, closed_setups=50)
    assert result == {"meets_n_floor": "0", "evidence_label": ""}


# ==========================================================================
# B2 - the leaderboard can be read by family and by regime
# ==========================================================================
def test_b2_the_default_view_keeps_its_exact_grain():
    """The offline tuner reads THAT file into live scoring weights, so a
    changed key there would change what the tuner sees without anyone deciding
    to. The finer readings are separate files."""
    fixture = _fixture(B1)
    rows = copy.deepcopy(fixture["attribute_rows"])
    views = legacy.build_tracker_attribute_leaderboard_views(rows)
    default = legacy._build_tracker_attribute_leaderboard_rows(copy.deepcopy(rows))

    assert _clean(views["default"]) == _clean(default)
    assert set(views["default"][0]) == set(default[0])
    # And it is still the frozen shape, column for column.
    assert set(views["default"][0]) - set(fixture["leaderboard_rows"][0]) == B1_NEW_COLUMNS


def test_b2_a_finer_view_splits_the_same_evidence():
    """Fail-before-fix: `build_tracker_attribute_leaderboard_views` does not
    exist and the builder takes no extra grouping."""
    fixture = _fixture(B1)
    rows = copy.deepcopy(fixture["attribute_rows"])
    for index, row in enumerate(rows):
        row["setup_family"] = "avwape_to_first_dev" if index < 30 else "post_earnings_candle_break"
        row["market_regime_label"] = "bullish_weak" if index % 2 else "bearish_weak"

    views = legacy.build_tracker_attribute_leaderboard_views(rows)

    assert len(views["family"]) > len(views["default"]), "a finer key makes more groups"
    assert "setup_family" in views["family"][0]
    assert "market_regime_label" in views["regime"][0]
    # Every row of every view still states its floor (B1) - a finer view has
    # smaller groups by construction, which is exactly why that column travels.
    for name, view in views.items():
        for row in view:
            assert row["meets_n_floor"] in {"0", "1"}, name


def test_b2_columns_are_read_by_name_not_position():
    """`extra_group_fields` PREPENDS to `group_cols`, so positional indices
    would silently shift every column one place the first time a finer view was
    built - a leaderboard whose `side` column held a setup family."""
    fixture = _fixture(B1)
    rows = copy.deepcopy(fixture["attribute_rows"])
    for row in rows:
        row["setup_family"] = "avwape_to_first_dev"

    view = legacy._build_tracker_attribute_leaderboard_rows(
        rows, extra_group_fields=("setup_family",)
    )
    assert view
    for row in view:
        assert row["side"] == "LONG"
        assert row["priority_bucket"] == "favorite_setup"
        assert row["setup_family"] == "avwape_to_first_dev"
        assert row["attribute_key"].startswith(("trend.", "setup."))


def test_b2_an_absent_grouping_field_is_ignored_rather_than_crashing():
    fixture = _fixture(B1)
    rows = copy.deepcopy(fixture["attribute_rows"])
    view = legacy._build_tracker_attribute_leaderboard_rows(
        rows, extra_group_fields=("a_column_that_does_not_exist",)
    )
    assert _clean(view) == _clean(
        legacy._build_tracker_attribute_leaderboard_rows(copy.deepcopy(rows))
    )


def test_b2_view_files_are_siblings_named_by_their_dimension():
    from project_paths import MASTER_AVWAP_SETUP_ATTRIBUTE_LEADERBOARD_FILE as base

    for name in legacy.ATTRIBUTE_LEADERBOARD_VIEWS:
        path = legacy.attribute_leaderboard_view_path(name)
        assert path.parent == base.parent
        assert path.name == f"{base.stem}_by_{name}{base.suffix}"
        assert path != base


# ==========================================================================
# B3 - fictional horizons leave the scan-factor leaderboard
# ==========================================================================
def _scan_factor_inputs(observations):
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                # `_prepare_scan_factor_history_frame` builds its own private
                # columns, so the fixture supplies the PUBLIC ones it reads:
                # symbol, last_close and a date column.
                "symbol": obs["symbol"],
                "side": obs["side"],
                "last_close": 100.0,
                "last_trade_date": obs["scan_date"],
                "trend_20d": "up",
            }
            for obs in observations
        ]
    )
    return frame, observations


def _observation(index, *, stale, side_return, symbol=None):
    return {
        "observation_id": f"obs-{index}",
        "scan_row_id": f"{(symbol or f'S{index}')}:2026-08-03",
        "scan_date": "2026-08-03",
        "symbol": symbol or f"S{index}",
        "side": "LONG",
        "horizon_sessions": 5,
        "side_return_pct": side_return,
        "raw_return_pct": side_return,
        "spy_relative_side_return_pct": None,
        "win": side_return > 0,
        "future_scan_date": "2026-08-10",
        "stale_horizon": stale,
    }


def test_b3_a_stale_horizon_row_is_dropped_and_counted():
    """`future_idx = idx + horizon` indexes this SYMBOL'S OWN scan rows, not
    exchange sessions: live medians are horizon 5 -> 64 sessions and horizon 10
    -> 73, with 42-45% of rows spanning more than twice their declared horizon.
    `stale_horizon` has been computed since R10.D and nothing filtered on it.

    Fail-before-fix: the stale row is inside the averages and no count is
    published.
    """
    pytest.importorskip("pandas")

    observations = [_observation(index, stale=False, side_return=1.0) for index in range(12)]
    observations.append(_observation(99, stale=True, side_return=99.0, symbol="ZZZ"))
    frame, obs = _scan_factor_inputs(observations)

    rows = legacy.build_scan_factor_leaderboard_rows(
        frame, obs, min_observations=1, reference_date="2026-08-03"
    )

    assert rows, "the leaderboard must still be built"
    row = rows[0]
    assert row["stale_horizon_observations_dropped"] == 1
    assert row["observations_before_stale_filter"] == 13
    assert "dropped" in row["stale_horizon_drop_note"]
    # The 99% stale row is not inside the average.
    assert row["avg_side_return_pct"] == pytest.approx(1.0)


def test_b3_an_unmeasurable_drift_is_KEPT():
    """Uncertainty is not grounds for deletion: `stale_horizon` is None when
    the drift could not be measured, and None must not match the drop."""
    pytest.importorskip("pandas")

    observations = [_observation(index, stale=None, side_return=2.0) for index in range(6)]
    frame, obs = _scan_factor_inputs(observations)

    rows = legacy.build_scan_factor_leaderboard_rows(
        frame, obs, min_observations=1, reference_date="2026-08-03"
    )
    assert rows
    assert rows[0]["stale_horizon_observations_dropped"] == 0
    assert rows[0]["observation_count"] == 6


def test_b3_does_not_reselect_the_future_row():
    """Step (a) ONLY. Re-selecting the future row by exchange session would
    redefine every historical number the tracker has produced, and the
    2026-07-01 signal weights were justified against this file - that is its own
    plan.md sec-7 promotion, not this packet."""
    import inspect

    source = inspect.getsource(legacy.build_scan_factor_observation_rows)
    assert "future_idx = idx + horizon" in source, (
        "the horizon indexing is deliberately unchanged in this packet"
    )


# ==========================================================================
# B4 - the tier tracker grades the tier that shipped
# ==========================================================================
def test_b4_the_assigned_tier_is_written_at_assignment_time():
    """`_tier_for_priority_bucket` re-derives S/A from the bucket alone, so the
    tracker graded a tier the trader never saw. The shipped tier is decided
    after the expected-R demote, the per-symbol de-dupe and the best-swing
    merge.

    Fail-before-fix: no row carries `assigned_tier`.
    """
    best = {"symbol": "AAA", "score": 120, "expected_r": 0.5, "priority_bucket": "favorite_setup"}
    actionable = {"symbol": "BBB", "score": 100, "expected_r": 0.4, "priority_bucket": "favorite_setup"}
    report = {"symbol": "CCC", "score": 60, "expected_r": 0.1, "priority_bucket": "near_favorite_zone"}

    sections = legacy._priority_partition_tier_rows(
        actionable_rows=[actionable],
        report_rows=[report],
        high_conviction_rows=[],
        best_swing_rows=[best],
    )

    by_label = {section["label"]: section["rows"] for section in sections}
    assert by_label["S Tier"][0][legacy.ASSIGNED_TIER_FIELD] == "S"
    assert by_label["A Tier"][0][legacy.ASSIGNED_TIER_FIELD] == "A"
    assert by_label["B Tier"][0][legacy.ASSIGNED_TIER_FIELD] == "B"


def test_b4_a_demoted_row_is_not_stamped_S():
    """The exact disagreement B4 exists for: a favorite-bucket row held out of
    S/A for a poor expected R derives as "S" and shipped as nothing."""
    demoted = {
        "symbol": "AAA",
        "score": 120,
        "expected_r": legacy.TIER_S_DEMOTE_EXPECTED_R_BELOW - 1.0,
        "priority_bucket": "favorite_setup",
    }
    sections = legacy._priority_partition_tier_rows(
        actionable_rows=[],
        report_rows=[],
        high_conviction_rows=[],
        best_swing_rows=[demoted],
    )
    by_label = {section["label"]: section["rows"] for section in sections}

    assert by_label["S Tier"] == []
    assert demoted.get(legacy.ASSIGNED_TIER_FIELD) is None
    # The derivation, which is what the tracker used to grade it by, still
    # says S - that is the number this packet makes visible.
    assert legacy._tier_for_priority_bucket(demoted["priority_bucket"]) == "S"
    assert legacy.tier_for_tracker_row(demoted) == ("S", "derived_from_bucket")


def test_b4_the_grader_prefers_the_assigned_tier_and_says_which():
    assert legacy.tier_for_tracker_row({"assigned_tier": "A", "priority_bucket": "favorite_setup"}) == (
        "A",
        "assigned",
    )
    assert legacy.tier_for_tracker_row({"priority_bucket": "favorite_setup"}) == (
        "S",
        "derived_from_bucket",
    )
    assert legacy.tier_for_tracker_row({"priority_bucket": "general"}) == ("", "derived_from_bucket")


def test_b4_the_fallback_is_kept_for_the_months_of_rows_without_the_column():
    """Grading old rows by the old rule is honest about what could be known;
    inventing an assigned tier for them would not be."""
    import inspect

    source = inspect.getsource(legacy.tier_for_tracker_row)
    assert "_tier_for_priority_bucket" in source


# ==========================================================================
# B5 - calibration reads structure points, not the proven-quality score
# ==========================================================================
def test_b5_the_record_stores_the_structure_points():
    """`apply_expected_r_ranking` overwrites `row["score"]` with the
    proven-quality score - which already has realized win rate and profit
    factor blended in - and keeps the pre-blend structure points in
    `row["static_score"]`. The record stored only the overwritten one, so the
    calibration fitted realized R against a number that already contained
    realized performance: a feedback loop.

    Fail-before-fix: `static_score` is not on the record.
    """
    record = legacy.build_tracker_setup_record(
        {
            "symbol": "AAA",
            "side": "LONG",
            "score": 64.0,
            "static_score": 120.0,
            "priority_bucket": "favorite_setup",
        },
        {"last_close": 50.0},
        {},
        "2026-09-01T00:00:00",
        None,
        scan_date="2026-09-01",
    )
    assert record is not None
    assert record["priority_score"] == 64.0
    assert record["static_score"] == 120.0


def test_b5_the_helper_prefers_the_stored_structure_points():
    stored = legacy._expected_r_static_points_from_record(
        {"priority_score": 64.0, "static_score": 120.0, "recent_tracker_score_delta": 5}
    )
    assert stored == 120.0, "the stored points are used as-is, deltas already excluded"


def test_b5_an_old_record_still_calibrates_by_the_old_path():
    """Months of records have no `static_score`. Approximating their structure
    points is honest about what can be recovered; refusing them would throw
    away the calibration sample."""
    derived = legacy._expected_r_static_points_from_record(
        {"priority_score": 100.0, "recent_tracker_score_delta": 5, "setup_type_score_delta": 3}
    )
    assert derived == 92.0


def test_b5_reports_how_much_of_each_path_the_sample_used():
    """The two paths fit different things, so a run that mixes them is fitting a
    blend of two definitions. The changeover has to be visible."""
    counts = legacy.expected_r_calibration_source_counts(
        {
            "setups": {
                "a": {"static_score": 120.0, "priority_score": 64.0},
                "b": {"priority_score": 90.0},
                "c": {},
            }
        }
    )
    assert counts == {
        "stored_static_score": 1,
        "derived_from_priority_score": 1,
        "unusable": 1,
    }


# ==========================================================================
# B6 - the representative exit template is named
# ==========================================================================
def test_b6_the_default_moves_nothing():
    """An empty constant means "first match in scenario order", which is exactly
    what the code has always done."""
    assert legacy.REPRESENTATIVE_EXIT_TEMPLATE_ID == ""

    scenarios = [
        {"stop_reference_label": "LOWER_1", "exit_template_id": "full_band2", "tradeable": True},
        {"stop_reference_label": "LOWER_1", "exit_template_id": "half_band2_trail", "tradeable": True},
    ]
    assert legacy._representative_scenario(scenarios, "LOWER_1") is scenarios[0]


def test_b6_a_pinned_template_is_chosen_explicitly(monkeypatch):
    monkeypatch.setattr(legacy, "REPRESENTATIVE_EXIT_TEMPLATE_ID", "half_band2_trail")
    scenarios = [
        {"stop_reference_label": "LOWER_1", "exit_template_id": "full_band2", "tradeable": True},
        {"stop_reference_label": "LOWER_1", "exit_template_id": "half_band2_trail", "tradeable": True},
    ]
    assert legacy._representative_scenario(scenarios, "LOWER_1") is scenarios[1]


def test_b6_a_pinned_template_nobody_carries_falls_back_rather_than_vanishing(monkeypatch):
    """A setup with no representative R falls through to the cross-variant
    average, which is the number this selection exists to avoid."""
    monkeypatch.setattr(legacy, "REPRESENTATIVE_EXIT_TEMPLATE_ID", "a_template_nobody_has")
    scenarios = [
        {"stop_reference_label": "LOWER_1", "exit_template_id": "full_band2", "tradeable": True},
    ]
    assert legacy._representative_scenario(scenarios, "LOWER_1") is scenarios[0]


def test_b6_no_matching_stop_label_is_still_nothing():
    assert legacy._representative_scenario(
        [{"stop_reference_label": "UPPER_1", "tradeable": True}], "LOWER_1"
    ) is None


def test_b6_the_expected_r_note_names_its_template():
    """The realized half of Expected R is the representative scenario's R, and
    which EXIT PLAN that is was never stated. Fail-before-fix: the note ends at
    the sample count."""
    note = legacy._format_expected_r_note(
        {"expected_r": 0.4, "prior_r": 0.4, "realized_r": None, "blend_weight": 0.0}
    )
    assert "exit template" in note
    assert legacy.representative_exit_template_label() in note


def test_b6_the_summary_reports_the_template_it_measured():
    summary = legacy._summarize_tracker_setup_outcome(
        {
            "side": "LONG",
            "favorite_signals": [],
            "scenarios": {
                "a": {
                    "stop_reference_label": legacy._protective_band_label("LONG"),
                    "exit_template_id": "full_band2",
                    "tradeable": True,
                    "status": "closed",
                    "total_r": 0.8,
                },
            },
        }
    )
    assert summary["representative_exit_template_id"] == "full_band2"
