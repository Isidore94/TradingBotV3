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
