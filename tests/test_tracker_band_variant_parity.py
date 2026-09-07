"""Phase 0.10 B-2 - the champion's tracker record does not move.

The fixture was frozen on the code as it stood BEFORE the band-variant shadow
block (`scripts/build_tracker_band_variant_parity_fixture.py`, plan.md
section 5). Every key it records must still hold its exact value with the shadow
present. New keys are allowed - that is the whole point of a shadow block - but
a moved value is a champion change wearing an evidence-only label, and this file
exists to make that impossible to ship by accident.

Two halves:

* the parity half, which is a pure regression pin and passes both before and
  after the edit;
* the shadow half, which describes what the block must DO and fails until it
  exists.

Why parity is not obvious here: `_summarize_tracker_setup_outcome` averages
`total_r` across every tradeable non-experimental scenario, and that average
feeds `build_tracker_setup_type_rows` -> `apply_tracker_setup_type_adjustments`
-> `row["score"]`. A shadow stop appended to the same `scenarios` dict without a
fence therefore changes a LIVE PRIORITY SCORE. Measured on this fixture before
the fence existed: eight of the sixteen recorded summary values moved, including
`avg_total_r` (-0.0790 -> -0.0755) and `tradeable_scenario_count` (8 -> 12),
while `representative_total_r` stayed put. Appending after the champion's
candidates is necessary and is not sufficient.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from conftest import load_fixture_contract

ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_tracker_band_variant_parity_fixture import measure  # noqa: E402
from master_avwap_lib import execution_convention as ec  # noqa: E402
from master_avwap_lib import selection_policy as sp  # noqa: E402

FIXTURE = "tracker_record_band_variant_parity_v1"
VARIANT_LABELS = {"VARIANT_LOWER_1", "VARIANT_UPPER_1"}

#: The policies this fixture was FROZEN under (2026-08-26). Packet ST7
#: (2026-09-06, decision 0019) moved all three defaults, so the parity read
#: names them: a characterization re-read under a policy it was not frozen on
#: measures a different thing and proves nothing about the shadow fence, which
#: is what this file exists for. The `measured_default` fixture below re-runs
#: the same fence under whatever the defaults currently are, so neither the
#: frozen pin nor today's behaviour goes unchecked.
FROZEN_POLICIES = {
    "execution_convention": ec.EXECUTION_LITERAL_LEVEL_V1,
    "level_knowledge": ec.LEVEL_KNOWLEDGE_SAME_SESSION_V1,
    "selection_policy": sp.SELECTION_CLOSED_FIRST_V1,
}


@pytest.fixture(scope="module")
def contract():
    return load_fixture_contract(FIXTURE)


@pytest.fixture(scope="module")
def measured(contract):
    rules = contract["rules_under_test"]
    bars = list(contract["bars"])
    return {
        "long": measure(bars, dict(rules["long_row"]), **FROZEN_POLICIES),
        "short": measure(bars, dict(rules["short_row"]), **FROZEN_POLICIES),
    }


@pytest.fixture(scope="module")
def measured_default(contract):
    """The same recipe under the CURRENT defaults - no policy named."""
    rules = contract["rules_under_test"]
    bars = list(contract["bars"])
    return {
        "long": measure(bars, dict(rules["long_row"])),
        "short": measure(bars, dict(rules["short_row"])),
    }


def _assert_subset(expected, actual, path: str, tolerance: float) -> None:
    """Every key/value in ``expected`` survives unchanged in ``actual``.

    Deliberately one-directional: an ADDED key is the shadow block doing its
    job, a changed one is a champion change.
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected an object, got {type(actual).__name__}"
        for key, value in expected.items():
            assert key in actual, f"{path}.{key} disappeared"
            _assert_subset(value, actual[key], f"{path}.{key}", tolerance)
        return
    if isinstance(expected, list):
        assert isinstance(actual, list), f"{path}: expected a list, got {type(actual).__name__}"
        assert len(actual) == len(expected), (
            f"{path}: list length moved {len(expected)} -> {len(actual)}"
        )
        for index, value in enumerate(expected):
            _assert_subset(value, actual[index], f"{path}[{index}]", tolerance)
        return
    if isinstance(expected, bool) or isinstance(actual, bool):
        assert actual == expected, f"{path}: {expected!r} -> {actual!r}"
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        assert actual == pytest.approx(expected, abs=tolerance), f"{path}: {expected} -> {actual}"
        return
    assert actual == expected, f"{path}: {expected!r} -> {actual!r}"


# ---------------------------------------------------------------------------
# Parity. Passes before and after the shadow block; fails if the champion moves.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["long", "short"])
def test_every_pre_existing_record_key_is_unchanged(contract, measured, side):
    _assert_subset(
        contract["expected"][side]["record"],
        measured[side]["record"],
        f"{side}.record",
        contract.tolerance,
    )


@pytest.mark.parametrize("side", ["long", "short"])
def test_the_outcome_summary_that_feeds_scoring_is_unchanged(contract, measured, side):
    """These sixteen values reach `row["score"]`. None of them may move."""
    _assert_subset(
        contract["expected"][side]["outcome_summary"],
        measured[side]["outcome_summary"],
        f"{side}.outcome_summary",
        contract.tolerance,
    )


@pytest.mark.parametrize("side", ["long", "short"])
def test_the_champions_own_scenarios_are_neither_added_to_nor_reordered(
    contract, measured, side
):
    expected = contract["expected"][side]["record"]["scenarios"]
    actual = measured[side]["record"]["scenarios"]
    champion_ids = [
        key
        for key in actual
        if str(actual[key].get("stop_reference_label") or "") not in VARIANT_LABELS
    ]
    assert champion_ids == list(expected), (
        "the champion's scenario ids changed or moved: "
        f"{list(expected)} -> {champion_ids}"
    )


@pytest.mark.parametrize("side", ["long", "short"])
def test_the_shadow_scenarios_come_last(contract, measured, side):
    """A shadow stop must never displace a champion stop in the ordering.

    `_summarize_tracker_setup_outcome` picks the representative scenario by
    label, so ordering does not decide it today - but `setup_status` and every
    "first tradeable" reader do, and appending is what keeps them honest.
    """
    labels = [
        str(scenario.get("stop_reference_label") or "")
        for scenario in measured[side]["record"]["scenarios"].values()
    ]
    variant_positions = [index for index, label in enumerate(labels) if label in VARIANT_LABELS]
    if not variant_positions:
        pytest.skip("no shadow scenarios on this record")
    assert min(variant_positions) == len(labels) - len(variant_positions), (
        f"shadow scenarios are interleaved with the champion's: {labels}"
    )


@pytest.mark.parametrize("side", ["long", "short"])
def test_the_shadow_fence_holds_under_the_current_defaults_too(
    contract, measured_default, side
):
    """ST7 moved the replay's defaults; the FENCE is what must not move.

    The frozen numbers above are a v1 characterization and are read under v1 by
    name. This test asserts the property the fixture actually protects - a
    band-variant scenario never enters a champion aggregate and never displaces
    a champion scenario - under whatever the defaults are today, so a future
    policy change cannot quietly unfence the shadow.
    """
    expected_ids = list(contract["expected"][side]["record"]["scenarios"])
    scenarios = measured_default[side]["record"]["scenarios"]
    labels = [
        str(scenario.get("stop_reference_label") or "")
        for scenario in scenarios.values()
    ]
    champion_ids = [
        key
        for key in scenarios
        if str(scenarios[key].get("stop_reference_label") or "") not in VARIANT_LABELS
    ]
    assert champion_ids == expected_ids
    variant_positions = [i for i, label in enumerate(labels) if label in VARIANT_LABELS]
    assert variant_positions, "no shadow scenario was built under the defaults"
    assert min(variant_positions) == len(labels) - len(variant_positions)

    # The champion aggregate counts the champion's own non-experimental
    # scenarios and no shadow one. The COUNT is a fence fact, not a v1 number:
    # ST7 moved which fills and levels the replay books, never how many
    # scenarios a record carries.
    summary = measured_default[side]["outcome_summary"]
    frozen = contract["expected"][side]["outcome_summary"]
    assert int(summary["tradeable_scenario_count"]) == int(
        frozen["tradeable_scenario_count"]
    )


# ---------------------------------------------------------------------------
# The shadow block itself. Fails until B-2 lands.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["long", "short"])
def test_the_record_carries_both_anchor_variant_blocks(measured, side):
    record = measured[side]["record"]
    for key in ("current_anchor_variant", "previous_anchor_variant"):
        assert key in record, f"{key} missing from the setup record"
        assert isinstance(record[key], dict)
    current = record["current_anchor_variant"]
    assert current["formula_version"] == "avwap_bands_oneoption_bb20_v1"
    assert current["vwap"] is not None
    assert current["stdev"] is not None
    assert set(current["bands"]) == {
        "UPPER_1",
        "LOWER_1",
        "UPPER_2",
        "LOWER_2",
        "UPPER_3",
        "LOWER_3",
    }
    # There is no previous anchor on this fixture, so the mirror block has to
    # say so rather than invent one.
    assert record["previous_anchor_variant"].get("bands") == {}
    assert record["previous_anchor_variant"].get("reason")


def test_a_long_gets_the_lower_shadow_stop_and_a_short_the_upper(measured):
    long_labels = {
        str(scenario.get("stop_reference_label") or "")
        for scenario in measured["long"]["record"]["scenarios"].values()
    }
    short_labels = {
        str(scenario.get("stop_reference_label") or "")
        for scenario in measured["short"]["record"]["scenarios"].values()
    }
    assert "VARIANT_LOWER_1" in long_labels and "VARIANT_UPPER_1" not in long_labels
    assert "VARIANT_UPPER_1" in short_labels and "VARIANT_LOWER_1" not in short_labels


@pytest.mark.parametrize("side", ["long", "short"])
def test_the_shadow_scenarios_are_tagged_and_graded(measured, side):
    scenarios = [
        scenario
        for scenario in measured[side]["record"]["scenarios"].values()
        if str(scenario.get("stop_reference_label") or "") in VARIANT_LABELS
    ]
    assert scenarios, "no shadow scenario was built"
    for scenario in scenarios:
        assert scenario["stop_source_type"] == "band_variant"
        # Graded by the existing per-bar machinery, with no further edit.
        assert scenario["status"] not in ("", None)
        assert scenario["last_action"]


@pytest.mark.parametrize("side", ["long", "short"])
def test_the_challengers_sigma_is_the_wider_one_this_soon_after_an_anchor(measured, side):
    """The study's central claim, pinned as arithmetic on a real record.

    Seven sessions after the anchor the champion's accumulated running
    deviation is 0.586 while the 20-close Bollinger sigma is 1.339 - 2.3x - and
    that gap is exactly why the trader's OneOption screenshots looked better
    early. It is not a rule the code enforces; it is a measurement, and it is
    here so a later "the variant stopped out less often" claim cannot be read
    without it.
    """
    record = measured[side]["record"]
    champion_sigma = record["current_anchor_entry"]["stdev"]
    variant_sigma = record["current_anchor_variant"]["stdev"]
    assert champion_sigma == pytest.approx(0.5857, abs=0.001)
    assert variant_sigma == pytest.approx(1.3390, abs=0.001)
    assert variant_sigma > champion_sigma * 2


def test_a_wider_sigma_does_not_mean_a_further_stop(measured):
    """The trap in the fairness rule, caught on this very fixture.

    "The wider band is stopped out less often by construction" is only true
    when entry sits INSIDE the band. This fixture's short is entered above both
    upper bands, and the wider sigma pushes the upper band UP - toward the
    entry - so the challenger's stop is 0.159 away where the champion's is
    0.971. Six times TIGHTER, from the wider formula.

    So T1/T3 may not assume a direction. Any stop-out comparison has to
    condition on the entry's position relative to the band, and the risk floor
    is what stands between this record and a 3,138-share position.
    """
    entry = measured["short"]["record"]["entry_price"]
    scenarios = measured["short"]["record"]["scenarios"].values()
    variant = next(s for s in scenarios if str(s.get("stop_reference_label")) in VARIANT_LABELS)
    champion = next(
        s
        for s in scenarios
        if str(s.get("stop_reference_label")) == "UPPER_1"
    )
    assert abs(entry - variant["stop_reference_level"]) == pytest.approx(0.1593, abs=0.001)
    assert abs(entry - champion["stop_reference_level"]) == pytest.approx(0.9666, abs=0.001)
    assert abs(entry - variant["stop_reference_level"]) < abs(
        entry - champion["stop_reference_level"]
    )
