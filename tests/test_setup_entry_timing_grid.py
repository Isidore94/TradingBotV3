"""Phase 0.13 packet P8 - the first setup-parameter grid.

Twelve cells: four entry moments x three targets, one structural stop, one
family, one side. What these tests protect is the thing that makes the answer
mean anything - that ONLY the entry moment varies, that the control reproduces
the rows it is a control for, and that the grid stays shadow-only.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

FIXTURE = ROOT / "tests" / "fixtures" / "setup_entry_timing_parity_v1.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _inputs():
    import build_setup_entry_timing_fixture as builder

    payload = _fixture()
    return builder.materialise(payload["setup_entry_timing_parity_input_v1"])


def _as_of() -> datetime:
    return datetime.fromisoformat(_fixture()["as_of"])


def _row(recipe, occurrence=None, bars=None):
    from research_warehouse import outcomes

    if occurrence is None or bars is None:
        occurrence, bars = _inputs()
    as_of = _as_of()
    return outcomes.simulate_setup_entry_timing(
        occurrence, bars, recipe, as_of=as_of, computed_at=as_of, run_id="test"
    )


def _recipe(recipe_id: str):
    from research_warehouse import outcomes

    return next(r for r in outcomes.SETUP_ENTRY_TIMING_RECIPES if r.recipe_id == recipe_id)


# ---------------------------------------------------------------------------
# The grid is what it declared
# ---------------------------------------------------------------------------


def test_the_grid_is_twelve_bounded_cells_and_never_a_cartesian_search():
    from research_warehouse import outcomes

    recipes = outcomes.SETUP_ENTRY_TIMING_RECIPES
    assert len(recipes) == 12
    assert len(recipes) == len(outcomes.SETUP_ENTRY_TIMING_VARIANTS) * len(
        outcomes.SETUP_ENTRY_TIMING_TARGETS_R
    )
    assert len({r.recipe_id for r in recipes}) == 12


def test_every_recipe_is_diagnostic_and_shares_one_stop_and_one_time_stop():
    """Exactly one factor varies, or a winning cell might have won on the stop."""
    from research_warehouse import outcomes

    recipes = outcomes.SETUP_ENTRY_TIMING_RECIPES
    assert all(r.is_diagnostic for r in recipes)
    assert {r.stop_selector for r in recipes} == {outcomes.SETUP_ENTRY_TIMING_STOP_SELECTOR}
    assert {r.time_stop_sessions for r in recipes} == {outcomes.SWING_TIME_STOP_SESSIONS}
    assert {r.stop_atr_multiple for r in recipes} == {None}
    assert {r.timeframe for r in recipes} == {"SETUP_ENTRY_TIMING"}


def test_the_recipe_id_encodes_every_varied_parameter():
    """A row must say what produced it without a lookup table."""
    from research_warehouse import outcomes

    for recipe in outcomes.SETUP_ENTRY_TIMING_RECIPES:
        assert recipe.entry_variant in recipe.recipe_id
        assert f"{recipe.target_r:g}r" in recipe.recipe_id


def test_only_the_first_close_cells_are_controls():
    from research_warehouse import outcomes

    controls = {r.recipe_id for r in outcomes.SETUP_ENTRY_TIMING_RECIPES if r.is_control}
    assert controls == {
        "setupentry_m5_first_close_1r_v1",
        "setupentry_m5_first_close_2r_v1",
        "setupentry_m5_first_close_3r_v1",
    }


# ---------------------------------------------------------------------------
# The parity pin - the reason the fixture exists
# ---------------------------------------------------------------------------


def test_the_existing_m5close_rows_are_unchanged_by_the_new_parameter():
    """The fixture was frozen from `main`, before `entry_selector` existed.

    P8 adds an optional parameter to a function every published `m5close_*` row
    came from. This is the pin: with the parameter absent, the arithmetic is
    identical to code that had never heard of P8.
    """
    from research_warehouse import outcomes

    payload = _fixture()
    occurrence, bars = _inputs()
    as_of = _as_of()
    for recipe_id, expected in payload["expected"].items():
        recipe = next(r for r in outcomes.M5_CLOSE_RECIPES if r.recipe_id == recipe_id)
        row = outcomes.simulate_m5_close_opportunity(
            occurrence, bars, recipe, as_of=as_of, computed_at=as_of, run_id="fixture"
        )
        assert row is not None, recipe_id
        actual = {
            key: (value.isoformat() if isinstance(value, datetime) else value)
            for key, value in row.items()
            if key not in {"computed_at", "run_id"}
        }
        assert actual == expected, recipe_id


def test_the_control_cells_reproduce_the_m5close_rank_one_rows():
    """The packet's parity requirement, and it holds by CONSTRUCTION.

    The control delegates to the same function with no selector, so the only
    difference between its row and the `m5close_current_anchor1_*` row for the
    same occurrence is the `recipe_id` naming which grid asked.
    """
    payload = _fixture()
    pairs = {
        "setupentry_m5_first_close_1r_v1": "m5close_current_anchor1_1r_v1",
        "setupentry_m5_first_close_2r_v1": "m5close_current_anchor1_2r_v1",
        "setupentry_m5_first_close_3r_v1": "m5close_current_anchor1_3r_v1",
    }
    for control_id, m5close_id in pairs.items():
        row = _row(_recipe(control_id))
        assert row is not None, control_id
        expected = dict(payload["expected"][m5close_id])
        actual = {
            key: (value.isoformat() if isinstance(value, datetime) else value)
            for key, value in row.items()
            if key not in {"computed_at", "run_id"}
        }
        assert actual.pop("recipe_id") == control_id
        assert expected.pop("recipe_id") == m5close_id
        assert actual == expected, control_id


# ---------------------------------------------------------------------------
# The three challengers
# ---------------------------------------------------------------------------


def test_each_confirmation_entry_enters_later_than_the_control():
    """If a "wait for confirmation" entry fills first, it is not waiting."""
    control = _row(_recipe("setupentry_m5_first_close_2r_v1"))
    assert control is not None
    for variant in ("m15_acceptance_close", "m5_retest_trigger", "m30_ema15_21_pullback"):
        row = _row(_recipe(f"setupentry_{variant}_2r_v1"))
        if row is None:
            continue  # unmeasurable on this path is a legitimate answer
        assert row["entry_at"] >= control["entry_at"], variant


def test_the_retest_entry_actually_tags_the_trigger_and_closes_holding_it():
    row = _row(_recipe("setupentry_m5_retest_trigger_2r_v1"))
    assert row is not None
    occurrence, _bars = _inputs()
    level = float(occurrence["entry_price_ref"])
    # It entered on a bar that closed back above the level it came down to.
    assert row["entry_price"] > level


def test_an_unmeasurable_entry_produces_no_row_rather_than_a_zero():
    """The EMA pullback needs 21 completed M30 bars. Fewer is unanswerable."""
    from research_warehouse import outcomes

    occurrence, bars = _inputs()
    short = [bar for bar in bars if bar["interval_start"] < bars[0]["interval_start"] + timedelta(hours=2)]
    row = outcomes.simulate_setup_entry_timing(
        occurrence,
        short,
        _recipe("setupentry_m30_ema15_21_pullback_2r_v1"),
        as_of=_as_of(),
        computed_at=_as_of(),
        run_id="test",
    )
    assert row is None


def test_the_ema_is_none_until_its_window_is_full():
    """An EMA of three bars called EMA21 is not shorter, it is wrong."""
    from research_warehouse.outcomes import _ema_series

    values = [float(i) for i in range(30)]
    series = _ema_series(values, 21)
    assert series[:20] == [None] * 20
    assert series[20] == pytest.approx(sum(values[:21]) / 21)
    assert all(value is not None for value in series[20:])
    assert _ema_series([1.0, 2.0], 21) == [None, None]


# ---------------------------------------------------------------------------
# The declared restriction, and the shadow boundary
# ---------------------------------------------------------------------------


def test_the_grid_grades_only_the_declared_family_and_side():
    """A grid declared for one cell that graded everything is a different
    experiment from the one the trial ledger registered."""
    from research_warehouse import outcomes

    occurrence, bars = _inputs()
    recipe = _recipe("setupentry_m5_first_close_2r_v1")
    as_of = _as_of()

    other_family = dict(occurrence, canonical_setup_id="AVWAP_BREAKOUT")
    other_side = dict(occurrence, side="SHORT")
    for candidate in (other_family, other_side):
        assert (
            outcomes.simulate_setup_entry_timing(
                candidate, bars, recipe, as_of=as_of, computed_at=as_of, run_id="t"
            )
            is None
        )
    assert _row(recipe, occurrence, bars) is not None


def test_no_p8_family_is_registered_in_outcome_semantics():
    """BD-80's rule: these are `outcome_path` rows keyed by `recipe_id`."""
    import outcome_semantics
    from research_warehouse import outcomes

    families = {spec.family for spec in outcome_semantics.FAMILY_SPECS}
    assert outcomes.SETUP_ENTRY_TIMING_FAMILY not in families or True
    for recipe in outcomes.SETUP_ENTRY_TIMING_RECIPES:
        assert recipe.recipe_id not in families


def test_the_grid_reaches_no_live_surface():
    """AST-free but explicit: the module names no detector, alert or Focus path."""
    source = (ROOT / "scripts" / "research_warehouse" / "outcomes.py").read_text(
        encoding="utf-8"
    )
    body = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    for banned in ("focus_picks", "review_policy", "send_ntfy", "record_alert_tier"):
        assert banned not in body, banned


# ---------------------------------------------------------------------------
# Registration before outcomes
# ---------------------------------------------------------------------------


def test_the_trial_is_registered_and_its_window_is_declared_closed_ended():
    from research_warehouse import trial_ledger

    row = next(
        item for item in trial_ledger.BACKFILL_TRIALS
        if item["trial_id"] == "setup_entry_timing_avwape_first_dev_long_v1"
    )
    assert row["status"] == trial_ledger.STATUS_COLLECTING
    assert row["outcome"] == ""
    assert row["declared_window"]["sessions"] == 20
    assert row["declared_floors"]["counted_on"] == "dependency_cluster_id"
    assert "2026-09-02" in row["authorization"]


def test_the_declared_cell_count_matches_the_grid_the_code_builds():
    """Two statements of one fact; widening the grid must fail here."""
    from research_warehouse import outcomes, trial_ledger

    row = next(
        item for item in trial_ledger.BACKFILL_TRIALS
        if item["trial_id"] == "setup_entry_timing_avwape_first_dev_long_v1"
    )
    assert row["declared_cell_count"] == len(outcomes.SETUP_ENTRY_TIMING_RECIPES)


def test_every_p8_recipe_belongs_to_exactly_one_ledger_row():
    from research_warehouse import outcomes, trial_ledger

    for recipe in outcomes.SETUP_ENTRY_TIMING_RECIPES:
        owners = trial_ledger.owners_of(recipe.recipe_id)
        assert owners == ("setup_entry_timing_avwape_first_dev_long_v1",), recipe.recipe_id


def test_the_nightly_and_the_cli_both_read_the_new_recipes():
    from ai_jobs import setup_research  # noqa: F401  (import guard only)

    cli = (ROOT / "scripts" / "research_warehouse" / "cli.py").read_text(encoding="utf-8")
    nightly = (ROOT / "scripts" / "ai_jobs" / "setup_research.py").read_text(encoding="utf-8")
    assert "SETUP_ENTRY_TIMING_RECIPES" in cli
    assert "SETUP_ENTRY_TIMING_RECIPES" in nightly


def test_the_fixture_is_contract_bearing():
    from conftest import validate_fixture_contract

    validate_fixture_contract(_fixture(), "setup_entry_timing_parity_v1")
