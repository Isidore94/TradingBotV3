"""Packet 2: fair, bounded entry comparisons over Packet 1 facts.

The comparison layer is deliberately a pure reader of existing P8/M5 outcomes,
Packet 1 forward measures, and the append-only trial ledger.  These contracts
keep a later entrant, a no-trigger, and a thin cell visible rather than letting
only successful simulated exits reach the research surface.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))


def _forward(
    opportunity_id: str,
    variant: str,
    *,
    state: str = "complete",
    mfe: float | None = 2.0,
    mae: float | None = -1.0,
    close: float | None = 0.5,
    time_to_mfe: float | None = 20.0,
    source: str = "observed",
) -> dict:
    """A hand-pinned Packet 1 published row, never an exit-simulator result."""
    return {
        "opportunity_id": opportunity_id,
        "attempt_id": f"{opportunity_id}|{variant}",
        "entry_variant": variant,
        "symbol": opportunity_id.split("|")[1],
        "side": "LONG",
        "trigger_knowledge_time": "2026-09-01T10:00:00-04:00",
        "source_knowledge_basis": source,
        "anchor_knowledge_basis": source,
        "windows": {
            "60m": {
                "state": state,
                "reason": "complete_coverage" if state == "complete" else state,
                "coverage": {"expected_bars": 12, "observed_bars": 12, "missing_bars": 0},
                "mfe_pct": mfe,
                "mae_pct": mae,
                "close_pct": close,
                "time_to_mfe_minutes": time_to_mfe,
            }
        },
    }


def _attempt(
    opportunity_id: str,
    variant: str,
    *,
    state: str = "complete",
    mfe: float | None = 2.0,
    mae: float | None = -1.0,
    close: float | None = 0.5,
    time_to_mfe: float | None = 20.0,
    cluster: str | None = None,
    population: str = "all_scanner",
    session: str = "2026-09-01",
    source: str = "observed",
) -> dict:
    return {
        "opportunity_id": opportunity_id,
        "attempt_id": f"{opportunity_id}|{variant}",
        "dependency_cluster_id": cluster or opportunity_id,
        "entry_variant": variant,
        "window": "60m",
        "state": state,
        "mfe_pct": mfe,
        "mae_pct": mae,
        "close_pct": close,
        "time_to_mfe_minutes": time_to_mfe,
        "symbol": opportunity_id.split("|")[1],
        "session_date": session,
        "population": population,
        "source_knowledge_basis": source,
        "anchor_knowledge_basis": source,
        "coverage": {"expected_bars": 12, "observed_bars": 12, "missing_bars": 0},
    }


def test_p8_declared_variants_make_one_attempt_per_opportunity_and_keep_no_trigger():
    """P8's 12 exit recipes are four entry variants, not 12 independent tries."""
    import entry_comparison
    from research_warehouse.outcomes import SETUP_ENTRY_TIMING_RECIPES

    opportunity = {
        "occurrence_id": "p8|ACME|2026-09-01",
        "symbol": "ACME",
        "side": "LONG",
        "dependency_cluster_id": "p8-cluster-1",
        "canonical_setup_id": "AVWAPE_TO_FIRST_DEV",
    }
    rows = entry_comparison.adapt_p8_attempts(
        [opportunity],
        [
            _forward("p8|ACME|2026-09-01", "m5_first_close"),
            _forward("p8|ACME|2026-09-01", "m15_acceptance_close", state="no_trigger", mfe=None, mae=None, close=None, time_to_mfe=None),
        ],
        recipes=SETUP_ENTRY_TIMING_RECIPES,
    )

    assert {row["entry_variant"] for row in rows} == {
        "m5_first_close", "m15_acceptance_close", "m5_retest_trigger", "m30_ema15_21_pullback"
    }
    assert len(rows) == 4
    assert {row["opportunity_id"] for row in rows} == {"p8|ACME|2026-09-01"}
    assert {row["dependency_cluster_id"] for row in rows} == {"p8-cluster-1"}
    control = next(row for row in rows if row["entry_variant"] == "m5_first_close")
    waited = next(row for row in rows if row["entry_variant"] == "m15_acceptance_close")
    absent = next(row for row in rows if row["entry_variant"] == "m5_retest_trigger")
    assert control["is_control"] is True
    assert waited["state"] == "no_trigger"
    assert absent["state"] == "missing_data", "unknown input is not a fabricated fill"


def test_existing_m5_entry_bearing_occurrences_share_the_normalized_attempt_interface():
    """The same report can read established M5 occurrences without recoding them."""
    import entry_comparison
    from research_warehouse.outcomes import M5_CLOSE_RECIPES

    occurrence = {
        "occurrence_id": "m5|BETA|2026-09-01",
        "symbol": "BETA",
        "side": "LONG",
        "dependency_cluster_id": "m5-cluster-1",
    }
    rows = entry_comparison.adapt_m5_occurrence_attempts(
        [occurrence],
        [_forward("m5|BETA|2026-09-01", "m5_first_close")],
        recipes=M5_CLOSE_RECIPES,
    )

    assert len(rows) == 1
    assert rows[0]["opportunity_id"] == occurrence["occurrence_id"]
    assert rows[0]["attempt_id"] == "m5|BETA|2026-09-01|m5_first_close"
    assert rows[0]["dependency_cluster_id"] == "m5-cluster-1"
    assert rows[0]["population"] == "all_scanner"
    assert rows[0]["window"] == "60m"
    assert rows[0]["mfe_pct"] == pytest.approx(2.0)


def test_m5_adapter_keeps_a_missing_data_attempt_when_a_base_occurrence_has_no_forward_row():
    """A base M5 opportunity with no captured forward row still counts honestly."""
    import entry_comparison
    from research_warehouse.outcomes import M5_CLOSE_RECIPES

    occurrence = {
        "occurrence_id": "m5|GAP|2026-09-01",
        "symbol": "GAP",
        "side": "LONG",
        "dependency_cluster_id": "m5-gap-cluster",
    }
    rows = entry_comparison.adapt_m5_occurrence_attempts(
        [occurrence], [], recipes=M5_CLOSE_RECIPES
    )

    assert len(rows) == 1
    assert rows[0]["opportunity_id"] == "m5|GAP|2026-09-01"
    assert rows[0]["entry_variant"] == "m5_first_close"
    assert rows[0]["state"] == "missing_data"
    assert rows[0]["reason"] == "missing_forward_measure"


def test_summary_reports_full_denominators_distribution_coverage_and_exclusions():
    """The published cell says what is measured, missing, and excluded."""
    import entry_comparison

    attempts = [
        _attempt("o1|AAA", "m5_first_close", mfe=1.0, mae=-0.5, close=0.2, time_to_mfe=10),
        _attempt("o2|BBB", "m5_first_close", mfe=3.0, mae=-1.5, close=0.7, time_to_mfe=30, session="2026-09-02"),
        _attempt("o3|CCC", "m5_first_close", state="no_trigger", mfe=None, mae=None, close=None, time_to_mfe=None, session="2026-09-03"),
        _attempt("o4|DDD", "m5_first_close", state="missing_data", mfe=None, mae=None, close=None, time_to_mfe=None, session="2026-09-04"),
    ]
    summary = entry_comparison.summarise_attempts(
        attempts, useful_move_pct=2.0, min_opportunities=1, min_sessions=1
    )["cells"]["m5_first_close"]

    assert summary["opportunity_count"] == 4
    assert summary["trigger_count"] == 2
    assert summary["trigger_rate"] == pytest.approx(0.5)
    assert summary["measurable_count"] == 2
    assert summary["measurable_coverage"] == pytest.approx(0.5)
    assert summary["missed_opportunities"] == 2
    assert summary["mfe_pct"]["median"] == pytest.approx(2.0)
    assert summary["mfe_pct"]["p10"] == pytest.approx(1.2)
    assert summary["mfe_pct"]["p90"] == pytest.approx(2.8)
    assert summary["mae_pct"]["median"] == pytest.approx(-1.0)
    assert summary["useful_move_frequency"] == pytest.approx(0.5)
    assert summary["close_pct"]["median"] == pytest.approx(0.45)
    assert summary["time_to_mfe_minutes"]["median"] == pytest.approx(20.0)
    assert summary["distinct_sessions"] == 4
    assert summary["distinct_symbols"] == 4
    assert summary["exclusions"] == {"no_trigger": 1, "missing_data": 1}


def test_duplicate_scans_bars_and_recipe_rows_collapse_to_one_dependency_cluster():
    """Three rendered rows from one event are one correlated observation."""
    import entry_comparison

    attempts = [
        _attempt("o1|AAA", "m5_first_close", mfe=2.0, cluster="episode-a"),
        _attempt("o1|AAA", "m5_first_close", mfe=2.0, cluster="episode-a"),  # repeat scan/bar
        _attempt("o1|AAA", "m15_acceptance_close", mfe=3.0, cluster="episode-a"),  # alternate recipe
        _attempt("o2|BBB", "m5_first_close", mfe=1.0, cluster="episode-b", session="2026-09-02"),
    ]
    summary = entry_comparison.summarise_attempts(
        attempts, useful_move_pct=2.0, min_opportunities=1, min_sessions=1
    )

    assert summary["independent_clusters"] == 2
    assert summary["cells"]["m5_first_close"]["opportunity_count"] == 2
    assert summary["cells"]["m5_first_close"]["independent_clusters"] == 2
    assert summary["deduplicated_rows"] == 1


def test_distinct_opportunity_ids_in_one_dependency_cluster_count_once_for_distributions_and_pairs():
    """A re-observed episode cannot earn two distribution samples or paired votes."""
    import entry_comparison

    attempts = [
        _attempt("o1|AAA", "m5_first_close", mfe=2.0, cluster="one-episode"),
        _attempt("o1|AAA", "m15_acceptance_close", mfe=3.0, cluster="one-episode"),
        _attempt("o2|AAA", "m5_first_close", mfe=2.0, cluster="one-episode"),
        _attempt("o2|AAA", "m15_acceptance_close", mfe=3.0, cluster="one-episode"),
    ]
    summary = entry_comparison.summarise_attempts(
        attempts, useful_move_pct=2.0, min_opportunities=1, min_sessions=1
    )
    comparison = entry_comparison.compare_variants(
        attempts,
        baseline="m5_first_close",
        challenger="m15_acceptance_close",
        window="60m",
        min_opportunities=1,
        min_sessions=1,
    )

    assert summary["cells"]["m5_first_close"]["distribution_count"] == 1
    assert summary["cells"]["m5_first_close"]["mfe_pct"]["median"] == pytest.approx(2.0)
    assert comparison["paired_shared_triggered_count"] == 1


def test_public_p8_adapter_refuses_an_unregistered_recipe_or_entry_variant_id():
    """The P8 adapter may read the ledger, but not become an alternate registry."""
    from types import SimpleNamespace

    import entry_comparison

    rogue = SimpleNamespace(recipe_id="rogue_entry_grid_v1", entry_variant="rogue_wait", is_control=False)
    occurrence = {"occurrence_id": "o1|AAA", "symbol": "AAA", "side": "LONG"}
    with pytest.raises(ValueError, match="unauthorized recipe"):
        entry_comparison.adapt_p8_attempts([occurrence], [], recipes=[rogue])


def test_public_m5_adapter_refuses_an_unregistered_recipe_or_entry_variant_id():
    """The M5 adapter has the same ledger gate even though exits do not fan out."""
    from types import SimpleNamespace

    import entry_comparison

    rogue = SimpleNamespace(recipe_id="rogue_entry_grid_v1", entry_variant="rogue_wait", is_control=False)
    occurrence = {"occurrence_id": "o1|AAA", "symbol": "AAA", "side": "LONG"}
    with pytest.raises(ValueError, match="unauthorized recipe"):
        entry_comparison.adapt_m5_occurrence_attempts(
            [occurrence], [_forward("o1|AAA", "rogue_wait")], recipes=[rogue]
        )


def test_paired_improvement_and_all_opportunity_coverage_are_not_the_same_claim():
    """A waiting entry may improve shared fills while serving fewer opportunities."""
    import entry_comparison

    attempts = [
        _attempt("o1|AAA", "m5_first_close", mfe=2.0),
        _attempt("o1|AAA", "m15_acceptance_close", mfe=3.0),
        _attempt("o2|BBB", "m5_first_close", mfe=2.0, session="2026-09-02"),
        _attempt("o2|BBB", "m15_acceptance_close", state="no_trigger", mfe=None, mae=None, close=None, time_to_mfe=None, session="2026-09-02"),
    ]
    comparison = entry_comparison.compare_variants(
        attempts, baseline="m5_first_close", challenger="m15_acceptance_close", window="60m"
    )

    assert comparison["paired_shared_triggered_count"] == 1
    assert comparison["paired_mfe_pct_delta_median"] == pytest.approx(1.0)
    assert comparison["all_opportunity_denominator"] == 2
    assert comparison["baseline_all_opportunity_coverage"] == pytest.approx(1.0)
    assert comparison["challenger_all_opportunity_coverage"] == pytest.approx(0.5)
    assert comparison["winner"] == "not_evaluated", "a paired winner cannot hide a missed opportunity"


def test_population_partitions_and_liked_veto_comparisons_require_matched_inputs():
    """Unreviewed is its own population; unmatched review evidence is not a result."""
    import entry_comparison

    attempts = [
        _attempt("a1|AAA", "m5_first_close", population="all_scanner"),
        _attempt("l1|BBB", "m5_first_close", population="liked", source="observed"),
        _attempt("v1|CCC", "m5_first_close", population="vetoed", source="reconstructed"),
        _attempt("t1|DDD", "m5_first_close", population="actual_trade"),
        _attempt("u1|EEE", "m5_first_close", population="unreviewed"),
        _attempt("v2|FFF", "m5_first_close", population="vetoed", source="reconstructed"),
    ]
    summary = entry_comparison.summarise_attempts(
        attempts, useful_move_pct=2.0, min_opportunities=1, min_sessions=1
    )

    assert set(summary["populations"]) == {"all_scanner", "liked", "vetoed", "actual_trade", "unreviewed"}
    review = summary["review_comparison"]
    assert review["status"] == "not_evaluated"
    assert review["reason"] == "unmatched_timing_source_or_window"
    assert review["liked_count"] == 1
    assert review["vetoed_count"] == 2


def test_liked_veto_comparison_refuses_coverage_or_entry_convention_mismatches():
    """Same source/window is insufficient when the two paths were measured differently."""
    import entry_comparison

    liked = _attempt("l1|AAA", "m5_first_close", population="liked")
    vetoed_coverage = _attempt("v1|BBB", "m5_first_close", population="vetoed")
    vetoed_coverage["coverage"] = {"expected_bars": 12, "observed_bars": 6, "missing_bars": 6}
    coverage_review = entry_comparison.summarise_attempts(
        [liked, vetoed_coverage], useful_move_pct=2.0, min_opportunities=1, min_sessions=1
    )["review_comparison"]

    vetoed_convention = _attempt("v2|CCC", "m5_first_close", population="vetoed")
    liked["entry_convention"] = "next_completed_m5_close_v1"
    vetoed_convention["entry_convention"] = "signal_bar_close_v1"
    convention_review = entry_comparison.summarise_attempts(
        [liked, vetoed_convention], useful_move_pct=2.0, min_opportunities=1, min_sessions=1
    )["review_comparison"]

    assert coverage_review == {
        "status": "not_evaluated",
        "reason": "unmatched_coverage_or_entry_convention",
        "liked_count": 1,
        "vetoed_count": 1,
    }
    assert convention_review == coverage_review


def test_outliers_and_below_floor_or_immature_cells_never_name_a_winner():
    """One huge MFE is sensitivity evidence, not a confident headline."""
    import entry_comparison

    attempts = [
        _attempt("o1|AAA", "m5_first_close", mfe=1.0),
        _attempt("o1|AAA", "m15_acceptance_close", mfe=1000.0),
        _attempt("o2|BBB", "m5_first_close", mfe=1.0, session="2026-09-02"),
        _attempt("o2|BBB", "m15_acceptance_close", mfe=1.0, session="2026-09-02"),
        _attempt("o3|CCC", "m5_first_close", mfe=1.0, session="2026-09-03"),
        _attempt("o3|CCC", "m15_acceptance_close", mfe=1.0, session="2026-09-03"),
    ]
    comparison = entry_comparison.compare_variants(
        attempts,
        baseline="m5_first_close",
        challenger="m15_acceptance_close",
        window="60m",
        min_opportunities=30,
        min_sessions=20,
        trial_status="collecting",
    )

    assert comparison["status"] == "not_evaluated"
    assert comparison["winner"] == "not_evaluated"
    assert comparison["reasons"] == ["below_sample_floor", "below_session_floor", "trial_immature"]
    assert comparison["outlier_sensitivity"]["mean_mfe_pct"] == pytest.approx(334.0)
    assert comparison["outlier_sensitivity"]["median_mfe_pct"] == pytest.approx(1.0)
    assert comparison["outlier_sensitivity"]["headline_eligible"] is False


def test_frozen_declaration_rejects_post_outcome_amendment_and_keeps_failed_history():
    """A result cannot turn a changed plan into prospective confirmation."""
    import entry_comparison

    declaration = {
        "trial_id": "entry-quality-v1",
        "primary_metric": "mfe_pct",
        "window": "60m",
        "baseline": "m5_first_close",
        "variants": ["m5_first_close", "m15_acceptance_close"],
        "meaningful_effect": 0.5,
        "coverage_rule": "complete_or_partial_with_reported_coverage",
        "min_opportunities": 30,
        "min_sessions": 20,
        "evaluation_date": "2026-10-01",
        "failure_criteria": "no paired advantage with coverage loss bounded at 10%",
    }
    frozen = entry_comparison.freeze_declaration(declaration, frozen_at="2026-09-01T00:00:00+00:00")
    assert frozen["status"] == "frozen"
    with pytest.raises(ValueError, match="post-outcome.*new trial"):
        entry_comparison.amend_declaration(
            frozen, {"primary_metric": "close_pct"}, outcome_seen_at="2026-09-15T00:00:00+00:00"
        )
    history = entry_comparison.preserve_trial_history(
        [{**frozen, "status": "abandoned"}, {"trial_id": "old-failed", "status": "concluded", "outcome": "failed"}]
    )
    assert [row["trial_id"] for row in history] == ["entry-quality-v1", "old-failed"]
    assert [row["status"] for row in history] == ["abandoned", "concluded"]


def test_only_ledger_authorized_recipes_run_and_family_lifetime_looks_remain_visible():
    """The comparison cannot become a second registry or reset multiplicity."""
    import entry_comparison
    from research_warehouse import trial_ledger
    from research_warehouse.outcomes import SETUP_ENTRY_TIMING_RECIPES

    recipe_id = SETUP_ENTRY_TIMING_RECIPES[0].recipe_id
    context = entry_comparison.authorized_recipe_context(recipe_id, trial_ledger.BACKFILL_TRIALS)
    assert context["trial_id"] == "setup_entry_timing_avwape_first_dev_long_v1"
    assert context["family_lifetime_variants_examined"] >= len(SETUP_ENTRY_TIMING_RECIPES)
    assert context["multiplicity_contract"]
    with pytest.raises(ValueError, match="unauthorized recipe"):
        entry_comparison.authorized_recipe_context("invent_a_1000_cell_search_v1", trial_ledger.BACKFILL_TRIALS)
    with pytest.raises(ValueError, match="unauthorized recipe"):
        entry_comparison.prepare_authorized_evaluation(
            recipe_ids=[recipe_id, "invent_a_1000_cell_search_v1"], ledger_rows=trial_ledger.BACKFILL_TRIALS
        )
