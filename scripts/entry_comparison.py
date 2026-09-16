"""Pure, bounded comparisons of Packet 1 entry-quality measurements.

This reader is deliberately downstream of both the warehouse simulator and the
new fixed-window measurement.  It does not fetch bars, select a setup, write a
trial, or decide a winner.  Its job is narrower: keep every declared entry
variant on the same opportunity denominator and make the coverage, dependence,
and declared research limits visible.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

import evidence_stats
from research_warehouse import trial_ledger


COMPARISON_SCHEMA = "entry_quality_comparison_v1"
DECLARATION_SCHEMA = "entry_quality_declaration_v1"
DEFAULT_WINDOW = "60m"
ALL_SCANNER = "all_scanner"
NON_TRIGGER_STATES = frozenset({"no_trigger", "missing_data", "invalid_entry", "unavailable", "pending"})
MEASURABLE_STATES = frozenset({"complete", "partial"})


def _text(value: Any) -> str:
    return str(value or "").strip()


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _quantile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _session_date(row: Mapping[str, Any], forward: Mapping[str, Any]) -> str:
    for source in (row, forward):
        value = _text(source.get("session_date"))
        if value:
            return value
        moment = _text(source.get("trigger_knowledge_time"))
        if moment:
            return moment[:10]
    return ""


def _recipe_variants(recipes: Iterable[Any]) -> dict[str, bool]:
    """One entry variant per declared grid axis, never one row per exit recipe."""
    variants: dict[str, bool] = {}
    for recipe in recipes:
        variant = _text(getattr(recipe, "entry_variant", ""))
        if not variant:
            continue
        variants[variant] = variants.get(variant, False) or bool(getattr(recipe, "is_control", False))
    return variants


def _require_authorized_recipes(recipes: Iterable[Any]) -> tuple[Any, ...]:
    """Keep public adapters inside the existing ledger's declared recipe set."""
    declared = tuple(recipes)
    for recipe in declared:
        recipe_id = _text(getattr(recipe, "recipe_id", ""))
        authorized_recipe_context(recipe_id, trial_ledger.BACKFILL_TRIALS)
    return declared


def _forward_index(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        opportunity_id = _text(row.get("opportunity_id"))
        variant = _text(row.get("entry_variant"))
        if opportunity_id and variant:
            # A duplicate published measurement is the same attempt, not another sample.
            indexed.setdefault((opportunity_id, variant), row)
    return indexed


def _normalise_attempt(
    occurrence: Mapping[str, Any],
    *,
    opportunity_id: str,
    variant: str,
    is_control: bool,
    forward: Mapping[str, Any] | None,
    window: str,
) -> dict[str, Any]:
    source = forward or {}
    payload = source.get("windows", {}).get(window, {}) if isinstance(source.get("windows"), Mapping) else {}
    state = _text(payload.get("state")) if isinstance(payload, Mapping) else ""
    if not state:
        state = "missing_data"
    coverage = payload.get("coverage") if isinstance(payload, Mapping) else None
    if not isinstance(coverage, Mapping):
        coverage = {"expected_bars": 0, "observed_bars": 0, "missing_bars": 0}
    return {
        "opportunity_id": opportunity_id,
        "attempt_id": f"{opportunity_id}|{variant}",
        "dependency_cluster_id": _text(occurrence.get("dependency_cluster_id")) or opportunity_id,
        "entry_variant": variant,
        "is_control": bool(is_control),
        "window": window,
        "state": state,
        "reason": _text(payload.get("reason")) if payload else "missing_forward_measure",
        "mfe_pct": payload.get("mfe_pct") if isinstance(payload, Mapping) else None,
        "mae_pct": payload.get("mae_pct") if isinstance(payload, Mapping) else None,
        "close_pct": payload.get("close_pct") if isinstance(payload, Mapping) else None,
        "time_to_mfe_minutes": payload.get("time_to_mfe_minutes") if isinstance(payload, Mapping) else None,
        "symbol": _text(occurrence.get("symbol")) or _text(source.get("symbol")),
        "side": _text(occurrence.get("side")) or _text(source.get("side")),
        "session_date": _session_date(occurrence, source),
        "population": _text(occurrence.get("population")) or ALL_SCANNER,
        "source_knowledge_basis": _text(source.get("source_knowledge_basis")) or _text(occurrence.get("source_knowledge_basis")),
        "anchor_knowledge_basis": _text(source.get("anchor_knowledge_basis")) or _text(occurrence.get("anchor_knowledge_basis")),
        "coverage": dict(coverage),
    }


def adapt_p8_attempts(
    opportunities: Iterable[Mapping[str, Any]],
    forward_rows: Iterable[Mapping[str, Any]],
    *,
    recipes: Iterable[Any],
    window: str = DEFAULT_WINDOW,
) -> list[dict[str, Any]]:
    """Give every bounded P8 entry variant one row for every opportunity.

    P8 has three target recipes for each entry rule.  Targets are not entry
    alternatives here, so their recipe rows collapse to the four predeclared
    entry variants before any comparison is made.
    """
    declared = _recipe_variants(_require_authorized_recipes(recipes))
    if not declared:
        raise ValueError("P8 recipes must declare entry variants")
    indexed = _forward_index(forward_rows)
    attempts: list[dict[str, Any]] = []
    for occurrence in opportunities:
        opportunity_id = _text(occurrence.get("occurrence_id")) or _text(occurrence.get("opportunity_id"))
        if not opportunity_id:
            raise ValueError("P8 occurrence needs an opportunity identity")
        for variant, is_control in declared.items():
            attempts.append(
                _normalise_attempt(
                    occurrence,
                    opportunity_id=opportunity_id,
                    variant=variant,
                    is_control=is_control,
                    forward=indexed.get((opportunity_id, variant)),
                    window=window,
                )
            )
    return attempts


def adapt_m5_occurrence_attempts(
    occurrences: Iterable[Mapping[str, Any]],
    forward_rows: Iterable[Mapping[str, Any]],
    *,
    recipes: Iterable[Any],
    window: str = DEFAULT_WINDOW,
) -> list[dict[str, Any]]:
    """Normalise existing M5 entry-bearing occurrences to the same reader shape.

    The old M5 close recipe library varies exits, not entry timing.  It therefore
    supplies authorization context but is never expanded into one entry attempt
    per exit recipe.
    """
    _require_authorized_recipes(recipes)
    indexed = _forward_index(forward_rows)
    attempts: list[dict[str, Any]] = []
    for occurrence in occurrences:
        opportunity_id = _text(occurrence.get("occurrence_id")) or _text(occurrence.get("opportunity_id"))
        if not opportunity_id:
            raise ValueError("M5 occurrence needs an opportunity identity")
        variants = [variant for (identity, variant) in indexed if identity == opportunity_id]
        if not variants:
            # Existing M5 close recipes share one entry convention.  A missing
            # Packet 1 row is still an attempt in that convention, not absence.
            variants = ["m5_first_close"]
        for variant in dict.fromkeys(variants):
            attempts.append(
                _normalise_attempt(
                    occurrence,
                    opportunity_id=opportunity_id,
                    variant=variant,
                    is_control=variant == "m5_first_close",
                    forward=indexed.get((opportunity_id, variant)),
                    window=window,
                )
            )
    return attempts


def _deduplicate_attempts(attempts: Iterable[Mapping[str, Any]], *, window: str | None = None) -> tuple[list[dict[str, Any]], int]:
    unique: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    duplicates = 0
    for row in attempts:
        copied = dict(row)
        if window is not None and _text(copied.get("window")) != window:
            continue
        key = (
            _text(copied.get("population")) or ALL_SCANNER,
            _text(copied.get("opportunity_id")),
            _text(copied.get("entry_variant")),
            _text(copied.get("window")),
        )
        if not key[1] or not key[2]:
            raise ValueError("an attempt needs opportunity_id and entry_variant")
        if key in unique:
            duplicates += 1
            continue
        unique[key] = copied
    return list(unique.values()), duplicates


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    return {
        "mean": sum(values) / len(values),
        "median": _quantile(values, 0.5),
        "p10": _quantile(values, 0.1),
        "p90": _quantile(values, 0.9),
    }


def _cluster_representatives(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep one stable observation per dependency cluster, never the best one."""
    chosen: dict[str, dict[str, Any]] = {}
    for row in rows:
        cluster = _text(row.get("dependency_cluster_id")) or _text(row.get("opportunity_id"))
        key = (_text(row.get("opportunity_id")), _text(row.get("attempt_id")))
        existing = chosen.get(cluster)
        if existing is None or key < (_text(existing.get("opportunity_id")), _text(existing.get("attempt_id"))):
            chosen[cluster] = row
    return list(chosen.values())


def _cell(rows: list[dict[str, Any]], *, useful_move_pct: float) -> dict[str, Any]:
    opportunity_ids = {_text(row.get("opportunity_id")) for row in rows}
    clusters = {_text(row.get("dependency_cluster_id")) or _text(row.get("opportunity_id")) for row in rows}
    sessions = {_text(row.get("session_date")) for row in rows if _text(row.get("session_date"))}
    symbols = {_text(row.get("symbol")) for row in rows if _text(row.get("symbol"))}
    exclusions = Counter(_text(row.get("state")) or "missing_data" for row in rows if _text(row.get("state")) not in MEASURABLE_STATES)
    measurable = [row for row in rows if _text(row.get("state")) in MEASURABLE_STATES and _finite(row.get("mfe_pct")) is not None]
    distribution_rows = _cluster_representatives(measurable)
    triggered = [row for row in rows if _text(row.get("state")) not in NON_TRIGGER_STATES]
    mfe = [_finite(row.get("mfe_pct")) for row in distribution_rows]
    mae = [_finite(row.get("mae_pct")) for row in distribution_rows]
    close = [_finite(row.get("close_pct")) for row in distribution_rows]
    time_to_mfe = [_finite(row.get("time_to_mfe_minutes")) for row in distribution_rows]
    mfe_values = [value for value in mfe if value is not None]
    mae_values = [value for value in mae if value is not None]
    close_values = [value for value in close if value is not None]
    time_values = [value for value in time_to_mfe if value is not None]
    evidence = evidence_stats.summarize(
        mfe_values,
        symbols=[_text(row.get("symbol")) for row in distribution_rows],
        sessions=[_text(row.get("session_date")) for row in distribution_rows],
        excluded=exclusions,
        clip=None,
    )
    measurable_count = len(measurable)
    denominator = len(opportunity_ids)
    return {
        "opportunity_count": denominator,
        "independent_clusters": len(clusters),
        "trigger_count": len(triggered),
        "trigger_rate": len(triggered) / denominator if denominator else None,
        "measurable_count": measurable_count,
        "distribution_count": len(distribution_rows),
        "measurable_coverage": measurable_count / denominator if denominator else None,
        "missed_opportunities": denominator - measurable_count,
        "mfe_pct": _distribution(mfe_values),
        "mae_pct": _distribution(mae_values),
        "close_pct": _distribution(close_values),
        "time_to_mfe_minutes": _distribution(time_values),
        "useful_move_frequency": (
            sum(value >= useful_move_pct for value in mfe_values) / measurable_count if measurable_count else None
        ),
        "distinct_sessions": len(sessions),
        "distinct_symbols": len(symbols),
        "exclusions": dict(sorted(exclusions.items())),
        "clustered_uncertainty": evidence.get("bootstrap"),
        "evidence_floor": {"n": evidence.get("n"), "min_n": evidence.get("n_floor"), "meets_n_floor": evidence.get("meets_n_floor")},
    }


def _review_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    liked = [row for row in rows if _text(row.get("population")) == "liked"]
    vetoed = [row for row in rows if _text(row.get("population")) == "vetoed"]
    timing_signatures = lambda values: {
        (_text(row.get("window")), _text(row.get("source_knowledge_basis")), _text(row.get("anchor_knowledge_basis")))
        for row in values
    }
    if not liked or not vetoed or not timing_signatures(liked).intersection(timing_signatures(vetoed)):
        return {
            "status": "not_evaluated",
            "reason": "unmatched_timing_source_or_window",
            "liked_count": len(liked),
            "vetoed_count": len(vetoed),
        }
    comparison_signatures = lambda values: {
        (
            _text(row.get("window")),
            _text(row.get("source_knowledge_basis")),
            _text(row.get("anchor_knowledge_basis")),
            int((row.get("coverage") or {}).get("expected_bars") or 0),
            int((row.get("coverage") or {}).get("observed_bars") or 0),
            int((row.get("coverage") or {}).get("missing_bars") or 0),
            _text(row.get("entry_convention")),
        )
        for row in values
    }
    if not comparison_signatures(liked).intersection(comparison_signatures(vetoed)):
        return {
            "status": "not_evaluated",
            "reason": "unmatched_coverage_or_entry_convention",
            "liked_count": len(liked),
            "vetoed_count": len(vetoed),
        }
    return {
        "status": "discovery_only",
        "reason": "matched_timing_source_and_window",
        "liked_count": len(liked),
        "vetoed_count": len(vetoed),
    }


def summarise_attempts(
    attempts: Iterable[Mapping[str, Any]],
    *,
    useful_move_pct: float,
    min_opportunities: int = evidence_stats.MIN_REPORTABLE_N,
    min_sessions: int = 20,
) -> dict[str, Any]:
    """Summarise attempts without pooling scanner, review, and trade populations."""
    rows, duplicates = _deduplicate_attempts(attempts)
    populations: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_text(row.get("population")) or ALL_SCANNER].append(row)
    for population, members in grouped.items():
        cells = {
            variant: _cell(variant_rows, useful_move_pct=useful_move_pct)
            for variant, variant_rows in sorted(_group_by_variant(members).items())
        }
        populations[population] = {
            "cells": cells,
            "opportunity_count": len({_text(row.get("opportunity_id")) for row in members}),
            "status": "below_floor" if any(cell["opportunity_count"] < min_opportunities or cell["distinct_sessions"] < min_sessions for cell in cells.values()) else "discovery_only",
        }
    primary = populations.get(ALL_SCANNER, {"cells": {}})["cells"]
    clusters = {_text(row.get("dependency_cluster_id")) or _text(row.get("opportunity_id")) for row in rows}
    return {
        "schema": COMPARISON_SCHEMA,
        "cells": primary,
        "populations": populations,
        "independent_clusters": len(clusters),
        "deduplicated_rows": duplicates,
        "review_comparison": _review_comparison(rows),
    }


def _group_by_variant(rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_text(row.get("entry_variant"))].append(row)
    return grouped


def compare_variants(
    attempts: Iterable[Mapping[str, Any]],
    *,
    baseline: str,
    challenger: str,
    window: str,
    min_opportunities: int = evidence_stats.MIN_REPORTABLE_N,
    min_sessions: int = 20,
    trial_status: str | None = None,
) -> dict[str, Any]:
    """Report paired fills and all-opportunity coverage as different questions."""
    rows, duplicates = _deduplicate_attempts(attempts, window=window)
    # An entry comparison is a scanner/opportunity question.  Review/trade cohorts
    # remain separately labelled in ``summarise_attempts`` and never enter this pair.
    rows = [row for row in rows if (_text(row.get("population")) or ALL_SCANNER) == ALL_SCANNER]
    by_variant: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if _text(row.get("entry_variant")) in {baseline, challenger}:
            by_variant[_text(row.get("entry_variant"))][_text(row.get("opportunity_id"))] = row
    base_rows = by_variant.get(baseline, {})
    challenger_rows = by_variant.get(challenger, {})
    opportunities = set(base_rows).union(challenger_rows)
    paired_by_cluster: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for opportunity in sorted(set(base_rows).intersection(challenger_rows)):
        base_row, challenger_row = base_rows[opportunity], challenger_rows[opportunity]
        if (
            _text(base_row.get("state")) not in MEASURABLE_STATES
            or _text(challenger_row.get("state")) not in MEASURABLE_STATES
            or _finite(base_row.get("mfe_pct")) is None
            or _finite(challenger_row.get("mfe_pct")) is None
        ):
            continue
        base_cluster = _text(base_row.get("dependency_cluster_id")) or opportunity
        challenger_cluster = _text(challenger_row.get("dependency_cluster_id")) or opportunity
        if base_cluster != challenger_cluster:
            continue
        paired_by_cluster.setdefault(base_cluster, (base_row, challenger_row))
    paired = list(paired_by_cluster.values())
    deltas = [
        _finite(challenger_row.get("mfe_pct")) - _finite(base_row.get("mfe_pct"))  # type: ignore[operator]
        for base_row, challenger_row in paired
    ]
    challenger_mfe = [
        _finite(row.get("mfe_pct"))
        for row in _cluster_representatives(
            row for row in challenger_rows.values() if _text(row.get("state")) in MEASURABLE_STATES
        )
    ]
    challenger_values = [value for value in challenger_mfe if value is not None]
    sessions = {
        _text(row.get("session_date"))
        for row in list(base_rows.values()) + list(challenger_rows.values())
        if _text(row.get("session_date"))
    }
    reasons: list[str] = []
    if len(paired) < min_opportunities:
        reasons.append("below_sample_floor")
    if len(sessions) < min_sessions:
        reasons.append("below_session_floor")
    if trial_status in {trial_ledger.STATUS_REGISTERED, trial_ledger.STATUS_COLLECTING}:
        reasons.append("trial_immature")
    denominator = len(opportunities)
    baseline_triggered = sum(_text(row.get("state")) in MEASURABLE_STATES for row in base_rows.values())
    challenger_triggered = sum(_text(row.get("state")) in MEASURABLE_STATES for row in challenger_rows.values())
    eligible = not reasons
    return {
        "schema": COMPARISON_SCHEMA,
        "baseline": baseline,
        "challenger": challenger,
        "window": window,
        "all_opportunity_denominator": denominator,
        "paired_shared_triggered_count": len(paired),
        "paired_mfe_pct_delta_median": _quantile(deltas, 0.5),
        "baseline_all_opportunity_coverage": baseline_triggered / denominator if denominator else None,
        "challenger_all_opportunity_coverage": challenger_triggered / denominator if denominator else None,
        "distinct_sessions": len(sessions),
        "deduplicated_rows": duplicates,
        "status": "discovery_only" if eligible else "not_evaluated",
        "winner": "not_evaluated",
        "reasons": reasons,
        "outlier_sensitivity": {
            "mean_mfe_pct": sum(challenger_values) / len(challenger_values) if challenger_values else None,
            "median_mfe_pct": _quantile(challenger_values, 0.5),
            "headline_eligible": eligible,
        },
    }


def freeze_declaration(declaration: Mapping[str, Any], *, frozen_at: str) -> dict[str, Any]:
    """Create a pure, immutable-shaped declaration; ledger registration stays elsewhere."""
    required = ("trial_id", "primary_metric", "window", "baseline", "variants", "evaluation_date", "failure_criteria")
    missing = [name for name in required if not declaration.get(name)]
    if missing:
        raise ValueError(f"declaration missing {', '.join(missing)}")
    frozen = deepcopy(dict(declaration))
    frozen.update({"schema": DECLARATION_SCHEMA, "status": "frozen", "frozen_at": frozen_at})
    return frozen


def amend_declaration(
    declaration: Mapping[str, Any], changes: Mapping[str, Any], *, outcome_seen_at: str | None = None
) -> dict[str, Any]:
    """Allow pre-outcome drafting only; an outcome-era change needs a new trial."""
    if outcome_seen_at and changes:
        raise ValueError("post-outcome amendment requires a new trial")
    amended = deepcopy(dict(declaration))
    amended.update(dict(changes))
    return amended


def preserve_trial_history(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the append-only history intact, including abandoned and failed rows."""
    return [deepcopy(dict(row)) for row in rows]


def authorized_recipe_context(recipe_id: str, ledger_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Resolve one declared recipe against the existing ledger; never register it."""
    rows = list(ledger_rows)
    owners = trial_ledger.owners_of(recipe_id, rows)
    if len(owners) != 1:
        raise ValueError(f"unauthorized recipe: {recipe_id}")
    trial_id = owners[0]
    owner = next(row for row in rows if _text(row.get("trial_id")) == trial_id)
    family = _text(owner.get("family"))
    lifetime = sum(int(row.get("declared_cell_count") or 0) for row in rows if _text(row.get("family")) == family)
    return {
        "trial_id": trial_id,
        "family": family,
        "recipe_id": recipe_id,
        "family_lifetime_variants_examined": lifetime,
        "multiplicity_contract": {
            "declared_cell_count": int(owner.get("declared_cell_count") or 0),
            "family_lifetime": True,
            "widening_rule": "k>10 requires 99% holdout interval and family-median holdout comparison",
        },
        "status": _text(owner.get("status")),
    }


def prepare_authorized_evaluation(
    *, recipe_ids: Iterable[str], ledger_rows: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Validate a bounded, already-authorized recipe list without running anything."""
    rows = list(ledger_rows)
    return [authorized_recipe_context(recipe_id, rows) for recipe_id in recipe_ids]


__all__ = [
    "ALL_SCANNER",
    "COMPARISON_SCHEMA",
    "DECLARATION_SCHEMA",
    "adapt_m5_occurrence_attempts",
    "adapt_p8_attempts",
    "amend_declaration",
    "authorized_recipe_context",
    "compare_variants",
    "freeze_declaration",
    "prepare_authorized_evaluation",
    "preserve_trial_history",
    "summarise_attempts",
]
