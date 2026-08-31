"""Nightly setup stop/target research, with optional local-AI narration.

All numbers are produced by code from warehouse outcomes.  The model receives
one bounded fact pack after an evidence floor is met.  It may explain and
propose a test; it cannot write a policy or touch the trading desk.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

_log = logging.getLogger(__name__)

FACTS_SCHEMA = "setup_stop_target_research_facts_v1"
NARRATION_SCHEMA = "setup_stop_target_research_narration_v1"
FACT_SOURCE_ID = "setup_research.facts"
MIN_SYMBOLS = 5
MIN_SESSIONS = 5
MAX_POLICY_ROWS = 80
MAX_CONTEXT_ROWS = 80


def _now(value: datetime | None = None) -> datetime:
    moment = value or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _load() -> tuple[list[dict], dict[str, dict], dict[str, dict[str, str]], dict[str, Any]]:
    from research_warehouse import market_bias_context, occurrences, outcomes
    from research_warehouse.store import ResearchStore

    store = ResearchStore.open()
    if store is None:
        raise RuntimeError("research warehouse is disabled")
    recipes = [recipe.recipe_id for recipe in outcomes.M5_CLOSE_RECIPES]
    latest = list(outcomes.latest_outcomes(store, recipe_ids=recipes).values())
    years = {(_now().year - 1), _now().year}
    occurrence_map: dict[str, dict] = {}
    for year in years:
        occurrence_map.update(occurrences.latest_occurrences(store, year))
    context_rows = store.read_rows(
        "setup_market_context",
        columns=["occurrence_id", "timeframe", "bias_definition_id", "env_key"],
        occurrence_ids=list(occurrence_map),
    )
    contexts: dict[str, dict[str, str]] = {}
    for row in context_rows:
        if str(row.get("bias_definition_id")) != market_bias_context.BIAS_DEFINITION_ID:
            continue
        contexts.setdefault(str(row.get("occurrence_id")), {})[str(row.get("timeframe"))] = str(
            row.get("env_key") or "unknown"
        )
    coverage = {
        "outcomes": len(latest),
        "occurrences": len(occurrence_map),
        "context_rows": len(context_rows),
        "bias_definition_id": market_bias_context.BIAS_DEFINITION_ID,
    }
    return latest, occurrence_map, contexts, coverage


def _summarize(rows: list[dict]) -> dict[str, Any]:
    from evidence_stats import summarize

    usable = [row for row in rows if row.get("net_r") is not None]
    unresolved: dict[str, int] = {}
    for row in rows:
        if row.get("net_r") is None:
            reason = str(row.get("result_state") or "UNRESOLVED")
            unresolved[reason] = unresolved.get(reason, 0) + 1
    stats = summarize(
        [row.get("net_r") for row in usable],
        symbols=[str(row.get("symbol") or "") for row in usable],
        sessions=[str(row.get("session_date") or "") for row in usable],
        stop_flags=[str(row.get("first_hit") or "") == "STOP" for row in usable],
        unresolved=unresolved,
    )
    stats["win_rate"] = (
        round(sum(float(row["net_r"]) > 0.0 for row in usable) / len(usable), 4)
        if usable
        else None
    )
    symbols = len({str(row.get("symbol") or "") for row in usable if row.get("symbol")})
    sessions = len({str(row.get("session_date") or "") for row in usable if row.get("session_date")})
    stats["eligible"] = bool(stats.get("meets_n_floor") and symbols >= MIN_SYMBOLS and sessions >= MIN_SESSIONS)
    stats["eligibility_rule"] = (
        f"n >= 30, at least {MIN_SYMBOLS} symbols, and at least {MIN_SESSIONS} entry sessions; "
        "still discovery, never confirmation"
    )
    return stats


def _trimmed_mean_score(row: Mapping[str, Any]) -> float:
    value = ((row.get("stats") or {}).get("clipped") or {}).get("trimmed_mean")
    return float(value) if value is not None else -999.0


def build_fact_pack(
    outcomes_rows: list[Mapping[str, Any]],
    occurrences_by_id: Mapping[str, Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, str]],
    *,
    coverage: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    moment = _now(now)
    enriched: list[dict] = []
    for raw in outcomes_rows:
        row = dict(raw)
        occurrence = occurrences_by_id.get(str(row.get("occurrence_id"))) or {}
        entry_at = row.get("entry_at")
        session = entry_at.date().isoformat() if isinstance(entry_at, datetime) else str(entry_at or "")[:10]
        enriched.append(
            {
                **row,
                "family": str(occurrence.get("canonical_setup_id") or "UNKNOWN"),
                "side": str(occurrence.get("side") or ""),
                "symbol": str(occurrence.get("symbol") or ""),
                "session_date": session,
                "contexts": dict(contexts.get(str(row.get("occurrence_id"))) or {}),
            }
        )

    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in enriched:
        grouped.setdefault((row["family"], row["side"], str(row.get("recipe_id") or "")), []).append(row)
    policies = []
    for (family, side, recipe_id), members in grouped.items():
        policies.append(
            {
                "family": family,
                "side": side,
                "recipe_id": recipe_id,
                "stats": _summarize(members),
            }
        )
    policies.sort(
        key=lambda row: (
            0 if row["stats"].get("eligible") else 1,
            -_trimmed_mean_score(row),
            -int(row["stats"].get("n") or 0),
            row["family"],
            row["recipe_id"],
        )
    )

    context_groups: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for row in enriched:
        for timeframe, env_key in row["contexts"].items():
            context_groups.setdefault(
                (row["family"], row["side"], str(row.get("recipe_id") or ""), timeframe, env_key), []
            ).append(row)
    context_cells = []
    for (family, side, recipe_id, timeframe, env_key), members in context_groups.items():
        stats = _summarize(members)
        if stats.get("eligible"):
            context_cells.append(
                {
                    "family": family,
                    "side": side,
                    "recipe_id": recipe_id,
                    "timeframe": timeframe,
                    "env_key": env_key,
                    "stats": stats,
                }
            )
    context_cells.sort(
        key=lambda row: (
            -_trimmed_mean_score(row),
            -int(row["stats"].get("n") or 0),
        )
    )
    eligible = sum(1 for row in policies if row["stats"].get("eligible"))
    return {
        "schema": FACTS_SCHEMA,
        "generated_at": moment.isoformat(timespec="seconds"),
        "entry_contract": "first completed M5 close in the next regular session",
        "market_context_timeframes": ["M5", "M30", "H1", "H4", "D1"],
        "data_contract": {
            "planned_stop_or_risk_required": False,
            "m1_used": False,
            "bid_ask_used": False,
            "earnings_fundamentals_used": False,
            "same_bar_rule": "STOP_FIRST",
            "numbers_written_by": "deterministic code",
        },
        "gate": {
            "eligible_policy_cells": eligible,
            "met": eligible > 0,
            "note": "Every result is discovery. A met floor allows narration, not a live rule change.",
        },
        "coverage": dict(coverage or {}),
        "policies": policies[:MAX_POLICY_ROWS],
        "policy_cells_dropped_from_pack": max(0, len(policies) - MAX_POLICY_ROWS),
        "market_context_cells": context_cells[:MAX_CONTEXT_ROWS],
        "context_cells_dropped_from_pack": max(0, len(context_cells) - MAX_CONTEXT_ROWS),
        "not_a_control_signal": (
            "This report cannot change scanners, scores, alerts, Focus, watchlists, stops, targets, or orders."
        ),
    }


def render_markdown(pack: Mapping[str, Any]) -> str:
    lines = [
        f"# Setup stop/target research - {str(pack.get('generated_at'))[:10]}\n\n",
        f"Entry: {pack.get('entry_contract')}.\n\n",
        f"Gate: {pack.get('gate', {}).get('eligible_policy_cells', 0)} eligible policy cells. "
        f"{pack.get('gate', {}).get('note', '')}\n\n",
        "| setup | side | recipe | n | trimmed mean | win rate | eligible |\n",
        "|---|---:|---|---:|---:|---:|---:|\n",
    ]
    for row in pack.get("policies") or []:
        stats = row.get("stats") or {}
        clipped = stats.get("clipped") or {}
        lines.append(
            f"| {row.get('family')} | {row.get('side')} | {row.get('recipe_id')} | "
            f"{stats.get('n', 0)} | {clipped.get('trimmed_mean')} | {stats.get('win_rate')} | "
            f"{stats.get('eligible')} |\n"
        )
    lines.append(f"\n{pack.get('not_a_control_signal')}\n")
    return "".join(lines)


def _evidence_package(pack: Mapping[str, Any]) -> dict[str, Any]:
    encoded = json.dumps(pack, sort_keys=True, default=str).encode("utf-8")
    source_sha = hashlib.sha256(encoded).hexdigest()
    source = {
        "source_id": FACT_SOURCE_ID,
        "label": "Deterministic setup stop/target research facts",
        "status": "available",
        "observed_at": pack.get("generated_at"),
        "content_through": str(pack.get("generated_at"))[:10],
        "content_through_basis": "warehouse outcomes available when the pack was built",
        "session_date": str(pack.get("generated_at"))[:10],
        "sha256": source_sha,
        "truncated": False,
        "content": dict(pack),
    }
    package = {
        "schema_version": "ai_evidence_package_v2",
        "generated_at": pack.get("generated_at"),
        "session_date": str(pack.get("generated_at"))[:10],
        "selected_scopes": ["setup_research"],
        "scope_labels": ["Setup stop/target research"],
        "source_count": 1,
        "sources": [source],
        "coverage": {
            "counts": {"requested": 1, "usable": 1, "stale": 0, "truncated": 0},
            "note": "Every number was computed by code. Explain it; do not invent or recompute it.",
        },
        "safety_contract": {
            "purpose": "advisory explanation and suggestions for a future registered test",
            "forbidden_effects": ["scanner scores", "watchlists", "alerts", "bot state", "orders", "live stops"],
        },
        "scope_caveats": [
            "Every result is post-hoc discovery, never confirmation.",
            "Say not enough data when the eligible flag is false.",
            "Recommend at most three next tests. Never call a recipe optimal unless its eligible flag is true.",
        ],
    }
    canonical = json.dumps(package, sort_keys=True, separators=(",", ":"), default=str)
    package["evidence_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    package["package_id"] = package["evidence_hash"][:16]
    return package


def _narrate(pack: Mapping[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    import ai_summary

    if not ai_summary.local_provider_enabled():
        raise RuntimeError("local AI provider is not configured; deterministic facts still exist")
    package = _evidence_package(pack)
    result = ai_summary.request_ai_summary(
        provider="local",
        model=ai_summary.local_model("medium"),
        api_key="",
        evidence=package,
        timeout_seconds=900,
    )
    return {
        "schema": NARRATION_SCHEMA,
        "generated_at": _now(now).isoformat(timespec="seconds"),
        "facts_sha256": package["sources"][0]["sha256"],
        "facts_package_id": package["package_id"],
        "model": result.get("model", ""),
        "narration": result.get("summary") or {},
        "note": "Advisory words over deterministic facts. No live rule was changed.",
    }


def _publish(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def _superseding(path: Path) -> Path:
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.stem}.{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def _default_root() -> Path:
    from ai_jobs import store

    return store.retros_dir() / "setup_research"


def run_setup_research(
    *,
    session_date: str = "",
    now: datetime | None = None,
    root: Path | None = None,
    narrate: bool = True,
    inputs=None,
    **_ignored: Any,
) -> dict[str, Any]:
    moment = _now(now)
    try:
        outcome_rows, occurrence_map, contexts, coverage = inputs or _load()
        pack = build_fact_pack(
            outcome_rows, occurrence_map, contexts, coverage=coverage, now=moment
        )
        target_root = Path(root) if root is not None else _default_root()
        stamp = session_date or moment.date().isoformat()
        json_path = _superseding(target_root / str(moment.year) / f"{stamp}.json")
        outputs = [str(_publish(json_path, json.dumps(pack, indent=1, sort_keys=True, default=str) + "\n"))]
        outputs.append(str(_publish(json_path.with_suffix(".md"), render_markdown(pack))))
    except Exception as exc:  # noqa: BLE001
        return {"status": "failed", "model": "", "reason": f"setup research failed: {exc}", "outputs": []}

    base = (
        f"{pack['coverage'].get('outcomes', len(outcome_rows))} recipe outcome(s); "
        f"{pack['gate']['eligible_policy_cells']} eligible policy cell(s)"
    )
    if not pack["gate"]["met"] or not narrate:
        suffix = "; no model called below the evidence floor" if not pack["gate"]["met"] else ""
        return {"status": "ok", "model": "", "reason": base + suffix, "outputs": outputs}
    try:
        narration = _narrate(pack, now=moment)
        narration_path = _superseding(json_path.with_name(f"{json_path.stem}.narration.json"))
        outputs.append(str(_publish(narration_path, json.dumps(narration, indent=1, sort_keys=True, default=str) + "\n")))
    except Exception as exc:  # noqa: BLE001
        _log.info("Setup research narration unavailable (%s).", exc)
        return {"status": "degraded_no_narrative", "model": "", "reason": f"{base}; narration absent: {exc}", "outputs": outputs}
    return {"status": "ok", "model": str(narration.get("model") or ""), "reason": base + "; narrated", "outputs": outputs}


__all__ = ["FACTS_SCHEMA", "NARRATION_SCHEMA", "build_fact_pack", "render_markdown", "run_setup_research"]
