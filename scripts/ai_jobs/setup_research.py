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
#: Retained as the pack's overall policy-row ceiling, and now only a ceiling:
#: the eligible block is published WHOLE (it has never approached this - the
#: 2026-08-31 pack had 9) and the ineligible block carries its own smaller cap.
#: A run that ever pushed the eligible block past this would be reporting a
#: different problem, so the assertion below states it rather than truncating
#: the answer silently.
MAX_POLICY_ROWS = 80
MAX_CONTEXT_ROWS = 80
#: How many INELIGIBLE policy cells ride along under the eligible block. They
#: are kept because "measured and thin" is a different fact from "not measured",
#: and bounded because the 2026-08-31 pack spent 71 of its 80 rows on n=1 cells
#: sorted to the top by trimmed mean - the reader saw +2.9R nine rows under the
#: real answer.
MAX_INELIGIBLE_POLICY_ROWS = 40

#: Family roles, until the setup registry of packet P7 owns them.
#:
#: Appendix C is normative and already says what these two are: General/Untagged
#: is a "Diagnostic fallback" that "must not become a pooled 'setup' edge", and
#: Favorite Zone Watch is a "Watch state" that is "never counted as a triggered
#: trade setup". The 2026-08-31 pack pooled both anyway - GENERAL over 735
#: occurrences and FAVORITE_ZONE_WATCH over 486 - because nothing in this job
#: knew their roles.
#:
#: Deliberately a SMALL EXPLICIT MAP and not a heuristic: anything not named
#: here is a trade setup, so a family added tomorrow is measured rather than
#: silently excluded, and excluding a real setup would need someone to type its
#: name. **P7's registry replaces this map**; when it lands, this constant goes
#: and the role comes from the registry row.
ROLE_TRADE = "TRADE"
NON_TRADE_FAMILY_ROLES = {
    "GENERAL": "FALLBACK",
    "FAVORITE_ZONE_WATCH": "WATCH_STATE",
}
NON_TRADE_ROLE_REASONS = {
    "FALLBACK": "Appendix C: diagnostic fallback - must not become a pooled 'setup' edge.",
    "WATCH_STATE": "Appendix C: watch state - never counted as a triggered trade setup.",
}


def family_role(family: str) -> str:
    """The role this family grades under. TRADE unless explicitly named."""
    return NON_TRADE_FAMILY_ROLES.get(str(family or "").strip().upper(), ROLE_TRADE)


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
    coverage.update(_coverage_state(store, latest, occurrence_map))
    return latest, occurrence_map, contexts, coverage


def _coverage_state(store, outcome_rows, occurrence_map) -> dict[str, Any]:
    """What has been measured, so an absent family reads correctly.

    Three facts, and every one of them can be UNKNOWN rather than zero:

    * **buckets covered** - `cli._run_outcomes` simulates ONE of 32 symbol
      buckets per firing, so a family can be missing because its symbols have
      not come up yet. That is the opposite conclusion from "measured and
      flat", and the pack could not tell them apart.
    * **families with zero outcome rows** - occurrences exist, outcomes do not.
      Named, because a family that simply is not in the table below reads as one
      with nothing to say.
    * **first M5 session in the lake** - the earliest entry any simulation could
      ever have had. Everything before it is out of reach, not flat.

    Every failure here degrades to a stated absence: the fact pack is the
    product and a coverage read must never cost it.
    """
    state: dict[str, Any] = {}
    try:
        from research_warehouse import outcome_coverage

        state.update(outcome_coverage.coverage_state(getattr(store, "root", None)))
    except Exception as exc:  # noqa: BLE001
        state["outcome_bucket_coverage_note"] = f"coverage record unreadable: {exc}"

    try:
        measured = {str(row.get("occurrence_id") or "") for row in outcome_rows}
        by_family: dict[str, bool] = {}
        for identity, occurrence in occurrence_map.items():
            family = str(occurrence.get("canonical_setup_id") or "UNKNOWN")
            by_family[family] = by_family.get(family, False) or (str(identity) in measured)
        state["families_without_outcomes"] = sorted(
            family for family, seen in by_family.items() if not seen
        )
        state["families_with_outcomes"] = sorted(
            family for family, seen in by_family.items() if seen
        )
    except Exception as exc:  # noqa: BLE001
        state["families_without_outcomes_note"] = f"not computed: {exc}"

    state.update(_first_m5_session(store))
    return state


def _first_m5_session(store) -> dict[str, Any]:
    """The earliest M5 partition in the lake, read from the MANIFEST.

    Partition names, never bar rows: the answer is a month, and materialising
    a month of M5 bars to learn its own name would be the exact mistake the
    month-keyed read rules exist to prevent (BD-66/BD-69). The manifest
    already carries every sealed file's partition.
    """
    try:
        resolved = store.manifest.resolve(dataset="bar_m5")
        months = set()
        for entry in resolved.entries:
            text = str(getattr(entry, "file_path", "") or "")
            for part in text.replace("\\", "/").split("/"):
                if part.startswith("month="):
                    months.add(part[len("month="):])
        if not months:
            return {"first_m5_session": None, "first_m5_session_note": "no sealed bar_m5 partitions"}
        return {"first_m5_session": min(months), "m5_months_in_lake": len(months)}
    except Exception as exc:  # noqa: BLE001
        return {"first_m5_session": None, "first_m5_session_note": f"not readable: {exc}"}


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
    # EPISODES, beside n. `n` counts outcome ROWS, and the ERD cardinality
    # table says what that is: `setup_occurrence` -> `outcome_path` is 1:N, and
    # "alternative recipes/horizons are correlated diagnostics of ONE episode;
    # they are never summed as independent samples". `dependency_cluster_id` is
    # the episode unit for evidence floors (occurrences.dependency_cluster_id,
    # which deliberately excludes the setup family so simultaneous variants on
    # one move share one cluster).
    #
    # The FLOOR STILL COUNTS ROWS in this packet, on purpose: moving it is a
    # change to which cells are eligible - i.e. to what the model is allowed to
    # narrate - and it belongs in its own packet with its own before/after.
    # Publishing both is what makes that packet decidable. See BD-80.
    stats["n_episodes"] = len(
        {
            str(row.get("dependency_cluster_id") or "")
            for row in usable
            if row.get("dependency_cluster_id")
        }
    )
    stats["eligible"] = bool(stats.get("meets_n_floor") and symbols >= MIN_SYMBOLS and sessions >= MIN_SESSIONS)
    stats["eligibility_rule"] = (
        f"n >= 30 OUTCOME ROWS, at least {MIN_SYMBOLS} symbols, and at least "
        f"{MIN_SESSIONS} entry sessions; still discovery, never confirmation. "
        "`n_episodes` is reported beside `n` and does NOT yet gate: rows on one "
        "episode are correlated diagnostics, so n_episodes is the honest sample "
        "size and moving the floor onto it is a separate, scoped change (BD-80)."
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
                # Already on the occurrence rows `_load` fetches; carried so a
                # cell can count episodes as well as rows.
                "dependency_cluster_id": str(occurrence.get("dependency_cluster_id") or ""),
                "contexts": dict(contexts.get(str(row.get("occurrence_id"))) or {}),
            }
        )

    # Non-trade families are EXCLUDED and REPORTED, never silently omitted.
    # Appendix C already says what they are; nothing in this job knew, so the
    # 2026-08-31 pack pooled GENERAL and FAVORITE_ZONE_WATCH as setup edges.
    # Their counts still travel, because an absent family with no explanation
    # reads as a family with no data.
    non_trade_rows: dict[str, list[dict]] = {}
    trade_rows: list[dict] = []
    for row in enriched:
        role = family_role(row["family"])
        if role == ROLE_TRADE:
            trade_rows.append(row)
        else:
            non_trade_rows.setdefault(row["family"], []).append(row)
    non_trade_families = [
        {
            "family": family,
            "role": family_role(family),
            "reason": NON_TRADE_ROLE_REASONS.get(family_role(family), ""),
            "outcome_rows": len(members),
            "episodes": len(
                {str(item.get("dependency_cluster_id") or "") for item in members if item.get("dependency_cluster_id")}
            ),
            "occurrences": len({str(item.get("occurrence_id") or "") for item in members}),
        }
        for family, members in sorted(non_trade_rows.items())
    ]

    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in trade_rows:
        grouped.setdefault((row["family"], row["side"], str(row.get("recipe_id") or "")), []).append(row)
    all_policies = []
    for (family, side, recipe_id), members in grouped.items():
        all_policies.append(
            {
                "family": family,
                "side": side,
                "recipe_id": recipe_id,
                "stats": _summarize(members),
            }
        )

    # TWO BLOCKS, because one sorted list buried the answer. Sorting everything
    # by trimmed mean put nine real cells above 71 n=1 cells reading +2.9R, and
    # the 80-row cap then dropped 508 more without saying which. The eligible
    # block is complete and ordered as before; the ineligible block is bounded
    # and ordered by n DESC first - the same shape the context-cell path uses -
    # so what rides along is the thickest evidence that has not cleared the
    # floor, never the luckiest single trade.
    eligible_cells = [row for row in all_policies if row["stats"].get("eligible")]
    ineligible_cells = [row for row in all_policies if not row["stats"].get("eligible")]
    eligible_cells.sort(
        key=lambda row: (
            -_trimmed_mean_score(row),
            -int(row["stats"].get("n") or 0),
            row["family"],
            row["recipe_id"],
        )
    )
    ineligible_cells.sort(
        key=lambda row: (
            -int(row["stats"].get("n") or 0),
            -_trimmed_mean_score(row),
            row["family"],
            row["recipe_id"],
        )
    )
    kept_ineligible = ineligible_cells[:MAX_INELIGIBLE_POLICY_ROWS]
    # `policies` is kept, same key and same meaning, so every existing reader
    # and the published-pack shape are unchanged: it is the two blocks in order.
    policies = eligible_cells + kept_ineligible

    context_groups: dict[tuple[str, str, str, str, str], list[dict]] = {}
    for row in trade_rows:
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
    eligible = len(eligible_cells)
    # WHERE THE CORRELATION ACTUALLY IS. Measured on the live lake 2026-09-01:
    # 9,372 outcome rows rest on 599 occurrences and 287 dependency clusters -
    # 15.6 recipe rows per occurrence, and 1,804 of 3,436 clusters carry more
    # than one family.
    #
    # Inside a single (family, side, recipe) cell, n and n_episodes were EQUAL
    # in all 756 cells: one row per occurrence per recipe, so the per-cell
    # episode count is not where the double-counting lives. The ERD's 1:N
    # warning is about reading cells TOGETHER - nine ATR variants of one family
    # are nine readings of the same 33 moves, not 297 samples. So the pack
    # publishes the shape of its whole evidence base as well, because that is
    # the number a reader comparing rows needs. See BD-80.
    trade_occurrences = {str(row.get("occurrence_id") or "") for row in trade_rows}
    trade_episodes = {
        str(row.get("dependency_cluster_id") or "")
        for row in trade_rows
        if row.get("dependency_cluster_id")
    }
    evidence_shape = {
        "outcome_rows": len(trade_rows),
        "distinct_occurrences": len(trade_occurrences),
        "distinct_episodes": len(trade_episodes),
        "rows_per_occurrence": (
            round(len(trade_rows) / len(trade_occurrences), 2) if trade_occurrences else None
        ),
        "note": (
            "Cells are alternative recipes over the SAME occurrences. Reading two "
            "cells as independent evidence double-counts the move underneath them: "
            "the episode count is the honest denominator for anything pooled across "
            "cells. Within one cell, n and n_episodes are equal by construction."
        ),
    }
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
        "evidence_shape": evidence_shape,
        "policies": policies,
        "eligible_policies": eligible_cells,
        "ineligible_policies": kept_ineligible,
        # Honest per block. The total is what a reader of the old single field
        # expects, and it is still the number of cells that exist minus the
        # number published - it just no longer hides WHICH kind was dropped.
        "policy_cells_dropped_from_pack": max(0, len(all_policies) - len(policies)),
        "eligible_policy_cells_dropped": 0,
        "ineligible_policy_cells_dropped": max(0, len(ineligible_cells) - len(kept_ineligible)),
        "non_trade_families": non_trade_families,
        "non_trade_families_note": (
            "Excluded from every policy and context cell above, by role. Their "
            "counts are published so an absent family reads as excluded rather "
            "than as unmeasured. Packet P7's setup registry will own this map."
        ),
        "market_context_cells": context_cells[:MAX_CONTEXT_ROWS],
        "context_cells_dropped_from_pack": max(0, len(context_cells) - MAX_CONTEXT_ROWS),
        "not_a_control_signal": (
            "This report cannot change scanners, scores, alerts, Focus, watchlists, stops, targets, or orders."
        ),
    }


_POLICY_HEADER = (
    "| setup | side | recipe | n | episodes | trimmed mean | win rate |\n"
    "|---|---:|---|---:|---:|---:|---:|\n"
)


def _policy_rows(rows) -> list[str]:
    out = []
    for row in rows or []:
        stats = row.get("stats") or {}
        clipped = stats.get("clipped") or {}
        out.append(
            f"| {row.get('family')} | {row.get('side')} | {row.get('recipe_id')} | "
            f"{stats.get('n', 0)} | {stats.get('n_episodes', '-')} | "
            f"{clipped.get('trimmed_mean')} | {stats.get('win_rate')} |\n"
        )
    return out


def render_markdown(pack: Mapping[str, Any]) -> str:
    """The pack a person reads, opening with the part that cleared the floor.

    It used to open with one table sorted by trimmed mean across every cell,
    eligible or not. On 2026-08-31 that put nine real cells - all
    AVWAPE_TO_FIRST_DEV/LONG, all negative - above 71 single-trade cells
    reading +2.9R, and the 80-row cap then dropped 508 more without saying
    which kind. A reader skimming the top of that file learned the opposite of
    what the evidence said.

    Now: the eligible block first and whole, the bounded ineligible block
    beneath it under a heading that says what it is, the excluded non-trade
    families named, and the coverage line. Every count that was dropped is
    printed next to the block it was dropped from.
    """
    gate = pack.get("gate") or {}
    eligible = pack.get("eligible_policies")
    ineligible = pack.get("ineligible_policies")
    if eligible is None and ineligible is None:
        # A pack written before the split. Read it as it was published rather
        # than inventing a division its author never made.
        eligible = [row for row in (pack.get("policies") or []) if (row.get("stats") or {}).get("eligible")]
        ineligible = [row for row in (pack.get("policies") or []) if not (row.get("stats") or {}).get("eligible")]

    lines = [
        f"# Setup stop/target research - {str(pack.get('generated_at'))[:10]}\n\n",
        f"Entry: {pack.get('entry_contract')}.\n\n",
        f"Gate: {gate.get('eligible_policy_cells', 0)} eligible policy cells. "
        f"{gate.get('note', '')}\n\n",
    ]
    lines += _coverage_lines(pack)
    lines.append("## Eligible policy cells\n\n")
    if eligible:
        lines.append(_POLICY_HEADER)
        lines += _policy_rows(eligible)
    else:
        lines.append(
            "No cell cleared the evidence floor. That is not a flat result - it "
            "is an absent one.\n"
        )
    dropped_eligible = int(pack.get("eligible_policy_cells_dropped") or 0)
    if dropped_eligible:
        lines.append(f"\n{dropped_eligible} eligible cell(s) did not fit this pack.\n")

    lines.append("\n## Below the evidence floor (thickest first, not best first)\n\n")
    lines.append(
        "These have NOT cleared the floor. They are ordered by n so what rides "
        "along is the most-measured evidence, never the luckiest single trade, "
        "and no number here may be read as a finding.\n\n"
    )
    if ineligible:
        lines.append(_POLICY_HEADER)
        lines += _policy_rows(ineligible)
    else:
        lines.append("None.\n")
    dropped_ineligible = int(pack.get("ineligible_policy_cells_dropped") or 0)
    if dropped_ineligible:
        lines.append(
            f"\n{dropped_ineligible} further cell(s) below the floor are not shown.\n"
        )

    families = pack.get("non_trade_families") or []
    if families:
        lines.append("\n## Excluded families (not trade setups)\n\n")
        lines.append("| family | role | outcome rows | episodes | why |\n")
        lines.append("|---|---|---:|---:|---|\n")
        for entry in families:
            lines.append(
                f"| {entry.get('family')} | {entry.get('role')} | "
                f"{entry.get('outcome_rows')} | {entry.get('episodes')} | "
                f"{entry.get('reason')} |\n"
            )
        lines.append(f"\n{pack.get('non_trade_families_note', '')}\n")

    lines.append(f"\n{pack.get('not_a_control_signal')}\n")
    return "".join(lines)


def _coverage_lines(pack: Mapping[str, Any]) -> list[str]:
    """What was measured, so "not measured yet" reads differently from "flat".

    Absent keys print nothing rather than a zero: a pack written before this
    block existed must not claim zero bucket coverage.
    """
    coverage = pack.get("coverage") or {}
    parts: list[str] = []
    shape = pack.get("evidence_shape") or {}
    if shape.get("outcome_rows") is not None:
        parts.append(
            f"Evidence shape: {shape.get('outcome_rows')} outcome row(s) over "
            f"{shape.get('distinct_occurrences')} occurrence(s) and "
            f"{shape.get('distinct_episodes')} episode(s) "
            f"({shape.get('rows_per_occurrence')} rows per occurrence). "
            "Cells below are alternative recipes over the SAME occurrences - "
            "reading two of them as independent evidence double-counts the move "
            "underneath.\n"
        )
    buckets = coverage.get("outcome_buckets_covered")
    total = coverage.get("outcome_bucket_count")
    if buckets is not None and total:
        firings = coverage.get("outcome_firings_considered")
        parts.append(
            f"Bucket coverage: {buckets} of {total} symbol bucket(s) simulated in "
            f"the last {firings if firings is not None else total} outcome firing(s). "
            "A family absent below may simply not have been in a covered bucket yet.\n"
        )
    elif coverage.get("outcome_bucket_coverage_note"):
        parts.append(f"Bucket coverage: {coverage['outcome_bucket_coverage_note']}\n")
    zero = coverage.get("families_without_outcomes")
    if zero:
        shown = ", ".join(str(name) for name in zero[:12])
        more = f" (+{len(zero) - 12} more)" if len(zero) > 12 else ""
        parts.append(
            f"Families with occurrences but ZERO outcome rows: {len(zero)} - {shown}{more}. "
            "Not measured yet, which is not the same as measured and flat.\n"
        )
    first_m5 = coverage.get("first_m5_session")
    if first_m5:
        parts.append(f"First M5 session in the lake: {first_m5}.\n")
    return [f"{part}\n" for part in parts]


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
