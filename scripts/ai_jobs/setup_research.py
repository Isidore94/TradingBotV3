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
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

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
#: different problem.
#:
#: R1: the comment said "the assertion below states it" and there is no such
#: assertion, and `MAX_POLICY_ROWS` itself is read by nothing - the eligible
#: block is bounded by `MAX_INELIGIBLE_POLICY_ROWS` and by the floor, not by
#: this. It is kept because the number is the stated intent and deleting it
#: would delete the intent with it, but a comment describing a guard that does
#: not exist is worse than no comment: the guard would have to be written, and
#: writing it is a behaviour change nobody has authorized.
MAX_POLICY_ROWS = 80
MAX_CONTEXT_ROWS = 80
#: How many INELIGIBLE policy cells ride along under the eligible block. They
#: are kept because "measured and thin" is a different fact from "not measured",
#: and bounded because the 2026-08-31 pack spent 71 of its 80 rows on n=1 cells
#: sorted to the top by trimmed mean - the reader saw +2.9R nine rows under the
#: real answer.
MAX_INELIGIBLE_POLICY_ROWS = 40

#: Why this job knows about roles at all. The MAP itself is gone (R1).
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
#: name. **P7's registry HAS replaced the map** (merged 2026-09-02):
#: `family_role` below reads `setup_registry.fact_pack_role`, and what survives
#: here is only the WORDING - the reason each non-trade role is excluded, which
#: the pack prints beside the family it excluded.
ROLE_TRADE = "TRADE"
NON_TRADE_ROLE_REASONS = {
    "FALLBACK": "Appendix C: diagnostic fallback - must not become a pooled 'setup' edge.",
    "WATCH_STATE": "Appendix C: watch state - never counted as a triggered trade setup.",
}


def family_role(family: str) -> str:
    """The role this family grades under. TRADE unless the registry says otherwise.

    P3 shipped this as a two-entry map of its own, because the setup registry did
    not exist yet. It does now (P7), so the map is gone and this reads the ONE
    table - the merge of the two branches on 2026-09-02 is what made the swap
    possible, and P7 named it as owed to whichever landed second.

    The wording is unchanged on purpose. The registry keeps Appendix C's spelling
    (`TRADE_SETUP`) and `fact_pack_role` translates it back to this pack's own
    `TRADE`, so the fact pack's output is byte-identical while the ontology has
    one owner instead of two. An unknown family still grades as TRADE: a registry
    gap must not silently reclassify live evidence.
    """
    from setup_registry import fact_pack_role

    return fact_pack_role(family)


def _now(value: datetime | None = None) -> datetime:
    moment = value or datetime.now(timezone.utc)
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _load() -> tuple[list[dict], dict[str, dict], dict[str, dict[str, str]], dict[str, Any]]:
    from research_warehouse import market_bias_context, occurrences, outcomes
    from research_warehouse.store import ResearchStore

    store = ResearchStore.open()
    if store is None:
        raise RuntimeError("research warehouse is disabled")
    # The M5-close grid, the Phase 0.12 B3 higher-timeframe LRSI study and the
    # Phase 0.13 P8 entry-timing grid. Reading a recipe id here only widens a
    # read of rows the warehouse has already published; it authorizes nothing.
    # Every HTF and P8 row is diagnostic.
    recipes = [
        recipe.recipe_id
        for recipe in (
            tuple(outcomes.M5_CLOSE_RECIPES)
            + tuple(outcomes.HTF_LRSI_RECIPES)
            + tuple(outcomes.SETUP_ENTRY_TIMING_RECIPES)
            + tuple(outcomes.AFTER_LIKE_RECIPES)
        )
    ]
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
    # The grid is carried out with the rows so the pack can state WHICH recipe
    # ids these outcomes came from (R3) - two packs from one night differed by
    # 3,067 rows because the grid had changed under them, and neither said so.
    coverage = {**coverage, "recipe_ids": recipes}
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


def _meets_eligibility_floors(
    summary: Mapping[str, Any],
    *,
    symbols: int | None = None,
    sessions: int | None = None,
) -> bool:
    """The ONE eligibility rule, so every block in this pack states it once.

    `evidence_stats.summarize` deliberately does NOT set `eligible` — the n
    floor is its business and the symbol/session floors are this pack's. Reading
    `summary["eligible"]` straight off a `summarize` result therefore reads a key
    that is never present, which is `False` for every cell no matter how large:
    the after-like block did exactly that, so a 60-episode, 60-symbol,
    28-session cell reported ineligible. When `symbols`/`sessions` are not given
    they are taken from `summarize`'s own `counts` block, which is where it puts
    the distinct tallies.
    """
    counts = summary.get("counts") or {}
    symbol_count = int(counts.get("symbols") or 0) if symbols is None else int(symbols)
    session_count = int(counts.get("sessions") or 0) if sessions is None else int(sessions)
    return bool(
        summary.get("meets_n_floor")
        and symbol_count >= MIN_SYMBOLS
        and session_count >= MIN_SESSIONS
    )


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
    # Publishing both is what makes that packet decidable. See BD-81.
    stats["n_episodes"] = len(
        {
            str(row.get("dependency_cluster_id") or "")
            for row in usable
            if row.get("dependency_cluster_id")
        }
    )
    stats["eligible"] = _meets_eligibility_floors(stats, symbols=symbols, sessions=sessions)
    stats["eligibility_rule"] = (
        f"n >= 30 OUTCOME ROWS, at least {MIN_SYMBOLS} symbols, and at least "
        f"{MIN_SESSIONS} entry sessions; still discovery, never confirmation. "
        "`n_episodes` is reported beside `n` and does NOT yet gate: rows on one "
        "episode are correlated diagnostics, so n_episodes is the honest sample "
        "size and moving the floor onto it is a separate, scoped change (BD-81)."
    )
    return stats


def _trimmed_mean_score(row: Mapping[str, Any]) -> float:
    value = ((row.get("stats") or {}).get("clipped") or {}).get("trimmed_mean")
    return float(value) if value is not None else -999.0


AFTER_LIKE_PREFIX = "afterlike_"


def after_like_block(rows) -> dict:
    """P10 C3: what the trader's likes did, by day offset and entry rule.

    Cells are keyed by (offset, entry) and NOT by family, and that is a measured
    limit rather than a choice. `outcome_path` has no column for the linked
    occurrence, so the published row carries the like episode - which is what its
    `occurrence_id` is for these rows - and not the setup family behind it. The
    family is recoverable by joining the bronze `like_occurrence_link` dataset,
    and the block SAYS SO rather than leaving a reader to wonder why the split
    they were promised is absent.

    The EPISODE is the unit. For an after-like row the `occurrence_id` IS the
    like episode (`afterlike|SYMBOL|SIDE|date`), so distinct ids are distinct
    likes, and a cell that counted rows would count one opinion once per cell.

    Nothing here is eligible for a verdict until the registered window closes -
    `eligible` reports the floors, and the trial ledger holds the window.
    """
    import evidence_stats

    cells: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        recipe_id = str(row.get("recipe_id") or "")
        if not recipe_id.startswith(AFTER_LIKE_PREFIX):
            continue
        rest = recipe_id[len(AFTER_LIKE_PREFIX) :]
        if not rest.startswith("d"):
            continue
        head, _, tail = rest.partition("_")
        offset_text = head[1:]
        if not offset_text.isdigit():
            continue
        entry = tail.rsplit("_2r_v1", 1)[0]
        cells.setdefault((int(offset_text), entry), []).append(row)

    out = []
    for (offset, entry), cell_rows in sorted(cells.items()):
        episodes = {str(row.get("occurrence_id") or "") for row in cell_rows}
        episodes.discard("")
        values = [
            float(row["net_r"])
            for row in cell_rows
            if row.get("net_r") is not None
        ]
        # The desk's ONE statistics contract, and the same call the other blocks
        # make: the episode is the sample, so symbols and sessions are counted on
        # the episode key rather than on the row.
        summary = (
            evidence_stats.summarize(
                values,
                symbols=[str(row.get("occurrence_id") or "") for row in cell_rows],
                sessions=[
                    str(row.get("entry_at") or "")[:10] for row in cell_rows
                ],
                stop_flags=[
                    str(row.get("first_hit") or "") == "STOP" for row in cell_rows
                ],
            )
            if values
            else {}
        )
        out.append(
            {
                "day_offset": offset,
                "entry": entry,
                "n": len(cell_rows),
                "n_episodes": len(episodes),
                "trimmed_mean_r": (summary.get("clipped") or {}).get("trimmed_mean"),
                # `summarize` does not compute a win rate - `_summarize` adds it
                # afterwards, and this does the same rather than reaching into
                # that function, so both blocks state one definition once each.
                "win_rate": (
                    round(sum(value > 0.0 for value in values) / len(values), 4)
                    if values
                    else None
                ),
                "stop_rate": summary.get("stop_rate"),
                # THE FLOORS, not a key `summarize` never sets. This read
                # `summary.get("eligible")`, which is always absent, so every
                # cell of this grid reported ineligible regardless of its size.
                # `symbols` here is the EPISODE key, as the call above passes it.
                "eligible": _meets_eligibility_floors(summary),
                "meets_n_floor": bool(summary.get("meets_n_floor")),
                "evidence_label": summary.get("evidence_label", "discovery"),
            }
        )
    return {
        "schema": "after_like_v1",
        "question": (
            "For a D1 name the trader LIKED, which day after the like and which "
            "entry rule gives the best net R?"
        ),
        "cells": out,
        "episodes": len({str(row.get("occurrence_id") or "") for cell in cells.values() for row in cell}),
        "family_split": (
            "NOT AVAILABLE from these rows. `outcome_path` has no column for the "
            "linked occurrence, so a family split needs a join against the bronze "
            "`like_occurrence_link` dataset."
        ),
        "read_before_the_window_closes": (
            "No cell here may be read for a verdict before the registered window "
            "closes - see the `after_like_entry_grid_v1` trial ledger row. A cell "
            "that looks good early is the reason the window was fixed at "
            "registration."
        ),
        "not_a_control_signal": (
            "Diagnostic rows only. Nothing here reaches a detector, score, alert, "
            "watchlist, Focus list, review queue or review_policy.json."
        ),
    }


def build_fact_pack(
    outcomes_rows: list[Mapping[str, Any]],
    occurrences_by_id: Mapping[str, Mapping[str, Any]],
    contexts: Mapping[str, Mapping[str, str]],
    *,
    coverage: Mapping[str, Any] | None = None,
    recipe_ids: Sequence[str] | None = None,
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
    # the number a reader comparing rows needs. See BD-81.
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
        # WHICH CODE BUILT THIS, and over WHICH GRID (R3). Two packs from one
        # night disagreed by 3,067 outcome rows - 9,372 at 03:55 on the
        # pre-merge checkout and 12,439 at 04:30 on `main`, because P8's grid
        # had landed in between - and nothing in either pack said so. A reader
        # comparing them had no way to tell a real change in the evidence from a
        # change in the code that measured it.
        "built_by_commit": _built_by_commit(),
        # What was LOADED, from the caller - never re-derived from the module
        # here. Re-deriving would state the grid this CODE knows about rather
        # than the grid these ROWS came from, which is the one thing the field
        # exists to distinguish. Empty means the caller did not say.
        "recipe_ids": list(recipe_ids or ()),
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
        # P10 C3, built from the RAW rows rather than the enriched ones: an
        # after-like row's `occurrence_id` is a LIKE EPISODE, so the family
        # enrichment above would file every one of them under UNKNOWN.
        "after_like": after_like_block(outcomes_rows),
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


def _narration_coverage_lines(narrated: Mapping[str, Any] | None) -> list[str]:
    """The "Narration" heading and the one line that says what was left out.

    N3: the narration is bounded (`_bounded_narration_view`), so the pack on
    disk carries cells the words were never written over. A reader of the
    markdown alone has no other way to know that, and a coverage line that
    appeared only in the json would be a rule nobody reads.

    Absent `narrated` prints NOTHING rather than "0 of 0": no narration was
    attempted (the gate was not met, or `narrate=False`), and claiming coverage
    of a narration that does not exist is worse than saying nothing.
    """
    if not narrated:
        return []
    kept = _int_or_zero(narrated.get("eligible_policy_cells"))
    total = _int_or_zero(narrated.get("of"))
    return [
        "\n## Narration\n\n",
        f"Narration covers {kept} of {total} eligible cells, selected by "
        f"{narrated.get('selected_by') or NARRATION_SELECTED_BY}; "
        f"{max(total - kept, 0)} omitted for size. The selection is a SIZE rule "
        "and never a ranking by result (gate #43) - the cells the model was not "
        "shown are in this pack, above, and are neither better nor worse than "
        "the ones it was.\n",
    ]


def render_markdown(
    pack: Mapping[str, Any], *, narrated: Mapping[str, Any] | None = None
) -> str:
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

    lines += _narration_coverage_lines(narrated)
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


class NarrationTooLarge(RuntimeError):
    """The narration view will not fit the local model. Refuse, do not send."""


# The per-cell prose that is identical in every cell by construction: each is a
# module constant interpolated into `stats`, not a measurement. Written once per
# cell they were half the narration view; the dotted paths address them inside
# the nested `stats` block where they live.
_SHARED_CELL_PROSE = (
    "stats.eligibility_rule",
    "stats.n_floor_note",
    "stats.profit_factor.convention",
    "stats.bootstrap.interval",
    "stats.schema",
)


def _dig(row: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(row, Mapping) or part not in row:
            return _MISSING
        row = row[part]
    return row


def _drop(row: Any, path: str) -> Any:
    """`row` without `path`, copying only the containers along the way."""
    head, _, rest = path.partition(".")
    if not isinstance(row, Mapping) or head not in row:
        return row
    copied = dict(row)
    if rest:
        copied[head] = _drop(copied[head], rest)
    else:
        copied.pop(head, None)
    return copied


_MISSING = object()


def _hoist_shared_conventions(
    cells: list[Any],
) -> tuple[list[Any], dict[str, Any]]:
    """Lift the prose every cell shares out of the cells, once.

    Returns the trimmed cells and the hoisted block. A path is hoisted only when
    EVERY cell carries it and they all agree - one dissenting cell keeps the path
    inline on all of them, because a convention stated once must be true of
    everything under it. With no cells nothing is hoisted and the block is empty.
    """
    if not cells:
        return list(cells), {}
    conventions: dict[str, Any] = {}
    for path in _SHARED_CELL_PROSE:
        values = [_dig(cell, path) for cell in cells]
        first = values[0]
        if first is _MISSING or any(value != first for value in values):
            continue
        conventions[path] = first
    if not conventions:
        return list(cells), {}
    trimmed = []
    for cell in cells:
        for path in conventions:
            cell = _drop(cell, path)
        trimmed.append(cell)
    conventions["_note"] = (
        "These are stated once and are true of every cell in "
        "`eligible_policies`; they were removed from the cells themselves, "
        "where they had been repeated verbatim."
    )
    return trimmed, conventions


#: How the bounded view says it chose its cells. ONE string, copied into the
#: view's `narrated` block, the pack markdown and the nightly ledger reason, so
#: a reader of any one of the three knows the basis without the other two.
NARRATION_SELECTED_BY = "evidence count descending, then recipe_id/family/side"


def _encoded_chars(value: Any) -> int:
    """The size of `value` exactly as `_evidence_package` encodes and hashes it."""
    return len(json.dumps(value, sort_keys=True, default=str).encode("utf-8"))


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _policy_cell_order_key(cell: Any) -> tuple[int, str, str, str]:
    """Order the eligible policy cells for the bounded view. A SIZE RULE ONLY.

    **Gate #43 is a refusal, not a check**: no cell of a frozen research grid
    may be read for a verdict before its declared window closes, including by
    the code that assembles the narration and including if an early cell looks
    good. So the only thing this key is allowed to know is HOW MUCH evidence a
    cell rests on - never how that evidence turned out. `mean_r`, `win_rate`,
    `profit_factor`, `expectancy`, the bootstrap bounds and the trimmed means
    are all deliberately absent: a cell that looks good early is exactly what
    the frozen window protects, and ranking the narration by result would hand
    the model the flattering half of the grid and call it a selection.

    `stats.n` is the outcome-row count the eligibility floor itself gates on
    ("n >= 30 OUTCOME ROWS"), so what survives the budget is the most-measured
    evidence. `stats.n_episodes` sits beside it and is equal on all 619 cells of
    the 2026-09-04 pack; it is the honest sample size and does not yet gate
    (BD-81), so if the floor ever moves onto it this key moves with it.

    Ties break on `recipe_id`, then `family`, then `side` - identifiers, so the
    order is total and the same list comes back on every run of the same pack.
    """
    stats = cell.get("stats") if isinstance(cell, Mapping) else None
    if not isinstance(stats, Mapping):
        stats = {}
    identity = cell if isinstance(cell, Mapping) else {}
    return (
        -_int_or_zero(stats.get("n")),
        str(identity.get("recipe_id") or ""),
        str(identity.get("family") or ""),
        str(identity.get("side") or ""),
    )


def _after_like_order_key(cell: Any) -> tuple[int, str, int]:
    """The same size rule for the after-like grid, whose cells are FLAT.

    Their evidence count is a TOP-LEVEL `n_episodes` - there is no `stats`
    wrapper on this grid - so reading `stats.n` here would silently order every
    cell as zero. Gate #43 covers this grid by name (`after_like_entry_grid_v1`,
    still `collecting`), so no result statistic is in this key either.
    """
    identity = cell if isinstance(cell, Mapping) else {}
    return (
        -_int_or_zero(identity.get("n_episodes")),
        str(identity.get("entry") or ""),
        _int_or_zero(identity.get("day_offset")),
    )


def _narration_budget(budget: int | None = None) -> int:
    if budget is not None:
        return int(budget)
    import ai_summary

    return int(ai_summary.local_evidence_budget_chars())


def _fill_to_budget(
    cells: list[Any], *, budget: int, used: int
) -> tuple[list[Any], int]:
    """Add cells in the given order while the NEXT one still fits.

    Sizes are measured per cell rather than by re-encoding the whole view each
    time: a JSON list contributes each element's own encoding plus the two-char
    `", "` separator between them, and `json.dumps` renders a nested value
    exactly as it renders it alone. 619 re-encodings of a 78,000-char view would
    be 48 MB of work inside a nightly job for the same answer.
    """
    kept: list[Any] = []
    for cell in cells:
        cost = _encoded_chars(cell) + (2 if kept else 0)
        if used + cost > budget:
            break
        kept.append(cell)
        used += cost
    return kept, used


def _bounded_narration_view(
    pack: Mapping[str, Any], budget: int
) -> tuple[dict[str, Any], int]:
    """The view, cut to `budget`, plus the size of its fixed head.

    Select, then fill. The head - everything that is not a cell - is encoded
    first, because it is what makes the rest readable: without the gate, the
    coverage and the conventions the cells are numbers with no basis. Cells are
    then added in `_policy_cell_order_key` order until the next one would cross
    the budget, and the after-like ELIGIBLE cells (P10 C3) follow under the same
    rule.

    The head is measured with `narrated` holding the TOTALS rather than the
    kept counts, so the placeholder can only ever be longer than the truth: K is
    at most N, so it cannot have more digits, and the finished view is therefore
    never larger than what was budgeted for. The head size is returned because
    the refusal needs it - "the head alone is too big" and "the head fits but no
    cell does" are different failures and only one of them is about the grid.
    """
    view = _whole_narration_view(pack)
    policy = sorted(view["eligible_policies"], key=_policy_cell_order_key)
    after_like = sorted(view["after_like_eligible"], key=_after_like_order_key)
    view["eligible_policies"] = []
    view["after_like_eligible"] = []
    view["narrated"] = {
        "eligible_policy_cells": len(policy),
        "of": len(policy),
        "selected_by": NARRATION_SELECTED_BY,
        "after_like_cells": len(after_like),
        "of_after_like": len(after_like),
    }
    head_chars = _encoded_chars(view)
    kept_policy, used = _fill_to_budget(policy, budget=budget, used=head_chars)
    kept_after_like, _used = _fill_to_budget(after_like, budget=budget, used=used)
    view["eligible_policies"] = kept_policy
    view["after_like_eligible"] = kept_after_like
    view["narrated"]["eligible_policy_cells"] = len(kept_policy)
    view["narrated"]["after_like_cells"] = len(kept_after_like)
    return view, head_chars


def narration_view(
    pack: Mapping[str, Any], *, budget: int | None = None
) -> dict[str, Any]:
    """What the model is asked to narrate - a VIEW, never the whole pack, and
    BOUNDED to what the local model can actually read.

    N3 (2026-09-05): the view used to carry every eligible cell, and the whole
    of it or nothing was sent. Gate #59's lake recompute took the grid from
    23,802 recipe outcomes to 141,299 overnight, and with it the eligible block
    from 128 cells to 619 - 658,292 chars against a 78,119-char budget. The
    ledger had already read `narration absent` on 09-02, 09-03 and 09-04. No
    budget a 64k-context model can read will ever fit 658k chars, so "raise the
    budget" is not a fix; the view has to SELECT, and it has to say that it did.

    The selection is a SIZE rule and never a ranking by result - see
    `_policy_cell_order_key` for why gate #43 makes that binding - and the
    `narrated` block states K of N in the view, the pack markdown and the
    nightly ledger line, so nothing reads as "these were the findings".
    """
    view, _head_chars = _bounded_narration_view(pack, _narration_budget(budget))
    return view


def _whole_narration_view(pack: Mapping[str, Any]) -> dict[str, Any]:
    """Every eligible cell, deduplicated but UNBOUNDED - the input to the cut.

    Kept separate from `narration_view` so the bounding step has something whole
    to select from, and so the refusal can price the whole view against the head
    without a second construction of either.

    The pack is the deterministic product and it grew: P3 added the ineligible
    block, the excluded families and the coverage detail, and the recipe grid
    grew again with P8. On 2026-09-01 the whole pack encoded to ~442,000 chars
    (~176,800 tokens) against a 65,536-token window, and the server SHEARED it -
    three nights running, three superseding packs, no narration.

    A narration does not need the pack. It needs what a person would read first:

    * the GATE - whether there is enough evidence to say anything at all;
    * COVERAGE and EVIDENCE_SHAPE - what this was computed over;
    * every ELIGIBLE policy cell, because those ARE the finding;
    * the excluded-families block, because a family excluded by role is a fact
      about the question and not about performance;
    * COUNTS of what was dropped, so the model can say "and 71 thin cells were
      not shown" rather than being handed 71 thin cells.

    Deliberately absent: the ineligible block's ROWS (bounded at 40 and still the
    bulk of the text, and the reason they are published is for a human to scan,
    not for a model to average) and the raw outcome list (12,439 rows on the
    2026-09-02 pack - the input to the arithmetic, never its answer).

    The eligible cells are also DEDUPLICATED, which is where most of the
    remaining size went. Every cell repeats the same four prose constants - the
    eligibility rule, the n-floor note, the profit-factor convention and the
    bootstrap interval - and on the 2026-09-01 pack that was ~900 identical
    characters inside each 1,900-character cell, 30,000 chars of one paragraph
    written 33 times. They are stated ONCE under `conventions` and removed from
    the cells, so nothing is lost and the view halved (65,816 -> ~33,000 chars),
    which is the difference between six cells of headroom and sixty.

    A constant that DISAGREES between cells is never unified: it stays inline on
    the cells that differ and is left out of `conventions` entirely. Hoisting a
    definition two cells do not share would silently restate one of them.
    """
    eligible = list(pack.get("eligible_policies") or ())
    ineligible = list(pack.get("ineligible_policies") or ())
    contexts = list(pack.get("market_context_cells") or ())
    trimmed, conventions = _hoist_shared_conventions(eligible)
    return {
        "schema": pack.get("schema"),
        "generated_at": pack.get("generated_at"),
        "built_by_commit": pack.get("built_by_commit"),
        "recipe_ids": list(pack.get("recipe_ids") or ()),
        "entry_contract": pack.get("entry_contract"),
        "data_contract": dict(pack.get("data_contract") or {}),
        "gate": dict(pack.get("gate") or {}),
        "coverage": dict(pack.get("coverage") or {}),
        "evidence_shape": dict(pack.get("evidence_shape") or {}),
        "non_trade_families": list(pack.get("non_trade_families") or ()),
        "non_trade_families_note": pack.get("non_trade_families_note"),
        "eligible_policies": trimmed,
        # P10 C3: the after-like grid's ELIGIBLE cells only. The ineligible ones
        # are in the pack on disk; a cell under the floor is a cell the narration
        # must not describe, and handing the model twenty thin cells is how the
        # prompt grew past the window in the first place (R3).
        "after_like_eligible": [
            cell
            for cell in ((pack.get("after_like") or {}).get("cells") or ())
            if cell.get("eligible")
        ],
        "conventions": conventions,
        "not_a_control_signal": pack.get("not_a_control_signal"),
        "omitted": {
            "ineligible_policies": len(ineligible),
            "market_context_cells": len(contexts),
            "outcome_rows": int((pack.get("coverage") or {}).get("outcomes") or 0),
            "eligible_policy_cells_dropped": pack.get("eligible_policy_cells_dropped"),
            "ineligible_policy_cells_dropped": pack.get(
                "ineligible_policy_cells_dropped"
            ),
            "why": (
                "The ineligible cells, the market-context cells and the raw "
                "outcome rows are all in the pack on disk. They are omitted here "
                "because they are input rather than finding, and sending them is "
                "what sheared the prompt."
            ),
        },
    }


_BUILT_BY_COMMIT: str | None = None


def _built_by_commit() -> str:
    """The short commit of the code building this pack, or "unknown".

    Read ONCE per process and cached: it cannot change while the process runs,
    and a subprocess per pack would be a git call inside a nightly job for a
    value that is constant.

    Fails OPEN to "unknown". A missing commit is a slightly less traceable pack;
    a raise here would cost the pack entirely, and the pack is the product.
    """
    global _BUILT_BY_COMMIT
    if _BUILT_BY_COMMIT is not None:
        return _BUILT_BY_COMMIT
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=10,
        )
        _BUILT_BY_COMMIT = (result.stdout or "").strip() or "unknown"
    except Exception:  # noqa: BLE001 - provenance never costs the pack
        _BUILT_BY_COMMIT = "unknown"
    return _BUILT_BY_COMMIT


def _evidence_package(pack: Mapping[str, Any]) -> dict[str, Any]:
    """The package the model receives. Its content is the VIEW, not the pack.

    The hash is taken over WHAT WAS ACTUALLY SENT, so a narration can always be
    traced to the exact bytes that produced it - hashing the pack while sending
    a view would make that traceability a lie.

    Refuses over budget rather than sending and hoping. `ai_summary` already
    computes what the local model can read (`local_evidence_budget_chars`, capped
    to the context window); a prompt above it is not a longer answer, it is a
    silently sheared one, and output generated from a sheared prompt is not
    trustworthy even when it validates.

    Since N3 the view CUTS itself to that budget, so the refusal narrows to the
    one case a cut cannot answer: the fixed head plus the FIRST cell does not
    fit. That is a fact about the head or the model's window, never about the
    grid being large, and the message names the head's size so the ledger line
    says which. A pack with no eligible cells at all is not this failure - the
    gate stops the job before the model is called.
    """
    import ai_summary

    budget = int(ai_summary.local_evidence_budget_chars())
    view, head_chars = _bounded_narration_view(pack, budget)
    encoded = json.dumps(view, sort_keys=True, default=str).encode("utf-8")
    narrated = view["narrated"]
    if len(encoded) > budget or (narrated["of"] and not narrated["eligible_policy_cells"]):
        raise NarrationTooLarge(
            f"the narration view is {len(encoded)} chars against a budget of "
            f"{budget}; its fixed head alone is {head_chars} chars, so not even "
            f"the first of the {narrated['of']} eligible cell(s) fits. Refusing "
            "to send a prompt the model would shear. The deterministic pack is "
            f"published and complete. Raise "
            f"'{ai_summary.LOCAL_EVIDENCE_BUDGET_SETTING_KEY}' or the model's "
            "num_ctx; narrating fewer cells is already what the view does."
        )
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
        "content": view,
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
        # N3: the same block that is inside the view the hash covers. A reader
        # who opens the narration alone and never the pack must still be able to
        # see that it was written over K of N cells and on what basis.
        "narrated": dict(package["sources"][0]["content"]["narrated"]),
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


#: The supersession suffix :func:`_superseding` appends: `<date>.json` first,
#: then `<date>.1.json`, `<date>.2.json`. Captured once here because the READER
#: has to undo exactly what the writer did, and the two drifting apart is the
#: whole of R4 B1.
_ORDINAL_SUFFIX = re.compile(r"^(?P<stem>.+)\.(?P<ordinal>\d+)$")


def pack_sort_key(path: Path) -> tuple[str, int]:
    """`(session stem, supersession ordinal)` - the order the packs were written.

    **Never sort these names as strings.** In ASCII `.` is 0x2E and `1` is 0x31,
    so `"2026-09-01.1.json" < "2026-09-01.json"` and `sorted(...)[-1]` hands back
    the FIRST pack written for the day - the one every re-run superseded. Both
    Weekend Prep readers did that, and on 2026-09-03 the live store held three
    packs for 2026-09-01: the reader took the original (no `eligible_policies`
    key at all) while the `.2` pack had 33 eligible cells.

    The ordinal is parsed as an INTEGER, so a tenth re-run sorts after a ninth;
    a string sort would put `.10` before `.9`. The session stem sorts first, so a
    re-run of yesterday never outranks today's first pack.
    """
    stem = path.stem
    match = _ORDINAL_SUFFIX.match(stem)
    if match is None:
        return (stem, 0)
    return (match.group("stem"), int(match.group("ordinal")))


def latest_pack_path(root: Path) -> Path | None:
    """The newest fact pack under `root`, or None when there is not one.

    Narrations live beside the packs and are not packs; a `.md` render is not
    either. Returns None rather than raising for a root that does not exist -
    "the research has not run yet" is an answer, and every caller of this is a
    read that must never take its page down.
    """
    root = Path(root)
    if not root.is_dir():
        return None
    packs = [
        path
        for path in root.rglob("*.json")
        if "narration" not in path.name
    ]
    if not packs:
        return None
    return max(packs, key=pack_sort_key)


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
            outcome_rows,
            occurrence_map,
            contexts,
            coverage=coverage,
            recipe_ids=(coverage or {}).get("recipe_ids"),
            now=moment,
        )
        target_root = Path(root) if root is not None else _default_root()
        stamp = session_date or moment.date().isoformat()
        # The narration's coverage is decided BEFORE the markdown is written,
        # not after it is narrated: the `.md` is published once, beside one pack
        # for the date (gate #40), and re-rendering it after the model answered
        # would either write a second file or rewrite a published one. The cut
        # is deterministic from the pack and the budget alone - no model call is
        # involved in deciding it - so knowing it early costs nothing, and a
        # failure to compute it leaves the line off rather than costing the pack.
        will_narrate = bool(narrate and pack["gate"]["met"])
        narrated: dict[str, Any] | None = None
        if will_narrate:
            try:
                narrated = dict(narration_view(pack)["narrated"])
            except Exception as exc:  # noqa: BLE001 - a coverage line never costs the pack
                _log.info("Setup research narration coverage unavailable (%s).", exc)
        json_path = _superseding(target_root / str(moment.year) / f"{stamp}.json")
        outputs = [str(_publish(json_path, json.dumps(pack, indent=1, sort_keys=True, default=str) + "\n"))]
        outputs.append(
            str(
                _publish(
                    json_path.with_suffix(".md"), render_markdown(pack, narrated=narrated)
                )
            )
        )
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
        # STATUS OK, AND NO RETRY (R3). The deterministic pack IS this job's
        # product; the narration is words over it. Returning
        # `degraded_no_narrative` under `max_attempts=3` made the runner re-run
        # the WHOLE job twice more - and this job is a ten-minute lake pass, so
        # on 2026-09-01 one truncated prompt produced three superseding packs at
        # 03:55, 04:30 and 05:00, 29 minutes of reads, and three identical
        # failures. Re-reading the lake cannot fix a prompt that is too long.
        #
        # The digest already works this way and says so in the runner: its facts
        # are written even when the model is down. This is the same shape, with
        # one difference that matters - the digest's retry is CHEAP and this
        # one is not, so this returns ok rather than a status the runner will
        # re-attempt at all.
        #
        # If a narration retry is ever wanted it must read the pack already on
        # disk and call the model again. It must never re-enter the lake.
        _log.info("Setup research narration unavailable (%s).", exc)
        return {
            "status": "ok",
            "model": "",
            "reason": f"{base}; narration absent: {exc}",
            "outputs": outputs,
        }
    # N3: the ledger line says HOW MUCH was narrated, because "narrated" alone
    # read the same on the 47-cell night and on the 619-cell one.
    covered = narration.get("narrated") or {}
    told = (
        f"; narrated {_int_or_zero(covered.get('eligible_policy_cells'))} of "
        f"{_int_or_zero(covered.get('of'))} eligible cell(s)"
        if covered
        else "; narrated"
    )
    return {"status": "ok", "model": str(narration.get("model") or ""), "reason": base + told, "outputs": outputs}


__all__ = ["FACTS_SCHEMA", "NARRATION_SCHEMA", "build_fact_pack", "render_markdown", "run_setup_research"]
