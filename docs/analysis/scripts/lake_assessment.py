"""Lake assessment 2026-09-04: read the repaired research lake and report.

Run from the repo root:
    .venv\\Scripts\\python.exe docs\\analysis\\scripts\\lake_assessment.py

Writes results to docs/analysis/lake_assessment_output.json.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

from research_warehouse.store import ResearchStore
from research_warehouse.outcomes import (
    SWING_HOUSE_V1, CONTROL_FIXED_1R2R_V1, CONTROL_TIME_ONLY_V1,
    M5_CLOSE_RECIPES, HTF_LRSI_RECIPES,
    TERMINAL_RESULT_STATES, latest_outcomes,
)
from research_warehouse.occurrences import latest_occurrences
from research_warehouse import outcome_coverage, trial_ledger
from evidence_stats import (
    MIN_REPORTABLE_N, lately_window,
)
from ai_jobs.setup_research import MIN_SYMBOLS, MIN_SESSIONS
import setup_registry

OUT = REPO / "docs" / "analysis" / "lake_assessment_output.json"
results = {}


def _finite(values):
    out = []
    for v in values:
        try:
            if v is None or v == "":
                continue
            f = float(v)
            if math.isnan(f) or math.isinf(f):
                continue
            out.append(f)
        except (TypeError, ValueError):
            continue
    return out


def wilson_lb(wins, total, z=1.96):
    if total == 0:
        return 0.0
    p = wins / total
    denom = 1 + z * z / total
    centre = p + z * z / (2 * total)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (centre - spread) / denom


print("Opening store...")
store = ResearchStore.open()
print(f"Store root: {store.root}")

# ── Q1: Integrity ──────────────────────────────────────────────────────────

print("\n=== Q1: Integrity ===")
q1 = {}

# bar_m5 grain check
for month_label in ("month=2026-08", "month=2026-09"):
    rows = store.read_rows("bar_m5", month_label, columns=["symbol", "interval_start", "provider", "revision_id"])
    grains = [(r["symbol"], str(r["interval_start"]), r.get("provider", ""), r.get("revision_id", "")) for r in rows]
    total = len(grains)
    unique = len(set(grains))
    dupes = total - unique
    q1[f"bar_m5_{month_label}_rows"] = total
    q1[f"bar_m5_{month_label}_dupes"] = dupes
    print(f"  bar_m5 {month_label}: {total} rows, {dupes} duplicate grains")

# bar_derived M15 constituent check
print("  Checking bar_derived M15...")
for month_label in ("month=2026-08", "month=2026-09"):
    part = f"timeframe=M15/{month_label}"
    try:
        rows = store.read_rows("bar_derived", part,
                               columns=["constituent_count", "constituent_expected"])
        violations = sum(1 for r in rows
                         if r.get("constituent_count") is not None
                         and r.get("constituent_expected") is not None
                         and int(r["constituent_count"]) > int(r["constituent_expected"]))
        q1[f"bar_derived_M15_{month_label}_rows"] = len(rows)
        q1[f"bar_derived_M15_{month_label}_violations"] = violations
        print(f"  bar_derived M15 {month_label}: {len(rows)} rows, {violations} constituent violations")
    except Exception as e:
        q1[f"bar_derived_M15_{month_label}_error"] = str(e)
        print(f"  bar_derived M15 {month_label}: error {e}")

# outcome view row count
print("  Reading outcome_path (latest view)...")
outcome_view = latest_outcomes(store)
q1["outcome_view_rows"] = len(outcome_view)
print(f"  outcome_path latest view: {len(outcome_view)} rows")

# coverage check
cov_path = outcome_coverage.coverage_path(store.root)
if cov_path.exists():
    lines = cov_path.read_text().strip().splitlines()
    recompute_lines = [l for l in lines if "outcomes_recompute" in l]
    buckets_covered = set()
    for line in recompute_lines:
        try:
            obj = json.loads(line)
            buckets_covered.add(obj.get("bucket"))
        except Exception:
            pass
    q1["recompute_buckets_covered"] = len(buckets_covered)
    q1["recompute_lines"] = len(recompute_lines)
    print(f"  Recompute coverage: {len(buckets_covered)}/32 buckets, {len(recompute_lines)} lines")

# AMBIGUOUS_BAR and CENSORED shares
state_counts = Counter()
for row in outcome_view.values():
    state_counts[row.get("result_state", "UNKNOWN")] += 1
q1["result_state_distribution"] = dict(state_counts)
ambiguous_share = state_counts.get("AMBIGUOUS_BAR", 0) / max(len(outcome_view), 1)
q1["ambiguous_bar_share"] = round(ambiguous_share, 4)
print(f"  Result states: {dict(state_counts)}")
print(f"  AMBIGUOUS_BAR share: {ambiguous_share:.2%}")

# Per-recipe row counts for implausibility check
recipe_counts = Counter()
for row in outcome_view.values():
    recipe_counts[row.get("recipe_id", "?")] += 1
q1["rows_per_recipe_top10"] = dict(recipe_counts.most_common(10))
q1["rows_per_recipe_bottom10"] = dict(recipe_counts.most_common()[-10:])
print(f"  Recipes with rows: {len(recipe_counts)}")

results["Q1"] = q1

# ── Load occurrences ───────────────────────────────────────────────────────

print("\n=== Loading occurrences ===")
occ_view = latest_occurrences(store, 2026)
print(f"  {len(occ_view)} current occurrences")

# Build episode map
occ_episodes = {}
for oid, occ in occ_view.items():
    cid = occ.get("dependency_cluster_id", oid)
    occ_episodes[oid] = cid

# ── Q2: Swings ─────────────────────────────────────────────────────────────

print("\n=== Q2: Swings ===")
swing_recipes = {SWING_HOUSE_V1.recipe_id, CONTROL_FIXED_1R2R_V1.recipe_id, CONTROL_TIME_ONLY_V1.recipe_id}

# Group outcomes by (setup, side, recipe)
swing_cells = defaultdict(list)
for key, row in outcome_view.items():
    rid = row.get("recipe_id", "")
    if rid not in swing_recipes:
        continue
    oid = row.get("occurrence_id", "")
    occ = occ_view.get(oid, {})
    setup = occ.get("canonical_setup_id", "UNKNOWN")
    side = occ.get("side", "UNKNOWN")
    state = row.get("result_state", "")
    if state not in TERMINAL_RESULT_STATES:
        continue
    swing_cells[(setup, side, rid)].append(row)

q2_table = []
for (setup, side, rid), rows in sorted(swing_cells.items()):
    nr = _finite([r.get("net_r") for r in rows])
    n = len(rows)
    episodes = len({occ_episodes.get(r.get("occurrence_id", ""), r.get("occurrence_id", "")) for r in rows})
    if not nr:
        continue
    wins = sum(1 for x in nr if x > 0)
    wr = wins / len(nr) if nr else 0
    wlb = wilson_lb(wins, len(nr))
    mean_r = sum(nr) / len(nr) if nr else 0
    med_r = sorted(nr)[len(nr) // 2] if nr else 0
    r_s5 = _finite([r.get("r_at_s5") for r in rows])
    r_s10 = _finite([r.get("r_at_s10") for r in rows])
    mean_s5 = sum(r_s5) / len(r_s5) if r_s5 else None
    mean_s10 = sum(r_s10) / len(r_s10) if r_s10 else None

    # Check eligibility
    symbols = len({occ_view.get(r.get("occurrence_id", ""), {}).get("symbol", "") for r in rows})
    sessions_set = set()
    for r in rows:
        ea = occ_view.get(r.get("occurrence_id", ""), {}).get("event_at")
        if ea:
            sessions_set.add(str(ea)[:10])
    n_sessions = len(sessions_set)
    eligible = (len(nr) >= MIN_REPORTABLE_N and symbols >= MIN_SYMBOLS and n_sessions >= MIN_SESSIONS)

    q2_table.append({
        "setup": setup, "side": side, "recipe": rid,
        "n": n, "n_with_net_r": len(nr), "n_episodes": episodes,
        "win_rate": round(wr, 3), "wilson_lb": round(wlb, 3),
        "mean_r": round(mean_r, 3), "median_r": round(med_r, 3),
        "mean_r_at_s5": round(mean_s5, 3) if mean_s5 is not None else None,
        "mean_r_at_s10": round(mean_s10, 3) if mean_s10 is not None else None,
        "symbols": symbols, "sessions": n_sessions,
        "eligible": eligible,
    })

q2_table.sort(key=lambda x: -x["wilson_lb"])
results["Q2"] = q2_table
print(f"  {len(q2_table)} cells; {sum(1 for x in q2_table if x['eligible'])} eligible")
for row in q2_table[:10]:
    print(f"    {row['setup']:40s} {row['side']:5s} {row['recipe']:25s} n={row['n']:4d} ep={row['n_episodes']:4d} WR={row['win_rate']:.1%} WLB={row['wilson_lb']:.1%} meanR={row['mean_r']:+.2f}")

# ── Q2 market context splits ──────────────────────────────────────────────

print("\n  Loading market context...")
ctx_rows = store.read_rows("setup_market_context", "year=2026")
ctx_by_occ = defaultdict(list)
for r in ctx_rows:
    ctx_by_occ[r.get("occurrence_id", "")].append(r)

# For eligible swing_house_v1 cells, split by D1 bias
eligible_setups = [(r["setup"], r["side"]) for r in q2_table
                   if r["eligible"] and r["recipe"] == SWING_HOUSE_V1.recipe_id]

q2_splits = []
for setup, side in eligible_setups:
    cell_rows = swing_cells.get((setup, side, SWING_HOUSE_V1.recipe_id), [])
    by_bias = defaultdict(list)
    for row in cell_rows:
        oid = row.get("occurrence_id", "")
        ctx = ctx_by_occ.get(oid, [])
        d1_ctx = [c for c in ctx if c.get("timeframe") == "D1"]
        if d1_ctx:
            bias = d1_ctx[0].get("bias_label", "unknown")
        else:
            bias = "no_context"
        by_bias[bias].append(row)

    for bias, brows in by_bias.items():
        nr = _finite([r.get("net_r") for r in brows])
        eps = len({occ_episodes.get(r.get("occurrence_id", ""), "") for r in brows})
        if len(nr) < MIN_REPORTABLE_N or eps < MIN_SYMBOLS:
            continue
        wins = sum(1 for x in nr if x > 0)
        q2_splits.append({
            "setup": setup, "side": side, "d1_bias": bias,
            "n": len(nr), "n_episodes": eps,
            "win_rate": round(wins / len(nr), 3),
            "wilson_lb": round(wilson_lb(wins, len(nr)), 3),
            "mean_r": round(sum(nr) / len(nr), 3),
        })

results["Q2_splits"] = q2_splits
if q2_splits:
    print(f"  {len(q2_splits)} context splits clearing episode floor")

# ── Q3: Day trades ─────────────────────────────────────────────────────────

print("\n=== Q3: Day trades (M5-close recipes) ===")
m5_recipe_ids = {r.recipe_id for r in M5_CLOSE_RECIPES}

m5_cells = defaultdict(list)
for key, row in outcome_view.items():
    rid = row.get("recipe_id", "")
    if rid not in m5_recipe_ids:
        continue
    oid = row.get("occurrence_id", "")
    occ = occ_view.get(oid, {})
    setup = occ.get("canonical_setup_id", "UNKNOWN")
    side = occ.get("side", "UNKNOWN")
    state = row.get("result_state", "")
    if state not in TERMINAL_RESULT_STATES:
        continue
    m5_cells[(setup, side, rid)].append(row)

# Aggregate by recipe
q3_by_recipe = defaultdict(list)
for (setup, side, rid), rows in m5_cells.items():
    q3_by_recipe[rid].extend(rows)

q3_table = []
for rid, rows in sorted(q3_by_recipe.items(), key=lambda x: -len(x[1])):
    nr = _finite([r.get("net_r") for r in rows])
    mfe = _finite([r.get("mfe_r") for r in rows])
    episodes = len({occ_episodes.get(r.get("occurrence_id", ""), "") for r in rows})
    if not nr:
        continue
    n = len(nr)
    mean_r = sum(nr) / n
    mfe_above_1 = sum(1 for x in mfe if x >= 1.0) / len(mfe) if mfe else 0
    med_mfe = sorted(mfe)[len(mfe) // 2] if mfe else None
    time_to_mfe = _finite([r.get("time_to_mfe_min") for r in rows])
    med_ttm = sorted(time_to_mfe)[len(time_to_mfe) // 2] if time_to_mfe else None

    q3_table.append({
        "recipe": rid, "n": n, "n_episodes": episodes,
        "mean_net_r": round(mean_r, 3),
        "mfe_above_1r_share": round(mfe_above_1, 3),
        "median_mfe_r": round(med_mfe, 3) if med_mfe is not None else None,
        "median_time_to_mfe_min": round(med_ttm, 1) if med_ttm is not None else None,
    })

q3_table.sort(key=lambda x: -(x.get("median_mfe_r") or 0))
results["Q3_by_recipe"] = q3_table[:20]

# Confirm negative mean net_r for n>500 recipes
neg_recipes = [r for r in q3_table if r["n"] > 500 and r["mean_net_r"] < 0]
print(f"  {len(q3_table)} recipes with terminal outcomes")
print(f"  Recipes with n>500 and negative mean net_r: {len(neg_recipes)}")
for r in neg_recipes[:5]:
    print(f"    {r['recipe']:50s} n={r['n']:5d} meanR={r['mean_net_r']:+.3f} medMFE={r.get('median_mfe_r','?')}")

# By entry hour (ET)
q3_by_hour = defaultdict(list)
for key, row in outcome_view.items():
    rid = row.get("recipe_id", "")
    if rid not in m5_recipe_ids:
        continue
    state = row.get("result_state", "")
    if state not in TERMINAL_RESULT_STATES:
        continue
    entry_at = row.get("entry_at") or row.get("trigger_at")
    if entry_at:
        try:
            if isinstance(entry_at, datetime):
                dt = entry_at
            else:
                dt = datetime.fromisoformat(str(entry_at).replace("Z", "+00:00"))
            # ET is UTC-4 for EDT
            hour = (dt.hour - 4) % 24 if dt.tzinfo else dt.hour
            q3_by_hour[hour].append(row)
        except Exception:
            pass

q3_hours = []
for hour in sorted(q3_by_hour.keys()):
    rows = q3_by_hour[hour]
    nr = _finite([r.get("net_r") for r in rows])
    mfe = _finite([r.get("mfe_r") for r in rows])
    if not nr:
        continue
    q3_hours.append({
        "hour_et": hour, "n": len(nr),
        "mean_net_r": round(sum(nr) / len(nr), 3),
        "median_mfe_r": round(sorted(mfe)[len(mfe) // 2], 3) if mfe else None,
    })
results["Q3_by_hour"] = q3_hours
print("  By hour (ET):")
for h in q3_hours:
    print(f"    {h['hour_et']:02d}:00 n={h['n']:5d} meanR={h['mean_net_r']:+.3f} medMFE={h.get('median_mfe_r','?')}")

# By family
q3_by_family = defaultdict(list)
for key, row in outcome_view.items():
    rid = row.get("recipe_id", "")
    if rid not in m5_recipe_ids:
        continue
    state = row.get("result_state", "")
    if state not in TERMINAL_RESULT_STATES:
        continue
    oid = row.get("occurrence_id", "")
    occ = occ_view.get(oid, {})
    setup = occ.get("canonical_setup_id", "UNKNOWN")
    q3_by_family[setup].append(row)

q3_families = []
for setup, rows in sorted(q3_by_family.items(), key=lambda x: -len(x[1])):
    nr = _finite([r.get("net_r") for r in rows])
    mfe = _finite([r.get("mfe_r") for r in rows])
    if len(nr) < 30:
        continue
    q3_families.append({
        "setup": setup, "n": len(nr),
        "mean_net_r": round(sum(nr) / len(nr), 3),
        "mfe_above_1r_share": round(sum(1 for x in mfe if x >= 1.0) / len(mfe), 3) if mfe else 0,
        "median_mfe_r": round(sorted(mfe)[len(mfe) // 2], 3) if mfe else None,
    })
q3_families.sort(key=lambda x: -(x.get("median_mfe_r") or 0))
results["Q3_by_family"] = q3_families
print(f"  {len(q3_families)} families with n>=30")

# ── Q4: HTF LRSI ──────────────────────────────────────────────────────────

print("\n=== Q4: HTF LRSI ===")
htf_recipe_ids = {r.recipe_id for r in HTF_LRSI_RECIPES}

q4_table = []
for recipe in HTF_LRSI_RECIPES:
    rows = [row for row in outcome_view.values()
            if row.get("recipe_id") == recipe.recipe_id
            and row.get("result_state") in TERMINAL_RESULT_STATES]
    nr = _finite([r.get("net_r") for r in rows])
    episodes = len({occ_episodes.get(r.get("occurrence_id", ""), "") for r in rows})
    n = len(nr)
    mean_r = sum(nr) / n if nr else None
    wins = sum(1 for x in nr if x > 0) if nr else 0
    wlb = wilson_lb(wins, n) if n else 0
    eligible = n >= MIN_REPORTABLE_N
    q4_table.append({
        "recipe": recipe.recipe_id,
        "timeframe": recipe.htf_timeframe,
        "cross": f"{recipe.cross_direction}{recipe.cross_level}",
        "n": n, "n_episodes": episodes,
        "mean_net_r": round(mean_r, 3) if mean_r is not None else None,
        "wilson_lb": round(wlb, 3),
        "floor_met": eligible,
    })

results["Q4"] = q4_table
for r in q4_table:
    status = "PASS" if r["floor_met"] else "FAIL"
    print(f"  {r['recipe']:50s} n={r['n']:4d} ep={r['n_episodes']:4d} meanR={r.get('mean_net_r','?'):>7} WLB={r['wilson_lb']:.3f} {status}")

# ── Q5: Declared grids ────────────────────────────────────────────────────

print("\n=== Q5: Declared grids (collection status) ===")
ledger_path = store.root / trial_ledger.DIAGNOSTICS_DIRNAME / trial_ledger.LEDGER_FILENAME
q5 = {}
if ledger_path.exists():
    entries = []
    for line in ledger_path.read_text().strip().splitlines():
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    q5["ledger_entries"] = len(entries)
    for entry in entries:
        tid = entry.get("trial_id", "?")
        status = entry.get("status", "?")
        cells = entry.get("declared_cells", "?")
        window = entry.get("declared_window", {})
        q5[tid] = {
            "status": status,
            "declared_cells": cells,
            "window": window,
        }
        # Count rows per trial
        recipe_prefix = entry.get("recipe_id_prefix", "")
        if recipe_prefix:
            matching = sum(1 for row in outcome_view.values()
                           if (row.get("recipe_id") or "").startswith(recipe_prefix)
                           and row.get("result_state") in TERMINAL_RESULT_STATES)
            q5[tid]["rows_collected"] = matching
        print(f"  {tid}: status={status}, cells={cells}, window={window}")

# For P8 (SETUP_ENTRY_TIMING_RECIPES) and P10 (after_like)
entry_timing_ids = set()
for recipe in M5_CLOSE_RECIPES:
    if recipe.entry_variant:
        entry_timing_ids.add(recipe.recipe_id)

# Count after-like rows
afterlike_rows = [row for row in outcome_view.values()
                  if (row.get("recipe_id") or "").startswith("afterlike_")]
q5["afterlike_rows_collected"] = len(afterlike_rows)
print(f"  After-like rows collected: {len(afterlike_rows)}")

results["Q5"] = q5

# ── Q6: Likes ──────────────────────────────────────────────────────────────

print("\n=== Q6: Likes ===")
q6 = {}
try:
    like_table = store.read_table("like_occurrence_link", "month=2026-08")
    like_rows_08 = like_table.to_pylist()
except Exception:
    like_rows_08 = []
try:
    like_table = store.read_table("like_occurrence_link", "month=2026-09")
    like_rows_09 = like_table.to_pylist()
except Exception:
    like_rows_09 = []
like_rows = like_rows_08 + like_rows_09
q6["total_like_links"] = len(like_rows)

# Parse basis
basis_counts = Counter()
for row in like_rows:
    payload = row.get("payload", "")
    if isinstance(payload, str):
        try:
            p = json.loads(payload)
        except Exception:
            p = {}
    else:
        p = payload or {}
    basis = p.get("basis", "unknown")
    basis_counts[basis] += 1

q6["basis_distribution"] = dict(basis_counts)
linked = sum(v for k, v in basis_counts.items() if k != "none")
unlinked = basis_counts.get("none", 0)
q6["linked"] = linked
q6["unlinked"] = unlinked
print(f"  Total like links: {len(like_rows)}")
print(f"  Linked: {linked}, Unlinked (basis=none): {unlinked}")
print(f"  Basis distribution: {dict(basis_counts)}")
results["Q6"] = q6

# ── Q7: Lately ─────────────────────────────────────────────────────────────

print("\n=== Q7: What is working lately ===")
start_date, end_date = lately_window()
print(f"  Lately window: {start_date} to {end_date}")

# Filter occurrences by event_at in the lately window
lately_occ_ids = set()
for oid, occ in occ_view.items():
    ea = occ.get("event_at")
    if ea:
        d = str(ea)[:10]
        if start_date <= d <= end_date:
            lately_occ_ids.add(oid)
print(f"  Occurrences in lately window: {len(lately_occ_ids)}")

# Swing lately
q7_swing = []
for (setup, side, rid), rows in swing_cells.items():
    if rid != SWING_HOUSE_V1.recipe_id:
        continue
    lately_rows = [r for r in rows if r.get("occurrence_id") in lately_occ_ids]
    nr_all = _finite([r.get("net_r") for r in rows])
    nr_lately = _finite([r.get("net_r") for r in lately_rows])
    if not nr_all or len(nr_all) < MIN_REPORTABLE_N:
        continue
    eps_all = len({occ_episodes.get(r.get("occurrence_id", ""), "") for r in rows})
    eps_lately = len({occ_episodes.get(r.get("occurrence_id", ""), "") for r in lately_rows})

    wins_all = sum(1 for x in nr_all if x > 0)
    wins_lately = sum(1 for x in nr_lately if x > 0)
    q7_swing.append({
        "setup": setup, "side": side,
        "n_all": len(nr_all), "ep_all": eps_all,
        "wr_all": round(wins_all / len(nr_all), 3),
        "wlb_all": round(wilson_lb(wins_all, len(nr_all)), 3),
        "mean_r_all": round(sum(nr_all) / len(nr_all), 3),
        "n_lately": len(nr_lately), "ep_lately": eps_lately,
        "wr_lately": round(wins_lately / len(nr_lately), 3) if nr_lately else None,
        "wlb_lately": round(wilson_lb(wins_lately, len(nr_lately)), 3) if nr_lately else None,
        "mean_r_lately": round(sum(nr_lately) / len(nr_lately), 3) if nr_lately else None,
    })

q7_swing.sort(key=lambda x: -(x.get("wlb_all") or 0))
results["Q7_swing"] = q7_swing
print(f"  Swing families with n>=30: {len(q7_swing)}")

# Day trade lately
q7_m5 = {}
all_m5_rows = []
lately_m5_rows = []
for key, row in outcome_view.items():
    rid = row.get("recipe_id", "")
    if rid not in m5_recipe_ids:
        continue
    if row.get("result_state") not in TERMINAL_RESULT_STATES:
        continue
    all_m5_rows.append(row)
    if row.get("occurrence_id") in lately_occ_ids:
        lately_m5_rows.append(row)

nr_all = _finite([r.get("net_r") for r in all_m5_rows])
nr_lately = _finite([r.get("net_r") for r in lately_m5_rows])
mfe_all = _finite([r.get("mfe_r") for r in all_m5_rows])
mfe_lately = _finite([r.get("mfe_r") for r in lately_m5_rows])

q7_m5 = {
    "n_all": len(nr_all),
    "mean_net_r_all": round(sum(nr_all) / len(nr_all), 3) if nr_all else None,
    "mfe_above_1r_all": round(sum(1 for x in mfe_all if x >= 1.0) / len(mfe_all), 3) if mfe_all else None,
    "n_lately": len(nr_lately),
    "mean_net_r_lately": round(sum(nr_lately) / len(nr_lately), 3) if nr_lately else None,
    "mfe_above_1r_lately": round(sum(1 for x in mfe_lately if x >= 1.0) / len(mfe_lately), 3) if mfe_lately else None,
}
results["Q7_m5"] = q7_m5
print(f"  M5 close all: n={q7_m5['n_all']} meanR={q7_m5['mean_net_r_all']}")
print(f"  M5 close lately: n={q7_m5['n_lately']} meanR={q7_m5['mean_net_r_lately']}")

# ── Q8: Coverage and blind spots ──────────────────────────────────────────

print("\n=== Q8: Coverage and blind spots ===")
q8 = {}

# Occurrences with no outcome
occ_with_outcomes = {row.get("occurrence_id") for row in outcome_view.values()}
occ_no_outcome = {oid for oid in occ_view if oid not in occ_with_outcomes}
q8["occurrences_no_outcome"] = len(occ_no_outcome)
q8["occurrences_total"] = len(occ_view)
print(f"  Occurrences with no outcome: {len(occ_no_outcome)} of {len(occ_view)}")

# Why no outcome?
no_outcome_statuses = Counter()
for oid in occ_no_outcome:
    occ = occ_view[oid]
    no_outcome_statuses[occ.get("status", "unknown")] += 1
q8["no_outcome_statuses"] = dict(no_outcome_statuses)
print(f"  No-outcome status distribution: {dict(no_outcome_statuses)}")

# Registry setups with zero occurrences
registry = setup_registry.registry()
occ_setups = {occ.get("canonical_setup_id") for occ in occ_view.values()}
registry_zero = set()
for key, entry in registry.items():
    sid = entry.get("canonical_id", key.split("@")[0] if "@" in key else key)
    if sid not in occ_setups:
        registry_zero.add(sid)
q8["registry_setups_zero_occurrences"] = sorted(registry_zero)
print(f"  Registry setups with 0 occurrences: {len(registry_zero)}: {sorted(registry_zero)[:10]}")

# Symbol concentration of eligible cells
eligible_symbols = Counter()
for (setup, side, rid), rows in swing_cells.items():
    nr = _finite([r.get("net_r") for r in rows])
    if len(nr) < MIN_REPORTABLE_N:
        continue
    for r in rows:
        oid = r.get("occurrence_id", "")
        sym = occ_view.get(oid, {}).get("symbol", "?")
        eligible_symbols[sym] += 1

q8["top_symbols_eligible"] = dict(eligible_symbols.most_common(15))
print(f"  Top symbols in eligible cells: {dict(eligible_symbols.most_common(10))}")

# Capture mode split
capture_modes = Counter()
for occ in occ_view.values():
    mode = occ.get("capture_mode", "UNKNOWN")
    capture_modes[mode] += 1
q8["capture_mode_split"] = dict(capture_modes)
print(f"  Capture mode split: {dict(capture_modes)}")

results["Q8"] = q8

# ── Write output ───────────────────────────────────────────────────────────

OUT.write_text(json.dumps(results, indent=2, default=str))
print(f"\nResults written to {OUT}")
