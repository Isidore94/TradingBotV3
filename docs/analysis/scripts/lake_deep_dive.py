"""Deep-dive checks on lake assessment findings."""

from __future__ import annotations
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

from research_warehouse.store import ResearchStore
from research_warehouse.outcomes import (
    SWING_HOUSE_V1, TERMINAL_RESULT_STATES, latest_outcomes,
)
from research_warehouse.occurrences import latest_occurrences

store = ResearchStore.open()
occ_view = latest_occurrences(store, 2026)
outcome_view = latest_outcomes(store)

# 1. Investigate 0% WR on swing_house_v1
print("=== Swing house v1: result state distribution ===")
for side in ("LONG", "SHORT"):
    states = Counter()
    net_rs = []
    gross_rs = []
    for key, row in outcome_view.items():
        if row.get("recipe_id") != SWING_HOUSE_V1.recipe_id:
            continue
        oid = row.get("occurrence_id", "")
        occ = occ_view.get(oid, {})
        if occ.get("side") != side:
            continue
        states[row.get("result_state", "?")] += 1
        nr = row.get("net_r")
        gr = row.get("gross_r")
        if nr is not None:
            try:
                net_rs.append(float(nr))
            except (TypeError, ValueError):
                pass
        if gr is not None:
            try:
                gross_rs.append(float(gr))
            except (TypeError, ValueError):
                pass

    print(f"\n  {side}:")
    print(f"    States: {dict(states)}")
    print(f"    net_r: n={len(net_rs)}")
    if net_rs:
        print(f"      mean={sum(net_rs)/len(net_rs):.3f} min={min(net_rs):.3f} max={max(net_rs):.3f}")
        print(f"      wins (>0): {sum(1 for x in net_rs if x > 0)}")
        print(f"      wins (>-0.01): {sum(1 for x in net_rs if x > -0.01)}")
    if gross_rs:
        print(f"    gross_r: n={len(gross_rs)} mean={sum(gross_rs)/len(gross_rs):.3f} wins={sum(1 for x in gross_rs if x > 0)}")

    # Sample rows
    terminal = [(key, row) for key, row in outcome_view.items()
                if row.get("recipe_id") == SWING_HOUSE_V1.recipe_id
                and occ_view.get(row.get("occurrence_id", ""), {}).get("side") == side
                and row.get("result_state") in TERMINAL_RESULT_STATES]
    print(f"\n    Sample terminal rows (first 5):")
    for _, row in terminal[:5]:
        print(f"      state={row.get('result_state')} net_r={row.get('net_r')} gross_r={row.get('gross_r')} "
              f"target_r={row.get('target_r')} mfe_r={row.get('mfe_r')} "
              f"sym={occ_view.get(row.get('occurrence_id',''),{}).get('symbol')}")

# 2. Check which families have swing outcomes
print("\n\n=== Which families have swing_house_v1 outcomes? ===")
families = Counter()
for key, row in outcome_view.items():
    if row.get("recipe_id") != SWING_HOUSE_V1.recipe_id:
        continue
    oid = row.get("occurrence_id", "")
    occ = occ_view.get(oid, {})
    families[occ.get("canonical_setup_id", "?")] += 1
for fam, cnt in families.most_common():
    print(f"  {fam}: {cnt}")

# 3. Check all families across ALL swing recipes
print("\n\n=== Families across ALL swing recipes ===")
swing_recipes = {"swing_house_v1", "control_fixed_1r2r_v1", "control_time_only_v1"}
family_recipe_counts = defaultdict(lambda: Counter())
for key, row in outcome_view.items():
    rid = row.get("recipe_id", "")
    if rid not in swing_recipes:
        continue
    oid = row.get("occurrence_id", "")
    occ = occ_view.get(oid, {})
    fam = occ.get("canonical_setup_id", "?")
    family_recipe_counts[fam][rid] += 1

for fam in sorted(family_recipe_counts.keys()):
    counts = family_recipe_counts[fam]
    print(f"  {fam:40s} house={counts.get('swing_house_v1',0):5d} "
          f"1r2r={counts.get('control_fixed_1r2r_v1',0):5d} "
          f"time={counts.get('control_time_only_v1',0):5d}")

# 4. Like links - check the bronze dataset more carefully
print("\n\n=== Like links bronze ===")
for part in ("month=2026-08", "month=2026-09"):
    try:
        t = store.read_table("like_occurrence_link", part)
        print(f"  {part}: {t.num_rows} rows, columns={t.column_names}")
    except Exception as e:
        print(f"  {part}: {e}")

# Also check bronze_like_occurrence_link
for part in ("month=2026-08", "month=2026-09"):
    try:
        t = store.read_table("bronze_like_occurrence_link", part)
        print(f"  bronze_ {part}: {t.num_rows} rows")
    except Exception as e:
        print(f"  bronze_ {part}: {e}")

# 5. Check the bar_m5 row counts vs checkpoint claim
print("\n\n=== Bar counts ===")
for month_label in ("month=2026-08", "month=2026-09"):
    t = store.read_table("bar_m5", month_label)
    print(f"  bar_m5 {month_label}: {t.num_rows} rows")

# 6. Check occurrence families overall
print("\n\n=== Occurrence families ===")
occ_families = Counter()
for occ in occ_view.values():
    occ_families[occ.get("canonical_setup_id", "?")] += 1
for fam, cnt in occ_families.most_common():
    print(f"  {fam:40s} {cnt}")

# 7. Check capture_mode field on occurrences
print("\n\n=== Capture mode samples ===")
sample = list(occ_view.values())[:3]
for s in sample:
    print(f"  columns: {list(s.keys())[:20]}")
    print(f"  capture_mode: {s.get('capture_mode', 'NOT_PRESENT')}")
    break
