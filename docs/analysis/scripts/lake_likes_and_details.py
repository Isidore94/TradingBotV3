"""Check like links, bar_derived details, and swing gross_r."""

from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))

from research_warehouse.store import ResearchStore
from research_warehouse.outcomes import SWING_HOUSE_V1, TERMINAL_RESULT_STATES, latest_outcomes
from research_warehouse.occurrences import latest_occurrences
from research_warehouse import like_links

store = ResearchStore.open()

# 1. Like links from bronze, read through the PRODUCTION reader (Q3.4).
#
# This block used to read a `basis` key with an "unknown" default. The real
# field is `match_basis`
# (`LikeLink.as_payload`), so every link on the lake counted as `unknown` and the
# 2026-09-04 process review printed that as a fact about the lake. An audit that
# invents a field name measures itself. `like_links.basis_of` is the only reader,
# and an unreadable payload stops this script with the row printed rather than
# quietly filling a bucket.
print("=== Like links (bronze_like_occurrence_link) ===")
total_links = 0
basis_counts = Counter()
for part in ("month=2026-08", "month=2026-09"):
    try:
        rows = store.read_table("bronze_like_occurrence_link", part).to_pylist()
    except Exception as e:
        print(f"  {part}: {e}")
        continue
    total_links += len(rows)
    for row in rows:
        payload = row.get("payload", "")
        if isinstance(payload, str):
            try:
                p = json.loads(payload)
            except Exception as e:
                raise SystemExit(f"AUDIT ERROR: undecodable like-link payload in {part}: {row!r} ({e})")
        else:
            p = payload
        try:
            basis_counts[like_links.basis_of(p)] += 1
        except ValueError as e:
            raise SystemExit(f"AUDIT ERROR: unreadable like-link payload in {part}: {row!r} ({e})")

print(f"  Total like links: {total_links}")
print(f"  Basis distribution: {dict(basis_counts)}")

# 2. bar_derived completeness check
print("\n=== bar_derived COMPLETE share ===")
for tf in ("M15", "M30", "H1", "H2"):
    for month in ("month=2026-08", "month=2026-09"):
        part = f"timeframe={tf}/{month}"
        try:
            rows = store.read_rows("bar_derived", part, columns=["aggregation_status"])
            status_counts = Counter(r.get("aggregation_status", "?") for r in rows)
            total = len(rows)
            complete = status_counts.get("COMPLETE", 0)
            print(f"  {part}: {total} rows, {complete} COMPLETE ({complete/total*100:.1f}%)" if total else f"  {part}: 0 rows")
        except Exception as e:
            print(f"  {part}: {e}")

# 3. Swing targeted row detail
print("\n=== Swing house v1 TARGETED rows ===")
occ_view = latest_occurrences(store, 2026)
outcome_view = latest_outcomes(store)
for key, row in outcome_view.items():
    if row.get("recipe_id") != SWING_HOUSE_V1.recipe_id:
        continue
    if row.get("result_state") == "TARGETED":
        occ = occ_view.get(row.get("occurrence_id", ""), {})
        print(f"  sym={occ.get('symbol')} side={occ.get('side')} net_r={row.get('net_r')} gross_r={row.get('gross_r')} mfe_r={row.get('mfe_r')}")

# 4. How many swing occurrences are OPEN vs terminal?
print("\n=== Swing house v1 open vs terminal ===")
open_count = 0
terminal_count = 0
for key, row in outcome_view.items():
    if row.get("recipe_id") != SWING_HOUSE_V1.recipe_id:
        continue
    if row.get("result_state") in TERMINAL_RESULT_STATES:
        terminal_count += 1
    else:
        open_count += 1
print(f"  Terminal: {terminal_count}, Open: {open_count}")

# 5. Check occurrence capture_mode via bar_m5 sampling
print("\n=== Capture mode on bar_m5 (sample) ===")
sample_rows = store.read_rows("bar_m5", "month=2026-09", columns=["capture_mode", "symbol"])
modes = Counter(r.get("capture_mode", "?") for r in sample_rows[:10000])
print(f"  Sample: {dict(modes)}")
