# Swing house v1 — 0/257 wins investigation

**Date:** 2026-09-04  
**Scope:** Read-only analysis of `swing_house_v1` outcome rows in the research lake  
**Verdict:** The simulator logic is correct. The band data pipeline is 99.15% empty,
so 942 of 947 occurrences run a fallback path that has no target and can never win.

---

## Evidence summary

| Fact | Value |
|------|-------|
| `swing_house_v1` occurrences with outcomes | 947 |
| `feature_snapshot_daily` rows with non-null AVWAP bands | 234 / 27,579 (0.85%) |
| Occurrences that received bands from the snapshot join | 5 / 947 (0.53%) |
| Occurrences that fell to the no-target fallback path | 942 / 947 |
| Terminal fallback rows (STOPPED) | 255 |
| Open fallback rows (still running) | 687 |
| Control recipe (`control_fixed_1r2r_v1`) win rate on the same occurrences | 45.3% blended |
| Rows with MFE > 1R that still ended STOPPED | 101 |

## Root cause chain

### 1. The band data pipeline is 99.15% empty

`feature_snapshot_daily` (year=2026) has 27,579 rows. Only 234 have non-null values in
`avwape_upper_1` through `avwape_lower_3`. The join in `cli._bands_by_occurrence` matches
on `(symbol, trigger_date)`, so an occurrence whose trigger session has no snapshot bands
gets `bands=None`.

942 of 947 swing occurrences hit this case.

### 2. Without bands, the fallback path has no target

In `outcomes.simulate_swing` (line 709–743), the managed path requires `band_1 is not None
and band_2 is not None`. When bands are absent, the code falls to `_walk_plain`:

```
target_price = None          # recipe.target_r is None for swing_house_v1
if band_3 is not None:       # band_3 is also None
    target_price = band_3
_walk_plain(..., target_price=None, ...)
```

`_walk_plain` with `target_price=None` can only exit via:
- **STOPPED** — 2 consecutive closes beyond the structural stop level
- **EXPIRED** — 18 sessions elapse

`STATE_TARGETED` requires `target_hit`, which requires `target_price is not None`.
So winning is structurally impossible on this path.

### 3. The 101 high-MFE STOPPED rows confirm the simulator works correctly

101 rows reached MFE > 1R (some above 10R) but ended STOPPED. Without a target to lock
in gains, the position rides the full path until the stop fires. Examples:

| Symbol | Side | MFE (R) | Gross R | Bands |
|--------|------|---------|---------|-------|
| TDC | SHORT | 15.9 | −12.1 | NO |
| USFD | LONG | 15.9 | −6.7 | NO |
| BEN | LONG | 13.5 | −15.1 | NO |
| CFG | LONG | 9.5 | −21.0 | NO |

These positions saw large favorable excursions but had no mechanism to capture them.

### 4. The one TARGETED row (PCG LONG) has inverted band geometry

PCG is one of only 5 occurrences with bands. Its bands are all below entry:

| Field | Value |
|-------|-------|
| Entry price | 17.84 |
| AVWAPE value | 16.90 |
| Band 1 (trail) | 17.19 (R = −1.89) |
| Band 2 (partial) | 17.49 (R = −1.03) |
| Band 3 (runner target) | 17.78 (R = −0.17) |
| Stop price | 17.50 |

For this LONG, the "target" (band 3) is below entry, so reaching it gives negative gross R.
The MFE = 0.0 is correct: price reached 17.78 without ever going above 17.84. The managed
path correctly recorded this as TARGETED with `gross_r = −0.60` (blended 50% partial at
band 2 + 50% runner at the close where the trail fired).

### 5. The other 4 banded rows

| Symbol | Side | Band geometry | State | Gross R | MFE |
|--------|------|---------------|-------|---------|-----|
| ACHC | LONG | 2 of 3 above entry | STOPPED | −6.0 | 0.0 |
| DELL | LONG | 2 of 3 above entry | OPEN | — | 0.82 |
| DUOL | LONG | All 3 above entry | OPEN | — | 1.40 |
| NOMD | LONG | 2 of 3 above entry | OPEN | — | 0.77 |

ACHC stopped before price reached band 2. The 3 OPEN rows are still running — too early
to judge.

## What this is NOT

- **Not a simulator logic defect.** The `_walk_managed` and `_walk_plain` implementations
  are correct for the inputs they receive. The managed path's partial/trail/runner logic
  is internally consistent.
- **Not bad occurrences.** The control recipe (`control_fixed_1r2r_v1`) runs on the same
  947 occurrences with a fixed 1R stop and 2R target and shows a 45.3% blended win rate.
  The setups find real edges.
- **Not a cost-model problem.** Gross R (before transaction costs) is also 0 wins.

## What it IS

A **band data pipeline gap**: `feature_snapshot_daily` writes AVWAP band values for < 1%
of its rows, starving the managed-path simulator of the levels it needs to define targets,
partials, and trails.

## Recommendations

1. **Diagnose why `feature_snapshot_daily` bands are 99% null.** The column names exist
   (`avwape_upper_1` through `avwape_lower_3`); the values are not being written. Check the
   snapshot builder's band-population step — it may be gated on a condition most symbols
   fail, or the AVWAP band computation may not run for most of the universe.

2. **Do not tune the simulator until the band pipeline is fixed.** Every parameter change
   is meaningless while 99.5% of rows take the no-target fallback.

3. **After fixing bands, recompute outcomes for the affected partition(s)** using
   `research_warehouse.cli recompute-outcomes`.

4. **The fallback path itself is debatable.** When bands are missing, should the simulator
   skip the row (return None), or run with no target? Currently it runs to a guaranteed
   loss. Returning None would be more honest — the recipe *requires* bands to function.

---

*Read-only investigation. No code, data, detector, score, alert or policy was changed.*
