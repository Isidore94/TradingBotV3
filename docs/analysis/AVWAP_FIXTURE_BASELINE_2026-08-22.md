# AVWAP golden-fixture baseline — R10.V step 1

**plan.md Phase 0.7 / R10.V step 1.** Frozen 2026-08-22 night on branch
`phase05-integration-blitz`, before any part of the daily-bar unit repair.

Step 1 has two jobs: prove that no golden fixture reads the live parquet store,
and record what every fixture is *now* so that step 5's re-freeze is a scoped,
explainable change rather than a diff nobody can bound. This file is that record.

---

## 1. No golden fixture reads the live store — measured, not inspected

Inspection is not proof, so the readers themselves were wrapped. A pytest plugin
replaced `builtins.open`, `Path.open`, `Path.read_bytes` and `pandas.read_parquet`
with versions that record any access resolving inside

- `C:\TradingBotData\data\daily_bars` (1,958 files), and
- `C:\TradingBotData\data\intraday_bars`,

and the **entire suite** was run under it.

| Run | Result |
|---|---|
| `pytest tests/ -q -p liveguard` | **4205 passed / 19 subtests**, exit 0 |
| live-store accesses recorded | **0** |

The stop condition attached to this step — *"stop if any golden fixture turns out
to read the live parquet"* — therefore does **not** fire, and the packet proceeds
as specified. Every fixture carries its own bars.

*Scope of the claim:* this proves no test **read** those two directories during
that run. It does not prove a test cannot reach them under a different
environment (the suite redirects `LOCALAPPDATA` and the home folder through
`conftest`, so a path that resolves elsewhere would not be caught by a guard
watching these two roots). It is evidence about the suite as it is configured and
run, which is what step 1 asks for.

## 2. Fixture inventory as of this commit

SHA-256 (first 16) over the file bytes. `git` is the actual freeze; this table
exists so a later reader can tell at a glance which fixtures were expected to
move and which were not.

| fixture | sha256 | bytes | AVWAP role |
|---|---|---|---|
| `aggressive_watchlist_candidates_v1.json` | `bf9f0c4c015480e2` | 4,977 | none |
| `auto_pick_focus_gate_v1.json` | `31b62611eb2a2996` | 5,421 | none |
| `bounce_entry_quality_v1.json` | `422b61d0e3d6229f` | 13,718 | none |
| `d1_zone_arms_golden_v1.json` | `a305c95b9d94d31d` | 25,844 | **levels as input** |
| `greatness_candidate_from_d1_v1.json` | `2a8ab34e801ae2e4` | 14,180 | **levels as input** |
| `journal_rebuild_trades_v1.json` | `94d069acfc920e62` | 35,548 | none |
| `laguerre_rsi_v1.json` | `1143d497e6cea524` | 2,678 | none |
| `m5_strength_functions_v1.json` | `d3b6cee2f3bab60b` | 39,241 | none |
| `mixed_unit_avwap_v1.json` | `69175d7bd1eb3157` | 13,969 | **computes** (new, §4) |
| `r3_swing_quality_v1.json` | `e20dbc1c6fa4f2aa` | 6,792 | **levels as input** |
| `regime_pause_sweep_v1.json` | `ab5c17c166417864` | 114,494 | session VWAP, M5 |
| `sector_cohort_v1.json` | `68f0f2e06f66acef` | 88,388 | none |
| `technical_integrity_replay_v1.json` | `54a856e0a00809c8` | 18,637 | none |
| `technical_integrity_scoring_v1.json` | `760e868221010ed7` | 7,823 | none |
| `warehouse_avwap_bands_v1.json` | `9a0897b1c185f74a` | 5,512 | **computes** |

Only two fixtures put bars through `calc_anchored_vwap_bands`:
`warehouse_avwap_bands_v1` (bars stored as positional arrays including volume)
and the new `mixed_unit_avwap_v1`. Three more carry **already-computed** AVWAP
levels as inputs and never recompute them. `regime_pause_sweep_v1` carries M5
bars and session VWAP — a different seam (`chart_snapshot.session_vwap_series`),
untouched by this packet.

## 3. What this predicts about step 5

Step 5 says "re-freeze every AVWAP-derived golden fixture that changed". Given
§1 and §2, the prediction is worth stating in advance so that a surprise is
visible as a surprise:

- **The backfill (step 4) cannot move any fixture.** Fixtures feed fixed bars;
  rewriting the parquet store changes what the *desk* computes, not what the
  suite computes. If a fixture moves at step 4, something reads the live store
  and §1's proof has expired.
- **Only steps 2–3 can move one**, and only through code: the schema columns and
  the `_normalize_daily_bar_frame` collision preference. Those touch the frame
  the fixtures do not supply (`source`, `volume_unit`) and the dedup rule.
  `warehouse_avwap_bands_v1` exercises the band formula, which this packet does
  not change, so the expectation is that **it does not move either**.
- **The σ formula is not to be swapped** (plan.md §5). `mixed_unit_avwap_v1`
  now guards it directly: an independent reimplementation of the running-deviation
  variant must agree, and the distribution-stdev variant must *disagree* on this
  fixture — otherwise the guard cannot discriminate and the fixture is rebuilt
  with more trend.

If step 5 turns out to be a no-op, that is a result, and it will be recorded as
one rather than quietly skipped.

## 4. The new fixture: `mixed_unit_avwap_v1`

Twenty hand-constructed daily bars for a synthetic symbol (`MIXQ`), anchored at
bar 0, spliced at bar 12. Three series with **identical prices**, differing only
in the volume column.

| series | volume | vwap | σ | UPPER_2 |
|---|---|---|---|---|
| `shares` | Yahoo consolidated shares throughout | 42.263138 | 1.259667 | 44.782472 |
| `mixed` | bars 12+ divided by 100 (IB round lots spliced on) | **41.301207** | **0.607158** | **42.515523** |
| `lots` | every bar divided by 100 (uniform rescale — the control) | 42.263138 | 1.259667 | 44.782472 |

- **The splice costs −0.9619 (−2.28%) on VWAP and −2.2669 on UPPER_2**, and
  collapses σ to **0.482×** its true value. On a $45 name that is a 2.27-point
  error in a band the tracker replays targets against — the same shape as the
  live measurement (vwap move median 1.03%, p90 5.99%, max 138.8%).
- **The uniform rescale costs nothing**: `lots` reproduces `shares` to 0.0 on
  vwap and 1.3e-15 on σ, inside the declared 1e-9 tolerance. AVWAP is a
  volume-weighted *ratio*, so a constant factor cancels. **This is the whole
  argument for C-prime in one row of a table**: if the store were uniformly
  mis-scaled there would be nothing to repair, and a ×100 conversion on the IB
  path would therefore not have fixed anything — it would have replaced a
  visible error with an invisible one.

The fixture **pins the wrong answer on purpose**. Its `mixed` numbers are what
the live store produces today; they do not move when the store is repaired,
because the fixture feeds fixed frames. That is what makes it the control for
step 5: any AVWAP-derived fixture that *does* move must be explained against it.

Rebuild with `scripts/build_mixed_unit_avwap_fixture.py` (refuses to overwrite
without `--force`); the consuming test is `tests/test_mixed_unit_avwap_golden.py`
(10 tests). Both guards were proven to discriminate before this was committed: a
0.0001 drift in one expectation fails the comparison, and editing an input bar
without re-freezing the hash fails the Milestone 3 contract loader with a
`raw input hash mismatch`.
