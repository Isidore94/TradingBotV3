# Lake assessment 2026-09-04

> **Correction, later 2026-09-04:** Treat the conclusions below as historical,
> not validated current advice. The audit scripts read `basis`, but the bronze
> payload field is `match_basis`. A later bounded read found 77 distinct likes:
> 41 matched and 36 unmatched (84 stored versions), refuting the broken-link claim.
> Missing bands explain fallback use, but no-target expiry can still be profitable;
> TARGETED does not itself mean positive net return. The M5-close study can hold
> for 18 sessions and is not the live day-trade alert population. Neither its MFE
> nor the control win rate proves edge or identifies exits as the cause of losses.
> The probability claim below is invalid and must not be reused. R1–R4 below
> confer no authorization or promotion readiness. See the source-backed
> [process review](PROJECT_PROCESS_REVIEW_2026-09-04.md) before using these results.
> The historical scripts and saved JSON have not been corrected or rerun.

## Verdict card

1. **Integrity PASSES** — 0 duplicate grains in bar_m5; 32/32 outcome buckets recomputed; 141,774 outcome rows in the current view. 15 bar_derived M15 constituent violations in August (0.002%), acceptable.
2. **Swing house recipe 0/257 wins** (n=257 terminal, 0 episodes won, WLB=0.0%) — the control_fixed_1r2r recipe on the SAME occurrences is 64.6% WR short / 37.4% long (n=532, floor met). The structural stop + trailing-band management loses every resolved trade; the occurrence population itself has signal. *Goal 5: swing setups get sharper — the house recipe needs investigation before any headline uses it.*
3. **Every M5-close recipe with n > 500 has negative mean net_r** (24 of 24, range -0.21 to -1.58 R) — confirmed across all hours and families. The M5 entry-next-session design produces trades that move favourably (median MFE 0.87-2.12 R) but the exits cannot capture the MFE. *Goal 4: day trading is the biggest prize — MFE confirms the opportunity exists; exit research (the P8 and after-like grids) is the path.*
4. **HTF LRSI: 16/16 cells pass the n floor, all carry negative mean net_r** (-0.21 to -0.68 R) — no cell shows edge. *Gate #28: the study has data but no promotion case.*
5. **P8 (entry timing) and P10 C (after-like) are collecting** — 506 after-like rows so far; 20-session windows not closed; gates #37 and #43 are refusals. *Cannot be read yet.*
6. **74 like links exist, 0 linked to an occurrence** — all 74 carry `basis=unknown`; the linker is writing rows but not matching. *Goal 2: teach the bot what the trader likes — the link is broken.*
7. **Lately (20 sessions) M5 mean net_r is worse than full sample** (-0.57 vs -0.39 R, n=32,721 vs 57,305). *No family changed rank; the shift is inside the lower bounds.*
8. **706 occurrences have no outcome** — 611 CLOSED (resolved before bars arrived), 69 OPEN (too recent), 26 UNTRADEABLE. 57 registry setups have zero occurrences (all are warehouse-canonical names the detector has never emitted under that ID). Capture is 100% LIVE for bar_m5.

---

## Q1 — Integrity

| Check | Result | Verdict |
|---|---|---|
| bar_m5 2026-08 duplicate grains | 0 of 1,818,148 | PASS |
| bar_m5 2026-09 duplicate grains | 0 of 257,438 | PASS |
| bar_derived M15 2026-08 constituent_count <= expected | 15 violations of 605,909 (0.002%) | PASS (marginal) |
| bar_derived M15 2026-09 constituent_count <= expected | 0 of 82,494 | PASS |
| outcome_path current view | 141,774 rows | expected (up from 137,439 post-recompute; +4,335 from today's build) |
| recompute coverage buckets | 32/32, all carry outcomes_recompute-bNN | PASS |
| AMBIGUOUS_BAR share | 3,474 of 141,774 (2.45%) | acceptable |

Result-state distribution across all recipes: STOPPED 61,573; TARGETED 34,920; EXPIRED 10,185; TRUNCATED 5,223; AMBIGUOUS_BAR 3,474; OPEN 26,399.

The 15 M15 constituent violations in August are bars where the derived bar claims more M5 constituents than expected for its timeframe. At 0.002% this does not compromise any downstream statistic, but the cause should be traced (likely a session boundary edge case in the aggregator).

No repair artefacts found: row counts per recipe are proportional to the occurrence population size, and the AMBIGUOUS_BAR share is stable across recipes.

## Q2 — Swings

### All swing cells, sorted by Wilson lower bound

| Setup | Side | Recipe | n | n episodes | WR | WLB | Mean R | Median R | Mean r_at_s5 | Mean r_at_s10 | Symbols | Sessions | Eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AVWAPE_TO_FIRST_DEV | SHORT | control_fixed_1r2r_v1 | 158 | 158 | 64.6% | 56.8% | +0.10 | +1.00 | — | — | 131 | 20 | YES |
| AVWAPE_TO_FIRST_DEV | LONG | control_fixed_1r2r_v1 | 374 | 374 | 37.4% | 32.7% | -0.58 | -1.00 | — | — | 266 | 21 | YES |
| AVWAPE_TO_FIRST_DEV | LONG | swing_house_v1 | 214 | 214 | 0.0% | 0.0% | -3.24 | -2.33 | — | — | 172 | 21 | YES |
| AVWAPE_TO_FIRST_DEV | SHORT | swing_house_v1 | 43 | 43 | 0.0% | 0.0% | -4.63 | -3.65 | — | — | 42 | 15 | YES |

**Only AVWAPE_TO_FIRST_DEV has swing outcomes** because `PRIMARY_RECIPE_BY_SETUP` maps only two families (AVWAPE_TO_FIRST_DEV and POST_EARNINGS_CANDLE_BREAK) to swing_house_v1, and POST_EARNINGS_CANDLE_BREAK's outcomes are in the intraday recipe. The other 14 families with occurrences (AVWAP_RETEST 927, AVWAP_BREAKOUT 732, AVWAP_BAND_BOUNCE 554, etc.) carry NO swing outcome rows.

**The 0% win rate on swing_house_v1 is the headline finding.** 0 wins out of 257 terminal rows, including at the gross level (0 gross wins). The one TARGETED row (PCG LONG) has net_r = -0.71 and MFE = 0.0 — a TARGETED state with zero favourable excursion is self-contradictory and points to a simulation defect in the `partial_at_band2_trail_band1_run_band3` management path. Meanwhile, the control_fixed_1r2r recipe on the same 532 occurrences has a 45.3% blended win rate, proving the underlying signal exists.

690 swing_house_v1 outcomes remain OPEN (not yet resolved); when those resolve, the 0% could change — but 0/257 is statistically extreme enough (p < 10^-77 under even a 10% base rate) to warrant investigating the simulator before trusting any future outcomes.

**Eligible families clearing the pack's floors** (`_meets_eligibility_floors`: n >= 30, symbols >= 5, sessions >= 5): all four cells above. No other family has swing outcomes.

### Market context splits

Only 2 splits clear both the n and episode floors:

| Setup | Side | D1 bias | n | n episodes | WR | WLB | Mean R |
|---|---|---|---|---|---|---|---|
| AVWAPE_TO_FIRST_DEV | LONG | bullish | 171 | 171 | 0.0% | 0.0% | -3.38 |
| AVWAPE_TO_FIRST_DEV | LONG | bearish | 43 | 43 | 0.0% | 0.0% | -2.68 |

Both carry 0% WR, consistent with the parent. The split adds no information beyond confirming the parent finding persists across market bias.

## Q3 — Day trades

### Confirmation: every M5-close recipe with n > 500 has negative mean net_r

24 of 24 such recipes confirmed negative. The five deepest:

| Recipe | n | Mean net_r | MFE >= 1R | Median MFE | Med time-to-MFE |
|---|---|---|---|---|---|
| m5close_atr0.5_1r_v1 | 2,356 | -1.577 | 53.4% | 1.92 | — |
| m5close_atr0.5_2r_v1 | 2,356 | -1.311 | 53.6% | 2.08 | — |
| m5close_atr0.5_3r_v1 | 2,356 | -1.121 | 53.9% | 2.12 | — |
| m5close_atr1_1r_v1 | 2,356 | -0.588 | 38.2% | 1.22 | — |
| m5close_atr1_2r_v1 | 2,356 | -0.468 | 38.3% | 1.29 | — |

**By hour (ET):** all 57,305 terminal M5-close rows enter in the 09:xx hour (the "next session first completed M5 close" is always the open). There is no hour split because the entry rule is a single moment.

**By family (n >= 30):**

16 families produce M5-close outcomes. The top 5 by median MFE:

| Setup | n | Mean net_r | MFE >= 1R | Median MFE |
|---|---|---|---|---|
| MID_EARNINGS_SECOND_DEV_HOLD | 1,179 | -0.467 | 36.1% | 1.12 |
| POST_EARNINGS_AVWAP_BOUNCE | 3,906 | -0.370 | 34.2% | 0.98 |
| AVWAPE_TO_FIRST_DEV | 13,176 | -0.392 | 33.2% | 0.86 |
| AVWAP_RETEST | 11,547 | -0.352 | 34.9% | 0.90 |
| AVWAP_BAND_BOUNCE | 5,832 | -0.349 | 35.4% | 0.88 |

**What this means for the trader:** the M5-close recipes enter at the open and hold with various stop/target levels. The prices DO move favourably (53% of the tightest-stop recipes see MFE >= 1R), but the trade's own exit policy (fixed stop, fixed target, 18-session time stop) does not capture the move. The MFE is the OPPORTUNITY; the negative net_r is the gap between the opportunity and what the recipe's exit captures. This is exactly the question the P8 entry-timing grid was designed to answer: whether an alternative entry (M15 acceptance, M5 retest, M30 EMA pullback) narrows that gap. Until P8's 20-session window closes, the negative mean net_r is a statement about THIS exit, not about the setups themselves.

## Q4 — HTF LRSI

All 16 cells pass the n >= 30 floor. None shows a positive mean net_r.

| Recipe | TF | Cross | n | n episodes | Mean net_r | WLB | Floor |
|---|---|---|---|---|---|---|---|
| htf_lrsi_h2_down80_2r_v1 | H2 | down80 | 3,274 | 1,667 | -0.211 | 0.346 | PASS |
| htf_lrsi_h1_down50_2r_v1 | H1 | down50 | 3,447 | 1,835 | -0.358 | 0.339 | PASS |
| htf_lrsi_h2_down50_2r_v1 | H2 | down50 | 3,310 | 1,683 | -0.373 | 0.339 | PASS |
| htf_lrsi_h1_down80_2r_v1 | H1 | down80 | 3,377 | 1,799 | -0.303 | 0.335 | PASS |
| htf_lrsi_h2_up50_2r_v1 | H2 | up50 | 3,084 | 1,595 | -0.396 | 0.327 | PASS |
| htf_lrsi_m30_down80_2r_v1 | M30 | down80 | 4,156 | 2,150 | -0.424 | 0.313 | PASS |
| htf_lrsi_h1_up50_2r_v1 | H1 | up50 | 4,187 | 2,149 | -0.453 | 0.307 | PASS |
| htf_lrsi_h2_up20_2r_v1 | H2 | up20 | 3,192 | 1,654 | -0.503 | 0.304 | PASS |
| htf_lrsi_h1_up20_2r_v1 | H1 | up20 | 4,121 | 2,117 | -0.447 | 0.299 | PASS |
| htf_lrsi_m30_down50_2r_v1 | M30 | down50 | 4,196 | 2,149 | -0.447 | 0.297 | PASS |
| htf_lrsi_m30_up50_2r_v1 | M30 | up50 | 4,508 | 2,273 | -0.507 | 0.283 | PASS |
| htf_lrsi_m30_up20_2r_v1 | M30 | up20 | 4,522 | 2,285 | -0.495 | 0.279 | PASS |
| htf_lrsi_h4_up50_2r_v1 | H4 | up50 | 459 | 233 | -0.413 | 0.221 | PASS |
| htf_lrsi_h4_up20_2r_v1 | H4 | up20 | 463 | 231 | -0.491 | 0.195 | PASS |
| htf_lrsi_h4_down50_2r_v1 | H4 | down50 | 706 | 360 | -0.592 | 0.188 | PASS |
| htf_lrsi_h4_down80_2r_v1 | H4 | down80 | 712 | 347 | -0.676 | 0.148 | PASS |

The best cell (H2 down-80, mean -0.21 R) is still well negative. The LRSI cross by itself does not produce a tradeable edge on any timeframe. The study has data; it has no promotion case.

## Q5 — Declared grids (collection status only)

| Trial ID | Status | Cells | Rows collected | Window closes |
|---|---|---|---|---|
| setup_entry_timing_avwape_first_dev_long_v1 (P8) | collecting | 4 entries x 3 targets = 12 | not separately countable (mixed into M5-close) | ~20 sessions after 2026-09-02 ≈ 2026-09-30 |
| after_like_entry_grid_v1 (P10 C) | collecting | 5 offsets x 4 entries = 20 | 506 outcome rows | ~20 sessions after 2026-09-02 ≈ 2026-09-30 |

Gates #37 and #43 are refusals: no cell may be read for a verdict before the window closes. The rows are accruing normally.

## Q6 — Likes

| Metric | Value |
|---|---|
| Total bronze like_occurrence_link rows | 74 |
| basis = unknown | 74 (100%) |
| basis = exact_family | 0 |
| basis = any_family | 0 |
| basis = none | 0 |

All 74 like-link rows carry `basis=unknown`. This means the linker wrote the rows but did NOT populate the `basis` field, so it is impossible to distinguish linked from unlinked likes. The linker is either writing placeholder rows or the basis field was not implemented in the payload writer.

**Impact on goal 2 ("teach the bot what the trader likes"):** the like → occurrence join is the foundation of the after-like grid (P10 C) and the "what the liked names had in common" feature (goal 3). With 0% of links carrying a valid basis, the downstream grids are grading against unresolved matches. This should be investigated before the after-like window closes.

## Q7 — What is working lately

Lately window: 2026-08-10 to 2026-09-04 (20 trading sessions).

### Swing (swing_house_v1, eligible families only)

| Setup | Side | n all | WR all | WLB all | n lately | WR lately | WLB lately |
|---|---|---|---|---|---|---|---|
| AVWAPE_TO_FIRST_DEV | LONG | 214 | 0.0% | 0.0% | 140 | 0.0% | 0.0% |
| AVWAPE_TO_FIRST_DEV | SHORT | 43 | 0.0% | 0.0% | 33 | 0.0% | 0.0% |

No rank change — 0% is 0% in both windows. The finding is invariant to the window.

### Day trades (all M5-close recipes pooled)

| Window | n | Mean net_r | MFE >= 1R share |
|---|---|---|---|
| Full sample | 57,305 | -0.386 | — |
| Lately (20 sessions) | 32,721 | -0.565 | — |

The lately window's mean net_r is worse (-0.57 vs -0.39), but with n=32,721 the difference (-0.18 R) is inside the lower bound of the full sample's distribution. No family changed rank between the two windows. The shift is not actionable.

## Q8 — Coverage and blind spots

### Occurrences with no outcome row

| Reason | Count |
|---|---|
| CLOSED (resolved before M5 bars arrived) | 611 |
| OPEN (too recent to resolve) | 69 |
| UNTRADEABLE | 26 |
| **Total** | **706 of 6,897 (10.2%)** |

The 611 CLOSED occurrences are setups that triggered and resolved before the warehouse could capture their M5 bars. This is structural — the warehouse captures bars from the live tee, and a setup that closed before the tee saw it has no path to simulate. It is not a bug.

### Registry setups with zero occurrences

57 of the registry's entries have zero occurrences. These are all warehouse-canonical IDs (lowercase with underscores) that the detector never emits under that exact ID — the detector uses UPPER_CASE family tags, and the registry's crosswalk from uppercase to lowercase has not been wired into the occurrence pipeline. Examples: `avwap_band_bounce` (vs `AVWAP_BAND_BOUNCE`, 554 occurrences), `avwap_breakout` (vs `AVWAP_BREAKOUT`, 732).

### Symbol concentration

The eligible swing cells are well-distributed: 172 symbols in the LONG cell, 131 in the SHORT cell. The top symbol (NMRK) appears in only 4 of 214 rows (1.9%). No concentration risk.

### Capture mode

bar_m5 is 100% LIVE capture in September. The `capture_mode` field is not on the occurrence schema, so a BACKFILL vs LIVE split of occurrences cannot be reported from the current schema. The entire outcome population was built from LIVE-captured bars.

---

## Recommendations

Each is mapped to a numbered goal from decision 0016, with evidence and the gate it would need.

### R1. Investigate the swing_house_v1 simulator (goal 5, goal 8)

**Evidence:** 0 wins in 257 terminal rows, including 0 gross wins. One TARGETED row has MFE = 0.0, which is self-contradictory. The control_fixed_1r2r recipe on the same occurrences has 45.3% WR.

**What would overturn it:** a code review of `_simulate_swing_house_v1` showing the management path correctly implements partials at band2 and trailing at band1, and a manual trace of 10 sampled rows confirming the gross_r calculation.

**Gate:** none needed — this is a defect investigation, not a promotion. If confirmed, the fix needs plan.md sec 5's golden-result fixture before landing, and the outcomes would need another forced recompute (BD-98).

### R2. Fix the like_occurrence_link basis field (goal 2, goal 3)

**Evidence:** 74 rows, 100% `basis=unknown`. The after-like grid (P10 C, 506 rows collecting) grades against these links.

**What would overturn it:** the `unknown` value being a legitimate third match quality (neither exact nor none), documented in `like_links.py`. Currently the code defines `BASIS_EXACT_FAMILY`, `BASIS_ANY_FAMILY` and `BASIS_NONE`; `unknown` is not one of them.

**Gate:** plan.md P10 C owns the like grid; fixing the linker should happen before the 20-session window closes (~2026-09-30) or the collected evidence grades against unresolved matches.

### R3. Extend swing outcomes to all 14 detector families (goal 5, goal 1)

**Evidence:** 14 families with occurrences (AVWAP_RETEST 927, AVWAP_BREAKOUT 732, etc.) carry zero swing outcome rows because `PRIMARY_RECIPE_BY_SETUP` maps only 2 families.

**What would overturn it:** a plan.md decision that the other families should not be graded under swing_house_v1 (e.g. because they have no structural stop level). That decision does not exist today.

**Gate:** plan.md Phase 4.1 (the identity-graph freeze) is where the registry becomes authoritative and could assign primary recipes. Until then, adding entries to `PRIMARY_RECIPE_BY_SETUP` is a safe, additive change that needs only the sec 5 golden-fixture guard.

### R4. Read M5-close MFE by family to guide the priority switch (goal 4, goal 1)

**Evidence:** MFE >= 1R ranges from 33% to 36% across the top families, and median MFE is highest for MID_EARNINGS_SECOND_DEV_HOLD (1.12 R). MFE is the day-trade headline by decision 0016 answer 4.

**What would overturn it:** the P8 entry-timing grid showing that an alternative entry captures more of the MFE, making the current per-family ranking obsolete.

**Gate:** V4 owns the priority switch. This evidence is ready for V4 to consume once the swing_house_v1 simulator is trusted.

---

## Cannot be answered yet

| Question | What it needs | When |
|---|---|---|
| Which entry variant captures the most M5 MFE? | P8 window closes | ~2026-09-30 (20 sessions from 2026-09-02) |
| Does a like predict a better entry 3-5 days later? | P10 C window closes + like basis field fixed | ~2026-09-30 |
| Do the other 14 families produce swing edge? | R3 (extend PRIMARY_RECIPE_BY_SETUP) | awaiting authorization |
| Is the 0% swing WR real or a simulator defect? | R1 investigation | next session |
| Lately ranking across families | V4 priority switch, which requires trusted swing and M5 outcomes | after R1 |

---

## Commands run

All from the repo root, `.venv\Scripts\python.exe`:

```
docs\analysis\scripts\lake_assessment.py    # Q1-Q8 main analysis
docs\analysis\scripts\lake_deep_dive.py     # swing detail, like links, capture mode
docs\analysis\scripts\lake_likes_and_details.py  # bronze link check, bar_derived completeness
```

Output: `docs/analysis/lake_assessment_output.json` (full JSON of all tables).
