# Master AVWAP — "Depth & Complexity" Integration Plan

> Companion to `thingstointegrate.txt`. This is the macro plan + step-by-step
> guide for integrating the six feature requests. Written so it can be picked up
> across multiple sessions (e.g. if usage limits reset). **Read the "Guiding
> principles" and "Architecture map" first, then jump to whichever Phase you're
> working on.** Each phase has a checklist + concrete code hooks.
>
> Status legend: ☐ not started · ◐ in progress · ☑ done
>
> Last updated: 2026-06-21
>
> **Working this plan with a non-Claude agent (Codex/Cursor/etc.) or a human?**
> Read §0–§1 for intent, then **§8 (verified anchors — trust these over §1 line
> numbers), §9 (glossary), and §10 (working protocol).** Then pick a single ticket
> from §11 and execute it end-to-end. Each ticket is self-contained.

---

## 0. Guiding principles (Aaron's intent, in his words)

1. **Don't force everything into scoring yet.** The repeated theme in
   `thingstointegrate.txt`: *"we dont need to directly incorporate these into the
   scoring yet but we need a place to start looking at these additions and see how
   valuable they might be."* So the default path for every new signal is
   **compute → store → study in the setup tracker → only then (maybe) score.**
2. **Relative strength is the north star.** Aaron's personal edge: *"ensuring a
   stock is at least as strong as its industry (or as weak for shorts)."* Stocks
   get "baited" by being carried by a strong industry. He wants the strongest
   stocks, or stocks that were very strong 1–2 weeks ago then pulled back.
3. **The setup tracker is ground truth.** It already drives Expected-R ranking.
   New ideas should be added to it as *study setups* (like the existing control /
   discovery namespace) before they touch the live score. See
   `memory/tracker-methodology-fixes.md` and `memory/expected-r-ranking-system.md`.
4. **Persist what stays relevant forever.** Horizontal high-volume levels and
   Ichimoku cloud lines "stay relevant forever" → store them per-symbol in the
   Google Drive / cloud folder so they accumulate over time.
5. **Two run modes, one core.** Everything flows through
   `master_avwap_lib/runner.py::run_master` (work = `master_avwap_mini_pc.py`,
   home = `gui.py`). Intraday is `bounce_bot.py`. Keep new logic in the shared
   core so both surfaces benefit. See `memory/user-trading-workflow.md`.

---

## 1. Architecture map (where everything lives)

| Concern | Location |
| --- | --- |
| Shared scan core | `scripts/master_avwap_lib/runner.py` → `run_master()` |
| Monolith (most logic, being split out) | `scripts/master_avwap_lib/legacy.py` (~1.1 MB) |
| Expected-R ranking (pure, new module) | `scripts/master_avwap_lib/expected_r.py` |
| Config constants (exposed from legacy) | `scripts/master_avwap_lib/config.py` |
| Mini-PC entry (work) | `scripts/master_avwap_mini_pc.py` |
| GUI entry (home) | `scripts/gui.py` → `gui_app/app.py` → `master_panel.py` |
| Intraday bot | `scripts/bounce_bot.py`, `scripts/bounce_bot_lib/` |
| **Reusable relative-strength + ETF maps** | Import from `scripts/bounce_bot_lib/rrs.py` (`real_relative_strength`, `load_sector_etf_map`, `load_and_update_industry_etf_map`, `resolve_sector_etf`, `resolve_industry_ref_etf`). ⚠️ `rrs.py` is only a **re-export shim** (`expose_legacy_names`); the real implementations live in `scripts/bounce_bot_lib/legacy.py` (see §8). Import from `rrs.py`, but **edit in `bounce_bot_lib/legacy.py`**. |
| ETF map JSON stores (already exist) | `scripts/project_paths.py:79-80` → `SECTOR_ETF_MAP_FILE`, `INDUSTRY_ETF_MAP_FILE` under `DATA_DIR`, with repo→data seed/sync at `project_paths.py:347-348` & `432-433`. Phase 3 reuses these — don't reinvent. |
| Paths / cloud folder | `scripts/project_paths.py` |
| Tests (333+, keep green) | `tests/` |

### Key data structures already available inside `run_master`
- `symbols` — union of long + short watchlist symbols (the scan universe).
- `daily_frames_by_symbol[sym]` — **every scanned symbol's daily OHLCV DataFrame**
  (`scripts/master_avwap_lib/runner.py:393`). This is the single most useful hook:
  universe-wide strength (Feat 1), D1 horizontal levels (Feat 5), cloud lines
  (Feat 6) all key off this.
- `priority_rows` — only the symbols that produced a *flagged setup*. Today's
  market-prep strongest/weakest decile reads from here, which is the bug (Feat 1).
- **D1 RS-vs-SPY math** lives in `assess_daily_relative_strength(daily_rows,
  last_trade_date, side, spy_benchmark, lookback_days)` at `legacy.py:22666` (math
  at `legacy.py:22713`: `rs = 0.35*(1d excess vs SPY) + 0.65*(5d excess vs SPY)`).
  Note: it consumes **`daily_rows`** = a list of `{"date","close",...}` dicts, *not*
  a DataFrame — so B1 must adapt each `daily_frames_by_symbol[sym]` DataFrame into
  that row shape (or factor out a DataFrame-native helper).
- **`spy_benchmark`** is *not* a `run_master` local variable. It is the SPY dict
  pulled from the market-regime snapshot:
  `market_regime_snapshot["benchmarks"]["SPY"]` (keys `one_day_return_pct`,
  `five_day_return_pct`), assembled around `runner.py:1266`. Thread that dict into
  any new universe-strength helper.

### The "study-first" pattern to copy
The control/discovery system (`memory/tracker-methodology-fixes.md`, item #2) is
the template for everything study-first: a **separate namespace** in the tracker
(`tracker["control_setups"]`) that the live ranking can never see, plus a
dedicated report file. New study signals (Feat 4/5/6) should follow this shape:
record observations to a side namespace + write a `*_study.txt` report; do **not**
touch the live score until the data says it's worth it.

---

## 2. Cross-cutting building blocks (build these once, reuse everywhere)

These are prerequisites shared by multiple features. Build them first.

### B1. Universe strength table (feeds Feat 1, 3)
A function that, given `daily_frames_by_symbol` + `spy_benchmark`, returns a row
per scanned symbol with: `symbol`, `rs_d1`, `1d_ret`, `5d_ret`, `13w_ret`,
`26w_ret`, `atr20`, plus (once Feat 3 lands) `industry_etf`, `rs_vs_industry`.
This decouples "strength ranking" from "did it flag a setup".

### B2. Multi-timeframe bar fetch (feeds Feat 4)
Master AVWAP only fetches `1 day` bars today
(`legacy.py:_fetch_live_daily_bars`). Add an analogous `fetch_intraday_bars(ib,
sym, bar_size, duration)` for `1 hour` (and 4h via resample of 1h, since IBKR has
no native 4h). Mirror the IBKR-first / Yahoo-fallback pattern. Cache per run.

### B3. Per-symbol level store in the cloud folder (feeds Feat 5, 6)
A JSON store under the Drive/cloud structure (via `project_paths.py`), one record
per symbol, holding: high-volume horizontal levels (with first-seen date, relvol
bucket, touch count) and flat-cloud (Leading Span B) lines. Levels persist and
accumulate across runs; tolerance is ATR-scaled. Schema-versioned like the
existing tracker files.

### B4. Study namespace in the setup tracker (feeds Feat 4, 5, 6)  ☑ DONE (2026-06-21, see T6)
A `tracker["study_setups"]` namespace (mirroring `control_setups`) + a
`master_avwap_study.txt` report, so new setup ideas (1h/4h trend, level
respect/break, compression break, trendline break) get measured for hit-rate and
realized R **before** they're allowed to affect scoring.

---

## 3. Phased roadmap (the 6 items → phases)

Ordered by (a) value to Aaron, (b) dependency, (c) effort. Each phase is
independently shippable + testable.

| Phase | Feature(s) from txt | Why this order |
| --- | --- | --- |
| **P1** ✅ | #1 Strongest/weakest 10% for market prep | **DONE 2026-06-21.** Smallest, already half-built, fixes a "not working" complaint. Builds B1. |
| **P2** ✅ | #2 TOP becomes a *secondary* setup | **DONE 2026-06-21.** Small, removes a known regression (TOP suppressing other patterns). |
| **P3** ☑ | #3 Industry/sector relative strength | **DONE 2026-06-22.** Daily scan + market prep + flag-gated D1 boost + BounceBot M5 directional industry RRS. |
| **P4** ☑ | #4 1h/4h trend detection (boost only) | **DONE 2026-06-22.** B2 intraday fetch + 1h/4h SMA retest detection + flag-gated boost + study rows. |
| **P5** ☑ | #5 D1 high-volume horizontal levels | **DONE 2026-06-22.** B3 level store + study-first HV level events; scoring penalty still later. |
| **P6** | #6 Cloud lines, compression breaks, trendline breaks | Needs B3 + B4. Most research-y; pure study-first. |

---

## 4. Per-phase step-by-step guides

### Phase 1 — Strongest / weakest 10% for market prep  ☑ DONE (2026-06-21, see T1)
**Problem:** `_market_prep_strength_decile_rows` ranks only `priority_rows`
(flagged setups), so market-prep "strongest/weakest" is empty/misleading. Aaron
wants the strongest/weakest *of the whole scanned universe* for prep.

Steps:
1. ☑ **Built B1** `build_universe_strength_rows(daily_frames_by_symbol,
   spy_benchmark, *, sides_by_symbol=None)` in `legacy.py` (right after
   `assess_daily_relative_strength`). One row per scanned symbol, reusing the
   `0.35/0.65` RS-vs-SPY math (the RS score is side-independent), plus 13w/26w
   trailing returns + atr20. Added helpers `_daily_frame_to_rows` (DataFrame→row
   dicts the dict-based helpers expect) and `_trailing_return_pct`.
2. ☑ Threaded `daily_frames_by_symbol` + `spy_benchmark` + `sides_by_symbol`
   from `run_master` into `build_market_prep_payload` (new kwargs; old
   `priority_rows` kept as a fallback so existing callers/tests still work).
3. ☑ `build_market_prep_payload` now builds universe rows and feeds them to
   `_market_prep_strength_decile_rows` (via `strength_source_rows`), falling back
   to `priority_rows` only when no frames are supplied. Decile fraction unchanged.
4. ☑ Added the "recently strong, now pulling back" sub-list
   (`_market_prep_pullback_rows`: 26w/13w > 0 AND 5d < 0) + a new
   `recently_strong_now_pulling_back` market-prep section.
5. ☑ Notes now report the universe count ("N of M symbols with RS data"). The
   report writer + GUI/mini-PC render sections generically from
   `MARKET_PREP_SECTION_DEFINITIONS`, so the new section + counts flow through with
   no surface-specific change.
6. ☑ **Tests:** 5 new cases in `tests/test_master_avwap_setups.py` (universe
   decile ignores flagged rows, tie-break by symbol, insufficient-history skip,
   no-frames fallback, pullback list). Full suite 338 green. (mini-PC status test
   not needed — rendering is generic.)

Acceptance: with a watchlist of e.g. 80 names, market prep lists ~8 strongest /
~8 weakest by D1-vs-SPY across **all 80**, not just the flagged ones.

### Phase 2 — TOP pattern as a *secondary* setup  ☑ DONE (2026-06-21, see T2)
**Problem (txt #2):** "TOP pattern shouldnt necessarily be a pattern in itself if
it overwrites the ability for other patterns to be studied. it should be
considered a secondary setup."

Steps:
1. ☑ Found it: `_derive_setup_family` (`legacy.py:14656`) checked TOP *above*
   sma_breakout/band_bounce/extreme_move/retest, collapsing those to `top_pattern`.
2. ☑ Demoted: moved the TOP block below all specific entry patterns (above the
   generic `avwap_breakout`/`favorite_zone`/`general` fallbacks). Added
   `top_secondary` to the priority row. The TOP score bonus was already applied
   independently of the family (`legacy.py` ~score build), so the boost survives.
   TOP watchlist preserved (`_priority_should_track_top_pattern` /
   `_priority_is_top_strength_watchlist` now gate on the flags, not the family).
3. ☑ A TOP + 50SMA-retest setup now records as the SMA-breakout family (tracker
   studies the real pattern) with `top_secondary=True` context.
4. ☑ **Tests:** 5 new in `tests/test_master_avwap_setups.py`. Full suite 352 green.

### Phase 3 — Industry & sector relative strength  ☑ DONE (2026-06-22, see T3)
**Goal (txt #3, #6 of his prose):** find stocks **stronger than their industry**
(D1 + M5), surface **strongest industries**, and boost swing setups where a
leader retests (e.g. 15EMA) while its industry turns up.

Steps:
1. ☐ **Inventory existing infra** in `bounce_bot_lib` (`rrs.py`):
   `load_sector_etf_map`, `load_and_update_industry_etf_map`,
   `resolve_industry_ref_etf`, `real_relative_strength`. Decide: import/share vs
   lift into a shared module both bots use (prefer a small shared module to avoid
   a master→bounce dependency).
2. ☐ **Curate the industry ETF list** (liquid only). Seed from the txt:
   TAN=solar, SMH/SOXX=semis, DRAM=memory, HACK=cyber (CRWD/PANW), plus sectors
   XLF/XLK/XLE/XLV/XLI/XLY/XLP/XLU/XLB/XLRE/XLC. Expand to cover major industries
   (biotech XBI, banks KRE, retail XRT, oil services OIH, homebuilders XHB,
   transports IYT, gold GDX, China KWEB, software IGV, internet FDN, etc.). Store
   in a JSON map in the cloud folder so it's editable without code changes.
3. ☐ **Map each scanned symbol → its industry ETF.** Reuse
   `resolve_industry_ref_etf`; fall back to sector ETF; cache the mapping.
4. ☐ **Compute `rs_vs_industry`** on D1: symbol return − industry-ETF return over
   1d/5d/13w (same excess-return shape as RS-vs-SPY). Add to B1 rows.
5. ☐ **Compute industry strength ranking** (each industry ETF vs SPY) → a
   "strongest industries" market-prep section.
6. ☐ **Scoring boost (bounded):** when a swing setup fires AND the stock leads its
   industry AND the industry itself is strengthening on D1, add a capped boost.
   Gate it behind a config flag so it can be tuned/disabled. Mirror the existing
   `daily_relative_strength_bonus` mechanism.
7. ☐ **Intraday (bounce_bot):** wire the same stock-vs-industry check on M5 so
   intraday entries prefer names leading their group (uses bounce_bot's existing
   `real_relative_strength`).
8. ☐ **Tests:** ETF map resolution, `rs_vs_industry` math, industry ranking,
   boost is bounded + flag-gated.

> **Shipped.** Seeded the editable industry ETF map in `bounce_bot_lib/legacy.py`
> (defaults merge into the existing `DATA_DIR/industry_etf_map.json` without
> overwriting user edits) and exposed cache/map helpers through `rrs.py`. Master
> AVWAP now reads the shared symbol classification cache, resolves each scanned
> symbol to industry ETF with sector fallback, fetches unique mapped ETF D1 bars,
> adds `industry_etf` + `rs_vs_industry` + ETF return fields to universe rows, and
> writes a new `strongest_industries` market-prep section. Priority rows, AI state,
> D1 feature CSV/history, tracker attributes, and focus payloads all carry the new
> industry RS context. The D1 score boost is bounded (`+10`) and gated by
> `feature_flags.industry_relative_strength_scoring_enabled` (default off). BounceBot
> M5 candidate scoring now gives a bounded directional industry-RRS bonus instead
> of a flat "any industry data" bump. Added 7 focused tests; full suite 358 green
> (`python -m unittest discover tests`).

Note on the "super strong industry" nuance: when an industry is *very* strong,
a slightly weaker member is OK (rotation candidate) — encode this as: industry
RS very high → relax the stock-vs-industry penalty; otherwise require the stock
to lead. Keep this as a tunable, study it in the tracker before trusting it.

### Phase 4 — 1h / 4h trend detection  ☑ DONE (2026-06-22; study-first + small boost)
**Goal (txt #4):** detect 1h/4h pullback→retest of 20/50/100/200 SMA that forms
up/down trends; boost stocks in a 1h **and** 4h uptrend. Reuse existing
uptrend/downtrend detection.

Steps:
1. ☑ **Build B2** intraday fetch (`1 hour`, duration enough for 200-SMA on 4h —
   ~40 trading days of 1h bars; resample 1h→4h). IBKR-first, Yahoo-fallback.
2. ☑ Locate the existing trend/uptrend detector (grep `uptrend`/`downtrend` in
   `legacy.py`) and apply it to the 1h and 4h frames.
3. ☑ Detect the specific setup: price pulled back to one of 20/50/100/200 SMA,
   then resumed trend (retest-and-go). Tag `htf_trend_1h`, `htf_trend_4h`,
   `htf_retest_sma`.
4. ☑ **Boost (small, bounded, flag-gated):** stock in 1h **and** 4h uptrend
   (long) gets a modest boost; opposite for shorts.
5. ☑ **Study in parallel:** record these as `study_setups` (B4) to measure
   whether the retest-and-go actually precedes follow-through, before increasing
   the boost.

> **Shipped.** Added normalized intraday fetch (`fetch_intraday_bars`) with
> IBKR-first/Yahoo fallback, deterministic 1h→4h resampling, 1h/4h SMA-stack trend
> labels, recent SMA retest detection, and a bounded `+6` boost gated by
> `feature_flags.htf_trend_scoring_enabled` (default off). Priority rows, AI state,
> D1 feature CSV/history, tracker attributes, focus payloads, and the priority
> report carry `htf_*` context. Confirmed retest-and-go rows are recorded into the
> isolated `study_setups` namespace via `study_rows`. Added 2 focused HTF tests +
> scoring-config coverage; full suite 360 green (`python -m unittest discover tests`).
6. ☐ **Tests:** synthetic 1h frames → correct trend label + retest detection;
   boost bounded + gated.

### Phase 5 — D1 high-volume horizontal levels  ☑ DONE (2026-06-22, study-first)
**Goal (txt #5):** from D1 OHLC + volume, derive horizontal S/R from
high-relative-volume candles (relvol ≥2 & <3 = "red", ≥3 = "green" per Aaron's
PineScript). Persist per-symbol. Use as S/R: **don't enter right at a big level.**
Tolerance ≈ `0.05 × ATR20` (tunable).

Steps:
1. ☑ **Build B3** per-symbol level store (cloud JSON, schema-versioned).
2. ☑ Port the PineScript: `relvol = volume / SMA(volume,50)`. For each candle
   with relvol in [2,3) record its **high** (and symmetrically its low) as a
   level (red); relvol ≥3 → stronger level (green). Keep first-seen date + relvol
   bucket + running touch/respect count.
3. ☑ **Tolerance / clustering:** merge levels within `±tol` where
   `tol = LEVEL_TOL_ATR_FRACTION × ATR20` (start 0.05). Cluster nearby levels so
   the list stays clean; strengthen a level each time price revisits within tol.
4. ☑ **Flag non-earnings high-vol days** separately (Aaron: anchor AVWAPs on
   them, study relevance). Reuse the earnings calendar already in the scan to
   exclude earnings days.
5. ☑ **Study-first:** record "approaching/at a green level" and "level break"
   events to the `study_setups` namespace + `master_avwap_study.txt`. Measure:
   how often levels hold, how often breaks lead to sharp moves.
6. ☐ **Scoring (later, after study validates):** penalize a long entry sitting
   just under a strong green level (poor R:R) / short under support; small,
   bounded, flag-gated.
7. ☑ **Tests:** relvol bucketing, level extraction from a synthetic frame,
   tolerance clustering, touch-count increment, earnings-day exclusion.

> **Shipped.** Added `scripts/master_avwap_lib/levels.py` with inclusive
> relvol-SMA bucketing, high+low candidate extraction, ATR-scaled clustering,
> schema-versioned JSON stores under `DATA_DIR/levels`, idempotent touch/respect/
> break recomputation, adjacent-session earnings-origin marking, and
> non-earnings green HV anchor flags. Master AVWAP now enriches priority rows,
> AI state, D1 feature CSV/history, tracker attributes, focus payloads, and the
> priority report with `hv_level_*` context, then records proximity/break rows
> into isolated `study_setups` without changing live score. Added focused
> `tests/test_levels.py` coverage plus scanner integration coverage; full suite
> 366 green (`python -m unittest discover tests`).

### Phase 6 — Cloud lines, compression & trendline breaks  ☐ (pure study-first)
**Goal (txt #6):** Ichimoku **Leading Span B** flat-cloud lines as S/R
(persisted, fairly exact); define **compression** as a function of AVWAPE ± stdev
bands (or a prior AVWAPE's bands) and detect **compression breaks**; begin
testing **trendline breaks**. Aaron: *"start adding real depth and complexity …
we dont need to directly incorporate these into the scoring yet."*

Steps:
1. ☐ **Leading Span B lines:** port the PineScript — `leadLine2 =
   avg(highest(52), lowest(52))`, displaced 26; detect "flat" via
   `SMA(leadLine2,8) == leadLine2` (within tol). Store flat segments as horizontal
   levels in the B3 store (these are "rather exact" → tight tol). Avoid entries
   right at a flat cloud line.
2. ☐ **Compression metric:** define compression as band width
   (`(UPPER_k − LOWER_k)/AVWAPE`) contracting over N sessions, relative to the
   anchor's own history (and/or vs the previous anchor's bands). Emit a
   `compression_score` + a `compression_break` event when price expands out of a
   contracted state.
3. ☐ **Trendline breaks:** fit recent swing-high / swing-low trendlines (pivot
   detection) and emit break events. Keep simple + deterministic first.
4. ☐ **Study-only:** everything in this phase goes to `study_setups` + the study
   report. No scoring impact until the tracker shows edge. This is the sandbox
   Aaron explicitly asked for.
5. ☐ **Tests:** flat-cloud detection, compression contraction/break on synthetic
   bands, trendline pivot fit + break, all isolated to the study namespace (live
   ranking untouched).

---

## 5. Testing & rollout checklist (every phase)
- ☐ New pure logic lives in a focused module or a clearly-scoped legacy block,
  with unit tests added under `tests/`.
- ☐ Run the full suite (`unittest`, currently 366 green) — no regressions.
- ☐ Anything that can affect ranking is **flag-gated** and **bounded**, off by
  default until studied.
- ☐ Study namespaces never leak into Expected-R / calibration / live ranking
  (same isolation guarantee as `control_setups`).
- ☐ Update `memory/` notes when a phase ships (what's live vs staged).
- ☐ Persisted stores (levels, ETF map) are schema-versioned + written to the
  cloud folder via `project_paths.py`.

## 6. Open questions for Aaron (resolve as we go)
- Universe for Feat 1/3 strength = current long+short watchlists only, or a
  broader leaders list? (Default: scanned watchlist union.)
- 4h bars: synthesize from 1h (no native IBKR 4h) — acceptable? (Default: yes.)
- Level tolerance start value `0.05×ATR20` — confirm after eyeballing output.
- Industry ETF map: start from the seed list above; Aaron to add/remove names.

## 7. Progress log
- 2026-06-21 — Plan created. Architecture mapped. Starting Phase 1.
- 2026-06-21 — Verified all code anchors against the working tree; corrected the
  §1 map (rrs.py is a shim, RS fn is `assess_daily_relative_strength`,
  `spy_benchmark` source, ETF maps already persisted). Added §8 verified reference
  index, §9 glossary, §10 multi-agent working protocol, §11 standalone task
  tickets (so the plan can be executed by Codex / other agents / a human, not just
  Claude). No code changed yet.
- 2026-06-21 — Added §12 worked implementation design for Feat 5 (HV horizontal
  levels) + Feat 6 (cloud lines): accumulate high+low of ≥3 rvol candles (not
  `valuewhen`), idempotent touch-stats, Span-B +26 displacement fix, new pure
  `levels.py`, store schema, tests. Found existing reuse (`summarize_anchor_compression`
  :3546, `_find_trendline_pivots` :15282, `compute_atr_from_ohlc` :2428).
- 2026-06-21 — Aaron resolved §12.7 tunables: relvol inclusive (match TV); keep
  red tier at lower weight, green is major S/R; tolerance ATR20-scaled (0.05 horiz
  / 0.02 cloud, not ADR%); 3y backfill. T5/T7 now turnkey except forward-window k
  (defaulted 5, non-blocking).
- 2026-06-21 — **T1 / Phase 1 SHIPPED.** Universe strength table + strongest/weakest
  decile across the full scan + "recently strong, now pulling back" list, wired
  through `build_market_prep_payload` and `runner.py`. Deleted the dead duplicate
  `build_market_prep_payload`/`format_market_prep_payload_report`. 5 new tests; full
  suite 338 green (`unittest`; pytest is NOT installed — §10b corrected). Committed
  as `6fb081b` (bundled with prior staged Expected-R/tracker work, disclosed).
- 2026-06-21 — **T6 / B4 SHIPPED.** `study_setups` isolated namespace +
  `master_avwap_study.txt` report, recorded via `update_setup_tracker_from_scan`'s
  new `study_rows` kwarg (no existing caller changed). 9 new tests incl. isolation.
  Full suite 347 green. T4/T5/T7 unblocked. Committed `d324da4`.
- 2026-06-21 — **T2 / Phase 2 SHIPPED.** Demoted TOP to a secondary tag in
  `_derive_setup_family` (real pattern keeps the family; `top_secondary` flag + bonus
  ride along). 5 new tests; full suite 352 green. **Also catalogued ~20 shadowed
  duplicate `legacy.py` functions in §8b#1** (handoff hazard — edit the last `def`).
  Next at that point was T3 (now shipped 2026-06-22).
- 2026-06-22 — **T3 / Phase 3 SHIPPED.** Industry ETF defaults + shared cache/map
  helpers exposed through `rrs.py`; Master AVWAP D1 universe rows now include
  stock-vs-industry context and market prep includes strongest industries; D1 boost
  is bounded and flag-gated off by default; BounceBot M5 candidate scoring now gives
  a bounded directional industry-RRS bonus. 7 new/updated focused tests; full suite
  358 green.
- 2026-06-22 — Confirmed GitHub `origin/main` contains T1 (`6fb081b`), T2
  (`9796b86`), and T3 (`d791b02`). Updated stale §8/§9/§10 references to reflect
  completed market-prep wiring and `study_setups`. Starting T4 on side branch
  `codex/master-avwap-mini-pc-journal-market-prep`.
- 2026-06-22 — **T4 / Phase 4 SHIPPED.** Added B2 intraday fetch, 1h→4h
  resampling, aligned 1h/4h SMA retest detection, `htf_*` output fields, a
  bounded flag-gated `+6` boost (default off), and `study_setups` producer rows.
  2 focused HTF tests + scoring-config coverage; full suite 360 green.
- 2026-06-22 — **T5 / Phase 5 SHIPPED.** Added B3 `DATA_DIR/levels/<SYM>.json`
  level stores, HV horizontal extraction/clustering/touch stats in new
  `levels.py`, earnings-origin/non-earnings HV anchor flags, `hv_level_*` scanner
  context, and study-only proximity/break rows. No live score penalty added.
  5 focused level tests + scanner integration coverage; full suite 366 green.

---

## 8. Verified reference index (ground truth, verified 2026-06-21)

> The §1 map is conceptual. **These anchors were grep-verified against the working
> tree on the date above.** Function/constant *names* are stable; *line numbers
> drift* as `legacy.py` grows — so before editing, re-grep the **name** (e.g.
> `grep -n "def assess_daily_relative_strength" scripts/master_avwap_lib/legacy.py`)
> and trust the fresh result. If §1 and §8 disagree, trust §8.

### 8a. Anchor table

| Symbol / thing | Location (verified) | Notes |
| --- | --- | --- |
| `run_master()` core | `scripts/master_avwap_lib/runner.py` | builds `daily_frames_by_symbol` (`runner.py:339`, populated `:393`) and `priority_rows` (`:340`) |
| SPY benchmark dict | `runner.py:~1266` | `market_regime_snapshot["benchmarks"]["SPY"]` → keys `one_day_return_pct`, `five_day_return_pct`. This is the `spy_benchmark` arg every RS call expects. |
| `build_market_prep_payload` (call site) | `runner.py` (re-grep) | **Updated by T1/T3.** Now passes `daily_frames_by_symbol`, `spy_benchmark`, `sides_by_symbol`, universe strength rows, and industry context into market prep. |
| `build_market_prep_payload` (def) | `legacy.py` (re-grep) | **Single live def after T1 cleanup.** Takes universe rows and industry rows/context; ranks full-universe strength, pullbacks, and strongest industries. |
| `_market_prep_strength_decile_rows` | `legacy.py` (re-grep) | Ranks whichever strength rows it is handed; T1 now feeds full-universe rows when daily frames are available. |
| `MARKET_PREP_STRENGTH_DECILE_FRACTION` | `legacy.py:22265` | `= 0.10` |
| `assess_daily_relative_strength` | `legacy.py:22666` (math `:22713`) | `(daily_rows, last_trade_date, side, spy_benchmark, lookback_days)`. Consumes **`daily_rows` = list of `{date, close,...}` dicts**, not a DataFrame. Returns dict incl. `daily_relative_strength_score`, `daily_relative_strength_bonus`, `symbol_one_day_return_pct`, `symbol_five_day_return_pct`. |
| `daily_relative_strength_bonus` (scoring hook) | `legacy.py:23714` (param), applied `:23787`, persisted `:23900` | The bounded, directional boost pattern to **mirror** for Feat 3/4 boosts. |
| `write_top_strength_watchlist_report` | `legacy.py:23316` | Keep working when TOP is demoted (Phase 2). |
| setup-family machinery | `_normalized_setup_family_text` `legacy.py:4948`, `_canonical_tracker_setup_family` `:5001`, `_tracker_priority_bucket` `:5024`, `MAIN_SWING_SETUP_FAMILIES` used `:5037` | Where Phase 2 decides primary family vs. secondary tag. |
| `_fetch_live_daily_bars` | `legacy.py:13906` | `(ib, symbol, days)`, IBKR-first/Yahoo-fallback. **Mirror this** for B2 `fetch_intraday_bars`. |
| `control_setups` namespace (the study-first template) | created `legacy.py:4873`, loaded `:4925`, recorded/pruned `:9899`–`:9941`, `_prune_control_setups` `:10056`, `_collect_control_episode_observations` `:10126` | Copy this exact shape for `study_setups` (B4). |
| `study_setups` namespace | `legacy.py` (re-grep `_default_setup_tracker_payload`, `update_setup_tracker_from_scan`, `build_study_discovery_rows`) | **DONE by T6.** Isolated study namespace + `master_avwap_study.txt`; producers for T4/T5/T7 pass `study_rows`. |
| RS / ETF infra (bounce_bot) | impls in `scripts/bounce_bot_lib/legacy.py`: `real_relative_strength:1531`, `load_sector_etf_map:475`, `_load_industry_etf_map_file:498`, `load_and_update_industry_etf_map:518`, `resolve_sector_etf:550`, `resolve_industry_ref_etf:567` | Imported via the `rrs.py` shim. **Edit in legacy.py, import from rrs.py.** |
| Shared/cloud data root | `project_paths.py:61` `PERSISTENT_DATA_DIR` (env `TRADINGBOTV3_DATA_DIR` → local `shared_data_dir` → default), `DATA_DIR = PERSISTENT_DATA_DIR/"data"` (`:64`) | New B3/B5 stores go under `DATA_DIR`. This *is* the "Google Drive / cloud folder." |
| ETF map files | `SECTOR_ETF_MAP_FILE`/`INDUSTRY_ETF_MAP_FILE` `project_paths.py:79-80`; repo→data seed/sync `:347-348`, `:432-433` | Phase 3 reuses; add new stores to the same sync lists so they propagate. |
| Tests | `tests/` — **366 tests green** as of T5 (`python -m unittest discover tests`) | "keep green" baseline; no `pytest.ini`/`conftest.py`. |

### 8b. Gotchas that will bite a cold-start agent
1. **`legacy.py` has ~20 SHADOWED DUPLICATE top-level functions** (older copies
   overridden by a later, maintained copy — Python keeps the **last** `def`).
   **ALWAYS edit the copy with the HIGHEST line number.** Verify before editing:
   `grep -n '^def NAME' scripts/master_avwap_lib/legacy.py` → edit the last hit.
   List every duplicate: `grep -oE '^def [A-Za-z_][A-Za-z0-9_]*' \
   scripts/master_avwap_lib/legacy.py | sed 's/^def //' | sort | uniq -d`.
   Known duplicated names (as of 2026-06-21; T1 already removed the
   `build_market_prep_payload`/`format_market_prep_payload_report` pair):
   `build_priority_setup_summary`, `analyze_sma_breakout_setup`,
   `_priority_is_preferred_custom_setup`, `_priority_is_high_conviction`,
   `_priority_best_swing_trade_rows`, `apply_final_priority_buckets`,
   `apply_priority_rejection_score_caps`, `assess_previous_day_range_break`,
   `assess_priority_directional_obstacles`, `build_priority_scoring_attribute_context`,
   `build_tracker_entry_attributes`, `evaluate_theta_put_candidate`,
   `write_master_avwap_focus_feed`, `write_priority_setup_report`,
   `_write_priority_score_ranked_rows`, `_register_tracker_attribute`,
   `_tracker_attribute_is_present`, `_normalize_tracker_attribute_value`,
   `_normalize_focus_priority_entry`, `_focus_priority_bucket_sort_value`.
   (Cleanup candidate: delete the dead copies — out of scope per ticket, do it as a
   standalone PR with the full suite green.)
2. **Edit bounce_bot RS/ETF code in `bounce_bot_lib/legacy.py`, not `rrs.py`.**
3. `assess_daily_relative_strength` wants **`daily_rows` (list of dicts)**; the
   universe data is **DataFrames** (`daily_frames_by_symbol`). Convert, or extract
   a DataFrame-native core both call paths share.
4. The "cloud folder" is `PERSISTENT_DATA_DIR` (resolved at import), **not** a
   hardcoded Drive path. Always go through `project_paths.py`.
5. `legacy.py` is ~1.1 MB; prefer adding **new focused functions** near the
   relevant anchor over weaving into long existing functions. New *pure* logic
   (level extraction, compression metric, trendline fit) should go in **new small
   modules** under `scripts/master_avwap_lib/` (like `expected_r.py`) and be
   imported into legacy/runner — easier to unit-test and review.

---

## 9. Glossary (for an agent/dev without this project's context)

- **RS (relative strength):** here, return *in excess of SPY*. Project formula:
  `0.35×(1-day excess) + 0.65×(5-day excess)`. >0 = outperforming SPY.
- **rs_vs_industry:** same idea but excess vs the symbol's **industry ETF** instead
  of SPY (Feat 3). Aaron's core edge: a long should lead its group, not just ride it.
- **AVWAP / AVWAPE:** Anchored VWAP; "AVWAPE" = the earnings-anchored AVWAP used
  heavily in this bot. **stdev bands** = ±k·σ bands around an AVWAP; "compression"
  (Feat 6) = those bands narrowing over N sessions.
- **relvol (relative volume):** `volume / SMA(volume, 50)`. Aaron's levels: relvol
  in **[2,3) = "red"** level, **≥3 = "green"** (stronger) level (Feat 5).
- **ATR20:** 20-period Average True Range; the volatility unit for tolerances
  (e.g. level merge tol `≈ 0.05×ATR20`).
- **TOP pattern:** a recently-added strength pattern; per Feat 2 it must become a
  **secondary tag/boost**, not a primary `setup_family` that hides other patterns.
- **setup_family:** the canonical label for *which* pattern fired (e.g.
  `sma_breakout`, `post_earnings_*`). The tracker studies hit-rate/R **per family**,
  so collapsing everything to "TOP" destroys that signal.
- **Expected-R ranking:** the live ranking system driven by the setup tracker
  (see `memory/expected-r-ranking-system.md`). "R" = realized reward in units of
  initial risk.
- **control_setups / study_setups:** *side* tracker namespaces the live ranking can
  **never** see, used to measure a new idea's edge before it's allowed to score.
  Both exist now; `study_setups` is the producer target for T4/T5/T7.
- **Timeframes:** **M5** = 5-min, **D1** = daily, **1h/4h** = the HTF trend frames
  Aaron cares about (Feat 4). IBKR has no native 4h → resample from 1h.
- **decile:** top/bottom 10% slice (the strongest/weakest list, Feat 1).
- **Leading Span B:** Ichimoku line `avg(highest(52), lowest(52))` displaced 26;
  its **flat** segments are strong horizontal S/R (Feat 6).

---

## 10. Working protocol for multi-agent / non-Claude execution

**Goal of this section:** let Codex, Cursor, another LLM, or a human pick up a unit
of work and finish it safely without the accumulated chat context.

### 10a. Unit of work
- **One ticket (§11) = one branch = one PR.** Don't batch phases.
- Branch name: `feat/avwap-<phase>-<short-slug>` (e.g.
  `feat/avwap-p1-universe-strength-decile`). The current working branch already
  follows a `codex/...` convention — match whatever the assignee tool uses.

### 10b. Environment & test command (Windows, repo root `c:\Users\aaron\TradingBotV3`)
**pytest is NOT installed in `.venv`** — the suite is `unittest`-based. Use unittest:
```powershell
# full suite (baseline: 366 passing as of 2026-06-22, through T5):
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
# one file while iterating:
.\.venv\Scripts\python.exe -m unittest tests.test_master_avwap_setups
# one test method:
.\.venv\Scripts\python.exe -m unittest tests.test_master_avwap_setups.MasterAvwapSetupTests.test_market_prep_decile_ranks_full_universe_not_just_flagged
```
(If you prefer pytest, `pip install pytest` first — it will also discover `tests/test_*.py`.)
Byte-compile a changed module fast: `.\.venv\Scripts\python.exe -m py_compile scripts\master_avwap_lib\legacy.py`.

### 10c. Non-negotiable invariants (every ticket must hold these)
1. **Study-first.** New signals (Feat 4/5/6) record to a **side namespace** +
   `*_study.txt` report; they do **not** touch the live score until the tracker
   shows edge. Mirror `control_setups`.
2. **Study isolation.** `study_setups`/`control_setups` must **never** feed
   Expected-R, calibration, or the live ranking. Add a test asserting isolation.
3. **Scoring changes are flag-gated + bounded + off-by-default.** Any boost/penalty
   sits behind a config flag and has a hard cap (mirror `daily_relative_strength_bonus`).
4. **Persisted stores are schema-versioned** (a `"schema_version"` field) and live
   under `DATA_DIR` via `project_paths.py`; add them to the repo→data sync lists
   (`project_paths.py:347-348` / `432-433`) so both machines get them.
5. **Don't break the suite.** Full suite green before "done."
6. **Re-grep anchors before editing** (line numbers drift; see §8 header).

### 10d. Definition of Done (per ticket)
- [ ] Code matches the ticket's "Files to touch" and respects §10c invariants.
- [ ] New unit tests added (listed in the ticket) and passing.
- [ ] `python -m pytest -q` fully green (no regressions).
- [ ] If a phase shipped: tick its boxes in §3/§4, append a dated line to §7,
      and update the relevant note in `memory/` (live vs. staged).
- [ ] PR description links the ticket and states what's live vs. study-only.

---

## 11. Standalone task tickets (hand one at a time to any coding agent)

Each ticket is written to be **pasted directly** into another agent as its whole
task. They are ordered by dependency. **Ticket T1 below is fully worked as the
canonical example;** T2+ are scoped headers that point back to the detailed steps in
§4 plus the verified anchors in §8 — expand any one into the T1 format before
handing it off.

### Ticket template (use this shape for every ticket)
```
TITLE:        <imperative, one line>
PHASE:        <P# / B#>   DEPENDS ON: <tickets that must land first>
CONTEXT:      <2–4 lines: the user-facing goal + why, from §0/§9>
FILES:        <verified file:anchor list from §8 — re-grep names first>
STEPS:        <numbered, concrete edits>
TESTS:        <test file + the specific cases to add>
ACCEPTANCE:   <observable behavior that proves it works>
INVARIANTS:   <which of §10c apply>
OUT OF SCOPE: <explicitly what NOT to do in this ticket>
```

### T1 — Universe strength table + Phase 1 strongest/weakest decile  ☑ DONE (2026-06-21)
*(This is the worked example. Build B1, then fix Feat 1.)*

> **Shipped.** `build_universe_strength_rows` + `_daily_frame_to_rows` +
> `_trailing_return_pct` added near `assess_daily_relative_strength` in `legacy.py`;
> `_market_prep_pullback_rows`/`_market_prep_pullback_details` + a
> `recently_strong_now_pulling_back` section added; the live `build_market_prep_payload`
> now takes `daily_frames_by_symbol`/`spy_benchmark`/`sides_by_symbol` and ranks the
> decile across the whole universe (falls back to `priority_rows` if no frames given);
> `runner.py` threads those in at the call site. **Also deleted the dead duplicate
> `build_market_prep_payload`/`format_market_prep_payload_report` (the §8 footgun).**
> 5 new tests in `tests/test_master_avwap_setups.py`; full suite 338 green.
```
TITLE:   Rank strongest/weakest 10% across the FULL scanned universe for market prep
PHASE:   B1 + P1     DEPENDS ON: none
CONTEXT: Market-prep "strongest/weakest" currently reads only flagged setups
         (priority_rows), so it's empty/misleading. Aaron wants the strongest/
         weakest decile of the WHOLE scanned universe (long+short watchlist union),
         by D1 RS-vs-SPY, for pre-market prep. Also surface "was strong 1–2 weeks
         ago, now pulling back" — his preferred buy.
FILES:
  - scripts/master_avwap_lib/legacy.py
      * NEW build_universe_strength_rows(...) near assess_daily_relative_strength (:22666)
      * _market_prep_strength_decile_rows (:23387)   # repoint to universe rows
      * build_market_prep_payload  -> EDIT THE LIVE DEF AT :26242 (NOT :20816)
  - scripts/master_avwap_lib/runner.py
      * call site :1678  # thread daily_frames_by_symbol + spy_benchmark in
      * SPY dict source ~:1266 (market_regime_snapshot["benchmarks"]["SPY"])
      * daily_frames_by_symbol available from :393
STEPS:
  1. Add build_universe_strength_rows(daily_frames_by_symbol, spy_benchmark,
     last_trade_dates) -> list[dict], one row per scanned symbol:
     {symbol, rs_d1, ret_1d, ret_5d, ret_13w, ret_26w, atr20}. Reuse the
     0.35/0.65 excess-vs-SPY math (factor a DataFrame-native core out of
     assess_daily_relative_strength, OR adapt each DataFrame to the {date,close}
     row list it expects — see §8b#3).
  2. In runner.py:1678, pass daily_frames_by_symbol and the SPY benchmark dict
     into build_market_prep_payload (add params to the :26242 def; keep
     priority_rows for the existing fields → backward compatible).
  3. In build_market_prep_payload (:26242), build universe rows and feed THEM to
     _market_prep_strength_decile_rows (strongest = rs_d1 desc, weakest = asc),
     keeping fraction = MARKET_PREP_STRENGTH_DECILE_FRACTION (0.10). Ties → symbol asc.
  4. Add a "recently strong, now pulling back" sub-list: ret_26w (or ret_13w) > 0
     AND ret_1d (or ret_5d) < 0; cap the list length.
  5. Update the market-prep report writer + mini-PC status to show the UNIVERSE
     count, not the flagged-setup count.
TESTS (extend tests/test_master_avwap_setups.py and
       tests/test_master_avwap_mini_pc_status.py):
  - universe of N synthetic symbols -> correct top/bottom 10% by rs_d1
  - tie-break by symbol
  - empty / insufficient RS rows handled (no crash, empty section)
  - pullback sub-list picks the strong-then-weak fixture only
ACCEPTANCE: with an ~80-name watchlist, market prep shows ~8 strongest / ~8
  weakest by D1-vs-SPY across all 80 (not just flagged), plus the pullback list.
INVARIANTS: §10c #5, #6. (No scoring change, no new persisted store, no study ns.)
OUT OF SCOPE: industry RS (that's T3), any scoring boost, 1h/4h.
```

### T2 — Demote TOP to a secondary tag/boost  (Phase 2)  ☑ DONE (2026-06-21)
- Expand §4 "Phase 2" into the template. Key anchors (§8): setup-family machinery
  `legacy.py:4948/5001/5024`, `write_top_strength_watchlist_report:23316`, plus
  grep `TOP` across `legacy.py`, `runner.py`, `master_avwap_shared.py`. Net effect:
  primary `setup_family` = the real pattern that fired; attach `top_secondary=True`
  + a bounded boost; tracker records the base family (not "TOP"). DEPENDS ON: none.

> **Shipped.** The single chokepoint was `_derive_setup_family` — TOP was checked
> *above* sma_breakout/band_bounce/extreme_move/retest. Moved the TOP block **below**
> all specific entry patterns (still above the generic `avwap_breakout`/`general`
> fallbacks), so a real pattern wins and TOP-only setups still track as `top_pattern`.
> Added a `top_secondary` flag to the priority row (TOP fired but family is something
> else); the TOP score bonus was already applied independently of family, so the boost
> survives demotion automatically. Relaxed `_priority_is_preferred_custom_setup` and
> `_priority_should_track_top_pattern` to gate on the `top_pattern_*` flags instead of
> the family (so TOP context + watchlist survive demotion; the strength watchlist was
> already flag-based). 5 new tests; full suite 352 green. **Discovered ~20 shadowed
> duplicate functions in `legacy.py` — documented in §8b#1.**

### T3 — Industry/sector relative strength  (Phase 3, B1)  ☑ DONE (2026-06-22)
- Expand §4 "Phase 3." Reuse `bounce_bot_lib/legacy.py` RS/ETF fns via the `rrs.py`
  shim (§8); the JSON ETF maps already exist (`project_paths.py:79-80`). Add
  `industry_etf` + `rs_vs_industry` to the B1 rows (T1), an industry-strength
  ranking section, and a **flag-gated bounded** boost mirroring
  `daily_relative_strength_bonus` (`legacy.py:23714`). DEPENDS ON: T1.

> **Shipped.** See Phase 3 + progress log notes above. Daily scan now computes
> symbol-vs-industry D1 context, market prep surfaces strongest industries, the
> D1 score boost is feature-flagged off by default, and BounceBot M5 candidate
> scoring now prefers directionally aligned industry RRS.

### T4 — B2 intraday fetch + 1h/4h trend boost & study  (Phase 4)  ☑ DONE (2026-06-22)
- Expand §4 "Phase 4." Build `fetch_intraday_bars(ib, sym, bar_size, duration)`
  mirroring `_fetch_live_daily_bars` (`legacy.py:13906`); resample 1h→4h. Boost is
  small/bounded/flag-gated; record retest-and-go to `study_setups`. DEPENDS ON: B4 (T6).

> **Shipped.** See Phase 4 + progress log notes above. The boost is gated off by
> default; the study namespace records HTF retest candidates for outcome discovery.

### T5 — B3 level store + D1 high-volume horizontal levels  (Phase 5)  ☑ DONE (2026-06-22)
- **→ Full worked algorithm in §12.1.** New schema-versioned per-symbol JSON under
  `DATA_DIR` (+ sync lists). Accumulate **high AND low** of every qualifying ≥2 rvol candle
  (not `valuewhen`), ATR-scaled clustering (`0.05×ATR20`), touch stats recomputed
  each run (idempotent), earnings-day exclusion. **Study-first** (record
  approach/break events; no scoring yet). DEPENDS ON: B4 (T6).

> **Shipped.** See Phase 5 + progress log notes above. The level store and
> study producers are live; the scoring penalty remains intentionally deferred.

### T6 — B4 study_setups namespace + master_avwap_study.txt  (cross-cutting)  ☑ DONE (2026-06-21)
- **Build this early — T4/T5/T7 depend on it.** Clone the `control_setups`
  template (§8: `legacy.py:4873/9899/10056/10126`) into a parallel `study_setups`
  namespace + a `master_avwap_study.txt` report. **Add a test asserting it never
  feeds Expected-R/calibration/live ranking** (§10c #2). DEPENDS ON: none.

> **Shipped.** `study_setups` added to `_default_setup_tracker_payload` + loader +
> save round-trip; `update_setup_tracker_from_scan` gained a `study_rows=None` kwarg
> that records into the isolated namespace (`study:` prefix, `is_study=True`,
> `study_kind`), pruned (`_prune_study_setups`, `STUDY_SETUP_KEEP_DAYS=200`/
> `MAX_RECORDS=4000`) + recomputed each scan like control. `build_study_discovery_rows`
> + `write_master_avwap_study_report` → `MASTER_AVWAP_STUDY_FILE`
> (`master_avwap_study.txt`), reusing the namespace-agnostic
> `_collect_control_episode_observations` / `_summarize_control_observation_group`.
> **Producers come later: T4/T5/T7 just build `study_rows` and pass them in** — no
> existing caller changed (kwarg defaults to None). 9 new tests in
> `tests/test_study_setups.py` incl. an isolation test (study setups don't change
> control/Expected-R discovery). Full suite 347 green.

### T7 — Cloud lines, compression breaks, trendline breaks  (Phase 6)  ☐
- **→ Cloud-line algorithm (with the +26 displacement fix) in §12.2.** Pure
  study-only. Leading Span B flat segments into the B3 store; compression reuses
  the **existing** `summarize_anchor_compression` (`legacy.py:3546`) + a break
  event; trendline reuses the **existing** `_find_trendline_pivots`
  (`legacy.py:15282`). All events → `study_setups` only. DEPENDS ON: T5 (B3), T6 (B4).

### Suggested execution order for parallel agents
`T6` and `T1` first (independent, unblock the most). Then `T2`, `T3` (needs T1).
Next `T7` (needs T5 + B4, both now done).

---

## 12. Worked implementation design — high-volume levels & cloud lines (Feat 5 & 6)

> This is the part T5/T7 were hand-waving. Everything here is **study-first**
> (no scoring until the tracker shows edge) and built on verified anchors (§8).
> New pure logic goes in a **new module `scripts/master_avwap_lib/levels.py`**
> (keep it out of the 1.1 MB `legacy.py`; import it where needed). Bar frames are
> `daily_frames_by_symbol[sym]`: DataFrame, columns
> `["datetime","open","high","low","close","volume"]`, ascending (`legacy.py:1372`).
> ATR via `compute_atr_from_ohlc` (`legacy.py:2428`, `ATR_LENGTH=20`).

### 12.0 Why the TradingView scripts don't port literally
- **HV levels:** the PineScript uses `ta.valuewhen(cond, high, 0)` → only the
  **most recent** qualifying candle's high, redrawn each bar. We want to
  **accumulate** levels: iterate *all* bars, and for *each* qualifying ≥2 rvol candle record
  **both its high and its low** as persistent levels. Different algorithm.
- **Cloud lines:** Span B is plotted with `offset=+26`. The flat segment that price
  reacts to **today** was computed **26 bars ago**. Comparing today's price to
  today's `mid52` checks the wrong level — must use the displaced (effective) range.
- **Persistence:** levels "stay relevant forever," but fetch windows overlap run to
  run. Naive increment double-counts touches. Fix: persist level **identity +
  first-seen** only; **recompute touch/break stats from history every run**
  (idempotent).

### 12.1 High-volume horizontal levels

**relvol** — **CONFIRMED inclusive of the current bar** (matches Aaron's
TradingView `ta.sma(volume,50)`; do not exclude the current candle):
```
vol_sma50[i] = mean(volume[i-49 .. i])         # inclusive of bar i; needs >=50 prior bars
relvol[i]    = volume[i] / vol_sma50[i]
```
Buckets (Aaron: "typically over 3 rvol") — **both kept**:
- `green` (**major S/R**, primary weight): `relvol >= HV_RELVOL_GREEN (=3.0)`
- `red`   (**kept but lower weight**, leans study-first): `2.0 <= relvol < 3.0`

**Candidate extraction** — for each qualifying candle emit **two** candidates:
```
LevelCandidate(price=high, origin_side="high", bucket, relvol, date, earnings_origin)
LevelCandidate(price=low,  origin_side="low",  bucket, relvol, date, earnings_origin)
```
(`origin_side` is provenance only — polarity flips on break, so proximity is what
matters, not "is it support or resistance".)

**Clustering** (greedy, ATR-scaled): sort candidates by price; merge any within
`tol = LEVEL_TOL_ATR_FRACTION (=0.05) × atr20`. Per cluster:
- `price` = **rvol-weighted mean** of members (gravitates to the heaviest-volume
  level — the "real" line); keep `band=[min,max]` of members for display/tol.
- `bucket` = strongest member (green > red); `relvol` = max member relvol.
- `first_seen` = earliest member date (persisted forever).

**Touch / respect / break stats** — recompute every run from the full available
history (idempotent), do **not** increment across runs:
```
band = price ± tol
touch  : a later bar with low <= price+tol AND high >= price-tol   -> touch_count++
break  : a later bar closing beyond price by > LEVEL_BREAK_ATR(=0.25)*atr20
         on either side -> break_count++ ; record fwd_return over next k bars
respect: a touch that did NOT break and reversed within k bars (the "held" case)
```
These per-level stats are exactly what the study report measures: *how often levels
hold vs. break, and how sharp the post-break move is* (Aaron's stated question).

**Strength** (for *later* scoring; study-only at first): older + more-touched +
higher bucket = stronger (Aaron: "respected more and more over time"). Green is the
**major** S/R weight, red is minor:
```
BUCKET_W = {"green": 1.0, "red": 0.35}   # green dominates; red still contributes
strength = BUCKET_W[bucket] + min(TOUCH_CAP, touch_count*TOUCH_W) + age_bonus(first_seen)
```

**Earnings exclusion / AVWAP-anchor spinoff:** the scan already has the earnings
calendar / `latest_release_map` (threaded into `build_market_prep_payload`). Mark a
candle `earnings_origin=True` if within ±1 session of a release. Non-earnings ≥3
rvol days additionally get emitted to an `hv_avwap_anchor_candidates` list — mirror
`EARNINGS_ANCHOR_CANDIDATES_FILE` (`project_paths.py:83`) — so the "anchor AVWAPs on
big non-earnings days and see if they're respected" idea can be studied later. (No
AVWAP math in this ticket — just flag the anchor dates.)

**History depth:** levels want years of data, but the live scan fetches few days.
Add a periodic **deep backfill** (`_fetch_live_daily_bars(ib, sym, days=HV_LEVEL_BACKFILL_DAYS≈750)`)
used only for the level store; thereafter the persisted store carries levels whose
origin candle has aged out of the normal window.

### 12.2 Cloud lines (Ichimoku Leading Span B, flat segments)

**Span B midpoint** (Donchian, length 52), per bar:
```
mid52[i] = ( max(high[i-51 .. i]) + min(low[i-51 .. i]) ) / 2
```
**Flat detection** (robust replacement for Pine's `sma(line,8)==line`): a run of
`>= CLOUD_FLAT_MIN_BARS (=8)` consecutive bars where `mid52` is constant within
`cloud_tol = max(CLOUD_TOL_ATR_FRACTION(=0.02)*atr20, CLOUD_TOL_PCT*price)`. (These
lines are "rather exact" → tight tol.) Collapse each run into one `FlatSegment`
with `value = mean(mid52 over run)`.

**The displacement fix (the important bit):** Span B is drawn `+CLOUD_DISPLACEMENT
(=26)` bars forward. So a segment computed over bars `[t0..t1]` is *effective* over
`[t0+26 .. t1+26]` (extends 26 bars past the last bar). Store both:
```
FlatSegment(value, computed_range=[date(t0),date(t1)],
            effective_range=[date(t0+26), date(t1+26 or +26 past last bar)], bar_count)
```
For "is price near a cloud line **now**", only consider segments whose
`effective_range` covers today (i.e. computed within the last 26 bars or ongoing).

Store as level `kind="cloud_flat"` in the same B3 store. "Don't enter right at a
flat cloud line" → unified proximity check below.

### 12.3 Shared store, module API, and unified proximity check
**Module `scripts/master_avwap_lib/levels.py`** (pure, unit-tested):
```
compute_relvol(df, vol_sma=50) -> pd.Series
extract_hv_levels(df, atr20, *, green=3.0, red=2.0, earnings_dates=()) -> list[LevelCandidate]
cluster_levels(candidates, atr20, *, tol_frac=0.05) -> list[Cluster]
compute_span_b_flats(df, atr20, *, length=52, displacement=26, min_bars=8, tol_frac=0.02) -> list[FlatSegment]
recompute_touch_stats(levels, df, atr20) -> None        # idempotent, per run
merge_into_store(store, clusters, flats, *, atr20) -> store  # preserves first_seen
levels_near(store, price, atr20) -> list[level]         # per-kind tol
levels_blocking_entry(store, side, entry_price, atr20) -> list[level]  # overhead res for longs / support for shorts
```
**Store** (`DATA_DIR/levels/<SYM>.json`, schema-versioned, added to the
`project_paths.py` repo→data sync lists at `:347-348`/`:432-433`):
```json
{ "schema_version": 1, "symbol": "NVDA", "updated": "2026-06-21", "atr20_at_update": 4.2,
  "levels": [
    {"kind":"hv_horizontal","price":123.45,"band":[123.1,123.8],"origin_side":"high",
     "bucket":"green","relvol":3.8,"first_seen":"2025-11-04","last_touch":"2026-06-12",
     "touch_count":5,"break_count":1,"earnings_origin":false,"strength":0.78},
    {"kind":"cloud_flat","price":110.2,"computed_range":["2026-04-01","2026-04-22"],
     "effective_range":["2026-05-09","2026-05-30"],"bar_count":15,"strength":0.6}
  ] }
```

### 12.4 Config (all new, off/neutral by default — §10c #3)
```
HV_LEVELS_ENABLED=False        HV_RELVOL_GREEN=3.0   HV_RELVOL_RED=2.0   HV_VOL_SMA=50
LEVEL_TOL_ATR_FRACTION=0.05    LEVEL_BREAK_ATR=0.25  HV_LEVEL_BACKFILL_DAYS=750
CLOUD_LINES_ENABLED=False      CLOUD_SPAN_B_LEN=52   CLOUD_DISPLACEMENT=26
CLOUD_FLAT_MIN_BARS=8          CLOUD_TOL_ATR_FRACTION=0.02   CLOUD_TOL_PCT=0.0005
LEVEL_SCORING_ENABLED=False    LEVEL_PROXIMITY_PENALTY_CAP=<small>
```

### 12.5 Study & (later) scoring
- **Study (now):** each run, for every scanned symbol near a stored level, record an
  event to `study_setups` (T6) + `master_avwap_study.txt`: `{symbol, kind, bucket,
  dist_atr, approached_from, outcome(held/broke), fwd_return_k}`. This answers "how
  often do levels hold / how sharp are breaks" before any scoring.
- **Scoring (later, gated by `LEVEL_SCORING_ENABLED`):** penalize an entry sitting
  just under a strong overhead level (long) / above support (short) via
  `levels_blocking_entry`; bounded by `LEVEL_PROXIMITY_PENALTY_CAP`, mirroring the
  `daily_relative_strength_bonus` mechanism (`legacy.py:23714`).

### 12.6 Tests (`tests/test_levels.py`, new)
- `compute_relvol`: known volumes → exact relvol; <50 bars → NaN/skip.
- `extract_hv_levels`: synthetic frame with one ≥3 and one [2,3) candle → 4 green +
  N red candidates, both high & low, correct buckets; earnings-day flagged.
- `cluster_levels`: three near prices within tol → one cluster, rvol-weighted price,
  correct band & first_seen; one far price → separate cluster.
- `recompute_touch_stats`: deterministic touch/break counts; **running twice gives
  identical counts** (idempotency guard).
- `compute_span_b_flats`: synthetic series with a flat midpoint run → one segment;
  **assert `effective_range == computed_range shifted +26 bars`** (displacement guard).
- `levels_blocking_entry`: long with a strong green level 0.1·ATR overhead → flagged;
  far level → not flagged.
- Isolation: level study events land only in `study_setups`, never live ranking (§10c #2).

### 12.7 Resolved decisions (Aaron, 2026-06-21) — build to these
1. ✅ **relvol SMA inclusive of the current bar** (match TradingView). Locked.
2. ✅ **Keep both tiers.** Green (≥3) is the major S/R; red [2,3) is kept but lower
   weight (`BUCKET_W` green 1.0 / red 0.35) and leans study-first. Locked.
3. ✅ **Tolerance is ATR-scaled, which already makes high movers whip wider.**
   Decision: base on **ATR20 (absolute dollars)**, not ADR%. Rationale: levels are
   absolute-price objects, so the natural noise band around a level *is* the dollar
   true range (ATR). Because `tol = FRACTION × ATR20` and ATR20 is per-stock, a
   high-mover (big ATR) automatically gets a proportionally wider tolerance — which
   is exactly Aaron's "higher movers whip around more." (ADR% is the right unit for
   *sizing/score normalization*, not for level proximity — keep it in mind for the
   scoring step, not here.) Values: **`LEVEL_TOL_ATR_FRACTION=0.05`** (horizontal —
   matches Aaron's own "0.05×20dayATR" note in `thingstointegrate.txt`) and
   **`CLOUD_TOL_ATR_FRACTION=0.02`** (cloud lines are "rather exact" → tighter), with
   a tiny `CLOUD_TOL_PCT` price floor so very high-priced names don't collapse to ~0.
   Tighten/loosen only after eyeballing real output.
4. ✅ **Deep-backfill depth ≈ 3 years** (`HV_LEVEL_BACKFILL_DAYS=750`). Locked.
5. ◐ **Forward window `k` for break/respect outcome** — defaulted to **5 sessions**
   (Aaron didn't specify). Easy to change; study report can emit k=5 and k=10 side
   by side so the better horizon is obvious. No build blocker.
