# TradingBotV3 Improvement Plan

## 1. Purpose and product boundary

TradingBotV3 should become a dependable, fully automated **decision-support system** for day trading and swing trading. It should do everything around the trade except place or manage orders:

- prepare the market before the session;
- build and maintain useful universes and watchlists;
- find, rank, explain, and monitor day-trade and swing-trade setups;
- run unattended and publish a phone-friendly report to the shared Google Drive folder;
- capture both bot picks and human picks before their outcomes are known;
- measure setup performance without look-ahead bias;
- connect alerts and picks to the journal;
- help discover, validate, promote, demote, and retire setups;
- remain fast and understandable while it runs all day.

This plan deliberately excludes execution. No phase should add order placement, automatic position sizing against a live account, or autonomous trade management. The application may calculate hypothetical entry, stop, target, and risk information, but it should continue to require the trader to execute independently.

### Confirmed operating model

The **main PySide6 GUI is the product and the primary runtime**. Auto Mode is not a mini-PC feature and must not be implemented as a second, weaker copy of the application. The intended operating model is:

- launch `launch_gui.py` / the main Qt GUI once;
- let one application runtime own market data, scanning, scheduling, ranking, alerts, research snapshots, and reports;
- keep **Auto Mode visible and central** whether the trader is at the desk or away at work;
- use the same Auto Mode decisions for the desktop opportunity feed and the Google Drive phone report;
- treat “Desk” and “Away” as presentation/notification profiles, not different trading algorithms;
- keep `master_avwap_mini_pc.py` only as an optional headless/compatibility host for the same services;
- spend minimal product effort on the mini-PC-specific window after its existing contract is stabilized.

The main implementation priority is therefore a GUI-owned `TradingRuntime`/`AutoModeCoordinator`, not further investment in a parallel mini-PC workflow.

## 2. Review snapshot

The repository is already a serious application rather than a small script:

- approximately 231 Python files and 102,000 Python lines;
- a PySide6 desktop UI plus legacy Tk/PyQt compatibility code;
- Master AVWAP swing scanning, BounceBot intraday monitoring, market prep, industry/relative-strength tools, a setup tracker, playbook research, a journal, autopilot scheduling, and Google Drive-backed output;
- 717 collected tests in the current environment;
- useful caching, atomic-write helpers, durable bar stores, tracker recompute memoization, earnings-history write deferral, outcome tracking, expected-R shrinkage, and provider fallback logic already exist.

The current test baseline is **715 passed, 2 failed, 7 warnings**. Both failures concern the mini-PC path no longer passing `include_theta=True` while tests still require it:

- `tests/test_master_avwap_mini_pc.py`
- `tests/test_public_entrypoints.py`

The warnings come from deprecated `QSortFilterProxyModel.invalidateFilter()` calls in the setup, theta, and journal proxy models.

The working tree also contains substantial active work. Implementation of this plan must preserve those changes and begin from a clean, intentional checkpoint. Only this plan file was added during the review.

## 3. What is already working well

These parts should be reused and strengthened rather than rebuilt:

1. **Clear product capabilities.** The scanner, setup tracker, market-prep services, journal, autopilot, and new Qt UI cover the correct product surface.
2. **Good test breadth.** Tests cover setup detection, tracker recomputation, expected R, bounce learning, earnings, market prep, journal imports, GUI feeds, and public entrypoints.
3. **Performance work has begun.** The tracker avoids recomputing sealed records, reuses indicator frames and band histories, skips unnecessary live-bar refreshes, and batches large earnings-history writes.
4. **Durable signal learning exists.** Human focus picks, bot tiers, bounce outcomes, pick feedback, control setups, regime conditioning, and setup-playbook studies are valuable foundations.
5. **Local/shared storage separation is partially correct.** Replaceable caches and diagnostic logs are local while selected durable data and phone reports are shared.
6. **The market-time abstraction handles timezone conversion and DST.** It is better than hard-coded Pacific-time scheduling.
7. **Scanner subprocess isolation is a good safety choice.** A native broker/data-library failure is less likely to close the desktop GUI.
8. **Market prep is relatively modular.** Its orchestrator/service layout is a useful model for refactoring the scanner cores.

The plan below keeps these strengths and removes the main scaling and reliability limits.

## 4. Highest-priority findings

### P0 — Correctness and unattended reliability

1. The repository is not currently green. The mini-PC/theta contract must be reconciled before other changes so failures introduced later are unambiguous.
2. Scheduling is spread across the Qt Master panel, `AutopilotService`, and `master_avwap_mini_pc.py`. Multiple implementations can disagree about due slots, tracker-write slots, shutdown, retries, holidays, and completed work.
3. Scheduling understands weekends and timezones but not the authoritative exchange calendar. Early closes and full market holidays are inferred indirectly from missing/stale SPY data.
4. Autopilot state is largely a mutable JSON snapshot. It needs a durable job ledger so a restart can distinguish queued, started, completed, failed, skipped, and stale work.
5. The scan subprocess returns as soon as a marker is printed while a deferred child task can remain alive. The GUI does not own a complete process lifecycle and cannot reliably cancel or reap every child at shutdown.
6. Many shared CSV/JSON/text files can be read and written by the GUI, scanner subprocess, mini-PC, and a second synced computer. Atomic replacement prevents partial files, but it does not prevent lost updates from two valid writers.
7. The journal SQLite database lives in the shared runtime directory. A cloud-sync folder is not a database replication protocol; concurrent machines or sync conflict copies can corrupt or fork the journal.
8. Important credentials, including Questrade and IBKR Flex tokens, can be stored as plaintext values in `local_settings.json`.
9. Exceptions are often caught broadly and suppressed. The three largest runtime areas inspected contain 163 broad `Exception` handlers. This keeps the UI alive but can make degraded data look like a valid empty result.
10. `project_paths.py` performs directory and drive-availability work at import time and can wait up to 120 seconds. Import side effects make testing, packaging, and recovery harder.

### P1 — Scanner speed and resource use

1. `run_master` processes symbols in one large sequential loop and calls `fetch_daily_bars` per symbol. Provider limits matter, but cache resolution, local reads, feature computation, and many enrichments can be batched or parallelized safely.
2. The 27,000-line `master_avwap_lib/legacy.py` and 8,700-line `bounce_bot_lib/legacy.py` modules make it difficult to identify repeated calculations and enforce boundaries. One duplicate top-level definition, `_priority_expected_r_text`, already exists in the Master legacy module.
3. The BounceBot loop has many per-symbol historical-data and cache access paths. Without one request coordinator it is hard to prove that broker pacing, priority symbols, retry backoff, and stale cache rules are optimal.
4. Master scan output is fanned out to many JSON/CSV/text files. Some are necessary views, but repeated full serialization and shared-drive writes add latency and sync churn.
5. The setup tracker is described in code as hundreds of megabytes and is still a large mutable JSON document. Loading it once per scan is an improvement, but rewriting the whole document remains an eventual bottleneck.
6. Several GUI panels maintain their own timers, watchers, and disk readers. The same output can be parsed repeatedly by different panels instead of being loaded once and published as application state.
7. Performance timing exists in logs, but there is no persisted, structured run profile with phase time, symbol count, cache hit rate, provider calls, bytes written, and errors. Optimization therefore lacks a stable before/after baseline.

### P1 — Evidence quality and setup discovery

1. There are many useful outcome tables, but there is no single immutable signal/event identity used across alerting, tracking, research, and journaling.
2. Strategy code and research code are close together in the legacy modules. A detector can drift away from its backtest version without a schema/version check making that mismatch obvious.
3. The playbook study includes costs, sample counts, means, medians, win rates, and t-statistics, and expected-R logic uses shrinkage. The next required step is formal time-based validation, leakage tests, confidence intervals, and multiple-hypothesis controls.
4. Universe history needs to be point-in-time. Using today's listed/optionable universe to evaluate past signals introduces survivorship and eligibility bias.
5. Bar provenance needs to travel with every signal. IBKR/yfinance fallback is useful operationally, but adjusted-price behavior, incomplete bars, splits, stale timestamps, and provider substitutions can change results.
6. Outcomes need consistent definitions by strategy horizon: alert-time entry versus next-bar entry, MFE, MAE, close R, target/stop ordering inside a bar, gap-through behavior, expiry, and realistic transaction costs.
7. Ranking currently combines rule scores, tracker adjustments, expected R, regimes, and other gates. That is powerful, but the contribution of each stage needs to be logged so the system can explain why a candidate was promoted, blocked, or demoted.

### P2 — Product focus and maintainability

1. The UI exposes many capable panels, but the primary trader question is still: **what deserves attention now, why, what changed, and how fresh is the evidence?** A unified opportunity inbox should become the main surface.
2. Dependencies are unpinned and there is no `pyproject.toml`, lint/type configuration, coverage gate, or repository CI workflow.
3. Both Qt and legacy UI stacks remain installed. This increases startup/import surface, packaging complexity, and the chance that fixes are applied to only one UI.
4. Configuration constants are distributed across Python modules, JSON, environment variables, and local settings without one typed settings model or validation report.
5. Existing roadmap documents cover consumer shipping and broker adapters. This plan should be implemented alongside those documents, not as a competing rewrite.

## 5. Target architecture

The long-term shape should be a modular monolith: one product, several well-defined internal services, and no network microservices required.

```text
Qt UI / phone report
        |
Opportunity service  <----  Health/status service
        |
Scheduler / job supervisor
        |
Scan orchestrators (day, swing, prep, research, journal)
        |
---------------------------------------------------------
| provider adapters | feature engine | setup detectors  |
| ranking/policy    | outcome engine | journal linker   |
---------------------------------------------------------
        |
Local operational store + immutable shared event exports
        |
IBKR / yfinance / calendars / SEC / news / shared Drive
```

Important rules:

- UI widgets consume services/models; they do not call broker SDKs or parse large files independently.
- Setup detectors are pure functions over versioned feature snapshots.
- A signal is stored before its outcome is known and never silently rewritten.
- Provider data has timestamps, source, adjustment policy, and freshness status.
- The scheduler owns job state and process lifecycle.
- SQLite databases remain local to one writer. Cross-machine sharing uses immutable events/snapshots or an explicit single-writer lease.
- Text, CSV, and JSON remain supported as human-readable exports, not the primary mutable database for large histories.

## 6. Canonical data model

Introduce a small set of versioned records before adding more setup logic.

### 6.1 Scan run

Every scan gets a `run_id` and a manifest containing:

- job type and trigger (`manual`, `scheduled`, `recovery`);
- app version/commit and configuration hash;
- start/end timestamps in UTC plus exchange-local date;
- requested, processed, skipped, failed, and stale symbols;
- provider request counts, retries, pacing waits, failures, and fallbacks;
- cache hit/miss counts by data type;
- timing by phase;
- input dataset versions and latest bar timestamps;
- output row counts and output paths;
- final status and structured error codes.

### 6.2 Feature snapshot

Each symbol/run snapshot should contain:

- `feature_snapshot_id`, `run_id`, symbol, side, and as-of timestamp;
- bar timestamp and whether the bar was complete when observed;
- daily/intraday provider and adjusted-price policy;
- setup-independent features: price/volume, ATR/ADR, trend, relative strength, sector/industry state, regime, catalyst/earnings distance, liquidity, gap, volatility, and key levels;
- a feature-schema version.

### 6.3 Signal event

Every candidate, including rejected/near-miss controls when sampled, should have:

- stable `signal_id`;
- detector ID and detector version;
- feature snapshot ID;
- first observed time, last observed time, and current lifecycle state;
- side, horizon (`day`, `swing`, `theta_research`), setup family, and trigger;
- proposed entry zone, invalidation, targets, and the price actually available at observation time;
- raw detector evidence;
- each ranking/gating adjustment and final rank;
- source cohort: bot, human focus, manual watchlist, autopilot, control, or imported;
- data-quality flags.

### 6.4 Outcome

Outcomes should be append-only and keyed by `signal_id`:

- evaluation policy version;
- entry policy and timestamp;
- MFE/MAE in price, percent, ATR, and R;
- close outcomes at strategy-specific horizons;
- target/stop/expiry result and ambiguous-bar flag;
- modeled commission, spread, and slippage;
- data completeness and provider;
- whether the result was available when a model/config change was made.

### 6.5 Opportunity

The UI/phone report should consume a normalized opportunity view:

- symbol, side, day/swing horizon, setup, current stage;
- rank, confidence band, expected R, sample count, and freshness;
- catalyst and regime context;
- concise “why now,” entry/invalidation, and “what would improve/invalidate this”;
- origin and whether the trader has liked, rejected, watched, or traded it;
- deduplication links to related signals from other detectors.

## 7. Implementation phases

### Phase 0 — Stabilize and capture the baseline

**Goal:** establish a known-good checkpoint before structural or strategy changes.

Tasks:

1. Resolve the mini-PC/theta mismatch. Confirm whether theta is now unconditional in `run_master` or whether the removed `include_theta` parameter was accidental. Align implementation and tests with one documented contract.
2. Replace the three deprecated Qt proxy invalidation calls with the supported filter-change API.
3. Run the full suite from a clean environment and record test count, duration, and warnings.
4. Create a tagged/checkpointed commit containing the current active work before starting refactors.
5. Add a deterministic smoke command for:
   - Qt app construction without external network calls;
   - one cached Master scan;
   - one headless autopilot scheduler tick;
   - one journal import/rebuild;
   - market-prep report generation.
6. Document the supported production entrypoints. Keep `launch_gui.py` and the headless scheduler; mark developer and legacy launchers explicitly.

Acceptance criteria:

- all tests pass with no unexpected warnings;
- smoke checks do not require live market hours;
- the mini-PC, manual Qt scan, and autopilot paths have the same documented theta/watchlist behavior;
- no active user changes are lost or overwritten.

### Phase 1 — Observability and performance baselines

**Goal:** make speed and unattended failures measurable before optimizing.

Primary files: `master_avwap_lib/runner.py`, `bounce_bot_lib/legacy.py`, `ui/services/autopilot_service.py`, `ui/services/scan_service.py`, and a new small diagnostics package.

Tasks:

1. Extend the existing phase timers into a structured run manifest.
2. Instrument provider calls, cache hits, bytes read/written, Drive write time, symbols processed, and per-symbol phase time.
3. Persist the last 30–90 run summaries locally and export a compact daily health summary to Drive.
4. Add an in-app Health page/status drawer showing:
   - last successful run per job;
   - next scheduled run;
   - IBKR status and last good data timestamp;
   - stale/failed symbols;
   - provider fallback counts;
   - shared-drive availability and sync-write errors;
   - orphan/running child processes;
   - queue depth and retry time.
5. Add repeatable benchmark fixtures using cached data at small, medium, and production-like universe sizes.
6. Record cold-cache and warm-cache baselines separately.
7. Add performance regression tests for pure hotspots. Avoid brittle whole-machine time assertions; test call counts, recomputation counts, and serialization volume.

Metrics to baseline:

- GUI time to usable window;
- Master scan total, p50/p95 symbol time, tracker time, earnings time, and report-write time;
- BounceBot full-cycle time and age of bars when each symbol is evaluated;
- market-prep duration by service;
- autopilot alert-to-report latency;
- tracker load/save size and duration;
- cache hit ratio and provider call count;
- skipped/failed/stale-symbol percentage.

Acceptance criteria:

- every production run writes one valid manifest even on failure;
- the phone report clearly says when data or a job is stale;
- the team can compare two commits using the same cached benchmark;
- no optimization is accepted without a measured before/after result.

### Phase 2 — One scheduler and a reliable autopilot supervisor

**Goal:** make unattended operation idempotent, restartable, and easy to diagnose.

Tasks:

1. Extract a pure `SessionPlan` that uses an exchange calendar, including holidays and early closes, and produces all due jobs for a market date.
2. Make the Qt scheduler, autopilot, and mini-PC entrypoint call the same scheduler engine.
3. Replace “slots done” JSON state with a local job ledger. Suggested columns:
   - job ID, market date, type, slot, attempt;
   - scheduled/start/end timestamps;
   - state, run ID, worker PID;
   - retry-after, error code, and message.
4. Make jobs idempotent by `(market_date, job_type, slot, config_hash)`.
5. Implement bounded retry policies by error class:
   - Drive unavailable;
   - IBKR disconnected/pacing;
   - provider rate limit;
   - stale/no market session;
   - bad/corrupt local state;
   - unexpected application failure.
6. Track spawned processes explicitly. On shutdown, cancel queued work, request graceful stop, wait for a bounded period, then terminate only owned children.
7. Move deferred theta enrichment into a separately tracked job instead of leaving it as an unowned tail after the success marker.
8. Add a heartbeat file that is atomically replaced and includes app version, machine ID, last tick, current job, last success, and next job.
9. Add a single-writer lease for shared mutable exports. A second machine may run read-only jobs but must not overwrite the active writer's state without lease expiry/takeover.
10. Keep a manual override: pause automation, skip next job, run now, retry failed job, and safely take over the writer lease.

Acceptance criteria:

- restarting during any job results in either a safe retry or a recorded completed job, never a duplicate silent write;
- early-close and holiday tests pass across DST boundaries;
- no child scan/enrichment process remains after a normal app shutdown;
- two simulated machines cannot overwrite the same mutable output concurrently;
- the phone report differentiates “no setups” from “scan failed,” “data stale,” and “job not run.”

### Phase 3 — Storage and cross-machine safety

**Goal:** reduce full-file rewrites, prevent sync conflicts, and keep a recoverable history.

Tasks:

1. Classify every file in `project_paths.py` as:
   - local cache;
   - local operational database/state;
   - shared immutable event/history;
   - shared latest-view export;
   - user-owned input/watchlist;
   - secret.
2. Move the journal SQLite database to the local application-data directory. Choose one synchronization design:
   - preferred for the current product: immutable per-machine import/event bundles plus a deterministic merge;
   - acceptable short term: one designated writer with lease and versioned backups;
   - future option: a real hosted database, only if remote access becomes necessary.
3. Migrate the large setup tracker from one mutable JSON blob to indexed tables or partitioned Parquet plus a small metadata database. Preserve current JSON/CSV exports during migration.
4. Store bars and feature snapshots in partitioned Parquet by timeframe/symbol/year. Update only affected partitions and retain provider/as-of metadata.
5. Add optimistic concurrency to watchlists and other user-owned files: read version/hash, merge changes by symbol, then atomically replace. Never erase hand-entered symbols because an older process wrote last.
6. Use the existing atomic-write pattern consistently for local settings, autopilot state, watchlists, scorecards, and reports.
7. Add schemas and migration versions to every durable structured store.
8. Add automatic rotating snapshots and a restore/repair tool for the tracker and journal.
9. Validate storage on startup without blocking module import. The UI should open into a degraded state and retry Drive connection in the background.
10. Store secrets through Windows Credential Manager/keyring. Environment variables remain supported. Migrate and remove plaintext tokens from `local_settings.json` after successful secure storage.

Acceptance criteria:

- no SQLite file is concurrently synced as a live multi-writer database;
- interrupted writes leave either the previous valid version or the new valid version;
- tracker updates scale with changed records/partitions rather than total history size;
- every migration is repeatable and backed up;
- credentials are masked in UI/logs and absent from ordinary settings exports;
- loss of Drive does not freeze import or GUI construction for up to 120 seconds.

### Phase 4 — Scanner performance and data-provider architecture

**Goal:** materially shorten scans while respecting IBKR/provider limits and keeping identical results.

Tasks:

1. Implement the provider interfaces already described in `docs/BROKER_ADAPTERS.md`:
   - daily bars;
   - intraday bars;
   - quotes;
   - option chains/quotes;
   - market calendar;
   - execution import.
2. Build one `BarRepository` with local L1 cache, durable L2 store, freshness policy, provider fallback, and provenance. Remove detector-specific fetch logic gradually.
3. Split Master scan into explicit stages:
   - resolve universe/watchlists;
   - load earnings/catalysts;
   - fetch/refresh bars;
   - compute shared features once per symbol/timeframe;
   - run pure detectors;
   - rank/gate/deduplicate;
   - update outcomes/tracker;
   - render/export views.
4. Batch yfinance requests and prefetch local/durable frames. Use limited parallelism only for independent local computation and providers that permit it.
5. Add an IBKR request coordinator with pacing budget, priority queue, deduplication, cancellation, retry/backoff, and request metrics. Priority order should favor SPY/regime, currently visible/focus symbols, triggered candidates, then background symbols.
6. Compute shared indicators once per symbol/timeframe. Detectors receive the same immutable feature frame instead of recalculating ATR, averages, weekly context, AVWAP bands, and relative strength.
7. Update only symbols whose input fingerprint changed. Fingerprint inputs should include latest bar, earnings/catalyst version, detector/config version, and relevant regime snapshot.
8. Separate slow option/theta enrichment from the core equity opportunity scan. Publish the equity results first, then merge theta results when complete.
9. Consolidate report generation around one normalized run result. Render text/CSV/JSON views from that result without reparsing output files.
10. Add a process-level snapshot service for the GUI. Panels subscribe to a loaded model instead of polling and parsing the same files independently.
11. Profile before changing algorithms. Focus first on provider waits, tracker serialization, repeated indicator work, and shared-drive I/O rather than micro-optimizing Python formatting.

Performance targets, to be finalized after Phase 1 baselines:

- at least 50% faster warm-cache Master scan at the same symbol count and identical normalized results;
- at least 70% fewer redundant provider calls on a repeated same-session scan;
- tracker save time proportional to changed partitions/records;
- visible/focus-symbol intraday data kept within one scan cycle of the provider;
- GUI interactions remain responsive while all jobs run.

Acceptance criteria:

- golden-result tests show no detector/rank changes from performance-only work;
- provider pacing and fallback behavior have deterministic tests;
- all data rows expose source, freshness, and completion status;
- benchmarks demonstrate the target improvements or document the actual limiting resource.

### Phase 5 — Point-in-time evidence and research rigor

**Goal:** ensure setup improvements are based on evidence that could have existed at the decision time.

Tasks:

1. Save immutable feature and signal snapshots before outcomes are calculated.
2. Build point-in-time universe membership history: listed status, price/liquidity filters, optionability when relevant, and manual/universe inclusion source as of each date.
3. Define one bar-adjustment policy per use case. Preserve raw provider fields and split/dividend metadata. Add tests around splits and missing sessions.
4. Create explicit outcome policies for:
   - intraday alert entry on next tradable bar;
   - VWAP/level touch entries;
   - end-of-day exits;
   - multi-day swing entries and expiries;
   - gaps through stops/targets;
   - same-bar target/stop ambiguity.
5. Include realistic, configurable spread/slippage/commission assumptions. Report gross and net results.
6. Implement expanding-window or rolling walk-forward evaluation. Training/tuning dates must always precede evaluation dates.
7. Purge/embargo overlapping swing episodes so adjacent signals do not leak nearly identical future paths across train/test boundaries.
8. Report uncertainty, not only point estimates:
   - sample and unique-symbol counts;
   - bootstrap confidence interval for average/median R;
   - Wilson interval for hit rate;
   - profit factor with a stability warning for small samples;
   - max drawdown and losing streak;
   - performance by year/quarter and market regime.
9. Control setup mining risk. Track how many variants were tested, use false-discovery controls or a strict holdout, and require a minimum out-of-sample sample size.
10. Add near-miss and random controls matched by date, side, liquidity, and regime. Compare incremental edge over the candidate-generation baseline, not just raw profitability.
11. Prevent silent detector drift. Every live signal stores detector/config versions; research reports compare like versions or explicitly migrate them.
12. Introduce a setup lifecycle:
   - `IDEA` — defined but not used for ranking;
   - `SHADOW` — emitted and tracked without promotion;
   - `CANDIDATE` — promising out of sample, visible in research;
   - `PROVEN` — allowed to influence ranking;
   - `DEGRADED` — recent evidence below guardrails;
   - `RETIRED` — preserved historically but no longer emitted.
13. Use champion/challenger configuration. A challenger runs in shadow on the same inputs until it passes a written promotion gate.

Suggested promotion gate for a setup/configuration:

- no look-ahead or point-in-time test failures;
- sufficient out-of-sample trades and unique symbols for its horizon;
- positive net expectancy with an uncertainty bound that is not materially negative;
- stable behavior in more than one time block and no unacceptable regime concentration;
- better than matched controls and the current champion on predeclared primary metrics;
- understandable trigger, invalidation, and failure modes;
- live shadow results directionally consistent with research.

Acceptance criteria:

- any ranked live setup can be traced from bars to features to detector version to outcome policy;
- rerunning a historical research window from pinned inputs gives identical results;
- tuning code cannot see evaluation-period outcomes;
- research UI/reports clearly separate in-sample, out-of-sample, and live-shadow performance.

### Phase 6 — Ranking, deduplication, and setup portfolio quality

**Goal:** find the best opportunities without flooding the trader with correlated versions of the same idea.

Tasks:

1. Separate **candidate generation** from **ranking policy**. Detectors answer “does this pattern exist?”; the ranker answers “how much attention does it deserve now?”
2. Log every rank component:
   - base detector evidence;
   - historical expected R and uncertainty;
   - current trend/regime fit;
   - relative strength and industry contribution;
   - catalyst/earnings risk;
   - liquidity/spread/data quality;
   - crowding/correlation penalty;
   - freshness/trigger proximity;
   - human feedback adjustment, if enabled.
3. Calibrate scores to observed probabilities/expected R using out-of-sample data rather than treating point totals as inherently comparable.
4. Add a correlation/crowding layer. Cap repeated exposure to the same sector, industry, catalyst, or highly correlated move in the top list.
5. Deduplicate multiple detectors on one symbol into one opportunity with supporting setup evidence. Preserve the underlying signal IDs.
6. Model opportunity lifecycle: developing, ready, triggered, extended/missed, invalidated, expired. Stop showing expired candidates as current opportunities.
7. Compare ranks against future outcomes using ranking metrics such as top-K expectancy, precision at K, recall of large movers, and regret versus the best available candidate.
8. Keep separate day and swing rankings. Do not force their scores, holding periods, or sample gates into one scale.
9. Treat trader feedback as a separate feature with strict time ordering. A “like” may identify the trader's style; it must not rewrite the historical bot score.
10. Add explicit reasons for hard blocks. A promising chart blocked for earnings, stale data, low liquidity, or poor reward/risk should remain inspectable.

Acceptance criteria:

- the top list is stable under harmless report ordering changes;
- one symbol appears once per horizon, with combined evidence;
- each rank is explainable numerically and in plain language;
- top-K out-of-sample performance is measured against the pre-ranking candidate pool;
- no rank uses feedback or outcomes recorded after the signal time.

### Phase 7 — New setup research program

**Goal:** expand setup coverage systematically rather than adding unmeasured rules to the live ranker.

All new setups begin in `SHADOW`, use the canonical event/outcome model, and must pass Phase 5 promotion rules.

#### Day-trade research candidates

1. **Opening range break plus relative strength.** Break of a configurable 5/15/30-minute range, abnormal relative volume, SPY/sector confirmation, and a first-pullback variant.
2. **Gap-and-go quality.** Gap size normalized by ATR, premarket/early-session volume, opening hold, sector sympathy, and distance from higher-timeframe resistance.
3. **First pullback after an opening drive.** Strong impulse, orderly volume contraction, VWAP/EMA/previous-breakout support, and resumption trigger.
4. **Relative-strength divergence during an index pullback.** A stock holds or advances while SPY/sector retraces, then triggers when the index stabilizes. Existing regime-pause work is a strong starting point.
5. **Failed breakout / failed breakdown reversal.** Liquidity sweep beyond a prior day or opening-range extreme followed by reclaim, with strict volume and market-context filters.
6. **High-relative-volume compression expansion.** Intraday volatility contracts after a strong move, then expands with directionally aligned volume and sector strength.
7. **Earnings reaction continuation.** Separate BMO/AMC handling, gap acceptance, first-hour behavior, and post-earnings liquidity; never mix it blindly with ordinary gap setups.
8. **Sector leader/laggard rotation.** Rank stocks against both SPY and their sector/industry over several intraday windows and study whether fresh leadership transitions predict follow-through.
9. **Prior-day level reclaim/rejection.** Previous high/low/close and high-volume/AVWAP levels with relative-strength and volume confirmation.
10. **Time-of-day specialization.** Evaluate open, late morning, lunch, and power hour independently rather than assuming a setup has one all-day edge.

#### Swing-trade research candidates

1. **Post-earnings drift and first constructive pullback.** Gap/reaction quality, estimate/news catalyst, volume signature, AVWAP/EMA support, and days since earnings.
2. **Volatility contraction pattern.** Multi-day range and volume contraction near highs, relative-strength persistence, and breakout/retest variants.
3. **52-week breakout and retest.** Distinguish first breakout, failed breakout, successful retest, and extended entries.
4. **Relative-strength new high before price.** Stock/sector relative-strength line makes a new high while price remains below its own high, followed by price confirmation.
5. **Anchored-VWAP confluence.** Earnings, major gap, high-volume day, and yearly anchors cluster near a trend support/reclaim zone.
6. **Weekly 8/10 EMA trend pullback.** Weekly trend quality plus daily compression and a defined daily trigger.
7. **Undercut and reclaim.** Undercut a prior swing low or key moving average, reclaim with volume/relative strength, and measure short- and medium-horizon outcomes.
8. **Industry acceleration.** A stock setup receives shadow evidence when its industry moves from weak/neutral to leading across multiple windows.
9. **Base breakout quality.** Base duration, depth, volatility contraction, volume dry-up, overhead supply, and relative-strength persistence.
10. **Failed swing setup as opposite-side information.** Study whether invalidated high-quality long/short setups create a measurable reversal signal.

#### Research priorities

Start with ideas closest to existing data and code:

1. relative-strength divergence during index pauses;
2. post-earnings first pullback/drift;
3. opening-range breakout plus first pullback;
4. volatility contraction/base breakout;
5. failed breakout/reclaim.

These reuse existing regime, RRS, earnings, AVWAP, volume, daily-bar, and outcome infrastructure and therefore produce evidence faster than entirely new data domains.

### Phase 8 — Journal as the learning center

**Goal:** connect what the bot saw, what the trader chose, and what actually happened.

Tasks:

1. Attach `signal_id`, `opportunity_id`, and `run_id` to watchlist actions, favorites, alerts, and journal auto-tag candidates.
2. Match imported executions to the nearest eligible pre-trade signal using symbol, side, time, setup, and account. Keep match confidence and allow correction.
3. Record non-trades:
   - watched and passed;
   - missed due to work/away;
   - rejected with reason;
   - entered manually but absent from bot output.
4. Add decision-time fields: thesis, planned entry/invalidation/target, confidence, and whether the trade followed the plan.
5. Keep outcome and behavior metrics separate. A good process can lose; a bad process can win.
6. Add daily/weekly review views:
   - best/worst executed trades;
   - best missed opportunities;
   - setup and regime performance;
   - entry timing/slippage versus signaled zone;
   - planned versus actual risk;
   - repeated mistake/rejection reasons;
   - human picks versus bot picks versus overlap.
7. Add screenshot/chart references without storing large image blobs in the database. Use managed local/shared asset paths with hashes.
8. Learn from tag corrections, but do not let corrections alter earlier model evaluations retroactively.
9. Export a compact AI-review bundle with redacted account identifiers and no credentials.
10. Add journal integrity checks: duplicate execution IDs, missing commissions, currency conversion gaps, unmatched closes, and impossible quantities.

Acceptance criteria:

- a closed trade can be traced to the signals available before entry;
- manual corrections are audited and survive rebuilds;
- bot, human, overlap, missed, and executed cohorts can be compared without hindsight relabeling;
- journal rebuilds are idempotent and preserve annotations;
- account and secret data are excluded from research/AI exports by default.

### Phase 9 — Opportunity-first desktop and phone experience

**Goal:** surface the best stocks continuously without forcing the trader to inspect many independent panels.

Tasks:

1. Make an **Opportunity Inbox** the default trading view. Separate Day and Swing lanes, with sections for Ready Now, Developing, Triggered, and Recently Invalidated.
2. Each row/card should show:
   - symbol and side;
   - setup and horizon;
   - current price/trigger distance;
   - final rank, expected R range, evidence count;
   - regime/sector/catalyst badges;
   - data age;
   - one-line reason and invalidation.
3. Selecting an opportunity should show the full evidence stack, rank breakdown, related alerts, setup history, chart levels, and journal history.
4. Add fast actions and keyboard shortcuts: focus, reject with reason, open chart, copy symbol, add note, and mark executed.
5. Add alert throttling and escalation:
   - alert only on meaningful lifecycle/rank changes;
   - deduplicate repeated detector messages;
   - escalate when developing becomes ready/triggered;
   - quiet low-confidence/stale candidates.
6. Keep specialized panels for research and diagnostics, but do not require them for the normal trading workflow.
7. Redesign `autopilot_today.txt` as a stable compact format:
   - last successful refresh and freshness warning at top;
   - top day opportunities;
   - top swing opportunities;
   - changes since the prior report;
   - active trigger/invalidation levels;
   - health/failure summary;
   - next scheduled update.
8. Write a small latest report plus dated immutable archives. This provides phone simplicity and an audit trail.
9. Load heavy pages lazily and parse shared data once through the snapshot service.
10. Add degraded-state UI. Missing IBKR, stale yfinance, missing Drive, and partial scans should be obvious without making the whole application unusable.

Acceptance criteria:

- the trader can identify the top current day and swing ideas within a few seconds;
- every displayed opportunity has visible data age and status;
- repeated scans do not create duplicate alert noise;
- phone output remains readable without horizontal scrolling and never presents stale data as current;
- UI remains responsive during scan, universe rebuild, market prep, and journal import.

### Phase 10 — Incremental codebase modernization

**Goal:** make future setup and performance work safer without a high-risk rewrite.

Tasks:

1. Do not rewrite the legacy cores wholesale. Extract one tested vertical slice at a time.
2. Suggested Master extraction order:
   - storage and schemas;
   - provider/bar repository;
   - shared features/indicators;
   - individual detector families;
   - ranking policy;
   - tracker/outcomes;
   - report renderers.
3. Suggested BounceBot extraction order:
   - broker request coordinator;
   - bar cache and scan-cycle state;
   - pure bounce detectors;
   - RRS/regime features;
   - alert lifecycle/deduplication;
   - outcomes/learning;
   - legacy GUI adapter.
4. Remove the duplicate `_priority_expected_r_text` definition and add an AST-based test preventing duplicate top-level definitions in production modules.
5. Replace broad exception suppression at service boundaries with typed errors and structured degraded results. Keep a final crash boundary around worker/UI entrypoints.
6. Add typed dataclasses/protocols at module boundaries. Use dictionaries only at serialization and compatibility edges.
7. Consolidate configuration into validated typed settings with provenance (`default`, config file, environment, secure store, UI override).
8. Add `pyproject.toml` with formatting/lint rules, pytest configuration, and gradual type-check scope.
9. Pin direct dependencies with a reproducible lock/constraints process and automate dependency update testing.
10. Add CI for Windows and a fast non-GUI platform where practical:
    - import/compile check;
    - lint;
    - unit tests;
    - headless Qt tests;
    - packaging smoke test;
    - dependency/security audit.
11. Mark tests by category (`unit`, `qt`, `integration`, `network`, `broker`, `slow`) so the fast suite runs on every change and external tests run intentionally.
12. Retire the Tk/PyQt UI only after a written parity checklist passes. Then remove unused GUI dependencies and compatibility code in one controlled phase.

Acceptance criteria:

- extracted modules have clear ownership and do not import UI code;
- the fast local test suite remains comfortably under one minute;
- all supported entrypoints are tested;
- CI produces a reproducible installer/smoke artifact;
- legacy line count decreases over time without a behavior-changing big-bang rewrite.

## 8. File-level first moves

This is the recommended initial implementation order after the current work is checkpointed.

| Order | Area | Initial change | Main files |
|---:|---|---|---|
| 1 | Baseline | Reconcile theta contract and Qt warnings | `master_avwap_mini_pc.py`, `master_avwap_lib/runner.py`, proxy models, failing tests |
| 2 | Diagnostics | Add structured scan manifest and benchmark fixtures | `master_avwap_lib/runner.py`, new `diagnostics/`, tests |
| 3 | Scheduler | Extract pure session plan and job ledger | `market_session.py`, `autopilot_core.py`, `ui/services/autopilot_service.py`, `master_avwap_mini_pc.py` |
| 4 | Process lifecycle | Own scan and theta worker processes through completion/cancel | `ui/services/scan_service.py`, `master_avwap_lib/runner.py` |
| 5 | Secrets/settings | Atomic settings and secure credential backend | `project_paths.py`, `journal_importers.py`, journal/market-prep settings UI |
| 6 | Storage safety | Localize journal DB; add immutable sync/export strategy | `journal_store.py`, `journal_runner.py`, `project_paths.py` |
| 7 | Data boundary | Introduce bar provider/repository interfaces | new provider package, current daily/intraday fetch helpers |
| 8 | Event identity | Add run/feature/signal/outcome IDs and schemas | tracker, bounce outcomes, focus tracking, journal |
| 9 | Performance | Stage Master scan and batch/prefetch data | `master_avwap_lib/runner.py`, provider repository |
| 10 | Product | Build normalized opportunity model/inbox/report | `ui/models`, `ui/services`, `ui/panels`, `autopilot_core.py` |

## 9. Testing strategy

### Unit tests

- pure setup detectors and mirror symmetry;
- feature calculations and incomplete-bar behavior;
- ranking component calculations and hard blocks;
- session plans across timezone, DST, holidays, and early closes;
- outcome policies, especially gaps and ambiguous bars;
- schema migrations and optimistic concurrency;
- secure-setting masking and migration.

### Characterization/golden tests

- run a fixed cached symbol set through current and extracted scanners;
- compare normalized signals, rank components, tracker updates, and reports;
- explicitly approve intentional differences rather than updating snapshots blindly.

### Integration tests

- fake IBKR/yfinance providers with pacing, timeout, partial data, and fallback cases;
- Drive disappears during read/write and later returns;
- two writer instances compete for a lease;
- app restart during each job stage;
- journal imports duplicate/rotated broker data;
- subprocess crashes before and after the completion marker.

### Research correctness tests

- feature snapshot cannot read bars after its as-of time;
- universe membership is resolved as of the signal date;
- tuning and evaluation windows never overlap improperly;
- detector/config versions are retained;
- outcomes cannot influence original rank fields;
- live and research detector implementations match on identical snapshots.

### Performance tests

- provider call count on warm repeat scans;
- indicator computation count per symbol/timeframe;
- tracker reads/writes per scan;
- GUI model loads once per changed snapshot;
- memory ceiling for production-like cached fixtures.

## 10. Operational and data-quality guardrails

1. Never label a run successful when required inputs are stale or missing.
2. Never substitute a provider silently; attach source and warning to the result.
3. Never overwrite user-owned watchlist changes from a stale in-memory copy.
4. Never tune a live score directly from the same outcomes used to report its performance.
5. Never change a detector without incrementing its version/config hash.
6. Never delete historical signals when a setup is retired.
7. Never put credentials, account numbers, or raw broker payloads in AI prompts or phone reports by default.
8. Never let report-generation failure erase the prior valid report; publish atomically.
9. Never treat a missing outcome as a loss or exclude it silently. Report coverage and reason.
10. Never allow decision-support work to grow into execution without an explicit separate project decision.

## 11. Success metrics

### Reliability

- scheduled-job completion rate;
- percentage of market sessions with a fresh phone report by each planned slot;
- mean recovery time after Drive/provider/IBKR failure;
- stale-data and partial-scan rate;
- duplicate or lost shared-write incidents;
- orphan process count.

### Speed

- cold/warm Master scan duration at fixed universe sizes;
- BounceBot cycle duration and bar age;
- GUI startup and page-switch latency;
- tracker update duration and bytes written;
- provider calls per processed symbol.

### Opportunity quality

- top-5/top-10 net expectancy by day and swing horizon;
- precision/recall for predefined large-move outcomes;
- expected-R calibration error;
- MFE captured and MAE by setup;
- performance stability by time block, regime, sector, and liquidity;
- fraction of top ideas invalidated, expired, or stale before actionable.

### Learning quality

- percentage of signals with complete immutable snapshots/outcomes;
- live-shadow versus research agreement;
- number of setup variants promoted, degraded, and retired under written gates;
- human-only, bot-only, and overlap cohort performance;
- journal-to-signal match coverage and correction rate.

### User usefulness

- time from alert to fresh phone report;
- alerts per session and duplicate-alert rate;
- percentage of alerts reviewed/focused/rejected/traded;
- best missed opportunity coverage;
- number of manual steps required for market prep and end-of-day review.

## 12. Recommended delivery sequence

The best practical sequence is:

1. **Weeks 1–2: trust the runtime.** Phase 0, structured manifests, health status, scheduler contract, and child-process ownership.
2. **Weeks 3–4: protect the data.** Storage classification, local journal DB, secure secrets, writer leases, atomic/versioned state.
3. **Weeks 5–7: make scans faster.** Provider repository, staged scan pipeline, batched fetches, shared features, incremental recompute, snapshot service.
4. **Weeks 8–10: make results scientifically defensible.** Canonical signal/outcome IDs, point-in-time datasets, walk-forward validation, setup lifecycle.
5. **Weeks 11–12: improve the daily experience.** Opportunity Inbox, deduplicated lifecycle alerts, clearer phone report, journal linking.
6. **Ongoing: research challengers.** Add new day/swing setups only through shadow tracking and promotion gates.
7. **Ongoing: extract legacy slices.** Refactor only where tests and measured bottlenecks justify it.

The time boxes are directional, not deadlines. Each phase should ship in small, green commits. Reliability, event identity, and point-in-time evidence should precede aggressive strategy tuning; otherwise the bot can become faster at producing results that cannot be trusted or reproduced.

## 13. Definition of done for the overall goal

TradingBotV3 reaches the intended product goal when:

- it can start before the session and run unattended through all planned jobs;
- a phone report always states what ran, how fresh it is, what failed, and the best current day/swing ideas;
- the desktop continuously presents a small, deduplicated, explainable opportunity queue;
- every opportunity is traceable to versioned data, features, detector logic, rank adjustments, and later outcome;
- setup research is point-in-time, out-of-sample, cost-aware, and protected from uncontrolled setup mining;
- human picks, bot picks, missed trades, and actual trades are linked in the journal without hindsight rewriting;
- shared data survives restarts, sync interruptions, and multiple machines without silent corruption;
- scan speed is measured and materially improved while golden results remain stable;
- the large legacy modules are shrinking incrementally behind tested service boundaries;
- the system remains decision support only and never executes trades.

## 14. Deep function and feature audit addendum

### 14.1 Audit scope

The production tree currently contains approximately:

- 166 Python modules under `scripts/` and `market_prep/`;
- 94,000 production lines;
- 3,555 functions/methods;
- 114 classes;
- 372 broad `except Exception` or bare-exception boundaries.

It would be counterproductive to write 3,555 isolated micro-suggestions. Most methods participate in a smaller number of stateful workflows, and optimizing one method without understanding its callers can make the bot less reliable. The deeper audit therefore traced functions by **feature pipeline**, then reviewed the hot/stateful functions individually. All production modules fall into the subsystem catalog in Section 17.

A runtime-data snapshot during this review explains why request/I/O architecture is urgent: shared lists were roughly 253 longs, 94 shorts, plus 25/25 bot picks; the two automation ownership stores disagreed substantially; the bounce-candidate CSV was roughly 214 MB; group-strength history was roughly 27 MB/310,000 rows; and environment history was roughly 4 MB. These are observations, not fixed limits, but they show that the current per-symbol sequential/request-and-full-file model has outgrown its original scale.

### 14.2 Most important revised conclusion

The requested SPY-pullback/RS behavior is already partially implemented four times:

1. `BounceBot.check_regime_pause_setups()` detects a pause and records names that defy it.
2. `BounceBot.entry_assist_auto_tick()` opens and closes automatic pullback/bounce measurement windows.
3. `BounceBot._build_spy_context_windows()` plus `_summarize_environment_scan()` scores historical intraday RS/RW behavior during weak/strong/compressed SPY bars.
4. `BounceBot.rank_window_movers()` powers the interactive RS Window over a manually or automatically selected SPY interval.

These should not remain separate ranking systems. They use overlapping data but different candidate pools, thresholds, scores, output formats, and lifecycle rules. The best next feature is a single **Market State + Relative Strength Engine** used by all four surfaces and by Auto Mode.

### 14.3 Current Auto Mode semantic conflicts

The current code has several reasonable behaviors that conflict when combined:

- `AutopilotService._start_watchlist_build()` replaces the Auto Pilot-owned part of `longs.txt` and `shorts.txt` after the open.
- `BounceBot._maybe_refresh_auto_populated_watchlists()` independently rotates another bot-owned part of those files every 30 minutes.
- `AutopilotService._maybe_add_near_extreme_names()` appends additional Master swing names during SPY pauses.
- BounceBot can remove symbols through its VWAP removal rules.
- Master AVWAP and focus/D1 watchlists add still more scan membership dynamically.
- when Auto Pilot is OFF, `_maybe_add_near_extreme_names()` and `_maybe_suggest_watchlists()` still write `autolongs.txt`/`autoshorts.txt` for measurement.

The individual ownership protections are thoughtful, but there are now two ownership models (`autopilot_written` and `auto_watchlist_membership.json`) plus direct append/remove operations. The correct fix is one `CandidateRegistry`/`WatchlistManager` with provenance and desired scan membership. Text files become exports/compatibility inputs, not the coordination database.

Auto Mode must have explicit semantics:

- **OFF:** no automatic user-facing list mutations, scheduled scans, or alerts. Optional shadow research may continue only when a separate “collect research while Auto is off” setting is enabled.
- **ON — Desk profile:** full automatic data/ranking/alerts and reports; desktop notifications are primary.
- **ON — Away profile:** identical decisions and candidate state; phone report cadence and notification filtering change. No strategy logic changes.

This removes the confusing situation where OFF still means several different kinds of automation remain active.

## 15. Central Auto Mode design

### 15.1 Application ownership

Create one application composition root, tentatively `TradingRuntime`, owned by `ui.app.MainWindow` rather than by a panel. It should own:

- `MarketDataCoordinator`;
- `MarketStateEngine`;
- `RelativeStrengthEngine`;
- `CandidateRegistry`;
- `OpportunityEngine`;
- `MasterScanService`;
- `IntradayScanService`;
- `MarketPrepService`;
- `JournalService`;
- `AutoModeCoordinator`;
- `ReportPublisher`;
- `HealthService`.

Panels receive these services. Closing or hiding a panel must not create, destroy, pause, or duplicate the trading runtime.

Today, `AutopilotPanel` constructs `AutopilotService`, `BouncePanel` owns `BounceService`, and other panels construct their own workers/timers. This panel-owned service pattern should be inverted incrementally.

### 15.2 Main-GUI layout

Auto Mode should be visible from every page:

- a persistent top-bar Auto toggle;
- `OFF`, `AUTO — DESK`, or `AUTO — AWAY` state;
- current SPY market-state badge and confidence;
- last completed 5-minute bar and data age;
- current job and next job;
- top long and short opportunity counts;
- degraded/failure badge;
- one click to open the full Auto dashboard.

The existing Auto Pilot page should become an **Auto Dashboard**, not mainly an activity-log/scheduler page. Recommended layout:

1. live SPY state and chart;
2. active pullback/bounce state machine;
3. strongest defiant longs and weakest defiant shorts;
4. opportunities that became Ready/Triggered recently;
5. current bot-managed scan pool and why each name is present;
6. scheduled work/health;
7. activity/debug log collapsed by default.

### 15.3 Event-driven Auto Mode

Replace the single 30-second `_tick()` method as the place where all policy decisions accumulate. Keep a lightweight heartbeat, but schedule explicit events:

- application started;
- shared storage available/unavailable;
- IBKR connected/disconnected;
- market session starting/open/closing/closed;
- new complete 5-minute bar;
- new daily bar;
- universe stale/refreshed;
- Master scan completed/failed;
- SPY state changed;
- pullback started/deepened/stabilized/resumed/failed;
- opportunity entered/exited a lifecycle stage;
- report publish due;
- outcome evaluation due.

Each event handler should be idempotent and persist its job/event ID. This will simplify `_tick()`, eliminate repeated polling work, and make failures recoverable.

## 16. SPY pullback and relative-strength engine

### 16.1 Desired behavior

On a strong bullish day, Auto Mode should detect when SPY begins a real pullback, continuously rank which stocks resist that pullback best, then distinguish between:

- a stock that was already strong and stays strong;
- a stock that gives back much less than expected;
- a stock that makes a new high while SPY pulls back;
- a stock that looks strong only because its sector is carrying it;
- a stock that is strong but too extended to offer a useful entry;
- a stock whose strength disappears before SPY stabilizes;
- a stock that confirms when SPY resumes.

For bearish days, use the exact signed inverse: detect an SPY bounce, find stocks that remain weakest or make new lows, and require downside confirmation when SPY rolls back over.

The system must emit two different things:

- **screening evidence:** “this stock is defying SPY”; useful for ranking and research;
- **entry-ready event:** a later price/market-state confirmation with a defined invalidation.

The current code correctly began separating those concepts by making regime-pause batches informational. The new engine should finish that separation.

### 16.2 One side-symmetric coordinate system

Avoid maintaining separate long and short algorithms. For every directional feature, use `side_sign`:

- `+1` for long/RS;
- `-1` for short/RW.

Convert raw movement into favorable-direction movement:

```text
aligned_return       = side_sign * raw_return
aligned_residual     = side_sign * (stock_return - expected_return)
aligned_spy_impulse  = side_sign * spy_impulse
aligned_sector_move  = side_sign * sector_return
aligned_extreme_gain = side_sign * change_in_session_extreme
```

All ranking, thresholds, lifecycle transitions, and tests operate on the aligned values. Only labels and price-level presentation branch by side. Add mirror/property tests that negate/mirror bars and require identical decisions with the opposite side.

### 16.3 Market-state inputs

Use only complete bars for state transitions. Maintain both current observation and last-complete-bar time.

Required SPY features:

- open-to-now return and return versus prior close;
- SPY return normalized by daily ATR and rolling intraday realized volatility;
- session VWAP and VWAP deviation bands;
- distance from VWAP, opening range, prior high/low, and current HOD/LOD;
- slope over 5, 15, and 30 minutes;
- pullback/bounce depth from the most recent impulse extreme;
- number and duration of countertrend bars;
- volume expansion/contraction during impulse and pullback;
- percentage of the session above/below VWAP and bands, both full-session and recent-window;
- sector ETF participation/breadth using already cached sector bars;
- data freshness and missing-bar coverage.

Optional later inputs:

- market breadth/advance-decline data from a reliable provider;
- QQQ/IWM/DIA confirmation;
- volatility-index context;
- internals such as TICK, only if a dependable source becomes available.

The first implementation should not wait for optional data.

### 16.4 Market-state machine

Replace the binary `_detect_spy_pause_start()` result with a pure, versioned state machine:

```text
PREOPEN
  -> OPENING_DISCOVERY
  -> BULL_IMPULSE / BEAR_IMPULSE / RANGE
  -> COUNTERMOVE_ARMED
  -> COUNTERMOVE_ACTIVE
  -> STABILIZING
  -> TREND_RESUMED
  -> RANGE or REGIME_FAILED
```

Suggested bullish logic, inverted by `side_sign` for bearish logic:

1. **BULL_IMPULSE:** SPY has a meaningful aligned move, is structurally above VWAP/prior close, and the state persists for a minimum number of complete bars.
2. **COUNTERMOVE_ARMED:** rate of change weakens, a lower close/opposite candle occurs, or no new high occurs for a configured number of bars.
3. **COUNTERMOVE_ACTIVE:** retracement exceeds a noise floor measured in SPY ATR/realized volatility, lasts at least one complete bar, and remains below a maximum depth/duration.
4. **STABILIZING:** downside slope flattens, a higher low/micro reclaim appears, volume contracts, or the counter-move fails to extend.
5. **TREND_RESUMED:** SPY breaks the stabilization pivot or the prior micro high in the original direction.
6. **REGIME_FAILED:** SPY loses a structural boundary such as VWAP/opening range/prior close, retraces too much of the impulse, or the market regime changes.

All thresholds must be configuration values recorded with the signal. Initial values should be conservative and then tuned through shadow outcomes. A single red/green candle should arm a counter-move, not automatically prove one.

Use hysteresis/debounce so a VWAP-band percentage near 60% cannot flip the market repeatedly on successive bars.

### 16.5 Pullback episode record

Persist one immutable episode record plus append-only updates:

- episode ID and detector version;
- direction;
- impulse start/extreme;
- armed/active/stabilizing/resumed/failed timestamps;
- SPY prices, VWAP, ATR/volatility, depth, duration, and volume at each transition;
- regime and confidence at start/end;
- symbols evaluated and their rank snapshots;
- data coverage/freshness;
- eventual SPY continuation/failure outcome.

This makes “stocks strong during SPY pullbacks” a measurable setup rather than an ephemeral alert.

### 16.6 Candidate universe: broad screen then live focus

The current pause sweep ranks only `self.longs` or `self.shorts`; Entry Assist uses manual plus auto lists; the environment scan uses the BounceBot scan set; the RS Window uses Entry Assist candidates. This inconsistency can hide the best stock before it reaches a watchlist.

Use a two-stage funnel:

1. **Broad universe stage:** use the existing batch yfinance 5-minute refresh and durable daily context to score the entire screened universe cheaply every 5–15 minutes.
2. **Live focus stage:** promote the best candidates plus user Focus Picks and active Master setups into a capped IBKR pool refreshed every complete 5-minute bar.

The broad stage should never overwrite user watchlists. It writes candidate/opportunity records. The live-stage membership is derived from those records and exposed in the UI.

Suggested live-pool priority:

1. active/triggered opportunities;
2. human Focus Picks;
3. top pullback-defiance candidates;
4. current Master favorite/near-favorite setups;
5. bot shadow candidates needed for outcomes;
6. lower-ranked background names on a slower cadence.

### 16.7 Stock features during the counter-move

For every aligned stock/SPY window, compute:

- stock return over the exact aligned timestamps;
- SPY return over those timestamps;
- sector/industry reference return;
- rolling beta-adjusted expected stock return;
- residual return versus SPY and versus sector;
- RRS normalized by stock and SPY volatility;
- stock giveback from its pre-pullback extreme;
- giveback as a fraction of its preceding impulse;
- HOD/LOD distance and whether it made a new favorable extreme;
- percentage of counter-move bars closing in the favorable direction;
- VWAP/EMA/opening-range/previous-level hold status;
- relative volume during the original move and counter-move;
- spread/liquidity/data coverage;
- 5/15/30/60-minute RS/RW persistence;
- daily/weekly and Master setup context;
- catalyst/earnings risk and sector crowding;
- whether the candidate is extended beyond a configurable ATR/ADR distance from a valid entry reference.

Do not use simple raw percent excess as the only measure. A 1% move means something different for a low-volatility mega-cap and a high-volatility small-cap.

### 16.8 Ranking model

For the first version, use a transparent percentile/composite score, not a black-box model and not hundreds of unexplained magic points.

Suggested components:

- 30% beta/volatility-adjusted residual during the SPY counter-move;
- 20% giveback resistance / favorable new extreme;
- 15% multi-window RRS persistence;
- 10% sector/industry residual;
- 10% volume/liquidity quality;
- 10% higher-timeframe setup quality and expected R;
- 5% trigger proximity/freshness;
- explicit penalties for extension, stale/missing bars, imminent earnings, spread, and correlated crowding.

Weights are hypotheses. Store every component, track outcomes, and calibrate them out of sample. Use cross-sectional percentiles within a liquidity/volatility cohort so one raw scale does not dominate.

Recommended tiers:

- **Defiant:** favorable/flat while SPY counter-moves and top residual percentile;
- **Holding:** pulls back materially less than expected and preserves structure;
- **Watching:** positive evidence but insufficient duration/coverage;
- **Fading:** lost relative strength during the episode;
- **Ready:** Defiant/Holding plus SPY stabilization and stock micro-trigger;
- **Invalid:** stock or SPY structure failed.

### 16.9 Entry confirmation and invalidation

Do not alert every Defiant stock as an entry. A Ready event should require:

- SPY state is stabilizing or trend-resumed, unless studying the more aggressive early-entry variant;
- stock remains above/below its valid structure in the aligned direction;
- stock breaks a counter-move micro pivot, reclaims VWAP/EMA, or makes a fresh favorable extreme;
- reward/risk to the next obstacle is acceptable;
- data is fresh and the relevant bar is complete;
- the opportunity is not already extended or expired.

Persist both the early Defiant observation and the later Ready trigger so their outcomes can be compared.

### 16.10 Alerts and presentation

One pullback episode should generate at most:

1. one state alert: “SPY bullish pullback active”;
2. one ranked batch update when the top set changes materially;
3. per-symbol Ready alerts only on lifecycle transition;
4. one resume/failure summary.

The Auto dashboard should show a continuously updating table without turning every refresh into an alert. The Drive report should show:

- SPY episode state/depth/duration;
- top Defiant/Holding stocks with residual and HOD/LOD status;
- Ready names with trigger/invalidation;
- names that faded or invalidated since the last report;
- freshness and next update.

## 17. Function-level disposition for the central intraday path

| Existing function | Finding | Recommended disposition |
|---|---|---|
| `real_relative_strength()` | Useful ATR-normalized foundation but endpoint-sensitive and returns only two scalars | Keep compatibility wrapper; add a typed multi-window feature result with aligned coverage, beta/sector residual, slope, and completed-bar metadata |
| `_spy_vwap_regime_stats()` | Full-session fraction is useful but reacts slowly and can flip at one hard threshold | Extract pure `MarketStateFeatures`; add recent-window stats, confidence, hysteresis, and data-quality flags |
| `_classify_spy_vwap_regime()` | Four labels are too coarse for pullback lifecycle | Retain as a display compatibility mapping derived from the richer state |
| `update_auto_market_environment()` | Correct responsibility, but coupled to bot state and manual override | Move decision logic to pure engine; service applies/publishes changes |
| `get_auto_regime_reading()` | Good read-only UI contract | Return the central immutable market snapshot rather than recomputing a parallel view |
| `_maybe_refresh_auto_regime_while_paused()` | Keeps data alive, but scanning pause and market-data pause are conflated | MarketDataCoordinator keeps SPY alive independently; Auto/scan state only controls downstream work |
| `_detect_spy_pause_start()` | Any opposite candle can trigger; returns only a datetime; no depth/confidence/failure state | Replace with the episode state machine; keep a temporary adapter for tests/UI |
| `_window_change_pct()` | Compares first available opens even when SPY/stock timestamps differ | Align exact complete timestamps, report coverage, and use close-to-close or explicitly versioned entry policy |
| `check_regime_pause_setups()` | Desired idea, but chooses one side from a coarse regime and can treat neutral as long | Consume central episode events; skip or separately study neutral; evaluate both sides for research but promote only aligned side |
| `_sweep_regime_pause_bangers()` | Good evidence capture; pool and raw-percent scoring are limited | Replace with cross-sectional defiance ranking over live focus pool; preserve observation/outcome hooks |
| `_record_regime_pause_banger()` | Correctly separates tracking from loud alerts | Generalize to canonical signal event; retain compatibility export |
| `_record_regime_pause_observation()` | Counts episodes but stores limited context | Store episode ID, rank/components, timestamps, feature version, and outcome coverage |
| `_emit_regime_pause_summary()` | Batch summary is the right alert shape | Emit only material top-set/lifecycle changes through OpportunityEngine |
| `start_entry_window()` / `end_entry_window()` | Useful manual tool but maintains a second window state | Manual action creates/pins an episode view in the central engine; automatic episodes use same record |
| `_rank_entry_window_side()` | Simple and explainable, but only raw percent excess and narrow pool | Delegate to RelativeStrengthEngine; keep raw excess as one displayed component |
| `_rank_trailing_movers()` | Useful fallback in chop | Add volatility/sector adjustment, coverage, and one cached multi-window calculation |
| `rank_window_movers()` | Excellent research/exploration surface | Make it call the same feature/rank engine for arbitrary intervals; allow full-universe cached snapshots |
| `entry_assist_board_snapshot()` | Correct consolidated UI concept but recomputes pieces from bot internals | Convert to serialization of central immutable snapshot; no direct algorithm calls from GUI timer |
| `entry_assist_auto_tick()` | Implements the wanted automation but duplicates pause state | Remove policy after central state machine is live; keep adapter during migration |
| `_build_spy_context_windows()` | Valuable bar-by-bar context history | Reuse state-machine labels per bar and avoid separate threshold semantics |
| `_summarize_environment_scan()` | Captures persistence, but has large hard-coded point weights and repeated sorting | Store raw aggregates; rank through versioned configurable policy; validate weights out of sample |
| `_record_environment_focus_history()` | Good learning trail but rewrites JSON frequently | Append canonical observations/outcomes; export compact daily view periodically |
| `run_rrs_scan()` | Does too much: fetch, aggregate, classify, score, group, history, GUI | Split fetch/features/rank/publish; calculate all requested timeframes in one pass |
| `_get_cached_bars()` | Prevents duplicate calls within a cycle but has no explicit freshness contract | BarRepository returns frame plus freshness/source/completeness; incremental refresh/subscription |
| `request_historical_bars()` | Synchronous wait per request, weak cancellation, timeout can leave provider work alive | Central IBKR coordinator with cancellation, priority, pacing, metrics, and bounded concurrency |
| `_prune_latest_bars_for_cycle()` | Clearing the cache forces repeated full-history downloads | Retain immutable history and update only deltas; mark freshness instead of deleting good frames |
| `build_atr_cache()` | Sequential per-symbol IBKR daily requests can dominate cold starts | Read ATR from durable daily frames; batch missing data via provider repository; compute once per new daily bar |
| `wait_for_candle_close()` | Blocking one-second loop cannot be cancelled and may wait a full cycle at a boundary | Replace with scheduler/stop-event wait to next complete bar plus small provider-settle delay |
| `BounceBot.run_strategy()` | Central loop is understandable but monolithic and `while True` has no stop event | Add lifecycle stop token immediately; then split one cycle into named stages with metrics and isolated failure policy |
| `BounceService.start()` / `stop()` | `stop()` disconnects the bot but the strategy loop can continue/reconnect; board timer is not stopped | P0 fix: real `bot.stop()`, join owned threads, stop all timers, clear snapshots, test repeated start/stop |
| `AutopilotService._tick()` | Too many policies and polling duties in one broad try/except | Thin heartbeat around event/job coordinator; structured failure per job |
| `_maybe_add_near_extreme_names()` | Similar to pause engine but checks only top swing rows through a second yfinance fetch | Fold near-extreme evidence into the central pullback rank; do not directly append files |
| `_start_watchlist_build()` | Good batch open scan; direct list replacement conflicts with later rotation | Write candidate records; CandidateRegistry derives bot scan pool and preserves provenance |
| `_maybe_suggest_watchlists()` | Valuable shadow measurement, but Auto OFF semantics are unclear | Gate under explicit shadow-research setting; never imply OFF is fully inactive while writing silently |
| `score_autopilot_picks()` | Conditions performance on picks that later created confirmed Bounce events | Add outcomes for every pick and every ranked-but-not-selected control; keep alert-conversion metrics separately |
| `write_watchlist_file()` / `append_watchlist_symbols()` | Simple but non-atomic and vulnerable to stale-writer loss | One atomic, locked/version-aware WatchlistManager; text files are generated views |
| `render_away_report()` / `write_away_report()` | Useful phone surface; currently list/log oriented and direct-written | Render normalized opportunities and health; archive versions; atomic publish; show freshness prominently |
| `get_previous_day_extremes()` | Contains a latent `datetimde` typo | Fix under Phase 0 with a regression test; audit low-use legacy helpers for similar dormant defects |
| `RequestQueue` | Defined but apparently not used by BounceBot requests | Remove after verifying callers or replace with the real shared request coordinator; do not maintain dead infrastructure |

Additional requirements for these dispositions:

- all stock-versus-SPY move ratios must use the same aligned timestamps; do not align for RRS and then calculate a companion move ratio on the unaligned stock series;
- side scoring must never use `abs(...)` where sign carries RS versus RW meaning;
- `RequestQueue`, if retained temporarily, must not call `process_queue()` while holding the same non-reentrant lock—its current shape would deadlock if actually used;
- manual Entry Assist actions must use cached snapshots or managed workers, not potentially blocking IBKR requests on the Qt thread;
- broad opening ranking must fail explicitly when SPY is missing/stale rather than substituting a zero benchmark move;
- daily-context cache must be replaced/reset atomically on a new market date; updating an old dictionary after a partial reload can retain yesterday's contexts.

## 18. Feature-by-feature improvement catalog

### 18.1 Main Qt shell and Trading Desk

Relevant modules: `launch_gui.py`, `scripts/gui.py`, `ui/app.py`, `ui/panels/trading_desk.py`, theme/state modules.

Improvements:

- make `ui.app` the one production composition root;
- move Auto Mode toggle/state into the persistent shell;
- construct services before panels and inject them;
- lazily construct Research, Journal, and other heavy pages after first selection;
- persist window geometry, selected page, table layouts, and filters in one batched settings write;
- replace multiple direct `get_local_setting()` reads with a cached typed settings object;
- show a single global data-as-of time and degraded-state indicator;
- make shutdown await/cancel runtime jobs and report failures instead of swallowing all panel exceptions;
- ensure one widget is never ambiguously owned/reparented by multiple stacked layouts;
- remove the old Tk/PyQt launch path after parity, reducing import and packaging cost.

### 18.2 BounceBot setup detection

Relevant modules: `bounce_bot_lib/legacy.py`, `learning.py`, feedback/alerts/RRS bridges.

Keep and improve:

- VWAP, dynamic VWAP, EOD VWAP, EMA, ORB, H1 color, impulse, AVWAP focus, and regime-pause detectors should become pure detector functions over one shared symbol snapshot;
- calculate VWAP/EMA/ATR/relative-volume/session structure once, then pass features to every enabled detector;
- replace per-detector DataFrame preparation with one prepared frame;
- attach exact detector version, bar-completeness, input IDs, and rejection reasons;
- treat “approaching,” “candidate,” “confirmed,” and “expired” as one lifecycle rather than unrelated messages;
- add cooldowns by signal identity/price zone instead of symbol-only sets;
- preserve near misses with sampled controls, not every rejected evaluation;
- evaluate long/short mirror symmetry automatically;
- replace hard-coded constants with typed versioned config and UI-safe ranges;
- keep learning adjustments out of raw detector truth: detector fires first, ranking/muting policy applies afterward.

Efficiency:

- one 5-minute snapshot per symbol per bar;
- one multi-timeframe aggregation cache;
- one sector/industry classification lookup;
- one outcome update pass;
- only priority/live names receive every-bar IBKR refresh;
- batch broad-universe discovery independently.

### 18.3 Master AVWAP and swing scanning

Relevant modules: `master_avwap_lib/runner.py`, `legacy.py`, `levels.py`, `expected_r.py`, tracker, theta, outputs.

Improvements:

- split the sequential per-symbol loop into bar acquisition, shared feature computation, pure setup evaluation, ranking, tracker, and export stages;
- prefetch/batch missing daily bars and use limited parallelism for local feature computation;
- use one canonical `SymbolDailySnapshot` for ATR, trend, SMA/EMA, weekly, AVWAP, earnings, high-volume levels, industry, and regime features;
- prevent report builders from reaching back into storage/provider code;
- record every score adjustment as a named component and remove duplicated score mutations;
- resolve the duplicate `_priority_expected_r_text` definition;
- make hard blocks, score caps, expected-R adjustments, and final bucket assignment one ordered policy pipeline;
- separate equity scan completion from theta enrichment;
- update only changed tracker partitions instead of a large JSON document;
- link every swing row to the same Opportunity model used by intraday Auto Mode;
- add freshness and input coverage to the GUI; stale Master rows must never look live;
- treat regime-pause evidence as one feature with measured incremental value, not an automatic additive bonus forever.

### 18.4 Setup Tracker and scoring tuner

Relevant modules: tracker functions in Master legacy, `analyze_master_avwap_scoring.py`, tracker UI/models.

Improvements:

- move from mutable setup dictionaries to versioned setup/snapshot/outcome tables;
- distinguish first observation, first actionable trigger, and later upgraded rank;
- compute outcomes from immutable entry policies;
- retain delisted/removed symbols and missing-data reasons;
- show sample count, unique symbols, time periods, confidence intervals, and coverage beside WR/PF;
- avoid multiple correlated setups on one symbol/date dominating evidence;
- use time-decay only as a reported/configured policy and compare it with non-decayed results;
- tuner writes recommendations, never silently activates them from the same dataset;
- require walk-forward/holdout validation before config promotion;
- display score contribution ablation: what ranking would be without each feature family;
- compact/archive closed records by partition, not by deleting evidence.

### 18.5 Playbook study and move forensics

Relevant modules: `setup_playbook_study.py`, `move_forensics.py`, `setup_docs.py`.

Improvements:

- share live detector implementations or a single feature contract so research cannot drift;
- label detector/config version in every episode;
- enforce point-in-time universe, earnings, and industry context;
- include entry timing, ambiguous stop/target bars, gaps, slippage, and costs;
- use walk-forward periods and matched controls;
- account for multiple setup variants tested;
- report performance concentration by symbol/year/regime;
- use `move_forensics` only to generate hypotheses until a separate forward/holdout test confirms them;
- show setup overlap so ten similar detectors do not masquerade as ten independent edges;
- add an automatically generated “failure anatomy” report: common conditions before large losses or missed large moves.

### 18.6 Universe builder and industry scanner

Relevant modules: `universe_builder.py`, `industry_scanner.py`, RS-window industry feed.

Improvements:

- maintain point-in-time universe membership snapshots;
- record why each symbol passed/failed each screen;
- batch metadata and avoid per-symbol `fast_info` calls where possible;
- separate stable daily universe build from fast intraday candidate ranking;
- use dollar volume, price, spread proxy, data completeness, and optionality as explicit fields;
- make custom groups and classifications versioned and editable in the GUI;
- compute sector/industry boards once per daily/intraday snapshot and share them with Master, BounceBot, Market Prep, and RS Window;
- expose breadth, acceleration, persistence, and leadership-change features, not just current RS;
- flag small groups where median/score is unstable;
- cache by input fingerprints and write outputs atomically.

### 18.7 Focus Picks, pick feedback, and human tracking

Relevant modules: `focus_picks.py`, `human_focus_tracking.py`, `pick_feedback.py`, Focus UI/service/feed.

Improvements:

- record a stable pick ID, decision timestamp, observed price, source opportunity/alert, side, category, and rank snapshot immediately;
- avoid using the later daily close as an implicit decision entry for intraday picks;
- preserve add/remove/category history rather than only current membership;
- distinguish “interested,” “planned,” “entered,” “passed,” “missed,” and “invalidated”;
- require a reason taxonomy for dislike/pass but allow free text;
- compare human, bot, overlap, and randomized/matched cohorts at identical timestamps;
- never feed later human feedback into the original historical bot score;
- use feedback to personalize presentation/ranking only after minimum evidence and with an off switch;
- surface recurring preferences/mistakes in the journal weekly review.

### 18.8 Alert Center and opportunity lifecycle

Relevant modules: alert center, alert models/widgets, focus routing, setup detail.

Improvements:

- stop parsing importance primarily from formatted text; emit typed `AlertEvent` fields;
- centralize tier, sound, favorite, dedupe, and feed-gate policy;
- dedupe by signal/lifecycle transition, not raw string;
- keep status/health messages out of the opportunity alert feed;
- show first-seen/current/last-changed times and data age;
- group supporting detectors under one opportunity;
- preserve dismissed/rejected state across refresh/restart;
- add snooze and “alert only when Ready” controls;
- measure alert volume, review rate, time-to-review, and false/stale-alert rate;
- maintain separate audit history even when the visible feed is capped.

### 18.9 RS Window and Entry Assist

Relevant modules: `rs_window_panel.py`, `rs_window_feed.py`, `spy_m5_chart.py`, `entry_assist_board.py`.

Improvements:

- keep the draggable historical-window analysis—it is useful for research and trader intuition;
- use the same central engine for selected and live windows;
- shade detected SPY impulse/counter-move/stabilization episodes on the chart;
- show rank change from pullback start to now;
- show aligned data coverage and exclude misleading missing-bar comparisons;
- include residual versus sector, volatility-adjusted residual, giveback ratio, HOD/LOD distance, volume, setup tier, and extension risk;
- add “pin this episode,” “promote to Focus,” and “why ranked here” actions;
- update the model only when snapshot ID changes, not every timer tick;
- move daily strength/industry joins to background snapshot construction;
- retain historical selected-window analyses as research annotations when requested.

### 18.10 Market Prep

Relevant modules: `market_prep/orchestrator.py`, services, report builder, Qt market-prep/ticker-lookup panels.

Improvements:

- run independent network services concurrently under bounded timeouts;
- create one source-result schema with `ok/stale/partial/error`, fetched-at, effective date, coverage, and cache age;
- share earnings, prices, metadata, sector boards, and calendar data with scanners through repositories instead of fetching separately;
- add authoritative exchange holiday/early-close calendar to all scheduling;
- deduplicate and cluster headlines by underlying event;
- prioritize changes since last prep rather than repeating the entire report;
- score source confidence separately from event importance;
- make AI text a cited summary of deterministic data, never an untraceable ranking input;
- cache prompts/responses by source snapshot hash;
- simplify the 2,500-line report builder into section renderers over typed view models;
- publish compact “today risks/catalysts/no-trade windows” into the central Auto dashboard and opportunities.

### 18.11 Ticker Lookup

Relevant modules: ticker lookup service/panels/feed.

Improvements:

- execute independent SEC/news/earnings/price/peer calls concurrently;
- show source age and failure per section;
- reuse central metadata/catalyst stores;
- cache by ticker plus source version;
- separate deterministic risk facts from heuristic verdict;
- connect lookup results to the selected Opportunity and journal notes;
- allow background prefetch for top opportunities;
- prevent speculative/low-signal headline heuristics from hiding raw evidence entirely.

### 18.12 Journal and broker imports

Relevant modules: `journal_store.py`, analytics/importers/runner/walkaway, Qt journal UI.

Improvements:

- move live SQLite off the synced Drive directory;
- enable appropriate local SQLite WAL/busy timeout and explicit migrations;
- replace full delete/rebuild of all trades with incremental rebuild by affected account/symbol/time range;
- preserve annotations and manual corrections under rebuild using stable IDs;
- store broker tokens in OS credential storage;
- attach signal/opportunity/pick IDs automatically;
- capture missed/passed/unexecuted opportunities, not only fills;
- separate trade outcome from process adherence;
- add execution-to-plan slippage, entry timing, MAE/MFE, scale-in/out, and opportunity-cost views;
- validate duplicate IDs, currencies, commissions, corporate actions, options multipliers, and unmatched positions;
- create redacted AI/research exports by default.

### 18.13 Storage, settings, and Google Drive

Relevant modules: `project_paths.py`, watchlist utilities, caches, every direct CSV/JSON writer.

Improvements:

- remove drive waiting and directory mutation from module import;
- introduce a storage service with readiness/degraded signals;
- cache local settings in memory and batch/atomically write changes—`UiState.save()` currently causes repeated read/write cycles;
- classify files as local cache, local DB, shared event, shared latest export, user input, or secret;
- apply atomic write plus version/hash/lease where multiple writers exist;
- create per-machine immutable event files and deterministic merge rather than syncing live databases;
- put schema version and producer version in durable files;
- retain last-known-good report/snapshot and publish a visible stale marker if refresh fails;
- add backup/restore/integrity UI;
- measure Drive write latency/conflict incidents.

### 18.14 Process, threading, and shutdown

Relevant modules: `BounceService`, `ScanService`, Autopilot workers, Qt worker services, launcher/shutdown code.

Improvements:

- every long-lived loop receives a stop event;
- every service owns and joins its threads/processes;
- use Qt thread-pool/worker abstractions or a shared executor rather than many unmanaged daemon threads;
- never mutate Qt widgets from worker threads;
- cancel IBKR requests on timeout;
- stop all service timers together;
- split scan success from deferred theta completion rather than abandoning a marked child process;
- store process handle/PID/run ID and reap it;
- bound task queues and coalesce duplicate refresh requests;
- propagate structured cancellation versus failure;
- test start/stop/restart and close-during-each-operation repeatedly.

### 18.15 Dependencies, packaging, and quality

Relevant files: requirements files, packaging notes, tests, compatibility launchers.

Improvements:

- pin direct dependencies and generate constraints/lock data;
- add `pyproject.toml` for pytest/lint/type/build configuration;
- add Ruff/formatter and a gradual type-check boundary;
- add Windows CI and headless Qt smoke tests;
- mark network/broker/slow tests;
- add coverage reporting by subsystem, with emphasis on scheduler/state/storage rather than a vanity global target;
- add static checks for duplicate definitions, unreachable/dead compatibility helpers, direct non-atomic shared writes, and forbidden UI-to-broker imports;
- remove unused `RequestQueue`-style infrastructure after call-graph verification;
- retire legacy GUI stacks after parity;
- package only the main GUI plus optional headless host.

## 19. Highest-return efficiency plan

Implement in this order because it reduces both runtime and code complexity:

1. **One complete-bar cache:** never request the same symbol/timeframe dataset four times in one cycle.
2. **One multi-timeframe RRS pass:** fetch 5-minute bars once, aggregate 15m/1h once, calculate profiles once, then publish all views.
3. **Durable ATR/daily features:** eliminate sequential IBKR ATR cold-start requests when daily frames already exist.
4. **Incremental bar refresh:** retain history and append/replace the latest complete bar instead of clearing caches and downloading five days repeatedly.
5. **Two-stage universe funnel:** batch broad yfinance scoring, cap the high-frequency IBKR pool.
6. **One classification/group cache:** never write classification CSV once per new symbol; batch and flush once.
7. **One market snapshot:** SPY regime, pullback, RRS, Entry Assist, and RS Window consume it.
8. **One candidate registry:** remove competing file writers and repeated file reads.
9. **One opportunity snapshot for UI/report:** panels do not reparse outputs independently.
10. **Incremental tracker/journal storage:** update changed records/partitions rather than whole histories.
11. **Concurrent market prep:** bounded parallel calls for independent sources.
12. **Structured profiling:** verify provider wait, computation, and Drive I/O improvements after each packet.

Do not start with low-level pandas/numpy micro-optimizations. Network waits, repeated provider requests, whole-file serialization, and duplicated pipelines are much larger opportunities.

## 20. Claude-ready implementation packets

Each packet should be a separate branch/commit series with tests green before the next packet. Claude should read this plan and the named tests before editing.

### Packet A — Runtime ownership and truthful Auto semantics

Goal: main GUI owns one runtime; Auto OFF/Desk/Away behavior is explicit.

Primary files:

- `ui/app.py`
- `ui/panels/autopilot_panel.py`
- `ui/services/autopilot_service.py`
- `ui/services/bounce_service.py`
- `ui/panels/bounce_panel.py`
- new runtime/coordinator modules

Work:

1. Add a real stop event and join path to BounceBot.
2. Fix `BounceService.stop()` to stop all timers and worker threads.
3. Move service construction to `MainWindow`/`TradingRuntime` and inject panels.
4. Add persistent Auto control to top bar.
5. Define OFF, Desk, Away, and optional shadow-research semantics.
6. Prevent OFF from mutating bot files unless shadow research is explicitly enabled.
7. Add structured health/status snapshot.

Tests:

- repeat start/stop/restart without leaked strategy threads;
- close app during reconnect/scan/watchlist build;
- mode transition truth table;
- no bot-owned watchlist writes in strict OFF;
- Desk/Away produce identical opportunity decisions.

### Packet B — Pure SPY state machine

Goal: one tested market-state/pullback engine.

Primary files:

- new `market_state.py`
- `bounce_bot_lib/legacy.py` compatibility adapters
- `market_session.py`
- Auto/Bounce services and SPY chart

Work:

1. Define typed bar/features/state/episode records.
2. Extract VWAP regime features.
3. Implement state transitions, confidence, hysteresis, and invalidation.
4. Persist episode events.
5. Adapt current regime labels and `_detect_spy_pause_start()` temporarily.
6. Shade episodes in SPY chart and expose state in the global header.

Tests:

- strong bullish impulse → pullback → stabilization → resume;
- bullish failure through structure;
- bearish mirror of every sequence;
- chop does not create repeated episodes;
- incomplete/stale/missing bars cannot transition state;
- DST/early close/session rollover.

### Packet C — Relative Strength Engine and broad/live funnel

Goal: rank the best defiant stocks during each SPY episode efficiently.

Primary files:

- new relative-strength engine/repository modules
- RRS functions in Bounce legacy
- `autopilot_core.py`
- universe/industry data services
- RS Window feed/panel

Work:

1. Build aligned multi-window stock/SPY/sector features.
2. Add side-symmetric scoring and extension/data-quality gates.
3. Batch broad-universe snapshots and promote top live candidates.
4. Calculate all RRS timeframes in one pass.
5. Make regime pause, Entry Assist, environment scan, and RS Window call this engine.
6. Store rank components and full considered cohort.

Tests:

- exact timestamp alignment and missing coverage;
- volatility/beta normalization;
- long/short mirror equivalence;
- full-universe candidate can outrank a preexisting watchlist name;
- same cached input gives identical results across all four UI/features;
- provider-call and indicator-computation count reductions.

### Packet D — Candidate registry and Opportunity lifecycle

Goal: replace competing watchlist mutations and alert floods.

Primary files:

- new candidate/opportunity store
- `autopilot_core.py`
- `ui/services/autopilot_service.py`
- Bounce/Master focus integration
- alert center and opportunity UI

Work:

1. Define candidate provenance/membership leases.
2. Import manual text lists as user-owned sources.
3. Convert open scan, auto-populate, near-extreme, Focus, Master, and outcome-required membership into sources.
4. Derive live scan pool without overwriting manual input.
5. Add Developing/Defiant/Holding/Ready/Triggered/Invalid/Expired stages.
6. Emit typed transition alerts and compatibility text exports.

Tests:

- simultaneous sources do not duplicate a symbol;
- removing one source retains other memberships;
- user-owned entries survive every automation pass;
- stale writers cannot erase current state;
- one lifecycle transition produces one alert;
- restart reconstructs identical active opportunities.

### Packet E — Complete pick/outcome measurement

Goal: measure every considered and selected stock without conditional bias.

Primary files:

- autopilot pick/scorecard functions
- bounce outcomes/learning
- human focus tracking
- journal linker

Work:

1. Assign pick/signal/opportunity IDs at decision time.
2. Store observed price/time and rank components.
3. Evaluate all selected picks, not only those later confirmed by BounceBot.
4. Sample matched rejected/control candidates.
5. Separate candidate quality, alert conversion, entry trigger, and post-trigger outcome.
6. Link fills and passes/misses.

Tests:

- picks without bounce alerts still receive outcomes;
- later outcomes cannot change original rank/pick fields;
- manual/bot/overlap cohorts share identical entry policies;
- missing data reported as missing, not loss/exclusion;
- restart/idempotent outcome update.

### Packet F — Master scanner staging and tracker storage

Goal: speed swing scans and make their output compatible with the Opportunity Engine.

Primary files:

- `master_avwap_lib/runner.py`
- staged extractions from Master legacy
- tracker/storage/expected-R modules
- data provider repository

Work:

1. Add golden normalized-run fixture.
2. Stage acquisition/features/detectors/ranking/tracker/export.
3. Batch/prefetch bars and compute shared features once.
4. Move tracker to incremental versioned storage.
5. Store rank components and detector versions.
6. Publish normalized Opportunities.
7. Split theta enrichment into a separate owned job.

Tests:

- golden results unchanged for performance-only changes;
- bounded provider calls;
- incremental tracker equivalence;
- partial provider failure produces explicit partial result;
- equity scan completes/publishes independently of theta.

### Packet G — Market Prep, Journal, and platform consolidation

Goal: complete the supporting product around the central runtime.

Work:

1. Concurrent typed market-prep source results.
2. Shared catalyst/earnings/calendar repositories.
3. Local journal DB plus deterministic shared exports.
4. Secure secrets.
5. Signal-to-journal linking.
6. Atomic settings and shared writes.
7. CI, dependency locking, package spec, and legacy UI retirement checklist.

Tests follow the integration/security/storage requirements in Sections 9, 10, and 18.

## 21. Revised first ten implementation moves

Given the clarified priority, the recommended immediate order is now:

1. checkpoint current work and restore the green test baseline;
2. fix BounceBot/BounceService start-stop lifecycle and the dormant `datetimde` defect;
3. make the main GUI own the runtime and put Auto Mode in the global header;
4. define truthful OFF/Desk/Away/shadow semantics;
5. add structured per-cycle profiling and provider-call counts;
6. extract the pure SPY state/pullback engine with mirror tests;
7. unify regime-pause, Entry Assist, environment scan, and RS Window ranking;
8. calculate all RRS timeframes from one fetched snapshot;
9. replace competing watchlist writers with the Candidate Registry;
10. measure every candidate/pick through canonical outcomes before tuning weights.

The mini-PC-specific UI/scheduler should receive only compatibility fixes required to keep tests and the optional headless path working. New product functionality belongs in the main GUI runtime and automatically becomes available to any later headless host through shared services.

## 22. P0 correctness findings from the deeper Master/research audit

These are not stylistic refactors. They can change which setups appear profitable or how live candidates are ranked. Fix them before using tracker/tuner results to change live scoring.

### 22.1 Moving-level look-ahead

Several cross, bounce, and first-deviation evaluations compare older bars against the **latest/final** AVWAP or band value rather than the value that existed on each older bar. A moving AVWAP band must be evaluated as a time series.

Affected areas include:

- current/previous AVWAP event detection in Master legacy and `runner.py`;
- `assess_first_dev_break_quality()` and related cross/rejection functions;
- dynamic AVWAP target evaluation during tracker recompute.

Required fix:

- compute point-in-time AVWAP/band arrays once;
- at historical bar `i`, compare price only with band value `i` or a prior-known value under the explicitly versioned execution rule;
- for a next-bar entry, use levels known at or before the prior bar close;
- add an invariance test: appending future bars cannot change whether an older signal fired;
- add a changing-band fixture where final-band comparison and point-in-time comparison intentionally disagree.

The band variance calculation itself also needs a versioned audit. Current implementations update variance with a custom order-dependent formula rather than standard final weighted variance or the usual weighted-Welford update (`weight × (x - old_mean) × (x - new_mean)`). This can systematically tighten bands. Do **not** silently replace it because every historical detector/target is calibrated to the current levels. Instead:

- name and freeze the current formula as a legacy band version;
- implement a standard weighted-variance candidate and a TradingView/reference parity fixture;
- vectorize cumulative VWAP/variance arrays once per anchor rather than repeated row/`.iloc` loops;
- run both versions in shadow over the same point-in-time snapshots;
- compare signal frequency, level error, entry geometry, and out-of-sample outcomes;
- promote only through a versioned detector/level migration with full recalibration.

### 22.2 Same-day scans masquerading as prior-day history

Master history is appended on every hourly run. Multi-day pattern logic can compare `history[-1]` and accidentally treat an earlier scan from the same market date as the previous day.

Required fix:

- key daily history by `(market_date, symbol, detector/config version)`;
- retain intraday observations separately;
- derive “previous day” strictly from the previous completed market session;
- make hourly reruns idempotent;
- test several same-date scan times followed by the next market date.

The AVWAP signal CSV also needs stable IDs/upserts instead of full concatenate/rewrite without robust deduplication.

### 22.3 Live/backfill parity is broken

The live `run_master` path and historical/backfill evaluator do not currently call exactly the same detector and gate sequence. Examples found in the audit:

- SMA-breakout analysis appears in historical evaluation but is absent from the corresponding live path;
- a post-earnings sessions-since-gap input is omitted in one live summary call;
- the short near-favorite quality gate is applied in backfill but not equivalently in live evaluation.

Required fix:

- create one `evaluate_symbol_asof(snapshot, config)` entrypoint;
- live scan passes the latest point-in-time snapshot;
- backfill passes a snapshot sliced at each historical as-of time;
- both produce the same typed detector/gate/rank components;
- report-only enrichment occurs afterward;
- add golden live/backfill parity tests for every promoted detector family.

Side normalization must also become strict. A current compatibility helper maps every value other than exact `SHORT`—including empty strings and typos—to `LONG`. That can silently convert corrupt/unlabeled rows into long candidates. Use a `Side` enum and return `UNKNOWN/validation error` for invalid input. Only an explicit legacy-ingestion adapter may apply a default, and it must attach a data-quality warning and count it in the run manifest.

### 22.4 Adaptive backfill leakage

Historical backfill can recompute an old setup all the way through the present, then allow those now-known outcomes to affect scoring of later historical dates. Some filters can also see a full present-day frame while evaluating an earlier as-of date.

Required fix:

1. build immutable chronological signal snapshots using only data available at each timestamp;
2. label outcomes in a separate pass;
3. at simulated date `T`, train/adapt only on outcomes that matured before `T`;
4. purge/embargo overlapping swing episodes;
5. save the training cutoff and model/config version with every historical rank;
6. test that adding future outcomes cannot change earlier ranks.

### 22.5 Tracker identity collisions and pseudo-samples

The current tracker study identifier can omit study family/version, so different study setups on the same symbol/day/anchor/bucket can overwrite each other. Aggregations can also count repeated daily observations of one episode as independent samples. Conversely, an episode key can merge a genuine later re-entry under the same earnings anchor.

Required identity:

```text
detector_id + detector_version + symbol + side + event_timestamp
+ anchor_id + entry_policy + episode_sequence
```

Rules:

- observation snapshots are not new episodes;
- rank upgrades remain attached to the original episode;
- a genuine re-entry requires detector-specific rearm/cooldown and a new episode sequence;
- performance reports show episode N, unique-symbol N, unique-session N, and effective N;
- study/control/live cohorts cannot overwrite one another.

### 22.6 Score/bucket inconsistency

The expected-R ranking stage can overwrite a score after final buckets and bucket-upgrade persistence have already run, without rerunning/synchronizing every dependent field. This can produce a row whose displayed score, `priority_score`, bucket, and persisted state disagree.

Required fix:

- make one ordered ranking pipeline produce an immutable `RankingDecision`;
- raw setup quality, evidence adjustments, expected-R blend, hard blocks, caps, and final bucket run once in declared order;
- never mutate score after finalization;
- store both pre-expected-R and final score as separate named fields;
- derive bucket from final score/policy only;
- assert invariants before publishing.

Also integrate and test freshness behavior. Unit tests for freshness math are insufficient while the live days-since-signal path effectively returns zero/no decay.

There is an additional recursive-calibration risk: tracker `priority_score` is populated after expected-R/Proven Quality Score can replace `row['score']`. Later expected-R calibration treats that stored value, minus only selected deltas, as if it were independent structural quality and can refit the prior from it. The supposed static-score tiebreaker can likewise already include recent/setup-type outcome feedback. This lets outcome-derived scoring feed back into its own prior.

Persist immutable, separate fields:

- `structural_score` — detector/market features only, before any learned outcome adjustment;
- `model_score` / `expected_r` — out-of-fold learned estimate;
- `policy_score` — final presentation/ranking score after declared policy;
- `model_version`, `training_cutoff`, `prediction_as_of`;
- each adjustment component.

Calibrate only from out-of-fold structural predictions and subsequently matured outcomes. Never train on a field that already contains the model's earlier output.

### 22.7 Scan-factor horizon and multiple-testing problems

Current factor horizons can use the next **appearance of a symbol in feature history**, not the next completed market session. When multiple hourly observations exist, later same-day rows can replace the decision-time row. SPY-relative outcomes may be absent unless SPY is coincidentally present in the feature history.

Required fix:

- canonical market-session index and benchmark series independent of candidate membership;
- retain the first/decision-time snapshot plus intentional later lifecycle snapshots;
- evaluate fixed 1/2/3/5/10-session horizons from the exchange calendar;
- use date/sector/liquidity-matched controls;
- cluster by symbol/session and correct for the number of tested factors/bins;
- require confidence intervals and out-of-sample lift, not `n >= 8` alone.

### 22.8 Auto-tuning currently risks in-sample promotion

Tracker update/export, tuner reload, and recommendation application are tightly coupled and can perform duplicate full I/O. More importantly, recommendations derived from the current evidence can be auto-applied without a later untouched validation period.

Required workflow:

```text
fit -> recommendation -> shadow config -> forward/holdout validation
-> manual promotion -> versioned activation -> rollback option
```

The active config must never change merely because a same-run in-sample recommendation exists.

### 22.9 Playbook controls are not fully reproducible or independent

Findings:

- baseline sampling uses Python `hash(symbol)`, which is randomized per process;
- overlap suppression and `last_exit` make the nominal every-five-session baseline dependent on earlier episode duration;
- overlapping symbol/date setup outcomes are treated too much like independent observations;
- near-window winners can remain open/excluded while quick stops are closed, creating right-censor bias;
- very small risk denominators can create extreme R values;
- a nominal 252-day breakout can operate with substantially less than 252 prior sessions;
- earnings BMO/AMC mapping needs decision-time-aware session handling;
- empty output writers can leave a stale old result visible.

Required fix:

- stable digest-based sampling seed;
- a clean date/liquidity/side matched control independent of strategy exits;
- detector-specific risk floors and executable entry/exit policies;
- matured fixed horizons plus survival/right-censor reporting;
- clustered block bootstrap and walk-forward validation;
- minimum required lookback enforced literally or detector renamed;
- explicit BMO/AMC availability rules;
- atomic empty output that replaces stale reports with a valid “no rows” generation.

### 22.10 Universe and industry data can silently broaden or bias results

Findings:

- optionability-source failure can become an empty list that silently disables the filter and broadens the universe;
- unknown market caps can bypass a market-cap floor, and failed cap lookups can be cached as zero for a week;
- current partial daily bars can enter volume/trend screens;
- `auto_adjust=False` makes long-horizon trend filters split-sensitive;
- one missing price symbol can trigger an all-or-nothing redownload/cache rewrite;
- manual includes bypass screens without a structured override reason/expiry;
- current universe membership creates survivorship bias in historical research;
- industry scan can request the full classification universe in one unchunked yfinance call;
- member frames need a common completed-session as-of date;
- partial-day volume versus completed-day average creates time-of-day bias;
- missing industry RS can be treated as aligned evidence rather than unknown.

Required fix:

- last-known-good/fail-closed behavior for critical universe filters;
- incremental per-symbol/session price cache with coverage manifest;
- adjusted prices for trend features and unadjusted prices where execution levels require them;
- complete-session ADV and freshness checks;
- explicit unknown versus pass/fail;
- audited manual override reason/expiry;
- dated point-in-time universe snapshots;
- chunk/retry/cache industry fetches;
- coverage/as-of diagnostics and catastrophic-count-drop guard;
- never award an alignment bonus for missing group data.

Two small but worthwhile universe fixes belong in the first optimization pass:

- hoist the option-symbol `set(...)` construction outside membership comprehensions instead of rebuilding it per listed symbol;
- calculate price/volume metrics on jointly valid, date-aligned rows and use mean daily dollar volume (`close × volume`) rather than `last_close × mean(volume)`.

### 22.11 Detector-specific semantics requiring correction

The deep audit found several live detector details that should be handled when each detector is extracted:

1. **Representative risk plan is too universal.** Setup candidate payloads use an AVWAPE stop and BAND2/BAND3 targets across detector families. For a second-deviation breakout, BAND2 can already be behind the entry; that outcome does not represent the detector. Give every detector family a versioned entry, invalidation, target, and time-stop policy. Attach/update the final candidate payload after final ranking so stored score components are not stale.
2. **TOP pattern can use an incomplete weekly candle.** Split last completed weekly context from current-week preview. Preview signals must be labeled repaintable. Require sufficient weekly warmup; test rising/falling moving-average conditions explicitly. Bound how deep a level undercut may be before a reclaim still counts, rather than allowing arbitrary-depth recovery. Add a researched short mirror rather than assuming the long rule inverts automatically.
3. **SMA breakout uses current ATR for historical retests and is long-only.** Use each historical bar's point-in-time ATR, bounded pierce/reclaim semantics, volume/RS confirmation, and explicit rearm. Add an exact short-side implementation and mirror tests. `abs(low - level)` alone misses legitimate bounded undercut/reclaim behavior.
4. **Missing intraday VWAP can act like confirmation.** Mid-earnings and all similar evidence must be tri-state (`PASS`, `FAIL`, `UNKNOWN`). Missing required evidence cannot promote a Favorite setup.
5. **Synthetic H4 and current H1 bars can be partial.** Aggregate fixed exchange-session buckets, mark completeness, and use only closed H1/H4 bars for confirmed trend/setup decisions. A current partial bar may be shown as preview only. Missing prior slope/SMA warmup is `UNKNOWN`, not an implicit pass.
6. **High-volume level history needs knowledge time.** Store both `effective_from` and `known_at`; historical evaluation filters on what was knowable then. Do not back-displace a level's first-known time into the past. Use point-in-time ATR for level origins. Do not finalize truncated forward-return statistics near the dataset end; queue immature outcomes until their full horizon exists.
7. **Setup tag-and-hold pierces need bounds.** A touch/hold cannot permit unlimited penetration and still mean the same setup. Express tolerance in point-in-time ATR/percent, retain maximum adverse pierce as a feature, and study thresholds in shadow.
8. **Second-deviation continuous states need rearm.** Prevent a detector from immediately creating another independent episode merely because the same boolean condition remains true after a prior episode times out. Require an explicit reset/cross-back/cooldown.

The tracker execution simulator also needs an explicit order-knowledge model:

- validate geometry at entry: a long target must be above entry and a short target below; stop geometry must also be directionally valid;
- if a generic band target is already behind entry, mark it `STALE/INVALID_TARGET` or choose the detector's next valid target—never count it as an immediate target hit;
- a fixed level known before the bar may interact with the next bar's OHLC under the declared fill/ambiguity policy;
- a dynamic AVWAP/band level finalized by the current bar cannot be executed earlier within that same bar;
- a close-confirmed failure becomes actionable no earlier than the next tradable open/bar, not at the close that first reveals it;
- store signal-known time, order-active time, fill policy, and ambiguous-bar status.

Add geometry, same-bar, gap-through, dynamic-level, and next-bar execution tests for long and short scenarios.

For every detector, add a small contract page/table containing:

- required inputs and missing-data behavior;
- complete versus preview bar policy;
- point-in-time level policy;
- trigger/rearm/expiry;
- entry/invalidation/targets/time stop;
- side support and mirror status;
- detector/config version;
- live/backfill parity fixture.

## 23. Runtime and platform defects to fold into Phase 0/1

### 23.1 Outside-hours stale scanning

`SCAN_OUTSIDE_MARKET_HOURS=True`, the immortal strategy loop, and `_spy_session_bars()` choosing the date of the last available bar can allow evening/premarket cycles to treat stale bars as the active session.

Required fix:

- exchange-calendar session gate;
- latest SPY bar must match expected market date and freshness tolerance;
- separate live scanning, premarket prep, and post-close outcome jobs;
- no live opportunity alert from a stale prior session;
- UI/report must show `CLOSED`, `STALE`, or `NO CURRENT SESSION`, not a live regime.

### 23.2 SPY fast lane must precede broad symbol work

The current strategy cycle can warm ATR and run repeated broad RRS work before checking the SPY pullback. On a large list or during timeouts, the pullback can be over before it is evaluated.

Required architecture:

- SPY incremental bar callback/event triggers MarketStateEngine immediately;
- pullback state transitions are independent of slow universe scanning;
- current live-focus candidates update against that event within a defined latency budget;
- broad discovery may finish later and join an active episode with coverage noted.

### 23.3 Duplicate five-day/M5 fetch paths

`run_rrs_scan()` populates cached bars, while `request_and_detect_bounce()` can fetch the same five-day five-minute data again through a different path and cache key. Direct request paths do not all share pacing, cleanup, and mapping behavior.

Required fix:

- one bootstrap/incremental BarRepository request per symbol/timeframe;
- detectors accept the cycle snapshot and cannot fetch;
- all timeouts call provider cancellation and tombstone late callbacks;
- tests assert provider request counts for 25/100/350-symbol fixtures.

### 23.4 Universe truncation and request budgets

`load_universe_pool()` reads lists in fixed order and truncates, allowing the long universe to dominate the first 1,200 names. Existing auto-populate caps can place 150/50 or 100/100 names into high-frequency lists, far beyond a dependable sequential IBKR request budget.

Required fix:

- deduplicated union before sampling/ranking;
- stratify by side/liquidity/sector rather than file order;
- broad batch stage may cover the full universe;
- live IBKR stage has an explicit measured request budget, initially closer to 15–25 per side plus priorities;
- candidate demotion/promotion is event-driven and visible.

### 23.5 Freshness by newest mtime is unsafe

`universe_built_at()` uses the newest companion-file mtime, so one fresh file can hide stale or missing files. Similar logic affects auto-list clearing.

Required fix:

- generation manifest lists every required artifact, hash, count, and timestamp;
- generation becomes current only after all required artifacts validate;
- consumers reject mixed generations;
- never infer a multi-file generation from `max(mtime)`.

### 23.6 Multiple GUI schedulers can overlap

`MasterAvwapPanel` and `AutopilotService` each own a `ScanService`/scheduler. Manual actions are another entry path. They can launch overlapping scans that write identical report/tracker outputs.

Required fix:

- one application-owned ScanCoordinator;
- dedupe by job/session/config generation;
- serialize writer-heavy Master jobs;
- all panels subscribe to the same job state;
- manual request can raise priority but cannot create a duplicate run.

### 23.7 Startup does unnecessary work before Qt can recover

`scripts/gui.py` imports the legacy Tk application before deciding to launch Qt; every main page is constructed eagerly; `project_paths` may wait/migrate at import time before `QApplication` exists.

Required fix:

- import legacy UI only inside the `--ui tk` branch;
- show Qt shell quickly, then initialize runtime with progress/degraded recovery;
- lazy-create noncentral pages;
- move Drive readiness/migration into startup jobs;
- use one atomic settings save and implement/remove unused state fields;
- refresh global status from live runtime state.

### 23.8 Away report needs verified publication

The current writer catches errors and returns a path; caller can record a successful heartbeat even if Drive write failed. Direct overwrite is not atomic and one latest file has no generation audit.

Required publisher:

- render/validate locally;
- queue in a local outbox;
- atomic publish to a generation-stamped archive and latest alias;
- retry transient Drive locks;
- hash/readback verification;
- separate `last_attempt` from `last_verified_success`;
- never clear the previous valid report on failure.

The actionable report section must also enforce ranking invariants. A negative expected-R or hard-blocked row cannot appear as an unqualified “Top” or “High Conviction” pick. It may appear only in a clearly labeled research/watch/blocked section with the conflict explained. Add a publication test that fails when bucket label, final score, expected R, hard blocks, or data freshness contradict one another.

### 23.9 Full Market Prep is missing from the main Qt product

The Qt Research view exposes mainly Master-derived prep while the richer macro, earnings, Fed, Treasury, news, SEC, roadmap, and AI workflow remains in legacy UI. Market Prep can also resolve repo-root watchlists instead of the configured shared watchlists.

Required fix:

- port full prep command center to Qt;
- use `project_paths`/CandidateRegistry, never a separate path universe;
- schedule deterministic prep premarket and changes during the session;
- fetch widest required windows once and slice locally;
- bounded-concurrent independent providers;
- optional asynchronous AI enrichment;
- source failure becomes `UNKNOWN/INCOMPLETE`, never “Noisy” or “Clean” by default.

### 23.10 Journal hot paths and correctness

Deep review findings:

- some GUI load/rebuild/export work is synchronous;
- trade listing can perform a regime lookup/connection per trade;
- a seven-day Questrade import can refetch accounts and rebuild/tag the full journal once per day;
- tagging repeatedly scans broad context;
- raw `net_pnl` can be aggregated across currencies while `pnl_usd` is absent;
- timestamps need canonical UTC plus market-local trade date;
- Questrade token in URL parameters can leak through persisted exception text.

Required fix:

- one managed range import and one incremental rebuild transaction;
- join/cache regimes once and index tagging context;
- base-currency conversion with rate provenance and missing-FX state;
- managed background UI jobs;
- startup/post-close reconciliation;
- central secret redaction before logging or persisting errors;
- OS credential store migration.

### 23.11 Alert parsing and deduplication

Free-text parsing can infer fake symbols from batch messages, and Alert Center lacks one canonical event-ID dedupe policy.

Required fix:

- typed alerts from producer to UI;
- stable event/episode/opportunity ID;
- severity, lifecycle transition, freshness, reason codes, and display text as separate fields;
- batch events carry explicit symbol arrays rather than reparsing prose;
- dedupe/cooldown at central policy layer;
- visible feed and immutable audit history separated.

## 24. Recommended initial SPY-engine parameters for shadow testing

These values are starting hypotheses, not production truth. Put them in a versioned config, run in shadow, and tune only through distinct-event walk-forward evidence.

### Strong trend entry/exit

- aligned day return at least `max(0.35%, 0.25 × SPY daily ATR%)`;
- price and three-bar VWAP slope on the aligned trend side;
- at least 8 of last 12 complete closes on the aligned side of VWAP;
- composite trend score at least 0.65 for two complete bars to enter strong state;
- remain strong until score is below 0.35 for two complete bars.

### Counter-move start

- drawdown/bounce from latest impulse extreme at least `max(0.12%, 0.30 × SPY M5 ATR%)`;
- at least two of the last three closes countertrend, or an adverse EMA8 cross;
- preceding impulse at least 0.60 M5 ATR;
- no material structural failure through VWAP;
- one episode ID anchored to the impulse extreme and first adverse complete bar.

### Resumption/failure

- episode lasts at least two complete bars;
- no new countertrend extreme for one bar;
- aligned close through prior-bar high/low and back through EMA8;
- at least 35% retrace of the counter-move;
- fail on two closes roughly 0.25 ATR through VWAP, regime exit, depth over 1.5 ATR, stale data, or duration over 8–12 bars;
- cooldown until a new trend extreme or three complete bars.

### Candidate quality

- exact latest timestamp alignment;
- at least 80% episode-bar coverage;
- positive aligned episode RRS and positive aligned day RRS for initial promotion;
- no strong opposite VWAP/structure violation;
- liquidity/data-quality gates;
- short rows display SSR, borrow/locate availability if known, squeeze/gap risk, and `unknown` when unavailable. These are decision-support warnings, not execution logic.

## 25. Additional tests Claude should add before optimization/tuning

### Runtime/orchestration

- start/stop/restart leaves exactly zero or one strategy worker as expected;
- Stop during startup cannot install a late bot;
- all timers stop and snapshots clear;
- no live requests/alerts outside an actual exchange session;
- disconnect aborts the current provider batch immediately;
- scan timeout/cancellation reaps the child process;
- older theta/output generation cannot overwrite newer output;
- failed/cancelled/skipped/successful jobs remain distinct and retry correctly;
- wrap-up completes only after every required dependency succeeds.

### SPY/RS engine

- false single-candle pause does not start an episode;
- valid bullish/bearish episode lifecycle;
- neutral never defaults to long;
- stale/incomplete SPY bars cannot transition state;
- stock/SPY timestamps align and missing coverage is rejected;
- signed long/short mirror property;
- flat stock during aligned SPY counter-move earns positive defiance;
- sector-driven versus stock-specific residual separation;
- one evidence count per distinct episode regardless of scan/timeframe count;
- top-set material-change alert dedupe;
- fixed provider-call/cycle deadlines at 25/100/350 names.

### Master/research correctness

- appending future bars cannot change an old signal;
- changing AVWAP band uses its point-in-time value;
- live/backfill golden parity;
- repeated same-day hourly scan idempotency;
- detector-family/version-safe study IDs;
- distinct re-entry after explicit rearm;
- future outcome cannot change prior historical rank;
- benchmark horizons use market sessions;
- partial daily bars excluded;
- BMO/AMC mapping at decision time;
- stable baseline across Python processes;
- censored/open cohorts reported correctly;
- universe provider failure uses last-known-good/fail-closed policy;
- two processes cannot corrupt/lose tracker or watchlist state.

The current targeted tests passing is not proof that these issues are absent; several existing tests intentionally lock in the current methodology. Behavior changes must update those tests deliberately and add the missing invariants rather than simply preserving every current result.
