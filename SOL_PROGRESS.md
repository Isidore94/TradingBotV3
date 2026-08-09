# Checkpoint marker

[`plan.md`](plan.md) owns all status, roadmap, and promotion policy. The full
implemented/remaining inventory lives in Section 3, and the ordered work queue
in Section 12. This file is only the small, frequently refreshed checkpoint
stamp; it must not duplicate the roadmap.

## Current checkpoint

- Branch **`testing-week-2026-08-10`** (2026-08-09), built off `main` `7d85a27`
  for the 2026-08-10 testing week: Mon-Wed Auto/Away + baseline testing with no
  trading, Thu-Fri live-session tests per plan.md sec 6. **Not merged to main,
  and no PR opened.** Four packets, merged one at a time with the full suite
  green after each:
  1. `chart-perf-c` — chart background loading, bar cache, stall watchdog, Drive
     reads off the paint paths. 2071 passed.
  2. `chart-review-workspace` — two-keystroke trader decision capture under
     `scripts/ui/annotations/`, the Chart Review panel, veto cohort grading.
     Conflict-free. 2172 passed.
  3. `9037c5f` cherry-picked from `integration-test` — the previously
     UNREVIEWED packaging WIP (PyInstaller spec, Qt runtime hook, the
     `launch_gui.py` frozen `sys.path` guard, the atomic-write temp sweep).
     Reviewed here; see the two fixes below. 2176 passed.
  4. `claude/das-warehouse-defects-2n9uql` — warehouse Phases 1-8 + both defect
     passes. Only SOL_PROGRESS.md conflicted; resolved by hand keeping both
     narratives (the warehouse block is under its own subheading below).
     2487 passed.
  5. `claude/a4-verify-a3-orchestrate-9c8kop` (2026-08-09, merge `06d8429`) —
     the A4/A5 stream, folded in on the trader's call after this branch had
     shipped without it. Brings A4 D1 paint lines (`scripts/chart_levels.py`,
     the Lines button and its machine-local prefs, levels on the chart
     snapshot), the packaging guards and `launch_gui.py --selftest`,
     pytest-timeout, the `d1_trendline_survey` / `d1_level_store_survey`
     read-only tools, the Lines-button height-neutrality fix, A5 click-to-arm
     routed through `AlertCenterPanel` so `PriceAlertService` keeps its single
     writer, and `d1_level_feed`'s shared mtime-cached ai_state loader (the
     last two are the trader-approved fence exceptions recorded in
     `docs/HANDOFF_A4_PACKAGING_2026-08-09.md` §10.6). Six files conflicted —
     see that merge commit for the per-file resolution. 2582 passed.
  6. `claude/testing-production-blockers-oek3aj` (2026-08-09, cut from
     `testing` @ 59128c5) — the capture-stream hardening packet from the Sol
     5.6 verification review, folded in here after the trader's call. Six
     commits, one line each:
     - `98414ce` Chart Review captures a judgement and must never grant alert
       privileges: the capture rail's LIKE writes an annotation only, never a
       watchlist, a Focus list or a price alert.
     - `f1a7019` a torn annotation write may cost its own row, never the one
       after it: `annotations/store.py` heals a torn tail before appending and
       fsyncs, so a half-written last line cannot corrupt the next append.
     - `53ab6dd` the chart bar cache re-stats the durable store on a memory
       hit, so a scanner publish is noticed without a desk restart.
     - `c11008a` a characterization fence that only reads the name tags is not
       a fence: the veto-cohort test now pins every field of every row against
       the pre-change `main` output.
     - `1327a17` its own fix of the same 9037c5f sweep defect packet 3 fixed —
       reconciled against `b7615b7` below rather than kept alongside it.
     - `210affa` the branch's checkpoint stamp.
     Three files conflicted (`scripts/project_paths.py`,
     `tests/test_project_paths.py`, `SOL_PROGRESS.md`); everything else
     auto-merged.
- **Gate on this branch: 2582 passed, 5 skipped, 7 subtests; junit
  `failures=0 errors=0`; smoke 7/7; `launch_gui.py --selftest` 30/30 exit 0.**
  Linux container, Python **3.12.3**, `TZ=America/Vancouver
  QT_QPA_PLATFORM=offscreen`. The desk runs 3.14 on Windows, so **this number
  is not the desk gate** — see the owed list. The spec-drift negative control
  was re-run after the merge: pulling `research_warehouse` from the spec fails
  5 of the merged file's 16 tests (both suites' package censuses and both asset
  sweeps), green again restored.
- Three fixes were made on this branch, each its own commit:
  - **Warehouse parquet reads** (`store.py`): `pq.read_table` builds a dataset
    around the file, so every part under `year=NNNN/` came back with a synthetic
    dictionary-typed `year` column from the directory name. Compaction sealed
    that column into the merged file, and the startup reconcile then died on
    `ArrowTypeError` — which its `except (OSError, pa.ArrowInvalid)` does not
    catch, so a crashed compaction took the next startup down. All three sites
    now read the file itself; the catch widened to `pa.ArrowException`. **This
    reproduces on `claude/das-warehouse-defects-2n9uql` unchanged** under the
    pinned pyarrow 22.0.0 — the merge did not introduce it, and that branch's
    reported 2088-passed gate must have run against a different pyarrow.
  - **Startup sweep ownership** (`project_paths.py`): 9037c5f's sweep deleted
    any six-hour-old `.<anything>.<8>.tmp` in whole directories, one of which is
    the cloud-synced shared home that other programs write into. It now takes
    canonical targets, not directories, and the import-time call derives them
    from this module's own path constants, so a file it cannot name is never a
    candidate. Both original leaks are still swept.
    **Reconciled with `oek3aj`'s independent fix of the same defect
    (`1327a17`) when packet 6 merged.** Both branches converged on the same
    design — `dotted_for` naming canonical targets instead of directories, and
    an `_owned_staging_targets()` derived from the module's own UPPER_CASE
    `Path` constants. `b7615b7`'s shape was kept (the parameterized
    `_owned_staging_targets(directories)`, so the swept set is stated at the
    call site rather than hardcoded twice), with **one thing ported in from
    `1327a17`: the `value.suffix` filter.** That is not cosmetic. Nine
    suffixless `Path` constants sit directly in the three swept directories —
    `DATA_DIR`, `OUTPUT_DIR`, `LOG_DIR`, `RUNTIME_DATA_DIR`,
    `LOCAL_MACHINE_CACHE_DIR`, `ALERT_REVIEW_EVENTS_DIR`,
    `MASTER_AVWAP_LEVELS_DIR` and the two bar stores — and they are
    directories, not staging targets. Admitting them built delete patterns for
    names like `.daily_bars.<token>.tmp` and `.logs.<token>.tmp`: shapes no
    writer here ever stages, so files the project could not claim to own. With
    the filter the owned set is 90 real file targets and those nine are
    excluded. The invariant now holds in both directions: a file the module
    cannot name as its own *staging target* is never a deletion candidate, and
    both original leaks (the bounce-candidates CSV via `staged_for`, the
    earnings-history temps via `dotted_for`) are still swept. The merged test
    file is the union of both sides' cases (14 tests): `b7615b7`'s stricter
    selectivity test was kept over `1327a17`'s inertness-only version and
    extended with its stranger case, and `1327a17`'s derivation test was kept
    alongside for its named-constant and `suffix` assertions, which are what
    pin the ported fix.
  - **Spec drift** (`packaging/`): the warehouse merge trips three frozen-exe
    rebuild triggers at once. The spec now derives its asset mirror from
    `FIRST_PARTY_PACKAGES` (so `research_warehouse/exploration_cohort.txt` ships
    instead of silently going missing), collects `duckdb` when installed and
    says so when not, and adds `desk_link` — which `ui.services` still imports.
    `tests/test_packaging_spec_drift.py` executes the spec with the collectors
    stubbed and fails on any package or asset that is in neither the spec nor a
    reasoned allowlist. 0.08s; it converts CLAUDE.md rebuild triggers 2-4 into a
    commit-time failure.
- **Still owed before Thursday** (none of it is doable from a Linux container):
  1. **Windows/3.14 re-baseline.** Re-run the suite and smoke on the desk and
     replace the 2496 figure above. The warehouse review's test-gate caveat
     still stands.
  2. **Frozen exe rebuild + engine exercise.** No PyInstaller build was
     attempted here. The spec changed and duckdb is new, so triggers 1, 2 and 3
     all fired: rebuild, launch, and exercise the engines — "it launched" is not
     evidence, the failure mode is a lazy import weeks later. Cheaper now that
     the A4 stream is in: `dist\TradingBotV3\TradingBotV3.exe --selftest`
     should print `selftest OK: 30/30 checks passed (frozen)` and exit 0, which
     is what retires the click-through. The unfrozen run passes here (30/30);
     only the frozen Windows one is still owed.
  3. **Broker-marked IB run (BD-25).** `ib_capture.build_ib_transport` still has
     no offline test and no live run; its socket behaviour is unconfirmed.
  4. **Warehouse confirmation-register answers** — the trader items in
     `docs/ULTIMATE_SETUP_DATABASE_PLAN.md`, including the deliberately empty
     `exploration_cohort.txt`.
  5. **13c mid-session restart drill** (audit HEALTHY with a nonzero backfill
     count). Until it runs, 13c is not `LIVE_VALIDATED`.
  6. **Phase 1 unattended week** — the exit gate still needs its real week.
- Still unmerged and needing a trader decision:
  `scoring-flagging-evidence-guardrails` (+1, scoring code). Neither the A4
  stream nor the production-blockers packet is on that list any more —
  `claude/a4-verify-a3-orchestrate-9c8kop` is the superset of
  `claude/a4-paint-lines-packaging-nug5km` and came in via `06d8429`, and
  `claude/testing-production-blockers-oek3aj` is packet 6 above.
- **Carried over from `oek3aj`'s own stamp, since nothing else records it:**
  that branch independently measured the same timezone sensitivity this ledger
  notes, counting **16** UTC-sensitive session-window tests against
  `TZ=America/Los_Angeles` and confirming the identical failures reproduce on
  unmodified `59128c5` — environmental, not a branch regression. Its stamp also
  filed the post-summary `multitasking`/`run_strategy` non-exit as known,
  pre-existing and unfixed, recommending `network` markers or explicit teardown
  in a follow-up packet; **that recommendation is superseded** by this branch's
  `tests/conftest.py` `_make_multitasking_inert()` fix, which removes the
  worker threads at the source rather than marking the tests that leak them.
- **Correction to this ledger's shutdown note (verified 2026-08-09).** The
  dead-port proxy workaround printed above is a placebo: yfinance rides
  `curl_cffi`, which ignores `HTTP_PROXY` / `HTTPS_PROXY` entirely, so a
  request still reaches the live internet with the proxy pointed at a dead
  port. The hang itself is real — a Qt test constructs the desk, whose
  universe-self-heal and industry-board threads make live calls, and ~300
  non-daemon `multitasking` workers then park in `threading._shutdown` for
  ~20 min *after* the summary prints. **Verdict protocol instead:** always run
  with `--junitxml=report.xml` and treat the printed summary plus junit
  (`failures=0 errors=0`) as the verdict, then reap the process — or wrap it as
  `python -c "import pytest,os,sys; os._exit(pytest.main(sys.argv[1:]))" tests/ -q --junitxml=report.xml`,
  which is how this checkpoint's gate was first taken. A post-summary hang, or
  a reaped 124/137, is never a test failure. **The hang itself is now fixed**
  (below), so plain `pytest tests/ -q` exits on its own again; keep the junit
  habit anyway, since it is what makes a verdict readable after a reap.
- **Shutdown hang fixed in `tests/conftest.py` only.** The blocker was never
  the desk's own threads — universe-self-heal and industry-board refresh are
  both daemon threads. It was `multitasking`, yfinance's fan-out helper, which
  creates its workers `daemon=False`; a few hundred of them parked in
  `threading._shutdown`. conftest now creates a `threads=0` multitasking pool
  at import, which makes `@multitasking.task` run inline and create no worker
  at all. yfinance's own `set_max_threads()` call cannot undo it — that only
  writes MAX_THREADS for *future* pools, verified against multitasking's
  source. Differential proof: identical 2582/5/7 both ways, but plain pytest
  exits 0 in **91s wall** with it and was still hung at **420s** with the one
  call commented out. No project file touched and no project function
  monkeypatched.
- **The suite is still not hermetic**, and that part is deliberately left for
  after the testing week: constructing the desk in a Qt test fires timers that
  make live outbound yfinance calls mid-run. Results are unaffected and the
  hang is gone, but the calls do still go out — now serially, on the calling
  thread. The real fix is making desk construction inert under pytest
  (`app.py`'s 2500ms `_self_heal_universe` single-shot plus its 30-min timer,
  and `IndustryBoardService.start`'s startup timer). That was **not** attempted
  on the eve of the testing week: every candidate seam either needs an autouse
  patch of functions that other tests legitimately assert on, or a
  test-awareness check inside production startup code.

### Merged into this branch: research warehouse Phases 1-8

Carried in from `claude/das-warehouse-defects-2n9uql`; the entries below are
that branch's own checkpoint, unchanged. They describe warehouse state, not
`main` state.

- Branch `claude/das-warehouse-defects-2n9uql` (2026-08-04), the defect pass on
  top of the Phases 1-8 review branch. Closes every S1/S2/S3 defect in
  `docs/RESEARCH_WAREHOUSE_REVIEW_2026-08-04.md`: feature windowing D5
  (BD-58/BD-59), backfill D6/D7 (BD-60), build-job coverage D19 (BD-61), the
  D14-D18 edge cases (BD-62), and the BD-20 live-tee wiring (BD-63).
  Linux build agent, `TZ=America/Vancouver QT_QPA_PLATFORM=offscreen`:
  **2088 passed, 2 skipped, 5 subtests** (that agent's baseline on the
  unmodified review branch was 2061 + the same 2 skips, so this adds 27 tests).
  **Not yet run on Windows/3.14** — the review's test-gate caveat still stands
  and that run must re-baseline this number. Also still owed before the pilot:
  one broker-marked IB run (BD-25) and the trader's confirmation-register items.
  NOTE for a Linux agent: the suite needs `TZ=America/Vancouver`. Naive bar
  timestamps are localized with the *system* timezone, so a UTC container fails
  7 desk-timezone tests (`test_vold_recorder`, `test_autopilot_core`,
  `test_technical_integrity`) that are green on the desk.
- Branch `claude/das-warehouse-phase-1-0gis7e` (2026-08-04), building the
  research warehouse Phases 1-8 on top of the merged Phase 0. Gate after
  Phase 8: **+237 warehouse tests** on the 1814-test baseline (adds
  `test_warehouse_seal.py`, `test_warehouse_manifest.py`,
  `test_warehouse_quarantine.py`, `test_warehouse_retire.py`,
  `test_warehouse_import.py`, `test_warehouse_tee.py`,
  `test_warehouse_spool.py`, `test_warehouse_pacer.py`,
  `test_warehouse_backfill.py`, `test_warehouse_aggregate.py`,
  `test_warehouse_avwap_parity.py`, `test_warehouse_features.py`,
  `test_warehouse_occurrence.py`, `test_warehouse_outcomes.py`,
  `test_warehouse_queries.py`, `test_qt_warehouse_readout.py`,
  `test_warehouse_restore.py`); smoke **7/7**. Measured on the Linux build
  agent: **2049 passed, 2 skipped, 5 subtests** (that agent's own baseline was
  1812 + the same 2 skips), so the desktop gate should read **2051 passed, 5
  subtests** — confirm on the next Windows run.
- Warehouse **Phase 1 landed** (store core, plan sec 19.2): the 4-step seal
  protocol (`store.py`), `manifest_log.jsonl` read authority (`manifest.py`),
  the frozen 13-table pyarrow schemas + deterministic occurrence/anchor keys
  (`schemas.py`), per-symbol/per-partition quarantine with clean-remainder
  publish (the tracker-blackout regression), compaction as one atomic manifest
  line, `_retired/` GC that skips files in use, startup reconciliation of
  crash artifacts, and the Phase-1 ERD (`docs/RESEARCH_WAREHOUSE_ERD.md`).
  Still shadow-only: no detector, score, ranking, or alert path imports it,
  and the store is a total no-op when `research_store_dir` is unset.
- Warehouse **Phase 2 landed** (bronze wraps + daily snapshots):
  `ingest_existing.py` wraps the sec 19.0 inventory into `bronze_*` datasets
  (tracker + scenario CSVs, bounce ledgers, `alert_review_events`, shadow and
  regime/RS artifacts, `technical_integrity_events`, job ledger/heartbeat/run
  manifests, earnings anchors + calendar history, and the four trader
  watch/level JSONs) with the source path, source file hash, and offset
  watermark on each manifest line — a re-run with no source change writes
  nothing. Daily snapshots populate `universe_membership_daily` (first capture
  wins, never backfilled) and `level_state_daily` (HV level stores,
  `d1_level_feed` SMA/trendline state, trader watch JSONs), and the durable
  per-symbol D1 Parquet store projects into `bar_d1` as a wrapped read,
  completed sessions only. Legacy writers are untouched;
  `scripts/research_warehouse/exploration_cohort.txt` is deliberately empty
  pending item 5 of the plan's trader confirmation register.
- Warehouse **Phase 3 landed** (M5 tee + coverage/gaps + spool):
  `bar_archive.py` archives BounceBot's already-fetched
  `latest_bars["<SYM>|5 D|5 mins"]` cache into `bar_m5` at zero provider cost
  (the module has no provider client at all — asserted by an AST test),
  completed bars only, idempotent per (symbol, interval_start); it also writes
  `scan_coverage` keyed by the run manifest's `run_id` (so coverage reconciles
  against run manifests by construction) and `collection_gap` rows that keep
  `NOT_COLLECTED_BY_POLICY` distinct from `MISSING`/`PARTIAL`. `spool.py` adds
  the sec-8.4 ownership split: the GUI-owned writer appends to `.open`
  segments, the CLI seals only `.closed` ones, with the 5 GB / 7-day cap, the
  fixed shedding order (D1/M5 never shed), and shed evidence surfacing as
  explicit gap rows.
- Warehouse **Phase 3b landed** (pacer + backfill + seed): `pacer.py` is the
  one process-wide arbiter — champion requests are counted, never delayed or
  queued; capture runs in a token bucket of the published floor minus observed
  champion consumption, yields instantly to champion activity and to IB error
  162/366, and honours the 15-second identical-request cooldown. Client IDs are
  asserted at connect (1003 retired, 1010 streamer, 1011 backfill, mini-PC
  refused). Capture errors are tagged `capture=True` and never reach
  `_IBKR_HISTORICAL_FAILURE_COUNT` — a test imports the champion module and
  proves the counter is untouched. `backfill.py` runs the ETH-inclusive
  (`useRTH=0`) nightly M5 job, the weekly universe sweep, and the trickled
  60-day yfinance seed with a per-symbol resume ledger; all are idempotent
  across the ~23:45 TWS restart and record paced-out/no-response work as gap
  rows. `ib_capture.py` is the only socket module.
- Warehouse **Phase 4 landed** (sessions + aggregation): `exchange_calendar.py`
  states the NYSE calendar as versioned rules (holidays with the NYSE
  observance rule, the three 13:00 ET early closes, DST handled by the zone),
  verified against the published 2025-2027 calendars; `aggregate.py` publishes
  `trading_session` rows and derives `bar_derived` — session-anchored M15 (26),
  M30 (13) and H1 (6 full + a 30-minute 15:30-16:00 stub) from canonical M5
  under explicit `aggregation_contract_id`s (half days use the half-day
  variant), plus W1 from canonical D1 that publishes only once the week's final
  session closes and flags short weeks. Derived H1 boundaries match IB native
  `useRTH=1` bars, which is the sentinel parity check; forming buckets are
  never derived, and a bucket with no constituents produces no row.
- Warehouse **Phase 5 landed** (tier-1 feature snapshots): `features.py`
  publishes `feature_snapshot_daily`, `feature_snapshot_intraday` and
  `anchor_instance` with exactly the frozen sec 7.1 columns. Every champion
  quantity is CALLED, never re-derived — `calc_anchored_vwap_bands` (parity
  1e-9 on a contract-bearing golden fixture, plus an AST assertion that the
  module holds no sigma math), `compute_indicator_frame` for the D1 EMA/SMA
  grid, and `BounceBot._calculate_vwap_bands` (called unbound) for the intraday
  session VWAP ±1σ. Snapshots are point-in-time and deterministic: recomputing
  from truncated history yields an identical row, and `input_manifest_hash` is
  built from the manifest's own file hashes.
- Warehouse **Phase 6 landed** (occurrences + outcomes): `occurrences.py`
  records detector output under the deterministic occurrence key — a rescan
  that changed nothing writes nothing, a rescan that changed something appends
  a linked revision, and 100 rescans still count 1 occurrence and 1 episode.
  `dependency_cluster_id` groups same-move variants (family excluded, side
  included). `outcomes.py` implements `house_default_v1` exactly: the net_r
  cost formula, STOP_FIRST primary with the TARGET_FIRST reading kept as
  `r_upper_bound`, the 18-session time stop, the swing/intraday checkpoint
  grids, MATURED as a derived predicate, and the five declared recipes
  (`swing_house_v1` with its band-2 partial / band-1 trail / band-3 runner
  simulated, two controls, the intraday bounce recipe, the ATR diagnostic).
  `intraday_bounce_v1` only ever runs from a linked bounce event.
- Warehouse **Phase 7 landed** (read path + readout): `queries.py` resolves
  every read from `manifest_log.jsonl` at query start — a query across a
  compaction returns the pre- or post-compaction row set and never the union,
  proven with a concurrent-compaction test — and publishes the slice readout
  (counts, mean R, checkpoints for the two slice setups) reporting rows,
  occurrences and **episodes** separately, matured apart from open, with the
  capture-mode split shown. DuckDB is pinned (`duckdb==1.5.5`, cp314 win_amd64
  wheel verified on PyPI) but strictly optional and read-only; pyarrow answers
  every slice query. The Research tab gains a read-only "Research Warehouse"
  panel that reads nothing until Refresh is pressed.
- Warehouse **Phase 8 code landed** (backup + build job + Health tiles):
  `backup.py` implements the 3-class policy (Class A mirrored to disk AND
  Drive, Class B append-only so a deletion is never propagated, Class C never)
  and the scripted restore check, which restores a partition to a NEW root,
  re-verifies every file against the manifest's recorded hash, and runs a
  canned query. `cli.py` adds `build` / `status` / `restore-check` with a
  single-flight lock (a live holder refuses even in-process; a dead holder's
  lock is reclaimed) and job-ledger registration.
  `scripts/ui/services/warehouse_service.py` computes exactly the six sec-18
  Health tiles from the ledger, with policy absence excluded from coverage
  defects.
- **Open (BD-52 / BD-20):** the 20-session pilot has NOT run — it is a live
  desk activity — and nothing calls the tee during a live session yet: the GUI
  must hand BounceBot's `latest_bars` to `capture_m5_tee` each cycle and the
  Health page must render the six tiles. Until then capture runs only from a
  manual build job.
- **Open gap (BD-44):** no detector adapter yet — Phase 6 proves the logic
  against constructed detections; nothing reads the tracker output into
  detection dicts.
- **Unverified (BD-25):** `ib_capture.build_ib_transport` — the real ibapi
  client — has no offline test and no broker-marked live run yet. Its socket
  behaviour must be confirmed on the desk before the pilot leans on it.
- **Open gap (BD-20):** nothing in the running desk calls the tee yet —
  `scripts/ui/services/warehouse_service.py` (GUI service + job-ledger
  registration + Health tiles) is still to be built, and the 20-session pilot
  depends on it.
- Builder decision log: `docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md`
  (BD-01..BD-52) records every implementation choice the locked plan left
  open, and ends with an **Open items for Sol / Fable** table (11 items:
  the unbuilt GUI service, the unverified ibapi client, the empty exploration
  cohort, two builder-stated favorite-zone definitions, null production
  context until Phase 6, DYNAMIC/EOD VWAP, unscheduled closures).

## Previous checkpoint (main, 2026-08-08 evening)

- Branch `main` (2026-08-08 evening, merged from `durability-catchup`,
  `local-ai-phase-0`, `local-ai-phase-1`). Gate: **2002 passed, 7 subtests**
  (adds `tests/test_durability_retry.py`, `tests/test_launch_guard.py`,
  `tests/test_ai_evidence_coverage.py`, `tests/test_ai_jobs_runner.py`,
  `tests/test_ai_jobs_store_window.py`, `tests/test_local_ai_provider.py`);
  smoke **7/7**.
- **Repair-and-merge program executed** against the checkpoint review's second
  review (`docs/CHECKPOINT_REVIEW_2026-08-08.md`, ADDENDUM). Two P0s in the
  tracker catch-up confirmed and repaired — the automatic path no longer runs
  the scoring tuner or the Expected-R prior refit (the manual GUI backfill
  still does), and the tracker stamps an explicit `data_session` vintage
  instead of inferring one from its write clock. Plus: bounded retries before
  either Tier B recovery path writes a permanent data gap, an honest `as_of`
  on an empty follow-up window, follow-up gap and outcome-coverage lines in
  the collection audit, a single-instance guard that sees the frozen build,
  three hard-rule gaps closed in the overnight AI runner, and an evidence
  packager that states what is missing rather than implying it.
- Merge strategy **amended to merge-early** per Sol: A `5d835ab`, B `b40cad7`,
  C `13f6e7b`, each green. `9037c5f` (WIP packaging) was not merged and stays
  on `integration-test`.
- **TradingBotV3 AI Jobs** was disabled during the repairs and re-enabled after
  a controlled proof on the real desk: 7 of 18 sources usable, 10 unfunded,
  1 missing, 5 stale, all stated in the published brief; ledger row `ok`.
- **Outstanding:** 13c is still not `LIVE_VALIDATED` — the mid-session restart
  drill (audit HEALTHY with a nonzero backfill count) needs a real session.
  Phase 1's exit gate needs its unattended week. The AI evidence budget
  (`MAX_TOTAL_EVIDENCE_CHARS` = 80,000) cannot fund ten real sources and is a
  trader decision, not a repair. The frozen-exe variant of the launch-guard
  drill is operator work. Phase 2 is stopped pending its fact-pack redesign.
- plan.md item **13c (durability & catch-up)** build-order steps 1-4 landed:
  15-minute repetition on the 07:00 launch task; the Master AVWAP tracker
  staleness override (reuses `backfill_setup_tracker_from_recent_sessions`,
  capped at the last *completed* session, pinned by a byte-identical
  characterization test); the Technical Integrity follow-up chain sweeper; and
  the breadth-ledger bar gap fill. Tier B rows carry
  `capture_mode: "backfill"` (absence means live) and
  `regime_collection_audit.py` reports live vs backfilled counts separately.
  Tier C (frozen snapshots, opening-range baselines, never-started
  predictions) is untouched. Step 5, the flagged preview lane, was not built.
- **Outstanding for 13c:** the mid-session restart drill (audit HEALTHY with a
  nonzero backfill count) is the remaining half of the exit gate, so 13c is
  not `LIVE_VALIDATED`. The task registration is **done** —
  `scripts/register_0700_autostart.ps1` was re-run on 2026-08-08 (06:00 PT,
  Mon-Fri, repeating every 15 min for 7.5h, launching from
  `C:\Users\Aaron\TradingBotV3`), and the fire-while-running drill passed:
  with a real python-launched desk up, both the task and the hardened guard
  reported "already running - nothing to do" and no second desk started.


## Previous checkpoint (main, 2026-08-03 evening)

- Branch `main` (2026-08-03 evening, merged from
  `ultimate-setup-database-plan`). Gate: **1814 passed, 5 subtests** (adds
  `tests/test_warehouse_config.py`); smoke **7/7**.
- `docs/ULTIMATE_SETUP_DATABASE_PLAN.md` is now the LOCKED implementation plan
  for the DAS research warehouse (Fabel ultracode review of the 2026-08-03
  draft: capture policy with IB pacing budgets, 13-table frozen schemas,
  Phases 0-8 build order, 28 locked decisions + a 6-item trader confirmation
  register). plan.md sec 12 gained trader-directed item **13a** scoping
  Phases 0-8 as shadow-only additive evidence capture.
- Warehouse **Phase 0 landed**: decision record
  `docs/decisions/0014-das-research-lake.md` (DAS lake = new append-only
  storage class; Drive stays operational-only) and
  `scripts/research_warehouse/config.py` (`research_store_dir` setting +
  `TRADINGBOTV3_RESEARCH_DIR` override, refusal of Drive-folder paths,
  `warehouse_enabled()` no-op guard, lake layout bootstrap, machine-local
  `research_spool` path). Phases 1-8 are code-complete on this branch; what
  remains is live work: wire the tee + Health tiles into the running desk
  (BD-20), then run the 20-session pilot (BD-52).

## Previous checkpoint (main, 2026-08-03 midday)

- Branch `main`, merge commit `29435d1` (2026-08-03). Combined gate:
  **1806 passed, 5 subtests**; smoke **7/7**. Working tree was clean and
  `origin/main` matched after push.
- `launch_gui.py` is now the only operator launcher. Obsolete GUI/satellite
  `.cmd`/`.command` wrappers were removed. Main versus Satellite is selected
  under Settings ▸ Desk Link and applied through a clean automatic restart;
  pairing host/port/token lives on that same page. Settings is split into
  scroll-safe General, BounceBot, and Desk Link tabs.
- Away's `autopilot_today.txt` is the one phone-facing Drive digest: verified
  safety/freshness first, numbered **BEST SWING TRADES** first among candidate
  content, then intraday opportunities and an `== OPERATIONS ==` tail. The
  atomic publish and last-good recovery code was not changed by the merge.
- Desk Link remains active in every Auto profile. The sticky snapshot carries
  the main Auto mode, and a satellite holding the Tier 2 lease can switch
  OFF/DESK/AWAY/EVENING through idempotent `set_auto_mode`.
- Focus price-alert Phases A–D are **IMPLEMENTED + GREEN**. Focus and Research
  share one main-only poll/push service; every crossing pushes to ntfy at
  urgent priority first, then produces persistent audible non-activating toasts
  on the main, full satellite, and mirror satellite. Today's triggers are
  sticky across reconnects. Satellite alert editing remains read-only until
  Phase E. Phone/main/satellite ordering and reconnect recovery still require
  the live two-machine check, so this is not `LIVE_VALIDATED`.

## Previous checkpoint (main)

- Branch: `main`, fast-forwarded 2026-07-30 evening from
  `regime-infrastructure-phase1` (which contains all of
  `milestone-1-observability`). One combined line: Milestone 1 observability,
  writer coordination, shadow evidence, perf/crash fixes, provider telemetry,
  the D1 stale-tail backfill, and the Regime Infrastructure Phase 1
  collection packets.
- Runtime: Python 3.14.6 on the desktop (upgraded from 3.14.2 the same
  evening; the mini-PC still needs the upgrade).
- Date: 2026-07-30
- Test baseline: **1678 passed, 5 subtests passed**
  (`.venv\Scripts\python.exe -m pytest tests/ -q`, pytest exit code 0)
- Qt thread-warning gate: **1678 passed, 5 subtests passed**
  (`-W "error::pytest.PytestUnhandledThreadExceptionWarning"`,
  pytest exit code 0; no warning suppression)
- Smoke: **7/7** (`scripts/smoke_check.py`, exit code 0)
- Live validation: **IN PROGRESS** — the July 13 session verified single-owner
  scheduled scans, durable run IDs/PIDs, accurate heartbeat state, M5 completed-
  candle processing, SPY shadow coverage, Greatness shadow coverage, and the
  composed operations audit (**6/6 healthy** before the Away-report check was
  added), and the 13:00 final scan
  completed successfully in 1111.4s with one durable worker/run ID and zero
  true log errors. The persisted focus feed had 166/166 v2-tagged rows and no
  empty tag lists. The expanded eight-check audit and transactional Away publication
  still need a restarted-app/live verification. Physical failure-matrix and
  two-machine Drive drills remain outstanding.
### Auto EVENING mode + phone price alerts landed 2026-08-01 (IMPLEMENTED — not LIVE-VALIDATED)

- Auto Mode cycle is now OFF -> DESK -> AWAY -> EVENING -> OFF. EVENING is the
  sleep-in profile: discovery identical to DESK (auto-populate picks stage for
  chart approval; nothing self-applies or recommends until the mode is turned
  off), plus an early open+30 Master AVWAP swing slot, 07:00/07:15/07:30
  strength-persistence checks on staged picks (held vs faded), and a morning
  briefing (environment + best D1s by expected R + held picks + overnight
  alerts) written to `evening_briefing.txt`, folded into the hourly phone
  report (EVENING publishes hourly like AWAY), and announced via push.
- Price-level alert watchlist (Focus and Research -> Price Alerts): trader-entered
  tickers with above/below levels, polled every minute from 1m bars including
  pre/post market, pushed to phone + watch over ntfy (outbound HTTPS only; no
  ports). Each side fires once per arm then disarms; entries are never
  auto-removed; only the explicit main-desk engine monitors, with the shared
  writer gate retained as defense in depth. Every crossing is urgent. New modules: `evening_mode.py`, `price_alerts.py`,
  `push_notify.py`, `ui/services/price_alert_service.py`,
  `ui/panels/price_alerts_panel.py`; runbook at `docs/EVENING_MODE_RUNBOOK.md`.
- Not live-validated: needs one real sleep-in session (arm the night before,
  verify the 07:00 early scan, the 07:30 briefing finalization, and an actual
  level-cross push reaching the phone/watch).

### Milestone 1 packets landed 2026-07-30 (IMPLEMENTED — none LIVE-VALIDATED)

Every item below is **IMPLEMENTED + GREEN**. None is **LIVE-VALIDATED**: all
evidence is single-machine and deterministic. The plan.md sec 6.2 physical
drills (two-machine collision, lease expiry, clock skew, sleep/wake) and the
sec 6.1 first-session checklist have **not** been run.

- Champion invariance is now executable, not prose: a raising, poisoned, or
  fully-enabled shadow engine leaves legacy SPY pause state and D1 trigger rows
  byte-identical, proven behaviourally and by AST assertions. Verified to fail
  when deliberately violated.
- Milestone 3 fixture contract is **enforced** rather than merely declared;
  `numeric_tolerance` and `raw_input_sha256` are now actually applied.
- Shared diagnostics I/O (`scripts/diagnostics/artifact_io.py`) with guaranteed
  temp cleanup, replacing hand-rolled `mkstemp` copies that leaked temp files.
- SPY shadow: `last_complete_bar_at` can no longer point at a bar that never
  completed; stale-vs-incomplete are separately counted; the cross-session
  dedupe latch no longer swallows a new session's first row. Episodes are
  emitted as replayable rows. **Absence of an episode is still not evidence of
  no episode** — the hook fires per bounce-scan cycle, not per completed bar.
- Greatness shadow: `source_trigger_id` carries real D1 provenance instead of a
  fabricated hash, gated behind a characterization fixture proving the produced
  candidate is byte-identical.
- Writer coordination: designated-writer authority (machine-local), kernel-owned
  local cross-process exclusion, and a fenced fail-closed lease. Verified by an
  independent adversarial suite (scenarios A–U, ~140 cases) using real corrupt
  files. **Publishing now requires a configured designated writer on each
  machine** — `scripts/writer_role.py` switches it, and exits non-zero when the
  machine cannot publish.
- Health: UNKNOWN is a first-class status with precedence
  UNHEALTHY > DEGRADED > UNKNOWN > HEALTHY; all 13 plan.md sec 6.3 dimensions
  are emitted, with unmeasured ones reported as UNKNOWN rather than omitted, so
  a partial payload can no longer roll up to HEALTHY.
- The operations audit now **streams the real shadow JSONLs** instead of
  trusting writer-maintained sidecars; a corrupt or truncated log is visible and
  marks the evidence non-promotable.
- BounceBot startup/shutdown now has generation-owned delivery guards, bounded
  retirement, tracked late workers, and a terminal failure latch. A stopped
  generation cannot re-arm Qt timers or overwrite a replacement generation,
  and a still-retiring startup prevents a second same-client-ID connection.
- Review-learning evidence now writes to one shard per stable, machine-local
  installation ID, protected by a local cross-process lock. Readers merge the
  read-only legacy ledger with all shards, deduplicate v2 record IDs, and keep
  hostname as diagnostic metadata rather than writer authority.
- W08/W09 session evidence now rotates each shadow log before a new session or
  configuration writes, atomically publishes replay-reconciled per-session
  summaries, reports eligible/incomplete sessions and Section 7 counters, and
  enforces bounded raw/summary retention. These counters cannot promote either
  challenger.

Known gaps, deliberately not claimed as done:

- Regime Infrastructure Phase 1 collection is **IMPLEMENTED + GREEN, not
  LIVE_VALIDATED** on `regime-infrastructure-phase1`: Technical Integrity
  appends +30/+60/+90 outcome windows and live-only 10:30/12:00 snapshots,
  SPY first-hour/ATR baseline evidence, and an independent verified-breadth
  M5 ledger; standalone adaptive Laguerre RSI is pure and golden-tested. The
  first live session must pass `scripts/regime_collection_audit.py`. Evidence
  remains **EXPLORATORY / NON-PROMOTABLE** until at least 40 instrumented
  sessions (60 preferred) support a point-in-time predictive study.
- Provider telemetry is **IMPLEMENTED + GREEN, not LIVE_VALIDATED**: schema-v2
  counters (lookup / cache-hit / per-provider attempt, success, failure /
  pacing-class throttle / fallback) are captured at the IBKR, Yahoo and
  Nasdaq boundaries across the declared inventory (daily_bars, intraday_bars,
  symbol_metadata, earnings_dates, earnings_calendar, theta_options) with a
  completeness contract - partial coverage, capture errors, orphan events or
  malformed values can never grade healthy, and failure ratios only ever use
  matching per-provider attempt denominators. The Health row honestly reports
  **UNKNOWN** until the first instrumented scan writes a manifest, and no
  live scan has run on this build yet.
- W08/W09 has not crossed a real session/configuration rollover on this build.
  The current audit therefore honestly reports zero finalized summaries and
  zero eligible sessions for both engines; the first restarted live session
  must validate rotation, counter preservation, and reconciliation.
- The review-log partition has not emitted a live v2 shard yet. The read-only
  survey found 491 valid legacy rows, zero malformed rows, and two real active
  installations (mini-PC `MainPC`, desktop `DESKTOP-IABHR62`) with no observed
  temporal/session overlap. The legacy file is now read-only; a restarted GUI
  must validate per-install writes and merged reads on both machines.
- No lease renewal loop; clock skew beyond the supported grace, Drive sync
  convergence, and the sub-millisecond pre-replacement window remain open and
  are documented in plan.md sec 4 and the runbook.

- Shadow engines: SPY state and Greatness both logging; neither is promoted.
- Registry adoption: open scan, auto-populate, and near-extreme writers now
  dual-write in shadow; text watchlists remain authoritative.
- Aggressive auto-populate: completed-M5 repeated HOD/LOD pressure and
  legacy-SPY-pullback extreme holders now feed regime-inverted long/short
  candidates, while scheduled rotation and triple-VWAP invalidation remain
  unchanged. Golden coverage here is narrow, not general: exactly one pure
  function, `autopilot_core.build_aggressive_regime_candidates`, is pinned to
  `tests/fixtures/aggressive_watchlist_candidates_v1.json` (consumed by
  `tests/test_auto_populate_watchlists.py`). The surrounding writer, rotation,
  and invalidation paths have no golden fixture.
- Technical Integrity v1: a research-only completed-M5 level-respect score now
  publishes market/sector/industry/stock hierarchy plus bullish/bearish break
  pressure. Auto regime and Technicals are visible on every GUI page; hover
  shows coverage and strongest/weakest industries, while clicking opens the
  searchable full hierarchy. Append-only predictions and
  outcomes feed daily plus on-demand point-in-time calibration, with a 100-event
  / 5-session / both-outcome review floor and no automatic tuning path.
- Scanner reliability: compact tracker scoring snapshot, true trigger-date
  freshness, invalid-side history sanitation, shared output-computation reuse,
  atomic signal/feature/watchlist writes, and detailed output timings landed.
- Away reliability: report + metadata publish transaction, hash/readback
  audit, honest GUI status, phone-sized scan/tracker health, hour-aligned
  07:00-through-close reports, persisted Away profile, simulated
  expiry/clock-skew/sleep-wake/failure tests, and swing-first candidate
  ordering with honest empty/unscanned states landed.
- Writer ownership is fail-closed at both role resolution and lease parsing,
  with local kernel exclusion and a fenced final ownership check. It remains
  cross-machine writer protection rather than distributed exclusion until the
  physical Section 6 drills pass.
- GUI trust foundation: the Industry Board now has single-flight startup/hourly
  refresh, atomic last-good files, snapshot freshness/Health evidence, and
  numeric strongest/weakest sorting. Master focus duplicates now merge by
  opportunity thesis/anchor instead of surviving solely because their buckets
  differ.
- Bounce/entry trust: BounceBot now defaults to Auto with a separate N/A user
  mode, logs every manual environment selection beside Auto's same-moment
  evidence, makes automatic completed-bar entry monitoring explicit, and hides
  manual pullback/bounce windows under Advanced. D1 Focus now accepts only
  final Favorite/High Conviction bucket upgrades; developing A/S target touches
  are named and logged as research-only evidence and cannot enter Alert Center
  or Auto/Away alert summaries. Generic champion D1 flags remain unchanged.
- Research/journal UX: every Setup Tracker, Day Trade Tracker, and Move
  Forensics row now has the same novice-friendly execution/evidence explanation;
  Setup Tracker summarizes qualified leaders in plain English. Journal schema
  v2 preserves append-only opportunity lifecycle events plus structured GUI
  reviews, with broker-derived Taken/Closed events idempotent across rebuilds.
- Optional A.I. review: a top-level A.I. Summary tab now attaches either
  ChatGPT/OpenAI or Claude/Anthropic to explicit user-selected evidence scopes.
  It previews the exact bounded package, stores saved keys in Windows Credential
  Manager, validates structured results and every source ID, and exports the
  summary/evidence/manifest without any write path back into bot decisions.
- Advisory industry RS/RW: the RS Window now excludes forming M5 bars and
  prior-session spillover from its automatic window, declares a deterministic
  primary industry, ranks all available intraday industry composites, and shows
  industry-vs-SPY plus stock-vs-industry with member/timestamp coverage. The
  atomic snapshot feeds replay and A.I. evidence only; production scoring,
  alerts, and promotion remain unchanged pending the roadmap gates. RS Window,
  Auto Pilot, and the swing-first Drive report expose the same source-board ID
  and flag any stale-source mismatch.
- Setup tags: v2 semantic family/trigger/confirmation tags are integrated with
  provenance while preserving the raw trigger signals separately.
- Next queue: finish the Section 6 operational drills, collect/audit Technical
  Integrity live evidence before considering any tuning, complete remaining
  CandidateRegistry writer adoption, and finish Greatness readiness gates.
