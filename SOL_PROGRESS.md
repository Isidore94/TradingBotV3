# Checkpoint marker

[`plan.md`](plan.md) owns all status, roadmap, and promotion policy. The full
implemented/remaining inventory lives in Section 3, and the ordered work queue
in Section 12. This file is only the small, frequently refreshed checkpoint
stamp; it must not duplicate the roadmap.

## Current checkpoint

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
- Previous `main` gate (2026-08-03 evening, merged from
  `ultimate-setup-database-plan`): **1814 passed, 5 subtests** (adds
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
