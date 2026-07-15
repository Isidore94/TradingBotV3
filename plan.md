# TradingBotV3 Master Roadmap

## Document status

This is the single authoritative implementation and product roadmap for TradingBotV3. It consolidates the former architecture/performance plan and the former high-conviction feature plan, then rebases the remaining work on the Sol3 implementation checkpoint.

Checkpoint used for this revision:

- Branch: `Sol3`
- Commit: `20cefb3`
- Date: 2026-07-11
- Reported suite status: 802 tests passing
- Verified repository evidence: Phase 0 foundations, most Phase 1 instrumentation, Phase 2 implementation, Packets A–D, the SPY pullback challenger, and the Greatness Monitor challenger are present in code and tests.

The reported 802-test baseline should remain the minimum green baseline. Run the suite before and after each implementation packet and record the exact count in the relevant commit or run manifest.

This roadmap distinguishes four facts that must never be collapsed into one status:

1. **Implemented** — code exists.
2. **Test-green** — deterministic automated tests pass.
3. **Live-validated** — behavior has been observed under real session, machine, provider, and storage conditions.
4. **Promoted** — the feature is authorized to affect production suggestions or user-facing decisions.

A shadow feature can be implemented and test-green without being live-validated or promoted.

---

## 1. Mission and product boundary

TradingBotV3 should be a fully automated trading decision-support system that does everything except execute orders.

It should:

- prepare the market before the session;
- discover high-quality swing and day-trade candidates;
- continuously monitor developing opportunities;
- determine which names are proving relative strength or weakness during meaningful SPY and sector tests;
- identify when a stock moves from “interesting” to genuinely actionable;
- present the best opportunities in the main Qt GUI;
- operate unattended in Auto/Away mode;
- publish an honest, current phone report through Google Drive;
- journal signals, decisions, and outcomes;
- measure setup expectancy without look-ahead or selection bias;
- help discover and validate new setups through controlled research.

The application does **not** execute trades. Broker order routing, consumer distribution, and commercial shipping are outside the current roadmap. Broker data imports may remain read-only inputs to the journal.

### Operating model

- The main Qt GUI is the primary product and composition root.
- Auto Mode is a central application state, not a collection of unrelated checkboxes.
- `OFF`, `AUTO-DESK`, and `AUTO-AWAY` must remain truthful and globally visible.
- The desk machine is the primary interactive runtime.
- The mini-PC is an optional unattended worker, not a separate product architecture.
- Both machines may read shared exports, but mutable shared outputs require explicit ownership.
- Desktop and Away mode must consume the same canonical opportunity snapshot.
- No feature may silently promote itself from research or shadow mode into production decision-making.

---

## 2. Status vocabulary and promotion rules

Use these labels in commits, the Health page, `SOL_PROGRESS.md`, and future handoffs.

| Status | Meaning | May affect live suggestions? |
|---|---|---:|
| `PLANNED` | Designed but not implemented. | No |
| `IMPLEMENTED` | Code exists but may lack complete tests. | No |
| `GREEN` | Deterministic tests pass. | Only if it is preserving existing production behavior |
| `SHADOW` | Runs on live inputs but cannot affect production output. | No |
| `LIVE_VALIDATED` | Passed documented real-session or operational acceptance checks. | Not automatically |
| `ADVISORY` | Visible in a clearly labeled research/advisory surface. | No loud alerts or ranking authority |
| `PROMOTED` | Approved as the production champion with rollback available. | Yes |
| `RETIRED` | Replaced or intentionally disabled. | No |

Promotion requires all of the following:

- a versioned configuration;
- deterministic tests;
- replayable evidence;
- defined success and failure metrics;
- live-session evidence across relevant regimes;
- comparison with the current champion;
- a rollback switch that does not require a code revert;
- explicit approval in a commit or release note.

Agreement with the legacy implementation is diagnostic, not the definition of correctness. The challenger is allowed to disagree, but the disagreement must be explainable and outcome-tested.

---

## 3. Current Sol3 state

### 3.1 Implemented and green

#### Phase 0 — baseline and dormant defects

- Test baseline restored to green.
- Dormant `datetimde` defect fixed with regression coverage.
- Duplicate top-level definition guard added.
- Deprecated Qt proxy invalidation replaced.
- Deterministic smoke command added.
- Entrypoint and lazy-import hygiene improved.
- Child processes and owned threads now have bounded shutdown behavior.
- Dependency metadata, pytest markers, narrow lint gates, and pinned constraints exist.

#### Phase 1 — core observability

- Master scans emit run manifests on success and failure.
- Phase timings and counters are recorded.
- Bounded local history exists.
- Strict side coercions and data-quality events are countable.
- The in-app Health page composes runtime, Away report, shadow, registry, and
  Industry Board freshness evidence.

Still remaining from Phase 1:

- stable benchmark/golden fixtures;
- trend reporting for timings, failures, provider calls, and coverage.

#### Phase 2 — unattended runtime implementation

- Durable job ledger with typed states.
- Bounded retry budgets by error class.
- Restart replay and stale-run marking.
- Scan-child ownership and bounded reaping.
- Verified atomic Away report replacement.
- Transactional Away report plus verification-metadata publication with
  readback validation and rollback of the previous verified pair.
- Hash readback verification and bounded report archive.
- Honest freshness headers and last-attempt versus last-success state.
- Swing-first phone-report ordering: the safety/freshness header remains first,
  then current swing opportunities lead every intraday candidate section;
  empty current-session and not-yet-scanned states are distinguished.
- Runtime heartbeat.
- Cross-machine writer-lease mechanism.
- Global `OFF` / `AUTO-DESK` / `AUTO-AWAY` header and service semantics.

Phase 2 is **implementation-complete**, not yet fully **live-validated**. Section 6 defines the remaining operational acceptance drills.

#### Packets A–D — pure foundation engines

- Runtime lifecycle and stop ownership.
- Side-symmetric SPY market-state engine.
- Timestamp-aligned relative-strength engine.
- Candidate registry with source leases, provenance, transitions, and atomic versioned persistence.

The registry has initial shadow adoption, but all legacy live writers have not yet migrated to authoritative sources.

#### Correctness and platform work already landed

- Stable digest sampling.
- Strict `Side` parsing and counted legacy coercions.
- Universe freshness based on the oldest required file.
- Lazy legacy-Tk import.
- Verified report publishing.
- Truthful global Auto profile.
- One single-flight Industry Board owner refreshes stale data at startup and
  hourly, preserves last-good outputs atomically, exposes snapshot freshness in
  Health, and supports true numeric strongest/weakest sorting.
- Master focus rows deduplicate by opportunity thesis/anchor rather than bucket,
  while preserving High Conviction/Favorite classification badges.
- BounceBot starts with Auto owning the active regime and the user's selector
  at N/A. Manual environment choices are session-only overrides recorded to an
  append-only learning log with the contemporaneous Auto reading; selecting N/A
  clears the override without hiding what Auto believes.
- Entry Assist presents completed-bar automatic pullback/bounce monitoring as
  the normal path. Strongest, weakest, and movers 30m remain on demand, while
  manual window controls live under Advanced diagnostics.
- D1 Focus routes final Favorite/High Conviction bucket-upgrade events only;
  armed/level-cross triggers remain developing evidence in the live stream.

### 3.2 Implemented and running in shadow

#### SPY pullback engine

Current role:

- consumes cached completed SPY M5 bars;
- runs beside the legacy pause detector;
- records state and agreement changes to `spy_state_shadow.jsonl`;
- cannot change live alerts, candidates, or ranking.

What it proves today:

- the pure state engine can run in the live runtime;
- state transitions and legacy disagreements can be collected safely.

What it does not yet prove:

- that episode timing is better than the legacy champion;
- that live bar completeness and timezone handling are always correct;
- that the engine improves stock selection or entries;
- that sector and candidate RS integration is production-ready.

#### Greatness Monitor

Current role:

- converts existing D1 trigger levels into persistent confirmation plans;
- consumes completed intraday bars through the existing D1 evaluation path;
- distinguishes touch, wick, close, acceptance, retest, failure, re-arm, and readiness;
- persists candidate state across restarts;
- records transitions to `greatness_shadow.jsonl`;
- cannot change existing D1 alerts.

What it proves today:

- a wick no longer has to equal confirmation;
- confirmation can be modeled as an ordered lifecycle;
- failed attempts can re-arm instead of consuming the day’s trigger;
- state can survive refreshes and restarts.

What it does not yet prove:

- continuous coverage independent of the legacy D1 scan cadence;
- correct same-day plan revisions or side changes;
- full multi-level D1 plan quality;
- RS, sector, volume, reward/risk, freshness, and anti-chase gates;
- superior alert precision or timeliness;
- production-ready candidate identity and lifecycle migration.

### 3.3 Not implemented or not complete

- Phase 1 Health page and benchmark fixtures.
- Phase 2 real-machine acceptance drills.
- Phase 3 storage reclassification, journal migration off Drive, and OS credential storage.
- Phase 4 provider repository, staged scanner, batching, and request coalescing.
- Point-in-time correctness work for moving levels, history keys, backfill leakage, tracker identity, score ordering, and factor horizons.
- Full CandidateRegistry authority over all candidate writers.
- Production integration of SPY state and relative strength across all surfaces.
- Greatness fast lane, complete readiness gates, mini-chart radar, and alert ladder.
- Point-in-time research and walk-forward promotion pipeline.
- Canonical opportunity ranking and deduplication.
- Opportunity Command Center and unified inbox.
- Market Prep Qt consolidation.
- Complete journal/outcome linking and feedback loop.
- Advanced setup research program.
- Full CI, packaging, and consumer-grade release work.

---

## 4. Assessment of the work so far

Fable implemented the correct half first. Runtime ownership, deterministic pure engines, manifests, a ledger, atomic publication, and explicit shadow mode are prerequisites for safe feature work. Building the SPY and Greatness challengers without immediately promoting them follows the methodology required by this roadmap.

The next risk is no longer lack of code. It is confusing green tests or the first interesting live examples with promotion evidence.

### Important cautions

#### The writer lease still needs adversarial validation

The lease substantially reduces accidental collisions, but a Drive-synchronized file is not automatically a distributed lock:

- acquisition currently reads state and then atomically replaces the lease file;
- two machines can race before synchronization converges;
- unexpected lease errors now fail closed; the live Drive behavior still needs
  the two-machine validation below;
- clock skew, sleep/wake, stale files, and delayed sync can change takeover behavior.

Until the two-machine drills in Section 6 pass, describe this as cross-machine writer protection, not a proof that clobbering is impossible.

#### Shadow coverage is inherited from the champion path

The Greatness hook currently runs when the legacy D1 trigger path evaluates a symbol. It therefore cannot yet prove that the proposed dedicated priority lane would have observed every confirmation on time. Add evaluation-coverage records before using absence of an event as evidence.

#### Current Greatness readiness is lifecycle readiness

The current readiness state is mainly based on confirmation-plan steps. The final product definition of `READY` must also include:

- SPY/sector context;
- relative strength or weakness;
- volume/participation quality;
- reward to the next real obstacle;
- logical invalidation;
- extension/no-chase state;
- data freshness and completeness;
- setup-specific hard risk gates.

Do not route current shadow `READY` events directly into loud production alerts.

#### Shadow logs must be auditable, not merely present

Each shadow artifact needs:

- schema and engine version;
- configuration hash;
- session and machine identity;
- completed-bar timestamp and evaluation timestamp;
- candidate/episode identity;
- enough inputs to replay the transition;
- coverage and error counters;
- daily summary and retention policy.

Improve the logs before accumulating weeks of evidence that cannot answer promotion questions.

---

## 5. Non-negotiable system invariants

These rules apply to every remaining phase.

### Data and time

- Only completed bars may satisfy completed-bar confirmation rules.
- All timestamps must have an explicit timezone and session interpretation.
- Every decision must carry an `as_of` time and data-health state.
- Market, sector, and stock comparisons must use aligned intervals.
- Missing data is uncertainty, never silent confirmation.
- Point-in-time research must use only information available at the simulated decision time.

### Identity and provenance

- Opportunity, candidate, setup, trigger, trade, and outcome identities must be stable and distinct.
- Every feature must record source, version, and calculation horizon.
- Manual/pinned candidates receive attention, not an unearned model-quality boost.
- Rescans and GUI refreshes must not erase valid lifecycle progress.

### Runtime and publication

- One component owns each timer, thread, process, job, and mutable export.
- Shutdown is bounded and testable.
- Retries are classified, bounded, and visible.
- A failed publish never destroys the last verified report.
- `OFF`, `DESK`, and `AWAY` accurately describe what work and publishing are active.
- The GUI and phone report consume the same versioned snapshot.

### Research and promotion

- Champion and challenger outputs remain separate until promotion.
- Historical artifacts are immutable and versioned.
- No tuning on the same outcomes used to claim improvement.
- Every policy change has a baseline, shadow period, acceptance gate, and rollback.
- Selected opportunities and rejected candidates are both retained to avoid selection bias.

### Product behavior

- The system may honestly recommend zero trades.
- A high score cannot cancel a hard risk or stale-data failure.
- A correct thesis that is too extended is `NO_CHASE`, not a top recommendation.
- Alerts explain what changed, what remains, invalidation, and actionability.
- No broker execution is added under this roadmap.

---

## 6. Live validation program

Automated tests establish logic. Live validation establishes that clocks, feeds, files, providers, machines, session boundaries, and user workflows behave as expected.

### 6.1 First-session checklist

Before the session:

1. Confirm branch and expected commit.
2. Confirm the worktree is clean or document intentional changes.
3. Run the deterministic smoke command.
4. Run the relevant focused tests; run the full suite if the baseline was not verified after the latest commit.
5. Record engine/config versions for SPY and Greatness.
6. Archive or rotate prior shadow logs without deleting them.
7. Confirm free disk space and diagnostics directory writability.
8. Start the main GUI early enough to observe premarket-to-open transitions.
9. Confirm Auto Mode state is the one actually intended.
10. Confirm no older TradingBot/scanner process is still running.

During the session:

- glance at `heartbeat.json` at open, midmorning, midday, and late day;
- verify its timestamp advances approximately every 30 seconds under normal operation;
- verify `current_job`, `next_job`, and `last_success` are credible;
- verify `spy_state_shadow.jsonl` advances only on meaningful state/agreement changes;
- verify `greatness_shadow.jsonl` records coherent transition chains;
- note visible SPY pullbacks, false breaks, retests, and standout RS/RW names for later comparison;
- do not tune thresholds in response to one live example;
- record data outages, delayed bars, restarts, sleep/wake events, and manual interventions.

After the session:

1. Stop through the normal GUI path.
2. Verify owned child-process count returns to zero.
3. Verify no scanner or worker remains orphaned.
4. Preserve the run manifest, heartbeat, job ledger, shadow logs, verified Away report metadata, and relevant market-data snapshot.
5. Run a daily audit summarizer.
6. Manually inspect every shadow transition during the first few sessions.
7. Record observations without changing champion behavior.

### 6.2 Phase 2 operational acceptance matrix

Run destructive or disruptive drills outside important market windows.

| Component | Controlled live test | Pass condition | Failure response |
|---|---|---|---|
| Heartbeat | Normal session, long scan, temporary provider stall, clean shutdown | Updates under normal runtime; job field explains long work; stops after process exits | Add stale/hung classification and Health alerting before relying on it remotely |
| Job ledger | Force one transient and one permanent test failure | Retry budget respects error class; permanent failure does not loop; restart marks stale work correctly | Keep legacy scheduling champion and fix ledger semantics |
| Child ownership | Start a scan, close GUI during work, restart app | Owned child exits within bounded grace/terminate path; no duplicate scan after restart | Treat as P0 regression |
| Atomic publish | Simulate render error, write error, and readback mismatch | Prior verified report remains intact; status reports failure honestly | Block Away promotion |
| Writer protection | Start Desk and mini-PC publishers nearly simultaneously | Exactly one verified writer wins; loser reports holder and does not overwrite | Change to fail-closed and strengthen coordination protocol |
| Lease expiry | Stop holder without release, wait through controlled TTL, start second machine | Takeover occurs only after defined expiry and is visible in metadata | Fix TTL/clock/recovery design |
| Clock skew | Compare machine clocks and simulate bounded skew | No premature takeover or indefinite lockout within supported skew | Add server-time/monotonic or explicit operator recovery |
| Sleep/wake | Sleep current holder and resume after TTL | Ownership and report freshness remain truthful | Require reacquisition before every publish |
| Freshness header | Stop successful work while runtime remains alive | Report says stale even though heartbeat is healthy | Fix freshness derivation |
| Shutdown | Close during startup, scan, and idle | Timers, workers, threads, and children all end; state remains readable | P0 rollback/fix |

### 6.3 Health page acceptance

The Health page should expose, without opening files manually:

- runtime profile and machine identity;
- heartbeat age;
- current and next job;
- last attempt and last verified success per job/export;
- job-ledger failures and exhausted retries;
- owned process/thread counts;
- writer-lease holder and expiry;
- report freshness and verification state;
- provider request, cache-hit, throttling, and failure counts;
- universe and market-data freshness;
- most recent scan manifest and phase timings;
- SPY and Greatness shadow engine versions, last evaluations, coverage, and errors;
- disk/storage warnings.

The page must show `UNKNOWN` when evidence is absent. It must not convert missing telemetry into a green state.

---

## 7. Shadow evidence and promotion ladder

### 7.1 Shared ladder

Every decision engine follows this order:

1. **Pure tests** — deterministic state, symmetry, persistence, and edge cases.
2. **Replay shadow** — historical sessions with no production influence.
3. **Live shadow** — real-time inputs, append-only evidence, no decisions.
4. **Audit** — labeled event review and champion/challenger comparison.
5. **Advisory UI** — clearly labeled challenger output visible to the user.
6. **Opt-in soft alerts** — non-ranking, non-loud notifications.
7. **Limited canary** — challenger affects a bounded surface with rollback.
8. **Promotion** — becomes champion only after gates pass.

Never jump directly from live shadow to loud alerts.

### 7.2 SPY pullback engine evidence plan

#### Improve logging first

Add:

- session ID, engine version, configuration hash, and machine ID;
- explicit episode IDs;
- impulse start/high/low, counter-move start, depth, stabilization, resumption, and failure timestamps;
- completed-bar timestamp and evaluation lag;
- stale/incomplete-bar counters;
- legacy detector output and reason;
- daily state-duration and transition summary;
- sufficient feature values to replay every transition.

#### Label live episodes

For each meaningful SPY impulse/counter-move/resumption:

- mark whether the trend premise was valid;
- mark whether the pullback was controlled, failed, or actually reversed regime;
- record false flips and missed episodes;
- measure onset and resumption timing versus the legacy detector;
- connect the episode to candidate RS/RW outcomes.

#### Initial evidence floor

Before advisory integration, collect at least:

- ten substantially complete sessions;
- multiple trend, chop, gap, low-volume, and reversal regimes;
- at least 30 manually reviewed meaningful counter-move episodes in total;
- evidence from both bullish and bearish sides, or explicitly limit the first advisory release to the validated side;
- clean behavior across open, midday, and late-day periods;
- no unexplained incomplete-bar or timezone transitions.

These are minimum evidence floors, not automatic promotion thresholds.

#### Metrics

- meaningful-episode precision and recall;
- false state-flip count;
- median and tail transition latency;
- stability/hysteresis under chop;
- stale-data behavior;
- percent of episodes with aligned stock/sector data;
- downstream improvement in leader/laggard ranking;
- champion/challenger disagreement outcomes.

#### Promotion gate

The SPY engine can enter advisory mode when transitions are reproducible, failure modes are understood, and it improves episode labeling or candidate timing without unacceptable added delay. It can become champion only after downstream ranking is replay-tested and live-canary results remain favorable.

### 7.3 Greatness Monitor evidence plan

#### Improve identity and coverage first

Before interpreting the log statistically, add:

- candidate ID including symbol, side, setup family, session, and plan version;
- plan-revision events when D1 levels change;
- evaluation records or daily counters for candidates checked, bars consumed, bars skipped, and reasons;
- monitoring cadence and last complete bar per candidate;
- source D1 trigger IDs and level provenance;
- engine/config version and data-health state;
- outcome linkage after each transition.

The current `symbol|session_date` storage key is insufficient if a side or plan changes during the session.

#### Manually audit transition chains

Review:

- wick through and immediate failure;
- close through with no hold;
- acceptance over multiple bars;
- clean break and retest;
- failed attempt followed by valid re-arm;
- repeated failures reaching the attempt cap;
- invalidation;
- restart persistence;
- same-day D1 plan change;
- long/short mirror behavior;
- late/extended confirmation.

#### Initial evidence floor

Before advisory cards:

- review the first 25 complete candidate transition chains manually;
- collect at least 50 meaningful level interactions;
- include at least 20 confirm/fail/re-arm outcomes;
- cover multiple setup families and both sides, or limit scope to validated families/sides;
- demonstrate that shadow coverage is timely enough for the claimed cadence;
- reproduce every reviewed event from stored bars and the recorded plan.

Before any loud `READY` alert, the full gate stack in Section 11 must exist and pass replay/live-canary evaluation.

#### Champion comparison

For every legacy wick alert, calculate:

- whether Greatness called it a test, failure, confirmation, or no event;
- subsequent favorable and adverse excursion;
- whether a later confirmation occurred;
- remaining reward/risk at both alert times;
- notification delay caused by close/accept/retest requirements;
- whether the later alert was still actionable;
- setup-family and market-regime context.

The objective is not merely fewer alerts. It is higher precision and expectancy while preserving enough entry opportunity.

---

## 8. Remaining roadmap: ordered milestones

The following order is dependency-driven. Do not start a later milestone merely because its UI is more visible.

## Milestone 1 — Finish observability and validate the runtime foundation

### Work

- Build the main Qt Health page described in Section 6.3.
- Add benchmark fixtures for scan duration, provider calls, cache hits, output write time, symbol coverage, and memory.
- Add a daily shadow/operations audit command that summarizes all three new artifacts and identifies schema, parse, timing, and coverage problems.
- Refresh `SOL_PROGRESS.md` at each promotion checkpoint.
- Run the Phase 2 operational acceptance matrix on the desk and mini-PC.
- Change writer coordination to fail closed for shared mutable output unless a deliberately configured emergency override is active.
- Document manual lease takeover and recovery.

### Live testing

- At least two normal full sessions.
- One controlled long-scan session.
- One off-hours failure/restart drill.
- One two-machine collision, expiry, and takeover drill.

### Exit gate

- Health state matches source artifacts.
- No orphan process or unbounded retry.
- Report freshness remains truthful through failures.
- Writer-protection limitations are resolved or explicitly bounded.
- Baseline performance is captured and reproducible.

---

## Milestone 2 — Supervised storage and secrets migration

This phase changes live trading data and must be performed while the user is present.

### Target classification

#### Machine-local authoritative data

- journal database;
- job ledger;
- run manifests and detailed diagnostics;
- caches;
- transient candidate state;
- provider response cache;
- secrets and credentials through the OS credential store.

#### Shared/read-mostly exports

- verified Away report;
- optional chart contact sheet;
- immutable dated summaries;
- explicitly versioned configuration that is safe to synchronize.

#### Never synchronized as a live database

- SQLite journal write-ahead files;
- actively mutated ledgers;
- temporary files;
- lock files intended for local process coordination;
- plaintext credentials.

### Migration sequence

1. Inventory every state path, writer, reader, format, size, and retention policy.
2. Classify authority and replication direction.
3. Stop all writers.
4. Create timestamped backups and checksums.
5. Validate current journal integrity and row counts.
6. Copy to the new local location; never move first.
7. Run schema migration against the copy.
8. Start in dual-read/old-write validation mode if practical.
9. Compare counts, key aggregates, recent trades, screenshots/attachments, and imports.
10. Switch authority explicitly.
11. Keep the old store read-only through an agreed rollback window.
12. Export immutable/sanitized summaries to Drive rather than synchronizing the live DB.
13. Move secrets to the Windows credential store and remove plaintext fallbacks only after retrieval is verified.

### Live testing

- Open, search, edit, and close the journal through the GUI.
- Import a representative broker file twice and verify idempotence.
- Restart the application and verify the same records.
- Simulate unavailable Drive and confirm the local journal remains fully functional.
- Restore a backup into a temporary location and verify it is usable.

### Exit gate

- No authoritative database is mutated by two machines.
- Migration reconciliation is exact or differences are explained.
- Rollback is tested.
- Secrets are not present in tracked files, logs, reports, or Drive exports.

---

## Milestone 3 — Golden-result and replay harness

This is the prerequisite for provider restructuring and the remaining correctness changes.

### Required fixture layers

#### Characterization fixtures

Capture current behavior exactly, including known quirks. These protect against accidental changes during refactoring.

#### Corrected expectation fixtures

For each known correctness defect, create small synthetic or hand-audited cases that define the desired behavior. These make intentional differences explicit instead of silently blessing current bugs.

#### Representative full-scan fixtures

Include:

- ordinary trend day;
- choppy day;
- gap/earnings-heavy day;
- partial/missing-data case;
- long and short candidates;
- multiple setup families;
- replayed Master output, tracker updates, and opportunity ranking inputs.

### Fixture contract

Each fixture records:

- raw input hashes and acquisition times;
- universe version;
- configuration and feature versions;
- provider/calendar assumptions;
- exact `as_of` time;
- expected outputs in stable sorted form;
- allowed numeric tolerances;
- intentional-difference approval notes.

### Exit gate

- Tests can replay a scan without network access.
- Refactors can prove semantic equivalence.
- Correctness fixes produce reviewed, narrow diffs.
- Performance comparisons use the same inputs.

---

## Milestone 4 — Provider repository and staged scanner

### Architecture

Create one provider/repository layer responsible for:

- request normalization;
- bounded concurrency and rate budgets;
- retries and backoff by error class;
- exact interval/session semantics;
- cache keys and freshness;
- request coalescing;
- batching where supported;
- response validation;
- provenance and coverage metrics;
- fixture/replay substitution.

### Staged Master scan

Use a funnel:

1. Load universe and cheap daily features once.
2. Apply liquidity, price, freshness, and coarse structure gates.
3. Compute expensive anchors/indicators only for survivors.
4. Run setup-family detectors on eligible subsets.
5. Build shared opportunity features once.
6. Rank, persist, and publish from canonical records.

### Performance targets

Set targets only after Milestone 1 benchmarks. Measure:

- wall-clock scan time;
- provider calls by endpoint;
- duplicate request rate;
- cache hit rate;
- symbols reaching each stage;
- feature and output time;
- peak memory;
- output equivalence.

### Live rollout

- Run legacy and staged scanners on identical snapshots.
- Compare exact outputs with the golden harness.
- Canary the repository beneath one low-risk scanner first.
- Expand by provider endpoint and feature family.
- Preserve a configuration rollback to the legacy path until multiple full sessions pass.

### Exit gate

- No unexplained result changes.
- Provider calls and total runtime improve materially against the captured baseline.
- Partial provider failures yield explicit incomplete status rather than a misleading complete scan.

---

## Milestone 5 — Point-in-time correctness and research repair

Correct historical evidence before using it to tune ranking or promote setups.

### Priority order

1. **Moving-level look-ahead** — reconstruct levels using only data available at each historical timestamp.
2. **Same-day history contamination** — prevent current-day scans from masquerading as prior-day evidence.
3. **Adaptive backfill leakage** — freeze policies and training windows before evaluating outcomes.
4. **Tracker identity collisions** — give setup occurrences stable event IDs; stop treating repeated scans as independent samples.
5. **Score/bucket ordering** — compute one canonical score, then assign buckets from the same versioned result.
6. **Factor horizons** — define prediction horizon and outcome target for each factor.
7. **Multiple testing** — account for repeated setup/factor searches and small samples.
8. **Auto-tuning leakage** — tune on training windows, select on validation, report once on untouched test periods.
9. **Universe/industry survivorship and freshness** — version membership and coverage point in time.
10. **Detector semantics** — correct setup-specific direction, anchors, and expiration rules.

### Data policy

- Never overwrite prior research results after logic changes.
- Version feature definitions, setup rules, outcome rules, and universe snapshots.
- Preserve both selected and rejected candidates.
- Separate discovery, validation, and final test periods.
- Report sample size, missingness, turnover, and regime coverage with every performance claim.

### Live testing

These changes should first run as parallel research outputs. Compare live candidate deltas, but do not use live outcomes to retune the same release.

### Exit gate

- Historical replay is point-in-time correct.
- Repeated observations of one setup are not counted as independent trades.
- Every intentional change from the characterization baseline is reviewed.
- Walk-forward results can be reproduced from immutable inputs.

---

## Milestone 6 — Make the foundation engines authoritative

### CandidateRegistry adoption

Migrate writers one at a time:

- open scan;
- auto-populate;
- D1/Master watch candidates;
- near-extreme candidates;
- human focus lists;
- VWAP and expiry removals;
- Setup Tracker lifecycle updates.

Rules:

- each source owns only its own lease;
- user entries cannot be deleted by automation;
- expiry and removal are typed transitions;
- text files become projections/exports, not competing stores;
- migration runs in dual-write comparison before authority switches.

### SPY and RS integration

After shadow evidence passes:

- publish one canonical market-state snapshot;
- publish one canonical SPY pullback episode record;
- compute stock and sector RS/RW over aligned episode windows;
- eliminate duplicate RS implementations across regime pause, Entry Assist, environment scan, and RS Window;
- calculate multi-window RRS in one pass;
- retain component scores and named penalties for explanation.

### Greatness dedicated priority lane

- Promote `NEAR_TRIGGER`, `TESTING_LEVEL`, and `CONFIRMING` candidates to every-completed-M5-bar monitoring.
- Keep broad `DISCOVERED` candidates on a slower cadence.
- Bound the fast pool by provider/runtime budget.
- Use cached shared bars; never fetch during chart rendering.
- Record reduced-cadence decisions when capacity is full.
- Decouple Greatness coverage from the legacy one-wick trigger path.

### Live testing

- Dual-read/dual-write registry comparisons.
- Shadow comparison of old and canonical RS on all four current surfaces.
- Fast-lane timing measurements versus actual completed bars.
- Restart, plan revision, side change, expiry, and capacity-limit drills.

### Exit gate

- One authoritative candidate lifecycle exists.
- All surfaces agree on SPY state and RS evidence for the same snapshot.
- Greatness evaluates near-trigger names on the promised cadence.
- Legacy text/watchlist adapters can be removed without losing user behavior.

---

## Milestone 7 — High-conviction readiness and canonical opportunity ranking

### Canonical opportunity model

Each opportunity should include:

- stable opportunity and candidate IDs;
- symbol, side, setup family, session, and thesis version;
- discovery source and reasons;
- ordered confirmation plan;
- current lifecycle stage;
- market snapshot and SPY episode ID;
- stock-versus-SPY and stock-versus-sector evidence;
- volume/participation evidence;
- entry zone, invalidation, obstacle, targets, and expected R;
- extension/actionability state;
- readiness, confidence, and hard-gate results;
- data freshness, coverage, and provenance;
- alert and user-feedback history.

### Greatness lifecycle

```text
DISCOVERED
  -> DEVELOPING
  -> NEAR_TRIGGER
  -> TESTING_LEVEL
  -> CONFIRMING
  -> READY
  -> ACTIVE / EXTENDED / EXPIRED

Side paths:
  FAILED -> REARMED
  any valid state -> INVALIDATED
```

An hourly Master scan defines or materially revises the thesis. Incremental intraday updates advance the plan. A UI refresh or ordinary rescan must not reset it.

### Independent score families

Keep separate, visible components:

1. Structural quality.
2. Live confirmation.
3. SPY/sector context.
4. Relative strength/weakness.
5. Participation/volume.
6. Entry quality and tradeability.
7. Evidence confidence.

Cap contributions by family so correlated indicators do not count as independent votes.

### Hard `READY` gate

No candidate reaches the highest tier unless:

- the structural thesis is current and valid;
- every mandatory confirmation step passes on eligible completed data;
- required RS/RW and volume rules pass;
- expected reward to the next real obstacle meets the setup minimum;
- logical invalidation is available;
- the entry remains actionable and not extended;
- market/sector context is not a hard conflict;
- no stale, incomplete, event, liquidity, or other risk block is active.

Scores rank candidates that pass. Scores do not cancel hard failures.

### SPY pullback leader/laggard model

For longs on strong days:

- measure how little the stock gives back during a defined SPY pullback;
- normalize by beta and volatility;
- reward holding VWAP, trigger, and relevant structure;
- prefer contracting sell volume and early resumption;
- include sector confirmation.

For shorts, invert exactly:

- measure failure to bounce during a defined SPY rebound;
- reward continued weakness below VWAP/trigger;
- prefer weak recovery volume and early renewed downside;
- require clean downside room and tradeability.

### Ranking pipeline

1. Hard eligibility and data-quality gates.
2. Candidate deduplication and thesis merge.
3. Setup-specific confirmation.
4. Independent evidence-family scoring.
5. Calibration from point-in-time walk-forward results.
6. Portfolio controls for sector/correlation concentration.
7. Actionability and anti-chase gate.
8. Top-K selection with an honest empty state.

### Promotion tests

- precision and expectancy at `READY`;
- precision@1 and precision@3;
- false-confirmation rate;
- median remaining expected R at notification;
- adverse/favorable excursion;
- missed-opportunity rate;
- readiness calibration;
- ranking churn;
- results by setup family, side, regime, and time of day.

### Exit gate

- Production and replay rankings are reproducible.
- Fewer than three qualifying trades produces fewer than three recommendations.
- A no-chase candidate cannot rank as Ready Now.
- Challenger improves quality without eliminating actionability.
- A configuration switch restores the prior champion immediately.

---

## Milestone 8 — Main GUI, Greatness Radar, and Auto/Away parity

The former GUI redesign concepts are folded into this milestone.

### Main application layout

The main Qt GUI should provide:

- global Auto Mode header;
- Trading Desk / Top Opportunities Command Center;
- D1 Development Radar;
- Alert Center and transition feed;
- RS/RW and Entry Assist views using shared evidence;
- Setup Tracker and research views;
- full Market Prep in Qt;
- Journal;
- Health page.

### Top Opportunities Command Center

Sections:

1. **Top 3 Ready Now**
2. **Confirming**
3. **One Step Away**
4. **Developing**
5. **Failed/Rearming**
6. **Extended/Expired**

Show fewer than three Ready candidates when fewer qualify.

### Candidate card

Each card shows:

- ticker, side, setup family, and stage;
- levels cleared and current blocker;
- exact next confirmation;
- entry zone, invalidation, obstacle, target, and expected R;
- SPY and sector test result;
- RS/RW and volume result;
- actionability/no-chase state;
- readiness and confidence separately;
- data complete-through time;
- what changed on the last transition.

### Mini-chart radar

Generalize the current SPY M5 chart into a reusable symbol chart showing:

- M5 candles with optional M1/M15 selection;
- VWAP and short-term trend overlays;
- opening range and prior-day/premarket levels;
- D1/AVWAP/SMA/trendline confirmation zones;
- entry, invalidation, obstacle, and target;
- touch, close, retest, failure, and Ready markers;
- SPY episode shading or a compact normalized RS trace.

Charts must render from the shared snapshot/cache and perform no provider call during paint.

### Alert ladder

| Tier | Behavior |
|---|---|
| Board-only | Developing state changes silently |
| Heads-up | One step away or close to the next level |
| Testing | Interacting with the important level |
| Confirming | Initial close occurred; waiting for acceptance/retest |
| Ready | All hard gates pass and entry is actionable; loud alert |
| Failure/Rearm | Quiet lifecycle update |
| No Chase | Informational only |
| Invalidation | Clear removal, especially for pinned names |

Deduplicate by typed transition identity, not message text.

### Human focus lists

- Keep separate long and short focus inputs.
- Allow quick add from Master AVWAP using side.
- Monitor human picks closely across bounce and RS/RW evidence.
- Preserve manual entries against automated removal.
- Track model quality and human-selection lift separately.
- Do not grant an objective score bonus merely because the user pinned the name.

### Auto/Away parity

The phone report must be a projection of the same opportunity snapshot:

- Ready Now;
- One Step Away;
- failures, invalidations, and no-chase changes;
- exact freshness and complete-through times;
- entry, invalidation, obstacle, and expected R;
- optional chart contact sheet generated from cached bars.

Publish on material state/rank changes plus a heartbeat interval, using verified atomic replacement.

### Live testing

- GUI responsiveness with maximum fast-lane candidates.
- Hidden chart panels do not stop monitoring.
- No data fetch occurs from paint/render paths.
- Setup Detail updates when a thesis version changes.
- Desktop and Away output match candidate IDs, order, evidence, and timestamps.
- Alert sounds and severity match lifecycle transitions.
- User can understand why a ticker is not ready without opening another tool.

### Exit gate

- One main GUI provides the complete desktop workflow.
- D1 Focus becomes a developing-opportunity product rather than a wick-cross feed.
- The loudest alerts are rare, explained, current, and actionable.
- Auto/Away mode presents the same truth as the desk.

---

## Milestone 9 — Journal and learning center

### Automatic linkage

Link:

- opportunity discovery;
- each lifecycle transition;
- alerts delivered;
- user actions: took, passed, missed, late, irrelevant, useful;
- imported trade execution records;
- favorable/adverse excursion;
- outcome at defined horizons;
- screenshots or chart snapshots when retained.

### Funnel measurement

Measure:

```text
Discovered -> Near Trigger -> Testing -> Confirming -> Ready -> Follow-through
```

Retain candidates that never became Ready so the system can study selectivity and missed opportunities.

### Personalization

Keep three distinct values:

- objective setup quality;
- user relevance/preference;
- historical fit with the user’s execution style.

Personalization may reorder equally qualified candidates. It must not rewrite historical expectancy or hide hard risk failures.

### Research questions

- Which D1 families most often reach Ready?
- Which prerequisite levels add useful selectivity?
- Does a retest improve expectancy or make alerts late?
- Which SPY pullback RS/RW patterns predict follow-through?
- How often do wick-only alerts fail?
- When does re-arming rescue a valid thesis?
- Which gates reduce false alerts with the least opportunity loss?
- Does human curation improve outcomes after controlling for setup and regime?

### Exit gate

- Every high-priority recommendation can be reconstructed.
- Outcome definitions are versioned and point-in-time safe.
- Journal operations remain fast and local if Drive is unavailable.
- Learning reports include coverage, sample size, and uncertainty.

---

## Milestone 10 — Controlled setup discovery program

New setups enter only through research and shadow stages.

Priority families:

- compression-to-expansion near meaningful levels;
- post-earnings constructive pullback;
- failed breakout/breakdown and undercut/reclaim;
- opening drive to first controlled pullback;
- trend-day leader/laggard continuation;
- multi-timeframe alignment/conflict;
- catalyst-aware continuation and risk;
- regime-specific swing continuation.

For each setup:

1. Write the thesis and failure mode.
2. Define point-in-time features and candidate universe.
3. Define entry, invalidation, target, expiry, and actionability.
4. Create synthetic and replay fixtures.
5. Run historical discovery/validation/test splits.
6. Run live shadow with complete candidate retention.
7. Compare against the existing opportunity portfolio.
8. Check incremental value after correlation with existing setups.
9. Promote only with a versioned rollback.

Do not reward a setup merely for increasing alert volume. Measure portfolio-level incremental expectancy and concentration.

---

## Milestone 11 — Platform consolidation, CI, and release engineering

### Codebase modernization

- Introduce a `TradingRuntime` composition root.
- Continue extracting pure domain modules from legacy files behind adapters.
- Centralize clocks, calendars, side parsing, paths, configuration, and snapshot publication.
- Remove duplicate scheduler, fetch, scoring, and alert logic only after parity tests.
- Keep expensive work outside the Qt thread.
- Replace prefix/string routing with typed events.

### CI

Required lanes:

- fast unit tests;
- deterministic smoke tests;
- golden/replay tests;
- Qt headless integration tests;
- lint/type gates introduced incrementally;
- migration tests;
- packaging/startup test;
- optional scheduled performance regression job.

### Release engineering

- pin and audit dependencies;
- document Python and OS support;
- provide reproducible environment setup;
- verify clean-machine startup;
- version schema migrations and configuration;
- provide backup/restore and rollback runbooks;
- keep broker execution and consumer shipping deferred.

### Exit gate

- A clean supported machine can install, start, smoke-test, and recover the application.
- CI blocks correctness regressions.
- Runtime health and migrations are supportable without reading source code.

---

## 9. Detailed Greatness Monitor requirements

This section remains the product specification for the flagship high-conviction workflow.

### Slow structural layer

Master/D1 discovery defines:

- direction and setup family;
- daily quality and conviction;
- AVWAP anchors and relevant daily levels;
- ordered levels that must clear;
- setup-specific confirmation policy;
- invalidation;
- obstacles and targets;
- catalyst/event context;
- expiration and re-arm policy.

It should be recalculated only when material structural inputs change.

### Fast confirmation layer

Each completed intraday bar updates:

- distance to next required level;
- touch/wick/close/acceptance/retest state;
- SPY/sector RS or RW;
- volume participation;
- VWAP/opening-range/short-term structure;
- expected R and next obstacle;
- extension/no-chase status;
- freshness and confidence;
- lifecycle stage.

### Ordered confirmation steps

Each step supports:

- a level or zone;
- side-aware comparison;
- touch, close, acceptance, retest-hold, or rejection condition;
- required timeframe and number of bars;
- optional RS, volume, and market predicates;
- mandatory versus supporting status;
- reset, expiry, and re-arm behavior;
- captured evidence on pass/fail.

### Multi-level greatness ladder

The user should see:

```text
[x] Reclaimed earnings AVWAP
[x] Held VWAP during SPY pullback
[>] Testing prior-day high
[ ] Needs completed M5 close
[ ] Needs hold/retest with positive RS
[ ] Final actionability and reward/risk gate
```

### Failure and re-arm

- A wick is a test, not a Ready event.
- A failed first attempt records evidence without consuming the entire day.
- Re-arm requires explicit reset time/distance and attempt limits.
- Structural invalidation prevents re-arm.
- Liquidity-sweep setups may interpret a failed break differently, but only through a setup-specific policy.
- Deduplication includes candidate, plan version, step, attempt, and transition.

### Actionability states

- `EARLY`
- `ACTIONABLE`
- `LATE`
- `NO_CHASE`
- `RESET_WATCH`

The product should celebrate a correct thesis without mislabeling a late entry as a current opportunity.

---

## 10. Testing strategy for all remaining work

### Unit tests

- pure state transitions;
- exact long/short mirrors;
- completed/incomplete bars;
- timestamp alignment;
- scoring component boundaries;
- lifecycle identity and deduplication;
- retries, leases, and expiry;
- migrations and serialization.

### Characterization and golden tests

- current scan output;
- detector output by setup family;
- candidate/watchlist exports;
- Alert Center routing;
- report contents;
- journal queries and imports;
- deliberate expected diffs for correctness fixes.

### Integration tests

- GUI/service start and bounded stop;
- scheduler-to-ledger lifecycle;
- provider-to-cache-to-scanner flow;
- candidate registry dual-write migration;
- scanner-to-opportunity-to-alert/report flow;
- desktop/Away snapshot parity;
- local journal with Drive unavailable.

### Replay tests

- wick failure followed by valid confirmation;
- clean break and retest;
- immediate extension after confirmation;
- SPY strong-day pullback with leader long;
- SPY weak-day rebound with laggard short;
- sector-confirmed move versus isolated false move;
- regime reversal;
- missing/stale data;
- no-trade session;
- restart in the middle of a candidate lifecycle.

### Research correctness tests

- no future level inputs;
- no same-day history contamination;
- immutable training/validation/test windows;
- stable event identity;
- no duplicate pseudo-samples;
- universe membership point in time;
- outcome coverage and missingness;
- reproducible multiple-testing controls.

### Performance tests

- provider calls per scan;
- duplicate fetch count;
- cache-hit rate;
- time by scan stage;
- GUI event-loop latency;
- chart repaint load;
- journal query latency;
- memory and file growth;
- bounded diagnostics retention.

---

## 11. Success metrics

### Reliability

- zero orphan workers after normal shutdown;
- zero unbounded retry loops;
- zero silent partial report replacement;
- zero unexplained cross-machine overwrite in controlled and live observation;
- truthful heartbeat, freshness, and last-success state;
- explicit incomplete status during provider/data failure.

### Performance

- lower Master scan wall time on identical fixtures;
- fewer duplicate provider calls;
- bounded memory and diagnostics growth;
- responsive GUI under maximum supported fast-lane load;
- reports and alerts published within their stated freshness window.

### Opportunity quality

- precision and expectancy at Ready;
- precision@1 and precision@3;
- false-confirmation rate;
- remaining expected R at notification;
- favorable/adverse excursion;
- actionability and no-chase accuracy;
- capture rate after failed attempt and re-arm;
- stable performance across regimes and sides.

### Research quality

- complete provenance and replayability;
- point-in-time-correct features;
- candidate/outcome coverage;
- calibrated confidence;
- out-of-sample promotion evidence;
- explicit sample size and uncertainty.

### User usefulness

- the user can identify the best current opportunities from one screen;
- every top candidate explains “why,” “why not yet,” “what next,” and “what kills it”;
- Auto/Away output matches the desktop;
- alert volume falls without losing high-quality opportunity capture;
- zero-trade days are represented honestly;
- manual focus and model picks can be compared fairly.

---

## 12. Immediate execution order

### Now — before promoting either shadow engine

1. Verify and record the 802-test baseline.
2. Improve shadow schemas, coverage counters, version/config metadata, and daily audit tooling.
3. Run the first-session checklist and preserve artifacts.
4. Complete the Phase 2 two-machine and failure drills.
5. Build the Health page.
6. Build golden/benchmark fixtures.
7. Keep legacy SPY and D1 alert behavior as champion.

### Next — after initial live evidence

8. Perform the supervised storage/secrets migration.
9. Introduce the provider repository behind golden parity tests.
10. Fix point-in-time research defects with intentional-diff fixtures.
11. Complete CandidateRegistry live-writer adoption.
12. Integrate canonical SPY/RS evidence in advisory/shadow mode.
13. Decouple Greatness into its dedicated priority lane.

### Then — after correctness and authoritative data paths

14. Build the canonical Opportunity and ranking pipeline.
15. Add the complete Greatness gate stack.
16. Build the Command Center, mini-chart radar, and alert ladder.
17. Publish the same snapshots to Auto/Away mode.
18. Link the full lifecycle to the journal and outcome engine.

### Later

19. Validate and promote advanced setup families one at a time.
20. Port remaining Market Prep functionality into Qt.
21. Finish CI, packaging, clean-machine recovery, and release documentation.
22. Reconsider consumer shipping or broader broker integration only after the internal product is stable; execution remains out of scope unless the product boundary is deliberately changed.

---

## 13. Definition of done

The roadmap is complete when:

1. The main GUI is the reliable central runtime and Auto Mode is truthful.
2. Desk and mini-PC can coexist without ambiguous ownership or shared-data corruption.
3. Every scan and job is observable, bounded, and recoverable.
4. Market data is cached, aligned, freshness-aware, and provider-efficient.
5. Historical research is point-in-time correct and reproducible.
6. One canonical candidate/opportunity lifecycle feeds every surface.
7. SPY pullback and RS/RW evidence are validated and consistently applied.
8. Greatness candidates are monitored continuously as they clear ordered levels.
9. Loud alerts require genuine confirmation, context, reward/risk, and actionability.
10. The Command Center may honestly show zero, one, two, or three Ready opportunities.
11. Desktop and Away mode show the same evidence and freshness.
12. The journal reconstructs what the bot knew and measures every stage of the funnel.
13. New setups are promoted only through replay, shadow, live validation, and rollback-controlled canaries.
14. The full supported test and CI suite is green.
15. The application remains a decision-support system and performs no executions.

Until all conditions are met, this document remains a living roadmap. Update statuses only when the corresponding implementation, green-test, live-validation, and promotion evidence actually exist.
