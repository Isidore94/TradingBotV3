# TradingBotV3 remaining roadmap

Last reconciled: **2026-08-10**

Authoritative for: **work that is not finished, validation gates, promotion rules,
and execution order**

Implemented history: [`CHANGELOG.md`](CHANGELOG.md)

Supporting-document index: [`docs/README.md`](docs/README.md)

This file intentionally does not repeat the implementation history. A feature with
code is recorded in `CHANGELOG.md`; any validation, evidence, promotion, or cleanup
still owed remains here. Detailed specifications under `docs/` are subordinate to
this roadmap. Section numbers 5–7 and 12 are retained deliberately so established
runbook and decision-record references stay valid.

## 1. Mission and product boundary

TradingBotV3 is a decision-support system for one trader. It prepares the market,
discovers swing and intraday candidates, monitors them, alerts, publishes an Away
report, records decisions and outcomes, and supports controlled research.

It never places or routes orders. Broker execution, consumer distribution, and any
automatic promotion of research output are outside the product boundary.

The operating topology is now simple:

- the Ryzen 7 8845HS main desk is the only always-on application and scan host;
- `launch_gui.py` starts the PySide6 Trading Desk in Main mode;
- the former mini-PC scanner and Desk Link satellite roles are retired and must stay
  unused until their code is removed in a deliberate cleanup packet;
- ntfy and the verified `autopilot_today.txt` digest are the remote surfaces;
- there is no cloud sync (decision 0015): `C:\TradingBotData` is a plain local folder
  and the DAS `\\MINI-PC\Trading Bot Data` is the durable storage tier;
- the Tk GUI remains a temporary compatibility path during migration.

## 2. Status vocabulary

These labels must not be collapsed:

| Status | Meaning | Production authority |
|---|---|---|
| `PLANNED` | Designed but no implementation exists. | None |
| `IMPLEMENTED` | Code exists. | None by itself |
| `GREEN` | Deterministic tests pass. | Only existing champion behavior |
| `SHADOW` | Runs on live inputs but cannot affect production decisions. | None |
| `LIVE_VALIDATED` | Passed the documented real-session or operational checks. | None by itself |
| `ADVISORY` | Visible as labeled research or decision support. | No loud-alert, gate, or ranking authority |
| `PROMOTED` | Explicitly approved as the production champion with rollback. | Yes |
| `RETIRED` | Intentionally disabled or replaced. | None |

Current code and test status belongs in `CHANGELOG.md`. Current branch and exact
test counts belong in `CURRENT_CHECKPOINT.md`. Only unfinished work belongs here.

## 3. Current-state summary

As of the reconciliation date:

- the active branch is `testing-week-2026-08-10`, not yet merged to `main`;
- the Windows desk gate is green at 2611 tests plus 7 subtests, smoke 7/7, and
  frozen self-test 29/29;
- subsequent 2026-08-10 presentation and phone-report fixes have not changed the
  recorded gate yet;
- the warehouse Phases 0–8, Chart Review A1–A5, durability steps 1–4, and Local-AI
  Phase 1 are implemented, but their remaining live gates below still apply;
- legacy SPY pause detection and D1 wick alerts remain the production champions;
- `market_state` and `greatness_monitor` remain shadow-only;
- the research warehouse and AI outputs remain additive/read-only and advisory.

See `CHANGELOG.md` for the full implemented inventory and revision history.

## 4. Authority and change control

When documents disagree, use this order:

1. this roadmap for remaining-work order, invariants, and promotion policy;
2. accepted decision records under `docs/decisions/`;
3. the locked warehouse specification where it is explicitly delegated authority;
4. active implementation specifications listed in `docs/README.md`;
5. historical reviews, handoffs, proposals, and superseded GUI plans.

Do not infer current status from a historical plan. Reconcile it through
`CHANGELOG.md` and this file.

`WISHLIST.md` is deliberately outside the authority chain. It records candidate
integrations and deferred ideas, but it never authorizes implementation or changes
the order below. Only an explicit trader decision may promote a wishlist item into
this roadmap.

## 5. Non-negotiable system invariants

### Data and time

- State transitions use completed bars only. A forming bar is a labeled preview.
- Missing or stale data is uncertainty, never confirmation.
- Point-in-time research may use only information available at the simulated
  decision time; timestamps carry explicit time zones.
- Never replace `calc_anchored_vwap_bands`' running-deviation sigma formula.

### Identity and provenance

- Stable identity must distinguish symbol, side, horizon, setup/thesis, anchor,
  attempt, and configuration where those dimensions matter.
- Every suggestion, alert, review, research row, and outcome must retain enough
  provenance to reconstruct what the system knew.
- User-entered watchlist names are never automatically removed.

### Runtime and publication

- One component owns each timer, thread, job, mutable store, or shared export.
- A failed publish never destroys the last verified report.
- Ambiguous ownership fails closed.
- The single-main topology does not authorize duplicate writers.

### Research and promotion

- Legacy SPY pause detection and D1 wick alerts stay champions until the Section 7
  gates pass.
- No detector, score, ranking, routing, or alert-behavior change lands without a
  golden characterization fixture first.
- Shadow, research, Technical Integrity, warehouse, review-learning, and AI outputs
  have zero production influence until separately promoted.
- `review_policy.json` ranks and annotates only; it has no suppression field.
- AI is one-way and evidence-grounded. It may summarize and propose tests, never
  mutate production state.

### Product behavior

- The app is decision-support only and never executes orders.
- Honest zero-opportunity and unknown-data states are preferable to filled panels.
- Desk, Away, alerts, journal, and AI must ultimately consume the same canonical
  opportunity facts.

## 6. Live validation program

Automated green tests do not satisfy live gates. The active checklist is
[`docs/FIRST_SESSION_CHECKLIST.md`](docs/FIRST_SESSION_CHECKLIST.md).

For the first live session on a new build, record:

- branch/commit, machine, Python, TWS/Gateway mode, home folder, research-store
  state, Auto profile, and market-session date;
- full pytest exit code, smoke result, and frozen self-test when a rebuild trigger
  applies;
- real run manifests, heartbeat, provider telemetry, shadow logs, job ledger,
  verified Away metadata, and capture audits;
- GUI responsiveness, chart freshness, alert delivery, clean shutdown, and restart
  behavior;
- every failure or unknown as evidence, without rewriting the acceptance result.

Physical two-machine and satellite checks from older runbooks are retired with the
topology. Writer fencing still requires deterministic tests, but no new live
two-machine gate blocks the single-main product.

## 7. Shadow evidence and promotion ladder

Promotion is a separate decision from implementation and live validation. Every
challenger requires:

1. a versioned configuration and stable identity;
2. golden/replay fixtures and a declared evidence window frozen before inspection;
3. complete coverage and data-quality accounting;
4. comparison with the active champion on the same inputs and outcome definition;
5. representative live sessions across relevant regimes, sides, and day parts;
6. explicit success, non-inferiority, and rollback criteria;
7. a bounded canary and one-switch rollback that does not require a code revert;
8. explicit trader approval recorded in the revision history.

### SPY pullback challenger

Still required before any promotion:

- prove completed-bar coverage rather than scan-cycle presence;
- label and reconcile episodes across session/config rollovers;
- compare episode timing, false pauses, missed pauses, and downstream candidate
  usefulness with the legacy pause detector;
- validate timezone, staleness, and restart behavior on live artifacts;
- integrate sector/candidate RS only as advisory evidence until its own gate passes.

### Greatness challenger

Still required before any promotion:

- a dedicated monitoring lane independent of legacy D1 scan cadence;
- same-day plan revision and side-change handling;
- complete multi-level confirmation, failure, re-arm, freshness, volume, RS/sector,
  reward/risk, and anti-chase gates;
- transition-chain audits and outcome comparison with legacy D1 wick alerts;
- evidence that alert precision improves without unacceptable delay or missed moves.

## 8. Detailed specifications retained under `docs/`

The roadmap owns priority and status. These files retain implementation detail that
would make this file unwieldy:

- research warehouse: `docs/ULTIMATE_SETUP_DATABASE_PLAN.md`,
  `docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md`, and
  `docs/RESEARCH_WAREHOUSE_ERD.md`;
- local AI: `docs/LOCAL_AI_AUTOMATION_PLAN.md`;
- Chart Review capture: `docs/CHART_REVIEW_WORKSPACE_PLAN.md`;
- setup doctrine: `docs/SETUPS_MAJOR.md` and `docs/SETUPS_TEST.md`;
- operations and packaging: the active runbooks listed in `docs/README.md`.

Their old phase lists do not reorder Section 12.

## 12. Remaining work, in execution order

The phases below are dependency order, not a menu. `CURRENT_CHECKPOINT.md` names the
one active item. Finish that item before moving down the list unless the trader
explicitly redirects the work. Elapsed evidence collection may run in parallel only
where the phase says so; it never authorizes an early promotion.

| Order | Build phase | Plain-English outcome |
|---:|---|---|
| **0 — NOW** | Validate and merge | Prove the testing-week build on the real desk and merge it safely |
| **1 — NEXT** | Reliable development baseline | Make tests offline/deterministic and close measured cleanup questions |
| **2** | Authoritative foundations | One correct provider, time, candidate, SPY/RS, and Greatness data path |
| **3** | Evidence and capture | Mature warehouse/AI/shadow evidence and capture trader commentary honestly |
| **4** | Canonical Opportunity | Build and validate one inspectable opportunity/ranking challenger |
| **5** | Delivery and lifecycle | Make alerts and every surface agree; reconstruct the whole decision lifecycle |
| **6** | Research payoff | Use the validated corpus to promote setups narrowly and finish the Qt product |
| **7 — LATER** | Consolidate and ship | CI, recovery, packaging, installer, and optional read-only broker adapters |

### Phase 0 — NOW: validate and merge the testing-week branch

No new feature or threshold work belongs in Phase 0.

1. **P0.1 Re-baseline the complete branch.** Run the full Windows suite after the
   post-gate code commits, smoke 7/7, and the frozen self-test when a packaging
   trigger applies. Record pytest's own exit code in `CURRENT_CHECKPOINT.md`.
2. **P0.2 Run one complete single-main live session.** Follow
   `docs/FIRST_SESSION_CHECKLIST.md`; preserve runtime, provider, shadow, review,
   chart, alert, warehouse, and shutdown artifacts. Do not tune from this session.
3. **P0.3 Validate Auto/Away and ntfy end to end.** Confirm the verified hourly
   report, safety/freshness header, swing-first ordering, quiet empty-swing behavior,
   best-swing phone push, late-opened chart freshness, and main-desk alert delivery.
4. **P0.4 Validate observability rollover.** Require real provider telemetry,
   SPY/Greatness rotation and summaries, valid per-installation review writes, and
   honest UNKNOWN/DEGRADED grades.
5. **P0.5 Run the durability restart drill.** Require a healthy regime audit with a
   nonzero backfill count, no duplicate desk, no pacing conflict, and no Tier-C
   reconstruction.
6. **P0.6 Start the elapsed evidence clocks.** Enable Local-AI Phase 1's five clean
   session mornings; run the warehouse broker-marked IB check, observe its live tee
   and six Health tiles, answer the pilot-relevant confirmation items (including the
   fixed cohort and favorite-zone definitions), and start the 20-session pilot.
7. **P0.7 Merge to `main`.** Only after a live-validation day passes, re-run the
   applicable gates, merge, and update all control documents.

Exit gate: the branch is green, one real session is documented, the application is
operationally safe on the single-main topology, and `main` contains the validated
build. The Local-AI, warehouse, regime, SPY, and Greatness evidence clocks may remain
in progress; their results gate later promotions, not the merge itself.

### Phase 1 — NEXT: remove known uncertainty from the development baseline

1. **P1.1 Make the test suite hermetic.** Stop Qt app tests from starting live
   universe/yfinance work; keep explicit network/broker markers and bounded teardown.
2. **P1.2 Resolve the measured D1 line-display defect.** After the testing week,
   decide the red-level threshold and total clutter budget from desk evidence. Ask
   before touching any fenced detector/scoring/alert-hosting file.
3. **P1.3 Adjudicate pending branches.** Review the scoring/flagging branch only with
   golden fixtures; discard or supersede obsolete documentation-only work after this
   consolidation.
4. **P1.4 Finish observability depth.** Add representative benchmark/golden fixtures
   and trends for timings, provider calls, failures, coverage, and scan-stage latency.
5. **P1.5 Do bounded repository hygiene.** Ignore generated desk JUnit output and
   remove retired Desk Link/satellite/mini-PC code only in an explicit, fully green
   cleanup packet. Do not mix cleanup with behavior changes.

Exit gate: tests are deterministic/offline by default, the chart-level policy is
intentional, open branches are resolved, benchmark evidence is stable, and retired
topology code no longer confuses the supported runtime.

### Phase 2 — FOUNDATION: create authoritative data paths before new ranking

Each item requires parity/rollback evidence before the next authority cutover.

1. **P2.1 Complete storage and secrets classification.** Preserve operational home,
   machine-local, and research-lake boundaries; migrate remaining live databases or
   secrets only with backup, dual-read verification, and rollback.
2. **P2.2 Introduce the provider repository.** Centralize IBKR/Yahoo fetches, cache
   keys, batching, request coalescing, pacing, source, and freshness behind golden
   parity tests. Do not change champion results.
3. **P2.3 Repair remaining point-in-time defects.** Cover moving levels, history
   keys, backfill leakage, tracker identity, score ordering, factor horizons,
   corporate actions, and survivorship with intentional-difference fixtures.
4. **P2.4 Make CandidateRegistry authoritative.** Migrate every live candidate writer,
   preserve manual names, prove expiry/restart/rollback, and retire duplicate text-
   file authority only after parity.
5. **P2.5 Integrate aligned SPY/sector/industry/stock RS as advisory evidence.** Expose
   complete-through time and provenance; unpromoted fields contribute exactly zero to
   production eligibility, score, order, sound, and delivery.
6. **P2.6 Give Greatness a dedicated completed-bar lane.** Establish continuous
   coverage, stable identities, revisions, and the evidence hooks required by
   Section 7 without changing legacy D1 alerts.

Exit gate: provider, point-in-time, candidate, market-state, and Greatness inputs are
stable and reconstructable. Production still uses the legacy champions.

### Phase 3 — EVIDENCE AND CAPTURE: finish what must mature over time

These lanes may collect in parallel with Phase 1–2 work because they are additive and
non-authoritative. Their analysis/cutover steps remain ordered here.

1. **P3.1 Complete Chart Review live acceptance.** Verify sub-five-second capture,
   chart provenance/fallback warnings, painted-level references, the one alert writer,
   and zero privileges from LIKE/veto/note annotations.
2. **P3.2 Complete warehouse live validation and the 20-session pilot.** Verify IB
   transport, tee/tiles, gaps, pacing, storage growth, backups, and restore. Then build
   the tracker-to-detection adapter, explicit bounce linkage, and only the additional
   context/VWAP fields demanded by registered consumers. Keep it shadow-only.
3. **P3.3 Complete Local-AI Phase 1, then redesign Phase 2.** After five clean
   unattended mornings, specify deterministic fact packs, evidence budgets, schema,
   failure behavior, and tests before writing the append-only digest format. Require
   ten clean digest sessions before later AI phases.
4. **P3.4 Accumulate and audit promotion evidence.** Continue regime infrastructure
   toward 40 instrumented sessions, and SPY/Greatness toward their Section 7 floors.
   Freeze windows before inspecting outcomes.
5. **P3.5 Build the live market commentary journal.** First define its relationship
   to Chart Review notes and the existing journal; then add one append-only, rapid
   intraday capture stream and advisory nightly summarization. It never becomes a
   detector, score, gate, or alert input.

Exit gate: capture paths are live-validated, warehouse and AI gates have honest
results, commentary is reconstructable, and promotion datasets are usable without
look-ahead or missing-denominator ambiguity.

### Phase 4 — CANONICAL OPPORTUNITY: build the challenger product

1. **P4.1 Freeze the identity graph and Opportunity snapshot contract.** Separate
   Objective Quality, Actionability Now, Expected R, and Personal Fit; make every
   input and blocker inspectable.
2. **P4.2 Build the canonical eligibility/ranking challenger.** Consume only the
   authoritative Phase-2 inputs; prove every unpromoted field has zero production
   contribution and retain a one-switch champion rollback.
3. **P4.3 Complete the Greatness/readiness gate stack.** Add remaining reward,
   freshness, volume, context, ordered levels, failure/re-arm, and anti-chase logic in
   pure tests, replay, and live shadow.
4. **P4.4 Build the advisory Command Center and Focus Workbench.** Show lifecycle
   lanes, compact dossiers, reasons/blockers, mini charts, and honest zero-Ready days.
5. **P4.5 Freeze and pass the ranking manifest.** Compare identical snapshots and
   outcomes, run the bounded GUI canary, and promote projection separately from rank.

Exit gate: one deterministic advisory Opportunity snapshot exists, the trader can
use it without changing production routing, and any promoted projection/rank has
passed its own manifest and rollback drill.

### Phase 5 — DELIVERY AND LIFECYCLE: make every surface agree

1. **P5.1 Build the typed delivery challenger.** Immediate, Heads-Up, Focus Changes,
   Developing, Research, and History remain separate; deduplicate/group bursts while
   protecting every user-armed hit.
2. **P5.2 Freeze and pass the delivery manifest.** Canary sound/severity/routing
   independently from ranking and retain complete History plus instant rollback.
3. **P5.3 Project one verified snapshot everywhere.** Desk, Focus, Alert Center,
   Auto/Away, phone report, journal, and AI must agree on snapshot/opportunity IDs,
   stage, rank, freshness, and champion/challenger status.
4. **P5.4 Complete lifecycle and journal linkage.** Join discovery, stages, reviews,
   Focus/watch actions, fills, no-trades, outcomes, screenshots, MFE/MAE, planned vs
   actual risk, and after-close reconciliation.
5. **P5.5 Build the Learning Center and controlled universe intake.** Keep objective
   edge, actionability, personal preference, execution, and discovery-source value
   separate; personalization may reorder only inside declared safe bands.

Exit gate: the complete decision lifecycle is reconstructable, alerts are useful and
bounded, every surface agrees, and preference cannot change objective truth or safety.

### Phase 6 — RESEARCH PAYOFF: learn and promote narrowly

1. **P6.1 Complete the warehouse post-slice research tools.** Add the registered
   setup/style readouts, Level Edge/recipe comparisons, and evidence packages only
   after the pilot validates the corpus.
2. **P6.2 Promote advanced setup families one at a time.** Each family requires a
   registered question, point-in-time corpus, replay, shadow, live evidence, bounded
   canary, approval, and rollback.
3. **P6.3 Continue Local-AI phases in order.** Journal enrichment, review-policy draft
   comparison, and periodic frontier synthesis remain advisory and start only after
   their predecessor gates pass.
4. **P6.4 Finish Market Prep migration into Qt.** Retire the Tk path only after parity,
   operational recovery, and clean-machine proof.

Exit gate: research produces trustworthy narrow improvements without leaking into
champions, and the supported interactive product is fully Qt.

### Phase 7 — LATER: consolidate and ship the internal product

1. **P7.1 Complete CI and clean-machine recovery.** Cover supported Windows/macOS
   tests, offline smoke, frozen regression, backup/restore, and operator recovery.
2. **P7.2 Finish packaging and release polish.** Icon, version metadata, windowed
   build decision, bundle trimming, installer, and release notes follow the frozen
   rebuild policy.
3. **P7.3 Revisit read-only broker adapters.** Only after the provider repository is
   stable; execution remains permanently out of scope.

Everything else stays in `WISHLIST.md` until explicitly promoted into this sequence.

## 13. Definition of done

The roadmap is complete when:

1. the single main desk is reliable, observable, recoverable, and live-validated;
2. data is freshness-aware, point-in-time correct, provider-efficient, and owned by
   one writer;
3. every surface consumes one canonical opportunity lifecycle and saved snapshot;
4. SPY/RS and Greatness behavior is promoted only after the evidence ladder passes;
5. alerts are current, typed, deduplicated, and protect every user-armed hit;
6. the journal reconstructs discovery, judgement, execution/no-trade, and outcome;
7. research and AI remain reproducible, advisory, and separated from champions;
8. new setups enter production only through fixtures, replay, shadow, live evidence,
   canary, approval, and rollback;
9. the supported test, smoke, packaging, and recovery gates are green; and
10. the application remains decision-support only and performs no execution.
