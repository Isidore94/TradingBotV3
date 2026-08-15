# TradingBotV3 remaining roadmap

Last reconciled: **2026-08-15**

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
| **0.5** | Trader refinement packets | Build the trader's 2026-08-14/15 desk requests in ranked order (R1–R8) |
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
   Cover the 2026-08-11 push policy too: the swing push must carry a favorite/
   high-conviction roster matching the Setup Tracker's rows for that hour, the D1
   push must name only the events since the previous push, and both must be silent
   while the desk sits in DESK or EVENING while a Research-tab price alert still
   fires from those modes.
   Also confirm the BounceBot scan window on a live day: Auto Pilot logs one resume
   at 06:00 and one pause at 13:30, `trading_bot.log` shows no symbol sweep between
   them and the close, and the session itself is unaffected — same alert count and
   the same IB connection held across the boundary.
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

### Phase 0.5 — trader refinement packets (promoted 2026-08-15)

The trader explicitly promoted the 2026-08-14 `WISHLIST.md` entries on 2026-08-15
and ranked the build order (R1 first, then R2; R3–R6 behind them). Each packet has
a specification under `docs/`; the file-scoped ask-first rule and the golden-fixture
invariant bind at edit time, packet by packet. Phase 1 work may interleave only
where a packet's own spec says the baseline item is a prerequisite (none currently
does).

**Build-order note (2026-08-15).** The original gate read "build work starts only
after P0.7 merges". The trader redirected twice on 2026-08-15 — first for R1, then
explicitly again for R2 — so both are built on their own branches ahead of the
testing-week merge, and P0's live gates are unchanged and still owed. The redirect
was packet-by-packet and does **not** carry forward: **R3 onward waits for the
trader to say so.**

On 2026-08-15 the trader added two new packets with their own explicit redirect:
**R7 (journal reliability + UX)** and **R8 (Weekend Prep)**, specced the same day.
Later that day the trader redirected again, in writing: **R7 code starts
immediately on `phase05-r7-journal-reliability-ux` cut from the R2 tip**, ahead
of the P0.7 merge — same pattern as the R1/R2 redirects. Rationale recorded:
R7/R8 touch journal and weekend surfaces, not the scanning/alerting/Focus path
whose live proofs Monday owes; the desk keeps running the R2 branch until the
validation day passes. P0's live gates are unchanged and still owed; merging R7
later brings the whole stack. The redirect does not authorize R3–R6.

R2's branch is cut from R1's and carries the R1.1 repair, so merging R2 brings the
testing week, R1, R1.1 and R2 together. The R1 and R2 live proofs are both owed and
are listed in `CURRENT_CHECKPOINT.md`.

1. **R1 Auto-mode matrix and quiet hours. — BUILT 2026-08-15, live proof owed.**
   Enforced the trader's OFF/DESK/AWAY/EVENING semantics (AWAY queues and never
   adopts into M5 Focus; EVENING stops scanning after its early block and gained
   the SPY ±1% wake alarm as the second documented push exception), one fail-open
   quiet-hours gate over every automatic starter (06:00–14:00 PT — a superset of
   the BounceBot scan window, see the spec §8), and the shared-scan/dead-Drive
   removal. Spec: `docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md`.
   Branch `phase05-r1-auto-modes-quiet-hours`; deterministic gate green
   (2773 passed / 19 subtests, smoke 7/7, source selftest 30/30 at the time, all exit 0);
   CLAUDE.md/AGENTS.md push policy, both runbooks, the first-session checklist
   and decision 0015 reconciled.
   **Remaining — the spec §6 live proofs, none yet run:** a ~21:00 boot that
   provably starts nothing; an EVENING day whose log shows the early block and
   then zero further slots; an AWAY session with picks staged-not-adopted and a
   clean drain on return; and one SPY-alarm firing (real or forced threshold).
2. **R2 M5 Focus adoption discipline and the M5 strength board. — BUILT
   2026-08-15, live proof owed.** The combined prev-day-extreme + session-VWAP
   gate applied at candidate build, staging refresh (queue eviction), and
   adoption; a provenance sidecar making auto picks the only legally removable
   entries; the scoped "Not today" verb; the triple-VWAP/Focus desync repair;
   and the TC2000-parity strength scanner over `universe_all.txt` via batched
   yfinance with one-click add-to-Focus. Spec:
   `docs/M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md`.
   Branch `phase05-r2-focus-gating-strength-board` (cut from R1, R1.1 merged
   forward); deterministic gate green (2865 passed / 19 subtests, smoke 7/7,
   source selftest 30/30 at the time, all exit 0; the current baseline after the
   R2.1 and R2.2 review passes is in `CURRENT_CHECKPOINT.md`). **This closes R1's
   recorded stale-drain gap**: the AWAY/EVENING→DESK drain now adopts only
   verdicts stamped after the flip itself, re-measured on the flip, with a
   failed re-measurement retrying rather than falling through (R2.2).
   **Remaining — the spec §8 live proofs, none yet run:** one session showing a
   staged pick evicted for a VWAP/PDH fallback, one adoption-time refusal, one
   clean "Not today" scoped removal that leaves the trader's other entries
   intact, and a board session the trader confirms matches the TC2000 scan's
   character (~20–40/side). Re-measure the board fetch during market hours on
   that session (spec §10 recorded 27.6 s on a Saturday — a floor, not a worst
   case). RVOL-for-survivors is specified but deliberately not built; decide it
   on that session.
3. **R3 Swing-quality demotion, pre-close honesty, and the dislike-feedback
   loop.** Demote-and-label (never hide) overextended swing rows using the
   trader's two v1 rules, the RVOL field + daytrade carve-out annotation, the
   reviewed-today badge from recorded decisions, structured dislike reasons
   counted by the review-learning scoreboard, and the authorized full-honesty
   pre-close bundle (tracker writes once post-close, a 12:45 PT slot,
   forming/completed stamps, STABLE+PREVIEW split, time-normalized volume
   thrust). The after-close investigation is COMPLETE — mechanisms recorded in
   the spec. Spec: `docs/SWING_QUALITY_AND_FEEDBACK_PLAN.md`. Exit: fixtures
   first, then the spec's one-week desk gates.
4. **R4 Desk chart unification.** CaptureRail (veto/like/note) on every chart
   surface, review/watch wiring for the RS/RW and Industry panels, Alert Center
   LIKE, armed price alerts and D1 watches painted as a toggleable levels family,
   the forming-bar source honesty fix for the early-morning gap distortion, and
   the reviewed-today badge rendered everywhere. The labeled Y axis already
   exists — recorded as answered. Spec: `docs/DESK_CHART_UNIFICATION_PLAN.md`.
5. **R5 M5 signal engines.** New pure indicator modules (TC2000-parity SMI,
   efficiency-LRSI under a non-colliding name, Heikin-Ashi reversal), the LRSI
   cross alert type, the HA+SMI+LRSI confluence alert (Focus-scoped), the
   first-candle ORB candidate flow, the AnyBounceWatch multi-level armed watch
   (including the new prior-anchor AVWAP line in the D1 scan output), and a
   shared completed-bars helper. Golden fixtures gate live wiring; packaging and
   frozen selftest update when `scripts/indicators` gains its first importer.
   Spec: `docs/M5_SIGNAL_ENGINES_PLAN.md`.
6. **R6 Small operational wins.** (a) AI-jobs visibility: reword the routine
   "(no arguments)" line in `scripts/run_ai_jobs.ps1` and add a read-only System
   Health row over `job_ledger.jsonl`/the dated log — the batch layer is never
   hosted in the GUI. (b) Evidence-ledger rotation: session-scoped rotation for
   `technical_integrity_events.jsonl` (~247 MB, fully re-parsed each boot)
   preserving its replay contract, then audit the other JSONL ledgers via the
   existing footprint check; `technical_integrity.py` is scoring-adjacent —
   ask first. (c) A bounded stall-watchdog diagnostic week (`ui_stall_watchdog`
   setting only) before prioritizing any worker-offload work. (d) **Auto journal
   is a mapping, not new work**: the trader's ask resolves to the QUEUED nightly
   `journal_import` slot (`docs/LOCAL_AI_AUTOMATION_PLAN.md` sec 6.4c — build
   only after the 6.4b live proof passes and the trader says go) plus the P3.5
   commentary journal; **the nightly slot half was promoted into R7 on
   2026-08-15** — see item 7; P3.5 is unchanged.
7. **R7 Journal reliability and UX (added 2026-08-15).** Tax-grade completeness
   from both brokers: stable execution/trade identity (annotations survive
   rebuilds), IBKR Flex as the primary historical source, wired Questrade
   activities, per-chunk partial persistence, a date-coverage ledger with a
   nightly `journal_import` runner slot that self-heals gaps (the promoted P3.3
   slice), position reconciliation against both brokers with append-only
   audit-trailed corrections, booked BoC CAD conversion, and the rebuilt Journal
   tab (account/tax-status selection that never silently blends, date-range +
   calendar + fees views, R-multiples with alert prefill, pyqtgraph analytics,
   surfaced walk-away). Spec: `docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md`.
   Dependencies: authorized to build now on `phase05-r7-journal-reliability-ux`
   cut from the R2 tip (trader redirect 2026-08-15, second of the day; see the
   preamble). Exit: the spec's deterministic gates plus its live gates (coverage
   complete since inception, statement reconciliation to the cent, one clean
   reconciliation week, zero orphaned annotations, five nightly slot runs).
8. **R8 Weekend Prep (added 2026-08-15).** A guided five-step weekend routine
   (week in review, focus-pick review, week-windowed walk-away with the auto-tag
   review, H1/D1/Monthly strength discovery on the R2 formula via a new pure
   module — `strength_scan.py` untouched — and week-ahead prep adopting the
   orphaned `market_prep` weekly engine), all manual-refresh with zero IB
   traffic, adopt-to-swing-Focus routing through the existing membership-tracked
   injection. Includes the standalone `app.py` nav-title bugfix as its first
   commit. Spec: `docs/WEEKEND_PREP_PLAN.md`. Dependencies: after R7 (shares
   `journal_feed`); discovery step blocked on the spec §5 filter approval.
   Exit: the spec's deterministic gates plus its one-real-weekend live proof.

Exit gate: each packet exits through its own spec; R1 and R2 land first per the
trader's ranking, then R7 before R8. A packet's live gates may overlap the next
packet's build only when no shared file is in flight.

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
   *Narrowed 2026-08-10:* evidence budgets and failure behavior are **done** — local
   calls cap at `ai_local_evidence_budget_chars` and raise on server-side prompt
   truncation. A fact-pack design packet is **proposed** in
   `docs/LOCAL_AI_AUTOMATION_PLAN.md` sec 6.4a. What remains owed: **trader answers to
   its six open questions**, then schema and tests. No digest schema may be built or
   frozen before those answers — the 2026-08-08 decision still stands.
   *Armed and built 2026-08-11:* the **ticker-briefs hardening packet**
   (`docs/LOCAL_AI_AUTOMATION_PLAN.md` sec 6.4b) was armed by the trader after the
   first overnight run and is **implemented**: project-then-budget evidence (TB-0),
   per-ticker failure isolation with an honest partial morning file (TB-1),
   deterministic membership-only skip (TB-2), resumable per-symbol completion (TB-3),
   and a per-session attempt cap (TB-4). Per the gate's own reset rules the
   **ticker-briefs five-session clock restarts at zero**; the `ai_summary` clock
   continues, because its code path is untouched.
   *First live night ran 2026-08-11 and the proof is **partial**, repaired 2026-08-12:*
   `ai_summary` succeeded first attempt; `ticker_briefs` briefed 101 of 182 symbols
   and was killed mid-batch, publishing no morning file. TB-0 is confirmed; TB-3 was
   proven broken and is fixed; TB-1/TB-2/TB-4 were never exercised. Three repairs
   landed — **TB-5** (a roster line is not evidence: 96.2% of the payload was ticker
   name-dumps, and removing them cuts 166 model calls to 49), **TB-3's stable
   `resume_key`**, and **TB-6** (the morning file is republished after every resolved
   symbol, so a hard kill no longer loses the night) — plus the scheduled task's
   `ExecutionTimeLimit`, which at `PT2H` against an 8-hour window was terminating the
   parent and letting a second concurrent runner start.
   **Still owed: live proof on the 2026-08-12 window**, and it is only interpretable
   once the desk stops sleeping — 4h39m of Modern Standby, trader-owned, ended the
   08-11 run. A night cut short by sleep is not evidence about this layer.
   *The nightly journal pull queued here 2026-08-11 was **promoted into Phase 0.5
   R7 on 2026-08-15** (trader go recorded in
   `docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md` §6, which honors the sec 6.4c design
   verbatim and supersedes the "after 6.4b proof" ordering). P3.3's remaining
   scope is the fact-pack/enrichment/policy-draft work only.*
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
   actual risk, and after-close reconciliation. *Narrowed 2026-08-15: the fills
   completeness, planned-risk capture, and broker after-close-reconciliation slice
   moved to Phase 0.5 R7; P5.4 retains the lifecycle joins, screenshots, and
   opportunity-identity linkage.*
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
   their predecessor gates pass. *Amended 2026-08-10:* the review-policy draft
   comparison runs frontier-vs-medium rather than local-large-vs-cloud — the local
   large tier is retired (no 27B-class model loads beside the running desk). The
   two-week side-by-side quality gate is unchanged.
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
