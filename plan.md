# TradingBotV3 remaining roadmap

Last reconciled: **2026-08-20**

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

- the active branch is `testing-week-2026-08-17`, not yet merged to `main` — the
  single consolidated release candidate since 2026-08-15, carrying testing-week,
  R1, R1.1, R2, R7 and R8. `testing-week-2026-08-10` and the R1/R7 branch names
  used below are historical: those branches were deleted once proven contained,
  and their tips remain reachable by SHA (see `CURRENT_CHECKPOINT.md` rollback
  points). `phase05-r2-focus-gating-strength-board` survives only as the desk's
  running branch until the Monday merge;
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
   applicable gates, merge, and update all control documents. Since the
   2026-08-15 consolidation this is **one** merge —
   `testing-week-2026-08-17` → `main` — followed by a full gate re-run on `main`
   including a clean-cache frozen rebuild, and only then the disarm / switch the
   desk / re-arm sequence. R7's and R8's own live gates follow **after** that
   merge and are not merge blockers. Exact steps: `CURRENT_CHECKPOINT.md`,
   "Monday sequence".

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

**Weekend redirect (2026-08-15).** The trader then explicitly authorized the
remaining packets on the consolidated release candidate: *"integrate the rest —
build R3 through R6 on the consolidated branch."* R3, R4, R5 and R6 therefore
build in that order on `testing-week-2026-08-17`, one packet at a time with its
fixtures, full deterministic gate, governance close-out and push complete before
the next starts. This redirect does not satisfy any live gate: R3's shadow week,
R6's watchdog week, R1/R2's eight proofs, R7's migration/backfill/reconciliation
sequence, and R8's real-weekend run all remain owed. After R6, only the explicitly
named R7/R8 review-deferral completions are authorized; true USD conversion stays
deferred pending a trader decision.

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
   **Remaining — the spec §6 live proofs, narrowed 2026-08-18.** The quiet boot
   **PASSED** on 2026-08-16 22:06, and AWAY staging-without-adoption **PASSED**
   across the 08-17 and 08-18 sessions. Still owed: the **drain on return** (the
   trader never flipped back to DESK, so that half of the AWAY proof is
   untested); an **EVENING day** whose log shows the early block and then zero
   further slots; and one **SPY-alarm firing** (real or forced threshold).
   Evidence and the exact log lines are in `CURRENT_CHECKPOINT.md`. Note
   recorded there and not fixed: `BouncePanel` connects to IB on every launch at
   any hour (`bounce_panel.py:280`), outside Auto Pilot and outside quiet hours —
   a behaviour decision left to the trader, not a gate.
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
   **Remaining — the spec §8 live proofs, narrowed 2026-08-18.** The **eviction
   proof PASSED** on 2026-08-18 (four logged evictions with per-symbol reasons;
   lines quoted in `CURRENT_CHECKPOINT.md`). Still owed, all three needing a DESK
   day because AWAY never adopts: one adoption-time refusal, one clean "Not
   today" scoped removal that leaves the trader's other entries intact, and a
   board session the trader confirms matches the TC2000 scan's character
   (~20–40/side). Re-measure the board fetch during market hours on
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
   first, then the spec's one-week desk gates. **Built so far (2026-08-16):**
   shadow-only quality stamps/report badge, the 12:45 preview slot, and actual-close
   ownership of the single scheduled tracker write, completed-bar STABLE beside
   live PREVIEW, and explicit row stamps. The dislike/reviewed-today loop remains
   deterministic work. **Structured dislike capture/counting and the
   reviewed-today table/report marker are now built.**

   **DETERMINISTIC WORK COMPLETE 2026-08-16.** The one remaining item, §4.3.5
   same-slot volume normalization, was **explicitly deferred by the trader on
   2026-08-16**: the D1 scoring seam has no intraday slot series, the faithful
   TC2000 baseline would need a 5-minute fetch across ~1,100 symbols, and the
   zero-fetch session-elapsed proration was offered and rejected as trading one
   dishonest reading for another. The 18-pt thrust bonus therefore keeps its
   full-day baseline as a **known, accepted** pre-close gap, characterized by
   `tests/fixtures/r3_swing_quality_v1.json`. Reopening it needs a fresh trader
   decision on the data seam. That decision is now recorded, so R4 is unblocked.

   **Owed — live gates only, none claimable from tests:** the §6 `would_demote`
   shadow week the Amendment requires before any row moves, the one-week
   12:45-vs-close list and STABLE-vs-PREVIEW churn comparison, and the
   scoreboard's first real-data curation cycle producing a threshold proposal.
4. **R4 Desk chart unification.** CaptureRail (veto/like/note) on every chart
   surface, review/watch wiring for the RS/RW and Industry panels, Alert Center
   LIKE, armed price alerts and D1 watches painted as a toggleable levels family,
   the forming-bar source honesty fix for the early-morning gap distortion, and
   the reviewed-today badge rendered everywhere. The recovered 2026-08-14 Alert
   Center contract is consolidated here too: "Not today" never cancels or mutes
   trader-armed alerts; the feed's existing Focus star becomes a labeled
   Like-to-Focus action; and display-only repetition/open-burst control reduces
   repeated rows without weakening detection, evidence, History, or armed-hit
   delivery. The labeled Y axis already exists — recorded as answered. Spec:
   `docs/DESK_CHART_UNIFICATION_PLAN.md`; historical source:
   `docs/ALERT_CENTER_QUALITY_PACKET.md`.

   **BUILT 2026-08-16, live proofs owed.** Sections 1–5 and 6.1–6.3 are green:
   trader-armed hits survive "Not today" (§6.1); `CaptureRail` lives in the
   snapshot popup and the Alert Center pane so every chart-opening host inherits
   capture, including the RS/RW and Industry boards which previously had none;
   armed price alerts and D1 level watches paint as a read-only `GROUP_ALERTS`
   levels family on the worker; the Yahoo forming-bar early print is suppressed
   for 15 minutes after the open and labeled when drawn; the reviewed-today
   marker renders on the snapshot, the Alert Center pane, RS/RW and Industry;
   the feed's star became a labeled Like→Focus verb; and one feed row per
   symbol/side/day folds repeats with a three-item escalation list and a
   30-minute open-burst digest. Three trader confirmations were taken before any
   §6.2/§6.3 code and are recorded in the spec's new §6.4.
   Deterministic gate: **3500 passed / 19 subtests, exit 0.**

   **Held ask-first, recorded not skipped:** the Focus Picks reviewed-today
   marker (that panel is editable watchlist *text*, so a marker injected there
   would land in data written back to the watchlists — a design decision the
   spec does not make), and §2.2's `review_host` for the boards (its remaining
   half is the setups table's advance-to-next-row flow, meaningless on a ranked
   board; §2.1's CaptureRail delivered what §2 was actually for).

   **Owed — the §8 exit gate, all live:** every entry point opening a chart with
   capture, watch controls and painted armed alerts; one desk morning confirming
   the forming-bar caveat replaced the inflated-gap rendering; a dislike recorded
   from the RS/RW board appearing as a badge everywhere that symbol renders that
   day; and §6.1's ignored-symbol armed-watch hit feeding and sounding while
   automatic Focus D1 interest for that same ignored symbol stays absent.
5. **R5 M5 signal engines. — §2, §5 and the FIRST of §3's engines BUILT;
   §3.2/§3.3 and §4 remain.** `scripts/indicators/smi.py`, `efficiency_lrsi.py` and
   `heikin_ashi.py` are built, pure and green (42 hand-computed tests), and
   `scripts/completed_bars.py` is now the one intraday completed-bar rule, with
   `weekend_strength` delegating to it and a characterization test proving the
   move changed nothing. Nothing imports `indicators` yet, so **no packaging
   trigger has fired**.

   All three §8 questions are answered and recorded in that spec: the confluence
   alert stays **M5 Focus only**; an ORB candidate is an **Alert Center
   annotation**; and the engines get **a new `M5_SIGNAL_TAG` family** rather
   than reusing `d1_flag` — main feed, no tier-gate bypass, not loud by default
   where the spec is silent, and **not privileged** against R4 §6.3, because an
   unproven engine must be foldable. Per-engine identity rides `bounce_type`.

   **§3.1 LRSI cross wired 2026-08-17.** `scripts/m5_signal_engines.py` is the
   pure bars→events seam; the detector gained `check_lrsi_cross_setups`, the
   `M5_SIGNAL_TAG` family, and its own `M5_SIGNAL_TYPE_DEFAULTS` toggle map kept
   deliberately OUT of `BOUNCE_TYPE_DEFAULTS` so `BOUNCE_LEARNING_TYPE_KEYS` is
   unchanged. **The packaging trigger fired and was fully discharged in that one
   commit**: `indicators` moved out of the drift allowlist into the spec's
   `collect_submodules`, four modules joined the selftest roster, and a
   clean-cache rebuild moved the frozen count **51 → 55**.

   **BUILT OUT 2026-08-18 (trader integration redirect).** §3.2's confluence,
   §3.3's first-candle ORB flow and §4's any-bounce watch are all built, wired
   and green, with the prior-anchor AVWAP line carried onto the zone-arms entry
   behind a golden fixture that passes unchanged (spec §9.2). The redirect is
   §8.2's own first reopen trigger. **All four new alert types default OFF**, so
   what §7's per-engine desk session now decides is which of them earns a
   default-on — audibility, not existence. **Owed, live only:** that session,
   for each engine, plus one observed any-bounce firing naming its level.

   **Superseded — the original ordering, kept because it explains the shape:**
   §7 ordered the remaining engines behind a desk session per engine — the confluence (§3.2) and
   first-candle ORB (§3.3) engines wire only after a session confirms the LRSI
   cross's alert volume is sane. §4's any-bounce watch is not behind that gate
   but needs the prior-anchor AVWAP line added to the D1 scan output, which is
   an **ask-first** edit to `master_avwap_lib/legacy.py`. BounceBot's ad-hoc
   completed-bars call sites still migrate opportunistically, never as a silent
   change to a shipped detector. Original scope follows.

   New pure indicator modules (TC2000-parity SMI,
   efficiency-LRSI under a non-colliding name, Heikin-Ashi reversal), the LRSI
   cross alert type, the HA+SMI+LRSI confluence alert (Focus-scoped), the
   first-candle ORB candidate flow, the AnyBounceWatch multi-level armed watch
   (including the new prior-anchor AVWAP line in the D1 scan output), and a
   shared completed-bars helper. Golden fixtures gate live wiring; packaging and
   frozen selftest update when `scripts/indicators` gains its first importer.
   Spec: `docs/M5_SIGNAL_ENGINES_PLAN.md`.
6. **R6 Small operational wins. — (a) BUILT 2026-08-17; (b) DECIDED 2026-08-17
   and narrowed to tests/docs; (c) diagnostic ACTIVE + evidence-led repair BUILT
   2026-08-20.** (a) AI-jobs visibility:
   the routine log line in `scripts/run_ai_jobs.ps1` no longer reads as a caller
   error, and `operations_audit` gained an **`ai_jobs` row** over the AI job
   ledger. It resolves the store by path and never imports `ai_jobs`, because
   that package is in `PACKAGES_NOT_IN_THE_BUNDLE` and System Health is frozen;
   four pins keep the duplicated rule honest. An unset store reads HEALTHY ("off
   by choice"), never UNKNOWN. (c) The `ui_stall_watchdog` **already existed** —
   `scripts/ui/stall_watchdog.py`, installed from `ui/app.py`, with setting +
   env overrides, a reader CLI and its own tests. Two measured AppHang events
   on 2026-08-20 justified activating it machine-locally and repairing the
   delivery seams it exposed; the watchdog implementation itself was not
   changed. **Its first day already paid for itself**: the watchdog caught the
   repair's own regression - both sweeps waited on input idleness with no upper
   bound, which starves the process's only collector while the desk is in use
   (8 GB in 90 minutes, then a 298 s and a 200 s freeze). Fixed 2026-08-21 by
   bounding both waits; see `CURRENT_CHECKPOINT.md` (tenth pass).
   The bounded diagnostic week remains the live gate, and it **begins at
   the 2026-08-21 relaunch** — the desk ran the pre-fix frozen exe until then,
   so every earlier `ui_stalls.jsonl` row is baseline, not evidence. The desk
   now launches from source; see `CURRENT_CHECKPOINT.md` (eighth pass).
   (b) Evidence-ledger rotation — **DECIDED 2026-08-17
   (delegated, R5 §8.1 pattern): do NOT rotate the live file now.** Measured
   that day: 370 MB / 318,040 rows / 25 sessions (~15 MB/session; the ~247 MB
   here and ~106 MB in `operations_audit.py` were both true when written and
   are stale by growth), and the full boot re-parse costs **2.2 s** —
   `_load_resolved_events` is strictly session-filtered, so every field it
   reconstructs (dedupe set, resolved ordering, followup horizons, snapshot
   markers, and the `latest_completed_bar_end` watermark, whose max only ever
   sees current-session rows) is untouched by the presence or absence of
   closed sessions. Rotation therefore buys ~2 s of boot time while risking
   the two readers that DO span sessions: the daily calibration replay wants
   full history deliberately, and `research_warehouse/ingest_existing.py`
   resumes each source from a (file SHA, max line-offset) watermark that
   in-place truncation or compaction silently breaks — rows below the old
   watermark would never reach bronze. The size fix is already owned
   elsewhere: the locked warehouse plan (§19) declares this ledger a bronze
   source whose "retention cleanup unlocks after verified ingestion" (Phase 3
   live evidence, not yet run). When that unlocks, implement it as
   **forward-only per-session segment files plus the monolith frozen in
   place** (each closed segment immutable = a clean ingest source; the frozen
   file's watermark stays valid forever) — never in-place truncation. What
   R6(b) is **CLOSED 2026-08-18**: item (2)'s read-only ledger audit and item
   (3)'s stale-comment fix both landed (`operations_audit._jsonl_ledger_rows`
   reports measured size, estimated rows and last write per ledger from the
   existing footprint walk, reading a 256 KB sample and writing nothing; no
   current size is hard-coded in code any more). Rotation stays declined; the
   reopen triggers below are unchanged. What it owed, for the record: **(1) the
   replay characterization fixture over
   `_load_resolved_events` is BUILT 2026-08-17** —
   `tests/fixtures/technical_integrity_replay_v1.json` +
   `tests/test_technical_integrity_replay.py`, 18 tests, every case in the
   specification below pinned, and **mutation-proven**: deleting the session
   filter fails 7 of them (including the watermark and the segmentation
   equivalence) and deleting the provenance strip fails 3. A **positive
   control** replays the same bytes against the next session, so "the filter
   excludes those rows" and "that field is unreachable" cannot be confused.
   `scripts/technical_integrity.py` was NOT edited. (2) the
   read-only JSONL-ledger audit via the existing footprint check; (3) the
   stale-size comment fix in `operations_audit.py` (ask-first applies to any
   `technical_integrity.py` edit; the fixture and audit touch tests/docs
   only). Fixture must pin: a started/resolved pair; an unresolved started
   recovering into pending with append-time provenance stripped; a resolved
   row with no started row suppressing stale state-seed pending; a followup
   chain with partial horizons (stays pending) and one fully complete
   (drops); all four snapshot-marker event types; a cross-session row of each
   type proven fully inert including its `as_of`; a truncated mid-flush line;
   the `(resolved_at, event_id)` sort tiebreak; and monolith-vs-segmented
   equivalence so the eventual segmentation is checkable rather than
   aspirational. Reopen triggers: warehouse Phase-3 verified ingest of this
   artifact passes (retention unlocks — implement the segment scheme then);
   boot replay measured >15 s or a session-rollover UI stall; diagnostics
   free space approaching the 5 GB floor with this file the driver; the
   calibration replay overrunning its overnight window (that is a separate
   windowing decision, not rotation).
   (c) **ACTIVE 2026-08-20 — evidence-led hang repair is BUILT; bounded live
   week owed.** Two Windows `AppHangB1` events (07:19 frozen exe, 14:16 source)
   triggered measurement rather than speculative tuning. The 3-second Qt
   health tick was parsing the 63.88 MB / 370,109-row AVWAP history (1.268 s
   warm); GUI report/audit work measured 0.540 s; aligned 30/60-second timers
   and fixed-boundary full GC amplified the stalls. The scanner now publishes a
   signature-validated compact active-bounce projection; health fallback is
   single-flight off-thread; GUI report/audit writes are serialized off-thread;
   Alert Center D1/earnings reads are memory-only with chart-worker prefetch;
   timer phases are staggered; full GC waits for input idleness. The existing
   watchdog is enabled machine-locally at 50 ms from the next launch. **Gate:**
   run it for one bounded live week and require no new Application Hang, no
   repeated repaired-seam culprit, and verified scan/report behavior. This is a
   delivery/responsiveness repair only — no detector, score, threshold or alert
   decision changed. (d) **Auto journal
   is a mapping, not new work**: the trader's ask resolves to the QUEUED nightly
   `journal_import` slot (`docs/LOCAL_AI_AUTOMATION_PLAN.md` sec 6.4c — build
   only after the 6.4b live proof passes and the trader says go) plus the P3.5
   commentary journal; **the nightly slot half was promoted into R7 on
   2026-08-15** — see item 7; P3.5 is unchanged.
7. **R7 Journal reliability and UX. — BUILT 2026-08-15, live gates owed.**
   Tax-grade completeness
   from both brokers: stable execution/trade identity (annotations survive
   rebuilds), IBKR Flex as the primary historical source, wired Questrade
   activities, per-chunk partial persistence, a date-coverage ledger with a
   nightly `journal_import` runner slot that self-heals gaps (the promoted P3.3
   slice), position reconciliation against both brokers with append-only
   audit-trailed corrections, booked BoC CAD conversion, and the rebuilt Journal
   tab (account/tax-status selection that never silently blends, date-range +
   calendar + fees views, R-multiples with alert prefill, pyqtgraph analytics,
   surfaced walk-away). Spec: `docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md`.
   Built on `phase05-r7-journal-reliability-ux`, cut from the R2 tip (trader
   redirect 2026-08-15, second of the day; see the preamble), in the spec's §9
   commit order. **Deterministic gates green: 3203 passed / 19 subtests, smoke
   7/7, frozen selftest 45/45 `(frozen)`, all exit 0.**

   **Owed, and none of it can start before Monday's validation day:** the
   trader-present finale — the live schema v2→v3 migration (dry-run report
   reviewed first, automatic file backup), the full backfill, account tax-status
   labeling applied to the live store, and reconciliation-week sign-off — then
   the spec's six live gates: coverage COVERED-or-NO_SESSION for every session
   day since inception, trade counts and commissions reconciling to one monthly
   statement per broker **to the cent**, one clean reconciliation week on both
   brokers, zero orphaned annotations (permanent SQL test), CAD totals
   spot-checked against published BoC rates for three dates, and ≥5 consecutive
   nightly `journal_import` ledger entries with coverage advancing and at least
   one observed self-heal.

   Two decisions taken at build time and recorded in the spec rather than left
   implicit: §5 fix 4's narrowing (only the unambiguous oversell is flagged;
   the naked sell is resolved by step 9's reconciliation) — **trader-approved as
   built** — and §4's three identity choices (the short `QT`/`IBKR` uid token,
   source precedence enforced at import time as well as in the migration, and a
   deterministic surrogate for a row with no execution id).

   **Release-candidate pre-flight fix pass (2026-08-16):** id-less Questrade
   identity now hashes only stable fill discriminators; v2 order-id-keyed
   partials are re-keyed before collapse and counted in the migration report;
   the GUI preparation gate reads persisted schema v3; the nightly slot refuses
   an existing pre-v3 database until that trader-present GUI migration; assembly
   orders fills by normalized instrument identity; and a failed Questrade
   activities cross-check makes the backfill/night fail. Deterministic regression
   coverage was added for all five findings; the live migration remains owed and
   untouched.
8. **R8 Weekend Prep. — BUILT 2026-08-15, live gate owed.** A guided five-step weekend routine
   (week in review, focus-pick review, week-windowed walk-away with the auto-tag
   review, H1/D1/Monthly strength discovery on the R2 formula via a new pure
   module — `strength_scan.py` untouched — and week-ahead prep adopting the
   orphaned `market_prep` weekly engine), all manual-refresh with zero IB
   traffic, adopt-to-swing-Focus routing through the existing membership-tracked
   injection. Includes the standalone `app.py` nav-title bugfix as its first
   commit — which turned out to be a **live crash**: the desk's Settings button
   raised `IndexError`, and eight nav titles from index 3 named the wrong page.
   Spec: `docs/WEEKEND_PREP_PLAN.md`. Built on `testing-week-2026-08-17`, cut
   from the R7 tip. **§5's filter table is trader-approved as proposed
   (2026-08-15)**, so the discovery step is no longer blocked.

   **OWED, not built (2026-08-20): the weekly trader-judgement synthesis.**
   Nightly deterministic grading of the veto cohort now runs
   (`ai_jobs.cohorts`, slot `veto_cohort_grading`) and the `trader_judgement`
   evidence scope exists but is **opt-in** — deliberately absent from
   `DEFAULT_SCOPES`. The cadence is decided (**weekly, on the weekend
   surface**, which is why it is recorded here rather than under the AI plan),
   but it is **gated on two weeks of graded rows** and is **NOT authorized to
   build**. Until then the scope is exercised by hand:
   `run_ai_jobs.py --scopes trader_judgement`. Live gate: one weekend where
   the graded cohort is read and the trader confirms the reasons ranked
   against forward returns are the ones they recognise.

   **Deterministic gates green after the adversarial repair pass: 3354 passed / 19 subtests, smoke 7/7, frozen
   selftest 49/49 `(frozen)`, all exit 0.**

   The release-candidate review closed A1–A19 and B1–B14. Weekend board state
   now persists across restart. The not-yet-built RRS-strength joins and Focus
   performance/pick-feedback/veto joins are retained as explicit future scope in
   the governing spec; they are not claimed by this build. R7 likewise defers
   true non-USD conversion, the Calendar year heatmap, and additional Analytics
   charts in its governing spec.

   **Owed: the one-real-weekend live proof** (spec §10) — the desk booting on a
   weekend with the tab present and no network activity until a button is
   pressed, zero IB traffic across the routine, all three boards refreshed with
   their per-timeframe wall clock recorded, a monthly board spot-checked for the
   absence of a current-month bar, one real Adopt verified in all four stores
   with nothing removed anywhere, one auto-tag confirm and one correction, a
   week-windowed walk-away, the week-ahead rendering only on its button press,
   progress surviving an app restart mid-routine, and the trader confirming the
   board character per timeframe before §5's filters count as proven. **The
   trader can run this as soon as the build lands** — it is read-only against
   their data and does not wait on Monday.

9. **Wishlist deep link into an external charting tool. — BUILT 2026-08-18
   (trader-directed).** Promoted from `WISHLIST.md` under the 2026-08-18
   integration redirect and built the same day: `scripts/external_chart_links.py`
   plus an "Open in TradingView" button on the arm bar, so every chart surface
   carrying that bar inherits it. The URL template is a machine-local setting,
   the symbol is validated before a URL exists, and a refused open is reported.
   Read-only in both directions — it opens a URL and reads nothing back, so no
   second source of truth about a symbol enters the system. TC2000 stays
   unwired by decision, not oversight (it answers no documented URL scheme).
   **No live gate**: the trader pressing the button once on the frozen desk is
   the whole proof, and the frozen selftest already covers the import.
   The rest of the wishlist triage produced no code: each remaining item needs
   one trader judgment, recorded in `docs/WISHLIST_OPEN_QUESTIONS.md`.

10. **Trader-directed integration set. — BUILT 2026-08-21 (trader-directed).**
   Four asks handed over together, each answered against a decision the trader
   made when the ambiguity was real rather than an AI's reading of it:
   (a) **"SMA incoming" veto reason** — vocabulary **v3**, an additive bump,
   with `canonical_veto_cohort` pooling every carried-over reason back to the
   earliest version carrying the identical definition, so the bump costs no
   sample and the v1/v2 split from the previous bump is recovered. Pooling is
   applied when the performance rollup is rebuilt, never at write time.
   (b) **Post-earnings + 2nd-dev claims on the like rail** — the three
   `post_earnings_*` families and `second_dev_breakout`, named by id rather
   than by admitting whole groups, because the ask was specific. The nine
   main-swing digits do not move; the extras continue on `0` then letters.
   (c) **RS/RW board under the M5 Strength Board** — the same
   `RrsSnapshotWidget`, a second listener on the one `rrsSnapshotChanged`
   signal, in a draggable splitter on the existing page. No new nav entry, no
   second fetch, no second chart widget.
   (d) **The candle that ate the chart** — `scripts/ui/bar_integrity.py`.
   Root-caused rather than restyled: the y-range is built from lows/highs
   while the body is drawn from opens/closes, so a bar violating
   `low <= open, close <= high` paints outside a viewport that still looks
   correct. Malformed bars now draw dashed and clamped, stay out of the scale,
   are counted on the chart and logged once to `bad_bars.jsonl` with their
   provenance.
   **Live gates owed:** (a) one veto committed on the desk under v3 and the
   pooled rollup read back; (b) one claim committed on a letter key; (c) the
   RS/RW half populating from a live BounceBot sweep; (d) confirmation from
   `bad_bars.jsonl` that the next occurrence is a malformed bar and not a
   well-formed aggregate row — the second case would move the fix into
   `bounce_bot_lib`, which is ask-first and was NOT touched here.
   **Diagnosed and deliberately not fixed:** `_get_cached_bars`
   (`bounce_bot_lib/legacy.py:8370-8371`) writes `latest_bars.setdefault(symbol,
   bars_ib)` for whatever duration/bar-size it just fetched, and
   `m5_chart_bars` falls back to that same plain key — so a symbol whose
   `|5 D|5 mins` key is missing can be charted from a daily or hourly series.
   Latent, not observed; detector-adjacent, so it needs the ask-first
   conversation and golden fixtures, not a quiet edit.

11. **Regime-pause "holding highs" - measured and expiring. - BUILT
   2026-08-21 (trader-directed).** The watch captioned MRK "holding highs" with
   a 75-minute-old high while price faded off it. Three defects, in increasing
   depth: the caption is a BATCH label applied to every symbol in the sweep;
   the qualifying predicate's third branch (`window_excess >= 0.20`) admits a
   name that is merely falling less than SPY; and nothing re-measures after the
   alert fires.

   **Built (no detector file touched):** `scripts/indicators/atr.py` (shared
   Wilder ATR) and `scripts/regime_pause_hold.py` (distance from the session
   extreme in ATR on completed bars, plus the queue verdict), wired into the
   Alert Center's existing 30s tick. Trader's thresholds: **1.0 ATR** and
   **15 minutes** from the later of the alert and the last new extreme, with a
   new extreme refreshing the clock. Expiry deletes from the QUEUE ONLY - the
   alert list, `alert_review_events.jsonl` (a `hold_expired` row) and the
   tracker's outcome rows keep it, which is what makes the rule gradeable.
   Uncertainty never deletes.

   **The detector gate, done in that order** (trader: "yes do it properly").
   `regime_pause_sweep_v1` was frozen against the unchanged detector first -
   four cases per side, each entering through a different branch of the
   defiance test - then the near-extreme condition was ADDED (never
   substituted, so the set can only shrink), then the fixture was re-frozen and
   a test now names exactly which rows left and which branch each had used.
   Three champion tests caught the first version dropping names whose ATR was
   unmeasurable; being AT the extreme needs no ATR and is holding regardless.
   The feed line gained a per-symbol measure, so the caption is no longer one
   batch phrase stamped on names that are not alike. Replayed on the day's real
   batch: 34% of longs and 28% of shorts drop, MRK and GFS among them.

   **The prior-day break and session VWAP** (trader, same day) were added on
   the same discipline: the fixture grew to six cases per side - each isolating
   one reason to be kept or dropped, each now two sessions because the new gate
   cannot be measured from one - frozen, changed, re-frozen, with a test naming
   which gate rejected which case. The pair is the M5 Focus adoption gate and
   is CALLED (`passes_focus_adoption_gate`), never restated; the numbers come
   from `regime_pause_hold.session_levels`. Three champion tests needed real
   fixtures rather than a change: no prior session, and no volume at all, so
   session VWAP was unmeasurable. Measured on the real batch: longs 38 -> 18,
   shorts 29 -> 18 across both gates.

   **Live gates owed:** a session where a "holding highs" row visibly leaves
   the queue within 15 minutes of the name rolling over; a row that keeps
   making new highs visibly surviving past 15 minutes; a read of `hold_expired`
   rows against forward outcomes to confirm the rule is not discarding winners;
   and a check that the tightened detector still produces a usable number of
   names on a normal day rather than a handful - it now passes fewer than half
   the longs it used to.

   **Named and NOT acted on:** `REGIME_BANGER_DAY_EXCESS_PCT` (0.75) and
   `REGIME_BANGER_WINDOW_EXCESS_PCT` (0.20) are still percentages, and the
   trader's own argument applies to them too - 0.75% is about nine ATR for the
   slowest name in that batch and two thirds of one for the fastest, so the day
   gate is biased toward fast movers. Changing it would move the flagged set in
   BOTH directions, which is a different decision from this one and needs its
   own fixture and its own trader call.

12. **GUI fluidity pass. - BUILT 2026-08-21 (trader-directed).** "I want this
   program to be very fluid to use." Measured first: 1843 stalls over 50 ms and
   1008 s blocked in 3h20m, plus the two GC freezes. The trader's own hypothesis
   - the DAS - was tested and ruled out (every hot path local; the share not
   mounted; a miss on it 0.0 ms). Fixed, in order of measured cost: per-widget
   `setStyleSheet` in the two busy lists (now `theme.qss` rules on properties),
   whole-list widget rebuilds (now diffed), `as_bar_dicts` on Qt (now memoized
   in `ChartDataService`), uncached settings and review-event parses (now
   stamp-cached), and the px/pt font arithmetic behind the `QFont` console
   flood. The stall watchdog now samples throughout a stall so the 56% it could
   only attribute to `app.exec()` names itself next time.

   **Live gates owed:** a full session compared against the same measurement -
   stalls per hour, median, p90, total blocked seconds, against 1843 / 238 ms /
   1.16 s / 1008 s, targeting no stall over 5 s and under ~60 s blocked; the
   working set after three hours (8.1 GB before the GC fix); and a console with
   no `QFont::setPointSizeF` lines. The method, the baseline and the reading
   guide are `docs/GUI_FLUIDITY_MEASUREMENT_RUNBOOK.md`; the first post-fix run
   already located the next target there (`live_state_for`, resolved per symbol
   per editor on every bounce alert). That last one **cannot be verified off the
   desk**: Qt warnings do not reach a piped stderr on Windows, so the fix rests
   on the arithmetic (unit-pinned) and the trader's next session is the proof.

Exit gate: each packet exits through its own spec; R1 and R2 land first per the
trader's ranking, then R7 before R8. **R7's code is complete**; what remains for
it is live evidence, which is why it does not close the phase on its own. A packet's live gates may overlap the next
packet's build only when no shared file is in flight.

### Phase 0.6 — R9: trade-review response packet (authorized 2026-08-22)

Source: `docs/analysis/TRADE_REVIEW_2026-08-21.md` §8–§9, its nine questions
answered on 2026-08-22 (Opus answer + Fable verification; working copies in the
session scratchpad). The trader answered the three decisions that needed him on
2026-08-22 and **authorized this packet in writing the same day** ("I authorize
you to queue a packet for opus to implement"). That authorization covers the
file-scoped ask-first rule for the files named below; anything outside them is
asked about again. Build order is the list order. Nothing here touches a
detector's or scorer's output; R9.5 is shadow-only by construction.

1. **R9.1 Universe write floor + `universe_rebuild` ledger event (operational P0)
   — BUILT 2026-08-22, GREEN; one live gate owed.**
   *Owed: one real rebuild on the desk that writes a `universe_rebuild` row with
   `refused: false` and a plausible before/after, confirming the ledger row and
   the snapshot directory appear on the live machine.* Built as specified:
   `universe_write_floor()` = `max(500, 50% of the prior universe_all.txt count)`
   with a missing, empty or **unreadable** prior failing OPEN (returns 0);
   `force=True` carves out the floor but never the zero-symbol refusal;
   `_record_universe_rebuild()` appends a deliberately **keyless**
   `universe_rebuild` row to `job_ledger.jsonl` on every write attempt (keyless
   so `JobLedger._replay` cannot turn evidence into a phantom QUEUED job);
   `_snapshot_universe_lists()` keeps the outgoing lists under a run-scoped name,
   bounded to the last 10. Manual carve-out wired at both entry points (trader
   decision 2026-08-22): the Universe tab's Build button forces, and
   `rebuild_universe_if_stale` forwards its existing `force`, so "Rebuild
   universe now" overrides while the scheduled stale tick does not. 14 tests in
   `tests/test_universe_builder.py`; suite 4073 passed / 19 subtests, exit 0.

   Original statement of the defect: `scripts/universe_builder.py::build_universe`
   refused to write only when the
   screen yields **zero** symbols. On 2026-08-20 13:31–13:35 PT a rebuild that
   priced ~25% of the listing overwrote a 1,487-name universe with ~370–590 and
   blinded the D1 scanner for all of 2026-08-21 (six in-session runs at 409–533
   symbols against a 1,088–1,513 band). Build: (a) refuse the write — keep the
   existing file — when the new count is below **max(500, 50% of the existing
   `universe_all.txt` count)**; a manual rebuild keeps a `force=True` carve-out
   exactly as the quiet-hours gate does; an unreadable prior file fails OPEN
   (write what we have — never leave no universe). (b) Write a `universe_rebuild`
   row to `job_ledger.jsonl` on **every** attempt, with pre/post counts per list
   and a `refused` flag, so the event is visible whether or not it was stopped.
   (c) Snapshot the consumed watchlists under a run-scoped name before writing.
   Tests: the 2026-08-20 shape (1,487 → ~400) must refuse; a normal 1,487 → 1,450
   must write; zero and unreadable-prior paths pinned.

2. **R9.2 The LIKE: always ask why, and stop parking the symbol — BUILT
   2026-08-22, GREEN; one live gate owed.**
   *Owed: one desk session in which a LIKE is filed and the symbol is still seen
   to alert afterwards (and, on an AWAY day, still reaches the hourly D1 push).*
   Built as specified. (a) `commit_like` refuses an empty or whitespace-only
   why; the claim key and double-click both route through `_prompt_for_why`,
   which selects and moves focus rather than committing; placeholder reads
   "why (required)"; the why is the row's existing `note` — no schema change.
   (b) `AlertChartReview.likeAdvanceRequested` is a new signal separate from
   `removeTodayRequested`; `AlertCenterPanel._advance_after_like` records
   `like_advance` and advances, touching neither `_ignored_symbols`, the
   symbol's other queued alerts, nor any auto-adopted Focus pick. The veto's
   retire-and-park path is unchanged. `review_learning.TAKE_ACTIONS` now
   contains `like_advance`. (c) `SymbolSnapshotDialog` already only advanced and
   needed no change; it inherits (a) through the shared rail. 13 new tests;
   four that pinned the superseded one-click/retire rule were rewritten to pin
   the new one, and `test_a_like_also_retires_the_chart` was deleted outright.
   Governing spec updated: `docs/CHART_REVIEW_WORKSPACE_PLAN.md` §7, plus the
   R4 gate row that now names which of the two "likes" it answers for.
   Suite 4085 passed / 19 subtests, exit 0.

   Original statement (trader decisions
   2026-08-22: *"if I like a chart I should always be prompted with why"*; parking
   removal = option (c) from the review's Q1, authorized as part of this packet).
   Measured first (2026-08-22): 40 of 52 `like_claim` rows retired the chart AND
   put the symbol on `alert_center_ignored_symbols.txt` for the day (34 symbols
   on 08-20, 6 on 08-21); a parked symbol also stops emitting `d1EventRecorded`,
   so on an AWAY day a LIKE silently drops the name from the hourly D1 phone
   push; and because the like is routed through `remove_today`, which
   `review_learning.REJECT_ACTIONS` classifies as a rejection, **every LIKE is
   currently counted as a dismissal by the review-learning loop.** Build, in
   `scripts/ui/widgets/capture_rail.py`, `scripts/ui/widgets/alert_chart_review.py`,
   `scripts/ui/panels/alert_center_panel.py` (and the symbol-snapshot host, which
   shares the rail):
   (a) **Why is required.** The claim digit / double-click selects the setup and
   moves focus to the why field; Enter commits; an **empty why does not commit**
   (same mechanic as the veto vocabulary's `note_required`). The chart stays until
   the why is given, then the existing advance runs. The why lands in the row's
   existing `note` field — no schema change. An ignorable prompt would recreate
   the empty-`dislike_reason` failure, which is why it is required; relaxing it
   is one constant.
   (b) **A LIKE advances the queue and nothing else.** It no longer emits
   `removeTodayRequested`; it takes an advance-only path (new signal or parameter,
   host's choice), does not touch `_ignored_symbols`, and leaves the symbol's
   other queued alerts alone. The review store records a new action for it
   (`like_advance` or similar — additive, free-form string) and
   `review_learning.build_episodes` must treat it as positive, never as a
   queue-clear. The veto's retire-and-park path is unchanged.
   (c) Both hosts. `SymbolSnapshotDialog` already only advances; it inherits (a)
   through the shared rail.
   Tests: a like with an empty why writes nothing and stays; a like with a why
   writes the row, advances, and the symbol is absent from the ignore set and
   still reaches `add_alert`; a veto still parks; `build_episodes` on a
   `like_claim`+`like_advance` pair yields a positive episode.
   **Parked as PLANNED, not authorized:** Q1(b), a one-click hand-off *request*
   from the rail to the Focus surface in the `vetoDayTradeRequested` shape. It
   grants the rail live reach, so it needs its own golden fixture (the placement
   request path) and its own rung before it is built.

3. **R9.3 Rebuild the setup scoreboard from the right stores — BUILT 2026-08-22,
   GREEN; no live gate (read-only analysis).**
   `scripts/setup_scoreboard.py` + `docs/analysis/SETUP_SCOREBOARD_2026-08-21.md`,
   classified in `docs/README.md`. Measured: 239,422 rows scanned, 14,452 finals,
   **6,907 in window over 20 sessions** (not 21 — one weekday has no finals, and
   the report says so). Exclusions applied *before* any ranking: **1,164 (16.9%)
   with no EOD close obtained** and **212 below the 0.1%-of-entry risk floor**,
   leaving **5,608 usable**. Condition (iii) is answered rather than noted: every
   one of the 1,164 has `eod_close` **exactly** equal to `entry_price` and **none**
   of the 5,743 settled finals does, so the zero mass is the writer defaulting a
   close it could not read — not a scratch population — and 563 of them are
   stopped-out trades scoring 0 instead of about −1R, which biases every mean
   upward. 251 never advanced a bar at all. Trimmed mean (10%) + median +
   stop-out rate sit beside every R; cells are ranked only at n ≥ 30; the swing
   block is measured against the file's own `baseline_every5` control and carries
   an explicit guard that a positive lift means *lost less than the control*.
   The regime axis the review called starved at n=130 is **5,608 rows across 5
   environments**. §5 declares the frozen forward window (40 sessions, must span
   bullish/bearish/chop, exclusions fixed in advance) — the only route by which
   anything here becomes §7 gate-2 eligible. It promotes and demotes nothing.
   19 tests in `tests/test_setup_scoreboard.py`; suite 4104 passed / 19 subtests,
   exit 0.

   Original statement.**
   Inputs: `data/runtime/intraday_bounce_outcomes.csv` `final` rows (6,907
   in-window, 21/21 sessions; bounce type from `event_id`'s trailing `-`-joined
   key; regime / `session_rvol` / sector / RRS / internals from `context_json`)
   and `output/reports/setup_playbook_episodes.csv` (127,926 rows, `stop`/`risk`/
   `net_r` on every row, its own `baseline_every5` control). Read with
   `chunksize`/`usecols`; report only cells with **n ≥ 30**. Hard conditions,
   each from a measured defect: (i) a **risk floor** — drop or cap rows with
   `risk_per_share` < 0.1% of entry and count them (penny stops produce ±655R:
   `regime_pause_rw` all-time mean −1.82 vs trimmed −0.28); (ii) **trimmed mean
   (10%) + median + stop-out rate** beside every R; (iii) explain the **16.9% of
   in-window finals with `close_r` exactly 0** (1,164 of 6,907; every large
   family's median is 0.000) before ranking anything; (iv) lift against
   `baseline_every5`; (v) **end by declaring the frozen evidence window for the
   next inspection** — this is the only path by which anything ever becomes §7
   gate-2 eligible. It cannot promote or demote anything (gate 2 is post-hoc
   here). Deliverable: a script under `scripts/` plus a report under
   `docs/analysis/`, classified in `docs/README.md`.

4. **R9.4 `thetalongs.txt` — BUILT 2026-08-22, GREEN; one live gate owed.**
   *Owed: one Master AVWAP scan on the desk in which DRAM reaches the theta
   report (or is honestly absent for a stated rule reason — earnings buffer, no
   weekly chain, support stack), labelled `via thetalongs.txt`.*
   `THETA_LONGS_FILE = LONGS_FILE.with_name("thetalongs.txt")` and
   `load_theta_long_symbols()` in `master_avwap_lib/legacy.py`; the file is
   optional and an absent **or unreadable** one returns `[]` with a warning, so
   it can cost those names but never the run. `resolve_scan_sides()` is the whole
   seam: `side` stays list membership for every detector, `theta_side` is LONG
   for anything on the list **regardless of long/short membership**, and a
   theta-only name resolves LONG rather than falling through to a phantom SHORT.
   The names join `symbols` (a name on no list is never scanned, so it could
   never be evaluated) but deliberately **not** `longs`. The two theta calls take
   `theta_side`; nothing else does. Rows carry `theta_list_source` and the report
   prints `| via thetalongs.txt`, since a short thesis appearing in a LONG-only
   sold-put section otherwise reads as a bug. The home-folder
   `thetalongs.txt` was created with DRAM in it. 14 tests in `tests/test_theta_longs_list.py`, including the
   characterization guarantee that an empty list moves no side at all; suite 4118
   passed / 19 subtests, exit 0. IB pacing budget untouched — one extra symbol.

   Original statement (trader 2026-08-22: keep the
   LONG-only gate; add the list; DRAM on it). `evaluate_theta_put_candidate`
   returns `None` unless `side == LONG`, and `side` is long-list membership
   (`scripts/master_avwap_lib/runner.py:597`), so a wheeled underlying on no list
   is never evaluated — the window's entire positive P&L (+$1,087.72, four DRAM
   short puts) was invisible to the engine. Build: an optional home-folder
   `thetalongs.txt` whose names reach the sold-put/PCS evaluation LONG-side
   regardless of long/short list membership; the theta report labels their
   provenance; the trader's own-names invariant applies (never auto-removed).
   Minimal wiring wins — a handful of names against the locked IB quote budget is
   negligible, and the locked pacing budget itself is untouched. Strategy
   context, so the engine's assumptions stay right: the trader is a **put
   seller** (wheel); calls are sold only on assigned shares.

5. **R9.5 `sector_cohort_divergence` — BUILT 2026-08-22, GREEN, AT SHADOW.**
   Golden fixture **frozen first**, per plan.md sec 5:
   `tests/fixtures/sector_cohort_v1.json` (via
   `scripts/build_sector_cohort_fixture.py`, which refuses to overwrite without
   `--force`), five hand-constructed cases isolating one rule each — fires short,
   fires long, two-qualifying-bars near miss, never-reaches-threshold, and one
   violent bar that reverses. It satisfies the repo-wide Milestone 3 fixture
   contract. *A defect the fixture caught in itself:* the first draft expressed
   each path as a cumulative move but let `path_pct[0]` be non-zero, which
   re-based every series and turned the gap-down case into a gap-up one; the
   generator now asserts `path_pct[0] == 0.0`.
   `scripts/sector_cohort_divergence.py` implements the rule verbatim
   (|spread| >= 0.75% persisting >= 3 consecutive **completed** bars, session
   only, unknown sector excluded, no benchmark = no observation), plus
   `member_entry()` reusing the archetype through
   `chart_snapshot.session_vwap_series`. Gate 1: `CONFIG_VERSION` +
   `config_hash()` (which excludes `enabled`, so switching it off is not a
   different engine). Gate 3: a coverage row on **every** run including quiet
   ones — otherwise a calm market and a dead collector look identical. Gate 7:
   `SECTOR_COHORT_DEFAULTS["enabled"]` ships **False**. Single-flight
   `run_shadow_pass` with an injected fetcher; the default is batched yfinance,
   **zero IB traffic**. Output is append-only JSONL at
   `diagnostics/shadow_evidence/sector_cohort/`. **First real day written
   2026-08-22 over the 2026-08-21 session: 20 ETFs measured, 78 benchmark bars,
   1,560 bars consumed, 11 cohort observations, XLU short from 10:35 ET.**
   27 tests; suite 4145 passed / 19 subtests, exit 0; selftest 56/56.
   **It stops at SHADOW.** Nothing reaches a detector, score, ranking, routing,
   alert, watchlist, Focus, the review queue or `review_policy.json` — pinned by
   an AST test, not a substring scan, because the module's own docstring names
   those surfaces in order to promise it avoids them. Evidence before it is
   discussable is unchanged: **>= 40 sessions across bullish, bearish and chop,
   window declared before inspection.**

   Original statement (trader 2026-08-22: yes; **after
   R9.1**, because on 2026-08-21 AEP was outside the universe and no live layer
   could have grouped it). Spec is the review's §6e, verbatim: ~20 sector/industry
   ETFs, every **completed** M5 bar, `spread = ETF move from open − SPY move from
   open`, fire when `|spread| ≥ 0.75%` persists across **≥3 consecutive completed
   bars**; session-only, re-derived never carried; UNKNOWN sector excludes; member
   entry timing reuses the archetype (first completed bar 10:00–11:30 ET below
   session VWAP via `chart_snapshot.session_vwap_series` and below the prior
   bar's low, session high set in the first three bars, prior-day low broken;
   stop = 6-bar swing high). Batched yfinance over the ETF set (the Strength
   Board template) — **zero IB traffic**, single-flight owner. Output is
   shadow-only JSONL under `diagnostics/shadow_evidence/sector_cohort/` with
   versioned config + `config_hash` (gate 1), coverage accounting (gate 3), and
   a single defaults-dict switch (gate 7). Ladder: PLANNED → IMPLEMENTED →
   **GREEN with the golden fixture frozen first** → SHADOW, and it stops there.
   Nothing reaches a detector, score, ranking, routing, alert, watchlist, Focus,
   the review queue or `review_policy.json`. Evidence before it is discussable:
   **≥40 sessions across bullish, bearish and chop, window declared before
   inspection.** Measured load at −0.75%: 16.5 observations/session.

**Deliberately not in this packet.** Re-keying `sma_incoming` off hotkey `0`
(review P7): measured 2026-08-22, v3 was live on the desk only from 12:19:42 PT
on 2026-08-21 and has had exactly one veto under it — the hypothesis is untested
at n=1; **re-check after one full session on v3** and only then decide. The
dislike-reason parser fix (review P7's other half), `policy_gate_check` honesty
(P8), dropping `guidance_score`/`take_prob` from display (P9), M5 run manifests
(P11), `event_id` on D1 alerts (P12), the family-namespace mapping table (P13 —
note `avwape_to_1stdev` exists **only** in the tracker namespace; the scanner's
emission store has zero rows of it, ever) and gate-7 for `regime_pause_*` (P4)
are all recommended and **unauthorized** until the trader says so.

Exit gate: R9.1–R9.4 green and on the desk; R9.3's report filed with its declared
window; R9.5 at SHADOW with its fixture frozen and its first JSONL day written.

**Status 2026-08-22: all five items BUILT and GREEN; the deterministic half of the
exit gate is met.** R9.3's report is filed with its declared window and R9.5's
fixture is frozen with its first JSONL day written. What remains is the "on the
desk" half — four live proofs, one per item, listed in `CURRENT_CHECKPOINT.md`:
a real rebuild writing a `universe_rebuild` row; a LIKE whose symbol is seen to
keep alerting; DRAM reaching (or being honestly absent from) the theta report
labelled `via thetalongs.txt`; and R9.5's shadow log growing over real sessions
toward its declared 40.

### Phase 0.7 — R10: Evidence Plane program (authorized 2026-08-22)

Source: the trader's 2026-08-22 evidence-quality brief (Fable synthesis v2 after
Sol's review). A **packetized** program, not blanket permission to modify the
named subsystems. The architectural objective is an immutable, point-in-time
**evidence plane**: capture facts once; record provenance, completeness and
uncertainty; derive replaceable views and reports; never let evidence collection
influence a live decision path.

**Ground rules (they bind every packet).**

1. No behavior change to any detector, scorer, gate, alert, watchlist, Focus
   store or `review_policy.json`. Golden fixtures run before and after each
   packet and must be byte-identical. A packet that cannot be built without
   touching such logic **stops and says so in the checkpoint**.
2. Ask-first: authorization covers evidence / provenance / presentation edits to
   the files each packet names — including `bounce_bot_lib/legacy.py` and
   `master_avwap_lib/legacy.py`. Any other file, or any non-evidence edit, is
   asked about first.
3. Every alleged defect is classified **PROVEN / REFUTED / UNKNOWN by
   reproduction** before its fix is designed.
4. Every new or changed store needs a writer/reader inventory (repo **and**
   warehouse ingestion), backward-compat plan, migration/canary, rollback,
   growth estimate, retention/segmentation, cold-push scope, health surface.
5. **Never rewrite history.** Append-only authorities with a schema NAME
   (`intraday_outcome_event_v1`, never "v2"); corrections are superseding
   events; known-bad legacy rows are tagged by a versioned reader-side rule
   registry (`evidence_rules.py`), never edited.
6. Missing data is uncertainty: `unresolved` with a reason, counted in every
   rollup beside n. **No path may write a number it did not measure.**
7. Completed bars only; `astimezone`, never `replace(tzinfo=None)`. New
   timestamps are tz-aware **UTC plus an explicit market-session identity**
   (`session_date`, exchange calendar including holidays and early closes).
8. One owner per timer/thread/store. Evidence maintenance is **not** gated on
   `auto_scanning_due` — that gate stops market activity and this is after-close
   recovery. It IS gated on zero IB traffic, cached/yfinance only, worker
   thread, idle cost.
9. Nothing expensive on the Qt thread.
10. Statistics on every evidence-facing summary: event / symbol / session
    counts; unresolved and excluded by reason; raw mean, median, trimmed mean
    (10%), p10/p90; PF with its all-win/all-loss convention stated; stop rate;
    concentration by symbol and session; raw and robust side by side; a
    session-block bootstrap interval; and a `discovery` vs `confirmation`
    label. **n ≥ 30 events is necessary, not sufficient.** The R clip is the
    existing **4R** where a view feeds ranking; the evidence report shows
    uncapped, 4R-clipped and trimmed together.
11. "No bare means" scopes to **evidence-facing** surfaces only (Daytrade
    Tracker, Setup Tracker and Focus Picks summaries, cohort performance CSVs,
    `setup_scoreboard.py`, the `review_learning` report). Existing scoring and
    promotion math is untouched.
12. **"Realizable R" is not a term this repo uses.** MFE/MAE are opportunity;
    each frozen exit policy is reported on its own; the best-of-policies number
    is labelled `oracle_best_ex_post_r` and is an upper bound, never a result.

**Trader decisions carried into the packets** (defaults; overridable in writing):
M5 Focus expiry stays as built (all M5 picks expire on the day roll, swing never
auto-removed — a survivor is a defect to diagnose, not a policy to design);
RVOL ≥ 1.2 lines are a Market-Journal-page overlay only, from the six symbols'
own daily bars, with the canonical D1 level store untouched; free-text journal
entries reach an AI scope **opt-in only**; a `launch_gui.py` single-instance
guard is authorized **only if** R10.0 proves concurrent instances; nightly and
derived reports live in a runtime report store with atomic last-good, while
`docs/analysis/` receives only deliberately frozen hand-committed audits.

**Commitments this program must not disturb.** R9.3's `setup_scoreboard.py` §5
declares a frozen forward window of 40 sessions. R10 must not alter it,
re-declare it, or measure it early; R10.C **extends that script** and prints the
declared window unchanged, stating that it did not measure it. R10 creates no
second scoreboard.

1. **R10.0 Read-only evidence audit — no code changes bar one.** Deliverable is
   a decision register: `docs/analysis/EVIDENCE_AUDIT_2026-08-22.md`, classified
   in `docs/README.md`, plus a checkpoint entry. Reproduce and classify every
   alleged defect (D1–D8 intraday outcomes, S1–S5 D1 tracker, F1–F6 Focus,
   C1–C4 capture/context) with the reproduction command and its numbers; for D1
   specifically establish **process lifetimes**, not start marks, and say
   whether two desks were alive at the same instant. Inventory every store the
   program will touch (writers, readers, timers, threads, schema + version,
   size, growth, downstream consumers including warehouse ingestion and cold
   push). Produce the **canonical data dictionary and family-namespace map**
   (the review's P13, now a prerequisite). Decide and write down: the authority
   between `bouncers.txt` and the outcome ledger; the exchange-session calendar
   source; the intrabar stop/target collision rule (predeclared conservative:
   stop first, ambiguous count reported); cost and slippage assumptions for
   simulated exits; the H1 legacy timestamp classification **by evidence**; and
   one written risk-floor definition reconciling R9.3's 0.1%-of-entry floor with
   the two existing 4R clips. Growth/retention contract per new store.
   **One code change is authorized here:** make `journal_import`'s nightly
   failure observable (non-empty `error`, traceback in the ledger row); its
   cause is reported, not fixed. **Stop after this packet and hand off** — the
   trader accepts the register before R10.A.
2. **R10.A P0 runtime and outcome integrity** (D1 if proven, D2, D3, D4, D7, D8),
   **plus the dated evidence snapshot** (trader instruction 2026-08-22: *"Any and
   all very important files that we use occasionally should go to the server with
   the massive HDD."*). The snapshot lands here because it is ground rule 4's
   backup-and-restore contract for every store R10 creates, so it must exist
   before the first ledger does.

   *Snapshot half BUILT 2026-08-22, GREEN.* Measured scope gap: the hourly cold
   push covers `data\daily_bars`, `data\intraday_bars`, `output`, `logs` and
   `away_report_archive` (~270 MB), and deliberately excludes the hot state that
   IS the evidence — `data\runtime` at **3.5 GB** (the 960 MB setup tracker plus
   its 939 MB `.bak`, the 203 MB outcome CSV, the journal SQLite, every outcome /
   cohort / Focus store), the **36 home-root evidence files**, `_tools`, and the
   machine-local diagnostics tree at **529 MB**. Decision 0015 stands, so the
   answer is a dated snapshot, never a move: `scripts/ops/evidence_snapshot.py`
   (tested) stages locally first and `scripts/ops/snapshot_to_das.ps1` robocopies
   to `\\MINI-PC\Trading Bot Data\backups\<YYYY-MM-DD>\`; an unreachable share
   exits 0 and leaves the staged copy, exactly like the cold push. Copy-while-hot:
   SQLite through the backup API, any file ≥ 256 MB must hold one size and mtime
   across a 60 s window or it is **skipped with a reason and counted** (never
   silently), and any file ≥ 64 MB is gzipped. `manifest.json` records size and
   SHA-256 per file. Retention 7 daily / 4 weekly / 12 monthly, `evidence_frozen/`
   permanent. `restore_from_das.ps1` restores **only into a scratch directory** —
   `restore()` refuses the home folder and the diagnostics tree outright, because
   a drill that overwrites live state is how a drill becomes an incident — and
   `--verify` re-hashes against the manifest. System Health gains an
   `evidence_snapshot` tile (absence is `unknown`, staleness degrades, a skipped
   file degrades). `push_cold_to_das.ps1` gains `data\runtime\evidence_ledgers`
   and both scripts' headers now say **two jobs, two scopes** so the next reader
   does not merge them. *Finding on the way:* `push_cold_to_das.ps1` existed
   **only** in `C:\TradingBotData\_tools` — the script protecting the evidence was
   itself unversioned; the repo copy is now the source of truth and a test
   compares the two byte for byte.

   *Ledger half: LARGELY BUILT 2026-08-23, GREEN.* Landed: the rule registry at
   v1 (`evidence_rules.py`, five rules, each re-measured against the live store -
   duplicates 742/609/430 and 394/345/300 exact, risk-floor 1,127,
   `h1_bar_start_v1` 9,623/9,914); the append-only ledger
   (`evidence_ledger.py`, `intraday_outcome_event_v1`, month-segmented, torn
   lines counted, writer identity on every row); the **dual-write canary** at
   the one CSV writer (fail-open, 50k/process, `evidence_ledger_dual_write=off`
   kills it); **no-fabrication finalization** (D2 - a stop-out finalizes at its
   stop, a measured trade at its last measured close, and a trade that saw
   nothing finalizes `unresolved` with a reason instead of a 0R); the
   **idempotent after-close sweep** (D3/D4 - needs no bars or IB, expires after
   3 completed sessions, files its own coverage) with a System Health tile; and
   **registration context** (D8 - family, engine version, day-part, RVOL,
   env_key, risk as % of price and as an ATR multiple), with the **tier emitted
   as its own `tier_assigned` ledger event** because every call site evaluates
   it *after* registration - which is why it was on 0 of 7,863 rows.

   *Sol's three reproduction blockers CLOSED 2026-08-23* (`137a4bf` lineage):
   the after-close scheduler gained two clocks and **two completion stamps** so
   a deferred sweep is retried rather than marked done, with a dedicated
   early-close seam (`scripts/market_early_close.py`) that leaves
   `market_calendar`/`market_session` untouched; finalization became **one
   transaction per trade** with a write-ahead intent, a disk re-read, a strict
   commit that raises, and `resolve_unfinished_finalizations()` settling
   interrupted attempts against the CSV; and the transaction is fenced across
   processes with `local_writer_lock`, with the authorized single-instance guard
   added to `launch_gui.py` as defence in depth. A failed commit is never
   reported as a finalization.

   *Still owed:* **one live weekday session with `outcome_sweep_autorun="on"`**
   (the mechanics canary - the switch stays OFF until the trader flips it);
   R9.5's shadow store aligned to month segments and `session_date`; a restore
   test of the ledger directory; the launch catch-up for the sweep (it runs only
   while the strategy thread is alive, so never in OFF); and the decision to make
   the ledger the authority, which needs the canary's own comparison first.
   Original text below.
   New append-only authority `intraday_outcome_events.jsonl`
   (`intraday_outcome_event_v1`, month-segmented); the pending dict becomes a
   reconstructable checkpoint, never the authority. One owner, one transaction:
   the BounceBot outcome worker owns every state transition and the Qt service
   may only *request* work. Bounded dual-write canary against the legacy CSV
   (no header widening — new fields go in `context_json`). **No fabricated
   values**: no session rows → `unresolved / no_bars_after_entry` with blank
   numerics; a stop-hit trade finalizes at its stop, never at the entry.
   Idempotent finalization shared by the after-close pass and the launch
   catch-up, expiring after 3 sessions without data, writing a coverage manifest
   to the ledger and the System Health tile. Registration events carry tier,
   family, engine version, day-part, RVOL, env_key, risk as % of price and ATR
   multiple, and the regime-pause/adoption-gate verdicts where computed.
   Single-instance guard only per the R10.0 verdict.
3. **R10.B Outcome semantics** (D5, D6, and the EAT/CAKE ask) - **BUILT 2026-08-24, GREEN; mechanics canary OWED** (one live session: LRSI registering gradeable rows, H1 stamping the bar close). Registry
   `outcome_semantics.py`, path capture `outcome_path.py`, fixture
   `outcome_path_eat_cake_v1`, health row `outcome_claim_kinds`,
   `evidence_rules.h1_bar_start_v2`. Typed registry
   keyed by family with a `claim_kind` — `entry_claim`, `annotation`,
   `information`, `unconfigured`. **Unknown families are `unconfigured`:
   counted loudly in the coverage manifest and the health tile, never given a
   manufactured trade.** Real signal bars for LRSI and ORB entry claims (the
   synthetic flat bar never reaches the ledger). H1 forward `entry_time` = bar
   close; legacy rows classified per R10.0's evidence rule. Path capture on
   every entry claim (MFE/MAE at 1/3/6/12/24/36/EOD bars, first-touch stamps,
   giveback, compact excursion path) so future exit models simulate offline
   with no refetch. Frozen exit policies reported each on its own —
   `eod_hold`, `trail_2bar_after_1r`, `vwap_close_after_1r`, `atr_1p5_trail` —
   plus `oracle_best_ex_post_r` as a labelled upper bound. Fixture: frozen EAT
   and CAKE M5 bars; the test asserts the honest calculation, not a desired sign.
4. **R10.C Robust deterministic evidence report** (extends R9.3; C4) - **BUILT 2026-08-24, GREEN.** `scripts/evidence_stats.py` is ground rule 10
   implemented once; the cohort CSVs and `setup_scoreboard.py` both read it.
   Bundle `setup_scoreboard_bundle_v1`, runtime report store with atomic
   last-good, `--freeze`, `--ledger`. R9.3's window reprinted unchanged and
   explicitly not measured. New
   `scripts/evidence_stats.py` implementing ground rule 10 exactly, used by
   every surface in ground rule 11. `setup_scoreboard.py` gains the ledger as
   input, the new axes, the frozen exit policies side by side, and a
   machine-readable bundle beside the Markdown. Its first section re-measures
   the four trader-named findings under the new discipline and says plainly
   whether each survives. Output to the runtime report store with atomic
   last-good; `--freeze` copies a dated audit into `docs/analysis/`.
5. **R10.D D1 setup tracker: point-in-time transition ledger** (S1–S4) - **BUILT 2026-08-24, GREEN.** `setup_tracker_ledger.py` + a digest sidecar
   (never a payload copy); `sessions_spanned` / `stale_horizon` measure S3a
   without re-selecting the future row; S3b fixed from cached daily bars,
   zero IB. **S2 did not reproduce on the current payload** - it needs a run
   during a live session - so the guard reports on every save.
   `setup_tracker_events.jsonl` (`setup_tracker_event_v1`, month-segmented)
   appended after every tracker run — `initial`, `transition`, `reopened`,
   `tombstone` — diffed via a small per-setup digest sidecar and **never by
   deep-copying the 960 MB payload**. Completed sessions only: no mark dated
   later than the run's `data_session`. `tier_outcomes` horizons in exchange
   sessions with `sessions_spanned` and a `stale_horizon` flag, and SPY-relative
   columns populated from cached daily bars (no IB).
6. **R10.E Focus provenance** (F1–F6). `focus_membership_events.jsonl`
   (`focus_membership_event_v1`) emitted by the one Focus writer, with a
   `membership_episode_id` and an owner of `trader` | `machine` |
   `unknown_legacy`; `expire_m5_if_new_day` emits `expired` per name it clears,
   so a survivor is a test failure **and** a visible gap. Enrichment is an
   asynchronous `enriched` revision from the worker and never blocks the Focus
   write or the Qt thread. Pick key includes category. Snapshots stamp the
   session observed and grade from the next completed session; a missed
   snapshot writes an explicit `observation_gap` row — membership is never
   reconstructed from current state. Outcome rows carry `days_on_list` and an
   age bucket; the rollup adds origin and age via `evidence_stats`.
7. **R10.F LIKE cohort grading** (C1, C3). `like_cohort_{picks,outcomes,performance}.csv`
   mirroring the veto trio; a deterministic `like_cohort_grading` slot
   **appended** after `veto_cohort_grading` (later phases append, never
   reorder). Stamps carry UTC + `session_date`, which makes the ET/PT mismatch
   moot.
8. **R10.G Market context ledger, auto-shift rows, calendar** (C2, season).
   Every auto-regime shift becomes a row. `daily_market_context.jsonl`
   (`daily_market_context_v1`), one row per session at close+grace, completed at
   next launch if missed with a `completed_late` flag and **never fabricated**.
   `config/market_calendar.json` multi-year capable, with a visible **degraded**
   state when the active year is not covered.
9. **R10.H Market Journal: store and two surfaces.** `market_journal.jsonl`
   (`market_journal_entry_v1`) behind one writer/service used by both surfaces:
   a "Journal" tab on the Trading Desk after "Capture" (M5 default in-session,
   Ctrl+Enter commits) and a left-nav "Market Journal" page (six D1 charts
   through the existing `ChartDataService` worker, the journal-only RVOL ≥ 1.2
   overlay, entries, an environment timeline with the auto-vs-manual agreement
   rate, the calendar strip, and the day-context table). The existing "Journal"
   page remains the trade/tax journal; the label difference is deliberate.
10. **R10.I Scheduled report slot and opt-in AI scope**, after two weeks of
    R10.A collection. An `evidence_report` slot appended to the runner
    (deterministic, no model) and an opt-in `market_journal` scope absent from
    `DEFAULT_SCOPES`. Nothing in this chain may reach a detector, score, alert,
    watchlist, Focus, the review queue or `review_policy.json`.

11. **R10.V Daily-bar unit repair** — **BUILT 2026-08-23, GREEN; one live scan
    day owed** (S1's mechanism; authorized by the trader's 2026-08-22 R10.0b
    decision as **option C-prime**). Runs **before** R10.D,
    because a point-in-time transition ledger built over a unit-mixed store
    would record the splice as history.

    *The defect.* IB returns regular-session daily volume in **round lots**
    (`whatToShow="TRADES"`, `useRTH=1`); Yahoo returns the consolidated session
    in **shares**. The store holds both, spliced. Measured: **1,227 of 1,737**
    comparable parquet files carry a >20× step (median 158×), and the ratio is
    symbol-dependent — SPY 1.0×, TSLA 56×, AAPL 81×, A 162×, NVDA 188× — so
    **no constant converts one unit into the other**. AVWAP bands are
    volume-weighted, so post-splice bars weigh ~1/100 and every AVWAP anchored
    before the splice freezes near its last pre-splice value: 30,003 of 60,519
    mark-days carry different levels. Stops did not move (0 of 9,331 — stored at
    scan time, never replayed), so the stop stayed fixed while the replayed
    target moved beneath it. Evidence:
    `docs/analysis/DAILY_BAR_VOLUME_CLIFF_2026-08-22.md`,
    `docs/analysis/EVIDENCE_AUDIT_2026-08-22.md` §S1b.

    *Interim measure, already landed* (`d031e89`): `daily_bars_source="yahoo"`
    pins the fetch so the store stops getting more mixed;
    `daily_volume_mixed_v1` in `scripts/evidence_rules.py` tags what is already
    written (13 of 15 manifest-covered sessions mixed, back to 2026-07-31; older
    sessions unmeasured, not clean).

    *Steps, each a commit — all seven landed 2026-08-22/23* (`7e0f217`,
    `e7b0fdf`, `b4497f2`, `069401c`, `5720c5b`, `d8d7a2a`, `740b591`).
    **This is a detector-input change, so plan.md §5 binds: fixtures first.**

    1. **Freeze the AVWAP golden fixtures** as they are, and prove none reads
       the live parquet. Add a **mixed-unit fixture** that feeds one Yahoo-unit
       and one IB-lot segment through `calc_anchored_vwap_bands` and pins the
       *current* (wrong) output, so the repair is a visible fixture change
       rather than a silent one. **Never swap the σ formula.**
    2. **Provenance on the store.** Parquet gains `source` (`yahoo` | `ibkr` |
       `cache`) and `volume_unit` (`shares` | `lots_rth` | `unknown`), written
       from the in-memory source the fetch already carries and today drops
       before the write. Existing rows read `unknown`. Arrow metadata carries
       `daily_bars_schema=v2`; every consumer reads v1 and v2, one test each.
    3. **Volume policy at the write seam.** Only `shares` volume is written; an
       IB-sourced frame contributes price columns and `volume=NaN` with
       `volume_unit=lots_rth`, never a rescaled number.
       `_normalize_daily_bar_frame`'s `keep="last"` becomes **prefer `shares`
       over `unknown` over NaN**, so a Yahoo row is never overwritten by an IB
       row again.
    4. **Backfill.** One batched yfinance sweep (`auto_adjust=False`, as today)
       over every parquet, rewriting rows whose `volume_unit != shares`, with a
       dated pre-backfill copy of the whole directory in
       `evidence_frozen/daily_bars_pre_backfill_<date>` and a manifest (files
       touched, rows rewritten, rows left `unknown`, per-file first-cliff date
       before and after). **Zero IB traffic.** The 221 unmeasurable files are
       refetched too and reported separately.
    5. **Re-freeze** every AVWAP-derived golden fixture that changed, in the
       same commit as the evidence showing *why* each moved; the step-1
       mixed-unit fixture is the control.
    6. **Health.** A tile reporting rows by `volume_unit`, files with any
       `unknown`, and a cliff detector (early/late median ratio > 20×) re-run
       nightly from the snapshot job, so a recurrence is loud the next morning.
    7. **Forming bar (S2).** The tracker catch-up trims
       `daily_frames_by_symbol[symbol]` to `<= data_session` before
       `recompute_tracker_setup_record` — evidence-only by construction (the
       docstring already claims it), fixtured by an 08-21-shaped run that must
       mark nothing for today. Ask-first is satisfied by this packet for that
       function only.

    *Exit gate.* Fixtures re-frozen with rationale; backfill manifest filed;
    **0 rows with `volume_unit != shares` that Yahoo can supply**; one live scan
    day on the repaired store. **No scoring, σ, ranking or threshold change
    anywhere.**

    *Gate correction (2026-08-23, measured).* The gate was written as "the cliff
    detector reads **0 files > 20×**". That is not achievable by any correct
    implementation, because **a 20× volume step is a real thing that happens to
    real stocks**: after a full single-source rewrite, 19 files still show one —
    DJT at its 2024-01-16 listing, OKLO's 2023-09-14 de-SPAC, POET, FFAI, QXO,
    SOXS — every row of them `source=yahoo`, so the step cannot be a unit
    artifact. The **unit** gate above is the falsifiable one; the cliff detector
    stays as a secondary signal, and a cliff in an all-`shares` file means
    "market event", not "defect". Measured after the applied backfill:
    **1,116,982 of 1,117,170 rows (99.98%) are `shares`**, cliffed files
    1,795 → 53, median residual ratio 158× → 29×, unmeasurable 0.

**Order (trader, 2026-08-22).** R9.4 first, then R10.0 in parallel; **R10.A
starts only after the trader accepts R10.0's decision register**; R9.5 after
R10.A. **R10.V (item 11) was authorized 2026-08-22 night and runs before
R10.D**, because a transition ledger built over a unit-mixed store would record
the splice as history. *Deviation on record:* R9.4 and R9.5 both landed on
2026-08-22 before
this program was registered (`36abb14`, `ba931a5`), so R9.5 did **not** adopt
conventions set by R10.A. Its store
(`diagnostics/shadow_evidence/sector_cohort/sector_cohort_shadow.jsonl`,
`sector_cohort_shadow_v1`) is append-only with a schema name, a `config_hash`
and a per-run coverage row, which is consistent with this program's rules but
was not derived from them. R10.0 inventories it with the other stores and names
any reconciliation R10.A should make.

Gates: a **mechanics canary** — one live session per packet that touches a live
writer (R10.A, R10.B, R10.E, R10.G, R10.H), with that packet's manifest or rows
visible on the desk. An **evidence-quality gate** — two weeks of R10.A/B
collection before any evidence-quality claim. Promotion and demotion remain with
Section 7 and R9.3's declared window; nothing in R10 promotes anything. Frozen
exe triggers (new runtime asset, new page, new top-level module) require a
rebuild plus `dist\TradingBotV3\TradingBotV3.exe --selftest`, with commit time
and exe mtime recorded side by side in the checkpoint.

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
