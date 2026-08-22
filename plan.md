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

3. **R9.3 Rebuild the setup scoreboard from the right stores (read-only analysis).**
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

4. **R9.4 `thetalongs.txt` — a theta-only long list** (trader 2026-08-22: keep the
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

5. **R9.5 `sector_cohort_divergence` to SHADOW** (trader 2026-08-22: yes; **after
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
