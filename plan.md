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
- the Tk GUI was removed on 2026-09-03 (assessment packet F2); `scripts/ui` is the only UI.

**What the program is for, in the trader's words (decision 0016, 2026-09-02).** The
trader answered twelve questions one at a time; the record is
[`docs/decisions/0016-trader-vision-and-priorities.md`](docs/decisions/0016-trader-vision-and-priorities.md)
and it is the tie-breaker for every prioritisation call. The short form:

1. Get **which names are shown** right before **when to enter**.
2. A name was right to show when it **moved** (a held D1 level then the move, for a
   swing; a held intraday level then the maximum favourable excursion, for a day
   trade); the trader's likes only say where to look. **Win rate** is the headline
   swing statistic, because losses run about 1.5x the best wins.
3. One click from any screen teaches the bot what the trader likes; words are
   optional; the click is always processed.
4. "What is working lately" (a rolling ~20 sessions, no regime label) lives on the
   **Trading Desk** and in Weekend Prep with a display-only priority switch - never a
   mute, never a filter. The Research tab is not a trader surface.
5. The trader sits on the Capture tab; the Alerts, D1 Focus, Armed tabs and the
   Universe page are unused; the Strength Board must match the trader's own TC2000
   scan (decision 0016 item 9) before it is compared to it.
6. Tagging is the slow part of journaling: the P6a tagger runs nightly, the trader
   corrects. Weekend Prep gets one Refresh, a verdict card and readable tables; the
   Market Journal is one box; Away Recap shows more names with charts for a 10-30
   minute evening review.

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

As of 2026-09-02:

- `main` is the running branch: the desk launches from source on `main` by trader
  decision (2026-08-26), and every Phase 0.13 packet (P0-P9, review rounds R1-R3) is
  merged. `CURRENT_CHECKPOINT.md`'s "Active state at a glance" block carries the
  measured baseline (6,091 tests, exit 0, on 2026-09-02) and the open live gates;
- the frozen exe is a verification artifact only, rebuilt when a packaging trigger
  is hit (last: P7's registry asset, 74/74 frozen self-test);
- the research warehouse Phases 0-8, Chart Review, durability and Local-AI Phases
  1-2 are implemented; their live gates are listed in the checkpoint;
- legacy SPY pause detection and D1 wick alerts remain the production champions;
- `market_state` and `greatness_monitor` remain shadow-only;
- the research warehouse and AI outputs remain additive/read-only and advisory;
- the trader's stated priorities are decision 0016 (Section 1 above).

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
| **0.8** | GUI fluidity Wave P1 | Repair the measured Standard-mode stalls and three verified GUI defects |
| **0.13** | Grade what the trader already said (P0-P10) | Every verdict gets a forward record; a like starts a five-session watch. **MERGED; live gates #29-#43 owed** |
| **0.16** | Capture and board rules (packet T1) | A veto with no box, a like that stays, a board click that queues nothing, the TC2000 board on M5 Focus. **BUILT; live gate #58 owed** |
| **0.14** | Names first (V1, V2, V3) | Decision 0016: the names shown come before the entry taken. **V1 and V2 merged, V3 on its branch; V2's item 3 (the AWAY Recap) is NOT BUILT** |
| **0.12** | Focus de-clutter + HTF LRSI research | Make the Focus feed, the Armed board and the Focus list readable again; ask in shadow whether a higher-timeframe LRSI entry pays |
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
named R7/R8 review-deferral completions are authorized. *(True USD conversion was
the one exception held back "pending a trader decision"; that decision arrived on
2026-08-24 and reversed the deferral — it is BUILT as packet W3, and R7's gates
1/3/6 are still owed.)*

R2's branch is cut from R1's and carries the R1.1 repair, so merging R2 brings the
testing week, R1, R1.1 and R2 together. The R1 and R2 live proofs are both owed and
are listed in `CURRENT_CHECKPOINT.md`.

1. **R1 Auto-mode matrix and quiet hours. — BUILT 2026-08-15, live proof owed.**
   Spec: `docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md`. Build record: `CHANGELOG.md`.
   **Remaining — the spec §6 live proofs, narrowed 2026-08-18.** The quiet boot **PASSED** on 2026-08-16 22:06, and AWAY staging-without-adoption **PASSED** across the 08-17 and 08-18 sessions. Still owed: the **drain on return** (the trader never flipped back to DESK, so that half of the AWAY proof is untested); an **EVENING day** whose log shows the early block and then zero further slots; and one **SPY-alarm firing** (real or forced threshold).

2. **R2 M5 Focus adoption discipline and the M5 strength board. — BUILT 2026-08-15, live proof owed.**
   Spec: `docs/M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md`. Build record: `CHANGELOG.md`.
   **Remaining — the spec §8 live proofs, narrowed 2026-08-18.** The **eviction proof PASSED** on 2026-08-18 (four logged evictions with per-symbol reasons; lines quoted in `CURRENT_CHECKPOINT.md`). Still owed, all three needing a DESK day because AWAY never adopts: one adoption-time refusal, one clean "Not today" scoped removal that leaves the trader's other entries intact, and a board session the trader confirms matches the TC2000 scan's character (~20–40/side). RVOL-for-survivors is specified but deliberately not built; decide it on that session.
   **Amended 2026-08-31 (trader): the board moved into the Desk's Strength window.** It is a collapsible section under `FocusStrengthBoard` in the alert column, starting closed, and the left-nav page is removed - so the board session owed above is now read from the Desk rather than from a page. Nothing about the gate, the fetch or the data changed; see the spec's 2026-08-31 addendum.

3. **R3 Swing-quality demotion, pre-close honesty, and the dislike-feedback loop.**
   Spec: `docs/SWING_QUALITY_AND_FEEDBACK_PLAN.md`. Build record: `CHANGELOG.md`.
   **DETERMINISTIC WORK COMPLETE 2026-08-16.** **DETERMINISTIC WORK COMPLETE 2026-08-16.** The one remaining item, §4.3.5 same-slot volume normalization, was **explicitly deferred by the trader on 2026-08-16**: the D1 scoring seam has no intraday slot series, the faithful TC2000 baseline would need a 5-minute fetch across ~1,100 symbols, and the zero-fetch session-elapsed proration was offered and rejected as trading one dishonest reading for another.
   **Owed — live gates only, none claimable from tests:** the §6 `would_demote` shadow week the Amendment requires before any row moves, the one-week 12:45-vs-close list and STABLE-vs-PREVIEW churn comparison, and the scoreboard's first real-data curation cycle producing a threshold proposal.

4. **R4 Desk chart unification.**
   Spec: `docs/DESK_CHART_UNIFICATION_PLAN.md`. Build record: `CHANGELOG.md`.
   **BUILT 2026-08-16, live proofs owed.** **BUILT 2026-08-16, live proofs owed.** Sections 1–5 and 6.1–6.3 are green: trader-armed hits survive "Not today" (§6.1); `CaptureRail` lives in the snapshot popup and the Alert Center pane so every chart-opening host inherits capture, including the RS/RW and Industry boards which previously had none; armed price alerts and D1 level watches paint as a read-only `GROUP_ALERTS` levels family on the worker; the Yahoo forming-bar early print is suppressed for 15 minutes after the open and labeled when drawn; the reviewed-today marker renders on the snapshot, the Alert Center pane, RS/RW and Industry; the feed's star became a labeled Like→Focus verb; and one feed row per symbol/side/day folds repeats with a three-item escalation list and a 30-minute open-burst digest.
   **Owed — the §8 exit gate, all live:** **Owed — the §8 exit gate, all live:** every entry point opening a chart with capture, watch controls and painted armed alerts; one desk morning confirming the forming-bar caveat replaced the inflated-gap rendering; a dislike recorded from the RS/RW board appearing as a badge everywhere that symbol renders that day; and §6.1's ignored-symbol armed-watch hit feeding and sounding while automatic Focus D1 interest for that same ignored symbol stays absent.

5. **R5 M5 signal engines. — §2, §5 and the FIRST of §3's engines BUILT; §3.2/§3.3 and §4 remain.**
   Spec: `docs/M5_SIGNAL_ENGINES_PLAN.md`. Build record: `CHANGELOG.md`.
   **BUILT OUT 2026-08-18 (trader integration redirect).** **Owed, live only:** that session, for each engine, plus one observed any-bounce firing naming its level.

6. **R6 Small operational wins. — (a) BUILT 2026-08-17; (b) DECIDED 2026-08-17 and narrowed to tests/docs; (c) diagnostic ACTIVE + evidence-led repair BUILT 2026-08-20.**
   The bounded diagnostic week remains the live gate, and it **begins at the 2026-08-21 relaunch** — the desk ran the pre-fix frozen exe until then, so every earlier `ui_stalls.jsonl` row is baseline, not evidence. What it owed, for the record: **(1) the replay characterization fixture over `_load_resolved_events` is BUILT 2026-08-17** — `tests/fixtures/technical_integrity_replay_v1.json` + `tests/test_technical_integrity_replay.py`, 18 tests, every case in the specification below pinned, and **mutation-proven**: deleting the session filter fails 7 of them (including the watermark and the segmentation equivalence) and deleting the provenance strip fails 3. (c) **ACTIVE 2026-08-20 — evidence-led hang repair is BUILT; bounded live week owed.** Two Windows `AppHangB1` events (07:19 frozen exe, 14:16 source) triggered measurement rather than speculative tuning. (d) **Auto journal is a mapping, not new work**: the trader's ask resolves to the QUEUED nightly `journal_import` slot (`docs/LOCAL_AI_AUTOMATION_PLAN.md` sec 6.4c — build only after the 6.4b live proof passes and the trader says go) plus the P3.5 commentary journal; **the nightly slot half was promoted into R7 on 2026-08-15** — see item 7; P3.5 is unchanged.

7. **R7 Journal reliability and UX. — BUILT 2026-08-15, live gates owed.**
   Spec: `docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md`. Build record: `CHANGELOG.md`.
   **OWED:** one live paste that survives a backfill. **TRADER DECISION RESOLVED 2026-08-28** ("i can easily get us yearly reports from questrade so long as we can process these files"): the 44 pre-retention days are recovered from a **statement file**, not from `/activities`. `scripts/journal_statement_import.py` is BUILT and reachable from Journal > Health; a statement never writes into a day a richer source covers, so no new coverage status was needed. *Owed:* the trader importing their own YTD file on the desk against the live journal, then reconciling one monthly statement to the cent (spec gate 2).
   **Owed, and none of it can start before Monday's validation day:** **Owed, and none of it can start before Monday's validation day:** the trader-present finale — the live schema v2→v3 migration (dry-run report reviewed first, automatic file backup), the full backfill, account tax-status labeling applied to the live store, and reconciliation-week sign-off — then the spec's six live gates: coverage COVERED-or-NO_SESSION for every session day since inception, trade counts and commissions reconciling to one monthly statement per broker **to the cent**, one clean reconciliation week on both brokers, zero orphaned annotations (permanent SQL test), CAD totals spot-checked against published BoC rates for three dates, and ≥5 consecutive nightly `journal_import` ledger entries with coverage advancing and at least one observed self-heal.
   **Questrade statement import (BUILT 2026-08-28, live gate owed).**
   `scripts/journal_statement_import.py` + Journal > Health > "Import statement
   file...". Reads .xlsx (stdlib, no new dependency) and .csv; one commission
   column taken as the complete cost; options resolved from the Description;
   midnight market-local timestamps so a date-only row is never given a session;
   day-level refusal of anything a richer source already covers. Measured drift
   against the trader's real YTD file: -$0.16 on $4,014 realised, commission
   exact. *Owed:* the trader importing that file on the desk.
   **Two-lane auto-tagging + tag adjust tools (BUILT 2026-08-28, live gate owed).**
   Trader-directed while evaluating this journal against their TradesViz
   subscription. `scripts/journal_trade_shape.py` tags a trade from its own
   timestamps and legs, so history imported from outside the scanner's lookback is
   no longer blank; plus a shared-header tag filter, `distinct_tags`, `rename_tag`
   and a Manage-tags dialog. No tag derives from the outcome. *Owed:* one desk
   session tagging real trades, renaming one and filtering on it. Nothing in this
   packet touches identity, migration, coverage or reconciliation, and the six
   live gates above are unchanged.
   **Release-candidate pre-flight fix pass (2026-08-16):** Deterministic regression coverage was added for all five findings; the live migration remains owed and untouched.

8. **R8 Weekend Prep. — BUILT 2026-08-15, live gate owed.**
   Spec: `docs/WEEKEND_PREP_PLAN.md`. Build record: `CHANGELOG.md`.
   **MACHINERY BUILT 2026-08-24, RUN GATED (packet W5); the live gate is OWED.** **MACHINERY BUILT 2026-08-24, RUN GATED (packet W5); the live gate is OWED.** `scripts/ai_jobs/synthesis.py` rolls both graded cohorts up through `evidence_stats` and narrates at medium tier over that rollup alone. The live gate below is unchanged.
   **OWED, not built (2026-08-20): the weekly trader-judgement synthesis.** **OWED, not built (2026-08-20): the weekly trader-judgement synthesis.** Nightly deterministic grading of the veto cohort now runs (`ai_jobs.cohorts`, slot `veto_cohort_grading`) and the `trader_judgement` evidence scope exists but is **opt-in** — deliberately absent from `DEFAULT_SCOPES`. The cadence is decided (**weekly, on the weekend surface**, which is why it is recorded here rather than under the AI plan), but it is **gated on two weeks of graded rows** and is **NOT authorized to build**. Live gate: one weekend where the graded cohort is read and the trader confirms the reasons ranked against forward returns are the ones they recognise.
   **Every deferred join is now built**: the RRS-strength symbol join and the picks↔outcomes join landed 2026-08-18, the veto mirror cohort as AI-P1 and the LIKE cohort as R10's packet 8b on 2026-08-24, and the last three — `human_focus_performance.csv`, `pick_feedback.jsonl` and `rrs_group_strength_extremes.csv` — as packet W2 the same day, closing the spec's §6 DEFERRED block. Building a view never validates it: §10's one-real-weekend gate covers all of them and is still owed. R7's true USD conversion is no longer deferred either (packet W3, the trader's recorded 2026-08-24 reversal); the Calendar year heatmap and the additional Analytics charts landed 2026-08-18.
   **Owed: the one-real-weekend live proof** **Owed: the one-real-weekend live proof** (spec §10) — the desk booting on a weekend with the tab present and no network activity until a button is pressed, zero IB traffic across the routine, all three boards refreshed with their per-timeframe wall clock recorded, a monthly board spot-checked for the absence of a current-month bar, one real Adopt verified in all four stores with nothing removed anywhere, one auto-tag confirm and one correction, a week-windowed walk-away, the week-ahead rendering only on its button press, progress surviving an app restart mid-routine, and the trader confirming the board character per timeframe before §5's filters count as proven.

9. **Wishlist deep link into an external charting tool. — BUILT 2026-08-18 (trader-directed).**
   **No live gate**: the trader pressing the button once on the frozen desk is the whole proof, and the frozen selftest already covers the import.

10. **Trader-directed integration set. — BUILT 2026-08-21 (trader-directed).**
   **Live gates owed:** (a) one veto committed on the desk under v3 and the pooled rollup read back; (b) one claim committed on a letter key; (c) the RS/RW half populating from a live BounceBot sweep; (d) confirmation from `bad_bars.jsonl` that the next occurrence is a malformed bar and not a well-formed aggregate row — the second case would move the fix into `bounce_bot_lib`, which is ask-first and was NOT touched here.

11. **Regime-pause "holding highs" - measured and expiring. - BUILT 2026-08-21 (trader-directed).**
   **Live gates owed:** **Live gates owed:** a session where a "holding highs" row visibly leaves the queue within 15 minutes of the name rolling over; a row that keeps making new highs visibly surviving past 15 minutes; a read of `hold_expired` rows against forward outcomes to confirm the rule is not discarding winners; and a check that the tightened detector still produces a usable number of names on a normal day rather than a handful - it now passes fewer than half the longs it used to.
   **With-trend rows auto-join M5 Focus - BUILT 2026-08-27 (trader rule, same morning).** **Live gate owed:** one DESK session on a directional day confirming the rows land in Focus without a chart, that "Not today" from the Focus surfaces still removes them, and a count of how many charts the rule saved. **Not built, the trader's call:** eviction when a placed name stops holding, and the same treatment for the other two queue fillers measured that morning (D1 flags at 54% of charts shown, LRSI crosses at 20% - `CHANGELOG.md` 2026-08-27).
   **Trader rule 2, same morning - BUILT 2026-08-27:** **Live gate owed:** a DESK session confirming the hidden count moves at show time, that a revealed name is badged `wrong side of VWAP`, and a before/after count of charts shown per hour against the 124-in-46-minutes baseline of 2026-08-27.
   **Trader rule 3, same morning - BUILT 2026-08-27:** **Live gate owed:** with the other two, one DESK session. **Not built, the trader's call:** the scanner still EMITS trend-contrary D1 shorts (it has `directional_sma_stack_aligned` and does not gate on it) - a detector change with golden fixtures first; and an IB fetch path for the forming daily candle of names outside the M5 scan set, which would spend the locked pacing budget per double-click (today those previews are Yahoo rows, labelled).
   **Trader rule 4, same morning - BUILT 2026-08-27: the M5 alert bar.** **Live gate owed:** one DESK session - the bar fills in alert order, Copy all pastes into TC2000, a click charts, clicking down the bar leaves the waiting count D1-only and unchanged. **Not built, the trader's call:** the 15-minute regime-pause expiry does not reach the bar (rows carry their time; the queue rule was "queue only"); and whether the bar should fold repeats per symbol.
   **Group RS/RW tape - REMOVED from the desk 2026-08-27 (trader decision), then REBUILT the same day - BUILT / GREEN on `claude/group-tape-rebuild`, one live gate owed.** **Group RS/RW tape - REMOVED from the desk 2026-08-27 (trader decision), then REBUILT the same day - BUILT / GREEN on `claude/group-tape-rebuild`, one live gate owed.** The rebuild was authorized as an Opus build session ("make me a prompt to get Opus to do it") and built to `docs/prompts/GROUP_TAPE_REBUILD_OPUS_PROMPT.md` packets T-1..T-4; that prompt's hard rules bound the build and all ten held (zero IB, no `legacy.py` change, completed today-only bars, UNKNOWN never invented, the RS Window tab untouched, fail-before-fix per file).
   **The rebuild - BUILT** Optional, later, and explicitly NOT built: industry = median member return over the same bars (the `industry_intraday_rs_snapshot` contract) instead of the ETF proxy - that needs member bars, which is an IB-budget question. **Live gate owed (one DESK session):** the tape moves every five minutes (not 10-30); the 06:30-07:00 read carries no overnight gap and windows that cannot answer yet are blank rather than zero; a stale or failed read says so on the callout line; a chip click still charts the ETF. **Separate finding, still parked:** the 27-minute scan cycle that day (302 symbols through IB in `rrs_scan`) - a cycle-time question, not a tape question.

12. **GUI fluidity pass. - BUILT 2026-08-21 (trader-directed).**
   **Live gates owed:** **Live gates owed:** a full session compared against the same measurement - stalls per hour, median, p90, total blocked seconds, against 1843 / 238 ms / 1.16 s / 1008 s, targeting no stall over 5 s and under ~60 s blocked; the working set after three hours (8.1 GB before the GC fix); and a console with no `QFont::setPointSizeF` lines.
   Exit gate: each packet exits through its own spec; R1 and R2 land first per the trader's ranking, then R7 before R8. **R7's code is complete**; what remains for it is live evidence, which is why it does not close the phase on its own. A packet's live gates may overlap the next packet's build only when no shared file is in flight.

13. **Today's swing picks - the trader's own vetted swing list. - BUILT 2026-08-31 (trader-directed, authorized in chat 2026-08-31).**
   *"At the end of the day I have a list of my top swing targets. I want a place to put them in so the bot knows my personal favourite picks. They will usually become focus picks too but these ones get special standing because I picked them by hand... put it at the very bottom of the M5 alerts tab, the tab is so long and I never use all of it. And the bot should scan the journal to know which ones I actually took."* Deliberately NOT the Master AVWAP like/dislike capture, which already exists.
   Built: `scripts/swing_favorites.py` (append-only store + session replay), `ui/services/swing_favorites_service.py` (the two writes + the journal join on a worker thread), `ui/widgets/swing_favorites_bar.py` (the strip), `project_paths.SWING_FAVORITES_FILE`. The swing Focus write-through goes FIRST and must not fail; the evidence row goes second and its failure is swallowed. No auto-adoption marker is ever written, so the pick stays the trader's and every automatic removal path stays off it. A removal appends a RETRACTION. The "took" badge is a display-only join against the TRADE journal over a bounded 10-day window, silent when the journal would need preparing to answer. Nothing reaches a detector, score, alert, watchlist ranking or `review_policy.json`; no phone push.
   Second pass the same day (trader): the strip and the alert list share a DRAGGABLE vertical split with its own settings key (`qt_m5_column_split_sizes_v1`), no collapse, and a chip area with a floor and no ceiling; Copy/Paste carry the trader's TC2000 list both ways. The Focus like-origin is `vetted`, so these grade as their own `human_focus_swing_vetted` sub-cohort in the existing 1/3/5/10-session human-focus tracker.
   **Live gate owed:** one desk session where the trader enters their real end-of-day swing list, the names appear in swing Focus as theirs (no marker, and "Not today"/desync repair leave them alone), the split drags and the size survives a restart, Paste takes a TC2000 list, one removal retracts without disturbing the earlier row, and a name they actually trade comes back marked "took" the next time the strip refreshes.
   **Open product questions, not built** (each additive, each the trader's call): the strip shows the CURRENT session only, so a pick typed after the close cannot carry its "took" badge into the next session; `swing_favorites.jsonl` is not in `ai_summary`'s overnight evidence pack; and nothing joins the list to per-setup journal statistics.

14. **Day-trade "passed on it" reasons in the capture window. - BUILT 2026-08-31 (trader-directed, authorized in chat 2026-08-31).**
   *"Many times I really like this stock for a daytrade but it has this ONE issue"* and the trader passes; they asked for a tickable reason list under the existing Note area, several reasons allowed per pass, plus the free-text note, and - when the M5 data is already in memory - the bars attached so an AI can read the chart back as it was, with the explicit fallback of the timestamp alone.
   Built: `EVENT_PASS` in `ui/annotations/store.py` with `record_pass_annotation`; a separate versioned vocabulary family (`ui/annotations/vocabularies/pass_reasons_v1.json`, `load_pass_vocabulary`); the M5 sidecar in `ui/annotations/pass_bars.py`; the "Passed - why?" block under Note in `ui/widgets/capture_rail.py` with Alt+P and digit toggles; `SymbolSnapshotWidget.cached_m5_bars` wired as the zero-fetch bar provider on all three capture hosts. A pass never retires the chart, writes no list, and reaches no detector, score, alert or `review_policy.json`.
   **Live gate owed:** one desk session where the trader records a real pass from the Alert Center capture tab - the ticked reasons and the note land in `trader_annotations.jsonl`, the chart stays up, and a pass taken while an M5 chart is drawn carries its bars in `trader_annotation_bars/`.
   **Both open questions are DECIDED (trader, 2026-08-31), so neither is pending work.** *"Reviewed today" stays OFF for a pass:* the trader's words - *"that flag feeds the scanner report and several badges. Making a pass count as reviewed touches scanner-side code, so it should be its own small job if you want it."* `pick_feedback._ANNOTATION_DECISIONS` therefore still lists `veto`/`like_claim`/`note` only, and a test pins that a pass does not mark a symbol reviewed. *A pass never closes the chart, and no option is needed:* *"if you pass AND want the chart gone, just hit veto after. You get both behaviors without a new rule."*

### Phase 0.6 — R9: trade-review response packet (authorized 2026-08-22)

Source: `docs/analysis/TRADE_REVIEW_2026-08-21.md` §8–§9, its nine questions
answered on 2026-08-22 (Opus answer + Fable verification; working copies in the
session scratchpad). The trader answered the three decisions that needed him on
2026-08-22 and **authorized this packet in writing the same day** ("I authorize
you to queue a packet for opus to implement"). That authorization covers the
file-scoped ask-first rule for the files named below; anything outside them is
asked about again. Build order is the list order. Nothing here touches a
detector's or scorer's output; R9.5 is shadow-only by construction.

1. **R9.1 Universe write floor + `universe_rebuild` ledger event (operational P0) — BUILT 2026-08-22, GREEN; one live gate owed.**
   *Owed: one real rebuild on the desk that writes a `universe_rebuild` row with `refused: false` and a plausible before/after, confirming the ledger row and the snapshot directory appear on the live machine.* Built as specified: `universe_write_floor()` = `max(500, 50% of the prior universe_all.txt count)` with a missing, empty or **unreadable** prior failing OPEN (returns 0); `force=True` carves out the floor but never the zero-symbol refusal; `_record_universe_rebuild()` appends a deliberately **keyless** `universe_rebuild` row to `job_ledger.jsonl` on every write attempt (keyless so `JobLedger._replay` cannot turn evidence into a phantom QUEUED job); `_snapshot_universe_lists()` keeps the outgoing lists under a run-scoped name, bounded to the last 10.

2. **R9.2 The LIKE: always ask why, and stop parking the symbol — BUILT 2026-08-22, GREEN; one live gate owed.**
   *Owed: one desk session in which a LIKE is filed and the symbol is still seen to alert afterwards (and, on an AWAY day, still reaches the hourly D1 push).* Built as specified.
   Measured first (2026-08-22): 40 of 52 `like_claim` rows retired the chart AND put the symbol on `alert_center_ignored_symbols.txt` for the day (34 symbols on 08-20, 6 on 08-21); a parked symbol also stops emitting `d1EventRecorded`, so on an AWAY day a LIKE silently drops the name from the hourly D1 phone push; and because the like is routed through `remove_today`, which `review_learning.REJECT_ACTIONS` classifies as a rejection, **every LIKE is currently counted as a dismissal by the review-learning loop.** Build, in `scripts/ui/widgets/capture_rail.py`, `scripts/ui/widgets/alert_chart_review.py`, `scripts/ui/panels/alert_center_panel.py` (and the symbol-snapshot host, which shares the rail): (a) **Why is required.** The claim digit / double-click selects the setup and moves focus to the why field; Enter commits; an **empty why does not commit** (same mechanic as the veto vocabulary's `note_required`). **Parked as PLANNED, not authorized:** Q1(b), a one-click hand-off *request* from the rail to the Focus surface in the `vetoDayTradeRequested` shape.

3. **R9.3 Rebuild the setup scoreboard from the right stores — BUILT 2026-08-22, GREEN; no live gate (read-only analysis).**

4. **R9.4 `thetalongs.txt` — BUILT 2026-08-22, GREEN; one live gate owed.**
   *Owed: one Master AVWAP scan on the desk in which DRAM reaches the theta report (or is honestly absent for a stated rule reason — earnings buffer, no weekly chain, support stack), labelled `via thetalongs.txt`.* `THETA_LONGS_FILE = LONGS_FILE.with_name("thetalongs.txt")` and `load_theta_long_symbols()` in `master_avwap_lib/legacy.py`; the file is optional and an absent **or unreadable** one returns `[]` with a warning, so it can cost those names but never the run.

5. **R9.5 `sector_cohort_divergence` — BUILT 2026-08-22, GREEN, AT SHADOW.**
   **Status 2026-08-22: all five items BUILT and GREEN; the deterministic half of the exit gate is met.** What remains is the "on the desk" half — four live proofs, one per item, listed in `CURRENT_CHECKPOINT.md`: a real rebuild writing a `universe_rebuild` row; a LIKE whose symbol is seen to keep alerting; DRAM reaching (or being honestly absent from) the theta report labelled `via thetalongs.txt`; and R9.5's shadow log growing over real sessions toward its declared 40.

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

5. **Never rewrite history.**
   Evidence maintenance is **not** gated on `auto_scanning_due` — that gate stops market activity and this is after-close recovery. It IS gated on zero IB traffic, cached/yfinance only, worker thread, idle cost. 9.

12. **"Realizable R" is not a term this repo uses.**

1. **R10.0 Read-only evidence audit — no code changes bar one.**

2. **R10.A P0 runtime and outcome integrity**
   *Sol's three reproduction blockers CLOSED 2026-08-23* (`137a4bf` lineage): the after-close scheduler gained two clocks and **two completion stamps** so a deferred sweep is retried rather than marked done, with a dedicated early-close seam (`scripts/market_early_close.py`) that leaves `market_calendar`/`market_session` untouched; finalization became **one transaction per trade** with a write-ahead intent, a disk re-read, a strict commit that raises, and `resolve_unfinished_finalizations()` settling interrupted attempts against the CSV; and the transaction is fenced across processes with `local_writer_lock`, with the authorized single-instance guard added to `launch_gui.py` as defence in depth.

3. **R10.B Outcome semantics**
   (D5, D6, and the EAT/CAKE ask) - **BUILT 2026-08-24, GREEN; mechanics canary OWED** (one live session: LRSI registering gradeable rows, H1 stamping the bar close).

4. **R10.C Robust deterministic evidence report**
   **CANARY OWED:** a second session read after a normal after-close sweep, confirming the policy breakdown and that no eod-hold cell absorbed a row with no EOD close.

5. **R10.D D1 setup tracker: point-in-time transition ledger**

6. **R10.E Focus provenance**
   (F1–F6) - **BUILT 2026-08-24, GREEN; mechanics canary OWED.** `focus_membership_events.jsonl` (`focus_membership_event_v1`) emitted by the one Focus writer, with a `membership_episode_id` and an owner of `trader` | `machine` | `unknown_legacy`; `expire_m5_if_new_day` emits `expired` per name it clears, so a survivor is a test failure **and** a visible gap.

7. **R10.F LIKE cohort grading**

8. **R10.G Market context ledger, auto-shift rows, calendar**
   (C2, season) - **BUILT 2026-08-24, GREEN; mechanics canary OWED.** Every auto-regime shift becomes a row. `daily_market_context.jsonl` (`daily_market_context_v1`), one row per session at close+grace, completed at next launch if missed with a `completed_late` flag and **never fabricated**. `config/market_calendar.json` multi-year capable, with a visible **degraded** state when the active year is not covered.

9. **R10.H Market Journal: store and two surfaces**
   - **BUILT 2026-08-24, GREEN; mechanics canary OWED.** Frozen exe rebuilt, `--selftest` 68/68 (frozen). **Extended 2026-08-27 on trader instruction (BUILT, three live gates owed).** The page had no `reload()` caller and the desk tab held a second service, so a day with five entries rendered empty; both fixed (show-once load, one `shared_journal_service()`). *Owed:* (a) a Desk-tab note appearing on the left-nav page with no Refresh, with its charts; (b) one real auto-mode flip producing a `[desk]` row with SPY's tape; (c) one nightly `ai_summary` packet naming `journal.chart_digests` and `journal.entries`.

10. **R10.I Scheduled report slot and opt-in AI scope**
   Spec: `docs/analysis/AI_DIRECTION_DECISIONS_2026-08-24.md`. Build record: `CHANGELOG.md`.
   **AWAY day recap and queue routing** - **BUILT 2026-08-24, GREEN offline; live mechanics canary OWED (not yet repeated).** The live AWAY day of 2026-08-25 correctly produced zero `shown` review impressions while the backing alert/evidence streams continued to fill, but the recap was empty because `MainWindow` never supplied either Alert Center backing list to `AwayRecapPanel.set_alerts`.

11. **R10.V Daily-bar unit repair**
   — **BUILT 2026-08-23, GREEN; one live scan day owed** (S1's mechanism; authorized by the trader's 2026-08-22 R10.0b decision as **option C-prime**). Runs **before** R10.D, because a point-in-time transition ledger built over a unit-mixed store would record the splice as history.

### Phase 0.8 — GUI fluidity Wave P1 (authorized 2026-08-26)

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.9 — GUI follow-ons from the 2026-08-26 live session (authorized 2026-08-26)

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.10 — AVWAP band challenger (authorized 2026-08-26)

Source: `docs/AVWAP_BAND_VARIANT_STUDY.md` (§2b the replicated formula, §4 the
harnesses, §4 T4 the pre-declared decision criteria). The trader replicated
OneOption's band on 2026-08-26 — `AVWAP(HLC/3) ± k · stdev(close, 20,
population)` — and authorized testing it in the setup tracker ("throw it into
the setup tracker and begin testing it out"). Build prompt:
`docs/prompts/AVWAP_BAND_CHALLENGER_OPUS_PROMPT.md`.

**Scope bound.** Shadow only. `calc_anchored_vwap_bands` stays frozen (decision
0008); no detector, score, rank, tier, alert, zone arm, Focus, queue or
`review_policy.json` behaviour changes; the champion's tracker outputs are
byte-identical with the shadow block present (parity fixture frozen first).
`legacy.py`/`runner.py` edits are limited to the additive ones the prompt
pre-authorizes; anything else asks first.

1. **B-0 Pure module + golden fixture.** *BUILT 2026-08-26 (`002f2a3`).*
   `scripts/indicators/avwap_band_variants.py` (`avwap_bands_oneoption_bb20_v1`),
   OKTA fixture frozen through `_normalize_daily_bar_frame`, discriminator tests
   against the champion (sigma 0.0 on a one-bar anchor vs the 10.28 read) and
   the killed sample-OHLC form (138.09 vs the 144.60 read), `None` below 20
   closes, AST fence against importing `master_avwap_lib`. First importer of
   `indicators/`: spec-drift 17 passed with no spec edit, `--selftest` 71/71.
2. **B-1 Fit/print script.** *BUILT 2026-08-26 (`13505d1`).*
   `scripts/avwap_band_variant_fit.py` — champion vs challenger per bar since an
   anchor, offline, writes nothing without `--csv`. Reproduces the study §2b S2
   column on OKTA.
3. **B-2 Tracker shadow.** *BUILT 2026-08-26 (`5613eec` fixture, `603333b`
   code).* Parity fixture frozen FIRST, before either fenced file was touched.
   The anchor-variant blocks, the appended `VARIANT_*` stop candidate,
   `master_avwap_band_variant_stats.csv` and the panel's "Band Variant" tab all
   landed. **Appending was not sufficient** — the champion's own averages moved
   and they reach `row["score"]` — so a trader-authorized fence
   (`_is_band_variant_scenario`, seven readers) keeps the shadow out of every
   champion aggregate; the parity fixture proves the champion's record is
   byte-identical. Tracker JSON growth measured: **9,982 bytes per new setup**,
   ≈144 MB (~15%) at the live 14,386-setup / 950.2 MB scale, accruing forward
   only. The study's "a few hundred bytes" estimate was ~30× low; capping the
   shadow to the non-experimental exit templates would cut it by a third and is
   a one-line change if the trader wants it.
4. **B-3 D1 chart overlay.** *BUILT 2026-08-26 (`3abf61d`).* Paint-lines group
   "AVWAP σ variant", built on the worker, anchored on the date the snapshot
   already resolved. Default OFF required a new
   `chart_levels.GROUPS_HIDDEN_BY_DEFAULT` + a `shown_groups` list in
   `PaintLinesPrefs`, because every group previously defaulted ON by design.
5. **B-4 Backfill** (next packet, after B-0..B-3 review): the level-quality
   study T1 and the playbook re-run T2, then the warehouse columns. NOT started.

**Finding that changes T1/T3's design.** A wider band is NOT automatically a
further stop: it is only stopped out less often when entry sits INSIDE it. On
the parity fixture's short — entered above both upper bands — the wider sigma
pushes the upper band toward entry and the challenger's stop lands 0.159 away
where the champion's is 0.971, six times tighter. Any stop-out or respect-rate
comparison must condition on the entry's position relative to the band.

Gates: T4's three criteria decide, and a pass is the input to a plan.md §7
promotion decision whose shape is an ADDITIONAL level family, never a swap of σ
inside the champion. ≥ 20 sessions of forward accrual owed before T3 counts.

## Phase 0.16 — Capture and board rules (packet T1, 2026-09-04) — BUILT, live gate #58 owed

Trader-authorized 2026-09-04 in their own words (quoted in full in
`docs/DESK_INTERNALS.md`, "T1 - the capture window is the why, and a look is not a
queue"). Tester-first: 48 tests committed red on `claude/t1-capture-and-board`
before any fix existed. **Note the name collision:** the sigma-band research
letters T1/T3/T4 above are a DIFFERENT thing; this is the capture-and-board packet.

1. **A rail veto retires with no box and one row.** `vetoRetireRequested` →
   `_retire_after_veto`; `removeTodayRequested` is the "✕ Not today" BUTTON's
   signal alone and that button is unchanged. Both verbs share one body
   (`_retire_review_alert(..., write_not_today_annotation=)`). **Lead ruling
   2026-09-04:** the day-trade veto retires through the box-free verb too, after
   its Focus placement, in that order.
2. **A like never advances.** `likeRecorded` → `_after_like`. The review event
   keeps the name `like_advance` - `review_learning.TAKE_ACTIONS` keys on the
   string.
3. **A board look holds no place and is never skip-counted.**
   `_is_manual_chart_look` on `MANUAL_CHART_TAG`; the M5-bar `skip` with
   `clicked_away_from_m5_alert` and the dequeued-D1 return-to-head rule are both
   untouched.
4. **The TC2000 board's parity rows auto-join M5 Focus.** DESK only, empty
   `failed_floors` only, the one adoption gate re-run on the row's own numbers,
   `_ignored_symbols` skipped, through the STORE (one `add_many` per side) +
   `mark_auto_adopted` and never `FocusService.add`, never removes, idempotent,
   one `strength_board_auto_focus` review event per refresh.
5. **Fix round 1 (reviewer NO-GO, blocker):** the auto-join also skips any name
   the trader took OFF a focus side today through ANY door.
   `FocusPickStore.declined_today`, recorded by the STORE on `remove`,
   `remove_everywhere`, `clear` and the fade under an additive `declined` key in
   `focus_auto_picks.json`, same-session only and pruned on load.
   `_ignored_symbols` alone let four other removal doors be undone by the next
   fifteen-minute refresh, re-injecting the name into `longs.txt` with it.

**Live gate (#58):** one DESK session where a double-click on a veto reason
retires the chart with no box and `trader_annotations.jsonl` gains ONE row; a like
leaves the chart up and the trader arms an alert on it before moving on;
"✕ Not today" still opens the box and advances; five clicks across the RS/RW and
TC2000 boards leave "queue clear" reading "queue clear"; and after the next
15-minute Strength refresh the TC2000 parity names are on M5 Focus with markers in
`focus_auto_picks.json`, and a "Not today" on one of them does not come back on
the refresh after that. **Fix round 1 adds one clause:** remove one of the
adopted names from the Focus list itself (not through "Not today") and confirm it
is still gone after the next refresh, and that `longs.txt` did not regain it.

## Phase 0.15 — Desk assessment packets (2026-09-03 evening, trader-authorized)

The evening assessment of 2026-09-03 (artifact "Where the Desk's Time Goes";
record in `CURRENT_CHECKPOINT.md` and `docs/DESK_INTERNALS.md`) measured the desk
after F1 and found the research tee thread at 101% of one core, the M5 scan cycle
preamble at 513-535 s against a 300 s candle, and 24 live gates owed on built work.
The trader authorized every packet in it. Status per packet:

1. **S1 - the tee (BUILT 2026-09-03 evening, BD-96).** Dedupe before work,
   persisted high-water mark, seal-side dedupe, `dedupe` CLI. Live gate #55: one
   post-restart session where `thread_cpu.jsonl` shows `warehouse-m5-tee` under
   5% of a core after the close and the day's spool holds one session of rows.
2. **The duplicated lake (REPAIRED 2026-09-03 22:29-22:42 PT with the trader's
   permission - 10,530,916 bar_m5 rows dropped, 25 + 4 derived/feature sessions
   recomputed, BD-97; outcomes for those months still owed).** Original text: Gate #56, the
   trader's commands in BD-97's runbook: `dedupe --dataset bar_m5 --apply`, then
   `rebuild-month --month 2026-08 --apply` and `--month 2026-09 --apply`
   (`retire_partition` + recompute of `bar_derived` and `feature_snapshot_intraday`,
   BD-97). Outcomes: `recompute-outcomes` (BD-98, `force` re-simulates terminal
   rows, one lock per bucket) RAN 2026-09-04 07:00-07:53 PT: 32/32 buckets,
   134,502 outcome rows superseded, 3,803 unchanged, no errors. Gate #56 MET in full.
3. **S3 - the thread gauge (BUILT 2026-09-03 evening).** Always on; verified by
   gate #55's read of `thread_cpu.jsonl`.
4. **S2 - the M5 cycle (INSTRUMENTED 2026-09-03 night; trim still measure-first).**
   The preamble line now names each RRS run and each engine sweep (`rrs_scan_5m`
   ... `engine_h1_color`); no detection change. After S1 reaches the desk, read
   one RTH morning of "Scan cycle N preamble" lines, then trim what the line names
   and decide on a detector process. Further edits to `bounce_bot_lib/legacy.py`
   remain ask-first.
5. **S4 - scan cadence (BUILT 2026-09-03 night).** DESK days run four scans
   (open+60, 13:00 ET, the 15:45 ET preview, the close slot that writes the
   tracker); AWAY/EVENING keep the hourly ladder for the phone digest;
   `desk_scan_cadence: "hourly"` restores the ladder. Live check: one DESK day
   whose run manifests show four `master_scan` runs.
6. **E1 - validation week (TRADER DECISION).** No new packets until gates #53,
   #54, #51, #52, #39 and #41 are closed with the trader watching.
7. **E2 - bar source (RESOLVED 2026-09-03 evening: it is a PIN, not a defect).**
   The desk's `local_settings.json` carries `daily_bars_source: "yahoo"`, the
   R10.0b §1.3 interim pin (`master_avwap_lib.daily_bars_source_pin`), so every
   D1 scan's daily bars come from Yahoo by configuration; IB serves intraday bars
   and the champion's M5 loop. `CLAUDE.md`'s market-data line now says so.
8. **F1 - the control documents (NEXT COMMIT).** Archive `CURRENT_CHECKPOINT.md`
   and `CHANGELOG.md` past their 1,500-line rule, move BUILT phases out of this
   section's work queue, cut `CLAUDE.md` to rule + pointer where
   `docs/DESK_INTERNALS.md` holds the story.
9. **F2 - dead weight (BUILT 2026-09-03 night, two items handed back).** The Tk
   GUI, its shims, the Tk journal/market-prep tabs, `TickerMover.py` and `PyQt5`
   are removed (19 files). `evidence_snapshots/` already had retention (7/4/12,
   `snapshot_to_das.ps1`) - the assessment was wrong there. The 498 MB `.corrupt`
   copy was deleted 2026-09-03 22:30 PT with the trader's permission.
   `technical_integrity_events.jsonl` rotation: DECLINED 2026-08-17 (R6(b)) until
   the warehouse's verified ingest of it passes; `bronze_technical_integrity_events`
   now runs nightly, so the trigger has fired and the segment scheme is OWED as its
   own packet. The six ATR implementations stay: two are in fenced formula files.
10. **F3 - the operational storage tier (STEP 1 BUILT 2026-09-04, decision 0017).**
    `scripts/tracker_store.py` mirrors every tracker save into a SQLite record
    store beside the JSON (shadow, default ON, never costs the save); no reader
    moves until gate #57 (five parity-clean live saves). Step 2 moves readers one
    at a time, narrowest first, each fail-before-fix; the CSV stores follow as
    their own packets after the tracker's step 2 is live.

## Phase 0.14 — Names first (decision 0016)

**Status at 2026-09-02, after round R4 Part A.** V1, V2 and V3 are all merged to
`main` (V3 fast-forwarded from `claude/v3-keep-it-honest` the evening of
2026-09-02); R4 Part A is on `claude/r4-fixes`. What is NOT built, in one place so
nobody has to reconstruct it from four entries:

| Packet | Item | State |
|---|---|---|
| V1 | 1 Strength Board = TC2000 | **BUILT** (gates #44, #45). R4 A7/A8 made the RVOL session-relative, dropped the forming daily bar and widened the daily window to `2y` |
| V1 | 2 `held_run_score` | **BUILT AND WIRED** (R4 A9/A10): the D1 dimension is fed, the tracker's second formula is deleted, and the M5 alert row carries the suffix. FOUR of the tracker's nine tabs fill (Bounce Types, Combos, Time of Day, Environment) and five read BLANK - the four `master_avwap_*` Swing tabs because the outcome log cannot be asked those dimensions, `rrs_alignment` because it is reachable and not derived yet (`UNDERIVED_DIMENSIONS` splits the two) |
| V1 | 3 phone digest ranks across buckets | **BUILT** (R4 A11, horizon corrected in fix round 1) - Wilson lower bound on the family's realized win rate at ONE declared horizon (`SWING_DIGEST_HORIZON_SESSIONS` = 5, stale-horizon rows dropped the way the scan-factor leaderboard drops them), expected R as tiebreak, near cap after the ranking |
| V1 | 4 Working-lately + priority switch | **NOT BUILT** - this is V4 |
| V2 | 1 nightly auto-tagging | **BUILT** (gate #46) |
| V2 | 2 Weekend Prep | (a)(b)(c)(e) **BUILT** (gate #49). R4 A13/A14/A15/A18 fixed the take rate, moved the 775 ms read off the Qt thread, gave Discovery a real `reload` and its six buttons the exit, stopped Confirm-all confirming a blank, added the per-row edit and put every table on a ten-row floor. The takes table and the collapsed notes are still owed |
| V2 | 3 AWAY Recap | **NOT BUILT** - this is V4 |
| V2 | 4 Market Journal one box | **BUILT** (gate #47). The Desk tab landed with V2; the LEFT-NAV PAGE landed with R4 A16, and R4 A17 moved the session roll to the open |
| V2 | 5 hide the dead tabs | **BUILT** (gate #48) |
| V3 | 1 win rate leads | **PARTIAL** (R4 B3 wired five surfaces). WIRED: the AWAY digest ranking (A11), `setup_docs.family_record_sentence` and its two renderers (B2), the Master AVWAP setups table's **Family Win %** column, the Setup Tracker's **Last 30 Days** tab, and all four Weekend Prep cohort tables (veto, like, pass, rejection), each sorting by the Wilson lower bound. ONE Wilson: `swing_headline.WILSON_Z`. STILL OWED: **the Setup Tracker's Setup Types tab** - and the reason is measured, not scheduling. `master_avwap_setup_type_stats.csv` carries no win column (only `target_hit_rate` / `stop_rate`, different questions), and `master_avwap_tier_outcomes.csv` cannot be joined at that table's grain: its 184 rows collapse to 71 (side, bucket, family, zone) groups, so one joined rate would repeat across up to six rows and read as each row's own. Giving it an honest win rate needs the tracker export to carry one |
| V3 | 2 day-trade headline | **BUILT** - surfaces real since R4 A10, and since R4 B4 every number on the Daytrade Tracker names its own basis: the champion tier is a COLUMN (PROVEN / MUTED / active from the learning state, blank for a segment it never saw - live 4 / 2 / 185 / 104 of 295 rows), the aggregator's verdict is headed **Verdict (edge score)**, and the My Decisions tabs carry Held 30m / Held x Ran through the same helper on `held_run_score.ALL_DIRECTIONS`, a pooled cell accumulated from the EPISODES and never an average of the two sided cells |
| V3 | 3 one `LATELY_SESSIONS` | **BUILT AND COMPLETE** (R4 B6). `review_learning.DEFAULT_WINDOW_SESSIONS` IS `LATELY_SESSIONS` and its cutoff walks the exchange calendar; Weekend Prep's week is `evidence_stats.WEEK_SESSIONS` (5). The state key, the report header, the CLI flag, the System Health audit and the Daytrade Tracker status line all say **sessions**, and a literal scan test fails if a `window_days` comes back |
| V3 | 4 one annotation writer | **BUILT** - all five surfaces have a writer since R4 A5, and since R4 B5 every VERB stamps the screen: `commit_pass` bypassed `_record` entirely, so a day-trade pass was the one row that could not say where it came from. The guard is now behavioural (one test per real click handler, reading the written row) rather than a scan of `_record`'s source text, which a verb that never calls it satisfied |
| V3 | 5 research on a trader surface | **BUILT**, and CORRECTED by R4 B1 - the reader picked the OLDEST pack of a superseded day (`sorted(...)[-1]` is an ASCII sort), so the verdict card read a 47-cell pack in the older shape and printed "no cell has cleared the evidence floor" while the current pack had 33 that had |
| V3 | 6 docs | **BUILT** |

**The two largest owed items are V1's Working-lately + priority switch and V2's
AWAY Recap, and both are V4.** Every rule they need is already written down - the
switch reorders and never withholds (CLAUDE.md), "lately" is `LATELY_SESSIONS`,
the headline statistics are `swing_headline` and `held_run_score` - so what is
missing is the surface, not the decision. **The priority switch is not built, and
CLAUDE.md no longer claims a test for it** (R4 B3): the
identical-visible-rows test is owed WITH the switch.


Decision `docs/decisions/0016-trader-vision-and-priorities.md` is the tie-breaker
for this phase: **when two packets compete, the one that improves WHICH NAMES ARE
SHOWN beats the one that improves WHEN TO ENTER.**

### Phase 0.14 packet V2 — The loop closes (2026-09-02) — items 1, 4 and 5 BUILT; 2 and 3 NOT BUILT

Authorized by the trader pasting the packet. Requires V1, which was merged first.

**Item 1 — nightly auto-tagging. BUILT.** `journal_auto_tag` runs P6a's plan every
night at the recorded 0.70 threshold, right after `journal_import` and before
every other slot. That position is an INSERT and the second and last sanctioned
exception to "later phases append; they never reorder" — the import puts the
night's trades in, and every cohort slot below reads them. It never touches a
confirmed row (the refusal lives in the STORE) and a failed write is reported
LOUDLY, because the journal is the one store on this desk that may not fail
quietly. The Journal nav button carries the review count, computed off-thread and
started from `showEvent`.

**Item 4 — the Market Journal capture is one box and one Enter. BUILT.** The
timeframe picker and the Save button leave the SURFACE; nothing leaves the
SCHEMA. Plain Enter saves through an event filter (a `QShortcut` on Return would
fire for the whole panel); Ctrl+Enter still works. The entry is dated to the
SESSION IT IS ABOUT — today while today trades, the last session that traded on a
weekend or holiday — and `written_after_the_session` is still COMPUTED.

**Item 5 — hide the dead surfaces, keep the code. BUILT.** One setting,
`qt_show_unused_tabs`, default OFF, hiding the Alerts / D1 Focus / Armed tabs and
the Universe page. **Hidden is not removed**: `setTabVisible`, no index shifts,
every timer still visibility-gated, and a test proving every rail shortcut is
panel-scoped, bound once, and not owned inside a hidden tab.

**Item 2 - Weekend Prep. (a), (b), (c) and (e) BUILT; the rest of (c) owed.**

* **(a) ONE Refresh** drives every step. The click starts each page's own reader
  and returns - measured under 50 ms - and names the steps as they start. The five
  per-page buttons left the layout and stay as objects, because `reload()` uses
  each one as its own single-flight guard.
* **(b) The verdict card**, five to eight lines from a PURE builder
  (`scripts/weekend_verdict.py`): take rate, blind spots and leaks BY NAME, the
  best liked claim and weakest veto reason at h3, the week's net and win rate
  (**confirmed tags only**), and the tag-review count. Every measured line carries
  its n; a cohort under n=5 is named as thin and never ranked; a missing input
  says so instead of printing a zero.
* **(c) The RS/RW prose is retired** - it duplicated a live board with a Saturday
  snapshot. The log SCANS are kept, uncalled, and say so in capitals in their own
  docstrings so nobody "fixes" a blank page by wiring the wall of text back.
* **(e) "Tag this week"**, a sixth step: the week's provisional and needs_review
  trades, confirm-all-shown and confirm-selected through the store's own API, ten
  visible rows, read on a worker, and a failed write reported LOUDLY.

**Still owed by item 2:** the takes/watch-conversion table (the summary is still
text), the ten-visible-rows pass over the OTHER tables, and the collapsed
"how to read this" notes.

**Item 3 - the AWAY Recap. NOT BUILT.** It is still the forward-looking digest
assembly - best-swing block, classified D1 alerts, staged picks, Focus lists -
with no outcomes, no charts, no "what moved", no "alerts that were right", no
"your names" and no "Review these" walk-through. All four of the packet's blocks
and the chart-on-click door remain.

**Live gate (#46):** one nightly run that tags new trades, and the Journal nav
button showing the count the next morning.

**Live gate (#47):** one Market Journal entry written from the desk tab in one
Enter, filed against the right session.

**Live gate (#48):** a desk session with the four surfaces hidden and every rail
hotkey still firing.

### Phase 0.14 packet V1 — Names first (2026-09-02) — items 1 and 2 BUILT; 2's surfaces, 3 and 4 NOT BUILT

Authorized by the trader pasting the packet.

**Item 1 — the Strength Board becomes the trader's TC2000 scan. BUILT.**

1. **Relative volume**, `AVG(V / mean(V78 ... V1170), 12)`, POSITIONAL exactly as
   TC2000 is. Blank, never zero, under sixteen sessions of history.
2. **The fetch period grew from `5d` to `1mo`** and had to: the RVOL needs 1,182
   bars and `5d` holds about 390, so every RVOL would have been blank.
3. **The floors** — price over $5, above the D1 200 and 100 SMA, above the M5
   15 EMA — each a NAMED boolean with the sentence that failed. The timeframes
   are a stated ASSUMPTION (decision 0016 records both as open); one line
   corrects either.
4. **The universe** is `universe_all.txt` plus the four watchlists. The D1 SMAs
   come from a second batched daily download; still **zero IB traffic**.
5. **A row that misses a filter is GREYED, not dropped**, with what it missed in
   its tooltip, behind a default-on "TC2000 parity" toggle that hides them.
6. **One window, two sections, RS/RW first** — the RS/RW board moved out of the
   tab stack into the strength column, in a scroll area, because hosted bare its
   minimum took the column's floor from 190 px to 452.
7. **Golden `tc2000_parity_v1`**: five symbols, sixteen sessions, expected values
   computed by a SECOND naive implementation written from the trader's formula
   lines rather than from the module under test. All five agree to four decimals.

**The fence on `strength_scan.py` is NARROWED, not lifted.** It was frozen whole
by the R8 spec ("stop and ask the trader first"); the trader asked, naming the
file. The test now asserts the seven FORMULA functions are byte-identical to the
R8 baseline — stronger than "no edits", which could be satisfied by not touching
the file while the numbers moved underneath it.

**Item 2 — `held_run_score`. THE SCORE IS BUILT; ITS THREE SURFACES ARE NOT.**

`scripts/held_run_score.py` computes P(level held in the first 30 min) x
trimmed-mean MFE_R of the held ones, per (bounce_type, time_bucket,
market_environment, d1_setup_present), over a rolling 20 sessions with
`evidence_stats` floors. Shadow only: the champion tier, the mutes and the PROVEN
stamp are untouched, and a test asserts the champion never imports it.

**NOT BUILT and owed by this item:** the Daytrade Tracker column and sort, the M5
alert-bar row suffix (`alert_suffix` exists and is tested; nothing calls it yet),
and the Alert Center ordering switch.

**Items 3 and 4 — NOT BUILT.**

3. The phone digest still ranks the best-swing block inside the favourite bucket
   only. Decision 0016 answer 8 says the best pick is often in `near_favorite_zone`,
   so the cream is still not being sent.
4. There is no "Working lately" section on the Trading Desk and no priority
   switch. `review_learning`'s callouts, the tracker's per-family outcomes and the
   four verdict cohorts all still require leaving the desk to read.

**Live gate (#44):** one DESK session where the Strength section matches the
trader's TC2000 list on the same minute for the top ten names, with the parity
toggle on.

**Live gate (#45):** the RS/RW section opens above the Strength section in the
alert column and neither widens the column.

### Phase 0.13 packet P3 — The fact pack tells the truth (2026-09-01) — BUILT, live gate owed

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P7 — One name per setup (2026-09-01) — BUILT, no live gate

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, no live gate**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 — Trader decisions of 2026-09-01 (packet P0) — BUILT, live gate owed

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P1 — Grade what you already said (2026-09-01) — BUILT, live gate owed

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P2 — Show me (2026-09-01) — BUILT, live gate owed

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P4 — The variables you are not looking at (2026-09-01) — BUILT, live gate owed

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P5 — Pass and not-today get graded (2026-09-01) — BUILT, live gate owed

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P6 — Preference to trade (2026-09-01) — BUILT, live gate owed

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P6a — Tag the backlog (2026-09-01) — BUILT, live gate owed

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P8 / Phase 6.1 addendum — First setup-parameter grid (2026-09-02) — BUILT, live gate owed

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P9 — Quick like (2026-09-02) — BUILT, live gate owed

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P10 — What happens after I like it (2026-09-02) — BUILT, live gates owed

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gates owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 review round R2 (2026-09-02) — TWO GUARDS, BUILT

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 review round R1 (2026-09-02) — BLOCKERS FIXED, ALL PACKETS MERGED

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.11 — Theta premium optimization (authorized 2026-08-31) — BUILT, live gate owed

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.12 — Focus de-clutter + higher-timeframe LRSI research (authorized 2026-09-01)

Long form moved to [`docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md`](docs/ROADMAP_ARCHIVE_PHASES_0.8-0.13.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 1 — NEXT: remove known uncertainty from the development baseline

1. **P1.1 Make the test suite hermetic.** Stop Qt app tests from starting live
   universe/yfinance work; keep explicit network/broker markers and bounded teardown.
   *Built 2026-08-18 (offline tripwire, IB/yfinance/market-prep stubs, `network`/
   `broker` opt-outs) and **completed 2026-08-24** (packet W1): the last unbounded
   half of teardown is closed. Measured with a thread-recording plugin over a full
   run: **22 tests left a thread running past their own teardown and 19
   `run_strategy` threads were still alive when the session ENDED** - the standing
   crowd `conftest.py`'s garbage-collection block already named and said it did not
   join. `conftest.retire_leaked_bounce_bots` now calls BounceBot's own
   `stop(timeout=...)` for any strategy loop a test leaves behind and FAILS the
   leaking test if one survives; re-measured after the fix, **0 scanner threads
   survive the session** and the other seven leaks (`scan-*-drain`,
   `qt-health-audit`, `industry-board-refresh`, the Desk Link reader) all end on
   their own before it. The wall-clock flake class (two panel tests that failed
   only between 06:30 and 07:00 PT, inside the open-burst digest window) was
   repaired 2026-08-23 by pinning those tests' clocks; verified still pinned.
   Determinism evidence: three consecutive full runs, identical pass counts,
   exit 0.*
2. **P1.2 Resolve the measured D1 line-display defect.** After the testing week,
   decide the red-level threshold and total clutter budget from desk evidence. Ask
   before touching any fenced detector/scoring/alert-hosting file.
3. **P1.3 Adjudicate pending branches.** Review the scoring/flagging branch only with
   golden fixtures; discard or supersede obsolete documentation-only work after this
   consolidation.
4. **P1.4 Finish observability depth.** Add representative benchmark/golden fixtures
   and trends for timings, provider calls, failures, coverage, and scan-stage latency.
   *BUILT 2026-08-24 (packet W7).* `scripts/diagnostics/observability_trends.py`
   reads the run manifests and `ai_job_ledger.jsonl` that already exist and folds
   them into a trend: per-phase latency against the window before it, the
   `provider.<family>` counter tree with cache-hit and failure rates, run and job
   failure counts with the errors quoted, and coverage from the scan's own
   `symbols_processed`. **Zero new measurement** — nothing is instrumented, timed
   or run during a scan, and an AST test keeps it that way. Frozen by the golden
   fixture `observability_trends_v1`, whose inputs are hand-written to contain
   each shape the reader has a rule for (a phase with no baseline, a phase absent
   from one run, a family with no attempts, a failed run, a mixed job record)
   rather than a copy of one machine's diagnostics. Its first live read named
   two real failures the desk had not been counting: `journal_import` 9 of 12
   (the dead Questrade refresh chain, a trader action) and `ticker_briefs` 11 of
   30.
5. **P1.5 Do bounded repository hygiene.** Ignore generated desk JUnit output and
   remove retired Desk Link/satellite/mini-PC code only in an explicit, fully green
   cleanup packet. Do not mix cleanup with behavior changes.
   *DONE 2026-08-24 (packet W8), in one commit with no behavior change.* Removed:
   the `desk_link` package (7 modules), `ui/satellite.py`, `ui/desk_role.py`, both
   `ui/services/desk_link_*` modules, `master_avwap_mini_pc.py`, the Settings ▸
   Desk Link tab, the `--satellite`/`--link-token`/`--satellite-desk`/`--desk-role`
   flags, the control banner, and 70 tests across 7 deleted files. `desk_report.xml`
   is ignored. **The edit reached eight methods in `alert_center_panel.py`**, which
   houses alert code, so the file-scoped ask-first rule was invoked and the trader
   authorized full removal on 2026-08-24 before anything was touched. What SURVIVES
   deliberately: the generic `read_only` mode on the price-alert board and panel,
   which is a widget capability with its own tests rather than satellite plumbing,
   and now has no production caller. Packaging triggers fired by design — the spec's
   `desk_link` entry is gone, the exe was rebuilt and
   `dist\TradingBotV3\TradingBotV3.exe --selftest` returned **70/70 (frozen)**,
   exit 0.

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
   *Narrowed 2026-08-27:* the tracker-to-detection adapter and the first demanded
   context dataset are **BUILT / GREEN**. The adapter reads the small transition
   ledger plus the scenario CSV, admits every canonical tracker family with usable
   geometry, and never parses the 1 GB snapshot. Five point-in-time Auto Market Bias
   views (M5/M30/H1/H4/D1) now attach to each studied occurrence. What remains here
   is the live warehouse canary/pilot and BD-43's explicit BounceBot occurrence link.
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
   *Satisfied and BUILT 2026-08-24 (packet W4):* the six answers were given and
   recorded in `docs/analysis/OFFLINE_BUILD_AUTHORIZATION_2026-08-24.md` §1, so the
   2026-08-08 decision is met rather than waived. `scripts/ai_jobs/digest.py` writes
   the two artifacts per session — a deterministic fact pack (zero LLM, written even
   when the model is down) and a medium-tier narration that reads the fact pack and
   nothing else — and `daily_digest` is APPENDED last in `default_slots()`.
   **Still owed here: the ten clean digest sessions plus the trader spot-audit of at
   least three packs against raw evidence.** Building the ledger never marks that
   gate met. The enrichment (P3) and policy-draft (P4) machinery is BUILT and
   RUN-GATED under packet W6; what P3.3 retains is the gates, not the code.
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
   *Partial build 2026-08-27, shadow-only:* the first bounded stop/target comparison
   is implemented for tracker D1 occurrences: next session's first completed M5 close
   is entry; structural-stop ranks 1–3 and 0.5/1.0/1.5 ATR controls are crossed with
   1R/2R/3R targets under STOP_FIRST and the existing cost model. A deterministic
   nightly fact pack is always written; medium local AI may narrate only after n>=30,
   five symbols and five sessions. Corpus accumulation, pilot validation, registered
   holdout work and every promotion gate remain owed.
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

**Long form.** The verbatim build narrative for Phases 0.5–0.7 moved to
[`docs/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md`](docs/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md) on
2026-08-28. Every numbered item and every owed gate stayed here, unabridged; only the
description of work already built moved. That file is evidence — if it disagrees with
this section, this section wins.

### Lint (2026-08-31, trader-directed) - CLOSED

`ruff` 0.16.5 is installed and pinned, and `ruff check .` reports **All checks
passed**, down from 1,703 findings on its first run. The backlog that stood here -
74 unused imports and one `F821` in an alert file - is closed; both were swept on
the trader's explicit yes. "ruff clean" is now a claim this repo can make.

Keep it that way: run `.venv\Scripts\python.exe -m ruff check .` before a commit,
alongside the test suite. The narrow select in `pyproject.toml` (`E9`, `F63`,
`F7`, `F82`, `F401`) is deliberate - widen it as the legacy cores shrink, not
before.

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
