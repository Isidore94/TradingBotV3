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
| **0.8** | GUI fluidity Wave P1 | Repair the measured Standard-mode stalls and three verified GUI defects |
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

Source: `docs/GUI_REDESIGN_PLAN_2026-08-25.md` §11.1 / Wave P1. The trader
promoted **Wave P1 only** on 2026-08-26. Waves U1–U3, S1 and the experimental
Snappy mode (P2) remain PROPOSAL and are not authorized; do not build from them.

Naming note: the proposal calls this "Wave P1" while Phase 1 below already uses
`P1.x` item ids. Items here are numbered **G-P1.x** so the two never collide.

**Scope bound.** Presentation and threading only. No detector, scorer, alert,
queue, scheduler, evidence stream or storage behavior may change, and no read
may be added or removed — only moved off the Qt thread. `alert_center_panel.py`
stays fenced under the file-scoped ask-first rule; the trader's 2026-08-26
pre-authorization in that file covers the quick-journal symbols attachment and
nothing else.

1. **G-P1.0 Three verified defects.** *Built and pinned 2026-08-26 (`db99271`).*
   The AWAY Recap called `load_focus_map(side)` against a keyword-only
   signature, so it reported the Focus lists unreadable on every run; its
   adoption-gate line called `mover_state(side, None, None, None)`, which can
   only return UNKNOWN, and rendered that as a verdict; the Desk quick-journal
   write dropped the chart symbol `write_entry` already accepts.
2. **G-P1.1 Weekend Prep off the Qt thread.** *Built 2026-08-26 (`d050ee1`).*
   The measured 8.45 s freeze. `WeekReviewPage` and `FocusReviewPage` now read
   on an owned single-flight worker; last-good survives a refresh and a failed
   read is stated, never blank. Panel shutdown joins every page.
3. **G-P1.2 Focus mover-state memo.** *Built 2026-08-26 (`0f04240`).* 36
   repeating stalls / 5.93 s. Resolved once per (symbol, side) per mover-refresh
   cycle, discarded by the poll signal that produces a newer measurement. A
   failed measurement is never cached.
4. **G-P1.3 Interaction id on the stall log.** *Built 2026-08-26 (`6bd7eef`).*
   `scripts/ui/interaction_trace.py` plus stamping in the stall watchdog, wired
   at page select, the Journal inner tab and the chart request. Diagnostics
   only; a test parses the module and fails if it can ever sleep, wait or start
   a thread. **Owed:** `first_paint` and `chart_ready` marks, which need the
   receiving paint path instrumented rather than the emit seam; and the Alert
   Center inner tab, which is fenced.
5. **G-P1.4 Convert hot `QTableWidget` rebuilds.** *Built 2026-08-26
   (`49744a7`).* System Health's three tables are written in place rather than
   rebuilt, with scroll position held across the update and selection still
   surviving by check id.
6. **G-P1.5 Audit every `reload()`** reachable from a click or page selection.
   *Audit DONE, one fix landed 2026-08-26 (`49744a7`); the remainder is listed
   below and is NOT done.*

   Fixed: `WarehouseReadoutPanel.refresh` read the DAS research lake inline —
   the only read in the audit that leaves the machine, against a share known to
   drop — and blanked its table on every failure path. Now on a single-flight
   worker, keeping last-good on failure while still clearing on a successful
   empty read.

   Audited clean: `WeekAheadPage` and `DiscoveryPage` both refresh through
   service signals, so `weekend_prep_panel.py` is fully off the Qt thread.

   **Still owed — eight panels with a click-reachable read and no worker at
   all:** `setup_tracker_panel` (12 IO call sites), `industry_panel` (6),
   `master_avwap_panel` (4), `master_market_prep_panel` (3), `theta_panel` (2),
   `watchlists_panel` (2), `rs_window_panel` (1), and `universe_panel` (has a
   worker, reload unaudited). Each needs the same treatment and its own
   fail-before-fix test. None was touched: a partial conversion of a page is
   worse than an honest list of which pages still need one.

   **Order them by MEASURED blocked time, not by that IO-call count.** The
   2026-08-26 pre-fix session (`CURRENT_CHECKPOINT.md` carries the full table)
   says the two costliest non-GC sites left are `widgets/data_table.py:35`
   (7.9%, 115 s) and `models/theta_table_model.py:72` (5.4%, 79 s — and the
   single worst stall of the day at 49.25 s), followed by
   `watchlist_utils.py:33`'s `read_text` (3.9%) and `project_paths.py:165`
   (2.1%). `theta_panel` is second-to-last on the IO-count list and near the
   top on the one that matters.

8. **G-P1.7 The cyclic GC is the largest addressed-by-nothing cost.** **NOT
   STARTED, and not authorized here** — `_GuiGcController` is a live scheduling
   component, not presentation. Recording it because the measurement is
   unambiguous: `collector(2)` and `collector(0)` together took **17.1%
   (248 s)** of the 2026-08-26 session's blocked time, and the desk was observed
   at ~1 GB after ~8.5 hours the same day. Same subsystem as the 2026-08-21
   incident (8 GB in 90 min, 298 s then 200 s sweeps). Any work here is a
   trader decision and needs its own authorization.
7. **G-P1.6 The HealthPanel audit thread outlived its panel.** *Fixed 2026-08-26
   (`49744a7`), found by the G-P1.5 audit and pre-dating this wave.*
   Constructing the panel starts a daemon thread that emits a Qt signal back
   into it; `shutdown` stopped the timer and never joined the thread, so it
   could emit into a freed C++ object — an access violation, not a Python
   `RuntimeError`, so the guard at the emit could not catch it. Intermittent:
   4 runs in 6 segfaulted an unrelated Qt test two files later. **Worth a
   sweep:** any other panel that starts a bare `threading.Thread` and emits a
   Qt signal back into itself has the same defect. This wave fixed the one it
   tripped over, not the class.

9. **G-P1.8 The 2026-08-31 desk lockup: a burst of one signal is one reaction.**
   *Built 2026-08-31, branch `claude/focus-refresh-storm`; live gate 19 owed.*
   ~500 s of GUI-thread blockage in a 16-minute session, worst stall **44.3 s**,
   Windows Not Responding, the desk killed twice. Cause: the DESK drain adopted
   **45 staged picks one at a time** and five `focusChanged` listeners each
   treated one add as a full rebuild. Fixed by coalescing at every listener
   (`ui.timer_utils.SignalCoalescer`, 200 ms leading-edge window) while the store
   keeps emitting per mutation; by making `FocusSideEditor.refresh()` the diff it
   already claimed to be (it still emptied and refilled the flow layout on every
   call); by narrowing `record_bounce_alert` to one chip; and by capping the
   drain at `AUTO_ADOPT_BATCH_LIMIT` (10) adoptions per cycle — pacing only, no
   pick dropped, a deferred pick never marked seen.

   **Authorization:** the trader approved the drain cap and the redraw slowdown
   on 2026-08-31, and approved the `alert_center_panel.py` feed-rebuild
   coalescing separately under the file-scoped ask-first rule. That fence is
   otherwise unchanged and this authorization does not extend past those two
   edits.

   **Deliberately NOT done, and why** (the packet allowed the cheapest 80% here):

   * **The table model resets.** `SetupTableModel`, `TrackerTableModel` and
     `ThetaTableModel` still `beginResetModel`/`endResetModel` on every
     `set_rows` instead of emitting `dataChanged` for the cells that changed.
     Left alone on measurement, not on effort: `setup_tracker_panel`,
     `theta_panel` and `daytrade_tracker_panel` own **no timer at all**, so
     those tables rebuild on an explicit refresh or a service signal, never per
     tick. The 2026-08-31 delegate samples came from *repaints*, and the burst
     that drove them is the one now coalesced. Converting to row-identity diffs
     also has to preserve sort and selection, which is its own packet.
   * **`fit_columns` / `apply_width_rule`.** `data_table.py:170`
     (`resizeColumnsToContents`) and `:135` (`classify_columns`' per-cell
     `model.data`) both appear in the 2026-08-31 samples and both still run a
     full measurement on every table rebuild. `data_table.py:35` was already the
     costliest non-GC site of the 2026-08-26 session (7.9%, 115 s), so this is a
     known, measured, unconverted cost — it belongs with G-P1.5's owed panels
     rather than with a lockup fix.
   * **The GUI-thread GC controller.** Untouched by design: its ~600 ms young
     sweeps that morning were a *symptom* of this churn, and G-P1.7 above still
     says any work there is a separate trader decision.

Gates: **the live-session soak in the proposal's §11.3 is OWED and cannot be
discharged by any test run.** Its acceptance targets are stall count, p90 and
worst-case blocked time measured over a real desk session with the watchdog
enabled — deterministic tests prove the reads moved, not that the desk feels
different. Re-run the §14 performance workflow on the same sequence and compare
against the 2026-08-25 capture (264 stalls, 117.3 ms median, 205.1 ms p90,
8.45 s worst, 46.0 s blocked in ~45 min) before calling Wave P1 done. No
packaging trigger applies to the work landed so far.

### Phase 0.9 — GUI follow-ons from the 2026-08-26 live session (authorized 2026-08-26)

Source: `docs/GUI_REDESIGN_PLAN_2026-08-25.md` §15 decisions 9, 10, 11, 14,
accepted by the trader on 2026-08-26 ("i authorize all changes") with the
recommended answers. Waves U1–U3, S1 and Snappy P2 are still NOT authorized.
Same scope bound as Phase 0.8: presentation and threading only; no detector,
scorer, alert, queue, scheduler, evidence or storage behavior changes;
`alert_center_panel.py` stays fenced (file-scoped ask-first). Each item gets
its own fail-before-fix test and a soak between fluidity slices.

1. **G-P2.0 Table width rule** *(BUILT 2026-08-27, `1fd9e6e`)* (proposal §12,
   §3.4 A). Tables stretch to the
   available width, the widest text column takes the slack, identifiers elide
   in the MIDDLE. Apply through the shared shell, not per panel; first on
   Weekend Prep ▸ Focus pick review (`human_foc…`) and AWAY Recap (`Line`).
2. **G-P2.1 AWAY Recap as a return surface** *(BUILT 2026-08-27, `a5fa6a9`)*
   (§8.3, decision 9). Hide-and-count
   scanner status rows (blank symbol, `WATCH`) in the recap panel only; a
   visible `Chart` action plus `Enter` on the selected row; symbol-less rows
   rendered distinctly with no chart action. The Alert Center's backing list
   is not changed.
3. **G-P2.2 Desk Journal route** *(BUILT 2026-08-27, `fd76923`; the trader
   approved the exact diff in chat before the fenced edit)* (§5.3, decision 10).
   One shortcut that selects
   the Journal tab and focuses the composer, plus a hint on the tab label. No
   second row under the charts; a verb-row verb only if the trader asks for a
   mouse route. Touches the fenced file: ask before the edit.
4. **G-P2.3 Next fluidity slice, in measured order** *(NOT STARTED - gated on
   SOAK 1)* (§11.1, decision 14):
   `DataTable.fit_columns` bounded measurement; the Theta refresh (explain the
   3.0 s → 26.6 s → 49.2 s growth first, then parse on a worker and diff rows
   into the model); `watchlist_utils.read_text` off Qt; `project_paths` `stat`
   measured before touched; then the eight G-P1.5 panels one whole page at a
   time. The panel-thread sweep (G-P1.6's class; candidates in proposal §2)
   rides along.
5. **G-P2.4 GC measurement packet** *(NOT STARTED)* (decision 11). Measurement
   FIRST: what
   produces the cyclic garbage, sweep cost per generation, growth per hour.
   **No scheduling change is authorized by this item** - a change to
   `_GuiGcController` needs its own ask with the measurement in hand.
6. **G-P2.5 The desk's 8-13 GB memory jumps** *(BUILT 2026-08-27 on
   `claude/warehouse-build-memory`; ONE live gate owed)*. Trader-authorised
   through `docs/analysis/OPUS_BUILD_PROMPT_DESK_MEMORY_2026-08-27.md`, which
   rests on the 2026-08-27 (10:00) investigation entry in
   `CURRENT_CHECKPOINT.md`. Three causes, all three fixed:
   - **Session-scoped warehouse reads.** `ResearchStore.read_rows` filters in
     Arrow before `to_pylist`, and the three `bar_m5` readers use it
     (`aggregate.build_derived_bars`, `features.build_intraday_snapshots`,
     `cli._run_outcomes`). Measured on the live lake: the month partition is
     8,704,108 rows / 408 MB / **15.4 GB** as dicts, against **0.53 GB** for a
     full session and **0.31 GB** for a 20-symbol outcome read. Equivalence is
     asserted against a longhand reference implementation of the old read, not
     assumed. BD-74.
   - **The 1.03 GB tracker snapshot.** `ingest_artifact` hashes the file in
     chunks and answers the watermark BEFORE `read_bytes`, and a SNAPSHOT over
     64 MB is stored whole but not `json.loads`-ed. For `setup_tracker` that
     loses nothing measurable - it declares neither `event_keys` nor `id_keys`,
     so the parse fed only the `quality` flag, and a test asserts the parsed
     and skipped rows are identical. BD-73.
   - **The BounceBot `self.data[reqId]` leak.** Five request paths freed the
     ready event and left the bar buffer (~206 KB each, ~400 a cycle, 1.5-2 GB
     a session). They now free both, on the success AND timeout branches, and
     `historicalData` drops bars for an unknown reqId instead of re-creating a
     buffer nobody will free. The trader authorised this one `legacy.py` edit
     and nothing else in that file; it was verified LIKE a detector change -
     the golden fixtures and all 411 BounceBot tests pass unchanged.

   **Live gate owed (one DESK session, after the trader restarts):** the first
   swing-scan slot's build keeps the desk under **3 GB** working set
   (`Get-Process -Id <pid> | select WorkingSet64`, sampled across the window
   the lake manifest shows for that build); the manifest still gains the same
   datasets for that session; and the desk's baseline stops creeping between
   builds.

   **Decisions, not owed work:** moving `run_build` into a child process was
   considered and NOT done - the in-process single-flight lock, the spool seal
   and the ledger's `_record_job` all assume one process, and the filtering
   removes the growth on its own (BD-74). It remains available if the live gate
   shows it is still wanted; that is the trader's call.

   **Observed in the same session, unchanged, NOT authorised here:** the RRS
   scan's O(n^2) intraday profile (CPU, not memory); the
   `_poll_focus_d1_interest` -> `FocusSideEditor.refresh` GUI stalls
   (`focus_picks_panel.py:441`, 392 s on 2026-08-27); and the RS-window
   `_auto_tick` reading 1,412 daily parquet files on the GUI thread
   (`rs_window_feed.py:745`, 92 s). Separate packets.

Gates: the Phase 0.8 live soak still comes first; **SOAK 1 (after G-P2.2, before
G-P2.3) is OWED and is the gate on item 4** (item 6 is independent of it - it is
warehouse and BounceBot memory, not GUI threading) - see `CURRENT_CHECKPOINT.md` for the
command and the baseline numbers; each G-P2.3 slice is then followed by a soak
against the archived 2026-08-26 baseline. Build prompt:
`docs/prompts/GUI_PHASE_0_9_OPUS_PROMPT.md` (two soak stops inside it; run
after the Phase 0.10 session, same checkout - which, note, cost a stash
collision on 2026-08-26: one build session per checkout).

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

### Phase 0.13 packet P3 — The fact pack tells the truth (2026-09-01) — BUILT, live gate owed

Authorized by the trader pasting the packet. Built on `claude/p3-fact-pack-truth`.
Shadow-only: nothing reaches a detector, score, alert, Focus list or watchlist.
Recorded as BD-81 … BD-85.

1. **Episodes beside rows.** `n_episodes` on every cell; the floor still counts rows.
   The measurement showed the per-cell count equals `n` in all 756 cells, so the pack
   also publishes `evidence_shape` and BD-81 names the follow-up as a CROSS-CELL floor.
2. **The eligible block leads**, with a bounded ineligible block ordered thickest-first
   and per-block drop counts.
3. **Non-trade families excluded and reported** by an explicit role map, until P7's
   setup registry owns it.
4. **Coverage published** — buckets covered, families with zero outcome rows, first M5
   session — so "not measured yet" reads differently from "measured and flat".
5. **`slice_readout` can read every family** without widening the pinned `SLICE_SETUPS`;
   the readout panel gains a family filter and four already-computed columns.

**Owed:** the optional `cell_history` block over the sibling packs on disk.

**Live gate (#32):** one overnight `setup_research` run whose Markdown opens with the
eligible block, shows `n_episodes` beside `n`, names the excluded families and prints
the bucket-coverage line; and the trader confirms the readout panel lists more than two
families.
### Phase 0.13 packet P7 — One name per setup (2026-09-01) — BUILT, no live gate

Authorized by the trader pasting the packet. Built on `claude/p7-setup-registry`.
Two READ-ONLY modules; **nothing in production imports either**, and no runtime
behaviour changed.

1. **`scripts/setup_registry.py`** + frozen `setup_registry_v1.json` (57 entries,
   `setup_id@version`), generated by `scripts/build_setup_registry.py` from FIVE
   naming sites - the packet named four; `legacy.py`'s `*_STUDY_FAMILY` constants
   are the fifth, and eight of those families are named nowhere else. Appendix C's
   role vocabulary; eight `known_divergences` recorded rather than resolved.
2. **`scripts/research_warehouse/trial_ledger.py`** - one append-only row per
   registered grid, written before any outcome is inspected, never rewritten. Four
   grids backfilled with their real authorization pointers.

**P4.1 IS WHERE THE REGISTRY BECOMES AUTHORITATIVE.** Until then it describes what
the code already believes. P4.1 owns: choosing which spelling is identity for each
of the eight divergences, and filling the columns P7 deliberately left blank
(supported sides, timeframe roles, the exact completed-bar trigger, the primary
recipe).

**Owed and NOT part of this packet:** replacing packet P3's two-entry role map with
`setup_registry.fact_pack_role` - that map is on `claude/p3-fact-pack-truth` and not
on `main`, so the two-line swap belongs to whichever of the two branches merges
second. Same for `HTF_LRSI_RECIPES`, whose grid the ledger already declares.

**Live gate: none.** Green tests are the whole gate; this packet changes no runtime
behaviour.
### Phase 0.13 — Trader decisions of 2026-09-01 (packet P0) — BUILT, live gate owed

Authorized by the trader in chat on 2026-09-01 (three quoted decisions). Built on
`claude/p0-apply-decisions`. Nothing here is a threshold change and nothing reaches a
detector's scoring.

1. **BANGER retired.** A top-alert class with a matcher and no producer: the literal
   token match, the tier-gate bypass, the always-sound and both repetition escalations
   are removed. The `banger` review-event column survives as a constant `False` so the
   historical rows and the schema id are unchanged. PROVEN is the top class.
2. **LRSI M5 alerts retired, evidence kept.** `LRSI_M5_ALERTS_RETIRED` gates the emit
   seam only. Detection, the candidate row, `intraday_bounce_outcomes.csv`, the tier
   and the PROVEN stamp keep running; the detection toggles stay `True` because they
   gate detection, not delivery. The "measure them on different timeframes" half of
   the decision is the Phase 0.12 packet B warehouse study, already built.
3. **Clicking away is a pass** — recorded in `docs/DESK_INTERNALS.md`; no code change.
   The `clicked_away_from_m5_alert` reason string is frozen (`review_learning` keys
   on it).

**Owed and deliberately not built:** LRSI as a context suffix on other M5 alerts (the
"boost" the trader described). `_format_bounce_alert_message` takes no bars, so it
needs plumbing through the champion alert callers — a champion-path change, not a
display tweak. Bring it back as its own packet if the trader still wants it.

**Live gate (#29):** one DESK session with no LRSI line on the M5 alert bar, `lrsi`
rows still arriving in `intraday_bounce_outcomes.csv` that day, and no BANGER branch
left in the alert path (grep).
### Phase 0.13 packet P1 — Grade what you already said (2026-09-01) — BUILT, live gate owed

Authorized by the trader pasting the packet. Built on `claude/p1-grade-what-you-said`.
Evidence-side only: nothing here reaches a detector, score, alert, watchlist, Focus
list, review queue or `review_policy.json`.

1. **The human-focus pick key carries its category.** One name on both the swing and
   the M5 list is now two rows and two graded cohorts; before, whichever list was
   snapshotted second was silently discarded and `human_focus_swing_vetted` had zero
   rows. The weekend-prep join and `journal_walkaway` follow the rows.
2. **A like merges into its cohort on the click**, through the same helper the veto
   uses. The nightly slot stays; both merges are idempotent.
3. **A pre-versioning veto pools with the version that introduced its code**, so a
   reason added in a later vocabulary no longer grades as its own cohort forever.
4. **The scoreboard grades every explicit decision** (seven action families, ~640
   decisions) and carries a third callout class, `r_gap`, that asks the R question
   without consulting the take rate. Chart Review's coded vetoes feed the
   `dislike_reason` dimension through a measured join.

**Reported, not forced:** the swing-favorites Focus write-through already existed and
worked (QFIN, 2026-08-31, proves it), and QFIN's `focus_swing_manual` origin is history
rather than a code path — `FOCUS_LIKE_ORIGIN` became `"vetted"` ten minutes after that
like. The packet's `lrsi_cross_20` R gap is real on the un-fixed action sets and closes
once they are corrected; the `r_gap` class is pinned to those literal numbers so it is
proven either way.

**Live gate (#30):** one Weekend Prep opened after the next scan showing a
`human_focus_swing_vetted` row, a like merged on the day it was captured, one pooled
`compressed` cohort, and the `r_gap` callout present in `review_preference_state.json`.
### Phase 0.13 packet P2 — Show me (2026-09-01) — BUILT, live gate owed

Authorized by the trader pasting the packet. Built on `claude/p2-show-me`. Six display
changes, each read-only over a file something else already writes; nothing reaches a
detector, score, alert, Focus list, review queue or `review_policy.json`.

1. **Weekend Prep's judgement tables show the robust half** — median, trimmed mean,
   symbols, sessions, top share, block CI, evidence label — one horizon at a time,
   floor-clearing rows first and by trimmed mean, sub-floor rows greyed and last.
2. **The week page names its callouts** instead of printing two integers.
3. **"My Decisions"** — one tab per scoreboard dimension beside the Daytrade Tracker,
   read off-thread, with a probation badge by set membership.
4. **The five AI phase gates** get a strip on the A.I. Summary page
   (`ai_jobs/gate_counters.py`), every number read from the source that owns it.
5. **A take-rate suffix on the M5 bar row**, from the CACHED guidance only.
6. **A repetition fold on the M5 bar** — presentation only; the bar's
   "deletes nothing, mutes nothing, records nothing, withholds nothing" contract is
   unchanged and its docstring says so.

**Reported, not forced:** the packet assumed guidance is computed before the M5 emit. It
is not — the emit precedes `_queue_score`, so the suffix reads the cache and is silent
for a symbol the desk has not charted yet. Computing it there would put a two-file stat
and a 34 KB JSON re-read on the Qt thread per alert.

**Live gate (#31):** one DESK session where the trader opens each of the six surfaces
and `ui_stalls.jsonl` charges no seconds to any of them.
### Phase 0.13 packet P4 — The variables you are not looking at (2026-09-01) — BUILT, live gate owed

Authorized by the trader, including an explicit yes to the first edit of
`master_avwap_lib/legacy.py` (file-scoped ask-first rule) and to all six Half B items.

**Half A, capture-only.** The attribute leaderboard gets a Qt tab (read off-thread; the
export is 19.7 MB) with the sample floor visible, and twelve variables already on the
record gain attribute keys. A contract-bearing golden frozen from the pre-change code
proves the priority score, bucket and expected R are unchanged.

**Half B, each behind its own fixture.** B1 sample floor on the leaderboard; B2 family
and regime views as sibling files; B3 stale-horizon rows dropped with the count
published; B4 the shipped tier written at assignment time; B5 calibration on structure
points; B6 the representative exit template named.

**Still owed and NOT part of this packet:** re-selecting the scan-factor future row by
exchange session (B3 step b) redefines every historical number and is a full sec-7
promotion; so is pinning `REPRESENTATIVE_EXIT_TEMPLATE_ID` to the documented house exit.
Any weight change consequent on B3's new numbers is likewise a sec-7 promotion.

**Live gate (#33):** one desk scan after which the Attributes tab opens off-thread with
the floor flag visible, the scan-factor coverage line shows the stale-horizon drop
count, and the expected-R note names its template.
### Phase 0.13 packet P5 — Pass and not-today get graded (2026-09-01) — BUILT, live gate owed

Authorized by the trader pasting the packet. Built on `claude/p5-pass-cohorts`. Two new
cohorts complete the set: every verdict the trader can record now has a forward record.

1. **`pass_cohort`** over the annotation log's `pass` rows. Multi-select, so a pass
   grades in k code cohorts AND the pooled `pass_all` — the code cohorts OVERLAP and must
   never be summed. Identity on write is (vocab_version, reason_code). Beside the daily
   horizons it carries a same-session grade when a bar sidecar exists.
2. **`rejection_cohort`** over `pick_feedback.jsonl`. `not_today` and `dislike` are
   separate cohorts whose numbers are never combined into a verdict (the family's
   pooled BASE row exists and is labelled, never read as either); `unfavorite` is not
   graded; the free-text reason is
   carried and never coded.
3. Two nightly slots appended, two Weekend Prep tables, both files added to the evidence
   report and the `trader_judgement` scope (with the like file, which was also missing).

**Owed and NOT part of this packet:** the same-session grade cannot currently be computed
— the bar sidecar holds only bars from BEFORE the pass, so the entry bar it asks for is
never in it. Every row says so through `intraday_unmeasured_reason`. Whether entry should
instead be the last completed close AT the pass is a definition change and the trader's
call; coding the free-text dislike reasons is likewise its own packet with a vocabulary
behind it.

**Live gate (#34):** the trader records two real passes and one not-today on the desk;
the next morning both cohorts have rows and Weekend Prep shows them.
### Phase 0.13 packet P6 — Preference to trade (2026-09-01) — BUILT, live gate owed

Authorized by the trader pasting the packet. Built on `claude/p6-preference-to-trade`.
Three stores each held a third of one question; nothing put the three on one row.

1. **Exact-id auto-tag candidates.** A fifth `AutoTagger` source, `trader_capture`, over
   the statements the trader already made about the symbol INSIDE the trade's own window.
   Ranked above every fuzzy lane, rejections prefixed, `context_row_id` carried as a
   reader's pointer only - plan.md P5.3/P5.4 keep the canonical id.
2. **`preference_trade_outcomes`** - a nightly deterministic slot and Weekend Prep table:
   one row per statement, joined to the journal and to the cohort paper grade, every row
   rendering its match confidence or "no match".
3. **An honest empty-dimension banner** on the journal's "My setups" group below 10%
   confirmed-tag coverage. The group is never hidden.

**Owed and NOT part of this packet:** the canonical opportunity id (P5.3/P5.4) is what
would turn the report's stated confidence into a link; coding the free-text reasons is a
separate packet; and whether `market_journal` should remain in the nightly scope - the
comment corrected here is what surfaced it - is the trader's call.

**Live gate (#35):** the trader imports a real day and one trade shows a `trader_capture`
candidate with a linked event; the nightly report lists that day's likes with a
traded/not-traded column.
### Phase 0.13 packet P6a — Tag the backlog (2026-09-01) — BUILT, live gate owed

Authorized by the trader: *"let's get Opus to do the tagging and I can review after"*.
Built on `claude/p6a-tag-backlog`. One trade in 193 carried a setup tag the trader typed.

1. **`tag_status` on `trade_annotations`** - `confirmed` / `provisional` /
   `needs_review`, arriving through the store's additive migration list, existing rows
   defaulting to `confirmed`.
2. **`scripts/journal_bulk_tag.py`** - the SINGLE authorized exception to invariant I7.
   Dry run by default, idempotent, refuses a confirmed row in the store, never writes
   `tag_corrections`, appends an inert `APPLY_PROVISIONAL_TAG` adjustment per tag.
   Threshold 0.70. **Run on 2026-09-01: 24 applied, 132 marked `needs_review`.**
3. **The review surface** in the Trades tab, and the analytics split ("my setups" is
   confirmed-only; "provisional setups" is its own group).

**Owed and NOT part of this packet:** the Journal's trade list still loads on the Qt
thread (`TradesTab.reload`, `AnalyticsTab._reload`) - measured, reported, and untouched
here; moving it to a worker is its own packet. Coding the 132 `needs_review` trades needs
either scan files that reach back further or the trader's own words.

**Live gate (#36):** the trader opens the Provisional filter on the desk and confirms or
edits at least ten; the "my setups" chart populates from confirmed rows only.

### Phase 0.13 packet P8 / Phase 6.1 addendum — First setup-parameter grid (2026-09-02) — BUILT, live gate owed

Authorized by the trader pasting the packet on **2026-09-02**; that paste date is the
grid's authorization pointer and is recorded in its trial-ledger row. Built on
`claude/p8-param-grid`, off `main` AFTER Phase 0.12, P3 and P7 landed - the packet
declared those as preconditions and they were not met until that morning's merge.

**Declared family:** `AVWAPE_TO_FIRST_DEV`, LONG - the registry's
`avwape_to_first_dev@1` (P7). 840 occurrences over 622 dependency clusters, the
largest cell in the lake.

**Declared question:** does an entry that waits for confirmation (M15 acceptance
close, M5 retest of the trigger, or M30 EMA15/21 controlled pullback) earn more net R
per episode than the first completed M5 close of the next session, under one
structural stop?

**Declared cells (12):** 4 entry moments x 3 targets, stop fixed at
`current_anchor:1`. **Declared floors:** n_episodes >= 30, >= 5 symbols, >= 5 entry
sessions, counted on `dependency_cluster_id`. **Declared window:** the first 20
trading sessions after the packet landed, fixed at registration.

Shadow only: every recipe is `is_diagnostic=True`, nothing is registered in
`outcome_semantics` (BD-80), and no row reaches a detector, score, alert, Focus list
or review queue. Recorded as **BD-88** and **BD-89**.

**Owed and NOT part of this packet:** the conditioning axis. The packet allows ONE
ATR-normalised bucket from the daily feature snapshot, attached point-in-time, three
buckets and not a lattice - it is not built, because nothing yet says the question
needs it, and a conditioning axis added before the unconditioned answer exists is
three more looks against the same k.

**Live gate (#37):** one overnight run publishes rows for every declared cell inside
the 20-minute reserve; the trial-ledger row exists with status `collecting`; and **no
cell is read for a verdict before the declared window closes.**

### Phase 0.13 packet P9 — Quick like (2026-09-02) — BUILT, live gate owed

Authorized by the trader pasting the packet, on their own decision: a like should
be able to say *"something about this was good"* without naming the setup.

1. **Alt+L** writes `like_claim` with `like_mode: "quick"`, no claim, no why.
   Supersedes R9.2(a)'s why-required for THIS PATH ONLY; Alt+K is untouched. The
   chart retires, `like_advance` is recorded, the symbol is marked reviewed - and
   nothing is placed, because a like carries zero privileges (P3.1).
2. **The bars**, on an M5 chart, through the writer Pass already uses.
3. **`like_mode`** as a picks column, so quick and claimed can be split later
   without rewriting a row. The cohort stays `like_unclaimed`.
4. **`sidecar_completion`**, a deterministic nightly slot that finishes a capture
   sidecar to the session close from the lake or the desk cache, into a NEW file.
   **This answers gate 34's open definition question** - "the first completed
   close after the click" is now a real bar, so the definition does not change.
5. A quick like contributes a LINK to the auto-tagger, never a tag; Weekend Prep
   and the AI scope both say the unclaimed cohort is not a setup's edge.

**Live gate (#39):** one DESK session where the trader quick-likes one swing
chart and one M5 chart - both rows in `trader_annotations.jsonl` with
`like_mode` quick, the M5 one carrying `m5_bars_ref`, both charts retired,
nothing in Focus; the next morning `like_cohort_picks.csv` holds both, the M5 one
has `m5_bars_completed_ref`, and its intraday columns are numbers.

### Phase 0.13 review round R2 (2026-09-02) — TWO GUARDS, BUILT

Authorized by the trader pasting the review.

1. **An empty `assigned_tier` cell is absent, not a tier called NAN.** The live
   feature-history file has no such column; the first scan after P4 widens it and
   every older row reads back as a float NaN, which is TRUTHY and stringifies to
   `"nan"`. `tier_for_tracker_row` now accepts only the vocabulary the stamper
   writes (S, A, B) and treats everything else as absent. Landed before the 07:30
   scan.
2. **A link is not a tag at any seam.** One predicate rejects link candidates in
   the bulk lane, the bulk top pick, Accept/Accept-all and `tag_confidence` - R1
   had covered only `auto_tag_summary`. A pass now carries ALL its codes.

Plus the stale sentences, the atomic overlap-note write, the trade-scoped
adjustment query, the Qt-thread backlog read, and four DESK_INTERNALS entries.

**No new live gates.** Gate 38 additionally watches the Setup Tracker's
current-picks count after the first scan (the NAN fix) and the Weekend Prep
backlog toggle line in `ui_stalls.jsonl`.

### Phase 0.13 review round R1 (2026-09-02) — BLOCKERS FIXED, ALL PACKETS MERGED

Authorized by the trader pasting the review. Eleven blockers across P4, P5, P6, P7 and
P8, each reproduced before it was fixed, then eight merges onto `main` in the order the
trader set: P0, P1, P2, P4, P5, P6, P6a, P8. **Every Phase 0.13 packet is now on
`main`.**

Two gates changed status rather than closing:

- **#33 (P4)** is now SATISFIABLE. It asked for a tier-tracker session; the assigned
  tier never reached the feature history, so the gate could not have passed however the
  session went.
- **#37 (P8)** is now SATISFIABLE. It asked for a trial-ledger row and nothing in
  production wrote one.

**One new gate, #38:** one DESK session on the merged tree after the restart, stall
watchdog quiet on every new surface.

**Owed and NOT done in this round:** the full suite with the `ai_jobs_runner` writer
lock FREE. It was held from 22:00 straight through by the nightly run, and the 32 tests
that stand down under it are explicitly not being called a baseline.

### Phase 0.11 — Theta premium optimization (authorized 2026-08-31) — BUILT, live gate owed

The theta sold-put/PCS report surfaces ~$0.25 credits with untradeable spreads
because the target is literally $0.25 (`$100 / 4 contracts`), the final sort
prefers the lowest qualifying strike (the cheapest option), spreads are only a
soft capped penalty, and the quote budget is spent in `base_score` order with no
premium-richness thinking. Trader decisions (2026-08-31 chat) lock the fix:

1. **T1 Relative floor.** Minimum credit 0.5% of the strike ($1 on a $200
   stock), ideal tier 1.0% ($2), absolute floor $0.40/contract. Below-floor rows
   leave the report. The $100/4-contract framing becomes display-only.
2. **T2 Ranking.** Support first (major SMAs above the strike: 1 required
   unchanged, 2 a big boost, then the covered stack), then yield per market day,
   then spread as a heavy monotonic spectrum (no new hard block — trader:
   "spreads are a spectrum … #1 priority is still areas of support"). The
   strike-ascending sort key is removed.
3. **T3 PCS time.** Credit spreads extend to 15 market days (3 weeks); sold
   puts stay at 10.
4. **T4 Budget allocation.** Enrichment work list orders `thetalongs.txt` names
   first, then estimated premium capacity (ATR%-based, no new network calls),
   then `base_score`. Nothing is dropped; the support-only fallback stays.
5. **T5 Surfaces.** Report + Qt panel carry credit % of strike, yield/week,
   spread %, and the SMA-above-strike count.

Build prompt: [`docs/prompts/THETA_PREMIUM_OPUS_PROMPT.md`](docs/prompts/THETA_PREMIUM_OPUS_PROMPT.md)
(scope fence for `legacy.py`, fail-before-fix tests, IB pacing untouched).
Eligibility rules (≥3 supports, ≥1 major SMA, earnings buffer) and R9.4
`theta_side` semantics are unchanged. Universe coverage already holds at
evaluation time (universe longs join full scans); T4 is allocation, not reach.

**Status 2026-08-31: T1-T6 BUILT and GREEN on `claude/theta-premium`.** Sold-put
credit is judged at >= 1.0% of the strike (recommended) / >= 0.5% (cusp) with a
$0.40 absolute floor, and a quote under both leaves the report. The final sort is
tier -> major SMAs above the strike -> support quality -> yield per market day ->
spread, with the strike-ascending key removed and the spread penalty uncapped.
PCS reaches 15 market days. The quote budget is ordered thetalongs -> estimated
premium capacity (ATR%-based, no new network call) -> base_score. The report and
the Qt panel carry credit %, yield/week, spread % and the SMA-above-strike count.

6. **T7 The spread credit scales with the underlying too. — DECIDED and BUILT
   2026-08-31.** The open question was put to the trader with its arithmetic and
   answered in as many words: *"Yes it should scale with price of the underlying."*
   The credit/width ratio does not scale, because `_pcs_long_strike_choices` caps
   the width at 10 points however expensive the stock is, so the 20% target credit
   stops growing at $2.00 - 1.36% of a $37 short strike and 0.31% of a $644 one.
   `theta_pcs_credit_floor(short_strike)` is now a hard minimum: 0.5% of the short
   strike or the $0.40 absolute floor, whichever is larger, sharing the sold-put
   constants so "the percent floor" has one definition. Under it the spread leaves
   the report; above it the credit/width ratio still decides recommended-vs-cusp.
   The RECOMMENDED percent (1.0%) is deliberately NOT applied here - 1% of a $644
   strike is a $6.44 credit on a 10-wide spread, a 64% credit/width bar no real
   market pays, so using it would delete every expensive spread rather than rank
   it. The report's PCS rows now carry the same `premium=` line as sold puts, with
   `credit_width_pct` alongside `credit_pct`.

   *Consequence, stated rather than discovered on the desk:* expensive credit
   spreads will mostly disappear unless their credit genuinely scales. If the
   trader wants those opportunities back, the lever is the WIDTH cap in
   `_pcs_long_strike_choices` (`max(10.0, preferred_width)`), not the floor -
   widening a $700-stock spread to ~17 points would let a 20% ratio pay $3.50 and
   clear 0.5%. That changes capital at risk per contract, so it was not done
   without asking.

Gate: one desk scan whose theta report shows percent-floored, support-first
rows, with `via thetalongs.txt` labelling intact.

### Phase 0.12 — Focus de-clutter + higher-timeframe LRSI research (authorized 2026-09-01)

Two independent packets, authorized by the trader in chat on 2026-09-01. Packet
A changes the desk; Packet B is a shadow research lane with zero desk cost.

#### Packet A — Focus alert de-clutter — BUILT, live gate owed

The Focus D1 feed had become unreadable, the Armed inventory accumulated
forever, and Focus itself only ever grew.

1. **A1 Pullback-only automatic Focus alerts.** `_poll_focus_d1_interest`
   evaluates the PULLBACK set only - 15EMA reject, AVWAPE and 1σ bounce. The
   EXTENSION set (new 5d/20d extreme, SMA break, AVWAPE / 1σ break) no longer
   fires automatically at all; the trader arms the ones they want per symbol and
   `_poll_d1_event_watches` remains the single path that fires one. The gate is
   at the flag-GENERATION seam - an extension kind is never evaluated, so
   nothing has to be suppressed downstream. Supersedes the 2026-08-05
   one-extension-per-day ration, which had nothing left to ration.
2. **A2 Armed alerts expire, in TRADING days.** A manually armed 5-day extreme
   watch gets 5 sessions; a 20-day one gets 10; every other armed thing - D1
   level watches, any-bounce watches, manual price alerts - gets 10. The clock
   is `market_calendar.trading_days_between`, never weekday arithmetic. Expiry
   runs at the head of the poll that already owns each store, so no new timer
   appears. **Uncertainty never deletes**: a date the calendar cannot reason
   about keeps the entry armed. Every expiry appends a row naming store, symbol,
   kind, `armed_at` and `expired_at`. A price alert is DISARMED rather than
   deleted - it leaves the Armed surface and keeps its levels, note and history,
   so plan.md sec 5's "user-entered names are never auto-removed" still holds.
3. **A3 Focus picks fade.** A pick that has fired no alert and printed no
   pullback event for 10 trading days moves to a FADED list. The clock starts at
   add time and is reset by a fired Focus D1 flag, an armed-watch hit, or the
   trader's own "keep in Focus" on the review chart. It applies to swing AND M5
   picks, the trader's own included - an explicit trader authorization to
   auto-remove a hand-typed name, scoped to Focus alone, through the store's own
   removal path so a hand-maintained watchlist line is untouched. Fading a
   hand-vetted swing pick appends a RETRACTION row, never an edit. It is
   reversible: "★ Restore to Focus" (fresh clock) and "✕ Discard". The check
   runs on the day roll and a half-hourly timer, never inside the 60 s poll.
4. **A4 Buttons and counts.** "Review ▶" is now "Focus pick review (N)", with
   "Faded review (N)" beside it. The faded walkthrough goes through
   `_enqueue_review_alert` - the one door - with `FOCUS_FADED_TAG`, which
   bypasses movers-only the way `FOCUS_REVIEW_TAG` does (a faded pick is by
   definition one that has not been moving). Counts repaint through the board's
   existing `SignalCoalescer` at the listener.

Gate: one desk session where the D1 Focus feed carries pullbacks only, an armed
extension watch still fires, an expired watch leaves the Armed board with a row
behind it, and a faded pick can be restored and discarded from the chart.

#### Packet B — Higher-timeframe LRSI entry research — BUILT, shadow only

"Is there something there" evidence for entering Focus-style setups on LRSI
crosses at M30/H1/H2/H4. Research lane only: it reaches no detector, score,
alert, Focus list or review queue, and promotion remains sec 7's job.

1. **B1 H2 exists.** 120 minutes joins `TIMEFRAME_MINUTES` and
   `DERIVED_TIMEFRAMES`. The locked plan CUT H2 for having no consumer; B3 is
   one, which is the cut's own reopen condition (BD-78). RTH is 6.5 h, so H2 and
   H4 end each session with a stub - published as evidence, EXCLUDED from the
   oscillator's input.
2. **B2 The short legs are unmirrored, and that is a decision.** The efficiency
   formula clamps at 0, so the mirrored-close idiom and `cross_down` are
   different features rather than a transform of one. The study reads ONE series
   for all four legs: cross-up 50/20 for longs, cross-down 50/80 for shorts.
   Rationale, cost and fixture in BD-79. Live `CROSS_LEVELS` unchanged.
3. **B3 A bounded 16-recipe diagnostic grid.** 4 timeframes × 4 entries, one
   stop model (the signal bar's extreme + 0.25 ATR on the SAME timeframe,
   following `DIAGNOSTIC_ATR_STOP_V1`) and one target (2.0R). Never a Cartesian
   search. Alternative recipes on one occurrence stay correlated diagnostics of
   ONE episode. It reads the occurrences and canonical M5 bars the nightly has
   already materialised, so it adds simulation and not a second data pass.
4. **B4 Nothing is registered in `outcome_semantics`.** These rows are warehouse
   `outcome_path` rows keyed by `recipe_id`; they never acquire a bounce family
   and never reach `claim_kind`. BD-80 records the reopen trigger.

Gate: one overnight `setup_research` run producing HTF rows inside the existing
reserve, then a first read of whether any cell clears the evidence floor.

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
