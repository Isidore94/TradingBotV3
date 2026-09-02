# TradingBotV3 documentation index

Last reconciled: **2026-08-27**

Start here when a supporting detail is not in the four root documents. This index
classifies every maintained Markdown document so a historical plan cannot be mistaken
for current status.

## Root control documents

| File | Purpose |
|---|---|
| [`README.md`](../README.md) | Setup, launch, and operator orientation |
| [`CHANGELOG.md`](../CHANGELOG.md) | Authoritative implemented inventory and revision history |
| [`plan.md`](../plan.md) | Authoritative remaining work, invariants, gates, and order |
| [`CURRENT_CHECKPOINT.md`](../CURRENT_CHECKPOINT.md) | Active item, branch, working state, and verification checkpoint |
| [`WISHLIST.md`](../WISHLIST.md) | Candidate integrations; never an implementation queue |
| [`CLAUDE.md`](../CLAUDE.md) / [`AGENTS.md`](../AGENTS.md) | Agent operating context; kept as identical copies |
| [`BRANCH_HISTORY.md`](BRANCH_HISTORY.md) | Branch provenance: what every development branch held, where it landed, and the containment proof required before deleting one. Not a roadmap |

If a supporting document claims a different implementation status, the root
`CHANGELOG.md`/`plan.md` pair wins.

## Active operator runbooks

These describe actions an operator may perform now.

| File | Use |
|---|---|
| [`prompts/GUI_PHASE_0_9_OPUS_PROMPT.md`](prompts/GUI_PHASE_0_9_OPUS_PROMPT.md) | Paste-ready Opus build prompt for `plan.md` Phase 0.9 (G-P2.0 table width rule through one shell, G-P2.1 AWAY Recap return surface, G-P2.2 Desk Journal route with the fenced-file ask, G-P2.3 fluidity slice in measured order with two soak stops, G-P2.4 GC measurement only). Waves U1–U3/S1/P2 stay unauthorized; run after the AVWAP challenger session in the same checkout |
| [`prompts/GROUP_TAPE_REBUILD_OPUS_PROMPT.md`](prompts/GROUP_TAPE_REBUILD_OPUS_PROMPT.md) | Paste-ready Opus build prompt for the group RS/RW tape rebuild (plan.md Phase 0.5 item 11): a pure `group_rrs` module with a parity test against `legacy.real_relative_strength`, a 5-minute Yahoo-batched zero-IB service in the Strength Board's shape, a 30/60/90 strip face, hidden -> shown. Forbids `legacy.py`, IB and RS Window changes. **BUILT 2026-08-27** on `claude/group-tape-rebuild` (`c4fa8c3`, `3dbff23`), all ten hard rules held; kept as the record of what was authorized and what was deliberately left out |
| [`prompts/AVWAP_BAND_CHALLENGER_OPUS_PROMPT.md`](prompts/AVWAP_BAND_CHALLENGER_OPUS_PROMPT.md) | Paste-ready Opus build prompt for Phase 0.10 packets B-0..B-3 (module + fixture, fit script, tracker shadow stops + stats + panel section, D1 overlay), with the pre-authorized `legacy.py`/`runner.py` edits enumerated |
| [`prompts/THETA_PREMIUM_OPUS_PROMPT.md`](prompts/THETA_PREMIUM_OPUS_PROMPT.md) | Paste-ready Opus build prompt for `plan.md` Phase 0.11 (theta premium optimization, T1–T6): percent-of-strike credit floor, support-first ranking with spread as a spectrum, PCS to 3 weeks, richness-ordered quote budget, report/panel columns. Fenced to the named theta surface of `legacy.py` |
| [`AWAY_SCANNER_RUNBOOK.md`](AWAY_SCANNER_RUNBOOK.md) | Single-main Auto/Away operation and report recovery |
| [`EVENING_MODE_RUNBOOK.md`](EVENING_MODE_RUNBOOK.md) | Sleep-in EVENING mode and ntfy setup |
| [`DESK_TESTING_PLAN.md`](DESK_TESTING_PLAN.md) | Plain-language testing sequence the trader follows step by step; rendered read-only in the desk at Settings ▸ Testing Plan. Restates `CURRENT_CHECKPOINT.md`'s owed live proofs for a human reader and must be updated in the same pass whenever those change |
| [`FIRST_SESSION_CHECKLIST.md`](FIRST_SESSION_CHECKLIST.md) | Required real-session validation for a new build |
| [`GUI_FLUIDITY_MEASUREMENT_RUNBOOK.md`](GUI_FLUIDITY_MEASUREMENT_RUNBOOK.md) | Measure how fluid the desk actually is: one command over the stall log, the 2026-08-21 baseline to beat, the targets, and how to read the result |
| [`REGIME_INFRASTRUCTURE_PHASE1_RUNBOOK.md`](REGIME_INFRASTRUCTURE_PHASE1_RUNBOOK.md) | Live regime-evidence collection and evidence-floor rules |
| [`MACOS_SETUP.md`](MACOS_SETUP.md) | Supported macOS setup and known Windows-only operations |
| [`packaging/README.md`](../packaging/README.md) | Frozen build triggers, process, and self-test |

## Active technical and product references

These retain detailed contracts or doctrine. They do not own roadmap order or current
status.

| File | Role |
|---|---|
| [`ULTIMATE_SETUP_DATABASE_PLAN.md`](ULTIMATE_SETUP_DATABASE_PLAN.md) | Locked warehouse architecture and Phase 0–8 contract; phases are implemented, live pilot remains |
| [`RESEARCH_WAREHOUSE_BUILD_DECISIONS.md`](RESEARCH_WAREHOUSE_BUILD_DECISIONS.md) | Builder decisions and current warehouse live/human open items |
| [`RESEARCH_WAREHOUSE_ERD.md`](RESEARCH_WAREHOUSE_ERD.md) | Warehouse dataset identity and read contract |
| [`LOCAL_AI_AUTOMATION_PLAN.md`](LOCAL_AI_AUTOMATION_PLAN.md) | Local-AI implementation specification; Phase 2 redesign and later phases remain. **Section 7 (2026-08-20)** owns deterministic cohort grading and the opt-in judgement scope; **Section 8 (2026-08-27)** owns the deterministic nightly setup stop/target fact pack and gated medium-tier narration |
| [`AUTO_MODES_AND_QUIET_HOURS_PLAN.md`](AUTO_MODES_AND_QUIET_HOURS_PLAN.md) | Phase 0.5 R1 — **BUILT 2026-08-15, live proof owed**: mode matrix, quiet hours, shared-scan removal, EVENING SPY wake alarm. §8 records the build-time decisions |
| [`M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md`](M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md) | Phase 0.5 R2 — **BUILT 2026-08-15, live proof owed**: the combined PDH+VWAP Focus adoption gate, queue eviction, provenance sidecar, scoped "Not today", the desync repair, and the M5 strength board. §10 records the measured fetch cost |
| [`SWING_QUALITY_AND_FEEDBACK_PLAN.md`](SWING_QUALITY_AND_FEEDBACK_PLAN.md) | Phase 0.5 R3 — **DETERMINISTIC WORK COMPLETE 2026-08-16, live gates owed**: shadow-only `would_demote` quality classifier, relvol + daytrade annotation, reviewed-today badge, pre-close honesty bundle (12:45 preview slot, post-close tracker write, STABLE+PREVIEW, bar stamps), structured dislike reasons counted by the scoreboard. §4.3.5 volume-thrust normalization is **trader-deferred**; the shadow week, churn comparison and first curation cycle are owed |
| [`DESK_CHART_UNIFICATION_PLAN.md`](DESK_CHART_UNIFICATION_PLAN.md) | Phase 0.5 R4 — **BUILT 2026-08-16, live proofs owed**: CaptureRail on every chart surface, Alert Center LIKE, armed alerts painted as a toggleable levels family, the early-morning forming-bar source-honesty fix, the reviewed-today marker, the labeled Like→Focus verb, and display-only feed repetition/open-burst control. §6.4 records the three trader confirmations; the Focus Picks marker and §2.2's `review_host` are held ask-first |
| [`M5_SIGNAL_ENGINES_PLAN.md`](M5_SIGNAL_ENGINES_PLAN.md) | Phase 0.5 R5 — **CODE COMPLETE 2026-08-18**: §2's pure indicator modules, §5's shared completed-bars rule, §3.1's LRSI cross, §3.2's confluence, §3.3's first-candle ORB flow, §4's any-bounce watch and §8.3's carried prior-anchor AVWAP line are all built and green. **The four newest alert types default OFF**, so §7's per-engine desk session now decides audibility rather than existence. §8.1/§8.2/§8.3 record the lane, ordering and plumbing decisions; §9.2 carries the final build state |
| [`JOURNAL_RELIABILITY_AND_UX_PLAN.md`](JOURNAL_RELIABILITY_AND_UX_PLAN.md) | ACTIVE spec for Phase 0.5 R7 — **BUILT 2026-08-15**, deterministic gates green, live gates owed: tax-grade broker import (Flex-primary IBKR, Questrade activities, coverage ledger + nightly self-heal, reconciliation, identity fixes) and the rebuilt five-tab Journal. §3 is the verified root-cause register; §4's identity notes and §5 fix 4's narrowing carry the build-time decisions |
| [`WEEKEND_PREP_PLAN.md`](WEEKEND_PREP_PLAN.md) | ACTIVE spec for Phase 0.5 R8 — **BUILT 2026-08-15**, deterministic gates green, one live gate owed (a real weekend run): the guided five-step weekend routine, the H1/D1/Monthly strength boards on the fenced M5 formula, and adds-only adoption into swing Focus. §5's filter table is trader-approved as proposed; §11 carries the measured yfinance probe |
| [`WISHLIST_OPEN_QUESTIONS.md`](WISHLIST_OPEN_QUESTIONS.md) | ACTIVE reference from the 2026-08-18 wishlist triage — one blocking trader question per unbuilt wishlist item, written down instead of guessed at. Read beside `WISHLIST.md`; nothing in it is authorized work, and an item leaves it by the trader answering its question |
| [`CHART_REVIEW_WORKSPACE_PLAN.md`](CHART_REVIEW_WORKSPACE_PLAN.md) | Chart Review schema, capture boundaries, and implementation record |
| [`GUI_REDESIGN_PLAN_2026-08-25.md`](GUI_REDESIGN_PLAN_2026-08-25.md) | **PARTLY AUTHORIZED 2026-08-26 — §11.1 Wave P1 (built, `plan.md` Phase 0.8) and §15 decisions 9–14 (`plan.md` Phase 0.9)**; revised the same evening to the measured baseline and the live-session findings. Waves U1–U3, S1 and the experimental Snappy mode (P2) remain PROPOSAL and must not be built from. Originally: tested 4K/90%-width professional-desk redesign covering every page and nested tab, Veto-first chart layout, Market Journal workflow, Settings offloads, measured fluidity defects, and an experimental bounded-cache Snappy mode. It changes no runtime behavior and enters the build sequence only if the trader promotes it into `plan.md` |
| [`AVWAP_BAND_VARIANT_STUDY.md`](AVWAP_BAND_VARIANT_STUDY.md) | **Governing spec for `plan.md` Phase 0.10 (authorized 2026-08-26)**: OneOption's anchored-VWAP band replicated the same evening from three OKTA hover readings — `AVWAP(HLC/3) ± k·stdev(close, 20, population)`, a Bollinger width on an anchored centre (§2b, with the kill record) — then graded against the frozen champion in three shadow harnesses (level-quality backfill, playbook re-run, setup-tracker shadow stops) under pre-declared criteria. Never a swap of `calc_anchored_vwap_bands` (decision 0008) |
| [`DURABILITY_CATCHUP_PLAN.md`](DURABILITY_CATCHUP_PLAN.md) | Built durability design and remaining live restart gate |
| [`REVIEW_LEARNING_LOOP.md`](REVIEW_LEARNING_LOOP.md) | Review evidence, scoreboard, and annotation-only AI policy contract |
| [`SETUPS_MAJOR.md`](SETUPS_MAJOR.md) | AI-stated production setup doctrine for trader correction |
| [`SETUPS_TEST.md`](SETUPS_TEST.md) | AI-stated study/research setup doctrine for trader correction |
| [`BROKER_ADAPTERS.md`](BROKER_ADAPTERS.md) | Deferred provider/broker boundary design; execution remains out of scope |
| [`SHIP_READINESS.md`](SHIP_READINESS.md) | Packaging/cleanup direction, subordinate to the current single-main topology |
| [`prompts/TRADE_ANALYSIS_OPUS_ULTRACODE_PROMPT.md`](prompts/TRADE_ANALYSIS_OPUS_ULTRACODE_PROMPT.md) | Reusable operator prompt for an Opus trade-analysis session on the desk: the setup scoreboard read, the earliness audit, and the AEP DT case. Analysis-only by construction - it forbids detector/scoring/alert edits and requires every promote/demote to name its `plan.md` §7.1 rung. It authorizes nothing and marks no gate met |

## Historical evidence: do not use as the current roadmap

These files remain because they preserve decisions, review findings, measurements, or
the reasoning behind implemented work. Their phase lists and “next” sections are
superseded by `plan.md`.

| File | Historical value |
|---|---|
| [`DESK_INTERNALS.md`](DESK_INTERNALS.md) | **ACTIVE reference, not history.** The verbatim long form of every `Core loop / data flow` rule in `CLAUDE.md` — the incident, the measurements and the trader conversation that produced it. Split out 2026-08-28 because that section was 42 KB (~10,600 tokens), 65% of a file that loads into *every* session. `CLAUDE.md` keeps the rule and this keeps the reason; the rules bind from `CLAUDE.md` alone. **Read the matching entry before changing the behaviour a rule governs, and change both places when a rule changes.** |
| [`ROADMAP_ARCHIVE_PHASES_0.5-0.7.md`](ROADMAP_ARCHIVE_PHASES_0.5-0.7.md) | The verbatim build narrative for `plan.md` Section 12's Phases 0.5, 0.6 and 0.7, moved out 2026-08-28. Those three completed refinement programs were 72% of the roadmap while describing work that is BUILT. **All 89 numbered items and every gate clause stayed in `plan.md`, unabridged** — only the implementation description moved, and it is also in `CHANGELOG.md` and each packet's spec. Evidence, not authority: an owed gate is owed because `plan.md` says so. If the two disagree, `plan.md` wins |
| [`CHECKPOINT_ARCHIVE_2026-08.md`](CHECKPOINT_ARCHIVE_2026-08.md) | The dated working record from `CURRENT_CHECKPOINT.md` for **2026-08-25 and earlier** (95 entries, 379 KB), split out 2026-08-27. The active checkpoint had reached 7,901 lines and made the mandatory documentation read ~260k tokens, so it was no longer followable. Evidence for one specific past decision — **never load it as context**, and never treat an entry here as an open gate. Open gates live in the active checkpoint's "Active state at a glance" block |
| [`CHANGELOG_ARCHIVE_2025-11_2026-08-19.md`](CHANGELOG_ARCHIVE_2025-11_2026-08-19.md) | `CHANGELOG.md`'s revision history from **2026-08-19 back to the initial system in 2025-11** (36 entries), split out 2026-08-27. What currently exists is the `Current implemented inventory` section of `CHANGELOG.md`; this is how it got there. Same rule — read it for one question, do not load it |
| [`analysis/OPUS_BUILD_PROMPT_DESK_MEMORY_2026-08-27.md`](analysis/OPUS_BUILD_PROMPT_DESK_MEMORY_2026-08-27.md) | Paste-ready builder prompt for the desk's 8-13 GB memory jumps (investigation in the 2026-08-27 (10:00) checkpoint entry): session-scoped Arrow reads for the three warehouse readers that `to_pylist()` the whole month of `bar_m5` inside the desk, a chunked-sha / no-parse path for the 1 GB tracker snapshot ingest, and the trader-authorised `legacy.py` fix that frees `self.data[reqId]` on every IB request path. Fail-before-fix tests, equivalence proof, measured before/after, one live gate owed. **Authorises exactly that packet; the subprocess move and the GUI-stall items stay out of scope**. **BUILT 2026-08-27** on `claude/warehouse-build-memory` (`c6c0dc8`): 15.4 GB -> 0.53 GB per session read, the 1.03 GB tracker no longer read to answer UNCHANGED, and the five leaking BounceBot request paths freed (BD-73, BD-74). Kept as the record of what was authorised, what was deliberately left out, and the one live gate still owed |
| [`analysis/SOL_ATTACK_2026-08-24.md`](analysis/SOL_ATTACK_2026-08-24.md) | Frozen adversarial reproduction pass over the 2026-08-24 build slate: exact commands, seven proven blocker classes, three surgical repairs, four report-only blockers, refuted attack claims, and the post-close AWAY/outcome-sweep evidence. It promotes nothing and marks no live gate met. **Frozen and never edited — one reading in it is superseded:** its C5 "second production sweep missed its own clock" was taken at 14:21 and the sweep ran at 14:27:36 (656/656, 0 failed). The correction is in `CHANGELOG.md`, `plan.md` R10.A and the 2026-08-25 evening (4) checkpoint entry |
| [`CHECKPOINT_REVIEW_2026-08-08.md`](CHECKPOINT_REVIEW_2026-08-08.md) | Review and merge rulings for durability/local-AI branches |
| [`HANDOFF_A4_PACKAGING_2026-08-09.md`](HANDOFF_A4_PACKAGING_2026-08-09.md) | A4/A5 and packaging handoff/desk verification record |
| [`RESEARCH_WAREHOUSE_REVIEW_2026-08-04.md`](RESEARCH_WAREHOUSE_REVIEW_2026-08-04.md) | Warehouse defect review; later BD entries record the repairs |
| [`D1_TRENDLINE_SURVEY.md`](D1_TRENDLINE_SURVEY.md) | Measurement record supporting the paint-line decision |
| [`FOCUS_PRICE_ALERTS_PROPOSAL.md`](FOCUS_PRICE_ALERTS_PROPOSAL.md) | Implemented price-alert design; satellite portions retired |
| [`MULTI_MACHINE_DESK_PROPOSAL.md`](MULTI_MACHINE_DESK_PROPOSAL.md) | Retired Desk Link design and implementation history |
| [`GUI_PRODUCT_PLAN.md`](../GUI_PRODUCT_PLAN.md) | July 2026 product design; implemented portions are in `CHANGELOG.md` |
| [`GUI_TRADE_DISCOVERY_LEARNING_PLAN.md`](../GUI_TRADE_DISCOVERY_LEARNING_PLAN.md) | Detailed learning/Command Center design reference; old phase order is superseded |
| [`GUI_LEARNING_PROGRESS.md`](../GUI_LEARNING_PROGRESS.md) | Superseded July capture-readiness checkpoint |
| [`ALERT_CENTER_QUALITY_PACKET.md`](ALERT_CENTER_QUALITY_PACKET.md) | Historical P1.6 Alert Center packet recovered from `671ee57`; R2 absorbed its auto-pick provenance/scoped-removal outcome, while its remaining armed-alert, labeled-Focus, and display-only repetition contracts are consolidated into the active R4 spec |
| [`analysis/AVWAP_FIXTURE_BASELINE_2026-08-22.md`](analysis/AVWAP_FIXTURE_BASELINE_2026-08-22.md) | R10.V step 1: the golden-fixture baseline taken before the daily-bar unit repair. Proves by instrumented run - not by inspection - that **no test in the suite reads the live parquet store** (4,205 tests under a guard that wraps `open`/`Path.open`/`read_bytes`/`read_parquet`, zero accesses), inventories every fixture with its SHA-256 and its AVWAP role, predicts step 5's blast radius in advance (the backfill cannot move a fixture; only the step 2-3 code can), and records the new `mixed_unit_avwap_v1` numbers - a splice costs -2.28% on VWAP and halves sigma, a uniform rescale costs **nothing**, which is the whole argument for C-prime |
| [`analysis/DAILY_BAR_VOLUME_CLIFF_2026-08-22.md`](analysis/DAILY_BAR_VOLUME_CLIFF_2026-08-22.md) | R10.0b, **read-only**: 1,227 of 1,737 measurable daily-bar parquet files carry a >20x volume cliff (median 158x), the ratio is symbol-dependent at 56x-188x so no single rescale repairs it, `bounce_bot_lib` has the IB round-lot rescale and `master_avwap_lib` (which owns the store) does not, and the parquet keeps no source provenance. AVWAP bands are volume-weighted, so this is **live detector input outside R10's authorization** - four fix options with their golden-fixture impact, and the trader decides. Nothing was changed |
| [`analysis/POST_ATTACK_AUTHORIZATION_2026-08-25.md`](analysis/POST_ATTACK_AUTHORIZATION_2026-08-25.md) | Trader decisions after Fable's review of Sol's pass (ACCEPT WITH BLOCKERS): Decision A — a sweep-finalized trade is usable at its `stop_exit_r` (stop-outs) or last-measured-close R, reported as separate policies never blended; Decision B — ask-first answered for two evidence-side `legacy.py` repairs (milestone stop erasure; signal-bar `bar_time` match); Decision C — AWAY recap wiring, correction of the wrongly-recorded "FAILED" sweep canary (it ran: 656/656 at 14:27:36), and §2.3 restored to half-done. **Hand-committed and frozen; marks no gate met** |
| [`analysis/OFFLINE_BUILD_AUTHORIZATION_2026-08-24.md`](analysis/OFFLINE_BUILD_AUTHORIZATION_2026-08-24.md) | Trader authorization (2026-08-24 night, after the R10 slate completed): the §6.4a digest questions ANSWERED (both-metrics side by side; env×day-part×side slices; shadow excluded; narration disposable; 16 KB cap; empty fact pack on non-sessions) unlocking Phase 2; two recorded reversals approved (R7 true-USD conversion, LOCAL-AI P3/P4 machinery with runs gated); Wave 1 offline slate defined; authority cutovers (plan.md P2.x) deliberately held for after the merge day. **Hand-committed and frozen; waives no live gate** |
| [`analysis/AI_DIRECTION_DECISIONS_2026-08-24.md`](analysis/AI_DIRECTION_DECISIONS_2026-08-24.md) | Trader decision record from the 2026-08-24 evening conversation: the overnight summaries' intended reader is a later LLM (briefs NOT parked; the day-view artifact remains the unbuilt Phase 2 digest, blocked on §6.4a sign-off, not data); the intraday market journal ask maps onto R10.G/H/I; walk-away and setup-tracker AI reads approved as opt-in scopes over deterministic outputs only (never the raw tracker — TB-0/TB-5); build authorization for R10.B–H, the two scopes, and the new AWAY day-recap packet, plus one precisely-scoped override (R10.I's build sequencing waived, its evidence-quality claims gate NOT waived). **Hand-committed and frozen; it promotes nothing and waives no live gate** |
| [`analysis/AI_LAYER_REVIEW_2026-08-24.md`](analysis/AI_LAYER_REVIEW_2026-08-24.md) | AI-layer situation analysis and optimization design, every §1 brief number re-measured at the source: the deterministic `veto_cohort_grading` slot is the only PROVEN slot (the graded veto cohort is the layer's only finding — labelled **discovery**, 1-session horizon only), the two model slots are UNKNOWN with no measured reader, `journal_import` has 0 lifetime `ok` rows, and the binding constraint is input poverty (dead Questrade refresh chain: 0/142 days covered; 1 confirmed annotation vs 220 auto-tag candidates). Five commit-sized packet proposals (mirror-cohort join, auto-tag backlog toggle, nightly-status honesty, Questrade health surface, stale-caveat fix), a stop-doing list headed by parking `ticker_briefs` pending the trader's readership answer, and §7's open trader decisions. **Read-only; hand-committed and frozen; it authorizes nothing** |
| [`analysis/EVIDENCE_AUDIT_2026-08-22.md`](analysis/EVIDENCE_AUDIT_2026-08-22.md) | R10.0 decision register for `plan.md` Phase 0.7 — every alleged evidence defect reproduced and classified PROVEN / PROVEN* / REFUTED / UNKNOWN with its command and numbers, the store inventory, the six-namespace family map, and the decisions R10.A-R10.I are built on (ledger-over-`bouncers.txt` authority, session calendar, stop-first intrabar collision, frozen slippage, the evidence-based H1 bar-start rule, and one reconciled risk-floor definition). **Read-only; hand-committed and frozen.** Its §8 holds the open trader questions, and the program stops until the register is accepted |
| [`analysis/SETUP_SCOREBOARD_2026-08-21.md`](analysis/SETUP_SCOREBOARD_2026-08-21.md) | R9.3 output, generated by [`scripts/setup_scoreboard.py`](../scripts/setup_scoreboard.py) — the setup scoreboard rebuilt from `intraday_bounce_outcomes.csv` finals and `setup_playbook_episodes.csv` rather than the review store, so it carries a real stop, a real R, and the regime/RVOL/sector axes the review called starved. **Read-only; it promotes and demotes nothing** (§7 gate 2 is post-hoc here). Its §5 declares the frozen forward window that is the only route to gate-2-eligible evidence. Regenerate with `python scripts/setup_scoreboard.py --out <path>` |
| [`analysis/TRADE_REVIEW_2026-08-21.md`](analysis/TRADE_REVIEW_2026-08-21.md) | Measurement record for the 2026-07-24…08-21 window: data inventory, the setup scoreboard (zero promotions, zero demotions — plan.md §7 gate 2 is unsatisfiable post-hoc), the AEP 2026-08-21 case study, and the `sector_cohort_divergence` study spec. **Its §8 queue items were proposals; on 2026-08-22 the trader accepted five of them as `plan.md` Phase 0.6 / R9 and the rest stay unauthorized**, and its §2 records why `%MFE>2%` rankings are confounded with volatility |

## Architecture decision records

The files under [`decisions/`](decisions/) are accepted constraints, not progress
trackers:

1. [decision-support only; no execution](decisions/0001-decision-support-only-no-order-execution.md);
2. [champion/challenger shadow promotion ladder](decisions/0002-champion-challenger-shadow-promotion-ladder.md);
3. [IBKR primary with Yahoo fallback](decisions/0003-ibkr-primary-yahoo-fallback-market-data.md);
4. [PySide6 product UI with Tk retained during migration](decisions/0004-pyside6-consumer-ui-tk-legacy-during-migration.md);
5. [plain-file operational home-folder storage (cloud-sync premise superseded by 15)](decisions/0005-cloud-synced-home-folder-file-storage.md);
6. [writer-lease fencing for shared exports](decisions/0006-writer-lease-fencing-for-shared-exports.md);
7. [completed bars for state transitions](decisions/0007-completed-bars-only-for-state-transitions.md);
8. [frozen anchored-VWAP sigma formula](decisions/0008-frozen-anchored-vwap-sigma-formula.md);
9. [golden fixtures before detector changes](decisions/0009-golden-fixtures-before-detector-changes.md);
10. [AI review policy is annotation/ranking only](decisions/0010-ai-in-the-loop-review-policy-annotation-only.md);
11. [one-way evidence-grounded AI advisory](decisions/0011-one-way-evidence-grounded-ai-advisory.md);
12. [layered requirements with pinned constraints](decisions/0012-layered-requirements-with-constraints-pin.md);
13. [root-roadmap authority](decisions/0013-plan-md-authority-hierarchy.md);
14. [separate DAS research lake](decisions/0014-das-research-lake.md);
15. [no cloud sync — the DAS file server is the durable storage tier](decisions/0015-no-cloud-sync-das-file-server-storage.md);
16. [the trader's vision and priorities, in their own words](decisions/0016-trader-vision-and-priorities.md).

## Maintenance rule

After every repository change and before handoff:

1. update `CURRENT_CHECKPOINT.md` with the active item, working state, and verification;
2. add completed behavior/contract/architecture changes to `CHANGELOG.md`;
3. remove, narrow, or advance `plan.md` while retaining any live/promotion gate;
4. update `WISHLIST.md` only for trader-directed idea changes or promotions;
5. update a supporting spec only when its contract or rationale changed;
6. classify every added, removed, renamed, or reclassified Markdown file here;
7. keep `CLAUDE.md` and `AGENTS.md` identical after instruction changes.
