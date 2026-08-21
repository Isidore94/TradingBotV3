# TradingBotV3 documentation index

Last reconciled: **2026-08-20**

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

If a supporting document claims a different implementation status, the root
`CHANGELOG.md`/`plan.md` pair wins.

## Active operator runbooks

These describe actions an operator may perform now.

| File | Use |
|---|---|
| [`AWAY_SCANNER_RUNBOOK.md`](AWAY_SCANNER_RUNBOOK.md) | Single-main Auto/Away operation and report recovery |
| [`EVENING_MODE_RUNBOOK.md`](EVENING_MODE_RUNBOOK.md) | Sleep-in EVENING mode and ntfy setup |
| [`DESK_TESTING_PLAN.md`](DESK_TESTING_PLAN.md) | Plain-language testing sequence the trader follows step by step; rendered read-only in the desk at Settings ▸ Testing Plan. Restates `CURRENT_CHECKPOINT.md`'s owed live proofs for a human reader and must be updated in the same pass whenever those change |
| [`FIRST_SESSION_CHECKLIST.md`](FIRST_SESSION_CHECKLIST.md) | Required real-session validation for a new build |
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
| [`LOCAL_AI_AUTOMATION_PLAN.md`](LOCAL_AI_AUTOMATION_PLAN.md) | Local-AI implementation specification; Phase 2 redesign and later phases remain. **Section 7 (2026-08-20)** owns the deterministic `veto_cohort_grading` slot and the opt-in `trader_judgement` scope, and states what is deliberately not built |
| [`AUTO_MODES_AND_QUIET_HOURS_PLAN.md`](AUTO_MODES_AND_QUIET_HOURS_PLAN.md) | Phase 0.5 R1 — **BUILT 2026-08-15, live proof owed**: mode matrix, quiet hours, shared-scan removal, EVENING SPY wake alarm. §8 records the build-time decisions |
| [`M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md`](M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md) | Phase 0.5 R2 — **BUILT 2026-08-15, live proof owed**: the combined PDH+VWAP Focus adoption gate, queue eviction, provenance sidecar, scoped "Not today", the desync repair, and the M5 strength board. §10 records the measured fetch cost |
| [`SWING_QUALITY_AND_FEEDBACK_PLAN.md`](SWING_QUALITY_AND_FEEDBACK_PLAN.md) | Phase 0.5 R3 — **DETERMINISTIC WORK COMPLETE 2026-08-16, live gates owed**: shadow-only `would_demote` quality classifier, relvol + daytrade annotation, reviewed-today badge, pre-close honesty bundle (12:45 preview slot, post-close tracker write, STABLE+PREVIEW, bar stamps), structured dislike reasons counted by the scoreboard. §4.3.5 volume-thrust normalization is **trader-deferred**; the shadow week, churn comparison and first curation cycle are owed |
| [`DESK_CHART_UNIFICATION_PLAN.md`](DESK_CHART_UNIFICATION_PLAN.md) | Phase 0.5 R4 — **BUILT 2026-08-16, live proofs owed**: CaptureRail on every chart surface, Alert Center LIKE, armed alerts painted as a toggleable levels family, the early-morning forming-bar source-honesty fix, the reviewed-today marker, the labeled Like→Focus verb, and display-only feed repetition/open-burst control. §6.4 records the three trader confirmations; the Focus Picks marker and §2.2's `review_host` are held ask-first |
| [`M5_SIGNAL_ENGINES_PLAN.md`](M5_SIGNAL_ENGINES_PLAN.md) | Phase 0.5 R5 — **CODE COMPLETE 2026-08-18**: §2's pure indicator modules, §5's shared completed-bars rule, §3.1's LRSI cross, §3.2's confluence, §3.3's first-candle ORB flow, §4's any-bounce watch and §8.3's carried prior-anchor AVWAP line are all built and green. **The four newest alert types default OFF**, so §7's per-engine desk session now decides audibility rather than existence. §8.1/§8.2/§8.3 record the lane, ordering and plumbing decisions; §9.2 carries the final build state |
| [`JOURNAL_RELIABILITY_AND_UX_PLAN.md`](JOURNAL_RELIABILITY_AND_UX_PLAN.md) | ACTIVE spec for Phase 0.5 R7 — **BUILT 2026-08-15**, deterministic gates green, live gates owed: tax-grade broker import (Flex-primary IBKR, Questrade activities, coverage ledger + nightly self-heal, reconciliation, identity fixes) and the rebuilt five-tab Journal. §3 is the verified root-cause register; §4's identity notes and §5 fix 4's narrowing carry the build-time decisions |
| [`WEEKEND_PREP_PLAN.md`](WEEKEND_PREP_PLAN.md) | ACTIVE spec for Phase 0.5 R8 — **BUILT 2026-08-15**, deterministic gates green, one live gate owed (a real weekend run): the guided five-step weekend routine, the H1/D1/Monthly strength boards on the fenced M5 formula, and adds-only adoption into swing Focus. §5's filter table is trader-approved as proposed; §11 carries the measured yfinance probe |
| [`WISHLIST_OPEN_QUESTIONS.md`](WISHLIST_OPEN_QUESTIONS.md) | ACTIVE reference from the 2026-08-18 wishlist triage — one blocking trader question per unbuilt wishlist item, written down instead of guessed at. Read beside `WISHLIST.md`; nothing in it is authorized work, and an item leaves it by the trader answering its question |
| [`CHART_REVIEW_WORKSPACE_PLAN.md`](CHART_REVIEW_WORKSPACE_PLAN.md) | Chart Review schema, capture boundaries, and implementation record |
| [`DURABILITY_CATCHUP_PLAN.md`](DURABILITY_CATCHUP_PLAN.md) | Built durability design and remaining live restart gate |
| [`REVIEW_LEARNING_LOOP.md`](REVIEW_LEARNING_LOOP.md) | Review evidence, scoreboard, and annotation-only AI policy contract |
| [`SETUPS_MAJOR.md`](SETUPS_MAJOR.md) | AI-stated production setup doctrine for trader correction |
| [`SETUPS_TEST.md`](SETUPS_TEST.md) | AI-stated study/research setup doctrine for trader correction |
| [`BROKER_ADAPTERS.md`](BROKER_ADAPTERS.md) | Deferred provider/broker boundary design; execution remains out of scope |
| [`SHIP_READINESS.md`](SHIP_READINESS.md) | Packaging/cleanup direction, subordinate to the current single-main topology |

## Historical evidence: do not use as the current roadmap

These files remain because they preserve decisions, review findings, measurements, or
the reasoning behind implemented work. Their phase lists and “next” sections are
superseded by `plan.md`.

| File | Historical value |
|---|---|
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

## Architecture decision records

The files under [`decisions/`](decisions/) are accepted constraints, not progress
trackers:

1. decision-support only; no execution;
2. champion/challenger shadow promotion ladder;
3. IBKR primary with Yahoo fallback;
4. PySide6 product UI with Tk retained during migration;
5. plain-file operational home-folder storage (cloud-sync premise superseded by 15);
6. writer-lease fencing for shared exports;
7. completed bars for state transitions;
8. frozen anchored-VWAP sigma formula;
9. golden fixtures before detector changes;
10. AI review policy is annotation/ranking only;
11. one-way evidence-grounded AI advisory;
12. layered requirements with pinned constraints;
13. root-roadmap authority;
14. separate DAS research lake;
15. no cloud sync — the DAS file server is the durable storage tier.

## Maintenance rule

After every repository change and before handoff:

1. update `CURRENT_CHECKPOINT.md` with the active item, working state, and verification;
2. add completed behavior/contract/architecture changes to `CHANGELOG.md`;
3. remove, narrow, or advance `plan.md` while retaining any live/promotion gate;
4. update `WISHLIST.md` only for trader-directed idea changes or promotions;
5. update a supporting spec only when its contract or rationale changed;
6. classify every added, removed, renamed, or reclassified Markdown file here;
7. keep `CLAUDE.md` and `AGENTS.md` identical after instruction changes.
