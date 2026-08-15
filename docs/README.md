# TradingBotV3 documentation index

Last reconciled: **2026-08-15**

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
| [`LOCAL_AI_AUTOMATION_PLAN.md`](LOCAL_AI_AUTOMATION_PLAN.md) | Local-AI implementation specification; Phase 2 redesign and later phases remain |
| [`AUTO_MODES_AND_QUIET_HOURS_PLAN.md`](AUTO_MODES_AND_QUIET_HOURS_PLAN.md) | Phase 0.5 R1 — **BUILT 2026-08-15, live proof owed**: mode matrix, quiet hours, shared-scan removal, EVENING SPY wake alarm. §8 records the build-time decisions |
| [`M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md`](M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md) | ACTIVE spec for Phase 0.5 R2: adoption gate/eviction, auto-pick provenance, TC2000-parity strength board |
| [`SWING_QUALITY_AND_FEEDBACK_PLAN.md`](SWING_QUALITY_AND_FEEDBACK_PLAN.md) | ACTIVE spec for Phase 0.5 R3: demote-and-label quality filter, pre-close honesty bundle (investigation record), dislike-feedback loop |
| [`DESK_CHART_UNIFICATION_PLAN.md`](DESK_CHART_UNIFICATION_PLAN.md) | ACTIVE spec for Phase 0.5 R4: capture on every chart, painted armed alerts, forming-bar honesty, reviewed-today badge |
| [`M5_SIGNAL_ENGINES_PLAN.md`](M5_SIGNAL_ENGINES_PLAN.md) | ACTIVE spec for Phase 0.5 R5: SMI/efficiency-LRSI/HA indicators, new M5 alert types, AnyBounceWatch, first-candle ORB |
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
