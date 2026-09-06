# TradingBotV3 documentation index

Last reconciled: **2026-09-05** (repo cleanup: historical documents moved under `archive/`).

One line per file. Status and decisions live in the root control set, never here; if a
row and `CHANGELOG.md`/`plan.md` disagree, the root pair wins.

## Root control documents

| File | Purpose |
|---|---|
| [`README.md`](../README.md) | Setup, launch, operator orientation |
| [`CLAUDE.md`](../CLAUDE.md) / [`AGENTS.md`](../AGENTS.md) | Agent operating rules; identical copies (edit CLAUDE.md, re-copy) |
| [`CURRENT_CHECKPOINT.md`](../CURRENT_CHECKPOINT.md) | Active item, branch, baseline, open gates, next action |
| [`plan.md`](../plan.md) | Remaining work in order, invariants (§5), live validation (§6), promotion (§7) |
| [`CHANGELOG.md`](../CHANGELOG.md) | `Current implemented inventory` (search it) plus the last two build days |
| [`WISHLIST.md`](../WISHLIST.md) | Ideas and their open trader questions; never authorized work |
| [`BRANCH_HISTORY.md`](BRANCH_HISTORY.md) | What each branch held, where it landed, the containment proof before deleting one |

## Runbooks — actions an operator performs now

| File | Use |
|---|---|
| [`AGENT_TEAM.md`](AGENT_TEAM.md) | How a session plans, builds, reviews and merges through `.claude/agents/` |
| [`FIRST_SESSION_CHECKLIST.md`](FIRST_SESSION_CHECKLIST.md) | Live-session validation for a new build |
| [`DESK_TESTING_PLAN.md`](DESK_TESTING_PLAN.md) | Trader's step-by-step testing sequence; shipped in the exe, rendered at Settings ▸ Testing Plan |
| [`AWAY_SCANNER_RUNBOOK.md`](AWAY_SCANNER_RUNBOOK.md) | Auto/Away operation and report recovery |
| [`EVENING_MODE_RUNBOOK.md`](EVENING_MODE_RUNBOOK.md) | EVENING mode and ntfy phone setup |
| [`GUI_FLUIDITY_MEASUREMENT_RUNBOOK.md`](GUI_FLUIDITY_MEASUREMENT_RUNBOOK.md) | One command over the stall log; the baseline and targets |
| [`REGIME_INFRASTRUCTURE_PHASE1_RUNBOOK.md`](REGIME_INFRASTRUCTURE_PHASE1_RUNBOOK.md) | Regime-evidence collection and evidence floors |
| [`MACOS_SETUP.md`](MACOS_SETUP.md) | macOS setup; its cloud-mount sections are dead since decision 0015 |
| [`packaging/README.md`](../packaging/README.md) | Frozen-build triggers, process, selftest |

## Active specifications — contracts and doctrine, not status

| File | Owns |
|---|---|
| [`DESK_INTERNALS.md`](DESK_INTERNALS.md) | The long form of every `CLAUDE.md` core-loop rule: incident, measurements, trader words. Read the entry before changing what a rule governs |
| [`ULTIMATE_SETUP_DATABASE_PLAN.md`](ULTIMATE_SETUP_DATABASE_PLAN.md) | Locked research-warehouse plan (Phases 0–8, 28 locked decisions in §23) |
| [`RESEARCH_WAREHOUSE_BUILD_DECISIONS.md`](RESEARCH_WAREHOUSE_BUILD_DECISIONS.md) | Warehouse builder decisions BD-01…; add one when you make one |
| [`RESEARCH_WAREHOUSE_ERD.md`](RESEARCH_WAREHOUSE_ERD.md) | Warehouse dataset identities and read contract |
| [`LOCAL_AI_AUTOMATION_PLAN.md`](LOCAL_AI_AUTOMATION_PLAN.md) | Nightly local-AI layer; §7 cohort grading, §8 fact pack and narration |
| [`REVIEW_LEARNING_LOOP.md`](REVIEW_LEARNING_LOOP.md) | Review evidence → scoreboard → annotation-only `review_policy.json` |
| [`CHART_REVIEW_WORKSPACE_PLAN.md`](CHART_REVIEW_WORKSPACE_PLAN.md) | Trader-annotation schema, veto vocabulary, forward-tracking cohorts |
| [`AUTO_MODES_AND_QUIET_HOURS_PLAN.md`](AUTO_MODES_AND_QUIET_HOURS_PLAN.md) | Phase 0.5 R1: mode matrix, quiet hours, phone push |
| [`M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md`](M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md) | Phase 0.5 R2: Focus adoption gate, provenance, M5 strength board |
| [`SWING_QUALITY_AND_FEEDBACK_PLAN.md`](SWING_QUALITY_AND_FEEDBACK_PLAN.md) | Phase 0.5 R3: swing quality classifier, pre-close honesty, dislike reasons |
| [`DESK_CHART_UNIFICATION_PLAN.md`](DESK_CHART_UNIFICATION_PLAN.md) | Phase 0.5 R4: CaptureRail on every chart, armed-alert paint, repetition control |
| [`M5_SIGNAL_ENGINES_PLAN.md`](M5_SIGNAL_ENGINES_PLAN.md) | Phase 0.5 R5: pure indicators, completed-bars rule, LRSI/confluence/ORB engines |
| [`JOURNAL_RELIABILITY_AND_UX_PLAN.md`](JOURNAL_RELIABILITY_AND_UX_PLAN.md) | Phase 0.5 R7: broker import, reconciliation, the five-tab Journal |
| [`WEEKEND_PREP_PLAN.md`](WEEKEND_PREP_PLAN.md) | Phase 0.5 R8: guided weekend routine and strength boards |
| [`AVWAP_BAND_VARIANT_STUDY.md`](AVWAP_BAND_VARIANT_STUDY.md) | Phase 0.10/0.19: the OneOption band challenger and its shadow harnesses |
| [`DURABILITY_CATCHUP_PLAN.md`](DURABILITY_CATCHUP_PLAN.md) | Self-healing launch task, deterministic backfill, never-reconstruct boundary |
| [`SETUPS_MAJOR.md`](SETUPS_MAJOR.md) / [`SETUPS_TEST.md`](SETUPS_TEST.md) | AI-stated production and research setup doctrine, under trader review |
| [`BROKER_ADAPTERS.md`](BROKER_ADAPTERS.md) | Provider/broker boundary; execution stays out of scope |
| [`SHIP_READINESS.md`](SHIP_READINESS.md) | Packaging and cleanup direction |
| [`SWING_SIMULATOR_INVESTIGATION_2026-09-04.md`](SWING_SIMULATOR_INVESTIGATION_2026-09-04.md) | Why the swing bands were null; the earnings-anchor bridge and gate #59 |
| [`analysis/PROJECT_PROCESS_REVIEW_2026-09-04.md`](analysis/PROJECT_PROCESS_REVIEW_2026-09-04.md) | Codex review that became Phase 0.18 (Q1–Q5); authorizes nothing |
| [`analysis/LAKE_ASSESSMENT_2026-09-04.md`](analysis/LAKE_ASSESSMENT_2026-09-04.md) | Lake measurements, corrected later; do not reuse its conclusions. Its `scripts/` and JSON are pinned by `tests/test_q3_ai_grounding.py` |
| [`analysis/OFFLINE_BUILD_AUTHORIZATION_2026-08-24.md`](analysis/OFFLINE_BUILD_AUTHORIZATION_2026-08-24.md) | Trader authorization the AI jobs and their tests cite; frozen |

## Decision records — accepted constraints

[`decisions/`](decisions/) holds 18 short records, 0001 (decision-support only) to 0018
(deterministic stage before narration). Read one before changing a library, storage or
architecture choice; **0016** is the trader's vision and priorities and breaks every
prioritisation tie. Numbering is chronological.

## Archive — evidence, never context

[`archive/`](archive/) holds everything that is history: the checkpoint, changelog and
roadmap archives, the July GUI plans, the retired Desk Link design, the paste-ready
build prompts for phases already built, and the frozen August reviews under
`archive/analysis/`. Nothing in the code reads any of it. Open one file there to answer
one specific question; never load the folder as context, and never treat an entry
there as an open gate — open gates live only in `CURRENT_CHECKPOINT.md`. Relative
links inside archived files were not rewritten when they moved.

## Maintenance rule

After every repository change and before handoff: refresh `CURRENT_CHECKPOINT.md`,
record behavior/contract changes in `CHANGELOG.md`, advance `plan.md` while keeping
every owed gate, touch `WISHLIST.md` only for trader-directed idea changes, update the
governing spec when its contract changed, classify every added/moved/removed Markdown
file here, and keep `CLAUDE.md` and `AGENTS.md` identical. A document that stops being
current moves to `archive/`; it is never deleted, and never left beside the live specs.
