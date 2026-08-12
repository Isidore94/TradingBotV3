# TradingBotV3 — AI context index

TradingBotV3 is a Windows desktop decision-support system for one trader's day and
swing trading. It does everything except execute orders: pre-session market prep,
candidate discovery (D1 anchored-VWAP swing scans + intraday 5-min bounce detection),
live monitoring with alerts, unattended Auto/Away scanning with a phone report, a
journal, and a controlled research/promotion program for new setups. Order execution
is permanently out of scope (plan.md sec 1).

## Mandatory documentation workflow for every AI

Before proposing, planning, or changing anything, read in this order:

1. `CHANGELOG.md` — get the implemented inventory and do not rebuild landed work.
2. `plan.md` — read Sections 5–7, then follow the phase order in Section 12.
3. `CURRENT_CHECKPOINT.md` — identify the one active phase/item, branch, uncommitted
   work, last verified baseline, and immediate next action. Resume it before selecting
   new work unless the trader explicitly redirects you.
4. `docs/README.md` — open only the active specification, runbook, and decision
   records relevant to the selected item. Historical documents are evidence, not
   current authority.
5. Inspect the source, tests, Git status/history, and runtime artifacts needed to
   verify that the documentation still matches reality.

`WISHLIST.md` contains ideas, not authorized work. Never implement directly from it.
An item enters the build sequence only when the trader explicitly moves it into
`plan.md`.

Before editing, state the exact roadmap/checkpoint item, what already exists, what
remains, governing documents, expected files, tests, and whether the ask-first rule
applies. Do not skip to a later phase because it is easier or more interesting.

After every repository change, reconcile the documentation before handoff:

- always update `CURRENT_CHECKPOINT.md` with the active item, working state, and
  verification result (or explicitly state why the baseline is unchanged);
- update `CHANGELOG.md` when behavior, contracts, architecture, operations, or an
  implementation status changed;
- remove, narrow, or advance the corresponding `plan.md` work while retaining any
  live-validation or promotion gate still owed;
- update the governing detailed spec/decision record when its contract or rationale
  changed;
- update `WISHLIST.md` only for trader-directed idea additions, removals, or
  promotions; an AI may recommend a change but must not silently promote one;
- update `docs/README.md` whenever a Markdown file is added, removed, renamed, or
  reclassified;
- keep `CLAUDE.md` and `AGENTS.md` identical whenever operating instructions change.

Do not create another roadmap, progress ledger, handoff, or status file. The root
control set is `CLAUDE.md`/`AGENTS.md`, `CHANGELOG.md`, `plan.md`,
`CURRENT_CHECKPOINT.md`, `WISHLIST.md`, and `docs/README.md`.

## Core loop / data flow
- Entry: `launch_gui.py` → `scripts/ui/app.py` (PySide6 Trading Desk). Run as Main only — the Desk Link satellite role/system is retired (2026-08-08; see `CHANGELOG.md`); the code remains in-repo pending roadmap cleanup P1.5 but must stay unused. `scripts/gui.py --ui tk` is the legacy Tk UI kept during migration.
- Market data: IBKR TWS/Gateway on `127.0.0.1:7496` (`ibapi`) primary, `yfinance` fallback; bar source is tracked per scan. See `docs/BROKER_ADAPTERS.md`.
- Engines: `scripts/master_avwap.py` (+`master_avwap_lib/`) D1 AVWAP swing scanner; `scripts/bounce_bot.py` (+`bounce_bot_lib/`) intraday M5 bounce detector; `market_prep/` pre-session services.
- Inputs: plain-text watchlists (`longs.txt`, `shorts.txt`, `swinglongs.txt`, `shortswings.txt`) in the user-selected shared "home folder".
- Mutable state lives in that home folder — `C:\TradingBotData`, a plain LOCAL folder on the desk SSD. **There is no cloud drive: Google Drive/OneDrive were removed 2026-08-10 (decision 0015) and are no part of this system.** It holds compact operational state: watchlists, reports, JSONL/CSV evidence logs. Per-machine caches + diagnostics live under `%LOCALAPPDATA%\TradingBotV3` (`scripts/project_paths.py`).
- Storage tiers: desk SSD is local/staging; the **DAS file server `\\MINI-PC\Trading Bot Data` is the durable tier** (expandable to ~100TB) and holds the research lake, the AI store, and cold subtrees pushed hourly by `C:\TradingBotData\_tools\push_cold_to_das.ps1`. Write local first, move to the DAS after, so a file-server outage costs throughput and never correctness.
- Research warehouse (Phases 0–8 implemented; plan.md Phase 3 owns live evidence and post-slice work): very large research files (bar archives, feature/outcome Parquet) go to the DAS research lake at `research_store_dir` (`local_settings.json`; env `TRADINGBOTV3_RESEARCH_DIR`), configured 2026-08-10 to `\\MINI-PC\Trading Bot Data\research_lake` with a machine-local spool at `%LOCALAPPDATA%\TradingBotV3\research_spool` — a separate append-only storage class (decision 0014) that is NEVER inside the `C:\TradingBotData` home folder (`scripts/research_warehouse/config.py` refuses such paths; unset = warehouse fully disabled). The refusal now rests on storage-class separation and cold-push scope, not sync quota (decision 0015). Locked contract: `docs/ULTIMATE_SETUP_DATABASE_PLAN.md`. Builder-level implementation decisions are logged in `docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md`; dataset keys/identities in `docs/RESEARCH_WAREHOUSE_ERD.md`. Shadow-only additive evidence — zero detector/score/alert influence.
- Shadow engines `scripts/market_state.py` (via `market_state_bridge`) and `greatness_monitor` (via `greatness_shadow`) run beside the legacy champions and emit JSONL promotion evidence only.
- Review-learning loop: Alert Center decisions → `alert_review_events.jsonl` → `review_learning.py` scoreboard → AI-curated `review_policy.json` → chart annotations (queue ordering gated to FIFO). See `docs/REVIEW_LEARNING_LOOP.md`.
- Chart paint lines (A4, landed on `testing` 2026-08-09): `scripts/chart_levels.py` builds the D1 S/R stores, prev-day H/L and the projected D1 trendline into a `levels` payload on the ChartDataService **worker** — never the paint path — and `CandleChart.set_levels` draws them with stable ids and click-to-select (`levelSelected`). One paint-lines control (`ui/widgets/paint_lines_button.py`) shows/hides groups, machine-local, defaults all-on. Trendline availability is surveyed in `docs/D1_TRENDLINE_SURVEY.md`; measure it on the desk with `scripts/d1_trendline_survey.py`.
- Price alerts: the Focus tab and Research advanced view share one `PriceAlertService`; the main desk polls and pushes fired alerts to the phone at ntfy `urgent`. The satellite relay/toast layer and the planned satellite edit intents are retired with Desk Link. See `docs/FOCUS_PRICE_ALERTS_PROPOSAL.md` and `docs/EVENING_MODE_RUNBOOK.md`.
- Phone push policy (trader rule 2026-08-11): **AWAY is the only Auto mode that pushes**, and these price alerts are the single deliberate exception — they fire from every mode, including OFF. In AWAY the hourly swing push also carries the full favorite/high-conviction roster, and a second hourly push names the D1 level/event alerts since the previous one (the Alert Center classifies, `AutopilotService` aggregates and gates). Before adding any new ntfy sender, gate it on `auto_mode == AWAY` or state why it belongs with the price alerts.
- Auto/Away phone output: `autopilot_today.txt` is the single verified home-folder digest, with the safety/freshness header first, then numbered best swing trades, then intraday and condensed operations. Mode changes (OFF/DESK/AWAY/EVENING) are made on the main desk.
- Unattended: the separate mini-PC scanner role is retired (2026-08-08) — the 8845HS main desk is the only always-on machine and the only scan host, so no cross-machine IB budget question exists. `scripts/master_avwap_mini_pc.py` stays in-repo only as a slot/state scheduling template pending cleanup.

## Hard invariants (plan.md sec 5 — never violate)
- Decision-support only: never add order execution.
- Legacy SPY pause detection and D1 wick alerts are the champions; shadow engines must never influence live decisions until plan.md sec 7 promotion gates pass.
- No detector/scoring behavior change without golden-result fixtures first (plan.md Sections 5 and 7).
- Never swap `calc_anchored_vwap_bands`' σ formula — every consumer is calibrated to the running-deviation variant.
- Completed bars only for state transitions; a forming bar is preview. Missing data is uncertainty, never confirmation.
- User-entered watchlist names are never auto-removed (CandidateRegistry enforces this; keep it true in any new writer).
- One component owns each timer/thread/job/mutable shared export; a failed publish never destroys the last verified report.
- Point-in-time research uses only information available at the simulated decision time; timestamps carry explicit timezones.
- `review_policy.json` ranks and annotates only — it deliberately has no suppression field; do not add one.

## Tech stack + key deps
- Python ≥3.12 (desk `.venv` measured 3.12.13, a uv-managed CPython built 2026-08-07; the repo venv
  has no `pip` — install with `uv pip install -r … -c constraints.txt --python .venv\Scripts\python.exe`),
  Windows-first with macOS support (`docs/MACOS_SETUP.md`; same code, no fork — platform differences live in launchers, `project_paths.py`, and `ai_credentials.py`), repo-local `.venv`.
- `PySide6`/`qtawesome`/`pyqtgraph` — new Trading Desk UI (`PyQt5` remains only for legacy `TickerMover.py`); Tk — legacy GUI.
- `ibapi` — IBKR market data; `yfinance` — fallback bars; `pandas`/`pyarrow` — bar frames and arrow-backed columns.
- `feedparser` — news RSS for market prep; `openai` — provider-neutral one-way advisory summaries (`scripts/ai_summary.py`, `market_prep/services/ai_service.py`).
- `pytest` (markers: `network`, `broker`, `slow`, `qt`), `ruff` (narrow defect-class select), `pyinstaller` — packaging, via `packaging/tradingbotv3.spec`.
- Layered installs: `requirements-core.txt` (headless) ⊂ `-gui` ⊂ `-dev`, pinned by `constraints.txt` for reproducibility.

## Commands
- Test (before every commit): `.venv\Scripts\python.exe -m pytest tests/ -q` — must be fully green; current baseline lives in `CURRENT_CHECKPOINT.md`. Check pytest's own exit code, not a piped tail's. macOS/Linux: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ -q` (Qt tests need the offscreen platform when headless).
- Smoke (offline, deterministic): `.venv\Scripts\python.exe scripts/smoke_check.py` — 7/7.
- Run: `.venv\Scripts\python.exe launch_gui.py` (Windows) or `.venv/bin/python launch_gui.py` (macOS/Linux; `./setup_macos.command` once first). Keep the Settings ▸ Desk Link role on Main (satellite retired). IB TWS/Gateway runs on the main desk.
- Audits: `scripts/operations_audit.py` (runtime), `scripts/review_capture_audit.py` (capture readiness) — both also render in System Health.
- No deploy pipeline: the user runs the app from this repo on `main`. Never leave the working tree broken.

## Frozen exe rebuild policy
Build: `.venv\Scripts\pyinstaller.exe .\packaging\tradingbotv3.spec --noconfirm` → `dist/TradingBotV3/TradingBotV3.exe`
(onedir, ~400MB, ~4 min). `dist/` and `build/` are gitignored, so the exe is never a commit artifact —
rebuilding is verification only, and skipping it can never leave the tree broken.

- **Do NOT rebuild per commit.** ~4 min machine time plus 5-10 min of the user's click-through is not
  worth it on the ~90% of commits that cannot affect freezing. Logic changes inside existing modules
  are invisible to PyInstaller.
- **Rebuild before each merge to `main`** (same point as the plan.md sec 6 live-validation day), and
  immediately when a change hits a trigger below. Ask the user before spending their time on the
  click-through; the build itself is unattended.
- **Both guards are now BUILT** (2026-08-09, branch `claude/a4-paint-lines-packaging-nug5km`):
  - `tests/test_packaging_spec_drift.py` executes the spec with the PyInstaller API stubbed and
    asserts every top-level `scripts/` package is in its `collect_submodules` list and every
    non-`.py` runtime asset is covered by a `datas` rule. It found the spec five packages behind
    the tree (`ai_jobs`, `desk_link`, `gui_app`, `indicators`, `market_prep_gui`); `desk_link` is
    now bundled, and the other four are documented allowlist entries — each unreachable from
    `launch_gui.py`, the frozen entry point.
    **Fix the spec, never the test** — deliberate omissions go in its documented allowlists.
  - `launch_gui.py --selftest` (`scripts/selftest.py`) imports every lazily-loaded engine and loads
    every `__file__`-relative asset (theme.qss, the veto vocabulary), no window and no network,
    exiting non-zero with every failure named. Run it against the FROZEN exe:
    `dist\TradingBotV3\TradingBotV3.exe --selftest`. Expect `selftest OK: 29/29 checks passed (frozen)`
    and exit 0 — that is what replaces the trader's click-through (desk-verified 2026-08-09).
  - The two lists must never contradict each other: a package in `PACKAGES_NOT_IN_THE_BUNDLE` cannot
    also be in `selftest.LAZY_ENGINE_MODULES`, because the frozen exe genuinely does not contain it.
    The unfrozen suite cannot see such a clash — a repo checkout imports anything under `scripts/` —
    so `test_the_selftest_never_demands_a_package_the_bundle_excludes` now asserts the two are
    disjoint. It exists because `ai_jobs` was in both, the unfrozen selftest passed 30/30 all week,
    and the desk's first frozen run (2026-08-09) was the first execution anywhere to catch it.
  - Between them, triggers 2-4 below are now caught by the normal test run.
- **Triggers — a change of these kinds can break the bundle, so rebuild and run the frozen selftest:**
  1. New third-party dependency (`requirements-*.txt` / `constraints.txt`) — may need hiddenimports or `collect_data_files`. **Not** covered by the guards.
  2. New non-`.py` runtime asset. The spec mirrors every `FIRST_PARTY_PACKAGES` tree plus `config/`; an asset outside those silently goes missing. *(spec-drift test catches it)*
  3. New top-level package under `scripts/` that is imported lazily — the spec's `collect_submodules` list is hardcoded. *(spec-drift test catches it)*
  4. New dynamic import by string name (`importlib`, name-keyed panel/service lookup) in an uncollected package. *(add the module to `selftest.LAZY_ENGINE_MODULES` — but only if a frozen run can actually reach it; see the disjointness rule above)*
  5. Any change touching `__file__` / `ROOT_DIR` / `sys.path` — `ROOT_DIR` is `sys._MEIPASS` when frozen. **Not** fully covered; the selftest checks the phantom-root assumption only.
- Read `packaging/README.md` "Things that will bite you" before touching the spec or any of the above.
  The signature failure is a bundle that starts fine and dies at the first lazy import, so "it launched"
  is not proof; the selftest is what exercises the engines.

## Working agreement for agents
- Follow the mandatory documentation workflow above. `plan.md` owns build order;
  `CURRENT_CHECKPOINT.md` owns the active item. Do not re-implement anything in
  `CHANGELOG.md` or implement anything directly from `WISHLIST.md`.
- `main` is the trunk; branch per milestone/packet, merge back after a live-session validation day passes (plan.md sec 6).
- Commit small and green; push after each commit. If a task will exceed usage limits, commit and push so another agent can take over from a green state.
- First live session on any new build: run plan.md sec 6 checklist; do NOT tune thresholds from one session.
- **File-scoped ask-first rule** (checkpoint review 2026-08-08): any edit to a file housing detector/scoring/alert code is asked about BEFORE it is made — even for capture-side or evidence-only changes in that file. Ambiguity is the trigger to ask, not a license to judge.
- While unmerged branch code runs in production via a scheduled task (see `docs/CHECKPOINT_REVIEW_2026-08-08.md`): never switch branches on the desk without disarming that task first.

## Where to read more
- `CHANGELOG.md` — authoritative implemented inventory and revision history.
- `plan.md` — remaining roadmap and single source of truth for unfinished work. Sec 5 invariants, sec 6 live validation, sec 7 promotion ladder, sec 12 ordered work queue. Read before any feature work.
- `CURRENT_CHECKPOINT.md` — active item, branch, working state, and exact verification baseline.
- `WISHLIST.md` — trader-visible candidate integrations and deferred ideas; never an implementation queue.
- `docs/README.md` — classifies every supporting file as active runbook/reference or historical evidence.
- `GUI_TRADE_DISCOVERY_LEARNING_PLAN.md` (+ historical `GUI_LEARNING_PROGRESS.md` pointer) — preserved GUI learning design; never overrides plan.md Sections 5–7 or Phase order.
- `GUI_PRODUCT_PLAN.md` — historical consumer GUI product design reference.
- `docs/decisions/` — backfilled decision records; read before changing a library, storage, or architecture choice.
- `docs/REVIEW_LEARNING_LOOP.md` — how the AI reads review artifacts and writes `review_policy.json`.
- `docs/ULTIMATE_SETUP_DATABASE_PLAN.md` — LOCKED implementation plan for the DAS research warehouse (capture policy + IB pacing budget, 13-table schemas, Phases 0-8 build order, 28 locked decisions in its Section 23 — do not re-litigate them; open items live only in its confirmation register).
- `docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md` — the warehouse builder decision log (BD-01…): every implementation choice the locked plan left open, with rationale and reopen triggers. Read before changing warehouse internals; add a BD entry when you make a new one. `docs/RESEARCH_WAREHOUSE_ERD.md` is its dataset/identity map.
- `docs/SETUPS_MAJOR.md` / `docs/SETUPS_TEST.md` — AI-stated understanding of the production setups and the study/research setups, under trader review; fold corrections back in.
- `docs/FIRST_SESSION_CHECKLIST.md`, `docs/AWAY_SCANNER_RUNBOOK.md`, `docs/REGIME_INFRASTRUCTURE_PHASE1_RUNBOOK.md` — operational runbooks for live sessions.
- `docs/FOCUS_PRICE_ALERTS_PROPOSAL.md`, `docs/EVENING_MODE_RUNBOOK.md` — Focus price-alert delivery and ntfy phone setup. `docs/MULTI_MACHINE_DESK_PROPOSAL.md` is historical: Desk Link/satellites retired 2026-08-08.
- `docs/LOCAL_AI_AUTOMATION_PLAN.md` — local LLM batch layer on the always-on main desk: automated AI summaries, daily digest ledger, journal enrichment, review-policy curation, frontier synthesis. Advisory-only; no inference during market hours.
- `docs/DURABILITY_CATCHUP_PLAN.md` — durability design: self-healing launch task, deterministic backfill with `capture_mode` provenance, never-reconstruct boundary, and the Master AVWAP tracker staleness override.
- `docs/CHART_REVIEW_WORKSPACE_PLAN.md` — Chart Review workspace and trader decision capture: `trader_annotations.jsonl` schema v1, the versioned veto vocabulary, veto forward-tracking cohorts, and why a lookup never writes a watchlist. The stream is analysis-only evidence — it must never mute, suppress, score, gate, or alert.
- `docs/MACOS_SETUP.md` — running the desk on macOS (native TWS, Keychain keys). Its Google Drive mount-discovery sections are dead since decision 0015; the code is harmless and stays until a macOS run is actually needed.
- `docs/decisions/0015-no-cloud-sync-das-file-server-storage.md` — no cloud sync; the DAS is the durable tier. Read it before touching storage paths, the writer lease, or backup rules.
- `docs/SHIP_READINESS.md`, `docs/BROKER_ADAPTERS.md`, `packaging/README.md` — shipping direction and future multi-broker architecture.
- Runtime facts: main desk is an always-on Ryzen 7 8845HS mini-PC (32GB DDR5, Radeon 780M iGPU — local-LLM host) and does everything; the former i5-8600K/3080 Ti desktop is powered down most days (discord/chat, at most ad-hoc alternative scanning — never an always-on or writer role). Storage is a DAS file server at `\\MINI-PC\Trading Bot Data`, expandable to ~100TB, holding `research_lake/`, `ai_store/`, and the cold-pushed `data/`, `output/`, `logs/`, `away_report_archive/` subtrees. Full scan ≈ 28.5 min, network-bound (measured on the old desk); the 8845HS measured 17–21 min on 2026-08-10 over 1,097 symbols. Post-session artifacts under `%LOCALAPPDATA%\TradingBotV3\diagnostics\` (`run_manifests\`, `spy_state_shadow.jsonl`, `greatness_shadow.jsonl`, `job_ledger.jsonl`, `heartbeat.json`).

`AGENTS.md` is a copy of this file (symlinks don't survive Windows checkouts) — edit CLAUDE.md, then re-copy.
