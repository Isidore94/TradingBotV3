# TradingBotV3 — AI context index

TradingBotV3 is a Windows desktop decision-support system for one trader's day and
swing trading. It does everything except execute orders: pre-session market prep,
candidate discovery (D1 anchored-VWAP swing scans + intraday 5-min bounce detection),
live monitoring with alerts, unattended Auto/Away scanning with a phone report, a
journal, and a controlled research/promotion program for new setups. Order execution
is permanently out of scope (plan.md sec 1).

## Core loop / data flow
- Entry: `launch_gui.py` → `scripts/ui/app.py` (PySide6 Trading Desk). Run as Main only — the Desk Link satellite role/system is retired (2026-08-08, plan.md 7a note); the code remains in-repo pending a cleanup packet but must stay unused. `scripts/gui.py --ui tk` is the legacy Tk UI kept during migration.
- Market data: IBKR TWS/Gateway on `127.0.0.1:7496` (`ibapi`) primary, `yfinance` fallback; bar source is tracked per scan. See `docs/BROKER_ADAPTERS.md`.
- Engines: `scripts/master_avwap.py` (+`master_avwap_lib/`) D1 AVWAP swing scanner; `scripts/bounce_bot.py` (+`bounce_bot_lib/`) intraday M5 bounce detector; `market_prep/` pre-session services.
- Inputs: plain-text watchlists (`longs.txt`, `shorts.txt`, `swinglongs.txt`, `shortswings.txt`) in the user-selected shared "home folder".
- Mutable state lives in that cloud-synced home folder (Drive/OneDrive): watchlists, reports, JSONL/CSV evidence logs. Per-machine caches + diagnostics live under `%LOCALAPPDATA%\TradingBotV3` (`scripts/project_paths.py`).
- Research warehouse (in build, plan.md sec 12 item 13a): very large research files (bar archives, feature/outcome Parquet) go to the DAS research lake at `research_store_dir` (`local_settings.json`; env `TRADINGBOTV3_RESEARCH_DIR`) — a separate append-only storage class (decision 0014) that is NEVER inside the Drive home folder (`scripts/research_warehouse/config.py` refuses such paths; unset = warehouse fully disabled). Locked build plan: `docs/ULTIMATE_SETUP_DATABASE_PLAN.md` (Phases 0-8; Phase 0 landed). Shadow-only additive evidence — zero detector/score/alert influence.
- Shadow engines `scripts/market_state.py` (via `market_state_bridge`) and `greatness_monitor` (via `greatness_shadow`) run beside the legacy champions and emit JSONL promotion evidence only.
- Review-learning loop: Alert Center decisions → `alert_review_events.jsonl` → `review_learning.py` scoreboard → AI-curated `review_policy.json` → chart annotations (queue ordering gated to FIFO). See `docs/REVIEW_LEARNING_LOOP.md`.
- Price alerts: the Focus tab and Research advanced view share one `PriceAlertService`; the main desk polls and pushes fired alerts to the phone at ntfy `urgent`. The satellite relay/toast layer and the planned satellite edit intents are retired with Desk Link. See `docs/FOCUS_PRICE_ALERTS_PROPOSAL.md` and `docs/EVENING_MODE_RUNBOOK.md`.
- Auto/Away phone output: `autopilot_today.txt` is the single verified Drive digest, with the safety/freshness header first, then numbered best swing trades, then intraday and condensed operations. Mode changes (OFF/DESK/AWAY/EVENING) are made on the main desk.
- Unattended: the separate mini-PC scanner role is retired (2026-08-08) — the 8845HS main desk is the only always-on machine and the only scan host, so no cross-machine IB budget question exists. `scripts/master_avwap_mini_pc.py` stays in-repo as the slot/state scheduling template for plan.md 13b jobs.

## Hard invariants (plan.md sec 5 — never violate)
- Decision-support only: never add order execution.
- Legacy SPY pause detection and D1 wick alerts are the champions; shadow engines must never influence live decisions until plan.md sec 7 promotion gates pass.
- No detector/scoring behavior change without golden-result fixtures first (plan.md Milestone 3).
- Never swap `calc_anchored_vwap_bands`' σ formula — every consumer is calibrated to the running-deviation variant.
- Completed bars only for state transitions; a forming bar is preview. Missing data is uncertainty, never confirmation.
- User-entered watchlist names are never auto-removed (CandidateRegistry enforces this; keep it true in any new writer).
- One component owns each timer/thread/job/mutable shared export; a failed publish never destroys the last verified report.
- Point-in-time research uses only information available at the simulated decision time; timestamps carry explicit timezones.
- `review_policy.json` ranks and annotates only — it deliberately has no suppression field; do not add one.

## Tech stack + key deps
- Python ≥3.12 (desktop runs 3.14.6), Windows-first with macOS support (`docs/MACOS_SETUP.md`; same code, no fork — platform differences live in launchers, `project_paths.py`, and `ai_credentials.py`), repo-local `.venv`.
- `PySide6`/`qtawesome`/`pyqtgraph` — new Trading Desk UI (`PyQt5` remains only for legacy `TickerMover.py`); Tk — legacy GUI.
- `ibapi` — IBKR market data; `yfinance` — fallback bars; `pandas`/`pyarrow` — bar frames and arrow-backed columns.
- `feedparser` — news RSS for market prep; `openai` — provider-neutral one-way advisory summaries (`scripts/ai_summary.py`, `market_prep/services/ai_service.py`).
- `pytest` (markers: `network`, `broker`, `slow`, `qt`), `ruff` (narrow defect-class select), `pyinstaller` — packaging, via `packaging/tradingbotv3.spec`.
- Layered installs: `requirements-core.txt` (headless) ⊂ `-gui` ⊂ `-dev`, pinned by `constraints.txt` for reproducibility.

## Commands
- Test (before every commit): `.venv\Scripts\python.exe -m pytest tests/ -q` — must be fully green; current baseline lives in `SOL_PROGRESS.md`. Check pytest's own exit code, not a piped tail's. macOS/Linux: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/ -q` (Qt tests need the offscreen platform when headless).
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
- **Triggers — a change of these kinds can break the bundle, so rebuild and launch the exe:**
  1. New third-party dependency (`requirements-*.txt` / `constraints.txt`) — may need hiddenimports or `collect_data_files`.
  2. New non-`.py` runtime asset. The spec mirrors `scripts/ui/**` and `config/` only; anything elsewhere silently goes missing.
  3. New top-level package under `scripts/` that is imported lazily — the spec's `collect_submodules` list is hardcoded.
  4. New dynamic import by string name (`importlib`, name-keyed panel/service lookup) in an uncollected package.
  5. Any change touching `__file__` / `ROOT_DIR` / `sys.path` — `ROOT_DIR` is `sys._MEIPASS` when frozen.
- Read `packaging/README.md` "Things that will bite you" before touching the spec or any of the above.
  The signature failure is a bundle that starts fine and dies at the first lazy import, so "it launched"
  is not proof; exercise the engines.
- Two cheap guards are proposed but not built — a spec-drift pytest (asserts every `scripts/` package and
  non-`.py` asset is covered) and a `launch_gui.py --selftest` flag (imports every lazily-loaded engine,
  exits non-zero on failure). Building them would move triggers 2-4 into the normal test run and replace
  the click-through with a ~30s automated check. Propose them if packaging work comes up again.

## Working agreement for agents
- `plan.md` outranks everything; work its Section 12 queue top to bottom. Do not re-implement anything marked implemented; do not promote anything marked shadow without Section 7 evidence.
- `main` is the trunk; branch per milestone/packet, merge back after a live-session validation day passes (plan.md sec 6).
- Commit small and green; push after each commit. If a task will exceed usage limits, commit and push so another agent can take over from a green state.
- First live session on any new build: run plan.md sec 6 checklist; do NOT tune thresholds from one session.
- **File-scoped ask-first rule** (checkpoint review 2026-08-08): any edit to a file housing detector/scoring/alert code is asked about BEFORE it is made — even for capture-side or evidence-only changes in that file. Ambiguity is the trigger to ask, not a license to judge.
- While unmerged branch code runs in production via a scheduled task (see `docs/CHECKPOINT_REVIEW_2026-08-08.md`): never switch branches on the desk without disarming that task first.

## Where to read more
- `plan.md` — master roadmap and single source of truth. Sec 5 invariants, sec 6 live validation, sec 7 promotion ladder, sec 12 ordered work queue. Read before any feature work.
- `SOL_PROGRESS.md` — checkpoint ledger: current branch, test baseline, what already landed.
- `GUI_TRADE_DISCOVERY_LEARNING_PLAN.md` (+ `GUI_LEARNING_PROGRESS.md` stamp) — subordinate GUI learning program; never overrides plan.md secs 5-7 or the sec 12 order.
- `GUI_PRODUCT_PLAN.md` — consumer GUI product design.
- `docs/decisions/` — backfilled decision records; read before changing a library, storage, or architecture choice.
- `docs/REVIEW_LEARNING_LOOP.md` — how the AI reads review artifacts and writes `review_policy.json`.
- `docs/ULTIMATE_SETUP_DATABASE_PLAN.md` — LOCKED implementation plan for the DAS research warehouse (capture policy + IB pacing budget, 13-table schemas, Phases 0-8 build order, 28 locked decisions in its Section 23 — do not re-litigate them; open items live only in its confirmation register).
- `docs/SETUPS_MAJOR.md` / `docs/SETUPS_TEST.md` — AI-stated understanding of the production setups and the study/research setups, under trader review; fold corrections back in.
- `docs/FIRST_SESSION_CHECKLIST.md`, `docs/AWAY_SCANNER_RUNBOOK.md`, `docs/REGIME_INFRASTRUCTURE_PHASE1_RUNBOOK.md` — operational runbooks for live sessions.
- `docs/FOCUS_PRICE_ALERTS_PROPOSAL.md`, `docs/EVENING_MODE_RUNBOOK.md` — Focus price-alert delivery and ntfy phone setup. `docs/MULTI_MACHINE_DESK_PROPOSAL.md` is historical: Desk Link/satellites retired 2026-08-08 (plan.md 7a note).
- `docs/LOCAL_AI_AUTOMATION_PLAN.md` — local LLM batch layer on the always-on main desk (plan.md item 13b): automated AI summaries, daily digest ledger, journal enrichment, review-policy curation, frontier synthesis. Advisory-only; no inference during market hours.
- `docs/DURABILITY_CATCHUP_PLAN.md` — durability packet (plan.md item 13c): self-healing launch task, deterministic backfill with `capture_mode` provenance, never-reconstruct boundary, and the Master AVWAP tracker staleness override.
- `docs/MACOS_SETUP.md` — running the desk on macOS (native TWS, CloudStorage Drive mount, Keychain keys).
- `docs/SHIP_READINESS.md`, `docs/BROKER_ADAPTERS.md`, `packaging/README.md` — shipping direction and future multi-broker architecture.
- Runtime facts: main desk is an always-on Ryzen 7 8845HS mini-PC (32GB DDR5, Radeon 780M iGPU — planned local-LLM host, plan.md 13b); the former i5-8600K/3080 Ti desktop is powered down most days (discord/chat, at most ad-hoc alternative scanning — never an always-on or writer role). Full scan ≈ 28.5 min, network-bound (measured on the old desk). Post-session artifacts under `%LOCALAPPDATA%\TradingBotV3\diagnostics\` (`run_manifests\`, `spy_state_shadow.jsonl`, `greatness_shadow.jsonl`, `job_ledger.jsonl`, `heartbeat.json`).

`AGENTS.md` is a copy of this file (symlinks don't survive Windows checkouts) — edit CLAUDE.md, then re-copy.
