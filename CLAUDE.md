# TradingBotV3 — AI context index

TradingBotV3 is a Windows desktop decision-support system for one trader's day and
swing trading. It does everything except execute orders: pre-session market prep,
candidate discovery (D1 anchored-VWAP swing scans + intraday 5-min bounce detection),
live monitoring with alerts, unattended Auto/Away scanning with a phone report, a
journal, and a controlled research/promotion program for new setups. Order execution
is permanently out of scope (plan.md sec 1).

## Core loop / data flow
- Entry: `TradingBotV3_GUI.cmd` / `launch_gui.py` → `scripts/ui/app.py` (PySide6 Trading Desk). `scripts/gui.py --ui tk` is the legacy Tk UI kept during migration.
- Market data: IBKR TWS/Gateway on `127.0.0.1:7496` (`ibapi`) primary, `yfinance` fallback; bar source is tracked per scan. See `docs/BROKER_ADAPTERS.md`.
- Engines: `scripts/master_avwap.py` (+`master_avwap_lib/`) D1 AVWAP swing scanner; `scripts/bounce_bot.py` (+`bounce_bot_lib/`) intraday M5 bounce detector; `market_prep/` pre-session services.
- Inputs: plain-text watchlists (`longs.txt`, `shorts.txt`, `swinglongs.txt`, `shortswings.txt`) in the user-selected shared "home folder".
- Mutable state lives in that cloud-synced home folder (Drive/OneDrive): watchlists, reports, JSONL/CSV evidence logs. Per-machine caches + diagnostics live under `%LOCALAPPDATA%\TradingBotV3` (`scripts/project_paths.py`).
- Shadow engines `scripts/market_state.py` (via `market_state_bridge`) and `greatness_monitor` (via `greatness_shadow`) run beside the legacy champions and emit JSONL promotion evidence only.
- Review-learning loop: Alert Center decisions → `alert_review_events.jsonl` → `review_learning.py` scoreboard → AI-curated `review_policy.json` → chart annotations (queue ordering gated to FIFO). See `docs/REVIEW_LEARNING_LOOP.md`.
- Unattended: `scripts/master_avwap_mini_pc.py` runs scheduled scans on the mini-PC and publishes a phone status file. Keep Auto Pilot OFF there while the desktop scans (no cross-machine IB budget yet).

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
- Python ≥3.12 (desktop runs 3.14.6), Windows-first, repo-local `.venv`.
- `PySide6`/`qtawesome`/`pyqtgraph` — new Trading Desk UI (`PyQt5` remains only for legacy `TickerMover.py`); Tk — legacy GUI.
- `ibapi` — IBKR market data; `yfinance` — fallback bars; `pandas`/`pyarrow` — bar frames and arrow-backed columns.
- `feedparser` — news RSS for market prep; `openai` — provider-neutral one-way advisory summaries (`scripts/ai_summary.py`, `market_prep/services/ai_service.py`).
- `pytest` (markers: `network`, `broker`, `slow`, `qt`), `ruff` (narrow defect-class select), `pyinstaller` — future packaging.
- Layered installs: `requirements-core.txt` (headless) ⊂ `-gui` ⊂ `-dev`, pinned by `constraints.txt` for reproducibility.

## Commands
- Test (before every commit): `.venv\Scripts\python.exe -m pytest tests/ -q` — must be fully green; current baseline lives in `SOL_PROGRESS.md`. Check pytest's own exit code, not a piped tail's.
- Smoke (offline, deterministic): `.venv\Scripts\python.exe scripts/smoke_check.py` — 7/7.
- Run: `.\TradingBotV3_GUI.cmd` (or `launch_gui.py`); IB TWS/Gateway must be connected for data.
- Audits: `scripts/operations_audit.py` (runtime), `scripts/review_capture_audit.py` (capture readiness) — both also render in System Health.
- No deploy pipeline: the user runs the app from this repo on `main`. Never leave the working tree broken.

## Working agreement for agents
- `plan.md` outranks everything; work its Section 12 queue top to bottom. Do not re-implement anything marked implemented; do not promote anything marked shadow without Section 7 evidence.
- `main` is the trunk; branch per milestone/packet, merge back after a live-session validation day passes (plan.md sec 6).
- Commit small and green; push after each commit. If a task will exceed usage limits, commit and push so another agent can take over from a green state.
- First live session on any new build: run plan.md sec 6 checklist; do NOT tune thresholds from one session.

## Where to read more
- `plan.md` — master roadmap and single source of truth. Sec 5 invariants, sec 6 live validation, sec 7 promotion ladder, sec 12 ordered work queue. Read before any feature work.
- `SOL_PROGRESS.md` — checkpoint ledger: current branch, test baseline, what already landed.
- `GUI_TRADE_DISCOVERY_LEARNING_PLAN.md` (+ `GUI_LEARNING_PROGRESS.md` stamp) — subordinate GUI learning program; never overrides plan.md secs 5-7 or the sec 12 order.
- `GUI_PRODUCT_PLAN.md` — consumer GUI product design.
- `docs/decisions/` — backfilled decision records; read before changing a library, storage, or architecture choice.
- `docs/REVIEW_LEARNING_LOOP.md` — how the AI reads review artifacts and writes `review_policy.json`.
- `docs/FIRST_SESSION_CHECKLIST.md`, `docs/AWAY_SCANNER_RUNBOOK.md`, `docs/REGIME_INFRASTRUCTURE_PHASE1_RUNBOOK.md` — operational runbooks for live sessions.
- `docs/SHIP_READINESS.md`, `docs/BROKER_ADAPTERS.md`, `packaging/README.md` — shipping direction and future multi-broker architecture.
- Runtime facts: primary desktop i5-8600K/32GB; full scan ≈ 28.5 min, network-bound. Post-session artifacts under `%LOCALAPPDATA%\TradingBotV3\diagnostics\` (`run_manifests\`, `spy_state_shadow.jsonl`, `greatness_shadow.jsonl`, `job_ledger.jsonl`, `heartbeat.json`).

`AGENTS.md` is a copy of this file (symlinks don't survive Windows checkouts) — edit CLAUDE.md, then re-copy.
