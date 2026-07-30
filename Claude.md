# TradingBotV3 — agent operating guide

1. Never run out of usage limits before finishing a task. If a task will go
   over usage limits, save your work, commit, and push to GitHub so another
   agent (CODEX / Claude) can take over from a green state.

## Source of truth

- **`plan.md`** is the consolidated master roadmap (Codex 5.6 Sol,
  2026-07-11). Section 12 is the ordered execution list — work it top to
  bottom. Section 5 lists non-negotiable invariants. Section 6 is the
  live-validation program; Section 7 defines shadow-evidence floors and the
  promotion ladder.
- **`SOL_PROGRESS.md`** is the checkpoint ledger of what already landed.
- **`GUI_TRADE_DISCOVERY_LEARNING_PLAN.md`** is the subordinate product
  addendum for discovery/Focus/review/learning GUI work. Section 15 is its
  ordered phase list; `GUI_LEARNING_PROGRESS.md` is its checkpoint stamp. It
  never overrides `plan.md` sections 5-7 or the Section 12 order.
- Do not re-implement anything marked implemented; do not promote anything
  marked shadow without the Section 7 evidence.

## Branches

`Sol`, `Sol2` and `Sol3` no longer exist as branches — all of that work is
merged into `main` (verified: plan checkpoints `20cefb3` and `3443c69` are both
ancestors of `main`). `main` is the trunk: branch from it for a milestone or a
packet, then merge back after a live-session validation day passes (plan.md
sec 6). The user runs the app from this repo — never leave the working tree
broken.

## Verification gates (before every commit)

- `.venv\Scripts\python.exe -m pytest tests/ -q` — full suite green
  (baseline **1249 passed, 5 subtests passed**, ~38s). Check pytest's own exit
  code, not a piped tail's.
- `.venv\Scripts\python.exe scripts/smoke_check.py` — 7/7 deterministic
  checks, no network needed.
- Commit small and green; push to origin after each commit.

## Non-negotiable invariants (from plan.md sec 5 — do not violate)

- Legacy SPY pause detection and D1 wick alerts are the **champions**. The
  shadow engines (`market_state` via `market_state_bridge`,
  `greatness_monitor` via `greatness_shadow`) must never influence live
  decisions until the plan's promotion gates pass.
- No detector/scoring behavior change without golden-result fixtures first
  (plan.md Milestone 3).
- Never swap `calc_anchored_vwap_bands`' σ formula — every band consumer is
  calibrated to the current running-deviation variant.
- User-entered watchlist names are never auto-removed (CandidateRegistry
  enforces this; keep it true in any new writer).
- Completed bars only for state transitions; a forming bar is preview.
- Decision-support only: never add order execution.

## Key modules from the Sol line, now on `main` (all pure + tested)

- `scripts/market_state.py` — SPY pullback state machine (side-symmetric).
- `scripts/relative_strength.py` — aligned multi-window RS ranking engine.
- `scripts/candidate_registry.py` — provenance/lease watchlist store.
- `scripts/greatness_monitor.py` + `greatness_shadow.py` — staged D1
  confirmation engine, shadow-wired into the live trigger path.
- `scripts/job_ledger.py`, `scripts/writer_lease.py`,
  `scripts/diagnostics/` — Phase 2 runtime reliability + run manifests.
- `scripts/smoke_check.py` — deterministic smoke command.
- `scripts/operations_audit.py` — composed runtime + capture-readiness audit
  (central to plan.md Section 12 items 2 and 5).
- `scripts/review_capture_audit.py` — decision-log / scoreboard / policy-gate
  capture readiness, also rendered in System Health.
- `scripts/ui/panels/health_panel.py` — the System Health surface that renders
  both audits.

## Runtime facts

- Primary machine: single desktop PC (i5-8600K, 32GB). Mini-PC is secondary;
  keep Auto Pilot OFF there while the desktop scans (no cross-machine IB
  budget yet). Full scan ≈ 28.5 min on this box, network-bound.
- Session artifacts to inspect after any live day (all under
  `C:\Users\aaron\AppData\Local\TradingBotV3\diagnostics\`): `run_manifests\`,
  `spy_state_shadow.jsonl`, `greatness_shadow.jsonl`, `job_ledger.jsonl`,
  `heartbeat.json`.
- First live session on any new build: run plan.md sec 6 checklist; validate logging
  and lifecycle behavior first; do NOT tune thresholds from one session.
