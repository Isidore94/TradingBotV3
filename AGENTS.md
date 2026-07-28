# TradingBotV3 — agent operating guide

1. Never run out of usage limits before finishing a task. If a task will go
   over usage limits, save your work, commit, and push to GitHub so another
   agent (CODEX / Codex) can take over from a green state.

## Source of truth

- **`plan.md`** is the consolidated master roadmap (Codex 5.6 Sol,
  2026-07-11). Section 12 is the ordered execution list — work it top to
  bottom. Section 5 lists non-negotiable invariants. Section 6 is the
  live-validation program; Section 7 defines shadow-evidence floors and the
  promotion ladder.
- **`SOL_PROGRESS.md`** is the checkpoint ledger of what already landed.
- Do not re-implement anything marked implemented; do not promote anything
  marked shadow without the Section 7 evidence.

## Branches

Linear history: `main` → `Sol` → `Sol2` → `Sol3` (each contains the previous).
Work on `Sol3` or branch from it. Merge to `main` only after a live-session
validation day passes (plan.md sec 6). The user runs the app from this repo —
never leave the working tree broken.

## Verification gates (before every commit)

- `.venv\Scripts\python.exe -m pytest tests/ -q` — full suite green
  (baseline **802 passed**, ~25s). Check pytest's own exit code, not a piped
  tail's.
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

## Key modules added on Sol/Sol2/Sol3 (all pure + tested)

- `scripts/market_state.py` — SPY pullback state machine (side-symmetric).
- `scripts/relative_strength.py` — aligned multi-window RS ranking engine.
- `scripts/candidate_registry.py` — provenance/lease watchlist store.
- `scripts/greatness_monitor.py` + `greatness_shadow.py` — staged D1
  confirmation engine, shadow-wired into the live trigger path.
- `scripts/job_ledger.py`, `scripts/writer_lease.py`,
  `scripts/diagnostics/` — Phase 2 runtime reliability + run manifests.
- `scripts/smoke_check.py` — deterministic smoke command.

## Review-learning loop (Alert Center) — AI-in-the-loop by design

The visual alert review surface learns the trader's preferences in three
phases. Phases 0-1 are live on `main` (2026-07-28); **Phase 2 is deliberately
an AI review step, not hard-coded logic**: the user wants Fable or Sol to
read the artifacts below periodically and decide what to prioritize and how
to surface the best alerts.

- **Raw decision log**: `<shared home>/alert_review_events.jsonl`
  (`review_events.py`, schema `review_events_v1`). One row per decision -
  shown impressions, skip/remove/restore, focus adds + cross-focus toggles,
  favorite/dislike (with reason), watch arm/disarm/fired/expired, level
  arm/disarm/fired with the quick-fill source - each with dwell time, queue
  length, and structured alert context (tier, PROVEN/banger, bounce types,
  RRS, rvol, market environment). `event_id` joins to
  `intraday_bounce_candidates.csv` / `intraday_bounce_outcomes.csv`. The
  Master AVWAP setups table's ★/✕ (the trader's actual SWING decisions) log
  here too with `surface: "setups"` and the row's structured context -
  bucket, setup_family, setup_tags, expected_r, sector/industry RS - which
  the scoreboard aggregates as extra dimensions (`bucket`, `setup_family`,
  `setup_tag`, `expected_r_band`; policy rules may target them). Table
  actions carry no "shown" impression, so their take rates are blank by
  design - grade them by taken-vs-passed forward returns instead.
- **Aggregated scoreboard**: `<shared home>/review_preference_state.json` +
  `output/review_learning_report.txt` (`review_learning.py`, schema
  `review_learning_v1`). P(take|shown) per segment with n/(n+10) shrinkage,
  taken-vs-passed outcomes (close R via event join; 3/5-session forward
  returns for D1 names), blind spots / leaks (min 8 shown), watch
  conversion. Auto-rebuilt when stale by a GUI startup thread; on demand via
  `python scripts/review_learning.py`.
- **When reviewing as the AI**: read the report + state, cross-reference
  `pick_feedback.jsonl` dislike reasons, and write your decisions to
  `<shared home>/review_policy.json` (`review_policy.py`, schema
  `review_policy_v1`): per-(dimension, segment) rules with `priority_delta`
  (queue ordering, clamped +/-5), `annotation` (shown on the chart), and
  `watch_kind`/`fill_source` presets (hint only, never auto-armed).
  `python scripts/review_policy.py --draft` turns the scoreboard callouts
  into `review_policy_draft.json` as a starting point - curate it, then
  save as `review_policy.json`. The Alert Center picks the file up on mtime
  change, no restart needed (`review_guidance.py` scores each alert as
  take-prob*100 + segment-R*20 + delta*10; chart-watch hits always stay at
  the queue front). Rank and annotate only - never auto-suppress an alert
  (house rule: mute -> CAUTION, focus picks always surface), and the
  policy format deliberately has no suppression field - do not add one.
- Capture only starts once the GUI restarts onto a build >= c45d965; expect
  ~2-3 weeks of sessions before segment samples clear the n>=8 gates.

## Runtime facts

- Primary machine: single desktop PC (i5-8600K, 32GB). Mini-PC is secondary;
  keep Auto Pilot OFF there while the desktop scans (no cross-machine IB
  budget yet). Full scan ≈ 28.5 min on this box, network-bound.
- Session artifacts to inspect after any live day (all under
  `C:\Users\aaron\AppData\Local\TradingBotV3\diagnostics\`): `run_manifests\`,
  `spy_state_shadow.jsonl`, `greatness_shadow.jsonl`, `job_ledger.jsonl`,
  `heartbeat.json`.
- First live session on Sol3: run plan.md sec 6 checklist; validate logging
  and lifecycle behavior first; do NOT tune thresholds from one session.
