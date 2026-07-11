# Sol3 implementation checkpoint

This is a concise handoff record. [`plan.md`](plan.md) is the authoritative roadmap, promotion policy, and live-validation plan.

## Checkpoint

- Branch: `Sol3`
- Commit: `20cefb3`
- Date: 2026-07-11
- Last reported green baseline: **802 tests passing**
- Smoke command: `python scripts/smoke_check.py`

## Implemented and green

### Phase 0

- Green baseline and dormant-defect fixes.
- Duplicate-definition guard and modern Qt proxy API.
- Entrypoint/lazy-import hygiene.
- Deterministic smoke command.
- Owned runtime/thread/process shutdown behavior.
- `pyproject.toml`, pytest markers, narrow lint gates, and pinned constraints.

### Phase 1 core

- Structured run manifest on Master success or failure.
- Phase timings, counters, and bounded local history.

Remaining:

- in-app Health page;
- benchmark and golden-result fixtures.

### Phase 2 implementation

- Durable job ledger with typed states.
- Bounded retries by error class.
- Restart replay and stale-run marking.
- Scan-child ownership and reaping.
- Verified atomic Away report publishing and archive.
- Truthful freshness and last-attempt/last-success state.
- Runtime heartbeat.
- Cross-machine writer protection/lease.
- Global `OFF` / `AUTO-DESK` / `AUTO-AWAY` control.

Phase 2 is implementation-complete. The two-machine collision, expiry/takeover, heartbeat-under-stall, publish-failure, and shutdown drills in `plan.md` remain required before it is marked live-validated.

### Foundation engines

- Packet A: runtime lifecycle.
- Packet B: pure, side-symmetric SPY market-state engine.
- Packet C: aligned, beta/volatility-normalized relative-strength engine.
- Packet D: candidate registry with provenance, leases, typed transitions, and versioned persistence.

### Correctness/platform fixes already landed

- Strict and counted side parsing.
- Stable digest sampling.
- Universe freshness based on all required files.
- Lazy legacy-Tk import.
- Verified report publishing.
- Truthful Auto profile semantics.

## Live shadow challengers

### SPY pullback engine

- Live bridge runs on cached SPY bars.
- Legacy pause detector remains champion.
- Transition/agreement evidence writes to `spy_state_shadow.jsonl`.

### Greatness Monitor

- Persistent D1 confirmation plans and lifecycle.
- Touch/wick/close/accept/retest/fail/re-arm semantics.
- Restart-safe candidate state.
- Existing D1 alerts remain champion.
- Transition evidence writes to `greatness_shadow.jsonl`.

Neither shadow engine is authorized to affect live ranking or loud alerts. Promotion gates and evidence floors are in `plan.md`, Section 7.

## Operational artifacts

- `heartbeat.json` — normally updated every 30 seconds.
- `spy_state_shadow.jsonl` — SPY champion/challenger transitions.
- `greatness_shadow.jsonl` — D1 confirmation lifecycle transitions.
- run manifests — scan timings, counters, and completion state.
- job ledger — unattended job lifecycle and retry history.

## Immediate next work

1. Preserve and audit the first full live-session artifacts.
2. Add shadow coverage/version/config metadata and a daily audit summarizer.
3. Run the Phase 2 controlled operational drills.
4. Build the Health page.
5. Build benchmark and golden-result fixtures.
6. Perform Phase 3 storage/secrets migration only in a supervised session.
7. Keep the SPY and Greatness engines in shadow until the evidence gates pass.

## Larger remaining work

- Provider repository and staged scanner.
- Point-in-time research correctness repairs.
- Full CandidateRegistry adoption.
- Canonical SPY/RS integration.
- Dedicated Greatness fast lane and complete readiness gates.
- Canonical opportunity/ranking pipeline.
- Command Center, mini-chart radar, and alert ladder.
- Auto/Away snapshot parity.
- Journal/outcome learning loop.
- Controlled new-setup research program.
- Market Prep Qt consolidation, CI, and release engineering.

Do not use this file as a substitute for the detailed dependencies, live tests, and exit criteria in `plan.md`.
