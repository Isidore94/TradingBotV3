# Trade-discovery learning program — checkpoint

[`GUI_TRADE_DISCOVERY_LEARNING_PLAN.md`](GUI_TRADE_DISCOVERY_LEARNING_PLAN.md)
owns the product design, the phase order, and every promotion gate. This file
is only the checkpoint stamp for that program: what has landed, what the
measured baseline is, and what the next phase needs. It must not duplicate the
plan.

Authority order is unchanged: [`plan.md`](plan.md) outranks the GUI plan, which
outranks this file.

## Current checkpoint

- Phase: **0 — Capture readiness and baseline** (code work complete; the
  live-session tasks below are still open)
- Branch: `main`; audit base `41eabc4`
- Date: 2026-07-28
- Test baseline: **1209 passed, 5 subtests passed**
  (`.venv\Scripts\python.exe -m pytest tests/ -q`)
- Smoke: **7/7** (`.venv\Scripts\python.exe scripts/smoke_check.py`)
- Learning evidence status: **Exploratory / Non-Promotable**. The promotion
  clock has not started and cannot start before the Phase 3 identity/parity
  exit gate.

The suite baseline moved from 1189 passed + 1 failed to 1209 green: the
`_sanitize_existing_avwap_signal_rows` side normalizer assumed `Series.map`
preserved the `Side` enum, which arrow-backed string columns do not. It now
maps straight to the string value. No scoring, detector, or alert behavior
changed.

## Phase 0 — what landed

- **Task 3 — decision-log durability is observable.**
  `scripts/review_capture_audit.py` audits `alert_review_events.jsonl` for
  rows, sessions, actions, schema drift, rows missing a symbol, and malformed
  lines. The runtime reader still skips bad lines so a corrupt row can never
  cost the trader a session; the audit counts every one of them, because a
  ledger that quietly drops rows is exactly the silent failure this phase
  exists to surface.
- **Task 3 — writer attribution.** Review events now record the writing
  machine. The log lives in the shared home, so two machines appending to one
  trade date is the concurrent-writer hazard the roadmap treats as an
  immediate rollback trigger; the audit reports it as unhealthy.
- **Task 4 — System Health shows the learning surfaces.** The operations audit
  composes six capture-readiness checks: decision log, preference scoreboard,
  outcome join, policy/ordering gate, setup-scoring champion, and evidence
  label. Runtime status and capture readiness roll up separately — a
  cold-start ledger reads degraded without making the unattended runtime look
  broken — but a capture check that is outright unhealthy (a gate that stopped
  holding) does raise the operational verdict.
- **Task 6 — preference ordering is gated to annotation-only.** Guidance still
  scores every alert, annotates the chart, and stamps the score onto each
  impression; it contributes zero to queue position, so the active queue is
  the pre-guidance FIFO. The pre-gate ordering is preserved as a
  characterization test and restored by `ordering_mode="preference"` (or
  `TRADINGBOT_REVIEW_QUEUE_ORDERING=preference`) with no code revert. An
  unrecognized value fails closed. Armed chart-watch hits still lead the
  queue: that is a trader instruction, not a preference signal.
- **Task 7 — the scoring champion is snapshotted, not changed.** The audit
  records the active config's SHA-256, its attribute rules split by source,
  and the three sites where the existing `apply_changes=True` tuner runs
  today. Nothing invokes, redirects, or promotes a tuner result; Phase 1 does
  that only after golden characterization exists.
- **Task 8 — the promotability label ships with the evidence.** Every audit
  payload carries `Exploratory / Non-Promotable` plus the four reasons
  (episodes folded by `(trade_date, symbol)`, engagement counted as a take,
  no setup-chart impression denominator, one shared Swing/M5 outcome
  definition). System Health prints it beside the audit timestamp.

Read the current state at any time with:

```powershell
.venv\Scripts\python.exe scripts/review_capture_audit.py
```

## Phase 0 — still open

These are live-session tasks that code cannot close:

1. **Task 2** — restart the GUI onto this build so the decision log starts
   growing. At the 2026-07-28 audit the log did not exist yet.
2. **Task 5** — capture the current screenshots, queue ordering, alert and
   sound counts, Focus size, duplicate rate, missed-winner review, latency,
   and decision time as the champion baseline.
3. **Cold-start collection** — roughly two to three weeks of normal sessions
   before anything is tuned. The audit reports progress against a 10-session
   observability floor; the real promotion floors live in the plan's ranking
   and delivery manifests.

## Next

Phase 1 — runtime foundation, fixtures, and the read-only Focus quick win.
Its first task is the `plan.md` live-session and failure drills, and the
golden/benchmark fixtures for current detector, scoring, queue, alert, and
sound behavior. The tuner may not be rerouted to a shadow artifact before
those fixtures exist.
