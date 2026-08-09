# Durability & Catch-up Plan

Status: ACCEPTED into plan.md sec 12 as item 13c (trader-directed,
2026-08-08). **Steps 1-4 of sec 5 IMPLEMENTED + GREEN on branch
`durability-catchup` (2026-08-08); step 5 (preview lane) deliberately not
built.** Not yet `LIVE_VALIDATED`: the mid-session restart drill in the sec 5
exit gate still needs a live session. Subordinate to plan.md — never overrides
secs 5-7 or the sec 12 order. Motivating evidence: the week of 2026-08-03 —
desk never ran Monday, mid-session outage Thursday, late start Friday — cost
3 of 4 regime-collection sessions their HEALTHY audit and left Master AVWAP
setups stale for a full day (see sec 2.1).

## 1. Design principle: three tiers of durability

Every gap falls into exactly one tier. The tiers keep recovery honest — the
plan.md invariants (completed bars only; missing data is uncertainty, never
confirmation) and the Regime Phase 1 runbook's no-reconstruction rule are the
boundary lines, not obstacles.

- **Tier A — don't go down (process uptime).** The cheapest evidence is the
  evidence you never lost. Self-healing launch, watchdog visibility.
- **Tier B — deterministic backfill with provenance.** Anything that is a
  pure function of completed bars may be recomputed after an outage, but
  every backfilled row carries `capture_mode: "backfill"` (live rows carry
  `capture_mode: "live"` from the same change) so research can always
  separate them. Backfill appends; it never rewrites or deletes a live row.
- **Tier C — never reconstruct.** Anything whose value is *what the process
  observed at a moment in time* stays missed when missed: frozen intraday
  snapshots, opening-range baselines, and live predictions for level tests
  that never started. `missed_snapshot` markers remain the correct output.
  (The point-in-time replay engine in `analyze_technical_integrity.py`
  already serves the research need for "what would it have said" — the live
  ledger must not blur into it.)

## 2. The gaps, by engine

### 2.1 Master AVWAP setup tracker staleness (the reported defect)

`should_update_setup_tracker_now` (`master_avwap_lib/legacy.py`) gates
tracker + favorite-zone watchlist refresh to the final market hour onward, so
intraday scans cannot rewrite stored lists — deliberate and correct. But the
gate checks **wall-clock only**, not data recency: if the desk misses the
after-close window entirely (2026-08-03), every scan the next day is blocked
from refreshing until 15:00, and the desk trades on stale setups all session.

**Fix — the staleness override:** allow a tracker/watchlist refresh outside
the window when `tracker_last_update_session < last_completed_session`, using
**only completed D1 bars** (yesterday's session and earlier). This preserves
the gate's true purpose — no intraday rewriting from forming bars — while
removing the "missed one close, stale all day" failure. The catch-up refresh
is labeled in the scan summary and run manifest (`tracker_catchup: true`).
Golden-fixture note: this changes *when* the tracker updates, never *what* a
given dataset produces — same scoring path, same data vintage discipline.
A characterization test pins that: catch-up refresh from session N-1 bars ==
the after-close refresh from session N-1 bars, byte-identical tracker.

**Amended 2026-08-08 (checkpoint review second review — retraction).** The
"timing-only" claim above was true of the *replay* and false of the *call*.
Two defects, both now repaired on this branch:

1. **Scoring side effects.** `backfill_setup_tracker_from_recent_sessions`
   ends by running `run_priority_scoring_tuner(apply_changes=True)` and
   `calibrate_expected_r_prior_anchors(persist=True)`. Both rewrite **live**
   scoring inputs, so an unattended recovery firing minutes before a scan
   retuned the model on its way past — which is a behaviour change, not a
   timing change, and the characterization test could not see it because it
   mocked both calls into silence. The function now takes
   `run_scoring_side_effects`. The **manual GUI backfill keeps them** (the
   trader asked for the rebuild and is watching the result); the **automatic
   catch-up passes `False`**. The tests no longer mock the two calls away —
   the stubs record, and the tests assert which path invoked them.
2. **Data vintage.** `get_setup_tracker_last_update_session` read the session
   off `updated_at`, which is the wall clock at write time. That is only the
   data vintage while every write happens after its own session's close — the
   exact assumption catch-up breaks. A Friday-morning catch-up rebuilt
   Thursday's tracker and then reported *Friday* as reflected, suppressing the
   genuine Friday refresh for the whole next session. `save_setup_tracker_payload`
   now takes an explicit **`data_session`** — the completed session whose bars
   produced the payload — and `update_setup_tracker_from_scan` passes the scan
   date it evaluated, so both paths stamp the truth. `updated_at` remains the
   fallback for legacy payloads written before the field existed.

So the corrected claim is: the catch-up **replay** is timing-only and remains
byte-pinned; the **call** used to carry global refits, and no longer does on
the automatic path. Provenance is recorded, not inferred from a clock.

**Optional preview lane (flagged, default off):** intraday scans may surface
new candidates to a clearly-labeled preview surface (UI only, never written
to tracker/watchlist files, never alerting) so fresh setups are visible
before the final hour. A forming bar is preview — this lane inherits that
vocabulary. Ships only if the trader wants it after the staleness fix lands.

### 2.2 Process uptime (Tier A — the biggest lever)

The 07:00 scheduled task fires **once**; a crash at 11:00 stays down until a
human notices (2026-08-06). `launch_gui_auto.ps1` already has a correct
single-instance guard, so relaunch is idempotent.

- `register_0700_autostart.ps1`: add task **repetition every 15 minutes**
  until ~16:45 ET on weekdays. Result: any crash or missed boot self-heals
  within 15 minutes, all session long. One script change; re-run once to
  re-register.
- Heartbeat surfacing: `heartbeat.json` already exists — System Health shows
  staleness, and the ntfy ops digest (Local AI plan Phase 6.2) later makes it
  a same-day phone alert. No new watchdog process: the scheduled task **is**
  the watchdog; one owner per timer stays true.

### 2.3 Technical Integrity follow-up chains (Tier B)

Incomplete +30/60/90 chains (42 on 8/4, 1106 on 8/6, 691 on 8/7) are pure
functions of completed M5 bars — exactly what Tier B permits.

- **Close-of-day chain sweeper**: after the close (and on next-day startup if
  the close was missed), complete any `post_resolution_tracking_started`
  chain whose horizon windows fall within available completed bars; emit the
  follow-up rows with `capture_mode: "backfill"`. Where bars are genuinely
  unavailable, emit the existing explicit `data_gap` row instead — the audit
  already knows how to count those honestly.
- Pending-test resume across mid-session restart already persists state
  (`technical_integrity_state.json` pending map); add a characterization test
  that a restart between touch and resolution resolves identically to an
  uninterrupted run.
- `regime_collection_audit.py` learns to report backfilled chains as a
  separate count — a HEALTHY-with-backfill session is distinguishable from a
  fully-live one, and the 40-session floor counts only sessions per whatever
  standard the promotion study declares (that declaration stays in the
  study, not here).

**Amended 2026-08-08 (second review).** Three corrections to what shipped:

- **Bounded retry before the gap is permanent.** The sweep wrote its
  `data_gap` rows *and* its per-session sweep marker in one pass, after a
  **single** failed historical request — and the marker is what stops the
  session from ever being swept again. One pacing violation or momentary
  disconnect therefore cost that session's evidence for good. The fetch now
  retries a bounded number of times first (`scripts/durability_retry.py`;
  default 2 retries, short backoff, every attempt logged), and a persistently
  *empty* response counts as a failure worth retrying — on a single attempt it
  is indistinguishable from "this data does not exist". What is written after
  exhaustion is unchanged, and the gap reason now states how many attempts
  were made. A whole-sweep sleep budget caps the total backoff across symbols,
  because that loop holds the monitor lock. The same helper wraps the 2.4
  breadth fetch, which had the identical single-attempt defect.
- **Point-in-time `as_of` on an empty window.** A follow-up row for a window
  with no bars was stamped `as_of = resolution_bar_close` — up to 90 minutes
  before the absence could have been known. It is now the end of the window:
  the horizon target, clamped to the close for a horizon truncated by it,
  which is exactly the moment the monitor waits for before writing the row.
- **Gaps and outcome coverage reported on their own.** A session can be
  HEALTHY — every chain closed, every gap explicitly marked — and still hand
  the promotion study almost no usable outcomes, because a `data_gap` row
  satisfies the completeness check while carrying no displacement/MFE/MAE. The
  audit now prints a dedicated follow-up data-gap line (count, by horizon,
  reasons) and an **outcome coverage** figure: matured windows that actually
  carry metrics over matured windows expected. Reporting only — the HEALTHY
  verdict logic is deliberately untouched.

### 2.4 Breadth ledger (Tier B)

`vold_m5.jsonl` gaps (41/78 on 8/6, 73/78+3 gaps on 8/7): the recorder
already qualifies contracts via historical M5 fetch. On startup and at the
close, fetch the session's missing completed M5 bars for the qualified proxy
and append them with `capture_mode: "backfill"`, keeping unique bar-end
timestamps and actual contract provenance. Unfetchable bars keep explicit
`data_gap` rows. IB pacing: backfill uses the normal historical-data path
inside the existing single-desk budget; it is a handful of requests, not a
scan.

**Amended 2026-08-08 (second review).** As with 2.3, one failed request used
to finalise this session's permanent gap rows and its backfill marker
together. The fetch now goes through the shared bounded retry
(`scripts/durability_retry.py`) first; pacing cost rises from one request to
at most three on a failing session, and what is written after exhaustion is
unchanged.

### 2.5 Frozen snapshots and baselines (Tier C — unchanged on purpose)

`frozen_intraday_snapshot`, `opening_range_baseline`, and live predictions
for tests that never started are **not** recovered, ever. Their whole value
is what the live hierarchy said at that wall-clock moment; the runbook's
`missed_snapshot` rule stands. Tier A (sec 2.2) is how these get more
durable: the process is simply up at 10:30 far more often.

### 2.6 Scan slots (no change)

The skip-don't-pile-up slot policy stays: a missed hourly scan slot is
skipped, not replayed late — a scan's value decays in minutes, and pile-up
after recovery is worse than the gap. The staleness override (2.1) covers
the only case where a missed scan has next-day consequences.

## 3. Config keys (all `local_settings.json`, all default-on except noted)

| Key | Default | Meaning |
|---|---|---|
| `tracker_staleness_catchup` | `true` | 2.1 staleness override |
| `tracker_intraday_preview` | `false` | 2.1 preview lane (ships later, if wanted) |
| `ti_chain_backfill` | `true` | 2.3 close-of-day + startup chain sweeper |
| `breadth_backfill` | `true` | 2.4 breadth bar backfill |

Defaults are on because these are recovery paths with explicit provenance,
not behavior changes; the preview lane is the only opt-in.

As built (2026-08-08), all three shipped keys read through
`get_local_setting` and are honoured at the entry point of their own recovery
path, so setting any of them to `false` in `local_settings.json` restores the
pre-packet behaviour exactly. `tracker_intraday_preview` has no code yet
because step 5 was not built.

## 4. Invariant compliance

| Invariant | Compliance |
|---|---|
| Completed bars only; forming bar is preview | All backfill uses completed bars; preview lane is labeled preview and writes nothing |
| Missing data is uncertainty, never confirmation | Unfetchable bars stay `data_gap` — now only after bounded retry, so a transient hiccup is not recorded as permanent absence; Tier C stays missed; backfill is marked, never silent |
| No detector/scoring change without golden fixtures | The catch-up *replay* changes timing only and is pinned byte-identical for an identical data vintage; the automatic catch-up no longer runs the scoring tuner or Expected-R prior refit at all (see 2.1 amendment) |
| One owner per timer/thread/job | The scheduled task is the only relauncher; sweepers run inside the existing engines' own lifecycles |
| Failed publish never destroys last verified | Backfill appends; never rewrites live rows |
| Point-in-time research | `capture_mode` provenance keeps live vs backfill separable forever |
| Watchlist names never auto-removed | Catch-up refresh goes through the same CandidateRegistry path as the after-close refresh |

## 5. Build order (one branch, `durability-catchup`, small commits)

1. **Task repetition** (2.2) — script change + re-register; biggest
   evidence-per-effort win, zero invariant surface. **DONE** —
   `scripts/register_0700_autostart.ps1` (weekly trigger + grafted
   `.Repetition`, 15 min / 10 h). Operator step: re-run the script once to
   re-register; it was not registered on this desk at all as of 2026-08-08.
2. **Staleness override** (2.1) — the reported defect; characterization test
   first, then the gate change. **DONE** — `compute_setup_tracker_catchup_plan`
   + `_maybe_run_setup_tracker_catchup` invoking the existing
   `backfill_setup_tracker_from_recent_sessions(..., end_date=)`;
   `tests/test_tracker_staleness_catchup.py`.
3. **Chain sweeper + audit column** (2.3). **DONE** —
   `TechnicalIntegrityMonitor.sweep_incomplete_followups`, driven from the
   existing technical-evidence clock; `capture_mode` on every follow-up row;
   live/backfilled counts in `regime_collection_audit.py`;
   `tests/test_ti_chain_backfill.py`. The restart characterization test also
   uncovered and fixed a real defect: a pending level test recovered from the
   ledger inherited the *started* row's `as_of`, so a restart between touch and
   resolution stamped the resolution with the touch time.
4. **Breadth backfill** (2.4). **DONE** —
   `VoldSessionRecorder.backfill_session_bars` through the qualified-contract
   historical path, driven from the recorder's own thread;
   `tests/test_breadth_backfill.py`.
5. **Preview lane** (2.1, flagged off) — only on trader request. **NOT BUILT.**

Exit gate per step: full suite green + smoke 7/7; for 2.3/2.4 additionally a
session where `regime_collection_audit.py` reports HEALTHY with a nonzero
backfill count after a deliberate mid-session restart drill. Steps 1-4 met the
suite/smoke half on 2026-08-08 (**1876 passed, 5 subtests; smoke 7/7**); the
restart drill is the outstanding live half. Relationship to
item 13b (Local AI): independent code paths, but this packet should land
**before or alongside** 13b Phases 1+ — the digest ledger and the 40-session
evidence floor are both downstream of uptime.

## 6. Implementation notes (binding — verified touchpoints)

- **Reuse mandate (2.1):** `backfill_setup_tracker_from_recent_sessions(lookback_sessions)`
  already exists (`master_avwap_lib/legacy.py` ~22837) and does exactly the
  catch-up refresh — completed prior sessions, earnings context, D1 frames,
  CandidateRegistry-safe watchlist handling. Today it is reachable only from
  a manual GUI action (`master_avwap_lib/gui.py:3246`). The staleness
  override **invokes this existing function** (lookback = sessions since the
  tracker's last update, small cap) — do not build a parallel path.
  Cautions: it opens its own IB client (`client_id=1004`) — the auto-invoke
  must not run concurrently with a live scan (use the existing job/ledger
  serialization), and it must stay inside the single-desk IB budget.
- **Staleness detection (2.1):** tracker payload's `updated_at`
  (`load_setup_tracker_payload`, legacy.py ~4398/4420) vs
  `get_recent_market_session_dates(1)`. Gate location:
  `should_update_setup_tracker_now` (legacy.py ~1759) and its callsite in
  `master_avwap_lib/runner.py` (~1929, `setup_tracker_allowed`).
- **Task repetition (2.2):** weekly triggers don't accept repetition kwargs
  directly in Windows PowerShell 5.1 — build a `-Once` trigger with
  `-RepetitionInterval (New-TimeSpan -Minutes 15)`
  `-RepetitionDuration (New-TimeSpan -Hours 10)` and copy its `.Repetition`
  onto the weekly trigger before `Register-ScheduledTask`. The existing
  5-minute `ExecutionTimeLimit` is fine (the launcher exits after spawning);
  the single-instance guard in `launch_gui_auto.ps1` stays the idempotency
  mechanism.
- **Chain sweeper (2.3):** monitor internals live in
  `scripts/technical_integrity.py` (`_resolve_pending`, persisted pending
  map in `technical_integrity_state.json`); audit changes go in
  `scripts/regime_collection_audit.py`. `capture_mode` is an **additive**
  schema field — existing consumers must treat its absence as `"live"`.
  As built, the vocabulary (`CAPTURE_MODE_LIVE`, `CAPTURE_MODE_BACKFILL`,
  `row_capture_mode`) lives in `scripts/diagnostics/artifact_io.py` beside the
  other shared evidence-writing primitives, because two ledgers now use it;
  `technical_integrity` re-exports the names.
- **Breadth backfill (2.4):** `scripts/vold_recorder.py` already qualifies
  the proxy contract via historical M5 fetch; reuse that path for gap fill.
- Tests: `tests/test_tracker_staleness_catchup.py`,
  `tests/test_ti_chain_backfill.py`, `tests/test_breadth_backfill.py`, plus
  the 2.1 characterization test proving catch-up output ≡ after-close output
  for the same data vintage.
