# Auto-mode matrix and quiet hours — packet R1

Status: **ACTIVE specification** for `plan.md` Phase 0.5 **R1**. Authorized by the
trader on 2026-08-15 as the promotion of the 2026-08-14 wishlist entries; ranked
**first** in the Phase 0.5 build order (trader answer, 2026-08-15).

Build gates: no code from this spec lands before `plan.md` P0.7 (testing-week merge)
completes. Every detector/scoring/alert-hosting file named here falls under the
file-scoped ask-first rule at edit time. Current-state facts below were verified by
read-only recon on 2026-08-15; re-verify line numbers before editing.

## 1. The trader's mode semantics (2026-08-14, restated as a matrix)

Price alerts (Research/Focus `PriceAlertService`) remain the standing always-on push
exception in **every** mode, unchanged.

| Behavior | OFF | DESK | AWAY | EVENING |
|---|---|---|---|---|
| Scheduled Master AVWAP swing slots | no | yes (in-window) | yes (in-window) | **early slot + strength checks only, then none** |
| Open watchlist self-build / BounceBot sweep | no | yes (in-window) | yes (in-window) | early morning only, then quiet |
| Live alerts served in the GUI (toast/sound emphasis) | — | yes | **no — queue silently in Alert Center for return** | queue |
| Auto picks adopted into M5 Focus | no | yes (via staging + adoption gate) | **never** | **staged only; adoption happens on the flip to DESK** |
| Swing push + D1 events push (phone) | no | no | yes (existing policy) | no |
| SPY ±1% wake alarm (phone, every 5 min) | no | no | no | **yes — new** |
| Hourly `autopilot_today.txt` digest write | no | — (existing behavior) | yes | yes |

Quiet hours: **automatic** work runs only 06:30–14:00 PT (09:30 ET open through one
hour after the 13:00 PT close), weekdays. An app boot outside that window starts
nothing — no universe rebuild, no ATR/bar fetching, no BounceBot IB connect from
saved Auto state, no auto-arm — until either the window opens or the trader acts
manually. **Manual scans always work, at any hour.**

EVENING clarification (trader, 2026-08-14): EVENING is armed after the trading day
when the trader works a late shift and sleeps in. Its job is (a) have the day's
trades ready on waking and (b) wake the trader if the market moves. The existing
early open+30 slot, 07:00/07:15/07:30 strength checks, and morning briefing are the
"get the day ready" half and stay. Everything after them stops.

## 2. What the code does today (recon 2026-08-15)

- Mode lives in `AutopilotService` (`scripts/ui/services/autopilot_service.py:116`,
  doc block 83–107). The doc block's stated semantics predate the trader's new rules.
- **AWAY currently does the opposite of the new rule**: `stage_only = mode in
  ("DESK", "EVENING")` (`scripts/bounce_bot_lib/legacy.py:10967`), so AWAY calls
  `apply_auto_populated_watchlists` (`scripts/autopilot_core.py:1944`) which writes
  straight into `longs.txt`/`shorts.txt` — the exact files BounceBot's live sweep
  alerts on in the GUI.
- **EVENING currently scans all day**: `_maybe_run_swing_slot`
  (`autopilot_service.py:859-892`) and `_maybe_build_watchlists` (628–733) run
  unconditionally for any enabled profile; the EVENING early slot is additive.
- Staged DESK/EVENING picks are auto-adopted into M5 Focus by
  `AlertCenterPanel._poll_auto_pick_pending` → `_adopt_auto_pick_into_focus`
  (`scripts/ui/panels/alert_center_panel.py:1543-1657`) with no re-validation
  (packet R2 owns the adoption gate).
- **Boot work ignores the clock**: `MainWindow` schedules `_self_heal_universe` at
  +2.5 s and every 30 min (`scripts/ui/app.py:219-223`, 590–611) → yfinance sweep of
  the whole universe at any hour; `AutopilotService.__init__` (126–190) reconnects
  BounceBot to IB at boot whenever Auto was left ON; `autopilot_auto_arm_due`
  (`autopilot_core.py:292-307`) has no upper-hour bound, so a 21:00 weekday boot
  self-arms Auto Pilot.
- The proven gate pattern to replicate is `bouncebot_scan_window` /
  `bouncebot_scanning_due` (`autopilot_core.py:199-255`) with its edge-triggered
  consumer `_apply_scan_window` (`autopilot_service.py:547-579`): pure window
  function + reason string, weekend check, settings override, **fail-open** on a
  broken session lookup, manual resume survives ticks. Tests:
  `tests/test_bouncebot_scan_window.py`.
- **"Shared scan" is a literal no-op**: `use_shared_watchlists=True/False` resolve
  to the identical `(LONGS_FILE, SHORTS_FILE)` paths
  (`scripts/project_paths.py:573-574`; `scripts/master_avwap_lib/legacy.py:1992-2022`).
  The dead flag threads through ~13 files (`ui/app.py:467-469`,
  `ui/panels/master_avwap_panel.py` menu/scheduler strings,
  `ui/services/scan_service.py:119-167` + config hashes `shared-v1`/`local-v1`,
  `scan_worker.py:53,73,79`, `master_avwap_lib/runner.py:113-114,435-450,2590-2624`
  incl. a run-manifest counter at 2606, `master_avwap_lib/legacy.py` warm/backfill
  entries, `master_avwap_mini_pc.py` (already-retired file), and
  `tests/test_public_entrypoints.py:86`). Separately ~7 stale "shared Drive" strings
  remain (`writer_lease.py:1`, `ui/panels/autopilot_panel.py:35`,
  `autopilot_service.py:495,697,1380`, comments in `project_paths.py:514`,
  `master_avwap_lib/runner.py:1876`, `master_avwap_lib/legacy.py:4544`), and
  `project_paths.py:163-289` still runs a Google-Drive mount probe/wait
  **unconditionally at module import** — decision 0015 blessed only the macOS
  CloudStorage branch as intentionally-kept-dead, not this Windows path.
- SPY day-change source for the wake alarm already exists:
  `BounceBot._spy_session_bars(cached_only=True)` and the
  `day_pct = (close - prev_close) / prev_close * 100` formula
  (`scripts/bounce_bot_lib/legacy.py:5136-5184`); `AutopilotService` already reaches
  the live bot this way for the near-HOD check (`autopilot_service.py:982-991`).
- ntfy senders and gates today: `_push_swing_picks` (AWAY-gated, 1395),
  `_maybe_push_d1_events` (AWAY-gated, 1673), `PriceAlertService._notify` (the
  documented exception), `push_notify.send_push` (the one physical sender).

## 3. Design

### 3.1 One quiet-hours gate

Add `auto_scanning_window(reference)` and `auto_scanning_due(now) -> (bool, reason)`
in `scripts/autopilot_core.py`, mirroring `bouncebot_scanning_due` exactly: window =
session open → close + 60 min (06:30–14:00 PT on a normal day), weekend refusal,
margin/override settings, **fail-open** on session-lookup failure. Wire it into:

1. `MainWindow._self_heal_universe` — skip the rebuild outside the window (the
   30-min timer keeps ticking; the check is cheap).
2. `AutopilotService.__init__`'s resume-from-saved-state — outside the window, do
   not connect BounceBot; the tick loop connects when the window opens.
3. `autopilot_auto_arm_due` — add the upper bound so a late boot does not self-arm.
4. `_maybe_run_swing_slot` / `_maybe_build_watchlists` — refuse slots outside the
   window (today slots implicitly stay in-session; the gate makes it explicit).

Manual carve-out: every trader-initiated button (manual scan, manual BounceBot
resume, manual universe rebuild) bypasses the gate, matching how manual master-scan
buttons already bypass the AutoPilot scheduler.

### 3.2 EVENING stops scanning after the early block

In `_maybe_run_swing_slot` and `_maybe_build_watchlists`: when
`self._profile == AUTO_PROFILE_EVENING`, only the EVENING early slot (open+30) and
the `EVENING_STRENGTH_CHECK_SLOTS` work run; ordinary hourly slots and the open
self-build are refused with a logged reason. `_maybe_run_evening_prep` is unchanged.

### 3.3 AWAY: queue, never adopt, never emphasize

- AWAY keeps scanning and keeps writing watchlists (the away report, outcome
  evidence, and the alert stream depend on it), but:
  - the Alert Center auto-adopt poll (`_poll_auto_pick_pending`) refuses adoption
    while `auto_mode == AWAY`; staged picks stay pending;
  - live alert presentation is queued-quietly: alerts append to the Alert Center
    feed/history as today but suppress attention-demanding presentation while AWAY
    (exact widget wiring decided at build time with the ask-first review);
  - on the flip AWAY→DESK, the pending queue drains through the **R2 adoption
    gate**, so stale picks are re-validated (below yday HOD / below VWAP → dropped),
    not blindly adopted.
- EVENING staged picks behave the same: staged, never auto-adopted; they adopt on
  the wake-up flip to DESK through the same R2 re-check. This supersedes the
  2026-08-05 "adopt immediately in DESK/EVENING" rule for **EVENING only**; DESK
  keeps immediate adoption.

### 3.4 EVENING SPY ±1% wake alarm

New pure function `spy_move_alarm_due(day_pct, last_sent_at, now)` in
`autopilot_core.py` + `_maybe_push_spy_alarm(now)` in the tick loop:

- gated `auto_mode == EVENING`; active from session open; checked each tick;
- fires when `abs(day_pct) >= 1.0` (threshold settings-overridable) and at least
  300 s since the last send; repeats every 5 min while the condition holds, until
  the trader flips out of EVENING;
- `day_pct` from `bot._spy_session_bars(cached_only=True)` (champion data path, no
  shadow engine involvement); missing bars = no alarm (missing data is uncertainty);
- sends via `push_notify.send_push(..., priority="urgent")`; last-sent timestamp
  day-rolls in `AUTOPILOT_STATE_FILE` so a mid-EVENING restart does not double-fire;
- machine-local kill switch `push_evening_spy_alarm` (default on), matching the
  `push_away_swings` pattern.

This is the **second deliberate exception** to the AWAY-only push rule. When built,
update the CLAUDE.md/AGENTS.md phone-push-policy paragraph, `plan.md` P0.3's push
wording, and `docs/EVENING_MODE_RUNBOOK.md` in the same commit.

### 3.5 Shared-scan removal

- Collapse `run_shared_watchlist_scan`/`run_local_watchlist_scan` into one
  `run_watchlist_scan`; drop the `use_shared_watchlists` parameter everywhere it
  threads (both branches are provably identical, so this is a no-op removal, not a
  behavior change — assert with the existing public-entrypoint tests).
- One "Run Scan" menu action replaces the Shared/Local pair; rename the
  "shared-watchlist scheduler" strings.
- Fix the ~7 stale "shared Drive" strings (trader-visible panel text first).
- Remove the Windows Google-Drive mount probe/wait in `project_paths.py:163-289`
  (import-time side effect, unblessed by decision 0015); keep the macOS
  CloudStorage branch exactly as decision 0015 blessed it; add an amendment note to
  `docs/decisions/0015-no-cloud-sync-das-file-server-storage.md`.
- Before deleting the `use_shared_watchlists` run-manifest counter
  (`runner.py:2606`), grep manifest consumers; if any audit reads it, retire the
  counter in the same change.
- `scripts/master_avwap_mini_pc.py` is NOT this packet's scope — it stays until the
  P1.5 retired-topology cleanup.

## 4. Invariants and fenced files

- Fail-open on session-lookup failure everywhere: a broken clock must never sit out
  a trading day (matches `bouncebot_scanning_due`).
- One owner per timer: the SPY alarm and quiet-hours checks live inside the existing
  `AutopilotService._tick()`; no new timers.
- Champion paths untouched; the alarm reads the champion SPY bars only.
- Ask-first at edit time: `scripts/bounce_bot_lib/legacy.py`,
  `scripts/autopilot_core.py`, `scripts/ui/services/autopilot_service.py`,
  `scripts/ui/panels/alert_center_panel.py`, `scripts/ui/services/scan_service.py`,
  `scripts/master_avwap_lib/legacy.py`, `scripts/master_avwap_lib/runner.py`.

## 5. Tests

- Pure-window + edge-trigger + fail-open tests for `auto_scanning_due`, modeled on
  `tests/test_bouncebot_scan_window.py`.
- Mode truth-table extensions in `tests/test_auto_mode_semantics.py`: AWAY never
  adopts, EVENING never runs a post-early slot, OFF/quiet-hours boot starts nothing.
- SPY alarm: threshold, 5-min cadence, day-roll, EVENING-only gate, missing-bars
  refusal, kill switch (pattern: `tests/test_away_push_gating.py`).
- Shared-scan removal: entrypoint parity (same paths scanned before/after), no
  remaining references.

## 6. Exit gate

Deterministic tests green; then one live proof each: a boot at ~21:00 that provably
starts nothing (log evidence), an EVENING day whose log shows the early block and
then zero further slots, an AWAY session with picks staged-not-adopted and a clean
drain through the R2 gate on return, and one real SPY-alarm firing (or a forced
threshold test push). CLAUDE.md/AGENTS.md, `docs/AWAY_SCANNER_RUNBOOK.md`, and
`docs/EVENING_MODE_RUNBOOK.md` reconciled in the same packet.

## 7. Open questions

- AWAY "queue quietly": which exact presentation channels count as
  attention-demanding (sound, toast, tab badge)? Decide with the trader at build
  time; the queue/history behavior itself is settled above.
- Should the SPY alarm also cover a fast intraday reversal (e.g. crosses back
  through ±1%)? Not in v1; revisit after the first live EVENING week.
