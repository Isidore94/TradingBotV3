# Auto/Away runbook

Document role: **active operator runbook**

Topology: **single main desk**

Last reconciled: **2026-08-15** (packet R1: Away stages picks instead of applying
them, and queues alerts without a sound).

Current live-validation location: `plan.md` P0.3.

The Away profile changes scheduling and presentation; it does not place orders and it
does not use different scanner/scoring logic from Desk mode. The main desk is the only
scan host and publisher. The former mini-PC writer handoff and Desk Link satellite
procedures are retired.

## What Away does with picks and alerts (trader rule 2026-08-14)

Away keeps scanning, keeps building the watchlists and keeps writing the hourly
digest. Two things changed on 2026-08-14:

- **Auto-populate picks stage; they are never adopted while Away.** Until this
  rule, Away wrote picks straight into `longs.txt`/`shorts.txt` on the reasoning
  that nobody was present to approve them. Nobody is present to *prune* them
  either, so a pick applied at 09:00 spent the whole day alerting on a name no
  one had looked at. They now wait, and the flip back to DESK adopts the day's
  picks into M5 Focus together. Packet R2 adds the freshness re-check that
  drops the stale ones on the way through.
- **Alerts queue without a sound.** The Alert Center feed, the history and the
  D1 unread badge all keep filling — the badge is how you see at a glance what
  accrued — but nothing beeps at an empty desk. Sound returns the moment the
  mode is not AWAY, including if the state file becomes unreadable.

## What Away publishes

While `AUTO-AWAY` is active, `autopilot_today.txt` is published to the shared home at
each clock hour from 07:00 local through market close. Starting late catches up the
current hour; completed scans and important events may add updates.

The verified file order is fixed:

1. safety and freshness header;
2. numbered `BEST SWING TRADES` when readable setups exist;
3. intraday candidates;
4. condensed `== OPERATIONS ==` detail.

Publication is transactional: the report and verification metadata are written,
read back, and accepted together. A failed attempt preserves the last verified pair.
The latest build may send an ntfy notification for readable best-swing content; an
empty or unparseable swing section stays quiet. Since 2026-08-11 that push also
carries the **full favorite and high-conviction roster** (side-split, `near`
excluded, built from the whole feed rather than the ranked picks), and a **second
hourly push** names every stock that fired a D1 level or event alert since the
previous one — new-since-last-push, silent on an empty hour. Both are **AWAY-only**:
DESK, EVENING, and OFF push nothing here. There are exactly two exceptions to the
AWAY-only rule: the Research/Focus price alerts, which fire from every mode at
`urgent`, and EVENING's SPY ±1% wake alarm (2026-08-14). Machine-local kill
switches: `push_away_swings`, `push_away_d1_events`, `push_evening_spy_alarm`.

## One-time setup

1. Confirm the repo is on the intended validated branch/build.
2. Start TWS/Gateway and enable API access on `127.0.0.1:7496`.
3. Confirm the selected home folder is the intended local location (`C:\TradingBotData`;
   there is no cloud sync — decision 0015), and that the DAS `\\MINI-PC\Trading Bot Data`
   is reachable if you want cold data pushed during the session.
4. Designate this main machine as the writer:

   ```powershell
   .venv\Scripts\python.exe scripts\writer_role.py --designate-self
   ```

5. Configure ntfy under **Research → Price Alerts** and verify **Test Push**.
6. If the research warehouse is in use, confirm `research_store_dir` points outside
   the shared home and the Health page does not report a configuration refusal.

A missing or ambiguous writer role fails closed: no report, metadata, or lease is
modified.

## Before leaving the desk

1. Verify the system clock and time zone. Since 2026-08-14 the clock also decides
   whether automatic work may start at all: quiet hours confine it to weekdays,
   06:00–14:00 local (session open−30m through close+60m). A desk launched
   outside that window starts nothing until it opens — no universe rebuild, no
   IB connect from saved Auto state, no self-arm — and says so in the Auto Pilot
   log: `Auto Pilot is ON from saved state, but nothing starts yet`. That is the
   gate working. Manual buttons still work at any hour.
2. Launch only through:

   ```powershell
   .venv\Scripts\python.exe launch_gui.py
   ```

3. Keep Settings → Desk Link role on **Main**. Do not start a satellite or
   `master_avwap_mini_pc.py` scanner.
4. Confirm TWS/Gateway is connected and market data is flowing.
5. Review System Health for runtime, writer, provider, Away-report, shadow, review,
   and warehouse rows. An honest UNKNOWN is not a green check.
6. Switch Auto mode to **AWAY** and confirm the global header changes.
7. Open the current `autopilot_today.txt` and verify:
   - the session/date is current;
   - `complete-through` and source/freshness are plausible;
   - the safety header is first;
   - swings lead candidate content;
   - the operations tail does not report an unexplained active failure.
8. Confirm phone notifications are permitted for ntfy, including the Focus mode used
   while away.

## During the session

Use the phone digest as a summary, not as proof that every subsystem is healthy. The
report must say when a scan has not run, data is stale, a provider fell back, or the
last attempt failed while an older verified report remains.

Do not manually edit the report or its metadata. Do not start a second GUI to recover
from a delay. The scheduled launch task and single-instance guard own recovery.

## If the report stops updating

Check in this order:

1. Is the main desk awake and the GUI process running?
2. Is Auto mode still AWAY rather than OFF/DESK/EVENING?
3. Is TWS/Gateway connected?
4. Does System Health name a writer-role, provider, scan, or publish failure?
5. Does `autopilot_today.txt` still contain a valid last-verified header? Preserve it.
6. Inspect `%LOCALAPPDATA%\TradingBotV3\diagnostics\`:
   - `heartbeat.json`;
   - `job_ledger.jsonl`;
   - `run_manifests\`;
   - shadow and audit artifacts.
7. Inspect the rotating log in the selected home folder.

If a publish failed, fix the named cause and allow the next normal attempt to replace
the report. Never delete the last verified report to force a refresh.

If the GUI is not running, let the registered repeating launch task recover it. A
manual restart is acceptable only after confirming no process is active; the guard
must refuse a duplicate.

## End of day

1. Confirm the final report is current and verified.
2. Review failed/stale jobs, provider fallbacks, shadow coverage, review capture, and
   warehouse gaps.
3. Switch to DESK or OFF as intended.
4. Preserve artifacts required by `FIRST_SESSION_CHECKLIST.md` when this is a
   validation day.
5. Do not relabel an implemented feature `LIVE_VALIDATED` until its checklist row has
   real-session evidence.

## Retired procedures

The following older procedures are intentionally gone from this runbook:

- choosing a writer separately on a desktop and mini-PC;
- live/away day role reversal;
- two-machine shared-folder takeover and clock-skew drills;
- satellite relay, toast, control-lease, or edit-intent checks;
- shutting down a mini-PC after its scan window.

They remain historical context in `MULTI_MACHINE_DESK_PROPOSAL.md` and Git history,
not current operations.
