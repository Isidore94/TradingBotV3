# Away scanner runbook

The Away profile changes presentation and unattended scheduling only. It does
not place orders, and it uses the same scanner/scoring decisions as Desk mode.
While `AUTO-AWAY` is active, the Google Drive report publishes once each clock
hour from 07:00 local time through market close. Starting the app later catches
up the current hour; completed scans and important events can add extra updates.

## Before leaving the PC

1. Restart the GUI after installing code changes. Run only one intended Auto
   Pilot writer; keep the second computer's Auto Pilot off unless performing a
   controlled writer-collision drill.
2. Confirm IB is connected and the Universe row is fresh on the Auto Pilot
   page. Use **Reconnect IB Now** or **Rebuild Universe Now** if required.
3. Select `AUTO-AWAY`, then click **Write Report Now**. Do not leave until the
   Away report row says `verified` and has no red failure text.
4. Run this read-only check in a terminal:

   ```powershell
   .\.venv\Scripts\python.exe scripts\operations_audit.py --no-write
   ```

   During a live session, investigate any `UNHEALTHY` result. A pre-market
   `DEGRADED` item can be expected when today's work has not started yet, but
   its explanation must match reality.
5. Open the shared `autopilot_today.txt` once from the phone-facing Drive view.
   Confirm its Updated time, runtime machine/PID, Health line, last-scan line,
   tracker-write result, watchlists, and next scheduled slot.

## What the unattended path now protects

- Watchlist files are staged and atomically replaced, so a failed replace keeps
  the prior list instead of leaving a partial file.
- The Away report and its SHA-256 verification metadata publish as one logical
  transaction. A render, write, readback, or metadata failure restores the
  previous verified publication and is shown as a failure in the GUI.
- A cross-machine lease blocks an active second writer and fails closed when
  ownership cannot be checked. A two-minute clock-skew grace prevents slightly
  fast clocks from taking over at the nominal expiry.
- The phone report includes operations health, last scan duration/status, and
  whether a requested setup-tracker update actually occurred or was skipped.
- The operations audit treats missing metadata, stale reports, and hash changes
  as degraded or unhealthy instead of green.

## If the phone report stops updating

1. Treat the displayed picks as stale; the report explicitly says not to trade
   an hours-old update as current.
2. On return, check the Auto Pilot activity log and the red Away report row.
3. Run `scripts\operations_audit.py --no-write --json` and preserve the latest
   run manifest, heartbeat, job ledger, report metadata, and lease file before
   restarting.
4. Restart the GUI. Do not delete or force-take a lease while another machine
   may still be publishing. An intentional takeover must be a controlled
   recovery action.

## Still requires a physical validation drill

Automated tests cover competing holders, expiry, bounded clock skew,
sleep/wake-style reacquisition, render/write/readback failures, and rollback.
Google Drive synchronization cannot be certified by a single-machine test.
Before relying on two computers simultaneously, complete the Section 6 tests in
`plan.md`: near-simultaneous publish, expiry/takeover, clock comparison, and
sleep/wake while observing both machines and the phone-facing Drive copy.
