# Away scanner runbook

The Away profile changes presentation and unattended scheduling only. It does
not place orders, and it uses the same scanner/scoring decisions as Desk mode.
While `AUTO-AWAY` is active, the Google Drive report publishes once each clock
hour from 07:00 local time through market close. Starting the app later catches
up the current hour; completed scans and important events can add extra updates.

## One-time setup: name the designated writer on every machine

Publishing shared Drive output is gated on an explicit, machine-local
configuration. **A machine with no designated writer configured publishes
nothing** - it preserves the last verified report and reports the configuration
failure in the GUI, the log, and Health telemetry. There is deliberately no
"first machine wins" fallback, because a Drive-synchronized file cannot arbitrate
that race.

Do this once per computer, then restart the GUI.

1. Get the machine name that the app uses:

   ```powershell
   .\.venv\Scripts\python.exe -c "import socket; print(socket.gethostname())"
   ```

2. Edit `%LOCALAPPDATA%\TradingBotV3\local_settings.json` (create it if it does
   not exist; it is machine-local and is *never* synced through Drive - a synced
   role file would suffer exactly the convergence problem it is meant to solve).

   On the **desktop** (the machine that publishes), with `DESKTOP-NAME` replaced
   by the name printed in step 1 *on the desktop*:

   ```json
   {
     "designated_writer": "DESKTOP-NAME",
     "writer_role": "designated_writer"
   }
   ```

   On the **mini-PC** (read-only secondary) - note that `designated_writer`
   still names the *desktop*, which is how the mini-PC knows it is not the
   writer:

   ```json
   {
     "designated_writer": "DESKTOP-NAME",
     "writer_role": "secondary"
   }
   ```

   Keep the other keys already in the file; add these alongside them.

3. Verify on each machine:

   ```powershell
   .\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'scripts'); from writer_role import resolve_writer_role; r=resolve_writer_role(); print(r.role, '|', r.machine, '|', r.designated_writer)"
   ```

   The desktop must print `designated_writer`; the mini-PC must print
   `secondary`. A mini-PC that prints `designated_writer` is misconfigured and
   would compete with the desktop.

### What each machine needs

| | Desktop (designated writer) | Mini-PC (read-only secondary) |
|---|---|---|
| `designated_writer` | its **own** hostname | the **desktop's** hostname |
| `writer_role` | `designated_writer` | `secondary` |
| Shared Drive folder mounted | yes | yes (reads only) |
| Auto Pilot | may run and publish | may run, alert, and read; publishes nothing |
| System clock | network time, within **120 s** of the other machine | same |

This is what keeps the mini-PC a read-only secondary while the desktop scans:
its Auto Pilot may still run, alert, and read shared state, but every attempt to
publish the shared report *or to rewrite the shared watchlists*
(`longs.txt`, `shorts.txt`, `autolongs.txt`, `autoshorts.txt` - they live in the
same Drive folder) returns a refusal naming its role, and it never takes the
writer lease. A refusal is logged as
`Shared watchlist write refused: <file> is shared Drive state and this machine
may not write it: ...`.

### Clock requirements, and what happens outside them

The lease expiry is computed on the writer's clock and judged on the reader's,
so the two machines must agree about the time. The supported disagreement is
**120 seconds** (`writer_lease.DEFAULT_CLOCK_SKEW_SECONDS`), applied as a grace
on top of the nominal expiry.

- A machine whose clock is **fast** by more than the lease's *remaining* TTL
  plus that 120 s **will take over a lease that is still live** by the holder's
  clock. This is not only a 30-minute-skew problem: 5 minutes is enough late in
  a 10-minute TTL.
- A machine whose clock is **slow** stays locked out for the length of its own
  error beyond the nominal expiry. Bounded, but it can be tens of minutes.
- Lease timestamps carry an explicit UTC field (`acquired_at_utc`,
  `expires_at_utc`) alongside the local wall-clock ones, so the two annual DST
  transitions and a machine in a different timezone no longer inject an hour of
  apparent skew.

Keep both machines on the Windows time service. To check:

```powershell
w32tm /query /status
w32tm /resync
```

Health telemetry records `observed_clock_offset_seconds` whenever a lease
written by the *other* machine is read - the number for the plan.md sec 6.2
clock-comparison drill. Only the "their clock is ahead of ours" direction is
observable from a lease; our own clock running fast looks exactly like an old
lease, which is why this is a reported observation and not a correction.

### The lease is held for the publication, not continuously

There is **no renewal loop**. The lease is taken, the report is published, and
the lease is released on a clean exit; between publications no machine holds it.
The standing authority between publications is the configured designated writer
(Layer 1), not the lease. `writer_lease.renew()` exists for a caller that wants
to extend an in-progress hold; it keeps the fencing generation it already has,
so a renewal cannot fence off the renewing writer's own publish.

### Emergency takeover (rare, and never a default)

Use this when the desktop is dead or unreachable and the mini-PC must publish.

**Preconditions** - all of them, before you touch anything:

1. You have confirmed the desktop is not publishing. Not "probably" - confirmed:
   it is powered off, or its GUI is closed, or you can see its Away report row
   is not updating.
2. You accept that the desktop must not be restarted into Auto Pilot until you
   have undone this, or both machines will be configured as writers.

**Step 1 - see who actually holds the lease** (read-only, safe to run any time):

```powershell
.\.venv\Scripts\python.exe -c "import sys,json; sys.path.insert(0,'scripts'); import writer_lease as wl; from project_paths import AUTOPILOT_REPORT_FILE; from pathlib import Path; p=Path(str(AUTOPILOT_REPORT_FILE)+'.lease'); print('holder:', wl.holder_of(p)); print(json.dumps(wl.inspect_lease(p).get('payload', {}), indent=2))"
```

`holder: None` means the lease is free or expired - no takeover is needed, just
fix the role configuration in step 2 and publish. A holder string
(`HOSTNAME:PID:instance`) means somebody's claim is live until the `expires_at`
shown. An exception means the lease state is unverifiable, which also blocks and
does need the override below.

**Step 2 - configure the mini-PC as a time-bounded writer.** Edit its
`%LOCALAPPDATA%\TradingBotV3\local_settings.json`:

```json
{
  "designated_writer": "MINI-PC-NAME",
  "writer_role": "designated_writer",
  "writer_emergency_takeover": true,
  "writer_emergency_takeover_expires_at": "2026-07-30T18:00:00",
  "writer_emergency_takeover_reason": "desktop offline, publishing from mini-PC"
}
```

Rules the override must satisfy, or it is silently inert (by design):

- the value is an explicit true token (`true`, `"yes"`, `"on"`, `"1"`) **or**
  itself an ISO timestamp;
- an expiry parses as an ISO timestamp (a trailing `Z` or a `+02:00` offset is
  accepted);
- the expiry is **in the future**;
- the expiry is **within 12 hours** (`writer_role.MAX_OVERRIDE_WINDOW_HOURS`).
  A longer window is rejected with a message naming the ceiling: an override
  that lasts a century would re-break the other machine's lease on every publish
  until somebody remembered to remove it. If the emergency outlives the window,
  renew the expiry deliberately.

`""`, `"0"`, `"false"`, `"no"`, `"maybe"`, `"null"`, `[]`, `{}`, a past
timestamp, a true value with no expiry, and `9999-12-31T23:59:59` all evaluate
to *no override*.

**Step 3 - verify the override is live before relying on it:**

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'scripts'); from writer_role import resolve_writer_role, resolve_emergency_override; r=resolve_writer_role(); o=resolve_emergency_override(); print('role:', r.role, '| may_publish:', r.may_publish); print('override active:', o.active, '| expires:', o.expires_at); print('rejected because:', o.rejected_because or '(accepted)')"
```

You need `role: designated_writer`, `may_publish: True`, and
`override active: True`. If `rejected_because` is populated, fix what it names
and re-run - do not proceed.

**Step 4 - publish.** Restart the mini-PC GUI, select `AUTO-AWAY`, click
**Write Report Now**. The override is used only if the normal acquisition path
is actually blocked; a free or expired lease is taken normally and no takeover
is recorded.

**Step 5 - confirm afterwards.** A takeover that fired is recorded in four
places, and a later renewal cannot erase any of them:

```powershell
.\.venv\Scripts\python.exe -c "import sys,json; sys.path.insert(0,'scripts'); import writer_lease as wl; print(json.dumps(wl.writer_health_state(), indent=2))"
Get-Content "$env:USERPROFILE\...\autopilot_today.txt.meta.json"
Get-Content "$env:USERPROFILE\...\autopilot_today.txt.lease.takeover_audit.jsonl" -Tail 5
```

- the lease payload: `takeover: true`, `previous_holder`, `takeover_reason`;
- the publication metadata `autopilot_today.txt.meta.json`: `takeover`,
  `previous_holder`, `holder`, `generation`;
- `autopilot_today.txt.lease.takeover_audit.jsonl` next to the report - one
  append-only line per takeover, with who was displaced, who displaced them,
  when, why, and the fencing generation. If this record cannot be written the
  takeover is **refused** and the live lease is left alone; an unauditable
  takeover does not happen;
- `writer_health.json`: `emergency_override.active` and its expiry.

**Step 6 - stand down.** The override stops working by itself at its expiry -
nothing needs to be done for it to lapse, and after it lapses the mini-PC keeps
publishing only because `writer_role` still names it. So when the desktop is
back:

1. remove the three `writer_emergency_takeover*` keys from the mini-PC's
   `local_settings.json`;
2. set the mini-PC back to `"designated_writer": "DESKTOP-NAME"` and
   `"writer_role": "secondary"`;
3. restart the mini-PC GUI and re-run the step 3 verification - it must now
   print `role: secondary`, `may_publish: False`;
4. start the desktop's Auto Pilot and confirm its Away report row says
   `verified`.

**Never** delete the lease file by hand to "unstick" it. Deleting it while the
other machine may still be publishing removes the only signal that machine has
that somebody else took over, and leaves no audit trail. The override path above
is the supported way, and it records what happened. (The fencing generation's
high-water mark lives in `autopilot_today.txt.lease.generation` and survives a
deleted lease; that file grants nothing and naming a holder is not among its
contents.)

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
  previous verified publication and is shown as a failure in the GUI. So does a
  hard interrupt (`KeyboardInterrupt`, a thread abort) landing between the two
  replaces - the rollback runs and the interrupt is then re-raised, instead of
  leaving the report and its metadata describing different publications.
- If an *earlier* cycle did leave a torn pair - a report whose SHA-256 does not
  match its `.meta.json` - the next publish detects it, logs it, and reports it
  as `previous_pair_disagreement` in the result and in Health telemetry. It is
  not silently overwritten.
- The publication gate runs in a fixed order and every step is a precondition
  for the next: configured writer role -> machine-local cross-process lock ->
  hardened Drive lease and fencing validation -> render/stage -> re-verify
  ownership immediately before replacement -> replace report and metadata ->
  verified readback. Ownership is re-read from disk before the report replace
  *and* again before the metadata replace.
- **Cross-machine writer protection** (not distributed exclusion): the lease
  file names the current writer, its process instance, and a fencing
  generation, so a second machine sees an active writer and degrades honestly.
  A Google Drive-synchronized file is not a compare-and-swap lock, and two
  machines can still race before sync converges - nothing here proves clobbering
  is impossible. **The re-check before each replace narrows the ownership window
  but does not close it**: a writer fenced off in the sub-millisecond gap between
  the check and the `os.replace` still completes that replace. What the design
  does do is make every *ambiguous* state fail closed: unreadable bytes, a
  directory in the way, invalid JSON, an unknown schema, a missing holder or
  expiry, or an old-format `writer_lease_v1` lease with no process instance id
  all block instead of reading as "free". A two-minute clock-skew grace prevents
  slightly fast clocks from taking over at the nominal expiry - see the clock
  section above for what happens outside that bound.
- The fencing generation has a durable high-water mark
  (`autopilot_today.txt.lease.generation`) beside the lease, so it keeps rising
  across a clean release and a restart rather than resetting to 1. That makes
  the `generation` recorded in publication metadata usable for ordering two
  publications after the fact.
- **Real same-machine exclusion**: two processes on the designated PC see the
  same lease bytes with zero sync delay, which a read/replace lease cannot
  arbitrate at all. A Win32 named mutex plus an exclusive byte-range lock on a
  machine-local lock file are both held around lease acquisition *and* the whole
  publication transaction. Both are kernel-owned, so a process killed without
  releasing them frees them automatically - a hard kill can never wedge
  publishing. If *neither* OS primitive is available, acquisition fails closed
  rather than reporting the lock as held on the strength of an in-process lock
  that arbitrates nothing between processes.
- A previous owner's death is reported as
  `local_lock.abandoned_by_previous_owner`. `WAIT_ABANDONED` alone does not
  cover the common case - when the killed process held the only handle to the
  mutex name, the kernel destroys the object and the next waiter sees a clean
  acquisition - so an owner marker file, written under the lock and removed only
  on a clean release, is what makes a lone-process crash visible.
- Ownership is per *process instance*, not per hostname. A second GUI on the
  same PC, and a restarted process that the OS handed the same PID, are never
  treated as the previous writer. A hard-killed writer's lease is recovered only
  by expiring and being re-acquired afresh, never by a later process claiming it
  was already its own.
- A clean GUI exit releases the lease (`AutopilotService.shutdown`, with
  `MainWindow.closeEvent` as a backstop), so the other machine is not locked out
  for the rest of the TTL. Release only ever drops a lease this process instance
  can prove is its own.
- `<diagnostics>\writer_health.json` is the single Health artifact: configured
  designated writer, this machine and its role, hostname/PID/instance id, local
  lock state, lease holder/instance/acquired/renewed/expires, fencing
  generation, last ownership or configuration failure, read-only reason,
  emergency-override state and expiry, any observed clock offset, and the owner
  plus generation of the last verified publication. It is written atomically on
  every publish attempt, and an absent, corrupt, or **stale** artifact reads as
  *unhealthy*, never as green. The last-verified-publication and last-failure
  records are both retained: a refusal does not erase the record of the last
  good publication, and a success does not erase the last failure. `last_renewal_at`
  is populated only by a real renewal, never as an alias for the acquisition time.
- The phone report includes operations health, last scan duration/status, and
  whether a requested setup-tracker update actually occurred or was skipped.
- The operations audit treats missing metadata, stale reports, and hash changes
  as degraded or unhealthy instead of green.

## If the phone report stops updating

1. Treat the displayed picks as stale; the report explicitly says not to trade
   an hours-old update as current.
2. On return, check the Auto Pilot activity log and the red Away report row.
   The refusal text names the cause: a configuration failure ("no designated
   writer is configured..." / "read-only secondary"), another machine holding
   the lease (it names the holder), an unverifiable lease, or another process on
   this machine.
3. Read the Health artifact - it answers most of the above without guessing:

   ```powershell
   .\.venv\Scripts\python.exe -c "import sys,json; sys.path.insert(0,'scripts'); import writer_lease as wl; print(json.dumps(wl.writer_health_state(), indent=2))"
   ```

   `"status": "missing"` or `"corrupt"` means the telemetry itself is gone or
   half-written - treat that as unhealthy, not as "nothing to report".
   `"status": "stale: ..."` means the artifact is more than 150 minutes old:
   this machine stopped reporting, and what it last said is history, not health.
4. Run `scripts\operations_audit.py --no-write --json` and preserve the latest
   run manifest, heartbeat, job ledger, report metadata, lease file, and
   `writer_health.json` before restarting.
5. Restart the GUI. Do not delete or force-take a lease while another machine
   may still be publishing. An intentional takeover must be a controlled
   recovery action - follow the **Emergency takeover** procedure above (see who
   holds it, configure a time-bounded override, verify, publish, confirm the
   audit trail, stand down) rather than deleting the lease by hand, so the
   takeover is auditable.

## Still requires a physical validation drill

Automated tests cover competing holders, expiry, bounded clock skew,
sleep/wake-style reacquisition, render/write/readback failures, and rollback.
Google Drive synchronization cannot be certified by a single-machine test.
Before relying on two computers simultaneously, complete the Section 6 tests in
`plan.md`: near-simultaneous publish, expiry/takeover, clock comparison, and
sleep/wake while observing both machines and the phone-facing Drive copy.

For the sec 6.2 clock-comparison drill you need the numbers above: the supported
skew is **120 s**, timestamps carry an explicit UTC anchor, `w32tm /query /status`
on both boxes is the measurement, and `observed_clock_offset_seconds` in
`writer_health.json` is what the software itself saw. Known limitations to
record rather than to be surprised by: a clock fast beyond the supported window
can take over a live lease; there is no renewal loop, so the lease is held for
the publication rather than continuously; and the pre-replace ownership check
leaves a sub-millisecond residual window that only a real compare-and-swap could
close.
