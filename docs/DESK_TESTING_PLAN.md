# Desk testing plan — what to check, and when

Document role: **active operator runbook.** Last reconciled: **2026-08-15**.

> **This file restates the owed live proofs from `CURRENT_CHECKPOINT.md` for a
> human reader.** It is not a second source of truth. Whenever those proofs
> change — one passes, one is added, one is reworded — this file must be updated
> **in the same pass**, or the desk will be reading yesterday's instructions.

You can open this any time from the desk: **Settings ▸ Testing Plan**.

---

## How to read this

Every step tells you four things:

- **WHEN** — the moment to do it.
- **DO** — what to click or type. Nothing else.
- **GOOD** — the exact line, file, or screen thing that means it worked.
- **BAD** — what failure looks like, and exactly what to copy to the AI.

If something is confusing, it is the document's fault, not yours. Copy the step
and what you saw to the AI and say "this didn't match".

**Two file locations you will need:**

| What | Where |
|---|---|
| Auto Pilot log | `C:\Users\Aaron\AppData\Local\TradingBotV3\logs\autopilot.log` |
| Bot log | `C:\Users\Aaron\AppData\Local\TradingBotV3\logs\trading_bot.log` |
| Your watchlists / reports | `C:\TradingBotData\` |

To read a log: open it in Notepad, press **Ctrl+End** to jump to the newest
lines at the bottom. That is where today's activity is.

---

# 1. Tonight (optional) — the quiet-boot check

**The question this answers:** if you open the desk late at night, does it
correctly do *nothing* until the morning?

Before this was fixed, launching at 9pm woke the whole machine up — it swept
every ticker through Yahoo, connected to IB, and switched Auto Pilot on against
a market that had been closed for hours.

### WHEN
Any weekday evening after about 3pm. Around 9pm is ideal. **Skip it on a
weekend** — the weekend rule hides what you are trying to see.

### DO
1. Make sure Auto Pilot is **ON** (status bar at the bottom shows `Auto: DESK`
   or `Auto: AWAY`, not `Auto: OFF`). If it is OFF, turn it on, then close the
   desk.
2. Close the desk completely.
3. Open it again the normal way.
4. Wait two minutes.
5. Open `autopilot.log` and press Ctrl+End.

### GOOD
Near the bottom you should see a line like:

```
Auto Pilot is ON from saved state, but nothing starts yet - quiet hours -
outside the 06:00-14:00 automatic-work window. BounceBot connects when the
window opens; manual scans and rebuilds work now.
```

And you should **NOT** see any of these:

- `Starting BounceBot`
- `Universe is stale ... rebuilding`
- `07:00 auto-arm: Auto Pilot ON for the day`
- any `Swing scan started for slot ...`

**Then check the manual path still works:** click **Run Scan** (Master AVWAP
page, or Ctrl+R). It should start normally. Quiet hours are only meant to stop
things the machine starts by itself — anything *you* click must still work at
any hour. You can cancel or let it finish; either is fine.

### BAD
Any of the forbidden lines above appearing, **or** the manual scan refusing to
start.

**Copy to the AI:** the last 40 lines of `autopilot.log`, and say whether the
manual scan ran.

---

# 2. Monday during the session

These six checks need a live market. You do not need to do them all in one day —
but each one needs the market open, so they cannot be done over the weekend.

Some need a specific Auto mode. The mode button is in the status bar at the
bottom of the desk; clicking it cycles **OFF → DESK → AWAY → EVENING → OFF**.

---

## 2.1 Quiet hours, the daytime half

**The question:** during the session, does automatic work actually run?

This is the other half of the tonight check. Tonight proves it stays asleep;
this proves it still wakes up.

### WHEN
Any time after 6:30am on Monday, once the desk has been open a few minutes.

### DO
Open `autopilot.log`, Ctrl+End.

### GOOD
A line like:

```
Automatic work resumed - inside the 06:00-14:00 automatic-work window.
```

and normal activity after it — swing scans starting, the watchlist build, and so
on.

### BAD
The log still saying `Automatic work paused` after 6:30am, or no new lines at
all since last night.

**Copy to the AI:** the last 60 lines of `autopilot.log` and the current time.

---

## 2.2 An EVENING day — does it stop after breakfast?

**The question:** EVENING is meant to get the morning ready and then go quiet.
Does it actually stop, instead of scanning all day?

### WHEN
A day you are happy to leave the desk in EVENING mode past about 8am. It does
not have to be a real sleep-in day — you can set EVENING deliberately just to
watch this.

### DO
1. Set the mode to **EVENING** before 7am (or the night before).
2. Leave it there until at least 9:30am.
3. Open `autopilot.log`, Ctrl+End.

### GOOD
Early on, the morning work happens: an `07:00` swing scan, the 07:00 / 07:15 /
07:30 strength checks, and the morning briefing.

Then, for each hourly slot that would normally have run, exactly one line like:

```
Evening mode: swing slot(s) 09:00 not run - Evening scans the 07:00 early slot
and the strength checks, then stops for the day. Flip to DESK to resume the
hourly schedule.
```

**A quiet log after the morning block is the mode working, not a broken
scheduler.** That is the whole point of the change.

Also confirm the BounceBot sweep is still running — you should still be getting
intraday alerts. That is deliberate: EVENING stops the *scheduled swing scans*,
not the M5 sweep.

### BAD
Ordinary hourly swing scans still starting after the morning block (lines like
`Swing scan started for slot 09:00`), **or** no `not run` lines at all.

**Copy to the AI:** the last 100 lines of `autopilot.log` and say what time you
set EVENING.

---

## 2.3 An AWAY day — do picks wait for you?

**The question:** while you are away, does the machine hold its picks instead of
dumping them into your watchlists, and does it stay silent?

### WHEN
A session you are actually away for, or any session you are willing to leave in
AWAY for an hour or two.

### DO
1. Note what is currently in `C:\TradingBotData\longs.txt` and `shorts.txt`
   (open them, or just note how many lines).
2. Set the mode to **AWAY**.
3. Leave it for at least an hour of live market.
4. Come back, and **before changing anything**, check those two files again.
5. Then set the mode back to **DESK** and watch the Alert Center for a moment.

### GOOD
- **While away:** `longs.txt` and `shorts.txt` are unchanged by auto picks. The
  desk makes no sound when alerts arrive. The Alert Center feed and the **D1
  Focus** tab's unread number both keep climbing — that number is how you see
  what you missed.
- **On the flip back to DESK:** a status message like
  `N auto pick(s) added to M5 Focus for today (...)`.
- **Important and new:** not everything that queued up will be adopted. Picks
  that no longer qualify are refused on purpose. Fewer names arriving than you
  expected is the gate doing its job.

To see the refusals, open `trading_bot.log`, Ctrl+End, and look for:

```
Focus gate refused N staged pick(s) at adoption: SYM (not above session VWAP), ...
```

### BAD
- New symbols appearing in `longs.txt` / `shorts.txt` while you were away.
- The desk beeping while in AWAY.
- Every single queued pick being adopted with no refusals logged *and* the
  market having clearly moved against some of them.

**Copy to the AI:** the `Focus gate refused` lines from `trading_bot.log`, and
say roughly how many picks you expected versus how many arrived.

---

## 2.4 The SPY wake alarm

**The question:** if the market moves hard while you are asleep in EVENING mode,
does the phone actually wake you?

### WHEN
Either a real day when SPY moves 1% or more, **or** any EVENING day if you force
it (below). Forcing it is the reliable option — do not wait for a 1% day.

### DO — the forced version (recommended)
1. Ask the AI to set `push_evening_spy_alarm_pct` to something tiny like `0.05`.
   (It is a machine-local setting; you do not need to edit files yourself.)
2. Set the mode to **EVENING** during the session.
3. Wait up to five minutes, watching your phone.
4. **When done, ask the AI to put the threshold back to `1.0`.** Do not leave it
   low — you will be woken constantly.

### GOOD
- An urgent notification on the phone titled something like
  `SPY +0.42% - market is moving`.
- If the move persists, it repeats — but **never sooner than five minutes**.
- The moment you flip out of EVENING, it stops.
- `autopilot.log` shows `Evening SPY alarm sent: SPY +0.42% on the day.`

### BAD
- No push at all (first check ntfy works — the Test Push button in
  Focus ▸ Phone Price Alerts).
- Repeats faster than every five minutes.
- Alarms continuing after you leave EVENING.
- **An alarm before the market opens.** That was a real bug and is fixed; if you
  see one, it is important.

**Copy to the AI:** the `Evening SPY alarm` lines from `autopilot.log`, the
times the pushes arrived on your phone, and the threshold you set.

---

## 2.5 Pick eviction and the adoption re-check

**The question:** when a pick the machine staged goes bad before you accept it,
does the machine drop it?

This one is mostly a log check — it is deliberately silent on the desk, because
you asked not to be bothered by it.

### WHEN
Any session, after about 9am, so the machine has had time to stage picks and
then re-check them at least once. The re-check runs every 30 minutes.

### DO
Open `autopilot.log`, Ctrl+End, and search (Ctrl+F) for `Focus gate`.

### GOOD
At least one line like:

```
Focus gate evicted 2 staged long pick(s): ABCD (not above session VWAP),
EFGH (not above yesterday's high)
```

The reason in brackets is the useful part — it tells you *why* the machine
changed its mind.

You may also see, at candidate-build time:

```
Focus gate refused 5 long candidate(s): ...
```

That is normal — it is the same rule applied earlier.

### BAD
No `Focus gate` lines at all after a full session. That would mean either
nothing was ever staged (possible on a very quiet day — check whether any picks
appeared at all), or the gate is not running.

**Copy to the AI:** all `Focus gate` lines from `autopilot.log`, plus roughly
how many auto picks you saw during the day.

---

## 2.6 "Not today" on an auto pick

**The question:** can you throw back a pick the machine gave you, without it
touching anything you chose yourself?

### WHEN
Any time after the machine has adopted at least one auto pick into M5 Focus
(check the Focus Picks page — if there are names there you did not add, you are
ready).

### DO
1. Before you start, note what is in your **M5 Focus** and **Swing Focus** lists.
2. Bring up an auto-adopted pick's chart in the Alert Center.
3. Look at the middle button on the row of three.
4. Click it.
5. Check both Focus lists again.

### GOOD
- On an **auto** pick, the button reads **`✕ Not today - drop pick`**.
- Clicking it removes **only** that one name from M5 Focus, on that one side.
- Your swing list is untouched. The other side is untouched. Every name you
  added yourself is untouched.
- Now open a chart for a name **you** added. The same button should read plain
  **`✕ Not today`** — and clicking it must **not** remove it from Focus. It only
  clears it from today's feed.

That difference is the safety rule: the machine can take back its own picks, and
can never take back yours.

### BAD
- Anything you added yourself disappearing from Focus.
- The swing entry disappearing when you dropped an M5 pick.
- The button reading `- drop pick` on a name you typed in yourself.

**Copy to the AI:** which name you clicked, what the button said, and what
disappeared. If one of your own names vanished, say so first — that is the most
serious failure on this page.

---

## 2.7 The strength board, first real look

**The question:** does the new board actually resemble your TC2000 scan?

### WHEN
Mid-session, once the market has been open at least an hour or so.

### DO
1. Click **Strength Board** in the left-hand menu.
2. Look at the status line at the top and the two lists.
3. Click **Refresh** and watch the status line.
4. Compare the names against your TC2000 scan.

### GOOD
- **Roughly 20–40 names per side.** That is the number to expect. Wildly more or
  wildly fewer is the thing to report.
- The status line names a time and counts, e.g.
  `Strength board 10:15:03: 31 long / 27 short (1487 of 1506 measurable)`.
- Refresh completes in well under a minute.
- The names broadly overlap what TC2000 shows you. It will not match exactly —
  it is a re-implementation — but the *character* should be familiar.

### Two things to judge and report

1. **Do the names look right?** This is a judgement only you can make. If they
   look wrong, say how — too many junk names, missing obvious movers, wrong side.
2. **Do you miss an RVOL column?** The board deliberately does not show relative
   volume yet. Use it for a session and see whether you reach for it. If you do,
   say so and it gets added for the ~20–40 names per side that survive the cut —
   that is cheap. Adding it for all 1,506 is not.

**Also worth timing:** it was measured at 27.6 seconds for the full universe,
but that was on a Saturday with quiet servers. If a mid-session refresh feels
much slower, that is worth knowing.

### BAD
- Both sides empty, or one side empty, on an ordinary trading day.
- The status line saying `last refresh FAILED: ...`.
- Hundreds of names per side.

**Copy to the AI:** the status line text, roughly how many names per side, and
your judgement on whether the names look like your TC2000 scan.

---

# 3. Monday after the close

## 3.1 The first-session checklist

**WHEN:** after the market closes, same day as the checks above.

**DO:** work through `docs/FIRST_SESSION_CHECKLIST.md`. It is the formal
acceptance list — machine details, provider telemetry, the Away report, shutdown
and restart behaviour. It already has rows for the quiet hours, AWAY, EVENING
and SPY-alarm checks above.

**GOOD:** every row filled in, including the ones that came out UNKNOWN or
failed. An honest UNKNOWN is a result. Do not tidy anything up to make the page
look better — a wrong "pass" costs more than a recorded failure.

**BAD:** rows left blank because you were not sure. Write what you saw instead.

---

## 3.2 The frozen rebuild and its self-test

**Already done and green as of 2026-08-15 09:58** —
`selftest OK: 30/30 checks passed (frozen)`, exit 0.

You only need to repeat this if code changed after that. **Ask the AI: "has any
code landed since the frozen rebuild? if so, rebuild and run the frozen
selftest."** It takes about four minutes and needs nothing from you.

Why it matters: the packaged app can start fine and then die the first time it
reaches something it failed to bundle. That has happened twice — once the app
could not scan at all for two full days before anyone noticed. The frozen
self-test is what catches it.

---

## 3.3 The merge

**WHEN:** only after §3.1 passes and the live checks above came out good.

**DO:** **tell the AI: "Monday's session passed — do the P0.7 merge."** Do not
run git commands yourself. There are three branches that have to go into `main`
in a specific order, and the AI has the order and the gates written down.

**GOOD:** the AI reports the merge done, plus a fresh full-test run, smoke check
and — if any code moved — a fresh frozen self-test.

### One thing that will probably go wrong, and is fine

There is a known flaky test:
`test_warehouse_seal.py::test_stale_staged_files_are_quarantined_not_deleted`.
It fails about half the time for a clock-precision reason on Windows, and it has
nothing to do with any recent work.

**If the test run fails and that is the only failure, it is not a reason to stop
the merge.** Re-run it. If any *other* test fails, that is real — stop and tell
the AI.

---

# If something goes wrong and you are not sure

Copy this to the AI, filling in what you can:

```
Testing plan step: <e.g. 2.3 AWAY day>
What I did:
What I expected (from the plan):
What actually happened:
Log lines (last 40 from autopilot.log or trading_bot.log):
```

You do not need to diagnose it. Reporting what you saw is the whole job.
