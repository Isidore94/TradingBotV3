# Desk testing plan — what to check, and when

Document role: **active operator runbook.** Last reconciled: **2026-08-25**.

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

# 0. Where things stand — read this first

**Last updated: 2026-08-25**, after the post-close AWAY and outcome-sweep
checks.

**One live check failed, one did not — that second reading was wrong, is
corrected here, and you have since accepted it (2026-08-25 evening).**

- The AWAY review queue correctly stayed empty and the evidence/feed stores
  continued to fill, but the **AWAY Recap was empty** because the main window
  never handed it the day's alerts. **Repaired the same evening**; the check is
  owed again on a real AWAY day, not repeated on the old build.
- **The after-close outcome sweep DID run, and you accepted it.** `swept_at
  2026-08-25T14:27:36-07:00`, 656 pending / 656 finalized / 0 expired / 0 failed
  / 0 commit failures / 0 still open. The earlier "did not run" reading was
  taken at 14:21, six minutes before the sweep started. It started **52 minutes
  after its 13:35 due time, for a cause nobody has found yet** — the due logic
  itself is correct. You accepted that on 2026-08-25, so **this check is done**.
  Nothing was changed about scheduling; the desk now logs one timing line per
  scan cycle, so if the delay happens again the log will name what caused it.

So: do not use an empty recap as a pass, and the sweep check needs no repeat.

The 2026-08-17 and 08-18 AWAY sessions proved staging and hourly reporting. The
2026-08-25 AWAY session tested the newer no-review-queue/recap contract: queue
routing behaved and the recap had no input. Checks that need you at the desk are
still waiting for the right kind of day.

## What is finished

| Check | Result | What proved it |
|---|---|---|
| §2.5 pick eviction | **PASSED** | `trading_bot.log` on 08-18 at 10:31, 11:40, 12:11 and 12:48 shows `Focus gate evicted N staged pick(s)` with a reason beside every symbol |
| §1 quiet boot | **PASSED** | The 08-16 22:06 launch logged `nothing starts yet - weekend - quiet hours` and nothing automatic ran until 06:00 the next morning |
| §2.3 AWAY day | **HALF DONE** | The **staging half is proved**: two full days (08-17, 08-18) of picks staying staged with every hourly phone report present, and on 08-25 the queue correctly accumulated nothing while the feed, History, the D1 badge and the hourly reports filled. The **flip-back-to-DESK half is owed** — a populated recap and no chart-review backlog on the return. It was relabelled PASSED on 2026-08-25 and is restored here |

**One thing changed in this document because of what you saw.** The 08-16 boot
also logged `IB: connected`. That is **expected and fine**: the BounceBot panel
connects to IB every time the desk opens, at any hour, and always has. It is not
Auto Pilot waking up. The line that would mean Auto Pilot woke up is
`Starting BounceBot` — and that one was correctly absent. §1's GOOD list below
now says so.

## The 2026-08-19 DESK morning: the gate never ran

Read this before §2.5-§2.8 — it is why they are still owed after a DESK day.

Every adoption attempt crashed inside the gate itself
(`TypeError: can't subtract offset-naive and offset-aware datetimes`), the Alert
Center refused each pick fail-closed, and **121 picks were refused every 30
seconds all session with zero adoptions**. The cause was a clock, not a
judgement: the stored *measured bar* carries a timezone offset and the desk's
clock did not, so the two could not be compared at all.

**What that means for these proofs.** Nothing about the PDH/VWAP rule was
exercised on 2026-08-19, so §2.5 and §2.6 are **not** partially done — they are
untouched, and they are re-owed in full on the fixed build. The one thing the
morning did prove is that fail-closed works: a gate that could not answer
adopted nothing.

**What to look for on the next DESK day**, beyond the checks below:

- **Adoptions happen at all.** Within a poll cycle or two of the first staged
  picks, names should appear in M5 Focus and the status line should say
  `N auto pick(s) added to M5 Focus for today`.
- **`focus_auto_picks.json` is not empty.** It records what the machine adopted;
  on 08-19 its `picks` map stayed `{}` all day.
- **The log is readable.** If the gate ever fails again you should see **one**
  traceback and **one** `Focus gate check unavailable for N staged pick(s) this
  cycle` line per 30-second cycle — not one traceback per pick. If you see a
  flood, that is a regression worth reporting on its own.

## What is still waiting

| Check | Needs |
|---|---|
| §2.5 adoption happens, §2.6 adoption refusal, §2.7 "Not today", §2.8 strength board | **one DESK day** (re-owed in full: 2026-08-19 crashed before the gate ran) |
| §2.7a strength-board sorting and charts (new, 2026-08-19) | the same DESK day |
| §2.10 movers-only chart review (new, 2026-08-19 evening) | the same DESK day |
| §2.2 EVENING stop, §2.4 SPY wake alarm | **one EVENING night** |
| §2.11 the rebuilt review pane (new, 2026-08-20) | the same DESK day |
| §2.12 the phone actually wakes you (new, 2026-08-20) | the same EVENING night, **with Sleep Focus on** |
| §2.13 the graded veto cohort (new, 2026-08-20) | **one weekend**, after two weeks of vetoes |
| §2.3 the AWAY recap half (the DESK flip back) | **one full AWAY day** — the recap feed repair landed 2026-08-25 |
| ~~the after-close outcome sweep on a restarted desk~~ | **DONE — you accepted it on 2026-08-25** ("52min late is fine"). It ran clean: 656/656, 0 failed, 0 commit failures. The 52-minute delay is accepted as known, not explained; the desk now logs a timing line per scan cycle so that if it happens again, the log names what caused it |

Still: **one DESK day and one EVENING night** — §2.11 and §2.12 ride the two
already owed. §2.13 is the only genuinely new sitting, and it cannot happen
until the cohort has forward returns to grade.

## Two things were broken, and both are fixed

You would not have seen either one on screen.

1. **Three swing scans died mid-run** — 08-17 at 07:30 and 10:00, 08-18 at 12:00.
   Each had already done 8 to 30 minutes of real work and then hit a locked file
   at the moment it wrote its market-prep report, and threw the whole scan away.
   The lock was the desk's own Market Prep page reading that file. The writer now
   waits a moment and tries again instead of giving up, and if it ever fails
   again `autopilot.log` will **name the reason on the same line** instead of
   just saying "exited with code 1".
2. **The Monday 06:00 universe rebuild failed** with a message about `datetime`.
   One odd row of data from Yahoo aborted the whole rebuild. It now skips the bad
   row and carries on — and, separately, a rebuild that finds *nothing* now
   refuses to overwrite your good universe files rather than blanking them.

**The desk was rebuilt and re-tested after both fixes** (all tests green, smoke
7/7, packaged self-test `31/31 (frozen)`). The build you will run for the DESK
day and the EVENING night is that one.

# 1. Tonight (optional) — the quiet-boot check

**The question this answers:** if you open the desk late at night, does it
correctly do *nothing* until the morning?

Before this was fixed, launching at 9pm woke the whole machine up — it swept
every ticker through Yahoo, connected to IB, and switched Auto Pilot on against
a market that had been closed for hours.

> **Already PASSED on 2026-08-16 22:06.** Keep this step for future builds; you
> do not need to repeat it before the DESK day or the EVENING night.

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

**`IB: retrying` and `IB: connected` are fine and expected.** The BounceBot panel
connects to IB every time the desk opens, whatever the hour, and always has —
that is not Auto Pilot starting up. The line that *would* mean Auto Pilot started
the bot is `Starting BounceBot`, and that is the one that must be absent. This
tripped up the 2026-08-16 reading, so it is spelled out here.

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

# 1b. The journal is rebuilt, and it has NOT touched your data yet

### WHEN

Any time. This is a read-only look, and it is deliberately the only thing R7
asks of you before Monday.

### DO

Open the desk and go to **Journal**. Before migration, the page must say that
preparation is required, show **Prepare Journal database**, and keep Trades,
Calendar, Analytics, Health, and Fees disabled. **Do not click the button yet.**

### GOOD

The preparation message is visible, the button is enabled, and all five tabs are
disabled. That proves merely opening Journal does not migrate the live database.

### BAD

The page starts migration by itself, the tabs are enabled before preparation,
the page fails to open, or anything raises. Any of those is a real problem and
worth stopping for.

### DO NOT, YET

Do **not** click **Prepare Journal database**, run the full backfill, or use
**Pull today now**, **Backfill Questrade gaps**, or **Retry failed Questrade
days**. Those are
the trader-present steps and they wait until after Monday's validation day
passes. The migration takes an automatic file backup and there is a dry-run
report to read first; none of that is a reason to run it early.

---

# 1c. This weekend (optional, and nothing waits on it)

### WHEN

Any weekend. Unlike everything else here, this one does not wait for Monday —
the tab is read-only against your data and starts nothing until you press
something.

### DO

Open the desk on a Saturday or Sunday. There is a new **Weekend Prep** page
between Journal and Universe. Walk the five steps down the left rail.

On **Discovery**, press Refresh on each of Hourly, Daily and Monthly. Note
roughly how long each takes.

### GOOD

- Nothing fetches until you press a button — no network noise on the desk log
  between opening the tab and your first press, and **zero IB traffic** all the
  way through.
- Each board shows a line like "1506 offered, 1440 measurable, 360 in the top
  25%, 41 after filters". On Monthly, measurable should be visibly **lower**
  than offered — recent listings have too little history and are excluded
  honestly rather than scored.
- Pick one name on the Monthly board and check it has no bar for the current
  month.
- Adopt one name. It should appear in swing Focus, in `swinglongs.txt`, in the
  membership file and in `pick_feedback.jsonl` with `origin="weekend_prep"` —
  and **nothing anywhere should have been removed**.
- Close the app halfway through and reopen it: the rail remembers where you got
  to.

### BAD

Anything fetching on its own. Any IB connection. Any name disappearing from a
watchlist. Boards whose character does not match what you would expect from a
strength scan on that timeframe — that last one is a judgement only you can
make, and §5's filters are not proven until you have made it.

---

# 2. Monday during the session

These checks need a live market. You do not need to do them all in one day —
but each one needs the market open, so they cannot be done over the weekend.

**2.1–2.7 are the ordinary checks**: does the desk do the right thing on a
normal day. **2.8 is the provocations**: does it do the right thing when the
day goes wrong. Do the ordinary ones first.

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

## 2.3 An AWAY day — does the recap replace the review queue?

> **HALF DONE. The staging half is proved; the recap half is owed.** On
> 2026-08-25 the review queue correctly accumulated nothing while the ordinary
> feed, History, the D1 badge, the evidence streams and the hourly phone output
> continued to fill — that half behaved. The recap itself was empty because it
> was never handed those alerts. **That was repaired the same evening**, so this
> section is now waiting for one real AWAY day rather than for engineering.

**The question:** while you are away, does the machine keep recording and
reporting the day without leaving hundreds of charts to review, then give you a
complete end-of-day recap?

### WHEN
The recap feed repair landed 2026-08-25. Use one full AWAY session and check it
after the close, before changing modes.

### DO
1. Set the mode to **AWAY** before alerts begin and leave it there through the
   close.
2. Confirm the hourly phone reports arrive and contain alerts.
3. After the close, open **AWAY Recap**.
4. Check that the recap covers today's alerts and that returning to DESK does
   not produce an old chart-review backlog.
5. Open one alert row (double-click or Enter). The desk's usual chart popup
   should appear — the same one the Strength Board opens. Try a swing row too.

**Start the desk BEFORE the session, not after.** The recap can only show what
the running program itself saw. If you restart after the close, the day's alerts
went with the old process and the page will be empty through no fault of its
own.

### GOOD

- The recap is populated, session-scoped to today and in the same order the
  alerts occurred.
- The alerts table has rows in it — time, symbol, side, tier, a `D1` mark on the
  D1 ones — and opening a row charts that symbol.
- History, the D1 unread badge, the backing alert list and evidence stores still
  contain the day.
- The chart-review queue stayed empty in AWAY, so there is no hundreds-chart
  drain on return.

### BAD

- An empty recap when the hourly reports or History show alerts.
- Yesterday's alerts mixed into today's recap, ordinary and D1 alerts out of
  order, or a review backlog appearing when you return.
- Any missing History/evidence rows. Queue routing is presentation only.

**Copy to the AI:** a screenshot of AWAY Recap, today's hourly report, and the
matching History rows. Do not call an empty recap a quiet day when either source
contains alerts.

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

> **Eviction PASSED on 2026-08-18** — `trading_bot.log` shows
> `Focus gate evicted N staged pick(s)` at 10:31, 11:40, 12:11 and 12:48
> with a reason beside every symbol. The **adoption re-check** half is
> still owed: it needs a DESK day, because AWAY never adopts.

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

---

# 2.8 Provocations — try to break it on purpose

Everything above checks that the desk does the right thing when the day goes
normally. These check what it does when the day does not. They come from an
outside review of the code, and each one targets something that has either
already gone wrong once or is one small mistake away from going wrong.

Do as many as you have appetite for. Each is short. **None of them can lose
money** — the worst case is a name you have to re-add by hand.

---


## 2.7a The strength board: sorting and charts (new, 2026-08-19)

**The question:** can you read the board as charts instead of as a list?

### WHEN
Any time the board has rows — same session as §2.7 is ideal.

### DO
1. Click each column heading in turn: **Symbol, Strength, Day %, vs VWAP, Last**.
   Click one of them twice.
2. Click a row (a single click, not a double).
3. Click a different row, then a row on the other side.
4. With a sort applied, press one row's **Add to Focus**.

### GOOD
- Every heading sorts, and a small arrow shows which column and which direction.
  Clicking the same heading again flips the arrow.
- Rows with a blank cell (`—`) stay at the BOTTOM whichever way the arrow points.
  A blank is a name the board could not measure, not a zero.
- **Sorting is instant and the status line does not change.** A sort must never
  trigger a refresh — if the status time updates when you sort, report it.
- Selecting a row opens the usual chart popup on that symbol, with the same
  levels and buttons every other board's chart has. Selecting another row
  re-points the same window rather than opening a second one.
- Selecting on the other side clears the first side's highlight.
- After sorting, **Add to Focus** adds the symbol on that row — not its
  neighbour — and a name that has fallen back through VWAP is still refused with
  its reason.

### BAD, and worth reporting
- The Add button adds the wrong symbol after a sort.
- A second, third, fourth chart window stacking up as you click through rows.
- The board refetching (status line time changing) when you only sorted.


## 2.10 Movers only in chart review (new, 2026-08-19 evening)

**The question:** does the review queue now show you only the names that are
actually moving — and does it tell you the truth about what it held back?

### WHEN
Any session, once alerts are flowing. Mid-morning is ideal.

### DO
1. Watch the review chart for a while without touching anything.
2. Look at the right-hand end of the button row under the chart.
3. Click the `N hidden (inside yesterday's range) - show` line once.
4. Now press **Review ▶** on the Focus board (the deliberate focus-picks review).

### GOOD
- Every chart that comes up on its own is a long **above** yesterday's high or a
  short **below** yesterday's low. Names inside yesterday's range do not appear.
- A `MOVING` tag sits beside the reviewed-today badge on those charts.
- A name the desk could not measure (no prior session, no bars yet) **still
  appears**, tagged `unmeasured`. That is deliberate: a data gap must never
  blank your review.
- The hidden line states a real count, and it grows as the session goes on.
- One click shows exactly those names — and the filter stays off for the rest of
  the day, so you are not fighting it.
- **Review ▶ shows everything**, filtered or not. Your own list is your own list.
- Nothing disappeared anywhere else: the Alert Center feed, the history and the
  D1 badge still carry every name, and no watchlist changed.

### BAD, and worth reporting
- An empty review queue with **no** hidden line — that would mean the filter
  swallowed everything silently.
- A hidden count that does not match what appears when you click it.
- Any name vanishing from the FEED (not the chart queue) — the filter must only
  affect what is charted.
- Review ▶ showing a filtered subset of your Focus picks.

### WHY THIS IS SAFE TO TRY MID-SESSION
It hides charts; it changes nothing else. No alert is muted, no push is
suppressed, nothing is removed from any store, and the file that ranks your
alerts is untouched. If you dislike it, one click reveals everything for the day.

## 2.11 The rebuilt review pane (new, 2026-08-20)

**The question:** can you actually read the charts now, and did anything you
rely on go missing when the pane was rearranged?

### WHEN
Any DESK day, first time you work the review queue. Ride it on the §2.5 day.

### DO
1. Look at the pane with **no alert charted** (queue clear). It should show a
   placeholder telling you how to get a chart — not a title, a "waiting" line
   and an arm bar spread across a screen of blank space.
2. Chart something. The candles should own the pane; under them there should be
   the arm bar and **one** row of verbs, nothing else.
3. Type a ticker into the box under the chart and press Enter.
4. Arm an M5 watch and a D1 event from the hotbuttons under the chart.
5. Press **Alt+V** from the charts — not from the Capture tab. Then a digit.
6. Press **Alt+K**, then a digit.
7. Open the **Capture** tab and read it without scrolling.

### GOOD
- The empty pane says something useful instead of nothing.
- Charts dominate; one slim verb row under the arm bar.
- Alt+V raises the Capture tab, highlights a veto reason, and the digit files
  it. The chart then **disappears as "not for today"** and the next one comes up.
- Alt+K does the same for a setup claim, and also moves to the next chart.
- Veto, Like and Note all sit **side by side** on a wide window, and all nine
  veto reasons are visible without scrolling.
- The Armed tab title carries a count.

### BAD, and worth reporting
- **Any Alt key doing nothing.** That is the failure this rearrangement risked:
  a shortcut bound inside a hidden tab never fires, and two bindings for one key
  make Qt fire neither. Silence is the symptom.
- Enter in a note field not committing it.
- A note jumping you to the next chart — only veto and like should do that.
- The hypothetical-stop control reappearing (deliberately removed).
- The charts still unreadable, which would mean the arm bar came back too big.

## 2.12 Does the phone actually wake you? (new, 2026-08-20)

**The question:** urgent priority alone does **not** override iOS Sleep Focus —
ntfy has no Apple critical-alert entitlement. So this is a phone-configuration
test, not a desk test, and it has never been run.

### WHEN
Before an EVENING night you actually intend to sleep through. Five minutes.

### DO
1. Work through the **Sleep breakthrough checklist** in
   `docs/EVENING_MODE_RUNBOOK.md` — ntfy allowed in Settings ▸ Focus ▸ Sleep,
   notifications not set to Deliver Quietly, topic not muted in the app.
2. Turn **Sleep Focus on** on the phone. Put it down.
3. On the desk: Focus ▸ Phone Price Alerts ▸ **Test wake alert (urgent)**.

### GOOD
- It sounds **through** Sleep Focus, and the message on screen tells you that is
  what should have happened.

### BAD, and worth reporting
- Silence, or a notification that arrives without sound. Work back up the
  checklist and re-run. If it will not break through after that, iOS is
  refusing it and the remaining options are outside this app.
- Note the ordinary **Test Push** proves nothing here — it goes out at `high`,
  not the `urgent` your real alerts use. Passing it is not passing this.

## 2.13 Is the veto cohort telling you anything? (new, 2026-08-20)

**The question:** the desk now grades your vetoes forward every night. Do the
reasons that score badly match the ones you would name yourself?

### WHEN
**A weekend, after roughly two weeks of vetoing.** Not before — the returns
need horizons to mature, and on day one every cohort is empty by construction.

### DO
1. `.venv\Scripts\python.exe scripts\run_ai_jobs.py --scopes trader_judgement --force`
2. Read the summary it publishes.
3. Open `veto_cohort_performance.csv` and look at the reasons by forward return.

### GOOD
- The reasons that saved you money are ones you recognise as good calls.
- The write-up states its two caveats: that only "Main swing" claims are
  offered, and that some vetoed names were day-traded anyway.

### BAD, and worth reporting
- A reason ranked confidently off two or three rows — sample size is the whole
  risk here, and the write-up should say so rather than assert.
- Any suggestion that a veto reason should mute, filter, rank or gate anything.
  This stream is analysis-only and must stay that way.
- Cohorts split across `v1`/`v2` of the same reason making a number look thin.
  That is expected after the vocabulary bump and is recorded — worth knowing,
  not worth alarm.

## 2.8a Can the machine steal one of your picks?

**The defect this hunts:** the machine used to be able to relabel a name YOU
added as machine-owned, after which its own cleanup could delete it. Fixed —
this is the check that it stays fixed.

**DO**
1. Wait until the machine has staged some picks (AWAY or EVENING, any session).
2. Pick one of the staged names — or if you cannot see them, pick any symbol
   the bot is likely to stage.
3. Add that same symbol to **M5 Focus yourself**, by hand, on the same side.
4. Flip to **DESK** so the queue drains.
5. Open that name's chart in the Alert Center.

**GOOD** — the middle button reads plain **`✕ Not today`**, *not*
`✕ Not today - drop pick`. The name is still yours. Clicking it clears the feed
and leaves the name in Focus.

**BAD** — the button reads `- drop pick`, or clicking it removes your name from
Focus. **That is serious; stop and report it.**

**Copy to the AI:** the symbol, which side, what the button said.

---

## 2.8b A pick that has gone stale in the queue

**The question:** if picks sit in the queue a long time, does the flip back
re-check them against the current tape, or adopt what was true an hour ago?

**DO**
1. Leave the desk in **AWAY** for at least 45 minutes of live market — longer
   is better.
2. Flip to **DESK**.
3. Watch the Alert Center status line, then read `trading_bot.log` (Ctrl+End).

**GOOD** — some picks are refused, with reasons naming either VWAP/yesterday's
level or the measured bar:

```
Focus gate refused 3 staged pick(s) at adoption: ABCD (measured 9 M5 bars ago (limit 2)), ...
```

Adopting *fewer* names than were queued is the correct outcome.

**Also good, and worth knowing about:** if the re-check itself cannot reach the
data, the status line says so —

```
Auto picks left staged - could not re-check them against the current tape (...). Retrying in 60s.
```

Nothing adopts while that is happening. It retries for about five minutes, then
waits for the next 30-minute refresh. Picks sitting unadopted after a failed
re-check is the *correct* outcome, not a stuck desk.

**BAD** — every queued pick adopted with no refusals after a long AWAY stretch,
or picks adopted whose charts clearly no longer qualify.

---

## 2.8c Delete the provenance file while the desk is running

**The question:** if the file recording which picks are the machine's goes
missing, does the desk fail toward *your* ownership?

**DO**
1. With picks in M5 Focus, delete
   `C:\TradingBotData\focus_auto_picks.json`.
2. Close and reopen the desk.
3. Open a previously auto-adopted pick's chart.

**GOOD** — the button reads plain `✕ Not today`. With no marker, every entry
reads as yours and nothing automatic can remove it. Losing the file makes the
desk *more* cautious, never less.

**BAD** — anything disappearing from Focus by itself, or the `- drop pick`
label surviving the file's deletion.

**Variation worth doing:** instead of deleting it, open it in Notepad and
corrupt it (delete a brace). Same expected outcome.

---

## 2.8d The five-second mode-cache race

**The defect this hunts:** the desk re-reads the Auto mode at most every five
seconds. Flip DESK → AWAY and an alert arriving in that window can still beep.

**DO**
1. In **DESK** with sound on, flip to **AWAY**.
2. Listen for the next five to ten seconds.

**GOOD** — at most one beep in the first ~5 seconds, then silence.

**BAD** — beeping continuing well past ten seconds in AWAY.

This is a known, bounded five-second lag rather than a bug — report it only if
the silence never arrives.

---

## 2.8e Break the phone push on purpose

**The question:** when ntfy misbehaves, does the alarm back off, or does it
hammer / go silent forever?

**DO** — pick whichever is easiest:
- turn off wifi on the desk for a few minutes during an EVENING test with the
  threshold forced low; **or**
- set the ntfy topic in Settings to nonsense, run the alarm, then set it back.

**GOOD** — `autopilot.log` shows attempts **slowing down**, not repeating every
30 seconds:

```
Evening SPY alarm outcome UNKNOWN (attempt 1) - it may have reached the phone: ...
Evening SPY alarm REJECTED (attempt 2): ntfy HTTP 404
```

The gap between attempts grows to a maximum of one every five minutes. When you
fix it, the next alarm sends normally.

**BAD** — an attempt every 30 seconds, or nothing ever retrying once the
connection is back.

**Copy to the AI:** all `Evening SPY alarm` lines and what you broke.

---

## 2.8f A failed data chunk

**The question:** when Yahoo fails part of a fetch, does the desk lose that
chunk or the whole board?

**DO** — during a strength-board refresh, briefly drop wifi, then restore it
and press **Refresh** again.

**GOOD** — the status line shows a lower `measurable` count, or says the refresh
failed while **still showing the previous board**. A failed refresh must never
blank the board.

**BAD** — an empty board with no explanation, or a status line claiming a
successful refresh with no names.

---

## 2.8g TC2000 versus the board — by symbol list, not by feel

**The question:** does the board actually reproduce your scan? "Looks about
right" is not an answer anyone can act on.

**DO**
1. At the same moment, run your TC2000 scan and refresh the strength board.
2. Export or copy **both symbol lists** — TC2000's, and the board's (use
   double-click or copy the visible names).
3. Save them somewhere as two plain lists, noting the time.

**GOOD** — a large overlap. Some difference is expected; this is a
re-implementation, not the same product.

**What to report:** the two lists, plus the time. The useful questions are
answerable only from the sets: *which names does TC2000 have that the board
misses*, and *which does the board invent*. A list of misses points at a
specific filter; "the character looked off" points at nothing.

**Copy to the AI:** both lists as text, the time, and the side.

# 3. After the close

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

**Already done and green on 2026-08-15** —
`selftest OK: 49/49 checks passed (frozen)`, exit 0, on the repaired R7/R8 build.

The build was made only after deleting both `build/` and `dist/`, so no cached
module could preserve an obsolete roster. A changing count is normal as checks
are added; the **`(frozen)`** suffix and exit 0 are what matter.

You only need to repeat this if code changed after that. **Ask the AI: "has any
code landed since the frozen rebuild? if so, rebuild and run the frozen
selftest."** It takes about four minutes and needs nothing from you.

Why it matters: the packaged app can start fine and then die the first time it
reaches something it failed to bundle. That has happened twice — once the app
could not scan at all for two full days before anyone noticed. The frozen
self-test is what catches it.

---

## 3.3 The merge

**WHEN:** only after §3.1 passes and the live checks above came out good —
which now means **after the DESK day and the EVENING night**, not before.

**DO:** **tell the AI: "Monday's session passed — do the P0.7 merge."** Do not
run git commands yourself. There are three branches that have to go into `main`
in a specific order, and the AI has the order and the gates written down.

**GOOD:** the AI reports the merge done, plus a fresh full-test run, smoke check
and — if any code moved — a fresh frozen self-test.

### ~~One thing that will probably go wrong~~ — no longer true

This section used to say `test_warehouse_seal.py` fails about half the time for
a clock-precision reason, and that re-running it was fine. **That was fixed on
2026-08-15** — it was a real defect, not flakiness, and it is gone.

**There is now no test allowed to fail.** If the run is not fully green, stop and
tell the AI, whichever test it is. "Just re-run it" is no longer an answer.

---

## 3.4 If Monday goes badly — going back

**WHEN:** only if the session shows something genuinely broken and you want the
old build back for Tuesday.

**DO:** **tell the AI: "roll back to the pre-R1 build."** Do not do it yourself
while the desk is running — the 07:00 task has to be disarmed first, or it can
start the app from a half-swapped folder. The AI has that order written down, and
the whole path was rehearsed unattended on 2026-08-15.

**GOOD:** the app starts and the self-test passes.

**The one thing that will look wrong and is not:** the rolled-back build reports

```
selftest OK: 30/30 checks passed
```

**30, not 31 — and that is correct.** The 31st check is the one that makes sure
this testing plan is inside the packaged app, and this file did not exist in the
older build, so neither did the check. A lower count after a rollback is the
count going back in time with everything else. It is *not* evidence of a broken
rollback.

**BAD:** any other number, a failure line, or a non-zero exit.

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
