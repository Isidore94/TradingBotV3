# 0021 — One Day Review page, a Week Review first step, and an overnight voice

Date: 2026-09-17

Sets the shape of the trader journal program (`plan.md` Phase 0.33). Amends none of
decisions 0001–0020; it relies on 0011 (one-way, evidence-grounded AI), 0016 (the
trader's priorities) and 0018 (the overnight stage order).

## Context

The trader, 2026-09-17, on the Market Journal and Daily Recap pages: *"daily recap and
market journal feel less than ideal … I want to see what actually worked that I passed on
or what worked that I liked but didn't enter. basically instant walk away analysis … it
also needs some sort of chart system to show me when I commented on it … Market journal
just sucks … I don't need to see the SPY auto modes pasted in there … there's just too much
shit in these tabs and it's laggy … this should be a simple 'what worked, what didn't and
what was your process' … these systems should ALL intertwine into one easy to use trader
journal that's mostly automated."* And later the same evening: *"it's VERY important we
maximize the overnight AI runs to bridge gaps and to really make this program feel alive
and have it adapt to what we want. in addition the local AI can have a voice somewhere
where it offers ideas of what we can improve based on what it reads."*

Measured on the live desk that evening: 34 of the 77 Market Journal rows were the desk's
own `Auto mode X -> Y` rows, and the 2026-09-16 nightly narration repeated them; the Daily
Recap streams a 476 MB CSV on every open; the Market Journal builds four charts on the
first entry click.

## Decision

The trader answered twelve questions on 2026-09-17. Each answer is binding:

1. **One page, Day Review, replaces Market Journal and Daily Recap.** One date at a time.
2. **Week Review is the FIRST step of Weekend Prep** (the existing `week_review` step); the
   other steps stay.
3. **The desk stops writing auto-mode flips into the journal.** Flips go to the Auto Pilot
   log only; the existing rows are hidden everywhere and kept out of every AI input. The
   ledger is append-only, so they are filtered, never deleted.
4. **The walk-away grades all four populations:** liked but never traded; passed / clicked
   away / vetoed / not today; traded then left early; claimed D1 picks.
5. **One SPY chart always; a name's chart on click.**
6. **The local model writes each day's story overnight; the frontier model writes the
   week's** (one paid call per week) — the OpenAI (ChatGPT) API, the trader's choice the
   same evening. A Redo button reruns a day outside market hours.
7. **The forecast (the trader's scheduled ChatGPT output) may be pasted at any time**,
   before or after the close; it is read only in the overnight run, to summarise the day
   it is about. "Paste weekly forecast…" becomes "Paste daily forecast…". The trader
   supplied one example (a dated "Market Morning Brief" with a dashboard, an Intraday
   playbook with bullish-continuation and bearish-reversal conditions, one- and two-week
   turbulence scores and a ranked-signals bottom line); it is the reader's fixture.
8. **One day summary of the trader's own thoughts, plus a rolling D1 view** of what they
   believe about the bigger picture and whether each open thesis is still true.
9. **The AI's ideas live on a card on Day Review and Week Review** with Keep / Dismiss;
   process ideas and program ideas; kept program ideas are listed for the trader to copy
   into `WISHLIST.md`. The AI never writes WISHLIST, Focus, alerts, scores or policy.
10. **Build order is bones first:** page and speed, then walk-away, then charts, then the
    day story, then the week, then ideas, then mood/process fields.
11. **`plan.md` keeps a short invariants section** (§5–§7) and otherwise holds only this
    program; the old roadmap is archived verbatim.
12. **The lag is on opening the tab and clicking an entry**, not the whole desk; the
    packets measure those two paths with the desk bench.

## Consequences

- The two journal stores stay separate (the Market Journal is what the trader thought,
  the Journal is what they traded). Day Review shows both; no writer joins them.
- Every AI output in this program is a closed JSON schema whose sentences cite source ids;
  a failed or ungrounded output leaves the last verified file untouched; local inference
  stays inside the off-hours window.
- The Daily Recap's Review tab moves to Research > Results and its Staged picks table to
  the Auto Pilot page; nothing is deleted until the page gates pass (TJ-8).
- Mood and process fields are added to the entry schema now (optional, tolerated by every
  reader) so the data starts accumulating; timed emotion prompts and mood statistics are a
  later, trader-directed phase.

## Amendment 2026-09-19 — closing the review loop (trader)

A read-only audit of the live stores on 2026-09-18 found three broken links in the loop
*decide → say → trade → judge → tell*: real trades carry money and no meaning (215 trades,
one confirmed setup tag, no stop, no note); nothing grades the trader's market reads or
compares them with their picks (the thesis store is empty and its only writer left the
screen in TJ-1); and ~95% of the day's decisions are D1 calls measured with a same-session
M5 ruler. The trader, 2026-09-19: *"I agree with everything in this plan and want to build
it with the exception of I want to be forced to label my trades around 0900 as per trade
mentor."* Binding, as `plan.md` TJ-9 … TJ-13:

13. **Trade labelling is forced at 09:00 Pacific through the Trade Mentor.** The previous
    session's trades are all listed, Save waits for an answer or an explicit answer state
    per field, and an unanswered section rides on the later cards that day. The confirm is
    the trader's write; the machine only suggests. AWAY still prompts nothing.
14. **"Were you right" is a measured row, never a model's opinion.** A pure grader extracts
    each stated stance with its exact span and grades it against the benchmark's measured
    move; the overnight story may only narrate those rows. This amends answer 8's reading
    of the day story, not its shape.
15. **Congruence is printed, never pushed and never acted on:** the trader's D1 view beside
    the desk's D1 label, the side mix of their likes and the bias of their fills.
16. **Swing calls get a swing ruler:** an Earlier-calls table over five sessions,
    against-you-first beside ran-after, ATR units, one versioned real-miss rule, and a
    decision made outside a session belongs to the NEXT exchange session.
17. **A five-line deterministic report card heads Day Review**; the week and the month
    re-cut the same lines. It computes no new statistic.
18. **The night serves the trader first.** Inside decision 0018's narration/model stages
    the cheap trader-facing slots run before `ticker_briefs` and `ai_summary`; the stage
    boundaries themselves do not move. Decision 0018 is amended with TJ-13, not before.
