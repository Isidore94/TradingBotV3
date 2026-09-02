# 0016 — The trader's vision and priorities, stated in their own words

Date: 2026-09-02

Amends nothing in decisions 0001–0015; every constraint there still binds.
This record exists because the build had grown faster than the statement of
what it is for. On 2026-09-02 the trader answered twelve questions, one at a
time, and these are the answers. They are the tie-breaker for every
prioritisation call until the trader says otherwise. The mandatory read in
`CLAUDE.md` points here; `plan.md` Section 1 summarises it.

## Context

By 2026-09-02 the desk captured almost every preference the trader could
express (veto with reason, like with claim, pass, not-today, swing picks,
armed watches, Focus membership) and graded most of them forward, but the
grades reached the trader on screens they never open, the joins between a
preference and a real trade did not exist, and the research warehouse varied
exits without ever varying entries. Phase 0.13 (packets P0–P10) began to close
those gaps. Before going further the trader was asked what the program is for.

## The goals, in priority order

1. **Trade from this program only.** Preparation, discovery, monitoring,
   alerts, journal. Never order execution (decision 0001).
2. **Teach the bot what the trader likes with one click, from any screen.**
   Words are optional. The click alone is a fact the bot must process.
3. **The bot works out why.** What the liked names had in common, how to search
   for more of them, and when the best entry was — which may be three to five
   sessions after the like.
4. **Day trading is the biggest prize.** Fast chart review; alerts worth
   taking; knowing which alert types work *for this trader*, not in general.
5. **Swing setups get sharper.** Ranked, tracked, and the trader learns which
   variables matter and which they are not yet looking at.
6. **Two-tier AI.** Local models digest nightly; a frontier model reads the
   digests periodically and says what works. The trader remains the executor.
7. **It works on away days.** Day-job days, weekends and the trader's own
   market thoughts feed the same loop.
8. **Honest numbers.** Shadow first, sample floors, nothing changes a live
   alert until it has earned it (decisions 0002, 0009, 0010, 0011).
9. **The desk stays fast.**

## The twelve answers (2026-09-02)

1. **First thing to get right: picking the right names to look at.** Entry
   timing comes second. Scanner and scoreboard work outranks entry-grid work.
2. **How a name is scored as "right to show":** *both*, but *moved* comes
   first; the trader's likes only say where to look. Objective quality leads;
   personal fit follows.
3. **What "moved" means for a swing:** the D1 support (long) or resistance
   (short) level holds, then the move follows. The trader gives swings room;
   losses run about 1.5× the best wins, so **win rate is the number that
   matters**, not average R. This is the tracker's stop-at-a-level,
   two-closes rule, so the measurement already exists.
4. **What "moved" means for a day trade:** the intraday level holds, then the
   name runs. Rank by **maximum favourable excursion** — the most the move
   offered — not by any exit; exiting well is the trader's job. A day trade
   lasts minutes to hours. **The best day trades are also swings** (partial
   profit, hold the rest), so an M5 alert on a name that also carries a D1
   setup outranks the same alert on a name that does not.
5. **How "what works" comes back:** (a) a "what is working lately" summary the
   trader can reach without leaving the Trading Desk, and (b) a switch that
   ranks those things higher. The switch may reorder and lift; it may never
   mute or hide (decision 0010).
6. **"This market regime" needs no definition.** "Lately" is a rolling
   window (about 20 sessions). No regime label.
7. **Screens actually used:** the Trading Desk, sitting on the **Capture** tab
   almost all the time; the Journal when needed. Wanted more: Market Journal
   (if simpler), Away Recap (if it did the work), Weekend Prep (if it made
   journaling easy). **Never opened:** Research, Universe, the Alerts tab, the
   D1 Focus tab, the Armed tab. The Strength tab loses to the trader's own
   TC2000 scan; the RS/RW board should sit where Strength is. Consequence:
   the "what is working lately" summary belongs on the Trading Desk and in
   Weekend Prep, **not** in the Research tab.
8. **The phone on a work day:** keep what it does now (hourly best-swing
   digest included). Make the picks better — the best pick is often in the
   *near* bucket, not the favourite bucket, so the cream is not being sent.
9. **The trader's own strength scan (TC2000), long side, mirror for shorts:**
   - relative strength = average over the last 12 bars of `((C/O)−1)×100`,
     times `((C + C50)/2) / ATR50`; keep the **top 25 %**;
   - relative volume = `AVG(V / mean(V at the same bar offset over the prior
     15 sessions), 12)`; keep the **top 50 %**, and a pick must also be in the
     top 50 % of today's volume;
   - then price above VWAP, price above $5, above the 200 and 100 SMA, above
     the 15 EMA. "Those picks are all good in the moment."
   The desk has the strength formula, the 25 % cut, the VWAP check and the
   15 EMA; it lacks the time-of-day relative volume, the $5 / SMA floors, and
   scans a different universe. (Open: the timeframe of the 15 EMA and of the
   100/200 SMA — assumed M5 and D1 respectively until the trader says.)
10. **Journaling's slow part is tagging.** The bot should auto-tag every night
    and the trader corrects. Weekend Prep's week-in-review was slow to build,
    its first screen is a wall of text whose three CALLOUT lines are the only
    part that matters, its tables show three rows at a time, and each table
    has its own refresh. Wanted: **one Refresh for the whole tab**, a short
    verdict card on top, tables not prose, ten visible rows.
11. **Market Journal:** one box, one Enter, one or two thesis entries a day.
    The AI may read the chart data after the close if it needs it.
12. **Away Recap on a work evening:** 10–30 minutes. Show *more*: more names
    and charts than a top-five, the M5 alerts that were right, the Focus names
    and how they did. Not a list of tickers.

## Decision

- Prioritise work in the order of the goals above, and within a goal by the
  answer that names it. When two packets compete, the one that improves
  **which names are shown** beats the one that improves **when to enter**.
- Win rate, not mean R, is the headline statistic on every trader-facing swing
  surface; MFE after a held level is the headline on every day-trade surface.
  Mean R and the robust columns stay beside them, never replaced.
- "What is working lately" is a Trading Desk surface with a 20-session window
  and a display-only priority switch. It is never a filter and never a mute.
- The Research tab is the builder's surface, not the trader's. Nothing the
  trader must see may live only there.
- Unused surfaces (Alerts tab, D1 Focus tab, Armed tab, the Universe page) are
  candidates for removal or folding, **after the trader confirms each one**;
  this record does not authorise their removal.

## Consequences

- Phase 0.13 P10 (likes start a five-session watch) stands, but sequenced
  after the scanner/scoreboard items this record ranks above it.
- The trader's TC2000 scan becomes the specification for the Strength Board's
  filters; the desk's variant must match it before it is compared to it.
- The auto-tagger built in P6a becomes a nightly slot on new trades.
- Weekend Prep, Away Recap and the Market Journal are redesigned to the
  answers above; their existing specs are amended, not replaced.
