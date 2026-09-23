# Changelog

One line per change, newest first. The commit message holds the detail. Keep
about 40 lines and delete the oldest. The full old changelog and inventory are in
`notes/CHANGELOG.md`, local only.

- 2026-09-23 Nightly AI reports name tickers: the measured report's swing examples show symbol, side, setup and date (not an occurrence hash); daily digest facts v3 add a capped names block (top D1 setups, settled swings, M5 alerts, journal trades) and the narration must name those tickers with their numbers or it is not published.
- 2026-09-23 AGENTS.md: the full suite takes 4-7 min, so run it in the background (a foreground run was killed at 99%).
- 2026-09-23 AGENTS.md test commands: offscreen Qt on Windows too, the main venv path for worktrees, and rerun suite failures on `main` before blaming a branch.
- 2026-09-23 Mentor asks once: a trade is asked about on one card only. Any answer opens its Save; leaving the card files what was typed and marks it `MENTOR_ASKED`; local AI fills blanks from the trader's words or leaves them blank. Fixes the DRAM trade asked every hour.
- 2026-09-23 Pullback dip gate (`pullback_sma_reclaim_v2`): M15/M30 Pullback triggers need price to have come to the SMA (0.2 x D1 ATR20 or a close break-and-reclaim, within 15 M30/30 M15 bars) and an LRSI flip under 20 before the 80 cross. Stops the ARM 09-23 no-pullback alert; QDEL 09-23 short still fires (real-tape goldens).
- 2026-09-23 AGENTS.md: builders commit WIP and resume existing worktrees, short code comments, Mentor data map row, and search tests before reversing a rule.
- 2026-09-23 Capture tab fills its space (sections stretch to the full tab on one line; picklists grow to show every reason). Veto vocabulary v4 adds Bad industry (q), Too early (w), Not remotely good (e); v3 reasons unchanged and pooled.
- 2026-09-23 M5 swing context: an M5 bar row on a name+side that is also a D1 swing setup shows `· D1 <grade>` (New when ungraded) and ★ when claimed; with prioritise on, those rows draw first (claimed, then grade). Display only, fed from the setups table's model reset.
- 2026-09-22 Close-scan freshness: a daily-bar cache hit now also needs the latest completed session (not just a file touched <30 min ago), so the 13:00 PT close scan no longer publishes on yesterday's bar after the 12:45 preview. Intraday unchanged; a failed refresh still falls back to the cache.
- 2026-09-22 "Working now" strip under Working-lately in the M5 column: today's M5 alerts by setup grade with average R since they fired (completed cached bars only, first stop touch = -1R, missing data = no data); tooltip per alert. Display only.
- 2026-09-22 Parallel tests: pytest-xdist, `pytest -n 8` runs the suite in ~4 min (was ~28). Three font-order-dependent tests fixed (desk icon fonts loaded up front; compact setups table drops Points and trims Bucket before it overflows).
- 2026-09-22 Setup grades: PROVEN/A/B/C/D/New from the trackers' real results (swing: tracker win rate + low bound; day trade: +1R before -1R). Shown on the setups table and M5 bar, best first; the priority switch now defaults ON.
- 2026-09-22 Slim-down review repairs: restored binding safety rules and complete owed gates, corrected STATUS/TODO, and set Codex defaults to Sol lead, Luna max manager and Luna xhigh workers.
- 2026-09-22 Repo slim-down: rewrote AGENTS.md to 8 KB; CLAUDE.md imports it; moved checkpoint/plan to STATUS/TODO and history to gitignored notes/. Initial Codex routing: gpt-6-luna for small work, gpt-6-sol for hard builds/reviews.
- 2026-09-22 Four work streams integrated on main (`cf9c6b2a`): AI-R1/R2/R3 overnight
  AI repair, SP1/SP2 setup-score repair, AR-1/2A/2B/3 alert review follow-up,
  TJ-17A–D recap learning.
- 2026-09-21 Per-trade Mentor Save: an answered trade is stored at once and never asked
  again (`182f3e08`).
- 2026-09-21 TJ-9E: the Mentor tells an exit from an entry. One exit box, plus a night
  slot that drafts why/felt/watching blind to the outcome (`7230b30d`).
- 2026-09-20 TJ-7: mood and process fields, reported only, never acted on (`b63db7af`).
- 2026-09-20 TJ-6: up to three grounded AI ideas a night, kept only by the trader's
  click (`a7809d7c`).
- 2026-09-20 TJ-5: Week Review first in Weekend Prep (`1b9d77e0`).
- 2026-09-20 TJ-12: a six-line report card heads Day Review (`843a3f02`).
- 2026-09-20 TJ-4: the day pack and the overnight day story (`d929e34f`).
- 2026-09-20 TJ-16: prediction ledger vs baselines, plus a blind word tagger (`c4a760e5`).
- 2026-09-20 TJ-14B: Mentor question registry, budget of three (`161e905c`).
- 2026-09-20 TJ-10: read grader, prediction ledger, congruence lines (`57b44ca9`).
- 2026-09-20 TJ-9Q: Questrade fill classifier; a sold put is a sale (`86c64f96`).
- 2026-09-20 TJ-13B: night-only large-model probe (`e54c8203`).
- 2026-09-19 TJ-14A: Mentor card with a forced prediction click (`e8c04f88`).
- 2026-09-19 TJ-15: D1 miss contrast (`fb3f55e9`).
- 2026-09-19 TJ-3: note markers on Day Review charts (`72647104`).
- 2026-09-19 TJ-11/11F: walk-away v2 and the decision session (`a89ec7d5`, `f00ec302`).
- 2026-09-19 TJ-9: forced 09:00 trade labels (`8077a758`).
- 2026-09-19 TJ-13A: night-only local inference, night slates (`9eaae1dd`).
- 2026-09-18 TJ-2A/2B: session bars and walk-away tables.
- 2026-09-18 TJ-1/1L: one Day Review page in two columns (`e00b734a`).
- 2026-09-16 Phase 0.32 entry quality finished (`5bc09528`).
