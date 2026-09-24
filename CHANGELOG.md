# Changelog

One line per change, newest first. The commit message holds the detail. Keep
about 40 lines and delete the oldest. The full old changelog and inventory are in
`notes/CHANGELOG.md`, local only.

- 2026-09-23 Day Review "Review my day (5 min)" walk (`ui/widgets/recap_walk.py`, cards in `recap_walk_cards.py`): a button by the glance strip and a closed-day banner (plus a nav dot) open a card-at-a-time walk that replaces the page body - trades (chart, R/P&L, grade and environment at entry, what-if, the awake Mentor questions and exit drafts), up to 3 real misses (recipe, lately, where to look, when the desk showed it), a good pass, calls, the market environment verdict and the lesson (rule check, Keep/Stop/Try + mood, tomorrow's rule, streak); ~10 cards by teaching value, keys, resume per session, every read and save on a worker, the day record rebuilt after; saved clues now draw on the page's name and trade charts.
- 2026-09-23 Day Recap chart clues (`ui/widgets/clue_marker.py`, not wired to a page yet): clue mode on `CandleChart` snaps a click to the nearest completed bar (tz-aware time), a tag-chip form saves via `record_clue` on a worker ("saved" / "not saved") with a bar OHLCV + SPY same-bar close snapshot (`record_clue(context=...)`, optional), `draw_clues` shows saved clues as tag-initial markers, `clues_for` reads them, `mark_clue_flow` wires it all for the Walk.
- 2026-09-23 Day Recap rule loop (`recap_rule_loop.py`): today's rule + streak on a status-bar chip and at the top of the Market Prep page / daily prep markdown (worker reads; hidden when none), and a Mentor `rule_reflection` question ("kept it or broke it?") on a closed trade the rule's tag can check (hold_winners, respect_stop, no_trade_first_15m, size_down_in_chop), once per trade, max 3 a day, unknown data never asks; answers read into the day record's recap.
- 2026-09-23 Week Review coach: Weekend Prep > Week Review shows your edge / your leaks (top/bottom 3 rows with n >= 10 from the day-record week files), repeats, rules kept, a 4-week trend, a week picker and a Month rollup; "Ask the AI" saves plain-words questions (`WEEK_QUESTIONS_FILE`) and the new night slot `week_questions` answers them from record excerpts only, every claim cited (uncited claims are not shown), and writes `records/week-<W>-frontier.md`.
- 2026-09-23 Day Recap data layer: `recap_store.py` is the one writer of the trader's recap inputs (card answers, Keep/Stop/Try + mood, rule, rule check, chart clues, environment verdict) in `day_recap_events.jsonl`; `day_session_record.py` writes one never-pruned point-in-time record per session (`day_review/records/<date>.json` + `.md`) and weekly rollups, rebuilt for the last 5 sessions by the night facts slot. No UI yet.
- 2026-09-23 Day Recap coach, findability (`scripts/recap_findability.py`, read-only, not wired to a page yet): each notable name's traits at pick time (unknown if recorded later), a plain-words recipe, how it did over 20 sessions split by environment (n < 10 = too few to tell), where the desk shows such names, when the desk first surfaced it; plus the session's environment timeline and label vocabulary. CLI `python scripts/recap_findability.py --date YYYY-MM-DD`.
- 2026-09-23 Recap point-in-time history: every setup-grades write also appends a dated, tz-aware snapshot (`SETUP_GRADES_HISTORY_DIR/<date>.jsonl`, unchanged content skipped, 600-day prune; reader `setup_grades_history.grades_as_of`), and the opening regime keeps a per-session row for the first read and the directional anchor (`AUTO_OPENING_REGIME_HISTORY_FILE`; reader `opening_regime_history.opening_regime_for`). Evidence only.
- 2026-09-23 Day Review cleanup (Day Recap step A): one clock (desk zone, HH:MM; UTC note times fixed), plain words, a glance strip over a "Details" report card, ◀ ▶ + Alt+Left/Right + remembered session, J/K rows, ONE miss table with five filter chips and a single-click name chart, trade charts, bars-file checks off the Qt thread, non-modal forecast paste, and a no-trade day is not an error.
- 2026-09-23 Test-run guard (conftest): one `-n` > 1 pytest run per machine (OS lock; a second run waits up to 30 min, naming the holder), and `-n` capped to 4 on weekdays 06:00-13:30 PT. Two parallel runs beside the live desk bluescreened the machine. `TBV3_TEST_LOCK=0` / `TBV3_TEST_WORKERS_CAP=off` bypass.
- 2026-09-23 GUI gc freezes: yfinance ticker threads (strength board, autopilot) now close their own SQLite cache connections instead of leaving them for the GUI-thread young sweep (5 s -> 0.1 s beside a busy thread); journal-health read closes its connection; chart daily-bars cache capped at 512 symbols (LRU); `thread_cpu.jsonl` now logs memory, object count and gc sweep time each minute (`thread_cpu_gauge.py --memory`).
- 2026-09-23 Snappy arms: arm clicks show ⏳ queued at once and arm on one ordered worker (ARMED / FAILED + reason; a second click cancels; drained at shutdown). Scanner-child M5 bars, zone arms and the warehouse tee cache are read off the Qt thread; bars not fetched yet are unknown and never fire or clear a watch.
- 2026-09-23 Hide Oil & Gas / Real Estate (default ON, one shared switch): hidden from the setups table, Alert Center feeds (no row, chart or sound; Focus and armed watches still show) and the phone report (`Hidden: N` line). Display only; unknown sector shows; everything is still recorded.
- 2026-09-23 Wall gate: review charts within 1 ATR20 of an SMA in their path or of the D1 trendline (or on its break day) are hidden and followed up by auto-armed `sma_break`/`ema15_reject` or an M15/M30 Pullback (cap 20; anything we cannot follow shows as `at wall`). Replay CLI `scripts/wall_gate_replay.py`.
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
