# Status

Overwrite this file; don't append to it. Keep it under 3 KB. History lives in `git log`.

**Updated:** 2026-09-25 17:45 PT

- **Live on `main`:** the whole 2026-09-24 wishlist plan (P0-1 to P2-11, gates #209-#256)
  and the night AI fixes (`af1cc04c`, gate #257). The per-trade Mentor asks setup,
  thesis, stop and target once per trade (`9f33023f`).
- **Live on `main` (merged and pushed 2026-09-25 17:40 PT, desk restarted 17:43 PT):** the
  plan to 8/10, phases 1-4 (`7d160b4b`): P1 stop backfill (applied: 23 trades), P2 mentor
  stop first + bulk confirm, P3 import retry (task registered) + Health, P4 night fixes,
  P5 journal truth lines + native R, P6 grades vs the tape, P7 rip-weak + Pop outcomes,
  P8 phone/sound notices + gated M5 watch feed (trader's level rule), P9 Alert Show
  filter, P10 options chase, P11 M5 + D1 facets, P12 weak-variant tag. First
  `permutation_report.json` written 2026-09-25 (37 keys). Not built: P13 (after the
  09-30 points outcomes) and P14 (ask-first `bounce_bot_lib`).
- **Night window:** the night AI starts 22:00 PT. No builders, merges or test runs
  22:00-02:00 PT.
- **Desk:** runs from source on `main`; restarted 2026-09-25 17:43 PT on `7d160b4b`.
- **Last full suite:** 2026-09-25 Phase 4 branch: 12,242 passed, 3 known red
  (`test_tj17d_chosen_change.py`) plus two Qt load flakes that pass alone. Ruff clean,
  smoke 7/7, selftest 104/104 source and frozen on every phase branch.
- **Owed:** TLT/USO daily bars (scan-side, ask first); halted-name refetch; movers summary
  into the night AI; phone-brief rule line; pre-Aug entry_at (outcome code, ask first);
  review of the per-trade Mentor Save (`182f3e08`).
- **Next action:** read gates #257-#263 on the next sessions; Saturday's `setup_keys_narration`
  reads the new report; P13 after 09-30.
- **Trader actions owed:** restart the desk; set Risk per trade ($) in Settings > General;
  fill `trading_plan.md`; run the permutation backfill before Saturday; after the Phase 1 merge: run
  `scripts
egister_ai_jobs_task.ps1` once (07:00 import retry, gate #258) and
  `journal_stop_backfill.py` then `--apply` (desk down); confirm setup tags (P2 screen); Task Scheduler "run whether logged on or not" on the night task;
  `map_freshness.py --apply`; desk down + market closed: `journal_pnl_repair.py` (#205),
  the options-journal repair (#162), `journal_questrade_gaps.py --statement`; the
  Saturday large-model probe; live click checks.
