# Status

Overwrite this file; don't append to it. Keep it under 3 KB. History lives in `git log`.

**Updated:** 2026-09-25 17:00 PT

- **Live on `main`:** the whole 2026-09-24 wishlist plan (P0-1 to P2-11, gates #209-#256)
  and the night AI fixes (`af1cc04c`, gate #257). The per-trade Mentor asks setup,
  thesis, stop and target once per trade (`9f33023f`).
- **In flight:** the plan to 8/10 on every goal (`TODO.md`, trader's go 2026-09-25).
  **Phases 1, 2 and 3 are done and wait for the trader's merge, in order:**
  `claude/p8-phase1-2026-09-25` (P1 stop backfill, P2 mentor stop first + bulk confirm,
  P3 import retry + Health, P4 night fixes), `claude/p8-phase2-2026-09-25` (P5 journal
  truth lines + native R, P6 grades vs the tape), `claude/p8-phase3-2026-09-25` (P7
  rip-weak + Pop outcomes, P8 phone/sound notices, P9 Alert Show filter, P10 options
  chase). Ten reviewers GO. Phase 4 (P11 facets, P12 weak-variant tag) is on
  `claude/p8-phase4-2026-09-25`, suite running. Not built: P8's automatic M5-watch feed
  (a Movers rule forbids board writes; needs the trader's word), P13 (after 09-30), P14
  (ask-first `bounce_bot_lib`).
- **Night window:** the night AI starts 22:00 PT. No builders, merges or test runs
  22:00-02:00 PT.
- **Desk:** runs from source on `main`. Restart is the trader's call and is owed: batch 3
  (Credential Manager move, M5 key sidecar) is not live until then.
- **Last full suite:** 2026-09-25 Phase 3 branch: 12,096 passed, 3 known red
  (`test_tj17d_chosen_change.py`) plus one Qt chip-height flake that passes alone. Ruff clean, smoke 7/7, selftest 104/104 source and frozen.
- **Owed:** TLT/USO daily bars (scan-side, ask first); halted-name refetch; movers summary
  into the night AI; phone-brief rule line; pre-Aug entry_at (outcome code, ask first);
  review of the per-trade Mentor Save (`182f3e08`).
- **Next action:** integrate Phase 1; read gate #257 tomorrow; the first
  `permutation_report.json` lands Saturday only if the scratch backfill has run.
- **Trader actions owed:** restart the desk; set Risk per trade ($) in Settings > General;
  fill `trading_plan.md`; run the permutation backfill before Saturday; after the Phase 1 merge: run
  `scripts
egister_ai_jobs_task.ps1` once (07:00 import retry, gate #258) and
  `journal_stop_backfill.py` then `--apply` (desk down); confirm setup tags (P2 screen); Task Scheduler "run whether logged on or not" on the night task;
  `map_freshness.py --apply`; desk down + market closed: `journal_pnl_repair.py` (#205),
  the options-journal repair (#162), `journal_questrade_gaps.py --statement`; the
  Saturday large-model probe; live click checks.
