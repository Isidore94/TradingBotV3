# Status

Overwrite this file; don't append to it. Keep it under 3 KB. History lives in `git log`.

**Updated:** 2026-09-25 16:00 PT

- **Live on `main`:** the whole 2026-09-24 wishlist plan (P0-1 to P2-11, gates #209-#256)
  and the night AI fixes (`af1cc04c`, gate #257). The per-trade Mentor asks setup,
  thesis, stop and target once per trade (`9f33023f`).
- **In flight:** the plan to 8/10 on every goal (`TODO.md`, trader's go 2026-09-25).
  **Phase 1 and Phase 2 are done and wait for the trader's merge, in order:**
  `claude/p8-phase1-2026-09-25` (P1 stop backfill, P2 mentor stop first + bulk confirm,
  P3 import retry + Health, P4 night fixes) then `claude/p8-phase2-2026-09-25` (P5 journal
  truth lines + one native-currency R, P6 grades vs the tape). Six reviewers GO. Phase 3
  (P7 rip-weak, P8 phone/sound, P9 Show filter, P10 options chase) and Phase 4 (P11
  facets, P12 weak-variant tag) are built or in review on `claude/p8-phase3/4-*`. Each
  phase merges after reviewer GO, full suite and frozen selftest.
- **Night window:** the night AI starts 22:00 PT. No builders, merges or test runs
  22:00-02:00 PT.
- **Desk:** runs from source on `main`. Restart is the trader's call and is owed: batch 3
  (Credential Manager move, M5 key sidecar) is not live until then.
- **Last full suite:** 2026-09-25 Phase 2 branch: 11,986 passed, 3 known red
  (`test_tj17d_chosen_change.py`) plus one Qt layout flake that passes alone. Ruff clean, smoke 7/7, selftest 104/104 source and frozen.
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
