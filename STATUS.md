# Status

Overwrite this file; don't append to it. Keep it under 3 KB. History lives in `git log`.

**Updated:** 2026-09-25 14:00 PT

- **Live on `main`:** the whole 2026-09-24 wishlist plan (P0-1 to P2-11, gates #209-#256)
  and the night AI fixes (`af1cc04c`, gate #257). The per-trade Mentor asks setup,
  thesis, stop and target once per trade (`9f33023f`).
- **In flight:** the plan to 8/10 on every goal (`TODO.md`, trader's go 2026-09-25).
  Phase 1 (P1 stop backfill, P2 mentor stop first + bulk confirm, P3 import retry, P4
  night fixes) is building on `claude/p8-*` branches with Opus builders. Phases 2-4 follow
  in order. Each phase merges after reviewer GO, full suite and frozen selftest.
- **Night window:** the night AI starts 22:00 PT. No builders, merges or test runs
  22:00-02:00 PT.
- **Desk:** runs from source on `main`. Restart is the trader's call and is owed: batch 3
  (Credential Manager move, M5 key sidecar) is not live until then.
- **Last full suite:** 2026-09-25 night AI fixes: green except the known red
  (3 tests in `test_tj17d_chosen_change.py`, `test_st6 ... empty_snapshot`). Ruff clean,
  smoke 7/7, selftest 104/104 source and frozen.
- **Owed:** TLT/USO daily bars (scan-side, ask first); halted-name refetch; movers summary
  into the night AI; phone-brief rule line; pre-Aug entry_at (outcome code, ask first);
  review of the per-trade Mentor Save (`182f3e08`).
- **Next action:** integrate Phase 1; read gate #257 tomorrow; the first
  `permutation_report.json` lands Saturday only if the scratch backfill has run.
- **Trader actions owed:** restart the desk; set Risk per trade ($) in Settings > General;
  fill `trading_plan.md`; run the permutation backfill before Saturday; confirm setup tags
  (P2 screen); Task Scheduler "run whether logged on or not" on the night task;
  `map_freshness.py --apply`; desk down + market closed: `journal_pnl_repair.py` (#205),
  the options-journal repair (#162), `journal_questrade_gaps.py --statement`; the
  Saturday large-model probe; live click checks.
