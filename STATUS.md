# Status

Overwrite this file; don't append to it. Keep it under 3 KB. History lives in `git log`.

**Updated:** 2026-09-25 19:25 PT

- **Live on `main`:** the 2026-09-24 wishlist plan (gates #209-#256), the night AI fixes
  (`af1cc04c`, gate #257) and the plan-to-8 round 1, phases 1-4 (`7d160b4b`, gates
  #258-#263). Desk restarted 2026-09-25 17:43 PT on `7d160b4b`.
- **Built, awaiting the trader's merge:** round 2 Phase A on `claude/p8b-phaseA-2026-09-25`
  (tip `5d10ddaf`): A1 perf tooling (`desk_perf_report.py --day`, bench ops for Alert
  Center x1000 / Working-lately / M5 chart / Movers board, Movers tick timing), A2 GC
  re-freeze after each full sweep + stat TTLs, A5 known-red tests fixed (the tests were
  wrong) + FlowLayout guard + one Credential Manager path, B1 night
  telemetry (tokens per slot, digest goal/unread lines, Health rows; gate #264), A4 one R
  (native, `r_definition` migration), A6 Alert Center split step 1 (10,125 -> 9,412
  lines), B6 noise report + Best-right-now log (gate #265), B4 missing-inputs chip (gate
  #266), C4a `setup_age` facet. Reviewer: one blocker (settings writers could build on
  the 1 s cache), fixed in `5d10ddaf` with a failing-first test.
- **Perf baseline (09-24 market hours):** GUI blocked 1305 s, 3276 stalls, p90 743 ms;
  full GC 306 ms/min, young 156 ms/min; add_alert 7.3 ms/alert at 1,000; Working-lately
  build 16-28 s (worker). First live delta: `desk_perf_report.py --day <next session>
  --compare 2026-09-24`.
- **Night window:** the night AI starts 22:00 PT. No builders, merges or test runs
  22:00-02:00 PT.
- **Last full suite:** 2026-09-25 Phase A tip: 12,404 passed, 2 order-dependent Qt
  flakes (`test_ws_10a_scan_freshness`, `test_ws_wl_watchlist_tab`;
  both pass alone twice). Ruff clean, smoke 7/7, selftest 104/104 source and frozen
  (frozen built at `37efbeff`; no packaging trigger since).
- **Owed:** TLT/USO/HYG daily bars (scan-side, ask first); halted-name refetch; movers
  summary into the night AI; phone-brief rule line; pre-Aug entry_at (outcome code, ask
  first); review of the per-trade Mentor Save (`182f3e08`); weekly-options facet needs a
  theta store (only 43 of 6,462 theta picks ever got an option quote: IB option data).
- **Next action:** trader merges Phase A and restarts; read gates #257/#258 (tonight's
  ledger, Monday 07:00), #259-#266 on Monday's session; Phase B per `TODO.md`.
- **Trader actions owed:** merge Phase A (`git merge --no-ff claude/p8b-phaseA-2026-09-25`
  on `main`, push, restart the desk); set Risk per trade ($) in Settings > General; fill
  `trading_plan.md`; run the permutation backfill before Saturday; confirm setup tags
  (P2 screen); Task Scheduler "run whether logged on or not" on the night task;
  `map_freshness.py --apply`; desk down + market closed: `journal_pnl_repair.py` (#205),
  the options-journal repair (#162), `journal_questrade_gaps.py --statement`; the
  Saturday large-model probe; live click checks; consider rotating the market-prep
  OpenAI key (a test read it once in the A5 suite run).
