# Status

Overwrite this file; don't append to it. Keep it under 3 KB. History lives in `git log`.

**Updated:** 2026-09-25 20:55 PT

- **Live on `main`:** round 1 (`7d160b4b`, gates #257-#263) and round 2 Phase A
  (`81272d42`, merged 20:45 PT on the trader's word, gates #264-#266): perf tooling
  (`desk_perf_report.py --day`, bench ops, Movers tick timing), GC re-freeze after each
  full sweep + stat TTLs, known-red tests fixed, FlowLayout guard, one Credential Manager
  path, night telemetry, one R, Alert Center split step 1, noise report + Best-right-now
  log, missing-inputs chip, `setup_age` facet. Reviewer's one blocker (settings writers
  could build on the 1 s cache) fixed in `5d10ddaf`.
- **Desk:** restarted 2026-09-25 20:50 PT on `81272d42` by the lead on the trader's word;
  IB disconnected (TWS logged out). Startup shows a pre-existing swallowed warning
  (`DayReviewPanel.eventFilter` before `entry_text` exists, from `8f82213a`); one-line
  fix in Phase B.
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
- **Next action:** Saturday 06:37 PT: read gates #257/#264 from tonight's ledger, perf
  report for the 20:50 restart; then Phase B builders per `TODO.md`. #258 Monday 07:00;
  #259-#263, #265, #266 on Monday's session.
- **Trader actions owed:** set Risk per trade ($) in Settings > General; fill
  `trading_plan.md`; run the permutation backfill before Saturday; confirm setup tags
  (P2 screen); Task Scheduler "run whether logged on or not" on the night task;
  `map_freshness.py --apply`; desk down + market closed: `journal_pnl_repair.py` (#205),
  the options-journal repair (#162), `journal_questrade_gaps.py --statement`; the
  Saturday large-model probe; live click checks; consider rotating the market-prep
  OpenAI key (a test read it once in the A5 suite run).
