# Status

Overwrite this file; don't append to it. Keep it under 3 KB. History lives in `git log`.

**Updated:** 2026-09-28 08:40 PT (movers side tint + D1 trend gate merged, trader's word)

- **Live on `main`:** round 1, round 2 Phase A (`81272d42`), Phase B + S (`2d59aace`),
  S10a + S7 (`3802c0d7`), p9 phase 1 regime frame (`849390a2`), p9 phase 2 long
  setups + momentum universe + longs off + S9/P14 (`55bbeb26`), p9 phase 3 favourite-zone
  LONG gutted for Long leaders (`d382986e`), p9 phase 4 swing table, strength + under AVWAPE promotion + runner dip watch (`79e8d658`), p10 research lake history + SPY/QQQ/IWM
  auto regimes 2019+ + setups-by-regime backtester (`1b56f34f`, research only); rolling RRS everywhere, cutoff 1.0,
  `TRADINGBOTV3_RRS_ENGINE=desk` = old formulas; Movers side tint + D1 SMA trend gate
  (`13b2e2cb`); gates #257-#301.
- **GUI/startup delivery:** approved 4K GUI design in gui.md; Journal, Research,
  Day Review, supporting pages and Desk polish; Windows font-startup fix.
  Source and frozen selftests 105/105. Long combined Qt test runs still hit
  worker-lifetime crashes (not a harness fix).
- **Desk:** next launch uses main; live 4K/resource and first-usable checks are in GATES.
- **Night window:** the night AI starts 22:00 PT. No builders, merges or test runs
  22:00-02:00 PT.
- **Owed:** rolling-RRS leftovers still on % (industry board, autopilot open-scan RS,
  setup_group_context, D1 group strength; bounce CSV lacks `rrs_engine`, ask first); TLT/USO/HYG daily bars (scan-side, ask first); halted-name refetch; movers
  summary into the night AI; phone-brief rule line; pre-Aug entry_at (outcome code, ask
  first); review of the per-trade Mentor Save (`182f3e08`); weekly-options facet needs a
  theta store (only 43 of 6,462 theta picks ever got an option quote: IB option data).
- **Gates read 2026-09-26 05:50 PT:** #257 passed (night of 09-25: ideas ok, setup_research
  "narration absent", empty-evidence enrichment carries no tags, no event 2004). #264:
  tokens on 12 model rows; the digest lines exist but describe the goal-less 09-24 night
  (B1b); Health rows need a desk session. Econ brief still rejected (P4b).
- **Next action:** Monday: gates #300-#301 (rolling RRS, Movers); gates #297-#298 (runner dips armed, a fire in the Alert Center); `desk_perf_report.py --day 2026-09-28 --compare 2026-09-24`,
  gates #258-#287. Left: B10, B2, C4a, Phase C.
- **Trader actions owed:** set Risk per trade ($) in Settings > General; fill
  `trading_plan.md`; run the permutation backfill before Saturday; confirm setup tags
  (P2 screen); Task Scheduler "run whether logged on or not" on the night task;
  `map_freshness.py --apply`; desk down + market closed: `journal_pnl_repair.py` (#205),
  the options-journal repair (#162), `journal_questrade_gaps.py --statement`; the
  Saturday large-model probe; live click checks; consider rotating the market-prep
  OpenAI key (a test read it once in the A5 suite run).
