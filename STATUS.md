# Status

Overwrite, don't append. Under 3 KB. History is `git log`.

**Updated:** 2026-09-30 PT (Trade Mentor app P0 in flight)

- **Live on `main`:** round 1, round 2 Phase A (`81272d42`), Phase B + S (`2d59aace`),
  S10a + S7 (`3802c0d7`), p9 phase 1 regime frame (`849390a2`), p9 phase 2 long
  setups + momentum universe + longs off + S9/P14 (`55bbeb26`), p9 phase 3 favourite-zone
  LONG gutted for Long leaders (`d382986e`), p9 phase 4 swing table, strength + under AVWAPE promotion + runner dip watch (`79e8d658`), p10 research lake history + SPY/QQQ/IWM
  auto regimes 2019+ + setups-by-regime backtester (`1b56f34f`, research only); rolling RRS everywhere, cutoff 1.0,
  `TRADINGBOTV3_RRS_ENGINE=desk` = old formulas; Movers side tint + D1 SMA trend gate
  (`13b2e2cb`), Yahoo download guard (`56ce3965`),
  Movers boxes + Dip anchors, daytime focus preview, night AI idle-probe skip +
  Saturday summary reads every slice (`1751c338`); gates #257-#306.
- **GUI:** 4K design (gui.md) live; selftests 105/105; long combined Qt test runs
  still hit worker-lifetime crashes.
- **In flight:** Trade Mentor app P0 shell (`claude/mentor-app-p0-shell-2026-09-30`,
  not merged; gate #311). Plan: TODO "Trade Mentor app".
- **Desk:** next launch uses main; live 4K/resource and first-usable checks are in GATES.
- **Night window:** the night AI starts 22:00 PT. Its model runs on the RTX 5080 host
  (`ai_remote_gpu_ssh_alias` = `claude-host`, ssh tunnel on 11435); falls back to local Ollama
  (also mid-run). One pass + one recheck, then unload and shut down; last firing 05:30. No builders, merges or test runs
  22:00-02:00 PT.
- **Owed:** rolling-RRS leftovers still on % (industry board, autopilot open-scan RS,
  setup_group_context, D1 group strength; bounce CSV lacks `rrs_engine`, ask first); TLT/USO/HYG daily bars (scan-side, ask first); halted-name refetch; movers
  summary into the night AI; phone-brief rule line; pre-Aug entry_at (outcome code, ask
  first); review of the per-trade Mentor Save (`182f3e08`); weekly-options facet needs a
  theta store (43 of 6,462 picks quoted).
- **Night read 2026-09-28:** econ brief rejected 4 nights ("1 p.m." auction time, P4b);
  Sat day_review_show (names XLK) and regime_read ("W bearish") rejected twice, reply text
  not kept; setup_research narration absent. #264 Health rows need a desk session.
- **Next action:** Tue 07:30 gate #305; gates #300-#301 (rolling RRS, Movers); gates #297-#298 (runner dips armed, a fire in the Alert Center); `desk_perf_report.py --day 2026-09-28 --compare 2026-09-24`,
  gates #258-#287. Left: B10, B2, C4a, Phase C.
- **Trader actions owed:** set Risk per trade ($) in Settings > General; fill
  `trading_plan.md`; run the permutation backfill before Saturday; confirm setup tags
  (P2 screen); Task Scheduler "run whether logged on or not" on the night task;
  `map_freshness.py --apply`; desk down + market closed: `journal_pnl_repair.py` (#205),
  the options-journal repair (#162), `journal_questrade_gaps.py --statement`; the
  Saturday large-model probe; live click checks; consider rotating the market-prep
  OpenAI key (a test read it once in the A5 suite run).
