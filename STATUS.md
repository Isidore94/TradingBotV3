# Status

Overwrite, don't append. Under 3 KB. History is `git log`.

**Updated:** 2026-09-30 evening PT (Mentor P0-P15 merged; first full night with the coach brief is 2026-10-01)

- **Live on `main`:** rounds 1-2 (`81272d42`, `2d59aace`, `3802c0d7`), p9 phases 1-4
  (`849390a2`, `55bbeb26`, `d382986e`, `79e8d658`), p10 research lake + auto regimes
  (`1b56f34f`, research only); rolling RRS, cutoff 1.0 (`TRADINGBOTV3_RRS_ENGINE=desk` =
  old); Movers tint + SMA gate (`13b2e2cb`), Yahoo guard (`56ce3965`), Movers boxes, Dip
  anchors, focus preview, idle-probe skip (`1751c338`); gates #257-#306.
- **GUI:** 4K design live; selftests 105/105; long Qt runs still crash workers.
- **Merged 2026-09-30, live at the next desk + app restart:** Trade Mentor app P0-P15 (`e9c77690`, `e864ddb9`, `b2402479`, `19b837d3`) + Pause AI (`8f7664fc`): chat on gemma4:12b with native tools, auto-attach from plain language, `/pick /vetoes /tape /check /news /book /mirror /tilt /debate /hypotheses /brief /issues /recaps /paste /feel /night /scorecard`; memory stands on the night; `mentor_app_enabled` on by default; frontier OFF (trader). Gates #311-#318, #321-#326, #329, #332-#334.
- **Desk:** next launch uses main; live checks are in GATES.
- **Night throughput (2026-09-30):** slot retries x3, budget 360, every care name briefed nightly and last on the slate (#331), nightly permutation report + per-setup sentences (#330, shadow only). Parked: Mentor recall index (worktree only). Next: morning pre-brief queue; rule audit once plan lines exist.
- **gpt-oss:20b vs gemma4:12b:** effort `high`, 8k think tokens live. Probe 2 10-01 02:05 runs both, then flip `ai_local_model_medium`.
- **Night window:** starts 22:00 PT on the RTX 5080 host (`claude-host`, tunnel 11435). No local
  models on the mini-PC (trader 2026-09-30): host down = facts only; rerun a missed night with
  `--session <date> --force`. One pass + one recheck, then shut down; last firing 05:30. No
  builders, merges or test runs 22:00-02:00 PT.
- **Owed:** trial ledger index (nightly permutation search adds ~370 grids/3 MB a night, +8 s search per night of history); rolling-RRS leftovers on % (industry board, open-scan RS, setup_group_context,
  D1 group strength; bounce CSV lacks `rrs_engine`, ask first); TLT/USO/HYG bars (ask
  first); halted-name refetch; movers into the night AI; phone-brief rule line; pre-Aug
  entry_at (ask first); review of the Mentor Save (`182f3e08`); theta store.
- **Night read 2026-09-28:** econ brief rejected 4 nights (P4b); Sat show/regime_read
  rejected twice; setup_research narration absent. #264 needs a desk session.
- **Next action:** Tue 07:30 gate #305; gates #300-#301, #297-#298; `desk_perf_report.py
  --day 2026-09-28 --compare 2026-09-24`, gates #258-#287. Left: B10, B2, C4a, Phase C.
- **Trader actions owed:** Risk per trade ($) in Settings > General; confirm setup tags
  (P2); night task "run whether logged on or not"; `map_freshness.py --apply`; desk down +
  market closed: `journal_pnl_repair.py` (#205),
  options-journal repair (#162), `journal_questrade_gaps.py --statement`; Saturday
  large-model probe; live click checks; consider rotating the market-prep OpenAI key.
