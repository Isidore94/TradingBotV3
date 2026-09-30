# Status

Overwrite, don't append. Under 3 KB. History is `git log`.

**Updated:** 2026-09-30 PT (Mentor P13 routing built)

- **Live on `main`:** rounds 1-2 (`81272d42`, `2d59aace`, `3802c0d7`), p9 phases 1-4
  (`849390a2`, `55bbeb26`, `d382986e`, `79e8d658`), p10 research lake + auto regimes
  (`1b56f34f`, research only); rolling RRS, cutoff 1.0 (`TRADINGBOTV3_RRS_ENGINE=desk` =
  old); Movers tint + SMA gate (`13b2e2cb`), Yahoo guard (`56ce3965`), Movers boxes, Dip
  anchors, focus preview, idle-probe skip (`1751c338`); gates #257-#306.
- **GUI:** 4K design live; selftests 105/105; long combined Qt runs still hit
  worker-lifetime crashes.
- **Merged 2026-09-30, live at the next desk restart:** Trade Mentor app P0-P12 (`370c6efb`, `8f7664fc` Pause AI, `e9c77690` P7-P12, `152e765f` plan inference): chat, `/plan` `/drop`, `/pick`, `/vetoes`, `/tape`, `/check`, `/news`, `/book` (Questrade + IBKR), `/mirror`, `/tilt`, `/debate`, `/hypotheses`; gates #311-#318, #321-#327. `mentor_app_enabled` off until #312; frontier OFF.
- **In flight:** Mentor P13 plain-language routing (`claude/mentor-app-p13-routing-2026-09-30`, gate #329): auto-attach, `journal_pack`, checklist, gemma4:12b native tools; trader runs `mentor_eval.py --live`.
- **Desk:** next launch uses main; live checks are in GATES.
- **Night throughput (2026-09-30):** slot retries x3, budget 360, all Focus names briefed nightly, last on the slate (`claude/night-focus-briefs-2026-09-30`, #330). In flight: Mentor night recall index. Next: per-setup condition reports, morning pre-brief queue.
- **gpt-oss:20b:** trader's word 2026-09-30; effort `high`, 8k think tokens live; on the mini-PC too. Probe 2 10-01 02:05, then flip `ai_local_model_medium`.
- **Night window:** the night AI starts 22:00 PT. Its model runs on the RTX 5080 host
  (`ai_remote_gpu_ssh_alias` = `claude-host`, ssh tunnel on 11435). No local models on the mini-PC
  (trader 2026-09-30): host down = facts only; rerun a missed night with `--session <date> --force`. One pass + one recheck, then unload and shut down; last firing 05:30. No builders, merges or test runs
  22:00-02:00 PT.
- **Owed:** rolling-RRS leftovers on % (industry board, open-scan RS, setup_group_context,
  D1 group strength; bounce CSV lacks `rrs_engine`, ask first); TLT/USO/HYG bars (ask
  first); halted-name refetch; movers into the night AI; phone-brief rule line; pre-Aug
  entry_at (ask first); review of the Mentor Save (`182f3e08`); theta store.
- **Night read 2026-09-28:** econ brief rejected 4 nights (P4b); Sat day_review_show and
  regime_read rejected twice; setup_research narration absent. #264 needs a desk session.
- **Next action:** Tue 07:30 gate #305; gates #300-#301, #297-#298; `desk_perf_report.py
  --day 2026-09-28 --compare 2026-09-24`, gates #258-#287. Left: B10, B2, C4a, Phase C.
- **Trader actions owed:** Risk per trade ($) in Settings > General; permutation backfill
  before Saturday; confirm setup tags (P2); night task "run whether logged on or not";
  `map_freshness.py --apply`; desk down + market closed: `journal_pnl_repair.py` (#205),
  options-journal repair (#162), `journal_questrade_gaps.py --statement`; Saturday
  large-model probe; live click checks; consider rotating the market-prep OpenAI key.
