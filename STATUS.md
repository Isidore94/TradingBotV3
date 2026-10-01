# Status

Overwrite, don't append. Under 3 KB. History is `git log`.

**Updated:** 2026-10-01 00:45 PT (night fixes + P19 merged; restart desk + app; probe 2 at 02:05)

- **Live on `main`:** rounds 1-2, p9 phases 1-4, p10 research lake + auto regimes (research only); rolling RRS, cutoff 1.0 (`TRADINGBOTV3_RRS_ENGINE=desk` =
  old); Movers tint + SMA gate (`13b2e2cb`), Yahoo guard (`56ce3965`), Movers boxes, Dip
  anchors, focus preview, idle-probe skip (`1751c338`); gates #257-#306.
- **GUI:** 4K design live; selftests 105/105; long Qt runs still crash workers. Hitches: 09-30
  blocked 2,643 s, two thirds GIL waits under the Setup Tracker paint; fixes merged
  09-30 night, live at the next desk restart; gate #337 on the first session.
- **Merged 2026-09-30, live at the next desk + app restart:** Trade Mentor app P0-P15 (`e9c77690`, `e864ddb9`, `b2402479`, `19b837d3`) + Pause AI (`8f7664fc`): chat on gemma4:12b with native tools, auto-attach from plain language, `/pick /vetoes /tape /check /news /book /mirror /tilt /debate /hypotheses /brief /issues /recaps /paste /feel /night /scorecard`; memory stands on the night; `mentor_app_enabled` on by default; frontier OFF (trader). Gates #311-#318, #321-#326, #329, #332-#334.
- **Merged 2026-09-30 (P16 eval gaps):** live eval tool hit 100 %, checklist 100 %; gate #335 owed.
- **Merged 2026-09-30 (P17 tape packs):** `/rs`, `/alerts`, M5 bars from the ~28-min spool tee (a desk-side 60 s M5 publisher is P18); live 108-question eval tool hit 100 %, 0 errors (gate #336). In flight: P18 journal/habits/reads/routines + M5 publisher.
- **Desk:** next launch uses main; live checks are in GATES.
- **Night (2026-09-30):** retries x3, budget 360, care names briefed nightly, last (#331), nightly permutation report (#330, shadow). Merged 10-01 (#338): P19 night rejections: econ time needs its id, regime read 1 retry, day show fits, idea with unknown ids dropped. Next: morning pre-brief queue; rule audit.
- **gemma4:12b is the night model** (live since 10-01, thinking OFF on main; `ai_local_reasoning_effort=high`, `ai_local_reasoning_tokens=30000` live but ignored while off). Probe 2 reports: `TradingBotV3-probeeport-20260930-*.md`: gemma4 fast, 2 grounding rejections; gpt-oss:20b at high cut at the 23k ceiling on every slot. In flight: `claude/gemma4-thinking-high-2026-10-01` (thinking at the set effort, `none` = off), not merged: measure 1-2 slots after a live night first.
- **Night window:** starts 22:00 PT on the RTX 5080 host (`claude-host`, tunnel 11435). No local
  models on the mini-PC (trader 2026-09-30): host down = facts only; rerun a missed night with
  `--session <date> --force`. One pass + one recheck, then shut down; last firing 05:30. No
  builders, merges or test runs 22:00-02:00 PT.
- **Owed:** trial ledger index (nightly permutation search adds ~370 grids/3 MB a night, +8 s search per night of history); rolling-RRS leftovers on % (industry board, open-scan RS, setup_group_context,
  D1 group strength; bounce CSV lacks `rrs_engine`, ask first); TLT/USO/HYG bars (ask
  first); halted-name refetch; movers into the night AI; phone-brief rule line; pre-Aug
  entry_at (ask first); review of the Mentor Save (`182f3e08`); theta store.
- **Next action:** Tue 07:30 gate #305; gates #300-#301, #297-#298; `desk_perf_report.py
  --day 2026-09-28 --compare 2026-09-24`, gates #258-#287. Left: B10, B2, C4a, Phase C.
- **Trader actions owed:** Risk per trade ($) in Settings > General; confirm setup tags
  (P2); night task "run whether logged on or not"; `map_freshness.py --apply`; desk down +
  market closed: `journal_pnl_repair.py` (#205),
  options-journal repair (#162), `journal_questrade_gaps.py --statement`; Saturday
  large-model probe; live click checks; consider rotating the market-prep OpenAI key.
