# Status

Overwrite, don't append. Under 3 KB. History is `git log`.

**Updated:** 2026-10-01 18:35 PT (Mentor day-1 fixes merged; restart desk + app)

- **Live on `main`:** rounds 1-2, p9 phases 1-4, p10 research lake + auto regimes (research
  only); rolling RRS, cutoff 1.0 (`TRADINGBOTV3_RRS_ENGINE=desk` = old); Movers tint + SMA
  gate, Yahoo guard, Movers boxes, Dip anchors, focus preview, idle-probe skip; gates #257-#306.
- **GUI:** 4K design live; selftests 105/105; long Qt runs still crash workers. Desk hitch
  fixes merged 09-30 night (GIL waits under the Setup Tracker paint); gate #337 on the first session.
- **Trade Mentor app (merged 09-30, P0-P17 + Pause AI):** chat on gemma4:12b with native
  tools, auto-attach from plain language, `/pick /vetoes /tape /check /news /book /mirror
  /tilt /debate /hypotheses /brief /issues /recaps /paste /feel /night /scorecard /rs /alerts`;
  memory stands on the night; `mentor_app_enabled` on by default; frontier OFF (trader).
  Gates #311-#318, #321-#326, #329, #332-#336.
- **Merged 2026-10-01 (live at the next desk + app restart):** P18 journal mode, habits/routines,
  `reads_pack`, 60 s M5 publisher, trade-intent gate (#341-#344); P20 plain brief/simple replies (#345);
  Mentor UI: card box hides, Tape/Tilt/Mirror/Scorecard, Dock into the desk Mentor tab, Clear,
  A-/A+ (#346); book ignores broker-flat journal opens; day-1 fixes: short WIN/LOSS, alert
  follow-through, VWAP side, plan infer, facets, pause unload, vs-peers, follow-up carry.
- **Night (2026-09-30):** retries x3, budget 360, care names briefed nightly (#331), nightly
  permutation report (#330, shadow). Merged 10-01 (#338): P19 night rejections (econ time
  needs its id, regime read 1 retry, day show fits, idea with unknown ids dropped), digest
  cap, mentor digest retry. Next: morning pre-brief queue; rule audit once plan lines exist.
- **gemma4:12b is the night model** (live since 10-01, thinking OFF).
  In flight, not merged: `claude/gemma4-thinking-high-2026-10-01`.
- **Night window:** 22:00 PT on the RTX 5080 host (`claude-host`, tunnel 11435; the app uses
  11436). No local models on the mini-PC: host down = facts only; rerun a missed night with
  `--session <date> --force`. No builders, merges or test runs 22:00-02:00 PT.
- **Owed:** trial ledger index; rolling-RRS leftovers on % (ask first); TLT/USO/HYG bars
  (ask first); halted-name refetch; movers into the night AI; phone-brief rule line; theta store.
- **Next action:** gates #305, #300-#301, #297-#298, #258-#287. Left: B10, B2, C4a, Phase C.
- **Trader actions owed:** Risk per trade ($) in Settings > General; confirm setup tags (P2);
  night task "run whether logged on or not"; `map_freshness.py --apply`; desk down + market
  closed: `journal_pnl_repair.py` (#205), options-journal repair (#162),
  `journal_questrade_gaps.py --statement`; live click checks; rotate the market-prep OpenAI key.
