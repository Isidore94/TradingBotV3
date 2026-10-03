# Status

Overwrite, don't append. Under 3 KB. History is `git log`.

**Updated:** 2026-10-02 PT (data compaction merged)

- **Live on `main`:** rounds 1-2, p9 phases 1-4, p10 research lake + auto regimes (research
  only); rolling RRS, cutoff 1.0 (`TRADINGBOTV3_RRS_ENGINE=desk` = old); Movers tint + SMA
  gate, Yahoo guard, Movers boxes, Dip anchors, focus preview, idle-probe skip; gates #257-#306.
- **GUI:** 4K design live; selftests 105/105; long Qt runs still crash workers. Desk hitch
  fixes merged 09-30 (gate #337 on the first session).
- **Trade Mentor app (merged 09-30, P0-P17 + Pause AI):** chat on gemma4:12b with native
  tools, auto-attach from plain language, `/pick /vetoes /tape /check /news /book /mirror
  /tilt /debate /hypotheses /brief /issues /recaps /paste /feel /night /scorecard /rs /alerts`;
  memory stands on the night; `mentor_app_enabled` on by default; frontier OFF (trader).
  Gates #311-#318, #321-#326, #329, #332-#336.
- **Merged 2026-10-01 (live at the next desk + app restart):** P18 journal mode, habits/routines,
  `reads_pack`, 60 s M5 publisher, trade-intent gate (#341-#344); P20 plain replies (#345);
  Mentor UI: Tape/Tilt/Mirror/Scorecard, Dock tab, Clear, A-/A+ (#346); book ignores
  broker-flat journal opens; day-1 fixes (short WIN/LOSS, alert follow-through, VWAP side, plan
  infer, facets, pause unload, vs-peers, follow-up carry).
- **Night (2026-09-30):** retries x3, budget 360, care names briefed nightly (#331), nightly
  permutation report (#330, shadow). Merged 10-01 (#338): P19 night rejections, digest cap,
  mentor digest retry. Next: morning pre-brief queue; rule audit once plan lines exist.
- **Night model: gpt-oss:20b, effort medium, allowance 24k** (trader's word 2026-10-02; Mentor chat
  pinned to gemma4:12b via `mentor_model`). Merged: story rejection names the ids, schema
  closed to the pack's ids. First gpt-oss night 10-02.
- **Merged 2026-10-02:** AVWAPE quick test (Setup Tracker rows, both sides); docked Mentor
  hides off its desk page; plan-rule gate, shadow (#348). **Unmerged:**
  `claude/gemma4-thinking-high-2026-10-01`.
- **Data compaction (merged 2026-10-02, trader: "lose nothing"):** D1 scan peak ~3.4 GB (was
  ~15 GB), outputs byte-identical; lossless Parquet archive `d1_feature_history_archive.py`
  (D1 history + bounce outcomes + candidates, night slot `history_pack`, 30-day trim built but
  OFF until readers move to `read_history`); tracker detail archived before compaction;
  candidates clean-up packs before it deletes. Gates #349-#351. Next: move the D1 history
  readers to `read_history`, then switch trim on; bounce outcomes writer lock -> trim; tracker
  gate #57 count (1/5 clean on 10-02); the ~half of old stripped tracker records whose replay
  disagrees with the stored outcome (unexplained, report only).
- **Night window:** 22:00 PT on the RTX 5080 host (`claude-host`, tunnel 11435; app uses
  11436). Mini-PC's one local model: plan-rule gate (Kev-4B CPU). Host down: facts only;
  rerun a missed night with `--session <date> --force`. No builds/merges/tests 22:00-02:00 PT.
- **Owed:** trial ledger index; rolling-RRS leftovers on % (ask first); TLT/USO/HYG bars
  (ask first); halted-name refetch; movers into the night AI; phone-brief rule line; theta store.
- **Next action:** gates #305, #300-#301, #297-#298, #258-#287. Left: B10, B2, C4a, Phase C.
- **Trader owes:** Risk per trade ($) in Settings > General; confirm setup tags (P2);
  night task "run whether logged on or not"; `map_freshness.py --apply`; desk down + market
  closed: `journal_pnl_repair.py` (#205), options-journal repair (#162),
  `journal_questrade_gaps.py --statement`; live click checks; rotate the market-prep OpenAI key.
