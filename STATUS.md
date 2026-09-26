# Status

Overwrite this file; don't append to it. Keep it under 3 KB. History lives in `git log`.

**Updated:** 2026-09-26 PT (Phase B built)

- **Live on `main`:** round 1 (`7d160b4b`, gates #257-#263) and round 2 Phase A
  (`81272d42`, merged 20:45 PT on the trader's word, gates #264-#266): perf tooling
  (`desk_perf_report.py --day`, bench ops, Movers tick timing), GC re-freeze after each
  full sweep + stat TTLs, known-red tests fixed, FlowLayout guard, one Credential Manager
  path, night telemetry, one R, Alert Center split step 1, noise report + Best-right-now
  log, missing-inputs chip, `setup_age` facet. Reviewer's one blocker (settings writers
  could build on the 1 s cache) fixed in `5d10ddaf`.
- **Desk:** DOWN. Restarted 20:50 PT on `81272d42`, then closed cleanly 21:51 PT on the
  trader's word; restart only on the trader's word.
- **Perf:** 09-24 baseline GUI blocked 1305 s, full GC 306 ms/min, young 156; on the new
  code (idle evening) full GC 4.7 ms/min, young 13.5. Monday: `desk_perf_report.py --day
  2026-09-28 --compare 2026-09-24`.
- **Night window:** the night AI starts 22:00 PT. No builders, merges or test runs
  22:00-02:00 PT.
- **In flight:** `claude/p8b-phaseB-2026-09-26` (NOT merged; merge needs the trader's
  word): R1 Show, B0, B1b, B3, B8, B9, B11-B13, P4b, A6 steps 2-3, S1-S6, S8, S10b/c,
  S11; gates #267-#275. Reviewer GO. Full suite 12,685 passed + 2 known order flakes
  (both pass alone twice); ruff clean, smoke 7/7, selftest 105/105 source and frozen.
- **Owed:** TLT/USO/HYG daily bars (scan-side, ask first); halted-name refetch; movers
  summary into the night AI; phone-brief rule line; pre-Aug entry_at (outcome code, ask
  first); review of the per-trade Mentor Save (`182f3e08`); weekly-options facet needs a
  theta store (only 43 of 6,462 theta picks ever got an option quote: IB option data).
- **Gates read 2026-09-26 05:50 PT:** #257 passed (night of 09-25: ideas ok, setup_research
  "narration absent", empty-evidence enrichment carries no tags, no event 2004). #264:
  tokens on 12 model rows; the digest lines exist but describe the goal-less 09-24 night
  (B1b); Health rows need a desk session. Econ brief still rejected (P4b).
- **Next action:** trader: merge Phase B (yes/no); S10a (stop recording the retired
  H1 type?); S7 (shadow engines edit the ask-first `m5_signal_engines.py`: yes/no).
  Left for live days: A4b, B10, B7, C4a, B2, Phase C. #258 Monday 07:00.
- **Trader actions owed:** set Risk per trade ($) in Settings > General; fill
  `trading_plan.md`; run the permutation backfill before Saturday; confirm setup tags
  (P2 screen); Task Scheduler "run whether logged on or not" on the night task;
  `map_freshness.py --apply`; desk down + market closed: `journal_pnl_repair.py` (#205),
  the options-journal repair (#162), `journal_questrade_gaps.py --statement`; the
  Saturday large-model probe; live click checks; consider rotating the market-prep
  OpenAI key (a test read it once in the A5 suite run).
