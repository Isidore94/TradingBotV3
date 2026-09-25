# Status

Overwrite this file; don't append to it. Keep it under 3 KB. History lives in `git log`.

**Updated:** 2026-09-25

- **Live on `main`:** everything through the 2026-09-25 wishlist merge: P0-2 rest (scan-factor board 2x, batched Yahoo fetch, compression calibration on SQLite; gates #222-#225), P1-4 setup keys (perm_ columns, backfill, search, Research > Setup keys, Saturday narration; #226-#229), P1-5 best-right-now (#230-#233), P1-6 entry state/plan/risk (#234-#237), P1-7 trading plan + `plan_review` slot (#238-#241). Reviewers GO. Older merges are in `git log`.
- **Desk:** runs from source on `main`. Down; restart is the trader's call. 2026-09-24: universe restored to 1,455 (from snapshot 20260922T130004) and the tracker `.bak` + `.damaged` copies pruned (2.6 GB). Gates #209-#212.
- **Last full suite:** 2026-09-25 wishlist integration: 11,740 passed + 2 load flakes (15/15 alone). Ruff clean, smoke 7/7, selftest 101/101 source and frozen.
- **Compact desk** merged 2026-09-24: Settings > General > Desk layout (New default / Old). Charts 541 px vs 216 px. Gate #208.
- **Auto modes** merged 2026-09-23 (`48d6cebb`, reviewer GO): DESK sends nothing to the phone; EVENING scans all morning,
  keeps the queue empty, rings SPY ±1% + price alerts every 10 s until the mode changes, catch-up card on the flip out. Gate #207.
- **Journal overhaul** merged 2026-09-23 (`6c92b650`, reviewers GO on P&L and auto-fill): Flex times as New York, socket/Flex dup
  collapse, made-up-entry trades kept but not counted, repair CLIs; setup evidence + Mentor suggestion lane; nightly regime fill;
  stat cards, long vs short, new calendar, tag chips. Gates #205-206.
  Owed: Day Review / Weekend Prep / recap totals should use `journal_analytics.counts_in_pnl`; MFE/MAE skipped.
- **Owed from 09-23:** swing-scan memory numbers; halted-name refetch; movers summary into the night AI; phone-brief rule line;
  `_mentor_rule_lane` list_trades on the Qt thread; pre-Aug entry_at (outcome code, ask first).
- **Next action:** read gates #217-#241 live. In flight: P2-8 breadth, P2-9 looking back, P2-10 journal (second wishlist merge); then P2-11 cleanup and M5 setup-key stamping (trader OK 2026-09-25).
- **Trader actions owed:** desk down + market closed: `journal_pnl_repair.py` (#205) and the options-journal repair (#162);
  Questrade gap days via a statement import; the Saturday large-model probe, Mentor confirmations, and live
  click checks.
- **Reviews owed:** the per-trade Mentor Save (`182f3e08`).
