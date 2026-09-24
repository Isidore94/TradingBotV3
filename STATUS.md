# Status

Overwrite this file; don't append to it. Keep it under 3 KB. History lives in `git log`.

**Updated:** 2026-09-23

- **Live on `main`:** everything through the 2026-09-22 integration (`cf9c6b2a`):
  overnight AI repair (AI-R1/R2/R3), setup-score repair (SP1/SP2), alert review
  follow-up (AR-1/2A/2B/3), recap learning (TJ-17A–D), TJ-9E exit notes and the
  per-trade Mentor Save.
- **Desk:** it runs from source on `main`. It was down for the 09-22 update and was
  not restarted. The next start is the trader's call.
- **Last full suite:** 10,429 passed, 14 skipped, 1 known flaky test (G7 research tab,
  green when run alone). Ruff clean, smoke 7/7, source and frozen selftest 97/97.
- **09-22 merges:** repo slim-down, setup grades + 8-core tests (gate #190), Working now strip (#191).
- **Close-scan freshness fix** merged 2026-09-23: the close scan no longer reuses a cache missing today's bar. Gate #192. Follow-up owed: halted names refetch each evening scan (reviewer advisory).
- **M5 swing context** merged 2026-09-23: M5 alert rows show the D1 swing grade and claim star. Gate #193.
- **2026-09-23 integration** (`80d86afd`): Mentor asks once (gate #194), Pullback dip gate v2 (#195),
  Capture tab fill + veto v4 (q/w/e), AGENTS.md test lines. Suite 10,524 passed, 2 load flakes green alone; selftest 97/97 source and frozen.
  Mentor-ask-once merged without a reviewer round.
- **Wall gate + Oil & Gas / Real Estate hide** merged 2026-09-23: D1/Focus review charts at an in-path SMA or the D1 trendline (1 ATR20) hide and auto-arm a follow-up (cap 20, else shown tagged `at wall`); Oil & Gas / Real Estate hidden by one switch. Live at the next restart. Gate #196.
- **Perf fixes** merged 2026-09-23 (`f3b2bfc8`), live at the next restart: test-run guard (the 09-23 07:52 bluescreen came from two parallel
  `-n 8` suites beside the desk), queued arms + no bot RPC on the Qt thread (gate #197), yfinance SQLite gc freeze fix + memory in `thread_cpu.jsonl`.
  Suite 10,681 passed. Owed: memory numbers in the swing-scan phase log (the scan child hit 4.7 GB).
- **Day Recap coach** merged 2026-09-23, live at the next restart: cleaned Day Review, 5-min Walk with Mentor,
  day records + week rollups, find-it-next-time, grade/regime history, chart clues, rule loop, week view + Ask the AI
  (gates #198–201). Owed: rule line in the phone brief; `_mentor_rule_lane` list_trades on the Qt thread.
- **Waiting for the trader's word:** chart scroll-back zoom fix on
  `claude/chart-wheel-zoom-2026-09-21` (`11c3a339`). It is not on `main`. Merge it only
  while the desk is down.
- **Next action:** read the live gates on the next session and local-model night (`docs/GATES.md`).
- **Trader actions owed:** the options-journal repair (gate #162: desk down, market
  closed, daytime), the Saturday large-model probe, Mentor confirmations, and live
  click checks.
- **Reviews owed:** the per-trade Mentor Save (`182f3e08`).
