# Group RS/RW tape rebuild — build prompt (Opus)

Paste everything below the line into a fresh Claude Code session in the repo on
the main desk, model set to Opus. **One build session per checkout** — do not
run it while another Claude session is editing this checkout (two sessions
shared one checkout on 2026-08-26 and one stashed the other's work). Paste the
handoff back to the Fable session for review. Authorized: `plan.md` Phase 0.5
item 11, "Group RS/RW tape — rebuild" (trader, 2026-08-27, "make me a prompt to
get Opus to do it"). NOT authorized and not in this prompt: any change to
BounceBot (`scripts/bounce_bot_lib/legacy.py`), to the RS Window tab's payload,
to the scan cycle, or any new IB request.

---

You are rebuilding the **sector / industry RS/RW tape** across the top of the
Trading Desk in the TradingBotV3 repo on my trading desk. Read `CLAUDE.md`
first and follow its mandatory documentation workflow: `CHANGELOG.md` (the
2026-08-27 entries — five trader rules and the tape removal), `plan.md` §5–7
and Phase 0.5 item 11 (the parked rebuild plan — that is your spec),
`CURRENT_CHECKPOINT.md`. Build only what that item lists.

## Why (so the design makes sense)

I said: "often times the sectors and industry RS/RW thing at the top is totally
wrong and doesn't reflect what is actually strong over the last 30-60-90
minutes." The investigation (2026-08-27) found the maths right and the timing
wrong:

- The old tape refreshed only when a scan cycle's RRS pass finished
  (`compute_group_strengths` inside `run_rrs_scan`) — 10 to 30 minutes apart,
  frozen in between. It once flipped 31 minutes late.
- Its one intraday number was `real_relative_strength` over `RRS_LENGTH = 12`
  M5 bars on `5 D` of bars, so for the first hour the window straddled the
  overnight gap (06:36 read XLK +10.5 / XLC −18.6 — the gap, not the morning).
- "Industry" is an ETF proxy: 136 industries map to 49 ETFs.

The tape is currently HIDDEN on the desk (`TradingDeskPanel`:
`self.group_tape.setVisible(False)`), widget and wiring intact. You are giving
it a fresh, fast data source and a 30 | 60 | 90 face, then showing it again.

## Hard rules

1. **Zero IB traffic and no BounceBot change.** Do not edit
   `scripts/bounce_bot_lib/legacy.py`, `run_rrs_scan`, the scan cycle, or
   anything that requests historical bars from IB. Data comes from ONE batched
   `yfinance` download per tick, the way the Strength Board does it
   (`scripts/ui/services/strength_board_service.py` — copy its shape: a
   `QObject` owning one `QTimer`, single-flight worker thread, last-good on
   failure, `status_text`, bounded `shutdown`).
2. **Nothing expensive on the Qt thread.** The download and the maths run on
   the worker; the Qt side receives a finished payload through a signal.
   Widget variants live in `theme.qss` keyed on object names / dynamic
   properties, never per-widget `setStyleSheet`. Chips diff, never rebuild
   (the existing strip already rebuilds — if you touch that, make it diff).
3. **Completed bars only, today only.** Use `scripts/completed_bars.py`
   (`completed_m5_bars(bars, now=...)`, inclusive at the boundary, `astimezone`
   conversions — never `replace(tzinfo=None)`). Restrict to the current
   session's RTH bars so no window ever straddles the overnight gap.
4. **UNKNOWN, never invented.** 30 min needs 6 completed bars, 60 needs 12,
   90 needs 18, each PLUS the ATR warm-up the formula needs. A window without
   enough bars is `None` and the chip shows a blank segment, not zero and not
   "as many as we have" (`strength_scan.sma`'s reasoning applies). Missing
   data is uncertainty, never confirmation.
5. **One formula, proven equal.** Write the RRS maths as a pure module
   `scripts/group_rrs.py` (bar dicts in, floats out, no I/O, no Qt) that
   reproduces `bounce_bot_lib.legacy.real_relative_strength` (legacy.py ~:2573:
   `sym_move = close[-1] - close[-1-length]`, same for SPY, Wilder ATR of the
   bars EXCLUDING the last one over `length`, `power_index = spy_move /
   spy_atr`, `rrs = (sym_move - power_index * sym_atr) / sym_atr`). Ship a
   parity test that feeds identical bars to both and asserts equality to 1e-9
   — do not import `legacy` from the service.
6. **Quiet hours.** The timer is gated on `autopilot_core.auto_scanning_due`
   like every automatic starter (06:00–14:00 PT weekdays, fail-open on a
   session lookup it cannot answer); outside it the tape keeps last-good with
   its as-of and says so. A manual "Refresh" is never gated.
7. **The RS Window tab is untouched.** It keeps reading BounceBot's
   `rrsSnapshotChanged` payload; that answers a different question. Do not
   change `rs_window_feed.py`, `rrs_snapshot_widget.py` or the
   `industry_intraday_rs_snapshot` contract.
8. **Fail-before-fix.** Every test ships shown failing on the un-built code
   (stash, run, unstash) — say so per test file in the handoff.
9. **Never break the tree.** The desk runs `launch_gui.py` from this checkout.
   Commit small and green (`.venv\Scripts\python.exe -m pytest tests/ -q`,
   check pytest's own exit code), push after each commit. Branch:
   `claude/group-tape-rebuild` from the current HEAD of `claude/gui-phase-0-9`
   (do not rebase onto `main`). Run `scripts/smoke_check.py` (7/7) before the
   final push. No packaging trigger is expected — a new module under
   `scripts/` or `scripts/ui/services` is collected; say so in the handoff.
10. Chat to me in very short, simple lines (CLAUDE.md "How to talk to the
    trader"). Depth goes in docs and commit messages.

## Facts you will need

- Sector ETFs: `DEFAULT_SECTOR_ETF_MAP` in `legacy.py` (~:359) — the 11 SPDRs
  (XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU). Copy the mapping
  into your module or read it through a tiny import-free constant; do not
  import `legacy` for it.
- Industry ETFs: `project_paths.INDUSTRY_ETF_MAP_FILE`
  (`C:\TradingBotData\data\industry_etf_map.json`),
  `yahoo_industryKey_to_ref[<industry>].etf`; 49 distinct ETFs for 136
  industries. Load it once per tick on the worker; a missing or unreadable
  file means sectors only, stated in `status_text`.
- Universe per tick: SPY + 11 sector ETFs + the distinct industry ETFs — about
  61 symbols, ONE `yfinance.download(..., period="1d", interval="5m",
  group_by="ticker", threads=True)` call. Yahoo rate-limits bursts (a diagnostic
  run hit `YFRateLimitError` on the 12th single-ticker call); one batched call
  every 5 minutes is well inside what the Strength Board already does every 15.
  Keep a per-tick cooldown and never retry inside the tick.
- The strip: `scripts/ui/widgets/group_tape_strip.py`. `update_groups(payload)`
  expects `{"group_strength": {<timeframe>: {"sectors": [{"etf", "rrs", ...}],
  "industries": [...]}}}` and ranks chips by the LAST entry of
  `SPARK_TIMEFRAMES`. Change `SPARK_TIMEFRAMES` to `("90", "60", "30")` (reads
  left-to-right "where it has been → where it is now"), rank by the 30-minute
  read, and rewrite `rotation_callout` as "up on 30 while still down on 90"
  and its mirror. Add the payload's as-of and the service's `status_text` to
  the callout line so a stale or failed read is visible, never silent.
- The mount point: `TradingDeskPanel.tape_host` / `self.group_tape`
  (`scripts/ui/panels/trading_desk.py`). Disconnect the old
  `rrsSnapshotChanged → group_tape.update_groups` wiring, connect your
  service's signal, and set the tape visible again. Shut the service down in
  the desk's shutdown list (bounded join — `tests/test_shutdown_waits_are_
  bounded.py` enforces it).
- Existing tests to keep green and extend: `tests/test_qt_group_tape.py`
  (chips, empty payload, click-to-chart, the desk wiring, and the
  "hidden on the desk" test — rewrite that one to assert it is SHOWN again and
  fed by your service, and say so).

## Packets, in order

### T-1 — `scripts/group_rrs.py` (pure) + parity

`session_rrs(symbol_bars, spy_bars, *, now, length)` → float | None over the
last `length` COMPLETED bars of TODAY's session, aligned on timestamps (drop
bars either side lacks — the old `_align_bars_with_map` idea), `None` when
either side is short. `rrs_windows(symbol_bars, spy_bars, *, now)` → `{"30":
…, "60": …, "90": …}` for lengths 6 / 12 / 18. Tests: parity with
`legacy.real_relative_strength` on identical bars; a gap-straddling series
yields `None` for windows that would cross it; the forming bar is excluded;
unequal timestamps are aligned, not misread.

### T-2 — `scripts/ui/services/group_tape_service.py`

The Strength Board shape. Tick every 5 minutes inside quiet hours; one batched
download; compute the three windows for every ETF against SPY on the worker;
publish `{"group_strength": {"30": {...}, "60": {...}, "90": {...}},
"as_of": <last completed bar stamp>, "source": "yfinance", "status": ...}` on
a signal. Last-good on failure with the failure in `status_text`. Manual
`refresh_now()` bypasses the gate. Tests: never runs on the Qt thread (the
download function is called from a non-main thread — assert on
`threading.current_thread()`); exactly one download per tick; a failed download
keeps the previous payload and states the failure; the quiet-hours gate holds
and `refresh_now` ignores it; shutdown is bounded.

### T-3 — the strip and the desk

`SPARK_TIMEFRAMES = ("90", "60", "30")`, ranking by "30", the new callout with
as-of + status, chips DIFF in place (keyed by ETF), hidden → shown, service
wired and shut down. Tests: the strip renders three segments from the new
payload and blanks a `None` window; the desk shows the tape and it is fed by
the service, not by `rrsSnapshotChanged`; the RS Window tab still receives
`rrsSnapshotChanged` unchanged.

### T-4 — docs and handoff

`CHANGELOG.md` entry; `plan.md` Phase 0.5 item 11 → BUILT with the live gate
owed (one DESK session: the tape moves every 5 minutes, the 06:30–07:00 read
carries no gap, a stale read says so); `CURRENT_CHECKPOINT.md` with the exact
suite count and each fail-before-fix; `CLAUDE.md` ("The group RS/RW tape is
HIDDEN…" bullet → what it is now) and re-copy to `AGENTS.md`;
`docs/README.md` if you add a Markdown file. Then tell me to restart the desk
— you do not restart it.

## Not in this prompt

- Industry = median member return (needs member bars — an IB-budget question).
- Any change to the 27-minute scan cycle (`rrs_scan` 1084 s over 302 symbols
  on 2026-08-27) — a separate finding, parked.
- Anything in `legacy.py`.
