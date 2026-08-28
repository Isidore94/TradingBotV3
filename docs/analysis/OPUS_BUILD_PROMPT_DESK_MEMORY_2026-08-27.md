# Build prompt for Opus: stop the desk jumping to 10 GB (2026-08-27)

Paste everything below the line into a fresh Opus session in
`c:\Users\Aaron\TradingBotV3`. The investigation it rests on is the
"2026-08-27 (10:00) - INVESTIGATION ONLY" entry at the top of
`CURRENT_CHECKPOINT.md`; nothing has been changed yet.

---

You are the BUILDER on TradingBotV3 (a Windows PySide6 trading desk, decision
support only). Read `CLAUDE.md` first and follow its mandatory documentation
workflow: `CHANGELOG.md`, `plan.md` sections 5-7 and 12, `CURRENT_CHECKPOINT.md`
(start with the **2026-08-27 (10:00) memory investigation** entry at the top),
`docs/README.md`, then `docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md` and
`docs/ULTIMATE_SETUP_DATABASE_PLAN.md` sections 8 and 19 (the lake's read/seal
contract and the bronze inventory). Chat with the trader in the CLAUDE.md
"five-year-old" style: very short, one idea per sentence. Detail goes in docs
and commit messages, not chat.

## The problem, already measured - do not re-investigate

The desk process climbs to **8-13 GB** after every hourly swing-scan slot and
falls back minutes later. It is not the scanner: a fresh desk stays at
0.9-1.3 GB through a whole BounceBot preamble. Three causes, in priority order.

**Cause 1 - the post-scan warehouse build runs INSIDE the desk and
materialises the whole month of M5 bars as Python dicts.**
`ui/services/scan_service.py` `ScanService._handle_finished` ->
`start_warehouse_build` -> `research_warehouse.cli.run_build` on a thread in the
desk process. Inside it, three steps do
`store.read_table("bar_m5", "month=YYYY-MM").to_pylist()` and then keep ONE
session in Python:

- `scripts/research_warehouse/aggregate.py:277` (`build_derived_bars`)
- `scripts/research_warehouse/features.py:809` (`build_intraday_snapshots`),
  plus `bar_derived` the same way at `:817`
- `scripts/research_warehouse/cli.py:328` (`_run_outcomes`)

Measured on the lake 2026-08-27: `silver/bar_m5/month=2026-08` =
**8,175,471 rows, 384 MB parquet, 151 files**; `to_pylist()` costs
**1,627 bytes per row = 13.3 GB** if the month is fully held. The lake
manifest (UTC) shows the 09:00 slot's build running 09:14:43-09:28:43 PT,
which is exactly when the desk read 10.7 GB working set / 12.8 GB private
(09:25:55) and then 2.5 GB (09:29:03). It grows every day because the
partition is month-keyed - 08-21 saw 8 GB, today 10.7, and September will
reset and climb again. `ResearchStore.read_table` / `open_dataset` are in
`scripts/research_warehouse/store.py:~795-810`; the `bar_m5` schema carries
`symbol`, `session_id`, `interval_start`, `interval_end` columns.

**Cause 2 - the bronze snapshot ingest loads the 1.03 GB
`master_avwap_setup_tracker.json` whole.** `ingest_existing.py:365-432`
(`ingest_artifact` -> `_snapshot_row`): `source.read_bytes()`, `.decode`,
`json.loads` of the whole file, sha of the whole bytes, then the full text is
published as one bronze `payload` row. It runs in the desk on any build where
the tracker's sha changed (today at 08:06 PT, after the 07:43 scan rewrote it)
and adds several more GB on top of cause 1. `parsed` is used only for
`_parse_event_at` and `_first_value(id_keys)`.

**Cause 3 - a slow leak in BounceBot.** `scripts/bounce_bot_lib/legacy.py`
stores every IB historical reply in `self.data[reqId]` and only the RRS path
(`request_historical_bars`, ~10250-10285) pops it. `request_and_detect_bounce`
(~12465-12499), `build_atr_cache` (~11425-11455) and the three
`check_dynamic_vwap*_touches` (~13701, 13733, 13767) `del` only
`data_ready_events[reqId]`, so the raw bar list (measured **206 KB per 390-bar
request**) lives until the process exits: ~80 MB per scan cycle, 1.5-2 GB over
a session, and it is why the desk settles at 2.5 GB instead of 1 GB after a
build releases.

## What to build

### 1. Session-scoped reads in the warehouse build (cause 1)

Add ONE store-level helper (name it clearly, e.g.
`ResearchStore.read_rows(dataset, partition, *, filter=..., columns=...)`)
that filters in Arrow BEFORE `to_pylist` - `open_dataset(...).to_table(filter=
pyarrow.compute expression)` - and convert the three readers to it, each
filtering to exactly the rows that step needs:

- `build_derived_bars`: the sessions in `session_dates` (by `session_id` or an
  `interval_start` range per session).
- `build_intraday_snapshots`: the one `session_date`, for both `bar_m5` and
  the two `bar_derived` timeframes.
- `_run_outcomes`: read it carefully - an outcome walk runs FORWARD over a
  horizon that can cross sessions and it already selects partitions via
  `_m5_partitions_for(known, day)`. Filter by the occurrence symbols (they are
  already known as `symbols`) and the date range the horizons actually need;
  do not narrow to one day if the walk needs more. Prove equivalence, do not
  assume it.

Do NOT move `run_build` into a child process in this packet. It is a
reasonable follow-up, but the in-process single-flight lock, the spool seal
and the ledger `_record_job` all assume one process, and the filtering fix
alone removes the growth. If you believe the subprocess move is still needed
after measuring, say so in the checkpoint and stop; the trader decides.

### 2. The tracker snapshot (cause 2)

Make the snapshot ingest cheap for very large files without changing what the
lake stores:

- compute the sha in chunks from the file, compare to the watermark, and
  return `UNCHANGED` before reading the whole file into memory;
- for files above a size threshold (pick one, e.g. 64 MB, and record it as a
  BD entry) skip `json.loads` - `event_at` and `legacy_id` come out empty, the
  row keeps `payload_format=json` and quality `COMPLETE` only if you can still
  establish it is JSON cheaply (e.g. first/last non-space byte); otherwise mark
  what you can prove. Record the choice and its reopen trigger in
  `docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md` as a new BD entry. Do not
  change the bronze schema or the publish path.

### 3. Free `self.data[reqId]` on every request path (cause 3)

**The trader authorises this `legacy.py` edit, limited to freeing
`self.data[reqId]` (and `reqid_to_symbol` where set) on the same lines that
already delete `data_ready_events[reqId]`, on both the success and the timeout
branches.** Nothing else in that file. This is not a detector change - the
bars are consumed into `all_bars` / `df` / `atr` before the pop - but treat it
as one for verification: the golden fixtures and every BounceBot test must
pass unchanged, and the `historicalData` callback must still tolerate a late
bar arriving for a popped reqId (it re-creates the entry today; make sure that
late entry cannot itself leak - either drop late bars for unknown reqIds under
the lock or bound them).

## Rules that bind you

- Tests first, fail-before-fix, for every one of the three: a fake store with
  two sessions where the reader must materialise no more than one session's
  rows (count rows through the helper, or assert with `tracemalloc` peak that
  it scales with one session, not the partition); a snapshot ingest test with
  a large synthetic JSON that asserts `json.loads` is not called above the
  threshold and `UNCHANGED` short-circuits before `read_bytes`; a BounceBot
  test that `bot.data` has no entry for the reqId after each request path
  (success and timeout). Show the failing run, then the passing run.
- Equivalence for the warehouse readers: on a synthetic multi-session
  partition, the rows published by `build_derived_bars`,
  `build_intraday_snapshots` and `_run_outcomes` must be byte-identical before
  and after (compare the published tables, not counts).
- Never let the warehouse touch a detector, score, alert, watchlist, Focus or
  the review queue; it stays shadow-only evidence. Never write inside
  `C:\TradingBotData` from the warehouse.
- Full suite green before every commit: `.venv\Scripts\python.exe -m pytest
  tests/ -q` (check pytest's own exit code), plus
  `.venv\Scripts\python.exe scripts/smoke_check.py` = 7/7. No packaging
  trigger is expected (no new dependency, asset, package or `__file__` use) -
  say so explicitly if true.
- Branch from the current `claude/gui-phase-0-9` head into
  `claude/warehouse-build-memory`; commit small and green; push after each
  commit. Do not switch the desk's branch or restart the desk yourself.
- Reconcile the control set before handoff: `CHANGELOG.md` (what changed and
  the measured before/after), `CURRENT_CHECKPOINT.md` (active item, verification
  result, the live gate owed), `plan.md` (add the item under Phase 0.9 with its
  live gate; retain the subprocess follow-up as a decision, not work),
  `docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md` (BD entries for the read helper
  and the snapshot threshold), `docs/README.md` if any Markdown file is added.
  Keep `CLAUDE.md` and `AGENTS.md` identical if you touch either.

## Measure, do not assert

Before the change, record the baseline you can reproduce offline: run the
three readers against the live lake partition in a scratch process with
`tracemalloc` and report peak bytes (expect ~13 GB for a full `to_pylist`; if
your machine cannot hold that, measure on a slice and extrapolate, and say so).
After the change, report the same peak for the same steps. Then the live gate,
which you record as OWED and do not mark met: after the trader restarts the
desk on the new commit, the first swing-scan slot's build keeps the desk under
3 GB working set (`Get-Process -Id <pid> | select WorkingSet64`, sampled
through the build window the lake manifest shows), the manifest still gains
the same datasets for that session, and the desk's baseline stops creeping
between builds.

## Out of scope - do not touch

The RRS scan's O(n^2) intraday profile (CPU, not memory), the
`_poll_focus_d1_interest` -> `FocusSideEditor.refresh` GUI stalls
(focus_picks_panel.py:441) and the RS-window `_auto_tick` reading 1,412
parquet files on the GUI thread (rs_window_panel.py / rs_window_feed.py:745)
were all observed in the same session. They are real, they are separate
packets, and they are not authorised here. Note them in the checkpoint as
observed, unchanged.

When you are done, end with the checkpoint's measured before/after table and
the one live gate still owed. Do not report completion until the suite is
green, the docs are reconciled and the branch is pushed.
