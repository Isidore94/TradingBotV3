# Wishlist

**SWEEP IN PROGRESS (trader, 2026-09-12 afternoon): every item below is being built as one feature
dump on branch `claude/wishlist-sweep-2026-09-12`.** The per-item status (packet, branch, MERGED tip,
lead decisions, deferrals) is the `CURRENT_CHECKPOINT.md` entry "2026-09-12 - WISHLIST SWEEP"; a new
session checks that table against this file and continues from the first row not MERGED. The item
texts below are the specs and stay as written; a `SWEEP:` line under an item's heading is the only
edit made here as it lands.

**Start here, Fable (2026-09-12): read 10K, then its linked items.**
10K is the trader's clarified integration plan: one measured daily review, faithful
market-thesis summaries, and a compact handoff for a frontier model. It connects
10D/E/F/I/J with items 5 and 7; their detailed contracts still apply. Item 11 remains
the separate workspace-memory work (built September 12). This plan starts no app build.
Older pasted instructions below describe their writing-date state, not today's build queue.
Recheck the checkpoint and code before acting on them; do not run their old cleanup,
merge or restart commands. Only trader-approved selections move into `plan.md`.

1. Finish up the GUI work

Resume Phase 0.22 (the desk reshape, G lane) as the lead session. Follow AGENTS.md and docs/AGENT_TEAM.md
(tester -> builder -> reviewer, own worktrees, scratch-script rule, chat to me in ten short lines).

Read first, in this order:
1. The memory file desk-reshape-plan-2026-09-06.md (auto-memory) - the full state, the merge idiom and the
   harness quirks.
2. CURRENT_CHECKPOINT.md's glance block, then plan.md Phase 0.22.
3. The packets .claude/packets/G4b.md, G5.md, G2b.md (each ends with a "Tester findings" section the
   builder must honour; G2b's findings may only exist in the tester's handoff - see step 4).

State when this prompt was written (2026-09-07 ~08:30 PT):
- main = origin/main at 6d9f4b43: ST1-ST7 (the other session, now closed) plus my G4, G0, G3, G1, G3b,
  G2a. Suite 7192 passed. The desk has NOT been restarted; a restart is my call.
- Red tests are pushed for G4b (claude/g4b-setup-tracker-detail 9879aa4a) and G5
  (claude/g5-research-results a79c7f20), both off 7e018c99. No builder has started on either.
- The G2b tester was still running in worktree .claude/worktrees/agent-adc6f553a1673a072 on branch
  claude/g2b-tracker-desk-away-columns and had NOT pushed: check that worktree for commits, push
  them, and if the tests are unfinished spawn a new tester for G2b.md.
- Gate numbers: mine continue from #85 (G2b takes #85; G4b and G5 next).

Do, in order:
a. Free the finished tester worktrees (git worktree remove -f -f; prune; delete any worktree-agent-*
   branch a refused spawn leaves behind - the "core.worktree redirect" refusal means remove that
   orphan worktree and retry).
b. Spawn builders: G4b (Opus), then G5 (Opus), then G2b once its tests are red and pushed. At most four
   Opus agents at once. Each builder: make the red tests pass without weakening them, merge origin/main,
   full suite with the AI lock probed and pytest's own exit code, ruff, smoke, selftest, docs per
   builder.md, push, handoff.
c. Reviewer by reproduction for G5 and G2b (trader-facing screens); G4b the lead may verify by running
   its tests and the revert proof.
d. Merge in the scratch worktree wt-g-merge on branch lead/merge-g (create it from main if it is gone),
   run the full suite there, then `git merge --ff-only lead/merge-g` from the DESK CHECKOUT
   (c:\Users\Aaron\TradingBotV3, clean first; my WISHLIST.md edits are mine - never touch them), push.
e. Then G7 (speed pass: defer first loads to first show - the Day-trade Tracker constructor read and
   Research's eager eight children; skip re-fits on unchanged data; re-measure against the G0 baseline
   desk_bench_baseline_2026-09-06.json in %LOCALAPPDATA%\TradingBotV3\diagnostics with
   scripts/ui/desk_bench.py; docs/GUI_FLUIDITY_MEASUREMENT_RUNBOOK.md section 7), and a small Journal
   packet (the Trades splitter opens 39/61 against its declared 3:2; the blank space above the table).
   Add nothing to CLAUDE.md (it is over its ~45 KB budget; trim in G7's docs pass).

Never edit master_avwap_lib/legacy.py without asking me first. Never run the bench with --platform
windows while the desk is up. Never restart the desk yourself. Report what landed on main, what is in
review, and whether a restart is owed.




2. Fixing some daily bare cache issues. the prompt is below

Fix the forming candles in the daily-bar cache (packet FC1).

Read AGENTS.md's workflow first: the CURRENT_CHECKPOINT.md glance block, plan.md
sections 5-7, CHANGELOG's inventory for "daily bar cache", decision 0016, and the
ST3 entry in docs/DESK_INTERNALS.md. Use the tester/builder/reviewer process in
docs/AGENT_TEAM.md; branch `claude/fc1-forming-candles` off main; work in
isolated worktrees; live stores read-only except the one repair named below,
which runs on a COPY first and on the live cache only with my go. Do not touch
CLAUDE.md. Keep chat short and simple.

## Verified facts (2026-09-06/07, ST3 review)

- The tracker's per-symbol daily-bar cache lives under
  `%LOCALAPPDATA%\TradingBotV3\machine_cache\daily_bars` as one CSV per symbol
  (`legacy.py:2871 _daily_bar_cache_file`), 1,980 files. The seed, quarantine
  and freshness helpers are at `legacy.py:3161`, `:3178`, `:3369-3415`; the
  Yahoo fetch counters at `:18697-18732`. Verify every line before editing.
- 100 of the 1,980 files end in an INVALID candle: the last row's open is
  outside [low, high] (99 open-only, 1 with the close outside too; low > high
  never occurs). Exactly one bad row per file, always the last row. Dates:
  2026-09-04 x89, 2026-09-02 x3 (DEI, HWM, SGHC), 2026-09-01 x3 (CBZ, HIW,
  SHW), 2026-08-20 x2 (CALY, HXL), 2026-07-07 (PRKS), 2026-06-08 (MCW),
  2026-05-15 (TERN). Example: AEE 2026-09-04 O=105.81 H=106.965 L=106.115.
- That signature is a FORMING session bar written into the cache mid-session
  and never replaced by the completed bar. Under the old replay default the
  bar was read for fills, excursions and marks; under gap_aware_v2 (default
  since decision 0019) the replay books nothing on an invalid bar and counts it
  in skipped_bar_reasons, but every other reader of the cache (the scan's D1
  indicators, the band history, the SMA floors, the session-horizon outcomes)
  still sees it as a real close.
- plan.md section 5: "State transitions use completed bars only. A forming bar
  is a labeled preview." `scripts/completed_bars.py` is the one completed-bar
  rule. The candle invariant is low <= open, close <= high.

## Required result

1. Find the writer. Name the function(s) that write a symbol's daily-bar cache
   and the source of the last row on each of the seven dates (the Yahoo fetch
   returns today's partial bar while the session is open; the cache refresh
   runs from the scan loop). State, with file:line, why the completed-bar rule
   was not applied at that seam and why a later refresh did not overwrite the
   row.
2. Never write a forming bar. The cache writer drops any row whose session is
   not completed per `scripts/completed_bars.py` (market-local, inclusive of
   the close) AND any row that breaks the candle invariant, counting each in
   the run manifest (`daily_bars_forming_dropped`, `daily_bars_invalid_dropped`)
   and logging one line per scan with the totals. A dropped row is logged with
   symbol and date at DEBUG. No reader change is needed if the writer is
   fixed; if a reader also needs a guard, say which and why.
3. Repair the 100 files. A CLI `python -m master_avwap_lib.daily_bar_cache
   repair` (or the existing `_quarantine_corrupt_daily_bar_cache` seam if it
   fits - say which) that scans every cache file, removes an invalid or
   forming LAST row, refetches that session's completed bar through the
   existing provider path with the desk's Yahoo pin, writes the file
   atomically, and reports symbol / date / old row / new row. Dry run by
   default; `--apply` writes; it prints project_paths.DATA_DIR first and
   aborts if a target is under C:\TradingBotData. Run it dry on the live cache
   and put the 100-line report in the handoff; run `--apply` only after I say go.
4. Say what the bad rows touched. On a COPY of the tracker mirror, count the
   setups whose replay window includes one of the 100 bad (symbol, date) pairs
   and whether the bar sat on a fill, a mark or a band level; the next
   persisted tracker write rebuilds every record, so no tracker repair is
   needed - state that and the count.

## Tests (fail first)

A forming bar for the current session is not written; a completed bar is; a row
breaking the invariant is dropped and counted; a refresh after the close
replaces a previously-cached partial row; the repair CLI removes only an invalid
LAST row, never an interior one, never a valid one; dry run writes nothing;
`--apply` writes atomically; the manifest fields exist; the counts reconcile
(kept + forming_dropped + invalid_dropped == fetched).

## Invariants

Completed bars only. Missing data is uncertainty, never confirmation. Yahoo
stays the pinned daily source. No detector, score or alert change; the cache
writer is not a scoring file, but if the fix has to sit in `legacy.py` the
ask-first rule applies - quote this prompt as the yes for the writer seam only.
Reconcile CHANGELOG, the checkpoint (gate #85 or the next free number - check
with the layout lane), DESK_INTERNALS; not CLAUDE.md.

Return what changed, the dry-run report, what the bad rows touched, and whether
a restart is owed.



3. Exit frameworks split by setup (packet EF1). the prompt is below

Split the Setup Tracker's exit-framework comparison by setup family (packet EF1).

Read AGENTS.md's workflow first: the CURRENT_CHECKPOINT.md glance block, plan.md
sections 5-7, CHANGELOG's inventory for "exit framework" and "M5", decision 0016,
and the M5 entry in docs/DESK_INTERNALS.md ("the control, study and
exit-framework populations"). Use the tester/builder/reviewer process in
docs/AGENT_TEAM.md; branch `claude/ef1-exit-frameworks-by-family` off main; own
worktrees; live stores read-only; scratch-script rule (TRADINGBOTV3_DATA_DIR set
BEFORE any import). Do not touch CLAUDE.md. Keep chat short and simple.

## Why (trader, 2026-09-08)

The trader asked whether taking profit at the 3rd band beats the 2nd for the
1st-dev breakout study. The tracker cannot answer: the exit-framework export
pools every scan row by (template, side, bucket) and never by setup, so "full at
band 3 loses" (LONG favorite: 47% win, -0.15 R, 46% stopped out, n_closed 8,018)
is a whole-population answer. A 1st-dev breakout starts one band from its
target; an AVWAPE bounce starts two. The right exit is probably per setup.

## Verified facts (2026-09-08)

- Writer: `legacy.py:13283 build_exit_framework_stats_rows`, grouped on the
  4-tuple at `:13316` (`framework_family`, `exit_template_id`, `side`,
  `priority_bucket`); columns `EXIT_FRAMEWORK_STATS_COLUMNS` at `:13260`; file
  `EXIT_FRAMEWORK_STATS_FILE` at `:427`; written in the guarded save pass at
  `:13538`. Verify every line before editing.
- The rows it groups come from `_flatten_tracker_scenarios` (`:7351`), and
  every flattened row ALREADY carries `setup_family` (`:7364`). The split is a
  key change, not a new reader - and a second scenario walker is forbidden
  (`test_band_variant_fence_guard.py`).
- Templates: `SETUP_EXIT_TEMPLATES` at `:1083` - `full_band2`, `full_band3`,
  `half_band2_trail_band1`, `half_band2_band3_trail_band1`, plus two
  `experimental` ones (`comparison_apr2026`). `experimental` is a COLUMN.
- Reader: `ui/panels/setup_tracker_panel.py` - `EXIT_FRAMEWORK_COLUMNS` (:283),
  the table built at :549, rows ranked by `_rank_exit_frameworks` (:1362) and
  capped at 300 (:1418), the population sentence
  `exit_framework_population_sentence` (:1816, reviewer-corrected wording:
  "the same setups MINUS the template's own filter").
- Live export today: 24 rows, 4 templates x 2 sides x 2 buckets (+ 8
  experimental). Splitting the SAME file by family would multiply it past the
  300-row cap and change the grain of a shipped table.
- Tests: `tests/test_m5_exit_framework_stats.py` (13 tests, including the
  byte-identical champion-aggregate pin and the "a raising export never costs
  the save" pin) and `tests/test_m5_setup_tracker_control_study_tabs.py`.
- Floor: `evidence_stats.MIN_REPORTABLE_N` (30). Win rate leads with n and the
  ONE Wilson bound; the sort is the bound (decision 0016).

## Required result

1. A SECOND export, `master_avwap_exit_framework_by_family.csv`, beside the
   pooled one, same columns plus `setup_family` FIRST, grouped on the 5-tuple.
   Built by the SAME builder with the grouping key as a parameter, so the two
   files can never disagree on a rate. The pooled file stays byte-identical
   (pin it). Written in the same guarded save pass; a raising by-family export
   never costs the tracker save or the pooled file.
2. The pooled row and its family rows reconcile: for every (template, side,
   bucket), sum of family `n`, `n_closed`, `wins`, `losses`,
   `n_expired_unmeasured`, `n_filtered_by_experiment` equals the pooled row.
   A family missing from the tracker's `setup_family` is grouped as
   `unlabelled`, never dropped, and counted.
3. Study and control families are IN the file with their `population`
   labelled (`champion` / `study` / `control`), read from the family's existing
   bucket/role, never inferred from the name. Confirm first (recon, file:line)
   that study records such as `1stdev_breakout` carry exit scenarios; if a
   population has none, the row is absent and the tab's sentence says so.
4. The Exit frameworks tab gets ONE control above its table: a family picker
   whose first entry is "All setups (pooled)" - the current table, unchanged -
   and then every family in the by-family file, sorted by name. Picking a
   family shows that family's rows only (4 baseline + experimental, 2 sides x
   2 buckets), ranked by the same Wilson bound. The population sentence names
   the family, its `n_closed`, and prints `below floor` when every row is
   under `MIN_REPORTABLE_N`. Below-floor rows are shown, never hidden; nothing
   is sorted by mean R.
5. Shadow only. No detector, score, alert, template, stop rule or default
   exit change. Nothing here promotes a template; T4's criteria decide.

## Tests (fail first)

Two setups in two families produce family rows whose sums equal the pooled
row; the pooled file is byte-identical with and without the new export; a
family-less setup lands in `unlabelled`; a study family is labelled `study`;
the save pass writes both files; a raising by-family export costs neither the
save nor the pooled file; the picker's first entry renders the pooled rows
unchanged (golden); picking a family filters to that family and sorts by the
bound; a below-floor family prints `below floor` and still shows its rows;
the 300-row cap is never hit by a single family's view.

## Invariants

The writer is in `master_avwap_lib/legacy.py`, a scoring file: the ask-first
rule applies - quote this prompt as the yes for the exit-framework export seam
ONLY (`build_exit_framework_stats_rows`, its columns, its file constant and the
save-pass call), nothing else in that file. One scenario reader. Win rate
first, n and the Wilson bound beside it, the sort is the bound. Counts are
integers at the table's own grain. Reconcile CHANGELOG, the checkpoint (next
free gate number - #90 or later, check the glance block), DESK_INTERNALS's M5
entry, and docs/README.md if a doc is added; not CLAUDE.md.

Return what changed, the reconciliation counts from a COPY of the live tracker
mirror, the 1st-dev breakout family's four baseline rows, and whether a
restart is owed.


4. Keep the desk snappy all day (packets SN1-SN6). the prompt is below

Cut the M5 scanner's hold on the interpreter so the desk stays snappy at the
close, without losing any scan, alert, board or evidence row (packets SN1-SN6).

Read AGENTS.md's workflow first: the CURRENT_CHECKPOINT.md glance block, plan.md
sections 5-7, CHANGELOG's inventory for "thread_cpu", "stall watchdog",
"SignalCoalescer" and "RRS", decision 0016, the F1 warehouse-child entry and
"Every ticker click lands on the centre chart" in docs/DESK_INTERNALS.md, and
the memory file desk-lag-at-close-2026-09-08-run-strategy-gil.md. Use the
tester/builder/reviewer process in docs/AGENT_TEAM.md; one branch per packet
off main (`claude/sn1-scanner-process`, `claude/sn2-incremental-bars`,
`claude/sn3-one-rrs-pass`, `claude/sn4-feed-diff`, `claude/sn5-symbol-breath`,
`claude/sn6-fast-lane-order`); own worktrees; live stores read-only;
scratch-script rule (TRADINGBOTV3_DATA_DIR set BEFORE any import). Do not touch
CLAUDE.md. Keep chat short and simple. Report artifact:
https://claude.ai/code/artifact/07dfd0a3-1487-4645-b25a-b73c9dce96df

## Why (trader, 2026-09-08)

The desk was "really quite laggy" at the close. Measured live at 13:18 PT: the
bounce scanner thread `Thread-4 (run_strategy)` held 0.62 of a core on average
in hour 13 and 71-88% per minute at the close, climbing from 0.35 at 06:00; the
GUI thread got 0.10-0.15. 13,031 GUI stalls over 50 ms that day (four hours hit
the 2,000-record cap, so undercounted), 245-365 blocked seconds per hour, 253 s
in the first 18 minutes after the close, one 30.6 s stall at 13:01:53.

## Verified facts (2026-09-08)

- The scanner is a THREAD in the desk process: `bounce_bot_lib/legacy.py:13597
  run_strategy`, started from `ui/services/bounce_service.py`. The GUI shares
  the interpreter lock with it; the stall culprit is `app.py:1345 main` (the
  event loop waiting for the lock) with almost no other samples.
- Cycle 24 preamble (12:56): 658.4 s = focus_fast_lane 332.7 s + rrs_scan_5m
  99.8 s + rrs_scan_15m 64.7 s + rrs_scan_1h 61.7 s + rrs_scan_gui 49.0 s +
  14 other 50.6 s; then "Monitoring 328 strongest/weakest symbols"; cycle 25
  started 13:10, so a cycle is ~25 min and `wait_for_candle_close` at
  `:13850` returns at once - the loop never rests while scanning is enabled.
- The fast lane (`:13566 _scan_human_focus_fast_lane`) scanned 258-259 names,
  alphabetically: `focus_auto_picks.json` 107 auto-adopted picks +
  `focus_longs.txt` 64 + `focus_shorts.txt` 129. Trader picks and auto picks
  share it with no ordering.
- `request_and_detect_bounce` (`:12580`) requests "5 D" of 5-minute bars from IB
  for EVERY symbol EVERY cycle (~206 KB, ~390 bars), waits up to 15 s, then
  `_dedupe_bars(_bars_to_ib(...))`, `pd.DataFrame`, `pd.to_datetime` over all
  of it. 586 symbol scans a cycle = ~120 MB fetched and ~230,000 rows parsed
  per 25 minutes.
- `run_rrs_scan` (`:10795`) runs FOUR times per cycle (5m, 15m, 1h, then the
  GUI's selected timeframe, which is one of the three) and each run walks the
  full universe, rebuilds `_build_intraday_rrs_profile` from the same five days
  of bars, aggregates, and aligns sector and industry ETFs. 275 s of pure CPU
  per cycle.
- `alert_center_panel.py:5640 _ignore_alert_symbol` calls `:1980
  _rebuild_feed`, which destroys and reconstructs up to MAX_FEED_ITEMS (250)
  M5 and MAX_D1_FEED_ITEMS (100) D1 `AlertFeedItem` widgets on the GUI thread
  after every veto (4.0-4.1 s each at 13:16) and on every coalesced
  `focusChanged` (24.2 s at 13:01:22). The feed already keeps `_feed_rows`
  keyed on (symbol, side). The comment at `:737` names this as the "rebuild,
  not diff" exception under the ask-first rule.
- Not the cause: the 13:00 close slot's tracker-write scan child (3.9 GB WS)
  and `analyze_master_avwap_scoring.py --apply` (5.9 GB WS, 100% of a core)
  squeezed free RAM to 3.6 GB of 29.5; pages/sec 0, disk 0.9%.
- `trading_bot.log` rotates at ~1 MB and held only hours 12-13; the durable
  evidence is `%LOCALAPPDATA%\TradingBotV3\diagnostics\thread_cpu.jsonl` and
  `ui_stalls.jsonl`.

## Required result

1. SN1 - the scanner in its own process. `run_strategy` runs in a below-normal
   CHILD PROCESS owned by `bounce_service` (the F1 warehouse-child recipe, never
   a thread); the desk reads its alerts, RRS snapshots, cached M5 bars and
   Focus adoptions over the existing files plus one queue; every GUI callback
   crosses the boundary as a message. Same detector, same bars, same alerts,
   same timing; the child dies with the desk and never outlives it; one child
   per desk. This is the biggest change and lands LAST, after SN2-SN6 have
   proven themselves on a live day.
2. SN2 - new bars only. Fetch the five days ONCE per symbol per day, keep the
   frame, then request only bars since the last completed one; the day roll
   and a bar-identity check (last bar's dt and close) refetch the whole window
   when the delta does not line up. The frame the detectors read is IDENTICAL
   to a fresh fetch - a golden fixture pins fresh == cached-plus-delta. A
   partial last bar is the fault to test.
3. SN3 - one RRS pass. One walk of the universe produces the 5m, 15m and 1h
   results from the same aligned 5-minute series; the GUI pass reuses the
   matching timeframe's result; `_build_intraday_rrs_profile` is cached per
   symbol keyed on its last bar's dt. The three per-timeframe results are
   BYTE-IDENTICAL to today's four-pass output (characterization test on a
   recorded bar set).
4. SN4 - diff the feed. A veto removes THAT row from `_feed_rows`; a Focus
   change restyles the star (`focusOn` property) on the rows it touches; a new
   alert inserts one row; the digest row and the repetition fold update in
   place. `_rebuild_feed` remains for the tier-mode switch and the day roll
   only. The visible rows after any sequence of veto / focus / new-alert are
   identical to a full rebuild (the test compares both).
5. SN5 - breathe. `_stop_event.wait(0.02)` after each symbol's compute in the
   fast lane and the main sweep. Pacing only; nothing produced changes. One
   line, first to land, the stopgap until SN1.
6. SN6 - order the fast lane. The trader's own Focus names (no marker in
   `focus_auto_picks.json`) are scanned first, then the auto-adopted ones, then
   the sweep; the SET is unchanged, every name still scanned every cycle.

Order of landing: SN5, SN6, SN4, SN3, SN2, then SN1. Each packet measures
itself: `thread_cpu.jsonl`'s `run_strategy` core fraction and `ui_stalls.jsonl`'s
blocked seconds per hour on the next live day, against 2026-09-08's numbers
above.

## Tests (fail first)

- SN2: fresh-fetch frame == cached-plus-delta frame for a recorded symbol,
  including across a day roll; a partial last bar triggers a whole-window
  refetch; the IB request count per symbol per cycle drops from 1 x "5 D" to 1
  x delta after the first cycle of the day.
- SN3: the three timeframe outputs from one pass equal the recorded four-pass
  outputs byte for byte; `run_rrs_scan` is entered once per cycle.
- SN4: after veto / focus toggle / new alert / repeat, the feed's visible rows
  (symbol, side, order, star state, repeat badge) equal a full rebuild's; a
  veto destroys ONE widget, not the layout; the coalescer seam still fires once.
- SN5: `ScanCycleClock` still never sleeps; the wait is on the stop event so
  shutdown latency is unchanged.
- SN6: the fast-lane order is trader picks, then auto picks, then everything
  else; the scanned set equals today's.
- SN1: the child starts below normal priority, dies with the desk, reconnects
  the desk to its queue after a child restart, and a full recorded cycle's
  alerts, RRS snapshot and Focus adoptions match the in-thread run.

## Invariants

SN1, SN2, SN3 and SN5/SN6 edit `bounce_bot_lib/legacy.py`, a detector file, and
SN4 edits `alert_center_panel.py`, an alert file: the ask-first rule applies to
each - quote this prompt as the yes for the named seams ONLY (`run_strategy`'s
loop pacing and fast-lane ordering, `request_and_detect_bounce`'s fetch and
cache, `run_rrs_scan`'s pass structure, `_rebuild_feed` / `_insert_item_into` /
`_ignore_alert_symbol`), nothing else in those files. No detector, score,
threshold, alert gate, tier, fold or evidence row changes; golden fixtures
before any bar-frame or RRS change (plan.md sections 5 and 7). Completed bars
only; a forming bar is preview. The sigma formula of `calc_anchored_vwap_bands`
is untouched. Manual buttons are never gated. Reconcile CHANGELOG, the
checkpoint (next free gate number - check the glance block), DESK_INTERNALS (a
new entry "SN - the scanner's hold on the interpreter"), and docs/README.md if
a doc is added; not CLAUDE.md.

Return what changed, the before/after `run_strategy` core fraction and blocked
seconds per hour from a live day, the IB request count per cycle, and whether
a restart is owed.


4. Rank the Master AVWAP setups by a POINT system (trader, 2026-09-08) - IN BUILD

Approved 2026-09-08 ("put it in wishlist.md then start working on it block by block").
Presentation only, the ST6 pattern: a switch that REORDERS the setups table and never
hides a row; legacy.py untouched. Favourite + near (+ high conviction) buckets are ranked,
every other row keeps its order after them. Blocks are removed here as they land on main.

- Blocks 1-3 (module, Points column, switch, all four inputs) and the docs: DONE on main
  2026-09-08. Restart the desk to see it (gate #90).
- Tracking + self-correction: DONE on main 2026-09-08. The log, the grade line on the setups
  status row (terciles by points vs the tracker's 5-session outcomes), and a proposed weight
  file. The desk uses the proposed weights only when you tick "... > Points: learned weights".
  Tick it once the grade line shows a lift and each third has 30+ rows.
- Block 4 (open, your call): rank the AWAY digest's swing picks by points too (today it
  ranks by the family's Wilson bound). Tune the weights after a few desk days - they are
  named constants at the top of scripts/setup_points.py.


5. Close the preference-learning loop - Fable 5.1 decision queue (2026-09-09)

STATUS: PROPOSED ONLY. The trader asked Codex to place this assessment and suggested
fixes here so Fable 5.1 can decide what to do. This is NOT permission to implement
these blocks or change live policy. Fable should select, narrow, combine or reject
blocks; move only trader-approved work into plan.md before Claude builds it.
Do not treat imperative wording in the proposed fix instructions as build approval.

Goal
Help the trader align what they like with what actually works, and challenge dislikes
that reject good opportunities. Keep four facts distinct: interest in a name, a claimed
setup, a decision to pass/reject, and an executed trade. A like is a hypothesis, not proof
of edge. A later rising price does not by itself prove a rejected entry was sensible.

Read narrowly before choosing or building
- CURRENT_CHECKPOINT.md glance and the relevant gates; plan.md Sections 5-7,
  Phase 0.13 P1/P5/P6/P9/P10 and Phase 0.22 ST5/ST6; search CHANGELOG inventory
  for those features. Much of this is already built: do not create a second pipeline.
- docs/REVIEW_LEARNING_LOOP.md, relevant sections of docs/LOCAL_AI_AUTOMATION_PLAN.md,
  docs/CHART_REVIEW_WORKSPACE_PLAN.md, docs/WEEKEND_PREP_PLAN.md, and the matching
  entries in docs/DESK_INTERNALS.md before changing their behavior.
- Follow AGENTS.md and docs/AGENT_TEAM.md for any approved implementation, including
  tests before fixes and independent reproduction on copies. Recheck source locations.
  The file-scoped ask-first rule still applies to detector/scoring/alert files; this
  wishlist entry is not that approval. No automatic restart or live data repair is implied.

What already works - verified on main ff4126aa, September 9
- Quick and claimed likes reach trader_annotations.jsonl; quick likes retain
  like_mode=quick and grade as like_unclaimed. Claimed likes name a setup. Missing
  legacy mode reads claimed. Neither proves a trade was taken.
- Live annotation snapshot: 115 likes (26 quick, 27 explicit claimed, 62 legacy),
  373 vetoes, one pass. pick_feedback.jsonl: 600 likes, 35 dislikes, 329 not_today,
  164 unfavorites. swing_favorites.jsonl: ten add events. These are event counts,
  NOT independent opportunities or a common denominator.
- September 8 overnight ledger confirms successful veto/like/pass/rejection grading,
  sidecar completion, preference-to-trade report, evidence report and narrated digest.
  Like outcomes: 94 rows, 55 with h3 returns; veto outcomes: 320 rows, 201 with h3.
  Four pass outcome rows belong to ONE pass with overlapping cohorts, not four passes.
  Different dates and grains mean raw capture/outcome count differences are not loss rates.
- ai_summary.py trader_judgement scope includes veto, like, pass and rejection
  performance plus annotations. The machinery can inspect both endorsements and
  rejections. Packages are bounded; available data does not prove every row was narrated.
- Working-lately and Setup Tracker provide the independent "what works" side.
  Points grading and trader-gated learned weights already exist. Reuse them; do not
  add preference bonuses to points or detector scores as an incidental part of this work.

Proposed block A - Repair the weekend verdict first (confirmed defect)
Owner seams: scripts/weekend_verdict.py best_cohort_line/build_verdict;
scripts/ui/panels/weekend_prep_panel.py _read_like_cohort/_read_veto_cohort/_read_verdict.
The readers return horizon, n and avg_return (formatted percent). The verdict asks for
avg_r_h3 and n_h3, then prints R. The actual reader-to-verdict call was reproduced:
120 like performance rows, eight h3 rows with n>=8; 101 veto rows, fifteen h3 rows with
n>=8; BOTH verdict lines incorrectly say "nothing with enough behind it yet." Eligible
cells include pooled/side-specific rows, not eight/fifteen independent strategies.

Claude's fix instructions IF SELECTED:
Use a typed/raw numeric report contract shared by the card and readers; format only at
the display edge. Select the intended horizon and side explicitly. Do not convert percent
returns into R. "Weakest veto reason" currently uses min(return); the reasons worth
questioning for missed gains are the rejected names with HIGH side-adjusted returns.
Choose wording and polarity together; exclude overall pools from a ranking of reasons.
Keep thin/unknown results explicit and reuse evidence_stats floors.
Tests: real-shaped CSV -> actual reader -> verdict with mature and thin cells; correct
horizon/side; positive and negative returns on both sides; highest rejected gains identified
as missed opportunities; overall pools excluded; no percentage labelled R. Prove failure
before the fix, then reproduce with staged live files. Do not merely rename test fixtures
to match the old broken assumptions.

Proposed block B - Make the actual-trade comparison symmetric
Owner: scripts/preference_trade_outcomes.py and its existing Weekend Prep consumer.
The live report has 647 statements: 527 feedback likes, 110 annotation likes, nine swing
favorites and one pass. Sixteen statements match 12 distinct trades inside ten sessions.
It currently excludes veto/not_today/dislike statements, although those are graded
elsewhere. It also does not retain quick/claimed mode in a dedicated report column.

IF SELECTED: extend this report, not a parallel store. Include each explicit rejection
with its original source identity, reason/version and side/horizon. Preserve quick versus
claimed likes. Retain match confidence and report ambiguity rather than asserting a
symbol/date coincidence is the same thesis. Keep P&L summed once per trade even when
several statements match it. Separate "window still open," "no match after the window,"
and "journal/matching unavailable"; none alone proves a deliberate skip. Keep separate
verdicts separate and never turn unfavorite into a negative judgement.
Tests: all channels, legacy modes, opposite-side/same-symbol decisions, multiple statements
per trade, incomplete ten-session windows, calendar uncertainty, duplicate prevention,
retractions and missing journal evidence. Reuse ST5 trade_level_summary and exposure rules.

Proposed block C - Verify the full night and connect its coaching inputs
Owners: scripts/ai_jobs/runner.py, scripts/ai_summary.py, existing digest/evidence readers,
and the existing status/Weekend Prep surfaces. Identify the smallest seams before editing.
At inspection the latest ledger ended after September 8 daily_digest success. It did NOT
prove completion of that night's later ai_summary, journal_enrichment or policy draft.
The local morning brief was last modified September 5; ai_trade_enrichment had zero rows.
This is a publication/coverage question, not a proven scheduler or model failure.

IF SELECTED: inspect the runner, process, logs and output timestamps to find the actual
cause before fixing anything. Show stale or missing advice explicitly. No direct consumer
of preference_trade_outcomes.csv was found in the searched ai_jobs/ai_summary code;
verify that gap, then feed a bounded deterministic summary with source pointers and match
coverage into the existing AI package. Job success counts alone are not the joined evidence.
The optional weekly_synthesis slot is explicitly NOT in default_slots; decide its cadence
rather than assuming it already runs automatically.
Tests: deterministic grades survive narration failure, stale advice is labelled, partial
coverage named, source references resolve, bounded selection never ranks by favorable
results, and the model cannot write live policy, watchlists, scores or alerts. Preserve
stage ordering and digest audit gates. Do not enable a model job or schedule as a side effect.

Proposed block D - Capture watchlist intent without inventing it
Plain swinglongs.txt/shortswings.txt are not the same as dated Today's swing favorites.
WatchlistEditorPanel._write_symbols writes membership and notifies its peer; it does not
write a dated preference/reason event at that seam. Existing AI membership evidence cannot
reconstruct exactly when or why a manual change happened.

Integration suggestion: a dated append-only add/remove event for a TRADER edit, linked to
the existing capture/evidence system. Preserve source, side, observation time and optional
reason without prompting on every edit. Membership means interest, never a setup claim or
position. Keep machine injection distinct. Do not invent historic addition times, convert
removal into dislike, or let an evidence append failure prevent the watchlist save.
Before selecting a design, inventory every writer (including edits outside the app),
deduplication and machine provenance. An observed external diff is labelled as such, not
asserted to be a trader decision. Reuse an existing schema if it expresses this honestly.
Tests if built: manual vs machine writes, external unknown provenance, unchanged saves,
add/remove/re-add, partial failures, and preservation of hand-entered symbols.

Proposed block E - Fix identity and journal coverage before strong claims
review_learning.build_episodes still groups by (trade_date, symbol), allowing distinct
sides or D1/M5 theses to merge. The ordering gate is annotation-only for a reason. Align
any identity repair with the existing canonical-opportunity roadmap; do not mint a competing
ID solely for a new report. Characterize current aggregates and quantify the restatement
on copies before changing joins or lifting a gate.

Journal snapshot: 205 trades; annotation tags: one confirmed, 27 provisional, 145
needs_review; zero positive planned-risk entries. All 12 matched trades lack planned risk.
The import ledger also reports 23 position mismatches and 198 unresolved self-heal items;
those counts were NOT audited in this assessment and do not prove 198 bad trades.
Suggestions: expose and resolve the existing tag backlog, capture planned risk prospectively,
and investigate reconciliation coverage through the journal's existing tools. Do not guess
old risk from an outcome, confirm machine tags automatically, or treat missing data as zero.
Use existing status partitions and name exclusions before comparing personal performance.
Tests for any selected fixes must preserve append-only broker authority, trade-level money,
trader-owned tags, option exposure semantics and explicit unknowns.

Proposed block F - One recurring coaching view, built from existing evidence
Integrate into Weekend Prep/Research Results and the existing overnight package rather
than another dashboard, model job or evidence ledger by default. Fable should decide
whether this belongs nightly, weekly, or both with one shared computed source.

Four questions:
1. What I like that works.
2. What I like that fails.
3. What I reject that was worth rejecting.
4. What I reject that deserves another look.

Join those observations to the SAME Working-lately/Setup Tracker evidence snapshot used
on the desk. Each lesson names the setup/reason, side/horizon, sample size, coverage,
uncertainty and source pointers. Distinguish actual trading outcomes from paper returns.
Compare suitable contemporaneous opportunities, not different regimes or entry policies.
A timing rejection needs an intraday path; a risk objection needs adverse movement and
entry/stop knowledge. Positive h3 return alone does not refute either. Repeated clicks and
overlapping reason cohorts cannot inflate sample size. An unmatched trade link is not a
certain missed trade. Counterexamples and thin evidence must be surfaced alongside wins.

The output should propose a small testable habit, then measure it on fresh sessions under
a predeclared window and outcome definition. It must not learn merely to flatter the trader
or bury disliked setups. No live rank/policy promotion until the existing Section 7 gates
and an explicit decision are satisfied. Tests: conflicting likes/results, rejected winners,
thin/missing groups, opposite sides, repeated statements, incomplete windows, stale tracker
snapshots, and narration that cannot introduce ungrounded numbers or trading instructions.

Suggested order and Fable 5.1's decision
A first; C's runtime investigation can precede any build; B then explicit C integration;
choose D/E only after checking the existing roadmap; F only over trustworthy joined evidence.
Fable should give the trader a short selection and explain what already exists, what is
broken, what each selected block changes, and what remains evidence-gated. Mark deferred
blocks here. For approved work, advance plan.md and reconcile checkpoint/CHANGELOG/specs
per AGENTS.md; do not create another roadmap or committed assessment file. Leave unrelated
wishlist entries and the trader's pre-existing edits intact.


6. Theta pick tracker - a Setup Tracker for the theta plays (trader, 2026-09-10)

STATUS: CANDIDATE. Brief only; plan and build later when the trader points Claude here.

Goal: know which theta picks work, which support combos and relative-strength reads
predict it, and whether the current theta scoring is any good.

- What exists: the D1 scan writes `master_avwap_theta_puts.txt` (theta rows + put credit
  spreads) from `thetalongs.txt`; rows carry support labels (SMA_50/100/200, current and
  previous AVWAPE, previous first-dev), a support quality score (`_theta_support_quality`),
  daily/industry RS bonuses and the IB option quote. Nothing records what happened AFTER.
- Track, per pick and per day it appears: symbol, scan date, support set (which labels held
  and their distance in ATR), RS flags, the score and rank, the chosen strike/expiry/premium.
  Grade at the sold put's expiry (and at 5/10/20 sessions): did price hold above the strike,
  max adverse excursion in ATR, which supports broke first.
- Readouts, the Setup Tracker way: win rate with `n` and the ONE Wilson bound, cut by
  support combo and by RS flag; a grade line for the score (terciles by score vs realized
  hold rate, like the point-system grade line) so the trader can see if the scoring is
  working. Shadow only - never changes the scan, the score or the report.
- Home: a new CSV store beside `master_avwap_tier_outcomes.csv`, written in the tracker's
  guarded save pass; a Theta tab in the Setup Tracker. Reuse `evidence_stats` for the
  window and `swing_headline` for the bound; no new statistic.
- Open questions for the trader: grade at expiry only, or also mark-to-market per session?
  Does a pick that repeats on several days count once (first appearance) or once per day?


7. D1 market environments, tracked like the M5 ones (trader, 2026-09-10)

STATUS: CANDIDATE. Brief only.

Goal: relate the setups we track to the daily market environment - a compressed D1
market and a trending D1 market should show up as separate cells, the way M5 alerts
already carry `market_environment` into `held_run_score`.

- What exists: M5 has a per-alert `market_environment` label recorded by the alert
  context (BounceBot) and read verbatim by `held_run_score` - never re-derived. The D1
  side has `market_state.py` (shadow, JSONL evidence, REGIME_FAILED/RANGE states) but no
  daily label joined to the swing outcomes; `master_avwap_tier_outcomes.csv` rows carry no
  environment column.
- Build: one D1 environment label per session for SPY (and maybe QQQ/IWM), from completed
  daily bars only: `compressed` / `trending_up` / `trending_down` / `unknown`, with the
  rule in one pure module under `scripts/indicators/` (range vs ATR over N sessions, slope
  of a moving average, or the band width - pick ONE, name it, version it). Append-only
  daily store; never backfilled from a different rule without a version bump.
- Join: stamp the label onto every tracker outcome row at SCAN time (point-in-time - the
  label of the scan date, never the exit date). Then the Setup Tracker, the Working-lately
  snapshot and the point-system grade line can cut by environment. Unknown is its own cell,
  never pooled.
- Invariants: shadow evidence, zero detector/score/alert influence, calendar walked in
  sessions, golden fixture for the label rule before anything reads it. Ask first before
  touching `legacy.py` for the stamp.
- Open question: should the environment also gate which weights the point system learns
  (one weight set per environment) or only label the readouts? Label first.


8. Setups table star and X reflect the day's decisions (trader, 2026-09-10)

STATUS: CANDIDATE. Brief only. Presentation only - hides nothing, writes nothing new.

- Star (★): today it is filled only for Focus picks (`setup_delegate.set_focus_lookup`).
  Make it filled for anything in Focus AND anything the trader LIKED today (quick or
  claimed like, from the same ledgers `pick_feedback.reviewed_symbols_today` unions).
  Tooltip says which: "In Focus" / "Liked today" / both.
- X (✕): today it is a dim, always-drawn dislike button. Paint it BRIGHT RED when the
  trader has checked that symbol today and vetoed, disliked, passed or skipped it (any
  reject-family decision for that symbol on this trade date; a click away from an M5
  alert IS a pass and counts). Tooltip says which decision and when.
- Seams: the delegate reads one per-symbol lookup built off the Qt thread (the same
  cached, mtime-keyed read `reviewed_symbols_today` already does, split into `liked` and
  `rejected` sets); a new theme token for the red; the row is never hidden, re-ordered or
  muted by this - the movers-only filter and the priority switch are untouched.
- Refresh on the day roll and after each capture verb, coalesced (one repaint per burst).
- Open question: does a symbol both liked and vetoed today (e.g. "Veto D1 - but M5 today")
  show both marks? Proposed: yes, both - the two columns are independent facts.


9. Stop putting up longs below AVWAPE and shorts above it (trader, 2026-09-11)

STATUS: CANDIDATE. Brief only; plan and build later when the trader points Claude here.

Trader's words: "stop putting up longs below avwape and shorts above it."

- Meaning: a LONG candidate whose price sits under its AVWAPE line, or a SHORT candidate
  whose price sits over it, is on the wrong side of the anchor and should not be shown
  as a pick.
- Open questions for the trader: which surfaces - the D1 scan output, the setups table,
  the AWAY digest, the M5 alerts, or all of them? Hide the row, or keep it and mark it
  "wrong side"? Which AVWAPE - the current anchor only, or the previous one too?
- Any change here touches detector/scoring output, so the ask-first rule and the golden
  fixtures (plan.md sections 5 and 7) apply before anything is built.


10. Better daily discovery, entry watches, journals and review — Fable 5.1 queue (2026-09-11)

STATUS: PLANNED CANDIDATES. The trader asked for plans in this file, not implementation.
No new product phase is authorized by this entry. Fable can investigate and prepare a
concrete first selection; move only explicitly approved work into `plan.md` before building.
The governing process is AGENTS.md + docs/AGENT_TEAM.md, plan.md sections 5–7 and
decision 0016. Before code changes, read the matching DESK_INTERNALS entry, characterize
the current behavior, then use tester → builder → independent reviewer in isolated
worktrees. File-scoped detector/scoring/alert approval still applies to a selected build.
No desk restart, live-store repair or provider switch is part of this planning request.

Request map: 1 → A; 2 → B; 3/4/5 → C; 6 → D; 7 → E; 8 → F;
9/10 → G; 11 → H. This merges overlapping work without dropping any request.
Source anchors below were inspected at `main` 7635d09a; line numbers are starting points.
The existing item 9 was already uncommitted and is preserved. No live-session diagnosis
or new runtime baseline is claimed by this planning pass.

### 10A. Find useful D1 candidates before the final hour (request 1)

Trader observation: the strongest Master AVWAP updates seem to arrive in the final hour
or at EOD, even when the app says it updated earlier. Treat this as an unresolved report
of freshness AND usefulness; a recent file timestamp proves neither fresh input bars nor
good discovery. Do not assume the scheduler is the cause or promise an earlier EOD signal.

First packet: trace one candidate end to end at the open, midday, final hour and close:
scheduled/requested time → actual start → universe/side → provider/cache → last input bar
and completed/forming status → detector result → export publication → desk read/render.
Use exchange-session boundaries, including early closes. Compare scan manifests, logs and
report contents, not just mtimes. Daily bars are intentionally pinned to Yahoo; IB serves
intraday. Identify which earlier runs reused yesterday's completed D1 inputs, which used
a labelled current-session preview, and which failed, queued or published old output.

Then split any fix into (1) truthful freshness/publication and (2) discovery coverage.
Show last successful scan separately from latest input-bar time and latest shown result.
Preserve the last good report on failure, labelled stale. Check existing early/intraday
slots before adding a timer. An intraday candidate may be a preview, but only completed
bars can confirm a state transition. Do not relax the champion's rules to manufacture
more names; use the existing intraday monitoring path where D1 confirmation is unavailable.

Acceptance: replay several recorded sessions with only data known at each checkpoint.
For names that later moved, record first eligibility, first publication, delay and reason
for absence; also include failed candidates, false positives and total universe coverage.
Maximum later movement is a retrospective measure, never an input to the earlier scan.
Freeze the comparison window/rules first; separate an eligible name delivered late from
a pattern that only became valid late. If historic intermediate snapshots do not exist,
collect them prospectively and label the missing proof rather than reconstructing certainty.
Tests: stale cache/new mtime, failed publish, queued runs, partial universe, timezone/early
close, forming-bar preview, last-good preservation and GUI responsiveness. Live gate:
one complete session trace and then representative day-part comparisons, not one good close.

Related work: item 2 daily-cache integrity; item 4 scanner pacing (SN5/SN6 already built,
SN1–SN4 still candidates); item 9 wrong-side AVWAPE proposal is a separate decision.
Governing docs: BROKER_ADAPTERS, AUTO_MODES_AND_QUIET_HOURS, SWING_QUALITY_AND_FEEDBACK,
GUI_FLUIDITY_MEASUREMENT_RUNBOOK. Recon anchors: `master_avwap_lib/runner.py:886`
calls the daily fetch and `:2569` writes the post-scan watchlist output;
`master_avwap_lib/legacy.py:3402`/`:3415` check cache day/mtime, and `:2128` owns the
last-hour/close windows. These are inspection targets, not a proven explanation of the
trader's late discoveries. Keep the date/mtime check separate from bar-content freshness.

### 10B. TC2000 picks reach the M5 watchlist automatically (request 2)

This is a verify/repair candidate first: the inventory already records TC2000-parity board
rows joining M5 Focus (T2, 2026-09-04). Distinguish the bot's TC2000-style Strength Board
from an external TC2000 export/paste; do not promise a direct TC2000 connection that has
not been found. The current wording is assumed to mean the board's picks, with external
lists accepted through the existing paste/import route if that is what the trader uses.

Verified owner: `ui/panels/alert_center_panel.py:5906` wires `boardChanged` and the initial
attachment; `_auto_adopt_strength_board` at `:5938` checks parity floors, adoption gate
and ignored/taken-off names, batches membership writes and stamps machine provenance.
Start with `test_m5_strength_characterization`, `test_t1_capture_and_board_focus`,
`test_qt_alert_center` and `test_focus_auto_pick_provenance`.

Trace board publication → eligible long/short names → mode/adoption gate → persisted
membership → actual BounceBot scan universe. Surface “watching”, “staged” or the measured
reason it is not adopted. Repair a broken link at its existing owner; do not add a second
poller or append repeatedly to the trader's plain-text lists. G will give all these names
one visible Watchlist home. Scanner inclusion and Focus adoption are distinct contracts;
if the trader wants every board row scanned even when adoption fails, present that exact
change as a separate selection rather than silently bypassing the gate.

Tests/live gate: long and short, repeat refresh/restart, removed then reappearing row,
trader-owned overlap, UNKNOWN/stale bars, DESK/AWAY/EVENING/OFF, no manual-name deletion,
and proof that accepted names actually enter a scan. AWAY must still stage, not adopt.
Read M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN and AUTO_MODES_AND_QUIET_HOURS_PLAN.

### 10C. One opt-in H1/H4 retester with named triggers (requests 3, 4, 5)

Trader intent: on selected “TOP weekly pattern” names, wait for a better entry instead
of jumping in. Add a quick **H1/H4 retester** arm button below the chart, using the shared
arm surface. A like or weekly-pattern tag alone does not arm it. Keep the existing tag
meaning intact; this watch expresses entry timing, not a new setup claim or an order.

Reuse candidates, not assumed finished alerts: `bounce_bot_lib/legacy.py:2387` builds
completed H1 bars and `:2589` evaluates H1 15-EMA riding; its H1 alert emission is retired.
`master_avwap_lib/legacy.py:28105` aggregates H4 by session. M5 retest/trendline tags at
`bounce_bot_lib/legacy.py:4095` and D1 trendline logic at
`master_avwap_lib/legacy.py:20634` are not a persisted, opt-in H1/H4 retester. Preserve
retired H1 and M5 LRSI emissions and their evidence; implement a separately selected
watch seam rather than flipping global retirement flags. Reuse indicator math only
after checking that its rule means the requested bounce, not merely riding an EMA.

Proposed small steps, not one giant detector:
1. H1 15-EMA bounce first: selected symbol/side, timeframe, expiry and a visible reason.
2. Add H4 pullbacks and LRSI pullback/recovery as separately named options in the SAME
   watch. Do not require all conditions at once or emit three alerts for one episode.
3. Add a direct trendline-break option for a chart-generated line, plus **break then
   retest** for trendlines and compression boundaries. A direct break alert and a retest
   alert are different choices; “watch for a retest” must not fire just because it broke.

Before tests, freeze a versioned rule sheet: regular/extended hours and H1/H4 session
alignment; EMA warmup; proximity tolerance in ATR; touch and rejection/reclaim definition;
LRSI pullback/recovery levels; break close threshold; retest window; invalidation; expiry;
cooldown/re-arm and handling multiple matching reasons. Proposed retest sequence is
armed → completed-bar break → later retest → completed-bar confirmation → fired/invalid/
expired. Keep one event per watch/episode, recording all measured reasons. A same-candle
break/touch with unknown ordering must not claim a confirmed retest.

Persist the selected trendline/boundary identity, endpoints, version and knowledge time.
Never backdate a newly redrawn line or silently substitute another line into an armed
watch. Project the frozen line forward; explicitly show invalidation/re-arm if needed.
Use completed, session-aligned bars; forming H1/H4 bars are previews and end-session
stubs cannot confirm these patterns. Missing history is “not measured”. Do not revive
the retired M5 LRSI emitters or promote the warehouse's HTF LRSI study as a live signal.

Phone behavior: trader-armed retester notifications should follow the existing armed
Research/Focus alert delivery path in every mode. Record this proposed exception to
routine AWAY-only push in the mode spec when selected; route through one sender with
persistent deduplication and quiet-hours behavior stated. No automatic arming is implied.

Tests: symmetric sides, exact EMA-15 input, completed/forming/stub bars, gaps and invalid
candles, stale data, no lookahead from pivots, line redraw, compression boundary changes,
same-bar ambiguity, repeated polls, restart during retest, expiry in trading sessions,
disarm/re-arm and one phone event. Replay on fixed fixtures before an opt-in live check.
Measure usefulness later; a pattern firing is not proof that it offers a profitable entry.
Read M5_SIGNAL_ENGINES_PLAN, DESK_CHART_UNIFICATION_PLAN, AUTO_MODES_AND_QUIET_HOURS
and existing expiry/price-alert contracts. Reuse the owners cited above.

### 10D. Market Journal tells the market story and challenges my thesis (request 6)

Existing foundation: `market_journal.py:78` keeps text, symbols, session and actual write
time; `ui/services/market_journal_service.py:77` owns writes, `:200` reads captures and
`:249` reads recorded market context. `ui/panels/market_journal_panel.py:219` is the
existing reader. Reuse it. G3/G7 made notes readable and charts lazy; that is not the
requested daily-to-quarterly synthesis. `ai_jobs/runner.py:452` owns the nightly stages;
the optional `weekly_synthesis` at `:760` is not automatically in the default schedule.
Existing trade enrichment is about trades and must not become this market narrative.

Primary scope: SPY, QQQ, IWM, VXX, TLT and USO, plus explicitly mentioned major markets.
Tell the sequence: what I expected, what the market did, what changed, what remains open
and which evidence would change my mind. Relate the trader's chosen tactics to the stated
environment: put selling, directional swings, pure day trading, or sitting out. Do not
infer a strategy recommendation from a ticker moving or from a strategy's later payoff.

Build in steps:
1. A daily story beside the price-action captures: original dated notes plus deterministic
   index facts and links back to each source. Clearly distinguish the trader's words,
   measured price action and AI interpretation. No note means no invented trader thesis.
2. A compact active-thesis view inside Market Journal: claim, original timestamp, intended
   horizon, catalyst/condition, caution stance, invalidation if actually stated, and later
   supporting/contradicting notes. Missing timing/invalidation stays unstated. Offer one
   or two grounded questions, such as whether the stated condition for caution still holds.
   A proposed interpretation is editable and never overwrites the original thought.
3. Weekly summaries from daily summaries; monthly from weekly summaries and uncovered
   daily periods; quarterly from monthly summaries. Keep session coverage explicit at
   month/quarter boundaries so overlapping weeks do not count the same day twice. Offer
   yearly later only if useful. Open theses carry forward even when their source is older.

Prefer the existing overnight AI runner and local models. Optional API-backed synthesis
uses the existing provider configuration when selected; no new key is needed for planning.
Use bounded incremental inputs, stable source IDs/hashes, a schema/prompt version, cached
unchanged results and per-job token/time caps. Changed notes invalidate their day and
affected parent summaries only. Reuse summaries but retain source pointers and a bounded
check of original notes to prevent repeated summarization from changing their meaning.
Never discard contradictory notes to fit a budget; report coverage and truncation.
Use current job-ledger/status surfaces for last success, pending/refused/stale/failed and
bounded retry. Model failure leaves notes/charts and the last verified summary usable.

Tests: absent notes/bars, opposing notes, late-written notes, superseding edits, ambiguous
predictions, unfulfilled predictions, unsupported AI claims, corrected price inputs,
overlapping weeks, incomplete periods, repeat runs, budget exhaustion and model failure.
No model writes Focus, alerts, scores, policy or executed trades. Live acceptance: trace
each daily claim to its note/bar, then inspect a weekly/monthly rollup and an unresolved
thesis; a retrospective note must never look like a prediction made before the event.
Governing docs: LOCAL_AI_AUTOMATION_PLAN, REVIEW_LEARNING_LOOP, decision 0016 and
DESK_INTERNALS “The Market Journal is what the trader thought”. Link item 7's proposed
D1 environment labels if built, but qualitative synthesis does not depend on that study.

### 10E. Journal uses my setup notes to tag the trades I actually took (request 7)

Already built: `journal_analytics.py:173` AutoTagger and its trader-capture lane,
`journal_bulk_tag.py:184` / `:290` plan/apply provisional tags, and the nightly
`journal_auto_tag` slot. `ai_jobs/enrichment.py:151` provides gated advisory enrichment;
the runner's `journal_enrichment` slot at `:719` does not authorize confirmed-tag writes.
Item 5C/5E already covers job completion and the tag backlog. Repair or extend these
paths; do not build another auto-tagger. September 9's assessment counts are historical,
not today's proof that the job failed or succeeded.

First inspect recent ledger results, unmatched trades, current provisional/needs-review
counts and the actual input package for one trade with a relevant stock note. Follow the
note or like event through identity/time matching to a visible tag recommendation.
Explicit setup claims outrank fuzzy inference; a quick like is interest, not a setup.
Free text may support an AI suggestion with the source quote/ID, match basis, confidence
and unknowns. Do not infer tags from future returns or match opposite sides merely because
the ticker agrees. Date-only broker fills cannot establish an intraday note window.
Option direction comes from legs/exposure, not the LONG label on a purchased option.

Use the existing advisory enrichment destination for inferred text-based suggestions.
Only the existing bounded provisional writer may populate unconfirmed setup tags on
closed trades; retain its store-level refusal to overwrite confirmed tags. A broader
write privilege would require a separate explicit contract decision. Let the trader
confirm/correct in the existing Journal/Tag Week flow; show evidence and abstain when weak.
Tests: claimed/quick likes, conflicting notes, pre/post-trade timing, ambiguous/date-only
fills, unknown setup name, confirmed/provisional tags, repeated nights, failed enrichment,
and no outcome leakage. Reuse `test_journal_bulk_tag`, `test_v2_journal_auto_tag_slot`
and `test_ai_enrichment_and_policy_draft`; preserve digest audit gates and lane priority.
Live gate: a closed trade with a known note reaches a traceable suggestion; confirmation
remains the trader's. Read JOURNAL_RELIABILITY_AND_UX_PLAN §2/auto-tagging and LOCAL_AI.

### 10F. Replace AWAY Recap with a visual Daily Recap (request 8)

Verified gap: `ui/app.py:711` hands the recap a capped process-scoped alert list, so a
restart or midnight crossing is not a complete session record. `_RecapWorker` in
`ui/panels/away_recap_panel.py:72` reads the current report, pending picks and Focus;
`away_recap.py:109` assembles those inputs without historical outcome ranking. Merely
renaming the heading or filtering today's data by the selected date cannot meet this ask.

Build the session reader first over existing durable evidence/outcome stores and dated
scan artifacts. Inventory coverage per source; if a required snapshot is not retained,
add the smallest append-only capture at its existing owner after selection. Historical
membership that was never recorded stays unknown. Do not load the giant tracker on the
GUI thread or start a competing outcome grader. This reader also feeds item 5F coaching.

Then replace the existing recap page with Daily Recap, available for every Auto mode and
still selectable/sortable by exchange session. Suggested default: latest completed session,
with Today explicitly provisional, and a 1/2/3 prior-session lookback (default 3).
Four visual views, all opening the shared chart at the original observation/decision:
1. What worked today: sort by maximum favorable movement OR movement held at EOD.
2. Recent swing picks: the previous 1–3 sessions' picks that followed through promptly.
3. My likes and swing favorites: winners AND failures, including all my swing picks.
4. My rejected picks that worked: what I missed, with my original reason alongside it.

Keep max favorable excursion, EOD return and actual journal P&L separate. Use existing
versioned measurement conventions where available, state the reference price/time and
side-adjust the return. R is unavailable without known risk, and “max profit” in a paper
view is labelled best available movement, not money I earned or an achievable fill.
For recent swings, propose first next-session favorable movement plus next-session close
as the explicit “instant follow-through” read; show the selected 1–3-session end as a
separate column. Ratify the precise definition before tests; never choose it after seeing
which metric makes a pick look good. A decision made midday cannot take credit for the
morning's high. Unknown intraday timing yields unavailable, not an assumed opening entry.

Pin each view's cohort/date window, label observation age and pending horizons, retain
capture IDs/source/category/side, dedupe at the opportunity grain and link repeated clicks
without inflating n. Likes, claimed likes, passes, vetoes and not-today/dislike remain
separate facts; unfavorite is not a rejection. Show rejection reasons and adverse movement:
a later rise alone does not prove a timing/risk veto wrong. Raw single-example review is
allowed below statistical floors; claims about best setups or edge still use evidence_stats.

Chart review should show decision marker, relevant level, later path and the original note,
with quick previous/next and an easy view of failures. Preserve AWAY staged-pick management
and its no-return-queue behavior while replacing the page; Daily Recap does not imply
routine phone pushes in DESK/EVENING/OFF. Explicit display sorting must never rerank live
candidates. Link item 5A's verified verdict-unit defect before reusing those summaries.
Tests: restart/midnight/session change, old date with current files, missing source, short
returns, late picks, repeated likes, retractions, overlapping cohorts, incomplete horizons,
and source counts that reconcile. Reuse `test_away_day_recap`, return-surface tests and
existing preference/human-focus graders. Live gate: review one past and one current session
after restart, trace winning AND losing examples to their source and chart.
Read AUTO_MODES_AND_QUIET_HOURS, CHART_REVIEW_WORKSPACE, REVIEW_LEARNING_LOOP and
the current outcome-semantics contracts. This is descriptive learning, not a new ranking model.

### 10G. One Trading Desk Watchlist for Focus and positions (requests 9, 10)

Resolve the placement overlap in favor of the trader's final instruction: the main
**Watchlist tab belongs on Trading Desk**. Journal can link to the Positions view of that
same component; do not create two independent lists. Retire the standalone Chart Review
and Focus Picks navigation pages only after their useful actions are reachable here.
Keep the Trading Desk's Visual Alert Review chart and capture verbs; “Chart Review tab”
does not mean deleting that chart, evidence or the Focus services behind monitoring.

Existing seams: `ui/panels/watchlists_panel.py:52` edits shared/swing lists and exposes
bot-owned lists; `ui/app.py:83`/`:84` register the separate pages; `journal_panel.py:67`
has Trades/Calendar/Analytics/Health/Fees, not a position watchlist; shared
`ui/services/price_alert_service.py:31` already owns armed price monitoring.

Proposed views in one list: My watchlist, M5/TC2000, Swing favorites, Open positions and
All. Show source badges and side/horizon without turning source into priority. Easy
single-symbol add and paste-many, with side/horizon selection and useful duplicate handling.
Manual “positions today” entry is a monitoring note unless the trader explicitly saves
an execution using the Journal's existing trade-entry flow; never manufacture a trade
just to add a ticker. Journal-derived positions are read-only projections of open/partly
closed exposure, with account, quantity/exposure, source and last sync time shown.
Unknown or stale sync must not silently remove a position. Closed journal positions leave
the auto view after verified refresh while independently hand-added watch names survive.

Click a row → the shared visual chart with capture and alert controls. Keep the arm bar
under the chart. Existing price alerts go to the phone through the current service; C's
retester can join the same controls later. Preserve price-alert identity, disarm/expiry,
Focus provenance/adoption gates, fade/restore, swing favorite retractions and keyboard
shortcuts. UI consolidation must not change who owns a name or what the scanner watches.
No alert is automatically armed just because a broker position exists.

Build membership/projection and actions first, then move navigation and saved-layout
routes. Maintain the existing Focus writer/service internally. Inventory old page-only
actions before removal, including arm, fade restore, paste, reasons and strength access.
Tests: manual + auto + journal overlap, separate sides/accounts/options, partial closes,
failed sync, external watchlist edit, restart, selection retention, shortcuts when old
pages are hidden, no lost alerts and no broker writes. Use existing watchlist/Focus tests
and Qt render checks at the trader's size plus one windowed size. Live gate: paste a name,
see an open journal position, chart each, arm/disarm a phone alert and retain both after
restart. Read M5_FOCUS_GATING, DESK_CHART_UNIFICATION and JOURNAL_RELIABILITY_AND_UX.

### 10H. More chart history without making the desk slow (request 11)

Treat “200 candles is not enough” as the requested outcome, not a verified universal cap.
Inventory each chart's provider request, cache retention, payload truncation and visible
viewport. Separate how many bars exist from how many are initially visible. Reuse
ChartDataService; never raise every provider request blindly or fetch from the paint path.

Verified differences: `chart_snapshot.py:310` loads the stored D1 history; `:488` builds
indicators before applying the display tail, whose default is 90 sessions (`:26`).
`ui/panels/chart_review_panel.py:67`/`:347` requests 520 D1 sessions already.
`ui/widgets/symbol_snapshot_dialog.py:424` and the Alert Center at `:2546`/`:2715`
request two M5 sessions. `ui/services/chart_data_service.py:184` passes supplied bars
through, and `ui/widgets/candle_chart.py:660` clips/downsamples the viewport rather than
imposing a universal 200-candle cap. Fix request/display seams by surface; do not confuse
symbol-count cache limits or the scan's AI OHLC excerpt with chart-history limits.

Proposed UI: useful initial zoom plus Load older / pan-left loading with a clear oldest
available date and provider limit. Suggested starting targets, pending provider/replay
checks: at least 1,000 available D1 candles and 500 H1/H4 candles; M5 history loads in
bounded session chunks rather than one huge startup request. These are targets, not a
claim that a provider supplies them. Preserve zoom/selection as older bars arrive, cancel
stale symbol requests, cache off-thread, dedupe overlapping chunks and leave charts usable
on provider failure. Fetch enough warmup before the visible span for EMA/LRSI/AVWAP and
retain actual anchor history; do not fabricate earlier bands from a truncated series.

Tests: >200 bars reachable, initial view unchanged, chunk overlap/order/timezones, sparse
history, exhausted provider, rapid ticker changes, warmup/anchor correctness and bounded
paint/memory cost. Live gate: pan further back on D1/M5/H1/H4 with the scanner running,
check oldest dates and freshness and compare GUI responsiveness to the existing bench.
Read BROKER_ADAPTERS, DESK_CHART_UNIFICATION and GUI_FLUIDITY_MEASUREMENT_RUNBOOK.

### 10I. Connect thesis, actual trades and setup evidence by market environment (2026-09-11 follow-up)

Trader asks: can the Market Journal, Journal and Setup Tracker together answer “what
works in what market environment?”, and are we using the local AI well? This is the
shared evidence connection for 10D/E/F and existing item 7, not a fourth journal or a
replacement tracker. The trader intends to work through the wishlist over the coming
week after Fable resets; prepare this dependency alongside those plans, not after all
three have been built independently. This entry implements nothing or promotes no model.

Verified starting point (source inspection, 2026-09-11):
- `scripts/ai_jobs/briefs.py:37` DEFAULT_SCOPES already includes market_conditions,
  setup_trackers, journal_review and market_journal. `scripts/ai_summary.py:868` supplies
  recorded context, chart digests and original notes. Do not propose merely adding access
  to data it already receives; inspect how much of each source actually survives packaging.
- `scripts/market_context_ledger.py:92` records measured daily context and distinguishes
  late completion; regime-shift evidence distinguishes machine reads and user overrides.
  A close-time summary cannot serve as the market state known at a morning entry.
- `scripts/ai_jobs/digest.py:209` reads environment/day-part keys from recorded rows.
  The digest's declared v1 slice is environment × day-part × side, without setup-family
  slicing. This is useful evidence, not the requested full setup/environment/trade join.
- `scripts/journal_analytics.py:919` provides separate setup and regime breakdowns.
  Journal regime fields exist, but separate totals do not establish how one specific
  setup performed under one market state. Missing labels remain unset.

Proposed shared contract, before building the new screens:
1. Keep two distinct accounts: the trader's thesis (what was expected) and the observed
   market state (what was measured). A thesis carries its original note ID, symbols/index,
   write time, horizon, stance and explicit conditions. An AI extraction is a suggestion
   with source pointers, never the authoritative market label or a silently confirmed
   trader claim. Preserve revisions; a later correction must not rewrite earlier intent.
2. Define a small, versioned environment vocabulary through item 7. Start with existing
   recorded states and one or two measured dimensions, such as trend/range and volatility,
   only where available. Separate daily and intraday context. Name the benchmark and
   mapping rule (SPY primary, explicit QQQ/IWM context where appropriate); do not choose
   whichever index explains a winner best after the fact. Avoid dozens of thin categories.
3. Attach the state known at opportunity observation and actual trade entry, separately.
   Record context ID/version, observed_at/available_at, relevant setup/occurrence ID,
   source and match certainty. Link a thesis by its scope and validity window, not by
   ticker/date alone. A market view may cover many names; a stock-specific note may not.
   No explicit thesis is a valid state. A changed intraday stance requires a new version.
4. Reuse current opportunity/trade/capture identities and named store owners. One thesis
   may link many opportunities; one opportunity may link several decisions/fills. Count
   a trade's money once, even with several tags or thesis links. Ambiguous matches stay
   ambiguous. Date-only broker fills cannot select a midday regime; use only justified
   prior/completed-session context or report the intraday link unknown.
5. Backfill only what contemporaneous evidence establishes, on copies first. Label a
   later historical reconstruction and keep it out of claims about what was known live.
   Do not stamp today's regime or today's AI interpretation onto an old trade as fact.

One drillable answer in the existing Daily Recap/Research views, with separate populations:
- Opportunity evidence: how all recorded eligible examples of setup S behaved in state E,
  including names the trader never took. This estimates the setup's observed behavior.
- Personal execution: how confirmed actual trades in S/E did, including losses, fees and
  valid risk measures. Keep paper movement, alternative exit-policy returns and broker
  money separate; a hypothetical best move is never the trader's expected profit.
- Thesis review: what the trader expected, what later supported/contradicted it, and
  whether the chosen tactic matched the trader's own stated plan. “Market call right”,
  “setup held” and “trade profitable” are different verdicts. Missing entry/risk/timing
  can prevent an execution diagnosis; do not explain every loss as poor discipline.

Each cell shows n, distinct sessions/symbols, coverage, window, outcome definition and
uncertainty through evidence_stats. Compare against the same setup's overall baseline
and suitable contemporaneous opportunities. Account for overlapping trades, repeated
signals, concentration and differing hold periods; do not pool theta, day trades and
swings into a single win rate. Start with descriptive association, not causal claims.
Choose cuts before reading outcomes and validate promising observations on later sessions
under the existing trial/promotion rules. Thin personal history can still show examples,
but cannot justify a “best environment” claim borrowed from the bot's larger sample.

Local AI's best role in this connection:
- Code computes joins, prices, returns, coverage and statistics. The model extracts
  tentative thesis structure, collates notes, explains the computed results, highlights
  contradictions and proposes a small question to test. It never invents measurements.
- Feed a bounded joined fact pack with source IDs and examples, not just three unrelated
  reports and a request to infer links. Show both supporting and opposing examples under
  a fixed selection rule, with omitted counts. This extends existing packaging/runner
  owners; no competing nightly pipeline or result-based narration selection is implied.
- Reuse 10D's incremental local daily summaries and cached period rollups. Reserve any
  optional frontier review for compact difficult/periodic cases after the trader selects
  the provider and budget. A larger model cannot repair absent timestamps or wrong joins.
- Check effective use, not activity: job completion/refusal/failure, source coverage in
  the actual package, artifact freshness, elapsed time, token counts where measured,
  unsupported-claim rate on an audited sample, and whether the report reaches the trader.
  A successful deterministic digest is not proof that later narration or enrichment ran.
  A model output file is not proof that its claims are useful or that a user saw it.
- Keep current gates and bounded retries. Explain refused/stale output in existing status
  surfaces. Audit a small set of notes → extracted claims → joins → numerical facts →
  narrated conclusions, then inspect counterexamples and compare against a simple
  deterministic summary. Spend more inference only where it adds a traceable benefit.

Read-only local-AI snapshot (2026-09-11; these are dated observations, not permanent status):
- Configured model: `gemma3:12b-tbv3ctx-64k`. The configured AI store is
  `\\MINI-PC\Trading Bot Data\ai_store`. Its `logs/ai_job_ledger.jsonl` records
  11/11 OK rows for session September 8 and 16/16 for September 9 and September 10.
  No failed/degraded ledger rows were found in those inspected sessions. This supersedes
  item 5C's September 9 uncertainty about later completion, not its general audit proposal.
- Session September 10: journal enrichment reports 3/3 trades; ticker briefs reports
  265 briefs, 91 model calls and zero failed calls. Some membership-only names are
  intentionally skipped; no model call is warranted when there is no grounded evidence.
- The AI summary's successful row says “NOT synthesized”. Inspect the reason and actual
  output before calling this a runtime defect; successful component reports do not prove
  the combined coaching answer exists. This is a concrete next quality-audit target.
- `digests/facts/2026/2026-09-10.json` reports 481 usable outcomes and 16 environment/side
  slices, but explicitly has no journal block and no setup-family slice. Those outcome
  counts are not counts of actual journal trades. They cannot establish the three-way join.
- `retros/setup_research/2026/2026-09-10` artifacts report 632 eligible cells, 63 narrated;
  the inspected narration has a generic summary and empty candidate/lesson/risk arrays.
  Bounded coverage is intentional, not a reason to remove the size rule. Audit whether
  the selected evidence supports useful specific conclusions and whether abstention is
  warranted. More calls or longer output is not the acceptance criterion.

Tests before any selected implementation: mid-session regime/thesis changes, EOD
lookahead, late notes, overlapping theses, opposite sides, one trade/many tags, ambiguous
or date-only fills, missing context, short/theta exposure, repeated observations,
partial exits, sparse/concentrated cells, model omissions/fabricated references and
rerun idempotence. Golden existing aggregates remain unchanged; new views have explicit
population definitions and no detector/score/alert influence. Live gate: trace one
winning and one losing opportunity and trade to their original thesis/context, then
verify a later-session comparison without retrospectively relabelling the inputs.

Read the current LOCAL_AI_AUTOMATION_PLAN, JOURNAL_RELIABILITY_AND_UX_PLAN,
REVIEW_LEARNING_LOOP, item 7 and matching DESK_INTERNALS evidence/AI contracts before
selecting source owners. Expected seams are the existing context ledger, journal/capture
joins, evidence summaries, AI packaging and Daily Recap/Research readers. Define the
shared contract early; ship the deterministic join before asking the AI to explain it.

### 10J. Trade Mentor — a steady, low-friction stream of trader context (2026-09-11)

Trader request: a Settings checkbox named **Trade Mentor**. While enabled, ask hourly
for an M5 market read, at 08:00 and 12:00 Pacific for a D1 read, and around 10:00 for
missing information on the previous session's trades. Skip missed prompts when away;
ask at the next scheduled hour. Accept natural-language replies and have local AI put
the information into the right fields. Purpose: reduce procrastination and feed 10D/E/I
with consistent, time-stamped thoughts. This is a planning addition for next week's work,
not an instruction to change the running desk now.

Existing foundations: `ui/panels/settings_panel.py:48` and `ui/state.py` own Settings;
`ui/services/market_journal_service.py:77` owns market-note capture; `market_journal.py:78`
records actual write time; `journal_store.py` owns trade fields and annotations;
`ui/panels/journal/trades_tab.py` owns trade editing. Existing local-AI packaging and
the overnight runner can read the resulting evidence. A dedicated mentor scheduler and
reliable free-text-to-field workflow have not been verified as existing; inspect before
adding them. Do not mistake ordinary overnight advisory enrichment for interactive form
filling. Verify existing target/thesis field support before proposing a schema migration.

Scheduling contract to select before implementation:
- Persist the checkbox, default OFF. Show the next prompt and a simple Pause today.
  Interpret the trader's “PST” as Pacific wall time (`America/Los_Angeles`, including
  daylight saving), not fixed UTC-8. State that assumption in Settings and test DST.
- Proposed hourly M5 schedule: whole hours during the exchange's regular session,
  starting at 07:00 Pacific on normal days and ending before the actual close. This
  excludes the opening half-hour and after-close prompts; make the window adjustable
  if the trader wants those. Weekends/holidays have no automatic prompts.
- Fixed D1 prompts: 08:00 and 12:00 Pacific on exchange sessions. Keep the noon read
  on an early-close day if the trader is present, labelled post-close, since the trader
  explicitly requested that time. “D1 read” describes the trader's horizon; it does not
  assert that today's D1 candle is complete. Show the current chart as a labelled preview.
- At 10:00 Pacific, inspect the PREVIOUS EXCHANGE SESSION's actual trades and ask only
  for missing material fields. Monday normally reviews Friday. No trades or complete
  entries means no repair questionnaire. Missing broker coverage means “journal not
  ready”, not “no trades”; one bounded retry within the current slot may be considered.
- At 08:00/12:00 combine M5 and D1 into one card with separately stored answers. At
  10:00 combine the current read and the missing-trade-data task into one card with
  separate sections; never stack dialogs or merge market notes into trade notes.
- One scheduler owns all slots. Persist slot identity (session/date, scheduled instant,
  prompt kind), delivery and answer state. A restart, timer drift or clock correction
  cannot duplicate a slot. Proposed response window: until the next hour, then expire
  unanswered prompts. A new hour replaces an untouched card; never discard typed text.
  Preserve a draft separately without representing it as a submitted market read.
- AWAY, paused, locked/asleep or otherwise absent: skip; no queue and no catch-up burst
  on return. Re-evaluate presence at each future slot; missing one must not disable the
  day. Resume/restart midway through a missed slot does not suddenly demand its answer.
  Define presence from explicit mode/lock signals first, with a configurable idle grace;
  a trader quietly watching charts is not necessarily away. Do not record raw activity.
- Keep Trade Mentor independent of the scanner's Auto setting: it does not enable scans
  or change DESK/AWAY adoption. Proposed behavior is opt-in, present-user prompts in
  DESK/EVENING/OFF, none in AWAY. Its fixed-time/early-close schedule is an explicit
  proposed exception to the broad automatic-starter quiet-hours rule; record that narrow
  contract in AUTO_MODES_AND_QUIET_HOURS/AGENTS/CLAUDE when selected. No phone push is
  included in this request. Manual “Give a read” remains available at any time.

Capture experience:
- Use a small, non-modal card near the chart; no focus stealing or blocked trading
  controls. One text box, Submit, “Read unchanged”, and Skip. Keep action keys scoped
  to the card so typing or trading shortcuts elsewhere cannot submit an answer.
- Default question: “What changed? What do you expect next? What would change your
  mind?” Show the previous read and relevant index chart as context. One sentence is
  enough; optional detail can cover SPY/QQQ/IWM, other indexes, preferred tactics and
  caution. Do not force answers to every dimension each hour.
- “Read unchanged” writes a NEW, explicit reaffirmation referencing the previous read,
  at the current time. It is not a copied prediction or an independent thesis sample.
  An unanswered prompt means no observation, not unchanged, neutral or bearish.
- Link each submitted read to its prompt, actual response time, horizon and available
  chart/context snapshot through 10I. A response at 09:50 cannot claim to describe the
  market at 09:00. Capture raw text first using the existing store owner; chart capture
  and parsing happen off the GUI thread. Failed journal writes are visible and retryable.

The 10:00 trade check and AI form filling:
- Name the trade clearly (symbol, side/exposure, account and entry/date as needed) and
  show fields already known. Ask for thesis/reason, original stop or invalidation,
  target IF one existed, and relevant support: setup claim, levels, timeframe, market
  thesis and reason for choosing the tactic. Ask only what is missing; “all support
  data” is contextual evidence, not an ever-growing compulsory questionnaire.
- Treat **not supplied**, **explicitly no fixed target**, **not remembered**, and
  **not applicable** distinctly. No fixed target is a complete answer. Apply the same
  honesty to a stop: no stop is not a numeric zero or permission to invent one. A later
  current stop and the original plan are different facts. Allow free text such as
  “exit if the H1 level fails”; do not invent a precise price from it.
- Local AI extracts a schema-checked draft into the relevant form fields, with exact
  source spans and units. It must distinguish underlying price, option premium, dollars
  of risk and percent; ambiguity stays blank with one focused follow-up. Deterministic
  validation checks types/units; confidence is not proof. Do not guess position sizing
  or risk from future price movement, exit price or broker money.
- Save the original reply immediately, then show the populated fields for one combined
  Save/Correct action through the existing trader-edit service. Do not ask permission
  for every extracted field. This is a trader-reviewed edit, not a new unattended AI
  writer of confirmed tags/risk. Preserve existing values and show any proposed conflict;
  model output never silently overwrites a trader entry. A wrong trade link cannot be
  repaired by simply giving its text a high confidence score.
- If the local model is busy/offline, keep the reply and offer manual completion; queue
  bounded parsing without repeated prompts. Give a small on-demand parsing task priority
  only through the existing model resource owner; do not launch a competing model load
  that starves scanners or overnight jobs. Measure latency before selecting the model.
- All next-day answers carry actual write time and “recalled after the session” status.
  Never backdate them or present remembered risk as a documented pre-entry plan. Keep
  any analysis using recalled risk separately labelled from prospectively recorded risk.
- Cap the morning task by time/trade count (suggest five minutes or three incomplete
  trades, configurable). Keep the rest in the existing Journal completeness view with
  an explicit count. Do not silently forget it, but do not nag again hourly about it.

Suggested refinements, separate from the requested schedule:
1. **Ask for changes, not essays.** Reuse known data; one useful new sentence beats an
   hourly repeated form. Never reward word count, trading frequency or a winning answer.
2. **Catch intent before results.** Offer a tiny “Why this trade / invalidation” capture
   when the trader explicitly records a trade or arms a watch. Do not assume an armed
   watch is a taken trade. If broker sync arrives late, label capture timing honestly.
3. **One useful follow-up.** If a read names a concrete condition, offer one bounded
   check after new measured evidence contradicts/supports it. Phrase it as a question,
   not a trade instruction. This is opt-in extra scope, with cooldown and no new scanner.
4. **Close the loop briefly.** Put “what I expected / what happened / one lesson” in the
   existing Daily Recap. Carry one trader-chosen habit into the next session and review
   it against future examples, including counterexamples, rather than creating a grade
   for every thought. Daily/weekly Mentor feedback reuses 10D/I, not a separate AI report.
5. **Measure burden and usefulness.** Show completed opportunities to give a read out of
   prompts actually delivered while present, skip reasons only when known, missing-field
   reduction and typical completion time. Separately count scheduled-but-away slots.
   No guilt streaks, punitive scores, escalating reminders or forced answers. A pause
   should be easy; record the coverage gap instead of manufacturing data.

Build in small selected steps: scheduler/presence + raw market reads first; missing-field
questionnaire second; validated AI draft filling third; useful coaching last. 10I's
identity/time contract must precede joins, but Mentor's raw capture need not wait for
every environment study or summary feature to finish. Coordinate Settings, Journal and
shared chart edits with 10D/E/G. Reconcile LOCAL_AI_AUTOMATION_PLAN,
JOURNAL_RELIABILITY_AND_UX_PLAN, AUTO_MODES_AND_QUIET_HOURS and matching DESK_INTERNALS
entries; mirror AGENTS/CLAUDE only if the selected contract changes operating rules.

Tests/live acceptance for a selected build: DST, holiday/early close, previous-session
lookup, 08:00/10:00/12:00 collisions, skipped hour then next-hour delivery, lock/sleep,
restart, Auto OFF vs Mentor OFF, pause, idle-but-present, existing typed draft, duplicate
submit, no-target vs missing, conflicting/mis-unit numbers, wrong-trade association,
stale broker import, AI unavailable/invalid output, confirmed-field preservation and
post-session provenance. Test the real scheduler with an injected clock and the actual
store/form path; do not sleep through hours in tests. Live check: miss one slot while
away, return without backlog, answer the next, fill one trade with “no target”, correct
one AI field, and trace both saved records into the next grounded AI package. No app
change, API call or runtime test is part of this planning update.

### 10K. One measured review for the trader, local AI and frontier model (2026-09-12)

STATUS: TRADER-REQUESTED INTEGRATION PLAN, NOT IMPLEMENTED. This conversation authorizes
planning in WISHLIST, not app changes, live repairs, a model/provider change or automatic
frontier calls. Fable uses this as the integration brief for selected work in items
5/7 and 10D/E/F/I/J, then records the bounded build selections in `plan.md`. Existing
contracts in those entries remain; this section supersedes their disconnected build
ordering and earlier assumptions that a successful AI job means useful advice exists.

#### Product outcome and division of work

The trader cannot watch every setup or variant. The program should record the observed
opportunities, measure what worked and failed, preserve the trader's changing market
view, and make the answers quick to read. The Daily Review and the frontier handoff
must be two views of the SAME versioned facts, with local-AI explanations alongside.
“Daily Review” here is 10F's Daily Recap, not a second page or a replacement tracker.

- **Python measures:** eligibility, coverage, prices, outcome paths, speed, returns,
  setup/variant comparisons and market features. It computes the best/worst tables.
- **Local AI distills:** the core of the trader's notes, conditions and concerns;
  concise explanations of the computed tables; contradictions and explicit missing
  measurements. It does not need to reinvent the analysis or crunch raw rows itself.
- **The trader reads:** charts, numerical results and a short local-AI review in the
  app, with originals one click away and an easy way to correct an interpretation.
- **The frontier model reasons:** receives those compact facts, the trader's intent,
  and the relevant Setup Tracker evidence/version. It can test explanations and ask
  deeper questions without re-reading giant stores to reconstruct routine answers.

The learning question is “which setup and variant worked, how, and in what conditions?”
It covers recorded names the trader never took as well as actual trades. Post-earnings
success is the trader's stated observation and a useful first example, not a verified
edge claim or a reason to select only winning examples. Nothing here changes live
detectors, scores, alerts or orders; evidence/promotion rules remain in force.

#### Starting point: what exists and what the September 12 audit found

Read the named seams again before building; these are dated findings, not permanent
runtime status. The September 11 session ran overnight into September 12 Pacific.

- The nightly default already includes Market Journal, trade journal and setup reports
  (`ai_jobs/briefs.py:37`, `ai_summary.py:763`). Its latest package held 28 market notes
  and 26 chart digests. It lacked the daily machine-context and user-environment sources.
  Access to several sources is not the thesis/context/setup/trade join in 10I.
- Main summaries for September 10 and 11 took 309.0 and 264.1 minutes; both final
  synthesis calls timed out at 900 seconds. September 9 synthesized 25/25 slices;
  September 11 read 54/54 but published unsynthesized findings. `briefs.py:250` still
  returns OK after `map_reduce.py:483` falls back. Fix meaningful completion/status,
  not just the green count; do not assume an output-length error caused a read timeout.
- **Confirmed enrichment contract defect:** six distinct trades over September 9–11
  have blank `summary` and `tags` in `ai_trade_enrichment`, although jobs report success.
  `enrichment.py:353/:363` reads fields excluded by `ai_summary.py:433/:484`'s schema.
  A read-only reproduction with a real validated summary returned blank text. Existing
  blank rows also satisfy `_trades_for_session`'s “already done” check at `:268`.
  The separate deterministic provisional-tag job exists and must not be replaced.
- September 11 ticker work recovered its one failed symbol: 91 analyzed, 246
  membership-only, zero failed after retry. The original failed manifest row remains
  valid history. Three sampled briefs largely restate membership/truncation. Digest
  narration repeats its computed headline; research narrates 63/660 eligible cells
  with largely generic advice. More tokens or model calls are not proof of usefulness.
- Broker import failed three times, with other import work partially successful;
  source completeness needs a separate check. No planned risk was recorded on the
  14 distinct preference-matched trades in that night's report. Never fill it from
  their results. No Journal UI reader of the enrichment table was found; the manual
  AI Summary page does not load the nightly main review (`ai_summary_panel.py:390`).

Audit sources: `ai_store/logs/ai_job_ledger.jsonl`, September 9–11 main-summary JSON
and paired evidence under `ai_store/briefs/2026`, September 11 digest/research packs,
`C:\TradingBotData\ai_morning_brief.txt`, and a read-only query of the journal database
at its `project_paths.JOURNAL_DB_FILE` path. Inspect bounded fields, not raw credential-
bearing broker error URLs. Older 5C/10I completion counts do not supersede this audit.

#### One numerical contract, with separate answers rather than one “best” score

Freeze these definitions and their availability map BEFORE examining the comparison
results. Reuse existing measures where equivalent; otherwise label the gap. Never
silently substitute an endpoint return for maximum movement or mix exit policies.

| Requested answer | Proposed report meaning and controls |
|---|---|
| Total profit | Actual closed-trade broker net P&L, counted once per trade and with currency/fees/partial exposure handled by current Journal rules. Rank actual trades or aggregate setups within the same declared population/window; show n. Hypothetical recipe net R is a separate result, never dollars earned or a realizable portfolio total. A paper-dollar comparison would need a separately selected capital, sizing, overlap and cost convention. |
| Biggest opportunity | Maximum favorable movement AFTER observation/defined entry within the chosen horizon, side-adjusted, plus adverse movement. Use measured MFE_R only with known risk; show percent or ATR as separate named units where available. Label it best available movement, not an achievable fill or profit taken. |
| Quickest result | Proposed first view: time to the recipe's predeclared first target, with target-hit count/rate and unhit/pending/unknown counts beside median elapsed trading minutes among hits. Compare only equivalent targets/entry rules. Keep `time_to_mfe_min` as a separate hindsight timing fact; fast failures must not disappear from the cohort. No universal 1R threshold is assumed when risk is missing. |
| End of day | Side-adjusted mark at the stated session close, with its entry/reference clock and existing exit-policy convention. Keep an EOD hold measure separate from a stopped strategy's realized result. An entry at the close has no same-session forward opportunity; use unavailable and an explicitly named next-session measure. |
| Last day or two | Two independent controls: which observation sessions are included, and how long each observation is followed. Proposed observation choices: selected session or latest two completed sessions. Forward choices include entry-session close and exact next 1/2 exchange-session endpoints, with pending horizons shown. Preserve 10F's existing 1–3-session lookback as an additional view; never rename a 48-hour calendar period “two sessions”. |

Individual examples may be sorted by the chosen measured result for retrospective
review. Setup/variant/environment cells show good AND bad results, n, distinct sessions
and symbols, missing coverage, policy/version and uncertainty via `evidence_stats`.
Use its floors before “best setup” claims; ties/thin samples say no clear leader.
An impressive single move is still visible below the floor, as an example only.
Keep champion/study/control, entry/exit recipes, band families, long/short, day/swing/
theta and actual trades distinct. Pair variant comparisons on shared occurrence IDs;
report unpaired counts and overlapping exposure instead of adding correlated trials.

Existing owners: `research_warehouse/outcomes.py:171/:912` already has session/minute
checkpoints, MFE/MAE, `time_to_mfe_min`, `first_hit_at`, gross/net R and maturity;
`:982` defines entry-session EOD. `master_avwap_lib/session_horizon_outcomes.py:82/:372`
has exact session endpoint returns, NOT speed or MFE. `evidence_stats.py:431` owns
statistics. Warehouse data can be disabled/unreachable; it must show unavailable,
not block the desk or relocate the research store. Inventory all recorded families/
variants and their eligible/measured/pending/missing counts before claiming coverage.
Do not promise to measure setups or variants that were never captured.

#### Market thoughts, weekly forecast and chart facts

Keep three source kinds: **trader thesis**, **external model forecast**, and **measured
market state**. The local summary is a fourth, derived interpretation with source IDs.

- Accept the trader's hourly natural-language reads through 10J's raw-first capture.
  Preserve actual write time, horizon, index/symbol scope and later revisions. Extract
  expectation, supporting cues, concerns, preferred tactics and stated invalidation.
  Keep “maybe”, alternative scenarios and unstated fields; do not strengthen a tentative
  note into a confident prediction. “Read unchanged” is reaffirmation, not a new thesis.
- Add an explicit paste/import route for the weekly ChatGPT forecast inside Market
  Journal, using its existing writer where the schema permits. Save the original text,
  source/model if supplied, original creation time if known, import time, target week,
  assumptions/scenarios and source links if supplied. Unknown creation time stays unknown.
  A forecast is outside commentary, not market fact or the trader's adopted view; an
  explicit adoption links to a new trader statement. Later imports cannot count as
  information known at an earlier entry. No ChatGPT connection, automatic pull or paid
  API is assumed; imported instructions cannot change app policy or start actions.
- Summarize cross-market reasoning: SPY, QQQ, IWM, bonds/yields and other explicitly
  named markets. An index mentioned in prose must not inherit a selected stock chart's
  identity. Keep each claim's scope. Separate yields from bond-price proxies such as
  TLT; show source, unit and observation time, and do not invent a yield feed.
- Use cached bars and `market_journal_capture.py:280` chart digests plus
  `market_context_ledger.py:88` and ChartDataService. A compact chart fact includes
  symbol/timeframe/session, data known through, completed/preview status, named measured
  levels/features with units and rule versions, and a pointer to the original chart/bars.
  Compress repetitive candles into computed facts; do not discard the evidence needed
  to check a claimed trend or manufacture an earlier pivot from later candles.
- Daily summary: what the trader expected, what changed in that view, what the measured
  market did, and which questions remain. Preserve disagreements with the weekly forecast
  rather than blending them away. Reuse 10D's incremental daily/weekly/monthly/quarterly
  rollups, unique session coverage, open-thesis carry-forward and source-linked corrections.

#### Measure environment, then connect it to setup performance

Implement 10I's identity/time contract before joining screens. Attach the environment
known at opportunity observation and the environment known at actual entry separately.
Use the original benchmark/rule, never the index or closing state that best explains
a winner afterwards. A late reconstruction is labelled and excluded from forward proof.

Start with a small set of measured dimensions from item 7, not a grid of dozens of
labels. Candidate dimensions grounded in the trader's words: compression/expansion;
trend/range and lower-high structure; asymmetric downward versus upward impulse size
and speed; and days since a known earnings event. “Downward wedge” remains a trader
description until a causal, versioned geometric rule is selected and tested. Bond/yield
and cross-index context stays unmeasured unless the needed dated data exists. Keep
dimensions independent where possible; do not choose thresholds by their later profit.

For each setup/variant, show the same metrics by environment and against its overall
baseline on comparable dates/hold conventions. Separate (a) all recorded opportunities,
(b) the trader's chosen/rejected names and reasons, and (c) actual execution. Reuse the
10I thesis/context/opportunity/trade links and count money once. “My market call was
right”, “the setup moved”, and “my trade made money” remain different conclusions.

The local AI may flag a gap: “your notes often mention weaker rallies, but no measured
field captures that.” Each suggestion names source notes, existing proxy if any,
missing observations, and a small proposed measurement/test with cost and coverage.
Store such proposals in existing research outputs, not live detector settings. Fable/
trader selects a versioned shadow study before new collection or trial grids; register
trials before reading outcomes and test promising associations on later sessions.
No automatic feature mining, confidence-based promotion or AI changes to scoring.

#### Shared report, UI and compact frontier handoff

Extend the existing evidence/digest/entry-index owners with versioned, source-linked
sections. First inventory which old readers must remain compatible; do not fill the
current digest's empty journal/swing sections with values of another grain. No second
grader, timer, competing ledger or giant tracker read on the Qt thread.

The logical report has separate sections for market thoughts/forecast, measured market
context, opportunity/setup results, preference decisions, actual trades, and missing
evidence. Each numeric cell has an ID, metric/units, population, window, reference clock,
source path/row IDs, version and measured/pending/unknown state. Narration cites these
IDs; it never owns their numeric values. Snapshots have a common report ID and as-of
time across UI and export; later matured results are explicit new/superseding versions,
not backdated knowledge or an implicit rebuild of immutable historic digest packs.

- **Daily Review:** reuse 10F's page, with a visible local-AI review area/tab alongside
  its numeric tables. Show selected session, horizon, metric, coverage and output age.
  Click a cell/example to the shared chart at the observation; click a thesis/forecast
  to its original. Show failures and counterexamples beside winners. A missing model
  leaves tables/charts useful; a failed import leaves a visible coverage warning.
- **Frontier handoff:** offer export/copy of a small readable brief and versioned JSON,
  a report manifest, relevant Setup Tracker snapshot/version, and drill-down paths to
  detailed partitions. Include the numerical answers, condensed notes/forecast, observed
  context, competing explanations, unresolved questions and missing fields. Do not make
  the frontier model recompute basic tables or infer joins from unrelated prose reports.
  Export is user-initiated; no automatic external upload or frontier call is selected.
- **Bounded work:** proposed first budget is 32 KiB UTF-8 for the headline handoff plus
  manifest (detailed partitions remain available separately). Measure actual token use
  as well as bytes; provider limits may be stricter. Ratify a measured bound before
  tests. Reuse unchanged source hashes and incremental period summaries; cap local task
  size/time, retain omitted counts, preserve open concerns and avoid a full-history
  reread each night. Summaries retain links to originals to prevent meaning drift.
- **Selection honesty:** the requested numeric best/worst tables are explicitly
  retrospective, result-selected views, with the full denominator, fixed metric/window
  and comparison coverage. They do not become a representative research sample. The
  existing N3 `setup_research` narration selection remains n-descending, never R-ranked;
  do not change it to implement this report. Any new compact review section needs its
  own explicit selection/omission contract and balanced examples, not a hidden change
  to the N3 pack or the Setup Tracker's current sort. Thin cells cannot be promoted.

#### Fable's integration sequence and acceptance

These are bounded selections for the existing roadmap, not a parallel progress ledger.
Fable should present the first concrete code scope after recon; no runtime changes are
authorized by this entry. Coordinate shared Journal/chart edits, and retain all live
validation and promotion gates. The dependency order within this work is:

1. **Repair trustworthy output.** Reproduce enrichment failure before fixing its
   schema/extractors, refusal of empty success and retry/supersession of existing blank
   records. Verify a real saved suggestion reaches the trader. Distinguish published,
   synthesized, useful-empty/abstained, failed and partial outputs. Inspect broker
   completeness and the synthesis timeout using captured inputs; do not bypass gates,
   raise timeouts blindly or change model as a substitute for this investigation.
2. **Define and publish shared measured results.** Freeze metric/population/time/identity
   contracts and a coverage inventory. Reuse the session/outcome readers, add only proven
   missing measures, and publish the shared report without requiring narration. Start
   an end-to-end slice with one recorded post-earnings family and one comparator chosen
   before reading results; include losses and unknowns. Expand to all supported families
   and recorded variants before claiming whole-universe coverage. Theta joins when its
   own capture exists; its absence must not block the first useful report.
3. **Preserve and condense intent.** Add weekly-forecast import and source-faithful
   thesis extraction, plus 10J raw capture first. This can run alongside independent
   numerical work once identities are agreed. Daytime capture need not wait for inference;
   current market-hours inference protection stays. Interactive parsing requires 10J's
   separately selected resource contract, latency proof and trader-reviewed field writes.
4. **Join measured context and show the daily review.** Add the small versioned market
   feature set, point-in-time joins and setup/environment comparisons. Wire the SAME
   report to 10F's tables/charts and local-AI review. A view with sparse evidence should
   still work and say what is unknown. Preserve manual/auto provenance and chart controls.
5. **Deliver and audit the frontier pack.** Add the compact export and incremental
   rollups. From the pack alone, a reader should answer: which recorded setups moved
   most, hit their target fastest, held best to the close, and followed through after
   1/2 sessions; what the trader expected/feared; and what remains unmeasured. A query
   whose data is absent must receive that answer instead of an invented winner. Source
   drill-down is for verification/deeper work, not routine reconstruction of the answer.
6. **Improve measurement deliberately.** Review missing-feature suggestions, select
   small shadow studies, and check later sessions. Keep the local layer mainly faithful
   summarization and explanation; deeper interpretation belongs to the frontier review.

Before each build, follow AGENTS/AGENT_TEAM's tests → builder → independent reproduction
in isolated worktrees. Governing specs: LOCAL_AI_AUTOMATION_PLAN (mission, digest,
stages and N3), REVIEW_LEARNING_LOOP, JOURNAL_RELIABILITY_AND_UX_PLAN, warehouse plan/
decisions, AUTO_MODES_AND_QUIET_HOURS, chart-unification, relevant DESK_INTERNALS entries,
and plan.md §§5–7/P6. Existing owner candidates are the source registry/digest/index,
market journal service/capture/context, outcome readers/evidence_stats, journal services
and the recap/research UI. Names are starting points; verify actual ownership before
assigning files. No new runtime module/store is authorized merely by this list.

Acceptance must cover: correct long/short and option exposure; missing risk/fees/FX;
partial closes and duplicate decision links; unhit targets and missing bars; entry-close
EOD and early closes; two-session selection versus two-session follow-through; stale/
unavailable warehouse/import; multiple simultaneous theses and mid-session changes;
late forecast imports and corrected notes; lookahead-free pivots/earnings; UI/export
cell parity; no fabricated citations/numbers; timeout, model-offline and empty output;
and supersession without history loss. Audit a winning AND losing example from original
note/forecast → known context → opportunity/trade → measured result → displayed sentence.
Confirm the compressed version preserves uncertainty and contradictory notes. Compare
time/tokens with the current multi-hour review and test whether an unfamiliar reader
can find the five requested answers without opening raw stores. Tests alone do not
close live gates; inspect a real night's report and a later matured session together.

This edit changes planning only. Checkpoint/glance is refreshed; CHANGELOG, plan.md,
governing runtime specs and app baseline stay unchanged because no implementation,
contract in force or promotion changed. No new Markdown/status file is introduced.

### Fable's selection order and optional additions

Read 10K first for the shared review/AI work. Its dependency sequence replaces the
separate recap → tagging → narrative ordering below where they overlap; preserve the
independent discovery/watchlist/chart work and do not duplicate the readers or screens.
Remaining broad sequence (proposal, not a roadmap promotion):
1. A/B delivery diagnosis and 10K's verified-output repairs; recheck the dated 5C findings.
2. Follow 10K's shared facts/intent/context/UI/export sequence for D/E/F/I/J and items 5/7.
   Define their identities early; raw note capture can proceed alongside independent
   numerical work. Coordinate shared Journal/UI edits serially.
3. G's one Watchlist and H's history/loading foundation remain separate, reusable work.
4. After H, C's H1/H4 watches advance one trigger at a time with their own rule fixtures
   and recorded selection. A recap or narrative build does not authorize a new detector.

Optional suggestions based on the existing wishlist, not extra authorized work:
- Put “data as of / last success / why absent” in the existing relevant status strips.
  This connects A/B/E/F without another health dashboard.
- Link F's examples to one small, dated habit/thesis to test next, as item 5F proposes;
  retain counterexamples and review it on fresh sessions rather than declaring success now.
- Connect item 6's theta study and item 7's D1 environments to D/F only when built and
  measured. Keep qualitative market stance, hypothetical pick outcomes and traded money
  distinct. Neither study should delay fixing stale scans or missing watchlist names.
- Keep item 8's like/reject badges source- and horizon-aware when integrating G; a pass
  can be shown without becoming a veto or “Reviewed today”. Item 9's AVWAPE-side filtering
  stays a separate detector/display decision and must not sneak in as a discovery repair.

Before each selected build, Fable records scope, actual source owners, golden/failing
tests, any remaining trader decision and live acceptance gate in the existing plan/docs.
Afterward reconcile checkpoint/CHANGELOG/specs, advance only completed work and keep live
gates open. No new roadmap, packet-status file or committed assessment is needed. This
planning pass changes no runtime contract, so plan.md and the app baseline remain unchanged.


11. Bring JumpStarter's workspace memory changes here (trader, 2026-09-11)

STATUS: BUILT 2026-09-12 on `main` (trader: "integrate the memory changes"; plan.md Phase
0.25, gate #93 owed). Written 2026-09-11 as CANDIDATE: trader explicitly asked to add the memory changes from their
JumpStarter GitHub repo to this wishlist. Plan the adaptation; do not run a blanket
retrofit, overwrite this repo's instructions or install a memory system in this task.
This is development-agent recall, separate from Trade Mentor and the bot's market memory.

Verified source: [Isidore94/JumpStarter](https://github.com/Isidore94/JumpStarter),
GitHub `main` at `664e08348f0cd1e234518f610185c983036d850a` on 2026-09-11,
matching the clean local checkout at `C:\Users\Aaron\JumpStarter`.
Read the integrated M1 change and its follow-ups, not just the initial patch:
`48d86e9` (hierarchical memory), `3234bd2` (index authority), `556cffd`
(maintenance/guidance), `664e083` (integration and verification).
Primary references at that revision: `CLAUDE.md` “Workspace memory”, `MEMORY.md`,
`memory/`, `docs/INTERNALS.md` “Workspace memory is request-grounded”,
`docs/CODEX_NOTES.md`, root `.codex/agents/*.toml` and CHANGELOG's M1 entry.
M1 changed JumpStarter's own workspace guidance; its templates were not changed.
Do not assume `jumpstart init` or copying templates installs these memory changes.

What to carry over, adapted to TradingBotV3:
- A small `MEMORY.md` routing index: name → detail file → trigger keywords, not stored
  facts. Detail areas for people, project knowledge and decisions, plus dated daily
  notes and prunable temporary context. Read the index, then only matching details.
- Before recalling earlier decisions/preferences/work, use the narrow relevant memory
  sources; JumpStarter limits recall to five sources and cites file, provenance and date.
  This does not cap the source investigation needed for a real implementation or audit.
- Keep only non-re-derivable knowledge. Each durable detail line has a source, date and
  `[stated]`, `[observed]`, `[inferred]` or `[suggested]` provenance. Do not turn a model
  guess into a trader statement. Do not duplicate git history, live counts, generated
  plans, broker data, keys or machine status that should be checked at its real source.
- Supersede obsolete detail beside its dated replacement while retaining the old line
  struck through. Fix stale routing; an index or future search database is only a locator.
  Update the index alongside detail changes and consolidate before its source-defined
  size caps, rather than letting daily logs grow into another giant startup brief.
- Preserve the source's distinction between direct trader corrections and inferred
  lessons. JumpStarter's evidence threshold for inferred standing lessons is at least
  three weighted independent signals across two sessions (signals older than 30 days
  count half). In this repo that threshold alone must NEVER authorize a detector change,
  overwrite an accepted decision, promote a wishlist idea or bypass the ask-first rule.
  Failure memories describe what broke and what fixed it; they are not executable orders.
- Adapt role guidance: read-only recon/review reports sourced proposed memory corrections
  to the lead; memory writes stay inside each worker's authorized scope and integration
  stays with the lead. Preserve TradingBotV3's existing model routing and isolated workers.

Reconcile authority before copying anything:
`CURRENT_CHECKPOINT.md` remains the current-work brief; `plan.md` remains build order and
promotion authority; CHANGELOG remains implemented history; accepted decisions/specs
remain the contracts. Memory points to those sources instead of competing with them.
JumpStarter's “detail is authoritative” applies to memory detail versus its routing
index, not memory versus verified code or this repo's control set. Its idle-boot rule
must not weaken this repo's mandatory narrow reads once a task exists. Explicit user
directions and verified current evidence still win. Inventory any existing Claude-local
memory before migrating; do not bulk-copy stale auto-memory or another project's notes.

Expected selected implementation: add the local index and narrowly seeded detail files;
edit CLAUDE then regenerate byte-identical AGENTS; adapt relevant existing agent guidance;
classify new Markdown in docs/README and reconcile checkpoint/CHANGELOG. Update an
existing internals entry or decision record if the authority contract needs explanation.
No new roadmap, handoff or status ledger. Do not import JumpStarter's sample operator
facts or approval record as though they were TradingBotV3 decisions.

Verification: inspect the final integrated source revision again; static checks for
resolvable routes, provenance/date/source, caps, stale/conflicting entries and identical
CLAUDE/AGENTS. In fresh Claude and Codex sessions, ask a bounded prior-preference question
and verify that only the relevant memory files are read and the answer cites its source;
ask a live-status question and verify it reads the checkpoint/code instead of stale memory.
Check that read-only helpers write nothing and neither recall nor a suggested lesson
authorizes app work. Use docs/config checks; no trading-engine test run or desk restart
is needed for a memory-only adaptation. JumpStarter's own passing tests do not verify
the adaptation here. Coordinate with the already-listed root instruction-file trim.
