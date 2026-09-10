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
