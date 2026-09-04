# Roadmap archive — Phases 0.8, 0.9, 0.11, 0.12 and the Phase 0.13 packets, long form

Verbatim build narrative for BUILT phases of [`plan.md`](../plan.md) Section 12, moved
here on 2026-09-03 by the F1 docs packet (the roadmap was 1,884 lines, ~1,300 of them
describing work already built).

**`plan.md` keeps the roadmap; this keeps the story.** Each moved phase leaves a stub in
`plan.md` with its status at the move, and every live gate still owed is a numbered row
in `CURRENT_CHECKPOINT.md`'s open-gates table. Phase 0.10 and Phase 0.14 stay in
`plan.md` in full. The earlier archive is
[`ROADMAP_ARCHIVE_PHASES_0.5-0.7.md`](ROADMAP_ARCHIVE_PHASES_0.5-0.7.md).

**This is evidence, not authority.** An owed gate is owed because the checkpoint table
says so, not because a sentence here does.

---

### Phase 0.8 — GUI fluidity Wave P1 (authorized 2026-08-26)

Source: `docs/GUI_REDESIGN_PLAN_2026-08-25.md` §11.1 / Wave P1. The trader
promoted **Wave P1 only** on 2026-08-26. Waves U1–U3, S1 and the experimental
Snappy mode (P2) remain PROPOSAL and are not authorized; do not build from them.

Naming note: the proposal calls this "Wave P1" while Phase 1 below already uses
`P1.x` item ids. Items here are numbered **G-P1.x** so the two never collide.

**Scope bound.** Presentation and threading only. No detector, scorer, alert,
queue, scheduler, evidence stream or storage behavior may change, and no read
may be added or removed — only moved off the Qt thread. `alert_center_panel.py`
stays fenced under the file-scoped ask-first rule; the trader's 2026-08-26
pre-authorization in that file covers the quick-journal symbols attachment and
nothing else.

1. **G-P1.0 Three verified defects.** *Built and pinned 2026-08-26 (`db99271`).*
   The AWAY Recap called `load_focus_map(side)` against a keyword-only
   signature, so it reported the Focus lists unreadable on every run; its
   adoption-gate line called `mover_state(side, None, None, None)`, which can
   only return UNKNOWN, and rendered that as a verdict; the Desk quick-journal
   write dropped the chart symbol `write_entry` already accepts.
2. **G-P1.1 Weekend Prep off the Qt thread.** *Built 2026-08-26 (`d050ee1`).*
   The measured 8.45 s freeze. `WeekReviewPage` and `FocusReviewPage` now read
   on an owned single-flight worker; last-good survives a refresh and a failed
   read is stated, never blank. Panel shutdown joins every page.
3. **G-P1.2 Focus mover-state memo.** *Built 2026-08-26 (`0f04240`).* 36
   repeating stalls / 5.93 s. Resolved once per (symbol, side) per mover-refresh
   cycle, discarded by the poll signal that produces a newer measurement. A
   failed measurement is never cached.
4. **G-P1.3 Interaction id on the stall log.** *Built 2026-08-26 (`6bd7eef`).*
   `scripts/ui/interaction_trace.py` plus stamping in the stall watchdog, wired
   at page select, the Journal inner tab and the chart request. Diagnostics
   only; a test parses the module and fails if it can ever sleep, wait or start
   a thread. **Owed:** `first_paint` and `chart_ready` marks, which need the
   receiving paint path instrumented rather than the emit seam; and the Alert
   Center inner tab, which is fenced.
5. **G-P1.4 Convert hot `QTableWidget` rebuilds.** *Built 2026-08-26
   (`49744a7`).* System Health's three tables are written in place rather than
   rebuilt, with scroll position held across the update and selection still
   surviving by check id.
6. **G-P1.5 Audit every `reload()`** reachable from a click or page selection.
   *Audit DONE, one fix landed 2026-08-26 (`49744a7`); the remainder is listed
   below and is NOT done.*

   Fixed: `WarehouseReadoutPanel.refresh` read the DAS research lake inline —
   the only read in the audit that leaves the machine, against a share known to
   drop — and blanked its table on every failure path. Now on a single-flight
   worker, keeping last-good on failure while still clearing on a successful
   empty read.

   Audited clean: `WeekAheadPage` and `DiscoveryPage` both refresh through
   service signals, so `weekend_prep_panel.py` is fully off the Qt thread.

   **Still owed — eight panels with a click-reachable read and no worker at
   all:** `setup_tracker_panel` (12 IO call sites), `industry_panel` (6),
   `master_avwap_panel` (4), `master_market_prep_panel` (3), `theta_panel` (2),
   `watchlists_panel` (2), `rs_window_panel` (1), and `universe_panel` (has a
   worker, reload unaudited). Each needs the same treatment and its own
   fail-before-fix test. None was touched: a partial conversion of a page is
   worse than an honest list of which pages still need one.

   **Order them by MEASURED blocked time, not by that IO-call count.** The
   2026-08-26 pre-fix session (`CURRENT_CHECKPOINT.md` carries the full table)
   says the two costliest non-GC sites left are `widgets/data_table.py:35`
   (7.9%, 115 s) and `models/theta_table_model.py:72` (5.4%, 79 s — and the
   single worst stall of the day at 49.25 s), followed by
   `watchlist_utils.py:33`'s `read_text` (3.9%) and `project_paths.py:165`
   (2.1%). `theta_panel` is second-to-last on the IO-count list and near the
   top on the one that matters.

8. **G-P1.7 The cyclic GC is the largest addressed-by-nothing cost.** **NOT
   STARTED, and not authorized here** — `_GuiGcController` is a live scheduling
   component, not presentation. Recording it because the measurement is
   unambiguous: `collector(2)` and `collector(0)` together took **17.1%
   (248 s)** of the 2026-08-26 session's blocked time, and the desk was observed
   at ~1 GB after ~8.5 hours the same day. Same subsystem as the 2026-08-21
   incident (8 GB in 90 min, 298 s then 200 s sweeps). Any work here is a
   trader decision and needs its own authorization.
7. **G-P1.6 The HealthPanel audit thread outlived its panel.** *Fixed 2026-08-26
   (`49744a7`), found by the G-P1.5 audit and pre-dating this wave.*
   Constructing the panel starts a daemon thread that emits a Qt signal back
   into it; `shutdown` stopped the timer and never joined the thread, so it
   could emit into a freed C++ object — an access violation, not a Python
   `RuntimeError`, so the guard at the emit could not catch it. Intermittent:
   4 runs in 6 segfaulted an unrelated Qt test two files later. **Worth a
   sweep:** any other panel that starts a bare `threading.Thread` and emits a
   Qt signal back into itself has the same defect. This wave fixed the one it
   tripped over, not the class.

9. **G-P1.8 The 2026-08-31 desk lockup: a burst of one signal is one reaction.**
   *Built 2026-08-31, branch `claude/focus-refresh-storm`; live gate 19 owed.*
   ~500 s of GUI-thread blockage in a 16-minute session, worst stall **44.3 s**,
   Windows Not Responding, the desk killed twice. Cause: the DESK drain adopted
   **45 staged picks one at a time** and five `focusChanged` listeners each
   treated one add as a full rebuild. Fixed by coalescing at every listener
   (`ui.timer_utils.SignalCoalescer`, 200 ms leading-edge window) while the store
   keeps emitting per mutation; by making `FocusSideEditor.refresh()` the diff it
   already claimed to be (it still emptied and refilled the flow layout on every
   call); by narrowing `record_bounce_alert` to one chip; and by capping the
   drain at `AUTO_ADOPT_BATCH_LIMIT` (10) adoptions per cycle — pacing only, no
   pick dropped, a deferred pick never marked seen.

   **Authorization:** the trader approved the drain cap and the redraw slowdown
   on 2026-08-31, and approved the `alert_center_panel.py` feed-rebuild
   coalescing separately under the file-scoped ask-first rule. That fence is
   otherwise unchanged and this authorization does not extend past those two
   edits.

   **Deliberately NOT done, and why** (the packet allowed the cheapest 80% here):

   * **The table model resets.** `SetupTableModel`, `TrackerTableModel` and
     `ThetaTableModel` still `beginResetModel`/`endResetModel` on every
     `set_rows` instead of emitting `dataChanged` for the cells that changed.
     Left alone on measurement, not on effort: `setup_tracker_panel`,
     `theta_panel` and `daytrade_tracker_panel` own **no timer at all**, so
     those tables rebuild on an explicit refresh or a service signal, never per
     tick. The 2026-08-31 delegate samples came from *repaints*, and the burst
     that drove them is the one now coalesced. Converting to row-identity diffs
     also has to preserve sort and selection, which is its own packet.
   * **`fit_columns` / `apply_width_rule`.** `data_table.py:170`
     (`resizeColumnsToContents`) and `:135` (`classify_columns`' per-cell
     `model.data`) both appear in the 2026-08-31 samples and both still run a
     full measurement on every table rebuild. `data_table.py:35` was already the
     costliest non-GC site of the 2026-08-26 session (7.9%, 115 s), so this is a
     known, measured, unconverted cost — it belongs with G-P1.5's owed panels
     rather than with a lockup fix.
   * **The GUI-thread GC controller.** Untouched by design: its ~600 ms young
     sweeps that morning were a *symptom* of this churn, and G-P1.7 above still
     says any work there is a separate trader decision.

Gates: **the live-session soak in the proposal's §11.3 is OWED and cannot be
discharged by any test run.** Its acceptance targets are stall count, p90 and
worst-case blocked time measured over a real desk session with the watchdog
enabled — deterministic tests prove the reads moved, not that the desk feels
different. Re-run the §14 performance workflow on the same sequence and compare
against the 2026-08-25 capture (264 stalls, 117.3 ms median, 205.1 ms p90,
8.45 s worst, 46.0 s blocked in ~45 min) before calling Wave P1 done. No
packaging trigger applies to the work landed so far.

### Phase 0.9 — GUI follow-ons from the 2026-08-26 live session (authorized 2026-08-26)

Source: `docs/GUI_REDESIGN_PLAN_2026-08-25.md` §15 decisions 9, 10, 11, 14,
accepted by the trader on 2026-08-26 ("i authorize all changes") with the
recommended answers. Waves U1–U3, S1 and Snappy P2 are still NOT authorized.
Same scope bound as Phase 0.8: presentation and threading only; no detector,
scorer, alert, queue, scheduler, evidence or storage behavior changes;
`alert_center_panel.py` stays fenced (file-scoped ask-first). Each item gets
its own fail-before-fix test and a soak between fluidity slices.

1. **G-P2.0 Table width rule** *(BUILT 2026-08-27, `1fd9e6e`)* (proposal §12,
   §3.4 A). Tables stretch to the
   available width, the widest text column takes the slack, identifiers elide
   in the MIDDLE. Apply through the shared shell, not per panel; first on
   Weekend Prep ▸ Focus pick review (`human_foc…`) and AWAY Recap (`Line`).
2. **G-P2.1 AWAY Recap as a return surface** *(BUILT 2026-08-27, `a5fa6a9`)*
   (§8.3, decision 9). Hide-and-count
   scanner status rows (blank symbol, `WATCH`) in the recap panel only; a
   visible `Chart` action plus `Enter` on the selected row; symbol-less rows
   rendered distinctly with no chart action. The Alert Center's backing list
   is not changed.
3. **G-P2.2 Desk Journal route** *(BUILT 2026-08-27, `fd76923`; the trader
   approved the exact diff in chat before the fenced edit)* (§5.3, decision 10).
   One shortcut that selects
   the Journal tab and focuses the composer, plus a hint on the tab label. No
   second row under the charts; a verb-row verb only if the trader asks for a
   mouse route. Touches the fenced file: ask before the edit.
4. **G-P2.3 Next fluidity slice, in measured order** *(NOT STARTED - gated on
   SOAK 1)* (§11.1, decision 14):
   `DataTable.fit_columns` bounded measurement; the Theta refresh (explain the
   3.0 s → 26.6 s → 49.2 s growth first, then parse on a worker and diff rows
   into the model); `watchlist_utils.read_text` off Qt; `project_paths` `stat`
   measured before touched; then the eight G-P1.5 panels one whole page at a
   time. The panel-thread sweep (G-P1.6's class; candidates in proposal §2)
   rides along.
5. **G-P2.4 GC measurement packet** *(NOT STARTED)* (decision 11). Measurement
   FIRST: what
   produces the cyclic garbage, sweep cost per generation, growth per hour.
   **No scheduling change is authorized by this item** - a change to
   `_GuiGcController` needs its own ask with the measurement in hand.
6. **G-P2.5 The desk's 8-13 GB memory jumps** *(BUILT 2026-08-27 on
   `claude/warehouse-build-memory`; ONE live gate owed)*. Trader-authorised
   through `docs/analysis/OPUS_BUILD_PROMPT_DESK_MEMORY_2026-08-27.md`, which
   rests on the 2026-08-27 (10:00) investigation entry in
   `CURRENT_CHECKPOINT.md`. Three causes, all three fixed:
   - **Session-scoped warehouse reads.** `ResearchStore.read_rows` filters in
     Arrow before `to_pylist`, and the three `bar_m5` readers use it
     (`aggregate.build_derived_bars`, `features.build_intraday_snapshots`,
     `cli._run_outcomes`). Measured on the live lake: the month partition is
     8,704,108 rows / 408 MB / **15.4 GB** as dicts, against **0.53 GB** for a
     full session and **0.31 GB** for a 20-symbol outcome read. Equivalence is
     asserted against a longhand reference implementation of the old read, not
     assumed. BD-74.
   - **The 1.03 GB tracker snapshot.** `ingest_artifact` hashes the file in
     chunks and answers the watermark BEFORE `read_bytes`, and a SNAPSHOT over
     64 MB is stored whole but not `json.loads`-ed. For `setup_tracker` that
     loses nothing measurable - it declares neither `event_keys` nor `id_keys`,
     so the parse fed only the `quality` flag, and a test asserts the parsed
     and skipped rows are identical. BD-73.
   - **The BounceBot `self.data[reqId]` leak.** Five request paths freed the
     ready event and left the bar buffer (~206 KB each, ~400 a cycle, 1.5-2 GB
     a session). They now free both, on the success AND timeout branches, and
     `historicalData` drops bars for an unknown reqId instead of re-creating a
     buffer nobody will free. The trader authorised this one `legacy.py` edit
     and nothing else in that file; it was verified LIKE a detector change -
     the golden fixtures and all 411 BounceBot tests pass unchanged.

   **Live gate owed (one DESK session, after the trader restarts):** the first
   swing-scan slot's build keeps the desk under **3 GB** working set
   (`Get-Process -Id <pid> | select WorkingSet64`, sampled across the window
   the lake manifest shows for that build); the manifest still gains the same
   datasets for that session; and the desk's baseline stops creeping between
   builds.

   **Decisions, not owed work:** moving `run_build` into a child process was
   considered and NOT done - the in-process single-flight lock, the spool seal
   and the ledger's `_record_job` all assume one process, and the filtering
   removes the growth on its own (BD-74). It remains available if the live gate
   shows it is still wanted; that is the trader's call.
   **SUPERSEDED 2026-09-03 (packet F1, trader-authorized, BD-95): the build DOES
   run in a child process now**, because the problem it solves turned out to be
   CPU rather than memory - the build thread held the GIL in 82.7% of py-spy
   samples and froze the desk for a morning. None of the three concerns above
   bit: `single_flight` is a lock FILE keyed on a pid with dead-holder reclaim
   (verified live - a `-m research_warehouse.cli build` run was refused by the
   desk's own in-flight build), `seal_spool` never touches the active `.open`
   segment because it already belongs to another writer, and `JobLedger` is an
   append-only file that replays. The memory gate above still stands as written.

   **Observed in the same session, unchanged, NOT authorised here:** the RRS
   scan's O(n^2) intraday profile (CPU, not memory); the
   `_poll_focus_d1_interest` -> `FocusSideEditor.refresh` GUI stalls
   (`focus_picks_panel.py:441`, 392 s on 2026-08-27); and the RS-window
   `_auto_tick` reading 1,412 daily parquet files on the GUI thread
   (`rs_window_feed.py:745`, 92 s). Separate packets.

Gates: the Phase 0.8 live soak still comes first; **SOAK 1 (after G-P2.2, before
G-P2.3) is OWED and is the gate on item 4** (item 6 is independent of it - it is
warehouse and BounceBot memory, not GUI threading) - see `CURRENT_CHECKPOINT.md` for the
command and the baseline numbers; each G-P2.3 slice is then followed by a soak
against the archived 2026-08-26 baseline. Build prompt:
`docs/prompts/GUI_PHASE_0_9_OPUS_PROMPT.md` (two soak stops inside it; run
after the Phase 0.10 session, same checkout - which, note, cost a stash
collision on 2026-08-26: one build session per checkout).


---

### Phase 0.13 packet P3 — The fact pack tells the truth (2026-09-01) — BUILT, live gate owed

Authorized by the trader pasting the packet. Built on `claude/p3-fact-pack-truth`.
Shadow-only: nothing reaches a detector, score, alert, Focus list or watchlist.
Recorded as BD-81 … BD-85.

1. **Episodes beside rows.** `n_episodes` on every cell; the floor still counts rows.
   The measurement showed the per-cell count equals `n` in all 756 cells, so the pack
   also publishes `evidence_shape` and BD-81 names the follow-up as a CROSS-CELL floor.
2. **The eligible block leads**, with a bounded ineligible block ordered thickest-first
   and per-block drop counts.
3. **Non-trade families excluded and reported** by an explicit role map, until P7's
   setup registry owns it.
4. **Coverage published** — buckets covered, families with zero outcome rows, first M5
   session — so "not measured yet" reads differently from "measured and flat".
5. **`slice_readout` can read every family** without widening the pinned `SLICE_SETUPS`;
   the readout panel gains a family filter and four already-computed columns.

**Owed:** the optional `cell_history` block over the sibling packs on disk.

**Live gate (#32):** one overnight `setup_research` run whose Markdown opens with the
eligible block, shows `n_episodes` beside `n`, names the excluded families and prints
the bucket-coverage line; and the trader confirms the readout panel lists more than two
families.

**Amended by R3 (2026-09-02) — gate 32 was never reachable as written.** The pack this
phase grew is the same pack the narration was sending whole, so from 2026-08-31 every
run sheared its prompt and published nothing but siblings. Gate 32 asks about the
**Markdown**, which was always written and is unaffected; it is still owed and still
satisfiable. What R3 adds is **gate 40**: one overnight run that publishes **exactly
one** pack for the date, with a narration beside it. Read them together on the same
night — the Markdown answers 32, the file count answers 40.

**Not built here, and named so it is not rediscovered as an idea:** a narration retry
that re-reads the pack from disk and calls the model again. It would be correct and it
is cheap, and it is deliberately absent because the retry that existed was the fault —
a second attempt is only safe once something can vary between the two, and today
nothing can.
### Phase 0.13 packet P7 — One name per setup (2026-09-01) — BUILT, no live gate

Authorized by the trader pasting the packet. Built on `claude/p7-setup-registry`.
Two READ-ONLY modules; **nothing in production imports either**, and no runtime
behaviour changed.

1. **`scripts/setup_registry.py`** + frozen `setup_registry_v1.json` (57 entries,
   `setup_id@version`), generated by `scripts/build_setup_registry.py` from FIVE
   naming sites - the packet named four; `legacy.py`'s `*_STUDY_FAMILY` constants
   are the fifth, and eight of those families are named nowhere else. Appendix C's
   role vocabulary; eight `known_divergences` recorded rather than resolved.
2. **`scripts/research_warehouse/trial_ledger.py`** - one append-only row per
   registered grid, written before any outcome is inspected, never rewritten. Four
   grids backfilled with their real authorization pointers.

**P4.1 IS WHERE THE REGISTRY BECOMES AUTHORITATIVE.** Until then it describes what
the code already believes. P4.1 owns: choosing which spelling is identity for each
of the eight divergences, and filling the columns P7 deliberately left blank
(supported sides, timeframe roles, the exact completed-bar trigger, the primary
recipe).

**Owed and NOT part of this packet:** replacing packet P3's two-entry role map with
`setup_registry.fact_pack_role` - that map is on `claude/p3-fact-pack-truth` and not
on `main`, so the two-line swap belongs to whichever of the two branches merges
second. Same for `HTF_LRSI_RECIPES`, whose grid the ledger already declares.

**Live gate: none.** Green tests are the whole gate; this packet changes no runtime
behaviour.
### Phase 0.13 — Trader decisions of 2026-09-01 (packet P0) — BUILT, live gate owed

Authorized by the trader in chat on 2026-09-01 (three quoted decisions). Built on
`claude/p0-apply-decisions`. Nothing here is a threshold change and nothing reaches a
detector's scoring.

1. **BANGER retired.** A top-alert class with a matcher and no producer: the literal
   token match, the tier-gate bypass, the always-sound and both repetition escalations
   are removed. The `banger` review-event column survives as a constant `False` so the
   historical rows and the schema id are unchanged. PROVEN is the top class.
2. **LRSI M5 alerts retired, evidence kept.** `LRSI_M5_ALERTS_RETIRED` gates the emit
   seam only. Detection, the candidate row, `intraday_bounce_outcomes.csv`, the tier
   and the PROVEN stamp keep running; the detection toggles stay `True` because they
   gate detection, not delivery. The "measure them on different timeframes" half of
   the decision is the Phase 0.12 packet B warehouse study, already built.
3. **Clicking away is a pass** — recorded in `docs/DESK_INTERNALS.md`; no code change.
   The `clicked_away_from_m5_alert` reason string is frozen (`review_learning` keys
   on it).

**Owed and deliberately not built:** LRSI as a context suffix on other M5 alerts (the
"boost" the trader described). `_format_bounce_alert_message` takes no bars, so it
needs plumbing through the champion alert callers — a champion-path change, not a
display tweak. Bring it back as its own packet if the trader still wants it.

**Live gate (#29):** one DESK session with no LRSI line on the M5 alert bar, `lrsi`
rows still arriving in `intraday_bounce_outcomes.csv` that day, and no BANGER branch
left in the alert path (grep).
### Phase 0.13 packet P1 — Grade what you already said (2026-09-01) — BUILT, live gate owed

Authorized by the trader pasting the packet. Built on `claude/p1-grade-what-you-said`.
Evidence-side only: nothing here reaches a detector, score, alert, watchlist, Focus
list, review queue or `review_policy.json`.

1. **The human-focus pick key carries its category.** One name on both the swing and
   the M5 list is now two rows and two graded cohorts; before, whichever list was
   snapshotted second was silently discarded and `human_focus_swing_vetted` had zero
   rows. The weekend-prep join and `journal_walkaway` follow the rows.
2. **A like merges into its cohort on the click**, through the same helper the veto
   uses. The nightly slot stays; both merges are idempotent.
3. **A pre-versioning veto pools with the version that introduced its code**, so a
   reason added in a later vocabulary no longer grades as its own cohort forever.
4. **The scoreboard grades every explicit decision** (seven action families, ~640
   decisions) and carries a third callout class, `r_gap`, that asks the R question
   without consulting the take rate. Chart Review's coded vetoes feed the
   `dislike_reason` dimension through a measured join.

**Reported, not forced:** the swing-favorites Focus write-through already existed and
worked (QFIN, 2026-08-31, proves it), and QFIN's `focus_swing_manual` origin is history
rather than a code path — `FOCUS_LIKE_ORIGIN` became `"vetted"` ten minutes after that
like. The packet's `lrsi_cross_20` R gap is real on the un-fixed action sets and closes
once they are corrected; the `r_gap` class is pinned to those literal numbers so it is
proven either way.

**Live gate (#30):** one Weekend Prep opened after the next scan showing a
`human_focus_swing_vetted` row, a like merged on the day it was captured, one pooled
`compressed` cohort, and the `r_gap` callout present in `review_preference_state.json`.
### Phase 0.13 packet P2 — Show me (2026-09-01) — BUILT, live gate owed

Authorized by the trader pasting the packet. Built on `claude/p2-show-me`. Six display
changes, each read-only over a file something else already writes; nothing reaches a
detector, score, alert, Focus list, review queue or `review_policy.json`.

1. **Weekend Prep's judgement tables show the robust half** — median, trimmed mean,
   symbols, sessions, top share, block CI, evidence label — one horizon at a time,
   floor-clearing rows first and by trimmed mean, sub-floor rows greyed and last.
2. **The week page names its callouts** instead of printing two integers.
3. **"My Decisions"** — one tab per scoreboard dimension beside the Daytrade Tracker,
   read off-thread, with a probation badge by set membership.
4. **The five AI phase gates** get a strip on the A.I. Summary page
   (`ai_jobs/gate_counters.py`), every number read from the source that owns it.
5. **A take-rate suffix on the M5 bar row**, from the CACHED guidance only.
6. **A repetition fold on the M5 bar** — presentation only; the bar's
   "deletes nothing, mutes nothing, records nothing, withholds nothing" contract is
   unchanged and its docstring says so.

**Reported, not forced:** the packet assumed guidance is computed before the M5 emit. It
is not — the emit precedes `_queue_score`, so the suffix reads the cache and is silent
for a symbol the desk has not charted yet. Computing it there would put a two-file stat
and a 34 KB JSON re-read on the Qt thread per alert.

**Live gate (#31):** one DESK session where the trader opens each of the six surfaces
and `ui_stalls.jsonl` charges no seconds to any of them.
### Phase 0.13 packet P4 — The variables you are not looking at (2026-09-01) — BUILT, live gate owed

Authorized by the trader, including an explicit yes to the first edit of
`master_avwap_lib/legacy.py` (file-scoped ask-first rule) and to all six Half B items.

**Half A, capture-only.** The attribute leaderboard gets a Qt tab (read off-thread; the
export is 19.7 MB) with the sample floor visible, and twelve variables already on the
record gain attribute keys. A contract-bearing golden frozen from the pre-change code
proves the priority score, bucket and expected R are unchanged.

**Half B, each behind its own fixture.** B1 sample floor on the leaderboard; B2 family
and regime views as sibling files; B3 stale-horizon rows dropped with the count
published; B4 the shipped tier written at assignment time; B5 calibration on structure
points; B6 the representative exit template named.

**Still owed and NOT part of this packet:** re-selecting the scan-factor future row by
exchange session (B3 step b) redefines every historical number and is a full sec-7
promotion; so is pinning `REPRESENTATIVE_EXIT_TEMPLATE_ID` to the documented house exit.
Any weight change consequent on B3's new numbers is likewise a sec-7 promotion.

**Live gate (#33):** one desk scan after which the Attributes tab opens off-thread with
the floor flag visible, the scan-factor coverage line shows the stale-horizon drop
count, and the expected-R note names its template.
### Phase 0.13 packet P5 — Pass and not-today get graded (2026-09-01) — BUILT, live gate owed

Authorized by the trader pasting the packet. Built on `claude/p5-pass-cohorts`. Two new
cohorts complete the set: every verdict the trader can record now has a forward record.

1. **`pass_cohort`** over the annotation log's `pass` rows. Multi-select, so a pass
   grades in k code cohorts AND the pooled `pass_all` — the code cohorts OVERLAP and must
   never be summed. Identity on write is (vocab_version, reason_code). Beside the daily
   horizons it carries a same-session grade when a bar sidecar exists.
2. **`rejection_cohort`** over `pick_feedback.jsonl`. `not_today` and `dislike` are
   separate cohorts whose numbers are never combined into a verdict (the family's
   pooled BASE row exists and is labelled, never read as either); `unfavorite` is not
   graded; the free-text reason is
   carried and never coded.
3. Two nightly slots appended, two Weekend Prep tables, both files added to the evidence
   report and the `trader_judgement` scope (with the like file, which was also missing).

**Owed and NOT part of this packet:** the same-session grade cannot currently be computed
— the bar sidecar holds only bars from BEFORE the pass, so the entry bar it asks for is
never in it. Every row says so through `intraday_unmeasured_reason`. Whether entry should
instead be the last completed close AT the pass is a definition change and the trader's
call; coding the free-text dislike reasons is likewise its own packet with a vocabulary
behind it.

**Live gate (#34):** the trader records two real passes and one not-today on the desk;
the next morning both cohorts have rows and Weekend Prep shows them.
### Phase 0.13 packet P6 — Preference to trade (2026-09-01) — BUILT, live gate owed

Authorized by the trader pasting the packet. Built on `claude/p6-preference-to-trade`.
Three stores each held a third of one question; nothing put the three on one row.

1. **Exact-id auto-tag candidates.** A fifth `AutoTagger` source, `trader_capture`, over
   the statements the trader already made about the symbol INSIDE the trade's own window.
   Ranked above every fuzzy lane, rejections prefixed, `context_row_id` carried as a
   reader's pointer only - plan.md P5.3/P5.4 keep the canonical id.
2. **`preference_trade_outcomes`** - a nightly deterministic slot and Weekend Prep table:
   one row per statement, joined to the journal and to the cohort paper grade, every row
   rendering its match confidence or "no match".
3. **An honest empty-dimension banner** on the journal's "My setups" group below 10%
   confirmed-tag coverage. The group is never hidden.

**Owed and NOT part of this packet:** the canonical opportunity id (P5.3/P5.4) is what
would turn the report's stated confidence into a link; coding the free-text reasons is a
separate packet; and whether `market_journal` should remain in the nightly scope - the
comment corrected here is what surfaced it - is the trader's call.

**Live gate (#35):** the trader imports a real day and one trade shows a `trader_capture`
candidate with a linked event; the nightly report lists that day's likes with a
traded/not-traded column.
### Phase 0.13 packet P6a — Tag the backlog (2026-09-01) — BUILT, live gate owed

Authorized by the trader: *"let's get Opus to do the tagging and I can review after"*.
Built on `claude/p6a-tag-backlog`. One trade in 193 carried a setup tag the trader typed.

1. **`tag_status` on `trade_annotations`** - `confirmed` / `provisional` /
   `needs_review`, arriving through the store's additive migration list, existing rows
   defaulting to `confirmed`.
2. **`scripts/journal_bulk_tag.py`** - the SINGLE authorized exception to invariant I7.
   Dry run by default, idempotent, refuses a confirmed row in the store, never writes
   `tag_corrections`, appends an inert `APPLY_PROVISIONAL_TAG` adjustment per tag.
   Threshold 0.70. **Run on 2026-09-01: 24 applied, 132 marked `needs_review`.**
3. **The review surface** in the Trades tab, and the analytics split ("my setups" is
   confirmed-only; "provisional setups" is its own group).

**Owed and NOT part of this packet:** the Journal's trade list still loads on the Qt
thread (`TradesTab.reload`, `AnalyticsTab._reload`) - measured, reported, and untouched
here; moving it to a worker is its own packet. Coding the 132 `needs_review` trades needs
either scan files that reach back further or the trader's own words.

**Live gate (#36):** the trader opens the Provisional filter on the desk and confirms or
edits at least ten; the "my setups" chart populates from confirmed rows only.

### Phase 0.13 packet P8 / Phase 6.1 addendum — First setup-parameter grid (2026-09-02) — BUILT, live gate owed

Authorized by the trader pasting the packet on **2026-09-02**; that paste date is the
grid's authorization pointer and is recorded in its trial-ledger row. Built on
`claude/p8-param-grid`, off `main` AFTER Phase 0.12, P3 and P7 landed - the packet
declared those as preconditions and they were not met until that morning's merge.

**Declared family:** `AVWAPE_TO_FIRST_DEV`, LONG - the registry's
`avwape_to_first_dev@1` (P7). 840 occurrences over 622 dependency clusters, the
largest cell in the lake.

**Declared question:** does an entry that waits for confirmation (M15 acceptance
close, M5 retest of the trigger, or M30 EMA15/21 controlled pullback) earn more net R
per episode than the first completed M5 close of the next session, under one
structural stop?

**Declared cells (12):** 4 entry moments x 3 targets, stop fixed at
`current_anchor:1`. **Declared floors:** n_episodes >= 30, >= 5 symbols, >= 5 entry
sessions, counted on `dependency_cluster_id`. **Declared window:** the first 20
trading sessions after the packet landed, fixed at registration.

Shadow only: every recipe is `is_diagnostic=True`, nothing is registered in
`outcome_semantics` (BD-80), and no row reaches a detector, score, alert, Focus list
or review queue. Recorded as **BD-88** and **BD-89**.

**Owed and NOT part of this packet:** the conditioning axis. The packet allows ONE
ATR-normalised bucket from the daily feature snapshot, attached point-in-time, three
buckets and not a lattice - it is not built, because nothing yet says the question
needs it, and a conditioning axis added before the unconditioned answer exists is
three more looks against the same k.

**Live gate (#37):** one overnight run publishes rows for every declared cell inside
the 20-minute reserve; the trial-ledger row exists with status `collecting`; and **no
cell is read for a verdict before the declared window closes.**

### Phase 0.13 packet P9 — Quick like (2026-09-02) — BUILT, live gate owed

Authorized by the trader pasting the packet, on their own decision: a like should
be able to say *"something about this was good"* without naming the setup.

1. **Alt+L** writes `like_claim` with `like_mode: "quick"`, no claim, no why -
   and a **BUTTON** on the chart's verb row and on the rail does the same behind a
   popup that takes an OPTIONAL note (trader follow-up, same day). The key never
   prompts; the button always does; cancel records nothing.
   Supersedes R9.2(a)'s why-required for THIS PATH ONLY; Alt+K is untouched. The
   chart retires, `like_advance` is recorded, the symbol is marked reviewed - and
   nothing is placed, because a like carries zero privileges (P3.1).
2. **The bars**, on an M5 chart, through the writer Pass already uses.
3. **`like_mode`** as a picks column, so quick and claimed can be split later
   without rewriting a row. The cohort stays `like_unclaimed`.
4. **`sidecar_completion`**, a deterministic nightly slot that finishes a capture
   sidecar to the session close from the lake or the desk cache, into a NEW file.
   **This answers gate 34's open definition question** - "the first completed
   close after the click" is now a real bar, so the definition does not change.
5. A quick like contributes a LINK to the auto-tagger, never a tag; Weekend Prep
   and the AI scope both say the unclaimed cohort is not a setup's edge.

**Live gate (#39):** one DESK session where the trader quick-likes one swing
chart and one M5 chart - both rows in `trader_annotations.jsonl` with
`like_mode` quick, the M5 one carrying `m5_bars_ref`, both charts retired,
nothing in Focus; the next morning `like_cohort_picks.csv` holds both, the M5 one
has `m5_bars_completed_ref`, and its intraday columns are numbers.

### Phase 0.13 packet P10 — What happens after I like it (2026-09-02) — BUILT, live gates owed

Authorized by the trader pasting the packet. Their two decisions govern it: a star
in Master AVWAP setups and a like in chart review are **the same thing** — one
bucket, graded together, the screen is a column — and *"anytime I like a D1 it
should be treated with respect ... if I like a stock one day it may not be for 3-5
days later that the best entry is."*

**Part A — one like, one dislike, note optional.**

1. **One writer** (`ui/annotations/verdicts.py`). Every like and dislike from any
   screen writes ONE annotation row carrying `surface`. The Master AVWAP ★ and ✕
   reached no graded cohort at all before this. Nothing existing changed meaning:
   the review event, the `pick_feedback` row and the Focus removal all still
   happen, and the annotation row is the addition whose failure is swallowed.
2. **The note is a SECOND row**, joined by `supersedes`, never an edit — and the
   CLICK ROW GOES FIRST, so Escape leaves the click counted. The box opens only
   where no quick button was used.
3. **One bucket, `surface` as a column** on the like picks CSV. Uncoded vetoes
   grade as `veto_uncoded`, never pooled with a coded cohort, and carrying no
   `vocab_version` because they cite no vocabulary.
4. **`note_vocabulary_audit`**, a deterministic nightly slot: the day's notes
   beside the vocabulary that exists, so recurring uncoded words are visible. It
   proposes no code and adds none.

**Part B — a like knows which setup it was.**

5. **B1** stamps the scanner row under the click. A capture click never fetches;
   a bare lookup stamps nothing.
6. **B2** `research_warehouse/like_links.py` — one row per like into
   `bronze_like_occurrence_link`, basis `exact_family` / `any_family` / `none`,
   window one session back and five forward. **Absence is a first-class fact.**
7. **B3** `queries.occurrence_features` — the round-1 audit's item 6, finally
   built. Point in time: never a later session, and never a later REVISION of the
   right session.

**Part C — what happened after the like.**

8. **C1** `after_like_entry_grid_v1` in the trial ledger, written before any
   outcome exists: 20 cells (5 offsets x 4 entries), one stop, one target, floors
   on the LIKE EPISODE, a 20-session window fixed at registration.
9. **C2** `simulate_after_like_entry` reuses P8's selectors and P8's exit machine;
   the offset restricts where the selector may look and never what the simulator
   sees. Parity with P8's control is pinned field-for-field.
10. **C3** an `after_like` block in the nightly pack, a "your likes: best day and
    entry so far" table on Weekend Prep (eligible cells only), and the eligible
    cells in R3's narration view.

**Three differences from the packet as written, each measured rather than
assumed:**

- The packet said B2 needed a "new frozen schema". The slice datasets ARE frozen
  (sec 7.1) and the bronze namespace exists so an additive artifact needs none;
  it is `bronze_like_occurrence_link` on the shared record (BD-90).
- The packet said B1's fields are "the same fields `setup_context_fields` already
  collects". They are not: that function has no `scan_date`, no
  `tracker_setup_id`, no canonical id, and spells the bucket `bucket`.
- **The `unlinked` bucket is a COUNT and not graded cells** (BD-93). The declared
  stop is `current_anchor:1`, which comes from the occurrence's tracker geometry,
  and a like the scanner never found has no anchor. A substitute stop would end
  the grid's one-stop model; dropping them silently would hide how many likes the
  scanner missed. They are counted by named reason.

**Live gate (#41):** one DESK session where a star in Master AVWAP, a like on the
chart-review rail and a "Not today" each leave exactly one annotation row with the
right `surface`; the note box appears only where no quick button was used; and
Escape leaves the click counted.

**Live gate (#42):** one overnight run writing `bronze_like_occurrence_link` rows
and after-like outcome rows inside the 20-minute reserve, with the
`after_like_entry_grid_v1` ledger row present and status `collecting`.

**Live gate (#43) is a REFUSAL, not a check:** no after-like cell may be read for
a verdict before the declared 20-session window closes — including by the agent
that built it, and including if an early cell looks good.

### Phase 0.13 review round R2 (2026-09-02) — TWO GUARDS, BUILT

Authorized by the trader pasting the review.

1. **An empty `assigned_tier` cell is absent, not a tier called NAN.** The live
   feature-history file has no such column; the first scan after P4 widens it and
   every older row reads back as a float NaN, which is TRUTHY and stringifies to
   `"nan"`. `tier_for_tracker_row` now accepts only the vocabulary the stamper
   writes (S, A, B) and treats everything else as absent. Landed before the 07:30
   scan.
2. **A link is not a tag at any seam.** One predicate rejects link candidates in
   the bulk lane, the bulk top pick, Accept/Accept-all and `tag_confidence` - R1
   had covered only `auto_tag_summary`. A pass now carries ALL its codes.

Plus the stale sentences, the atomic overlap-note write, the trade-scoped
adjustment query, the Qt-thread backlog read, and four DESK_INTERNALS entries.

**No new live gates.** Gate 38 additionally watches the Setup Tracker's
current-picks count after the first scan (the NAN fix) and the Weekend Prep
backlog toggle line in `ui_stalls.jsonl`.

### Phase 0.13 review round R1 (2026-09-02) — BLOCKERS FIXED, ALL PACKETS MERGED

Authorized by the trader pasting the review. Eleven blockers across P4, P5, P6, P7 and
P8, each reproduced before it was fixed, then eight merges onto `main` in the order the
trader set: P0, P1, P2, P4, P5, P6, P6a, P8. **Every Phase 0.13 packet is now on
`main`.**

Two gates changed status rather than closing:

- **#33 (P4)** is now SATISFIABLE. It asked for a tier-tracker session; the assigned
  tier never reached the feature history, so the gate could not have passed however the
  session went.
- **#37 (P8)** is now SATISFIABLE. It asked for a trial-ledger row and nothing in
  production wrote one.

**One new gate, #38:** one DESK session on the merged tree after the restart, stall
watchdog quiet on every new surface.

**Owed and NOT done in this round:** the full suite with the `ai_jobs_runner` writer
lock FREE. It was held from 22:00 straight through by the nightly run, and the 32 tests
that stand down under it are explicitly not being called a baseline.

### Phase 0.11 — Theta premium optimization (authorized 2026-08-31) — BUILT, live gate owed

The theta sold-put/PCS report surfaces ~$0.25 credits with untradeable spreads
because the target is literally $0.25 (`$100 / 4 contracts`), the final sort
prefers the lowest qualifying strike (the cheapest option), spreads are only a
soft capped penalty, and the quote budget is spent in `base_score` order with no
premium-richness thinking. Trader decisions (2026-08-31 chat) lock the fix:

1. **T1 Relative floor.** Minimum credit 0.5% of the strike ($1 on a $200
   stock), ideal tier 1.0% ($2), absolute floor $0.40/contract. Below-floor rows
   leave the report. The $100/4-contract framing becomes display-only.
2. **T2 Ranking.** Support first (major SMAs above the strike: 1 required
   unchanged, 2 a big boost, then the covered stack), then yield per market day,
   then spread as a heavy monotonic spectrum (no new hard block — trader:
   "spreads are a spectrum … #1 priority is still areas of support"). The
   strike-ascending sort key is removed.
3. **T3 PCS time.** Credit spreads extend to 15 market days (3 weeks); sold
   puts stay at 10.
4. **T4 Budget allocation.** Enrichment work list orders `thetalongs.txt` names
   first, then estimated premium capacity (ATR%-based, no new network calls),
   then `base_score`. Nothing is dropped; the support-only fallback stays.
5. **T5 Surfaces.** Report + Qt panel carry credit % of strike, yield/week,
   spread %, and the SMA-above-strike count.

Build prompt: [`docs/prompts/THETA_PREMIUM_OPUS_PROMPT.md`](docs/prompts/THETA_PREMIUM_OPUS_PROMPT.md)
(scope fence for `legacy.py`, fail-before-fix tests, IB pacing untouched).
Eligibility rules (≥3 supports, ≥1 major SMA, earnings buffer) and R9.4
`theta_side` semantics are unchanged. Universe coverage already holds at
evaluation time (universe longs join full scans); T4 is allocation, not reach.

**Status 2026-08-31: T1-T6 BUILT and GREEN on `claude/theta-premium`.** Sold-put
credit is judged at >= 1.0% of the strike (recommended) / >= 0.5% (cusp) with a
$0.40 absolute floor, and a quote under both leaves the report. The final sort is
tier -> major SMAs above the strike -> support quality -> yield per market day ->
spread, with the strike-ascending key removed and the spread penalty uncapped.
PCS reaches 15 market days. The quote budget is ordered thetalongs -> estimated
premium capacity (ATR%-based, no new network call) -> base_score. The report and
the Qt panel carry credit %, yield/week, spread % and the SMA-above-strike count.

6. **T7 The spread credit scales with the underlying too. — DECIDED and BUILT
   2026-08-31.** The open question was put to the trader with its arithmetic and
   answered in as many words: *"Yes it should scale with price of the underlying."*
   The credit/width ratio does not scale, because `_pcs_long_strike_choices` caps
   the width at 10 points however expensive the stock is, so the 20% target credit
   stops growing at $2.00 - 1.36% of a $37 short strike and 0.31% of a $644 one.
   `theta_pcs_credit_floor(short_strike)` is now a hard minimum: 0.5% of the short
   strike or the $0.40 absolute floor, whichever is larger, sharing the sold-put
   constants so "the percent floor" has one definition. Under it the spread leaves
   the report; above it the credit/width ratio still decides recommended-vs-cusp.
   The RECOMMENDED percent (1.0%) is deliberately NOT applied here - 1% of a $644
   strike is a $6.44 credit on a 10-wide spread, a 64% credit/width bar no real
   market pays, so using it would delete every expensive spread rather than rank
   it. The report's PCS rows now carry the same `premium=` line as sold puts, with
   `credit_width_pct` alongside `credit_pct`.

   *Consequence, stated rather than discovered on the desk:* expensive credit
   spreads will mostly disappear unless their credit genuinely scales. If the
   trader wants those opportunities back, the lever is the WIDTH cap in
   `_pcs_long_strike_choices` (`max(10.0, preferred_width)`), not the floor -
   widening a $700-stock spread to ~17 points would let a 20% ratio pay $3.50 and
   clear 0.5%. That changes capital at risk per contract, so it was not done
   without asking.

Gate: one desk scan whose theta report shows percent-floored, support-first
rows, with `via thetalongs.txt` labelling intact.

### Phase 0.12 — Focus de-clutter + higher-timeframe LRSI research (authorized 2026-09-01)

Two independent packets, authorized by the trader in chat on 2026-09-01. Packet
A changes the desk; Packet B is a shadow research lane with zero desk cost.

#### Packet A — Focus alert de-clutter — BUILT, live gate owed

The Focus D1 feed had become unreadable, the Armed inventory accumulated
forever, and Focus itself only ever grew.

1. **A1 Pullback-only automatic Focus alerts.** `_poll_focus_d1_interest`
   evaluates the PULLBACK set only - 15EMA reject, AVWAPE and 1σ bounce. The
   EXTENSION set (new 5d/20d extreme, SMA break, AVWAPE / 1σ break) no longer
   fires automatically at all; the trader arms the ones they want per symbol and
   `_poll_d1_event_watches` remains the single path that fires one. The gate is
   at the flag-GENERATION seam - an extension kind is never evaluated, so
   nothing has to be suppressed downstream. Supersedes the 2026-08-05
   one-extension-per-day ration, which had nothing left to ration.
2. **A2 Armed alerts expire, in TRADING days.** A manually armed 5-day extreme
   watch gets 5 sessions; a 20-day one gets 10; every other armed thing - D1
   level watches, any-bounce watches, manual price alerts - gets 10. The clock
   is `market_calendar.trading_days_between`, never weekday arithmetic. Expiry
   runs at the head of the poll that already owns each store, so no new timer
   appears. **Uncertainty never deletes**: a date the calendar cannot reason
   about keeps the entry armed. Every expiry appends a row naming store, symbol,
   kind, `armed_at` and `expired_at`. A price alert is DISARMED rather than
   deleted - it leaves the Armed surface and keeps its levels, note and history,
   so plan.md sec 5's "user-entered names are never auto-removed" still holds.
3. **A3 Focus picks fade.** A pick that has fired no alert and printed no
   pullback event for 10 trading days moves to a FADED list. The clock starts at
   add time and is reset by a fired Focus D1 flag, an armed-watch hit, or the
   trader's own "keep in Focus" on the review chart. It applies to swing AND M5
   picks, the trader's own included - an explicit trader authorization to
   auto-remove a hand-typed name, scoped to Focus alone, through the store's own
   removal path so a hand-maintained watchlist line is untouched. Fading a
   hand-vetted swing pick appends a RETRACTION row, never an edit. It is
   reversible: "★ Restore to Focus" (fresh clock) and "✕ Discard". The check
   runs on the day roll and a half-hourly timer, never inside the 60 s poll.
4. **A4 Buttons and counts.** "Review ▶" is now "Focus pick review (N)", with
   "Faded review (N)" beside it. The faded walkthrough goes through
   `_enqueue_review_alert` - the one door - with `FOCUS_FADED_TAG`, which
   bypasses movers-only the way `FOCUS_REVIEW_TAG` does (a faded pick is by
   definition one that has not been moving). Counts repaint through the board's
   existing `SignalCoalescer` at the listener.

Gate: one desk session where the D1 Focus feed carries pullbacks only, an armed
extension watch still fires, an expired watch leaves the Armed board with a row
behind it, and a faded pick can be restored and discarded from the chart.

#### Packet B — Higher-timeframe LRSI entry research — BUILT, shadow only

"Is there something there" evidence for entering Focus-style setups on LRSI
crosses at M30/H1/H2/H4. Research lane only: it reaches no detector, score,
alert, Focus list or review queue, and promotion remains sec 7's job.

1. **B1 H2 exists.** 120 minutes joins `TIMEFRAME_MINUTES` and
   `DERIVED_TIMEFRAMES`. The locked plan CUT H2 for having no consumer; B3 is
   one, which is the cut's own reopen condition (BD-78). RTH is 6.5 h, so H2 and
   H4 end each session with a stub - published as evidence, EXCLUDED from the
   oscillator's input.
2. **B2 The short legs are unmirrored, and that is a decision.** The efficiency
   formula clamps at 0, so the mirrored-close idiom and `cross_down` are
   different features rather than a transform of one. The study reads ONE series
   for all four legs: cross-up 50/20 for longs, cross-down 50/80 for shorts.
   Rationale, cost and fixture in BD-79. Live `CROSS_LEVELS` unchanged.
3. **B3 A bounded 16-recipe diagnostic grid.** 4 timeframes × 4 entries, one
   stop model (the signal bar's extreme + 0.25 ATR on the SAME timeframe,
   following `DIAGNOSTIC_ATR_STOP_V1`) and one target (2.0R). Never a Cartesian
   search. Alternative recipes on one occurrence stay correlated diagnostics of
   ONE episode. It reads the occurrences and canonical M5 bars the nightly has
   already materialised, so it adds simulation and not a second data pass.
4. **B4 Nothing is registered in `outcome_semantics`.** These rows are warehouse
   `outcome_path` rows keyed by `recipe_id`; they never acquire a bounce family
   and never reach `claim_kind`. BD-80 records the reopen trigger.

Gate: one overnight `setup_research` run producing HTF rows inside the existing
reserve, then a first read of whether any cell clears the evidence floor.
