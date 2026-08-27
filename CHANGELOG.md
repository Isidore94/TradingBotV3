# TradingBotV3 implemented history

Last reconciled: **2026-08-26** on `claude/gui-phase-0-9` (tip `714f717`, which
also carries Phase 0.9 G-P2.0/G-P2.1), at Phase 0.10's
review fixes - the AVWAP band challenger built, shadow-fenced, its fence guarded
at source and its export guarded against costing the tracker save; its three
forward gates owed.

Authoritative for: **what exists and the historical sequence of revisions**

Remaining work: [`plan.md`](plan.md)

This is a curated product history, not a raw commit dump. It reconciles the former
status sections in `plan.md`, the accumulated `CURRENT_CHECKPOINT.md` ledger, the GUI
plans, warehouse plans/reviews, dated handoffs, and Git history. Exact current test
counts remain in `CURRENT_CHECKPOINT.md`.

The labels retain their strict meanings: `IMPLEMENTED` means code exists, `GREEN`
means deterministic tests pass, `LIVE_VALIDATED` requires real-session evidence,
and `PROMOTED` requires an explicit champion decision. A feature can be implemented
and green while its live or promotion gate remains open in `plan.md`.

## Current implemented inventory

### 2026-08-26 night - Phase 0.10 review fixes: the shadow cannot cost the save, and the fence is no longer a hand-maintained list

**IMPLEMENTED / GREEN.** Fable's review of `002f2a3..292e335` returned GO with
two fixes owed before B-4 and one trader decision recorded. All three landed in
`ac9a952` on `claude/gui-phase-0-9`.

**The shadow export is guarded.** `export_setup_tracker_views` wrote the
band-variant CSV as its last statement with no guard, and
`update_setup_tracker_from_scan` runs `save_setup_tracker_payload` AFTER it - so
one malformed setup dict reaching `build_band_variant_stats_rows` would have
aborted the day's tracker save. That is the evidence store costing the thing it
records, which R10 forbids everywhere else in this codebase. The `try/except` +
`logging.warning` wraps the SHADOW write only: every champion export above it is
already on disk by then, and a champion export that fails must still fail
loudly - asserted as its own test.

**The fence is guarded at source.** Seven readers filter on
`_is_band_variant_scenario`, and three of those were found by the parity fixture
rather than by reading the code - so an eighth would not be found by reading
either. `tests/test_band_variant_fence_guard.py` walks the AST of `legacy.py`
and requires every scenario-iteration site to mention the fence inside its
enclosing function or to be named in `ALLOWED_UNFENCED` with its reason. Two
entries, both readers that MUST see the shadow: the stop rebuild on replay
(`_extract_tracker_stop_candidates_from_setup`, which sorts by label so
`VARIANT_*` still lands last) and sealed-record compaction
(`_compact_tracker_setup_record`, which strips the shadow's per-bar event log
exactly as it strips the champion's).

The detector is deliberately wider than the spelling the fence was written
against - `setup["scenarios"].values()`, `.get("scenarios", {}).values()` and a
local `working_scenarios.values()` all count - because a guard that only
recognizes today's spelling is passed by tomorrow's. It finds nine readers where
the narrow `(setup.get("scenarios") or {}).values()` pattern finds six. Proved
against real code rather than a mutation: pointed at `5613eec:legacy.py`, the
tree as it stood before the fence, it reports six unfenced readers. Four
companion tests keep the guard itself honest. It does NOT claim that mentioning
the helper means it was used correctly - a name in a function is not a proof
about its logic, and the parity fixture remains what proves the values did not
move.

**The shadow crosses the four BASELINE exit templates only** (trader decision,
2026-08-26). `_is_band_variant_stop` is the candidate-side twin of
`_is_band_variant_scenario`, kept beside it so the two spellings of "is this the
shadow" cannot drift apart, and `_build_tracker_scenarios` skips experimental
templates for such a stop. The champion is untouched and still crosses all six;
the experimental templates are a comparison framework for the CHAMPION's stops,
and a challenger inside them would be two variables at once. Re-measured:
**9,982 -> 6,524 bytes per new setup** (474 anchor blocks + 6,050 for four
variant scenarios), so **~144 MB -> ~89.5 MB, 15% -> 9.4%** at the live
14,386-setup / 950.2 MB scale, forward only - and 5,739 bytes once sealed-record
compaction strips the event logs. All four baseline templates remain, so the
stats table's per-template pairing is still possible.

Verification: **4995 passed / 19 subtests, exit 0** at `ac9a952`, and **5010
passed, exit 0** on the tip `714f717` once Phase 0.9's `a5fa6a9` (committed by a
concurrent session while this work ran) landed beneath it; smoke 7/7. Eleven
tests added, every one proved failing first. **Owed, unchanged**: T4's three
criteria, >= 20 sessions of forward accrual before T3 counts, and B-4 - which
these two fixes were the gate on.

### 2026-08-26 - AVWAP band challenger: a second formula, computed beside the champion and unable to reach it

**IMPLEMENTED / GREEN. Every T4 gate is OWED and no test discharges one.**
`plan.md` Phase 0.10, governing spec `docs/AVWAP_BAND_VARIANT_STUDY.md`, build
prompt `docs/prompts/AVWAP_BAND_CHALLENGER_OPUS_PROMPT.md`. Branch
`claude/avwap-band-challenger` off `claude/gui-p1-fluidity` at `88a34b7`.

`calc_anchored_vwap_bands` is untouched and frozen (decision 0008). Nothing in
this packet reaches a detector, score, rank, tier, alert, zone arm, Focus list,
review queue or `review_policy.json`.

**B-0 - the formula, replicated** (`002f2a3`).
`scripts/indicators/avwap_band_variants.py`
(`avwap_bands_oneoption_bb20_v1`): an anchored HLC/3 volume-weighted centre with
a 20-close **population** Bollinger sigma as its half-width - the form the trader
pinned on 2026-08-26 against OneOption / Option Stalker Pro. The two halves know
nothing about each other, so the sigma window deliberately reaches back BEFORE
the anchor; that is why the band is already wide on the anchor bar where the
champion's is exactly zero. `indicators/` shape throughout: completed bars in,
immutable aligned tuples out, no I/O, no pandas import, `None` below the
lookback - never padded, never 0.0, never a shorter window.

`tests/fixtures/avwap_band_variant_oneoption_v1.json` freezes OKTA
2026-04-01..06-05 from the durable store **through
`_normalize_daily_bar_frame`**, not an ad-hoc threshold, because that store's
OKTA volumes are mixed-unit. Neither golden row is affected and the fixture says
why. The expectations are the trader's hover readings, not this repo's output:
centre to 0.2% relative (126.78 here against 126.565 there - a consolidated-vs-IB
volume-feed gap), sigma to +/-0.02 absolute (18.039 vs 18.035).

Two discriminators are pinned as arithmetic: the champion's sigma is 0.0 on a
one-bar anchor where OneOption read 10.28, and the killed sample-OHLC form
predicts an upper of 138.09 on 2026-06-02 where the trader read 144.60. The
killed form lives in the TEST, not the module - no live code carries a formula
the study already killed. An AST test forbids the module from importing
`master_avwap_lib` at all.

**B-1 - the hover-comparison table** (`13505d1`).
`scripts/avwap_band_variant_fit.py SYMBOL ANCHOR_DATE [--lookback 20]` prints
both formulas per session since an anchor. Offline; writes nothing without
`--csv`, and then only into `OUTPUT_DIR/reports/`. The champion publishes only
its final bar, so its column comes from calling the frozen function once per
session on a truncated frame - a call, never an edit. An unmeasurable cell prints
EMPTY, because the champion's sigma really is 0.00 on the anchor bar and the two
must not look alike. Live read on OKTA reproduces the study's S2 column exactly.

**B-2 - the tracker shadow** (`5613eec` fixture, `603333b` code).
Golden fixture FIRST: `tracker_record_band_variant_parity_v1` was frozen on the
champion's code BEFORE either fenced file was touched, and it earned its keep.
`runner.build_anchor_band_variant_meta` computes the challenger from the same
frame and anchor index; `current_anchor_variant` / `previous_anchor_variant`
ride `symbol_entry` and the setup record; `_find_tracker_stop_candidates`
appends one `VARIANT_<protective>` candidate LAST with the champion's own
`close_failure_limit`; `master_avwap_band_variant_stats.csv` is written in the
existing export pass and read by a "Band Variant" tab on the Setup Tracker page.

**Appending after the champion's candidates was necessary and NOT sufficient**,
and the prompt assumed it was. `representative_total_r` is picked by label and
did not move - but `_summarize_tracker_setup_outcome` averages `total_r` across
every tradeable non-experimental scenario, and that average reaches
`build_tracker_setup_type_rows` -> `apply_tracker_setup_type_adjustments` ->
`row["score"]`. Measured on the frozen fixture before the fence: `avg_total_r`
-0.0790 -> -0.0755, `tradeable_scenario_count` 8 -> 12, eight summary values in
all, plus `daily_marks[1].scenario_events` 10 -> 15, the short's `setup_status`
CLOSED -> OPEN, and 12 -> 18 rows in the scenario and stats CSVs.
**Trader-authorized 2026-08-26**, `_is_band_variant_scenario` now fences seven
readers. The shadow is still graded - `_evaluate_tracker_scenario_bar` runs for
it exactly as before - its events simply stay off the champion's mark. The last
three fence sites were found by the fixture rather than by reading the code.

Two findings worth more than the code. The challenger's sigma is 1.339 where the
champion's is 0.586 seven sessions after an anchor (2.3x), which is why the
trader's screenshots looked better early. And **"the wider band is stopped out
less often by construction" is only true when entry sits INSIDE the band**: the
fixture's short is entered above both upper bands, the wider sigma pushes the
upper band UP toward entry, and the challenger's stop lands 0.159 away where the
champion's is 0.971 - six times TIGHTER, from the wider formula. T1 and T3 may
not assume a direction.

**Tracker JSON growth, measured**: 9,982 bytes per NEW setup (474 for the two
anchor blocks, 9,508 for six variant scenarios with their event lists) against a
live file of 950.2 MB holding 14,386 setups - about 144 MB, ~15%, if every setup
carried it. It accrues forward only; existing records do not grow until rebuilt.
The study estimated "a few hundred bytes per setup" and was ~30x low.

**B-3 - the D1 overlay, default OFF** (`3abf61d`).
`chart_levels.avwap_variant_levels` builds six sloped lines in the
`avwap_variant` group on the ChartDataService worker, never on the paint path,
anchored on the date the snapshot already resolved so the two lines on one chart
differ for one reason rather than two. **The paint-lines preference file had no
way to express a default-OFF group** - every group defaults ON there on purpose -
so `chart_levels.GROUPS_HIDDEN_BY_DEFAULT` names the exceptions and
`PaintLinesPrefs` gained a `shown_groups` list, letting both defaults live in one
file: an older preference file keeps the group off, the trader's own hidden
groups survive a rewrite beside it, and an unreadable file falls back to the
defaults rather than to "show everything". Four existing paint-lines tests now
assert the amended rule instead of the blanket one.

`indicators.avwap_band_variants` joined `selftest.LAZY_ENGINE_MODULES` - the
first lazy import of it from a path a frozen run can reach. `indicators` was
already in the spec's `collect_submodules`, so no spec edit was needed, verified
rather than assumed.

Verification across the packet: 4968 passed / 19 subtests, exit 0 (baseline
4902); smoke 7/7; `launch_gui.py --selftest` 71/71; spec-drift 17 passed.
**Owed**: T4's three criteria, including >= 20 sessions of forward accrual with
>= 40 finalized setups before T3 counts, and the B-4 backfills (T1 level quality,
T2 playbook re-run) which are the next packet and are NOT started.

### 2026-08-26 - GUI fluidity Wave P1: the desk stops reading its stores on the click

**IMPLEMENTED / GREEN. The live soak is OWED and no test discharges it.**
`plan.md` Phase 0.8, promoted by the trader from
`docs/GUI_REDESIGN_PLAN_2026-08-25.md` §11.1 on 2026-08-26. Presentation and
threading only: no detector, scorer, alert, queue, scheduler, evidence stream or
store changed behavior, and no read was added or removed - only moved.

**Three verified defects, each reproduced at source before it was touched.** The
AWAY Recap called `focus_picks.load_focus_map(side)` against a keyword-only
signature: TypeError on every run, absorbed by a fail-quiet `except` and shown to
the trader as "Focus lists unavailable". The page had therefore NEVER read the
Focus lists, and no amount of the files being present could have changed that.
The same page's adoption-gate line called `mover_state(side, None, None, None)`
against `(side, price, prev_high, prev_low)` - with nothing to compare it could
only return UNKNOWN, which was then rendered as a gate verdict for the symbol. It
now says the gate was not measured here and why, and points at the surfaces that
do measure it; UNKNOWN stays UNKNOWN. And the Desk quick-journal write (Ctrl+Enter,
the one used mid-session with a chart up) dropped the `symbols` field
`MarketJournalService.write_entry` has accepted since R10.H, so the entries most
likely to be about one name were the ones stored with no name.

The shared shape of the first two is worth keeping: **a `try/except` written to
keep a page from crashing had been absorbing a programming error and reporting it
as missing data.** Fail-quiet is right for a store that might not be there; it is
wrong for a call that can never work.

**Weekend Prep now reads on a worker.** `WeekReviewPage.reload` ran
`build_review_learning_state(window_days=7)` plus two RS log scans, and
`FocusReviewPage.reload` ran five CSV/JSONL reads and built five tables of cells,
all inside the click that selected the page - the worst measured stall on the
desk, 8.45 s frozen. `WalkawayPage` in the same file already owned a QThread, so
one `_ReadWorker` now serves all three. Deliberately NOT copied from it: that
page blanks its body to "Running walk-away..." while refreshing. **Clearing a
populated page to announce a refresh destroys the only copy of what it knew**,
most damagingly when the refresh then fails, so the new pages keep last-good
visible and put "refreshing" and any stated error in their own slot. On
`FocusReviewPage` a failed refresh keeps every row: the graded cohorts are the
whole forward record of the trader's own vetoes and likes. That page previously
had no error handling at all - a bad CSV propagated out of the click. Panel
shutdown, which named `walkaway` while that was the only threaded page, now joins
every page.

**The Focus board measures each mover state once per poll, not once per redraw**
(36 repeating stalls, 5.93 s). `_refresh_all` fires on things unrelated to
previous-day extremes - a BounceBot alert, an RS/RW snapshot, a side edit - and
each walked every chip through `AlertCenterPanel.mover_state`, reading the D1/M5
series per symbol per side. Memoized per (symbol, side), discarded by
`refresh_mover_flags`, which is not an arbitrary expiry but the signal that a
newer measurement exists. A FAILED measurement is never cached: a flag is
decoration over a measurement, and one transient miss must not switch it off
until the next poll.

**A stall record now says which click it belongs to.** `scripts/ui/interaction_trace.py`
plus stamping in `StallWatchdog._write`. The watchdog names the frame that held
the GUI thread, which is the wrong question whenever the modal frame sits inside
Qt's own event dispatch and names no application code. The trace is read from the
sampler thread and therefore holds **no lock** - live state is one module-level
tuple replaced whole, because a lock in a diagnostic could stall the thread it
exists to measure. It owns no timer and no thread, and a test PARSES the module
and fails on any call to sleep/wait/start/join/Thread/Timer/QTimer - the
`ScanCycleClock` rule, for the same reason. An empty interaction id means an
idle-desk stall, which is a fact about the stall rather than a gap. Wired at page
select, the Journal inner tab and the chart request.

**Fence discipline.** `alert_center_panel.py` is fenced under the file-scoped
ask-first rule; the trader pre-authorized only the quick-journal symbols
attachment, and the diff there is six added lines and no deletion. The mover memo
was implemented in the consumer rather than at its natural point inside that
file.

**The mover memo moved to its source.** The trader extended the fence
authorization, so the memo now lives in `AlertCenterPanel._measure_mover_state`
as well - the review queue asks the same question once per alert and now gets
the same answer for free. The design came from a measurement rather than a
guess: per (symbol, side), m5 materialization is 0.049 ms and everything after
it is 0.186 ms, so **79% of the cost is memo-able after materialization**. That
is why the key is the identity of the bars measured - session date plus the
length and last timestamp of both series - and not a clock. `mover_state` feeds
the movers-only review filter, which decides what the trader SEES; a time-based
cache would let a name that has just broken yesterday's high stay hidden until
it lapsed. A new bar is a new key.

**System Health stopped rebuilding itself**, and the warehouse readout stopped
reading a network share on the Qt thread. `_fill` and the checks table built a
fresh `QTableWidgetItem` per cell of three tables every 15 seconds - which is
also where the scroll position went, so a trader reading the bottom of the jobs
list was pulled back to the top mid-read, on a timer, with nothing on screen to
explain it. `WarehouseReadoutPanel.refresh` called `ResearchStore.open()` and
`slice_readout()` inline against the DAS lake; that share is known to drop, and
an SMB read against a dropped share blocks until it times out. It was the only
read in the whole audit that leaves the machine. It also blanked its table on
every failure path - an unreadable lake is not an empty lake - and now keeps
last-good on failure while still clearing on a successful empty read.

**The `reload()` audit is complete, and most of what it found is still owed.**
Fourteen panels have a reload/refresh plus file IO; eight own no worker at all.
One was fixed (above); `WeekAheadPage` and `DiscoveryPage` audited clean.
The other eight are named in `plan.md` under G-P1.5, `setup_tracker_panel`
first. Nothing was half-converted: a partial page is worse than an honest list.

**A latent crash the audit found (G-P1.6).** Adding a second
HealthPanel-constructing test file made an unrelated Qt test segfault two files
later - 4 runs in 6. `HealthPanel.shutdown` stopped the panel's timer and left
its audit thread running; that thread emits a Qt signal back into the panel, so
it could fire into a freed C++ object - an **access violation**, which the
`except RuntimeError` at the emit cannot catch because it is not a Python
exception. Reproduced at the committed HEAD with all work stashed, so it
pre-dates this wave. `shutdown` now joins the thread and a `_closing` flag stops
a refresh queued before shutdown (construction uses `singleShot(0, ...)`) from
starting a fresh one after it. **The class is not closed:** any panel that
starts a bare `threading.Thread` and emits a Qt signal back into itself has the
same defect.

Three shutdown lists in this wave named their threaded children by hand and had
each fallen behind: `WeekendPrepPanel` (named only `walkaway`), `ResearchPanel`
(missed the readout), and the `MainWindow` list the readout sits under. Two were
fixed by naming the missing child; the weekend prep one now iterates its pages.

**Every shutdown join is bounded (`e0f78ae`).** Found in live use, not by a
test: the trader closed the window on 2026-08-26, it "froze for a few seconds",
and the PROCESS OUTLIVED THE WINDOW. Four shutdown paths joined their reader
with a bare `worker.wait()`, which has no upper bound - two from this wave
(weekend prep, warehouse readout), two older (journal panel, weekend prep
service) - and the warehouse reader is on the DAS, the one read in the desk
that can block for minutes when the share is unwell, which is exactly when a
trader gives up and closes the app. `ui/read_worker.join_worker` (5 s default)
replaces all four. On timeout the worker is DISOWNED AND PARKED in a
module-level list rather than dropped, because dropping the last Python
reference to a running `QThread` destroys its C++ half mid-run - a crash, not a
leak; these are reads with no side effects and the process is leaving anyway.
`tests/test_shutdown_waits_are_bounded.py` is a source-level guard: a bare
`.wait()` on a shutdown path fails the suite. Tests 4897 -> 4902.

**The proposal is reconciled to the build (docs only, 2026-08-26 evening).**
`docs/GUI_REDESIGN_PLAN_2026-08-25.md` now records Wave P1 as BUILT with
commit ids, replaces its 45-minute fluidity sample with the archived full
pre-fix session (3350 stalls / 1457.5 s; by blocked time, not count), states
what Wave P1 can and cannot be expected to change against it, re-orders the
owed fluidity work by measured time (the two Qt table paths and the growing
Theta refresh first - those are Qt measurement costs, not reads, so a worker
does not fix them), folds in the trader's 2026-08-26 live findings (narrow
columns on every table page; AWAY Recap unusable as a return surface; the Desk
Journal undiscoverable) as a table-width RULE plus page decisions, adds the
build's standing constraints (bounded joins, panel threads, child lists,
never-blank refresh, the fence, the unwired paint marks), and records that
Smart App Control now reads OFF. **One premise of the 2026-08-25 draft was
refuted at source:** its "arm bar contract/source mismatch" - the arm bar is
under the chart by the trader's 2026-08-20 second-pass instruction
(`4c05de5`, "the hotbuttons return"), so the CLAUDE.md/AGENTS.md line placing
it on the Armed tab is the stale one. Flagged for the trader; not edited.
Waves U1-U3, S1 and Snappy P2 remain PROPOSAL. **The trader then authorized
all changes (same evening):** CLAUDE.md/AGENTS.md now say the arm bar is under
the chart, that SAC reads OFF and the source launch stays production by trader
decision, and carry a new rule that chat messages to the trader are written
very simply; `trading_desk.cmd`'s header matches; `plan.md` gained Phase 0.9
(table width rule, AWAY Recap return surface, Desk Journal route, the next
fluidity slice in measured order, a GC MEASUREMENT packet with no scheduling
change). Nothing in Phase 0.9 is built.

**AVWAP band challenger planned, replicated and authorized (same evening, docs
only).** The trader compared their anchored-VWAP bands with OneOption / Option
Stalker Pro's, which are wide from the anchor bar. A one-evening study
(`docs/AVWAP_BAND_VARIANT_STUDY.md`) replicated the vendor's band from three
OKTA hover readings: `AVWAP(HLC/3) ± k · stdev(close, 20, population)` - the
textbook Bollinger σ laid on an anchored HLC/3 centre, no anchor memory
(the anchored sample-OHLC form predicted 138.09 on 2026-06-02; the reading was
144.60). `plan.md` gained **Phase 0.10** (module + fixture, fit script, tracker
shadow stops + stats + panel section, D1 overlay off by default; backfills
after review) with the Opus build prompt at
`docs/prompts/AVWAP_BAND_CHALLENGER_OPUS_PROMPT.md`. Nothing is built; the
champion σ stays frozen (decision 0008) and any promotion would be an
additional level family, never a swap.

Tests 4844 -> 4902, exit 0 (4897 at `49744a7`, 4902 at `e0f78ae`); smoke 7/7.
No packaging trigger. **Owed:** the eight
panels under G-P1.5, the bare-thread sweep under G-P1.6, the
`first_paint`/`chart_ready` marks (which need the receiving paint path
instrumented rather than the emit seam), and **the §11.3 live soak, which is the
trader's to run and which no test discharges.**

### 2026-08-26 - the Phase 0.5 work is on `main`, and the branch chain is retired

**Three weeks of Phase 0.5 development became the trunk.** From 2026-08-04 the work
ran on a nested chain of branches rather than on `main`, because the trader was
running unmerged branch code in production through a scheduled task
(`docs/CHECKPOINT_REVIEW_2026-08-08.md`). The chain ended at
`testing-week-2026-08-24`, which contained every commit of its predecessors, and
`main` was a **strict ancestor** of it. The consolidation was therefore a
fast-forward: 354 commits, 480 files, no conflict, and no merge resolution
performed. `git merge-base --is-ancestor` proved the relationship before the merge
rather than after it.

**The code state on `main` is byte-identical to the state that was verified.** The
only non-`main` content added beside the fast-forward is Markdown, so the
4844-passed/19-subtest baseline recorded for `ed277a7` describes `main` exactly. It
was **not** re-run for this merge: the container this consolidation ran in has no
project virtualenv and Python 3.11 against a project floor of 3.12, so a run there
would have proved nothing. That is a stated limit of this entry, not a claim of
green.

**One unlanded document was brought in.** `claude/trade-analysis-opus-prompt`, a
single additive commit from 2026-08-22, contributes
`docs/prompts/TRADE_ANALYSIS_OPUS_ULTRACODE_PROMPT.md` - the Opus trade-analysis
prompt carrying the scoreboard read, the earliness audit and the AEP DT case. Its
context list still told the reader to load `SOL_PROGRESS.md`, which this repository
deleted when `CHANGELOG.md` and `CURRENT_CHECKPOINT.md` took over that role; the
reference now names the pair. The prompt is classified in `docs/README.md` and is
authorization for nothing.

**Three branches are cleared for deletion, and what they were is written down.** New
`docs/BRANCH_HISTORY.md` records every branch in the chain with its commit count,
date range, tip SHA and disposition, so deleting a merged branch never destroys the
only account of what it held. `claude/ticker-briefs-hardening-imcm8r` (94 commits),
`phase05-r2-focus-gating-strength-board` (150) and `phase05-integration-blitz` (308)
each hold no commit that is not on `main`, proved with
`git merge-base --is-ancestor` against `226fbac`. **The deletion itself did not
happen and is owed to the desk:** the cloud session's GitHub credential pushes but
refuses ref deletion with `HTTP 403`, with no proxy policy denial recorded, and the
GitHub MCP surface has no delete-branch counterpart to `create_branch`. The three
commands are in `docs/BRANCH_HISTORY.md`. `testing-week-2026-08-24` is **kept** -
the active GUI-optimization work continues on it.

**The Alert Center quality packet remains unmerged, by decision.**
`claude/alert-center-quality-packet-5btu3w` (8 commits, tip `57fcf47`, 2026-08-18)
builds the alert-delivery measurement surface `GUI_TRADE_DISCOVERY_LEARNING_PLAN.md`
sec 10.3/17 specify but never built: `scripts/alert_quality.py`,
`scripts/alert_delivery_events.py`, a delivery-capture emit in
`scripts/ui/panels/alert_center_panel.py`, a System Health surface, and its tests.
Two things block it and both are recorded rather than guessed at - it **edits alert
code**, so the file-scoped ask-first rule governs the merge itself; and it adds its
own `docs/ALERT_CENTER_QUALITY_PACKET.md` at the same path where `main` already
carries the *different* historical P1.6 packet recovered from `671ee57`, so a
content merge would silently destroy one of the two. Nothing on `main` depends on
it. No alert behavior changed in this consolidation.

### 2026-08-25 - the Questrade credential chain has one owner

**The chain kept snapping because several things spent it.** Questrade rotates
on every refresh - a success invalidates the access token it replaces and
consumes the refresh token it was given - so "Pull today now", the gap backfill
and the nightly slot were three consumers of one single-use chain. The live desk
showed it exactly: a Questrade import OK at 20:54:59, a year-wide backfill at
20:59, `400 Bad Request` on the refresh endpoint at 21:06:51, eleven minutes
after a fresh token was pasted.

`QuestradeImporter.refresh_access_token` now holds the machine-local writer lock
(`local_writer_lock`, the primitive the outcome finalizer uses), **re-reads the
token inside the lock** so a caller that waited spends what the winner left, and
writes the four rotated values in ONE save. `_authorized_get` answers a 401 that
is explained by someone else's rotation by picking up **their** new access token
rather than burning a refresh to rediscover it. A failed refresh still saves
nothing and leaves the stored token alone. New
`project_paths.save_local_settings()` writes several keys in one
read-modify-write via a temp file and `os.replace`; the settings file holds every
machine-local secret and a direct write could truncate it.

**A day whose cause was repaired can be retried again.** The attempt cap counts
failures against a day, not against a cause, so the 140 days that failed while
the chain was dead burned their budget and were skipped forever - a repaired
chain could never clear them. `journal_coverage.self_heal(include_exhausted=True)`
lifts the cap for one deliberate run and is passed only by the Health tab's
"Retry failed Questrade days"; the nightly keeps the cap. `attempts` is never
rewritten, and the run reports `reopened_exhausted`.

**Not built, and recorded as a trader decision:** 44 of the 45
`activities report trades the executions endpoint did not return` days have no
execution within +/-3 days and predate 2026-06-10, which is the executions
endpoint's retention horizon on both accounts. Retrying them can never work.
Importing them from `/activities` (lower fidelity, feeds tax) or labelling them
permanently uncovered is the trader's call and needs a new coverage status.
2026-08-13 is the one such day inside the retention window.

### 2026-08-25 - the scan cycle is timed, and the sweep canary is accepted

**The R10.A restarted-process outcome-sweep canary is MET** by the trader's
explicit acceptance ("52min late is fine", 2026-08-25) of the 2026-08-25 run:
656 pending / 656 finalized, 0 expired, 0 failed, 0 commit failures,
`pending_after 0`. The 52-minute start delay is accepted as KNOWN, not
explained - its cause remains UNKNOWN and the investigation stands. The two
fenced repairs' own canaries and R10 10a are separate and still owed.

**`ScanCycleClock` times each stage of a scan cycle** (trader-authorized
instrumentation in the fenced `bounce_bot_lib/legacy.py`; no scheduling,
detector, scorer or alert change). `run_strategy` marks eleven stages across the
preamble and logs one line per cycle, slowest stage first, before the existing
"Monitoring N" line. Every call in that stretch was silent on the normal path,
which is why the 2026-08-25 investigation could narrow a 92-minute stall no
further than "somewhere in the preamble". Stages past the named few are counted,
never dropped; a backwards clock reports 0.0s rather than a negative stage.
`_maybe_refresh_learning_after_close` now logs when it first finds work due, and
once per worker when it is waiting on an earlier one - and stays silent when
nothing is due. The instrument decides nothing, and a test parses the class to
assert it calls no `sleep`, `wait`, `start` or `Thread`.

### 2026-08-25 - the AWAY Recap draws the day's alerts and charts them

**Correction to the entry below.** "The AWAY Recap is wired to the Alert Center
backing list" was true at the input and overstated as a claim about the page:
`build_recap` has always returned `classified_alerts` and
`AwayRecapPanel._render` never read it, so the alerts reached the page and were
dropped. The only trace of a full AWAY day was the word "alert(s)" in the
summary line, and the page carried no chart at all - `symbolActivated` was
declared with no emitter and no host connection.

`AwayRecapPanel` now draws an **alerts table** - time, symbol, side, tier, a
`D1` flag and the trigger - in the order the day produced them, with no
re-ranking. A D1 row is flagged rather than merged away, because the Alert
Center keeps that feed separate. Activating an alert or swing row emits
`symbolActivated`, which `MainWindow` connects to
`alert_center.show_board_symbol`: **the same snapshot popup** the Strength
Board, RS/RW and Industry boards open (the R4 pattern), so the chart carries the
bot-backed series, the painted levels and the capture rail and no second chart
widget exists anywhere. A blank symbol asks for no chart.

The backing list remains PROCESS-scoped, so the R10 10a live gate needs a
restart before a session and the page read after its close. It is unmet.

### 2026-08-25 - record correction: the restarted-process outcome sweep RAN

The 2026-08-25 after-close outcome sweep was recorded as a FAILED canary in
`CURRENT_CHECKPOINT.md`, `plan.md` and `docs/DESK_TESTING_PLAN.md`. It ran.
`diagnostics/outcome_sweep_coverage.json`: `swept_at
2026-08-25T14:27:36-07:00`, pending_before **656**, finalized **656**, expired
0, failed 0, commit_failed 0, pending_after 0, recovered_from_csv 636; by reason
last_measured_bar 422 / stop_hit 214 / no_measurement_in_checkpoint 20.
Yesterday's 553 finals were untouched. The failure reading was taken at 14:21,
six minutes before the sweep started.

What is real is a **52-minute start delay of UNKNOWN cause**. The due logic is
correct - the sweep became due at close+35 (13:35) - and the top-of-loop check
that finds it due did not run until 14:27:36 because the strategy loop spent
12:55:00 to 14:27:36 inside one cycle whose whole preamble logs nothing. The
investigation is recorded and **no scheduling change was made or authorized**.
Acceptance of the canary is the trader's.

`docs/DESK_TESTING_PLAN.md` sec 2.3 is restored from PASSED to **HALF DONE**:
the AWAY staging half is proved (08-17, 08-18, and the 08-25 queue routing), the
flip-back-to-DESK half - a populated recap and no chart-review backlog on return
- is owed. `docs/analysis/SOL_ATTACK_2026-08-24.md` is a frozen report and is
deliberately NOT edited; the correction lives here.

### 2026-08-25 - two fenced evidence repairs in the outcome path (Decision B)

**Milestone recovery can no longer erase a recorded stop.**
`BounceBot._recover_measurements_from_csv` took the first row at the FURTHEST
milestone outright, so a 12-bar row saying `stop_hit=False` erased a 3-bar row
that had already recorded the stop and the trade finalized `last_measured_bar`
at a positive R. `stop_hit` is now `any()` across the trade's recoverable rows,
and where a stop exists the exit numbers come from the EARLIEST stop-hit row -
R10.0's stop-first decision: the trade was over at that bar. With no stop
anywhere the best-rank row still wins, and an unreadable `bars_elapsed` sorts
last among the stop rows rather than winning by accident. The scan stays
streaming and O(1) per trade. One champion expectation changed with the rule:
`test_a_backlog_stop_out_is_recovered_from_its_own_csv_rows` now takes its exit
from the 3-bar row. **The already-written rows are tagged, never rewritten**
(ground rule 5): new reader-side rule `evidence_rules.milestone_stop_erased_v1`,
conjunctive on `finalization.measurement_source == "legacy_csv_milestones"`,
with `unknown` as a real third answer. Live confirmation: 2026-08-24 finals tag
35 mixed / 172 clean / 907 unknown; 2026-08-25 tags 0 mixed / 737 unknown.

**Signal-bar recovery must match the event's bar.** `_signal_bar_dict` matched
only on close, so a cache shifted by one bar with two adjacent equal closes
returned the 06:30 bar for a 06:35 event. It now takes the event's `bar_time`
from all three call sites and requires `bar.dt == event.bar_time` as well; no
`bar_time`, or a bar with no `dt`, yields the fallback. The alert row, the tier,
the fold, the digest and the queue are untouched, and the golden fixtures are
byte-identical.

### 2026-08-25 - the AWAY Recap is wired to the Alert Center backing list

`MainWindow._select_page` now calls `_feed_away_recap()` when the AWAY Recap
page is selected. `AwayRecapPanel.set_alerts` had no caller anywhere, so a full
AWAY day ended in an empty recap while the backing list, History and every
evidence stream were full (Sol C1). The page is matched by TITLE, not index, so
a reorder cannot silently unwire it. The Alert Center's ordinary and D1 backing
lists are exported as one ordered stream - both are newest-first so both are
reversed, merged on `time_text`, with a D1 row flagged rather than merged away -
and converted to the mappings `away_recap._alert_rows` reads; the tier comes
from the Alert Center's own `extract_alert_tier`, and this page computes none.
`alert_center_panel.py` is untouched. Failure is quiet: a recap that cannot be
filled never costs the page switch. Known limitation: the backing list is
process-scoped and capped, not session-scoped. The R10 10a live gate - one real
AWAY day ending in a populated recap - is still owed.

### 2026-08-25 - Decision A: evidence usability under the sweep's own exit policies

**A sweep-finalized trade now counts under the policy that measured it.** The
after-close sweep finalizes with a blank eod-hold `close_r` by design
(`no_eod_close`), and `setup_scoreboard.usable` keyed on that blank, so all 656
of the 2026-08-25 finals were invisible to every evidence surface while what the
sweep DID measure sat unread in `context.exit`. `exit_policy_r` now derives three
frozen exit policies per final - `eod_hold` (the settled `close_r`), `stop_exit`
(the stored `context.exit.stop_exit_r`, only where `exit.stop_hit`), and
`last_measured` (`(last_measured_close - entry_price) / risk_per_share`, sign
flipped for a short) - attached as `r_eod_hold` / `r_stop_exit` /
`r_last_measured`. A row is usable when at least one policy measured it; the risk
floor and the R10.B claim split still apply; unresolved rows stay unusable and
are counted by reason. The policies are **never blended**: one table per policy
through `evidence_stats`, and the eod-hold ranking tables read `r_eod_hold` so a
row with no EOD close cannot widen their n. Report section 1a prints the
before/after and the policy behind every figure; the bundle gains
`sweep_exit_policies` and four coverage keys; the daily digest carries
`stop_exit_r` / `last_measured_r` beside `close_r`, each with its own n.
Live read of the 2026-08-25 slice: 0 usable -> 255 usable (`stop_exit` n=99 mean
-1.0R, `last_measured` n=255 mean +0.1546R, `eod_hold` n=0), 20 unresolved.
Nothing here promotes or demotes anything.

### 2026-08-25 - Sol adversarial repairs

**Hermetic teardown now fails on a failed stop (`0c62b63`).** The BounceBot
cleanup helper returns `stop()` exceptions to the autouse fixture even when the
worker thread has already joined, so a logged cleanup exception cannot produce a
green suite. A regression first reproduced the former false-green result.

**Blank unresolved outcomes are excluded from usable evidence (`8474383`).**
`setup_scoreboard.unsettled_close_mask` now treats missing and non-finite
`close_r` as unsettled alongside the legacy zero/entry sentinel. Such rows stay
unresolved and are never relabelled `fabricated_zero_v1`; the live 2026-08-24
slice therefore reports zero usable finals instead of 93 entry-like claims.

**Questrade activities use the documented DateTime request shape (`21fd55e`).**
The completeness cross-check now sends inclusive Pacific day bounds as full ISO
DateTimes with UTC offsets instead of bare dates. A failed cross-check still
fails coverage under R7 I2; the repair changes the invalid request, not that
honesty rule.

The frozen attack report also proves four blockers that were not changed: the
AWAY recap is never handed the Alert Center backing list; signal-bar recovery
can choose an earlier duplicate-close bar; milestone recovery can erase an
earlier stop fact; and the outcome sweep becomes due five minutes after its
owning strategy loop pauses. The latter two outcome paths are in fenced
`bounce_bot_lib/legacy.py`; no ask-first authorization was supplied. See
`docs/analysis/SOL_ATTACK_2026-08-24.md`.

### 2026-08-24 - Wave 1 offline slate

**Packet W8 - the retired topology is gone (plan.md P1.5).** The Desk Link
satellite role was retired on 2026-08-08 and its code stayed in the tree for
sixteen days "pending cleanup", which is exactly the state P1.5 exists to end: a
supported runtime carrying a role nobody may use.

Removed in one commit with no behavior change: the `desk_link` package (7
modules), `ui/satellite.py`, `ui/desk_role.py`, both `ui/services/desk_link_*`
modules, `master_avwap_mini_pc.py`, the Settings > Desk Link tab and its role
picker, the control banner, the `--satellite` / `--link-token` /
`--satellite-desk` / `--desk-role` flags, and 70 tests across 7 files.
`desk_report.xml` is now ignored.

**The file-scoped ask-first rule was invoked and answered first.** The removal
reaches eight methods in `alert_center_panel.py`, which houses alert code, and
two of them are decision paths rather than cosmetics: `apply_desk_link_intent`
wrote Focus, and `_alert_has_focus_privilege` is what the feed gate, the beep and
the relay all asked. Nothing was touched until the trader authorized full removal
on 2026-08-24. Partial removal was never an option - deleting the package while
`_relay_alert_popup` still imported it would leave the tree broken, which the
working agreement forbids outright.

**What deliberately SURVIVES**: the generic `read_only` mode on the price-alert
board and panel. It is a widget capability with its own tests, not satellite
plumbing; its only caller was the satellite, so it now has none, and its two
user-facing strings stopped naming a machine that does not exist. Saying this out
loud is the point - a silent half-removal is how the previous "pending cleanup"
lasted sixteen days.

**Packaging triggers fired by design.** `desk_link` was IN the bundle; the spec's
`FIRST_PARTY_PACKAGES` entry is gone, the spec-drift test passes against the
smaller tree, and the exe was rebuilt: `--selftest` returned **70/70 (frozen)**,
exit 0, and Smart App Control did not refuse this hash. That says nothing about
the next build - SAC verdicts are per file hash - and the desk still runs from
source, so this is verification rather than delivery.

Suite: 4868 -> **4798 passed**, exit 0. The drop is the 70 deleted tests and
nothing else.

**Packet W7 - observability depth over what is already on disk (plan.md P1.4).**
Every figure Phase 1's exit gate asks for was already MEASURED: `run_manifest_v1`
records per-phase seconds and the whole `provider.<family>.<event>[.<source>]`
counter tree, and `ai_job_ledger.jsonl` records every overnight job's outcome.
What was missing was a reader that turns them into a TREND, so "the scan feels
slower" becomes a number with an n beside it.

`scripts/diagnostics/observability_trends.py` folds the last N runs against the
N before them: per-phase median and p90 with the change against the baseline,
provider lookups/cache hits/attempts/failures per family and per source, run and
job failure counts with the errors quoted verbatim, and coverage from the scan's
own `symbols_processed`.

**Zero new measurement**, and an AST test keeps it so - the module calls no
clock, sleeps nowhere, and imports no live decision module. It opens files that
were written hours ago.

Three refusals carry the honesty: **n on every figure**, because a median over
two runs is not a trend; **a change needs both halves**, so a phase with no
baseline reports its recent median and NAMES the absence rather than computing a
change against nothing; and **absent is not zero** - a run that never recorded a
phase is counted in `runs_missing_phase` rather than contributing a zero, and a
family with no attempts has `failure_rate: null` with the reason beside it,
because zero failures out of zero attempts is not a 0% rate.

Frozen by the golden fixture `observability_trends_v1` under the Milestone 3
contract. Its inputs are hand-written to contain each shape the reader has a
rule for - a phase new to the recent window, a phase missing from one run, a
family that never dialled out, a failed run carrying an error, a job with a
mixed record - rather than a copy of one machine's diagnostics, which would
drift the moment the desk ran again.

**Its first live read found two real failures nobody was counting**:
`journal_import` failed 9 of 12 recorded runs (the dead Questrade refresh chain,
already a known trader action) and `ticker_briefs` 11 of 30. Neither is new
behaviour; what is new is that a single command says so.

**Packet W6 - LOCAL-AI Phase 3 and Phase 4 machinery, runs gated.** Both were
authorized ahead of their phase gates on the R10.I pattern, and they are gated
in DIFFERENT ways because their gates are different things.

**Phase 3, journal enrichment** (`ai_jobs/enrichment.py`, slot
`journal_enrichment`). Below Phase 2's ten-clean-digest-session counter it calls
no model and writes nothing - and it reads that counter from
`digest.digest_gate_state` rather than restating it, so the repo has one
definition of "ten clean digest sessions". Enriching a journal from a layer
whose own facts have never been audited is what the phase order exists to
prevent. Advisory fields only, and structurally so: it writes one new table,
`ai_trade_enrichment`, through the `JournalStore` API, and never opens the
trader's `trade_annotations` row (I7). The table is append-only - a re-run adds
a row rather than rewriting what an earlier night believed. Tags come from
`SETUPS_MAJOR.md`/`SETUPS_TEST.md`; anything outside that vocabulary is DROPPED
and counted, because an invented family name is a bucket nobody can compare
against anything.

**Phase 4, the policy draft** (`ai_jobs/policy_draft.py`, slot
`review_policy_draft`). This one RUNS while its gate is unmet, and that is not
an inconsistency: **the gate IS two weeks of drafts**, so a writer that refused
until the window passed would make the window unreachable. It writes
`review_policy_draft.json`, archives one copy per session so the comparison has
something to compare, and carries the NOT-MET statement in the draft's own
`notes` until the window closes - which is still only half the gate, since the
trader's quality sign-off is the other half and no counter answers it.
`review_policy.json` is never written, and no code path in the module resolves
it. Deltas come from the existing mechanical translator, clamped; the model may
only write the sentence a chart shows. Ranks and annotates only, no suppression
field, FIFO untouched.

**The boundary is walked, not asserted.** AST tests check that neither module
names the live policy file as a path token, that neither writes a trader-owned
journal field or calls the trader-facing saves, and that neither imports a live
decision module. Path scans deliberately ignore prose, so both modules stay free
to NAME the files they are forbidden to write in the sentences forbidding it.

**Packet W5 - the weekly synthesis: machinery built, runs gated.** LOCAL-AI
§7.3 listed this under "What is NOT built" with the cadence and the gate already
decided and only the authorization missing. The trader gave it for the MACHINERY
on 2026-08-24, on the R10.I pattern.

**The gate counts sessions in which a graded cohort row MATURED**, pooled across
the veto and LIKE cohorts. Sessions rather than rows, because two weeks means
two weeks of forward evidence and counting rows would let one busy afternoon of
vetoes clear a gate that exists to wait for the market to answer them. Pooled,
because the two cohorts are the halves of one judgement.

**Below the gate no model is called at all** - the deterministic rollup is
written with `SYNTHESIS GATE NOT MET.` as the first line of both artifacts and
everything labelled `discovery`. §7.2's reason for keeping `trader_judgement`
off the nightly slate applies harder here: a read over a stream still filling
narrates "too early" until a reader stops looking. Above the gate it is STILL
`discovery`, because the window was not declared in advance.

Every cell is one `evidence_stats` summary per (cohort x side x horizon), capped
at 40 with what the cap dropped printed. The Phase 2 digest rollup folds in once
a fact pack exists; before that it says so in words rather than rendering a zero.

**Not nightly**, and structurally so: it lives in a new `optional_slots()` that
`default_slots()` never reaches, constructed per call exactly as `--scopes` is,
and is invoked by `run_ai_jobs.py --weekly-synthesis`. **Not frontier** - medium
tier or nothing; Phase 5's frontier pass stays unauthorized. **Not a control
signal** - an AST test pins that no decision-surface path or live-decision module
is reachable from it, scanning only strings that could BE a path so the module
stays free to name `review_policy.json` in the sentence forbidding itself from
touching it.

R8's live gate - one weekend where the trader confirms the ranked reasons are
the ones they recognise - is unchanged and still owed.

**Packet W4 - the Daily Digest Ledger (LOCAL-AI Phase 2).** The 2026-08-08
trader decision forbade building or freezing any digest schema until six open
questions were answered. They were answered on 2026-08-24, so the decision is
MET rather than waived, and §6.4a's design is now `scripts/ai_jobs/digest.py`
with the answers frozen into `ANSWERS` and carried inside every pack - a reader
six months from now gets the rules with the record.

**Two artifacts per session, and that split is the design.** `facts/<YYYY>/<date>.json`
is written by code with zero LLM involvement and is written even when the model
is down; `narration/<YYYY>/<date>.json` is medium tier, reads the fact pack and
NOTHING else, and is simply absent when it fails (the slot returns
`degraded_no_narrative`, which the runner does not count as coverage, so the
next firing retries the narration). Because the narrator sees one bounded
document, the 2026-08-10 truncation failure - a model handed a sheared prompt
producing confident schema-valid output about evidence it never saw - cannot
recur here by CONSTRUCTION rather than by vigilance.

The six answers as built: **both win metrics side by side** (close-R is result,
MFE/MAE is opportunity, and no field combines them); **slices are env_key
(environment x day-part) x side**, no setup-family slice in v1; **shadow-engine
output excluded**, champion facts only, pinned by an AST test; narration
disposable and facts permanent; **16 KB hard cap where over-cap FAILS the job
and writes nothing**; and a non-session writes an EMPTY pack so the gap is
visible rather than looking like a missing file.

Every measured value carries `{value, n, source_id, selector, as_of}` and `n` is
mandatory - `measured()` raises without it, because a default would make the
omission invisible. Champion outcomes are read through
`setup_scoreboard.load_intraday_finals`, so the digest and the scoreboard cannot
drift into two definitions of "usable", and rows whose family does not CLAIM an
entry are never averaged as trades. Packs are append-only: a second run writes a
superseding SIBLING naming what it supersedes, and never edits. Rollups are a
read computed on demand (D8), weighting sessions by n rather than averaging
day-means.

**Three build decisions §6.4a left open** are recorded in that section rather
than taken silently: `MAX_SLICES = 16` so the pack fits its cap by construction
(worst measured shape 12.9 KB); one pointer per slice ROW with `value`/`n` per
metric cell; and `env_key` READ from the row's `context_json` rather than
re-derived. That last one caught a real boundary: computing the day-part meant
importing `bounce_bot_lib.learning`, the module that MUTES alert segments, and
an existing test bars every `ai_jobs` module from reaching into live decision
code. Reading the stamp the alert path already writes is both the boundary-safe
answer and the one that keeps a single definition of "midday".

**The exit gate is owed and building never marks it met**: ten consecutive
session days of digests, with the trader spot-auditing at least three packs
against raw evidence and finding no fabricated fact.
`digest.clean_digest_sessions` counts sessions - and counting is not passing. An
empty non-session pack deliberately does not count towards it.

**Packet W3 - true USD conversion, booked rather than estimated.** R7 deferred
this on 2026-08-18 for a good reason: the FX table booked CAD only, and inventing
a rate is exactly the dishonesty the currency refusal was built to prevent. The
trader reversed the deferral on 2026-08-24; nothing is invented, because the
table no longer books CAD only.

**The gap was upstream of the render seam.** `rates_needed_for_trades` asked only
for each trade's OWN currency, so a CAD-only session never had a USD observation
booked and could never be shown in USD however honest the display was. It now
asks for a USD observation on every session that has trades.

`book_cad_values` becomes `book_currency_values` and books both: `net_pnl_cad`
exactly as before, and `net_pnl_usd` as the booked CAD value divided by that
trade's OWN session rate - a trade taken on a 1.28 day is not valued at what a
1.42 day says. A USD-native trade books its own number with no rate at all.
`fx_usd_rate` and `fx_usd_rate_date` travel with it, and the date is the
EFFECTIVE observation after any weekend carry-back, so the figure is auditable.
`convert_amount` reads the column; it never divides at render.

Every I5 rule carries over: booked once at import; a missing observation renders
"unconverted", never 0 and never the native number relabelled; a rate that later
disappears CLEARS the booking rather than leaving a number nothing on disk
supports. `resolve_pnl_key` prefers the booked key only when EVERY closed row in
the selection carries one - a total mixing booked rows with estimated ones is
neither - and the manual USD/CAD field remains the labelled fallback for an
unbooked session. The Analytics `None`-bucket exclusion is untouched, CAD stays
the tax-grade value, and the I6 blended badge is untouched.

Three additive columns on `trades`, appended to `NEW_COLUMNS_V3`, which
`migrate_to_v3` applies idempotently on every open - so they reach an already-v3
database with no version bump and none of the trader-present preparation a real
migration requires. The `journal_rebuild_trades_v1` golden was re-frozen with an
`intentional_difference` naming exactly what moved: three new columns and
nothing else, with legs, opportunity events and the summary byte-identical.

R7 gates 1/3/6 are unchanged and still owed - building a conversion does not
validate it against a broker statement.

**Packet W2 - R8's DEFERRED block is empty.** Three streams the Weekend Prep
spec named as future scope and honestly did not claim now render, all under the
reader idiom the AI-P1 repair established: address a home-folder store by its
named `project_paths` constant, never by composing a filename.

* **Focus Pick Review - the picks' own forward record.** `human_focus_performance.csv`
  per cohort, side and horizon: n, win rate, mean and median side-adjusted
  return, profit factor, the symbol and session counts, and the session-block
  interval. The two cohort tables beside it are the judgement mirrors - what was
  thrown away, what was endorsed - and this is the thing being judged. It renders
  the WHOLE rollup rather than a week slice, and says so: the file carries no
  trade date, only the `updated_at` stamp of its last rebuild, so the spec's
  "filtered to the week" would have filtered on when the nightly pass RAN and
  emptied the table on any week it did not. The deviation is recorded in the
  spec rather than silently taken. Blank stays blank; an absent interval carries
  its reason.
* **Focus Pick Review - the week's verdicts.** `pick_feedback.jsonl` through
  `pick_feedback.load_pick_feedback` (one loader per file, never a second JSONL
  parser on a panel), scoped on `trade_date` - the session the verdict is ABOUT,
  never `ts`, which is when it was typed. The pane tallies the verdicts, says
  they are opinions rather than outcomes, and prints what its cap dropped.
* **Week in Review - sector and industry extremes.** `rrs_group_strength_extremes.csv`
  folded per `(group_type, group_key)`. The group log records **no bucket** - its
  writer emits the top and the bottom of each list with identical columns, unlike
  the symbol log - so the fold keeps BOTH extremes and the sign is what the
  reader reads. Nothing invents a direction the file never recorded. The cap
  prints what it dropped, like the symbol stream.

**Also removed: `_read_focus_week`**, dead since 2026-08-18 and still resolving
its CSVs under the home root. AI-P1 fixed the live reader and left the copy that
encoded the defect where a future edit could reach for it.

§10's one-real-weekend live gate is unchanged and still owed for all three -
building a view never validates it.

**Packet W1 - the test suite's teardown is bounded (plan.md P1.1).** The
hermetic-suite packet of 2026-08-18 closed the network half: an offline tripwire
that refuses and RECORDS any socket an unmarked test opens, adapter-boundary stubs
for IB, yfinance and the market-prep feeds, and `network`/`broker` as the only
opt-outs. The teardown half was still open, and `conftest.py`'s
garbage-collection block said so in its own honesty paragraph - "several tests
leave threads running past their own teardown ... and nothing in this block joins
them".

Measured before it was fixed, with a thread-recording plugin over a full run:
**22 tests left at least one thread alive past their own teardown, and 19
`run_strategy` threads were still alive when the session ENDED.** A BounceBot
strategy loop is not idle - it re-reads the watchlists, refreshes learning state
and reloads Focus picks - so every later test was sharing mutable state with a
scanner nobody was watching.

No new machinery was needed, only a call: BounceBot already implements
cooperative shutdown. `conftest.retire_leaked_bounce_bots` finds the owning bot
through the thread's own `_target.__self__` (a name is a label anyone can set;
the target IS the object that can be stopped), calls `stop(timeout=...)` once per
BOT rather than once per thread, and FAILS the leaking test by name if one
survives - the same rule the chart-worker drain already follows, because a
teardown that swallows its timeout cannot prove the quiescence it exists to
provide. Deliberately narrow: the other leaks the measurement found
(`scan-*-drain`, `qt-health-audit`, `industry-board-refresh`, the Desk Link
reader) finish on their own and were gone before the session ended, so a blanket
"no thread outlives its test" rule would have failed 22 tests to fix 19 threads.

Re-measured after the fix: **0 scanner threads survive the session**, 22 leaking
tests down to 7, and three consecutive full runs agree on their counts. The
wall-clock flake class named in the R10.V review - two panel tests that failed
only between 06:30 and 07:00 PT, inside the open-burst digest window - was
repaired on 2026-08-23 by pinning those tests' clocks rather than disabling the
digest, and is verified still pinned.

### 2026-08-24 - The AWAY recap, the like-cohort view, two opt-in scopes, and R10.I

**Packet 8 - an AWAY day ends in a recap, not a queue** (R1 trader amendment
2026-08-24; decision record §5). The trader returned from a full AWAY day to
**317 alerts waiting in the chart review queue**, plus 128 hidden inside
yesterday's range. In AWAY the queue no longer accumulates: `_enqueue_review_alert`
diverts, and that line is the ONE door into the queue - the auto-pick drain, the
D1 feed and the ordinary feed all arrive there. **What does not change is the
harder half**: `self._alerts`, the D1 feed and badge, History and every evidence
stream are written BEFORE that call and are asserted byte-identical between an
AWAY panel and a DESK one. DESK is untouched; EVENING keeps its queue (it is for
sleeping through the morning, and that queue is what the trader wakes up to);
the AWAY hourly phone pushes are untouched, per the resolved sub-decision.
Diverted alerts are **counted**, session-scoped, because "nothing accumulated"
and "nothing happened" must not look the same on the return.

The recap surface **writes nothing** - a test parses its AST and asserts it
makes no IO call and imports no store - and **ranks nothing**: the numbered
swings are the AWAY push's own ranking, the alerts the desk's own
classification, the staged picks what the machine already staged. Each section
states where its ORDER came from. A source that could not be read is NAMED
(a page empty from a failed open must not read as a quiet day); a digest line
that cannot be parsed is kept and marked. Every mutation is delegated:
`FocusService` adds, the Alert Center removes, `MarketJournalService` takes the
D1 write-up. The R2 adoption gate is SHOWN at click time and never blocks the
trader - it governs the machine's adoptions, not theirs.

**Packet 8b - the like cohort beside the veto cohort.** R10.F graded 45 LIKE
claims into 28 cohorts and nothing read the file. Focus Pick Review now shows
both: the two are the halves of one judgement, and keeping only the vetoes
leaves you the flattering half. Read by NAMED CONSTANT (the like trio moved into
`project_paths` beside the veto trio), because AI-P1 found this exact step
rendering an empty table for six days from a composed path. No canonical pooling
- that exists because the veto VOCABULARY is versioned, and a like's cohort is
its claimed setup id, which is not.

**Packet 9 - two opt-in evidence scopes** (decision record §3). `walkaway` and
`setup_performance`, registered but not nightly. `setup_performance` reads the
scoreboard's **output** and never the raw tracker: TB-0/TB-5 measured the
tracker's text projection contributing zero symbol-specific content while
starving every analysis it led, and a test asserts no source path names it.
Caveats are **derived from live source**, not retyped (the AI-P5 lesson) -
`setup_performance`'s reads the bundle's own claim-kind coverage, and a bundle
it cannot read yields "UNKNOWN for this package" rather than a remembered
enumeration.

**Packet 10 - R10.I, and it says it is scaffolding.** Built under the recorded
sequencing override; the claims gate was **not** waived and is not waived here.
Every report states its n, labels everything `discovery`, and prints
**COLLECTION WINDOW NOT MET ... Do not promote, demote, or change anything on
the strength of this file** as the first thing rendered. Pinned by test over an
empty ledger. A caller that cannot count sessions reads as unmet - the only
direction that cannot turn scaffolding into a finding by accident. The
`evidence_report` slot is appended last (it reads what the cohorts produced),
deterministic, no model. First live run: **n=13394 ledger rows over 1 of 10
sessions**, 44 cohort rows, 158 scoreboard rows, window not met. Plus the opt-in
`market_journal` scope - free-text entries reach an AI scope opt-in only, a
recorded trader decision, unchanged.

- **Frozen exe rebuilt**: packet 8 fired the trigger (new page, two modules);
  `--selftest` returns **70/70 (frozen)**, exit 0. SAC did not refuse this hash,
  which says nothing about the next one.
- **Canaries owed**: the AWAY routing needs one live AWAY day, and R10.B/E/G/H's
  remain open. Building never marks a live gate met.

### 2026-08-24 - R10.E/F/G/H: provenance, the other cohort, the tape, and the words

**R10.E - Focus membership becomes an episode.** Audit F5: 244 of 499
(symbol, side) pairs - **49%** - appear on two or more sessions, DOCN SHORT on
seven, and the snapshot store cannot say whether that is a name surviving the
day roll or the trader re-adding it. `focus_membership_events.jsonl`
(`focus_membership_event_v1`) records episodes, so a re-add after a departure
is a NEW `membership_episode_id`. Three refusals carry it: the pick key
includes the **category** (F3 - a name on both the swing and M5 list silently
lost a row, and the CSV's zero multi-source keys is the signature of that
collision, not evidence against it); no marker in a store with **no** markers
is `unknown_legacy`, never `trader` (F4 measured `focus_auto_picks.json` exists
for no historical date); and a missed snapshot is an `observation_gap` row,
because membership is never reconstructed from current state.
`expire_m5_if_new_day` emits **one row per name**, so a survivor is visible
rather than hidden inside a "cleared N". Emitted from the ONE Focus writer and
never at the pick's expense.

**R10.F - the likes get graded too.** Audit C1: 52 `like_claim` rows over two
sessions and **no** `like_cohort_*` file. The trader's rejections had a forward
record; their endorsements had none. `like_cohort` mirrors the veto trio
deliberately - same `_pick_key`, same first-of-day rule, same sideless refusal,
same delegate through path parameters - so a difference between the two cohorts
comes from the data and never from two implementations that drifted. The cohort
source is the **claimed setup id** rather than a reason code, and stamps carry
UTC plus `session_date` (ground rule 7). `like_cohort_grading` is **appended**
after `veto_cohort_grading`. **First live run: 45 claims merged and graded, 28
cohorts, 0 skipped for no side** - and because the rollup routes through R10.C's
`evidence_stats`, it arrived with the robust half already attached.

**R10.G - the machine's half of the day's record.** Audit C2:
`market_environment_annotations.jsonl` **did not exist**, so the regime the desk
operated under was unrecorded and unrecoverable. Every shift is now a row from
the ONE setter, keeping **both** the auto read and a trader override, because
the difference between them is the agreement rate and an agreement rate needs
the disagreements. `daily_market_context_v1` carries one row per session at
close+grace, completed at next launch if missed and flagged `completed_late` -
never fabricated, so a session nobody measured leaves a gap.
`config/market_calendar.json` overlays the computed rules for what a rules
engine cannot know; a year it does not cover reports **DEGRADED** on System
Health rather than silently falling through.

**R10.H - somewhere to write what you thought.** `market_journal.jsonl`
(`market_journal_entry_v1`) behind one service, two surfaces: a **Journal tab**
on the Trading Desk after Capture (M5 default, Ctrl+Enter commits) and a
left-nav **Market Journal** page (entries, the environment timeline with its
agreement rate, the calendar strip, R10.G's day-context row). The existing
"Journal" page stays the trade/tax journal - the near-identical labels are
deliberate: one records what you TRADED, the other what you THOUGHT.
After-the-fact entries are first class and **never backdated**: an entry written
Saturday about Friday carries both stamps, and `written_after_the_session` is
COMPUTED rather than claimed, so a caller cannot set it wrongly. Corrections
supersede and the original stays on disk. The agreement rate refuses to
flatter - zero comparable sessions is UNMEASURED, not 100%. A failed write is
reported as failed and never as saved.

- **Frozen exe rebuilt and verified**: the trigger fired (new page, seven new
  lazily-imported modules), `selftest.LAZY_ENGINE_MODULES` gained them all
  disjoint from `PACKAGES_NOT_IN_THE_BUNDLE`, and
  `dist\TradingBotV3\TradingBotV3.exe --selftest` returned **68/68 (frozen)**,
  exit 0. Smart App Control did not refuse this hash - which per its per-file
  nature says nothing about the next build.
- **Mechanics canaries owed** for R10.E, R10.G and R10.H (and R10.B's, still
  open). Building a packet never marks its live gate met.

### 2026-08-24 - R10.D: the tracker's transitions become an authority

- **The setup tracker is a 951 MB snapshot with no memory**, and audit S1
  measured what that costs: between one frozen pair, 218 setups changed status,
  2,737 CLOSED scenarios changed status or reason, 1,306 changed exit date, and
  AMCR LONG on 2026-07-28 went `TIME_STOP @ 46.69, R 0.577` to
  `TARGET_HIT @ 45.55, R 0.360` **on the same date**. A snapshot can answer
  only "what is true now". `setup_tracker_events.jsonl`
  (`setup_tracker_event_v1`, month-segmented) is the append-only authority
  beside it.
- **Four event types**, and the last two carry the distinctions that matter.
  `reopened` is named separately from `transition` because S1 measured 35
  CLOSED→OPEN and 1 UNTRADEABLE→OPEN in a single pair. `tombstone` says a setup
  **left the payload and says nothing about why** — it can leave because it
  closed, because the tracker pruned it, or because a partial read lost it, and
  a row that implied the flattering one would be worse than no row.
- **Never by deep-copying the payload.** A digest sidecar holds one 16-char
  hash per setup, so the diff is a dict comparison over ~10k short strings and
  the payload is read once, in place. A test asserts the setup dicts handed in
  are the same objects the caller holds.
- **The digest covers only state-bearing fields.** Most of the payload's
  hundreds of per-setup fields move every run — a price, a band, a note — so
  digesting whole records would emit a transition for every setup on every run
  and the stream would say nothing.
- **The sidecar is written LAST**, only after the ledger append succeeded, so a
  crash between the two costs a repeat of the run's diff rather than a hole in
  the stream. Re-emitting a transition is recoverable; dropping one is not. A
  sidecar written over a different field list re-seeds as `initial` rather than
  emitting a wave of false transitions.
- **One `run_summary` row per save**, because an event stream alone cannot
  distinguish "nothing changed" from "the run did not happen" — the first thing
  a reader needs when a day looks empty.
- **S2 is measured on every run and never repaired.** A tracker run during a
  session marks the FORMING bar, so a setup carries a close that does not exist
  yet. Rewriting a mark would be rewriting history (ground rule 5), so the
  count, the offending date and a sample ride on the run summary.
  **Reproduction note: S2 does NOT reproduce on the current payload.** The
  audit measured 2,739 offending setups on a `data_session` 2026-08-20 payload;
  the 2026-08-24 payload (`data_session` 2026-08-21, written Monday over a
  completed Friday) has **14,043 marks and zero later than its vintage**. The
  defect is intermittent by nature — it needs a run during a live session — not
  refuted.
- **S3a reproduced almost exactly, and its root cause is now named in code.**
  `future_idx = idx + horizon` indexes the symbol's OWN scan rows, not exchange
  sessions, so a name that appears on a watchlist irregularly has "5 sessions
  later" land far away. Live medians over 10,928 rows: horizon 1 → 1 session,
  3 → 5, **5 → 64**, **10 → 73**, with **42%** of rows spanning more than twice
  their declared horizon. New `sessions_spanned` and `stale_horizon` columns
  MEASURE and FLAG it. They deliberately do **not** re-select the future row:
  that would silently redefine every number the tracker has produced, which is
  a scoring change and not this packet's to make.
- **S3b reproduced exactly and is fixed.** `spy_forward_return_pct` and
  `spy_relative_side_return_pct` were non-null on **0 of 10,928** rows, because
  SPY only reaches the frame when it is itself a scanned symbol — and it never
  is. The benchmark now falls back to the durable daily-bar store's SPY.
  **Cached bars only: zero IB traffic** (ground rule 8). A machine whose store
  has no SPY leaves the columns null, which is what they already were.
- Golden fixtures byte-identical before and after (ground rule 1).

### 2026-08-24 - R10.C: one statistics discipline, and the numbers it moves

- **`scripts/evidence_stats.py` implements ground rule 10 exactly once**, and
  every ground-rule-11 surface routes through it. Until now the cohort rollup
  published a bare mean, a win rate and a profit factor; the scoreboard
  published quantiles but no concentration and no interval; nothing published
  an interval at all. A reader comparing two of them was comparing two
  disciplines.
- **What every summary now carries**: event / symbol / session counts;
  excluded and unresolved by reason beside n; raw mean, median, trimmed mean,
  p10/p90; uncapped and 4R-clipped side by side; profit factor with its
  convention; stop rate; concentration by symbol and session; a session-block
  bootstrap interval; and a `discovery` / `confirmation` label.
- **The refusals are the point.** A cohort with no losers reports **no** profit
  factor, not a large finite number - a PF with a zero denominator is a claim
  about a division nobody performed. A sample spanning one session gets **no**
  interval, because an interval over one block describes one day as though it
  were a range. A policy or a cell missing its input reports unmeasured, never
  zero.
- **The interval resamples whole SESSIONS**, not individual trades: trades
  inside one session share the tape, and resampling them individually would
  report a precision the data does not have. It seeds from the data itself, so
  two runs over identical inputs agree - a report that changes between runs
  cannot be checked by anyone.
- **`confirmation` can never be inferred from n.** A large post-hoc sample is a
  large discovery. Only a caller naming a window declared in advance gets the
  confirmation label. **n >= 30 is necessary, never sufficient**, and the old
  `reportable` column - which meant exactly the n floor under a name that
  claimed more than it measured - is now `meets_n_floor`.
- **The cohort performance CSVs gain ground rule 10's robust half**, appended
  so every existing reader keeps working: median, trimmed mean, p10/p90,
  symbol and session counts, top-symbol share, the interval and its basis, the
  evidence label. A cell that cannot carry an interval **prints why** rather
  than a blank a reader would take for an oversight.
- **The setup scoreboard applies R10.B's claim-kind split, and shows what it
  moved.** Measured on the 07-24..08-21 window: of 5,970 settled,
  above-the-floor rows, **4,442 were annotations and 526 were observations -
  only 1,002 were entry claims.** 83% of what earlier reports ranked was not a
  trade. New section 1b prints every affected family with its **before mean,
  its after mean, the rows removed and the claim kind that removed them**,
  because an unannounced move reads as a regression and an announced one reads
  as the fix working. `h1_ema10_bounce` (2,887 rows, -0.092R),
  `h1_blue_after_red` (1,197, -0.126R) and `h1_green_to_yellow` (354, -0.004R)
  leave entirely; so do `regime_pause_rs` (304) and `regime_pause_rw` (208).
  A family with no "after" prints a blank and the report says what that blank
  means - a family that never claimed an edge, not one whose edge vanished.
- **R9.3's frozen 40-session window is reprinted unchanged and the report says
  in words that it did not measure it.** The sessions it names have not
  elapsed, and a number taken from a window before it closes is not the
  evidence the window exists to produce.
- **A machine-readable bundle beside the Markdown**
  (`setup_scoreboard_bundle_v1`), from the SAME computation so the two cannot
  disagree; the **runtime report store** at `output/reports/evidence_reports/`
  with atomic last-good (a failed publish costs the new report, never the
  previous one); `--freeze` for a dated, hand-committed audit into
  `docs/analysis/`; and `--ledger` to count the R10.A evidence ledger
  **beside** the CSV rather than merging it - the CSV stays the authority
  during the canary, and a report that silently preferred one source would make
  the canary unreadable. First live run read 12,707 ledger rows.
- **The four frozen exit policies are reported per family, side by side and
  never blended.** Rows without a captured path are COUNTED (`paths_missing`)
  rather than quietly excluded: averaging only the trades that happen to carry
  a path is a different statistic wearing the same name. Every family currently
  shows `paths_missing` equal to its whole n, because path capture began with
  R10.B and no historical row has one.
- Golden fixtures byte-identical before and after (ground rule 1).

### 2026-08-24 - R10.B: the outcome store learns what its rows CLAIM

- **Every registered row was measured as a trade, and most of them are not
  trades.** `scripts/outcome_semantics.py` gives each family a declared
  `claim_kind` - `entry_claim`, `annotation`, `information`, `unconfigured` -
  and only an entry claim may carry an R, an exit policy or a path. Measured
  over the live store: **entry_claim 68,237, annotation 147,713, information
  35,407**. Nearly 60% of the store is H1 colour marks on bars that had already
  closed, and they were being averaged as trades. `regime_pause_rw`'s all-time
  mean of -1.82R across n=934 is the cost of that category error, not an edge.
- **`unconfigured` is never silently a trade.** It is the honest default for a
  family nobody declared, is counted loudly and NAMED in a new System Health
  row (`Outcome claim semantics`), and is excluded from every trade statistic.
  DEGRADED rather than unhealthy: nothing is broken, a statistic is simply not
  entitled to those rows yet.
- **The registry was corrected by reading the store, not the audit's prose.**
  The first draft invented two H1 family names (`h1_red_after_blue`,
  `h1_reversal`) and missed `h1_ema10_bounce` - the single largest family in
  the store at 92,477 rows - and `h1_green_to_yellow`. Enumerating the 27
  distinct level names is what found it.
- **Compound families are decided by their parts.** `_make_bounce_event_id`
  builds the family as the sorted level names joined by `-`, so splitting on it
  recovers the exact parts; that is construction, not similarity. Without it
  **158,053 live rows read as unconfigured**. A compound whose parts disagree
  about what they claim, or that contains one undeclared level, stays
  `unconfigured` - a row whose pieces disagree has not been classified.
  Matching is by whole name and never by prefix, because the store contains the
  trap: `h1_ema_15` is a bounce LEVEL (entry claim) while `h1_ema10_bounce` is
  a colour ANNOTATION.
- **LRSI produced zero outcome rows, and now produces gradeable ones** (audit
  D5a). `_emit_lrsi_cross_alert` built one synthetic bar with
  `open=high=low=close` and passed it everywhere; a long's stop comes from that
  bar's LOW, so stop == entry, risk == 0, and `_register_bounce_outcome`
  returned at its guard. The engine had been firing alerts nobody could ever
  grade. The REAL signal bar is now recovered by index from the same cached
  series the event was measured against - and a series that has moved yields
  the fallback rather than a mis-indexed bar, because a wrong bar produces a
  plausible stop from the wrong price. Applied to the confluence engine and the
  first-candle ORB flow too; D5b stays UNTESTED, because that flow has never
  fired and there is no row anywhere to check it against.
- **The alert row and the tier still see the flat bar**, deliberately. They
  feed `_evaluate_bounce_alert_quality`, so widening them would move alert
  tiers - a scoring change, which plan.md sec 5 forbids without golden fixtures
  first. Only the outcome registration, which is evidence, gets the real bar.
- **H1 entry stamps are the bar CLOSE going forward** (audit D6a: 6,439 of
  6,439 rows stamped on the bar START). An entry stamped an hour before the
  signal existed makes every entry-timing statistic over 82% of the store
  measure the wrong instant. Existing rows are NOT rewritten (ground rule 5).
- **`evidence_rules.h1_bar_start_v2`**, because the fix was invisible to v1: an
  H1 bar in PT starts at :30 and therefore also CLOSES at :30, so v1's
  family-AND-minute heuristic would report a false positive on every forward
  row. Rules are never edited in place, so the answer is a new NAME. Forward
  rows record an explicit `entry_time_basis`; a row with none falls back to v1.
- **Path capture** (`scripts/outcome_path.py`): MFE/MAE at 1/3/6/12/24/36/EOD,
  first-touch stamps, giveback, and a compact per-bar excursion in R, so a
  future exit model simulates offline with no refetch. Where one bar contains
  both the target and the stop, the **STOP is taken first** and the row says so
  (`stop_first_intrabar`) - OHLC carries no intrabar sequence, and assuming the
  favourable order manufactures profit out of an unknown, in one direction
  only. Attached to FINAL rows of entry claims only; anything else records an
  explicit `path_absent` with its reason, so "not a trade" and "something
  failed" stay distinguishable.
- **The four frozen exit policies each report on their own** - `eod_hold`,
  `trail_2bar_after_1r`, `vwap_close_after_1r`, `atr_1p5_trail` - and a policy
  missing its input reports **unmeasured, never zero**: one that silently
  degrades into a different policy publishes a number under the wrong name.
  `oracle_best_ex_post_r` is the best of them chosen with hindsight, labelled
  an upper bound in the payload itself, attributable to no policy (ground rule
  12). "Realizable R" appears in no emitted field.
- Fixture `outcome_path_eat_cake_v1` - 78 real M5 bars per symbol for
  2026-08-21, fetched once from yfinance (**zero IB traffic**) and frozen so
  every test runs offline. The tests assert the honest calculation and never a
  desired sign. It deliberately carries no VWAP column, which is what exercises
  `vwap_close_after_1r` reporting itself unmeasured.
- **Golden fixtures byte-identical before and after** (ground rule 1), verified
  by SHA-256 on all five. **R10.B touches a live writer, so its mechanics
  canary is owed** - one live session confirming LRSI now registers gradeable
  rows and H1 stamps land on the close.

### 2026-08-24 - AI-P2: the auto-tag backlog becomes drainable (trader-approved amendment)

- **R8 §6's locked decision was "journal hook is the weekly auto-tag review
  only". The trader approved widening it on 2026-08-24**, so the weekend
  auto-tag sub-pane gains a **default-off** "Show all pending proposals" toggle
  backed by `journal_feed.pending_tag_candidates()`. The weekly scope remains
  the default and is unchanged.
- **Why it was needed:** the store held 220 auto-tag candidate rows against
  **one** confirmed annotation, so the confirmation stream could only fill at
  the weekly trickle - and every analysis that reads the trader's own tags
  (per-setup performance, the AI layer's `journal_review` scope) waits behind
  it. **Correcting a number used earlier in this program's notes:** those 220
  rows span **48 closed trades**, not 220 review items - several proposals per
  trade - so the backlog is one sitting's work, not a month's.
- **The amendment widens what is LISTED and nothing else.** Same row shape,
  same inclusion rule, and the confirm→`accept_auto_tags` /
  correct→`correct_auto_tag` paths are taken unchanged; a characterization test
  pins that, because a toggle that changed how a confirmation is written would
  have quietly forked the trader's own annotation stream in two.
- **Newest first**, because a backlog is reviewed by memory: the trader can
  still say what they were thinking in March and cannot for 2023.
- **`already_tagged` is reported, not filtered**, and the cap (60) prints what
  it dropped. Accepting a suggestion does not delete its candidate row, so a
  confirmed trade keeps proposing; hiding those would also hide a trade that
  deserves a second tag, so the pane counts them and says so instead - which is
  what lets a burn-down visibly shrink.
- Toggling the backlog **never replays the walk-away**: the tag list moved into
  its own `_reload_tags()`, because walk-away is a market-history run behind a
  worker thread and this is a database read.

### 2026-08-24 - AI-P1: the mirror cohort, and the join that was reading nothing

- **Focus Pick Review now shows the graded veto cohort**, which its subtitle
  has promised since the step shipped while nothing loaded any `veto_cohort_*`
  file. R8 §6's last DEFERRED join. It is the MIRROR of the picks table and
  that is why it belongs there: the picks answer "how did what I took do", and
  only the cohort answers "how did what I threw away do".
- **The cohort is not week-scoped and loads first**, so a quiet week cannot
  also hide the whole graded record of the trader's vetoes.
- **Pooled through the one canonical function.** The rollup on disk is already
  grouped by `canonical_veto_cohort`, so calling it here is idempotent - but it
  is CALLED, never reimplemented, so a later vocabulary bump cannot leave the
  pane and the rollup disagreeing about which rows are the same reason.
- **Honesty carried over verbatim from the sibling join**: an unmeasured
  statistic renders BLANK, never 0.00%; a missing file is an explicit absent
  state ("this is an absent measurement, not a clean record") that leaves the
  rest of the page working; every row shows its n. Nothing derives a number the
  CSV does not carry (Phase 0.7 ground rule 6) - the caption states the horizon
  and the sign convention, and labels the table **discovery, not
  confirmation**. The two capture caveats travel with it: "Veto D1 - but M5
  today" writes an ordinary veto row, and a reason introduced by a later
  vocabulary keeps its own cohort.
- **A pre-existing defect found while wiring it: the Focus Pick Review step had
  been rendering an empty table on the live desk since it shipped on
  2026-08-18.** `_join_focus_week` composed its paths as
  `PERSISTENT_DATA_DIR / name`; that constant is the home ROOT while
  `human_focus_daily_picks.csv` and `human_focus_outcomes.csv` live under
  `data/runtime`. Both reads missed every time, and the function's own "a
  missing CSV is a quiet week" forgiveness turned the miss into a plausible
  blank page instead of an error. Both joins now use the NAMED CONSTANTS
  (`HUMAN_FOCUS_DAILY_PICKS_FILE`, `HUMAN_FOCUS_OUTCOMES_FILE`,
  `VETO_COHORT_PERFORMANCE_FILE`), which is the only spelling that cannot drift
  from where the writers put them. The landed tests hid it by patching
  `PERSISTENT_DATA_DIR` and writing the fixtures directly beneath it, so the
  fixture encoded the same wrong assumption as the code; they now redirect each
  file by its constant. Live read after the fix: **16 cohort rows and 605 focus
  pick rows for the 08-17 week, where the pane previously showed zero.**

### 2026-08-24 - AI-P4: a dead broker credential chain becomes visible

- **The Questrade refresh chain had been dead since 2026-08-19 and nothing on
  the desk said so.** 0 of 142 Questrade session days covered, 56 identical
  `500 Server Error ... oauth2/token` rows, one whole broker - including a
  TFSA - absent from the journal, from walk-away analysis and from everything
  the AI layer reads. It was found by opening `trade_journal.sqlite3` by hand.
  Questrade issues **single-use** refresh tokens, so a broken chain never heals
  itself and no retry brings it back: it needs the trader, which is exactly why
  it needed a surface rather than another log line.
- **New `scripts/journal_health.py`** classifies the chain into five states and
  is deliberately Qt-free and `ui`-free, so the Journal Health tab and
  `operations_audit` (which renders inside System Health, frozen) can share one
  verdict without either importing the other's world. A test asserts that
  import cleanliness rather than trusting it.
- **The honest states are the point.** An absent setting is `not_configured`,
  never `ok` - "nobody set this up" and "this is working" must not render the
  same. A database it cannot read is `unknown`, never `ok`. A token that never
  refreshed reports its age as **absent**, not as zero (Phase 0.7 ground rule
  6). A non-auth broker outage (503 on `/accounts`) does **not** read as a dead
  chain, because telling the trader to paste a token would spend the one thing
  the surface exists to spend - their attention.
- **A recorded auth failure outranks a fresh-looking stamp.**
  `journal_questrade_expires_at` records the last refresh that WORKED, so a
  chain that broke an hour ago still carries this morning's timestamp; a
  freshness check alone would have called the dead chain healthy for five days.
  Silence beyond **3 days** (Questrade's own refresh-token lifetime) is
  presumed dead on its own.
- **Two defects were found in this packet's own first run against live data**
  and are now pinned by tests: the headline reported the *coverage day* as the
  failure date (a year-spanning backfill marks every session day FAILED from
  one broken chain, so it claimed a 2025 outage for a chain that broke last
  week - `day` and `recorded_at` are now distinct fields); and comparing an
  aware caller clock against the naive stored stamp raised `TypeError`,
  normalized by ATTACHING the caller's zone to the naive side, never stripping
  the aware side (CLAUDE.md, `_gate_moment`).
- **The audit check takes its store as a parameter**, like every other check
  there. `test_capture_readiness_checks_reach_the_audit_and_never_read_the_shared_home`
  caught the first draft reading the trader's real journal from a sandboxed
  audit run; `build_operations_audit` now injects `journal_db_path`.
- The Journal ▸ Health banner is hidden unless the chain needs a hand - a
  permanently present "all good" strip is furniture - and its style lives in
  `theme.qss` as `QFrame#BrokerChainBanner`, so showing it costs a property set
  rather than a stylesheet parse.

### 2026-08-24 - AI-P3: the nightly journal slot stops being mute

- **Every `journal_import` ledger row ever written carried an empty reason.**
  The runner records `outcome["reason"]`; `run_nightly_journal_import` returned
  its findings under `messages` and no `reason` key at all. So the nine
  lifetime failures said only "failed", and diagnosing any of them meant
  opening `trade_journal.sqlite3` by hand - exactly the failure mode the ledger
  exists to prevent one level up. The night now returns a `reason` built from
  what it measured: executions imported, trades rebuilt, self-heal repaired and
  unresolved, positions reconciled and mismatched, and FX left unconverted.
- **A failure names its own source.** `run_journal_backfill` now returns
  `failures` beside `status`, and every one of its eight `had_errors = True`
  sites appends the thing that failed ("Questrade 51830546 2026-08-15..08-22:
  500 Server Error ... oauth2/token", "IBKR Flex: statement declared no span").
  `messages` mixes routine notes with failures; `failures` is only what made
  the run not-OK, so a caller can build a ledger reason without guessing which
  lines mattered. The reason names the first three and **prints the count it
  dropped** - a dead Questrade chain fails once per 31-day chunk, and a silent
  truncation would read as "that was all of it".
- **The alleged defect was REFUTED by reproduction.** The review recorded that
  a run which imports successfully but finds reconcile mismatches records
  FAILED and burns its attempts. It does not, and never did: only a reconcile
  *exception* sets `had_errors`, while mismatches are appended to `messages`
  and the run returns OK. `test_reconcile_mismatches_do_not_make_a_successful_import_a_failure`
  passed on first run and is kept as the regression that pins it.
  What was actually wrong was the muteness above - the reason the real cause
  (a dead Questrade OAuth refresh chain, 0 of 142 days covered) could sit
  undiagnosed for five nights.
- **Reconciliation itself is untouched.** MISMATCH is still written to
  `import_runs` exactly as before; only the slot-status reporting changed.
  Nothing here writes a number it did not measure (Phase 0.7 ground rule 6): a
  night whose reconciliation never ran says nothing about positions rather than
  claiming it checked none.

### 2026-08-24 - AI-P5: the picklist caveat stops being a hand-maintained fact

- **`trader_judgement`'s picklist caveat is now DERIVED from the picklist.**
  The scope ships two machine-written caveats as package data, and one of them
  states which setup claims the capture rail actually offers - a fact the model
  is never asked to infer, because a reader who does not know it reads a claim's
  absence as a trader preference. It was hand-written prose duplicating a
  code-owned list. `ai_summary._offered_claim_caveat()` reads
  `ui.annotations.setup_claims.offered_setup_claims()` - the same function the
  rail renders from - so admitting a claim updates the caveat by itself. The
  live text now names 13 claim types (Main swing's 9, plus the three
  post-earnings families and `second_dev_breakout`).
- **The premise this packet was authorized on was REFUTED, and the packet was
  built anyway for a different reason.** The review recorded the caveat as
  still saying "only the 'Main swing' group". It did not: the text had already
  been corrected when the picklist widened on 2026-08-21, and
  `test_the_two_caveats_travel_with_the_scope_as_data` already pinned the
  corrected content. What was true is the failure mode *behind* the alleged
  defect - the caveat only kept up because a human retyped it, and every test
  stayed green while it was stale. Pinning content catches a caveat that has
  gone stale; deriving it catches one that is about to.
- **The picklist definition moved out from behind Qt.** `MAIN_CLAIM_GROUP`,
  `EXTRA_CLAIM_IDS` and `offered_setup_claims()` moved from
  `ui/widgets/capture_rail.py` (which imports PySide6) to
  `ui/annotations/setup_claims.py` (Qt-free, and already the owner of the claim
  registry), because `ai_summary` runs headless in the overnight slate. The
  rail re-exports all three and delegates through the module, so existing
  imports are unchanged and a test that patches the source patches both - the
  rail and the caveat cannot disagree about what was offered.
- **A picklist it cannot read is declared UNKNOWN**, never a remembered list
  (plan.md sec 5: missing data is uncertainty, never confirmation).
- Docs reconciled en route: `LOCAL_AI_AUTOMATION_PLAN` §7.2 carried the same
  stale enumeration and now records that the caveat is derived rather than
  restating it; §6.4c still read "QUEUED - do not build yet" for the
  `journal_import` slot that ships **first** in `default_slots()`.

### 2026-08-23 - Sol's three blockers: the sweep becomes safe to enable

- **The autorun could never actually sweep.** The worker fired at close+10, the
  sweep correctly deferred to close+35, and the day was stamped done anyway
  because the refresh had succeeded. **Two jobs now have two clocks and two
  completion stamps**: the sweep is due at the real close + 35 minutes and
  stamps only when it swept; the refresh is due at close + grace and **waits for
  the sweep whose rows it reads**; a deferral or failure leaves the day open;
  a running worker does not start a second.
- **Early closes get a dedicated seam** (`scripts/market_early_close.py`): day
  after Thanksgiving, 24 December when it is a session, 3 July when the 4th is a
  weekday. `market_calendar` and `market_session` are **untouched** - they model
  every close as 16:00 ET on purpose and feed detectors, scanners and the
  overnight window. An unscheduled early close answers "regular", which makes
  the sweep wait longer rather than run early.
- **Finalization is one transaction per trade with a write-ahead intent**:
  machine-wide lock, disk re-read, intent committed **before** the append, the
  append skipped when the CSV already holds that final, then the finalization
  committed with `fsync` on file and directory. **A failed commit is not a
  finalization** - it is returned, counted and reported as such - and
  `resolve_unfinished_finalizations()` settles anything left mid-transaction
  against the CSV at load. Crash points covered: before the append, after it,
  during the temp write, during `os.replace`, after the commit, mid-batch.
- **The transaction is fenced across processes** with `local_writer_lock` (named
  mutex AND byte-range file lock, failing closed) and re-reads the authoritative
  disk state inside it. A test runs two real Python processes that both load the
  checkpoint before either commits: one finalizes, one skips, one row lands.
- **`launch_gui.py` gained the single-instance guard** (`scripts/single_instance.py`),
  so every launcher path is covered rather than only `launch_gui_auto.ps1`. It
  fails **open** when the machine has no exclusion primitive and **closed** when
  a desk holds the slot, exiting 0. `--selftest` and `--run-scan` are outside it;
  `--allow-second-instance` overrides. It is defence in depth: the outcome path
  stays correct with two desks running.
- **`_save_pending_bounce_outcomes` reports whether it landed** instead of
  swallowing silently; the finalization path uses a strict commit that raises.
- Recorded, not fixed: `market_session` resolves the desk's local zone to a
  **fixed offset** on Windows (no IANA key), so a session window for a date in
  another DST regime is an hour out. It cannot reach this scheduler, which only
  compares now against today's close; fixing it moves labels and slot times
  across the desk.

### 2026-08-23 - review round 1 part 2: the R10.A blockers

- **`outcome_sweep_autorun` defaults OFF** (trader decision). The sweep does not
  fire itself until its first live session is signed off; calling it by hand
  always sweeps, and it announces the reason once per process. Dual-write,
  registration context, tier capture and `unresolved`-instead-of-zero stay on.
- **BLOCKER-1: two finalizers, one lock.** The sweep (close+10, worker thread)
  and the per-symbol path (through close+30, scan thread) both mutated the
  pending dict with no lock over a non-atomic checkpoint whose loader answered a
  torn file with `{}`. Now: one re-entrant lock over every read-check-write, the
  sweep re-reads each entry under it, the per-symbol path consults the same
  finalized-id set, the sweep defers until close+35, the checkpoint is temp +
  `os.replace`, and an unreadable one is **quarantined and logged** rather than
  read as an empty backlog. Test: two threads, one final.
- **MAJOR-2: the backlog's own measurements are recovered.** `last_measured`
  landed on 08-23 and **0 of 576** checkpoint entries carry it, so the sweep
  would have called every backlog trade "no bars after entry" - including 563
  stop-outs whose milestone rows are in the CSV. Those rows are now recovered
  (read-only, furthest milestone wins, `last_close` reconstructed from the row's
  own numbers), and a trade with nothing to recover reads
  **`no_measurement_in_checkpoint`**, which is a different fact.
- **MAJOR-3: `close_r` means one thing everywhere** - R at the EOD close under
  `eod_hold`. Without bars through the close it is blank and the row is
  `unresolved`, never -1.0. The stop exit lives in `context.exit` as
  **`stop_exit_r` under a named fill assumption**, with `gap_through_stop` and
  `ambiguous_interval_bars` (R10.0's stop-first rule, counted not absorbed).
- **MAJOR-4: the measurement no longer comes from the forming bar.** The session
  frame is cut to completed bars through the one shared rule, and
  `replace(tzinfo=None)` is gone from the path. Authorization was conditional on
  proving the helper feeds no detector - it has one caller, asserted in a test.
  *Found on the way:* `completed_bars._TIME_KEYS` has no `datetime` key, so the
  obvious call drops every bar silently; the adapter is at the call site and the
  shared rule is untouched.
- **MAJOR-7: the learning refresh runs in SHADOW first.** Corrected finals move
  segment averages, which decide `muted`/`proven`, which decide suppression.
  `bounce_learning_refresh_mode` defaults to `shadow`: a state file beside the
  live one plus a diff of every segment whose verdict would move, live state
  frozen until the trader flips it.
- **MAJOR-5**: the canary cap is per **session-day** (a process-lifetime cap
  silenced the mirror after 8-14 days) and writes a `canary_capped` event.
  **MAJOR-6**: `pending_after` is answered from the row, since the mirror runs
  before the caller pops.
- Minors: sweep finals carry the measurement's bar count; `mfe_pct`/`mae_pct`
  stored where the stop branch reads them; the refresh date is stamped inside the
  worker so a raising sweep is retried; coverage rows carry an id and the file is
  written atomically; the ledger documents `session_date` (write session) versus
  `trade_date` (trading session).
- **Red runs recorded**: 64 failed / 17 passed against the pre-fix writer,
  81 passed against the fixed tree.

### 2026-08-23 - review round 1: a clock-dependent suite, and four R10.V corrections

- **Two tests failed only between 06:30 and 07:00 PT.** Inside the open-digest
  window an ordinary alert is legitimately folded into the digest row - no feed
  row, no beep - so two tests asserting an ordinary alert surfaces read correct
  behaviour as a failure. Both pin the clock now; the digest stays enabled, and
  two new tests cover the mechanism directly.
- **The share percentage was flattered by its own denominator**: the store is
  **1,136,420 rows and 98.29% shares**, not 1,117,170 and 99.98% - the
  measurement dropped files with no `volume_unit` column. Those 19,250 rows are
  counted as `no_column`, named in the health tile, and the reconciliation JSON
  is corrected with what it supersedes recorded in it.
- **`no_overlap`**: a file whose Yahoo history overlaps none of its dates is a
  named non-change now (AVNS, SATS, SKYT) instead of falling through as `ok`.
- **`--only-unfinished`** scopes a backfill re-run to the 63 files still holding
  a non-shares row. The nine "no Yahoo data" symbols are recorded as
  **unsettled** - a Sunday probe reproduces 404s for names like BK that cannot
  plausibly be delisted - and a weekday probe settles it.
- **The frozen pre-backfill copy is on the DAS**, 1,958 files, manifest SHA-256
  identical on both sides.

### 2026-08-23 - R10.A: the sweep is visible on the desk

- **An `outcome_sweep` System Health tile** reads the coverage the sweep files
  and never sweeps anything itself. **No file is `unknown`, not healthy** - a
  sweep that has never reported is indistinguishable from a sweep that never
  ran, and that indistinguishability is how 576 pending outcomes accumulated
  over two months.
- **Degrades on a backlog above 200** (D3 measured 576) or on a sweep older than
  four days - long enough to cover a long weekend, short enough that a silent
  week cannot pass.

### 2026-08-23 - R10.A / D8: registrations record what they can measure, tiers arrive on their own row

- **The tier was absent from 0 of 7,863 registered rows because of ordering,
  not oversight.** Every call site registers the outcome and evaluates the
  alert's tier *afterwards*, so at registration the tier does not exist yet.
- **It is emitted as its own `tier_assigned` ledger event** rather than
  back-filled onto a row that could not have known it - which is what an
  append-only store is for. Wired at all **8** sites where a quality verdict
  follows a registration, and a test walks the source to prove none was missed.
  Reordering a live alert path is a different kind of change and is not this
  packet's.
- **Everything D8 asks for that IS measurable at registration now is**: family,
  engine version, day-part (from the session, not the wall clock), session RVOL,
  `env_key` (environment + day-part, the pair the learning segments are keyed
  by), risk as a **percent of price** and as an **ATR multiple**. Each is
  measured or blank; none is estimated.
- **None of it can cost an alert.** A test that makes every accessor misbehave
  found one unguarded call - `get_market_environment()` - and it is guarded now.

### 2026-08-23 - R10.A / D3+D4: finalization no longer depends on being scanned again

- **`sweep_pending_bounce_outcomes()`** finalizes every pending outcome whose
  session is over, needing **no bars and no IB**. The backlog it exists for:
  **576 pending, 94 older than 2026-08-18, 17 from June, the oldest
  2026-06-22**. Finalization only ever happened inside
  `_update_pending_bounce_outcomes`, which runs for a symbol the scan is looking
  at right now, so a name that stopped being scanned was never finalized at all.
- **D4 is that same gap, confirmed.** 2026-08-21 has 409 `registered` through
  394 `12_bar` and **0 `final`**: the milestones ran all day and only the EOD
  pass was missing. Not an IB outage.
- **It runs in the existing after-close worker**, before the learning refresh
  that reads the rows, and a sweep failure is logged without costing the refresh.
- **Idempotent by construction.** Finalized ids live in the same checkpoint as
  the pending dict, so a restart or a second pass cannot write a second final.
  The memory is bounded at 5,000 - it is de-duplication, not a record.
- **Expiry is three completed sessions**, counted in sessions rather than days
  so a long weekend cannot expire a two-session-old trade, and **only for a
  trade that measured nothing**: one with evidence finalizes on that evidence
  however old it is.
- **It reports itself.** A coverage row goes to the ledger and to
  `diagnostics/outcome_sweep_coverage.json` - counts by reason, still-open,
  unparseable, expired. A sweep that reports nothing is indistinguishable from a
  sweep that never ran, which is how the backlog stayed invisible for two months.
- **A writer failure leaves the trade pending** rather than losing it, and
  `_is_eod_finalization_due` gained an injectable clock so the sweep and its
  tests share one - without it the "still open" branch could not be tested at all.

### 2026-08-23 - R10.A / D2: a finalization stops writing numbers it did not measure

- **The fabricated zero is gone.** The EOD writer defaulted `eod_close` to the
  entry price whenever it had no bars in hand, producing **1,164 of 6,907
  in-window finals with `close_r` exactly 0** - every one of them with
  `eod_close == entry_price`, and none of the 5,743 non-zero finals like that.
- **Three honest outcomes replace it.** Bars measured earlier with a **stop hit**
  among them finalize **at the stop** (`close_r = -1`) - that is the 563 stop-outs
  that were scoring 0R. Bars measured earlier with no stop finalize at the **last
  measured close**. Nothing ever seen after entry finalizes **`unresolved`** with
  blank numerics and the reason `no_bars_after_entry`.
- **The state now remembers what each bar measured** (`last_measured`), which is
  what makes an honest finalization possible without refetching anything: a
  trade whose own earlier rows recorded a stop cannot be finalized as a scratch.
- **How a row's numbers were arrived at rides in `context_json`**
  (`finalization.basis`, `.measured_bars`, `.reason`), never in a new column -
  the CSV header stays exactly as it was.
- **Legacy rows are untouched.** `unsettled_close_mask` and the registry's
  `fabricated_zero_v1` describe the same old signature and a test asserts they
  agree row for row; new `unresolved` rows deliberately do not match it, because
  they are blank rather than zero.

### 2026-08-23 - R10.A: the outcome ledger runs as a dual-write canary

- **Every row the BounceBot writes to `intraday_bounce_outcomes.csv` is mirrored**
  to the append-only ledger in `data\runtime\evidence_ledgers`. **The CSV stays the
  authority during the canary**: the mirror runs after the CSV write, cannot
  change it, and cannot fail it. The point of a canary is that the two can be
  compared before anything is asked to believe the new one.
- **One writer, one call site.** `_append_bounce_outcome_row` is the only place
  outcome rows are written, so it is the only place that mirrors - a test
  asserts there is exactly one call, because a second would be a second writer.
- **Fail-open everywhere.** A ledger that raises, a directory that cannot be
  made, a module that will not import: logged, and the CSV row still stands.
- **Bounded**: 50,000 rows per process, announced **once** on reaching the cap
  rather than once per row. A defect in the mirror costs disk space once.
- **No header widening.** `family` (derived from the id, which is where it has
  always been hiding), the canary marker, the source store and `pending_after`
  exist in the ledger row only; `BOUNCE_OUTCOME_COLUMNS` is untouched.
- **Kill switch** `evidence_ledger_dual_write="off"` in `local_settings.json`.
  Only `off` stops it - an unreadable setting leaves it running, because a
  canary that switches itself off on an unrelated failure proves nothing.

### 2026-08-23 - R10.A: the append-only ledger exists

- **`scripts/evidence_ledger.py`** - `EvidenceLedger`, month-segmented JSONL,
  with `intraday_outcome_ledger()` writing `intraday_outcome_event_v1` into
  `data\runtime\evidence_ledgers` (the directory the cold push already covers;
  a test asserts the two agree).
- **A caller cannot overwrite the ledger's own fields.** Schema name,
  `event_at`, `session_date` and writer identity are applied last: a row that
  can lie about who wrote it is not evidence. The caller's mapping is copied,
  never mutated.
- **Every row carries UTC and the market session.** A 20:30-local write on the
  21st is 03:30 UTC on the 22nd; only `astimezone` gets the session right, and
  the segment follows the session rather than the UTC month.
- **Every row says who wrote it** - host, pid, and a run id when the caller has
  one. When two desks ran concurrently on 2026-08-20 the outcome store could not
  say so and the duplicates had to be attributed by inference.
- **A torn line is counted, never skipped.** `ReadResult.coverage_note` reports
  `n=` with the unreadable count beside it, because a silently dropped row makes
  a gap look like an absence of events. A row that cannot say its session is
  excluded from a *window* and counted, but kept in an unwindowed read - it is a
  real row, just an unplaceable one.
- **Append-only in fact, not by convention**: a correction is a new row with
  `supersedes`, and a test asserts the file only ever grows by suffix.
- **13 months hot.** `cold_segments()` NAMES what is cold and never deletes it -
  moving it is the cold push's job and deleting it is nobody's.

### 2026-08-23 - R10.A: the rule registry reaches v1 (four more rules, all reproduced)

- **`h1_bar_start_v1`**, **`fabricated_zero_v1`**, **`duplicate_row_v1`** and
  **`risk_below_floor_v1`** join `daily_volume_mixed_v1` in
  `scripts/evidence_rules.py`. Every one was **re-measured against the live
  store** rather than trusted from the audit, and every one reproduces:
  duplicates **742 / 609 / 430** on 2026-07-24..08-21 and **394 / 345 / 300** on
  2026-08-07..08-21 (both exact), risk-below-floor **1,127** all-time,
  `h1_bar_start` **9,623 of 9,914** minute-30 rows.
- **The registry states its measured precision, not a round number.**
  `h1_bar_start_v1` is 9,623/9,914 - the family half of the rule is
  load-bearing, because 291 of 6,054 non-H1 rows also land on minute 30.
- **`duplicate_row_v1` requires its window and echoes it back.** The same
  allegation reproduced at 742 on one window and 394 on another, so a count from
  this store that travels without its window is not evidence.
- **`family_from_event_id` exists because the store has no `family` column.**
  Validating `h1_bar_start_v1` the first time I passed a family that did not
  exist and it tagged **0 of 9,914** rows - silently, because "no match" and "no
  data" looked identical. The rule now derives the family from the id when it is
  not supplied, and a test pins the derivation.
- Missing inputs are **unknown**, never a pass: an unreadable stamp, a missing
  `eod_close`, an entry price of 0. A date with no time carries no minute and is
  unknown rather than minute 0.
- **A time-of-day fragility in the chart suite, fixed.**
  `test_unscanned_symbol_fetches_todays_candle_without_persisting_it` forces
  `session_has_opened` True, so any run before 06:30 local - or on a weekend -
  lands inside the Yahoo early-print suppression window, the preview is
  correctly withheld and the test reads that as a failure. Found at 00:10 on a
  Saturday. The suppression window is `test_forming_bar_honesty.py`'s subject
  and is now pinned there rather than left to the clock. No product change.

### 2026-08-23 - R10.V step 7: a recompute cannot see a bar that came after its session

- **The tracker catch-up trims its daily frames to the session it is
  recomputing.** It held today's frames while replaying a past one, which is how
  a payload whose `data_session` said 2026-08-20 came to carry **2,739 setups
  with a snapshot dated 08-21** and 452 scenario exits on a bar that session had
  never seen (the S2 defect).
- **`<=`, not `<`**: the session being recomputed is the session we have.
- **An unparseable session does not trim.** This is a point-in-time guard, not a
  filter, and silently emptying every frame because a stamp was malformed would
  be a far worse failure than the one it prevents. A session before every bar
  leaves nothing to recompute, and the caller skips the setup rather than marking
  it from nothing.
- **The indicator cache is now keyed by (symbol, session)**, because the frame it
  is built from is session-dependent - caching by symbol alone would hand a later
  session's indicators to an earlier one, which is the same defect wearing a hat.
- Evidence-only by construction: the recompute writes the tracker payload, which
  nothing in the live decision path reads back as a signal.

### 2026-08-23 - R10.V steps 5 and 6: nothing to re-freeze, and a nightly unit check

- **Step 5 is a recorded no-op.** No AVWAP-derived golden fixture moved across
  the whole packet - the only addition under `tests/fixtures/` is the control
  step 1 created. That was predicted in advance and for a stated reason
  (fixtures feed fixed bars, so repairing the store changes what the desk
  computes and not what the suite computes), which is what makes it evidence.
- **A `daily_bar_units` System Health tile** reports rows by `volume_unit`,
  files by schema, files not all-shares, and the cliff count. It **reads and
  never measures**: the measurement takes ~7 s over 1,958 files and rides the
  nightly evidence-snapshot job instead, because a tile a human waits on is a
  tile nobody opens. A failure there is logged and never fails the backup.
- **`lots_rth` degrades; `unknown` does not.** A round-lot row means something
  got past a write seam that refuses IB volume - the splice starting again. The
  188 `unknown` rows are the known residue Yahoo cannot supply, named in the
  backfill manifest and unclearable by anyone; they are reported in full and set
  no status. An alarm nobody can clear is an alarm people learn to ignore.
- **The cliff count is reported and never sets the status**, because after the
  backfill 19 all-`yahoo` files still step >20x - a real market event, not a
  unit mix. A measurement older than two nights degrades, since it cannot answer
  today's question. No measurement at all is **unknown**, not clean.
- One cliff definition (`scripts/ops/daily_bar_cliff.py`) serves both the
  backfill manifest and the nightly check, so "over 20x" means the same thing in
  both places.

### 2026-08-23 - R10.V step 4: the daily store is repaired

- **99.98% of rows are now share-denominated** - 1,116,982 of 1,117,170 - and
  files carrying a >20x volume step fell from **1,795 to 53**, median residual
  ratio 158x to 29x, with 1,920 of 1,958 files on `daily_bars_schema=v2`. AAL:
  2026-07-24 74,218,900 -> 07-27 **93,953,900**, where it read 836,047 before.
- **Prices were not touched**; only volume and the two provenance columns moved.
  A verified frozen copy of the whole directory was taken first
  (`evidence_frozen\daily_bars_pre_backfill_2026-08-23`), and the run refuses to
  start if that freeze is incomplete. Zero IB traffic.
- **Two refusals, both learned from the dry run rather than guessed.** A file
  whose rows Yahoo covers under 90% is left alone - a rewrite would have changed
  **2 of EA's 787 rows**, which manufactures a boundary rather than removing one
  (13 files). A file the run would leave with a cliff it did not have, or a
  bigger one, is left alone (13 files). Both named in the manifest.
- **`CON_.parquet` holds `CON`**, and a batched download silently drops the odd
  ticker (BK returned empty in a batch and full alone), so every missing symbol
  gets one individual retry. After that, **9 symbols genuinely have no Yahoo
  data** and their files are untouched and named.
- **The exit gate was corrected against measurement.** "0 files > 20x" is not
  achievable by any correct implementation: 19 fully-rewritten, all-`yahoo` files
  still step >20x - DJT at its listing, OKLO's de-SPAC, POET, FFAI, QXO, SOXS -
  because a 20x volume step is a real market event. The gate is now **0 rows with
  `volume_unit != shares` that Yahoo can supply**, with the cliff detector as a
  secondary signal.
- **A reporting bug found by checking**: the applied manifest said 44
  cliffed-after where an independent scan said 53 - the nine no-data files kept
  their cliffs and were never counted. Fixed with a test that asserts the
  manifest reconciles with `scan_store()`, and a reconciliation note filed beside
  the manifest rather than the manifest being rewritten.
- New: `scripts/ops/daily_bar_cliff.py` (one cliff definition, shared by the
  backfill and step 6's health check; the boundary date is refined to the bar the
  step happened on, because a rolling median crosses about half a window early)
  and `scripts/ops/backfill_daily_bar_volume.py` (dry run by default).

### 2026-08-22 (night) - R10.V step 3: the store takes shares, and a collision prefers them

- **An IB row is written with its prices and NO volume**, never a rescaled
  number - the ratio is symbol-dependent, so a x100 conversion would replace a
  visible error with an invisible one. The row carries `volume_unit=lots_rth`,
  so the absence is explained rather than merely present.
- **A date collision prefers `shares` > `unknown` > blanked**, in either arrival
  order; among equals the later row still wins, which is the previous behaviour.
  `keep="last"` alone is how a share-denominated row was replaced by a
  round-lot one. **The rank follows the data, not the label**: a row labelled
  `shares` with no number cannot outrank one that has a number.
- **Deliberate exception: `unknown` legacy rows keep their volume** until step
  4's backfill. Blanking them would empty the volume column of the whole store,
  and an AVWAP with no weights is not a safer answer than one with an old weight
  - it is no answer at all, live, for every symbol.
- **A blanked row stays readable.** `dropna` disqualifies a row only for a
  missing price, and both weighting loops skip a blank exactly as they already
  skipped a zero - **NaN is not `<= 0`**, so one blank bar would otherwise poison
  the accumulation. **The sigma formula is untouched**; what changed is which
  bars enter it.
- **Three more readers made blank-safe**: `chart_snapshot.load_d1_bars` emits
  `0.0` (never NaN on the paint path), `avg_vol_20` reads 0 and therefore
  **rejects** an unmeasurable candidate at the liquidity gate instead of raising
  on `int(nan)`, and `last_volume` becomes `None` rather than 0 because it is a
  bucketed liquidity factor where 0 reads as "illiquid", not "unknown".
- **No golden fixture moved**, as predicted.

### 2026-08-22 (night) - R10.V step 2: the daily store records where each row came from

- **Provenance is per ROW, not per frame.** `source` (`yahoo` | `ibkr` |
  `unknown`) and `volume_unit` (`shares` | `lots_rth` | `unknown`) are columns
  on the durable daily parquet, because the store IS a merge of two sources and
  a frame-level attribute cannot survive one. The file carries
  `daily_bars_schema=v2` in its Arrow metadata, separating "predates provenance"
  from "has provenance and every row is unknown".
- **`cache` is not a source value.** A row read off disk is `unknown`, never
  `cache`: recording the reading path as the author would look like provenance
  while carrying none.
- **Rows that already know what they are are never relabelled**, and provenance
  is stamped before de-duplication, so step 3's collision rule decides between
  two rows that both know their unit.
- **An untouched file stays v1 on purpose** - the write is still skipped when
  the bars did not change, so nothing is quietly upgraded to a v2 full of
  `unknown`. Step 4's backfill converts them with a manifest.
- **Both fetch seams declare their source before normalizing.** Setting it after
  (as the first version did) stamped every row `unknown` while the frame said
  `yahoo`; `_set_daily_bar_source` stays a pure attribute setter, because making
  it backfill unknown cells would have relabelled old IB rows as Yahoo at the
  merge seam.
- **Every consumer reads v1 and v2, one test each** - the D1 scanner's durable
  loader, `chart_snapshot.load_d1_bars`, `human_focus_tracking` (the path
  `ai_jobs/cohorts.py` grades vetoes through), `setup_playbook_study`, plus two
  the cliff report's consumer table had missed: `ui/services/bar_cache.py` and
  `research_warehouse/ingest_existing.py`. The warehouse's `provider="UNKNOWN"`
  docstring is now understated; wiring it through is **owed, not done** - it is
  a warehouse change this packet does not authorize.
- **No golden fixture moved**, as the step-1 baseline predicted. One test
  asserted the old six-column contract and was updated to the new one.

### 2026-08-22 (night) - R10.V registered; the AVWAP fixtures are proven not to read the live store

- **The cliff packet is plan.md Phase 0.7 item 11 (R10.V)**, seven steps, and it
  runs **before R10.D**: a point-in-time transition ledger built over a
  unit-mixed store would record the splice as history.
- **Step 1's stop condition was measured, not inspected.** A pytest plugin
  wrapped `builtins.open`, `Path.open`, `Path.read_bytes` and
  `pandas.read_parquet` to record any access inside the live daily/intraday
  parquet roots, and the whole suite ran under it: **4,205 tests, zero
  accesses**. Every fixture carries its own bars, so the packet proceeds.
- **`mixed_unit_avwap_v1` is frozen, and it pins the WRONG answer on purpose.**
  Twenty hand-constructed daily bars, three series with identical prices
  differing only in volume. The splice (bars 12+ in IB round lots) costs
  **-2.28% on VWAP, -2.27 points on UPPER_2, and halves sigma to 0.482x**. The
  uniform rescale control costs **nothing** - `lots` reproduces `shares` to 0.0
  on vwap and 1.3e-15 on sigma, because a volume-weighted ratio cancels a
  constant factor. That row is the argument for C-prime: a x100 conversion on
  the IB path would have replaced a visible error with an invisible one.
- **The sigma formula gained a direct guard** (plan.md sec 5, never swap it): an
  independent reimplementation of the running-deviation variant must agree, and
  the distribution-stdev variant must *disagree* on this fixture - if the two
  ever agree, the fixture cannot discriminate and is rebuilt with more trend.
- **Step 5's blast radius is predicted before it runs**
  (`docs/analysis/AVWAP_FIXTURE_BASELINE_2026-08-22.md`): only two fixtures put
  bars through `calc_anchored_vwap_bands`, three carry already-computed levels as
  inputs, and the backfill cannot move any of them. If one moves at step 4, a
  test reads the live store and the proof has expired.

### 2026-08-22 (night) - the daily-bar unit problem gets a pin and a name

- **`daily_bars_source` pins the durable daily-bar store to one source.**
  `"yahoo"` pins; an absent key or anything else - including a typo - resolves to
  `"auto"`, which is exactly the previous behaviour. Read at
  `_fetch_live_daily_bars` (`master_avwap_lib/legacy.py`), the same seam as the
  IB failure circuit and **independent of it**: `_IBKR_HISTORICAL_YAHOO_ONLY` is
  a state repeated failures flip and each scan clears, this is a preference, and
  either alone routes daily bars to Yahoo. Announced once per scan, not once per
  symbol. **Intraday is deliberately not pinned.** Set on the desk. The 482
  fixture/AVWAP/tracker tests were run to show the golden fixtures do not move.
- **Why a pin and not a rescale.** IB returns regular-session volume in round
  lots (`whatToShow="TRADES"`, `useRTH=1`); Yahoo returns the consolidated
  session in shares. The measured ratio is symbol-dependent - SPY 1.0x, TSLA
  56x, AAPL 81x, A 162x, NVDA 188x - so a constant would make the store
  consistently wrong instead of visibly wrong. The trader chose **C-prime**:
  provenance first, a Yahoo-only durable store for volume second, refetch third.
- **`scripts/evidence_rules.py` is the reader-side rule registry** (R10 ground
  rule 5): history is never rewritten, so known-bad rows are TAGGED by a
  versioned rule name that is never edited in place. It reads and never writes,
  and it reaches no detector, score, gate, alert or Focus decision.
- **`daily_volume_mixed_v1` is its first rule.** A session is `mixed` if any run
  manifest that day reported a non-`yahoo` `provider.daily_bars.success.*`
  count, `shares` if all of them reported Yahoo, and `unknown` otherwise -
  derived from evidence the scans already wrote about themselves rather than a
  hard-coded date list. **`mixed` dominates** (one IB run contaminates the
  session, including 2026-08-20 where two desks ran concurrently and only one
  used IB) and **`unknown` beats `shares`** (a manifest we cannot read may have
  been the IB one). Measured on the live tree: **13 of 15 manifest-covered
  sessions are mixed**, back to 2026-07-31; only 08-03 and 08-17 are clean. That
  is wider than the two runs the trader named, which is the point of deriving it.
- **The rule states its own limit.** Manifests are pruned to 90 runs, so
  everything older reads `unknown` and reads `unknown` increasingly as time
  passes; `freeze_verdicts()` is how a rollup that must stay reproducible files
  what its numbers relied on. No rollup is wired to the tag yet, on purpose: the
  defect is confined to the **volume** column, so it moves volume-weighted AVWAP
  levels and not price-only readers, and the one existing rollup reads a store
  it has not been shown to touch. R10.A and R10.V build the consumers.
- **System Health names it.** The `daily_bar_source` tile reports the pin -
  healthy when pinned, **unknown** (not degraded) under the shipped `auto`
  default, because with `auto` the setting genuinely cannot say what a given
  scan wrote - and appends the manifest-derived history. History never sets the
  status: the pin governs what happens next, and a past that cannot be changed
  must not raise a permanent alarm.
- **The two 2026-08-22 evening reports are amended**, superseding their own
  numbers where Fable's field-level re-run beat them: the marks are stable
  (26,087 float32-to-float64 round-trips; 95% of the rest at or under 1.1 cents;
  genuine restatement 361 field-diffs on 136 symbol-dates), what moved is the
  **levels**, and the mechanism is a **splice** at 2026-07-29 (median x0.0088
  across 1,179 of 1,236 rewritten files). Stops did not move at all - 0 of 9,331
  stored anchor entries and stop references - because they are written at scan
  time and never replayed. A uniform rescale could not have done this: AVWAP is
  a volume-weighted ratio and scaling every weight cancels. Only a splice moves
  it, which is why the fix refuses IB volume rather than converting it.

### 2026-08-22 (evening) - the evidence snapshot is scheduled and stops duplicating itself

- **The rotated tracker `.bak` is excluded** by an explicit rule
  (`excluded_rotated_duplicate`), counted in the manifest like any other skip.
  Once the snapshot runs nightly, day N's main is day N+1's `.bak`; measured on
  the live scope that is one file, 939 MB source and ~133 MB compressed, saved
  every night. The on-disk `.bak` is never deleted - the tracker reads it back
  when the main payload is corrupt - and `exclude_rotated=False` is the switch a
  deliberate freeze uses.
- **`source_sha256` joins the manifest** beside the stored hash. The stored hash
  proves the archive is intact; only the source hash proves the content survived
  compression, and it is the only hash a restored file can be compared against.
  `verify()` stays on stored bytes so it remains cheap. For a SQLite copy the two
  differ by construction, because the backup API rewrites page layout.
- **`TradingBotV3 - Evidence snapshot` is registered**, daily at 20:30 PT - after
  the close, before the AI runner's 22:00 window, outside the 06:00-14:00 band
  where the launch task fires every 15 minutes. The task XML is exported into
  `scripts/ops/` so it is versioned like the scripts it launches, and a test
  parses its trigger and fails if the hour ever drifts into market hours.

### 2026-08-22 (evening) - the tracker rewrites settled outcomes: S1/S2 PROVEN

Evidence and documentation only; no runtime code changed.

- **The pre/post tracker pair is frozen.** `master_avwap_setup_tracker.json.bak`
  is rotated on every save, so the 2026-08-22 canary snapshot held the only pair
  that could ever prove S1/S2 and the next run would have destroyed it. Both
  payloads are now in `evidence_frozen/` locally and on the DAS, with stored and
  decompressed SHA-256 recorded; the `.bak` hash independently reproduces the one
  Fable measured.
- **S1 PROVEN, and worse than "historical exits move".** Across one run:
  218 setup-status transitions on 9,331 common setups, including 35 CLOSED→OPEN
  and 14 OPEN→UNTRADEABLE. Among 6,736 setups CLOSED in both runs, **2,737
  scenarios changed status or reason**, 1,306 changed exit date, and **2,618 had
  their event history dropped while status and R stayed identical**. A settled
  trade can be rewritten on its own historical date: AMCR LONG on 2026-07-28
  moves from `TIME_STOP @ 46.69, R 0.577` to `TARGET_HIT @ 45.55, R 0.360`.
- **S2 PROVEN to the unit.** A payload whose `data_session` is 2026-08-20 carries
  **2,739 setups marked 2026-08-21** and 452 scenario exit events on that forming
  bar. The prior run shows the same shape a day earlier, so it is systematic.
- **A correction to the claim that prompted the check:** of 1,309 same-dated
  historical closes that differ between runs, only **5 differ materially**; 1,304
  are float32-to-float64 precision from the Yahoo-to-IB source switch. S1 stands
  on status, reason and R; the precision half is a bar-source problem.
- Two verdicts of my own were wrong. **D5b becomes UNTESTED** - `orb_first_candle*`
  has never fired anywhere, and the rows I called "working as designed" belong to
  a different family. **D1d/D2b were a window mismatch, not a brief error**; on
  2026-08-07..08-21 the brief's figures are exact, and every number from that
  store now states its window.

### 2026-08-22 - R10.A (first half): the evidence that was on one disk now has a dated backup

`IMPLEMENTED` + `GREEN`; live gate is the nightly schedule plus one proven restore.

- **The measured gap.** `push_cold_to_das.ps1` mirrors ~270 MB of cold subtrees
  hourly and deliberately excludes hot state. What it excludes is the evidence
  this whole program exists to protect: `data\runtime` at **3.5 GB** (the 960 MB
  setup tracker plus its 939 MB `.bak`, the 203 MB outcome CSV, the journal
  SQLite, every outcome / cohort / Focus store), the **36 home-root evidence
  files**, `_tools`, and the machine-local diagnostics tree at **529 MB**. All of
  it existed on exactly one disk.
- **A snapshot, not a move.** Decision 0015 stands - hot files stay on the local
  SSD and are written local first - so `scripts/ops/evidence_snapshot.py` stages
  a dated copy locally and `snapshot_to_das.ps1` robocopies it to
  `\\MINI-PC\Trading Bot Data\backups\<YYYY-MM-DD>\`. An unreachable share exits
  0 and leaves the staged snapshot, exactly as the cold push does.
- **Copy-while-hot rules, because a backup taken mid-write is worse than none.**
  SQLite is copied through the backup API, not byte-for-byte, since the AI runner
  writes to the journal nightly. Any file at or above 256 MB must hold one size
  and mtime across a 60-second window or it is **skipped with a reason and
  counted in the manifest** - a snapshot that quietly omitted the 960 MB tracker
  would look identical to one that captured it. Any file at or above 64 MB is
  gzipped; ~2.4 GB of tracker and integrity JSON raw would be 60 GB a month for
  files whose daily diff R10.D will carry anyway.
- **`manifest.json`** records size and SHA-256 per stored file plus the skipped
  count by reason. `--verify` re-hashes every file against it.
- **Restore refuses the live store.** `restore()` raises rather than write into
  the home folder or the diagnostics tree; a drill that overwrites live state is
  how a drill becomes an incident. `restore_from_das.ps1 -DryRun` plans without
  writing, and prefers the local staging copy over the DAS since it is the same
  bytes on a faster disk and present even when the share is not.
- **Retention** 7 daily / 4 weekly / 12 monthly, computed as a pure function of
  the date list so pruning never depends on when it runs. `evidence_frozen/` is
  never pruned.
- **System Health** gains an `evidence_snapshot` tile: last snapshot date and
  age, files, bytes, skipped count, DAS reachability, last restore test. Absence
  reports **unknown**, not unhealthy - a machine that has not been scheduled yet
  is not a machine in trouble, and that is the repo's existing rule.
- **`push_cold_to_das.ps1` gains `data\runtime\evidence_ledgers`** (append-only
  and unboundedly growing, which is its shape) and both scripts' headers now
  state **two jobs, two scopes** with a test pinning it, so the next reader does
  not merge them.
- **A finding on the way:** `push_cold_to_das.ps1` existed **only** in
  `C:\TradingBotData\_tools` - the script protecting the evidence was itself
  unversioned, untested and unreviewable. The repo copy is now the source of
  truth, `_tools` holds an installed copy, and a test compares them byte for byte
  so an unreviewed edit to the running script cannot go unnoticed.
- `ops` joins `FIRST_PARTY_PACKAGES` in the spec: `operations_audit` imports
  `ops.evidence_snapshot` lazily and renders the frozen exe's System Health page,
  so a bundle without it would die at exactly the lazy import the spec-drift
  guard exists to catch.

### 2026-08-22 - R10.0: the evidence audit, and one failure made observable

`IMPLEMENTED` + `GREEN`. Read-only sweep; the only runtime change is the
observability fix below.

- **`docs/analysis/EVIDENCE_AUDIT_2026-08-22.md`** is the R10.0 decision
  register: every alleged evidence defect reproduced and classified with its
  command and numbers (12 PROVEN, 6 reproduced-but-a-number-differs, 3 REFUTED,
  4 UNKNOWN), the store inventory, the six-namespace family map, and the
  decisions R10.A-R10.I rest on - ledger-over-`bouncers.txt` authority, the
  session calendar, stop-first intrabar collision, frozen slippage, the
  evidence-based H1 bar-start rule, and one reconciled risk-floor definition
  (raw stored, 0.1%-of-entry analytic floor, existing 4R ranking clip, all three
  reported side by side and never substituted for one another).
- **Concurrent desks are proven** and the guard is authorized: on 2026-08-20 one
  pid overlapped three others, the worst for 3.8 hours, and the existing guard
  lives only in the scheduled-task PowerShell path. **It is not the duplicate
  fix** - the concurrent session supplies 25% of duplicate rows and no
  duplicated id was written twice within 5 seconds.
- **`ai_jobs/runner.py` no longer records a failure with a blank reason.**
  `run_nightly_journal_import` returns its explanation in `messages`; the
  runner's non-exception path passed only `reason=`, so the diagnostic was
  produced and dropped at the seam, and 20 nightly failures said nothing at all.
  `_failure_reason()` prefers `reason`, falls back to `messages`, and when a job
  fails with nothing to say records that fact naming the job. Successful rows
  are untouched. **The cause it was hiding, corrected the same day:** not a
  pending migration (that branch never runs here) but three nightly `had_errors`
  paths - a transient IBKR Flex failure, a Questrade `/activities` 400 on both
  accounts, and 19 reconcile mismatches - which mark a run FAILED *after* it has
  successfully imported, so the runner then retries it 3x per session.

### 2026-08-22 - R10 registered as plan.md Phase 0.7 (Evidence Plane program)

Roadmap only; no runtime file changed. Ten packets R10.0-R10.I with their ground
rules (append-only authorities with schema NAMES, never rewritten history;
unresolved-with-a-reason instead of fabricated numbers; UTC plus an explicit
session identity; one owner per store; evidence-facing statistics that report
counts, robust and raw side by side, concentration and a discovery/confirmation
label), the trader decisions that scope them, and their canary and
evidence-quality gates. Nothing in R10 promotes anything, and R9.3'''s declared
40-session window is a commitment R10 must not alter, re-declare or measure
early.

### 2026-08-22 - R9.5: `sector_cohort_divergence`, at SHADOW and stopping there

`IMPLEMENTED` + `GREEN` + `SHADOW`. It has no production authority and is not
proposed for any.

- **Why it exists.** On 2026-08-21, 25 of 26 electric utilities closed below
  their open (mean −2.78%, XLU −2.57%) while SPY closed −0.05%. AEP at −4.13%
  was the second-worst member of that cohort, and no surface on the desk ever
  named the sector. The archetype scan that found AEP is worthless without this:
  strip the utilities out of that session and it lost money.
- **The golden fixture was frozen FIRST** (plan.md sec 5), by
  `scripts/build_sector_cohort_fixture.py`, which refuses to overwrite without
  `--force` - a golden file that regenerates on a whim is not golden. Five
  hand-constructed cases, one rule each: fires short, fires long, a
  two-qualifying-bars near miss, never reaches the threshold, and one violent
  bar that reverses. It satisfies the repo-wide Milestone 3 fixture contract.
  **The fixture caught a defect in itself before the detector existed:** the
  first draft let `path_pct[0]` be non-zero while claiming each entry was a move
  from the session open, which silently re-based every series and turned the
  gap-down case into a gap-up one. The generator now asserts `path_pct[0] == 0.0`.
- **The rule, verbatim from the review's §6e.** `spread = ETF move from session
  open − SPY move from session open` on every **completed** M5 bar; fire when
  `|spread| >= 0.75%` has persisted across **>= 3 consecutive** completed bars.
  Session only, re-derived and never carried. An unknown sector excludes. No
  benchmark means no observation - a bare ETF move is not a divergence.
  `member_entry()` reuses the archetype through
  `chart_snapshot.session_vwap_series` rather than restating it, and reads only
  the session so far.
- **Gates.** 1: `CONFIG_VERSION` + a stable `config_hash()` that excludes
  `enabled`, so turning the watch off is an operational act and not a different
  engine. 3: a coverage row on **every** run, including quiet ones - without it
  a calm market and a dead collector look identical in the log. 7:
  `SECTOR_COHORT_DEFAULTS["enabled"]` ships **False**. Gates 2, 4, 5, 6 and 8
  are unmet and are not addressable by building it.
- **Zero IB traffic.** Batched yfinance over the ETF set, the M5 Strength Board's
  template, behind a single-flight lock with an injected fetcher so the vendor
  stays one seam wide.
- **First real day written 2026-08-22** over the 2026-08-21 session:
  `diagnostics/shadow_evidence/sector_cohort/sector_cohort_shadow.jsonl`,
  20 ETFs measured, 78 benchmark bars, 1,560 bars consumed, 11 cohort
  observations, XLU short from 10:35 ET.
- **It reaches no live surface**, pinned by an AST test rather than a substring
  scan - the module's own docstring names those surfaces in order to promise it
  avoids them, and a grep cannot tell a promise from a call.

### 2026-08-22 - R9.4: `thetalongs.txt`, so a wheeled name is actually evaluated

`IMPLEMENTED` + `GREEN`; one live gate owed (DRAM reaching the theta report on
a real scan, or being honestly absent for a stated rule reason).

- **The defect.** `evaluate_theta_put_candidate` returns `None` unless
  `side == "LONG"`, and `side` was long-watchlist membership. A wheeled
  underlying on neither trend list was therefore never evaluated at all - which
  is how the 2026-07-24..08-21 window's entire positive P&L (+$1,087.72, four
  DRAM short puts) stayed invisible to the engine built to find exactly that
  trade.
- **`thetalongs.txt`** is a new **optional** home-folder list beside the other
  watchlists (`THETA_LONGS_FILE`, `load_theta_long_symbols()`). Absent is the
  normal state and returns `[]`; **unreadable also returns `[]` with a warning**,
  so a locked file costs those names and never the whole Master AVWAP run. The
  trader owns it and nothing auto-removes from it (plan.md sec 5).
- **`resolve_scan_sides()` is the entire seam.** `side` is unchanged from list
  membership and is what every detector sees; `theta_side` is LONG for anything
  on the list **regardless of long/short membership**, and only the two premium
  evaluations receive it. A name can be a legitimate short thesis on the daily
  and still be one the trader will sell puts against.
- A symbol reachable only through `thetalongs.txt` resolves to **LONG**, not to
  a phantom SHORT - that list is a long-side list, and defaulting it the other
  way would hand every other detector a bearish thesis on a name the trader is
  bullish on. Its names join the scanned `symbols` set (a name on no list is
  never scanned, so it could never be evaluated) but deliberately **not**
  `longs`.
- **Provenance.** Rows carry `theta_list_source`, and the report prints
  `| via thetalongs.txt` on them; the rules header names the list too. A short
  thesis appearing in a LONG-only sold-put section otherwise reads as a bug.
- The home-folder `thetalongs.txt` was created with DRAM in it. The locked IB
  pacing budget is untouched - this is one extra symbol.

### 2026-08-22 - R9.3: the setup scoreboard, rebuilt from the stores that carry outcomes

`IMPLEMENTED` + `GREEN`. Read-only analysis; no live gate, and it promotes and
demotes nothing.

- **`scripts/setup_scoreboard.py`** reads `intraday_bounce_outcomes.csv` finals
  and `setup_playbook_episodes.csv` with `chunksize`/`usecols` (the first is
  ~200 MB and is never loaded whole), lifts `market_environment`, `session_rvol`,
  sector, industry and the RRS triple out of `context_json`, and takes the bounce
  type from the tail of `event_id`. Output:
  `docs/analysis/SETUP_SCOREBOARD_2026-08-21.md`.
- **The regime axis was never starved - it was in a different file.** The
  2026-08-21 review reported it at n=130 because it read the review store. The
  outcome store carries `market_environment` on 100% of its in-window rows:
  **5,608 usable rows across 5 environments**, plus RVOL and sector splits on the
  same rows, and a real stop and therefore a real R.
- **The 16.9% `close_r == 0` mass is a writer defect, not a scratch population.**
  Every one of the 1,164 in-window finals with `close_r` exactly 0 has
  `eod_close` **exactly** equal to `entry_price`; **none** of the 5,743 settled
  finals does. A real close does not land on the entry to the cent 1,164 times
  and never otherwise - the writer defaults `eod_close` to the entry when it
  cannot read one. 251 never advanced a bar. 563 are stopped-out trades that
  should score about −1R and score 0 instead, so treating them as scratches
  biases every mean **upward**. They are excluded and counted, never averaged.
- **Three statistics per cell, never one.** Trimmed mean (10%), median and plain
  mean with the stop-out rate beside them - a plain mean on a ratio with an
  unbounded numerator is what produced the review's phantom −1.82R. A 0.1%-of-entry
  risk floor removes 212 penny-stop rows; an unmeasurable risk excludes rather
  than passes. Cells are ranked only at **n ≥ 30**; thinner ones are printed
  marked `reportable = False` so the thinness is visible.
- **The swing block is measured against its own control** (`baseline_every5`)
  and carries an explicit guard: the control's trimmed R is −0.573, so a positive
  lift means *lost less than the control*, not *made money*. Most families' median
  `net_r` is about −1.0 - more than half of every family's episodes are full
  stop-outs - and the plain mean sits far above the trimmed mean nearly
  everywhere, so what positive numbers exist are carried by a thin tail.
- **The report ends by declaring the next window before it is measured** - 40
  sessions spanning bullish, bearish and chop, with the exclusions fixed in
  advance. That declaration is the only route by which anything in this file ever
  becomes plan.md §7 gate-2 eligible; everything above it is post-hoc and cannot
  move a rung.

### 2026-08-22 - R9.2: the LIKE asks why, and stops parking the symbol

`IMPLEMENTED` + `GREEN`; one live gate owed (a desk session where a liked
symbol is seen to keep alerting).

- **The why is required.** The claim key or a double-click now selects the setup
  and moves focus to the why field; Enter commits; an empty or whitespace-only
  why does not commit and the chart stays. Trader, 2026-08-22: "if I like a
  chart I should always be prompted with why". It is required rather than
  offered because the `dislike` rows are the counter-example - 31 of the most
  information-dense strings in the store, captured under a field nothing
  insisted on. The why lands in the row's existing `note`; no schema change.
- **A LIKE advances the queue and does nothing else.** It used to route through
  the Alert Center's "Not today" verb, which retires the chart *and parks the
  symbol for the day*. Measured over 2026-07-24..08-21: 40 of 52 `like_claim`
  rows put their symbol on `alert_center_ignored_symbols.txt`; a parked symbol
  also stops emitting `d1EventRecorded`, so on an AWAY day a LIKE silently
  dropped that name from the hourly D1 phone push. On 2026-08-21 the trader
  liked AEP short at 10:37:15 ET - the best day trade of that week - and the
  system's response to recognising it was to file a research row and take the
  chart away.
- **New signal, new action, one owner each.**
  `AlertChartReview.likeAdvanceRequested` is separate from
  `removeTodayRequested`; `AlertCenterPanel._advance_after_like` records
  `like_advance` and advances the queue, touching neither `_ignored_symbols`,
  the symbol's other queued alerts, nor any auto-adopted Focus pick. **The
  veto's retire-and-park path is unchanged.** The rail still places nothing -
  the explicit Focus verb remains the one thing that does.
- **`review_learning.TAKE_ACTIONS` gained `like_advance`.** Because the old
  route wrote `remove_today`, `REJECT_ACTIONS` had been scoring every like the
  trader ever filed as a dismissal - the strongest positive signal in the store,
  read as its opposite, on 40 of the window's 52 rows.
- `SymbolSnapshotDialog` already only advanced and needed no change; it inherits
  the required why through the shared rail.
- Four existing tests pinned the superseded one-click/retire rule and were
  rewritten against the new one; `test_a_like_also_retires_the_chart` was
  deleted, since the behavior it protected is the behavior the trader reversed.

### 2026-08-22 - R9.1: the universe can no longer collapse silently

`IMPLEMENTED` + `GREEN`; one live gate owed (a real rebuild writing its row on
the desk).

- **`build_universe` now has a write floor**, not just a zero guard. It refuses
  to overwrite the universe lists when the new `all` count falls below
  `max(500, 50% of the existing universe_all.txt count)`. The defect it closes:
  on 2026-08-20 13:31-13:35 PT a rebuild that priced roughly a quarter of the
  listing replaced a 1,487-name universe with a few hundred, and the D1 scanner
  ran 409-533 symbols for the whole of 2026-08-21 against its usual 1,088-1,513.
  AEP, a -4.1% short that session, was outside the universe on every in-session
  run. The absolute floor matters as much as the fraction: half of a universe
  that has already collapsed once is still a collapse.
- **Unreadable fails OPEN.** A missing, empty or unreadable prior universe
  returns a floor of 0 and the write proceeds. Refusing because the old file
  could not be *measured* would leave the desk with no universe at all, which is
  strictly worse than the partial one on offer. `_read_universe_count` raises on
  an unreadable file rather than reporting zero, so "unmeasurable" and "empty"
  are never conflated (plan.md sec 5).
- **`force=True` is the manual carve-out**, exactly as it is on the quiet-hours
  gate, and it is wired at both manual entry points: the Universe tab's Build
  button always forces, and `rebuild_universe_if_stale` forwards its existing
  `force`, so "Rebuild universe now" overrides a floor refusal while the
  scheduled stale tick cannot. `--force` on the CLI does the same. **It never
  carves out the zero-symbol refusal** - nothing makes writing an empty universe
  correct.
- **Every write attempt is now recorded.** `_record_universe_rebuild` appends a
  `universe_rebuild` row to `job_ledger.jsonl` with per-list before/after counts,
  the computed floor, and `refused` / `forced` flags, whether or not the write
  was allowed. The 2026-08-20 collapse left no trace anywhere on disk and had to
  be reconstructed afterwards from provider counters. The row is deliberately
  **keyless**: `JobLedger._replay` only reduces events carrying a `key`, so this
  is durable evidence in the same stream without inventing a phantom QUEUED job
  for `operations_audit` to report on. Writing it is best effort and can never be
  the reason a rebuild fails.
- **The outgoing lists are snapshotted** under a run-scoped
  `machine_cache/universe/snapshots/universe-<stamp>/` before each overwrite,
  bounded to the last 10 runs - recovery, not just detection.

### 2026-08-21 (seventh pass) - GUI fluidity: the hitching, measured and cut

`IMPLEMENTED` + `GREEN`; a live session is owed to confirm the numbers move.

- **The server was ruled out with measurements**, which is what the trader
  asked: every hot path is local, the GUI never reads the research store, the
  DAS was momentarily unreachable (and resolved again the same afternoon), and a miss on it costs 0.0 ms.
- **Per-widget stylesheets removed from both busy lists.** `AlertFeedItem` went
  from seven `setStyleSheet` calls per row to none; its variants are `theme.qss`
  rules selected by object name and an `alertKind`/`focusOn` property, backed by
  pre-mixed rgba tokens in `theme._derived_tokens`. Measured: 250 rows built in
  282 ms before, 167 ms after.
- **`FocusSideEditor.refresh` diffs instead of rebuilding.** Chips are reused,
  only arrivals are constructed, only departures are destroyed, and
  `FocusStatusChip.update_state` re-styles only when the accent actually moves.
- **`ChartDataService.cached_bar_dicts`** memoizes `as_bar_dicts` against the
  series object (LRU-bounded), so the Alert Center's D1 polls stop materializing
  ~490 dicts per symbol per tick on Qt - which that function's own docstring
  already forbade.
- **`_load_local_settings` is mtime-cached** (100 read/parse call sites; 100
  reads went 9.6 ms -> 0.7 ms) and **`load_review_events` is stamp-cached**
  (5.8 MB, 8809 rows; 80.8 ms -> 7.7 ms). Both hand out copies; both writers
  invalidate.
- **The `QFont` console flood is fixed at its cause.** The theme sizes fonts in
  px, so `pointSizeF()` is -1 and `setup_delegate`'s `+1.0` asked Qt for a
  zero-point font once per visible row per repaint. `_resized` scales in
  whichever unit the font carries; the favourite star, which was rendering at
  **1 point**, is right again.
- **`install_qt_message_rate_limit`** prints each distinct Qt message once,
  counts repeats and reports the tally at exit - never silencing a new one.
- **The stall watchdog now samples throughout a stall** and records the modal
  frame plus a `culprit_samples` histogram, instead of one stack captured at
  detection. 56% of stalls that could only be blamed on `app.exec()` should now
  name themselves.
- Test suite **4059 passed / 19 subtests**, exit 0; smoke 7/7; source selftest
  56/56.

### 2026-08-21 (sixth pass) - Prior-day break and session VWAP, on the same picks

`IMPLEMENTED` + `GREEN`; live gates owed (`plan.md` Phase 0.5 item 11).

- A regime-pause pick must now also have broken the PREVIOUS session's high
  (longs) or low (shorts) and be on the right side of session VWAP. That pair
  is the M5 Focus adoption gate, so `_sweep_regime_pause_bangers` CALLS
  `passes_focus_adoption_gate` rather than restating it.
- `regime_pause_hold.session_levels` reads the four numbers off the cached M5
  series: price and prior-session extremes from the bars, session VWAP from
  `chart_snapshot.session_vwap_series`. Completed bars only; every field
  independently optional, and UNKNOWN fails at the gate.
- The golden fixture grew to six cases per side, each isolating ONE reason to
  be kept or dropped, and each case is now two sessions because the new gate
  cannot be measured from one. Frozen before the change and re-frozen after;
  a test names which gate rejected which case.
- Three champion tests needed real fixtures rather than a change: their
  sessions had no prior day and their bars had **no volume at all**
  (`IbBar.volume` defaults to 0, so session VWAP was unmeasurable). Both are
  now what production actually sees.
- Measured on the day's real batch: longs 38 -> 18 and shorts 29 -> 18 across
  both gates. The prior-day half dropped 7 longs and 3 shorts; the VWAP half
  dropped none that day, which is expected - a name near its high is nearly
  always above its VWAP.
- Test suite **4032 passed / 19 subtests**, exit 0; smoke 7/7; source selftest
  56/56.

### 2026-08-21 (fifth pass) - The regime-pause gate, fixtures first

`IMPLEMENTED` + `GREEN`; live gates owed (`plan.md` Phase 0.5 item 11).

- Golden fixture `regime_pause_sweep_v1` frozen against the UNCHANGED detector
  first (plan.md sec 5). Four cases per side, each entering through a different
  branch of `still_trending or made_new_extreme or window_excess`; two are
  genuinely at their extreme and two are not.
- `_sweep_regime_pause_bangers` now also requires
  `regime_pause_hold.hold_state(...).holding`. Added, never substituted, so the
  flagged set can only shrink. The fixture's re-freeze records the diff: four
  flagged per side became two, and a test names which rows left and which
  branch each had used.
- **Being AT the extreme needs no ATR.** Three champion tests (12-bar sessions)
  caught the first version switching the detector off whenever an ATR(14) was
  unmeasurable - which is most of the first hour.
- The feed line carries a per-symbol measure instead of one batch phrase:
  `HTFL (new HOD), MRK (0.7 ATR)`. `ui/models/bounce.py` expands it per row and
  still reads a bare symbol as the old wording.
- `completed_bars` now reads attribute-shaped bars as well as dicts. BounceBot's
  cached series is `IbBar` objects, and a shared rule that only understood
  `bar.get` excluded every detector-side caller.
- Replayed against the day's real batch: **34% of longs and 28% of shorts drop**,
  including MRK (1.8 ATR off a 70-minute-old high) and GFS (1.3 ATR off).
- Test suite **4031 passed / 19 subtests**, exit 0; smoke 7/7; source selftest
  56/56.

### 2026-08-21 (fourth pass) - "Holding highs" is measured, in ATR, and expires

`IMPLEMENTED` + `GREEN`; the detector-side gate is NOT built (needs golden
fixtures - see `plan.md` Phase 0.5 item 11).

- `scripts/indicators/atr.py`: the shared Wilder ATR. The repo carried the rule
  twice already and the copies disagreed (`legacy._wilder_atr_last` is Wilder,
  `market_state._m5_atr` is a plain mean under the same name); neither was
  importable. Unmeasurable returns None, never 0.
- `scripts/regime_pause_hold.py`: how far price sits from its session extreme,
  in ATR, on completed bars only - plus the queue verdict. Tolerance 1.0 ATR,
  freshness 15 minutes from the later of the alert and the last new extreme.
  A level merely EQUALLED does not refresh the clock.
- The Alert Center expires stale regime-pause rows on its existing 30s chart
  tick and re-captions the ones it keeps with what is true now. Deletion is
  from the queue only; the alert list, the review-event stream (`hold_expired`)
  and the tracker rows are untouched.
- Measured on the day's own batch: 26% of flagged longs and 45% of shorts had
  an extreme more than 30 minutes old when they were captioned "holding
  highs"; MRK was 1.6 ATR off its high at fire time and 4.8 ATR off by the time
  it was read.
- Test suite **4020 passed / 19 subtests**, exit 0; smoke 7/7; source selftest
  56/56.

### 2026-08-21 (third pass) — The GUI garbage collector could be starved

`IMPLEMENTED` + `GREEN`; caught by the R6(c) diagnostic week on its first day.

- `d0aebd5` gated both GC sweeps on input idleness with no upper bound. With
  `gc.disable()` in force, that timer is the process's only collector, so a
  trader working continuously starved it completely: the desk reached **8 GB
  in 90 minutes** and froze for **298 s and then 200 s** in the sweeps that
  finally ran, releasing ~6 GB and returning to 1.9 GB.
- Both waits now carry a deadline in ticks — a young sweep waits at most ~10 s
  for quiet, a due full sweep at most ~3 min. Idleness still decides while
  inside the deadline, so the pause stays off clicks; at the deadline the
  sweep runs regardless. Activity may delay a sweep, never cancel one.
- Test suite **3992 passed / 19 subtests**, exit 0; smoke 7/7.

### 2026-08-21 (second pass) — Four trader-directed integrations

`IMPLEMENTED` + `GREEN`; four live gates owed (see `plan.md` Phase 0.5 item 10).

- **Veto vocabulary v3** adds "SMA incoming" (hotkey `0`) and changes nothing
  else. `canonical_veto_cohort` pools reasons whose definition is identical
  across versions, so the additive bump restarts no forward record and the
  eight cohorts the v1→v2 bump split are pooled again. Applied only when the
  performance rollup is rebuilt; pick and outcome rows keep the version they
  were captured under and are never rewritten.
- **The like+claim rail** now offers the three post-earnings families and the
  2nd-dev breakout alongside Main swing, named by setup id. The nine
  main-swing digits are unchanged; extras continue on `0` then letters, and
  each row's label starts with its own key so type-search and the shortcut
  agree.
- **The M5 Strength Board page carries the RS/RW board** beneath it in a
  draggable splitter — the same widget the Alert Center tab uses, fed by a
  second connection to one `rrsSnapshotChanged` signal. Display only: no
  second fetch, no second chart widget, both halves charting through the
  page's existing `symbolActivated`.
- **Malformed candles can no longer paint over a chart.** `ui/bar_integrity.py`
  enforces `low <= open, close <= high`; a bar that breaks it draws dashed and
  clamped, stays out of the y-range, is counted in a corner note and is logged
  once with its provenance to `diagnostics/bad_bars.jsonl`. A well-formed bar
  whose range dwarfs its series is observed in the same log and never redrawn.
- Test suite **3987 passed / 19 subtests**, exit 0; smoke 7/7; source selftest
  56/56.

### 2026-08-21 — Source launch is the production launch

`OPERATIONS`; no code change.

- The 2026-08-20 GUI hang repair (`d0aebd5`) was committed but not built, and the
  desk launches `dist\TradingBotV3\TradingBotV3.exe` — so the fix did not reach
  the running desk until it was relaunched on 2026-08-21. Every `ui_stalls.jsonl`
  row before that relaunch is a pre-fix baseline, not R6(c) evidence.
- Rebuilding produced a valid bundle (`ui` 120 submodules, `timer_utils`
  collected) that **Windows Smart App Control refuses to execute** — the open item
  from 2026-08-19, now on the live launch path rather than only on the frozen
  gate. SAC verdicts are per file hash: the 2026-08-20 bundle ran, the 2026-08-21
  one does not.
- The desk therefore runs from source (`.venv\Scripts\python.exe launch_gui.py`)
  and that is now the production launch path; the frozen exe is a verification
  artifact until the trader resolves SAC by signing or disabling it.
- Added `trading_desk.cmd` and a Desktop shortcut so the source launch needs no
  editor or terminal open. Console starts minimized and keeps the desk's log.
- Observed and unrepaired: the desk did not complete shutdown — hung
  (`Responding = False`, 4.0 GB) for 90 s, then survived windowless until
  terminated. Recorded as a diagnostic-week finding.

### 2026-08-20 (seventh pass) — Evidence-led GUI hang repair

`IMPLEMENTED` + `GREEN`; R6(c) live diagnostic week active.

- Confirmed two Windows `AppHangB1` events and measured the principal Qt-thread
  blockers: the 63.88 MB / 370,109-row AVWAP history parse on a 3-second health
  tick (1.268 s warm) and GUI-originated Away report/operations audit work
  (0.540 s measured).
- The Master scanner publishes a signature-validated, current-session
  `master_avwap_active_events.json`; Bounce health consumes it on a
  single-flight worker and falls back to the historical CSV off-thread only.
- GUI-originated Away report/audit publication is backgrounded, coalesced and
  serialized with every existing worker through one writer lock. Verified
  publish semantics and hourly state advancement are unchanged.
- Alert Center D1 watches are memory-only on Qt. D1 freshness/parsing and
  earnings-anchor reads use the shared chart worker pool; cold data remains
  UNKNOWN for one poll rather than blocking interaction.
- Major 30/60-second timers have distinct first phases, and generation-2 GUI
  collection waits for two seconds of input idleness (young collection waits
  250 ms). Periodic cadence is unchanged after the phase offset.
- Enabled the existing bounded `ui_stall_watchdog` machine-locally at 50 ms for
  the owed R6(c) week. It begins on the next desk launch.
- No detector/scoring/alert-decision behavior changed. Gate: 3945 passed + 19
  subtests, clean exit; smoke 7/7; source selftest 56/56. No frozen rebuild
  trigger applies.

### 2026-08-20 (sixth pass) — The veto cohort is graded nightly

`IMPLEMENTED` + `GREEN`.

- **`scripts/ai_jobs/cohorts.py`** (new) + slot `veto_cohort_grading` appended
  to the overnight runner. `update_veto_cohort_outcomes` had **zero callers**
  since it shipped; this is the caller. Deterministic, no model. Sideless picks
  are counted and named, never graded as LONG. Re-runs change only
  `updated_at`. 45 picks → 44 graded on the desk's real data.
- **Vocabulary-aware cohort key**: `veto_cohort_source(code, vocab_version)`
  → `veto_v2_<code>`; an omitted version keeps the historical unversioned form
  so existing rows are never rewritten. Splits eight unchanged cohorts across
  the v1→v2 bump — recorded as a known cost in
  `docs/CHART_REVIEW_WORKSPACE_PLAN.md`.
- **`trader_judgement` evidence scope**, opt-in (absent from `DEFAULT_SCOPES`
  and `TICKER_BRIEF_SCOPES`), sources in funding order with the raw annotation
  log last, and two machine-written caveats carried as package data. Run on
  demand via `run_ai_jobs.py --scopes trader_judgement`.
- **Review-event freshness in `review_capture_audit`**: newest merged `ts` in
  the summary, staleness counted in sessions (holiday-aware) against a
  2-session threshold. Confirms the store is healthy — 8,077 decisions over 19
  sessions, newest today; the legacy file's 07-30 silence was the shards taking
  over by design.
- Confirmed by inspection: the forward-return metric is close-to-close only, so
  the known IBKR/Yahoo volume-unit defect does not reach cohort numbers.
- Gate: 3942 passed, 0 failed. Smoke 7/7, selftest 56/56.

### 2026-08-20 (fifth pass) — The chart popup accepts keyboard input again

`IMPLEMENTED` + `GREEN`. Bug fix.

- **`SymbolSnapshotDialog` dropped `Qt.WindowDoesNotAcceptFocus`.** The flag
  means "may never hold keyboard focus", not "do not steal focus", so every
  field in the Master AVWAP chart popup was untypeable — clicking worked,
  typing did nothing. Pre-existing since the dialog was written.
- The non-stealing intent is kept by `WA_ShowWithoutActivating` +
  `show()`/`raise_()` with no `activateWindow()`, which is what governs the
  popup's appearance rather than its whole lifetime.
- The test asserts the **flag**, not a keystroke: the offscreen platform does
  not enforce OS focus rules, so a typing test passes either way.
- Gate: 3902 passed, 1 pre-existing flake. Smoke 7/7, selftest 56/56.

### 2026-08-20 (fourth pass) — Earnings markers, projection, and veto vocabulary v2

`IMPLEMENTED` + `GREEN`.

- **`scripts/earnings_projection.py`** (new, pure): median-cadence projection of
  the next report. Needed because the earnings cache holds **no future dates at
  all** (1,885 symbols, zero forward dates). `OVERDUE_GRACE_DAYS` keeps a
  just-passed projection instead of skipping a quarter — the NVDA case, where
  the first draft reported November for a report landing that week.
- **Earnings ribbon on the D1 chart**: `E` glyphs on a reserved top rail with a
  dotted connector to their candle, plus a projection pinned to the viewport's
  top-right. The axis is **not** extended (the projection sits a median 48
  sessions out; drawing it in place cost ~40% of candle width). Headroom is
  reserved for every symbol so the price scale never depends on whether a name
  has an earnings date. Built on the chart-data worker beside the levels.
- **`veto_reasons_v2.json`**: "S/R cluttered" → "Compressed" as a NEW code, not
  a rename. v1 remains on disk and loadable so existing rows stay readable;
  every surviving code keeps its meaning and its digit.
- **Like + claim is a numbered picklist**, Main swing only for now — same shape
  and same digit-commits-it behaviour as the veto list.
- Gate: 3899 passed, 1 pre-existing flake. Smoke 7/7, selftest 56/56.

### 2026-08-20 (third pass) — The review pane stops wasting the monitor

`IMPLEMENTED` + `GREEN`. Layout and capture-verb changes, trader-authorized.

- **~1240px reclaimed on an idle pane.** Measured: with no alert charted,
  `AlertChartReview` gave a one-line title 346px, the setup line 346px, the arm
  bar 346px and the verb row 346px at 2000x1900. The snapshot holds the pane's
  only expanding stretch, so hiding it left four `Preferred` widgets to split
  the slack. An expanding `EmptyState` now holds the chart's slot when the
  chart is hidden, and the other rows are pinned to `Maximum` vertically.
- **Capture rail: ~900px single column → ~379px of columns.** Sections flow via
  `FlowLayout`; symbol and side share a line; the veto list is sized from the
  vocabulary so all nine reasons are visible instead of six-plus-scrollbar.
- **LIKE retires the chart** like a veto does, in both the Alert Center queue
  and the snapshot popup (`snapshot_review_advance`). **NOTE still holds it.**
- **Hypothetical stop removed from the rail** — the control only;
  `EVENT_HYPO_STOP` stays in the annotation schema so existing evidence rows
  remain readable.
- Gate: 3860 passed, 1 pre-existing flake. Smoke 7/7, selftest 56/56.

### 2026-08-20 (second pass) — Veto becomes a verb, the hotbuttons return, D1 gets volume

`IMPLEMENTED` + `GREEN`. Four trader-authorized changes from one message, plus
one **open defect found and deliberately not fixed** (see
`CURRENT_CHECKPOINT.md`).

- **A veto retires the chart.** "When I click veto it should just disappear as
  'not for today'." `AlertChartReview._on_captured` routes `EVENT_VETO` to the
  existing "Not today" path. LIKE / hypothetical stop / note still hold the
  chart deliberately.
- **"Veto D1 - but M5 today".** A second veto button for the case the trader
  named: a bad daily chart on a name still worth a day trade. **The rail places
  nothing** — it emits `vetoDayTradeRequested` and `AlertCenterPanel` performs
  the M5 Focus add, preserving one writer per store. Place-then-retire ordering
  is load-bearing; a failed placement still retires the chart because the veto
  is already on disk. The annotation row is an ordinary veto — no new field, no
  schema change. Known limitation recorded rather than hidden: the veto cohort
  study counts a day-traded name as vetoed.
- **The arm bar returns under the chart** with the M5 hotbuttons, the D1 event
  hotbuttons and the type-a-ticker box. `docked_controls` splits into
  `dock_arm_bar` / `dock_capture_rail`; only the rail stays on a tab. Measured
  at the alert column's 420px the rail is 697px and the bar 131px, so this
  keeps 84% of yesterday's reclaimed height. The Armed tab returns to being the
  cross-symbol inventory; the verb-row armed line hides when the bar is docked.
- **D1 volume underlay** (`VolumeItem` in `candle_chart.py`). Translucent
  columns in the bottom 18% of the price view — **not** a stacked sub-plot,
  which would have taken back a fifth of the candles this pane just fought for.
  The picture is recorded once in normalized space and `paint` maps it onto the
  current view, so a pan/zoom/log-flip is a transform rather than a re-render,
  and volume never votes on the price range. **No fetch and no IB request**:
  `chart_snapshot.load_d1_bars` already carries volume from the durable daily
  store. Nothing measurable draws nothing, rather than a flat row of zeros.
  10 new tests, all failing before.
- **OPEN DEFECT (not fixed, needs a trader decision):** the daily store mixes
  IBKR and Yahoo volume with no unit normalization — IBKR rows measured 150-200×
  low against yfinance on the same sessions, affecting ~17% of stored symbols.
  Because `calc_anchored_vwap_bands` is volume-weighted, this distorts D1
  anchored VWAP on affected names, so the fix is a recalibration governed by
  plan.md sec 5 and the ask-first rule, not a bug fix. Full evidence, scale and
  the four options are in `CURRENT_CHECKPOINT.md`.
- Gate: **3847 passed / 19 subtests, 1 failed** — a pre-existing full-suite
  flake (`test_stale_d1_tail_triggers_one_backfill_with_cooldown`) that fails
  identically on the pre-change tree. Smoke 7/7, selftest 56/56, both exit 0.

### 2026-08-20 — The charts get their pane back, and the wake alert gets a test

`IMPLEMENTED` + `GREEN`. Two trader-authorized presentation/delivery changes.
No detector, scoring, gating or alert *decision* code was touched.

- **Alert Center review pane: charts, then one row.** The pane stacked title →
  setup text → charts → a two-row arm bar → a ~600px capture rail → the verb
  row, in the desk's narrow alert column. Trader, with a screenshot: "I cannot
  see the charts at all… I am ok with them being tabbed where alerts/D1
  focus/RSRW board is and clicking into them."
  - `AlertChartReview` gains `docked_controls` (default `True`). Docked is the
    historical stack, which `SymbolSnapshotDialog` and the Chart Review
    workspace keep; the Alert Center passes `False`, and the two control docks
    are `setParent(None)`-detached for the host to adopt. **Placement is a host
    decision now** — the widget no longer dictates it.
  - `AlertCenterPanel` hosts `arm_bar` on its existing **Armed** tab, above the
    inventory it fills, and `capture_rail` on a new scrolled **Capture** tab.
    The arm bar joins the inventory rather than becoming a sixth tab because
    "Arm" and "Armed" a millimetre apart on one strip is a misclick waiting to
    happen, and arming is deliberate enough to cost a click. Between the charts
    and the tab strip there is now exactly one row: the verb row, which
    advances the review queue and must not cost a click.
  - **The rail's five-second/no-mouse contract survives the move.** A
    `QShortcut` bound inside a hidden tab page never fires, so Alt+V / Alt+K /
    Alt+S / Alt+N are rebound at **panel** scope
    (`WidgetWithChildrenShortcut`), each raising the Capture tab before handing
    off to the rail's own handler. `CaptureRail` gains
    `bind_action_shortcuts` (default `True`) and a public
    `action_shortcuts()`; the Alert Center's rail binds none of its own,
    because two live bindings for one sequence is an ambiguous shortcut and Qt
    fires **neither**. It is a rebinding of the rail's own handler list, not a
    second copy. The 1-9 veto digits (bound on the reason list) and every
    Enter-to-commit path are untouched.
  - Armed state stays legible with the tab closed: `armedSummaryChanged` feeds
    a count into the Armed tab title *and* an always-visible line
    (`armed_summary`) on the verb row, replacing the arm bar's own "Nothing
    armed" text that went onto the tab with it. `clear()` now also drops the
    armed level chips, which it had always been the only `set_armed_*` call to
    omit.
  - **CaptureRail semantics are untouched** — a re-parenting, not a behavior
    change. It still records annotation rows and still never mutes,
    suppresses, scores, gates, alerts or writes a watchlist. The movers-only
    filter, the repetition fold and the adoption gate were not touched.
  - 11 new tests in `tests/test_qt_alert_capture.py`; 10 of them fail against
    the previous commit, and the eleventh is a regression guard on the
    unchanged recorder path.

- **A wake alert you can verify.** Audit first: both EVENING-permitted senders
  already push at ntfy's maximum — the Focus/Research price alerts
  (`price_alert_service._notify`) and the SPY ±1% wake alarm
  (`AutopilotService._maybe_push_spy_alarm`), both `priority="urgent"`. The gap
  was that the channel **test** went out at `high`, so "does an urgent push
  break through iOS Sleep Focus" had never been answerable.
  - `PriceAlertService.test_push(urgent=True)` sends one `urgent` push whose
    message says what should have happened ("This should have sounded through
    Sleep Focus…"), behind a new **Test wake alert (urgent)** button beside the
    existing Test Push. Same fail-quiet contract: `send_push` never raises, an
    unconfigured topic is reported rather than logged as a delivery.
  - **No new sender.** Nothing schedules it and only that button calls it, so
    the phone push policy is unchanged (AWAY remains the only Auto mode that
    pushes routine output; the price alerts and the SPY alarm remain the two
    deliberate exceptions).
  - `docs/EVENING_MODE_RUNBOOK.md` gains a **Sleep breakthrough checklist**:
    ntfy has no Apple critical-alert entitlement, so urgent priority alone
    cannot override Sleep Focus — the app must be in iOS Settings ▸ Focus ▸
    Sleep ▸ Allowed Apps, the topic must not be "Deliver Quietly", and the
    trader verifies with the new button while Sleep Focus is ON. Device steps
    are marked to-be-confirmed-on-desk.

- **Repaired a clock-fragile test.** `test_it_reads_the_gates_predicate_over_the_desks_own_bars`
  built an 11:00 session while `_measure_mover_state` read the real wall clock,
  so a bar stamped 10:50 was in the FUTURE at 07:34, was discarded as
  incomplete, and the assertion read UNKNOWN. It failed on this branch before
  any change here. The fixture now pins the clock; the test measures the
  predicate instead of the time of day.

- Gate: **3838 passed / 19 subtests, 0 failed**; process exit `0xC0000409` (the
  known intermittent Qt-teardown crash, after the summary printed). Smoke 7/7,
  source selftest 56/56, both exit 0. No packaging trigger hit — no new
  dependency, asset, top-level package or dynamic import — so the frozen exe
  was deliberately not rebuilt.

### 2026-08-17 — The technical-integrity replay contract is pinned (R6b)

`IMPLEMENTED` + `GREEN`. Tests and fixtures only — no source file was edited.

- **`tests/fixtures/technical_integrity_replay_v1.json`** +
  **`tests/test_technical_integrity_replay.py`** (18 tests) characterize
  `_load_resolved_events`: the boot-time reconstruction of in-memory state from
  the append-only ledger. The pre-existing `technical_integrity_scoring_v1`
  fixture covers the scoring math and would have stayed green through a change
  that corrupted every field here.
- Pinned: started/resolved pairing; unresolved starts recovering into pending
  with append-time provenance (`as_of`, `written_at`) **stripped**; a resolved
  row suppressing a stale state-seeded pending entry; partial vs complete
  follow-up horizon chains; all four snapshot-marker types; the
  `(resolved_at, event_id)` sort tiebreak, written in reverse order so an
  unsorted read fails; a truncated mid-flush final line costing that line only;
  and **monolith-vs-segmented equivalence**.
- **Cross-session inertness is proven both ways.** Rows from a LATER session
  are included, because a prior session sorts below every current row and
  deleting the filter would leave the watermark unchanged — the assertion would
  have passed while measuring nothing. A positive control replays the identical
  bytes against the next session and gets the later watermark, so "the filter
  excludes those rows" and "that field is never reachable" cannot be confused.
- **Mutation-proven**: removing the session filter fails 7 tests; removing the
  provenance strip fails 3. `scripts/technical_integrity.py` was restored
  byte-identical after each check.
- This is what makes the per-session segmentation committed by the R6(b)
  decision checkable rather than aspirational when warehouse Phase-3 retention
  unlocks.
- Gate: 3623 passed / 19 subtests, exit 0.

### 2026-08-17 — The overnight AI layer becomes visible (R6a)

`IMPLEMENTED` + `GREEN`.

- **`ai_jobs` row in `operations_audit`** — read-only visibility for the batch
  layer, which runs as a scheduled task against the repo checkout and had no
  System Health row at all. It resolves the AI store by path (env override,
  then the local setting) and **never imports `ai_jobs`**: that package is in
  `PACKAGES_NOT_IN_THE_BUNDLE`, so an import would work in a checkout and die
  in the frozen exe that actually renders System Health. Four pins keep the
  duplicated rule honest — the env key, the setting key and the ledger filename
  are each asserted equal to the batch layer's own, and a source-level test
  forbids the import.
- An **unset store reports HEALTHY**, not UNKNOWN: UNKNOWN means "could not
  measure", and a machine with no `ai_store_dir` has been measured — the layer
  is off by choice. Failure outranks degradation; freshness is graded in days
  because the layer is nightly; a tail line killed mid-flush costs that line
  rather than the whole ledger.
- **`run_ai_jobs.ps1`** routine log line reworded: the scheduled task passes no
  arguments, so `(no arguments)` labelled the normal nightly run as though a
  caller had forgotten something.
- Gate: 3604 passed / 19 subtests, smoke 7/7, both exit 0.

### 2026-08-17 — R5's first live engine: the LRSI efficiency crossing

`IMPLEMENTED` + `GREEN`; **live gate open.**

- **`scripts/m5_signal_engines.py`** — the pure seam between the R5 indicator
  modules and the live detector. Bars in, events out, no clock and no I/O, so
  the three rules the call site historically got wrong are testable without
  standing up a scanner: completed bars only through the one shared helper; the
  indicator warms across sessions while the *event* belongs to one; and shorts
  mirror by negating price rather than inverting the test, because the
  efficiency oscillator clamps at zero and a falling name reads LOW, never
  negative. `latest_lrsi_cross` fires only on the most recently completed bar,
  so one crossing is one alert rather than one per scan cycle.
- **`check_lrsi_cross_setups`** in `bounce_bot_lib/legacy.py`, on both the
  human-focus fast lane and the full cycle, beside the ORB and 8-EMA grind
  sweeps. It passes an aware market-local clock straight through to
  `completed_bars` instead of stripping the offset with `replace(tzinfo=None)`,
  which is the defect that helper was extracted to end.
- **`M5_SIGNAL_TAG` family** (R5 §8.1), defined once in `m5_signal_engines`
  because the detector cannot import from the UI and both sides must agree. It
  replaces the `"green"`/`"red"` colour previously passed as the callback tag;
  direction has always come from the feedback block, which is now asserted. No
  D1 routing, no chart-watch or entry-assist bypass, ordinary tier gate.
- **`M5_SIGNAL_TYPE_DEFAULTS`**, a toggle map kept deliberately out of
  `BOUNCE_TYPE_DEFAULTS`. `BOUNCE_LEARNING_TYPE_KEYS` derives from that dict, so
  adding engines on probation would have widened what the learning path treats
  as an established bounce type — a scoring change smuggled in as a feature. A
  taxonomy pin now fails if the two are ever tidied together.
- **Packaging trigger fired and discharged in the same commit**: `indicators`
  moved out of `PACKAGES_NOT_IN_THE_BUNDLE` into the spec's
  `FIRST_PARTY_PACKAGES`, and four modules joined the selftest roster. The
  clean-cache frozen rebuild moved the count **51 → 55**; the movement is what
  proves the build was real.
- Gate: 3590 passed / 19 subtests, smoke 7/7, `selftest OK: 55/55 (frozen)`,
  all exit 0.
- **Owed:** R5 §7's per-engine desk session. The confluence and first-candle ORB
  engines are deliberately NOT wired until one runs.

### 2026-08-16 — One shared completed-bar rule, and R5's lane decision

`plan.md` sec 5 says completed bars only for state transitions and a forming bar
is preview. That rule had been written once and implemented three times.
`scripts/completed_bars.py` is now the single definition of the intraday case —
`bar_start + bar_minutes <= now`, inclusive at the boundary, timezone-converted
with `astimezone` and never `replace(tzinfo=None)`.

`weekend_strength._completed_intraday` already had it right, so its logic was
**moved** rather than rewritten and it now delegates. Only the intraday rule is
shared: the NYSE-last-completed-session and month-identity rules stay where they
were, because they answer different questions and folding them in would have
been a behaviour change dressed as a refactor. A characterization test pins the
weekend board's behaviour with frozen survivor counts and was verified to go red
against a mutated boundary.

BounceBot's ad-hoc call sites (`bounce_bot_lib/legacy.py:4384-4386, 4533-4535`)
are deliberately **not** migrated: R5 §5 says they move opportunistically, never
as a silent behaviour change to a shipped detector. Their `replace(tzinfo=None)`
spelling is recorded as the defect it is — it discards a stamp's offset instead
of converting through it, so a zone-carrying bar is judged against a wall-clock
number that never meant the same instant.

R5's Alert Center lane question was also answered (Fable, trader-delegated;
trader may override) and recorded in that spec's §8.1 before any wiring: **a new
`M5_SIGNAL_TAG` family**, not a reuse of `d1_flag`. Main feed, no tier-gate
bypass, not loud by default where the spec does not say so, and **not privileged**
against R4 §6.3 — an unproven engine must be foldable. Per-engine identity rides
`bounce_type` so each engine stays separately countable in the feed and History,
which is what §7's per-engine desk session needs.

One correction was folded into that record: Focus privilege rides the symbol
rather than the tag, as the decision assumed, but `_alert_has_focus_privilege`
is membership **and** an open prev-day break on the alert's own side — stricter
than membership alone. "Focus member ⇒ never folded" is not what the code does.

Deterministic gate: 3562 passed / 19 subtests, exit 0.

### 2026-08-16 — R5's pure indicator modules

`scripts/indicators/` gained three pure, offline, completed-bars-only modules:
TC2000-parity `smi.py`, the TC2000 "LRSI" efficiency oscillator as
`efficiency_lrsi.py`, and `heikin_ashi.py` with a reversal classifier. Immutable
tuples out, no provider call, no clock, no ledger, each with its own
`FEATURE_VERSION`.

`efficiency_lrsi.py` is named apart from the pre-existing `laguerre_rsi.py`
deliberately: that module is Ehlers' Laguerre RSI with fractal-energy
modulation, an unrelated algorithm, and the trader calls this one "LRSI" only
because TC2000 does. A test asserts the two cannot be confused. Of the two
possible readings of the trader's LRSI source, the 0–100 scale is used, because
the spec pins that range and its crossing levels (up through 20, up through 50)
only make sense there.

An unmeasurable bar is `None` throughout, never a fabricated zero — a warm-up
window, a mid-window gap, an SMI range of zero, an EMA that did not move.

**Nothing imports these yet, so no packaging trigger has fired**; they stand
where `laguerre_rsi.py` does. All R5 wiring is unbuilt and blocked on the spec's
§8 Alert Center lane question. Two other §8 questions were answered by the
trader the same day: the confluence alert stays M5-Focus-only, and an ORB
candidate is an Alert Center annotation rather than a strength-board lane.
Deterministic gate: 3542 passed / 19 subtests, exit 0.

### 2026-08-16 — R4 desk chart unification built

Packet R4 is built to its authorized scope. Every chart surface now carries the
same capture controls, the trader's own armed alarms are drawn on the chart, the
early-morning gap distortion is fixed at its source, already-checked names are
marked wherever they render, and the Alert Center's feed stops stacking rows for
one ticker all day.

`CaptureRail` moved into `SymbolSnapshotDialog` and `AlertChartReview`, so every
host that opens a chart — the setups table, the RS/RW board, the Industry panel,
a typed lookup, the Alert Center — inherits veto/like/note/hypothetical-stop
without knowing anything about it. The RS/RW board and Industry panel previously
passed no review host at all and had no capture whatsoever. Capture stays
analysis-only: it writes `trader_annotations.jsonl`, emits no placement signal,
creates no other file, and does not advance the review queue. The Alert Center's
"Add to Focus Picks" verb remains the single explicit placement action.

Armed price alerts and armed D1 level watches render as a new `GROUP_ALERTS`
paint-lines family, built on the `ChartDataService` worker and toggleable
through the existing paint-lines control. Read-only throughout: the family
builder opens and writes no file, and the single-writer rule on
`price_alerts.json` is untouched. A disarmed side paints nothing; a D1 *event*
watch paints nothing at all, because it is a condition with no armed price and
inventing one would draw a level the trader never chose.

The forming D1 preview no longer paints a Yahoo daily "today" row as a candle
during the first 15 minutes after the open (`chart_yahoo_forming_suppress_minutes`,
zero disables). That row is a thin early print which both mis-stated the gap and
drove the chart's y-autoscale — the axis the trader suspected was fine, the data
was not. When a Yahoo-sourced preview *is* drawn it now says so. The IB M5 path
is unchanged and always preferred.

The reviewed-today marker renders on the snapshot header, the Alert Center pane,
the RS/RW window and the Industry board, joining the setups table from R3. It
rides display text only — sort roles and row payloads are untouched, so it
cannot become a ranking.

The feed's Focus star became a labeled `☆ Like → M5 Focus` / `☆ Like → Swing
Focus` action with the same semantics and the same signal. New pure module
`scripts/alert_repetition.py` gives one live feed row per symbol + side + market
day: a repeat updates that row in place, keeps its first-seen time, gains a
repeat count, and stays silent unless it escalates on a strictly higher tier,
the first BANGER, or the first PROVEN. Ordinary alerts in the first 30 minutes
after the open group into one digest row naming each symbol. Focus-privileged,
trader-armed, entry-assist and ready-D1 output bypass both the fold and the
digest entirely.

Nothing in the repetition control withholds anything: the backing alert list is
written before any repetition decision and never consulted by one, so History,
the evidence streams and the AWAY push are unaffected, and the chart review
queue is enqueued independently. No detector, score or threshold changed, and no
suppression field exists anywhere in this chain. Every failure path falls open
to a plain new row.

Three trader confirmations were taken before §6.2/§6.3 code was written and are
now decisions: a 30-minute open digest, no reason prompt on a like, and an
exhaustive three-item escalation list.

Two items are explicitly held for a trader decision rather than skipped: the
Focus Picks reviewed-today marker (that surface is editable watchlist text, not
a table, and a marker injected there would land in data written back to the
watchlists), and §2.2's `review_host` for the boards (its remaining half is the
setups table's advance-to-next-row flow, which has no meaning on a ranked
board). Deterministic gate: 3500 passed / 19 subtests, exit 0. Every §8 live
proof remains owed.

### 2026-08-16 — R3 deterministic work closed; §4.3.5 deferred by the trader

Packet R3's build is complete to its authorized scope. The shadow-only
`would_demote` classifier, the D1 `relvol` field with its annotation-only
`daytrade_candidate` carve-out, the reviewed-today badge built from recorded
decisions, the 12:45 PT preview slot with actual-close ownership of the single
scheduled tracker write, the completed-bar STABLE list beside the live PREVIEW
list with `bar_status` stamps, and structured dislike reason codes counted by
`review_learning.py` as a `dislike_reason` dimension are all in place. Nothing
demotes, hides, reorders or suppresses a live row: the classifier stamps
evidence only, and `review_policy.json` still has no suppression field.

The one unbuilt item, §4.3.5 time-normalized volume thrust, is now a recorded
trader decision rather than an open question. The Master D1 scan fetches daily
bars only, so `rvol.same_slot_baseline` has no same-slot intraday series to read;
supplying one would mean a 5-minute fetch across ~1,100 symbols and a new data
contract inside the scanner. A zero-fetch alternative — prorating the 20-session
full-day average by session-elapsed fraction — was offered and rejected, because
real intraday volume is U-shaped and a flat-profile proration would over-fire
mid-day and under-fire at the open and close. The trader deferred the change on
2026-08-16 pending a week of stamped evidence. The 18-pt thrust bonus therefore
keeps its full-day baseline as a known, accepted pre-close honesty gap, frozen in
`tests/fixtures/r3_swing_quality_v1.json`. Documentation only; no code changed in
this reconciliation.

R3's live gates remain owed and UNKNOWN: the `would_demote` shadow week required
before any row moves, the one-week 12:45-vs-close and STABLE-vs-PREVIEW churn
comparison, and the scoreboard's first real-data curation cycle.

### 2026-08-16 — "Not today" preserves trader-armed alerts

Alert Center dismissal no longer deletes trader-armed chart watches or defers
trader-armed persistent D1 level/event watches. A fired chart, armed-level, or
D1-event watch carries `CHART_WATCH_TAG`, bypasses the ignored-symbol feed filter,
renders in the live feed, and sounds. The automatic Focus-derived D1-interest
path remains separate: ignored Focus symbols are still skipped and
`FOCUS_D1_EVENT_TAG` receives no exemption. Producer tracing confirms the two
persistent watch stores are UI-armed/UI-persisted only. Deterministic evidence:
80 focused Alert Center/arm/watch tests and the full 3377-test suite plus 19
subtests pass; live confirmation remains owed.

### 2026-08-16 — R4 Alert Center contract recovered (documentation)

The historical P1.6 Alert Center quality packet was recovered byte-for-byte from
commit `671ee57`, read against the built R2 contract and active R4 spec, and
classified as historical evidence. R2 already owns the auto-pick provenance,
scoped removal, and `not_today`-rather-than-dislike outcomes. The active R4 spec
now retains the unabsorbed trader outcomes: "Not today" never cancels or mutes a
trader-armed alert; the feed's Focus star becomes a labeled Like-to-Focus action;
and repeated feed rows/open bursts are controlled on the presentation side only.
No alert behavior changed in that documentation recovery commit. Source inspection
found two additional ignored-symbol deferrals in the D1 level/event watch pollers;
the trader subsequently expanded the exact fenced-file approval to include them,
with the automatic-Focus carve-out and both-direction seam tests recorded above.

### 2026-08-16 — R7 journal pre-flight fix pass

The release-candidate spot-check closed five journal correctness gaps before any
further packet work. Id-less Questrade fills now derive identity only from order,
normalized symbol, timestamp, side, quantity and price, so fee or raw-payload
drift updates the same row. The v2→v3 migration recognizes Questrade rows whose
legacy uid used an order id, re-keys each partial fill to that same stable hash
before collapse, and reports the affected group/row counts. Fresh app launches
now read the persisted schema version instead of process-local store state; an
already-v3 journal opens directly, while the nightly slot refuses to migrate an
existing pre-v3 journal before the trader-present GUI preparation. Rebuild input
is sorted by normalized position identity and time, including mixed option-symbol
spellings, and a failed Questrade activities cross-check now propagates FAILED to
the run. The live journal and brokers were not touched.

### 2026-08-16 — R3 tracker ownership moves to the close

The swing schedule now adds an explicit close-minus-15-minute preview (12:45 PT
on a normal session). That slot never writes the setup tracker. Both scheduling
and the scanner's wall-clock fallback gate now begin at the actual market close,
leaving the 13:00 PT close slot as the sole ordinary tracker writer; later manual
runs and the existing completed-session catch-up remain recovery paths. The R3
fixture preserves the former 12:00+13:00 behavior and records the intentional
difference. This milestone does not yet claim the completed-bar STABLE report or
any live comparison proof.

### 2026-08-16 — R3 exposes completed STABLE beside live PREVIEW

Master AVWAP now performs a presentation-only second pass over each symbol's
already-fetched D1 frame, truncating it to the latest completed date through the
existing historical snapshot evaluator and daily-only ranking stages. The report
puts that STABLE Best Swing list immediately before the independent live PREVIEW
list. Priority, focus, D1-feature and new tracker records carry explicit
`bar_status` and presentation-mode fields; report rows print the status. No second
broker/HTF fetch, tracker mutation, watchlist change, alert change, or live-row
ranking change is introduced. The same-slot volume-thrust scoring change remains
unbuilt because this seam has no intraday slot series. Deterministic gate: 3367
passed / 19 subtests, exit 0; live churn comparison remains owed.

### 2026-08-16 — R3 setup dislikes become counted evidence

The Setups ✕ dialog now offers the existing versioned veto vocabulary, then an
optional detail prompt (required for `other`). Review-event rows store the
permanent code(s) and vocabulary version; the review-learning scoreboard counts a
new `dislike_reason` dimension. A day/signature-cached union of explicit decisions
across pick feedback, alert review events, and trader annotations supplies an
additive "Reviewed today" badge in the Setups table and a matching report group.
Impressions and hypothetical stops do not count. This is presentation/advisory
evidence only: no rank, score, filter, alert, or suppression field consumes it.
Deterministic gate: 3370 passed / 19 subtests, exit 0; first live curation remains
owed.

### Application, runtime, and data ownership

- PySide6 Trading Desk launched by `launch_gui.py`, with the legacy Tk UI retained
  as a compatibility path.
- Main-desk single-process ownership, bounded BounceBot startup/shutdown, generation
  guards, child-process reaping, runtime heartbeat, durable job ledger, typed retry
  budgets, stale-run marking, and a hardened single-instance launch guard that also
  sees the frozen executable.
- User-selected shared home folder for operational text/JSONL/CSV artifacts;
  machine-local settings, caches, and diagnostics under LocalAppData; a separate
  research-lake storage class outside that home folder.
- **No cloud sync (2026-08-10, decision 0015).** Google Drive/OneDrive were removed
  from the system entirely. `C:\TradingBotData` keeps its path and role as a plain
  local folder; the DAS file server `\\MINI-PC\Trading Bot Data` is the durable
  tier, holding the research lake, the AI store, and hourly cold-pushed subtrees.
  Documentation-only change: no path, behavior, or test changed.
- Designated-writer authority, local kernel exclusion, fenced writer lease, atomic
  publication, readback verification, last-good preservation, and bounded archives.
- Main desk is the sole always-on scanner. The former mini-PC scanner and Desk Link
  satellite topology are `RETIRED`; their code remains only pending cleanup.

### Scanning, candidates, and decision support

- Master AVWAP D1 swing scanning with earnings anchors, current/previous AVWAP
  families, running-deviation bands, focus buckets, Expected-R ranking, study tags,
  theta candidates, tracker history, and durable daily-bar storage.
- BounceBot completed-M5 detection with session VWAP/bands, EMA and prior-day
  levels, relative strength/weakness, regime-aware candidate discovery, tiering,
  alerts, outcome tracking, and the day-scoped M5 Focus path.
- BounceBot's sweep runs only inside the session window (open-30m to close+30m by
  default, weekdays); outside it Auto Pilot pauses scanning and holds the IB
  connection open. A manual resume survives until the next boundary.
- CandidateRegistry foundation with provenance, source leases, transitions, atomic
  versioned persistence, and partial shadow adoption. Full authority remains open.
- Industry Board with one single-flight owner, hourly refresh, atomic last-good
  snapshot, numeric sorting, freshness/Health integration, and advisory aligned
  industry-vs-SPY plus stock-vs-primary-industry fields.
- Auto-populate rules for both regimes, previous-day-extreme gating, DESK adoption
  into M5 Focus, and one extension notification per Focus name/day while pullback
  notifications stay active.
- Focus privileges begin only beyond the previous session's directional extreme;
  missing prior-day data grants nothing.
- D1 Focus routes final Favorite/High Conviction upgrades while developing trigger
  evidence remains research-only. Legacy D1 champion alerts are unchanged.

### Charts, review, alerts, and phone surfaces

- Chart-first review flow, current forming D1 preview, D1/M5 shared snapshot widget,
  log scale, crosshair/OHLCV readout, source/age strip, fallback warning, cache
  invalidation, background loading, prewarming, and stall watchdog.
- Chart Review workspace with lookup for any symbol, hidden-by-default Setups drawer,
  keyboard-first LIKE/veto/note/setup-claim capture, versioned veto vocabulary,
  append-only `trader_annotations.jsonl`, and isolated forward veto cohorts.
- Painted D1 S/R, previous-day H/L, projected trendline, SMA/EMA/AVWAP groups,
  machine-local visibility preferences, stable level IDs, click selection, and
  click-to-arm routed through the one `PriceAlertService` writer.
- Chart Review annotations cannot add Focus/watchlist membership or price alerts;
  LIKE records judgement only.
- Visual Alert Center and review queue, chart-armed watches, persistent History,
  structured review decisions, review scoreboard, and annotation-only/FIFO policy
  gate.
- Main-only price-level polling with cross-up/cross-down, one fire per arm, urgent
  ntfy push, persistent main-desk presentation, and manual re-arm.
- Auto modes OFF/DESK/AWAY/EVENING, honest global status, EVENING early scan and
  briefing, and one verified `autopilot_today.txt` with safety/freshness first,
  numbered best swings, intraday candidates, and condensed operations.
- The double-click symbol snapshot popup opens at desk height (2026-08-11): its size
  is taken from the hosting window's frame, or the screen's available area when the
  window is not yet measurable, never smaller than the former fixed 1180x760, and is
  centered on the desk window and clamped inside the screen. Opening geometry only —
  a trader resize survives subsequent double-clicks.
- On 2026-08-10, best swings gained an ntfy report notification; it stays quiet when
  the generated swing section contains no readable setups. Late-opened alerts now
  receive current bars, and the Chart Review Setups column defaults hidden with a
  visible restore control.
- Phone push policy, 2026-08-11 (trader rule): **AWAY is the only mode that pushes**,
  and the Research/Focus price alerts are the single deliberate exception — they keep
  their own always-on urgent channel, unchanged. The EVENING morning-briefing push and
  the retired Desk Link control-reclaim push are now silent outside AWAY; both still
  announce on the desk. The hourly swing push carries the **full favorite and
  high-conviction roster** under the ranked picks, built from the whole current feed
  rather than the top-ten slice, side-split, with `near` excluded and an explicit
  "did not fit" marker if the message ever exceeds the ntfy size ceiling; a roster with
  no ranked picks still sends. A **second hourly push names every stock that fired a D1
  level or event alert since the previous one** (armed D1 levels, D1 event watches,
  Focus D1 flags, and the scanner's ready D1 focus alerts), new-since-last-push rather
  than cumulative, silent on an empty hour, and cleared only on a delivered push so an
  ntfy failure never eats the events. The Alert Center classifies (it owns the D1
  routing rules) and Auto Pilot aggregates and gates, so the phone and the D1 Focus
  feed cannot disagree. Machine-local kill switches: `push_away_swings`,
  `push_away_d1_events`. **Extended 2026-08-14 (packet R1):** EVENING's SPY ±1%
  wake alarm is the *second* deliberate exception — urgent, repeating every five
  minutes while the move holds, stopping on the flip out of EVENING, kill switch
  `push_evening_spy_alarm`.
- Auto-mode matrix, 2026-08-14 (packet R1): discovery is identical in every mode;
  what differs is who is present to act. DESK adopts staged picks immediately;
  AWAY stages and never adopts and queues alerts silently (only the sound is
  suppressed); EVENING runs its early block and then stops scanning entirely,
  staging picks for the wake-up flip; OFF is the only mode that still
  self-applies. Quiet hours confine every automatic starter to weekdays,
  06:00–14:00 local; manual buttons are never gated.
- One Master AVWAP scan action, 2026-08-15 (packet R1). The Shared/Local pair read
  the identical two watchlist files, so `use_shared_watchlists` and the menu choice
  it drove were removed across thirteen files. Cloud-drive *store discovery* went
  with it (decision 0015 amendment); the mount-presence guard stays.

### Journal, explanations, and learning

- **R7/R8 adversarial release-candidate repair (2026-08-15).** Every verified
  A1–A19 and B1–B14 finding was closed before handoff. The repair normalizes
  broker-ledger casing and Flex dates; preserves shared Focus wiring and exact
  suggestion-row identity; scopes reconciliation clears to reachable brokers;
  bounds shutdown; migrates every execution leg and gives fills stable
  identities; makes coverage, quarantine, currency, FX ordering, token
  precedence, weekly identity, exit-window, empty-last-good, and OCC handling
  fail honestly; and restores the journal's missing pull/gap/retry controls,
  grouped tags and filters, reversible undo, atomic exports, and truthful labels.
  Expensive journal work now runs in a worker and re-renders from captured
  structured results without re-querying; migration starts only after an
  explicit **Prepare Journal database** click and remains visibly gated in the
  background. Weekend rollover, timezone conversion, failed-discovery state,
  Flex reuse, single-fetch boards, board persistence, and failure signaling are
  likewise pinned by regression tests. Account tax labels moved out of source
  into machine-local settings. No live journal database or broker was touched.

  Scope reconciliation is explicit: true non-USD-to-USD conversion, the
  Calendar year heatmap, additional Analytics charts, Weekend RRS-strength
  joins, and Weekend Focus performance/pick-feedback/veto joins remain deferred
  in their governing specs. They are not represented as shipped behavior. The
  repaired code tip is `dd201cd`; deterministic baseline is 3354 passed / 19
  subtests, smoke 7/7, frozen selftest 49/49, all exit 0. Live gates remain owed.

- **Weekend Prep (R8, 2026-08-15).** A guided five-step weekend routine with
  persisted progress: week in review, focus-pick review, week-windowed walk-away
  with the weekly auto-tag review, strength discovery on H1/D1/Monthly using the
  M5 formula through the fenced `strength_scan` functions, and the week-ahead
  prep from the `market_prep` weekly engine. Manual refresh only, zero IB
  traffic, adds-only adoption into swing Focus.

- **Tax-grade journal (R7, 2026-08-15).** Stable `BROKER:account:exec_id`
  execution identity; one security-type vocabulary across both brokers; anchored
  `trade_id` with an annotation re-key pass and `trade_aliases`;
  `CLOSED_PARTIAL` and a `SYNTHETIC_OPEN` marker instead of a fabricated inverse
  position; append-only `trade_adjustments` corrections re-applied at every
  rebuild; an `import_coverage` ledger with a bounded nightly self-heal; IBKR
  Flex as the primary history source including OptionEAE, OpenPositions and
  CashTransactions; Questrade activities and a trade-day cross-check; Bank of
  Canada FX booked once per (date, currency); reconciliation against both
  brokers' reported positions with trader-confirmed force-closes; a nightly
  `journal_import` slot at the front of the `ai_jobs` slate; and a five-tab
  Journal (Trades, Calendar, Analytics, Health, Fees) over one shared
  tax-grouped header.

- Journal schema v2 with append-only opportunity lifecycle events, idempotent broker
  Taken/Closed imports, structured reviews, free-form notes, tags, and analytics.
- Deterministic novice explanations across Setup Tracker, Day Trade Tracker, and
  Move Forensics, plus an evidence-floor-aware “What’s Working” summary.
- Review events partitioned by installation, merged/deduplicated by readers, capture
  audits, preference scoreboard, AI-curated `review_policy.json`, and a permanent
  no-suppression boundary.
- Technical Integrity research hierarchy with point-in-time predictions/outcomes,
  break pressure, calibration report, and no detector/watchlist/alert influence.
- Regime infrastructure evidence for SPY baseline, breadth, Technical Integrity
  follow-ups, and audit tooling. The evidence remains exploratory/non-promotable.

### AI and automation

- Provider-neutral A.I. Summary workspace for OpenAI and Anthropic, explicit evidence
  selection, bounded preview, credential-manager storage, structured/source
  validation, immutable evidence packages, and export-only results.
- Config-gated local OpenAI-compatible provider through Ollama, default off; small and
  medium model tiers verified on the Ryzen main desk with no market-hours inference.
  The local large tier is `RETIRED` (2026-08-10): 27B-class models no longer load
  beside the running desk on the 780M, so its jobs belong to the frontier model. Local
  calls are capped to the tier's context window and fail loudly on server-side prompt
  truncation.
- Separate off-hours `ai_jobs` process and scheduled task, job-ledger integration,
  deterministic evidence coverage, daily advisory summary, per-ticker briefs, full
  artifacts in `ai_store`, and bounded atomic `ai_morning_brief.txt` publication.
- Per-ticker briefs project each symbol out of a full-size base package and then
  ration the projection to the local context window; ticker-roster and bare-name
  lines are discarded as non-evidence, each symbol resolves independently, a symbol
  with no evidence beyond watchlist membership is answered without a model call,
  completions resume on a read-stamp-independent evidence key, the morning file is
  republished after every resolved symbol, and the slot spends at most three attempts
  a session.
- Local-AI Phase 0 is complete. Phase 1 implementation is complete; its five-session
  unattended live gate remains in `plan.md`.

### Durability and catch-up

- Repeating 06:00 Pacific weekday launch task through the session, protected by the
  existing single-instance guard.
- Master AVWAP tracker staleness catch-up from completed prior-session D1 data with
  explicit `data_session` vintage and no automatic scoring-tuner/prior-refit side
  effects.
- Technical Integrity follow-up and breadth-ledger deterministic backfill with
  bounded retries, explicit `capture_mode`, honest gap rows, and live/backfill audit
  separation.
- Frozen snapshots, never-started predictions, and other Tier-C evidence remain
  intentionally non-reconstructed.

### Research warehouse

- Phase 0: research-lake decision record, configuration, home-folder-path refusal, layout,
  and disabled-by-default no-op behavior.
- Phase 1: immutable Parquet store, four-step seal, append-only manifest authority,
  13 frozen schemas, quarantine, compaction, retirement, and crash reconciliation.
- Phase 2: idempotent bronze wraps, daily universe/level snapshots, and completed D1
  projection with source hashes and watermarks.
- Phase 3/3b: zero-extra-request M5 tee, coverage/gap rows, capped spool, capture-only
  pacer, IB backfill transport, nightly/weekly backfill, and trickled yfinance seed.
- Phase 4: versioned XNYS sessions and deterministic M15/M30/H1/W1 aggregation.
- Phase 5: point-in-time daily/intraday feature snapshots and anchor instances using
  champion calculations, including AVWAP parity at 1e-9.
- Phase 6: deterministic occurrence/revision/episode identity and versioned swing and
  intraday outcome simulation with costs, ambiguity bounds, partials, time stops,
  slippage, and open/truncated states.
- Phase 7: manifest-resolved read path and read-only Research panel; DuckDB remains
  optional and pyarrow can answer every slice.
- Phase 8: three-class backups, restore check, single-flight build/status CLI, job
  ledger, and six Health tiles.
- Defect passes repaired outcome supersession, management bounds, feature windows,
  per-bar backfill dedupe, pacing clocks, gap semantics, session identity, compaction
  reads, every job invoker, live tee wiring, and off-GUI-thread spool I/O.
- Phases 0–8 are code-complete on the testing-week branch. The broker check,
  confirmation items, and 20-session pilot remain open.

### Testing, packaging, and platform

- Broad pytest suite, deterministic smoke check, pytest markers, narrow Ruff gates,
  layered requirements with constraints, and Windows/macOS path handling.
- Provider telemetry at IBKR/Yahoo/Nasdaq boundaries with completeness contracts and
  honest UNKNOWN until measured.
- PyInstaller onedir spec, Qt runtime hook, asset/package drift test, lazy-engine
  `--selftest`, and a permanent guard preventing self-test from demanding packages
  deliberately excluded from the bundle.
- The first Windows frozen run found and closed an `ai_jobs` bundle-roster conflict;
  the current frozen self-test is 29/29.
- macOS launcher, CloudStorage Drive discovery, Keychain credentials, and machine-
  local path normalization.

### Shadow challengers

- Side-symmetric SPY market-state/pullback engine runs beside the legacy pause
  detector, emits replayable evidence, and cannot affect candidates, alerts, or rank.
- Greatness Monitor persists ordered touch/wick/close/acceptance/retest/failure/re-arm
  transitions beside legacy D1 alerts and cannot alter the champion path.
- Champion-invariance tests prove enabled, failing, or poisoned shadow engines leave
  production SPY/D1 results unchanged.

Neither challenger is promoted. Their remaining evidence gates are in `plan.md`.

## Revision history

### 2026-08-19 (evening) — chart review shows movers only

`IMPLEMENTED`, live proof owed (`docs/DESK_TESTING_PLAN.md` §2.10). Trader rule,
recorded verbatim as a dated addendum in the R2 spec: "a long inside yesterday's
range is probably chop", so chart review shows only longs above the previous
day's high and shorts below the previous day's low, Focus picks beyond their
previous-day extreme are flagged, and inside-range picks appear only on a
deliberate Focus review.

**One predicate.** `focus_adoption_gate.mover_state` is the adoption gate's own
extreme leg — a thin name over the same `prev_day_break_state` call — and
`focus_adoption_gate_state` now routes its extreme leg through it, so there is
exactly one implementation of "beyond yesterday's extreme" in the tree. A test
walks the whole input matrix and asserts the two entry points cannot disagree; a
filter with a private copy of the rule would eventually hide a name the machine
had just adopted. No session-VWAP leg: this asks the weaker question on purpose.

**The filter is presentation, and stays that way.** It lives in
`AlertCenterPanel._enqueue_review_alert`, the single door into the review queue.
It hides and counts — `N hidden (inside yesterday's range) - show`, one click
reveals exactly those names and turns the filter off for that session (day-scoped,
resets with the market date). It removes nothing from any feed, history or store,
mutes no alert or push, auto-removes no watchlist or Focus entry, writes nothing
to `review_policy.json`, and records nothing to the review-learning stream. Each
of those is a test.

**UNKNOWN shows, tagged `unmeasured`.** Missing data is uncertainty, never
confirmation, and a filter that failed closed would blank the review the moment
the daily store hiccuped — indistinguishable from "nothing qualifies".

**Two entry points bypass it entirely:** the deliberate Focus review
(`review_focus_picks`), because answering a request for the trader's own list
with a subset of it is the surface lying, and armed chart-watch hits, because
that is the exact condition the trader armed.

**The flag.** A Focus chip beyond its previous-day extreme on its own side
carries `MOVING`, in the same badge idiom as the existing `BOUNCE`/`RRS` flags;
the charted alert shows `MOVING` / `unmeasured` / `inside range` beside the
reviewed-today badge. It repaints off the Alert Center's existing 60-second D1
poll via a new `focusBreakStatesChanged` signal — no new timer, no new market
data, no IB traffic.

**Docs.** `CLAUDE.md`/`AGENTS.md` also correct a line that contradicted the R1
spec: OFF does nothing automatic at all, including no auto-pick adoption. The
old wording claimed OFF was the one mode where auto picks self-apply.

### 2026-08-19 — the adoption gate could not compare two clocks

`FIXED`, live proof re-owed. The first DESK morning adopted nothing: every
attempt raised `TypeError: can't subtract offset-naive and offset-aware
datetimes` inside `pending_pick_gate_ok`, the Alert Center refused each pick
fail-closed, and 121 picks were refused every 30 seconds from 08:07 onward.

**Root cause.** A stored verdict carries two stamps written by different paths.
`gate_bar_end` is the intraday profile's `as_of`, which
`_intraday_extreme_metrics` writes **always aware** — the provider's own offset
when it has one, market-local otherwise. `gate_checked_at` and the caller's
clock are plain `datetime.now()`, **naive**. So the wall-clock age check
(naive − naive) passed and the bar-lag check (naive − aware) raised. The gate did
not judge the picks wrongly; it never ran.

**Fix.** Every datetime `pending_pick_gate_ok` compares — the caller's clock,
both stored stamps and the flip barrier — is normalized at the seam through
`market_session.normalize_market_local_datetime`, which ATTACHES market-local to
a naive stamp and converts an aware one. Stripping offsets instead would have
ended the crash and kept the outage: an aware 11:05 ET bar read as naive against
an 08:07 PT clock is three hours "ahead of the tape", so every pick would still
have been refused, silently. A test pins that direction. `minutes_since_open`
carried the identical subtraction and is hardened the same way; every caller
passes a naive clock today, so its answers are unchanged.

**The log flood is bounded.** The refusal wrapper logged a traceback per pick,
so one systematic fault wrote 121 tracebacks every 30 seconds and rotated the log
that held the evidence. Now the first failure of a poll cycle carries the
traceback and the cycle ends with one WARNING naming the count and the exception.
Fail-closed semantics are unchanged.

**The retry investigation found no disagreement.** R2.2's 60-second, five-attempt
budget governs only a failed flip re-measurement; the desk was in DESK mode from
the start on 08-19, so it was never engaged. The 30-second cadence in the log is
the ordinary poll, and a refused pick is deliberately not marked seen so every
cycle re-attempts the queue — which is what made recovery automatic once the code
was fixed. Recorded in the R2 spec so it is not re-litigated.

### 2026-08-19 — the teardown crash is intermittent, and its attribution was stale

`RECORDED`, not fixed. Measured through Python's own `returncode` rather than a
shell that truncates `0xC0000409` to `127`: today's tip, **yesterday's unchanged
tip re-run today**, the suite minus `tests/test_ui_stall_watchdog.py`, and the
suite minus either of today's new test files all report every test passing and
all exit `0xC0000409`. So the crash is **intermittent** — the same commit read
clean yesterday — and the recorded discriminator (ignore the stall-watchdog tests
→ exit 0) no longer holds on this tree. The 2026-08-18 entry's "did not occur in
any run today" was a true observation read through a truncating shell; it is
corrected here rather than left to be mistaken for a fix. `ui/stall_watchdog.py`
is product code owed R6(c)'s diagnostic week and is NOT to be edited to make a
suite exit cleanly. Quote the summary line and the exit code together.

### 2026-08-19 — the strength board becomes readable

`IMPLEMENTED`, live proof owed (`docs/DESK_TESTING_PLAN.md` §2.7a). Two trader
requests against a board that was "just a lot of picks".

**Every column sorts on click**, with a visible indicator. Sorting is
presentation: it re-orders rows already in hand, never calls the service and
therefore can never cost a refetch — the board's data budget stays one batched
yfinance pull every 15 minutes and zero IB traffic. Qt's own `setSortingEnabled`
is deliberately not used, because the last column holds a per-row cell *widget*
and `QTableWidget` leaves cell widgets behind when it sorts — the Add button
would end up on its neighbour's row. Owning the order also puts blank cells last
in BOTH directions: an unmeasured field is an absence, not a small number. The
default order is unchanged and now stated by the indicator (longs
strength-descending, shorts ascending — strongest for that side first). Every add
still re-runs the adoption gate at click time.

**Selecting a row charts it** in the desk's existing snapshot popup — the same
one the RS/RW, entry and Industry boards open, owned by the Alert Center, so the
chart carries the same bot-backed series, painted levels and CaptureRail. No new
chart widget exists anywhere; `show_symbol_snapshot` already reuses one dialog
per owner, so re-selecting re-points that window instead of stacking dialogs.
Selecting on one side clears the other, and a refresh that keeps the same row
selected is not a new chart request. A docked always-visible chart is recorded as
the follow-up option: it needs a desk-layout decision about the two tables' width
rather than more wiring.

### 2026-08-18 — R7/R8's deferred visuals, and the one buildable wishlist item

`IMPLEMENTED`, live proof owed. Same branch and same redirect as the entry
below.

**The journal's per-group charts (R7 deferred scope).** The Analytics tab gains
a group picker, a bar chart of net by bucket, and a CSV of exactly what is
charted. Every bar carries its n as closed trades and a thin sample says so on
its own label. A bucket whose total cannot be converted is **excluded, never
drawn as zero** — None there means "mixed currencies, unconverted", and a zero
bar would claim the setup broke even. What the 12-bar cap drops is printed,
because a silent top-N reads as "that was all of them".

**The Calendar year heatmap (R7 deferred scope).** A pyqtgraph image of the
year, diverging red→white→green, **centred on zero and scaled to the largest
single day** so a good year and a bad one are drawn on the same footing. A day
with no trading stays blank rather than taking a break-even colour — a flat day
and a day the trader did not trade are different facts. The numeric grid stays
underneath and still filters the Trades tab on a click.

**The wishlist triage (trader-directed).** Every `WISHLIST.md` candidate was
assessed against the codebase. One was buildable: the external chart deep link,
now `scripts/external_chart_links.py` plus an **Open in TradingView** button on
the arm bar, with the URL template as a machine-local setting, symbol validation
before any URL is built, and a refused open reported rather than swallowed.
TC2000 is deliberately not wired — it answers no documented URL scheme, and a
dead `tc2000://` link would be worse than the honest gap. Every other item
needed exactly one trader judgment whose plausible answers lead to different
code; those are written down one per item in `docs/WISHLIST_OPEN_QUESTIONS.md`
instead of being guessed at. Nothing else was promoted into `plan.md`.

**R4's two held items resolved.** The Focus Picks reviewed-today marker is
built as a read-only line BESIDE the editors rather than a glyph inside them -
those editors hold watchlist text that is synced back, so a marker in a row is
one careless save from becoming a symbol name. §2.2's `review_host` for the
boards is now a recorded decision rather than a hold: a ranked board has no
"next row", and advancing through one would invent a queue the trader never
asked for.

**Gate for the whole blitz branch.** 3760 passed / 19 subtests, exit 0; smoke
7/7, exit 0; clean-cache frozen rebuild + `selftest OK: 56/56 checks passed
(frozen)`, exit 0, with `build/` and `dist/` deleted first and the build run
from the worktree so the desk's own `dist/` was never touched (exe mtime
22:02 postdates the commit at 22:00). The `0xC0000409` teardown crash the
testing-week entry warned about did **not** reproduce in any run on this
branch; that is reported, not claimed as fixed.

**Packaging.** `external_chart_links` joins the selftest roster because the
Alert Center imports it inside the click handler — the failure mode is a bundle
that starts fine and dies the first time the trader presses the button. Source
selftest 56/56.

### 2026-08-18 — R5 completed: confluence, first-candle ORB, any-bounce watch

`IMPLEMENTED`, live proof owed. Built on `phase05-integration-blitz` under the
trader's integration redirect of 2026-08-18, which is the override R5 §8.2
named as its own first reopen trigger.

**Two new M5 engines, wired and silent.** `m5_signal_engines` gained
`confluence_events`/`latest_confluence` (Heikin-Ashi reversal + SMI turn +
LRSI cross within a tunable 4-bar window, **M5 Focus symbols only**) and
`orb_events`/`latest_orb_events` (gap-up first candle sets the session extreme;
after an LRSI pullback below 50 it arms a new-extreme alert and an
informational recross). Both are **pure and stateless** — they recompute from
the session's completed bars, so a toggle flipped mid-session cannot wake a
state machine holding contents no session exercised, which was §8.2's actual
objection. All four new alert types default **OFF**, so the desk session §7
demands now gates audibility rather than existence.

**The any-bounce watch (R5 §4).** One armed request per symbol and side over
the whole level set — D1 1st-dev band, current and prior AVWAP, prior 1st-dev
band, D1 15/21 EMA, session M5 15/21 EMA, H1 15 EMA — evaluated with the two-bar
bounce idiom the D1 zone arms already use, firing once on the level that held
and then disarming. New `any_bounce_watches.json` store, owned by the Alert
Center panel like every other watch store; new **Any bounce** button on the arm
bar. A level the data cannot supply is absent, never fabricated.

**The prior-anchor AVWAP line (R5 §8.3), proven additive.** `prev_avwape` is
carried — not recomputed — from `prev_anchor_meta["vwap"]` onto the zone-arms
entry as an optional TOP-LEVEL key, never a `trigger_levels` arm, so the
shipped zone-arm alert rubric cannot gain a trigger. The golden characterization
fixture landed first and passes **unchanged** after the edit; a companion test
shows the trigger walker cannot see the key. No second
`calc_anchored_vwap_bands` call exists anywhere — the σ-formula invariant is not
approached.

**R6(b) closed out.** The read-only JSONL-ledger audit now reports each
diagnostics ledger's measured size, estimated rows and last write inside the
existing footprint check — reusing that walk, reading a 256 KB sample rather
than 370 MB, and writing nothing. The stale `~106 MB` comment is gone: no
current size is recorded in code any more, because the two that were had both
gone stale by growth within weeks. Rotation stays **declined** (plan.md 6(b)).

**Two hermeticity gaps found by the merge and fixed test-side.** The Fed
calendar adapter reached `federalreserve.gov` from the daily-prep orchestrator
in a full-suite run (it passed in isolation because the on-disk cache answered
first) — stubbed at `_fetch_text`, the one boundary both callers share. And the
new `PriceHistoryShapeTests` were measuring conftest's own offline stub rather
than `fetch_price_history`; they now take back the stashed original, which the
guard provides for exactly this case.

### 2026-08-18 — two live sessions, and the two defects they exposed

`IMPLEMENTED` + `GREEN`. Repair pass on `phase05-r2-focus-gating-strength-board`,
after AWAY sessions on 2026-08-17 and 2026-08-18. Live-proof results — one R2
PASS, one R1 PASS, one HALF-PROVEN, five UNKNOWN — are recorded in
`CURRENT_CHECKPOINT.md` with their log evidence and are **not** promoted here:
nothing became `LIVE_VALIDATED`.

**A reader holding a report open cost the desk three whole swing scans.**
2026-08-17 07:30 and 10:00, and 2026-08-18 12:00, each after 8 to 30 minutes of
real work. All three run manifests carry `"error": "PermissionError(13, 'Access
is denied')"` and a phase list ending at `output/signals`; the surviving
traceback names `legacy.py:2122`, the `os.replace` inside `_write_text_atomic`,
replacing `master_avwap_market_prep.txt`. It is a self-inflicted race:
`write_market_prep_files` writes the JSON first, the desk's own Market Prep
panel watches that JSON with a `QFileSystemWatcher` and re-reads the *report
text* on the change, and Windows' `open()` does not grant FILE_SHARE_DELETE — so
the replace landing milliseconds later is denied. Not the frozen `-c` spawn
class of 2026-08-13 (that failed one second in with exit code 2) and not a data
fault (the provider counters on the failed run match the successful one).

`_write_text_atomic` and `_write_dataframe_csv_atomic` now replace through a
bounded retry — ten attempts a tenth of a second apart — the same doctrine
`project_paths.SafeRotatingFileHandler` already applies to a locked log file at
rollover. A lock outliving the budget still raises: a report that cannot be
published must never be reported as published. `save_json` inherits it, since it
writes through `_write_text_atomic`. The file-scoped ask-first rule was applied;
the trader approved the `legacy.py` edit before it was made. No detector, score,
ranking or alert behaviour changed — only the publish step became lock-tolerant.

**The failure path now names its own cause.** `AutopilotService._on_scan_failed`
writes only `detail.splitlines()[0]` to `autopilot.log`, so "exited with code 1"
was the entire public record and identifying these three took the run manifests
plus a log that had since rotated. `scan_service` now lifts the child's final
unindented exception line onto that first line, bounded to 240 characters, and
leaves the message unchanged when the child dies without one (a native fault, a
kill) rather than quoting a random stack frame.

**One odd yfinance frame aborted the universe rebuild.** `Universe rebuild
failed: "['datetime'] not in index"` (2026-08-17 06:00:16), raised by the column
selection ending `fetch_price_history`'s per-symbol loop: yfinance normally
names the daily index `Date`, that chunk arrived with an unnamed index,
`reset_index()` produced `index`, and one malformed sub-frame killed the whole
rebuild while every other per-symbol fault there is skipped. It self-healed on
the ~60-minute retry, which is why it needed a test rather than a watch. The
date axis is now resolved by name (`Date`/`Datetime`/`index`/`level_0`) and then
by dtype, and an unusable frame is skipped and counted — five warnings plus one
total, so a systematic oddity reads as one line rather than 1,500.

**And a floor under that fail-soft.** `build_universe` wrote
`universe_all/longs/shorts` unconditionally, so a fetch outage that priced
nothing would have overwritten a good universe with an empty file — against the
sec 5 invariant that a failed publish never destroys the last verified report.
An empty screen now raises; the caller already logs and retries in ~60 minutes,
and the previous universe stays authoritative until a rebuild succeeds.

Fourteen new tests, every one verified to fail against the unfixed code,
including a Windows-only reproduction that holds a real read handle on the
destination while the write runs. Gates on the new candidate: **2935 passed / 19
subtests, smoke 7/7, `selftest OK: 31/31 checks passed (frozen)`**, all exit 0,
with `build/` and `dist/` deleted before the rebuild.

**Recorded, not fixed.** `BouncePanel.__init__` runs
`QTimer.singleShot(0, self.start)` (`bounce_panel.py:280`), so the desk connects
to IB on every launch at any hour, outside Auto Pilot and outside quiet hours.
That is what produced `IB: connected` at 22:06:41 on the quiet-boot night — not
an Auto Pilot BounceBot start, which is unconditionally announced by
`Starting BounceBot` and was absent. It contradicts the *wording* of the R1
quiet-hours proof (which said "no IB connect"), so that wording is corrected;
changing the behaviour is an R1 decision left to the trader.

### 2026-08-16 — R3 swing-quality classifier enters shadow

`IMPLEMENTED` as additive shadow evidence only; live gate owed. The classifier
stamps `would_demote` for directional EMA21 distance over 2 ATR and trade-side
zones beyond the first AVWAP band. It also carries the already-loaded D1 RVOL
and an annotation-only daytrade-candidate marker. The priority report duplicates
the calls in a bottom **NO LIVE CHANGE** section, the focus payload and feature
CSV retain the measurements, and the desk adds a `Stretched? (shadow)` badge.
Tests prove live Best Swing rows, ordering and S/A/B membership are identical
before and after the stamps. Nothing moves, hides, demotes, alerts or writes a
watchlist until the trader accepts the owed full-session shadow week.

### 2026-08-16 — Phase 0.5 remaining-packet pre-flight

The trader's 2026-08-15 weekend redirect authorized R3 through R6, in order, on
the consolidated `phase05-r8-weekend-prep` branch. No live gate was waived: R3's
shadow week and R6's watchdog week remain owed alongside the existing R1/R2,
R7 and R8 proofs.

The required pre-R3 suite exposed two journal-coverage tests whose synthetic
Questrade chunk inherited the wall-clock date and therefore became a weekend
`NO_SESSION` on Sunday. Their fixture now uses its existing fixed Monday date;
production journal behavior and live data paths are unchanged. Verification:
targeted **26 passed**, then full suite **3354 passed / 19 subtests**, both exit 0.

### 2026-08-15 — packet R8: Weekend Prep

`IMPLEMENTED` + `GREEN`. One live gate owed: a real weekend run. Built on
`phase05-r8-weekend-prep`, cut from the R7 tip `4420bbf`, in the spec's §9
commit order (`docs/WEEKEND_PREP_PLAN.md`).

A new top-level **Weekend Prep** page: a guided five-step routine matching the
trader's weekend ritual — week in review, focus-pick review, walk-away with the
weekly auto-tag review, strength discovery on H1/D1/Monthly, and the
forward-looking week-ahead prep. Progress persists across sittings in
`weekend_prep_state.json` (atomic write, pruned to eight weekends), keyed by the
Friday of the week containing the last completed session so Saturday and Sunday
resume the same routine rather than restarting it.

**The scanner reuses the formula rather than copying it.**
`scripts/strength_scan.py` is fenced by the spec and is not edited; a new pure
`weekend_strength` imports its functions and reimplements only the board
orchestration, which is where the three timeframes genuinely differ. "Completed
bars only" needs three rules: H1 by clock arithmetic, D1 by
`last_completed_session` (not "yesterday" — after a Monday holiday that is
Friday), and monthly by **month identity, never duration**, which is the only
test that is right on the 1st of a month.

**Filters** are the spec's §5 table, trader-approved as proposed. Session VWAP
is dropped above M5 rather than imitated: there is no session inside an H1, D1
or monthly bar for it to anchor to. Each leg is its own named function, and a
leg that cannot be measured fails with a reason rather than passing by default.

**Nothing starts itself and nothing is removed.** Neither the service nor the
panel owns a timer, and there is no removal call anywhere in the tab — both
asserted by parsing the source, not by promise. Adopt routes to swing Focus
through the existing membership-tracked injection with `origin="weekend_prep"`.
The R2 M5 adoption gate is deliberately not applied to weekend swing adds (§7);
a test asserts its absence so it is not later "restored".

#### Defects found while building, each by a test

| # | Defect | Where it would have shown |
|---|---|---|
| 1 | `app.py` kept three index-aligned structures; the titles tuple was one short | **Clicking Settings raised IndexError**, and eight titles from index 3 named the wrong page — live on the desk |
| 2 | `weekend_strength` read a bar's time from `timestamp`/`time`/`date`; `autopilot_core._frame_rows` emits `dt` | **Every board would have measured nothing** on a live desk while every hand-built unit test passed |
| 3 | A total fetch failure returned an empty board | It overwrote the last good board — "nothing is strong this week" is a claim about the market, not the provider |

Defect 1's two existing guard tests had been passing throughout: they compared
the positions of two string literals at indices 1 and 2, which were fine.
Defect 2 was caught only by the one test that went through the downloader
instead of around it.

#### Frozen packaging

`selftest OK: 49/49 checks passed (frozen)`, exit 0, **first attempt** — from a
deleted `build/` **and** `dist/`, following the stale-cache rule R7's close-out
wrote into the checkpoint. Roster additions: `market_prep.orchestrator` (lazily
imported inside the week-ahead worker, so nothing statically reachable pulls it
in any more), `weekend_strength`, the service and the panel.

### 2026-08-15 — packet R7: the tax-grade journal

`IMPLEMENTED` + `GREEN`. Live gates owed — see `plan.md` Phase 0.5 R7 and
`CURRENT_CHECKPOINT.md`. Built on `phase05-r7-journal-reliability-ux`, cut from
the R2 tip by the trader's second redirect of 2026-08-15, in the spec's §9
commit order (`docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md`).

The trader's report was two sentences: *"the journal misses trades, has trades
open — not acceptable; I need this for tax purposes too."* The register in the
spec's §3 found eighteen distinct causes behind them. What follows is what was
built, not what was intended.

**Identity (B4, B3, B6).** `execution_uid` stopped embedding the symbol and the
timestamp, so the same IBKR fill arriving over the socket during the session and
again in that night's Flex statement is one execution instead of two — the
mechanism that opened a ten-share position as twenty and left it open forever.
A new `journal_identity` module owns one security-type vocabulary shared by both
brokers, the assembler and the migration, so a position spelled `STOCK` by one
import and `NASDAQ` by another (Questrade's `listingExchange` fallback) is one
position again. `trade_id` is anchored to its opening execution rather than a
per-group sequence, and a re-key pass plus `trade_aliases` carries annotations
onto rebuilt trades — I4, which did not hold before and now has a permanent
zero-orphan test.

**Completeness (A3, A5, A6, A7, I2).** `import_coverage` records which
(broker, account, day) an import actually spanned, so a day nobody imported and
a day with no trades stopped being the same absence of rows. The Questrade pull
persists per (account, chunk), so one failing chunk no longer discards fills
already fetched. `journal_coverage.self_heal` repairs oldest first, bounded at
62 days a night and 5 attempts a day. The IBKR socket marks no coverage at all —
it sees only the current TWS session — and Flex marks from the statement's own
declared span. Option expiries, exercises and assignments now arrive as the
fills they really are, which is what finally closes an option that expired
worthless; dividends, interest, fees and FX land in `cash_transactions` and
deliberately never near `raw_executions`.

**Corrections (B7, I3).** `trade_adjustments` is an append-only audit trail with
a mandatory reason, re-applied on every rebuild so a correction survives the
next import. VOID, EDIT, ADD, REASSIGN_GROUP and FORCE_CLOSE, with undo as a
superseding record rather than a delete.

**Currency (B8, I5).** Rates are booked once per (date, currency) from the Bank
of Canada, never fetched at render, with weekend and holiday carry-back recorded
in `effective_date`. An unconverted trade is NULL — not zero, and not the native
number relabelled — and a mixed-currency total with anything unconverted is
**refused** with a reason rather than shown wrong.

**Reconciliation (B1, B2).** `journal_reconcile` compares the journal's net-open
against both brokers' reported positions. A journal-open-but-broker-flat
position produces a *suggested* force-close that the trader confirms; the
suggestion is stored outside `trade_adjustments` precisely so it cannot apply
itself.

**The nightly slot (I8).** `journal_import` is the first `JobSlot` in the
`ai_jobs` slate — the one sanctioned exception to the slate's no-reorder rule,
because both AI jobs read the journal. No new timer, no new thread, no new ntfy
sender, and a test asserts that against the parsed source.

**The Journal tab.** A shell over Trades, Calendar, Analytics, Health and Fees,
all reading through `ui.services.journal_feed` and none holding a store. The
shared header groups accounts by tax treatment and badges a blended selection
(I6). Health carries the coverage grid, the reconciliation confirm flow, and the
Flex token/query-id fields plus a backfill button — closing A1 and A9, where the
only complete import path was a CLI the trader never ran.

#### Defects found while building, each by a test

Recorded because they are the useful part of the record, not because they closed.

| # | Defect | Where it would have shown |
|---|---|---|
| 1 | The store's multiplier rule read `security_type` verbatim, so it knew `"OPT"` and missed Questrade's `"Option"` | **Every Questrade option trade's P&L was out by 100×** |
| 2 | The activities cross-check ran before the chunk was marked COVERED | A COVERED mark painted over the disagreement it had just found |
| 3 | The Flex parser counted the `OptionEAE` section container as a row (IBKR nests same-named elements) | A phantom option expiry, once those rows became executions |
| 4 | `list_active_adjustments` broke `created_at` ties by random uuid, and `created_at` is second-precision | Which of two same-second corrections won was a coin flip |
| 5 | `undo_adjustment` reused the real actions with an empty payload, on the theory that empty is inert | True for EDIT, false for FORCE_CLOSE: undoing a force-close left it force-closed |
| 6 | Per-group analytics summed native P&L under a converted headline | A breakdown that disagreed with the total directly above it |
| 7 | The nightly path rebuilt twice, the first time from executions already known to have holes | Wasted work, and a journal assembled from a state it was about to repair |

#### Frozen packaging

Three rebuilds. The first two reported **31/31 (frozen)** — the pre-existing
roster, passing with R7 code in the bundle. Extending `selftest.LAZY_ENGINE_MODULES`
by the fourteen journal modules did not change the frozen count until `build/`
was cleared, which is a finding in its own right: **a PyInstaller rebuild can
silently reuse a cached module**, and that is precisely the failure mode that let
"frozen selftest 30/30" be recorded three times in R1/R2 for runs that had never
happened. The clean rebuild reports **`selftest OK: 45/45 checks passed
(frozen)`**, exit 0, and `ui` collects 117 submodules against 109 before, which
is the new `ui/panels/journal/` package.

### 2026-08-15 — R2.3: a flip's identity is a counter, never its timestamp

The final external review round reproduced one defect on the R2.2 tip: two
DESK returns inside the same second shared their second-floored `_desk_flip_at`,
which was also the re-verification attempt's identity — so an in-flight run
begun for the first return could satisfy the second return's debt, and its
same-second stamp then passed the verdict barrier without a post-flip
measurement (`reverify_calls=1`, adoption on unseen tape). The mode button
cycles through the modes, so two same-second DESK entries are a normal rapid-
click gesture, not a race.

- The flip's identity is now `_desk_flip_generation`, a counter incremented on
  every DESK return; a finishing worker satisfies only its own generation.
  `_desk_flip_at` remains solely the verdict `not_before` barrier — the two
  jobs are now carried by two fields because one field provably could not do
  both.
- The failure side got the same treatment: a superseded run's failure no
  longer spends the newer flip's retry budget — the newer return owes its own
  attempt with its full allowance, matching how the success side already
  refused to let an old run answer a new debt.
- Two new tests run the round trip with the clock deliberately frozen inside
  one second; the same-second test was verified to fail against the un-fixed
  code (the external reviewer's exact reproduction) before the fix landed.
- Also from that round: the frozen executable had been built 21 seconds
  before the tip it claimed to represent. The rebuild now happens after the
  last code commit, and the checkpoint records commit time and executable
  mtime side by side so the ordering is visible on its face.

### 2026-08-15 — R2.2: the drain waits for a measurement taken since you came back

`IMPLEMENTED` + `GREEN`. First of four items from the final external review pass.

The AWAY/EVENING → DESK drain re-measured its queue before adopting, but the
lock was incidental. If that re-measurement **failed**, the next 30-second poll
fell straight through to the ordinary stored-verdict drain, and the only thing
left between a stalled feed and an adoption was the 2-bar lag bound.

Two independent mechanisms now, and the separation is the point:

- **The barrier.** The flip records its own moment (floored to the second, the
  resolution `gate_checked_at` carries), and adoption refuses any verdict
  stamped before it — `pending_pick_gate_ok(..., not_before=...)`. Nothing
  measured while the desk was unattended is adoptable by any path.
- **The retry.** A failed re-measurement waits 60 s and tries again, up to five
  times, instead of handing back to the drain. Giving up after that is safe
  because the barrier still holds: the ordinary 30-minute staging refresh stamps
  post-flip verdicts and becomes the recovery. The status line says which of the
  two it is doing.

A re-measurement also remembers which flip it answers, so a DESK → AWAY → DESK
round trip mid-flight is still owed one of its own rather than inheriting a
result whose bars predate the second return.

The 2-bar lag bound stays as defense in depth rather than as the lock.

### 2026-08-15 — R2.2: the desk runbook stops contradicting the checkpoint

Documentation only. Fourth of four items from the final external review pass.

`docs/DESK_TESTING_PLAN.md` claimed the 09:58 frozen build was **31/31**;
`CURRENT_CHECKPOINT.md` recorded **30/30**. The checkpoint was right, and the
evidence is in the build itself: the only selftest change between `e18757e` and
now is the `docs/DESK_TESTING_PLAN.md` asset check, added by commit `619be55` at
10:38 — so the runbook was claiming its own bundling had been verified before the
file existed. The runbook now states both counts and why they differ.

Two more contradictions closed in the same pass. Its merge section still told the
trader that a `test_warehouse_seal.py` failure was expected and could be re-run
past, three days after that defect was fixed and the checkpoint removed the
rerun-until-green carve-out; it now says no failure is acceptable. And it gained a
rollback section, because the rehearsed rollback to `e18757e` reports **30/30**,
not 31/31 — a 6am reader had no way to know that the count going *down* is the
count going back in time with everything else, rather than a broken rollback.

### 2026-08-15 — R2.2: the two-bar tolerance is accepted, and says so

Documentation and one test; no behavior changed. Third of four items from the
final external review pass.

`FOCUS_GATE_MAX_BAR_LAG = 2` lets a feed stalled by one or two bars adopt a name
that crossed back through session VWAP in the bars nobody saw. That is a
**trader-accepted exposure**, now written down in the constant's own comment and
in `docs/M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md` §11.2 rather than being
inferable only from the code: `max_bar_lag = 1` would refuse ordinary late
publication and have the desk declining most adoptions on a healthy feed.

The bound is named with it. An adopted pick is injected into the watchlists, so
BounceBot scans it; four completed M5 closes on the wrong side of
session/dynamic/EOD VWAP file a desync request and the Alert Center removes the
Focus entry — roughly four completed bars from adoption, unattended. A new test
pins both constants together, so the documented bound cannot quietly stop being
true.

### 2026-08-15 — R2.2: one quiet-hours boundary, one answer

`IMPLEMENTED` + `GREEN`. Second of four items from the final external review pass.

The quiet-hours gate had two fallback paths that spelled the same boundary
differently: `auto_scanning_due` compared against an inclusive datetime endpoint,
`AutopilotService._auto_work_due` used `hour < 14`. At exactly 14:00:00.000000 —
the one instant where those differ — the same clock produced "inside" from one
caller and "outside" from the other.

Both now build the window with `auto_quiet_hours_fallback_window` and compare
with `within_auto_scanning_window`, inclusive at both ends. Inclusive is the
correct side: this gate is permissive everywhere else too (close + 60 minutes,
widened to contain the sweep window, failing to a window rather than to
silence), and one extra microsecond of automatic work is waste while one refused
is a missed start. The new test pins the exact microsecond at both call sites,
in the ordinary path and in both fallback branches.

### 2026-08-15 — a Testing Plan tab, and the packaging trap it nearly walked into

`IMPLEMENTED` + `GREEN`. Documentation-and-viewer work; no engine touched.

`docs/DESK_TESTING_PLAN.md` is a plain-language runbook of the current testing
sequence — tonight's quiet-boot check, Monday's live proofs, then the
after-close checklist, rebuild and merge. Every step gives when to do it, what
to click, the exact log line or screen element that means it worked, what bad
looks like, and what to copy to the AI if it fails. It restates
`CURRENT_CHECKPOINT.md`'s owed proofs for a human reader and carries a header
line requiring it to be updated in the same pass whenever those change.

Settings gains a **Testing Plan** tab rendering that file read-only, with a
Refresh button and the file's last-modified time. It owns no timer, writes no
state and touches no engine. A missing, unreadable or empty file says so
plainly and shows **no cached copy** — a stale runbook read as current would
have the trader checking for log lines the build no longer prints.

**The packaging trap.** The plan is a runtime asset living **outside
`scripts/`**, and nothing existing would have caught that. The spec's
package-asset sweep only mirrors files inside `FIRST_PARTY_PACKAGES`, and
`test_packaging_spec_drift.py` only walks `scripts/`. The frozen desk would
have shipped showing "plan file not found" on the one page the trader opens
when nothing else is behaving — the same shape as the two defects that reached
the desk before (the `ai_jobs` roster clash and the `-c` scan spawn). Now
guarded three ways: an explicit `datas` rule with a hard `SystemExit` if the
file is absent at build time, a new selftest asset check, and a test asserting
the spec rule still exists. The view resolves through `sys._MEIPASS` when
frozen, since a frozen build has no `scripts/` tree to walk up from.

The frozen selftest is therefore **31/31**, not 30/30 — the new check is the
31st, and it was verified by an actual rebuild rather than assumed.

### 2026-08-15 — packet R2: the M5 Focus adoption gate and the strength board

`IMPLEMENTED` + `GREEN`; **not** `LIVE_VALIDATED` — the four live proofs in
`docs/M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md` §8 are owed. Built on branch
`phase05-r2-focus-gating-strength-board` (cut from R1, R1.1 merged forward) on
the trader's second explicit redirect ahead of P0.7.

**One gate, three call sites.** `scripts/focus_adoption_gate.py` holds the
combined rule — beyond yesterday's extreme AND on the right side of session
VWAP — and it runs at candidate build, at every staging refresh, and again at
adoption. Session VWAP comes from `chart_snapshot.session_vwap_series` over the
completed bars of the current session, carried on the profile as
`completed_session_vwap`; BounceBot's dynamic and EOD VWAP are deliberately not
used because they blend prior sessions and answer a different question.

The gate reads `last_complete`, not `last`. The previous filter measured the
prev-day break on the **forming** bar, so a pick could be admitted on a break
that bar then closed back inside. The golden fixture landed first and recorded
the effect: five of its seven baseline survivors now fail, each for a stated
reason, and nothing that failed the baseline now passes — the gate only
narrows.

**The queue is re-checked, not trusted.** Each 30-minute refresh re-runs the
gate over everything already staged: picks that have fallen back are evicted
with a logged reason, survivors are re-stamped. An evicted pick may re-propose
the same day if it re-qualifies — the queue says what qualifies now, not what
once did, and a name that pulled back at 10:00 and broke out at 13:00 is the
setup rather than the noise.

**Adoption reads a stored verdict** rather than measuring its own: the Alert
Center runs on the GUI thread and a staged pick is on no watchlist, so
BounceBot holds no bars for it. Failing, missing and >45-minute-old verdicts
are all refused, and a refusal does not mark the pick seen, so a stale verdict
costs one cycle rather than the pick. **This closes the stale-drain gap R1
recorded as an accepted limitation.**

**Focus entries now have an origin.** `focus_auto_picks.json` rides beside the
plain-text focus files (which do not change — the trader edits them by hand)
and marks only what the machine adopted. Absence of a marker reads as
user-entered, so every pre-R2 file is protected by default and a lost or
corrupt sidecar fails toward "the trader owns it". This is what makes
"user-entered names are never automatically removed" structural instead of
aspirational: without a per-entry origin, no removal verb could be written
safely at all.

**"Not today" says which of its two jobs it will do.** On an auto-adopted entry
it reads `✕ Not today - drop pick` and removes that one M5 entry on that one
side; on anything else it keeps its quieter feed-only meaning. The verdict is
recorded as `not_today`, not `dislike` — a same-day pass is not "this name is
bad", and the review-learning scoreboard must not learn that.

**The triple-VWAP desync is repaired by request, not by a second writer.**
BounceBot cuts a watchlist line on a worker thread, so it files a day-scoped
request and the Alert Center's existing poll performs the removal — one owner
per mutable store. The machine's own pick is removed; a name the trader typed
is left alone and the mismatch surfaced, because deleting it would break the
invariant and keeping it quietly would leave them trusting an entry nothing
scans.

**The strength board.** `scripts/strength_scan.py` implements the trader's
TC2000 formula (12-bar body sum × price level ÷ ATR50) with hand-computed
fixtures, plus the percentile cut and the VWAP/15EMA/prev-extreme filters. It
does not touch `real_relative_strength`; the existing RS/RW board is unchanged
beside it. `StrengthBoardService` owns one single-flight refresh on R1's
quiet-hours window and keeps the last good board through a failure;
`StrengthBoardPanel` is a new top-level desk page next to Focus Picks, with
per-row and side-aware adds that re-run the adoption gate at click time and
name any refusal with its reason.

**Transport measured before the cadence was chosen** (spec §10): 27.6 s for all
1,506 symbols at `period=5d`, 100% carrying ≥50 bars. `5d` rather than `1d`
because ATR50 and C50 need 51 completed bars and a `1d` window holds six at
07:00 PT — every symbol would be unmeasurable through the first four hours of
the session being traded. Zero IB traffic, so the locked pacing budget is
untouched. The 15-minute default stands with wide margin; the number was taken
on a Saturday and is recorded as a floor, to be re-measured live.

### 2026-08-15 — R1.1: the repair pass an independent review demanded

`IMPLEMENTED` + `GREEN`. Five code-verified defects from the R1 review, two of
them blockers that would have made the owed live proofs fail as written.

**The boot gate was cosmetic.** Quiet hours gated `AutopilotService.__init__`'s
resume but not `_ensure_bot_running`, which the tick calls every 30 seconds — so
a 21:00 launch logged "nothing starts yet" and connected BounceBot to IB half a
minute later. The gate now lives inside `_ensure_bot_running`, the one place
automation starts the bot, with `force=True` as the manual carve-out that
`force_reconnect` passes. The original test missed it by stopping the timer
before a tick could run; the new one runs a real tick with the clock frozen to a
weekday 21:00.

**The SPY alarm read yesterday's tape.** `_spy_session_bars` calls the last
cached bar's date "today", and the sweep is paused overnight, so on an Evening
morning after a ±1% day the cache still held that move — and the quiet window
opens 30 minutes before the bell. The alarm now refuses a series whose last bar
predates the day it is asked about. Roughly seven false urgent wake-ups per such
morning, none of them ever sent, because the alarm had not yet run live.

**A post-window relaunch silently cancelled the after-close wrap-up.** The quiet
refusal in `_maybe_run_swing_slot` returned before any slot resolution, so slots
still pending after the window closed — a crash, or the 4h39m machine sleep this
desk had on 2026-08-11 — stayed pending forever and `after_close_wrapup_due`
never fired. Slots are now resolved once the window closes, on the same
reasoning as Evening's refused slots. Before the window opens nothing is
resolved: those slots are still going to run.

**EVENING adopted picks immediately**, against the spec, the CLAUDE.md matrix,
the runbook and the CHANGELOG, all of which said it stages until the DESK flip.
It now refuses like AWAY, and stops beeping too — closing the spec §1 alert cell
the R1 build had left unimplemented. The trader is asleep; the SPY alarm is
EVENING's deliberate wake channel.

**The legacy Tk GUI died at construction.** Removing `get_shared_watchlist_paths`
with the shared/local vocabulary left `master_avwap_lib.gui` — which copies
legacy's globals wholesale — raising NameError. Invisible to the suite, which
imports these modules but never constructs them, and to the import-only frozen
self-test. New `tests/test_module_globals_resolve.py` statically resolves every
global that four never-constructed legacy modules read; it was verified to fail
on the un-fixed file before the fix went back in.

Hardening in the same pass: a NaN threshold no longer bypasses the alarm's
threshold test; the quiet-window ⊇ sweep-window containment is now structural
(`auto_scanning_window` widens itself to contain `bouncebot_scan_window`, so two
independent settings keys cannot be configured into contradiction);
`autopilot_auto_arm_due` takes `quiet_hours` so its test no longer depends on a
machine-local setting; the launch self-heal gate and the D1-feed beep site gained
coverage; Qt tests skip rather than silently pass without PySide6; and the
"an early close moves this window" docstring claim is corrected — no early-close
modelling exists anywhere, which is pre-existing and fail-open.

Recorded, not fixed: a corrupt `local_settings.json` still silently re-homes the
store to `%LOCALAPPDATA%`. And whether EVENING should also pause the BounceBot
sweep is now an explicit open question in the spec's new §9 rather than a
silently unbuilt matrix cell — pausing it would also stop the alert stream the
same matrix says EVENING should queue, and remove the prices the strength checks
read, so the build implemented the unambiguous cell and left this one.

### 2026-08-15 — packet R1: quiet hours, the auto-mode matrix, and one scan

`IMPLEMENTED` + `GREEN`; **not** `LIVE_VALIDATED` — the four live proofs in
`docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md` §6 are owed. Built on branch
`phase05-r1-auto-modes-quiet-hours` after the trader directed R1 ahead of the
P0.7 merge gate. Two commits: the behaviour, then the removal.

**Quiet hours.** `autopilot_core.auto_scanning_window` / `auto_scanning_due`
mirror the proven `bouncebot_scanning_due` pattern — pure window, reason string,
weekend refusal, settings override, fail-open on an unanswerable session lookup —
and now gate every automatic starter: the launch `_self_heal_universe` (which
fired 2.5 s after launch and every 30 minutes with no clock check at all), the
tick-loop universe heal, the boot resume that connected BounceBot to IB whenever
Auto was left ON, the daily 07:00 self-arm (previously unbounded above, so a
21:00 launch self-armed against a closed tape), the open watchlist build, and the
swing slots. Manual work is gated nowhere; `force=True` is the carve-out.
The window is 06:00–14:00 on a normal session and is deliberately a **superset**
of the BounceBot scan window — a literal open-at-06:30 gate would refuse the IB
connect at 06:10 while the sweep window said the sweep could run.

**EVENING prepares the morning and stops.** The open+30 early slot, the
07:00/07:15/07:30 strength checks and the briefing still run; every ordinary
hourly slot is refused and named once in the log, and the open self-build is
skipped without a sticky marker so the wake-up flip to DESK can still build.
Refused slots are marked *done* rather than left pending, because
`after_close_wrapup_due` requires every slot to be done and pending ones would
have silently cancelled the after-close wrap-up for the day.

**AWAY queues instead of adopting.** Auto-populate picks now stage in AWAY as
they already did in DESK and EVENING; OFF is the only remaining self-applying
mode. This reverses the 2026-08-05 rule for AWAY on the trader's reasoning:
nobody is present to *prune*, so a pick applied at 09:00 alerted unwatched all
day. The Alert Center refuses adoption while AWAY and marks nothing seen, so the
whole day drains on the flip to DESK — the same point where packet R2 will add
its freshness re-check. Alerts arrive without a sound; feed, history and the D1
unread badge keep filling, and an unreadable mode file resolves to OFF so a
broken read can never silence the desk.

**EVENING's SPY ±1% wake alarm** — the second deliberate exception to the
AWAY-only push rule, after the always-on price alerts. Reads the champion cached
SPY bars, repeats every five minutes while the condition holds, stops on the flip
out of EVENING, day-rolls its stamp in the Auto Pilot state file, and stamps only
a delivered push. NaN is refused explicitly, because `nan < threshold` is False
and no data at all would otherwise read as a 1% move. Kill switch
`push_evening_spy_alarm`, threshold override `push_evening_spy_alarm_pct`.

**One scan, not two.** `use_shared_watchlists` is gone from thirteen files and
one run-manifest counter: both branches resolved to `(LONGS_FILE, SHORTS_FILE)`
under the same label, so the Shared/Local choice the menu offered was never a
choice. `ScanService.run_shared_watchlist_scan`/`run_local_watchlist_scan`
collapse to `run_watchlist_scan`; the menu pair and the Ctrl+R "Run Shared Scan"
action become one "Run Scan"; the `run_master_with_shared_watchlists` alias is
gone; the scheduler's "shared-watchlist" wording and the false "local project
watchlists" label are gone. The manifest counter had no consumer — checked before
deleting. The job-ledger `config_hash` stays `"shared-v1"` deliberately: an
opaque idempotency token, not user-facing text.

**Cloud-drive store discovery removed.** `project_paths` probed `$GOOGLE_DRIVE`,
`~/My Drive`, `~/Google Drive` and the macOS CloudStorage accounts *at import*
and adopted the first writable one as the operational store whenever
`shared_data_dir` was unset — inert on this desk only because that setting
happens to be set, and directly against decision 0015. The fallback is now
plainly local. The mount-presence guard is kept and renamed
`_wait_for_shared_store` (decision 0015 blessed its macOS half), with the sync-
client instructions removed from its messages. No path moved: the desk still
resolves `C:\TradingBotData` from `local_config`. Decision 0015 carries a dated
amendment explaining which half was harmless and which was not.

Thirty-five new tests in `tests/test_auto_quiet_hours_and_modes.py`. Four
existing tests asserted behaviour this packet reverses and were updated to the
new rule rather than worked around; two more were inverted to prove cloud-drive
discovery can no longer happen.

### 2026-08-15 — trader refinement packets promoted; after-close mechanism identified

Documentation-only pass; no code, path, or test changed. The trader promoted the
2026-08-14 wishlist entries into `plan.md` **Phase 0.5** (packets R1–R6, ranked
R1 auto-modes/quiet-hours first, R2 Focus-gating/strength-board second) with five
ACTIVE specifications added under `docs/` and indexed in `docs/README.md`. Code
for these packets starts only after P0.7 merges.

Two findings from the read-only recon are recorded as knowledge about implemented
behavior:

- **Why Master AVWAP setups "totally change after the close."** The live scan
  applies no completed-bar guard to the daily frame, so AVWAP/sigma bands, ATR20,
  binary bounce/cross gating, two candle-shape score penalties, and an 18-point
  volume-thrust bonus (whose full-day-average denominator makes it structurally
  near-unfireable intraday) all move with today's forming D1 bar; the setup
  tracker is then written at 12:00 PT and wiped/rewritten when the 13:00 slot
  finishes ~13:20-13:28. Full mechanism list with evidence:
  `docs/SWING_QUALITY_AND_FEEDBACK_PLAN.md` §4. The trader authorized the
  "full honesty bundle" fix design; nothing is built yet.
- **"Shared scan" is a proven no-op**: `use_shared_watchlists=True/False` resolve
  to the identical watchlist paths, so its removal (Phase 0.5 R1) is a dead-flag
  cleanup, not a behavior change. **Removed 2026-08-15** — see below.

Running the frozen executable as the daily driver disabled the Master AVWAP D1
swing scan completely, for two sessions, without any visible symptom.

- **The defect.** `_run_master_scan_subprocess` spawned
  `[sys.executable, "-c", code]`. Under PyInstaller `sys.executable` is
  `TradingBotV3.exe`, so the flag and code string reached the application's own
  argument parser: `error: unrecognized arguments: -c import faulthandler; …`,
  exit code 2, **one second after each slot fired** against a scan that takes
  17-21 minutes. Every slot from 2026-08-12 07:30 through 2026-08-13 09:00
  failed; the last successful scan was 2026-08-11 13:23:59 (622 setup rows).
- **Why nobody saw it.** Everything that runs in-process was unaffected —
  BounceBot alerts fired, the 07:00 open scan rebuilt the watchlists, Auto Pilot
  wrote its reports — so the desk looked healthy. The cost surfaced one layer
  away: the overnight AI read 11 stale D1 sources and produced briefs about
  truncation.
- **The fix.** New `scripts/scan_worker.py` owns the scan invocation;
  `scan_service.scan_worker_command()` owns the transport, choosing
  `TradingBotV3.exe --run-scan <json payload>` when frozen and the unchanged
  `-c` form from source. `launch_gui.main` answers `--run-scan` before argparse,
  exactly where `--selftest` is handled. Both forms call `scan_worker.run`, so
  work and transport cannot drift apart. A malformed payload raises rather than
  defaulting — guessing would run a different scan than the one requested,
  including the setup-tracker write.
- **The guard that was missing.** `tests/test_scan_worker_spawn.py` really
  spawns a child process and waits for the completion marker, against a stub
  scanner so it stays offline. The spec-drift test inspects bundle *contents*
  and `--selftest` resolves *imports*; neither ever launched anything, which is
  why both passed while the desk could not scan. `scan_worker` is also added to
  the selftest's lazy-import roster.
- Verification: full Windows suite **2738 passed, 19 subtests**, exit 0; smoke
  **7/7**, exit 0. Eleven new tests.

### 2026-08-12 — first-night repair: roster noise, resume identity, crash-safe publish

The 2026-08-11 window was the ticker-briefs packet's owed live proof. It produced
126 briefs covering 101 of 182 symbols, never published a morning file, and exposed
three defects plus one machine fault. Advisory-only throughout: no detector,
scoring, or alert file is in the diff.

- **What the night actually did.** `ai_summary` succeeded first attempt at 22:02:53
  (~170 s, 10 usable sources) — a clean result against the previous night's six
  degraded rounds. `ticker_briefs` ran 22:04:33 → 01:20:08 with zero failures and
  was killed mid-batch. `ai_morning_brief.txt` still held the 2026-08-10 file.
- **TB-5 — a roster line is not evidence about the symbol.** `_extract_ticker_content`
  projected a text source by keeping every *line* containing the symbol, and the
  evidence files are human-readable reports full of copy-paste ticker blobs. Measured
  over the real 2026-08-11 packages: **307,630 of 319,687 projected chars (96.2%)
  were roster text**, median symbol-specific content **42 characters**, and
  `daily.master_events` contributed 174,994 roster chars against 479 chars of real
  content. Lines are now dropped when stripping ticker tokens and list punctuation
  leaves ≤15% residue, and when the line is the bare symbol (Auto Pilot's `longs`
  array is membership wearing a second hat). The residue test is deliberately not a
  ticker count: a tier row carrying eight tickers is pure signal. Measured effect on
  the same data — **166 model calls → 49**, projected payload 319,687 → 26,223 chars,
  and TB-2's membership-only skip now does what sec 6.4b scoped it to do.
- **TB-3 repaired — resume on the evidence, not on when it was read.** The manifest
  now carries a `resume_key` hashing only symbol, session, memberships, and source
  ids with their content. `evidence_hash` keeps its whole-package meaning for
  artifact identity, but it covers `generated_at` and every `as_of`, so it changed on
  every firing and the resume could never match. Manifest schema `v1` → `v2`; a row
  without a `resume_key` is regenerated, never reused.
- **Crash-safe publication.** The morning file is re-rendered and atomically
  republished after every resolved symbol, carrying an explicit
  "Run in progress at the time of writing" note that the final publish drops. A
  publish fault is logged and never costs the batch. The market-session block still
  suppresses publication outright — it is an unconditional stop for the whole job,
  and the last verified file stands.
- **Scheduled-task time limit was defeating its own concurrency guard.**
  `ExecutionTimeLimit` was `PT2H` against an 8-hour window. On 2026-08-11 the 22:00
  run was still briefing at 00:00, so Task Scheduler terminated its PowerShell parent
  and marked the task not-running, letting the 00:00 repetition start a **second**
  runner while the first instance's Python child continued. The session manifest
  records both: from 00:01:54 the rows interleave one-for-one, instance A continuing
  at list position 73 while instance B restarted from position 0, two 12B models
  resident on one iGPU, and 25 symbols briefed twice. Now `PT8H` — the window itself
  — in `scripts/register_ai_jobs_task.ps1` and on the live desk task.
- **Machine fault, not code (trader-owned).** The desk entered Modern Standby 60
  times during the window, 4h39m in total, including an unbroken 01:39:42 → 05:57:09.
  That killed the run and suppressed every task firing from 01:30 to 05:30.
- Verification: full Windows suite **2727 passed, 19 subtests**, exit 0; smoke
  **7/7**, exit 0. Seven new tests.

### 2026-08-11 — symbol snapshot popup opens at desk height

Trader ask: the chart popup that opens on a table double-click should use
essentially the full vertical space the rest of the program uses. It had opened at
a fixed 1180x760 regardless of monitor, so on the desk's screen the stacked D1 and
M5 charts were squeezed into roughly half the available height.

- `SymbolSnapshotDialog.__init__` now calls `_resize_to_desk_height()` instead of the
  hardcoded `resize(1180, 760)`. Height comes from the hosting window's
  `frameGeometry` (minus a title-bar allowance) when that window is visible, and from
  the screen's `availableGeometry` otherwise; it never falls below the old 760.
- The popup is centered horizontally on the desk window and clamped inside the
  screen's available area, so a multi-monitor desk cannot place it off-screen.
- Opening geometry only. The dialog is constructed once per panel and reused, so a
  manual resize persists across subsequent double-clicks within a session.
- Both charts already carry layout stretch 1, so the added height splits evenly
  between D1 and M5; no chart, data, or alert code was touched.
- Verification: full Windows suite **2687 passed, 19 subtests**, exit 0.

### 2026-08-11 — ticker-briefs hardening packet (TB-0..TB-4)

Armed by the trader after reading the first repaired overnight run
(`docs/LOCAL_AI_AUTOMATION_PLAN.md` sec 6.4b, PROPOSED → BUILT). Advisory-only:
nothing in this layer touches scanners, scores, watchlists, alerts, or bot state,
and no detector, scoring, or alert file is in the diff.

- **The measurement that changed the packet's premises.** `ticker_briefs` completed
  all 95 symbols in **5,962 s — ~63 s/call**, on the repaired `gemma3:12b-tbv3ctx`.
  The drafted premise of ~4.75 min/call and a window overrun is **obsolete**: there
  was no overrun. The real finding was content vacuity.
- **TB-0 — project first, budget second.** Every one of those 95 briefs was
  content-free. `run_ticker_briefs` built one base evidence package *already*
  budgeted to the local ceiling (22,000 chars) and projected each symbol out of that
  starved base, so the per-symbol-rich sources had been declared unfunded at 0 chars
  (`setups.current_tracker` 95,806, `setups.current_tiers` 77,124,
  `setups.bounce_learning` 17,995, `market.industry_intraday_rs` 17,833) and the
  funded tables sheared to about one row. MRVL's brief reads "1 of 19 requested
  source(s) usable", the one being its own watchlist membership. The base now carries
  the cloud ceiling so symbol rows survive projection, and the local budget is applied
  to each much smaller per-symbol package through `ai_summary.ration_projected_sources`
  — same unfunded/truncation vocabulary, same truncation tripwire on every local call.
  `run_daily_summary` is untouched and cloud payloads stay byte-identical.
- **TB-1 — per-ticker failure isolation and an honest partial morning file.** Each
  symbol's inference and export is its own unit, with the daily summary's single
  fed-back-error retry applied per symbol for the first time. The morning file
  publishes what completed and states `Briefed N of M. Failed: SYM (reason), …`
  before the first brief. Focus names lead the ordering, so a partial night covers
  Focus first. `ok` only when every symbol resolved; otherwise `degraded`, which the
  runner retries. A mid-batch window closure now publishes the partial instead of
  losing the night; the market session remains an unconditional stop, and the
  unreadable-watchlist refusal is unchanged.
- **TB-2 — membership-only symbols skip the model.** A symbol whose projected package
  holds nothing but `watchlists.membership` gets a deterministic one-line entry and no
  artifact set, and counts as resolved.
- **TB-3 — resumable completion.** Per-symbol completions are recorded in an
  append-only `ticker_briefs_manifest.jsonl` under
  `ai_store/briefs/<year>/<session>/`, keyed by `(session_date, symbol,
  evidence_hash)`. A re-fire regenerates only what changed, ending both the
  restart-at-symbol-1 waste and the duplicate four-file artifact sets; the morning
  file is re-rendered from the manifest, so clearing the failures upgrades `degraded`
  to `ok` on its own. An unreadable manifest regenerates rather than refusing.
- **TB-4 — per-session attempt cap.** `JobSlot.max_attempts` (3 for `ticker_briefs`,
  unlimited elsewhere) plus an identical-error early stop; on reaching either, the
  runner writes one terminal marker — an ordinary `skipped` row carrying
  `terminal: true`, deliberately not a new job status — and every later firing costs
  about a second. Only `failed` and `degraded_no_narrative` rows spend an attempt, so
  a cheap refusal from an unmounted share still self-heals, and `--force` overrides
  the marker. This ends the 11-consecutive-failure grind of 2026-08-09/10.
- Gate handling: separate five-session clocks. `ai_summary`'s clock continues; the
  `ticker_briefs` clock restarts at zero. Live proof owed at the next 22:00 window.
- **Testing-branch integration correction.** The first focused Windows gate after
  fast-forwarding the packet exposed that list evidence truncation measured retained
  rows before prepending its truthful truncation banner, allowing serialized source
  content to exceed the declared local character budget by the banner length. The
  truncator now includes the banner in its allocation. This is an evidence-packaging
  correction; detector, scoring, alert, and daily-summary call-site behavior remain
  unchanged.
- The full Windows gate also exposed a non-hermetic warehouse-tee assertion: it
  counted unrelated background `ResearchStore.open()` calls elsewhere in the pytest
  process although its contract concerns the capture object's own worker. The test
  now scopes the assertion to that worker; production warehouse behavior is
  unchanged.

### 2026-08-10 — testing-week usability and phone-report corrections

- Chart Review opens with its Setups column hidden and exposes a restore control.
- A newly opened alert receives current cached/fetched bars rather than scan-time bars.
- Best swing content can trigger a phone notification after report publication, with
  an explicit no-readable-setups quiet gate.
- The existing live market commentary journal request was recorded as roadmap item;
  it is not implemented.
- Consolidated repository guidance into implemented history, a phase-gated remaining
  roadmap, a precise current checkpoint, a classified documentation index, and a
  non-authoritative wishlist. `CLAUDE.md`/`AGENTS.md` now mandate the read/update
  sequence for every AI handoff.
- **Designated writer configured on the main desk.** `autopilot_today.txt` had not
  published since 2026-07-30 because the retired desktop was still the last recorded
  holder and no writer was named on the mini-PC; the lease correctly fail-closed
  rather than publishing from an unconfigured machine. Consequence: an entire Auto/Away
  session produced no phone digest and no swing push, since the push is tied to a
  *verified* publish. `writer_role.py --designate-self` fixed it.
- **Research warehouse enabled.** `research_store_dir` was unset, so a full session of
  capture was silently discarded. Now `\\MINI-PC\Trading Bot Data\research_lake`, with
  the sec-8.2 layout created and the machine-local spool at
  `%LOCALAPPDATA%\TradingBotV3\research_spool`.
- **Overnight AI jobs repaired.** Three independent faults, all found by reading the
  job ledger rather than the scheduler's hex code:
  (a) the task ran `pythonw.exe`, a GUI-subsystem binary, and exited `0xC0000142`
  with its stdout/stderr discarded — now a logged PowerShell wrapper
  (`scripts/run_ai_jobs.ps1`) over console `python.exe`, with the runner's real exit
  code propagated and both streams captured to `%LOCALAPPDATA%\TradingBotV3\logs\`;
  `register_ai_jobs_task.ps1` updated so re-registering cannot reintroduce it;
  (b) `ticker_briefs` had failed six consecutive nights with truncated JSON because
  the local server capped prompts at 2,048 tokens while the app sends up to 80,000
  chars of evidence — the medium tier now points at a derived `gemma3:12b-tbv3ctx`
  (`num_ctx 12288`), measured at 6,147 prompt tokens against 2,051 before;
  (c) after those failures the job then *skipped* every remaining run for reserving
  120 min against a shrinking window, so the ledger showed skips and hid the failures.
- **Local AI summarization made truthful about its own limits.** The evidence cap
  is now resolved per call site (`evidence_budget_for`): local calls use
  `ai_local_evidence_budget_chars` (default 22,000, derived from the 12288 context
  minus generation and scaffold, with headroom for the retry), while
  `MAX_TOTAL_EVIDENCE_CHARS` (80,000) stays the cloud ceiling — cloud request
  payloads remain byte-identical, test-asserted. A truncation tripwire compares the
  server's reported `usage.prompt_tokens` against what was sent and raises a named
  error instead of parsing output built on a sheared prompt; it is silent when the
  server omits usage, and raises rather than retries because a retry sends more.
  Token usage now reaches the job ledger for the daily summary and per-ticker slots.
  The local large tier is retired: 27B-class models no longer load beside the
  running desk on the 780M, so policy drafts and retros move to the frontier model.
  A Phase 2 design packet (sec 6.4a) is proposed and awaits trader sign-off; no
  digest schema was built or frozen.
- **Cloud sync removed from the system (decision 0015).** Google Drive/OneDrive are no
  longer part of the design. `C:\TradingBotData` is a plain local folder at the same
  path; the DAS `\\MINI-PC\Trading Bot Data` is the durable tier. Decisions 0005, 0006
  and 0014 carry superseded/amendment banners rather than being rewritten, since the
  mechanisms they justify still exist. Documentation and comments only — 2647 tests
  and 7/7 smoke unchanged. Known consequence: cloud sync was the only off-site copy of
  the Class A backup set, so off-site redundancy is now an explicit open gap.
- **BounceBot's intraday sweep is confined to the session window** (trader-directed,
  2026-08-10). Auto Pilot re-enabled scanning on every 30-second tick with no clock
  check of any kind, so a desk left running swept the watchlists — 95 to 150 names,
  measured at roughly eight full sweeps an hour — straight through the night and the
  weekend against prices frozen since the close. `bouncebot_scanning_due` now derives
  a window from `market_session` (session ± configurable 30-minute warm-up and
  wind-down; 06:00-13:30 on a normal Pacific session) and the sweep pauses outside it.
  The pause is `set_scanning_enabled(False)`, whose branch in the strategy loop skips
  `ensure_connected` and every symbol request, so the IB traffic stops while the
  connection stays up and the open needs no reconnect. Three details are deliberate:
  the check runs *before* the tick's weekend and Auto-Pilot-OFF short-circuits, which
  are the paths that would otherwise let a Friday sweep run all weekend; it acts only
  on a boundary transition, so a deliberate manual resume holds instead of being undone
  one tick later; and it fails **open**, because an unanswerable session lookup must
  never be the reason the bot sits out a trading day. Settings
  `qt_bouncebot_scan_session_only` (default on),
  `qt_bouncebot_scan_preopen_minutes`/`qt_bouncebot_scan_postclose_minutes` (default
  30). No detector, score, threshold, or alert rule changed — only *when* the existing
  scan runs. `SCAN_OUTSIDE_MARKET_HOURS` in `bounce_bot_lib/legacy.py` was left exactly
  as it was: it gates per-symbol bounce *detection*, not the data fetching that
  produces the traffic, so flipping it would have cost detection without saving requests.

### 2026-08-09 — testing-week integration, chart completion, and frozen proof

- Integrated chart performance, Chart Review capture, warehouse Phases 1–8 and
  defect repairs, A3 shared chart, A4 paint lines, A5 click-to-arm, Local-AI Phase 1
  completion, and capture-stream hardening.
- Added packaging spec-drift coverage and frozen self-test. The real Windows build
  exposed the excluded-`ai_jobs` contradiction; the roster and disjointness guard
  were corrected.
- Desk surveys confirmed 62/62 stored trendlines projectable/fresh and 0/171 red
  horizontal levels clearing the shared strength threshold across three symbols.
- Recorded the Windows desk gate: 2611 passed, 7 subtests, smoke 7/7, frozen 29/29.

### 2026-08-08 — single-main topology, durability, local AI, and captured judgement

- Retired the Desk Link/satellite and separate mini-PC operating roles; the Ryzen
  desk became the sole always-on scan and AI host.
- Built and repaired durability steps 1–4, including tracker-vintage honesty,
  bounded recovery, and frozen-process launch protection.
- Built Local-AI Phase 0 and the scheduled Phase 1 foundation, then hardened evidence
  budgets, missing-source reporting, session identity, and publication rules.
- Built Chart Review decision capture, veto vocabulary/cohorts, and the workspace
  shell; added chart background loading and stall protection.
- Reviewed/merged packaging work and recorded the testing-week branch.

### 2026-08-03 to 2026-08-04 — remote surfaces and research warehouse

- Consolidated Auto/Away output into one swing-first verified phone digest and added
  main-origin price alerts over ntfy.
- Implemented all three Desk Link tiers, then later retired the topology on 2026-08-08.
- Locked the Ultimate Setup Intelligence Database design and implemented warehouse
  Phases 0–8 plus two review/defect passes.
- Added the DAS research-lake storage class, immutable store, capture/aggregation,
  features, occurrences/outcomes, readout, backups, Health integration, and job
  invokers without production influence.

### 2026-07-30 to 2026-08-02 — observability, live controls, and platform support

- Finished Milestone-1 observability packets: champion-invariance guards, enforced
  fixture contracts, shared diagnostics I/O, shadow coverage/retention, writer
  coordination, honest Health, lifecycle ownership, review sharding, provider
  telemetry, and first-session runbooks.
- Added regime evidence collection, breadth ledger, Technical Integrity outcomes,
  stale-tail D1 recovery, and off-GUI Health work.
- Added DESK/AWAY/EVENING workflows, previous-day gates, chart watches, Focus review
  actions, phone price alerts, and the flagship post-earnings candle break.
- Added macOS setup, CloudStorage/Keychain support, machine-local path normalization,
  UI scaling, and the now-retired Desk Link implementation.

### 2026-07-22 to 2026-07-29 — chart-first desk and review learning

- Added broader RS/industry measurements, market internals, recalibrated Technical
  Integrity, and expanded Auto candidate coverage.
- Built D1/M5 snapshot charts across setups, RS, and industry surfaces, then added
  chart navigation, log scale, forming D1 preview, AVWAP/SMA overlays, and caching.
- Built the visual review queue, armed chart watches, D1 Focus toggles, alert dock,
  strength tape, and persistent D1 event alerts.
- Added review-event capture, preference scoreboard, AI policy handoff, annotation-
  only guidance, Focus strength board, and Phase-0 learning audits.
- Fixed recurring Python/Qt crash paths with faulthandler and GUI-thread GC ownership.

### 2026-07-10 to 2026-07-18 — runtime foundation and product trust

- Restored a deterministic baseline, removed dormant defects, added smoke checks,
  manifests, lifecycle ownership, job ledger, heartbeat, writer lease, and verified
  Away publication.
- Added pure SPY state, aligned RS, CandidateRegistry, and Greatness engines; wired
  SPY and Greatness only in shadow.
- Added trustworthy Industry Board refresh, Master opportunity dedupe, Auto-vs-user
  environment separation, automatic Entry Assist, final-upgrade D1 Focus, novice
  explanations, journal v2, provider-neutral A.I. Summary, and advisory industry RS.
- Reworked day-trade evidence, RVOL, setup tiers, D1 zone/rubric feeds, and daily-
  trend gates while retaining golden/evidence controls for behavior changes.

### 2026-07-01 to 2026-07-08 — research breadth and early Auto Pilot

- Expanded study/playbook families, tracker replay, industry indexes, universe
  building, broker journal imports, and Qt Universe/Industry surfaces.
- Added the setup encyclopedia, Bounce learning tiers, Expected-R ranking spine,
  Alert Command Center, delayed ORB/EMA/VWAP workflows, Auto Pilot, self-healing
  universe, outcome measurement, tracked auto watchlists, and pick feedback.

### 2026-06 — durable data, Qt desk, and Focus Picks

- Added durable D1/H1 stores, gap-aware delta fetching, cache warming, multi-year S/R
  levels, industry/HTF/cloud/structure studies, and theta support improvements.
- Began the Tk-to-PySide6 migration and built the Qt Trading Desk.
- Added FocusPickStore, top-level Focus UI, Master/Bounce integrations, D1 upgrade
  gates, human focus tracking, and the Human Picks tracker.
- Adopted Google Drive as the default operational shared-home pattern.

### 2026-03 to 2026-05 — unified desktop workflow

- Consolidated the AVWAP and BounceBot GUIs, shared home-folder watchlists and data,
  market-session scheduling, ranking/tracker tools, local caches, and swing lists.
- Added the original mini-PC scheduler (retired 2026-08-08), tracker synchronization,
  theta candidate ranking/explanations, D1 watchlist integration, and expanded Market
  Prep/AI reporting.

### 2026-02 — intraday and market-context expansion

- Integrated RRS into BounceBot, configurable EMA bounce monitoring, all-symbol
  bounce checks, sector/industry classification, earnings-gap anchors, GUI controls,
  and anchor persistence.

### 2026-01 and 2025-11/12 — initial system

- Established Master AVWAP and BounceBot, earnings-anchor refresh, AVWAP cross/bounce
  events, signal exports, yfinance fallback, grouped output, and early historical
  evaluation/labeling/trade-outcome tooling.
- Added moving-average and D1 summaries, TickerMover integration, trade logging, and
  the first dependency/runtime structure.

## Retired or superseded implementations

- Desk Link satellite relay/control and the separate mini-PC scanner role are retired
  as of 2026-08-08. The code remains pending a scoped cleanup.
- H1 alerts were retired; H1 now confirms D1 tracker picks.
- The old DESK approval queue for auto-populate was superseded by direct day-scoped
  M5 Focus adoption.
- The legacy shared review-event ledger is read-only; per-installation shards are the
  current writer path.
- The legacy Tk UI remains only for migration compatibility and is not the product
  direction.
- Historical plans and handoffs listed as such in `docs/README.md` are evidence, not
  current execution authority.
