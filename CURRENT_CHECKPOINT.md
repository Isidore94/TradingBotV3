# Current checkpoint

This file is the frequently refreshed active-work, branch, and verification stamp.

- Implemented inventory and revision history: [`CHANGELOG.md`](CHANGELOG.md)
- Remaining work and gates: [`plan.md`](plan.md)
- Supporting-document roles: [`docs/README.md`](docs/README.md)

---

## 2026-08-27 (morning) - trader rule BUILT: with-trend regime-pause rows auto-join M5 Focus; queue scan done

**Branch `claude/gui-phase-0-9`** (this session, on top of `fd76923`). Trader,
07:20: "I've been doing nothing but managing the bot all morning. There are
too many trades. New rules. 1. M5 holding highs on bullish days and M5 holding
lows on bearish days are auto added to the M5 focus lists. Then do a scan and
see what is the other primary type of chart recommended."

### Rule 1 - BUILT

`scripts/regime_pause_focus.py` (pure rule) + `AlertCenterPanel._auto_focus_regime_pause`
in `add_alert`. Detail and the exact behaviour boundary are in `CHANGELOG.md`
2026-08-27 and plan.md Phase 0.5 item 11. Detector, sweep and hold measurement
untouched. The alert-panel edit was trader-directed in chat (the ask-first
rule's authorisation).

| Measure | Value |
|---|---|
| Tests added | 30 (`test_regime_pause_focus.py` 18, `test_qt_regime_pause_auto_focus.py` 12) |
| Fail-before-fix | panel change stashed: 3 of 12 fail on the assertion, 9 pass (the "stays on the queue" cases hold either way) |
| Full suite | **5046 passed / 19 subtests, exit 0** (299 s; 5016 before this change) |
| `scripts/smoke_check.py` | 7/7 |
| Packaging trigger | none (new module is a plain `scripts/*.py`, collected by the existing rule; no dependency, asset or `__file__` change) |

**Live gate owed:** one DESK session on a directional day - rows land in Focus
with no chart, "Not today" from the Focus surfaces still removes them, and a
count of charts saved.

### The scan - what else fills the queue

`alert_review_events`, 06:33-07:19 today (46 min): **124 charts shown**, one
every 22 s; 40 skip, 60 "Not today"; 23 hidden / 74 waiting at 07:09.

| Chart type | Shown | Share |
|---|---:|---:|
| D1 flags - `d1_flag_long/short` (Master AVWAP D1 scanner) 41 + `focus_d1_event` 26 | 67 | 54% |
| M5 `lrsi_cross_20` / `lrsi_cross_50` | 25 | 20% |
| Regime-pause "holding highs" (now auto-Focus with-trend) | 21 | 17% |
| Armed chart watches | 11 | 9% |

Behind the `focus_d1_event` count: the auto-populate slot adopted **69 auto
picks into M5 Focus at 07:09** (20 "Bullish-day weakness", 13 "RS vs SPY", 36
PDH/PDL breaks; 47 shorts / 25 longs on a `bullish_weak` day), and every Focus
name is watched for every D1 event kind, which raised **102 `focus_d1_flag`
rows on 95 names** in the same window. The other primary chart type is the D1
flag; the LRSI cross is second. **Nothing changed for either** - the trader
decides whether to gate, cap or fold them, and which of the three auto-pick
families is worth 69 Focus names on a bullish morning.

---

## 2026-08-27 - ACTIVE: Phase 0.9, G-P2.0..G-P2.2 BUILT and committed; **SOAK 1 is the gate on G-P2.3**

**Branch `claude/gui-phase-0-9`, tip `fd76923`.** Governing document
`docs/GUI_REDESIGN_PLAN_2026-08-25.md` (§12, §8.3, §5.3, §15 decisions 9/10/14);
build prompt `docs/prompts/GUI_PHASE_0_9_OPUS_PROMPT.md`. Presentation and
threading only: no detector, scorer, alert, queue, scheduler, evidence-stream or
storage behaviour changed, and no read was added or removed.

### Verification

| Measure | Value |
|---|---|
| Baseline before the packet (`cc7dffa`) | 5010 passed / 19 subtests, exit 0 |
| After G-P2.0..G-P2.2 | **5016 passed / 19 subtests, exit 0** |
| `scripts/smoke_check.py` | 7/7 |
| Tests added | **37**, every one proved failing on the un-fixed code |
| Packaging trigger | none (no new dependency, asset, top-level package or `__file__` change) |

### Landed

| Commit | What |
|---|---|
| `1fd9e6e` | G-P2.0 - the §12 width rule through one shell, plus middle elision |
| `a5fa6a9` | G-P2.1 - AWAY Recap as a return surface |
| `fd76923` | G-P2.2 - `Ctrl+J` to the Desk Journal (fenced file; trader approved the diff in chat first) |

### How fail-before-fix was proved, per file

- `test_table_width_rule_pages.py` (2 tests) - fails on the ASSERTION with the
  three source files stashed: the cohort column and the `Line` column do not
  stretch. It exists separately from `test_table_width_rule.py` precisely so
  there is a page-level behavioural failure and not only an import error.
- `test_table_width_rule.py` (14) - fails at import, the helper not existing.
- `test_away_recap_return_surface.py` (15) - all 15 fail with the panel stashed.
- `test_desk_journal_route.py` (6) - all 6 fail with the panel stashed.
- `test_away_day_recap.py::test_a_blank_symbol_asks_for_no_chart` was
  STRENGTHENED rather than left alone: it would otherwise have passed because
  the blank-symbol row is now hidden, which is a different claim from "a blank
  symbol charts nothing". It now reveals the status rows first.

### Two things a later reader needs

1. **`measure_column_widths` is still `resizeColumnsToContents()`** - the 7.9% /
   115 s site of the 2026-08-26 measurement - and G-P2.0 now reaches it from two
   more pages. That is deliberate: it is ONE seam, and G-P2.3 item 1 bounds it
   there. **Do not soak-judge table cost until item 1 lands.**
2. **`Ctrl+J` must stay the only binding of that sequence.** Two live bindings
   for one sequence is an ambiguous shortcut and Qt fires NEITHER, silently.
   `test_desk_journal_route.py` greps every `QKeySequence("...")` under
   `scripts/ui` and fails if a second one appears.

### SOAK 1 - OWED, and it is the gate on G-P2.3

Not dischargeable by any test run. Work a normal session with the stall watchdog
on (`ui_stall_watchdog: true`), then run the
`docs/GUI_FLUIDITY_MEASUREMENT_RUNBOOK.md` command with `--compare` against
`%LOCALAPPDATA%\TradingBotV3\diagnostics\ui_stalls_prefix_baseline_2026-08-26.jsonl`
and record stalls / median / p90 / worst / total blocked in this file before
G-P2.3 begins. Archive `ui_stalls.jsonl` first or the session is compared
against itself.

| Baseline (2026-08-26 pre-fix, attended 06:15-10:25 window) | |
|---|---:|
| Stalls over 50 ms | 2000 |
| Total blocked | 1129 s (~19 min in ~4h10m) |
| Worst single stall | 49.25 s (`theta_table_model.py:72`) |
| SOAK 1 result | **not yet run** |

### Owed after the soak

- **G-P2.3**, in the measured order: bounded `fit_columns` measurement; the
  Theta refresh (explain the 3.0 s -> 26.6 s -> 49.2 s growth FIRST);
  `watchlist_utils.read_watchlist_symbols` off Qt; the `project_paths` `stat`
  MEASURED before touched; then Setups and `setup_tracker_panel` only, with the
  G-P1.6 panel-thread sweep riding along. SOAK 2 after it.
- **G-P2.4**, the GC measurement packet - measurement only; no
  `_GuiGcController` scheduling change is authorized.
- The Phase 0.8 live soak, still the trader's to run.
- A live desk session confirming the three surfaces above render on real data.

### Process finding, repeated here because it cost real time

Two Opus sessions ran in ONE checkout on 2026-08-26 and the Phase 0.10 session
stashed this session's in-flight G-P2.1 work to get a clean tree; it was
recovered from the stash (including the untracked test file, via `stash@{0}^3`)
and nothing was lost. The Phase 0.10 review-fix commits `ac9a952..cc7dffa` are
consequently in `claude/gui-phase-0-9`'s history rather than on the AVWAP
branch. **One build session per checkout; a second needs `git worktree`.**

## 2026-08-27 - ASSESSMENT (Fable) of both overnight Opus runs: GO; G-P2.2 landed mid-review; docs owed for G-P2.0/2.1/2.2

Reproduced on a detached worktree at the committed tip `cc7dffa`:
**5010 passed / 19 subtests, exit 0** (279 s) - the builder's number.

**Phase 0.10 review fixes (`ac9a952`) - verified at source and by reproduction.**
The shadow CSV write in `export_setup_tracker_views` is inside
`try/except` + `logging.warning`, shadow write only. The fence guard
(`tests/test_band_variant_fence_guard.py`) was pointed at `5613eec`'s
`legacy.py` here, independently: **6 unfenced readers before the fence, 0 now**,
9 readers seen, a two-entry documented allowlist (the stop rebuild on replay,
sealed-record compaction). The four-template cut is in `_build_tracker_scenarios`
via `_is_band_variant_stop`. B-4 is unblocked.

**Phase 0.9 G-P2.0 (`1fd9e6e`) - built and pinned.** One shell
(`data_table.apply_width_rule` + `MiddleElideDelegate`), applied to the six
`DataTable` users and to the raw `QTableWidget`s on AWAY Recap and Weekend
Prep; tests cover 1680 vs 2304 widths, tail-preserving deterministic elision,
tooltips. **Caveat, not a defect:** `measure_column_widths` still calls
`resizeColumnsToContents()` - the 7.9% / 115 s site - and G-P2.0 now reaches it
from two more pages. That is G-P2.3 item 1, which is next; do not soak-judge
table cost until it lands.

**Phase 0.9 G-P2.1 (`a5fa6a9`) - built and pinned.** Hide-and-count of
scanner-status rows, `Chart ▸` cell + `Enter`, symbol-less rows muted/italic
through a theme token (`setForeground`, no stylesheet). The Alert Center's list
untouched.

**Phase 0.9 G-P2.2 - committed as `fd76923` by the GUI session while this
assessment was being written** (it was an uncommitted fenced edit when the
review began, i.e. the session had stopped at the ask as instructed).
`alert_center_panel.py` carries the Ctrl+J route exactly in the prompt's shape
(panel scope, `WidgetWithChildrenShortcut`, label hint, no row) with
`tests/test_desk_journal_route.py`. It passes 6/6 - after one unexplained 6/6
failure on its first run here (4.3 s, then green three times at 1.3 s; likely a
collision with the other session still finishing). Watch it in the first full
suite that includes it; the 5010 reproduced above is the tip BEFORE it.

**Owed by the GUI session, because it stopped at the ask before its handoff:**
CHANGELOG entries for G-P2.0/G-P2.1 (the header mentions them; there is no
entry), `plan.md` Phase 0.9 Built stamps, and the §13 build status in the
redesign plan. Then **SOAK 1** (stall watchdog on, compare against
`ui_stalls_prefix_baseline_2026-08-26.jsonl`), then G-P2.3.

**Process finding, recorded so it is not repeated:** the two Opus sessions ran
in ONE checkout at once; the Phase 0.10 fix session found the GUI session's
in-flight G-P2.1 work red in the tree and stashed it to measure. Nothing was
lost, but a stash of someone else's work is a coin toss. One build session per
checkout; a second needs `git worktree`.

## 2026-08-26 night - Phase 0.10 review fixes APPLIED; B-4 is unblocked

**Branch `claude/gui-phase-0-9`** (cut from `292e335`, so the Phase 0.10 build
is in its history). Commit `ac9a952`. Both fixes the review below owed are done,
and the growth decision it recorded is taken.

| Measure | Value |
|---|---|
| Full suite, tip `714f717` | **5010 passed / 19 subtests, exit 0** |
| Full suite at `ac9a952`, before G-P2.1 landed under it | 4995 passed / 19 subtests, exit 0 |
| `scripts/smoke_check.py` | 7/7 |
| Tests added here | 11, every one proved failing first |

**1. The shadow export is guarded.** `export_setup_tracker_views` now wraps the
band-variant CSV write in `try/except Exception` + `logging.warning`. Only the
shadow write: every champion export above it is already on disk by then, and a
champion export that fails must still fail loudly - which is its own test.
Proved by a raising `build_band_variant_stats_rows`: all nine champion CSVs
still written, no shadow CSV (absent rather than half-written), the failure
logged, and `update_setup_tracker_from_scan` still reaching
`save_setup_tracker_payload` with the payload it was handed. That last one runs
through the REAL export, because the seam between the two is the thing under
test.

**2. The fence is guarded at source.** `tests/test_band_variant_fence_guard.py`
walks `legacy.py`'s AST: every scenario-iteration site must mention
`_is_band_variant_scenario` inside its enclosing function or be named in
`ALLOWED_UNFENCED` with its reason. Two entries, both readers that MUST see the
shadow - the stop rebuild on replay and sealed-record compaction. The detector
is wider than the spelling the fence was written against (`setup["scenarios"]
.values()`, `.get("scenarios", {}).values()` and a local
`working_scenarios.values()` all count), because a guard that only knows today's
spelling is passed by tomorrow's: it finds **9** readers where the narrow
pattern finds 6.

Fail-before-fix on real code rather than a mutation: pointed at
`5613eec:legacy.py` - the tree as it stood before the fence - the guard reports
**six** unfenced readers (`_flatten_tracker_scenarios`,
`_summarize_tracker_setup_outcome`, `_tracker_short_horizon_risk_per_share`,
`build_tracker_playbook_rows`, `build_tracker_setup_record`,
`recompute_tracker_setup_record`). Four companion tests keep the guard honest:
it must still see the known readers, its allowlist may not name a function that
no longer reads scenarios or carry a one-word reason, no read may sit outside a
function, and the fence helper must still exist under that name. **What it does
not claim**: that mentioning the helper means it was used correctly. The parity
fixture is what proves the values did not move.

**3. The shadow crosses the four BASELINE exit templates only** - the trader
decision the review recorded, taken. `_is_band_variant_stop` is the
candidate-side twin of `_is_band_variant_scenario`, kept beside it so the two
spellings cannot drift; `_build_tracker_scenarios` skips experimental templates
for such a stop. The champion still crosses all six. Re-measured:

| | Before | After |
|---|---:|---:|
| Bytes per new setup | 9,982 | **6,524** |
| Forward growth at 14,386 setups / 950.2 MB | ~144 MB (15%) | **~89.5 MB (9.4%)** |
| Once sealed-record compaction strips the event logs | - | 5,739 |

All four baseline templates are still present, so the stats table's
per-template pairing stays possible.

**Note on this checkout's state, because it explains two numbers.** The working
tree carried uncommitted Phase 0.9 G-P2.1 AWAY-recap work
(`away_recap_panel.py`, `test_away_day_recap.py`, an untracked
`test_away_recap_return_surface.py`) that was not this session's, and it was RED
on arrival - 14 failures / 18 errors, all in the AWAY-recap and Qt-page tests,
none of them in Phase 0.10. It was stashed so the review fixes could be measured
against a green tree; that is the **4995** figure, taken at `1fd9e6e` plus the
fixes. While this session worked, a concurrent one finished and committed that
work as **`a5fa6a9` "Phase 0.9 G-P2.1: AWAY Recap as a return surface"**, which
now sits directly under `ac9a952`. The suite was re-run on the combined tip:
**5010 passed, exit 0** (the +15 are G-P2.1's own tests). Nothing in these fixes
touches that panel, and nothing of the other session's work was lost.

**Owed, unchanged:** T4's three criteria, >= 20 sessions of forward accrual with
>= 40 finalized setups before T3 counts, and B-4 (the T1 level-quality backfill
and the T2 playbook re-run) - which the review's two fixes were the gate on, and
which is now unblocked.

---

## 2026-08-26 night - REVIEW of Phase 0.10 B-0..B-3 (Fable): GO with two fixes owed before B-4

Reviewed `002f2a3..292e335` by reproduction on a detached worktree, not from the
handoff: `pytest tests/ -q` **4968 passed / 19 subtests, exit 0** (234 s),
matching the builder's number. Verified at source: the parity fixture
`tracker_record_band_variant_parity_v1.json` was frozen in `5613eec` and is
byte-unchanged by `603333b` (only its builder script gained the two shadow
blocks), and its `numeric_tolerance` is 1e-9, so "every pre-existing key
unchanged" is effectively exact; the runner's previous-anchor block uses the
previous anchor's own index (`anchor_idx` is rebound at `runner.py:817`);
`stop_source_type` is carried through the recompute path
(`_extract_tracker_stop_candidates_from_setup`), so the fence holds after a
rebuild; the overlay group is in `GROUPS_HIDDEN_BY_DEFAULT` and the prefs file
keeps it off unless `shown_groups` names it. The branch base is
`d30b732` (the `gui-p1-fluidity` tip), and `claude/gui-phase-0-9` was cut from
`292e335`, so the two build branches are linear, not divergent.

**Two fixes owed, both small, before B-4 starts:**

1. **The shadow export can cost the tracker save.** `export_setup_tracker_views`
   (`legacy.py:11099`) writes `master_avwap_band_variant_stats.csv` as its last
   statement with no guard, and its caller (`:11541`) runs
   `save_setup_tracker_payload` AFTER it (`:11552`). A raising
   `build_band_variant_stats_rows` - one malformed setup dict is enough - would
   abort the day's tracker save. R10's rule: an evidence store is never allowed
   to cost the thing it records. Fix: `try/except Exception` +
   `logging.warning` around the shadow write only, and a fail-before-fix test
   that a raising builder still leaves the champion CSVs written and returns.
2. **The fence is a hand-maintained list.** Seven readers filter on
   `_is_band_variant_scenario`, three of them found by the fixture rather than
   by reading - which is exactly why an eighth reader will not be found by
   reading either. Fix: a source-level guard test in the shape of
   `test_shutdown_waits_are_bounded.py`: every
   `(setup.get("scenarios") or {}).values()` iteration site in `legacy.py`
   either references `_is_band_variant_scenario` within its enclosing
   function or is on a documented allowlist (the per-bar grader, the stop
   rebuild at `:1414`, the sealed-record compaction at `:11237` - sites that
   MUST see the shadow scenarios).

**One trader decision, recorded here, not made here:** the tracker JSON grows
9,982 bytes per new setup (~144 MB forward on the ~950 MB file) because six
variant scenarios are built per setup while the stats table pairs on ONE exit
template (`_band_variant_paired_scenarios`). Building the variant stop for the
four non-experimental templates only is the builder's one-line cut (~96 MB);
building it for the paired template only is ~24 MB. Recommendation: the
four non-experimental templates - keeps a per-template comparison possible at a
third less cost.

**Carried into the B-4 prompt, not a fix:** the builder's fairness finding -
"a wider band gets stopped out less often" is false when entry is outside the
band (the fixture's short: challenger stop 0.16 vs champion 0.97 from entry) -
so T1/T3 stop metrics condition on the entry's position relative to each band.

## 2026-08-26 - ACTIVE: Phase 0.10 AVWAP band challenger, packets B-0..B-3 BUILT

**Branch `claude/avwap-band-challenger`, off `claude/gui-p1-fluidity` at
`88a34b7`.** Governing spec `docs/AVWAP_BAND_VARIANT_STUDY.md`; build prompt
`docs/prompts/AVWAP_BAND_CHALLENGER_OPUS_PROMPT.md`. Shadow only:
`calc_anchored_vwap_bands` is untouched and frozen (decision 0008), and nothing
in this packet reaches a detector, score, rank, tier, alert, zone arm, Focus
list, review queue or `review_policy.json`.

### Verification

| Measure | Value |
|---|---|
| Baseline before the packet | 4902 passed / 19 subtests, exit 0 |
| After B-0..B-3 | **4968 passed / 19 subtests, exit 0** |
| `scripts/smoke_check.py` | 7/7 |
| `launch_gui.py --selftest` | **71/71** (was 70/70; `indicators.avwap_band_variants` added) |
| `test_packaging_spec_drift.py` | 17 passed, **no spec edit needed** |

### Landed

| Commit | What |
|---|---|
| `002f2a3` | B-0 - `scripts/indicators/avwap_band_variants.py` + the OKTA golden fixture + 21 tests |
| `13505d1` | B-1 - `scripts/avwap_band_variant_fit.py`, the hover-comparison table |
| `5613eec` | B-2 step 0 - the tracker parity fixture, frozen BEFORE either fenced file was touched |
| `603333b` | B-2 - the tracker shadow, its fence, the stats CSV and the panel tab |
| `3abf61d` | B-3 - the D1 overlay group, default OFF |

### The one place the prompt was wrong, and what was authorized

The prompt pre-authorized appending a `VARIANT_*` stop candidate after every
champion candidate, on the reasoning that `representative_total_r` is picked by
label and so nothing moves. `representative_total_r` did not move. Eight other
values did, and they reach a live score:
`_summarize_tracker_setup_outcome` averages `total_r` across every tradeable
non-experimental scenario, and that average feeds
`build_tracker_setup_type_rows` -> `apply_tracker_setup_type_adjustments` ->
`row["score"]`.

Measured on the frozen parity fixture before the fence existed:

| Key | Before | With the naive append |
|---|---:|---:|
| `avg_total_r` | -0.0790 | -0.0755 |
| `tradeable_scenario_count` | 8 | 12 |
| `daily_marks[1].scenario_events` | 10 | 15 |
| short `setup_status` | CLOSED | OPEN |
| scenario + stats CSV rows | 12 | 18 |

**The trader authorized the fence on 2026-08-26** ("Yes, add the filter").
`_is_band_variant_scenario` now filters seven readers: the outcome summary, the
scenario CSV flattener (which also feeds `master_avwap_setup_stats.csv`), the
attribute flattener, the short-horizon risk pick, `setup_status` in the record
builder AND in the forward replay's open/closed counts, and the per-bar daily
mark's `scenario_events`. The shadow is still GRADED - the per-bar evaluator
runs for it exactly as before. Three of the seven sites were found by the
fixture rather than by reading the code.

### Tracker JSON growth, measured

**9,982 bytes per NEW setup** - 474 for the two anchor blocks, 9,508 for six
variant scenarios and their event lists - against a live
`master_avwap_setup_tracker.json` of **950.2 MB holding 14,386 setups**. That is
about **144 MB, ~15%**, if every setup carried it; it accrues FORWARD only, so
existing records do not grow until they are rebuilt. The study estimated "a few
hundred bytes per setup" and was ~30x low. **Trader decision available:** capping
the shadow to the four non-experimental exit templates would cut it by a third
and is a one-line change. Not made unilaterally.

### Two findings that change how B-4 must be designed

1. The challenger's sigma is **1.339 where the champion's is 0.586** seven
   sessions after an anchor - 2.3x. That is why the trader's OneOption
   screenshots looked better early, and it is pinned as arithmetic in the parity
   test.
2. **"A wider band is stopped out less often by construction" is only true when
   entry sits INSIDE the band.** On the parity fixture's short - entered above
   both upper bands - the wider sigma pushes the upper band UP toward entry and
   the challenger's stop lands 0.159 away where the champion's is 0.971. Six
   times TIGHTER, from the wider formula. T1's touch/respect metrics and T3's
   stop-out rates must both be cut by the entry's position relative to the band.

### Owed

- **T4's three criteria in full.** T3 needs >= 20 sessions of forward accrual
  with >= 40 finalized setups before it counts; nothing has accrued yet.
- **B-4 is the next packet and is NOT started**: the T1 level-quality backfill,
  the T2 playbook re-run, then the warehouse columns.
- A live desk session with the "Band Variant" tab and the "AVWAP sigma variant"
  paint-lines group switched on, to confirm both render on real data. Neither is
  exercised by anything but tests today.
- The frozen-exe rebuild is **not** owed: the desk launches from source by
  trader decision (2026-08-26) and the unfrozen selftest covers the new lazy
  import. It becomes owed the moment the exe is production again.

### Immediate next action

The trader reviews this packet (the handoff was written for a Fable review pass).
B-4 waits on that review by the prompt's own instruction. The Phase 0.8 live
soak below is unchanged and still the trader's to run.

---

## 2026-08-26 - ACTIVE: GUI fluidity Wave P1 (Phase 0.8), every code item built; the live soak is what remains

**Branch `claude/gui-p1-fluidity`, off `main` at `53b9733`.** The trader
authorized **Wave P1 only** from `docs/GUI_REDESIGN_PLAN_2026-08-25.md` §11.1 on
2026-08-26 and it is now `plan.md` Phase 0.8 (items G-P1.0 … G-P1.7). Waves
U1-U3, S1 and Snappy P2 remain PROPOSAL and are NOT authorized.

**Evening update (2026-08-26): the trader authorized all changes** from the
reviewer's reconciliation. Applied: the proposal revision, CHANGELOG and this
file committed; CLAUDE.md/AGENTS.md corrected (arm bar under the chart; SAC
reads OFF, source launch stays production by trader decision; new rule - chat
messages to the trader are written VERY simply, five-year-old level);
`trading_desk.cmd` header corrected; **`plan.md` Phase 0.9** added (G-P2.0
table width rule, G-P2.1 AWAY Recap, G-P2.2 Desk Journal route, G-P2.3 next
fluidity slice in measured order + thread sweep, G-P2.4 GC measurement packet
- measurement only). Nothing in Phase 0.9 is built. Verification for this
evening's commits: docs, CLAUDE.md and a `.cmd` comment only - the 4902 / 19
subtests baseline is unchanged and was not re-run.

### Immediate next action

**The §11.3 live soak - the trader's to run.** Every code item in Wave P1 is
built. Enable `ui_stall_watchdog`, work a normal session, then compare the log
against the 2026-08-25 capture. Stall records now carry an `interaction_id`, so
an event-loop-only sample names the click behind it.

After that, the largest remaining known cost is the **eight panels listed under
G-P1.5 that still read on the Qt thread** - `setup_tracker_panel` first. That
is new work, not unfinished work: the audit is complete and says exactly which
pages need it.

### The measured PRE-FIX baseline, and how to compare against it

The stall watchdog was **already enabled** on this machine - the setting did not
need changing. Its log has been archived to
`ui_stalls_prefix_baseline_2026-08-26.jsonl` in the machine-local diagnostics
directory, so the next session writes a clean `ui_stalls.jsonl` that is entirely
post-fix. Nothing was deleted (an earlier `ui_stalls_prefluidity_2026-08-21.jsonl`
archive sits beside it).

**This morning's real session (2026-08-26, all pre-fix code)** is a far better
baseline than the 45-minute sample in the proposal. **Window correction
(reviewer, 2026-08-26 evening):** the archive's 2026-08-26 rows run from
**00:00 to 10:25**, not 06:15 - the desk ran overnight. 1212 stalls / 303 s
fell before 06:00 on an unattended desk (idle loop + GC), 138 / 26 s in
06:00-06:15; the attended 06:15-10:25 window is **2000 stalls / 1129 s in
~4h10m**. The totals below are the whole-day figures; no ranking changes
under either window.

| Measure | Pre-fix (2026-08-26 session) |
|---|---|
| Stalls >50 ms | **3350** |
| Median blocked | 169.8 ms |
| p90 | 617.9 ms |
| p99 | 3771.5 ms |
| Worst single stall | **49.25 s** |
| Total blocked | **1457.5 s** - 24 minutes of frozen desk in ~4h15m |

Where that time actually went - **by blocked time, not stall count**, which is
the ranking that matters and is not the same list:

| Share | Site | Status |
|---:|---|---|
| 42.6% (621 s) | `app.py:1029` = `app.exec()` | **Uninformative by construction.** Precisely the bucket G-P1.3 exists to resolve: from the next session these records carry an `interaction_id` naming the click behind them |
| 12.6% + 4.5% (248 s) | `app.py:833/841` = `collector(2)` / `collector(0)` | The cyclic GC sweeps. **Not addressed by Wave P1** - and the same subsystem as the 1 GB overnight growth observed 2026-08-26 |
| 12.5% (183 s) | `focus_picks_panel.py:419` = the mover chip update | **FIXED** (`0f04240` + `10a3008`) |
| 7.9% (115 s) | `widgets/data_table.py:35` | Not addressed |
| 5.4% (79 s) | `models/theta_table_model.py:72` | Not addressed - and it owns the single worst stall of the day, 49.25 s |
| 3.9% (57 s) | `watchlist_utils.py:33` = `path.read_text()` | Not addressed |
| 2.1% (30 s) | `project_paths.py:165` | Not addressed |

`health_panel.py:147` (the `_fill` cell loop, **fixed** in `49744a7`) does not
appear in today's top list but is the 4th most frequent culprit across the whole
2026-08-21..26 log at 973 stalls - it costs whenever the Health page is open.

**What this says about Wave P1's honest expected effect:** it removes one
measured 12.5% item plus the Health page's churn, and makes the 42.6%
`app.exec()` bucket legible for the first time. It does **not** touch the GC
(17.1%) or the two table paths (13.3%). Do not expect the total to halve.

### The proposal is reconciled to the build (docs only, 2026-08-26 evening)

`docs/GUI_REDESIGN_PLAN_2026-08-25.md` was revised so the next UI effort plans
against what is now true: Wave P1 BUILT with commit ids (§13), the archived
pre-fix session as the baseline (§3.2, by blocked time), the honest expected
effect of Wave P1 (§11.3: ~12.5% + Health churn; GC 17.1% and the two Qt table
paths 13.3% untouched - do not expect the total to halve), the owed fluidity
work re-ordered by measured time (§11.1 - `data_table.py:35` and the Theta
refresh are Qt measurement costs, NOT reads; the Theta refresh grew 3.0 s ->
26.6 s -> 49.2 s across three hourly file-watcher refreshes and that growth is
itself a finding), the trader's live findings (§3.4: narrow columns on every
table page; AWAY Recap unusable as a return surface; Desk Journal
undiscoverable) folded into a §12 table-width/middle-elision RULE and §8.3
page decisions, the build's standing constraints (§2), and the SAC environment
change (§3.5). **Deleted as now-false:** the recap's `load_focus_map` defect
line, the quick-journal "writes no symbol" gap, the §2 "arm-bar mismatch" and
§15 decision 3 / U1's arm-bar item.

**Two things the trader has to decide, neither decided here:**

1. **The CLAUDE.md/AGENTS.md arm-bar line is stale, not the source.** It says
   the arm bar lives on the Armed tab; `4c05de5` (2026-08-20 second pass,
   CHANGELOG "the hotbuttons return") put it back under the chart on the
   trader's own instruction, and `alert_center_panel.py:711` passes
   `dock_arm_bar=True` with that quote beside it. The 2026-08-25 proposal
   built a recommendation on the stale line. Correcting CLAUDE.md is an
   operating-instruction change and is left for the trader.
2. **Smart App Control reads OFF** (`VerifiedAndReputablePolicyState = 0`,
   `SAC_PreviousState = 1`, `SAC_EnforcementReason = 6`), while CLAUDE.md and
   `trading_desk.cmd` still say it is enforced. Both launchers start source,
   nothing is broken; whether the frozen exe becomes production again - with
   its rebuild-before-merge delivery gap - is the trader's call.

Seen in passing: System Health reports `daily_bars/yahoo: 4/5 attempts failed
(80%)`. Data-source finding, not a GUI one.

### Landed on this branch

| Commit | What |
|---|---|
| `cc2d2a3` | Sol's 2026-08-25 GUI review proposal, its checkpoint entry and its `docs/README.md` row, carried onto `main` |
| `db99271` | G-P1.0 - three verified defects (Focus lists never read; adoption-gate line measured nothing; quick journal dropped its symbol) |
| `d050ee1` | G-P1.1 - Weekend Prep reads on a worker; the measured 8.45 s freeze |
| `0f04240` | G-P1.2 - Focus mover state memoized per poll instead of per redraw (36 stalls / 5.93 s) |
| `6bd7eef` | G-P1.3 - interaction id stamped on every stall record |
| `10a3008` | G-P1.2b - the mover memo at its SOURCE, under the extended fence authorization |
| `49744a7` | G-P1.4 incremental health tables; G-P1.5 the lake off the Qt thread; G-P1.6 a daemon thread that outlived its panel |
| `e0f78ae` | Every shutdown join bounded (`join_worker`, 5 s) after the process outlived its window on the DAS reader; source-level guard test. (Recorded here by the reviewer - the commit carried no doc reconciliation) |
| *uncommitted* | **Docs only:** `docs/GUI_REDESIGN_PLAN_2026-08-25.md` reconciled to the build and the 2026-08-26 live session (see "The proposal is reconciled" below); CHANGELOG and this file updated. No code, no tests changed |

### Verification - read this before quoting a number

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4902 passed / 19 subtests, exit 0** (re-run 2026-08-26 evening by the reviewer on the uncommitted docs-only tree, 254.8 s, `.venv` 3.12.13; 4897 at `49744a7`, +5 guard tests at `e0f78ae`). Baseline was 4844; the new tests are the fail-before-fix pins for this work |
| `scripts/smoke_check.py` | **7/7**, re-run at each commit |
| `launch_gui.py --selftest` | Not re-run; unchanged from the `ed277a7` 70/70 |
| Packaging triggers | **None.** `scripts/ui/interaction_trace.py` is a new module inside the already-collected `ui` package and the spec-drift guard passes unchanged. No dependency, asset, top-level package, dynamic import or `__file__` change. No rebuild owed |

Every fix was written fail-before-fix: the test was shown failing on the current
code, then the code changed. Where a test itself was wrong (two were - a wrong
output key on `build_recap`, a cache warmed by the fixture's own setup) the test
was corrected and re-proved against the unfixed code by stashing the change.

### Live gates - OWED, and no test run can discharge them

The proposal's **§11.3 acceptance targets need a real desk session** with the
stall watchdog enabled, compared against the 2026-08-25 capture (264 stalls,
117.3 ms median, 205.1 ms p90, 8.45 s worst, 46.0 s blocked in ~45 min).
Deterministic tests prove the reads moved off the Qt thread; they cannot prove
the desk feels different. Wave P1 is not done until that soak runs.

Also owed inside G-P1.3: the `first_paint` and `chart_ready` marks, which need
the receiving paint path instrumented rather than the emit seam, and the Alert
Center inner tab, which is fenced.

### A latent crash the audit found, and what it implies

G-P1.6 was not on anyone's list. Adding a second HealthPanel-constructing test
file made an unrelated Qt test segfault two files later - 4 runs in 6 - and
bisecting it reached `HealthPanel.shutdown`, which stopped the panel's timer and
left its audit thread running. That thread emits a Qt signal back into the
panel, so it could fire into a freed C++ object: an **access violation**, which
the `except RuntimeError` guard at the emit cannot catch, because it is not a
Python exception.

It was reproduced at the committed HEAD with every uncommitted change stashed,
so it pre-dates this wave; the new tests only made it frequent enough to see.

**The class is not closed.** Any panel that starts a bare `threading.Thread` and
emits a Qt signal back into itself has the same defect. This wave fixed the one
it tripped over. A sweep is recorded in plan.md and is not done.

### Fence discipline

`scripts/ui/panels/alert_center_panel.py` is fenced under the file-scoped
ask-first rule. The trader pre-authorized ONE change there - attaching the chart
symbol to the quick journal write - and the diff in that file is six added lines
and no deletion. G-P1.2's natural memo point is also in that file
(`_measure_mover_state`). It was first implemented in the CONSUMER
(`focus_picks_panel.py`) to stay inside the authorization; **the trader extended
the authorization on 2026-08-26** and it now also lives at the source
(`10a3008`), so the review queue gets it too. The design was chosen from a
measurement rather than a guess: 79% of the per-(symbol, side) cost sits AFTER
bar materialization, so the memo is keyed on the identity of the bars it
measured - not on a clock, because `mover_state` decides what the trader SEES
and a time-based cache could hide a live break.

---

## 2026-08-26 evening - AUTHORIZED: Phase 0.10 AVWAP band challenger; OneOption's band replicated

**Replicated the same evening from three OKTA hover readings** (2026-05-29 and
2026-06-02): `AVWAP(HLC/3) ± k · stdev(close, 20, population)` - a Bollinger
width on an anchored HLC/3 centre, no anchor memory. The anchored sample-OHLC
form predicted 138.09 on 06-02 and the reading was 144.60; the 20-bar
population σ predicted 18.04 and the reading implies 18.035. Record in
`docs/AVWAP_BAND_VARIANT_STUDY.md` §2b. The trader then authorized building it
into the setup tracker: `plan.md` **Phase 0.10** (B-0 module + fixture, B-1 fit
script, B-2 tracker shadow stops + stats + panel section, B-3 D1 overlay off by
default; B-4 backfills after review). Build prompt for the Opus session:
`docs/prompts/AVWAP_BAND_CHALLENGER_OPUS_PROMPT.md`. **Nothing is built yet**;
the champion σ stays frozen (decision 0008) and the parity fixture comes first.
The Wave P1 live soak above is still owed and is unaffected. Verification
baseline unchanged - docs only. **Also written the same evening:** the Phase 0.9
build prompt `docs/prompts/GUI_PHASE_0_9_OPUS_PROMPT.md` (G-P2.0..G-P2.4 with
two soak stops, GC measurement only, fenced-file ask for the Journal route) -
to be run AFTER the Phase 0.10 session, because both build in this checkout
and the desk launches from it.

(Earlier the same evening this entry read "PROPOSAL recorded"; the paragraph
below is that record.)

### The proposal as first recorded

The trader compared the same earnings-anchored chart in TradingView (their own
`AVWAPE` script) and a second program whose 1σ band is wide from the very first
bar and ≈1.5× the champion's width thirteen bars later - and was the level the
last bar rejected. Instruction: replicate the second program's σ, then test it
against the trader's method, integrated into the setup tracker; **plan only, no
code.** The plan is `docs/AVWAP_BAND_VARIANT_STUDY.md`: the inference that any
one-price-one-volume-per-bar formula has σ = 0 on the anchor bar, so the other
program uses in-bar information (range or intraday prints), a percentage/ATR
width, or an earlier anchor; a fifty-candidate closed-form fit against hover
readings with a champion-vs-TradingView control run first; a frozen pure
`indicators/` module; and three shadow harnesses with criteria declared before
the first run. It touches nothing until the trader answers its §6 questions
(which program, which symbols/anchors, the hover readings) and promotes it into
`plan.md`. The champion σ stays frozen (decision 0008) throughout; a winning
variant would be an additional level family, never a swap. Active item, branch
and verification baseline are unchanged - no code or tests changed.

## 2026-08-26 - CONSOLIDATION: the trunk is `main` again

**Branch `main`.** Trader-directed branch cleanup. `testing-week-2026-08-24` was
fast-forwarded onto `main` (354 commits, 480 files); `main` had been a strict
ancestor of it, so no conflict was possible and none was resolved. One unlanded
document was merged; one branch was deliberately left open; three fully contained
branches were cleared for deletion but **not deleted** - see below.

### What the working branch is now

Active work continues on **`testing-week-2026-08-24`**, which was kept alive for the
GUI-optimization pass in flight. `main` now carries everything that branch carried as
of `ed277a7`. The next consolidation is a fast-forward again if that branch keeps
`main` as an ancestor.

### Verification - read this before quoting a number

| Check | Result |
|---|---|
| `pytest tests/ -q` | **NOT RE-RUN for this merge.** The `ed277a7` baseline of 4844 passed / 19 subtests, exit 0, describes `main` exactly |
| Code-state proof | `git diff --name-only ed277a7 HEAD` returns **only `.md` files** - the verified code tree is byte-identical |
| `scripts/smoke_check.py` | Not re-run; unchanged from the `ed277a7` 7/7 |
| `launch_gui.py --selftest` | Not re-run; unchanged from the `ed277a7` 70/70 |
| Packaging triggers | **None.** No dependency, no non-`.py` runtime asset, no new `scripts/` package, no dynamic import, no `__file__`/`ROOT_DIR`/`sys.path` change. No rebuild owed |

The consolidation ran in a cloud container with no project virtualenv and Python
3.11 against a project floor of 3.12. A suite run there would have proved nothing,
so none was claimed. The Markdown-only diff is what carries the baseline forward,
and it is checkable in one command.

**Live gates:** none marked met, none waived. This merge promotes nothing, changes
no detector, scorer or alert, and moves no row.

### Owed to the desk: three branch deletions

`claude/ticker-briefs-hardening-imcm8r`, `phase05-r2-focus-gating-strength-board`
and `phase05-integration-blitz` each hold no commit that is not on `main`
(`git merge-base --is-ancestor` against `226fbac`). The cloud session's GitHub
credential pushes but refuses ref deletion with `HTTP 403`, and the egress proxy
recorded no policy denial, so the refusal is token scope rather than a blocked host.
Commands and the re-prove step are in `docs/BRANCH_HISTORY.md`.

### Held open, deliberately

`claude/alert-center-quality-packet-5btu3w` (8 commits, tip `57fcf47`) stays
unmerged pending two trader answers: the file-scoped ask-first rule governs it
because it edits `scripts/ui/panels/alert_center_panel.py`, and it collides with
`main` at `docs/ALERT_CENTER_QUALITY_PACKET.md`, where a *different* historical file
already lives. Both are written up in `docs/BRANCH_HISTORY.md`.
## 2026-08-25 night (UI planning redirect) - full 4K desk review recorded; no build authorized

**Branch `testing-week-2026-08-24`, head `ed277a7`.** The trader explicitly
redirected to a planning-only review of every GUI page/tab for a main 4K monitor
using full height and about 90% width, followed by a measured speed plan and a
Settings offload plan. No application code, runtime contract, detector, scorer,
alert, schedule, store, or roadmap item was changed.

The supporting proposal is
[`docs/GUI_REDESIGN_PLAN_2026-08-25.md`](docs/GUI_REDESIGN_PLAN_2026-08-25.md).
It is deliberately marked **PROPOSAL — NOT AUTHORIZED WORK** and is indexed as
an active product reference, not another roadmap. `plan.md` remains unchanged;
implementation begins only if the trader accepts the decisions in the proposal
and promotes the selected work into the root roadmap.

### Review evidence

- Exercised the live PySide desk top-to-bottom across all 14 pages and every
  nested tab at the current 4K Windows logical viewport.
- Projected the stated 90%-width target as a 2304 × 1392 outer window / roughly
  2060 × 1308 Trading Desk content surface. Capture measured 1195 × 352 against
  a 759px content hint with setups hidden, and 679 × 352 with setups shown: the
  permanent 60/40 Focus Strength companion is the primary Veto-layout defect.
- Read the locked chart/capture, Chart Review, Trade Journal, Weekend Prep, and
  GUI-fluidity contracts before writing the proposal.
- Filtered the live stall log to the current GUI process: 264 stalls, 117.3 ms
  median, 205.1 ms p90, 8.45 s worst, 46.0 s total blocked in about 45 minutes.
  The worst freeze was Weekend Prep doing review-learning/CSV work on Qt; Focus
  mover-state refreshes and Desk mode reparents were also reproduced.
- Focused planning baseline: **116 passed** across the desk-layout, scale,
  Market Journal, AWAY Recap, fluidity/watchdog, and page-spec suites.
- `git diff --check` and Markdown/index checks are the only additional
  verification required for this docs-only change.

### Roadmap/baseline effect

None. The 4844-test / 19-subtest, smoke 7/7, selftest 70/70 baseline below is
unchanged, and every existing live gate remains owed exactly as recorded. The
immediate active implementation action therefore remains the Questrade-chain
live proof/trader decision below unless the trader promotes part of the GUI
proposal.

---

## 2026-08-25 night - the Questrade chain has ONE owner; stale failures are reachable again

**Branch `testing-week-2026-08-24`.** Trader-directed after a fresh token was
pasted and Journal Health stayed red. Three findings, two repairs, one decision
left to the trader.

### What was actually happening (reproduced from the live desk, read-only)

The token **worked**. `import_runs` shows `QUESTRADE OK` at **20:54:59**, and
today's single Questrade execution is in `raw_executions` for 2026-08-25. Then a
year-wide backfill ran at 20:59, and at **21:06:51** the next Questrade run
failed `400 Bad Request` on the refresh endpoint - eleven minutes after the
paste. **Disclosure:** one of the failed runs is the AI's - `pull_today()` was
run once headless to reproduce the button. It was rejected at the refresh, so it
rotated nothing, but it wrote a FAILED `import_runs` row.

Health was red for a **different** reason than the token. 185 FAILED Questrade
days: **140** carrying OAuth/400 messages from before the 2026-08-25 DateTime
repair, and **45** saying `activities report trades the executions endpoint did
not return`. No token can clear either set.

### Repair 1 - the chain snapped because it had several owners

Questrade rotates on every refresh: a success invalidates the access token it
replaces AND consumes the refresh token it was given. "Pull today now", the gap
backfill and the nightly slot are three consumers of one single-use chain, and
that is enough to break it. Three changes, in `journal_importers.py`:

* the refresh is **serialized across every process on this machine**
  (`local_writer_lock`, the primitive the outcome finalizer already uses);
* the token is **re-read inside the lock**, so a caller that waited spends what
  the winner LEFT rather than what it read before waiting;
* a **401 explained by someone else's rotation reuses their new access token**
  instead of burning a refresh to rediscover it. That cascade - our access token
  killed by their refresh, our 401 answered by spending a consumed refresh
  token - is the exact shape of the 21:06:51 failure.

Beside it, `project_paths.save_local_settings()` writes several keys in **one**
read-modify-write, through a temp file and `os.replace`. The rotation saves four
related keys, one of them a single-use token; four separate cycles were four
windows for a concurrent writer to drop it, and a direct `write_text` over a
file holding every machine-local secret could truncate it.

A failed refresh still saves nothing and **leaves the stored token alone** -
a rejected token may still be the good one, and clearing it makes the repair
"paste it again" every time.

### Repair 2 - a day whose CAUSE was fixed could never come back

The attempt cap counts failures against a DAY, not against a cause. The 140
OAuth-era days burned their budget while the chain was dead and were then
skipped forever, so a repaired chain could not clear them and the tile stayed
red on a solved problem. `journal_coverage.self_heal(include_exhausted=True)`
lifts the cap for ONE deliberate run; the Health tab passes it from **"Retry
failed Questrade days"** only. The nightly keeps the cap - a dead chain must not
eat the budget every night. **`attempts` is not rewritten** (ground rule 5): it
stays the record of what the day cost, and the run reports
`reopened_exhausted` so the trader sees a number rather than a silent change.

### Finding 3 - the 45 mismatches are real, and NOT repairable by retrying

Measured, read-only: **44 of the 45** have no Questrade execution within +/-3
days, so this is not a timezone skew between the two endpoints. They span
2025-11-13 to 2026-08-13 and cluster in 2025-11 (8), 2025-12 (12), 2026-01 (19),
2026-02 (5). The journal holds **no Questrade execution before 2026-06-10** on
either account, which is the executions endpoint's retention horizon; the
activities endpoint reaches back much further. So the executions endpoint can no
longer supply those trades **at all**, and every retry of them is futile.

**Left to the trader, deliberately not built.** The choices are to import those
days from `/activities` - a lower-fidelity source that lacks execution-level
detail, in a record that feeds tax - or to label them permanently uncovered and
stop implying they are actionable. That is a decision about the trade/tax
record, not a plumbing fix, and it needs a new coverage status either way.
**One exception worth a look: 2026-08-13 falls INSIDE the retention window**, so
that day may be genuinely recoverable and is not explained by retention.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4844 passed / 19 subtests**, pytest exit **0** |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| `launch_gui.py --selftest` | **70/70**, exit 0 |
| `test_packaging_spec_drift.py` | 17 passed, exit 0 - no packaging trigger |

4837 -> 4844: seven added.

**Live gates:** none marked met. The chain repair is proved by tests, not by a
live session; the next real proof is a paste that survives a backfill.

---

## 2026-08-25 evening (7) - trader accepts the sweep canary; the scan cycle is now timed

**Branch `testing-week-2026-08-24`.** Two trader decisions, both recorded here
verbatim because they are the only authority for what follows.

### 1. The R10.A restarted-process sweep canary is ACCEPTED

Trader, 2026-08-25: **"52min late is fine."** The 2026-08-25 run is therefore
the successful live weekday session R10.A owed - `swept_at
2026-08-25T14:27:36-07:00`, 656 pending / 656 finalized, 0 expired, 0 failed, 0
commit failures, `pending_after 0`, yesterday's 553 finals untouched. **This gate
is met by the trader's acceptance, not by an AI's judgement**, and the delay is
accepted as known rather than explained: its cause is still UNKNOWN and the
investigation stands.

What this does NOT do: it does not accept the two fenced repairs' own canaries
(the milestone recovery and `_signal_bar_dict`), which are separate and still
owed, and it does not touch R10 10a.

### 2. The scan-cycle timing instrument is authorized and BUILT

Trader, 2026-08-25: **"Yes you can add the timing log."** This is the ask-first
answer for `bounce_bot_lib/legacy.py` beyond Decision B's two repairs, and it is
scoped to instrumentation: **no scheduling change, no detector, scorer or alert
change.**

`ScanCycleClock` (module level, pure) records elapsed seconds per named stage
and formats them slowest-first. `run_strategy` builds one per cycle and marks
ten stages across the preamble - `watchlists`, `atr_cache`, `auto_regime`,
`focus_fast_lane`, `rrs_scan`, `regime_pause`, `entry_assist`, `auto_populate`,
`rvol_baselines`, `m5_and_h1_engines`, `symbol_sets` - then logs **one** line
before the existing "Monitoring N" line:

> `Scan cycle 41 preamble: 92.4s total: rrs_scan 88.1s, atr_cache 2.0s, +8 other 2.3s`

Once per cycle, so a quiet evening costs a few lines a night. Stages past the
named few are **counted, never dropped**: a breakdown that silently omits stages
reads as a complete account of the time and is not one. A backwards clock (a
Windows adjustment) reports 0.0s rather than a negative stage, which would read
as an instrument fault rather than a measurement.

Beside it, `_maybe_refresh_learning_after_close` now says **when it first finds
work due** - the timestamp the 2026-08-25 investigation had to infer from the
sweep's own stamp - and says **once per worker** when it is waiting on an
earlier one. Deliberately silent when nothing is due: that would be a line a
minute, all evening, saying nothing happened. The 60 s throttle, the due logic
and the worker are untouched.

**The instrument decides nothing, and a test enforces that**: the class is
parsed and asserted to call no `sleep`, `wait`, `start` or `Thread`. A timing
helper that could defer or skip would be a scheduling change wearing an
instrument's name, and scheduling is exactly what was not authorized.

Fail-before: `test_the_loop_logs_one_cycle_line_with_the_breakdown` red (the
class existed and `run_strategy` never built one - the same "present but never
called" shape as the recap's `set_alerts`).

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4837 passed / 19 subtests**, pytest exit **0** |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| `launch_gui.py --selftest` | **70/70**, exit 0 |
| golden fixtures + bounce/sweep sets | **456 passed**, exit 0 |

4830 -> 4837: seven added. No packaging trigger. The one-off 93% segfault has
not recurred in the four runs since.

**What it is expected to show.** If the 12:55-14:27 stall repeats, one line will
name the stage that ate it. If it does not repeat, the lines are a normal-cycle
baseline, which is the other half of the answer.

---

## 2026-08-25 evening (6) - the recap had no OUTLET: alerts drawn, rows chartable

**Branch `testing-week-2026-08-24`.** Trader-directed, after reading the live
page: *"nothing useful appears in the away journal. It's just a few tickers no
charts."* Correct, and packet 2 fixed only half of it.

**Correction to this checkpoint's own packet-2 entry.** It said the AWAY Recap
"is wired to the Alert Center backing list". That is true at the INPUT and was
too strong as a claim about the page: `build_recap` has always returned
`classified_alerts`, and `AwayRecapPanel._render` **never read it**. The page
drew three tables - best swings, staged picks, Focus names - so a whole AWAY
day's alerts left exactly one trace, the word "alert(s)" in the summary line.
Packet 2 connected a pipe to a blocked outlet.

**What the trader was actually looking at, measured.** The running desk is pid
36300, started **2026-08-24 21:48**, i.e. before any of today's commits, so
`set_alerts` had no caller in that process either. Run headless against today's
files, the recap says:

> `10 ranked swing(s), 0 alert(s), 54 staged pick(s) and 0 Focus name(s) for 2026-08-25.`

Ten swing rows and fifty-four staged tickers, and nothing else - "a few
tickers". And there was **no chart anywhere on the page**: `symbolActivated` was
declared and nothing emitted it, while `app.py` connected only the Strength
Board's.

**Built.**

- **An alerts table** - time, symbol, side, tier, a `D1` flag and the trigger -
  in the order the day produced them. No re-ranking, which is this page's own
  provenance rule. A D1 row is FLAGGED rather than merged away, because the
  Alert Center keeps that feed separate (it is untiered) and a reader has to be
  able to tell them apart.
- **Rows open the desk's existing snapshot popup.** `_activate_alert` /
  `_activate_swing` emit `symbolActivated`; `MainWindow` connects it to
  `alert_center.show_board_symbol` - **the same popup** the Strength Board,
  RS/RW and Industry boards use (the R4 pattern), so the chart carries the
  bot-backed series, the painted levels and the capture rail and **no second
  chart widget exists**. A blank symbol asks for nothing: an empty popup reads
  as a broken chart rather than an empty row.

Fail-before: seven new tests, all seven red (`panel.alerts` did not exist;
`symbolActivated` had no emitter and no host connection).

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4830 passed / 19 subtests**, pytest exit **0** |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| `launch_gui.py --selftest` | **70/70**, exit 0 |

4823 -> 4830: seven added. No packaging trigger. The 93% segfault seen once
earlier did not recur in either of the two runs since.

**Still true and still owed:** the backing list is PROCESS-scoped, so the test
is a restart BEFORE a session and the page AFTER its close. Restarting now shows
no alerts, because today's went with the old process. R10 10a's live gate is
unchanged and unmet.

---

## 2026-08-25 evening (5) - what held the strategy loop from 12:55 to 14:27 - evidence only, no fix

**Branch `testing-week-2026-08-24`.** Packet 5. **No code change**, no scheduling
change, nothing proposed as built. Logs and read-only reads only.

**The narrowing, from `logs/trading_bot.log` and `trading_bot.log.1`.** Cycle
boundaries (`Monitoring N strongest/weakest symbols for EMA bounces`, logged
once per cycle at `run_strategy` line ~13465) ran **11:46:54, 11:57:01,
12:20:37, 12:32:42, 12:47:21** - gaps of 10, 23, 12 and 15 minutes - and then
**14:27:36**. The last `wait_for_candle_close` line is **12:51:29** with 211 s
to go, so that cycle's wait ended at **12:55:00**, and no `Waiting for candle to
close` line appears after it. The `legacy.py` output between 12:56 and 13:10 is
**not** the strategy loop: it is the Master AVWAP runner's 13:02 swing scan and
its theta option enrichment, interleaved with `runner.py` lines on the same
thread.

**So the loop spent 12:55:00 -> 14:27:36 (92 minutes) inside ONE iteration**,
and specifically between the loop top and its cycle-boundary log line - the
preamble at `run_strategy` lines 13428-13465: the watchlist loads,
`emit_master_avwap_d1_flags`, `_prune_latest_bars_for_cycle`, `build_atr_cache`,
`update_auto_market_environment`, `run_rrs_scan`, `check_regime_pause_setups`,
`entry_assist_auto_tick`, the two `_maybe_refresh_*` calls, the five M5 engines
and `check_h1_color_setups`. **Every one of them is silent on the normal path**,
which is why 92 minutes produced no line at all.

**The consequence for the sweep is arithmetic, not a scheduling defect.**
`_maybe_refresh_learning_after_close` is called at the TOP of every iteration
with a 60 s throttle, and the sweep's `swept_at` is stamped on entry. The stamp
is 14:27:36, the same second as the cycle-boundary line, so the top-of-loop
check first found the sweep due 52 minutes after its 13:35 due time. The due
logic itself is correct.

**Candidates, and what the evidence does to them:**

1. **Mass IB request timeouts in `run_rrs_scan` / `build_atr_cache`** - the
   arithmetic fits (92 min / 15 s = 368 requests) but the evidence does **not**:
   each timeout logs `Timeout waiting for RRS data`, and the whole log holds
   **one**, at 13:01:25, on the swing-scan thread. **Refuted as the main cause.**
2. **Contention with the 13:02 swing scan** on the one IB connection - it ran
   13:02:03 to 13:10:19 and spent a 240-option-quote budget. Real, but it covers
   8 minutes of 92. **Contributing at most.**
3. **`_maybe_refresh_auto_regime_while_paused`** - it pops the SPY series and
   refetches once a minute, and logs nothing. It is only reached on the PAUSED
   branch, and the loop was not on that branch. **Not it.**
4. **A slow, non-timing-out IB path inside the preamble** - requests that return
   late rather than time out. Consistent with everything measured and
   **unfalsifiable from the current log**, because nothing in the preamble is
   instrumented.

**Cause: UNKNOWN**, narrowed to those lines. It is not called a defect and
nothing was changed.

**One thing the logs cannot settle, stated rather than smoothed over.** At
14:27:36 the loop reached the SCANNING branch, past the paused-branch
`continue`, although `set_scanning_enabled(False)` was logged at 13:30:11 and no
`Scanning enabled.` line follows it. Either the flag was restored without a log
line, or that line belongs to the tail of an in-flight cycle and the sweep's
stamp coincides. Separating them needs instrumentation.

**Sweep throughput beside the live loop (measured):**

| | 2026-08-24 | 2026-08-25 |
|---|---|---|
| sweep start | 16:45:43 | 14:27:36 |
| finalize complete | 16:47:18 | 14:46:35 |
| trades | 687 | 656 |
| wall clock | **1 m 35 s** | **19 m 0 s** |
| per trade | **0.14 s** | **1.74 s** |

**~12x slower, and the likely contention point is what else was running.** The
08-24 sweep ran at 16:45, long after the desk had gone quiet. The 08-25 sweep
started in the same second the strategy loop began a fresh scanning cycle, and
that cycle held the IB connection for the whole 19 minutes. Both threads append
to the same 200 MB outcome CSV and both take `local_writer_lock`. Two further
measured costs: `_recover_measurements_from_csv` scanned the whole CSV in **1 m
56 s** (14:27:36 -> 14:29:32 for 636 trades), and a **second** full scan ran at
14:31:16 for **one** trade - two whole-file passes within four minutes.

**Proposals (not built, not authorized):**

1. **A cycle-boundary INFO line** in `run_strategy` - cycle start, cycle end,
   and the after-close check's outcome - which is the one thing that would have
   answered this in a minute rather than an afternoon. `bounce_bot_lib/legacy.py`
   is fenced and Decision B names two repairs, neither of which is this, so it
   is **ASK-FIRST and was not written.** Recommended shape: per-stage elapsed
   time across the preamble, not only start/end, since start/end alone would
   have narrowed this no further than the log already did.
2. **Nothing about scheduling.** Deferring the sweep while a cycle is in flight,
   or checking the pause flag mid-cycle, are both scheduling changes and are
   explicitly not authorized.

---

## 2026-08-25 evening (4) - record correction: the restarted-process outcome sweep RAN

**Branch `testing-week-2026-08-24`.** Packet 4 (Decision C, bullets 2-3).
Documentation only; no code change, baseline unchanged from packet 3.

**What actually happened on 2026-08-25**, read from
`diagnostics/outcome_sweep_coverage.json` and `logs/trading_bot.log`:

| | |
|---|---|
| `swept_at` | **2026-08-25T14:27:36-07:00** |
| pending_before / finalized | **656 / 656** |
| expired / failed / commit_failed | **0 / 0 / 0** |
| pending_after | **0** |
| recovered_from_csv | 636 |
| by reason | last_measured_bar 422, stop_hit 214, no_measurement_in_checkpoint 20 |

Yesterday's 553 finals were untouched. Three documents said the canary
**FAILED**; they were written from a 14:21 read, six minutes before the sweep
started. The word is replaced by the observation, and acceptance is the
trader's: **no gate is marked met here.**

**The 52-minute start delay is real and its cause is UNKNOWN.** The sweep became
due at close+35 (13:35 PT) and the due logic is correct; the top-of-loop check
that finds it due did not run until 14:27:36. What is measured is in packet 5's
entry below. Nothing about the delay is called a scheduling defect, and no
scheduling change was made.

**`docs/DESK_TESTING_PLAN.md` sec 2.3 restored to HALF done.** Sol relabelled it
PASSED; it is not. The AWAY staging half was observed on 08-17/08-18 and again
on 08-25 (queue routing behaved: zero `shown` impressions while the backing
list, History, the D1 badge and the hourly phone reports filled). The
flip-back-to-DESK half - a populated recap, and no chart-review backlog on the
return - is still owed and now has code behind it (packet 2).

**Sol's frozen attack report is NOT edited.** `SOL_ATTACK_2026-08-24.md` records
what was true when it was frozen; the correction lives here, in `CHANGELOG.md`
and in `plan.md`.

**Verification baseline: unchanged from packet 3** - docs only.

---

## 2026-08-25 evening (3) - the two fenced evidence repairs (Decision B)

**Branch `testing-week-2026-08-24`.** Packet 3. `scripts/bounce_bot_lib/legacy.py`
was edited ONLY for the two repairs Decision B names. No alert, tier, fold,
digest or queue behavior changed; no scheduling change; the R5 wiring tests and
the C9 boundary tests are green untouched, and the golden fixtures are
byte-identical.

**B.1 - milestone recovery may not erase a recorded stop** (Sol T3, C4
reproduced verbatim as a test). `_recover_measurements_from_csv` took the first
row at the FURTHEST milestone outright, so a 12-bar row saying `stop_hit=False`
erased a 3-bar row that had already recorded the stop and the trade finalized
`last_measured_bar` at +0.5R. Now `stop_hit` is `any()` across the trade's
recoverable rows, and where a stop exists the exit numbers come from the
EARLIEST stop-hit row - R10.0's stop-first decision applied here: the trade was
over at that bar, and later rows describe price action after an exit that had
already happened. With no stop anywhere the best-rank row still wins. An
unreadable `bars_elapsed` cannot be ordered, so it sorts LAST among the stop
rows rather than winning by accident. The scan stays streaming and O(1) per
trade: the best-rank row, the earliest stop-hit row and the any() flag, never
the whole milestone history.

**One existing champion expectation changed, deliberately and named.**
`test_a_backlog_stop_out_is_recovered_from_its_own_csv_rows` has two stop-hit
rows; its exit is now the 3-bar row's (`mae_r` -1.5, `recovered_from` `3_bar`)
rather than the 12-bar row's (-1.8, `12_bar`). That is the rule Decision B.1
states, not a regression.

**The 35 already-written rows are TAGGED, never rewritten** (ground rule 5):
`evidence_rules.milestone_stop_erased_v1`. Reproduced read-only over the live
store and confirmed after: 2026-08-24 finals tag **35 mixed / 172 clean / 907
unknown**; 2026-08-25 tags **0 mixed / 737 unknown**, because that sweep's
trades measured in state rather than by recovery. The rule is conjunctive - a
final not recovered from milestone rows is `unknown`, not clean - and a clean
answer means "no evidence of erasure", never "no erasure", because it cannot see
a trade whose milestone rows were pruned.

**B.2 - signal-bar recovery must match the event's bar** (Sol T2, C3 reproduced
verbatim). `_signal_bar_dict` matched only on close, and a cache shifted by one
bar with two adjacent equal closes returned the 06:35 event's 06:30 bar. It now
takes the event's `bar_time` - passed from all three call sites - and requires
`bar.dt == event.bar_time` in addition to the close match. No `bar_time`, or a
bar with no `dt`, yields the fallback: missing data is uncertainty, never
confirmation, and the fallback is at least the event's own prices. The real call
path is unaffected -
`test_the_outcome_registration_gets_the_real_signal_bar` and
`test_the_alert_row_and_the_tier_still_see_the_flat_bar` pass unmodified.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4823 passed / 19 subtests**, pytest exit **0** |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| `launch_gui.py --selftest` | **70/70**, exit 0 |
| golden fixtures / R5 / sweep / outcome sets | **405 passed**, exit 0 |

**A method note that cost a rerun, recorded so it is not repeated.** The first
three suite runs were piped (`pytest ... | tail -5; echo $?`), which reports
**tail's** exit code, not pytest's - exactly what CLAUDE.md's Commands section
warns against. Run unpiped, the packet-3 suite **segfaulted at 93% (exit 139)**
with zero test failures. It did **not** reproduce: the rerun reached 100% and
exited 0, and the warehouse set that was executing at 93% passes 322/322 alone.
Recorded as an unexplained one-off crash, not as a pass and not as a defect.
Every count above is from an unpiped run whose own exit code was read.

**Canaries owed:** one live after-close sweep on the repaired recovery (the
2026-08-25 sweep would not exercise it - it recovered nothing from milestones);
and the first live LRSI/confluence/ORB registration after B.2, confirming a
gradeable row still lands.

---

## 2026-08-25 evening (2) - the AWAY Recap is wired to the Alert Center backing list

**Branch `testing-week-2026-08-24`.** Packet 2 (Decision C, first bullet). One
call at page selection; no detector, scorer, alert or scheduling change, and the
recap still writes nothing (its AST test is untouched and green).

**Reproduced first:** Sol's C1, now a test. `MainWindow` constructed
`AwayRecapPanel` and `set_alerts` had no caller anywhere, so a full AWAY day
ended in an empty recap while the backing list, History and every evidence
stream were full. Fail-before printed `backing 1 recap_input 0`.

**Built.** `MainWindow._select_page` calls `_feed_away_recap()` when the selected
page is `AWAY_RECAP_PAGE_TITLE` - matched by TITLE, not index, so a page reorder
cannot silently unwire it (the class of bug `test_qt_page_specs` exists for).
The method exports the Alert Center's two backing lists as ONE ordered stream:
both are newest-first so both are reversed, and they are merged on `time_text`,
because a recap of the day is the day and a D1 row travels flagged (`is_d1`)
rather than merged away. `away_recap._alert_rows` reads MAPPINGS, so the
`BounceAlert` dataclasses are converted here - handing them in raw would raise
inside the worker thread and leave the page blank in a way no assertion on
`_alerts` could see. The tier comes from the Alert Center's own public
`extract_alert_tier`; this page computes none. `alert_center_panel.py` was NOT
touched - no accessor was needed, so the fence was not approached.

Failure is quiet on purpose: a recap that cannot be filled must never cost the
page switch that asked for it.

**Known limitation, stated rather than papered over:** the Alert Center's
backing list is PROCESS-scoped and capped, not session-scoped. A desk left
running across midnight, or restarted mid-session, hands the recap what the
process saw rather than what the session produced. Fixing that is a session
boundary on the backing list itself - inside the fenced alert file - and is not
this packet's to make.

**Live gate still owed:** one real AWAY day ending in a populated recap
(R10 10a). Nothing here is marked passed.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4810 passed / 19 subtests**, exit 0 |

Suite 4808 -> 4810: two Qt regressions added. No packaging trigger.

**Immediate next action:** packet 3, the two fenced evidence repairs (Decision B).

---

## 2026-08-25 evening (1) - Decision A built: a sweep-finalized trade counts under the policy that measured it

**Branch `testing-week-2026-08-24`.** Packet 1 of the post-attack repair slate
(`docs/analysis/POST_ATTACK_AUTHORIZATION_2026-08-25.md`, Decision A). No
detector, scorer, alert or scheduling change; evidence-read only.

**Reproduced first (ground rule 3).** Over the live store, read-only:
`load_intraday_finals` for 2026-08-25 returned `in_window 656`, `unsettled 656`,
**`usable 0`**. Every one of the day's 656 sweep-finalized trades was invisible
to every evidence surface, because `usable` keyed on the eod-hold `close_r` the
sweep leaves blank BY DESIGN (`no_eod_close`), while what the sweep measured sat
in `context.exit` - `stop_exit_r` on 214 rows and `last_measured_close` on 636 -
which `_exit_policy_rows` never read (it reads `context.path.exit_policies`).

**Built.** `setup_scoreboard.exit_policy_r` derives three frozen exit policies
per final and `load_intraday_finals` attaches them as `r_eod_hold`,
`r_stop_exit`, `r_last_measured`:

- `eod_hold` - the settled `close_r`, blank wherever no EOD close was obtained;
- `stop_exit` - the STORED `context.exit.stop_exit_r`, read only when
  `exit.stop_hit` is true;
- `last_measured` - `(last_measured_close - entry_price) / risk_per_share`,
  sign flipped for a short. Arithmetic on stored numbers; an unreadable close, a
  non-positive risk or a direction that is not `long`/`short` yields no value at
  all rather than a guess.

`usable` is now "at least one policy has a measured value", still ANDed with the
risk floor and the R10.B claim split. Unresolved rows stay unusable and are
counted by reason (`coverage.unresolved`, `unresolved_by_reason`). The three
policies are **never blended**: `sweep_exit_policy_rows` builds one table per
policy through `summarise` -> `evidence_stats`, so every cell carries the full
ground-rule-10 statistics including the 4R clip, and an n always means "rows this
policy could measure". The eod-hold ranking tables now read `r_eod_hold` instead
of `close_r`, so a newly usable row with no EOD close cannot widen their n.

Report section **1a** prints the move rather than asserting it: usable under the
old rule, usable now, unresolved by reason, what each policy could measure, and a
per-family table per policy. The old 1a/1b became 1b/1c. The bundle gains
`sweep_exit_policies` + `sweep_exit_policy_note` and four coverage keys. The
digest inherits `usable` from the scoreboard as before and now carries
`stop_exit_r` / `last_measured_r` beside `close_r`, each with its own n, never
folded in - otherwise a sweep-finalized session would report n=0 close_r beside
656 usable rows (the R10.I "n=0 beside real data" class).

**Live check after (read-only, 2026-08-25 slice), recorded as observation:**

| | rows |
|---|---|
| in window | 656 |
| usable under the old rule | **0** |
| **usable now** | **255** |
| unresolved (`no_measurement_in_checkpoint`) | 20 |
| measurable by `eod_hold` / `stop_exit` / `last_measured` | 0 / 214 / 636 |
| excluded - below the risk floor | 18 |
| excluded - family does not CLAIM an entry (R10.B) | 363 |

Among the 255 usable: `stop_exit` n=99 mean **-1.0R**, `last_measured` n=255
mean **+0.1546R**, `eod_hold` n=0. The two are reported side by side and are not
averaged together. **These are observations, not a met gate.**

**Known limitation, stated not papered over:** `stop_exit` is -1.0 under the
sweep's stated fill assumption (only the touch is measured); it is stored that
way and is read, not recomputed.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4808 passed / 19 subtests**, exit 0 |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| `launch_gui.py --selftest` | **70/70**, exit 0 |
| golden fixtures | byte-identical (93 fixture/golden tests pass) |

Suite is 4801 -> 4808: seven added, none removed. No packaging trigger: no new
dependency, package, asset or dynamic import.

**Immediate next action:** packet 2, the AWAY Recap wiring (Decision C).

---

## 2026-08-25 afternoon - Fable review of Sol's pass: ACCEPT WITH BLOCKERS; trader decisions A/B/C recorded

**Branch `testing-week-2026-08-24`.** Independent review of `9edbf83..5a954d8`
(no repository change by the review itself). Sol's three repairs verified
minimal and correct (adjacent edge cases pass); suite 4801/4801 twice, smoke
7/7, selftest 70/70 source and frozen. Findings that change the record:

- **The restarted-process outcome sweep RAN today** - `swept_at
  2026-08-25T14:27:36-07:00`, 656 pending / 656 finalized / 0 failed / 0
  commit-failed / pending_after 0, yesterday's 553 finals untouched. The
  "canary FAILED" language in the 14:28 entry below, plan.md and
  DESK_TESTING_PLAN is superseded: the due logic is correct (offline
  `_after_close_jobs_due` at 13:36 PT -> sweep due), the sweep started 52 min
  late for an UNKNOWN cause (the loop was held inside one cycle from before
  13:30 to 14:27). The observation stands; acceptance is the trader's.
- **HIGH, new:** every sweep-finalized trade is invisible to every evidence
  surface (blank eod-hold `close_r` by design; `usable` keys on it;
  `_exit_policy_rows` never reads `context.exit`). Today: 656 finals, 0 usable.
- AWAY recap unwired (`set_alerts` has no caller) - one call in
  `_select_page` fixes it; T3 (stop erased by furthest milestone row: 0 of 84
  candidates today, 35 of 207 yesterday) and T2 (wrong duplicate-close bar)
  PROVEN and fenced; DESK_TESTING_PLAN sec 2.3 was relabelled PASSED by Sol
  from HALF done - restored below.

**Trader decisions (recorded in
`docs/analysis/POST_ATTACK_AUTHORIZATION_2026-08-25.md`):** A - stop-outs
count at `stop_exit_r`, other sweep finals at last-measured-close R, separate
policies never blended; B - ask-first answered for the two evidence-side
`legacy.py` repairs; C - recap wiring, record corrections, sec 2.3 restored.
Opus build prompt issued; build not started at this stamp.

**Verification baseline: unchanged** (`21fd55e` code state: 4801 passed / 19
subtests, exit 0; smoke 7/7; selftest 70/70) - docs only.

---

## 2026-08-25 post-close - Sol adversarial pass complete; live canaries failed

**Branch `testing-week-2026-08-24`. Active item: Phase 0 validation and blocker
removal.** The frozen attack record is
`docs/analysis/SOL_ATTACK_2026-08-24.md`. Three non-fenced blockers were repaired
and pushed in one commit each:

- `0c62b63` makes hermetic teardown fail when `BounceBot.stop()` raises, even
  after the thread has joined;
- `8474383` excludes blank/non-finite unresolved finals from usable outcome
  evidence; and
- `21fd55e` sends full inclusive offset-bearing DateTimes to Questrade
  `/activities`.

**The 2026-08-25 live AWAY canary did not pass.** Queue routing itself behaved:
the day had zero `shown` review impressions while `focus_d1_flag`,
`level_fired`, `d1_event_fired`, the backing alert list, History and the hourly
AWAY report continued to fill. The recap nevertheless had no input:
`MainWindow` constructs `AwayRecapPanel` but never calls `set_alerts`, and the
ordinary and D1 backing lists have no single ordered, session-scoped export.
R10 10a remains live-owed.

**The restarted outcome-sweep canary also did not pass.** The owning strategy
loop logged `Scanning paused` at 13:30:11 (close+30), before the sweep's 13:35
due time. At 14:21 the 2026-08-25 outcome slice had 656 registrations and zero
finals, and `outcome_sweep_coverage.json` still carried the prior day's
`swept_at`. The prior 2026-08-24 production sweep remains valid (687 finalized,
0 failed, 0 commit-failed), but it does not close the restarted-process gate.

Two further outcome-integrity attacks are **PROVEN and report-only** because the
smallest repairs touch fenced `scripts/bounce_bot_lib/legacy.py`: shifted caches
with duplicate closes can recover the preceding bar, and conflicting milestone
rows can erase an earlier `stop_hit=True`. Ask-first authorization is required
before either repair or the autorun scheduling repair.

**Verification after all three code repairs:** two full suites each reported
**4801 passed / 19 subtests**, exit 0; their JUnit pass sets were identical
(`only_run1=0`, `only_run2=0`). Smoke was **7/7**, source selftest **70/70**, and
the unchanged frozen executable selftest **70/70**, all exit 0. No packaging
trigger fired. Working tree is reconciled; no live or promotion gate is marked
met.

**Immediate next action:** obtain the trader's ask-first decision for the three
fenced outcome repairs, specify and build the ordered session-scoped AWAY recap
feed, then repeat both post-close canaries. Phase 0 is not merge-ready while
those blockers remain.

---

## 2026-08-25 morning - first night on the new code: what the artifacts say

**Branch `testing-week-2026-08-24`.** Fable read the overnight artifacts; no
code changed. Desk restarted 2026-08-24 21:48 PT with `Outcome sweep autorun is
ON` logged; the trader runs AWAY today, DESK not before Thursday.

1. **The R10.A sweep ran in production, clean - 2026-08-24 16:45 PT**, in the
   pre-restart process (which already carried the R10.A code under test).
   `diagnostics/outcome_sweep_coverage.json`: pending_before **687** (the
   backlog had grown from 576 during Monday's session), finalized 687, expired
   7, failed 0, commit_failed 0, still_open 0. By reason: stop_hit 283 +
   stop_hit:legacy_csv_milestones 112, last_measured_bar 183 + legacy 95,
   no_measurement_in_checkpoint 7, expired_no_data 7. Log line 16:47:18. This
   is the R10.A mechanics observation; a second live day on the restarted
   process closes it.
2. **The 22:00 slate ran all nine slots** (ledger, session 2026-08-24):
   ai_summary ok (10 sources, 1 stale); ticker_briefs ok 175/175, 57 calls;
   veto grading 130 picks / 21 cohorts; like grading 45 / 28; evidence_report
   ok "window NOT met - scaffolding"; daily_digest ok (13,560 bytes, over the
   8 KB target, under the 16 KB cap, **n=0 outcomes**); enrichment and
   policy_draft both correctly gated. journal_import failed x3 then capped -
   and for the first time the rows SAY WHY (AI-P3).
3. **The Questrade chain is ALIVE - the token paste is not needed.**
   `journal_questrade_expires_at` refreshed to 2026-08-25T00:30:03; Questrade
   executions land daily (166 total; 2 on 08-24, latest 14:07 ET). The 204
   FAILED coverage days are all one cause: the `/activities` cross-check
   returns **400 Bad Request** on both accounts and the cross-check DOWNGRADES
   an imported day to FAILED (R10.0 named this; it is now the dominant journal
   defect and burns the slot's three attempts nightly). Yesterday's UNHEALTHY
   badge read the 08-21 oauth failure as current. Fix packet owed: the
   activities request (likely date-range/format) and, separately, a failed
   cross-check should read "activities unknown" on a COVERED day, not FAILED.
4. **IBKR Flex failed both nightly attempts** ("Statement could not be
   generated at this time") - transient, self-heal retries.
5. **UNKNOWN, flagged for Sol:** the digest reports n=0 outcomes for 2026-08-24
   on the same day the sweep finalized 687 trades. Either the digest's source/
   session filter is wrong or the finals carry a different session_date - the
   R10.I "n=0 beside real data" bug class.

**Verification baseline: unchanged** (4798 passed / 19 subtests; smoke 7/7;
selftest 70/70) - no repository change. Today's AWAY session is the routing
canary; after close: recap, journal entry, `outcome_sweep` and
`daily_bar_units` tiles.

---

## 2026-08-24 night (11) - Wave 1 is COMPLETE (8 of 8)

**Branch `testing-week-2026-08-24`.** Every packet in
`docs/analysis/OFFLINE_BUILD_AUTHORIZATION_2026-08-24.md` §3 is built, committed
and pushed.

| Packet | State |
|---|---|
| W1 P1.1 hermetic teardown | **DONE** |
| W2 R8's last deferred joins | **DONE** |
| W3 R7 true USD conversion | **DONE** |
| W4 Phase 2 Daily Digest Ledger | **DONE** - ten-session gate owed |
| W5 weekly synthesis machinery | **DONE** - runs gated |
| W6 LOCAL-AI P3/P4 machinery | **DONE** - runs gated |
| W7 P1.4 observability depth | **DONE** |
| W8 P1.5 repository hygiene | **DONE** |

**W8 was held and then authorized.** The removal reaches eight methods in
`alert_center_panel.py`, which houses alert code, so the file-scoped ask-first
rule was invoked; the trader answered "full removal" on 2026-08-24 and it was
done in one commit with no behavior change. What deliberately survives is the
generic `read_only` price-alert mode - a widget capability with its own tests,
now with no production caller - and that is stated rather than left as a silent
half-job.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4798 passed / 19 subtests**, exit 0 |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| `launch_gui.py --selftest` | **70/70**, exit 0 |
| `dist\TradingBotV3\TradingBotV3.exe --selftest` | **70/70 (frozen)**, exit 0 |

The suite count fell from 4868 because W8 deleted 70 tests with the code they
covered; every other packet added tests. **Exe rebuilt 2026-08-24 21:16:21,
committed the same minute** - Smart App Control did not refuse this hash, which
says nothing about the next one, and the desk still runs from source.

**NOTHING here is live-validated.** Every gate below stays owed, and this wave
added three more: the **ten clean digest sessions plus a trader spot-audit of at
least three fact packs** (W4), the **two weeks of graded cohort sessions** before
the weekly synthesis narrates anything (W5), and the **two-week policy-draft
side-by-side plus the trader's quality sign-off** before any draft becomes the
live policy (W6). R7's gates 1/3/6 (W3) and R8 §10's one real weekend (W2) are
unchanged. Nothing merges to `main` until a live-session validation day passes.

**Deliberately NOT built**, per the authorization §4: plan.md P2.1-P2.6 and P6.4
(authority cutovers, held for after the merge), P1.2, P1.3, Phases 4-7, any
frontier call, and any run of the gated machinery before its gate.

---

## 2026-08-24 night (10) - W8 is HELD on the file-scoped ask-first rule

**Branch `testing-week-2026-08-24`.** Wave 1 is **seven of eight**: W1-W7 built,
committed and pushed. **W8 (P1.5 repository hygiene) is NOT started, and is
deliberately blocked pending a trader answer.**

**Why it is blocked.** P1.5 removes the retired Desk Link / satellite /
mini-PC code. Measured, that removal reaches into
`scripts/ui/panels/alert_center_panel.py`, which houses alert code, in **eight
methods** - and two of them are decision paths rather than cosmetics:

| Method | Lines | What it is |
|---|---|---|
| `attach_desk_link` | 3 | relay handle |
| `attach_remote_feed` | 14 | satellite feed |
| `desk_link_stream_symbols` | 10 | relay symbol list |
| `apply_desk_link_intent` | 30 | **writes Focus** (`focus_service.add` / `remove_everywhere`) |
| `_relay_alert_popup` | 28 | imports `desk_link.popup_payload` from the alert flow |
| `_alert_has_focus_privilege` | 15 | **the feed gate, the beep and the relay all ask this** |
| `_current_bot` | 17 | satellite fallback |
| `_poll_auto_pick_pending` | 169 | the auto-pick adoption path |

The file-scoped ask-first rule (checkpoint review 2026-08-08) says any edit to a
file housing detector/scoring/alert code is asked about BEFORE it is made, *even
for capture-side or evidence-only changes*, and that ambiguity is the trigger to
ask rather than a licence to judge. The Wave 1 prompt repeats it: *"stop and ask
if a packet forces it."* This packet forces it.

**Partial removal is not an option**, and that is why nothing was started rather
than half-done. P1.5's own rule is a fully green cleanup packet never mixed with
behavior changes; deleting `scripts/desk_link/` while
`alert_center_panel._relay_alert_popup` still imports it would leave the tree
broken, which is the one thing the working agreement forbids outright.

**Scope, measured, so the answer can be given once:** ~4,625 lines of source
(`scripts/desk_link/` 7 modules, `ui/satellite.py`, `ui/desk_role.py`, the two
`ui/services/desk_link_*` modules, `master_avwap_mini_pc.py`), ~2,057 lines and
**63 tests** deleted with them, the Settings > Desk Link control and the
`--desk-role` startup flag, the `desk_link` entry in
`packaging/tradingbotv3.spec` (it is currently IN the bundle), and the matching
allowlist entries in the spec-drift test and `selftest`. It fires the packaging
triggers by design, so it ends with a rebuild and a frozen `--selftest` with the
new count recorded.

**Nothing else in Wave 1 is blocked by this.** W1-W7 are complete and pushed.

| Check at this stamp | Result |
|---|---|
| `pytest tests/ -q` | **4868 passed / 19 subtests**, exit 0 |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

Every canary and live gate listed in the entries below is still owed.

---

## 2026-08-24 night (9) - Wave 1, packet W7: observability depth (P1.4)

**Branch `testing-week-2026-08-24`.** Active item: Wave 1. **W1-W7 DONE**; only
W8 (P1.5 repository hygiene) remains.

`scripts/diagnostics/observability_trends.py` folds the run manifests and the AI
job ledger into a trend - per-phase latency against the previous window,
provider cache-hit and failure rates per family and source, run/job failures
with errors quoted, coverage from the scan's own counter. **Zero new
measurement**, pinned by an AST test. Frozen by the golden fixture
`observability_trends_v1`, whose inputs are hand-written to contain each shape
the reader has a rule for.

**It found something on its first live read.** `journal_import` has failed 9 of
its last 12 recorded runs and `ticker_briefs` 11 of 30. The first is the dead
Questrade OAuth refresh chain, which is already a known TRADER action (paste a
fresh token into Journal > Health); the second is not yet explained. Neither is
new behaviour - what is new is that one command says so instead of nobody
counting.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4868 passed / 19 subtests**, exit 0 |
| `observability_trends.py --window 5` | ran against live diagnostics, exit 0 |

Selftest and the frozen bundle untouched: `scripts/diagnostics` is already a
bundled package and this adds a module inside it, not a new top-level one.

---

## 2026-08-24 night (8) - Wave 1, packet W6: P3/P4 machinery, runs gated

**Branch `testing-week-2026-08-24`.** Active item: Wave 1. **W1-W6 DONE**;
W7-W8 not started.

Two slots appended, gated in two different ways because their gates are
different things.

**`journal_enrichment`** refuses below Phase 2's ten-clean-digest counter - no
model, no write - and reads that counter from `digest.digest_gate_state` rather
than keeping a second copy. Advisory only: a new `ai_trade_enrichment` table
written through the `JournalStore` API, append-only, with the trader's
`trade_annotations` row never opened (I7). Tags outside the
`SETUPS_MAJOR`/`SETUPS_TEST` vocabulary are dropped and counted.

**`review_policy_draft`** RUNS while its gate is unmet, deliberately: the gate
IS two weeks of drafts, so refusing would make the window unreachable. It writes
`review_policy_draft.json`, archives one copy per session, and says NOT MET in
the draft's own notes. The live `review_policy.json` is never written and cannot
be resolved from that module.

**A new journal table, and no migration cost.** `NEW_TABLES_V3` is applied
idempotently by `migrate_to_v3` on every open, so the advisory table appears on
the live database at its next launch with no schema-version bump and no
trader-present preparation.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4854 passed / 19 subtests**, exit 0 |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

Selftest and the frozen bundle untouched (`ai_jobs` is not in the bundle).
**Both phase gates are owed**: ten clean digest sessions with a trader
spot-audit before enrichment runs at all, and the two-week side-by-side plus the
trader's quality sign-off before any draft becomes the live policy.

---

## 2026-08-24 night (7) - Wave 1, packet W5: the weekly synthesis, gated

**Branch `testing-week-2026-08-24`.** Active item: Wave 1. **W1-W5 DONE**;
W6-W8 not started.

`scripts/ai_jobs/synthesis.py` rolls both graded cohorts up through
`evidence_stats` (one summary per cohort x side x horizon, capped at 40 with the
drop printed) and folds in the Phase 2 digest rollup once a pack exists.
**Below its gate it calls no model at all** and writes scaffolding whose first
line is `SYNTHESIS GATE NOT MET.`; above it, every figure is still `discovery`.

The gate counts SESSIONS in which a graded cohort row matured, pooled across
veto and LIKE - not rows, which one busy afternoon of vetoes could clear.
Earliest pass is early September on the current cohort.

**Never nightly, and structurally so**: a new `runner.optional_slots()` that
`default_slots()` never reaches, invoked by `run_ai_jobs.py --weekly-synthesis`
(pair with `--force` on a weekend morning; the market-session block is never
skipped). No frontier call is authorized.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4836 passed / 19 subtests**, exit 0 |

Selftest and the frozen bundle untouched (`ai_jobs` is not in the bundle).
**R8's live gate for this - one weekend where the trader confirms the ranked
reasons are the ones they recognise - is unchanged and still owed.**

---

## 2026-08-24 night (6) - Wave 1, packet W4: the Daily Digest Ledger

**Branch `testing-week-2026-08-24`.** Active item: Wave 1. **W1-W4 DONE**;
W5-W8 not started.

LOCAL-AI Phase 2 is built. `scripts/ai_jobs/digest.py` writes two artifacts per
session - a deterministic fact pack (zero LLM, written even when the model is
down) and a medium-tier narration that reads the fact pack and nothing else -
and `daily_digest` is APPENDED last in `default_slots()`. The six §6.4a
questions were ANSWERED, so the 2026-08-08 decision is met rather than waived,
and the answers are frozen into every pack.

**One near-miss worth recording.** Computing the slice's day-part meant
importing `bounce_bot_lib.learning` - the module that mutes alert segments - and
an existing test bars every `ai_jobs` module from reaching into live decision
code. It caught it. The fix reads the `env_key` R10.A already stamps into
`context_json` instead, which is both boundary-safe and the only spelling that
keeps ONE definition of "midday" between the digest and the learning state. A
row written before that stamp existed gets `unknown` as its day-part, never a
guess.

**Sizing is by construction, not by truncation.** Measured: the busiest shape
(600 outcomes, 30 environments, 16 slices kept) renders at **12,900 bytes**,
inside the 16 KB hard cap. Over-cap fails the job and writes nothing.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4817 passed / 19 subtests**, exit 0 |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

Selftest and the frozen bundle are untouched: `ai_jobs` is a documented
`PACKAGES_NOT_IN_THE_BUNDLE` entry, so a new module inside it fires no
packaging trigger.

**The ten-clean-session gate is OWED**, plus the trader spot-audit of at least
three packs against raw evidence. `clean_digest_sessions` counts; counting is
not passing. Every earlier canary and live gate remains owed.

---

## 2026-08-24 night (5) - Wave 1, packet W3: USD is booked, not estimated

**Branch `testing-week-2026-08-24`.** Active item: Wave 1. **W1-W3 DONE**;
W4-W8 not started.

R7's 2026-08-18 USD deferral is reversed by the trader's recorded decision and
BUILT. The gap was never the render seam: `rates_needed_for_trades` asked only
for each trade's own currency, so a CAD-only session had no USD observation to
convert from. It now asks for one on every session with trades, and
`book_currency_values` (was `book_cad_values`) books `net_pnl_usd` from that
trade's OWN session rate, with `fx_usd_rate` and the EFFECTIVE `fx_usd_rate_date`
beside it. A missing observation stays "unconverted"; a disappearing one clears
the booking rather than leaving a stale number.

Three additive columns on `trades` via `NEW_COLUMNS_V3`, which `migrate_to_v3`
applies idempotently on every open - **no schema version bump, so no
trader-present migration is required** on the live database.

**Golden fixture re-frozen with a note.** `journal_rebuild_trades_v1` gained
three columns and nothing else; the diff is 28 added lines and one changed
`intentional_difference`, and legs, opportunity events and the summary are
byte-identical.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4795 passed / 19 subtests**, exit 0 |

Smoke and selftest untouched by this packet and not re-run. **R7 gates 1, 3 and
6 are unchanged and still owed** - building a conversion does not validate it
against a broker statement.

---

## 2026-08-24 night (4) - Wave 1, packet W2: R8's DEFERRED block is empty

**Branch `testing-week-2026-08-24`.** Active item: Wave 1. **W1 and W2 DONE**;
W3-W8 not started.

Focus Pick Review gained the `human_focus_performance.csv` rollup and the week's
`pick_feedback.jsonl` verdicts; Week in Review gained the
`rrs_group_strength_extremes.csv` stream beside the symbol stream it already
folds. `docs/WEEKEND_PREP_PLAN.md` §6's DEFERRED block now says nothing remains.

**One deviation is on the record rather than papered over.** The spec asked for
the performance rollup "filtered to the week"; it carries no trade date, only
the `updated_at` stamp of its last rebuild, so a week filter would filter on
when the nightly pass RAN and empty the table on any week it did not. It renders
whole, as-of stamped, saying it is not week-scoped - the same treatment the two
cohort tables already get.

Also removed `_read_focus_week`, dead since 2026-08-18 and still carrying the
wrong-directory defect AI-P1 repaired in the live reader.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4782 passed / 19 subtests**, exit 0 |

Smoke and selftest untouched by this packet (no runtime module added or moved)
and not re-run. **R8 §10's one-real-weekend live gate is unchanged and still
owed** for all three streams - building a view never validates it.

---

## 2026-08-24 night (3) - Wave 1, packet W1: the suite's teardown is bounded

**Branch `testing-week-2026-08-24`.** Active item: Wave 1 of
`docs/analysis/OFFLINE_BUILD_AUTHORIZATION_2026-08-24.md` §3. **W1 (plan.md
P1.1) is DONE**; W2-W8 not started.

**Measured, then fixed, then re-measured.** A full run under a thread-recording
plugin found **22 tests leaving a thread alive past their own teardown and 19
`run_strategy` threads still alive at session end** - exactly the standing crowd
`conftest.py`'s GC block already named as unjoined. `conftest`'s new
`retire_leaked_bounce_bots` calls BounceBot's own cooperative `stop(timeout=...)`
for any strategy loop a test leaves behind, and fails the leaking test if one
survives. After: **0 scanner threads survive the session**, 7 leaking tests left,
all of which end on their own before the session does.

The 06:30-07:00 PT flake class was already repaired on 2026-08-23 (the two panel
tests pin their clocks); verified, not re-fixed.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4771 passed / 19 subtests**, exit 0 |
| Determinism | three consecutive full runs, identical counts, exit 0 |
| Stray scanner threads at session end | 19 -> **0** |

Baseline before this packet was 4767 passed; the four new tests are
`tests/test_suite_hermetic_teardown.py`. Smoke and selftest unchanged by this
packet (no runtime module added or moved) and not re-run.

**Nothing here changes a live gate.** Every canary and live gate listed in the
two entries below is still owed.

---

## 2026-08-24 night (2) - offline-build authorization; digest questions answered

**Branch `testing-week-2026-08-24`.** After the R10 slate completed, the trader
directed building everything that does not directly require live testing.
Recorded in `docs/analysis/OFFLINE_BUILD_AUTHORIZATION_2026-08-24.md`
(classified in `docs/README.md`): the six 6.4a digest questions are ANSWERED
(unlocking Phase 2's fact pack + narration), R7's true-USD deferral is
reversed, LOCAL-AI P3/P4 machinery is authorized with runs gated, and Wave 1
(digest, synthesis machinery, R8's remaining joins, USD, P3/P4 machinery,
P1.1/P1.4/P1.5) is defined. Authority cutovers (plan.md P2.x) are deliberately
held for after the live validation day and merge. Build prompt issued to Opus;
build not started at this stamp.

**Verification baseline: unchanged** (`db36459` lineage: 4767 passed / 19
subtests, exit 0; smoke 7/7; selftest 70/70 unfrozen and frozen) - docs only.

---

## 2026-08-24 night - the R10 slate is COMPLETE (10 of 10)

**Branch `testing-week-2026-08-24`.** Every packet authorized in
`docs/analysis/AI_DIRECTION_DECISIONS_2026-08-24.md` Â§3-Â§5 is built.

| Packet | State |
|---|---|
| R10.B outcome semantics + path capture | BUILT - canary owed |
| R10.C evidence_stats + scoreboard | BUILT |
| R10.D tracker transition ledger | BUILT |
| R10.E Focus membership events | BUILT - canary owed |
| R10.F like_cohort_grading | BUILT |
| R10.G market context ledger | BUILT - canary owed |
| R10.H Market Journal + two surfaces | BUILT - canary owed |
| **AWAY day recap + queue routing** | **BUILT - canary owed** |
| **Like cohort on the weekend surface** | **BUILT** |
| **Two opt-in scopes** | **BUILT** |
| **R10.I evidence_report + market_journal scope** | **BUILT - claims gate binding** |

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4767 passed / 19 subtests**, exit 0 |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| `launch_gui.py --selftest` | **70/70**, exit 0 |
| `dist\TradingBotV3\TradingBotV3.exe --selftest` | **70/70 (frozen)**, exit 0 |
| Golden fixtures | byte-identical, SHA-256 verified |

**The AWAY queue no longer accumulates.** The trader's 317-alert return is what
this fixed. The routing sits at `_enqueue_review_alert`, the single door into
the review queue, and everything upstream - the backing alert list, the D1 feed
and badge, History, every evidence stream - is asserted byte-identical between
an AWAY panel and a DESK one. DESK unchanged, EVENING unchanged, the AWAY hourly
phone pushes untouched per the resolved sub-decision.

**R10.I is scaffolding and says so.** The override covered sequencing only. Its
first live run read **13,394 ledger rows over 1 of 10 required sessions** and
printed COLLECTION WINDOW NOT MET as the first line of the report. The clock
starts at the first live session after the `outcome_sweep_autorun="on"` flip.

**FIVE canaries now owed**, and building never marks a live gate met:
- R10.B: LRSI registering gradeable rows; H1 stamping the bar close.
- R10.E: membership events on an add/remove/day-roll.
- R10.G: a regime shift writing a row.
- R10.H: a journal entry round-tripping through both surfaces.
- **The AWAY routing: one live AWAY day ending with an empty review queue and a
  populated recap**, with History and the evidence streams unchanged.

Also still owed and unchanged: R10.V's live scan day, the R10.A mechanics
canary, R9's four live proofs, the 2026-08-21 fluidity gates, R7 gates 1/3/6,
R8 Â§10's one real weekend, and the Questrade token paste. Nothing merges to
`main` until a live-session validation day passes.

---

## 2026-08-24 evening - R10 build slate (packets 1-7 of 10 done)

**Branch `testing-week-2026-08-24`.** Building the slate authorized in
`docs/analysis/AI_DIRECTION_DECISIONS_2026-08-24.md` Â§4: R10.B-R10.H, the AWAY
day recap (Â§5), two opt-in scopes (Â§3) and R10.I's machinery under the recorded
sequencing override.

| Packet | State |
|---|---|
| **R10.B** outcome semantics + path capture | **BUILT** - canary owed |
| **R10.C** evidence_stats + scoreboard | **BUILT** |
| **R10.D** tracker transition ledger | **BUILT** |
| **R10.E** Focus membership events | **BUILT** - canary owed |
| **R10.F** like_cohort_grading | **BUILT** |
| **R10.G** market context ledger | **BUILT** - canary owed |
| **R10.H** Market Journal + two surfaces | **BUILT** - canary owed |
| AWAY day recap | not started |
| Two opt-in scopes | not started |
| R10.I machinery | not started |

| Check after R10.H | Result |
|---|---|
| `pytest tests/ -q` | **4708 passed / 19 subtests**, exit 0 |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| `launch_gui.py --selftest` | **68/68**, exit 0 |
| `dist\TradingBotV3\TradingBotV3.exe --selftest` | **68/68 (frozen)**, exit 0 |
| Golden fixtures | **byte-identical**, SHA-256 verified on all five |

**R10.B touches a live writer, so its mechanics canary is OWED.** Building a
packet never marks its live gate met. One live session is needed to confirm
that LRSI crossings now register gradeable outcome rows (the engine has been
firing ungradeable alerts since it shipped) and that H1 stamps land on the bar
close.

**The frozen exe was rebuilt and it RAN.** R10.H fired the trigger (a new
left-nav page and seven new lazily-imported modules), so the bundle was rebuilt
and `--selftest` returned 68/68 (frozen), exit 0. Smart App Control did not
refuse this hash. **That says nothing about the next build** - SAC verdicts are
per file hash - and the desk still runs from source, so this is verification
rather than delivery.

**Three more canaries are now owed: R10.E, R10.G, R10.H** - a live session
each, confirming Focus membership events appear on an add/remove/day-roll, that
a regime shift writes a row, and that a journal entry round-trips through both
surfaces. R10.B's is still open. Building a packet never marks its live gate
met.

**What R10.F found on its first run:** 45 LIKE claims merged and graded across
28 cohorts. Read it beside the veto cohort on the weekend surface - the two are
the halves of one decision, and this is the first time both have had a forward
record.

**File-scoped ask-first, answered in session:** R10.H edits
`alert_center_panel.py`, which houses alert code. The trader authorized the
packet explicitly on 2026-08-24 ("go ahead and do R10E R10F R10G R10H"). The
edit is presentation only - a tab, a text box, a save button, a shortcut - and
touches no alert, tier, fold, digest or queue behaviour.

**R10.D: one audit premise did NOT reproduce, and that is the finding.**
S2 (setups carrying a mark dated later than the run's `data_session`) was
measured at 2,739 setups by the audit on a 2026-08-20 payload. On the current
payload - `data_session` 2026-08-21, written Monday over a completed Friday -
there are **14,043 marks and zero later than the vintage**. The defect needs a
tracker run during a live session to appear, so it is intermittent rather than
refuted; the guard is built and reports whatever it finds on every save.
S3a reproduced almost exactly (horizon 5 -> median 64 sessions, 10 -> 73, 42%
of rows over 2x their declared horizon) and S3b reproduced exactly (0 of 10,928
SPY-relative values). S3b is fixed from cached daily bars, zero IB. S3a is
MEASURED and FLAGGED but the future-row selection is deliberately unchanged -
re-selecting it would silently redefine every number the tracker has produced.

**R10.C makes the claim-kind split visible in the numbers the trader reads.**
On the 07-24..08-21 scoreboard window, of 5,970 settled above-the-floor rows
**4,442 were annotations and 526 observations - only 1,002 were entry claims**.
83% of what earlier reports ranked was not a trade. Section 1b prints every
moved family before and after with the rows removed and the claim kind that
removed them, so nothing moves silently. R9.3's 40-session window is reprinted
unchanged and the report states it did not measure it.

**What R10.B measured that changes how the store should be read:** the live
outcome store is **entry_claim 68,237 / annotation 147,713 / information
35,407**. Nearly 60% of it is H1 colour marks on already-closed bars. Any
existing statistic over "the outcome store" that did not filter by claim kind
was averaging those in.

**Two corrections found by reproduction rather than reading** (ground rule 3):
the registry's first draft invented two H1 family names and missed the largest
family in the store; and compound families (`10_candle_high-vwap_lower_band`)
made 158,053 rows read as unconfigured until parts-based classification landed.

**Frozen exe: no rebuild trigger from R10.B.** `outcome_semantics.py` and
`outcome_path.py` are modules under `scripts/`, not new top-level packages, and
are imported by name from already-collected code. The new fixture is under
`tests/`, which the bundle does not ship.

**Still owed on this branch** (unchanged): R10.V's live scan day, the R10.A
mechanics canary, R9's four live proofs, the 2026-08-21 fluidity gates, R7
gates 1/3/6, R8 Â§10's one real weekend, and the Questrade token paste. Nothing
merges to `main` until a live-session validation day passes.

---

## 2026-08-24 late - autorun flipped ON by the trader; AWAY quiet rule recorded

**Branch `testing-week-2026-08-24`.** Two trader actions, both taken in
conversation with Fable:

1. **`outcome_sweep_autorun` is now `"on"`** in
   `%LOCALAPPDATA%\TradingBotV3\local_settings.json` (trader directed the
   flip; a dated `.bak-20260824` sits beside the file). Takes effect at the
   next desk restart â€” the next live weekday session IS the R10.A mechanics
   canary, and it starts the two-week R10.A/B collection clock gating R10.I
   claims. Predicted first sweep (measured read-only 2026-08-23): 576 pending
   â†’ 560 finalizing on recovered milestone rows (271 stop-outs), 16
   `no_measurement_in_checkpoint`. Observe the `outcome_sweep` System Health
   tile after the close.
2. **AWAY quiet rule** (after a live AWAY day left 317 alerts waiting in the
   review queue): an AWAY day ends in an EOD recap, not a queue â€” recorded as
   a dated trader amendment in `docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md` and
   folded into the AWAY day-recap packet's spec
   (`AI_DIRECTION_DECISIONS_2026-08-24.md` Â§5). Evidence streams keep filling
   unchanged; the sub-decision on the hourly phone pushes was RESOLVED by the
   trader in the same conversation â€” the Master AVWAP scans and both AWAY
   hourly pushes stay exactly as built, the rule covers only the desk's
   review-queue accumulation. CLAUDE.md/AGENTS.md matrix bullet updated with
   the BUILD PENDING marker.

**Verification baseline: unchanged** (`7fa655b`: 4587 passed / 19 subtests,
exit 0; smoke 7/7; selftest 58/58) â€” settings + docs only. Opus's build
program (R10.C onward) continues; packet 8 must absorb the AWAY amendment.

---

## 2026-08-24 evening - direction decisions recorded; R10.B-H build authorized

**Branch `testing-week-2026-08-24`.** Trader decisions from the evening
conversation are recorded in
`docs/analysis/AI_DIRECTION_DECISIONS_2026-08-24.md` (classified in
`docs/README.md`): the summaries' reader is a later LLM (briefs stay);
the intraday market journal ask is R10.G/H/I as already specced; walk-away and
setup-scoreboard AI reads approved as opt-in scopes over deterministic outputs;
a new **AWAY day-recap** requirement (after-the-fact D1 analysis entry, Focus
adjustment, best-of-day pick review) is authorized with its constraints stated;
and the trader authorized building **R10.B, C, D, E, F, G, H, the two scopes,
the recap packet, and â€” by a precisely-scoped sequencing override â€” R10.I's
machinery** (its evidence-quality claims gate is NOT waived; reports over the
unmet collection window must say so). A build prompt for Opus was issued; the
build has not started at this stamp.

**Verification baseline: unchanged** (`4df4bac`: 4547 passed / 19 subtests,
exit 0; smoke 7/7; selftest 58/58) â€” documentation-only change. Owed items
unchanged, including the R10.A mechanics canary and the Questrade token paste.

---

## 2026-08-24 - review pass + documentation reconciliation sweep (docs only)

**Branch `testing-week-2026-08-24`.** Fable's review pass over the five AI-P
commits (`9209a75`..`4df4bac`): **no blockers** â€” both refuted premises
verified genuine at the source, the 162 tests in the six touched files pass,
each commit records a full-suite green, branch in sync with origin. Follow-up
reconciliation for the docs the build commits did not reach: `CLAUDE.md` (LIKE
picklist definition now in `ui/annotations/setup_claims.py`; new bullet for the
nightly-journal reason semantics, the Questrade token paste point, and the
named-constant lesson from the blank Focus Pick Review page) with `AGENTS.md`
re-copied; `docs/CHART_REVIEW_WORKSPACE_PLAN.md` Â§6 (picklist caveat is
derived, not retyped); `docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md` (dated
addendum for AI-P3/AI-P4, gates 1/3/6 advanced and still owed live). plan.md
deliberately untouched: these packets were never plan.md work items â€” they
came from the frozen review doc, and every owed live gate is unchanged.

**Verification baseline: unchanged** from the last build commit (`4df4bac`:
**4547 passed / 19 subtests**, exit 0; smoke 7/7; selftest 58/58) â€”
documentation-only change, suite not re-run. Active build item remains the
R10.A mechanics canary; trader actions outstanding: the Questrade token paste
and this weekend's R8 Â§10 one-real-weekend run.

---

## 2026-08-24 - AI-layer review delivered (analysis only, no code changed)

**Branch `testing-week-2026-08-24`.** Trader-directed analysis task: classify
the four AI slots by reproduction, argue the binding constraint, and design the
layer's evolution against R10. Deliverable:
`docs/analysis/AI_LAYER_REVIEW_2026-08-24.md` (frozen, hand-committed;
classified in `docs/README.md`). Headline: `veto_cohort_grading` is the only
PROVEN slot; the two model slots are UNKNOWN (no measured reader, `ticker_briefs`
is 72 min and the whole model bill); `journal_import` has 0 lifetime `ok` rows;
binding constraint is input poverty â€” the dead Questrade refresh chain
(0/142 days covered, trader portal action to fix, paste point = Journal â–¸
Health â–¸ "Questrade refresh token") and the 1-annotation / 220-candidate
confirmation starvation. The weekend AI-ledger silence was REFUTED as a defect
(designed one-`no_session`-row-per-slot dedup, verified in the per-firing
logs). Five packet proposals (AI-P1â€¦P5) and the trader questions are in the
document's Â§5/Â§7 â€” none is authorized by the document itself.

**Working state:** no file under `scripts/` or `tests/` touched. Uncommitted:
the new analysis file, this entry, and the `docs/README.md` row (plus the
pre-existing untracked `desk_report.xml`, not mine, left alone).

**Verification baseline: unchanged** from the cut below (4518 passed / 19
subtests, smoke 7/7, selftest 58/58) â€” documentation-only change, suite not
re-run. The active build item remains the one below: the R10.A mechanics
canary (trader flips `outcome_sweep_autorun="on"` on a live weekday), which
also starts the two-week collection clock that gates R10.I.

---

## 2026-08-24 - AI-layer packets: AI-P5, P3, P4, P1, P2 all done

**Branch `testing-week-2026-08-24`.** Building the packets in
`docs/analysis/AI_LAYER_REVIEW_2026-08-24.md` Â§5, in the order AI-P5 â†’ AI-P3 â†’
AI-P4 â†’ AI-P1, with AI-P2 conditional on trader assent.

| Check after AI-P2 | Result |
|---|---|
| `pytest tests/ -q` | **4547 passed / 19 subtests**, exit 0 |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| `launch_gui.py --selftest` | **58/58**, exit 0 |

**Frozen exe: no rebuild trigger fires.** AI-P5 adds no dependency, no
non-`.py` runtime asset, no new top-level `scripts/` package and no dynamic
import by string name. The moved picklist stays inside `ui/`, which the spec
already collects. Stated rather than silently skipped, per CLAUDE.md.

**AI-P5's stated premise was REFUTED; the packet shipped for a different
reason.** The review recorded `ai_summary`'s caveat as still describing only
the "Main swing" group. It did not - the text was corrected when the picklist
widened on 2026-08-21, and a test already pinned the corrected content. The
real defect is one level up: the caveat was hand-maintained prose duplicating a
code-owned list, so it could only ever be correct *until the next widening*,
and every test would stay green while it was wrong. It is derived now. Two
genuinely stale doc lines were found and fixed en route (Â§7.2's enumeration,
Â§6.4c's "QUEUED - do not build yet" on a slot that ships first).

**AI-P3's stated premise was also REFUTED, and again the packet was worth
building.** Reconcile mismatches never caused a FAILED slot - only a reconcile
*exception* does, and the reproduction test passed on its first run (it is kept
as a regression). The real defect sat one layer over: the job returned its
findings under `messages` while the runner reads `reason`, so **every**
`journal_import` row ever written was mute. That is why the actual cause - a
dead Questrade OAuth refresh chain, 0 of 142 days covered - survived five
nights undiagnosed. The night now reports what it measured, and a failure names
its own source. Reconciliation logic is untouched.

**AI-P4 found the thing AI-P3 could not say.** The Questrade chain is DEAD on
the live desk and the new System Health row says so in words, with the repair
step. This is a **trader action**: get a fresh refresh token from the Questrade
portal and paste it into Journal > Health. Until then Questrade contributes
nothing to the journal, and 0 of 142 days stay uncovered.

**AI-P1 found a live-desk defect while wiring the mirror cohort.** Focus Pick
Review had been rendering an **empty table since 2026-08-18**: `_join_focus_week`
resolved its CSVs under `PERSISTENT_DATA_DIR` (the home root) while they live in
`data/runtime`, and the step's own "a missing CSV is a quiet week" forgiveness
made the miss look like a plausible blank page. Both joins now address their
files by the named constants. Live read after the fix: **16 cohort rows and 605
focus pick rows** for the 08-17 week, where the pane showed zero. R8 Â§10's
one-real-weekend gate is the thing that would have caught this.

**AI-P2 was approved by the trader in session and built.** The weekend
auto-tag sub-pane has a default-off "show all pending proposals" toggle;
`WEEKEND_PREP_PLAN.md` Â§1 records the amendment. Measured on the live store:
**48 closed trades carry the 220 candidate rows**, so the whole backlog fits
inside the 60-row view and is one sitting's work.

**Still owed on this branch** (unchanged by AI-P5, P3, P4 or P1): R10.V's live scan day, the
R10.A mechanics canary, R9's four live proofs, and the 2026-08-21 fluidity
gates. Nothing merges to `main` until a live-session validation day passes.

---

## 2026-08-24 week - branch `testing-week-2026-08-24`

Cut from `phase05-integration-blitz` at **`1a2fbde`**, which carries the whole
2026-08-17 week: R9 (five items), R10.0/R10.0b, the R10.V daily-bar unit repair
(all seven steps, backfill applied to the live store), the R10.A ledger half
(rule registry, ledger, dual-write canary, no-fabrication finalization, the
after-close sweep, registration context, health tiles) and both review rounds -
Fable's and Sol's three reproduction blockers.

**The tree is identical to `phase05-integration-blitz`**; only the branch name
changed, so the desk's source launch is unaffected. The week's work continues
here, and `phase05-integration-blitz` stays where it is as the packet branch.

**Nothing merges to `main` until a live-session validation day passes**
(plan.md sec 6). Owed on this branch: R10.V's live scan day, the R10.A
mechanics canary (one live weekday with `outcome_sweep_autorun="on"`), R9's four
live proofs, and the 2026-08-21 fluidity gates.

| Check at the cut | Result |
|---|---|
| `pytest tests/ -q` | **4518 passed / 19 subtests**, exit 0 |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| `launch_gui.py --selftest` | **58/58**, exit 0 |

---

## 2026-08-23 - Sol's three blockers are fixed. Autorun is safe to enable.

**Branch `phase05-integration-blitz`.** plan.md Phase 0.7 / R10.A. Sol reproduced
three ways the sweep was unsafe; all three are closed, with the reproductions
themselves turned into tests.

### Blocker 1 - autorun could never actually sweep

At close+10 the worker fired, the sweep correctly deferred to close+35, and the
worker stamped `_learning_refresh_date` anyway because the refresh had
succeeded. At close+35 nothing ran: the day was already marked done.

**Two jobs now have two clocks and two completion stamps.** The sweep is due at
the real close + 35 minutes, once, only when the switch is on, and its stamp
goes down **only when it actually swept** - a deferral or a failure leaves the
day open for the next tick. The refresh is due at close + grace, once, and
**waits for the sweep whose rows it reads** when the sweep is enabled and has
not run. A successful refresh can no longer mark the sweep complete; the two
are separate variables and a test pins that. A worker already running does not
start a second.

**Early closes.** NYSE closes 13:00 ET on 2026-11-27, and `market_calendar`
models every close as 16:00 ET on purpose. Rather than change that - it feeds
detectors, scanners and the overnight window - the scheduler asks a **dedicated
seam**, `scripts/market_early_close.py`: day after Thanksgiving, 24 December
when it is a session, and 3 July when the 4th is a weekday. Unscheduled early
closes are not modelled and answer "regular", which fails in the safe direction
(the sweep waits longer, never runs early). `market_calendar` and
`market_session` are untouched, and a test asserts it.

| day | close | sweep opens |
|---|---|---|
| normal Monday 2026-08-24 | 16:00 ET | 13:35 local |
| half day 2026-11-27 | **13:00 ET** | 11:35 local |
| Christmas Eve 2026-12-24 | **13:00 ET** | 11:35 local |
| post-DST Monday 2026-11-30 | 16:00 ET | close+35 |
| weekend | - | never |
| autorun OFF | - | never |

### Blocker 2 - a crash between the append and the commit duplicated finals

The sweep appended every row and committed one checkpoint at the end of the
batch. Sol injected a failure in `os.replace`: the sweep reported `finalized=1`
while the disk still held the trade as pending with no finalized memory, and the
restart wrote a second final. On the 576-row backlog a crash near the end would
have duplicated most of it.

**Finalization is now one transaction per trade, with a write-ahead intent:**

1. take the machine-wide lock;
2. re-read the checkpoint from **disk**;
3. record the intent (`finalizing`) and **commit it before appending**;
4. append - unless the CSV already has that final, which is what an interrupted
   attempt leaves behind;
5. record the finalization and commit, with `fsync` on the file and the
   directory.

**A failed commit is not a finalization.** `finalize_outcome_once` returns
`commit_failed`, the sweep counts it separately, and the coverage report never
claims it. `_save_pending_bounce_outcomes` still swallows its failures - that is
right for milestone bookkeeping - but it **returns whether it landed**, and the
finalization path uses `_commit_checkpoint`, which raises.

`resolve_unfinished_finalizations()` settles anything left mid-transaction
against the CSV at load: row present means finalized (do not append again), row
absent means nothing happened (stay pending). Neither branch invents an outcome
and neither loses one. **Crash points covered by tests: before the append,
after the append, during the temp write, during `os.replace`, after the commit,
and mid-batch.** Every one converges to exactly one durable final.

### Blocker 3 - two processes could both finalize

`_pending_lock` is an in-process `RLock`. Sol started two real Python processes;
both loaded the same pending entry and both wrote a final.

**The transaction now takes `local_writer_lock`** - the repo's existing
machine-local primitive: a named mutex **and** a byte-range file lock, both
taken, failing closed when neither is in force - and **re-reads the disk inside
it**. A test starts two real processes that deliberately both load the
checkpoint before either commits: one returns `finalized`, one `skipped`, and
the CSV holds one row.

**`launch_gui.py` gained the authorized single-instance guard**
(`scripts/single_instance.py`). `launch_gui_auto.ps1` already refused a second
desk but only on that path; R10.0 measured pid 31848 overlapping three others,
the worst by 3.8 hours. It fails **open** when the machine has no primitive -
refusing to start the trader's desk over that is the worse failure - and
**closed** when somebody holds the slot, exiting 0 like the PowerShell launcher.
`--selftest` and `--run-scan` are deliberately outside it. `--allow-second-instance`
is the escape hatch. **The guard is defence in depth, not the transaction**: the
outcome path stays correct with two desks running, which is what the two-process
test proves.

### A finding I did not fix, with its blast radius

`market_session.get_market_local_timezone` falls back to
`datetime.now().astimezone().tzinfo`, which on Windows is a **fixed offset** with
no IANA key - today's offset applied to every date. A session window for a
November date therefore reads -07:00 on a desk currently on PDT. It does not
reach this scheduler, which only ever compares *now* against *today's* close
where today's offset is by definition right. It would reach anything reasoning
about a session months away, and fixing it moves displayed labels and slot times
across the desk. Recorded in `test_after_close_schedule.py` with a test that
documents it; **not** changed here.

### Verification (wall clock PT, pytest's own exit codes)

| run | start | end | result |
|---|---|---|---|
| new adversarial tests (durability, schedule, guard) | 09:05 | 09:12 | **49 passed**, exit 0 |
| existing focused packet (5 files) | 09:12:48 | 09:12:50 | **81 passed**, exit 0 |
| `pytest tests/ -q` | 09:18:29 | 09:22:22 | **4518 passed / 19 subtests**, exit **0** |
| `scripts/smoke_check.py` | 09:22:36 | 09:22:38 | **7/7**, exit 0 |
| `launch_gui.py --selftest` | 09:23 | 09:23 | **58/58**, exit 0 |
| Sol repro 1 (two real processes) | 09:22:56 | 09:23:00 | exactly one final, exit 0 |
| Sol repro 2 (commit failure, six crash points) | 09:23:00 | 09:23:03 | exactly one final each, exit 0 |

**The live stores were never written.** Checkpoint mtime is still
2026-08-21 12:18:55 with 576 pending / 0 finalized / 0 finalizing; the outcome
CSV is unchanged at 202.7 MB. Every test drives temp copies.

**No frozen rebuild.** No new dependency, no new runtime asset, no new package,
no `__file__`/`ROOT_DIR`/`sys.path` change. `launch_gui.py` gained one lazy
import, so `single_instance` and `market_early_close` were added to
`selftest.LAZY_ENGINE_MODULES` - the frozen run now proves them rather than
inferring them - and the spec-drift guard is green.

### The gate that remains

**One live weekday session.** The switch stays OFF until the trader flips it:
`outcome_sweep_autorun="on"` in `local_settings.json`. What Monday should show
if it is flipped, measured read-only against the live stores: 576 pending, 560
finalizing on their own recovered milestone rows (271 of them stop-outs), 16
reading `no_measurement_in_checkpoint`.

---

## 2026-08-23 - what flipping the sweep switch would actually do (measured, read-only)

**Branch `phase05-integration-blitz`, HEAD `7b2ed18`.** Taken against the live
checkpoint and the live outcome CSV without writing anything, so the trader can
decide on numbers rather than on a description.

| | |
|---|---|
| pending entries in the checkpoint | **576** |
| ...carrying `last_measured` | **0** - every one predates the field, exactly as the review said |
| recoverable from their own CSV milestone rows | **560** (550 from `12_bar`, 6 from `6_bar`, 3 from `3_bar`, 1 from `1_bar`) |
| ...of those, **stop-outs** | **271** |
| not recoverable -> `no_measurement_in_checkpoint` | **16** |
| backlog by trade month | 2026-06: **17**, 2026-07: **27**, 2026-08: **532** |

**Without MAJOR-2's recovery all 576 would have finalized as having seen no
bars**, including those 271 stop-outs. With it, 560 finalize on their own
measured rows and 16 say plainly that the checkpoint has nothing to measure.

The 17 June entries are the ones the audit found (oldest 2026-06-22), still
sitting there - the sweep is what removes them, and it is off until the trader
flips it.

**Nothing was written by this measurement**: `finalized` memory is still empty
and all 576 are still pending.

---

## 2026-08-23 - review round 1, part 2: the R10.A blockers are fixed

**Branch `phase05-integration-blitz`.** BLOCKER-1, MAJOR-2 through MAJOR-7 and
the minors, plus the Monday posture the trader asked for.

### The Monday switch, first

**`outcome_sweep_autorun` defaults OFF.** The sweep does not fire itself until
its first live session is signed off; calling
`sweep_pending_bounce_outcomes()` by hand always sweeps. It announces itself
once per process with the reason, so a desk that is not sweeping says so rather
than looking like a desk whose sweep found nothing. Everything else stays on -
dual-write, registration context, tier capture, `unresolved` instead of a
fabricated zero.

### BLOCKER-1 - two finalizers, twenty minutes, no lock

The sweep fired at close+10 on the after-close worker while the scan thread kept
finalizing through close+30, both mutating the pending dict and both saving a
checkpoint written with a bare `write_text` - and the loader answered a torn
checkpoint with `{}`, silently discarding the whole backlog.

- **One re-entrant lock** over every read-check-write of the pending dict and
  the save. Re-entrant because a finalize inside it calls the row writer, which
  saves.
- **The sweep re-reads each entry under the lock** - the id list is stale by
  construction - and **the per-symbol path now consults the same finalized-id
  set**, so neither can write a second final.
- **The sweep defers until close+35**, past the scan window. The lock makes an
  overlap correct; deferring makes it rare.
- **The checkpoint is temp + `os.replace`**, and an unreadable one is
  **quarantined and logged at ERROR** instead of read as an empty backlog.
- Test: two threads finalizing the same trade produce **exactly one** final.

### MAJOR-2 - Monday's sweep would have libelled the whole backlog

`last_measured` landed on 2026-08-23 and **0 of the 576 checkpoint entries carry
it**, so every backlog trade - including the 563 stop-outs whose milestone rows
sit in the CSV - would have finalized "no bars after entry". Bars were seen; the
state never recorded them.

- A trade with no measurement now reads **`no_measurement_in_checkpoint`**,
  which is a different fact from "the session produced no bars".
- **Each backlog trade's own CSV milestone rows are recovered** (one read-only
  pass, furthest milestone wins) as the measurement basis, tagged
  `measurement_source=legacy_csv_milestones` with `recovered_from`. `last_close`
  is reconstructed from that row's own `close_r`, entry and risk - arithmetic on
  stored numbers, not an estimate. A row with nothing usable is **not**
  recovered, because that would only relabel an absence as a measurement.

### MAJOR-3 - `close_r` means one thing everywhere

It was `(last_close - entry)/risk` with bars in hand and `-1.0` without: the
same trade reporting a different number depending only on what the finalizer
had. **`close_r` is now always the `eod_hold` number.** Without bars through the
close it is **blank** and the row is `unresolved` - never -1.0, and never the
last mid-session close either.

What was measured is still written (mfe/mae, best/worst, the stop and target
flags). The exit-policy question lives in `context.exit`:

- **`stop_exit_r`** = -1.0 **under a named assumption** - only the touch is
  measured, and the assumption travels in the row rather than hiding inside a
  number someone will later read as observed;
- **`gap_through_stop`** (`mae_r < -1`), where that assumption is optimistic;
- **`ambiguous_interval_bars`** - bars whose own range holds both the stop and
  the 1R target, where R10.0's predeclared **stop-first** rule applies and the
  count is reported rather than absorbed.

### MAJOR-4 - the measurement was taken from the forming bar

The frame comes from a request with an empty `endDateTime`, so its last row is
still forming, and `_rows_after_bounce_entry_for_session` had no completed-bar
cut. It does now, through the one shared rule (`completed_bars`), inclusive at
the boundary. **Authorization was conditional on proving the helper feeds no
detector**: it has exactly one caller, `_update_pending_bounce_outcomes`, which
writes outcome rows and nothing else - asserted in a test, not just grepped.
`replace(tzinfo=None)` is gone from this path; `_naive_market_local` converts
first and says so.

*Found on the way:* `completed_bars._TIME_KEYS` does not include `datetime`,
which is what this frame calls its column - handing it `{"datetime": ...}`
returns None for every bar and silently drops the whole frame. The shared rule
is untouched; the adapter is at the call site.

### MAJOR-7 - the refresh runs in SHADOW first

Corrected finals move segment averages, and segment averages decide `muted` and
`proven`, which decide whether an alert is suppressed. **`bounce_learning_refresh_mode`
defaults to `shadow`**: the refresh writes a state file beside the live one and
a diff of every segment whose mute or proven verdict **would** move, with n and
before/after R, to `diagnostics/bounce_learning_shadow_diff.json`. The live
state is frozen until the trader sets `live`. A segment whose average moves
without changing either verdict is deliberately not listed - it would bury the
ones that reach an alert.

### The rest

- **MAJOR-5**: the canary cap is **per session-day**, not per process (3.6k-6.1k
  rows/day against 50,000 silenced the mirror after 8-14 days), and binding it
  writes a **`canary_capped` event into the ledger** rather than only a log line.
- **MAJOR-6**: `pending_after` is answered from the row - a `final` ends the
  trade by definition - because the mirror runs inside the row writer, before
  the caller pops, so a membership test said `true` on every final ever written.
- Minors: sweep finals carry the measurement's bar count; `mfe_pct`/`mae_pct`
  are stored where the stop branch reads them; `_learning_refresh_date` is
  stamped **inside** the worker so a raising sweep is retried; coverage rows
  carry an `event_id` and the coverage file is written atomically; the ledger
  now documents that `session_date` is the **write** session and `trade_date`
  the trading one.

### UNCLEAR-8 - the red runs, recorded

Against the pre-fix writer (`2e87f4c`) with today's tests:
**64 failed, 17 passed**. Against the fixed tree: **81 passed**. The red run
covers every new behaviour above - the lock, the deferral, the CSV recovery,
`close_r` semantics, the completed-bar cut, the shadow refresh and the cap.

| Check | Result |
|---|---|
| `pytest tests/ -q` (own exit code) | **4469 passed / 19 subtests**, exit **0** |
| red run on `2e87f4c` | **64 failed / 17 passed** |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

### Owed, and named

The launch catch-up for the sweep is not built, and the sweep only runs while the
strategy thread is alive (so never in OFF). With autorun off by default that
changes nothing on Monday; it is R10.A's remaining piece along with the
single-instance guard, R9.5's shadow-store alignment and the ledger restore test.

---

## 2026-08-23 - review round 1: the suite's clock, and four R10.V corrections

**Branch `phase05-integration-blitz`.** Fable's 06:46 PT run found **2 failed,
4422 passed** where mine said 4424 passed at 07:26. Both were right.

**The suite was clock-dependent between 06:30 and 07:00 PT.**
`RepetitionLedger.consider(now=None)` reads the live clock, and inside the open
digest window an ordinary alert is legitimately folded into the digest row - no
feed row, no beep. `test_desk_still_beeps` and
`test_focus_privilege_waits_for_the_previous_day_extreme` both assert an
ordinary alert surfaces, so inside that half hour they read correct behaviour as
a failure. **The same class as the forming-candle chart bug**, and the second
one this week. Both now pin the clock to a moment outside any session's first
half hour; **the digest stays enabled**, because disabling it would hide a real
regression here. Two new tests cover the mechanism directly - an ordinary alert
inside the window IS digested, a privileged one never is.

**Also corrected: `PYTEST_EXIT=$?` after a pipe reports the tail's exit code.**
CLAUDE.md warns about exactly this. Every run below writes to a file and checks
pytest's own code.

### R10.V - independently verified, four corrections

Fable re-derived the backfill against the frozen copy: **every price cell
identical** (max |delta| 0, 0 rows added or dropped), 08-20 volume matching
`yfinance(auto_adjust=False)` at 1.000 on seven names, 0 rows dated after 08-21,
`files_changed` agreeing with the mtime count.

1. **The share percentage was flattered by its own denominator.** The health
   measurement dropped files with no `volume_unit` column: **1,136,420 rows,
   98.29% shares**, not 1,117,170 and 99.98%. The 19,250 rows in 38 pre-column
   files exist and still feed an AVWAP; they are `no_column` now, named in the
   tile, and the reconciliation JSON is corrected (it also had a **file** count
   sitting inside a **row** map) with what it supersedes recorded in it.
2. **Three unlabelled non-changes.** AVNS, SATS, SKYT got a history back whose
   dates overlap none of theirs and fell through as `status=ok` - still v1,
   still cliffed, in no bucket. They report **`no_overlap`** now and their
   cliffs count.
3. **The nine "no Yahoo data" symbols are UNSETTLED, not unfixable.** A Sunday
   re-probe reproduces it - BK and VSCO answer 404 "Quote not found", which is
   not plausible for Bank of New York Mellon. `--only-unfinished` now scopes a
   re-run to the **63** files still holding a non-shares row.
4. **The frozen pre-backfill copy is on the DAS**: 1,958 files, 47.4 MB,
   manifest SHA-256 identical on both sides, receipt filed beside it.

Out of scope but recorded so they are not rediscovered as defects: **605 of
1,958 files end before 2026-08-21** (pre-existing staleness - the backfill added
no dates), and **EQR's 08-18..20 rows are zero-volume placeholders on Yahoo
itself**.

| Check | Result |
|---|---|
| `pytest tests/ -q` (own exit code) | **4431 passed / 19 subtests**, exit **0** |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

**Next: the R10.A blockers.** Until BLOCKER-1, MAJOR-2, MAJOR-3 and MAJOR-7 are
fixed and green, the sweep's automatic firing is to be **switched off by
default** (trader decision).

---

## 2026-08-23 - R10.A: what Monday should show

**Branch `phase05-integration-blitz`.** An `outcome_sweep` tile now reads the
coverage the sweep files, so D3's fix is verifiable on the desk rather than by
digging through a diagnostics directory. It reads and never sweeps. **No report
is `unknown`, not healthy** - that distinction is the whole lesson of the
backlog. It degrades above 200 pending (D3 measured 576) or on a sweep older
than four days.

### The three things to look at on the first live weekday

| where | what should be true |
|---|---|
| System Health -> **Daily bar units** | healthy, ~99.98% shares, no `lots_rth` row |
| System Health -> **Outcome finalization sweep** | reported after close+grace, pending falling |
| `data\runtime\evidence_ledgers\` | `intraday_outcome_events-202608.jsonl` growing with the CSV |

Two of those depend on a job that has not run yet: the **nightly evidence
snapshot at 20:30 PT** writes the daily-bar unit measurement, and the **sweep**
runs in the after-close worker on a weekday. Until each runs once, its tile
reads `unknown` - correctly.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4424 passed / 19 subtests**, exit **0** (was 4419; +5) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

---

## 2026-08-23 - R10.A / D8: the tier lands on its own row

**Branch `phase05-integration-blitz`.** The audit found `tier` on **0 of 7,863**
registered rows. Reading the call sites settled why: every one of them registers
the outcome and evaluates the alert's tier **afterwards**. The tier does not
exist at registration, so the field could never have been filled.

**So it is recorded when it exists**, as a `tier_assigned` ledger event carrying
the id, tier, muted/proven/banger and the reason. That is what an append-only
store is for: a fact learned later goes on a later row rather than being
retro-fitted onto one that predates it. Wired at all **8** sites where a quality
verdict follows a registration, and a test walks the module source to prove none
was missed. **Reordering the alert path is a different kind of change** - it
would move a live alert flow - and is not this packet's to make.

**What a registration can honestly say, it now says**: family, engine version,
day-part (from the session window, not the wall clock), session RVOL, `env_key`
(environment + day-part, the pair the learning segments are keyed by), risk as a
percent of price, and risk as an ATR multiple. Measured or blank - never
estimated.

**A test earned its keep.** Making every accessor in that function misbehave
found `get_market_environment()` unguarded, which would have let a mid-update
read raise **into the alert path**. Guarded.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4419 passed / 19 subtests**, exit **0** (was 4405; +14) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

---

## 2026-08-23 - R10.A / D3+D4: the pending backlog has a way out

**Branch `phase05-integration-blitz`.** 576 pending outcomes, 94 of them older
than 2026-08-18 and 17 from June, existed because finalization only ever happened
inside the per-symbol update - which runs for a symbol the scan is looking at
*right now*. A name that stopped being scanned was never finalized at all.

**D4 is the same gap.** 2026-08-21: 409 `registered`, 399/398/397/394 milestones,
**0 finals**. The milestones ran all day; only the EOD pass was missing. Not an
IB outage, and the audit's PROVEN* is now settled as D3's consequence.

**The sweep needs no bars and no IB.** It finalizes from what each trade already
measured (the D2 rules), runs in the existing after-close worker **before** the
learning refresh that reads the rows, and a failure in it is logged without
costing the refresh.

**Idempotent by construction**: finalized ids ride in the same checkpoint as the
pending dict, bounded at 5,000, so a restart or a second pass cannot write a
second final. **Expiry is three completed sessions** - counted in sessions, not
days, so a long weekend cannot expire a two-session-old trade - and only for a
trade that measured nothing. One with evidence finalizes on that evidence however
old it is.

**It reports itself** to the ledger and to `diagnostics/outcome_sweep_coverage.json`.
A sweep that reports nothing is indistinguishable from a sweep that never ran,
which is exactly how this backlog stayed invisible for two months.

**One thing the tests forced.** `_is_eod_finalization_due` read the live clock,
so with the sweep pretending it was 08:00 on 08-21 the real clock still said the
session was over and the "still open" branch could not be tested at all. It takes
an optional `now` now; every existing caller is unchanged.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4405 passed / 19 subtests**, exit **0** (was 4387; +18) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

**What to watch on the first live weekday after close+30:** the sweep line in the
log, `outcome_sweep_coverage.json` appearing, and the pending count falling from
its current backlog rather than growing.

---

## 2026-08-23 - R10.A / D2: no path writes a number it did not measure

**Branch `phase05-integration-blitz`.** The most consequential change in R10.A,
and the one that changes what future rows say.

**What was there:** with no bars in hand at EOD the writer set
`close_r = 0`, `mfe = mae = 0` and `eod_close = entry_price`. That is 1,164 of
6,907 in-window finals (16.9%), every one with `eod_close` exactly its entry and
none of the 5,743 non-zero finals like it. **251 never advanced a bar. 563 were
trades whose own earlier rows had already recorded a stop hit** - stop-outs
scoring 0R, which drags every mean over them upward.

**What replaces it - three outcomes, each measured:**

| situation | finalization |
|---|---|
| earlier bars, one of them a **stop hit** | at the **stop**, `close_r = -1` |
| earlier bars, no stop | at the **last measured close** |
| nothing ever seen after entry | **`unresolved`**, blank numerics, reason `no_bars_after_entry` |

The state carries `last_measured` now, which is what makes this possible without
refetching anything. A 0R is a number a mean will average in; `unresolved` is not.

**No header widening.** `finalization.basis`, `.measured_bars` and `.reason` ride
in `context_json`, so a reader that does not know about them does not see them.

**The legacy rows keep their meaning.** `unsettled_close_mask` (the scoreboard's
detector) and `fabricated_zero_v1` (the registry's rule) are asserted to agree
row for row on the old signature; new `unresolved` rows deliberately do not match
it, because blank is not zero.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4387 passed / 19 subtests**, exit **0** (was 4375; +12) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

**What to watch on the first live session after close+grace:** finals carrying
`status=unresolved` instead of a 0R, and any stop-out finalizing at its stop.

**Next in R10.A:** idempotent finalization shared by the after-close pass and the
launch catch-up, expiring after 3 sessions without data, with a coverage manifest
and a health tile (D3 - 576 pending, 94 older than 08-18, 17 from June; D4 is
that same gap, not an IB outage).

---

## 2026-08-23 - R10.A: the ledger is live beside the CSV

**Branch `phase05-integration-blitz`.** The dual-write canary is wired and green.
Nothing reads the ledger yet - that is the point.

**One writer, one call site.** `_append_bounce_outcome_row` is the only function
that writes an outcome row, so it is the only one that mirrors, and a test
asserts exactly one call exists. The mirror runs **after** the CSV write: during
the canary the CSV is still the authority, and the new store cannot change or
fail what the old one recorded.

**Fail-open, bounded, and switchable.** A raising ledger is logged and swallowed;
the cap is 50,000 rows per process and announces itself once rather than once per
row; `evidence_ledger_dual_write="off"` stops it, and **only** that value does -
an unreadable setting leaves it running, because a canary that switches itself
off on an unrelated failure proves nothing.

**`family` is recorded once.** The CSV has never had the column - it lives inside
the event id, and every rollup has re-derived it, which is the same trap that
made my first `h1_bar_start_v1` validation tag 0 of 9,914 rows. The ledger row
carries it.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4375 passed / 19 subtests**, exit **0** (was 4361; +14) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

**What to watch on the first live session:** `data\runtime\evidence_ledgers\intraday_outcome_events-202608.jsonl`
appears and grows with the CSV; the two agree row for row; no "row cap" warning.

**Next in R10.A:** no-fabrication finalization (D2 - a session with no rows is
`unresolved`, and a stop-hit finalizes at its stop, never at its entry), then
idempotent finalization with a coverage manifest (D3/D4).

---

## 2026-08-23 - R10.A: the ledger the outcome store will be believed from

**Branch `phase05-integration-blitz`.** The store itself, before any wiring: it
can be tested on its own, and the wiring is the part that touches a live writer.

`scripts/evidence_ledger.py` writes `intraday_outcome_event_v1` into
`data\runtime\evidence_ledgers`, month-segmented. The properties that make it an
authority rather than a log:

- **A caller cannot overwrite the ledger's own fields** - schema, `event_at`,
  `session_date`, writer identity are applied last. A row that can lie about who
  wrote it is not evidence.
- **Two clocks on every row.** UTC for machine order, market session for trading
  order; the segment follows the session, so a 20:30-local write on 31 August
  lands in August.
- **A torn line is counted.** Power loss mid-append leaves a partial line, and
  `coverage_note` reports it beside n - a silently dropped row makes a gap look
  like an absence. A row with no session is excluded from a *window* and counted,
  but kept in an unwindowed read.
- **Append-only in fact**: a test asserts the segment only ever grows by suffix,
  and a correction is a superseding row with both versions surviving.
- **13 months hot**; `cold_segments()` names what is cold and deletes nothing.

Concurrency is stated rather than implied: thread-safe in-process by a lock,
and across processes an append is one line opened in append mode - rows
interleave whole, ordering is not guaranteed, and `event_at` is what orders them.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4361 passed / 19 subtests**, exit **0** (was 4337; +24) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

**Next:** wire the BounceBot outcome worker to it as a bounded dual-write canary
against the legacy CSV - the pending dict becomes a reconstructable checkpoint,
never the authority.

---

## 2026-08-23 - R10.A begins: the rule registry reaches v1

**Branch `phase05-integration-blitz`.** With R10.V code-complete, the R10.A ledger
half starts with the piece every later part reads: the reader-side rules.

**Four rules joined `daily_volume_mixed_v1`**, and each was **re-measured against
the live 239,422-row outcome store** rather than trusted from the audit:

| rule | reproduced |
|---|---|
| `duplicate_row_v1` | **742 / 609 / 430** on 07-24..08-21 and **394 / 345 / 300** on 08-07..08-21 - both windows exact |
| `risk_below_floor_v1` | **1,127** all-time finals under 0.1% of entry |
| `h1_bar_start_v1` | **9,623 of 9,914** minute-30 registered rows |
| `fabricated_zero_v1` | 2,964 all-time finals (the audit measured 1,164 in-window) |

**The registry states its measured precision, not a round number** - 9,623/9,914,
because 291 of 6,054 non-H1 rows also land on minute 30 and the family half of
the rule is therefore load-bearing.

**`duplicate_row_v1` will not run without its window** and echoes it back. The
same allegation reproduced at 742 on one window and 394 on another; a count from
this store that travels without its window is not evidence.

**One trap found by measuring rather than asserting.** The outcome CSV has **no
`family` column** - it lives in the event id. My first validation passed a family
that did not exist, and the rule tagged **0 of 9,914** rows without complaining,
because "no match" and "no data" looked identical from the outside.
`family_from_event_id` now derives it, a test pins the derivation, and the second
run reproduced 9,623 exactly.

**A suite failure that was the clock, not the code.** Running just after local
midnight on a Saturday, `test_unscanned_symbol_fetches_todays_candle_without_persisting_it`
went red. It forces `session_has_opened` True while the real time is before the
06:30 open, which lands inside the Yahoo early-print suppression window - so the
preview was **correctly** withheld and the test read that as a failure. The
suppression window is `test_forming_bar_honesty.py`'s subject; it is now pinned
in this test, with the reason written down. Nothing in the product changed.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4337 passed / 19 subtests**, exit **0** (was 4319; +18) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

**Next:** the ledger itself - `intraday_outcome_events.jsonl`
(`intraday_outcome_event_v1`, month-segmented), one owner, one transaction.

---

## 2026-08-23 - R10.V step 7: the S2 defect is closed. **The packet is code-complete.**

**Branch `phase05-integration-blitz`.** All seven steps built and green.

**A recompute may not see a bar that came after its session.** The catch-up held
today's daily frames while replaying a past session, which is how a payload whose
`data_session` said 2026-08-20 carried **2,739 setups with a snapshot dated
08-21** and 452 scenario exits on a bar that session had never seen. The frames
are now trimmed to `<= target_scan_date` before the indicator frame **and** the
record - both, because an indicator frame built from untrimmed bars would carry
the future into a record built from trimmed ones. The indicator cache is keyed by
`(symbol, session)` for the same reason.

**`<=`, not `<`**: the session being recomputed is the session we have. **An
unparseable session does not trim** - this is a point-in-time guard, not a filter,
and emptying every frame over a malformed stamp is a worse failure than the one it
prevents. A session before every bar leaves nothing, and the caller skips the
setup rather than marking it from nothing.

**Residual, stated rather than buried:** on a live same-day scan `target_scan_date`
IS today, so today's forming bar is still admitted - unchanged behaviour, and a
separate question from S2. plan.md sec 5's "completed bars only for state
transitions" would narrow that further; it is not this packet's authorization.

### R10.V exit gate - where it stands

| gate item | state |
|---|---|
| fixtures re-frozen with rationale | **met** - nothing moved, recorded with its prediction |
| backfill manifest filed | **met** - manifest + reconciliation under `evidence_frozen\` |
| 0 rows with `volume_unit != shares` that Yahoo can supply | **met** - 1,116,982 of 1,117,170; the 188 remainder have no Yahoo data |
| one live scan day on the repaired store | **OWED** - first weekday scan after 2026-08-23 |
| no scoring, sigma, ranking or threshold change | **met** - no golden fixture moved |

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4319 passed / 19 subtests**, exit **0** (was 4304; +15) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

**What to watch on the first live scan day:** the `daily_bar_units` tile must stay
healthy (a `lots_rth` row means something got past the write seam), the run
manifest's `provider.daily_bars.success.*` must be yahoo-only, and the tracker
payload's `data_session` must match the newest `latest_snapshot` date.

---

## 2026-08-23 - R10.V steps 5 and 6: a recorded no-op, and the nightly unit check

**Branch `phase05-integration-blitz`.**

**Step 5 found nothing to re-freeze, and that is the result rather than a skip.**
No AVWAP-derived golden fixture moved across provenance columns, the volume
policy, the collision rule and a backfill that rewrote 1,116,982 rows; the only
addition under `tests/fixtures/` is step 1's control. It was predicted before any
of it ran, for a stated reason - fixtures feed fixed bars - and had one moved at
step 4 the conclusion would have been that a test reads the live store and step
1's proof had expired.

**Step 6's tile reads; it never measures.** The measurement is ~7 s over 1,958
files, so it rides the nightly evidence-snapshot job and the tile reads the file
it writes. A failure in the measurement is logged and never fails the backup it
rides on.

**Two states, one of them actionable.** A `lots_rth` row means something got past
a write seam that refuses IB volume - degraded. The **188 `unknown` rows** are the
residue Yahoo has no data for, named in the backfill manifest; nobody can clear
them, so they are reported in full and set no status. The **53 cliffed files** are
reported the same way: 19 of them are all-`yahoo` and still step >20x because a
20x volume step is a real market event. A measurement older than two nights
degrades; no measurement at all is **unknown**, not clean.

Live tile right now: `healthy | 1,116,982 of 1,117,170 rows are share-denominated
(99.98%); no row carries IB round-lot volume. 188 row(s) remain unmeasured...
53 file(s) still step >20x; in an all-shares file that is a market event, not a
unit mix.`

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4304 passed / 19 subtests**, exit **0** (was 4290; +14) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

**Next:** R10.V step 7 - the tracker catch-up trims its frames to the session it
is recomputing (the S2 defect).

---

## 2026-08-23 - R10.V step 4: the store is repaired. 99.98% of rows are shares.

**Branch `phase05-integration-blitz`.** The backfill ran against the live store
on the Saturday with no desk process running and no scan due (quiet hours are
weekdays), after a dry run over all 1,958 files.

| | before | after |
|---|---|---|
| rows in `shares` | â€” | **1,116,982 of 1,117,170 (99.98%)** |
| files carrying a >20x volume step | **1,795** | **53** |
| median step ratio | 158x | **29x** |
| files on `daily_bars_schema=v2` | 0 | **1,920 of 1,958** |
| unmeasurable | 0 | 0 |

AAL is the case the reports were written around: 2026-07-24 74,218,900 â†’
2026-07-27 **93,953,900**, where it read 836,047 this morning. Every sampled
file (AAL, NVDA, TSLA, AAPL, SPY) is v2, all-`shares`, all-`yahoo`.

**Prices were not touched.** Only `volume`, `source` and `volume_unit` were
written; open/high/low/close and the set of dates came out exactly as they went
in, because prices in this store are fine and a second unmeasured change inside
this one would be untraceable.

**A verified frozen copy came first**, `evidence_frozen\daily_bars_pre_backfill_2026-08-23`,
file-count and byte-total checked, and the run refuses to start if the freeze is
incomplete. Zero IB traffic: yfinance only, batched, `auto_adjust=False`.

**Two refusals came out of the dry run, not out of a guess.** Both are now
tested.

* **Coverage.** Yahoo returned a near-empty history for EA, TMHC, JHG, SATS and
  AVNS - a rewrite would have changed **2 of EA's 787 rows**. That does not
  repair a file; it manufactures a second unit boundary inside one. 13 files
  skipped, named, left exactly as they were.
* **Worsening.** Any file this run would leave with a cliff it did not have (or
  a bigger one) is left alone: a repair that can make a file worse is not a
  repair. 13 files skipped.

Also learned live: a batched download silently drops the odd ticker (**BK came
back empty in a batch and I retried it individually**), and `CON_.parquet` holds
`CON` - Windows cannot name a file after a device, and Yahoo has never heard of
`CON_`. Both fixed. After the retry pass, **9 symbols genuinely have no Yahoo
data** (BK, CPRX, CWAN, EXPI, IAC, LC, NUVL, PRA, VSCO - confirmed one at a time)
and their files are untouched and named.

**The exit gate as written could not be met, and the reason is not a defect.**
"0 files > 20x" is unachievable, because a 20x volume step is a real thing that
happens to real stocks: after a full single-source rewrite, 19 files still show
one - DJT at its 2024-01-16 listing, OKLO's 2023-09-14 de-SPAC, POET, FFAI, QXO,
SOXS - with **every row `source=yahoo`**, so the step cannot be a unit artifact.
plan.md's gate is corrected to the falsifiable one: **0 rows with
`volume_unit != shares` that Yahoo can supply**. The cliff detector stays as a
secondary signal, where a cliff in an all-`shares` file reads "market event".

**A reporting bug of mine, found by checking rather than by trusting.** The
applied run's manifest said 44 cliffed-after; an independent scan of the same
store said **53**. The store was right and the summary was nine short - the nine
files Yahoo had no data for kept their cliffs and were never added to the
after-count. Fixed, with a test that asserts the manifest reconciles against
`scan_store()`, and a reconciliation note filed beside the manifest rather than
the manifest being quietly rewritten. The manifest also carried the frozen
copy's UTC date while naming itself with the local date; both now use the run's
stamp.

**Artifacts** (machine-local, under `evidence_frozen\`): the pre-backfill copy,
`daily_bars_pre_backfill_manifest_2026-08-22.json` (per-file rows rewritten, rows
left unknown, first-cliff date and ratio before and after), and
`daily_bars_pre_backfill_reconciliation_2026-08-23.json`.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4290 passed / 19 subtests**, exit **0** (was 4265; +25) |
| live store scan | 53 cliffed / 1,958 measurable / 0 unmeasurable |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

**Next:** R10.V step 5 - re-freeze any AVWAP-derived golden fixture that moved
(predicted: none), then step 6's health tile and step 7's forming-bar trim.

---

## 2026-08-22 night - R10.V step 3: the store takes shares, and a collision prefers them

**Branch `phase05-integration-blitz`.** The behaviour-changing step, and the one
that makes the splice structurally impossible rather than merely unlikely.

**An IB row is written with its prices and NO volume.** Not a rescaled number:
the measured ratio is symbol-dependent (SPY 1.0x, TSLA 56x, AAPL 81x, A 162x,
NVDA 188x), so a x100 conversion would replace a visible error with an invisible
one. The row says `volume_unit=lots_rth`, so the absence is explained rather than
merely present.

**A date collision now prefers `shares` > `unknown` > blanked**, in either
arrival order. `keep="last"` handed the session to whichever scan ran last, which
is exactly how a share-denominated Yahoo row was replaced by an IB row measured
in round lots. Among rows of equal standing the later one still wins - the
previous behaviour, preserved. **The rank follows the DATA, not the label**: a
row whose unit says `shares` but whose number is missing cannot outrank one that
has a number.

**The deliberate exception, stated rather than buried: `unknown` legacy rows keep
their volume.** Blanking them would empty the volume column of the entire
existing store between this step and the step-4 backfill, and an AVWAP with no
weights is not a safer answer than one with an old weight - it is no answer at
all, for every symbol, live. The grandfathering ends at step 4's exit gate: zero
rows with `volume_unit != shares`.

**A blanked row stays readable.** `dropna` now disqualifies a row only for a
missing PRICE - dropping it for a missing volume would delete the price bar too -
and both weighting loops skip a blank exactly as they already skipped a zero.
That guard matters: **NaN is not `<= 0`**, so without it a single blank bar would
poison `cumVol` and take the whole level with it. **The sigma formula is
untouched** (plan.md sec 5); what changed is which bars enter it, on the rule the
function already applied to a zero.

**Three more readers were made blank-safe**, found by looking rather than by
waiting for a crash: `chart_snapshot.load_d1_bars` emits `0.0` (what it already
emits for a file with no volume column - one "no data" value, not two, and NaN
must never reach the paint path); `avg_vol_20` reads 0 when twenty bars are all
blank, which correctly **rejects** the candidate at the liquidity gate rather
than raising on `int(nan)`; `last_volume` becomes `None`, never 0, because it is
a bucketed liquidity factor and 0 reads as "illiquid" rather than "unknown".

**No golden fixture moved.** `git diff tests/fixtures/` is empty, as the step-1
baseline predicted for every step that does not touch the band formula.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4265 passed / 19 subtests**, exit **0** (was 4247; +18) |
| `git diff tests/fixtures/` | empty - ground rule 1 holds |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

**Next:** R10.V step 4 - the batched yfinance backfill, with a dated
pre-backfill copy in `evidence_frozen/` and a manifest. Zero IB traffic.

---

## 2026-08-22 night - R10.V step 2: the store records where each row came from

**Branch `phase05-integration-blitz`.** Provenance travels **with the row**, not
with the frame: `source` (`yahoo` | `ibkr` | `unknown`) and `volume_unit`
(`shares` | `lots_rth` | `unknown`) are columns, because the store IS a merge of
two sources and a frame-level attribute cannot survive one. The file additionally
carries `daily_bars_schema=v2` in its Arrow metadata, which is what separates
"this file predates provenance" from "this file has provenance and every row of
it is unknown".

**`cache` is deliberately not a source value.** Reading a row off disk tells you
it came off disk, not what wrote it, so a v1 row reads `unknown`/`unknown`.
Recording `cache` would look like provenance while carrying none - and the point
of the column is that `unknown` shows up in a rollup.

**A bug I wrote and caught with the test I wrote for it.** The first version set
the source *after* normalization at both fetch seams
(`_set_daily_bar_source(_normalize_daily_bar_frame(df), YAHOO)`), so every row
was stamped `unknown` while the frame said `yahoo` - two new columns of nothing
on every fresh fetch. The source is now declared **before** normalization at both
seams, and `test_the_yahoo_fetch_stamps_every_row_it_returns` exists so it stays
that way. `_set_daily_bar_source` remains a pure attribute setter: making it
backfill unknown cells looked tempting and would have relabelled old IB rows as
Yahoo at the merge seam (`legacy.py` ~15395), which is fabrication.

**Rows that already know what they are are never relabelled.** That is the whole
reason the mix was invisible; `_normalize_daily_bar_frame` fills blanks only, and
stamps before the de-duplication so step 3's collision rule has two rows that
both know what they are.

**An untouched file stays v1 on purpose.** `_persist_durable_daily_bars` still
skips a write when the bars did not change, so a v1 file is not quietly upgraded
to a v2 full of `unknown`. Step 4's backfill converts them, with a manifest
saying which.

**Every consumer reads both schemas, proven one test per consumer** - the D1
scanner's durable loader, `chart_snapshot.load_d1_bars`,
`human_focus_tracking` (which is also how `ai_jobs/cohorts.py` grades vetoes),
`setup_playbook_study`, and **two the cliff report's consumer table had missed**:
`ui/services/bar_cache.py` and `research_warehouse/ingest_existing.py`. Both read
by column name and are unaffected. The warehouse's `provider="UNKNOWN"` docstring
is now understated - v2 rows carry a real source - but wiring that through is a
warehouse change this packet does not authorize: **owed, not done**.

**No golden fixture moved**, which is what `AVWAP_FIXTURE_BASELINE_2026-08-22.md`
Â§3 predicted: `git diff tests/fixtures/` is empty. One test asserted the old
six-column contract and was updated to the new one - a contract change this
packet authorizes, made visible rather than worked around.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4247 passed / 19 subtests**, exit **0** (was 4215; +32) |
| `git diff tests/fixtures/` | empty - ground rule 1 holds |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

**Next:** R10.V step 3 - the write seam accepts only `shares`; an IB frame
contributes price columns and `volume=NaN`, never a rescaled number, and the
date-collision rule becomes prefer `shares` > `unknown` > NaN.

---

## 2026-08-22 night - R10.V registered, step 1 done: the fixtures are proven clean

**Branch `phase05-integration-blitz`.** The cliff packet is now plan.md Phase 0.7
item 11 (**R10.V**), and it runs **before R10.D** - a point-in-time transition
ledger built over a unit-mixed store would record the splice as history.

**The stop condition did not fire, and it was measured rather than inspected.**
The step said: stop and hand off if any golden fixture reads the live parquet. A
pytest plugin wrapped `builtins.open`, `Path.open`, `Path.read_bytes` and
`pandas.read_parquet` to record any access resolving inside
`C:\TradingBotData\data\daily_bars` (1,958 files) or `data\intraday_bars`, and
the **whole suite** ran under it: **4205 passed, 0 accesses**. Every fixture
carries its own bars. The claim's limit is stated in the record: it proves those
two roots were not read in that configuration, not that no path could ever
resolve elsewhere.

**`mixed_unit_avwap_v1` pins the wrong answer on purpose.** Twenty
hand-constructed daily bars, three series with **identical prices** differing
only in volume:

| series | vwap | sigma | UPPER_2 |
|---|---|---|---|
| `shares` (Yahoo throughout) | 42.263138 | 1.259667 | 44.782472 |
| `mixed` (bars 12+ in IB lots) | **41.301207** | **0.607158** | **42.515523** |
| `lots` (every bar /100 - control) | 42.263138 | 1.259667 | 44.782472 |

The splice costs **-2.28% on VWAP, -2.27 points on UPPER_2, and halves sigma
(0.482x)**. The uniform rescale costs **nothing** - `lots` reproduces `shares` to
0.0 on vwap and 1.3e-15 on sigma. That single row is the whole argument for
C-prime: a volume-weighted ratio cancels a constant factor, so if the store were
uniformly mis-scaled there would be nothing to repair, and the x100 conversion
option C originally proposed would have replaced a visible error with an
invisible one.

**Both guards were proven to discriminate before commit**: a 0.0001 drift in one
expectation fails the comparison, and editing an input bar without re-freezing
the hash fails the Milestone 3 contract loader with `raw input hash mismatch`.
The sigma formula now has a direct guard too - an independent reimplementation of
the running-deviation variant must agree, and the distribution-stdev variant must
*disagree* on this fixture, or the guard cannot discriminate.

**Step 5's blast radius is predicted in advance** (`docs/analysis/AVWAP_FIXTURE_BASELINE_2026-08-22.md`
Â§3), so a surprise will be visible as one: only two fixtures put bars through
`calc_anchored_vwap_bands`; three more carry already-computed levels as inputs;
the backfill **cannot** move any fixture, and if one moves at step 4 then
something reads the live store and the Â§1 proof has expired.

**Also repaired:** plan.md line 1171 contained a literal backspace character -
`Data\backups` written through one of my earlier heredocs became `Data\x08ackups`
and rendered as `Dataackups`. Fixed, and the whole repo swept for control
characters: **0 remaining**.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4215 passed / 19 subtests**, exit **0** (was 4205; +10) |
| `pytest tests/ -q -p liveguard` | 4205 passed, **0 live-store accesses** |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

**Next:** R10.V step 2 - provenance columns (`source`, `volume_unit`) on the
parquet with `daily_bars_schema=v2`, every consumer reading v1 and v2.

---

## 2026-08-22 night - R10.0b small items 2 and 3: the amendments, and the rule registry

**Branch `phase05-integration-blitz`.** Â§3 of the decision release, items 2 and 3.

**The two reports now supersede their own numbers.** My Amendment 2e said "1,309
differ, only 5 materially" - measured on same-dated historical closes only, which
undercounted. Fable's field-level re-run over 60,519 mark-days replaces it: 26,087
float32-to-float64 round-trips, 7,104 of the remaining 7,465 at or under 1.1 cents
(sub-penny prints and one-cent vendor disagreement about an extreme), genuine
restatement **361 field-diffs = 136 symbol-dates on 113 symbols, max 1.9%**, and
the ten largest close moves are SCCO at exactly x0.98814 - a dividend.

**S1 now names its mechanism.** "History rewritten" reads **targets rewritten by
re-weighted levels; stops and closes stable**. The 08-21 07:0x run spliced IB
round-lot volume onto Yahoo share history at 2026-07-29 (median x0.0088 in 1,179
of 1,236 rewritten files), so every AVWAP anchored before the splice froze near
its 07-28 value; 30,003 of 60,519 mark-days carry different levels, and **0 of
9,331** stored anchor entries and stop references moved, because those are written
at scan time and never replayed. The stop stayed put while the replayed target
moved beneath it. A uniform rescale could not do this - AVWAP is a volume-weighted
ratio, so scaling every weight cancels - which is the argument for R10.V refusing
IB volume rather than converting it. Both reports also now carry the trader's
answers beside the questions that asked for them.

**`scripts/evidence_rules.py` exists** (R10 ground rule 5): a reader-side registry
that tags known-bad rows by a versioned name instead of editing them. It reads and
never writes, and reaches no detector, score, gate, alert or Focus decision.

**`daily_volume_mixed_v1` is derived, not declared.** A session is `mixed` if any
run manifest that day reported a non-`yahoo` `provider.daily_bars.success.*`,
`shares` if all reported Yahoo, `unknown` otherwise. `mixed` dominates - one IB
run contaminates the session, which is exactly 2026-08-20, where two desks ran
concurrently and only one used IB - and `unknown` beats `shares`, because a
manifest we cannot read may have been the IB one.

**The result is wider than the two runs the trader named, and that is the point of
deriving it: 13 of 15 manifest-covered sessions are mixed**, back to 2026-07-31
(the edge of the 90-run retention); only 08-03 and 08-17 are clean, and everything
older reads `unknown` - and will read `unknown` increasingly as manifests prune.
`freeze_verdicts()` is how a rollup that must stay reproducible files what its
numbers relied on.

**No rollup is wired to the tag yet, deliberately.** The defect is confined to the
**volume** column, so it moves volume-weighted AVWAP levels and leaves price-only
readers alone - the veto-cohort grader reads the same parquet store and is not
affected. The one existing evidence rollup (`setup_scoreboard.py`) reads BounceBot
intraday outcomes, a store this has not been shown to touch, so tagging it would
assert an influence I have not proven. R10.A and R10.V build the real consumers.

**System Health** appends the manifest-derived history to the `daily_bar_source`
tile. History never sets the status: the pin governs what happens next, and a past
that cannot be changed must not raise a permanent alarm.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4205 passed / 19 subtests**, exit **0** (was 4182; +23) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

---

## 2026-08-22 night - R10.0b Â§1.3: daily bars pinned to Yahoo (interim)

**Branch `phase05-integration-blitz`.** The one live-path line the trader
authorized, and only that line.

`daily_bars_source` in `local_settings.json`: `"yahoo"` pins; absent or anything
else - including a typo - resolves to `"auto"`, which is exactly today's
behaviour. Read at `_fetch_live_daily_bars` (`master_avwap_lib/legacy.py`), the
same seam as the existing circuit breaker and **independent of it**:
`_IBKR_HISTORICAL_YAHOO_ONLY` is a *state* flipped by repeated IB failures,
this is a *setting*, and either alone routes daily bars to Yahoo. Announced
**once per scan** (latched on `reset_ibkr_historical_failure_circuit`), not once
per symbol - 1,500 identical lines would bury the fact rather than report it.
**Intraday is deliberately untouched.**

System Health gains a `daily_bar_source` tile. `auto` reports **unknown**, not
degraded: it is the shipped default, and with it we genuinely cannot say from
the setting alone which source a given scan wrote - the run manifests carry
that. `yahoo` reports healthy. **Set on the desk**; the tile reads healthy.

**Golden fixtures do not move** - 482 fixture/AVWAP/tracker tests pass unchanged,
as the release predicted, because they feed fixed bars.

**A bug I introduced and contained.** The first version wrote
`daily_bars_source` from inside `_measured_fixture`. `local_settings.json` is
redirected per *session*, not per test, so the write outlived the module and
made `_fetch_live_daily_bars` return Yahoo before `test_provider_counters` could
count an IB attempt - two unrelated tests went red. Now a module-scoped autouse
fixture that saves and restores.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4182 passed / 19 subtests**, exit **0** (was 4172; +10) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

---

## 2026-08-22 evening - R10.0b COMPLETE: the daily-bar volume cliff. STOP.

**Branch `phase05-integration-blitz`.** Read-only. **Nothing was changed** - no
fetch, normalize, scanner or level code touched, no parquet written. Report:
[`docs/analysis/DAILY_BAR_VOLUME_CLIFF_2026-08-22.md`](docs/analysis/DAILY_BAR_VOLUME_CLIFF_2026-08-22.md),
classified in `docs/README.md`. **The program stops here for the trader's
decision**, per the release's Â§3.

**Measured.** 1,958 parquet files, 1,737 measurable. **1,227 (71%) carry an
early/late volume ratio > 20x**, median **158x** - matching the release note's
1,227 exactly. First-drop dates scatter across months (07-27: 473, 06-04: 302,
05-26: 175, ...), so it is per-symbol and not one event.

**The finding that constrains every fix: the factor is NOT 100.** Parquet vs
Yahoo for 2026-08-20 - SPY 1.0x and ACHC 1.0x (shares, agree), but TSLA **56.5x**,
AAPL **81.3x**, A **161.9x**, NVDA **188.0x**. A blanket x100 backfill would
replace a wrong number with a differently wrong one and destroy the ability to
tell afterwards.

**Root cause located.** `bounce_bot_lib/legacy.py:630` has
`IB_HISTORICAL_VOLUME_LOT_SIZE = 100`, applied at `:11376`, with a comment
recording that this exact bug was already found and fixed once *in that engine*
("without this the ratio deflates ~100x and gates out every bounce alert").
**`master_avwap_lib` - which owns the daily-bar parquet - has no lot handling at
all.** And the parquet keeps **no provenance**: six columns, pandas-typing
metadata only. `fetch_daily_bars` carries a source on the in-memory frame and it
is dropped before the write, so no existing row can be asked which source made it.

**Blast radius.** Every in-session run on 2026-08-21 used IB
(1,222/422/468/460/459/450 vs 9-22 Yahoo), so **the 08-21 scan ran on cliffed
volume**; the post-close run reverted to Yahoo. AVWAP bands are volume-weighted,
so the D1 scanner, the setup tracker, `chart_snapshot`'s D1 payloads,
`human_focus_tracking._load_durable_daily_frame` and `ai_jobs/cohorts` all
inherit it. **The M5 Strength Board is NOT affected** - it does its own batched
yfinance 5m download and never reads this store.

**Four options in the report** with their golden-fixture impact. My
recommendation is **C: provenance first (source + unit column, plus the missing
rescale on the master_avwap IB path), then refetch history with the AVWAP
fixtures re-frozen in the same packet** - the only order in which the repair is
verifiable afterwards.

**Deliberately not established:** why the factor is 56x-188x rather than 100x. I
did not open the IB adapter's bar-request parameters, because that is live fetch
code and Â§3 says change nothing and hand off. The explanation is owed.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4172 passed / 19 subtests**, exit **0** |

**Three questions for the trader** (report Â§7): which option; whether the 08-21
scan output is to be treated as suspect; and whether the daily-bar fetch should
pin to one source until this is settled (a one-line change to a live path, so an
ask rather than something I would do unprompted).

**R10.A's ledger half remains not started**, gated on this decision per the
release's ordering.

---

## 2026-08-22 evening - Â§2 LANDED: snapshot scheduled, `.bak` excluded, source hash added

**Branch `phase05-integration-blitz`.** Trader answers to the audit's Â§8, applied.

- **Q2 â€” `.bak` excluded.** Once the snapshot runs nightly, day N's main IS day
  N+1's `.bak`. Excluded by an explicit rule carrying the reason
  `excluded_rotated_duplicate`, counted in the manifest like every other skip,
  never a silent omission. Measured on the live scope: exactly one file,
  **939 MB source / ~133 MB compressed saved per night**. `exclude_rotated=False`
  is the switch Â§0's frozen pair used. **The on-disk `.bak` is never deleted** â€”
  the tracker reads it back when the main payload is corrupt.
- **Q5 â€” 13 months hot accepted** (unchanged from the audit's proposal).
- **Â§2.8 â€” `source_sha256` added** beside the stored hash. The stored hash proves
  the archive; only the source hash proves the CONTENT survived compression, and
  it is the only hash a restored file can be compared against. `verify()`
  deliberately stays on stored bytes so it remains cheap â€” no decompression. For
  a SQLite copy the two are necessarily different: the backup API rewrites page
  layout, so comparing them would be wrong, and the code says so.
- **Scheduling authorized and done.** `TradingBotV3 - Evidence snapshot` runs
  `snapshot_to_das.ps1` daily at **20:30 PT** â€” after the 13:00 close, before the
  AI runner's 22:00 window, and far outside the 06:00â€“14:00 band where
  `TradingBotV3 0700 Launch` fires every 15 minutes. `StartWhenAvailable`,
  `IgnoreNew` on overlap, 3 h limit. Next run 2026-08-23 20:30. The task XML is
  **exported to `scripts/ops/`** so it is versioned like the scripts, and a test
  parses its trigger and fails if the hour ever drifts into market hours.

Neighbouring schedule, for the record: snapshot 20:30 â†’ cold push 21:05 (hourly)
â†’ AI jobs 22:00 â†’ launch 06:00.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4172 passed / 19 subtests**, exit **0** (was 4168; +4) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |

**Owed:** confirm the first *scheduled* run (2026-08-23 20:30) writes to
`\\MINI-PC\Trading Bot Data\backups\2026-08-23\` with a byte count â€” today's
canary was hand-run.

---

## 2026-08-22 evening - Â§0 EVIDENCE FROZEN, Â§1 AUDIT AMENDED (S1/S2 now PROVEN)

**Branch `phase05-integration-blitz`.** Consolidated R10.A release, steps Â§0 and
Â§1. Docs plus one irreplaceable copy operation; no runtime code changed.

### Â§0 â€” the pre/post tracker pair is frozen, on both disks

`master_avwap_setup_tracker.json.bak` is rotated on **every** tracker save
(`os.replace`, `master_avwap_lib/legacy.py:4907`), so the 2026-08-22 canary
snapshot held the only pre/post pair that can ever prove S1/S2 â€” and Monday's
07:00 run would have destroyed it. Frozen into
`evidence_snapshots\evidence_frozen\tracker_2026-08-20_vs_08-21\`, which pruning
never touches, and copied to `\\MINI-PC\Trading Bot Data\backups\evidence_frozen\`.

| file | run state | stored | stored SHA-256 | decompressed | decompressed SHA-256 |
|---|---|---|---|---|---|
| `â€¦tracker.json.bak.gz` | 2026-08-20 | 133,162,736 B | `f6f9eef32faf5b32â€¦` | 938,541,721 B | **`7777fd68f58732f0â€¦`** |
| `â€¦tracker.json.gz` | 2026-08-21 | 135,738,242 B | `802cc9ed8f3a9f3dâ€¦` | 960,488,317 B | `29a534b058c1d39dâ€¦` |

The `.bak` decompressed hash **independently reproduces Fable's `7777fd68â€¦`**.
Both DAS copies were re-hashed after the copy and match their stored hashes.

**Robocopy leg run by hand:** `backups\2026-08-22\` now exists on the DAS â€”
667 files, 682 MB (650.6 MiB), rc=1 (0â€“7 is success), 132 s. The manifest reads
back on the DAS as 666 files / 0 skipped.

### Â§1 â€” six audit amendments; the verdict counts move to 14/4/2/2

Amendment 2 in the audit carries the detail. The three that matter:

**S1 and S2 leave UNKNOWN for PROVEN, and S1 is worse than "exits move".**
218 status transitions on 9,331 common setups (OPENâ†’CLOSED 168, CLOSEDâ†’OPEN 35,
OPENâ†’UNTRADEABLE 14, UNTRADEABLEâ†’OPEN 1 â€” the release note's breakdown exactly).
Among 6,736 setups CLOSED in both runs, **2,737 scenarios changed status or
reason**, 1,306 changed exit date, and **2,618 had their `events` dropped while
status and `total_r` stayed identical**. The worst shape is a same-date rewrite:
AMCR LONG on 2026-07-28 goes `TIME_STOP @ 46.69, R 0.577` â†’ `TARGET_HIT @ 45.55,
R 0.360`. A trade that timed out is now on record as having hit its target.
S2 reproduces to the unit: **2,739** setups carry an 08-21 `latest_snapshot` in a
payload whose `data_session` is 08-20, with 452 exit events on that forming bar;
PRE shows the same shape one day earlier (2,834), so it is systematic.

**A correction to the release note's own mark claim.** Of 1,309 same-dated
historical closes that differ between the runs, **only 5 differ materially** â€”
**1,304 are float32â†’float64 precision** (`31.350000381469727` â†’ `31.35`) from the
Yahooâ†’IB switch. S1 stands (it is measured on status, reason and R, not float
tails), but the mark-level evidence is far smaller than "7,674 mark-days differ",
and the precision half belongs with Â§3's bar-source problem.

**Two of my own verdicts were wrong.** D5b becomes **UNTESTED**: `orb_first_candle*`
has zero rows anywhere â€” the flow has never fired â€” and the 5,053 rows I called
"working as designed" belong to `orb_breakout`/`orb_breakdown`, a different
family. D1d/D2b were a **window mismatch, not a brief error**: on 2026-08-07â€¦08-21
the brief's 394/345 and 300 are exact. Every outcome-CSV figure now states its
window.

Also: `h1_bar_start_v1` keys on `^h1_` (9,623/9,623 whole file, 6,439/6,439 in
window) â€” and I had implied no non-H1 row lands on minute 30, when **291 of 6,054
(4.8%) do**; the rule is conjunctive so it holds, but its precision is not 100%.
Duplicate-pair gaps bottom out at **76 s** with a 76â€“78 s cluster, which is a
sweep cadence and reinforces that concurrency is not the duplicate mechanism.
pid 32620 recurs across 08-11 and 08-21 â€” pid joins must be session-qualified.

**Next:** Â§2 (schedule the snapshot task, add `source_sha256`), then Â§3 (the
daily-bar volume cliff, read-only, then STOP).

---

## 2026-08-22 - R10.A (first half) LANDED: the evidence now has a dated backup

**Branch `phase05-integration-blitz`.** Trader instruction the same afternoon:
*"Any and all very important files that we use occasionally should go to the
server with the massive HDD."*

**Measured gap.** The hourly cold push covers ~270 MB and excludes hot state by
design. What it excludes is the evidence itself: `data\runtime` **3.5 GB**
(960 MB tracker + 939 MB `.bak`, 203 MB outcome CSV, journal SQLite, every
cohort / Focus store), **36 home-root evidence files**, `_tools`, and the
diagnostics tree at **529 MB** - all on one disk.

Decision 0015 stands, so this is a dated **snapshot**, never a move.
`scripts/ops/evidence_snapshot.py` (19 tests) stages locally then
`snapshot_to_das.ps1` robocopies to `backups\<YYYY-MM-DD>\`; unreachable share
exits 0 leaving the staged copy. SQLite via the backup API; anything â‰¥256 MB
must hold size+mtime for 60 s or is **skipped with a reason and counted**;
anything â‰¥64 MB is gzipped; `manifest.json` carries size + SHA-256 per file.
Retention 7/4/12, `evidence_frozen/` permanent. `restore_from_das.ps1` restores
only into a scratch dir - `restore()` **refuses** the home folder and diagnostics
outright. System Health gains an `evidence_snapshot` tile (absence = `unknown`).

**Finding:** `push_cold_to_das.ps1` existed **only** in `_tools` - the script
protecting the evidence was itself unversioned. The repo copy is now the source
of truth and a test compares the two byte for byte. It also gained
`data\runtime\evidence_ledgers`, and both headers say **two jobs, two scopes**.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4168 passed / 19 subtests**, exit **0** (was 4148; +20) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source `--selftest` | **56/56**, exit 0 |

### Mechanics canary â€” run for real, and the restore drill is done

First snapshot, 2026-08-22: **666 files, 4,033 MB source â†’ 682 MB stored (83%
compression), 0 skipped**, `--verify` re-hashed all 666 with 0 missing and 0
mismatched. Largest: the 960 MB tracker â†’ 136 MB, its 939 MB `.bak` â†’ 133 MB,
`master_avwap_setup_attributes.csv` 484 MB â†’ 44 MB, `technical_integrity_events.jsonl`
466 MB â†’ 25 MB, the 203 MB outcome CSV â†’ 20 MB. One SQLite through the backup API.

**Restore drill passed:** 666 files restored into a scratch folder and byte-compared
against the live originals â€” `intraday_bounce_outcomes.csv` (203 MB), the
`master_avwap_setup_tracker.json` (960 MB) and `trader_annotations.jsonl` all
**SHA-256 match**. Restoring into `C:\TradingBotData` was attempted deliberately
and **refused**, as designed. The drill is recorded, so the tile now reads
*healthy: last snapshot 2026-08-22 (0d ago), 666 files, 682 MB stored, DAS
reachable, last restore test 2026-08-22*. Scratch copy deleted afterwards.

**Frozen-exe trigger FIRED** (new top-level package `ops`, new non-`.py` runtime
assets). The automated half is green - the spec-drift guard caught both and
`ops` is now in `FIRST_PARTY_PACKAGES` because `operations_audit` imports it
lazily and renders System Health in the frozen build. **The frozen selftest was
not run: Smart App Control refuses the unsigned exe on this desk, so
`dist\TradingBotV3\TradingBotV3.exe --selftest` cannot execute.** Source launch
remains production.

**Owed for the snapshot half:** the only thing left is **scheduling** â€”
`snapshot_to_das.ps1` nightly after the AI runner, and confirming the first
scheduled run copies to the DAS (today's canary was run by hand and stopped at
staging, since the robocopy leg lives in the `.ps1`). The trader also has an
open decision on the 939 MB `master_avwap_setup_tracker.json.bak`: it is being
snapshotted every night at 133 MB compressed, and audit Â§8 Q2 asks whether it is
a deliberate rollback point or an accident. If it is an accident, excluding it
halves the nightly cost.

**R10.A's ledger half is NOT started** - the outcome ledger, one-owner
transaction, dual-write canary, no-fabrication finalization, registration
context and `evidence_rules.py` all remain.

---

## 2026-08-22 - R10.0 COMPLETE: the evidence audit, and the program stops here

**Branch `phase05-integration-blitz`.** Read-only sweep plus one authorized
observability fix. Deliverable:
[`docs/analysis/EVIDENCE_AUDIT_2026-08-22.md`](docs/analysis/EVIDENCE_AUDIT_2026-08-22.md),
classified in `docs/README.md`.

**R10.A does not start until the trader accepts the register.**

### Verdicts: 12 PROVEN, 6 PROVEN\* (number differs), 3 REFUTED, 4 UNKNOWN

Three findings change R10.A's design:

1. **Concurrency is PROVEN, so Â§2.5's guard is authorized â€” but it is not the
   duplicate fix.** On 2026-08-20 pid 31848 lived 07:46:01â†’12:45:09 PT and
   overlapped three other pids, the worst for **3.8 hours**. Every other
   in-window session is sequential restarts. A guard already exists but only in
   `launch_gui_auto.ps1` (the scheduled-task path); `launch_gui.py` and
   `trading_desk.cmd` have none, so any other launch route bypasses it.
   **However**, the concurrent session supplies only 184 of 742 duplicate
   `registered` rows (25%), and **0 of 609 duplicated ids were written within
   5 s of each other** (median gap 1,581 s). The guard is warranted; what
   removes duplicates is the ledger's keyed idempotent write.
2. **D4 is a finalization gap, not an outage.** 2026-08-21 produced 409
   registrations and **394 `12_bar` milestones** and **zero** finals. Tracking
   ran all day; only EOD finalization did not. Same mechanism as D3's 576-event
   backlog, so R10.A's idempotent finalization fixes both and neither needs an
   IB-outage story.
3. **Tier cannot be conditioned on.** `tier` is absent from the outcome store's
   `context_json` on **0 of 7,863** rows, so D8c's tierÃ—outcome inversion is not
   reproducible and must not be quoted until R10.A puts a tier in the ledger.

### Refuted as described

- **D5b ORB**: the ORB re-break **does** register outcomes (28/25 breakout,
  20/18 breakdown). The code already separates candidate â†’ re-break â†’
  recross and only the break claims an entry. D5 is **LRSI-only** (0 rows).
- **D6c** the "median 90 min" lag: `logged_at` is the write time, so that
  statistic measures the wrong thing (H1 502 min, non-H1 425 min). The
  bar-start defect is proven a better way: **6,439 of 6,439** H1 rows have
  `entry_time` minute == 30, and no non-H1 population does. That became
  evidence rule `h1_bar_start_v1`.
- **D1b** concurrency as the duplicate cause, above.

### Worse than stated

- **S3**: horizon 5 has a **median 65 business-day** span; horizon 10, **73**.
  SPY-relative columns are **0.0% non-null on all 9,967 rows**.
- **F5**: not four names â€” **244 of 499 (symbol,side) pairs (49%)** appear on the
  true M5 list across â‰¥2 sessions. *Caveat that travels with it:* the picks store
  is a snapshot and cannot distinguish "survived the roll" from "re-added", which
  is exactly why R10.E needs membership episodes.
- **F3 is PROVEN by code and invisible in data by construction**:
  `_pick_key` (`human_focus_tracking.py:171`) has no category and lines 290/468
  build dict comprehensions, so the collision is destroyed before the CSV is
  written. Its absence in the output is the signature, not a refutation.

### A trap that bit this audit

`human_focus_daily_picks.csv` `source` conflates **list and origin**:
`focus_swing_m5` is a **swing** row. A substring match on `m5` pulls 649 swing
rows into an M5 count and inflated F2/F5 until caught. The list must come from
the `focus_(swing|m5|pick)` **prefix**. R10.E should store the two separately.

### The one authorized code change

`ai_jobs/runner.py` recorded failed jobs with a blank `reason` because the
non-exception path passed only `reason=` while `run_nightly_journal_import`
returns its explanation in `messages`. `_failure_reason()` prefers `reason`,
falls back to `messages`, and when a job fails with nothing to say records that
fact naming the job. Successful rows untouched. **Two tests proved red first**
(`test_a_failing_job_records_the_messages_it_returned`,
`test_a_failing_job_with_nothing_to_say_still_says_so`); a third was green
throughout and guards against over-reach.

**AMENDED same day:** the cause I named was wrong.
`existing_journal_requires_migration()` is **False** on this desk and the Journal
page shows no preparation banner, so that refusal branch never runs. I attributed
the failure to the most legible string near it without checking whether the
branch was live â€” the audit's own rule, applied everywhere else and not here.

What is actually happening: the job **imports and is then marked failed**. The
2026-08-21 23:30 run imported **21 Questrade executions** and still returned
FAILED, because three `had_errors` paths fire nightly â€” `IBKR_FLEX` FAILED
(transient: connection refused 08-21, "statement could not be generated" 08-22),
`QUESTRADE_BACKFILL` PARTIAL (400 on `/v1/accounts/<id>/activities`, **both**
accounts, executions still imported), and `RECONCILE` MISMATCH (19 mismatches).
Because the status is FAILED the runner retries 3Ã— per session, re-requesting a
Flex statement each time. Audit Â§6a carries it; Â§8 Q1 is struck.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4148 passed / 19 subtests**, exit **0** (was 4145; +3) |

### Open questions blocking R10.A (audit Â§8)

1. ~~Run the journal migration~~ **STRUCK** â€” none is pending. Its replacement is
   *recommended, not authorized* and belongs to R7: reclassify a transient Flex
   failure and a cross-check-unavailable PARTIAL as `degraded` rather than
   `failed`; diagnose the Questrade `/activities` 400; surface the 19 reconcile
   mismatches with trade ids in Journal â–¸ Health.
2. Is `master_avwap_setup_tracker.json.bak` (939 MB) a deliberate rollback point
   or an accident?
3. Confirm the existing duplicate rows stay as history with a reader-side rule
   rather than a deduplicated copy.
4. Confirm the `launch_gui.py` guard, noting it will not remove duplicates.
5. Retention: 13 months hot then cold-push â€” longer?

---

## 2026-08-22 - R10 REGISTERED: Phase 0.7, the Evidence Plane program

**Branch `phase05-integration-blitz`.** Docs only. The trader authorized a
packetized evidence-quality program (Fable synthesis v2 after Sol's review); it
is now `plan.md` Â§12 **Phase 0.7 â€” R10**, ten packets R10.0â€¦R10.I with their
ground rules, trader decisions and gates.

**Correction to the brief's stated state.** It was written against HEAD
`22154dd` and says R9.4 and R9.5 "remain queued and authorized". Both had
already landed â€” `36abb14` (R9.4) and `ba931a5` (R9.5), both pushed. The
consequence for R10's ordering is on record in the phase text: the brief
sequences R9.5 *after* R10.A "using the evidence-plane conventions R10.A sets",
and R9.5 in fact shipped before this program was registered. Its store
(`diagnostics/shadow_evidence/sector_cohort/sector_cohort_shadow.jsonl`,
`sector_cohort_shadow_v1`) is append-only with a schema name, a `config_hash`
and a per-run coverage row â€” consistent with the program's rules but not derived
from them. R10.0 inventories it with the other stores and names any
reconciliation R10.A should make.

**Immediate next action:** R10.0, the read-only evidence audit. It is the one
packet that changes no behavior (bar the single authorized `journal_import`
observability fix), and the program **stops after it** for the trader to accept
its decision register before R10.A begins.

---

## 2026-08-22 - R9.5 LANDED: the R9 packet is code-complete

**Branch `phase05-integration-blitz`.** Fifth and last item. `sector_cohort_divergence`
is at **SHADOW and stops there**.

Golden fixture frozen **first** (`tests/fixtures/sector_cohort_v1.json`, five
cases isolating one rule each, satisfying the repo-wide Milestone 3 contract).
It caught a defect in itself before the detector existed: `path_pct[0]` was
allowed to be non-zero while claiming to be a move from the session open, which
re-based every series and inverted the gap case. Gate 1 `config_hash` (excludes
`enabled`), gate 3 coverage on every run including quiet ones, gate 7 shipped
**off**. Batched yfinance, single-flight, **zero IB traffic**.

**First real shadow day written:** 2026-08-21 session â€” 20 ETFs, 78 benchmark
bars, 1,560 consumed, **11 cohort observations, XLU short from 10:35 ET**.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4145 passed / 19 subtests**, exit **0** (was 4118; +27) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| `launch_gui.py --selftest` | **56/56**, exit 0 |
| frozen exe | not rebuilt; Smart App Control refuses it and the desk runs from source |

---

## R9 exit gate â€” deterministic half MET, four live proofs owed

All five items are BUILT and GREEN. R9.3's report is filed with its declared
window; R9.5's fixture is frozen and its first JSONL day is written. **What is
owed is the "on the desk" half, one proof per item:**

1. **R9.1** â€” a real rebuild writing a `universe_rebuild` row with
   `refused: false` and a plausible before/after, plus the snapshot directory
   appearing on the live machine.
2. **R9.2** â€” a session where a LIKE is filed and the symbol is still seen to
   alert afterwards; on an AWAY day, still reaching the hourly D1 phone push.
3. **R9.4** â€” a Master AVWAP scan where DRAM reaches the theta report labelled
   `via thetalongs.txt`, or is honestly absent for a stated rule reason
   (earnings buffer, no weekly chain, support stack).
4. **R9.5** â€” the shadow log growing over real sessions toward its declared 40,
   spanning bullish, bearish and chop. Nothing may move it off SHADOW before that.

**Also still owed, unchanged: the 2026-08-21 fluidity live gates** below â€” a
full session measured against `ui_stalls_prefluidity_2026-08-21.jsonl` via
`docs/GUI_FLUIDITY_MEASUREMENT_RUNBOOK.md`, and the `focus_picks_panel.py:419`
`live_state_for` hot spot that the first post-fix run named.

**Recommended, NOT authorized, not in R9:** fix the `eod_close` default in the
intraday outcome writer (R9.3 found it defaults to `entry_price`, biasing every
R in that store upward on 16.9% of finals).

---

## 2026-08-22 - R9.4 LANDED: `thetalongs.txt`

**Branch `phase05-integration-blitz`.** Fourth item of the R9 packet, tests
first (10 red, then the code, then 4 more on the report label).

`evaluate_theta_put_candidate` is LONG-only and `side` was list membership, so a
wheeled name on neither trend list was never evaluated â€” the window's whole
positive P&L (four DRAM short puts) was invisible to the engine built for it.
`resolve_scan_sides()` is the seam: `side` unchanged for every detector,
`theta_side` LONG for anything on the list regardless of membership, only the
two premium calls take it. Theta-only names join `symbols` but **not** `longs`,
and resolve LONG rather than a phantom SHORT. Rows carry `theta_list_source`;
the report prints `| via thetalongs.txt`.

`C:\TradingBotData\thetalongs.txt` **created, containing DRAM**. Absent or
unreadable both return `[]` â€” the list can cost its own names, never the run.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4118 passed / 19 subtests**, exit **0** (was 4104; +14) |

**Owed for R9.4:** one Master AVWAP scan on the desk where DRAM reaches the
theta report labelled `via thetalongs.txt`, or is honestly absent for a stated
rule reason (earnings buffer, no weekly chain, support stack). **Immediate next
action:** R9.5 (`sector_cohort_divergence` to SHADOW, golden fixture frozen
first).

---

## 2026-08-22 - R9.3 LANDED: the scoreboard, rebuilt from the right stores

**Branch `phase05-integration-blitz`.** Third item of the R9 packet. Read-only
analysis: `scripts/setup_scoreboard.py` +
`docs/analysis/SETUP_SCOREBOARD_2026-08-21.md`, classified in `docs/README.md`.

**The headline is that the review was wrong about its own starvation.** The
regime, RVOL and sector axes it reported at n=130 were never starved - they are
in `intraday_bounce_outcomes.csv`, on 100% of its in-window rows, alongside a
real stop and therefore a real R. Rebuilt: 239,422 rows scanned â†’ 14,452 finals
â†’ 6,907 in window over **20** sessions â†’ **5,608 usable** after the two
exclusions, split across 5 market environments at n=658-1,643 each.

**The 16.9% zero mass is a defect, and the report says so before it ranks
anything.** All 1,164 in-window finals with `close_r == 0` have `eod_close`
exactly equal to `entry_price`; none of the 5,743 settled finals does. 251 never
advanced a bar; 563 are stopped-out trades scoring 0 instead of â‰ˆ âˆ’1R, which
biases every mean upward. Excluded and counted, never averaged. **This is a
writer defect worth fixing and it is not in the R9 packet** - see below.

Trimmed mean (10%) + median + stop-out rate beside every R; 0.1%-of-entry risk
floor (212 rows); cells ranked only at n â‰¥ 30; swing block measured against
`baseline_every5` with an explicit guard that a positive lift means *lost less
than the control*. **Â§5 declares the frozen forward window** - 40 sessions
spanning bullish/bearish/chop, exclusions fixed in advance - which is the only
route to Â§7 gate-2-eligible evidence. It promotes and demotes nothing.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4104 passed / 19 subtests**, exit **0** (was 4085; +19) |

**Recommended, NOT authorized, not in R9:** fix the `eod_close` default in the
intraday outcome writer so it records "no close obtained" instead of the entry
price. Every R in that store is biased upward until it is fixed, and the
scoreboard can only work around it.

**Immediate next action:** R9.4 (`thetalongs.txt`).

---

## 2026-08-22 - R9.2 LANDED: the LIKE asks why, and stops parking the symbol

**Branch `phase05-integration-blitz`.** Second item of the R9 packet, tests
first (13 written, 11 red, then the code).

The why is required â€” the claim key and double-click select and move focus to
the why field, Enter commits, an empty why does not and the chart stays. And a
LIKE now takes an advance-only route (`likeAdvanceRequested` â†’
`_advance_after_like`, recording `like_advance`) instead of the "Not today"
verb, so it never reaches `_ignored_symbols`, never sweeps the symbol's other
queued alerts, and never drops an auto-adopted Focus pick. The veto's
retire-and-park path is unchanged. `review_learning.TAKE_ACTIONS` gained
`like_advance`, which stops 40 of the window's 52 likes being scored as
dismissals.

**Four existing tests pinned the rule the trader reversed** and were rewritten
against the new one; `test_a_like_also_retires_the_chart` was deleted rather
than adjusted, because its subject is the behavior that changed.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4085 passed / 19 subtests**, exit **0** (was 4073; +12 net) |

**Owed for R9.2:** one desk session in which a LIKE is filed and the symbol is
still seen to alert afterwards â€” and, on an AWAY day, still reaches the hourly
D1 phone push. **Immediate next action:** R9.3 (rebuild the setup scoreboard
from the outcome stores, read-only).

---

## 2026-08-22 - R9.1 LANDED: the universe can no longer collapse silently

**Branch `phase05-integration-blitz`.** First item of the R9 packet, built with
its tests first (12 written, all 12 red, then the code).

`scripts/universe_builder.py::build_universe` gained a write floor of
`max(500, 50% of the prior universe_all.txt count)`; a missing, empty or
**unreadable** prior fails OPEN; `force=True` carves out the floor and never the
zero-symbol refusal; every write attempt appends a keyless `universe_rebuild`
row to `job_ledger.jsonl` with per-list before/after counts, the floor, and
`refused`/`forced`; the outgoing lists are snapshotted (last 10 runs kept).

**Trader decision this pass:** wire the carve-out at both manual entry points.
`scripts/ui/panels/universe_panel.py` (Build button) always forces, and
`scripts/autopilot_core.py::rebuild_universe_if_stale` forwards its existing
`force` - so "Rebuild universe now" overrides a floor refusal and the scheduled
stale tick does not. Both are outside R9.1's named files and were asked about
before editing.

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4073 passed / 19 subtests**, exit **0** (was 4059; +14) |

`tests/test_universe_builder.py` carries all 14: 12 floor/ledger cases plus
2 that pin the manual carve-out wiring at both entry points.

**Owed for R9.1:** one real rebuild on the desk writing a `universe_rebuild` row
with `refused: false`, confirming the row and the snapshot directory appear on
the live machine. **Immediate next action:** R9.2 (the LIKE: always ask why,
stop parking the symbol). The 2026-08-21 fluidity live gates below remain owed
and unchanged.

---

## 2026-08-22 - R9 QUEUED: the trade review's nine questions, answered and authorized

**Branch `phase05-integration-blitz`, docs only - no runtime file changed.** The
2026-07-24..08-21 trade review (`docs/analysis/TRADE_REVIEW_2026-08-21.md`) left
nine questions. Two independent passes answered them on 2026-08-22 (Opus, then
Fable verifying Opus against the raw stores); the trader answered the three that
needed him and **authorized the response packet in writing**. It is now
**Phase 0.6 / R9 in `plan.md` Â§12**, five items in build order:
R9.1 universe write floor + ledger event (P0) â†’ R9.2 LIKE always asks why and
stops parking the symbol â†’ R9.3 scoreboard rebuilt from the outcome stores
(read-only) â†’ R9.4 `thetalongs.txt` â†’ R9.5 `sector_cohort_divergence` to SHADOW.

Findings that changed a number or a claim, all with raw rows in the scratchpad
`NINE_QUESTIONS.md` / `FABLE_ASSESSMENT.md`: every capture-rail LIKE is currently
counted as a **rejection** by `review_learning.REJECT_ACTIONS` because it routes
through `remove_today`; the review's "three SMA vetoes filed after v3 shipped" is
false - v3 was live on the desk only from **2026-08-21 12:19:42 PT** (one veto,
ever); `intraday_bounce_outcomes.csv` `close_r` carries penny-stop artifacts
(Â±655R) outside the window and a 16.9% exact-zero mass inside it; `bouncers.txt`
is never actually rotated (57 accumulated launch-blocks) and is not a subset of
the outcome store; `avwape_to_1stdev` exists only in the tracker namespace.

**Immediate next action:** Opus implements R9.1 first, on this branch, fixture
and tests first where the item says so. The 2026-08-21 fluidity live gates below
remain owed and unchanged. Working tree at this stamp: `plan.md`,
`CURRENT_CHECKPOINT.md` (this), plus the review session's `docs/README.md`
classification and `docs/analysis/` - all docs.

---

## 2026-08-21, fourteenth pass - THE HITCHING, MEASURED AND CUT

**Branch `phase05-integration-blitz`.** Trader: "do an assessment of what's
slowing things down... is it having to pull files from the server PC? I want
this program to be very fluid to use."

### The server was the first thing tested, and it is not the cause

| check | result |
|---|---|
| Where every hot path resolves | `C:\TradingBotData` and `%LOCALAPPDATA%` |
| GUI references to the research store | none outside two worker-thread warehouse tiles |
| `\\MINI-PC\Trading Bot Data` at the time | **momentarily unreachable** (WinError 3); it resolved again by 12:34 the same day, so the share drops and re-establishes rather than being unmounted |
| Cost of a miss on it | **0.0 ms** |
| Local `listdir` + 20 x `stat` | 0.1 ms + 0.2 ms |

### What it actually was

07:52-11:11 on the live desk: **1843 stalls over 50 ms, median 238 ms, p90
1.16 s, 1008 s blocked** (~8% of the session), plus the 298 s and 200 s GC
freezes already fixed in `ab219b5` - about a third of the session with the main
thread stuck.

**56% of stalls had no Python frame below `app.exec()`** - Qt's own C++. The
dominant Qt C++ work here turned out to be **stylesheet parsing**: `setStyleSheet`
appears 49 times across `scripts/ui/`, per widget, inside rebuild loops.
`FocusSideEditor.refresh` destroyed and rebuilt every chip (105 on D1 Focus),
each constructor parsing CSS; `AlertFeedItem` did it seven times per row with up
to 250 rows. That also explains the freezes: 105 widget teardowns per refresh is
the cyclic garbage the starved collector had to walk.

### Fixed, in order of measured cost

| change | measured |
|---|---|
| `AlertFeedItem`: 7 stylesheets -> 0, variants as `theme.qss` rules on `alertKind`/`focusOn` | 250 rows **282 ms -> 167 ms** |
| `FocusSideEditor.refresh` diffs; `FocusStatusChip.update_state`; re-style only when the accent moves | no rebuild on an unchanged board |
| `ChartDataService.cached_bar_dicts` memoizes `as_bar_dicts` per series (LRU 160) | ~490 dicts/symbol/poll -> once per series |
| `_load_local_settings` mtime-cached | 100 reads **9.6 ms -> 0.7 ms** |
| `load_review_events` stamp-cached | 5.8 MB, 8809 rows, **80.8 ms -> 7.7 ms** |
| `setup_delegate._resized` scales in the font's own unit | the `QFont` flood, and a 1-point star |
| `install_qt_message_rate_limit` | a storm costs one line + a tally |
| stall watchdog samples throughout, records modal frame + `culprit_samples` | the 56% names itself next session |

### The font bug, stated exactly

The theme sizes fonts in **px** on purpose (`theme.py` emits `"{size}px"`, for
device independence across the desk and the MacBook). `QFont.pointSizeF()`
returns **-1** for such a font, so `setup_delegate`'s three call sites computed:

| line | expression | result |
|---|---|---|
| `_favorite_star` | `-1 + 2.0` | a **1-point star** |
| `_dislike_mark` | `-1 + 1.0` | `setPointSizeF(0.0)` - **the console flood**, call ignored |
| `_chip` | `max(7.5, -1 - 1.0)` | pinned to 7.5, never relative |

Inside `QStyledItemDelegate.paint()`, once per visible row per repaint.

**Honest limit on the verification:** Qt warnings do not reach a piped stderr on
Windows - a canary `qWarning` in the test harness printed nothing - so "no
warnings in 600 paints" proves nothing and is not claimed. The fix rests on the
arithmetic, which is unit-pinned in `tests/test_ui_fluidity.py`, and the
trader's next session is the proof.

### Tests

Three existing tests asserted on inline `styleSheet()` strings - they pinned the
implementation, not the behaviour. They now assert the object name / property
that carries the styling, which is the real contract between a widget and
`theme.qss`. One asserted on label text where the chip now hides rather than
omits a label; it reads visibility instead.

`tests/test_ui_fluidity.py` and `tests/test_qt_widget_reuse.py` are new: caches
parse once and never go stale, cached values cannot be mutated by a caller,
chips are reused across a refresh, and an alert row owns no stylesheet.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4059 passed / 19 subtests**, exit **0** (was 4032; +27) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| frozen exe | not rebuilt; Smart App Control refuses the build and the desk runs from source |

### Owed

A full session measured the same way. **The method is now a runbook** -
`docs/GUI_FLUIDITY_MEASUREMENT_RUNBOOK.md` - so the next session does not have
to re-derive it: one command
(`python scripts/ui/stall_watchdog.py --compare <baseline>`), the baseline to
beat, the targets, and how to read the result. The desk was restarted onto the
fixed build at 12:12 and the pre-fix log archived to
`ui_stalls_prefluidity_2026-08-21.jsonl`.

**The first post-fix run already named the next target**, which is the histogram
earning its keep on day one: a 11,970 ms stall with **210 of 361 samples inside
`focus_picks_panel.py:419`** - that is `live_state_for`, not the widget work the
pass just fixed. The stack names the driver: `record_bounce_alert` â†’
`_refresh_all` â†’ `refresh`, so **every bounce alert resolves `mover_state` per
symbol across all four editors**. Fix candidates and why neither touches a
detector are written up in the runbook's section 4.

---

## 2026-08-21, thirteenth pass - THE PRIOR-DAY BREAK AND SESSION VWAP

**Branch `phase05-integration-blitz`.** Trader: "i think we should also demand
these picks have broken the previous HOD that day for longs, and previous LOD
for shorts. also they should be above vwap for longs and below vwap for
shorts."

### That rule already exists, so it is called rather than rewritten

"Beyond yesterday's extreme AND the right side of session VWAP" is the **M5
Focus adoption gate**, trader rule 2026-08-14, living in
`scripts/focus_adoption_gate.py`. The sweep now calls
`passes_focus_adoption_gate` with numbers read off its own cached M5 series by
`regime_pause_hold.session_levels`:

- **price** and the **prior session's high/low** come from the series itself.
  It is RTH, so those are the previous REGULAR session's extremes.
- **session VWAP** comes from `chart_snapshot.session_vwap_series` and nowhere
  else. `calculate_dynamic_vwap` / `calculate_eod_vwap` blend prior sessions
  and answer a different question.

UNKNOWN fails here exactly as it does on the Focus path: a cache holding only
today has no previous extreme to break, and "cannot measure" is not "passed".

### Fixtures first, again

`regime_pause_sweep_v1` grew to **six cases per side**, each isolating ONE
reason, and each is now **two sessions** because the new gate cannot be
measured from one:

| case | defiance | near extreme | levels gate | fate |
|---|---|---|---|---|
| HOLDS_AT_HIGH | pass | 0.33 ATR | open | kept |
| HOLDS_JUST_UNDER | pass | 0.73 ATR | open | kept |
| FELL_LESS_THAN_SPY | pass | 7.15 ATR | - | dropped, ATR |
| BOUNCING_OFF_LOW/_HIGH | pass | 6.25 ATR | - | dropped, ATR |
| INSIDE_PREV_RANGE | pass | 0.04 ATR | closed | dropped, prior extreme |
| BELOW_VWAP / ABOVE_VWAP | pass | 0.97 ATR | closed | dropped, VWAP |

Frozen against the unchanged sweep (four flagged per side), changed, re-frozen
(two per side). A test now asserts, per case, WHICH gate rejected it - and a
separate one asserts every case still clears the defiance test it was built
for, so no case can quietly stop being evidence while the fixture stays green.

The two VWAP cases needed volume weighting to exist at all: for a long inside
one ATR of its high to sit BELOW session VWAP, the day has to be thin on the
way up and heavy near the top. That is a real shape, and it is the only shape
where this half of the gate binds.

### Three champion tests needed better fixtures, not a change

They went red again, and both reasons were the fixtures being unlike
production:

1. **no prior session** - a single 12-bar day cannot answer "did it break
   yesterday's extreme", while `get_cached_5m_bars` asks for `5 D`;
2. **no volume at all** - `IbBar.volume` defaults to `0.0` and the helpers
   never set it, so session VWAP was unmeasurable and the gate correctly
   refused every name.

Both fixed in the fixtures: a quiet prior session, and a realistic default
volume on the bar constructors. Nothing in the gate was loosened to make them
pass.

### What it does to the real batch

Replayed against the 67 names actually flagged that morning, each at its own
flag time:

| | flagged | survive BOTH gates | dropped by ATR | dropped by prior-day/VWAP |
|---|---|---|---|---|
| longs | 38 | **18 (47%)** | 13 | 7 |
| shorts | 29 | **18 (62%)** | 8 | 3 |

Every levels-gate drop that day was "not above yesterday's high" / "not below
yesterday's low". **The VWAP half bound on nothing** - expected, because a name
within one ATR of its high is nearly always above its session VWAP. It costs
nothing and covers the wide-range case the fixture pins.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4032 passed / 19 subtests**, exit **0** |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| golden fixture | `regime_pause_sweep_v1` re-frozen; per-case rejection reason asserted |
| frozen exe | not rebuilt; Smart App Control refuses the build and the desk runs from source |

### Owed

The live gates in `plan.md` item 11, and one of them now matters more: the
detector passes **fewer than half** the longs it used to, so a normal session
has to confirm the list is still usable rather than a handful. The running desk
still predates every commit made today.

---

## 2026-08-21, twelfth pass - THE REGIME-PAUSE GATE, FIXTURES FIRST

**Branch `phase05-integration-blitz`.** Trader: "yes do it properly." So the
order was the one plan.md sec 5 asks for, and it earned its keep twice.

### 1. Freeze the baseline

`tests/fixtures/regime_pause_sweep_v1.json`, captured from the UNCHANGED
detector: four long cases and four short ones, each reaching the sweep through
a **different branch** of `still_trending or made_new_extreme or window_excess`.
Two per side are genuinely at their extreme; two are not - one drifting flat
while SPY falls, one bouncing off the day's low. The baseline flagged all four
per side and captioned every one of them "holding highs".

The harness drives the real `_sweep_regime_pause_bangers` on a real
`BounceBot`; only the bookkeeping (observations, tracker row, candidate event)
is stubbed. The method gained an injectable `now` so the replay does not depend
on the day it runs.

### 2. Add the condition

```python
hold = regime_pause_hold.hold_state(bars, side, now=moment)
if not hold.holding:
    continue
```

**Added, never substituted** - the defiance test still has to pass first, so
the flagged set can only shrink. It is handed the FULL cached series rather
than `sym_today`: an ATR(14) needs fifteen bars and this sweep fires when there
are nine. `hold_state` takes its ATR from everything supplied and its extreme
from the last completed bar's session.

### 3. The fixture failed, which is the point

Four flagged per side became two. `test_the_gate_dropped_exactly_the_documented_rows`
now names the survivors and the departures, and a companion test asserts the
dropped pair **still satisfies the old predicate** - so if either case ever
stops being evidence about this gate, the fixture cannot stay quietly green
while proving nothing.

### 4. Three champion tests caught a real defect in the first version

`test_regime_pause_flags_nonparticipating_weak_name`,
`test_regime_pause_inverts_for_bullish_tape` and
`test_spy_champion_scenario_is_a_real_exercise` all went red. Their sessions
are **12 bars**, so there is no ATR(14) - and the first version treated
unmeasurable as not-holding and dropped names that were making new lows on the
pause candle. That would have switched the detector off for most of the first
hour of every session.

**Being AT the extreme needs no ATR.** A name whose extreme was set on the last
completed bar is holding, full stop; only the DISTANCE needs a tolerance, and
inventing one is the thing not to do. Off the extreme with no ATR stays
UNMEASURABLE and does not qualify.

### 5. The caption stopped being a batch label

The feed line carries each symbol's own measure -
`HTFL (new HOD), MRK (0.7 ATR)` - and `ui/models/bounce.py` expands it per row
(`M5 regime-pause watch - 0.7 ATR off HOD`). A bare symbol still reads as the
old phrase, so lines written before today, and any symbol whose hold could not
be measured, keep their wording rather than acquiring a claim the parser
invented.

### What it does to the real batch

Replayed against the 67 names actually flagged that morning, each at its own
flag time:

| | flagged | still fires | dropped |
|---|---|---|---|
| longs | 38 | 25 (66%) | **13 (34%)** |
| shorts | 29 | 21 (72%) | **8 (28%)** |

Worst drops: TGB 4.8 ATR off a 155-minute-old high; AMBP 4.2 ATR; ECHO 2.7 ATR
off its low. **MRK: dropped, 1.8 ATR off a 70-minute-old high. GFS: dropped,
1.3 ATR off.** Both of the trader's screenshots are refused at fire time now.

### Also changed

`completed_bars` reads attribute-shaped bars as well as dicts. BounceBot's
cached series is `IbBar` objects, and a shared rule that only understood
`bar.get` silently excluded every detector-side caller - the bar's shape is a
producer detail, not a different rule.

### Named, and deliberately NOT acted on

`REGIME_BANGER_DAY_EXCESS_PCT` (0.75) and `REGIME_BANGER_WINDOW_EXCESS_PCT`
(0.20) are still percentages, and the trader's own argument applies to them:
0.75% was about nine ATR for the slowest name in that batch and two thirds of
one for the fastest, so the day gate is biased toward fast movers. Changing it
would move the flagged set in BOTH directions - a different decision, needing
its own fixture and its own trader call.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4031 passed / 19 subtests**, exit **0** (was 4020) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| golden fixture | `regime_pause_sweep_v1` re-frozen, diff named in the test |
| frozen exe | not rebuilt; Smart App Control refuses the build and the desk runs from source |

### Owed

The live gates in `plan.md` item 11, now including one that matters more after
a tightening: **confirm a normal day still produces a usable number of names**
rather than a handful. And the running desk still predates every commit made
today.

---

## 2026-08-21, eleventh pass - "HOLDING HIGHS" WAS NEVER MEASURED

**Branch `phase05-integration-blitz`.** Trader: "this stock is being recommended
because its M5 regime pause watch holding highs but from the bottom M5 chart you
can see... its not... at all."

### Three defects, not one

1. **The caption is a batch label.** `_emit_regime_pause_summary` writes one
   feed line per sweep and `ui/models/bounce.py` explodes it into one row per
   symbol, each captioned `M5 regime-pause watch - holding highs`. The
   per-symbol numbers exist in the `hit` dict and never leave it.
2. **The predicate does not mean "holding highs".** `legacy.py:5469` is
   `still_trending or made_new_extreme or window_excess >= 0.20`. Only the
   middle branch is about the extreme; the third admits a name that is falling,
   just less than SPY.
3. **Nothing re-measures.** One alert per symbol per day. MRK was flagged at
   08:30 and read at 09:40 with the original claim intact.

### Measured before proposing

All 67 names flagged that day, each at its own flag time (yfinance 5m, +/- one
bar against the detector's IB bars):

| | longs (38) | shorts (29) |
|---|---|---|
| extreme older than 30 min when flagged | **10 (26%)** | **13 (45%)** |
| more than 1% off the extreme | 8 (21%) | 5 (17%) |
| genuinely made a new extreme in-window | 20 (53%) | 12 (41%) |
| stalest | WT, HOD **170 min** old | ECHO, LOD 70 min old, 1.8% off |

**MRK: flagged 08:30 with a 75-minute-old high, 1.6 ATR off it. By 09:40, 4.8
ATR off with a 140-minute-old high.** GFS, the short in the earlier screenshot,
was 1.1 ATR off its low at fire. Both fail the new rule at fire time; both are
deleted by it later.

### Why ATR and not percent - the trader was right, with a number

M5 ATR across that single batch ran from **0.084% of price (HMC) to 1.160%
(CIFR)** - a **14x spread**. A 1% threshold is 12 ATR for HMC and 0.9 ATR for
CIFR, so no fixed percentage can serve both. Trader: "a stock like MRK moves
slower than say MU, we can't use the 1% rule."

### Built

- `scripts/indicators/atr.py` - the shared Wilder ATR. The repo already had the
  rule twice and the copies disagreed: `legacy._wilder_atr_last` is Wilder,
  `market_state._m5_atr` is a plain mean under the same name, and neither is
  importable as a shared rule. Unmeasurable is `None`, never 0 - a zero ATR
  turns "I don't know how fast this moves" into "it doesn't move".
- `scripts/regime_pause_hold.py` - distance from the session extreme in ATR on
  completed bars, plus the queue verdict. **1.0 ATR** and **15 minutes** from
  the later of the alert and the last new extreme (trader's numbers). The ATR
  may use earlier sessions for warm-up while the extreme is taken from the last
  completed bar's session, so both are right when two sessions are handed in.
  A level merely EQUALLED does not refresh the clock.
- `AlertCenterPanel._expire_stale_hold_alerts`, on the existing 30s chart tick
  (connected after the bar refetch, before the re-render). Rows that survive
  are **re-captioned with what is true now**; the feed row keeps the words it
  was born with.

**Deletion is from the queue only** - the trader's explicit call. The alert
list, `alert_review_events.jsonl` (a new `hold_expired` action) and the
tracker's outcome rows keep the row, so the rule itself stays gradeable.
**Uncertainty never deletes:** no bot, no bars, no ATR, no readable stamp all
mean KEEP.

### NOT built, deliberately

The detector-side gate. Making near-extreme-in-ATR a REQUIRED condition in
`check_regime_pause_setups` changes what a champion emits - the alert set, the
candidate events, the outcome records - and plan.md sec 5 requires golden
fixtures first. `bounce_bot_lib/legacy.py` is also ask-first. Nothing in that
file was touched. Sequence when authorized: fixtures of today's behaviour on
recorded bars, then the predicate, then re-run and diff deliberately.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **4020 passed / 19 subtests**, exit **0** (was 3992; +28) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| frozen exe | not rebuilt; Smart App Control refuses the build and the desk runs from source |

### Owed

The three live gates in `plan.md` item 11, and the detector-side decision
above. Also still true from the tenth pass: **the running desk predates all of
today's commits** and needs a restart to carry any of it.

---

## 2026-08-21, tenth pass â€” THE DIAGNOSTIC WEEK CAUGHT THE REPAIR ITSELF

**Branch `phase05-integration-blitz`.** R6(c) was activated to watch for
regressions. On day one it found one, and it was the repair's own.

### Measured, not theorised

| Time | What the desk did |
|---|---|
| 07:52 | relaunched from source on `d0aebd5`, 642 MB |
| 08:00-09:10 | ordinary use; working set climbs 817 MB â†’ 7.0 GB |
| **09:14:01** | **297,994 ms** stall (8,918 samples), unresponsive |
| **09:17:22** | **200 s** stall, still climbing - peaks at **8.1 GB** |
| 09:22 | responsive again at **1.96 GB** |

Roughly six gigabytes released in one pause. That is a garbage collection, and
its size is the whole story.

### The mechanism

`install_gui_thread_gc` calls `gc.disable()` process-wide - deliberate, and
older than this repair: automatic collection runs on whichever thread happens
to allocate, and a collection on a scanner thread that frees a cycle holding a
PySide6 wrapper runs a QObject destructor off the GUI thread, which corrupted
the heap in every session on 2026-07-29. So the 2-second GUI timer is **the
only collector in the process**.

`d0aebd5` then gated that timer on input idleness:

```python
if idle_ms < self.young_idle_ms:      # 250 ms
    return
if self.full_due and idle_ms >= self.full_idle_ms:   # 2 s
```

With no upper bound. A trader working the desk produces input every few hundred
milliseconds, so **neither branch is ever reached while the desk is in use** -
not the full sweep, not even the young one. Cycles accumulate with nothing else
in the process able to free them, and the first sweep after the trader finally
pauses has an eight-gigabyte heap to walk.

The intent was right - do not put the largest pause on top of a click. The
error was making activity able to cancel a sweep rather than only postpone it.

### The fix

Every wait now carries a deadline in ticks. Inside it, idleness decides exactly
as before; at it, the sweep runs regardless:

- `young_deadline_ticks=5` â†’ at most ~10 s at the production 2 s tick
- `full_deadline_ticks=90` â†’ at most ~3 min

The pre-repair code swept unconditionally every 2 s and every 60 s, so the
worst case here is a small multiple of what shipped for months - not a new
regime. Six tests pin it, including the two that fail on the old code:
continuous input cannot starve either sweep, and the full deadline is measured
from when the sweep came due rather than from process start.

**Rule for anything added here later:** a "wait for quiet" in this controller
must carry a bound. Unbounded, it is indistinguishable from "never collect".

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3992 passed / 19 subtests**, exit **0** |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| frozen exe | not rebuilt; Smart App Control refuses the build and the desk runs from source, so this is live at the next restart |

### Owed

**The running desk is still on the starved build** - it was launched at 07:52,
before this commit. It needs a restart to pick this up, together with the four
integrations in the ninth pass. Then: watch working set across a full session,
and confirm no stall over ~5 s survives at a repaired seam.

Two smaller things the same log named, neither addressed here:
`chart_data_service.py:221` (1.4 s) and `focus_picks_panel.py:396` (13.9 s at
08:05, 11.9 s at 08:56) are real blocking work on the GUI thread that the GC
noise was hiding.

---

## 2026-08-21, ninth pass â€” FOUR TRADER ASKS, ONE OF THEM A ROOT CAUSE

**Branch `phase05-integration-blitz`.** Trader handed over four integrations in
one message; each ambiguity was put back to them before code was written, and
their four answers are what got built. `plan.md` Phase 0.5 item 10 holds the
live gates.

### What was decided, not assumed

| Ask | The real choice | Trader's call |
|---|---|---|
| "SMA incoming" veto | a v3 bump re-cohorts eight unchanged reasons a second time | v3 **and** pool identical definitions |
| post-earnings + 2nd-dev claims | three `post_earnings_*` or the whole earnings group | the three, plus `second_dev_breakout` |
| RS/RW board placement | new nav page vs a second half of the Strength Board page | second half, draggable splitter |
| the giant candle | guard only, guard + log, or also hunt the source | guard, log, **and** hunt |

### The candle was a real defect, not a styling problem

A chart takes its y-range from bar **lows and highs** and draws the body from
**opens and closes**. Those are the same numbers only while
`low <= open, close <= high` holds. A bar that breaks it paints a solid column
through the entire viewport while the axis still reads perfectly normally -
which is exactly the GFS M5 screenshot: a full-height green bar over a session
scaled 47.0-48.6.

`scripts/ui/bar_integrity.py` now owns that judgement for both the renderer and
the data service. A malformed bar draws **dashed, hollow, caution-coloured,
body clamped into its own low/high**, is excluded from the scale, and is
counted in a bottom-left note. It is never silently dropped - missing data is
uncertainty, never confirmation - and `ChartDataService` logs each one once to
`bad_bars.jsonl` with symbol, timestamp, OHLC and the cache it came from.

**The honest limit:** if the offending bar turns out to be *well formed* and
merely an aggregate row (a daily bar in an M5 series), this guard does not
fire. That case is now OBSERVED rather than guessed at - `range_outliers()`
logs a bar whose range is both 6x the series median and half its whole range -
so the next occurrence names which failure it is.

**Diagnosed in BounceBot, deliberately not edited:** `_get_cached_bars`
(`bounce_bot_lib/legacy.py:8370-8371`) does
`self.latest_bars.setdefault(symbol, bars_ib)` for whatever duration and bar
size it just fetched, and `m5_chart_bars` falls back to that same plain key.
A symbol whose `|5 D|5 mins` key is missing can therefore be charted from a
daily or hourly series - most plausibly SPY, whose group-strength D1 fetch is
`6 M`/`1 day`. Latent, not observed. That file houses detector code, so it is
ask-first and stays untouched.

### The other three

- **Veto v3** carries "SMA incoming" on hotkey **0**; the nine existing digits
  do not move. `canonical_veto_cohort` pools by DEFINITION (code, label, hint,
  note rule), so v1/v2/v3 `volume_dry` grade as one cohort while `compressed`
  (new in v2) and `sma_incoming` (new in v3) grade on their own. Pooling is a
  reading applied at rollup time; picks and outcomes keep their captured
  version and are never rewritten, so it is reversible.
- **The like rail** offers Main swing plus the three post-earnings families and
  `second_dev_breakout`, keyed `1234567890qwerty...` in list order. A test
  fails loudly if an id in `EXTRA_CLAIM_IDS` is not one the registry names.
- **The Strength Board page** gained an RS/RW half in a vertical splitter, fed
  by a second listener on the existing `rrsSnapshotChanged` signal. No new nav
  entry, no second fetch, no second chart widget.

### Three tests were wrong before this and are fixed

`test_chart_review_workspace` and `test_veto_cohort` asserted the literal
string `veto_v2_...`, which the repo's own rule forbids ("never assert a
literal `vocab_version`; assert against the loaded vocabulary"). They now read
the version from the vocabulary, so the next bump will not have to edit them.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3987 passed / 19 subtests**, exit **0** (was 3945; +42 new) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| ruff | **not run** - not installed in this venv (`No module named ruff`) |
| frozen exe | **not rebuilt.** `veto_reasons_v3.json` is a new runtime asset (trigger 2), but the spec mirrors every non-`.py` file under each first-party tree and `test_packaging_spec_drift` passes. Moot in practice while Smart App Control refuses the build - the desk runs from source, so this commit is live at the next restart |

### Owed

The four live gates in `plan.md` item 10, on top of the unchanged R6(c)
diagnostic week. Above all: read `bad_bars.jsonl` after the next occurrence and
find out whether the bad bar is malformed or an aggregate - that answer decides
whether anything needs to change in `bounce_bot_lib` at all.

---

## 2026-08-21, eighth pass â€” THE REPAIR REACHES THE DESK, AND THE FROZEN EXE NO LONGER RUNS

**Branch `phase05-integration-blitz`.** No source change. This pass corrects a
delivery assumption, records a new machine constraint, and adds the launcher the
constraint forces.

### The seventh pass was green and undelivered

That pass closed with `frozen exe: **not rebuilt** â€” no trigger`. Correct by the
packaging rule and wrong in effect: **the trader launches
`dist\TradingBotV3\TradingBotV3.exe`**, so `d0aebd5` never reached the desk. The
process running overnight was the pre-fix binary.

Consequences, stated plainly:

- every `ui_stalls.jsonl` row written before **2026-08-21 07:52 PT** is a
  **pre-fix baseline**, not R6(c) evidence. The bounded diagnostic week starts
  at that launch, not at the seventh pass.
- the desk was hung when it was asked to close this morning:
  `Responding = False`, 4.0 GB working set, no exit after 90 s. The main window
  closed on a second request; the process then lingered **windowless and alive**
  until it was terminated. **Shutdown not completing is an open defect** â€” it is
  unrepaired by `d0aebd5` and belongs in the diagnostic week's findings.

Cheap check for "is the desk running this commit": PyInstaller writes module
names into the bundle in plaintext, so
`grep -ac timer_utils dist/TradingBotV3/TradingBotV3.exe` returned **0** against
the old bundle and **1** against the rebuild.

### Smart App Control refused this build too â€” the open item since 2026-08-19

The rebuild itself succeeded â€” `ui: 120 submodules` (was 119), `timer_utils`
present, spec assets unchanged. It will not start:

> Program 'TradingBotV3.exe' failed to run: **An Application Control policy has
> blocked this file**

This is **not new**. It is the open item recorded on 2026-08-19 midday
(`VerifiedAndReputablePolicyState = 1`, CodeIntegrity 3077/3118), re-confirmed
today. What is new is that it now costs something: that entry reasoned "the desk
is unaffected â€” the 07:00 task launches from source", and overnight the desk was
in fact launched **from the frozen exe**, so the block landed on the live launch
path rather than on a gate.

SAC verdicts are **per file hash**, which is the whole shape of this problem: the
2026-08-20 13:19 bundle ran all night, and the byte-different 2026-08-21 07:49
bundle is refused. A rebuild is therefore a coin toss, and "the frozen selftest
passed last week" says nothing about the next build. A routine Code Integrity
policy refresh is logged at 07:50:44 with **no change in any active policy** â€”
noted so the next reader does not mistake it for a cause.

The exits are unchanged and remain a **trader decision, not an AI's**: a
reputable code-signing certificate, or turning SAC off (irreversible without a
Windows reinstall).

While this holds, **the source launch is production**: a pushed commit is live at
the next restart and the frozen exe is a verification artifact only. The frozen
`--selftest` is unavailable as a gate for the same reason; the source selftest
still runs.

### Running now

`.venv\Scripts\python.exe launch_gui.py`, PID 27416, responding, ~680 MB. The
fix is confirmed live from the watchdog's own frames: `app.py:1056 main` /
`app.py:896 sweep`, where the pre-fix bundle logged `app.py:978 main` /
`app.py:860 _sweep`.

`trading_desk.cmd` (repo root) + a **Trading Desk** Desktop shortcut launch it
minimized, so no editor or terminal has to stay open. Both branches of the
script were exercised in a scratch copy â€” happy path exit 0, missing-venv path a
named error and exit 1 â€” rather than by starting a second desk against the same
mutable state.

### Verification baseline

**Unchanged from the seventh pass** (`pytest tests/ -q` â†’ 3945 passed, exit 0;
smoke 7/7; source selftest 56/56). No source file was touched here â€” the changes
are this file, `CHANGELOG.md`, `CLAUDE.md`/`AGENTS.md` and a new batch launcher
that nothing imports. The suite was **deliberately not re-run mid-session**: a
full-CPU run beside the live desk would manufacture exactly the stalls the
diagnostic week is measuring. Re-run it after the close if any doubt remains.

### Owed

Unchanged R6(c) diagnostic week, now actually started. Additionally: decide the
Smart App Control question, and treat the incomplete shutdown as a finding to
reproduce.

---

## 2026-08-20, seventh pass â€” THE DESK NO LONGER PARSES HISTORY ON A HEALTH TICK

**Branch `phase05-integration-blitz`.** Trader-authorized Phase-0 GUI
responsiveness repair and R6(c) diagnostic activation. Code is complete and
deterministically green; implementation commit **`d0aebd5`** is pushed and the
bounded live week is now running.

### Evidence, not a resource guess

Windows recorded two real `AppHangB1` events today: frozen
`TradingBotV3.exe` at **07:19** and source `python.exe` at **14:16**. The desk
had CPU, RAM and GPU headroom. The blocking work was application-side:

- `BounceService.refresh_health()` ran every 3 seconds on Qt and, whenever
  `avwap_signals.csv` changed, parsed the complete **63.88 MB / 370,109-row**
  history. A warm parse measured **1.268 s**.
- GUI-originated Away-report publication included the operations audit, which
  measured **0.540 s**; the Focus JSON was **13.88 MB** (about **0.079 s** to
  parse).
- independently-created 30/60-second timers retained the same phase, so their
  work arrived as a herd; the full generation-2 GUI garbage collection was
  also scheduled on the fixed 60-second boundary.

The 07:19 hang landed 63 seconds after `IB: connected`, consistent with the
first timer cohort. The 14:16 hang followed scan/wrap-up activity. These are
the causal seams repaired here; no detector, score, threshold, alert decision
or completed-bar rule changed.

### Repair

- The Master scanner now atomically publishes
  `master_avwap_active_events.json`, a current-session BOUNCE-only projection
  carrying the exact source size/mtime signature. GUI health validates that
  tiny projection. A missing/stale projection falls back to the historical CSV
  **once on a single-flight worker**, never on Qt; unchanged signatures reuse
  the last count.
- GUI-originated Away-report/audit writes now run on a background thread behind
  one publication lock. Requests coalesce, while hourly completion state still
  advances only after a verified publish. Existing scan/wrap-up workers share
  the same writer lock.
- Alert Center D1 watches now read only the shared chart service's memory cache.
  D1 store freshness, parsing and earnings-anchor resolution run through the
  existing two-thread chart pool; an unwarmed symbol is honestly UNKNOWN for a
  poll instead of freezing the desk.
- `timer_utils.start_staggered` gives the major 30/60-second jobs distinct first
  phases without changing their recurring intervals. The initial phase is
  never earlier than the timer's original cadence (except the intentionally
  fast 3-second health display).
- Cyclic GC still owns Qt-wrapper destruction on the GUI thread, but a young
  sweep waits for 250 ms without input and a due full sweep waits for 2 seconds
  of idleness. A click/wheel/key event cannot have a full heap sweep scheduled
  directly on top of it.
- Machine-local `ui_stall_watchdog=true`, threshold **50 ms**, is saved now.
  It takes effect at the next launch and writes bounded diagnostics to
  `%LOCALAPPDATA%\TradingBotV3\diagnostics\ui_stalls.jsonl`.

### Gate figures

| Check | Result |
|---|---|
| focused responsiveness/lifecycle/cache/mode tests | **75 passed**, exit 0 |
| Alert Center/watch regression slice | **128 passed**, exit 0 |
| service/chart/report regression slice | **182 passed**, exit 0 |
| `pytest tests/ -q` | **3945 passed / 19 subtests**, exit **0** |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| frozen exe | **not rebuilt** â€” no dependency, runtime asset, new top-level package, dynamic import or root-path trigger; the new module is an ordinary import inside the already-collected `ui` package. **Corrected 2026-08-21:** true about packaging risk, but the trader launches the exe, so this commit did not reach the desk until the eighth pass â€” see the top entry |

### Owed, live

Restart the desk, then run the R6(c) bounded diagnostic week. Require no new
Windows `Application Hang` event, no repeated >50 ms culprit at a repaired
seam, and no responsiveness regression while clicking/typing during scan and
wrap-up. Confirm the next Master scan writes the compact active-event file and
that Away reports still verify. Inspect the watchdog log after each session;
do not tune a detector or threshold from this diagnostic.

---

## 2026-08-20, sixth pass â€” THE VETO COHORT IS GRADED NIGHTLY

**Branch `phase05-integration-blitz`.** Agreed design, built to spec (W1â€“W5).

### W1 â€” the function with zero callers now has one

`update_veto_cohort_outcomes` shipped with the cohort packet and was **never
called**. Picks accumulated on every veto commit; nothing graded them.

`ai_jobs/cohorts.py` â†’ slot **`veto_cohort_grading`**, appended fourth
(5-minute reserve, 3 attempts). A slot rather than a step inside
`journal_import`, because the slot is the unit the runner already gives every
job â€” own ledger row, retry budget, reserve check, failure isolation â€” and
folding it in would make a grading failure read as a journal failure. Last,
not first: it costs seconds and the briefs must not lose window time to it.
**Deterministic â€” no model is called**, and a test asserts the provider is
never even consulted.

**Measured on the desk's real data:** 45 picks â†’ 44 graded outcome rows,
0 sideless. `performance_rows: 0`, correctly â€” every pick is from today, so no
horizon has matured yet.

**Sideless rows are counted and named, never graded.**
`human_focus_tracking._side_label` reads anything that is not "SHORTâ€¦" as LONG,
blank included, so handing it one would manufacture a directional claim the
trader never made. Only their presence stages a filtered copy; the healthy path
touches no extra file.

**Idempotence, stated precisely.** A re-run changes exactly one column â€”
`updated_at` â€” and nothing measured. Byte-identical is deliberately *not* the
claim: a provenance stamp is supposed to move. Writing the failure test
surfaced the mechanism behind it â€” a fully matured pick is never recomputed,
which is why patching the outcome computer to raise did not raise.

**The volume defect does not reach these numbers.** Confirmed by inspection:
`human_focus_tracking` contains no reference to volume, AVWAP or bands. The
forward return is close-to-close only.

### W2 â€” the cohort key carries its vocabulary version

`veto_cohort_source(code, vocab_version)` â†’ `veto_v2_compressed`. An omitted
version keeps the historical `veto_<code>`, which is what lets the 45 rows
already on disk keep grading in the cohort they were filed under â€” they are
not rewritten.

**Cost recorded, not hidden:** eight of nine v2 reasons are byte-identical to
their v1 entry, so this splits eight cohorts that could have been pooled. Right
way round (pooling stays recoverable from the key; a wrongly pooled cohort is
not), but it halves the sample per reason across the bump.

### W3 â€” `trader_judgement`, opt-in

Three sources in funding order â€” performance rollup, outcomes, then the raw
annotation log **last** (the same rule that stopped the setup tracker starving
its own scope). **Not** in `DEFAULT_SCOPES` or `TICKER_BRIEF_SCOPES`. Two
machine-written caveats travel with it as data: Main-swing-only claims, and
"Veto D1 â€” but M5 today" writing an ordinary veto row.

On demand: `run_ai_jobs.py --scopes trader_judgement`. The override is built
per call, so an opt-in scope cannot leak into the unattended slate by being set
once; unknown names are rejected at the CLI.

### W4 â€” review-event freshness, and the number that settles it

**The store is healthy: 8,077 decisions over 19 sessions, newest today.** The
legacy `.jsonl` going quiet on 07-30 was the shards taking over by design.

The audit already computed the newest merged timestamp and hid it in `details`.
It is now in the summary line System Health renders, and staleness is counted
in **sessions** via `market_calendar` (a Friday event read on Tuesday after a
long weekend is one session behind, not four days) with a 2-session threshold.
Unknown is never stale.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3942 passed / 19 subtests, 0 failed**; exit `0xC0000409` (known Qt-teardown crash after the summary). The intermittent `test_stale_d1_tailâ€¦` flake did not fire this run |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| frozen exe | **not rebuilt** â€” `ai_jobs` is in `PACKAGES_NOT_IN_THE_BUNDLE`, so a new module inside it is not a packaging trigger; spec-drift test passes |

### Owed, live

One weekend where the graded cohort is actually read (`--scopes
trader_judgement`) and the trader confirms the reasons ranked against forward
returns are the ones they recognise. Recorded against R8 in `plan.md`. The
weekly synthesis job is **not authorized** â€” cadence decided, gated on two
weeks of graded rows.

---

## 2026-08-20, fifth pass â€” THE CHART POPUP WAS UNTYPEABLE BY DESIGN

**Branch `phase05-integration-blitz`.** Trader: "i cant type in the master
avwap charts that I double click on in the notes section."

### One flag

`SymbolSnapshotDialog` set **`Qt.WindowDoesNotAcceptFocus`**. That flag does
not mean "do not steal focus" â€” it tells the window system the window may
**never hold keyboard focus**, so no widget inside it could receive a
keystroke. The note field, the veto note, the like note and the symbol box were
all dead: clicking in worked, typing did nothing.

**Pre-existing.** The flag has been there since the dialog was written; the
capture rail becoming the product is what made it matter.

### The intent was right, the mechanism was not

A chart popping up must not pull the caret out of a watchlist editor or the
live feed. That is `WA_ShowWithoutActivating`'s job (on Windows it maps to
`SW_SHOWNOACTIVATE`), together with `show()` + `raise_()` and **no**
`activateWindow()` in `show_symbol` â€” all kept. Those govern what happens when
the popup **appears**. `WindowDoesNotAcceptFocus` governed what could ever
happen afterwards, which is a different question and the wrong answer to it.

The two other users of the flag are correct and untouched: the price-alert
toast has no input and must never take focus, and the satellite window is
retired.

### Why the test asserts a flag and not a keystroke

**The offscreen platform does not enforce OS focus rules.** A test that focuses
the note field and types passes with the flag set *and* unset, so it would
never have caught this. The flag's absence is the contract, so the flag is what
is pinned â€” verified failing on the pre-change tree.

For the same reason the neighbouring "does not steal focus" test no longer
asserts `editor.hasFocus()`: offscreen has no show-without-activate, so that
assertion measured the test platform rather than the behaviour.

### One thing to confirm on the desk

Focus-stealing is the half that cannot be tested here. If the popup now pulls
the caret when it opens, that is this change and it is one line to revisit â€”
but `WA_ShowWithoutActivating` is the documented mechanism for exactly this and
is still set.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3902 passed / 19 subtests, 1 failed** â€” the pre-existing full-suite flake |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |

---

## 2026-08-20, fourth pass â€” EARNINGS ON THE CHART, AND VETO VOCABULARY v2

**Branch `phase05-integration-blitz`.** All trader-authorized, same day.

### The finding that shaped the earnings work

**The earnings cache holds NO future dates.** Measured on the desk's own
(`earnings_dates_cache.json`, refreshed 2026-08-20): **1,885 symbols, not one
forward date.** So "when does this name report next" is not a lookup â€” it is a
projection from the symbol's own cadence, and it is labelled `est` everywhere
it appears.

(A first read of that file suggested it was a year stale. It is not â€” the date
lists are stored **newest-first**, so a tail slice shows the OLDEST entries.
Median newest date is 2026-07-29. No staleness problem exists.)

`scripts/earnings_projection.py` is the math: pure, no I/O, no detector
contact. **Median** gap, not mean â€” one moved report would drag a mean around.
Gaps outside 40â€“200 days are dropped before the median (duplicated rows and
cache holes are not a rhythm). Measured cadence across the cache: **91 days**.

### Two things real symbols caught that fixtures would not have

1. **NVDA projected 08/19, one day before the reference.** The first draft
   rolled that forward a whole quarter and reported **November** for a report
   landing that week. `OVERDUE_GRACE_DAYS` (10) now keeps a just-passed
   projection and flags it **"E due"** instead.
2. **`MAX_PROJECTION_DAYS` was dead code** â€” a projection lands at most one
   cadence past the last report, and `MAX_CADENCE_DAYS` already bounds that at
   200, so a 200-day cap could never fire. Removed rather than left looking
   like a guard.

### Presentation, as the trader chose

- **E on a top ribbon**, dotted connector down to its own candle, never buried
  in price action. A report on a day the chart does not hold gets **no**
  marker â€” it is never nudged onto a neighbouring candle.
- **Reserved headroom on every symbol**, not only ones with an earnings date:
  otherwise two names at the same price draw at different scales. Without it a
  chart running to the top-right puts its E through the candles that made it â€”
  pinned by a test.
- **Projection pinned to the viewport's top-right, axis NOT extended.** It sits
  a median **48 sessions** past the last bar, so drawing it in place would cost
  ~40% of candle width to reach a date that is an estimate anyway.
- Built on the chart-data worker beside the levels, so the paint path still
  reads no caches; a failed lookup costs the markers, never the chart.

### Veto vocabulary v2

"S/R cluttered" â†’ **"Compressed"**, as a **NEW code in `veto_reasons_v2.json`,
not a rename.** v1's own description sets that rule: a code is never reused for
a different meaning, because rows already carry it â€” and "too many levels in
the path" is not "range too tight to work with". v1 stays on disk and stays
loadable; every surviving code keeps its meaning **and its digit**.

Two tests hardcoded `vocab_version == 1` and failed the moment v2 shipped. They
now assert against the loaded vocabulary â€” the property they were always about
is that a row stamps the list it was written from, not that the number is 1.

### Like + claim

A numbered picklist like the veto, **Main swing only, for now** (trader's
words). A combo hides every option until opened, which is the opposite of the
rail's five-second contract; Alt+K then a digit is now a whole like. The
earnings-cycle, study and playbook groups are unreachable from this rail while
this stands â€” re-admitting one is adding it to `MAIN_CLAIM_GROUP`.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3899 passed / 19 subtests, 1 failed** â€” the pre-existing full-suite flake, verified failing identically on the pre-change tree |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| frozen exe | **not rebuilt.** `veto_reasons_v2.json` is a new runtime asset (trigger 2) and `earnings_projection.py` a new lazily-imported module, but the spec-drift test passes â€” the spec mirrors every non-`.py` file under each first-party tree, and the loose module is collected exactly as `chart_levels` already is from the same call site |

---

## 2026-08-20, third pass â€” THE PANE WAS SPENDING 1240px ON WHITESPACE

**Branch `phase05-integration-blitz`.** Trader, with a full-desk screenshot:
"look at how inefficient this GUI is. this bot basically gets an entire 4k
monitor and we cant fit everything in cleanly?"

### The measurement, before any change

`AlertChartReview` at 2000x1900 with **no alert charted**:

| Row | Height | Needs |
|---|---|---|
| title (one line) | **346px** | 17 |
| setup line ("Waiting for the next ticker alert.") | **346px** | 16 |
| arm bar | **346px** | 107 |
| verb row | **346px** | 28 |

~1240px of a 4K screen on whitespace, for ~170px of content â€” in the state the
desk sits in **whenever the review queue is clear**.

**One-line cause.** The snapshot carries the pane's only expanding stretch, so
HIDING it left Qt with four `Preferred` widgets and a column of slack, which it
split equally. Charted, the same pane was already correct (chart 1212 of
1408px) â€” which is exactly why this never showed up in a charted screenshot.

**Fix.** An expanding `EmptyState` occupies the chart's slot whenever the chart
is hidden, so a stretch item is always present and the slack collects in one
place that explains how to get a chart. Title / setup line / arm bar pinned to
`Maximum` vertically â€” a `QLabel` defaults to `Preferred`, i.e. "I will happily
take more".

### The capture rail: 900px column â†’ 379px of columns

Sections now **flow** (the primitive the arm bar already uses): wide hosts put
veto / like / note side by side, the narrow Capture tab still stacks them with
nothing clipped. Symbol and side share one line. The veto list is sized **from
the vocabulary** instead of a hardcoded 190px cap, so all nine reasons are
visible â€” a surface built for two keystrokes cannot ask for a digit the trader
cannot see. Deliberately NOT a wrapped multi-column list: those labels only fit
in columns by eliding them.

### Capture verbs

- **LIKE now retires the chart**, like a veto â€” in the Alert Center queue and
  in the Master AVWAP snapshot popup, which already had
  `snapshot_review_advance` for exactly this.
- **NOTE still holds the chart.** It is written ABOUT the thing in front of
  you; a rail that skipped would make every note cost the trader that chart.
- **Hypothetical stop removed** from the rail. The **control only** â€”
  `ui.annotations.store` still builds and validates `hypo_stop` rows, because
  the stream is append-only evidence and rows already on disk have to stay
  readable. Re-adding it is a layout change, not a migration.

### Still open, deliberately

The **horizontal** split was not touched. The screenshot shows the Setups table
truncating columns ("Diagnostics & Li...", "AVWAP_BAND...") while the alert
column holds an empty chart â€” but that split is **persisted** (`qt_desk_split_sizes_v2`)
and may be one the trader dragged themselves, so re-weighting it silently would
overwrite their own choice. Needs a decision, not a guess.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3860 passed / 19 subtests, 1 failed** â€” the pre-existing full-suite flake, verified failing identically on the pre-change tree |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| frozen exe | **not rebuilt** â€” no packaging trigger hit |

---

## 2026-08-20, second pass â€” OPEN DEFECT: THE DAILY STORE MIXES TWO VOLUME UNITS

**Branch `phase05-integration-blitz`.** Found while wiring D1 volume bars.
**Not fixed â€” it needs a trader decision, and the fix is inside detector
input data (ask-first rule).**

### What is wrong

`data/daily_bars/*.parquet` carries volume from two sources with **no unit
normalization between them**. `master_avwap_lib.legacy._normalize_daily_bar_frame`
normalizes column names, dtypes and duplicate dates â€” and nothing else. IBKR
rows and Yahoo rows are appended into one column as-is.

Proven against a reference, NVDA, same file:

| Session | Daily store | yfinance | Ratio |
|---|---|---|---|
| 2026-05-18 | 934,776 | 146,280,900 | 156Ã— |
| 2026-05-19 | 823,818 | 140,948,200 | 171Ã— |
| 2026-05-20 | 940,980 | 184,201,600 | 196Ã— |
| 2026-05-27 | 167,601,200 | 167,601,200 | **1Ã— (exact)** |
| 2026-06-01 | 212,850,700 | 212,850,700 | **1Ã— (exact)** |

The Yahoo-sourced rows are exact. The IBKR-sourced rows are low by a
**variable** 150â€“200Ã—, so it is not a clean 100-share-lot conversion and a
constant rescale would be a guess. It alternates in blocks, following whichever
source answered on the day.

**Scale: 338 of 1,949 stored symbols (17.3%)** have a volume series straddling
two magnitudes (p90/p10 > 20Ã—). That is an upper bound â€” a genuinely spiky name
can trip the same test â€” but the mechanism is confirmed by reading the code,
not inferred from the statistic.

### Why it matters beyond the chart

`calc_anchored_vwap_bands` is **volume-weighted** over this frame's `volume`
column. A day under-reported 150Ã— contributes ~0.6% of its true weight, so on
an affected symbol the D1 anchored VWAP is effectively computed from the
Yahoo-sourced days alone. Every band consumer â€” events, zones, tracker
families, scoring history â€” sits downstream of that.

This is not a new regression. It has been true for as long as the store has
mixed sources; the volume bars only made it visible.

### Why nothing was changed

- The fix lands in detector **input** data. plan.md sec 5: no detector/scoring
  behavior change without golden-result fixtures first, and the file-scoped
  ask-first rule covers `master_avwap_lib/legacy.py`.
- Re-weighting AVWAP would move every band on affected symbols. That is a
  recalibration, not a bug fix, and it is the trader's call.
- The correction factor is not constant, so there is no safe silent repair.

### The decision owed

1. Normalize at the writer and **backfill** the store from one source â€” moves
   the bands, needs golden fixtures first.
2. Normalize at the writer for **new rows only** â€” stops the bleed, leaves a
   discontinuity mid-history.
3. Drop volume from the IBKR path and take it from Yahoo only â€” one unit, one
   source, at the cost of an extra fetch.
4. Leave it, and treat D1 volume (and the volume-weighting of D1 AVWAP) as
   approximate on affected names.

Until one is chosen: **the D1 volume underlay is honest about what it was
given and nothing more.** It draws the numbers in the store. On an affected
symbol roughly half the sessions will render as near-nothing columns. It does
NOT invent a correction.

### Also open (pre-existing, not mine)

`tests/test_chart_snapshot.py::test_stale_d1_tail_triggers_one_backfill_with_cooldown`
fails intermittently in a **full-suite** run and passes alone. Verified to fail
identically on the pre-change tree, so it is not caused by this work. It is a
threaded backfill test spinning the event loop against a 10s deadline; under
full-suite load it can miss it. Left alone deliberately â€” unlike this morning's
clock fixture, this one needs an investigation of shared state in a
chart/alert-adjacent widget, not a three-line repair.

---

## 2026-08-20, second pass â€” WHAT THE TRADER ASKED FOR AFTER USING IT

Four changes, all trader-authorized in one message.

1. **Veto retires the chart.** "When I click veto it should just disappear as
   'not for today'." A veto now takes the "Not today" path: recorded, removed
   from today's feed and chart queue, next chart up. LIKE, hypothetical stop
   and note still hold the chart â€” a note that skipped to the next symbol
   would cost the trader the chart they were writing it about.
2. **"Veto D1 - but M5 today".** "It may be a shit D1 chart but its a good
   daytrade." The rail does not place the name; it emits a REQUEST and the
   panel that owns the Focus store does the placement, same shape as
   BounceBot's desync request, one writer per store. Place first, retire
   second â€” retiring is what drops the alert object the placement needs. A
   failed placement still retires the chart, because the veto is already on
   disk. **Known limitation, deliberately not papered over:** the veto row is
   an ordinary veto with no new field, so the veto cohort study will count a
   day-traded name as vetoed. Making that queryable is a schema v2 decision.
3. **The arm bar comes back under the chart.** "I also need my m5 and D1 alert
   hotbuttons back on the bottom of the visual chart... I also need the ability
   to input a ticker manually as well." Only the capture rail stays on a tab.
   Measured at this column's 420px: rail 697px, arm bar 131px â€” sending only
   the rail away keeps 84% of the reclaimed height. `docked_controls` splits
   into `dock_arm_bar` / `dock_capture_rail`. The Armed tab is the
   cross-symbol inventory again; the verb-row armed line switches off with the
   bar docked (its own chips are right there), and the tab keeps its count.
4. **D1 volume bars** â€” an underlay in the bottom 18% of the price view, not a
   stacked sub-plot, so they cost no chart height. No fetch: the daily store
   already carries volume. **Read the open-defect entry above before trusting
   what they show.**

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3847 passed / 19 subtests, 1 failed** â€” the failure is the pre-existing full-suite flake named above, which fails identically on the pre-change tree |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| frozen exe | **not rebuilt** â€” no packaging trigger hit |

Fail-before-feature: all 10 volume tests and the new veto/layout tests were run
against the pre-change tree and failed there.

### A desk restart IS needed

Source-level only, as before.

---

## 2026-08-20, morning â€” THE CHARTS GET THE PANE. READ THIS FIRST.

**Branch `phase05-integration-blitz`.** Built on the desk's live checkout
(`C:\Users\Aaron\TradingBotV3`) while the desk was running. **Nothing was
restarted and no scheduled task was touched.**

### What the trader asked for, and got

> "I cannot see the charts at allâ€¦ I am ok with them being tabbed where
> alerts/D1 focus/RSRW board is and clicking into them. But I need to be able
> to see the charts."

The Alert Center review pane stacked title â†’ setup text â†’ charts â†’ a two-row
arm bar â†’ a ~600px capture rail â†’ the verb row, all in the desk's narrow alert
column. The charts â€” the point of the surface â€” got whatever was left.

**Now: charts, then one row.** The arm bar moved onto the existing **Armed**
tab, above the inventory it fills. The capture rail became a new scrolled
**Capture** tab. Under the charts there is exactly one row left: the verb row
(Remove/Skip/Not today/Add + queue count), which advances the review queue and
must never cost a click.

Tab strip is now `Alerts | D1 Focus | RS/RW Board | Armed | Capture`.

### Why the arm bar joined "Armed" instead of becoming a sixth tab

"Arm" and "Armed" a millimetre apart on one strip is a misclick waiting to
happen, and the controls and the list they produce are one subject. Arming is
also deliberate enough that a click is fine â€” unlike the verb row.

Armed state stays legible with that tab closed, in two places: a count in the
tab title (`Armed (2)`) and an always-visible line on the verb row
(`AlertChartReview.armed_summary`), which replaces the "Nothing armed" text
that went onto the tab with the bar.

### The keyboard contract moved WITH the rail

The rail's founding constraint is every capture under five seconds, no mouse.
**A `QShortcut` bound inside a hidden tab page never fires**, so moving the
rail would have killed Alt+V/K/S/N silently. They are rebound at **panel**
scope (`WidgetWithChildrenShortcut`), and each one raises the Capture tab
before handing off to the rail's own handler.

`CaptureRail` gained `bind_action_shortcuts` (default `True`) and a public
`action_shortcuts()`. The Alert Center's rail binds **none** of its own,
because two live bindings for one sequence is an ambiguous shortcut in Qt and
Qt fires **neither** â€” the failure mode is the keys going dead with nothing on
screen to say so. A test asserts the rail owns no duplicate of a key its host
took. The 1-9 veto digits and every Enter-to-commit path are untouched.

### Placement is a host decision now

`AlertChartReview(docked_controls=...)`, default `True` = the historical
single-column stack. `SymbolSnapshotDialog` and the Chart Review workspace
build their own rails and are unaffected; a docked `AlertChartReview` keeps
its rail, its own keys, and no duplicated armed line. Undocked, the two docks
are `setParent(None)`-detached (references kept, signals intact) so they cannot
paint over the charts before the host adopts them.

### What was NOT touched

CaptureRail semantics (still a recorder: no mute, suppress, score, gate, alert
or watchlist write â€” this was re-parenting only), the movers-only presentation
filter, the repetition fold, and every line of adoption-gate code.

### Second change: a wake alert the trader can verify

Audit confirmed both EVENING-permitted senders already push at ntfy's maximum
(`price_alert_service._notify` and `AutopilotService._maybe_push_spy_alarm`,
both `priority="urgent"`). The gap was the channel **test**, which went out at
`high` â€” so "will this break through iOS Sleep Focus" had never been
answerable.

New **Test wake alert (urgent)** button beside Test Push
(`PriceAlertService.test_push(urgent=True)`), sending one urgent push whose
message says what should have happened. **Not a new sender**: nothing
schedules it, only that button calls it, and the phone push policy is
unchanged. `docs/EVENING_MODE_RUNBOOK.md` gains a Sleep breakthrough checklist
â€” ntfy has no Apple critical-alert entitlement, so urgent priority alone cannot
override Sleep Focus; the device steps are marked to-be-confirmed-on-desk.

### One pre-existing failure repaired

`test_it_reads_the_gates_predicate_over_the_desks_own_bars` was **already red
on this branch before any change here** â€” it built an 11:00 session while
`_measure_mover_state` reads the real wall clock, so at 07:34 its M5 bar was in
the future, was correctly discarded as incomplete, and the assertion read
UNKNOWN. The fixture now pins the clock. It was measuring the time of day, not
the predicate.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3838 passed / 19 subtests, 0 failed**; process exit `0xC0000409` (the known intermittent Qt-teardown crash, measured through Python's `returncode`, raised after the summary printed) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| frozen exe | **deliberately not rebuilt** â€” no packaging trigger is hit (no new dependency, asset, top-level package or dynamic import). Smart App Control still blocks the built exe on this machine; see the 2026-08-19 midday entry, unchanged and unresolved |

Fail-before-feature: 10 of the 11 new Alert Center capture tests were run
against `d60cbaf` and **all 10 failed**; the eleventh is a deliberate
regression guard on the unchanged recorder path.

### A desk restart IS needed

Source-level only, so **the trader sees none of this until the desk is
restarted.** Nothing is urgent: what changed is where controls sit and one new
test button â€” nothing about what is detected, recorded, alerted or pushed.
Cleanest moment is the usual one: let the 07:00 scheduled task relaunch it from
source, or close the desk and relaunch via `scripts/launch_gui_auto.ps1`. No
task disarm is needed, because the branch is not changing.

### Owed, live

- One review session where the trader confirms the charts are readable, the
  four Alt keys still land in the rail from the charts, and the armed count is
  honest.
- The Sleep breakthrough checklist, run once on the phone with Sleep Focus ON,
  ending in a **Test wake alert (urgent)** that actually sounds. Until that
  passes, treat the SPY wake alarm as unverified on the device side.
- Everything owed by the 2026-08-19 entries below still stands.

---

## 2026-08-19, evening â€” MOVERS ONLY IN CHART REVIEW. READ THIS FIRST.

Built on the desk's live checkout (`C:\Users\Aaron\TradingBotV3`, branch
`phase05-integration-blitz`) while the desk was running. **Nothing was restarted
and no scheduled task was touched** â€” see "restart" below.

### The trader's rule, as recorded

> "A long inside yesterday's range is probably chop. Chart review should only
> show me longs above the previous day's high and shorts below the previous
> day's low. Focus picks that ARE beyond their previous-day extreme should be
> flagged - those are the ones actually moving. Inside-range picks appear only
> when I deliberately review focus picks."

Verbatim as a dated addendum in `docs/M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md`.

### One predicate, not two

`focus_adoption_gate.mover_state(side, price, prev_high, prev_low)` is the
adoption gate's **extreme leg alone** â€” a thin name over the same
`prev_day_break_state` call â€” and `focus_adoption_gate_state` now routes its own
extreme leg through it. There is exactly one implementation of "beyond
yesterday's extreme" in the tree, and a test walks the whole input matrix
asserting the two entry points cannot disagree. That is the point: a display
filter with a private copy of the rule would eventually hide a name the machine
had just adopted, and the trader would be reading a queue that disagreed with
their own Focus list.

No session-VWAP leg. The filter asks the weaker question deliberately â€” the
trader wants to *see* movers, not only the ones the machine would take.

### Where the filter lives, and what it will not do

`AlertCenterPanel._enqueue_review_alert` â€” the single door into the review
queue, so the D1 Focus feed, the auto-pick drain and the scanner alerts all pass
through it. Default ON.

- Longs and shorts inside yesterday's range: **not queued**.
- **UNKNOWN shows**, tagged `unmeasured`. Missing data is uncertainty; a filter
  that failed closed would blank the review the moment the daily store hiccuped.
- The withheld are counted on a clickable line, `N hidden (inside yesterday's
  range) - show`. One click shows exactly those names and turns the filter off
  **for that session** (day-scoped â€” tomorrow opens filtered again).
- **Bypassed entirely** by the deliberate Focus review (`review_focus_picks`)
  and by armed chart-watch hits.
- It **hides**: nothing leaves the feed, the history or any store; no alert,
  sound or push is muted; no watchlist or Focus entry is auto-removed;
  `review_policy.json` is untouched; nothing is written to the review-learning
  stream. Each of those is a test.

### The flag

A Focus chip beyond its previous-day extreme on its own side carries `MOVING`,
in the existing badge idiom (the same short uppercase word as `BOUNCE`/`RRS`).
The charted alert shows `MOVING` / `unmeasured` / `inside range` beside the
reviewed-today badge. It repaints from the Alert Center's existing 60-second D1
poll through a new `focusBreakStatesChanged` signal â€” **no new timer, no new
market data, no IB traffic**.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3822 passed / 19 subtests, 0 failed, 0 errors**; process exit **`0xC0000409`** (the intermittent Qt-teardown crash, measured through Python's `returncode` â€” bash shows it as `127`) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| frozen exe | **not executed** â€” Smart App Control still blocks the built exe on this machine (see the midday entry). Unchanged and unresolved; no new module was added to the selftest roster, so the expected count stays 56 |

Fail-before-feature: all 28 new tests were run against `e207851` in a throwaway
worktree first and **all 28 failed**; all 28 pass here.

### A desk restart IS needed

The running desk is executing the Python it imported at launch. Everything here
is source-level, so **the trader sees none of it until the desk is restarted.**
Nothing is urgent: the filter changes what is charted, not what is detected,
recorded or alerted, so the running session keeps working exactly as it did.

Cleanest moment: **after tomorrow's close, or before the 07:00 task on
2026-08-20** â€” the scheduled task launches from source, so letting the desk be
restarted the usual way picks it up with no extra step. To take it now: close the
desk and relaunch via `scripts/launch_gui_auto.ps1` (the task's own path). No task
disarm is needed, because the branch is not changing.

### Owed, live

`docs/DESK_TESTING_PLAN.md` Â§2.10 â€” one review session where the trader
confirms the queue shows only movers and the hidden-count line is honest.


---

## 2026-08-19, midday â€” the desk flipped to this branch

The trader flipped the desk to `phase05-integration-blitz` at 11:08 PT
(mid-session, deliberately â€” trader's call on a slow tape). The worktree
`..\TradingBotV3-blitz` is removed (it was clean and fully pushed); this main
checkout at `C:\Users\Aaron\TradingBotV3` now holds the branch. Sequence
executed: task disarmed â†’ desk closed by the trader â†’ checkout `198a2bd` â†’
gates â†’ manual launch via `scripts/launch_gui_auto.ps1` (the task's own path)
â†’ task re-armed (all three tasks `Ready`). New desk pid 13364, heartbeat
fresh, Auto Pilot resumed ON, slot 11:00 picked up at 11:09:30.

Gates on this checkout, 2026-08-19 ~11:00 PT:

| Check | Result |
|---|---|
| pytest | **3794 passed / 19 subtests, 0 failed**, process exit **0** (the intermittent `0xC0000409` did not occur this run) |
| smoke | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| frozen rebuild | clean-cache rebuild exit 0 â€” **but the exe could not be executed; see below** |

**NEW OPEN ITEM â€” Smart App Control blocks the freshly built exe.** Windows 11
Smart App Control (enforcing, `VerifiedAndReputablePolicyState=1`) refused to
run `dist\TradingBotV3\TradingBotV3.exe` built at ~11:05 ("An Application
Control policy has blocked this file"; CodeIntegrity events 3077/3118 at
11:07). The worktree's byte-different build had run fine at 09:20 the same
morning â€” SAC verdicts are per-hash cloud reputation, so they can differ
between rebuilds. **The desk is unaffected** (the 07:00 task launches from
source, and this flip was verified with the source selftest), but the frozen
gate cannot be relied on to *execute* on this machine until this is resolved.
Options are a trader decision: code-sign the exe (SAC needs a real signature
with reputation), stop using SAC (WARNING: once turned off it cannot be
re-enabled without reinstalling Windows), or accept that the frozen selftest
may intermittently be blocked and re-run it when the verdict clears. Recorded
here; not resolved.

---

## 2026-08-19 â€” the gate that could not tell the time.

**Branch: `phase05-integration-blitz`, pushed.** Everything
below happened in `..\TradingBotV3-blitz` (worktree since removed; see the
flip entry above).

### What the first DESK morning actually did

**Zero adoptions. 121 picks refused every 30 seconds from 08:07 onward.**
`focus_auto_picks.json` finished the day with an empty `picks` map, and the
failure logging rotated `trading_bot.log`.

**Root cause â€” one subtraction, two clocks.** A stored verdict carries two
stamps written by different paths:

| Field | Writer | Awareness |
|---|---|---|
| `gate_bar_end` | the intraday profile's `as_of` (`_intraday_extreme_metrics`) | **always aware** â€” the provider's own offset when it has one, market-local otherwise |
| `gate_checked_at` | the staging refresh's `datetime.now()` | **naive** |
| the caller's `now` | `AlertCenterPanel` â†’ `datetime.now()` | **naive** |

So `pending_pick_gate_ok`'s wall-clock age check (naive âˆ’ naive) passed, and its
bar-lag check (naive âˆ’ aware) raised
`TypeError: can't subtract offset-naive and offset-aware datetimes` â€” exactly the
line the traceback named. The Alert Center caught it and refused fail-closed,
which is correct behaviour for an unverifiable pick.

**The gate did not judge the picks wrongly. It never ran.** Nothing about the
PDH/VWAP rule was exercised on 2026-08-19, which is why Â§2.5 and Â§2.6 of the
testing plan are re-owed **in full** rather than being partly done.

**The fix.** Every datetime the gate compares â€” the caller's clock, both stored
stamps and the `not_before` flip barrier â€” is normalized at one seam
(`_gate_moment` â†’ `market_session.normalize_market_local_datetime`), which
ATTACHES market-local to a naive stamp and converts an aware one. Stripping the
offset instead would have ended the crash and kept the outage: an aware 11:05 ET
bar read as naive against an 08:07 PT clock is three hours "ahead of the tape",
so every pick would still have been refused â€” silently. A test pins that
direction, and every refusal path (stale clock, stale bar, future bar, pre-flip
verdict) is re-asserted unchanged.

`minutes_since_open` carried the identical subtraction one function away and is
hardened the same way. Every caller passes a naive clock today, so its answers
are unchanged; the scheduler is simply no longer where this class of bug gets
discovered live.

**The log flood is bounded.** The refusal wrapper logged a full traceback per
pick, so one systematic fault wrote 121 tracebacks every 30 seconds and rotated
the log holding the evidence. Now the first failure of each poll cycle carries
the traceback and the cycle ends with one WARNING naming the count and the
exception. The refusal is as loud as it was; fail-closed semantics are untouched.

### The retry investigation: design and code agree, no change made

R2.2's budget (`FLIP_REVERIFY_RETRY_SECONDS` = 60, `FLIP_REVERIFY_MAX_ATTEMPTS`
= 5) governs **only** a failed flip re-measurement â€” the `reverify_pending_picks`
fetch on an AWAY/EVENING â†’ DESK return. The desk was in DESK mode from the start
on 08-19, so no flip happened, no re-verification was owed, and that budget was
never engaged. Correctly.

The 30-second cadence in the log is the **ordinary poll**:
`_poll_auto_pick_pending` rides the Alert Center's 30s `_watch_timer`, and a
refused pick is deliberately not marked seen, so every cycle re-attempts the
whole queue. That is designed â€” "a stale verdict costs one cycle rather than the
pick" â€” and it is what makes recovery automatic once the code is fixed rather
than requiring a restart. Two mechanisms were being read as one; nothing
disagreed, so nothing was changed. Recorded in the R2 spec so it is not
re-litigated.

### The strength board, on the trader's two requests

**Sortable columns.** Every heading sorts, with a visible indicator, and clicking
the same heading flips it. Sorting is presentation: it re-orders rows already in
hand and never calls the service, so a header click cannot cost a refetch â€” the
board's budget stays one batched yfinance pull per 15 minutes and **zero IB
traffic**. Qt's own `setSortingEnabled` is deliberately not used: the last column
holds a per-row cell *widget*, and `QTableWidget` leaves cell widgets behind when
it sorts, so the Add button would end up on its neighbour's row. Owning the order
also puts blank cells last in **both** directions â€” an unmeasured field is an
absence, not a small number. The default order is unchanged and now stated by the
indicator (longs strength-descending, shorts ascending â€” strongest for that side
first). Every add still re-runs the adoption gate at click time.

**Charts on selection.** Selecting a row opens that symbol in the desk's existing
snapshot popup â€” the same one the RS/RW, entry and Industry boards use, owned by
the Alert Center â€” so it carries the same bot-backed series, painted levels and
CaptureRail. No new chart widget exists anywhere (R4's unification pattern), and
`show_symbol_snapshot` already reuses one dialog per owner, so re-selecting
re-points that window instead of stacking dialogs. Selecting on one side clears
the other; a refresh that keeps the same row selected is not a new chart request;
double-click still works.

**The docked chart is the follow-up option, not this build.** An always-visible
chart inside the board needs a desk-layout decision about what happens to the two
tables' width on that page â€” a judgement about the trader's screen, not a wiring
problem. The popup reuses a surface the trader already knows.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3794 passed / 19 subtests, 0 failed, 0 errors**. **The PROCESS exit code is `0xC0000409`, not 0** â€” see the finding below |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| Source selftest | **56/56**, exit 0 (roster unchanged â€” no new module is lazily imported) |
| Clean-cache frozen rebuild + frozen selftest | **`selftest OK: 56/56 checks passed (frozen)`**, exit 0. `build/` AND `dist/` deleted first, built from the worktree, so the desk's own `dist/` was never touched; exe mtime 09:20 postdates the commit at 09:15 |


### A correction about the teardown crash, with new evidence

Yesterday's report said the `0xC0000409` native crash "did not occur in any run
today". That observation was true and it was also **misleading**, because the
tool reporting it truncates the code: bash shows `127` for `0xC0000409`, and I
read a genuine `0` yesterday against truncated readings elsewhere. Measured
properly today, through Python's own `returncode`:

| Run | Summary line | Process exit |
|---|---|---|
| today's tip (`80279c3`) | 3794 passed, 0 failed | `0xC0000409` |
| **yesterday's tip (`e266f5f`), re-run today, unchanged code** | 3760 passed, 0 failed | `0xC0000409` |
| today's tip minus `tests/test_ui_stall_watchdog.py` | 3789 passed | `0xC0000409` |
| today's tip minus either new test file | 3776 / 3778 passed | `0xC0000409` |

Three things follow, and none of them is "my changes broke it":

1. **The same commit that read clean yesterday crashes today.** The crash is
   **intermittent**, not resolved and not introduced by this work.
2. **The recorded attribution is stale.** The testing-week note says ignoring
   `tests/test_ui_stall_watchdog.py` returns exit 0; on this tree it does not.
   Whatever the trigger is now, that single file is no longer a discriminator.
3. **Neither of today's new test files causes it** â€” removing either one leaves
   the crash exactly where it was.

**Not "fixed" by editing product code.** `scripts/ui/stall_watchdog.py` is
product code owed R6(c)'s diagnostic week, and making a suite exit cleanly is not
a reason to touch it. The standing rule holds: **quote the summary line AND the
exit code together; neither alone is the truth.** What is new is that the
summary line is now the one that has stayed stable across every run, and the exit
code is the one that moves.

### Live proofs â€” what today changed

**Re-owed in full** (the 08-19 session proved nothing about them, because the
gate crashed before it could judge anything):

- one adoption actually happening on a DESK day (new Â§2.5 check: names landing in
  M5 Focus, `focus_auto_picks.json` non-empty);
- one adoption-time refusal with its reason;
- one scoped "Not today" leaving the trader's other entries intact.

**Newly owed:**

- Â§2.7a â€” the board's sorting and chart-on-selection, on real rows.

**Unchanged and still owed:** the strength board's TC2000-character check, the
EVENING stop, the SPY wake alarm, and everything in R3â€“R8's ledger from the
08-18 report below.

### Putting this build on the desk

Same sequence as the 08-18 report below, with one number changed â€” the frozen
selftest count is **56/56**, unchanged from yesterday, because nothing added a
lazily-imported module today:

1. **Disarm the scheduled task first.**
2. Close the desk app, then in `C:\Users\Aaron\TradingBotV3`:
   `git fetch origin` â†’ `git checkout phase05-integration-blitz`.
3. `.venv\Scripts\python.exe -m pytest tests/ -q` (expect the figure above, exit
   0) and `.venv\Scripts\python.exe scripts/smoke_check.py` (7/7).
4. `.venv\Scripts\pyinstaller.exe .\packaging\tradingbotv3.spec --noconfirm`,
   then `dist\TradingBotV3\TradingBotV3.exe --selftest` â€” expect
   `selftest OK: 56/56 checks passed (frozen)`, exit 0.
5. Launch once by hand; confirm the desk opens on Main and the Focus tab looks
   normal.
6. **Re-arm the scheduled task.**

**On the next DESK morning, check the thing that failed:** within a poll cycle or
two of the first staged picks, names should appear in M5 Focus and the status
line should say `N auto pick(s) added to M5 Focus for today`. If the gate ever
fails again you should see **one** traceback and **one**
`Focus gate check unavailable for N staged pick(s) this cycle` line per 30
seconds â€” not a flood. A flood is a regression worth reporting on its own.


---

## The 2026-08-18 integration blitz report (still current for everything below)

**Branch: `phase05-integration-blitz`, pushed. The desk is untouched.** The main
checkout at `C:\Users\Aaron\TradingBotV3` is still on
`phase05-r2-focus-gating-strength-board`, its `dist/` was never rebuilt from
this work, and the 07:00 scheduled task will run exactly what it ran yesterday.
Everything below happened in a linked worktree at `..\TradingBotV3-blitz`.

### The first thing to know: most of the redirect was already built

The redirect asked for R3, R4, R5, R6, R7 and R8. When the branch audit ran, R3
through R8 were **already built** on `testing-week-2026-08-17` â€” 30-plus commits
from 2026-08-16/17 that the desk branch's own checkpoint had not caught up with.
So the blitz branch was cut from that lineage and the four newer R2 commits were
merged into it, rather than rebuilding landed work (which `CLAUDE.md` forbids).

That merge is itself a deliverable: **one branch now carries testing-week, R1,
R1.1, R2 (including the 08-18 defect fixes), R3, R4, R5, R6, R7 and R8.** Before
today, the desk branch and the release candidate had diverged.

### What was actually built today

| Packet | State before today | What landed |
|---|---|---|
| **R5 Â§3.2** | pure logic not written | Confluence engine (HA reversal + SMI turn + LRSI cross within 4 completed bars), **M5 Focus symbols only**, wired, **default OFF** |
| **R5 Â§3.3** | pure logic not written | First-candle ORB flow: candidate mark, post-pullback new-extreme break, informational LRSI recross â€” three separately toggleable types, all **default OFF** |
| **R5 Â§4** | not started | `AnyBounceWatch`: one armed request per symbol/side over nine levels, own store, Alert Center owns it, fires once naming the level that held then disarms; **Any bounce** button on the arm bar |
| **R5 Â§8.3** | decided, not built | `prev_avwape` carried onto the zone-arms entry as a top-level key, golden fixture first, fixture passes unchanged after the edit |
| **R6(b)** | decided + narrowed | Read-only JSONL-ledger audit inside the existing footprint check; the stale `~106 MB` comment removed. R6 is now fully closed |
| **R7 visuals** | deferred | Analytics per-group bar charts with honest n counts + a CSV of exactly what is charted; Calendar pyqtgraph year heatmap centred on zero |
| **R8 joins** | retained future scope | Week review folds the week's RS/RW extremes per symbol; Focus review joins picks WITH their outcomes, one row per pick |
| **R4 held items** | held ask-first | Focus Picks reviewed-today marker built as a line BESIDE the editors; `review_host` for the boards declined on the record |
| **WISHLIST** | 20+ candidates | One was buildable and is built (external chart deep link). Every other item has one blocking trader question, written down in `docs/WISHLIST_OPEN_QUESTIONS.md` |

### Every autonomous decision I made where a spec was ambiguous

1. **R5 Â§8.2 said do not wire Â§3.2/Â§3.3 until a desk session measures Â§3.1.** The
   redirect is that decision's own first reopen trigger ("the trader
   overrides"), so I wired them â€” and kept the substance by shipping **all four
   new alert types OFF** and writing both engines as **stateless** functions
   over the session's completed bars. Â§8.2's objection was a dormant state
   machine waking mid-session with contents nobody exercised; a function that
   recomputes from bars has nothing to carry. **What the desk session now
   decides is which toggles earn a default-on, not whether the code exists.**
2. **The ORB candidate mark does not seed the bounce outcome tracker.** Only the
   re-break does. Measuring an engine against events it never claimed were
   entries would corrupt the evidence the promotion ladder reads.
3. **The confluence is Focus-scoped at the sweep**, intersecting the watchlist
   with the human focus sets â€” the trader's framing was "on names I'm watching",
   and a perfect chart on a non-Focus name is silence.
4. **The any-bounce watch reuses `detect_zone_arm_triggers`' two-bar idiom**
   rather than inventing a bounce rule, so "bounce" means one thing system-wide.
   Its tolerance for a chart-armed watch (no scan measurement available) is
   0.15% of the level â€” deliberately small, and a named constant.
5. **R4's Focus Picks marker renders beside the editors, not inside them.** Those
   editors hold watchlist text that is synced back; a marker in a row is one
   careless save from becoming a symbol name. Same answer, no path to the data.
6. **`review_host` for the ranked boards is declined**, not deferred: a ranked
   board has no "next row", so advancing through it would invent a queue.
7. **TC2000 deep-linking is not wired.** It answers no documented URL scheme; the
   URL template is a setting, so it is one line of config away the day you tell
   me what your install answers to.
8. **The ledger audit estimates rows from a 256 KB sample** rather than counting
   370 MB of newlines on every System Health render. The field is called
   `estimated_rows` because that is what it is.
9. **Two hermeticity gaps were fixed test-side, not in product code**: the Fed
   calendar adapter (it reached the wire in a full-suite run and passed in
   isolation only because a cache answered first) and the new universe-shape
   tests (they were measuring conftest's own offline stub instead of the
   function under test).

### What the redirect asked for that I deliberately did NOT reopen

**R3 Â§4.3.5, the same-slot volume-thrust normalization.** The trader deferred it
explicitly on 2026-08-16 with a reason that today's redirect does not touch: the
D1 scoring seam has no intraday slot series, the faithful TC2000 baseline would
need a 5-minute fetch across ~1,100 symbols (a data-budget and contract change),
and the zero-fetch session-elapsed proration was offered and REJECTED because
real volume is U-shaped. Reopening it needs a fresh decision about the data
seam, not a permission â€” the blanket ask-first approval removes the asking, not
the missing judgment. The 18-point thrust bonus therefore keeps its full-day
baseline as a known, accepted pre-close gap, characterized by
`tests/fixtures/r3_swing_quality_v1.json`.

### Wishlist: built vs stubbed-with-a-question

**Built (1):** deep-link a symbol into an external charting tool.

**Stubbed with the blocking question stated (12), in
`docs/WISHLIST_OPEN_QUESTIONS.md`:** voice dictation (local vs cloud speech, and
what happens to a bad transcription); chart line-density presets (blocked on
P1.2's clutter budget â€” a desk-evidence decision, not a preference); read-only
mobile/web dashboard (who may read it, from where); self-hosted ntfy (is the
operational burden worth it); macOS scheduled jobs (will a Mac ever be the
unattended host); broader strength-board universe (explicitly gated on the R2
board proving itself); and the six research/data captures, which share one rule â€”
each needs a **registered consumer** before capture is justified.

Nothing on that list was implemented, and nothing was promoted into `plan.md`
except the one item that was built.

### The gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3760 passed / 19 subtests, 0 failed, 0 errors, exit 0 (2026-08-18 21:56 PT)** |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| Source selftest | **56/56** (55 â†’ 56: `external_chart_links` joined the roster) |
| Clean-cache frozen rebuild + frozen selftest | **`selftest OK: 56/56 checks passed (frozen)`, exit 0. `build/` AND `dist/` deleted first and built from the worktree, so the desk's own `dist/` was never touched; exe mtime 22:02 postdates the commit at 22:00** |

**The teardown crash is gone.** The `0xC0000409` native crash at interpreter
shutdown that the testing-week checkpoint says must be quoted alongside the
summary line **did not occur in any run today** â€” every full-suite run on this
branch exited 0. I did not fix it and cannot claim it is fixed; I can only
report that it stopped reproducing on this tree. If it returns, quote the
summary line and the exit code together, as that entry says.

### Live proofs now owed â€” the full ledger

**Nothing below has been observed.** UNKNOWN is a result and `plan.md` sec 6
requires recording it as one.

Inherited (8, unchanged except where the 08-17/08-18 AWAY sessions closed one):

- **R1 (3 open):** an EVENING day that stops after its early block; one SPY Â±1%
  alarm; the AWAYâ†’DESK **drain on return** (the trader never flipped back). The
  quiet boot PASSED 2026-08-16; AWAY staging-without-adoption PASSED both days.
- **R2 (3 open):** one adoption-time refusal; one scoped "Not today" that leaves
  other entries intact; one strength-board session matching the TC2000 scan's
  character. The eviction proof PASSED 2026-08-18.
- **R3 (3):** the `would_demote` shadow week **before any row moves**; the
  one-week 12:45-vs-close and STABLE-vs-PREVIEW churn comparison; the first
  real-data curation cycle.
- **R4:** the whole Â§8 exit gate.
- **R6:** the stall-watchdog diagnostic week.
- **R7:** the trader-present finale â€” dry-run migration report, live migration,
  full backfill, â‰¥10-trade statement audit, one clean reconciliation week, â‰¥5
  consecutive nightly ledger entries.
- **R8:** one real weekend run (does not wait for Monday).

Added by today's work:

- **R5, per engine:** one desk session confirming the **LRSI cross** volume is
  sane; then the same for the **confluence**; then for the **first-candle ORB**.
  Each session also decides whether that engine's toggle should default on.
- **R5 Â§4:** one observed any-bounce firing that names the level that held, and
  one re-arm after it.
- **R5 Â§8.3:** one scan whose zone-arms file actually carries `prev_avwape` for a
  symbol with a prior anchor (deterministic tests prove the shape, not the feed).
- **R7 visuals:** one look at the Analytics group chart and the year heatmap on
  real data, to confirm the thin-sample labels and the blank (untraded) days read
  the way you expect.
- **R8 joins:** one weekend where the focus-review table shows a pick whose
  horizons are still maturing, so the blank-not-zero rule is seen rather than
  trusted.

### Known weak spots â€” where I would look first if something misbehaves

1. **The four new alert types are OFF.** If you turn one on and the feed floods,
   that is the volume question Â§7 was written to ask â€” turn it back off and tell
   me the count; do not tune thresholds from one session.
2. **`_any_bounce_levels_for` reads `bot.d1_zone_arms`.** If the any-bounce watch
   never fires on D1 levels, check that the running BounceBot actually has that
   dict loaded; the session/H1 EMAs will still work without it, which is the
   design (a missing level is absent, never fabricated), but it looks like
   silence.
3. **The H1 15EMA needs 15 completed hours.** Early in a session it is legitimately
   absent, so an any-bounce watch armed at the open watches fewer levels than one
   armed in the afternoon.
4. **The confluence needs all three legs inside 4 bars.** If it never fires, that
   is far more likely to be the window than a bug; it is a parameter
   (`CONFLUENCE_WINDOW_BARS`) and the desk session is where it gets tuned.
5. **The Analytics group chart excludes buckets with no convertible total.** If
   the chart looks emptier than the table, read the note under it â€” it says how
   many were excluded and why.
6. **The Fed-calendar and universe stubs are conftest-side.** If a future test
   genuinely needs the real fetch, take back the stashed original the way
   `tests/test_universe_builder.py` now does, rather than removing the guard.

### Putting this build on the desk, when you choose to

Do these in order, on the main checkout, with the market closed:

1. **Disarm the scheduled task first.** Nothing else in this list is safe while
   it can fire: `Get-ScheduledTask -TaskName '<the launch task>' | Disable-ScheduledTask`.
2. Confirm the desk app is closed, then in `C:\Users\Aaron\TradingBotV3`:
   `git fetch origin` then `git checkout phase05-integration-blitz`.
3. Re-run the gate on the checkout you will actually run:
   `.venv\Scripts\python.exe -m pytest tests/ -q` (expect the figure above,
   exit 0) and `.venv\Scripts\python.exe scripts/smoke_check.py` (7/7).
4. Rebuild the exe if you run the frozen desk:
   `.venv\Scripts\pyinstaller.exe .\packaging\tradingbotv3.spec --noconfirm`,
   then `dist\TradingBotV3\TradingBotV3.exe --selftest` â€” expect
   `selftest OK: 56/56 checks passed (frozen)`, exit 0.
5. Launch once by hand and confirm the desk opens on Main, Auto mode reads what
   you left it at, and the Focus tab looks normal.
6. **Re-arm the scheduled task**: `Enable-ScheduledTask`.

To back out at any point: `git checkout phase05-r2-focus-gating-strength-board`,
rebuild, re-arm. Nothing in this branch writes to `C:\TradingBotData` differently
from the branch you are on now, and no store schema changed â€” the new
`any_bounce_watches.json` is created on first use and its absence is normal.

### What I did NOT do

- **No merge to `main`.** As instructed.
- **No live migration or backfill** (R7's trader-present finale) â€” built and
  tested against fixtures only, still behind their manual actions.
- **No threshold tuning** from any single session, and no live proof claimed
  from a deterministic test.
- **No changes to the desk's scheduled tasks, settings, or `C:\TradingBotData`.**


---

## THE WEEKEND OF 2026-08-15/16 â€” history, superseded by the report above

**The two stray remote branches are known and deliberately NOT merged
(trader decision, 2026-08-17).** A branch audit that day found exactly two refs
carrying commits absent from the release candidate, and both were ruled out:

- `scoring-flagging-evidence-guardrails` â€” one commit from 2026-08-03,
  "Tighten setup flags and add evidence boosts" (13 files, 704 insertions,
  `master_avwap` setup scoring/flagging plus a golden fixture). **Ignored by
  explicit trader decision.** It predates the consolidation, has never run
  alongside any of R1-R8, and merging a scoring change into a release candidate
  awaiting live validation would make the validation unreadable - a behaviour
  change could no longer be attributed. The branch is left in place, not
  deleted: ignoring is not discarding, and the work is still reachable if it is
  ever wanted.
- `claude/trading-system-review-e0p8ll` â€” one doc commit from 2026-08-09,
  `CONSOLIDATION_PLAN_2026-08-09.md`, describing a consolidation that has since
  actually happened. Superseded; no action.

Do not re-raise either as an open merge question. If a future audit wants to
revisit the scoring branch, that is a fresh trader decision, not a cleanup task.

**FIXED 2026-08-18 - the suite is hermetic.** The deterministic suite was only
ever ACCIDENTALLY offline: evening runs looked clean because R1's quiet-hours
gate was closed, and a market-hours run connected to IB, rebuilt a
1,536-symbol universe and pulled live SPY quotes. Mechanism of the fix: a
conftest tripwire refuses `socket.connect`/`connect_ex`/`create_connection` for
any test not marked `network`/`broker` (loopback included, because TWS is on
127.0.0.1:7496), records attempts so background-thread reaches fail at teardown
too, and names the calling frame; then eight external adapters are stubbed at
their boundaries - `ibapi EClient.connect`, ForexFactory, Treasury, yfinance,
the NASDAQ earnings fetch, and all five `universe_builder.fetch_*` entry
points. No product file was edited. The four Desk Link files are marked
`network` (a wire protocol whose transport IS the subject). Full suite in the
OPEN window, 11:19 PT: **3638 passed / 19 subtests, 0 failed, 0 errors**, and
no socket left the process.

**STILL OPEN, newly attributed, and it blocks a clean gate:** the pytest
PROCESS exits `0xC0000409` (STATUS_STACK_BUFFER_OVERRUN) - a native crash at
interpreter shutdown, AFTER every test has passed and the summary has printed.
It is not a test failure and it is not the hermetic work: `--ignore
tests/test_ui_stall_watchdog.py` returns **0** with 3633 passed, while ignoring
other Qt-heavy files (`test_qt_desk_layout`, `test_qt_journal_panel`) still
crashes. Deselecting only that file's two subprocess tests still crashes, so
the trigger is **importing `ui.stall_watchdog` into the shared process**, not
yesterday's subprocess isolation. `scripts/ui/stall_watchdog.py` imports
`PySide6.QtCore` at module scope and is product code owed R6(c)'s diagnostic
week - so this is NOT to be "fixed" by editing it to make a suite exit cleanly.
Until it is resolved, quote the summary line AND this exit code together;
neither alone is the truth.

**Branch renamed 2026-08-17: `phase05-r8-weekend-prep` â†’ `testing-week-2026-08-17`.**
Same commits, same SHAs, nothing merged or rebased â€” only the name moved, and the
old remote ref is deleted so there is exactly one name for one lineage. The old
name had stopped being true: it carried R1, R1.1, R2, R3, R4, R5, R7 and R8, not
a weekend-prep packet. Older documents and commit messages still say
`phase05-r8-weekend-prep`; those are accurate history, not stale pointers.

**One branch now carries everything.** Before this weekend the work lived on
`testing-week-2026-08-10` plus a ladder of packet branches. It is all collapsed
into **`testing-week-2026-08-17`**, which is **208 commits ahead of `main`** and
has both `main` and `phase05-r2-focus-gating-strength-board` as proven ancestors
(`git merge-base --is-ancestor` = 0 for each). Every deleted branch was verified
fully contained first; every rollback SHA is still reachable. There is exactly
**one thing to merge**, and it has no known conflict.

```
main (7d85a27)
  â””â”€ testing-week â†’ R1 â†’ R1.1 â†’ R2 (8d25c92) â†’ R7 â†’ R8 â†’ [R3, R4, R5 this weekend]
       = testing-week-2026-08-17          â† the single release candidate
```

**What got built, packet by packet.** 64 commits since the R2 tip; 14 of them in
the final session.

| Packet | State | One-line summary |
|---|---|---|
| **R1 + R1.1** | BUILT, 4 live proofs owed | OFF/DESK/AWAY/EVENING matrix, one fail-open quiet-hours gate over every automatic starter, EVENING SPY wake alarm |
| **R2** | BUILT, 4 live proofs owed | PDH+VWAP Focus adoption gate at build/refresh/adoption, provenance sidecar, scoped "Not today", M5 strength board |
| **R3** | **CLOSED 2026-08-16** | Shadow-only `would_demote` classifier, relvol + daytrade annotation, reviewed-today badge, 12:45 preview slot, post-close tracker write, STABLE+PREVIEW, structured dislike codes. **Â§4.3.5 volume-thrust deferred by trader decision** |
| **R4** | **BUILT 2026-08-16** | CaptureRail on every chart surface, painted armed alerts, forming-bar honesty fix, reviewed-today markers, labeled Likeâ†’Focus, feed repetition + open-burst digest |
| **R5** | **Â§2 + Â§5 + Â§3.1 built; Â§3.2/Â§3.3 behind a LIVE gate** | Three pure indicator modules, the one shared completed-bars rule, and the **LRSI cross engine wired live 2026-08-17** with its own `M5_SIGNAL_TAG` family and toggle map. The packaging trigger fired and was discharged: frozen count 51 â†’ 55 |
| **R6** | **(a) BUILT, (b) DECIDED + narrowed, (c) already existed** | AI batch layer now has a System Health row; rotation declined on measurement (see the R6(b) decision row); the stall watchdog was already built and owes only its diagnostic week. Item (d) already resolved into R7 |
| **R7** | BUILT, trader-present finale owed | Tax-grade journal from both brokers, rebuilt Journal tab |
| **R8** | BUILT, one weekend run owed | Guided weekend prep routine, H1/D1/M1 strength boards |

**Gates on the current tip** â€” all exit 0, all re-run this weekend:

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3638 passed / 19 subtests, 0 failed, 0 errors** (2026-08-18 11:19 PT, quiet-hours window OPEN, hermetic). **Process exit code is `0xC0000409`, not 0** - a native Qt-teardown crash after the summary prints, attributed to importing `ui.stall_watchdog`; see the entry above. The 3638/exit-0 row previously dated 2026-08-18 was written while its confirmation run was in flight: that run did land with 3638 passed, but its exit-code line never wrote, and the code was almost certainly already this crash. **Do not quote exit 0 for this suite until the teardown crash is resolved.** |
| `scripts/smoke_check.py` | **7/7** |
| clean-cache frozen rebuild + selftest | **`selftest OK: 55/55 checks passed (frozen)`** (2026-08-17; 51 â†’ 55 on the R5 roster growth) |

The frozen rebuild deleted `build/` **and** `dist/` first and ran from the
worktree, so the desk's own `dist/` was never touched. Exe mtime **22:08:48**
postdates the last code commit â€” provenance stated on its face, because a past
round shipped an exe built 21 seconds *before* its tip and an external review
correctly refused it.

**The count moved 49 â†’ 51, and that movement is the point.** The first rebuild
of the evening was taken at `6d81492`, then `7d97904` added `completed_bars.py`
and made `weekend_strength` reach it through a **function-level import** â€” the
exact shape PyInstaller can follow today and a refactor can quietly break, whose
failure mode is a bundle that starts fine and dies the first time a weekend
board filters a forming bar. `completed_bars` and `alert_repetition` were added
to `selftest.LAZY_ENGINE_MODULES` so the frozen run *proves* they import instead
of inferring it, and the count moving from 49 to 51 is what shows the rebuild
was real rather than a cached reuse.

`indicators.*` is deliberately **not** in the roster: it has no importer
anywhere and is listed in `PACKAGES_NOT_IN_THE_BUNDLE`, so the frozen exe
genuinely does not contain it. When R5's wiring gives it a real importer, that
entry is removed and its modules are added to the roster **in the same commit** â€”
the two lists must never contradict each other.

**Nothing live was touched all weekend.** No broker call, no journal write, no
desk-branch switch, no `main` push. The desk kept running
`phase05-r2-focus-gating-strength-board` from the main checkout throughout.

**Six trader decisions were taken and are recorded, not re-litigable:**

1. R3 Â§4.3.5 volume-thrust normalization â€” **deferred** (no intraday seam; a
   flat-profile proration was offered and rejected).
2. R4 open-burst digest window â€” **30 minutes**, zero disables.
3. R4 like-to-Focus â€” **one click**, no reason prompt.
4. R4 escalation list â€” **exhaustive at three**: higher tier, first BANGER,
   first PROVEN.
5. R5 confluence scope â€” **M5 Focus members only**.
6. R5 ORB candidate surface â€” **Alert Center annotation**, not a board lane.

Plus delegated to Fable and recorded (the trader may override any of them):
R5 gets a **new `M5_SIGNAL_TAG` family**, no tier bypass, foldable (spec Â§8.1);
**R5 Â§7 holds the WIRING of Â§3.2/Â§3.3, not their pure logic** â€” no wiring into
the live M5 loop even default-OFF until Â§3.1's desk session, pure
correlator/ORB-classifier code with fixtures may land now, and note nothing in
the UI can flip `m5_signal_toggles` anyway (spec Â§8.2, 2026-08-17); the
**prior-anchor AVWAP line is carried as an optional top-level `prev_avwape`
key** on the existing zone-arms entry â€” never a `trigger_levels` arm, absent
when no prior anchor, golden fixture over `build_d1_zone_arms` first, and the
value already exists at `runner.py:747` so no new band computation and no
`master_avwap_lib/legacy.py` edit at all (spec Â§8.3, 2026-08-17).

**The three held-ask-first items are TRIAGED 2026-08-17 (Fable, delegated) â€”
none needs a trader question:**

- the Focus Picks reviewed-today marker â€” **technical, decided: decoration
  only**, never in the document text, save-path byte-identity pinned by test;
  the only path back to the trader is if decoration proves impossible (R4 spec
  header note);
- R4 Â§2.2's `review_host` for the boards â€” **CLOSED, no build**: auto-advance
  on a re-ranking board advances to the wrong symbol; reopen only on a trader
  ask for a frozen review-queue mode (R4 spec header note);
- the completed-bars migration â€” **verified NOT a live bug**: every checked
  site strips the offset only *after* `get_market_local_now()` has converted
  to market time, so naive market-local compares against naive market-local
  and the answers are correct today. Migration stays opportunistic hygiene â€”
  it rides along with the next authorized `legacy.py` wiring edit behind an
  old-vs-new equivalence pin, and never opens that ask-first file on its own
  (R5 spec Â§5 note).

**â†’ Next session: see RESUME HERE in the table below.**

---

## Active work â€” read this before choosing a task

There may be only one active build item unless `plan.md` explicitly identifies an
elapsed evidence lane that can run in parallel.

| Field | Current value |
|---|---|
| Roadmap phase | **Phase 0.5 â€” R5 in progress, R6 and the review-deferral completions still to come.** R3 CLOSED and R4 BUILT 2026-08-16. R1 + R1.1 + R2 + R3 + R4 + R7 + R8 built; every live gate remains owed |
| **Active packet** | **R5 M5 signal engines** (`docs/M5_SIGNAL_ENGINES_PLAN.md`). **Â§2's three pure indicator modules and Â§5's shared completed-bars helper are BUILT and green.** The lane question that blocked the wiring is **ANSWERED** â€” spec Â§8.1: one new `M5_SIGNAL_TAG` family, main feed, **no tier-gate bypass**, not loud by default where the spec does not say, and **not** privileged against R4 Â§6.3 (foldable and digest-eligible). Per-engine identity rides `bounce_type`, not the tag. Â§9 carries build state and the packaging rules |
| **RESUME HERE** | **R5 Â§3.1 (LRSI cross) is WIRED, green and frozen-verified as of 2026-08-17 â€” see below. The next two engines are blocked on a LIVE gate, not on build effort.** **1. Â§7's per-engine desk session**: the confluence (Â§3.2) and first-candle ORB (Â§3.3) engines wire ONLY after one desk session confirms the LRSI cross's alert volume is sane. Do not wire them from deterministic tests. **2. R5 Â§4's any-bounce watch** is not behind that gate and can build next, but its prior-anchor AVWAP line is an **ask-first** edit to `master_avwap_lib/legacy.py` (D1 scan output) â€” ask before touching it. **3. R6 â€” (a) BUILT 2026-08-17, (c) was already built, (b) DECIDED 2026-08-17 and narrowed to tests/docs.** Rotation is declined on measurement; do **not** re-propose it without a reopen trigger from the decision row below. What R6(b) still owes: **(1) the replay characterization fixture is BUILT 2026-08-17** â€” `tests/fixtures/technical_integrity_replay_v1.json` + `tests/test_technical_integrity_replay.py`, 18 tests, mutation-proven (session filter removed â†’ 7 fail; provenance strip removed â†’ 3 fail), `scripts/technical_integrity.py` untouched. **(2) the read-only JSONL-ledger audit** via the existing footprint check; **(3) the stale-size comment fix** in `operations_audit.py` (~106 MB was a mid-July docstring, never a measurement â€” the audit measures live). The fixture and the audit touch tests and docs only; ask-first still binds any `technical_integrity.py` edit. Both stale sizes are resolved: measured **370 MB / 318,040 rows / 25 sessions** on 2026-08-17. **4. The review-deferral completions**: R8 Week-in-Review RRS-extremes join, R8 Focus Review joins (**join picks WITH their outcomes, not as separate rows**), Analytics per-setup/per-account charts with honest n counts and a CSV under each, and the Calendar pyqtgraph year heatmap. **Leave true USD conversion deferred** â€” the FX table books CAD only |
| **R4 close-out (2026-08-16)** | **BUILT, live proofs owed.** Â§6.1 armed-alert survival; CaptureRail in the snapshot popup and Alert Center pane (so the RS/RW and Industry boards, which had no capture at all, now inherit it); Alert Center LIKE as capture-not-placement; armed price alerts + D1 level watches painted as a read-only `GROUP_ALERTS` family on the worker; the Yahoo forming-bar early print suppressed 15 min after the open and labeled when drawn; the reviewed-today marker on snapshot/Alert pane/RS-RW/Industry; the labeled `â˜† Like â†’ M5 Focus` verb; and one feed row per symbol+side+day with a three-item escalation list and a 30-minute open-burst digest. Three trader confirmations recorded in the spec's Â§6.4. **Held ask-first:** the Focus Picks marker (editable watchlist *text*, not a table) and Â§2.2's `review_host` for the boards. **Owed:** the whole Â§8 exit gate, all live |
| **R3 close-out (2026-08-16)** | **DETERMINISTIC WORK COMPLETE.** The classifier stays shadow-only â€” `would_demote` stamps, nothing moves, hides or reorders a live row. Built: relvol + `daytrade_candidate` annotation, reviewed-today badge from recorded decisions only, the 12:45 PT preview slot, actual-close ownership of the single scheduled tracker write, STABLE+PREVIEW with `bar_status` stamps, and structured dislike codes counted as `review_learning`'s `dislike_reason` dimension. **Â§4.3.5 volume-thrust normalization is DEFERRED by explicit trader decision** â€” the D1 seam has no intraday slot series, a per-symbol 5-min fetch was refused as a data-budget/contract change, and a session-elapsed proration was offered and rejected because real volume is U-shaped. **Owed, live only:** the `would_demote` shadow week (required before any row moves), the one-week 12:45-vs-close and STABLE-vs-PREVIEW churn comparison, and the first real-data curation cycle |
| **Working branch** | **`testing-week-2026-08-17`** â€” since the 2026-08-15 consolidation this is **THE single release candidate**, carrying testing-week + R1 + R1.1 + R2 + R7 + R8 and every review-repair pass. It was cut from the R7 tip `4420bbf`, which was cut from the R2 tip `8d25c92`; the one R2 commit made after that cut (`fc4bcaf`) is now **merged in**, so `phase05-r2-focus-gating-strength-board` is a proven ancestor (`git merge-base --is-ancestor` = 0). Built in a linked worktree at `..\TradingBotV3-r8`. **The main `C:\Users\Aaron\TradingBotV3` checkout stays on the R2 branch, because the desk's scheduled task runs the desk from it.** Run tests with the main repo's venv python and the worktree as cwd |
| Desk branch | **`phase05-r2-focus-gating-strength-board`** at `fc4bcaf` â€” what the desk runs and what Monday's live proofs are observed against. It is kept **only until the Monday merge**; do not switch, rename or delete it before the scheduled task is disarmed |
| Scope | R5 Â§6 fences its files: `bounce_bot_lib/legacy.py`, `chart_watch.py`, `master_avwap_lib/d1_zone_arms.py`, `master_avwap_lib/legacy.py` (prior-anchor output), `alert_center_panel.py`, `bounce_service.py`. Edits outside the files the active packet's spec names are **ask-first**, fixtures first on anything detector/scoring/alert adjacent â€” the recovered-rule detour proved that pattern works. **Never edit `scripts/strength_scan.py`** |
| State | **3590 passed / 19 subtests, exit 0; smoke 7/7, exit 0; clean-cache frozen rebuild + `selftest OK: 55/55 checks passed (frozen)`, exit 0** (2026-08-17). **The packaging trigger finally fired and moved the count 51 â†’ 55**, which is the outcome the stale-build rule demands: R5 Â§3.1 gave `indicators` its first real importer, so it left `PACKAGES_NOT_IN_THE_BUNDLE`, entered the spec's `FIRST_PARTY_PACKAGES`, and four modules joined the selftest roster â€” all in one commit, the two lists still disjoint. `build/` **and** `dist/` were deleted first and the build ran from the worktree, so the desk's own `dist/` was never touched; exe mtime 18:59:35 postdates the commit at 18:57:23. Main desk checkout and live runtime untouched; no live broker call, no live journal write |
| Next action | See **RESUME HERE** above. Do not claim any live proof from deterministic tests |
| Do not start yet | **Phases 1â€“7 remain NOT authorized.** Do not run R7's live migration/backfill before Monday's validation day passes; do not claim any live proof from deterministic tests |
| **Owed live gates â€” the full ledger** | Nothing below has been observed. UNKNOWN is a result and `plan.md` sec 6 requires recording it as one. **R1 (4):** a ~21:00 boot that starts nothing; an EVENING day that stops after its early block; an AWAY session staging-not-adopting with a clean post-flip drain; one SPY Â±1% alarm. **R2 (4):** one staged pick evicted on a VWAP/PDH fallback; one adoption-time refusal; one scoped "Not today" leaving other entries intact; one strength-board session matching the TC2000 scan's character (re-measure the fetch during market hours). **R3 (3):** the `would_demote` shadow week **before any row moves**; the one-week 12:45-vs-close and STABLE-vs-PREVIEW churn comparison; the first real-data curation cycle. **R5 (1, and it blocks build work):** one desk session confirming the LRSI cross engine's alert volume is sane â€” Â§7's gate is per engine, so Â§3.2's confluence and Â§3.3's first-candle ORB stay unwired until it runs. **R4:** the whole Â§8 exit gate, including the two-direction Not-today/armed-watch check. **R6:** the stall-watchdog diagnostic week (the code was already built; only the week is owed). **R7:** the trader-present finale â€” dry-run review, migration, full backfill, â‰¥10-trade statement spot-audit, one clean reconciliation week, â‰¥5 consecutive nightly ledger entries. **R8:** one real weekend run (does **not** wait for Monday â€” read-only, starts nothing until a button is pressed) |
| **Live sessions 2026-08-17 + 2026-08-18 (merged in from the desk branch)** | Both days ran AWAY open-to-close on `c69b69c`. **R2 eviction PASSED** (four timestamped `Focus gate evicted N staged pick(s)` lines with per-symbol reasons), **R1 quiet boot PASSED with a note** (the `IB: connected` at 22:06:41 is `BouncePanel`'s launch auto-connect, not an Auto Pilot start), **R1 AWAY discipline HALF-PROVEN** (no DESK flip ever happened), and the other five proofs stay **UNKNOWN**. Two defects found and fixed on the desk branch and now merged here: an open report file aborting a whole swing scan (`_write_text_atomic` PermissionError), and one odd yfinance frame blanking the universe rebuild |
| Doc-only addendum (2026-08-15, late) | Phase 0.5 gained packets **R7 (journal reliability + UX)** and **R8 (Weekend Prep)**: specs written, WISHLIST/plan.md/docs README reconciled (incl. the P3.3 nightly-journal-pull promotion into R7 and the P5.4 narrowing). **Markdown-only â€” the release candidate, gates, and baseline above are unchanged** |
| **R6(b) decision (2026-08-17, delegated)** | Rotation of `technical_integrity_events.jsonl` is **declined for now** â€” measured 370 MB / 2.2 s boot re-parse, session-filtered replay makes closed sessions inert, and in-place rotation would break the warehouse ingest (SHA + line-offset) watermark; retention stays owned by the locked warehouse plan's after-verified-ingest cleanup, to be built as forward-only per-session segments with the monolith frozen. R6(b) narrows to the replay characterization fixture + read-only ledger audit. Recorded in `plan.md` item 6(b). **Markdown-only â€” the release candidate, gates, and baseline above are unchanged** |
| **R7 redirect (2026-08-15, second of the day)** | The trader explicitly authorized **R7 code to start now**, ahead of the P0.7 merge: branch **`phase05-r7-journal-reliability-ux` cut from the R2 tip** â€” same redirect pattern as R1/R2, recorded in `plan.md` Phase 0.5 preamble and the R7 spec header. Rationale: R7/R8 touch journal/weekend surfaces, not the scanning/alerting/Focus path Monday's proofs cover. **The desk keeps running the R2 branch via the scheduled task until the validation day passes â€” do not switch the desk branch without disarming that task.** R1/R2's eight live proofs remain owed and are inherited by the eventual stack merge. R7's own trader-present steps (live DB migration, full backfill) must NOT run on the desk before Monday's validation passes |
| **R3â€“R6 weekend redirect (2026-08-15)** | The trader explicitly authorized the remaining packets on this consolidated branch: *"integrate the rest â€” build R3 through R6 on the consolidated branch."* Build order is R3, R4, R5, R6, with per-packet governance and full-suite pushes. The redirect authorizes code; it does not discharge R3's shadow week or R6's watchdog week |

## R7 build progress â€” `docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md` Â§9

Each step is its own green commit, pushed. A step is not done until
`pytest tests/ -q` passes **by its own exit code**.

| Â§9 step | State | Evidence |
|---|---|---|
| 0 Characterization fixture | **DONE** | `tests/fixtures/journal_rebuild_trades_v1.json` + `tests/test_journal_characterization.py`; 2931 passed, exit 0 |
| 1 Hygiene (A10, B5, A4) | **DONE** | `tests/test_journal_import_hygiene.py` (34 tests); 2965 passed, exit 0 |
| 2 v3 migration + uid migration | **DONE** | `scripts/journal_migrate.py` + `tests/test_journal_migration.py` (26 tests); 2991 passed / smoke 7/7, exit 0 |
| 3 Group-key normalization | **DONE** | `scripts/journal_identity.py` + `tests/test_journal_identity.py` (34 tests); 3025 passed / smoke 7/7, exit 0. Golden regenerated with a note: 10 trades â†’ 9 |
| 4 Assembly changes | **DONE** | `tests/test_journal_assembly.py` (19 tests); 3044 passed, exit 0. Golden regenerated with a note: statuses and trade ids change, no P&L moves |
| 5 Adjustments API | **DONE** | `tests/test_journal_adjustments.py` (16 tests); 3060 passed / smoke 7/7, exit 0 |
| 6 Coverage ledger + partial persistence + self-heal | **DONE** | `scripts/journal_coverage.py` + `tests/test_journal_coverage.py` (21 tests); 3081 passed / smoke 7/7, exit 0 |
| 7 Activities + Flex OptionEAE/OpenPositions/CashTransactions | **DONE** | `tests/test_journal_cash_and_options.py` (25 tests); 3106 passed, exit 0 |
| 8 FX booking + analytics currency honesty | **DONE** | `scripts/journal_fx.py` + `tests/test_journal_fx.py` (15 tests); 3121 passed, exit 0. Golden regenerated with a note: the CAD trade books by identity |
| 9 Reconciliation against broker positions | **DONE** | `scripts/journal_reconcile.py` + `tests/test_journal_reconcile.py` (17 tests) |
| 10 Nightly `journal_import` JobSlot | **DONE** | `run_nightly_journal_import` + `tests/test_journal_nightly_slot.py` (11 tests); **3149 passed / smoke 7/7 / source selftest 31/31**, all exit 0 |
| 11 Shell + shared header + Trades tab | **DONE** | `ui/panels/journal/` package; `tests/test_journal_feed.py` (29) + `tests/test_qt_journal_panel.py` (25) |
| 12 Calendar + Analytics | **DONE to reconciled scope** | pyqtgraph equity curve, month grid, and walk-away in a worker; the year heatmap and additional analytics charts are explicitly deferred in the governing spec |
| 13 Health + Fees | **DONE** | coverage grid, reconciliation confirm flow, FX coverage, Flex/backfill controls (closes A1/A9); **3203 passed / smoke 7/7 / source selftest 31/31**, all exit 0 |
| **R8 Â§9 1-12** | **ALL DONE** | See the repaired R8 release candidate below: 3354 passed, smoke 7/7, frozen 49/49 |
| 14 Governance close-out | **DONE** | Frozen rebuild + `selftest OK: 45/45 (frozen)`, exit 0; CHANGELOG, `docs/README.md`, `WISHLIST.md`, `plan.md`, this file and `docs/DESK_TESTING_PLAN.md` reconciled |
| Pre-flight fix pass (2026-08-16) | **DONE** | Five trader spot-check findings closed; focused journal gate 145 passed; full suite 3375 passed / 19 subtests, all exit 0; live journal/brokers untouched |

### The R8 finale â€” one weekend, and it does not wait for Monday

Read-only against the trader's data, and the tab starts nothing until a button
is pressed. Spec Â§10: the desk boots on a weekend with the tab present and no
network activity until a press (log-verified); zero IB traffic across the
routine; H1/D1/M1 each refreshed with its wall clock recorded in the spec's Â§11;
the monthly board spot-checked for the absence of a current-month bar; one real
Adopt verified in Focus swing, `swinglongs.txt`, the membership file and
`pick_feedback.jsonl` with `origin="weekend_prep"`, and **nothing removed
anywhere**; one auto-tag confirm and one correction; a walk-away windowed to the
reviewed week; the week-ahead rendering only on its button press; the app closed
mid-routine and reopened with progress restored; and the trader confirming board
character per timeframe â€” until that, Â§5's filters are approved but not proven.

**R7 and R8 are built and their adversarial-review repairs are complete.** The
earlier R3â€“R6 hold is superseded by the trader's 2026-08-15 weekend redirect
recorded in the active-work table; R3 is now active. R8's Â§5 discovery filters
are trader-approved as proposed; the live weekend run still has to prove their
board character.

**The golden fixture is the packet's spine.** It freezes what `rebuild_trades`
does today, six known defects included, and it is regenerated only by
`tests/journal_characterization.py` with the change written into the fixture's
`intentional_difference` field in the same commit. It was verified to fail: a
trial `CLOSED_PARTIAL` status change turned three assertions red, and was
reverted.

**Step 1 finding â€” the ibapi timestamp gap is latent, not live.** The old parser
did not understand ibapi **10.x**'s `"20260804 09:31:00 US/Eastern"` execution
time and answered `pacific_now()` for it, which would have stamped every socket
fill with the import time. The desk is unaffected today: `constraints.txt` pins
**`ibapi==9.81.1.post1`**, whose `"20260804  09:31:00"` form the old parser did
read. So this is a defect that fires on an ibapi upgrade, not one already in the
live journal â€” recorded that way rather than as a live data-corruption finding.
Verified by running the pre-fix module directly against both spellings.

**Step 2 changed the golden once, on the record.** Schema v3 adds five columns
to every trade row (`net_pnl_cad`, `fx_rate`, `fx_rate_date`,
`reconcile_status`, `anchor_execution_uid`), all NULL or empty until steps 4, 8
and 9 populate them. **No assembled value moved**: legs, opportunity events and
the summary are byte-identical and every shared trade column matches, verified
column by column before regenerating. The note is in the fixture's
`intentional_difference` field, and the generator now **refuses to write a
changed golden without one**.

**Step 4's narrowing â€” APPROVED by the trader 2026-08-15, closed.** Â§5 fix 4
says a missing-opening-fill produces a `SYNTHETIC_OPEN` leg + `NEEDS_REVIEW`.
Built as: **only the unambiguous case is flagged** â€” a fill that closes more
than the journal knows is open, where the leftover is proof an opening fill is
missing. A plain sell with no position is *genuinely* ambiguous (a real short
entry, or a sale of shares bought before the import window), and nothing in the
execution distinguishes them; flagging every short would make the review queue
noise. That other half is caught by Â§9 step 9's reconciliation, where the broker
reporting flat against a journal that says short is the proof this step cannot
have. **This is a decided narrowing, not an open item** â€” do not re-litigate it
or "restore" the broader reading.

**The live journal DB has not been touched.** Everything above ran against
fixture and temporary databases. `journal_migrate.py` defaults to a dry run
against a throwaway copy, and a test asserts the live file is byte-identical
afterwards and that no backup is taken (because nothing changed). The real
migration is a trader-present step and waits for Monday.

### Broker credentials â€” DONE and live-verified (trader, 2026-08-15)

No longer waiting on the trader. Stored in machine-local settings and verified
**read-only**:

| Broker | State |
|---|---|
| IBKR Flex | `journal_ibkr_flex_token` / `journal_ibkr_flex_query_id` set. Verified: **372 trades**, 365-day window, **both accounts**, all four sections present (Trades, **OptionEAE**, **OpenPositions**, **CashTransactions**) |
| Questrade | Rotating-token chain stored and anchored on this desk. Auth OK; accounts **TFSA 51830546** and **Margin 29347316** |

Two standing constraints on this credential access, and they are not
negotiable while the migration is still owed:

- **Read-only against the live brokers, writes to fixture/temp DBs only.** Do
  not run any `journal_runner` path that writes the live store.
- **Do not trigger extra Questrade token refreshes.** The refresh chain is
  single-use rotating and anchored on this desk; every needless refresh risks
  breaking the trader's auth. Use only what a read-only verification needs.

### Tax status â€” partly decided (trader, 2026-08-15)

For the Â§9 step 11 labeling UI. The migration seeds `tax_status` from
`account_type` and never overwrites a `trader`-sourced value (I7):

| Account | Status |
|---|---|
| Questrade TFSA **51830546** | `TAX_FREE` |
| Questrade Margin **29347316** | `TAXABLE` |
| IBKR **U4867396** | `TAX_FREE` â€” TFSA, **currently unfunded and deliberately kept** |
| IBKR **U5102524** | `TAXABLE` â€” margin |

**All four confirmed by the trader 2026-08-15** and recorded in
`journal_migrate.TRADER_CONFIRMED_TAX_STATUS` as `tax_status_source='trader'`,
because a statement from the person who opened the account is a different kind
of fact from an inference off an account-type string â€” and only one of them may
never be overwritten (I7).

**U4867396 stays labeled while unfunded.** A zero balance is not zero history,
and an account that drops out of the tax grouping is an account whose past
trades quietly stop being counted.

An account nobody has decided about still stays blank and lands in the account
tree's own "Unlabeled" group. A guessed tax status is a wrong number in a tax
record.

**Deferred out of step 3, deliberately â€” one spec conflict.** Spec Â§5 fix 3 puts
"the manual-execution dialog gains real broker/account pickers" in this step,
but that dialog exists **only in the legacy Tk tab** (`scripts/journal_tab.py`),
which spec Â§7 says stays untouched â€” and the Qt panel has no manual-entry dialog
at all yet. The data layer already accepts a real broker/account
(`manual_execution_from_fields` honours them), so the missing half is purely
UI and belongs to the Qt Trades tab in **step 11**. Recorded rather than
silently skipped.

### Suite instability seen during R7 â€” READ BEFORE MONDAY

Two events, neither in a file R7 touches, both recorded because the merge gate
has **no rerun-until-green carve-out** and a 6am reader needs to know these
exist before deciding what a red run means.

| When | What | Reproduced? |
|---|---|---|
| During step 3 | One full run exited **3** â€” a crash, not a test failure | No. Next run green |
| During step 4 | `tests/test_desk_link_control.py::test_set_auto_mode_intent_round_trip_from_controller` **failed** | No. **1 failure in 10 full-suite runs** on this branch; 3/3 in isolation |

What is known: the Desk Link test drives a **real loopback TCP server** and
polls `_pump_until` against a **20-second wall-clock deadline**. Twenty seconds
is not an ordinary scheduling miss, which makes "just load" an unsatisfying
explanation â€” something stalled. `tests/conftest.py` already names the likely
family: leaked `bounce_bot_lib.legacy.run_strategy` worker threads that outlive
their tests, and its own honest verdict that "12/12 is a real improvement over
8/10 but it is not a proof of thread safety".

What is **not** known: whether R7 makes it more likely. R7 adds 123 tests and
~17s of runtime, which is more load on a load-sensitive test, so "R7 is
innocent" is a plausible claim and not a proven one. One full run at the R2 tip
was green â€” one run is not evidence of absence. No R7 file touches Qt, sockets,
or Desk Link.

**Context for Monday's gate decision, not a licence to ignore a failure:**
`tests/test_desk_link_control.py` guards **Desk Link, retired 2026-08-08**
(`CHANGELOG.md`) and kept in-repo only pending the P1.5 cleanup. Nothing the
desk runs today depends on it. That is worth knowing when weighing whether a
red run blocks the merge â€” it is *not* a reason to re-run until green, and the
flake stays **unattributed**.

**Do not treat either event as a known-flaky exemption on Monday.** P1.1 owns
suite hermeticity; if this recurs, it is worth a bounded investigation before
the merge rather than a re-run.

**Packaging, checked at the step-10 boundary.** The five new top-level modules
(`journal_identity`, `journal_migrate`, `journal_coverage`, `journal_fx`,
`journal_reconcile`) are **modules, not packages**, and every one is statically
reachable from the frozen entry point:
`ui/services/journal_import_service.py` â†’ `journal_runner` â†’ `journal_coverage`
/ `journal_fx` / `journal_reconcile` â†’ `journal_store` â†’ `journal_migrate` /
`journal_identity`. The spec-drift test passes and the **source** selftest
reports 31/31. **No packaging trigger fired**: no new third-party dependency
(`journal_fx` uses `requests`, already pinned), no new non-`.py` runtime asset,
no new top-level package, no dynamic string import, no `__file__`/`ROOT_DIR`
change. The **frozen** rebuild + frozen selftest are still owed before the merge
â€” CLAUDE.md requires them regardless of triggers, and they are the gate that has
historically caught what the suite could not.

**`ai_jobs` still is not in the frozen bundle**, and the new slot does not change
that: `default_slots()` imports `journal_runner` lazily inside the function, so
the roster/selftest disjointness rule is untouched.

### The R7 finale â€” trader-present, and all of it after Monday

Nothing below has happened. The build is complete; this is the part that needs
the trader and real data, in this order:

| # | Step | Note |
|---|---|---|
| 1 | **Read the migration dry-run report** â€” `python scripts/journal_migrate.py` (dry run is the default; it copies the DB to a temp file and leaves the live one byte-identical) | Look at the duplicate collapses and the annotation-orphan count before anything is applied |
| 2 | **Apply the migration** â€” open Journal and explicitly click **Prepare Journal database** | Runs backup, migration, and rebuild in a background worker. The tabs stay disabled and the status stays visible until it succeeds; this is when the four confirmed tax statuses land |
| 3 | **Full backfill** â€” Journal â–¸ Health â–¸ backfill, or `journal_runner --backfill-days 365` | Flex caps at 365 days; older history needs the one-time Flex file import (spec Â§8) |
| 4 | **Spot-audit â‰¥10 trades against statements**, then reconcile trade counts and commissions to **one monthly statement per broker, to the cent** | This is the gate that decides whether the journal is tax-grade |
| 5 | **One clean reconciliation week** on both brokers | Every mismatch fixed upstream or explained by an adjustment record |
| 6 | **â‰¥5 consecutive nightly `journal_import` ledger entries** with coverage advancing and at least one observed self-heal | |

**Questrade env-var cleanup**, if `QUESTRADE_REFRESH_TOKEN` is still set: local
settings win, but the env var is a first-boot seed only and a stale copy can be
mistaken for the live rotating token. The Health tab warns when it sees one.

**Nothing in R7's build touched the live journal database.** Every test ran
against fixture and temporary stores; `journal_migrate.py` defaults to a dry run
against a throwaway copy, and a test asserts the live file is byte-identical
afterwards.

## Live sessions 2026-08-17 and 2026-08-18 â€” what they proved, and two defects

Both days ran **AWAY from open to close**. That is the single fact that shapes
everything below: AWAY exercises staging, eviction, silent alert queueing and the
hourly phone reports, and it exercises **none** of adoption, "Not today", the
strength board, EVENING or the SPY alarm. Those did not fail â€” their triggering
conditions never occurred, so they stay **UNKNOWN**, which `plan.md` sec 6 counts
as a result.

### Proof results

| Proof | Result | Evidence |
|---|---|---|
| R2 eviction | **PASS** | `Focus gate evicted N staged pick(s)` in `trading_bot.log` / `trading_bot.log.1` on **2026-08-18 at 10:31, 11:40, 12:11 and 12:48**, each with per-symbol reasons â€” e.g. `Focus gate evicted 6 staged long pick(s): BMRN (not above yesterday's high and not above session VWAP), COO (not above session VWAP), DLTR (not above session VWAP), HLT (not above session VWAP), PAYC (not above session VWAP), SBH (not above session VWAP)`. Refusals at candidate build (`Focus gate refused N long candidate(s)`) appear in the same file and hour |
| R1 quiet boot | **PASS, with a note** (see below) | The 2026-08-16 22:06 launch logged `Auto Pilot is ON from saved state, but nothing starts yet - weekend - quiet hours until the next session` (autopilot.log 22:06:38) and nothing automatic ran until `Automatic work resumed - inside the 06:00-14:00 automatic-work window` at 2026-08-17 06:00:11 |
| R1 AWAY discipline | **HALF-PROVEN** | Two full sessions staged without adopting, and every hourly `Hourly Away swing report verified for HH:00` line is present on both days. **The flip-back-to-DESK half never ran** â€” the trader never flipped, so the R2.2 post-flip re-measurement is untested |
| R2 adoption refusal | **UNKNOWN** | AWAY never adopts, so `Focus gate refused N staged pick(s) at adoption` cannot appear |
| R2 scoped "Not today" | **UNKNOWN** | Needs an auto-adopted M5 entry; AWAY produced none |
| R2 strength board | **UNKNOWN** | Never opened during a session |
| R1 EVENING stop | **UNKNOWN** | No EVENING day ran |
| R1 SPY wake alarm | **UNKNOWN** | No EVENING day ran |

**No UNKNOWN above was upgraded.** A green suite does not move any of them, and
none may be written as `LIVE_VALIDATED` in `CHANGELOG.md` without its own
preserved evidence.

### The quiet-boot note: `IB: connected` at 22:06:41 â€” what it actually was

autopilot.log shows `IB: retrying` and then `IB: connected` at 2026-08-16
22:06:41, three seconds after the quiet-hours refusal. Answered from the code:

**It is neither of the two candidates.** It is not an Auto Pilot BounceBot start
â€” finding #1 of the R1 review has **not** regressed â€” and it is not a standalone
market-internals recorder, because none exists: the internals recorder lives
*inside* a running BounceBot (`bounce_bot_lib/legacy.py:8602-8625`), and the only
IB connect sites in the tree are `bounce_bot_lib/legacy.py:11535` and `:11637`
(inside `run_bot_with_gui`), `master_avwap_lib/legacy.py:1936` (the scan child)
and `journal_importers.py:412` (broker import).

It is a **third path**: `scripts/ui/panels/bounce_panel.py:280` runs
`QTimer.singleShot(0, self.start)` in `BouncePanel.__init__`, so the BounceBot
panel connects to IB on every launch at any hour, entirely outside Auto Pilot.
Its own tooltip says so (`bounce_panel.py:285`, "Auto-connects on launch").

Why that proves the R1 gate held rather than failed:

- An automatic start is **unconditionally announced**: `autopilot_service.py:576`
  logs `Starting BounceBot (IB connect + intraday scanning).` on every successful
  `service.start()`, and `_ensure_bot_running` (`autopilot_service.py:547-577`,
  the call at `:568`) is the only automated caller. That line is **absent** on
  08-16. The two previous real starts (08-14 12:48:31 and 08-14 23:32:15) both
  show it immediately *before* the same IB status pair â€” the contrast is the
  proof.
- The R1.1 fix is where it should be: the gate sits inside `_ensure_bot_running`
  (`autopilot_service.py:556-561`), not only at the boot resume
  (`autopilot_service.py:204-217`), so the 30-second tick cannot undo it.
- `IB: retrying` / `IB: connected` are emitted only by `bounce_service.py:865`
  and `:920`, both of which require an installed bot. `IB: connecting`
  (`bounce_service.py:403`) never reached the log because `_on_connection_changed`
  suppresses the first status when the previous one is `None`
  (`autopilot_service.py:2106-2117`).
- Nothing swept: a freshly started BounceBot begins with scanning disabled, and
  the window gate never enabled it â€” the same behaviour recorded for the
  2026-08-10 21:19 restart further down this file.

**Recorded, not fixed â€” the trader decides.** The desk connects to IB on every
launch regardless of hour. That is arguably right (a connection is cheap, and the
trader may want live charts at 22:00) but it contradicts the *wording* of the R1
quiet-hours proof row, which said "no IB connect". That row is now written
against what the build does. Making the panel's launch connect obey quiet hours
is a one-line change at `bounce_panel.py:280` plus a test; it is an R1 behaviour
change, so it waits for direction rather than riding along with a defect repair.

### Defect 1 â€” a reader holding a report open killed three whole swing scans

**Symptom.** `Swing scan for slot HH:MM FAILED: Master AVWAP scan process exited
with code 1.` on 2026-08-17 at 07:30 and 10:00, and 2026-08-18 at 12:00 (a
tracker-write slot), while neighbouring slots the same days succeeded.

**Root cause, with evidence.** All three run manifests record `"status":
"failed"`, `"error": "PermissionError(13, 'Access is denied')"`, and a phase list
ending at `output/signals` â€” the next phase, `output/reports`, never completed.
The surviving traceback (`trading_bot.log`, 2026-08-18 12:29:50) names the line:

```
File "scripts\master_avwap_lib\runner.py", line 2265, in _run_master_impl
    write_market_prep_files(market_prep_payload)
File "scripts\master_avwap_lib\legacy.py", line 21984, in write_market_prep_files
    _write_text_atomic(report_path, ...)
File "scripts\master_avwap_lib\legacy.py", line 2122, in _write_text_atomic
    os.replace(temp_path, path)
PermissionError: [WinError 5] Access is denied:
  'C:\TradingBotData\output\reports\.master_avwap_market_prep.txt.ifkokr1w.tmp'
  -> 'C:\TradingBotData\output\reports\master_avwap_market_prep.txt'
```

It is a **self-inflicted race**. Not a data or network fault: the failed 08-18
run's provider counters are normal (1,295 IBKR daily-bar successes against 1,293
on the successful 13:00 run). Not the frozen `-c` spawn class of 2026-08-13:
that failed one second in with exit code **2**, while these failed 8 to 30
minutes in with exit code 1, from a source-launched desk.
`write_market_prep_files` (`legacy.py:21978-21985`) writes the JSON first; the
desk's Market Prep panel watches that JSON with a `QFileSystemWatcher`
(`ui/panels/master_market_prep_panel.py:141-146`) and re-reads the **report
text** on the change (`:163` â†’ `ui/services/market_prep_feed.py:90-96`); and
Windows' `open()` does not grant FILE_SHARE_DELETE, so the `os.replace` landing
milliseconds later is denied. Reproduced directly on this desk: a plain read
handle on a destination file makes `os.replace` raise the identical
`[WinError 5] Access is denied`.

**Cost.** The whole scan died â€” tracker, reports, feature history, scan factors
and state â€” because one 60 KB report file was being read for a millisecond.

**Fix** (`c69b69c`; the trader approved the `legacy.py` edit before it was made):

1. `_write_text_atomic` and `_write_dataframe_csv_atomic` now replace through
   `_replace_with_retry` â€” ten attempts a tenth of a second apart. Same doctrine
   as `project_paths.SafeRotatingFileHandler`, which already tolerates a locked
   log file on rollover. A lock that outlives the budget still raises: a report
   that cannot be published must never be reported as published.
2. `ui/services/scan_service.py` lifts the child's own final exception line onto
   the **first** line of the `RuntimeError`, bounded to 240 characters, because
   `_on_scan_failed` writes only `detail.splitlines()[0]` to `autopilot.log`
   (`autopilot_service.py:1144`). The next occurrence reads `... exited with code
   1. PermissionError: [WinError 5] Access is denied: ...` instead of sending the
   reader to the run manifests and a log that may have rotated. No change to
   `autopilot_service.py` was needed â€” putting the cause on the first line was
   enough.

**Tests** (`tests/test_atomic_publish_under_reader_lock.py`,
`tests/test_scan_service_marker.py`): nine new, every one verified to fail
against the unfixed code, including a Windows-only reproduction that holds a real
read handle on the destination while the write runs.

**Not fixed, deliberately:** the panel still re-reads the report on every JSON
change, so the race can still *start*; the writer now survives it. Removing the
trigger as well is a UI change outside this pass.

### Defect 2 â€” one odd yfinance frame aborted the universe rebuild

**Symptom.** `Universe rebuild failed: "['datetime'] not in index"` (autopilot.log
2026-08-17 06:00:16). It self-healed on the ~60-minute retry â€” the universe was
rebuilt at 13:00 the same day â€” so the visible cost was a stale universe for one
session, which is exactly why it needed a test rather than a watch.

**Root cause.** `scripts/universe_builder.py:329` (pre-fix), the column selection
that ends `fetch_price_history`'s per-symbol loop. yfinance normally names the
daily index `Date`, so `reset_index()` yields a `Date` column the rename turns
into `datetime`; that chunk arrived with an **unnamed** index instead,
`reset_index()` produced `index`, and the selection raised pandas'
`KeyError: "['datetime'] not in index"` â€” the exact message the log carries. One
malformed sub-frame aborted the entire rebuild, while every other per-symbol
fault in that loop is skipped. The upstream response itself is not recoverable:
`trading_bot.log` has since rotated past 08-17.

**Fix** (`0d355b1`): the date axis is resolved by name (`Date` / `Datetime` /
`index` / `level_0`) and then by dtype; a frame with no usable date column is
skipped and counted rather than fatal, bounded to five warnings plus one total.

**And a floor under that fail-soft.** `build_universe` wrote
`universe_all/longs/shorts` unconditionally, so a fetch outage that priced
nothing would have overwritten a good universe with an empty file. `plan.md`
sec 5 â€” *a failed publish never destroys the last verified report* â€” so an empty
screen now raises; the caller already logs and retries in ~60 minutes, and the
previous universe stays authoritative until a rebuild succeeds. Trader approved.

**Tests** (`tests/test_universe_builder.py`): five new. The offending frame shape
was verified to fail against the unfixed code with the identical `KeyError`.

### Release candidate â€” 2026-08-18

Code changed, so this is a **new** release candidate and all three gates were
re-run against it.

| Check | Result | When |
|---|---|---|
| pytest | **2935 passed / 19 subtests**, exit 0 | 2026-08-18, on `c69b69c` |
| smoke | **7/7**, exit 0 | 2026-08-18, on `c69b69c` |
| frozen rebuild + selftest | **`selftest OK: 31/31 checks passed (frozen)`**, exit 0 | 2026-08-18, on `c69b69c` |

2921 â†’ 2935 is the fourteen new tests; no test was weakened or removed. The
frozen count stays 31: neither fix added a dependency, asset, package or dynamic
import, and the spec-drift test passes.

**Provenance, on its face:** last code commit `c69b69c` at **19:36:04**; last
commit of any kind `f2141c5` (Markdown only) at **20:00:01**;
`dist\TradingBotV3\TradingBotV3.exe` mtime **20:02:47** â€” the executable
postdates both. `build/` and `dist/` were **deleted before each of the two
builds**, so no cached module could have been reused.

The second build was not ceremony: `docs/DESK_TESTING_PLAN.md` is a **bundled
runtime asset** (the 31st selftest check exists because of it), and the doc pass
changed it after the first build. Rebuilding keeps the packaged Settings â–¸
Testing Plan page from rendering a superseded runbook. Both builds returned
`selftest OK: 31/31 checks passed (frozen)`, exit 0; the bundled copy at
`dist/TradingBotV3/_internal/docs/DESK_TESTING_PLAN.md` was confirmed to carry
the 2026-08-18 text.

### Next action â€” one DESK day and one EVENING night

Neither needs code. Both are written up for a human reader in
`docs/DESK_TESTING_PLAN.md`.

| Day | What it closes |
|---|---|
| One **DESK** session | R2 adoption refusal, scoped "Not today", the strength board's first real look â€” and the second half of AWAY discipline if the trader spends part of the day in AWAY and flips back |
| One **EVENING** night | EVENING stop (the early block runs, then each refused hourly slot is named once) and the SPY wake alarm |

The SPY alarm does not need a real Â±1% day: set `push_evening_spy_alarm_pct` low
for one night to force it, confirm one urgent push with repeats no sooner than
five minutes and silence after flipping out of EVENING, then **restore the
setting** â€” a forgotten low threshold wakes the trader on an ordinary move.

## Merge safeguards â€” read before Monday

### Repaired R7/R8 release candidate â€” code tip `dd201cd`

**`testing-week-2026-08-17` at code tip `dd201cd` is the repaired release
candidate.** The subsequent governance commit changes documentation only. All
three gates are green on the repaired tree:

| Check | Result |
|---|---|
| pytest | **3354 passed / 19 subtests**, exit 0 |
| smoke | **7/7**, exit 0 |
| frozen rebuild + selftest | **`selftest OK: 49/49 checks passed (frozen)`**, exit 0 |

`build/` **and** `dist/` were deleted before the clean rebuild, preserving the
stale-cache safeguard established during R7 close-out.

The adversarial review closed all A1â€“A19 and B1â€“B14 findings. Weekend board
snapshots now persist across restarts. The deliberately unshipped journal USD
conversion/year heatmap/additional analytics charts and Weekend RRS/Focus-review
joins are explicitly deferred in their governing specs rather than described as
implemented.

**Rollback for R8 alone: `4420bbf`** (the R7 tip). R8 is a strict superset, so
backing it out is a checkout.

**Step 1 is separately merge-worthy, and probably urgent.** `3c3c8e1` fixes a
**live crash**: on the branch the desk runs today, clicking **Settings** raises
`IndexError`, and eight nav titles from index 3 name the wrong page. It touches
only `ui/app.py` and two tests.

### ~~MERGE NOTE â€” expected conflict in this file~~ â€” ABSORBED 2026-08-15

The conflict is gone: `phase05-r2-focus-gating-strength-board` was merged into
this branch on 2026-08-15 rather than left for Monday morning. Monday's merge is
now **one** merge, not three, and it has no known conflict.

`fc4bcaf` (the R2 frozen-gate re-verification) was the only R2 commit outside
R7/R8 ancestry. It touches `CURRENT_CHECKPOINT.md` only; git auto-merged it
without a conflict, and **both** the R2 clean-cache re-verification note and the
R7/R8 sections are present above. The merge's whole contribution to this branch
is **7 inserted Markdown lines in this file** â€” verified with
`git diff --stat b154b8a HEAD`, no `.py` and no test touched.

**Gates re-run after the merge, in `..\TradingBotV3-r8`:**

| Check | Result |
|---|---|
| pytest | **3354 passed / 19 subtests**, exit 0 |
| smoke | **7/7**, exit 0 |
| frozen rebuild + selftest | **not re-run, deliberately** â€” the merge added Markdown only, so the frozen gate recorded on code tip `dd201cd` (**49/49 `(frozen)`**, exit 0, from a wiped `build/` and `dist/`) still describes this tree's code exactly |

### Branch consolidation â€” 2026-08-15

The repository was reduced to **three** branches so Monday has one thing to
merge. Everything deleted was proven fully contained first
(`git merge-base --is-ancestor <branch> testing-week-2026-08-17`); no commit was
lost, and every named rollback SHA is still reachable from this branch.

| Branch | Fate |
|---|---|
| `main` | kept â€” trunk, tip untouched at `7d85a27` |
| `phase05-r2-focus-gating-strength-board` | kept â€” the desk branch, until Monday |
| `testing-week-2026-08-17` | kept â€” **the** consolidated release candidate |
| `testing-week-2026-08-10`, `phase05-r1-auto-modes-quiet-hours`, `phase05-r7-journal-reliability-ux`, `testing`, `chart-review-workspace`, `chart-perf-c`, `integration-test`, `durability-catchup`, `local-ai-phase-0`, `local-ai-phase-1`, `repair-packet-2` | deleted, local and (where it still existed) on `origin` â€” all contained |

Worktrees `..\TradingBotV3-r7`, `..\TBV3-testing` and `..\TBV3-chart-review`
were confirmed clean and removed. `..\TradingBotV3-r8` and the main checkout are
the only two that remain.

**Three remote-only branches were deliberately NOT deleted** â€” each still holds
one commit that is in neither `main` nor this branch, so the trader decides:
`origin/scoring-flagging-evidence-guardrails` (`47a3e97` "Tighten setup flags and
add evidence boosts" â€” the only one of the three carrying code),
`origin/claude/trading-system-review-e0p8ll` (`18c9c93`) and
`origin/claude/wishlist-integration-analysis-2ixvy0` (`671ee57`). Two further
remote branches, `origin/claude/testing-production-blockers-oek3aj` and
`origin/claude/ticker-briefs-hardening-imcm8r`, **are** proven contained but
their deletion was refused by the tooling; they are safe to delete from the
GitHub UI at any time.

### R7 release candidate â€” `fe4fe73`

**`phase05-r7-journal-reliability-ux` at `fe4fe73` is a named release candidate**,
verified by all three gates on the tree that produced it:

| Check | Result | Command |
|---|---|---|
| pytest | **3203 passed / 19 subtests**, exit 0 | `.venv\Scripts\python.exe -m pytest tests/ -q` |
| smoke | **7/7**, exit 0 | `.venv\Scripts\python.exe scripts/smoke_check.py` |
| frozen rebuild + selftest | **`selftest OK: 45/45 checks passed (frozen)`**, exit 0 | `pyinstaller .\packaging\tradingbotv3.spec --noconfirm` then `dist\TradingBotV3\TradingBotV3.exe --selftest` |

**Rollback for R7 alone: `3339dd9`** â€” the step-10 tip, the last commit before
the Journal UI was rebuilt. Everything earlier in the stack keeps its own
rollback points in the table further down; the R7 branch is a strict superset of
the R2 tip `8d25c92`, so backing R7 out entirely is a checkout of that.

**The frozen build was made from the worktree** (`..\TradingBotV3-r7`), so its
`dist/` is the worktree's and the desk's own `dist/` â€” the R2 release candidate
it has been running â€” was never touched.

#### What the frozen run caught, and it is not nothing

Three rebuilds were needed. The first two reported **31/31**, the pre-existing
roster, passing with R7 code in the bundle. Extending
`selftest.LAZY_ENGINE_MODULES` by fourteen journal modules **did not change the
frozen count** until `build/` was deleted â€” a PyInstaller rebuild had silently
reused the cached module. That is exactly the failure shape that let "frozen
selftest 30/30" be recorded three times during R1/R2 for runs that never
happened. **Treat a frozen count that does not move after a roster change as a
stale build, not as a passing gate.** The clean rebuild reports 45/45, and `ui`
collects **117** submodules against 109 before â€” the new `ui/panels/journal/`
package.

### Previous release candidate (R1/R2)

> **Superseded 2026-08-18.** The current release candidate is `c69b69c` - see
> the 2026-08-18 section above for its gate figures and provenance. The table
> below is the 2026-08-15 R2.3 candidate, kept because the 08-17 and 08-18
> sessions' evidence belongs to *that* tree.

The 08-17 and 08-18 sessions ran **the tip of
`phase05-r2-focus-gating-strength-board` as of 2026-08-15**. The last
commit that changed code or tests before them is the R2.3 fix **"Give each return to the
desk an identity its timestamp cannot collide"** (`90ba0d4`, committed
2026-08-15 13:11:19 PT); everything after it until the 2026-08-18 defect repair
is documentation, so the running behaviour those two days exercised is exactly
the tree the three gates below were run against.

Stated that way on purpose: the SHA above is re-stated **only** because the
external provenance check needs commit time and executable mtime side by side.
**The rule is unchanged: if a commit changes code or tests, all three gates
re-run and this whole section is updated â€” a stale line here is worse than
none.**

| Check | Result | When |
|---|---|---|
| pytest | **2921 passed / 19 subtests**, exit 0 | 2026-08-15, after R2.3 |
| smoke | **7/7**, exit 0 | 2026-08-15, after R2.3 |
| frozen rebuild + selftest | **`selftest OK: 31/31 checks passed (frozen)`**, exit 0 | 2026-08-15, after R2.3 |

**Provenance, on its face:** last code commit `90ba0d4` at **13:11:19 PT**;
`dist\TradingBotV3\TradingBotV3.exe` mtime **13:13:54 PT** â€” the executable
postdates the last code commit. Commits after `90ba0d4` on this branch are
Markdown-only (verify with `git show --stat`); the R2.2 round's executable had
been built 21 seconds *before* its tip, which is why the ordering is now
recorded here explicitly rather than left derivable.

**Clean-cache re-verification (2026-08-15, evening).** R7's close-out discovered
that a PyInstaller rebuild can silently reuse cached modules (`build/` must be
wiped for a roster change to register). Because the R2 candidate's 31/31 above
predates that rule, the R2 checkout was re-verified with `build/` and `dist/`
deleted first: full rebuild from tip `8d25c92` (Markdown-only after `90ba0d4`),
frozen selftest **31/31, exit 0**. The gate stands on a clean build.

The R2.3 fix changed code, so it is a **new** release candidate and all three
gates were re-run against it â€” including the frozen rebuild, even though no
packaging trigger applied. The frozen one is never optional: it is the gate that
caught the `ai_jobs` roster clash and the `-c` scan-spawn defect when the suite
could not.

### Rollback points

**Read this first: the branch names below no longer exist.** The 2026-08-15
consolidation deleted them, but every SHA is still reachable from
`testing-week-2026-08-17`, so each remains a plain `git checkout <sha>` â€” a
detached checkout, not a revert. Nothing here got harder to roll back; the names
just stopped being branch heads.

| Point | SHA | What it is |
|---|---|---|
| Pre-everything | **`7d85a27`** | `main`. Last known-good merged trunk |
| Pre-R1 | **`e18757e`** | Former tip of `testing-week-2026-08-10`. The build that ran the desk before any Phase 0.5 work |
| Pre-R2 | `4389961` | Former tip of R1+R1.1, if only R2 needs backing out |
| Pre-R7 | `8d25c92` | The R2 tip R7 was cut from, if R7+R8 need backing out but R1/R2 do not |
| Pre-R8 | `4420bbf` | Former tip of `phase05-r7-journal-reliability-ux`, if only R8 needs backing out |
| Desk build | `fc4bcaf` | Tip of `phase05-r2-focus-gating-strength-board` â€” the build the desk runs and Monday's proofs are observed against. Still a live branch until the merge |

Ancestry is a single line with one merge at the end â€”
`main` â†’ `testing-week` â†’ R1 â†’ R2 (`8d25c92`) â†’ R7 â†’ R8, then `fc4bcaf` merged
in â€” so every row above is an ancestor of the consolidated tip.

**The rolled-back build reports `selftest OK: 30/30`, not 31/31, and that is
correct** â€” the 31st check is the one bundling `docs/DESK_TESTING_PLAN.md`, which
did not exist at `e18757e`. `docs/DESK_TESTING_PLAN.md` Â§3.4 now says so in plain
language, because a 6am reader watching the count drop would otherwise read a
successful rollback as a broken one.

### Rollback drill â€” EXECUTED 2026-08-15

Run once, unattended, with no desk process running:

| Step | Result |
|---|---|
| Disarm `TradingBotV3 0700 Launch` | `Ready` â†’ `Disabled` |
| Check out the pre-R1 rollback SHA `e18757e` | clean, no conflicts |
| Verify the rolled-back build starts | `selftest OK: 30/30 checks passed`, exit 0 (30 not 31: the testing-plan check did not exist at that SHA â€” the count moving is *correct*) |
| Return to the release candidate | back at `bf1ab89`, `selftest OK: 31/31` |
| Re-arm the launch task | `Disabled` â†’ `Ready` |

All three TradingBotV3 tasks confirmed `Ready` afterwards (`0700 Launch`,
`AI Jobs`, `Push cold data to DAS`).

**What the drill did NOT prove:** a full GUI launch. The selftest is the
designed proxy â€” it imports every lazily-loaded engine and loads every
`__file__`-relative asset with no window and no network â€” but it is not a
double-click. If the trader wants that certainty before Monday, one manual
launch at `e18757e` is the missing step; the mechanical path around it is
proven.

**The order matters and is the point:** disarm first. The launch task starts
the desk from source, so checking out another SHA while it is armed can have
the task launch a half-swapped tree.

### Live proofs are UNKNOWN until observed

Nothing in the tables below has been run on a live session. They are
**UNKNOWN**, and UNKNOWN is a result â€” `plan.md` sec 6 requires recording it as
such. A green test suite does not upgrade any of them, and none may be written
as PASS in `CHANGELOG.md` without preserved real-session evidence.

## Monday sequence â€” 2026-08-17

Do these in order. **Nothing merges until (a) and (b) both pass.**

**The trader can read all of this on the desk**: Settings â–¸ Testing Plan renders
`docs/DESK_TESTING_PLAN.md`, a plain-language version of the same sequence. That
file restates the proofs below for a human reader and **must be updated in the
same pass whenever they change**.

### (a) Run the live proofs on THIS build, during the real session

Both packets' proof tables are below â€” four for R1, four for R2. They are written
against the finished build, not against what either packet did mid-flight; the
AWAY proof in particular changed when R2 landed.

Two are already actionable outside the session: the R1 quiet-boot proof (a ~21:00
launch, which the trader is running the evening of 2026-08-15) and the R2 "Not
today" proof (needs an auto-adopted M5 entry, so it needs a session first).

Record every result, including UNKNOWNs, without rewriting the outcome
(`plan.md` sec 6).

### (b) Run the plan.md sec 6 first-session checklist

`docs/FIRST_SESSION_CHECKLIST.md`, which already carries the four R1 rows added
2026-08-15. It has **no R2 rows** â€” use the R2 proof table below alongside it
rather than assuming the checklist covers this build.

### (c) Only if both pass: P0.7 merges **one** branch into `main`

The 2026-08-15 consolidation replaced the three-branch ladder with a single
merge. There is no order to get wrong and no known conflict:

```
testing-week-2026-08-17  ->  main
```

That one branch carries testing-week + R1 + R1.1 + R2 + R7 + R8 and every
review-repair pass. The old ladder (`testing-week` â†’ R1 â†’ R2, each merged
separately) is gone along with those branch names; per-packet rollback is
preserved by SHA in the rollback-points table instead.

**Then, in this order â€” the desk is not switched until the gates pass on `main`:**

| # | Step | Note |
|---|---|---|
| 1 | Merge `testing-week-2026-08-17` into `main` | one merge, no expected conflict |
| 2 | Re-run **all** gates on `main`, including a **clean-cache** frozen rebuild | delete `build/` **and** `dist/` first â€” R7's close-out proved a rebuild silently reuses cached modules, and a frozen count that does not move after a roster change is a stale build, not a pass |
| 3 | Disarm `TradingBotV3 0700 Launch` | **before** touching the checkout â€” the task starts the desk from source and can launch a half-swapped tree |
| 4 | Switch the desk checkout to `main` | this is when `phase05-r2-focus-gating-strength-board` stops being needed |
| 5 | Re-arm `TradingBotV3 0700 Launch` | confirm all three tasks read `Ready` (`0700 Launch`, `AI Jobs`, `Push cold data to DAS`) |

**Gates to re-run at merge time, on `main` after the merge:**

| Gate | Command | Expected |
|---|---|---|
| Full suite | `.venv\Scripts\python.exe -m pytest tests/ -q` | **3370 passed / 19 subtests**, exit 0 â€” check pytest's own exit code, not a piped tail |
| Smoke | `.venv\Scripts\python.exe scripts/smoke_check.py` | **7/7**, exit 0 |
| Frozen rebuild | delete `build/` and `dist/`, then `.venv\Scripts\pyinstaller.exe .\packaging\tradingbotv3.spec --noconfirm` | exit 0, ~4 min unattended. **Required on `main` regardless of triggers** |
| Frozen selftest | `dist\TradingBotV3\TradingBotV3.exe --selftest` | **49/49**, exit 0, output ending `(frozen)` |

**R7 and R8's own live gates come AFTER this merge, not before it.** Nothing in
step (a) or (b) exercises them, and none of them is a merge blocker:

- **R8** â€” one real weekend run (spec Â§10). This one does not have to wait for
  Monday at all: it is read-only against the trader's data and starts nothing
  until a button is pressed.
- **R7** â€” the trader-present sequence in "The R7 finale" above, in order: read
  the migration dry-run, click **Prepare Journal database**, full backfill,
  the â‰¥10-trade statement spot-audit, one clean reconciliation week, and â‰¥5
  consecutive nightly `journal_import` entries. **None of it may start before
  Monday's validation day passes.**

**Subsequently authorized by the trader's 2026-08-15 weekend redirect:** R3,
R4, R5 and R6 now build on this consolidated branch in that order; the
active-work table owns their current state. Phases 1â€“7 remain open and are not
authorized this session.

**Is a packaging trigger pending? No â€” but rebuild anyway.** Checked all five
triggers across the whole stack (`e18757e..HEAD`): no new third-party dependency,
no new non-`.py` runtime asset, no new top-level *package* under `scripts/`
(`focus_adoption_gate.py` and `strength_scan.py` are modules, reached by static
analysis through eager imports; the two new UI files sit inside `scripts/ui`,
already collected), no new dynamic string import, and no `__file__`/`ROOT_DIR`/
`sys.path` change. The spec-drift test passes. **The rebuild is still required**
because CLAUDE.md mandates one before every merge to `main`, and because:

> **Correction, 2026-08-15:** every "frozen selftest 30/30" recorded for R1, R1.1
> and R2 was actually the **source** selftest (`launch_gui.py --selftest`, whose
> output carries no `(frozen)` suffix), against a `dist/` built 2026-08-13 that
> predated all three packets. **Resolved the same day â€” see the frozen rebuild
> below.** Re-run it at merge time only if code lands after that rebuild: this is
> the gate that has historically caught what the suite could not, finding the
> `ai_jobs` roster clash on 2026-08-09 and the `-c` scan-spawn defect on
> 2026-08-13.

### Frozen rebuild and REAL frozen selftest â€” 2026-08-15

Five rebuilds, all green. The first was the run three packets of notes had
mislabeled; the second was forced by the testing-plan asset; the third was the
R2.1 release candidate `bf1ab89`; the fourth was the R2.2 tip â€” built 21
seconds before its final commit, which the external review correctly refused as
provenance; the fifth is the current R2.3 candidate, built after `90ba0d4`.

| # | Time | Result |
|---|---|---|
| 1 | 09:58 | `selftest OK: **30/30** checks passed **(frozen)**`, exit 0 |
| 2 | 10:27 | `selftest OK: **31/31** checks passed **(frozen)**`, exit 0 |
| 3 | 11:0x | `selftest OK: **31/31** checks passed **(frozen)**`, exit 0 â€” on `bf1ab89` |
| 4 | 13:0x | `selftest OK: **31/31** checks passed **(frozen)**`, exit 0 â€” superseded: exe predated its tip by 21 s |
| 5 | 13:13 | `selftest OK: **31/31** checks passed **(frozen)**`, exit 0 â€” **current, after code commit `90ba0d4` (13:11:19)** |

Rebuilds 4 and 5 were run **without a packaging trigger**, because a code commit
makes a new release candidate and CLAUDE.md requires a rebuild before merging to
`main`. The count is unchanged at 31, which is the expected result: neither R2.2
nor R2.3 added a dependency, asset, package or dynamic import.

**31, not 30, and that is the point.** The Testing Plan tab renders
`docs/DESK_TESTING_PLAN.md`, a runtime asset that lives **outside `scripts/`**.
The spec's package-asset sweep only mirrors files inside `FIRST_PARTY_PACKAGES`,
and `test_packaging_spec_drift.py` only walks `scripts/` â€” so **neither would
have noticed it going missing**, and the frozen desk would have shipped showing
"plan file not found" on the one page the trader opens when nothing else is
behaving. Three things now guard it: an explicit `datas` rule with a hard
`SystemExit` if the file is absent at build time, a new selftest asset check
(the 31st), and a test asserting the spec rule still exists. Confirmed present
in the bundle at `dist/TradingBotV3/_internal/docs/DESK_TESTING_PLAN.md`.

That trigger is trigger 2 in the CLAUDE.md list ("new non-`.py` runtime asset"),
plus trigger 5 (`__file__`-relative resolution â€” the view resolves through
`sys._MEIPASS` when frozen, since a frozen build has no `scripts/` tree to walk
up from).

| Check | Result |
|---|---|
| `pyinstaller .\packaging\tradingbotv3.spec --noconfirm` | **exit 0** |
| `dist\TradingBotV3\TradingBotV3.exe --selftest` | **`selftest OK: 31/31 checks passed (frozen)`**, exit 0 |

The `(frozen)` suffix is the whole point: the source selftest prints the same
count without it, which is how three packets of notes recorded a run that had
never happened. Any future entry claiming a frozen result must quote the suffix.

What this build collected: `ui` 109 submodules, `bounce_bot_lib` 12,
`master_avwap_lib` 26, `market_prep` 23, `diagnostics` 6, `research_warehouse`
19, `desk_link` 7, `duckdb` 39, plus the three package assets
(`veto_reasons_v1.json`, `theme.qss`, `exploration_cohort.txt`). R2's two new
top-level modules (`focus_adoption_gate`, `strength_scan`) and its two new UI
modules are in the bundle and import cleanly under it â€” which is what this run
was needed to prove and no packaging-trigger analysis could.

The desk was running from source, so nothing had to be closed â€” and no desk
process was running at all for rebuild 4. `dist/` and `build/` are gitignored, so
this is verification only and never a commit artifact.

**This satisfies the frozen gate for the current tree** (rebuild 4, on the R2.2
tip). Re-run it at merge time only if code lands after that.

### ~~Known blocker for the merge gate~~ â€” FIXED 2026-08-15

`tests/test_warehouse_seal.py::test_stale_staged_files_are_quarantined_not_deleted`
no longer fails intermittently, and the merge gate has **no rerun-until-green
carve-out**. Any test failure on Monday is a real failure.

It was never flakiness. `reconcile` compared `st_mtime > cutoff` where
`cutoff = utc_now() - grace`, and Windows' system clock ticks about every
15.6 ms while NTFS stamps mtimes far more finely â€” so `utc_now()` could round
BELOW the mtime of a file written microseconds earlier, and that file read as
"from the future" and was never quarantined. The earlier "timing-sensitive
under suite load" note was wrong: load was never the variable, and it
reproduced in isolation at 3 failures in 6 runs.

Fixed in `store.py` with a 50 ms clock-granularity slack (trader-approved
before the edit; recorded as a warehouse build decision). Verified by 20
consecutive passes of the previously flaky test plus a new deterministic
reproducer that writes and reconciles back to back 25 times.

### R2.2 review pass â€” 2026-08-15 (four items from the final external review)

Four items, each its own green commit, plus one refinement of item 1 found while
reviewing it. Two changed behaviour, one is documentation with a test that keeps
it honest, one reconciled the desk runbook.

| # | What | Where |
|---|---|---|
| 1 | **The flip drain is explicitly locked.** The AWAY/EVENING â†’ DESK flip records its own moment; adoption refuses any verdict stamped before it (`pending_pick_gate_ok(..., not_before=)`). A failed re-verification now retries every 60 s, five times, instead of falling through to the ordinary stored-verdict drain â€” the 2-bar lag bound is defense in depth, no longer the only lock. Giving up after five is safe because the barrier holds and the 30-minute staging refresh stamps post-flip verdicts. A follow-up commit closed the DESK â†’ AWAY â†’ DESK mid-flight case: an attempt remembers which flip it answers, so a newer return is owed its own measurement rather than inheriting one whose bars predate it | `alert_center_panel.py`, `autopilot_core.py`, spec Â§11.1 |
| 2 | **One 14:00 boundary.** `auto_scanning_due` used an inclusive datetime endpoint, `_auto_work_due`'s fallback used `hour < 14`; at 14:00:00.000000 they disagreed. Both now call `within_auto_scanning_window` over `auto_quiet_hours_fallback_window`, inclusive at both ends. Test pins the exact microsecond at both call sites and was verified to fail against the old spelling | `autopilot_core.py`, `autopilot_service.py`, R1 spec Â§4 |
| 3 | **The two-bar tolerance is recorded as an accepted exposure**, with its backstop named: BounceBot's four-close triple-VWAP invalidation plus the desync repair removes a bad adoption within ~4 completed bars. A test pins both constants so the documented bound cannot quietly stop being true. No behaviour changed | `autopilot_core.py` comment, spec Â§11.2 |
| 4 | **The runbook stopped contradicting this file.** It claimed 31/31 at 09:58 where this file says 30/30 â€” the checkpoint was right, provable from the build: the only selftest change since `e18757e` is the testing-plan asset check added at 10:38, so the runbook was claiming its own bundling was verified before the file existed. Also removed its stale "known flaky test, just re-run it" carve-out and added the rollback section with the 30/30 explanation | `docs/DESK_TESTING_PLAN.md` |

**Not done, and deliberately:** item 3 offered `max_bar_lag = 1` as an
alternative. The trader's note left that as their call, so the accepted-exposure
documentation was built as written and the constant is unchanged. Switching it
later is a one-line change plus the golden-fixture update.

### R2 live proofs â€” one PASSED 2026-08-18, three still owed

From `docs/M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md` Â§8. **Eviction PASSED on
2026-08-18** with the log lines quoted in the 2026-08-18 section above; the other
three need a DESK day, which AWAY could not provide:

| Proof | What to look for |
|---|---|
| Eviction â€” **PASSED 2026-08-18** | One staged pick evicted for falling back through VWAP or the previous-day extreme: `Focus gate evicted N staged long pick(s): SYM (not above session VWAP)` in the Auto Pilot log. Silent on the desk by design â€” the log is the record |
| Adoption refusal | One pick refused at adoption, in `trading_bot.log`: `Focus gate refused N staged pick(s) at adoption`. A verdict older than 45 min reads `gate check is NN min old` |
| Scoped "Not today" | On an auto-adopted M5 entry the button reads `âœ• Not today - drop pick` and removes only that entry; the trader's own picks, the swing list and the other side are all still there afterwards. On a name the trader typed the button keeps its old feed-only wording and Focus is untouched |
| Strength board | A board session the trader confirms matches the TC2000 scan's character (~20â€“40/side). **Re-measure the fetch during market hours** â€” Â§10's 27.6 s was taken on a Saturday and is a floor, not a worst case. Decide the RVOL column then; it is specified but deliberately not built |

**Deferred deliberately:** RVOL for the surviving ~20â€“40 rows a side. Specified
in Â§9, not built â€” the trader decides on the first live board session whether
they miss it, and the fetch is cheap only at survivor scale.

### R1 live proofs â€” one PASSED, one HALF-PROVEN, two owed

**Quiet hours PASSED** on the 2026-08-16 22:06 boot (with the `IB: connected`
note resolved in the 2026-08-18 section above). **AWAY discipline is
HALF-PROVEN**: staging without adoption held for two full sessions, but the
flip back to DESK never happened. EVENING stop and the SPY alarm need an
EVENING night. Each is one observation on the desk:

| Proof | What to look for |
|---|---|
| Quiet hours â€” **PASSED 2026-08-16** | Launch at ~21:00 on a weekday with Auto left ON. `autopilot.log` says `Auto Pilot is ON from saved state, but nothing starts yet`; **no `Starting BounceBot` line**, no universe rebuild, no self-arm. A manual scan from the same desk still runs. **`IB: connected` on its own is expected and is not a failure** â€” `BouncePanel` connects on every launch at any hour (`bounce_panel.py:280`), outside Auto Pilot; the Auto Pilot start is the announced one |
| EVENING stop | An EVENING day: the open+30 slot and the 07:00/07:15/07:30 checks run, then one `Evening mode: swing slot(s) â€¦ not run` line per refused hourly slot and no further scan. The after-close wrap-up still fires |
| AWAY discipline â€” **HALF-PROVEN 2026-08-17/18** | An AWAY session: picks do not reach `longs.txt`/`shorts.txt`, alerts arrive silently while the feed and D1 badge fill, and the flip back to DESK adopts **only picks re-measured since the flip** â€” R2 changed this proof and R2.2 tightened it, so anything staged hours ago and no longer qualifying is refused rather than adopted. If the re-check itself fails, the status line says `Retrying in 60s` and **nothing adopts** â€” that is also a pass |
| SPY wake alarm | One real Â±1% EVENING day, or force it by setting `push_evening_spy_alarm_pct` low: an urgent push, a repeat no sooner than 5 minutes, and silence after flipping out of EVENING |

**~~Known limitation, deliberate~~ â€” CLOSED by R2 (2026-08-15).** The
AWAY/EVENINGâ†’DESK drain no longer adopts an un-revalidated backlog: every staged
pick carries a gate verdict from the most recent 30-minute refresh, and adoption
refuses anything failing, missing, or older than 45 minutes. The AWAY live proof
below is written against that behaviour, not the R1 behaviour it replaced.

### R1 build review â€” 2026-08-15 (independent five-dimension review; findings code-verified)

**All five findings are FIXED as of the R1.1 pass below.** The list is kept
because the defects are the useful record, not the fact that they closed.

Overall: the architecture is right, fail-open holds at every consumer, the manual
carve-outs are real, the alarm's dedupe/day-roll/restart mechanics are solid, the
shared-scan parity claim is proven against the base commit, no existing test was
weakened, and CLAUDE.md/AGENTS.md are byte-identical. But an **R1.1 fix pass is
required before the live proofs are attempted and before R2 stacks on top** â€”
the following were verified against the code, not just claimed:

1. **BLOCKER â€” the boot gate is defeated by the tick.** `_tick` calls
   `self._ensure_bot_running()` ungated (`autopilot_service.py:450`), so a 21:00
   boot with Auto left ON logs "nothing starts yet" and then connects BounceBot
   to IB 30 seconds later. Live proof #1 above will fail as written; every doc
   stating "no IB connect until the window opens" currently describes behavior
   the code does not have. The suite stayed green because the boot test stops
   the timer before a tick can run â€” the fix needs a test that runs a tick.
2. **BLOCKER â€” the EVENING SPY alarm fires on YESTERDAY's move pre-open.**
   `_maybe_push_spy_alarm` (`autopilot_service.py:1869-1872`) trusts
   `_spy_session_bars(cached_only=True)` with no bar-date check, and its only
   session gate is the quiet window, which opens 30 minutes before the open. On
   any EVENING morning after a Â±1% day, ~7 false urgent wake-ups fire on stale
   data before the first new-session bar (all night if quiet hours are disabled).
   Fix at the data read: refuse a series whose last bar predates `now.date()`.
   Every alarm test stubs `_spy_session_bars`; add one with stale-dated bars.
3. **IMPORTANT â€” a post-14:00 relaunch silently cancels the after-close
   wrap-up.** The quiet refusal in `_maybe_run_swing_slot`
   (`autopilot_service.py:953-955`) returns before any slot resolution, so slots
   still pending after 14:00 (crash or sleep before the close slot â€” a 4h39m
   sleep happened on this desk 2026-08-11) stay pending forever and
   `after_close_wrapup_due` never fires that day. Same rationale as the EVENING
   marked-done decision; apply it on the post-window side.
4. **IMPORTANT â€” EVENING picks still adopt into M5 Focus immediately.**
   `_poll_auto_pick_pending` refuses only AWAY
   (`alert_center_panel.py:1612`); the spec Â§1/Â§3.3, CLAUDE.md matrix, EVENING
   runbook, and CHANGELOG all state EVENING stages until the DESK flip. Make the
   code match the documented rule.
5. **IMPORTANT â€” the legacy Tk GUI dies at construction.** `gui.py:1040` still
   calls `get_shared_watchlist_paths`, which the removal deleted from
   `legacy.py`'s import block; `gui.py` acquires its globals from `legacy`, so
   construction raises NameError. One-line import fix. Invisible to the suite
   (tests import but never construct) and to the import-only frozen selftest.

### R1.1 repair pass â€” 2026-08-15 (all five findings closed)

| # | Fix | Proof |
|---|---|---|
| 1 | Quiet hours moved **into** `_ensure_bot_running`, the one place automation starts the bot; `force=True` is the manual carve-out and `force_reconnect` passes it | `test_the_tick_cannot_undo_the_boot_refusal` runs a real tick with the clock frozen to a weekday 21:00; `test_the_reconnect_button_starts_the_bot_at_any_hour` |
| 2 | The alarm refuses a SPY series whose last bar predates the day being asked about â€” stale cache is not a move | `test_yesterdays_cached_move_never_wakes_the_trader` (and the same +3% once today's tape prints it still fires) |
| 3 | `_resolve_slots_after_window` marks still-pending slots done once the window closes, so the after-close wrap-up survives a crash or a long sleep. Before the window opens nothing is resolved | `test_slots_left_pending_past_the_window_are_resolved` |
| 4 | `_poll_auto_pick_pending` refuses `("AWAY", "EVENING")`; EVENING also stops beeping, closing the spec Â§1 alert cell | `test_away_and_evening_refuse_to_adopt_staged_picks`, `test_evening_queues_alerts_without_a_sound` |
| 5 | `gui.py` uses `LONGS_FILE, SHORTS_FILE` instead of the deleted helper | New `tests/test_module_globals_resolve.py` statically resolves every global four never-constructed legacy modules read â€” verified to fail on the un-fixed file before the fix went back in |

Hardening taken in the same pass: NaN threshold guard on the alarm; the
quiet-window âŠ‡ sweep-window containment is now **structural** (`auto_scanning_window`
widens itself to contain `bouncebot_scan_window`, so two independent settings keys
cannot contradict each other); `autopilot_auto_arm_due` takes `quiet_hours` and the
arm test pins it, so a desk with quiet hours disabled no longer turns that test red;
`MainWindow._self_heal_universe`'s gate and the D1-feed beep site now have coverage;
the Qt tests **skip** instead of silently passing without PySide6; the false
"an early close moves this window" docstring claim is corrected (no early-close
modelling exists anywhere â€” pre-existing, and fail-open since the window is only
ever too long).

**Baseline after R1.1: 2785 passed / 19 subtests / smoke 7/7 / source selftest
30/30**, all exit 0. (Recorded at the time as "frozen"; it was the source run â€”
`launch_gui.py --selftest`, whose output carries no `(frozen)` suffix.)

Still owed, recorded not fixed: a corrupt `local_settings.json` silently re-homes
the store to `%LOCALAPPDATA%` (wants one loud stderr line plus atomic settings
writes); and the spec Â§1 EVENING **sweep** cell is now explicitly unresolved in
that spec's new Â§9 rather than silently unbuilt â€” the recommendation there is to
leave the sweep running, and the trader decides before the EVENING live proof is
recorded as passed.

Original hardening list from the review, for reference: NaN threshold
bypasses the alarm's threshold test (guard `threshold != threshold` like
`day_pct`); the quiet-windowâŠ‡sweep-window containment is enforced nowhere at
runtime (two independent settings keys; clamp or log the contradiction);
`test_autopilot_auto_arm_due_daily_hands_off_rules` reads the machine-local
`qt_auto_quiet_hours` setting and goes red on any desk that disables quiet hours
(pin `quiet_hours=True`); `MainWindow._self_heal_universe`'s gate and the D1-feed
beep site have zero coverage; five Qt tests silently pass (not skip) without
PySide6; the spec Â§1 matrix retains two EVENING cells (sweep "then quiet",
alerts "queue") the build never implemented and Â§8 never settled â€” reconcile or
build; a corrupt `local_settings.json` still silently re-homes the store to
`%LOCALAPPDATA%` (one loud stderr line + atomic settings writes); the
"early close moves this window" docstring/CHANGELOG claim is false â€”
`get_market_session_window` hardcodes regular hours (pre-existing, fail-open).

### Previous packet â€” ticker-briefs hardening (TB-0..TB-6)

| Field | Value |
|---|---|
| State | **Integrated and green on `testing-week-2026-08-10`**. **Live proof still owed: the 2026-08-12 22:00 window.** The 08-11 night proved TB-0, broke on TB-3, and exposed a task time limit that defeated its own concurrency guard plus 4h39m of machine sleep |
| Side item landed | **Snapshot popup opens at desk height** (2026-08-11) â€” UI geometry only |
| Side item landed | **Phone push policy + two richer pushes** (2026-08-11) â€” AWAY became the only pushing mode; R1 has since added EVENING's SPY alarm as the second exception |

A newly arriving AI resumes the active packet if it is unfinished. If it is complete,
it performs the stated next action. It does not select a different roadmap item
without explicit trader direction.

## Planning pass â€” 2026-08-15 (documentation only)

**Superseded the same day**: the trader then directed R1 to be built, and it was.
See the active-work table above. This section is kept for the recon findings it
records, which are still the current understanding.

The trader promoted the 2026-08-14 `WISHLIST.md` entries and directed a build
foundation for the next implementer. Recorded in this pass:

- **`plan.md` Phase 0.5 (R1â€“R6)** inserted with the trader's ranked order
  (R1 auto modes/quiet hours first, R2 Focus gating + strength board second) and
  five ACTIVE specs under `docs/` (indexed in `docs/README.md`).
- Eight trader decisions captured in the specs and `WISHLIST.md` (demote+label
  never hide; v1 extension rules; existing universe; build order; full pre-close
  honesty bundle; prior-anchor AVWAP line; checked = recorded decisions;
  Not-today removes just the M5 entry).
- **After-close investigation COMPLETE** (read-only): the live Master AVWAP scan
  scores today's forming D1 bar (no completed-bar guard in `runner.py`), and the
  setup tracker is written at 12:00 PT then wiped and rewritten by the ~13:24
  close-slot finish. Mechanisms with file:line evidence are in
  `docs/SWING_QUALITY_AND_FEEDBACK_PLAN.md` Â§4. No fix is built.
- Verification: Markdown-only pass â€” link resolution, `git diff --check`,
  control-document consistency. The recorded automated baseline (2738 passed /
  19 subtests / smoke 7/7 / source selftest 30/30) is **unchanged**.
- Housekeeping note: untracked `desk_report.xml` at the repo root is generated
  pytest JUnit output from the 2026-08-09 desk gate â€” left untracked; P1.5 owns
  gitignoring desk JUnit artifacts.

The active build item above (P0 live gates) is unchanged; Phase 0.5 code starts
only after P0.7 merges.

## Branch

Three branches exist, and that is the whole list (2026-08-15 consolidation):

| Branch | Tip | Role |
|---|---|---|
| `main` | `7d85a27` | trunk. Tip untouched; nothing is merged into it until Monday |
| `phase05-r2-focus-gating-strength-board` | `fc4bcaf` | **the desk branch** â€” what the scheduled task runs and what Monday's live proofs are observed against. Retired at merge step 4 |
| `testing-week-2026-08-17` | consolidated tip | **the release candidate** â€” testing-week + R1 + R1.1 + R2 + R7 + R8 + all review repairs. Worked in `..\TradingBotV3-r8` |

- State: **nothing merged to `main`; no PR recorded.**
- The consolidated branch is a strict superset of the desk branch (proven with
  `git merge-base --is-ancestor`), so the desk's source-run scheduled tasks are
  unaffected by the merge itself. The standing rule still holds: **disarm the
  scheduled task before switching branches on the desk.**
- Merge only after a `plan.md` Section 6 day passes â€” see the Monday sequence
  above.

## Last full Windows desk gate

Recorded at the 2026-08-09/10 desk re-baseline (`60119e8`):

| Check | Result |
|---|---|
| pytest | **2611 passed, 7 subtests passed**, exit 0 |
| JUnit | 2618 cases, 0 failures, 0 errors, 0 skipped |
| smoke | **7/7**, exit 0 |
| frozen self-test | **29/29**, exit 0 |
| Python | repo-local uv-managed **3.12** environment |

The frozen run found a real packaging-roster conflict: `ai_jobs` was deliberately
excluded from the bundle but required by self-test. The roster was corrected and a
permanent disjointness test added. The 29/29 figure is therefore the correct current
expectation, not the older 30/30 text in historical handoffs.

## Changes after that gate

The following commits landed after the recorded full gate and require coverage by the
next normal full run; none changes the frozen package inventory:

- `07395a0` â€” Chart Review Setups column defaults hidden and can be restored.
- `bfc8850` â€” a late-opened alert receives current bars.
- `4907b6f` â€” a published best-swing report can notify the phone.
- `1f41af1` â€” the swing push stays quiet when no readable setups exist.
- documentation consolidation: `CHANGELOG.md` for implemented history, `plan.md` for
  remaining work, `docs/README.md` for classification, and the renamed
  `CURRENT_CHECKPOINT.md` for active state;
- mandatory AI read/update workflow in `CLAUDE.md`/`AGENTS.md`, phase-gated roadmap
  ordering, and the new non-authoritative `WISHLIST.md`.

The documentation packet does not change the recorded automated baseline. Markdown
verification consists of link resolution, `git diff --check`, control-document
consistency, and confirmation that tracked edits remain Markdown-only.

## Re-baseline and desk configuration â€” 2026-08-10 (evening)

**P0.1 is satisfied for the four post-gate commits above.** Full Windows run on the
working tree:

| Check | Result |
|---|---|
| pytest | **2647 passed, 7 subtests passed**, exit 0 (109s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | not re-run â€” no packaging-trigger change since 29/29 |

Re-run after the decision-0015 documentation/comment pass: **2647 passed**,
**smoke 7/7**, unchanged. That pass edited Markdown, docstrings, comments, and two
user-facing strings only; no behavior, path, or test changed.

**Current baseline after the local-AI evidence-budget packet:**

| Check | Result |
|---|---|
| pytest | **2659 passed, 19 subtests passed**, exit 0 (104s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | not re-run â€” no packaging trigger (no new package, no new runtime asset, no new dependency) |

Twelve new tests cover the budget resolver and its fallbacks, the cloud ceiling
staying untouched, the derivation itself (worst-case retry prompt must fit the
context left after generation), the truncation tripwire firing/staying silent, and
ledger usage recording.

**Current baseline after the BounceBot scan-window packet (2026-08-10, late):**

| Check | Result |
|---|---|
| pytest | **2672 passed, 19 subtests passed**, exit 0 (104s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | not re-run â€” no packaging trigger (no new package, no new runtime asset, no new dependency) |

Thirteen new tests in `tests/test_bouncebot_scan_window.py` cover the window bounds,
the overnight and weekend refusals, the settings escape hatch and margin fallbacks,
and the four service transitions that matter: the close pauses a running sweep, an
after-hours start pauses on its first tick without needing a boundary crossing, a
manual resume survives subsequent ticks, and a broken session lookup changes nothing.

**Why this packet exists.** The trader reported the bot "running all night prompting
the API constantly". Reading the artifacts found two independent causes, and the loud
one was not the AI layer:

1. **BounceBot swept all night** â€” Auto Pilot's 30-second tick re-enabled scanning
   with no clock check, and `trading_bot.log` showed ~830-900 metric lines/hour for
   147 symbols, about eight full sweeps an hour, continuing hours past the close with
   IB answering `HMDS data farm connection is broken` and RRS timeouts. **Fixed here.**
2. **`ticker_briefs` retried all night** â€” see the open question below. **Not fixed.**

No metered API was involved in either: every unattended AI call is hardcoded
`provider="local"` against Ollama on localhost. OpenAI and Anthropic are reached only
from GUI buttons.

### Resolved â€” overnight AI job cadence (armed and built 2026-08-11)

The ticker-briefs hardening packet was **armed by the trader on 2026-08-11** after the
first overnight run and is **built** on this branch. The question below is kept because
its premises were partly wrong, and the correction is the useful part.

**What the first repaired night (2026-08-10/11) actually showed.** `ticker_briefs`
completed **all 95 symbols in 5,962 s â€” ~63 s/call**, not the ~4.75 min/call recorded
below. There was no window overrun. Instead **every one of the 95 briefs was
content-free**: the base evidence package was budgeted to the local ceiling *before*
the per-symbol projection, so the per-symbol-rich sources were unfunded at 0 chars
(`setups.current_tracker` 95,806 chars, `setups.current_tiers` 77,124,
`setups.bounce_learning` 17,995, `market.industry_intraday_rs` 17,833) and the funded
tables were sheared to about one row. MRVL's brief reads **"1 of 19 requested source(s)
usable"**, the one being its own watchlist membership. That is TB-0, and it was the
defect worth an hour and a half of GPU time to fix.

**Built:** TB-0 project-then-budget; TB-1 per-ticker failure isolation with an honest
partial morning file (`Briefed N of M. Failed: â€¦` in the header); TB-2 deterministic
membership-only skip; TB-3 resumable completion keyed by
`(session_date, symbol, evidence_hash)`; TB-4 a three-attempt per-session cap with an
identical-error early stop. `run_daily_summary` is untouched, so the two jobs now run
**separate five-session clocks**: `ai_summary`'s continues, `ticker_briefs`' restarts
at zero.

**Live proof owed â€” the next 22:00 window.** In the morning check: coverage counts
above one usable source per brief, statements citing real evidence, a morning-file
header stating the outcome, at most three `ticker_briefs` ledger rows for the session
(with a `terminal: true` row if it stopped early), and exactly one artifact set per
symbol under `ai_store/briefs/<year>/<session>/tickers/<symbol>/`.

**~~Known defect, reported not yet fixed (2026-08-11 evening review).~~ FIXED
2026-08-12 â€” and it fired live first.** TB-3's cross-firing reuse could never
trigger on the desk: the projected package's `evidence_hash` covers `generated_at`
and every source's read stamp, so identical evidence hashed differently on every
firing. On the night of 2026-08-11 a second runner instance restarted from symbol 1
and re-briefed 25 symbols, leaving 25 duplicate artifact sets on the DAS. The
manifest now carries a `resume_key` over stable fields only (symbol, session,
memberships, source ids + content); `evidence_hash` keeps its whole-package meaning
for artifact identity. Manifest schema `v1` â†’ `v2`; a row without a `resume_key` is
regenerated, never reused.

**Queued, not built (trader-approved 2026-08-11):** the **nightly journal pull** â€”
a third `journal_import` runner slot ahead of `ai_summary` so the summary reads a
journal already containing the session's trades. Spec with design decisions (Flex
over socket at night, Questrade token-rotation race stated, one-writer statement,
zero-execution `ok`) in `docs/LOCAL_AI_AUTOMATION_PLAN.md` sec 6.4c. Build only
after the 6.4b live proof passes and the trader says go.

**Integration correction (2026-08-11).** Fast-forwarding the hardening packet onto
the testing branch exposed a real 27-character ceiling overrun in the first focused
Windows run: list truncation budgeted retained rows before prepending its truncation
banner. `_truncate_to_budget` now budgets the banner too. Focused/full verification
is green: **74 focused tests**, **2687 full-suite tests plus 19 subtests**, and
**smoke 7/7**, all exit 0. The full gate also exposed a test-only warehouse-tee
hermeticity issue: its assertion observed every store open in the pytest process
rather than the tee worker it claimed to test. The assertion is now worker-scoped;
no warehouse runtime behavior changed.

<details>
<summary>The original open question, as written on 2026-08-10 (premises now corrected
above)</summary>

The 30-minute task repeat is **not** a work cadence; it is a retry ladder, and on a
healthy night sixteen of the seventeen firings read the ledger and exit in about a
second. Lengthening the interval would therefore save nothing and weaken the
self-heal. Two real defects sit behind the symptom instead:

- **A failing job has no attempt cap.** Only `ok` is a canonical completion, so a
  deterministic failure retries on every firing for the rest of the window. On the
  night of 2026-08-09/10 `ticker_briefs` failed **11 consecutive times at 9-16 minutes
  each â€” about 111 minutes of local inference that produced nothing.** A per-session
  attempt cap (2-3) would keep the self-heal for transient faults (NAS asleep,
  endpoint down) and end the grind.
- **`ticker_briefs` cannot finish as scoped.** It calls the model once per unique
  Focus/watchlist symbol â€” **95 today** â€” and publishes the morning file only after
  every one succeeds. At the observed ~4.75 min per call that is **~7.5 hours against
  an 8-hour window**, while the slot reserves only 120 minutes. It needs a symbol cap,
  incremental publication, or both.

Neither is fixed. Deferred deliberately: the 22:00 window on 2026-08-10 is the first
run with the repaired `gemma3:12b-tbv3ctx` model and is the live proof the AI-jobs
repair is owed, so the night was left alone rather than changed hours before it.

**Contingency drafted (2026-08-10, late):** the repair plan for both defects â€” plus
per-ticker failure isolation with an honest partial morning file, a deterministic
membership-only skip, and resumable per-symbol completion â€” is fully specified as the
**ticker-briefs hardening packet**, `docs/LOCAL_AI_AUTOMATION_PLAN.md` sec 6.4b, with
a pointer in `plan.md` P3.3. It is PROPOSED, not authorized: the trader arms it after
reading the 2026-08-11 morning ledger (or later five-session evidence). An arriving AI
must not build it without that direction. This documentation pass is Markdown-only;
the recorded automated baseline (2672 passed / smoke 7/7) is unchanged.

</details>

**Current baseline after the ticker-briefs hardening packet (2026-08-11):**

| Check | Result |
|---|---|
| pytest | **2682 passed, 5 skipped, 19 subtests passed**, exit 0 (106s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | not re-run â€” no packaging trigger (no new package, no new runtime asset, no new dependency) |

Recorded on a Linux container (Python 3.12, `TZ=America/Vancouver`,
`QT_QPA_PLATFORM=offscreen`); the 5 skips are the Windows-only cases the desk runs, so
the desk figure should read **2687 passed**. Fifteen new tests cover TB-0's
project-then-budget proof and its budget ceilings, the partial-publish header,
membership-only skip, resume-by-evidence-hash, and the attempt cap with its terminal
marker.

**Windows integration gate after the budget and hermeticity corrections:**

| Check | Result |
|---|---|
| focused | **74 passed**, exit 0 |
| pytest | **2687 passed, 19 subtests passed**, exit 0 (126s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | not re-run â€” no packaging trigger |

**Current baseline after the phone-push policy packet (2026-08-11):**

| Check | Result |
|---|---|
| pytest | **2720 passed, 19 subtests passed**, exit 0 (119s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | **29/29**, exit 0 â€” rebuilt at the trader's request, not by a packaging trigger |

Thirty-three new tests across `tests/test_away_push_roster_and_d1.py` (roster
membership, bucket-spelling collapse, the honest trim marker, and the D1 push
formatting/capping) and `tests/test_away_push_gating.py` (the AWAY-only gate on both
pushes, once-per-hour cadence, a failed send keeping its events, the kill switch, the
Alert Center classifier, and the panel signal firing on both D1 routing paths). Two
existing tests were updated rather than worked around: the Desk Link reclaim push now
declares AWAY (with a new sibling test proving it stays quiet in DESK), and the day-roll
test asserts yesterday's unsent D1 events are cleared.

**Live proof owed:** the next AWAY session â€” a swing push whose roster matches the
Setup Tracker's Favorite + High Conviction rows, a D1 push naming only events from that
hour, and silence on the swing/D1 channels while the desk sits in DESK or EVENING.

**Trader-verified on the phone, 2026-08-11 20:0x.** One real push built from the live
feed (593 rows, `data_date` 2026-08-11, source `focus`) delivered `ok: True`: five ranked
HC longs plus the full roster â€” HC 12 long / 7 short, FAV 30 long / 6 short, 55 names,
nothing trimmed. The D1 push is NOT yet proven: its queue only fills from live alerts in
the running desk.

**Documentation close-out (2026-08-11, Markdown only).** The push policy is now stated
where an operator or an arriving AI will actually meet it: `CLAUDE.md`/`AGENTS.md` core
loop (with the rule that a new ntfy sender must gate on AWAY or justify itself),
`docs/AWAY_SCANNER_RUNBOOK.md`, `docs/EVENING_MODE_RUNBOOK.md`, a `docs/FIRST_SESSION_CHECKLIST.md`
row, and `plan.md` P0.3. No file was added, removed, or reclassified, so `docs/README.md`
is unchanged; `WISHLIST.md` is untouched (no trader-directed idea moved). The recorded
baseline above still stands â€” this pass changed no code, path, or test.

### Desk rebuilt and relaunched onto the push-policy build â€” 2026-08-11 20:15

The frozen exe was the running desk (pid 35676, started 19:02); the python desk pid 32620
named earlier in this file was already gone. Rebuilt at the trader's request rather than
on a packaging trigger: graceful `CloseMainWindow`, `pyinstaller â€¦ --noconfirm` exit 0,
**frozen self-test 29/29 exit 0**, relaunch. **Running pid is now 2552** (started
20:15:20), heartbeat fresh at the 30-second cadence from 20:16:05. `dist/` is gitignored,
so the rebuild is verification only and no commit artifact.

### Desk restarted onto the scan-window build â€” 2026-08-10 21:19

The desk was closed gracefully (`CloseMainWindow`, so `closeEvent` ran its panel
shutdowns and released the writer lease) and relaunched through
`scripts/launch_gui_auto.ps1`, the same path the 06:00 task uses. **Running pid is now
32620** (started 21:19:22); it supersedes pid 17984 named below. Auto Pilot resumed ON
from saved state and BounceBot started and connected to IB as before.

Verified on the live desk immediately after:

- `bouncebot_scan_window` resolves to **06:00-13:30** from the real machine settings,
  with the verdict `False` at 21:20 and `True` at 09:45.
- **Zero `Metrics ->` sweep lines in `trading_bot.log` after the restart**, watched to
  fifteen minutes â€” the previous build would have run two full sweeps in that time.
  The whole log went quiet at 21:19:48 after the startup sequence (18 lines total, all
  of them start-up) against ~830-900 lines/hour beforehand.
- Sustained CPU fell from ~57% of a core to ~17% (and that figure still includes the
  start-up burst).
- `heartbeat.json` stays fresh at the 30-second cadence under the new pid, so the tick
  loop still reaches its end; `writer_role.py` still resolves
  `designated_writer / may publish True`, so the 07:00 publish proof is unaffected.

There is no "scanning paused" line in the Auto Pilot log, and that is the correct
outcome rather than a missing one: a freshly started BounceBot begins with scanning
already disabled, so the window gate simply never enables it and there is no state
change to announce. The startup IB traffic that remains (`$VOLD`/TICK recorder
contract verification) is the market-internals recorder, not the sweep.

Still owed by P0.3: the two live boundary crossings (a resume at 06:00, a pause at
13:30) and confirmation that the session itself is unchanged.

Three desk misconfigurations were found and fixed by inspecting the first
testing-week session's artifacts. All three were machine-local settings lost when the
old desktop was retired; none was a code defect:

1. **Designated writer was unset** â€” `autopilot_today.txt` had not published since
   2026-07-30, so the whole 2026-08-10 session produced no phone digest and no swing
   push. Fixed with `writer_role.py --designate-self` (NucBox_K8_Plus). The desk was
   restarted at 19:37 local to pick it up (pid 17984 then; superseded by the 21:19
   restart above â€” the designation is a saved setting and survives both), and
   `writer_role.py` now resolves `designated_writer / may publish True`, exit 0.
   **Not yet proven end to end:** `hourly_away_report_slot_due` returns nothing once
   the hour is past the session close, so no publish was due at restart time.
   `writer_health.json` consequently still carries its pre-fix 15:18 payload â€” that
   file is rewritten on a *publish attempt*, not at startup, so a stale copy here is
   expected and is **not** evidence the fix failed.
2. **`research_store_dir` was unset** â€” the warehouse was fully disabled and captured
   nothing. Now `\\MINI-PC\Trading Bot Data\research_lake`, layout created, and the
   restarted desk is the first process to run with it enabled. Capture is proven by
   the next scan writing under the lake, not by configuration alone.
3. **ntfy was already configured and works** â€” verified by test push (`ok: True`) at
   both `default` and `urgent` priority. Delivery to the iPhone banner/sound is an
   iOS-side setting and is **not yet confirmed by the trader**.

**AI jobs repaired 2026-08-10 (evening).** The task now exits 0 when run through the
scheduler. Details in `CHANGELOG.md`; the live proof is the 22:00 window tonight, and
`%LOCALAPPDATA%\TradingBotV3\logs\ai_jobs-<date>.log` will now carry any failure.
Two AI-layer caveats remain unproven and must be checked against tomorrow's ledger:

- ~~Context smaller than the evidence cap~~ â€” **closed the same evening.** Local
  calls now cap evidence at `ai_local_evidence_budget_chars` (22,000) and a
  truncation tripwire fails loudly if the server still sees less than was sent. The
  cloud ceiling is untouched.
- ~~The large tier cannot load~~ â€” **accepted and designed around.** The local large
  tier is retired (plan sec 2); policy drafts and retros belong to the frontier
  model. Revisit triggers recorded: Ollama Vulkan allocator work, ROCm on gfx1103,
  or more RAM.
- **Phase 2 design packet is PROPOSED, not approved.** `docs/LOCAL_AI_AUTOMATION_PLAN.md`
  sec 6.4a. Its six open questions need trader answers before any digest code is
  written â€” question 1 ("what counts as winning": R at scenario close, MFE/MAE, or
  both) is a trading judgement and is the one the whole fact pack hangs on.

### What the next session must confirm

Four fixes are configured and unit-verified but have **not** completed a live cycle.
None could be proven on the evening of 2026-08-10; all resolve by 09:00 on 08-11:

| Fix | Proof to look for | When |
|---|---|---|
| Designated writer | `autopilot_today.txt.meta.json` names `NucBox_K8_Plus` with a current `verified_at` â€” it still names the retired `DESKTOP-IABHR62` at 2026-07-30 | 07:00 publish |
| Swing phone push | an ntfy notification carrying numbered swings | 09:00 (push start hour) |
| Research warehouse | new files appearing under the lake root | first scan |
| AI jobs | `ai_jobs-20260811.log` records a completed `ai_summary` / `ticker_briefs` | 22:00-06:00 window |
| BounceBot scan window | **Requires a desk restart first** â€” the running pid predates the change. Then: one "scanning resumed" line at 06:00, one "scanning paused" at 13:30, and no symbol sweep in `trading_bot.log` after it | 06:00 and 13:30 |

If the 07:00 publish does not happen, read `writer_health.json` first: it will then be
fresh, and its `reason` names the exact gate that refused.

Still open on the desk, not blocking the week:

- `technical_integrity_events.jsonl` is ~247 MB and is never pruned (~10 MB/session).
- Off-site backup: cloud sync was the only off-site Class A copy (decision 0015).
- ~~One flaky test~~ **FIXED 2026-08-15.** `test_stale_staged_files_are_quarantined_not_deleted` was never flaky: `reconcile` compared a file's mtime against a coarser system clock, so a file written microseconds earlier read as "from the future". Both earlier notes here were wrong - it was not "observed once" (3 in 6) and not load-
  related (it reproduced in isolation). See the merge-safeguards section above.

## URGENT â€” the frozen desk cannot scan (found and fixed 2026-08-13)

The desk switched to `dist\TradingBotV3\TradingBotV3.exe` as its daily driver on
2026-08-12. The frozen build spawned its scan child as `sys.executable -c <code>`,
which under PyInstaller means `TradingBotV3.exe -c â€¦` â€” rejected by the app's own
argument parser, exit 2, one second after each slot fired. **Every Master AVWAP D1
swing scan failed from 2026-08-12 07:30 through 2026-08-13 09:00.** Last success:
2026-08-11 13:23:59, 622 setup rows.

Nothing else broke, which is why it went unnoticed: BounceBot, the 07:00 open scan,
Auto Pilot and the away report all run in-process. The visible cost was one layer
away â€” the overnight AI read 11 stale D1 sources.

**Code fix is committed and green** (`scripts/scan_worker.py`,
`scan_service.scan_worker_command`, `launch_gui --run-scan`, `selftest` roster,
`tests/test_scan_worker_spawn.py`), and the desk was **rebuilt 2026-08-13 11:00:25**
after the trader closed it:

| Check | Result |
|---|---|
| pytest | **2738 passed, 19 subtests**, exit 0 |
| smoke | **7/7**, exit 0 |
| frozen selftest | **30/30**, exit 0 â€” was 29/29; `scan_worker` is the added check |
| frozen `--run-scan` dispatch | **verified** â€” a deliberately malformed payload now fails inside `scan_worker.parse_payload`, where the old build answered `TradingBotV3.exe: error: unrecognized arguments: -c â€¦` |

**Still owed: one real slot on the desk** â€” `Swing scan for slot HH:MM finished at â€¦
(N setup rows)` in `autopilot.log`. Nothing before that proves a full scan runs
end to end under the frozen build; the checks above prove only that the child
starts and reaches the scanner. Until then the fallback is running from source
(`scripts/launch_gui_auto.ps1`), where the `-c` form is correct.

Also owed once a slot passes: the D1 sources have been stale since 2026-08-11
13:23:59, so tonight's AI window is the first that can read fresh evidence. A brief
that still cites truncation after a good scan day means something else is wrong.

## What the 2026-08-11 window measured, and what was repaired â€” 2026-08-12

The packet's owed live proof ran and is **partial**. Ledger and manifest evidence:

| | Result |
|---|---|
| `ai_summary` | **ok at 22:02:53**, first attempt, ~170 s, 10 usable sources â€” against six degraded rounds the night before |
| `ticker_briefs` | **no completion row.** 126 briefs / 101 unique symbols of 182, 0 failures, 22:04:33 â†’ 01:20:08, killed mid-batch |
| `ai_morning_brief.txt` | **never published** â€” still the 2026-08-10 file, because publication happened only after the loop |
| TB-0 | **Confirmed.** MDB's real brief: 7 of 19 usable, 0 unfunded (08-10 was 4 of 19 with 5 unfunded) |
| TB-1 / TB-2 / TB-4 | Not exercised â€” 0 failures, and every membership-only name sits past list position 100 |
| TB-3 | **Proven broken**, 25 symbols with two rows and two distinct `evidence_hash` values |

Three defects and one machine fault, all now addressed except the last:

1. **TB-5 â€” roster noise.** 96.2% of everything sent to the model (307,630 of
   319,687 chars) was ticker name-dumps matched line-wise; median symbol-specific
   content 42 chars; only 18 of 166 symbols had a real scan line. Fixed by a
   residue test, not a ticker count. Measured effect: **166 model calls â†’ 49**.
2. **TB-3** â€” see the repaired entry above.
3. **TB-6 â€” publication only after the loop.** Now republished after every resolved
   symbol, with an explicit in-progress note; the market-session block still
   suppresses publication outright.
4. **`ExecutionTimeLimit` was `PT2H` against an 8-hour window** â€” it terminated the
   22:00 run's parent at 00:00, freeing `IgnoreNew` so the 00:00 repetition started a
   second runner while the first instance's Python child kept going. The manifest
   shows the two interleaving one-for-one from 00:01:54. Now `PT8H` in
   `scripts/register_ai_jobs_task.ps1` **and applied to the live desk task**.
5. **Machine sleep â€” trader-owned, not code.** 60 Modern Standby transitions during
   the window, **4h39m asleep**, including an unbroken 01:39:42 â†’ 05:57:09 that
   killed the run and suppressed every firing from 01:30 to 05:30. The trader is
   raising the sleep setting. **Until that is confirmed, no overnight result is
   evidence about the AI layer.**

**The 2026-08-12 morning check.** Expect ~49 model calls against ~160 symbols (the
rest membership-only), roughly an hour of inference rather than 3.5, exactly one
`ticker_briefs` ledger row, a morning file dated 2026-08-12 **without** the
in-progress note, no duplicate artifact sets, and briefs that cite `daily.market_prep`
scan lines and `setups.tier_performance` rows rather than complaining about
truncation. `setups.current_tracker` is a known remaining gap: it arrives as one
JSON line, so line-based projection is still all-or-nothing for it.

## Immediate live gates

- **P0.1:** ~~run the complete Windows automated gate~~ â€” **done 2026-08-10**
  (2647 passed / smoke 7/7), and **re-run 2026-08-15 on the R1 branch**
  (2773 passed / 19 subtests / smoke 7/7 / source selftest 30/30, all exit 0).
  Re-run again before merge if further code lands.
- **P0.2â€“P0.4:** run the single-main session checklist, Away/ntfy validation, and
  observability rollover.
- **P0.5:** run the durability mid-session restart/backfill drill.
- **P0.6:** start Local-AI's five-session clock and the warehouse broker/live/pilot
  sequence.
- **P0.7:** merge only after the live-validation day and applicable rechecks pass.
  **One** branch now queues for `main` â€” `testing-week-2026-08-17`, which carries
  testing-week, R1, R1.1, R2, R7 and R8 together after the 2026-08-15
  consolidation. The gates to re-run, the clean-cache rebuild rule, and the
  desk-switch order are in the Monday sequence at the top of this file.

Do not add historical detail here. When a change lands, update `CHANGELOG.md`; when a
gate remains, update `plan.md`.
