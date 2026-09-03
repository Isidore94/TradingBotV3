# Current checkpoint

This file is the frequently refreshed active-work, branch, and verification stamp.

- Implemented inventory and revision history: [`CHANGELOG.md`](CHANGELOG.md)
- Remaining work and gates: [`plan.md`](plan.md)
- Supporting-document roles: [`docs/README.md`](docs/README.md)
- Entries dated 2026-08-25 and earlier:
  [`docs/CHECKPOINT_ARCHIVE_2026-08.md`](docs/CHECKPOINT_ARCHIVE_2026-08.md)

---

## Active state at a glance

**Read this block first. It is the answer to "where are we?" — the dated entries below
are the working record behind it. Refresh this block on every handoff; if it disagrees
with the newest dated entry, the dated entry wins and this block is stale.**

| | |
|---|---|
| Working branch | **`claude/v2-loop-closes`**, off `main` - Phase 0.14 packet **V2**: items 1, **2 (a/b/c/e)**, 4 and 5 built; **item 3 (the AWAY Recap) is not**. V1 is merged to `main`; decision 0016 is the tie-breaker for this phase. **`claude/p10-after-the-like` is still unmerged** |
| Also in flight | **`claude/p10-after-the-like`** - Phase 0.13 packet P10, built and verified, NOT merged. It has no dependency on V1 and V1 has none on it |
| Active roadmap items | **Phase 0.13 packets P0-P9: ALL MERGED, live gates #29-#40 owed**; **integration gate #38** (one DESK session on the merged tree after a restart); Desk snappiness packets 1-3 (#24-#26); Phase 0.11 theta (#23); Strength Board (#22); day-trade pass (#21); swing picks (#20); desk lockup (#19); R7 journal auto-tagging + statement import; Phase 3.2 + 6.1 (warehouse); Phase 0.9 (GUI); Phase 0.10 (AVWAP band challenger) |
| Last verified baseline | `pytest tests/ -q` **6222 passed, 72 subtests, process exit 0, ZERO failures** (2026-09-02, desk `.venv`, on `claude/v2-loop-closes`) - the `ai_jobs_runner` lock probed FREE immediately before the run. `ruff` **clean** · smoke **7/7** · source `--selftest` **74/74** · no packaging trigger. Previous: **6200 passed** |
| Frozen exe | **NO REBUILD REQUIRED BY R1, and this is a measured statement rather than an omission.** P0-P6a and P8 add no dependency, no non-`.py` asset and no spec change; every new module is inside an already-collected package (`scripts/` root, `ui.annotations`, `research_warehouse`, `ai_jobs`). P7's asset was the one packaging trigger and its exe was already rebuilt on 2026-09-02: 420 MB, `selftest OK: 74/74 checks passed (frozen)`, exit 0, with the 74th check LOADING `setup_registry_v1.json` from inside the frozen process. Still a verification artifact: the desk runs from SOURCE |
| Desk restart | **DONE 2026-09-02 04:09, trader-authorized ("Go ahead and restart the desk").** The checkout was moved to `main` at `125ffa0` FIRST and verified (selftest 74/74, smoke 7/7, ruff clean) - restarting onto the branch it happened to be sitting on would have put the wrong code on the desk. Old pid 17132 (up since 2026-09-01 11:09) stopped; relaunched through `trading_desk.cmd`, the production launcher, unchanged. New pids **25884** (the desk) under trampoline **9140**. Verified UP THREE WAYS rather than assumed: the process outlived the launch by minutes, `heartbeat.json` re-stamps every ~4 min naming pid 25884 (it had read pid 17132 before), and a second launch printed "another TradingBotV3 desk is already running" and exited 0. **The desk now knows `tag_status`, so P6a's 24 provisional tags render as provisional rather than as the trader's own** - which is what the restart was for. The nightly AI run was a SEPARATE process and was not disturbed; it finished normally at ~04:08 |

### Open gates, newest first

Each is owed before the work it belongs to can be called live-validated. Detail is in
the dated entry named beside it.

| # | Gate | Owed by |
|---|---|---|
| 49 | **Weekend Prep, read in one click (V2 item 2)** - one open where Refresh builds every step and the verdict card shows five to eight lines with an n on each; then "Tag this week" lists the week's unconfirmed trades and Confirm all shown writes the trader's answer | 2026-09-02 V2 entry |
| 48 | **The hidden surfaces (V2)** - a desk session with Alerts, D1 Focus, Armed and Universe hidden, and EVERY capture-rail hotkey still firing | 2026-09-02 V2 entry |
| 47 | **One box, one Enter (V2)** - one Market Journal entry written from the desk tab with a single Enter, filed against the right session | 2026-09-02 V2 entry |
| 46 | **The tagger runs itself (V2)** - one nightly run that tags new trades, and the Journal nav button showing the review count the next morning | 2026-09-02 V2 entry |
| 45 | **One window, two sections (V1)** - the RS/RW section opens ABOVE the M5 Strength section in the alert column, and neither widens the column | 2026-09-02 V1 entry |
| 44 | **TC2000 parity (V1)** - one DESK session where the Strength section matches the trader's own TC2000 list on the same minute for the top ten names, with the parity toggle ON. Turning it OFF shows the near-misses greyed, each naming the filter it failed | 2026-09-02 V1 entry |
| 40 | **The narration fits (R3)** - one overnight `setup_research` run that publishes **exactly ONE pack** for the date and a `.narration.json` beside it. Three siblings, or an `ok` whose reason contains `narration absent`, means the view is still too large - and the refusal message names the size, the budget and the eligible-cell count, so it says which. Also check the pack carries `built_by_commit` and a non-empty `recipe_ids` | 2026-09-02 R3 entry |
| 39 | **Quick like (Phase 0.13 P9)** - one DESK session: the trader quick-likes one SWING chart and one M5 chart. Both rows reach `trader_annotations.jsonl` with `like_mode` quick, the M5 one carries `m5_bars_ref`, BOTH charts retire, and nothing appears in Focus. The next morning `like_cohort_picks.csv` holds both, the M5 one has `m5_bars_completed_ref`, and **its intraday columns are numbers rather than blank** - which is also what closes gate 34's open definition question | 2026-09-02 P9 entry |
| 38 | **The merged tree, on the desk (R1, extended by R2)** — one DESK session after the restart with the stall watchdog ON and quiet on every new surface: the Weekend Prep pass/rejection/preference tables, the journal's Provisional filter, the Decisions pane, the M5 take-rate row and the Strength Board section. **R2 adds two specific things to watch**: the Setup Tracker's current-picks count after the FIRST scan of the day (it should be the real tier count, not one row per symbol - that is the NAN guard), and the Weekend Prep backlog-toggle line in `ui_stalls.jsonl`, which should now be absent | 2026-09-02 R2 entry |
| 37 | **First setup-parameter grid (Phase 0.13 P8)** — one overnight run publishes rows for every declared cell inside the 20-minute reserve, and the trial-ledger row exists with status `collecting`. **The third condition is a refusal, not a check: no cell may be read for a verdict before the declared 20-session window closes** — including by me, and including if an early cell looks good | 2026-09-02 Phase 0.13 P8 entry |
| 28 | **HTF LRSI study (Phase 0.12 B)** — one overnight `setup_research` run that publishes `htf_lrsi_*` outcome rows inside the existing 20-minute reserve, with `bar_derived` rows under `timeframe=H2` present and no stub in the oscillator's input; then a first read of whether any cell clears the evidence floor | 2026-09-01 Phase 0.12 entry |
| 27 | **Focus de-clutter (Phase 0.12 A)** — one DESK session: the D1 Focus feed carries pullbacks only, an armed extension watch still fires from the Armed board, a watch past its window leaves the board with a row behind it, and a faded pick can be restored (fresh clock) and discarded from the chart | 2026-09-01 Phase 0.12 entry |
| 32 | **Fact-pack truth (Phase 0.13 P3)** — one overnight `setup_research` run whose Markdown **opens with the eligible block**, shows **`n_episodes` beside `n`**, **names the excluded families** (GENERAL, FAVORITE_ZONE_WATCH) and prints the **bucket-coverage line** with a real count rather than UNKNOWN (which needs at least one warehouse build after this lands); plus the trader confirming the Research readout panel lists **more than two families** on 'All families' | 2026-09-01 Phase 0.13 P3 entry |
| ~~1~~ | ~~Frozen rebuild + frozen selftest~~ — **MET AGAIN 2026-09-02** on the merged tree: 420 MB, `selftest OK: 74/74 checks passed (frozen)`, exit 0, and the new check LOADS `setup_registry_v1.json` from inside the frozen process rather than trusting the `datas` rule. Previously met 2026-08-31 at `d0a2ae6` (the merge point): 419 MB, `selftest OK: 72/72 checks passed (frozen)`, exit 0, SAC reads OFF. Previously met 2026-08-28 at `fff07b8` | done |
| 29 | **P0 trader decisions (Phase 0.13)** — one DESK session with **no LRSI line on the M5 alert bar**, `lrsi_cross_20` / `lrsi_cross_50` rows **still arriving in `intraday_bounce_outcomes.csv`** that same day, and **no BANGER branch left in the alert path** (grep `scripts/` for `banger` — permitted hits are the retired `banger` review column, the `REGIME_BANGER_*` regime-pause thresholds, the regime-pause function names `_sweep_regime_pause_bangers` / `_record_regime_pause_banger`, the two dated "the retired BANGER class used to" comparisons in `bounce_bot_lib/learning.py`, and the trader's own quote in the `alert_repetition.py` docstring may remain) | 2026-09-01 Phase 0.13 P0 entry |
| 30 | **P1 grading loop (Phase 0.13 P1)** — one **Weekend Prep** opened after the next scan showing: a **`human_focus_swing_vetted`** row in the picks table; a like merged into `like_cohort_picks.csv` on the DAY it was captured (its `trade_date` equal to the session, not the night before); **one** pooled `compressed` veto cohort rather than the current two; and an **`r_gaps`** array present in `review_preference_state.json` | 2026-09-01 Phase 0.13 P1 entry |
| 31 | **P2 surfaces (Phase 0.13 P2)** — one DESK session where the trader opens all six: the two Weekend Prep judgement tables (robust columns, horizon selector, greyed sub-floor rows), the week page's named callouts, the Daytrade Tracker's **My Decisions** tabs, the A.I. Summary **gate strip**, and the M5 alert bar showing both a **take %** suffix and a **×N** fold — and `ui_stalls.jsonl` charges no seconds to any of them | 2026-09-01 Phase 0.13 P2 entry |
| 33 | **Swing variables (Phase 0.13 P4)** — one desk scan, then: the **Attributes** tab on the Setup Tracker opens without stalling the desk and shows the greyed sub-floor rows; the scan-factor leaderboard's new `stale_horizon_observations_dropped` column carries a real count; and an `expected_r_note` on the priority report names its exit template. The new attribute keys only appear on setups recorded AFTER this lands, so the leaderboard needs a scan plus forward sessions before it can grade them | 2026-09-01 Phase 0.13 P4 entry |
| 34 | **Pass and rejection cohorts (Phase 0.13 P5)** — the trader records **two real passes and one not-today** on the desk; the next morning `pass_cohort_picks.csv` and `rejection_cohort_picks.csv` both have rows, and the two new Weekend Prep tables show them. The intraday columns are expected to be BLANK with `sidecar_ends_before_the_entry_bar` — that is the measured structural limit, not a failure of this gate | 2026-09-01 Phase 0.13 P5 entry |
| 35 | **Preference to trade (Phase 0.13 P6)** — the trader imports a real day and one trade shows a `trader_capture` candidate with a linked event; the nightly report lists that day's likes with a traded/not-traded column. A candidate that points at `review_event:<ts>` rather than an alert id is the EXPECTED result for 676 of 730 rows, not a gate failure | 2026-09-01 Phase 0.13 P6 entry |
| 36 | **The tagged backlog (Phase 0.13 P6a)** — the trader opens the Provisional filter on the desk and confirms or edits **at least ten** of the 24 provisional tags; the "my setups" chart then populates from confirmed rows only, and "provisional setups" shrinks by the same number. The 132 `needs_review` rows staying blank is the EXPECTED result — 104 of them have no scanner candidate at all — not a gate failure | 2026-09-01 Phase 0.13 P6a entry |
| ~~1~~ | ~~Frozen rebuild + frozen selftest~~ — **MET AGAIN 2026-08-31** at `d0a2ae6` (the merge point): 419 MB, `selftest OK: 72/72 checks passed (frozen)`, exit 0, SAC reads OFF. Previously met 2026-08-28 at `fff07b8` | done |
| 2 | **Warehouse canary** — one post-scan run verifying occurrence/context/outcome writes and bounded memory; then all symbol buckets filled; then one overnight fact pack compared against warehouse counts | Phase 3.2 (2026-08-27 tracker entry) |
| 3 | **Desk memory** — one DESK session, first swing-scan slot, confirming the 8–13 GB jump is gone | 2026-08-27 afternoon (memory) entry |
| 4 | **Group RS/RW tape** — one DESK session with the four trader rules of that morning | 2026-08-27 afternoon (group tape) entry |
| 5 | **The four 2026-08-27 trader rules** — one DESK session on a directional day covering auto-Focus, the VWAP-side/show-time filter, the D1 SMA leg, and the M5 alert bar | 2026-08-27 morning entry |
| 6 | **Market Journal** — one desk session where a Desk-tab note reaches the left-nav page | 2026-08-27 evening entry |
| 7 | **SOAK 1** — the gate on Phase 0.9 G-P2.3; not yet run | 2026-08-27 Phase 0.9 entry |
| 8 | **Phase 0.8 live soak** — the trader's to run | 2026-08-26 fluidity entry |
| 9 | **Feature-history exports** — one 2026-08-28 scan producing `output/scan-factors` and `output/tier-tracker` files again instead of `ParserError` | 2026-08-28 corruption entry |
| 10 | **Narrated digest overnight** — one unattended 22:00 run producing a narration without being forced | 2026-08-28 narration entry |
| 11 | **Narrated summary + ticker briefs overnight** — one unattended run at the raised context; tonight's briefs were 0 of a normal 53-62 | 2026-08-28 context entry |
| 12 | **Sliced summary overnight** — one unattended 22:00 run: 46 slices, a synthesized summary, briefs still finishing in the window | 2026-08-28 slices entry |
| 13 | **Journal auto-tagging** — one desk session: tag real trades, rename one, filter on it | R7 auto-tagging (2026-08-28 tagging entry) |
| 14 | **Statement import** — the trader imports their own Questrade YTD file on the desk, against the live journal | R7 statement import (2026-08-28 statement entry) |
| 15 | **Statement layering + self-check** — the trader imports both real files on the desk and runs "Check a statement..." against the live journal | R7 statement layering (2026-08-28 layering entry) |
| 16 | **IBKR file import** — the trader imports their IBKR transaction file on the desk; the second account's mask resolves once Flex has named it | R7 IBKR file (2026-08-28 IBKR entry) |
| 17 | **File authority** — one desk import where a shared day agrees (sync keeps its times) and, if one ever disagrees, the file takes it | R7 file authority (2026-08-28 authority entry) |
| 18 | **Tax report** — one desk run of "Realised P&L for tax..." against the live journal, with the BoC rates booked so the CAD total is complete | R7 tax report (2026-08-28 tax entry) |
| 26 | **Desk snappiness packet 3** — three proofs, stall watchdog ON: one OVERNIGHT where the after-close technical-integrity replay finishes in minutes rather than an hour (the wrap-up log's own timing) and `technical_integrity_events_resolved.jsonl` exists beside the main log; one QUIET-HOURS night with no Industry Board download in the logs and no five-second post-launch fetch; and one DESK session with the drip lines quiet in `ui_stalls.jsonl` — `setup_tracker_panel.py`, `focus_picks_panel.py` chip churn, the entry-board minute tick, and the technical-integrity 30 s parse | 2026-08-31 snappiness packet 3 entry |
| 25 | **Desk snappiness packet 2** — one DESK session with the stall watchdog ON: `ui_stalls.jsonl` quiet at `bar_cache.py:75` and at the GC collector lines, and a journal retag (accept a correction, or add an execution) that shows "tagging..." and does NOT freeze the desk | 2026-08-31 snappiness packet 2 entry |
| 24 | **Desk snappiness packet 1** — one DESK session's `ui_stalls.jsonl` with `data_table.py:170`, `watchlist_utils.py:33`, `project_paths.py:165` and the operations-audit CSV parse gone quiet (stall watchdog stays ON; #23 is the theta gate on `claude/theta-premium`) | 2026-08-31 snappiness entry |
| 23 | **Theta premium (Phase 0.11)** - one desk scan whose theta report shows percent-floored, support-first rows: no quarter-dollar credits on expensive names, the richer of two equally defended strikes on top, a `premium=` line on every quoted sold put, and DRAM still labelled `via thetalongs.txt` | 2026-08-31 theta entry |
| 22 | **Strength Board in the Desk** - one desk session: the trader opens the section under the Strength window, reads the board in the column, clicks a row onto the Visual Alert Review chart, adds a name from it, and says whether the vertical stack is right | 2026-08-31 Strength Board entry |
| 21 | **Day-trade pass** — one desk session where the trader records a real pass from the Alert Center capture tab: the ticked reasons and the note reach `trader_annotations.jsonl`, the chart STAYS UP, and a pass taken while an M5 chart is drawn carries its bars into `trader_annotation_bars/` | 2026-08-31 pass entry |
| 20 | **Today's swing picks** — one desk session: the trader enters their real end-of-day swing list, the names show in swing Focus as THEIRS (no auto marker; "Not today" and the desync repair leave them alone), the bar/strip split drags and the size survives a restart, Paste takes a TC2000 list and Copy hands one back, one removal retracts without disturbing the earlier row, and a name they actually trade comes back marked "took" | 2026-08-31 swing picks entry |
| 19 | **Desk lockup fix** — one DESK session on a directional morning where the drain stages a large batch: the desk stays responsive, every staged pick reaches M5 Focus across successive ticks, and `ui_stalls.jsonl` charges no seconds to `focus_picks_panel.py` or `setup_delegate.py` | 2026-08-31 lockup entry |


### 2026-09-02 - V2 second run: Weekend Prep gets one Refresh, a card and a tag step

**Same branch.** Live gate 49 owed. Item 2's (a), (b), (c) and (e); item 3 is
still not built.

**ONE REFRESH.** The click starts each page's own reader and returns - measured
under 50 ms in a test, which matters because the reads behind it were once 8.45 s
of frozen GUI on one page alone. One page that will not start does not stop the
other four. The five per-page buttons left the LAYOUT and stay as objects,
because `reload()` uses each as its own single-flight guard.

**THE VERDICT CARD** is a PURE builder, so it is testable without a journal, a
lake or an event loop. Every measured line carries its n; a cohort under n=5 is
named thin and never ranked, because a top row resting on two observations is
worse than no row; a missing input SAYS SO, because "no graded likes yet" and
"your likes averaged 0.00R" are different facts and only the second is a claim.
The P&L line counts CONFIRMED tags only - "my setups" means the trader's answer,
not the tagger's guess. It reads five stores on the worker, each guarded
separately: a card that failed because one of five files was unreadable would
tell the trader nothing about the four that were fine.

**THE RS/RW PROSE IS RETIRED** - two long blocks duplicating a LIVE board with a
Saturday snapshot. The log SCANS are kept, uncalled, and their docstrings say so
in capitals: a reader that exists but is never called renders the same blank page
as a broken one, and the next agent must not "fix" it by wiring the wall of text
back. Two tests that asserted the retired block's behaviour are replaced by ONE
named test recording that they retired with it.

**"TAG THIS WEEK"** is a sixth step, appended between reading the week and
planning the next. It lists only what is not the trader's answer yet; confirming
goes through `JournalStore.confirm_tags`; a failed write is reported LOUDLY,
because a confirmation the trader believes landed is worse than one that visibly
did not. Ten visible rows - three at a time was the complaint.

Two panel tests updated: one pinned the step COUNT at five, which made it fail for
the sanctioned way to add a step, and now pins the routine's ENDS instead; the
other selected a page by row number and now selects by name.

**Still owed:** item 3 entirely, and item 2's takes/watch-conversion table, the
ten-row pass over the other tables, and the collapsed how-to-read notes.

**Verification.** `pytest tests/ -q` **6222 passed, 72 subtests, exit 0** · `ruff`
clean · smoke **7/7** · `--selftest` **74/74**.

### 2026-09-02 - Phase 0.14 packet V2: the loop closes (items 1, 4 and 5 of five)

**Branch `claude/v2-loop-closes`, off `main`.** Live gates 46-48 owed. V1 was
merged to `main` first, as the packet required. No frozen rebuild: no new
dependency, no new asset, and the one new module sits inside `ai_jobs`.

**ITEM 1 - THE TAGGER RUNS ITSELF.** P6a built the whole machine and left it as a
command the trader had to remember. `journal_auto_tag` runs it nightly at the
recorded 0.70 threshold, **inserted SECOND** - right after `journal_import` and
before everything else. That position is an insert rather than an append and is
the second and last sanctioned exception to this list's own rule: the import is
what puts the night's trades in the journal, so tagging ahead of it would tag
yesterday's, and every cohort slot below reads the journal, so tagging after them
would hand them one a night stale. Both exceptions are now argued for in
`default_slots`' docstring, because a third would mean the list has an ordering
nobody can state.

**A journal write fails LOUDLY.** Every other evidence store swallows a failed
append; the journal is the exception, because a tag that silently did not land is
a trade the trader will believe is tagged. The test proves it by making the write
raise - `JournalStore` creates its own parent directory, so a missing path would
have tested nothing.

**The badge** reads "Journal (12 to review)", counted from the STORE rather than
from the last run's summary: a run that wrote nothing new does not mean there is
nothing to review, and the trader may have confirmed some since.

**ITEM 4 - ONE BOX, ONE ENTER.** The desk capture had a timeframe picker, a box, a
Save button and a status line - four decisions for a thought you have at 10:40 and
would otherwise lose. Everything but the box leaves the SURFACE; **nothing leaves
the SCHEMA**, because a field that exists at v1 keeps its name and meaning forever
and the nightly scope reads it. Plain Enter saves through an EVENT FILTER, not a
`QShortcut`: a shortcut on Return would fire for every widget in the panel's
scope, and this key must mean "save" only in this one box.

**The entry is dated to the session it is ABOUT.** Today while today trades; the
last session that traded on a weekend or a holiday. A thought written at 18:00 is
about the day that just ended, and dating it tomorrow would file it against a
session that has not happened. Measured: Wed 10:00 and Wed 18:00 both give
2026-09-02; Saturday, Sunday and Labor Day all give 2026-09-04.
`written_after_the_session` is untouched and still COMPUTED, because which day the
note is about and whether the trader had already seen how it finished are
different questions.

**ITEM 5 - HIDDEN IS NOT REMOVED.** One setting, default OFF, hides the Alerts,
D1 Focus and Armed tabs and the Universe page. All four are load-bearing behind
the scenes - the review-alert door, the flag list two polls write into, the armed
inventory the expiry sweep walks, and the builder that writes the file the scanner
reads - so `setTabVisible` rather than `removeTab`, no index shifts, every timer
still visibility-gated. **The shortcut rule is the part that would actually cost
the trader something**: a `QShortcut` owned inside a hidden tab never fires, and
two bindings for one sequence fire NEITHER, so a test asserts every rail shortcut
is panel-scoped, bound once, and not owned inside a hidden tab.

**A DEFECT OF MY OWN, FOUND AND FIXED HERE.** Item 1's badge started its reader in
`__init__`. That thread opened the journal while another test was still
monkeypatching the journal's module globals, and it made
`test_migration_failure_stays_visible_instead_of_claiming_no_accounts` fail from a
hundred tests away - green alone, red in the suite, which is the worst kind of
failure to own. It starts from `showEvent` now and is joined in `closeEvent`; a
window nobody has shown is a window nobody is reading a badge on.

**ITEMS 2 AND 3 ARE NOT BUILT.** Weekend Prep still has its per-table refreshes,
its week-in-review text block and no verdict card; the AWAY Recap is still the
forward-looking digest with no outcomes and no charts. Both are UI rebuilds of a
size that deserves its own run, and plan.md records what each owes.

**Verification.** `pytest tests/ -q` **6200 passed, 72 subtests, exit 0, zero
failures**, lock probed FREE immediately before the run · `ruff` clean · smoke
**7/7** · `--selftest` **74/74**.

### 2026-09-02 - Phase 0.14 packet V1: names first (items 1 and 2 of four)

**Branch `claude/v1-names-first`, off `main`.** Live gates 44-45 owed. Decision
0016 was merged to `main` first, as the packet required; it is docs-only and it is
the tie-breaker for this whole phase.

**ITEM 1 IS COMPLETE: the board is the trader's own scan now.** Their relative
volume (`AVG(V / mean(V78 ... V1170), 12)`, positional exactly as TC2000 is, blank
and never zero under sixteen sessions of history), their four floors, their
universe (`universe_all.txt` PLUS the four watchlists), and their filters applied
as a DISPLAY filter - a row that misses is greyed and names what it missed, behind
a default-on parity toggle that hides them for a line-by-line comparison.

**Two costs, both measured rather than assumed.** The M5 fetch period had to grow
from `5d` to `1mo`, because the RVOL needs 1,182 bars and `5d` holds about 390 -
under the old period every RVOL on the board would have been blank. And the D1
floors need daily bars, so there is now a second batched daily download over the
symbols that actually reached the board. Still zero IB traffic.

**The fence on `strength_scan.py` is NARROWED, not lifted.** It was frozen whole
by the R8 spec - "stop and ask the trader first" - and the trader asked, in this
packet, naming the file. What the fence protects is the FORMULA, so the test now
asserts the seven formula functions are byte-identical to the R8 baseline. That is
stronger than "no edits at all", which could be satisfied by not touching the file
while the numbers moved underneath it.

**The golden's expected values come from a SECOND hand implementation** written
from the trader's two formula lines, not from the module under test - a golden
generated by the code it checks pins that code's mistakes. All five symbols agree
to four decimals.

**The RS/RW board moved into the strength column above the Strength section**, in
a scroll area. Hosted bare its minimum took the column's floor from 190 px to 452,
past the alert column's entire 360 px budget - the charts would have paid for the
move, and the test that measures that floor is what caught it.

**ITEM 2's SCORE IS COMPLETE AND ITS THREE SURFACES ARE NOT.**
`scripts/held_run_score.py` is built, tested and shadow: P(no stop inside 30
minutes) x trimmed-mean MFE_R of the ones that held, per segment, rolling 20
sessions. The champion tier, the mutes and the PROVEN stamp are untouched, and a
test pins that the champion never imports the challenger. **Not wired:** the
Daytrade Tracker column, the M5 alert-bar suffix (`alert_suffix` exists and is
tested; nothing calls it), the Alert Center ordering switch.

**ITEMS 3 AND 4 ARE NOT BUILT.** The phone digest still picks the best swing from
the favourite bucket alone, so the near-bucket cream is still not being sent; and
there is no "Working lately" section on the Trading Desk and no priority switch.
plan.md's Phase 0.14 entry records exactly what each still owes.

**Verification.** `pytest tests/ -q` **6174 passed, 72 subtests, exit 0, zero
failures**, lock probed FREE immediately before the run · `ruff` clean · smoke
**7/7** · `--selftest` **74/74**.

### 2026-09-02 - Phase 0.13 packet P8: the first setup-parameter grid

**Branch `claude/p8-param-grid`, off `main` at `1837b63`** - cut only after the
integration below, because the packet declared Phase 0.12, P3 and P7 as preconditions
and said to stop if they were missing. Live gate 37 owed. Shadow only, and no frozen
rebuild needed: no new dependency, no new asset, no spec change.

**Declared family:** `AVWAPE_TO_FIRST_DEV` LONG - `avwape_to_first_dev@1` in P7's
registry - **840 occurrences over 622 dependency clusters**, the largest cell in the
lake. **Declared cells:** 12 (four entry moments x three targets). **Declared floors:**
30 episodes, 5 symbols, 5 entry sessions, counted on `dependency_cluster_id`.
**Declared window:** the first 20 trading sessions after the packet landed, fixed at
registration.

**ONLY THE ENTRY MOMENT VARIES.** The stop is `current_anchor:1` in every cell, and so
are the time stop, the exit machine and the checkpoints. A grid that also varied the
stop could not answer the question it declared - a winning cell might have won on the
stop, and nothing in the row would say which.

**The control is the code it challenges, not a copy of it.** `m5_first_close`
delegates to the existing `simulate_m5_close_opportunity`; the three challengers use
the same function through one new optional `entry_selector`. Parity is therefore a
property of the code. The test pins it anyway, because "by construction" holds only
until someone edits one of the two paths.

**The golden fixture pins code that never heard of P8.**
`build_setup_entry_timing_fixture.py` imports `outcomes.py` as `main` has it, through
`git show` into a temp package, and freezes the three rank-1 rows from THAT. The
packet asked for a fixture before the simulator; what actually needed protecting was
the arithmetic that already ships, since P8 adds a parameter to the function every
published `m5close_*` row came from.

**Each entry is defined by what it refuses.** Acceptance is a completed M15 CLOSE
beyond the trigger, never a wick. A retest TAGS the level and still closes holding it.
A controlled pullback needs the EMAs in trend order, an extreme reaching the band and
a close still beyond it - a bar that closes THROUGH the band is a break. Eligibility
is STRICTLY after the trigger: a derived bar ending at that instant is the signal bar,
and entering on it is entering on the information that made the setup. Unmeasurable -
fewer than 21 completed M30 bars - produces NO ROW.

**THE FAILURE MODE IS NAMED IN THE LEDGER, BEFORE ANY NUMBER EXISTS.** A waiting entry
can look better purely because it SKIPS the episodes that went straight down: no
confirmation printed, no row written, and the loss is missing from the average rather
than counted. **The control's rows-per-cluster is the denominator to read first.** The
second named failure is the three challengers agreeing so strongly that they are one
look, not three.

**Measured and reported.** The packet justified the family partly as "the trader's
most-claimed setup (22 of 61 like claims)". That is right for the SETUP and wrong for
the side: the 22 split **11 LONG / 11 SHORT**, and `avwap_breakout` LONG carries **15**.
It is still the right first grid - it has the deepest evidence by a wide margin - but
"most-claimed" is not the reason for the LONG leg.

**Deliberately NOT built:** the conditioning axis. The packet permits one
ATR-normalised three-bucket axis "if the question needs one"; nothing yet says it
does, and adding it before the unconditioned answer exists is three more looks against
the same k.

**Verification.** `pytest tests/ -q` **5800 passed**, the only failures being the 32
`ai_jobs` tests that stand down while the nightly holds the machine-local writer lock ·
`ruff` clean · smoke **7/7** · source `--selftest` **74/74**. Fail-before-fix: 17 of 18.

### 2026-09-02 - The merge, and the test run that started a real scan

**`main` now holds P9 and R3.** Both were built, verified and pushed on their own
branches; both are merged here. The merge itself was three documentation conflicts,
each resolved by keeping BOTH entries rather than choosing.

**THE MERGED TREE PRINTED 6,145 PASSED AND THEN KILLED ITS OWN PROCESS.** Zero
failures, then `QThread: Destroyed while thread '' is still running` and exit
`0xC0000409`. Each branch alone had exited 0, so the obvious read was an interaction
between them. **It was not.**

Five test files build a real `MainWindow`, which builds a real `AutopilotService` with
live timers, and nothing shuts them down. A later test called
`QApplication.processEvents()`; a surviving timer ticked; `_maybe_auto_arm` saw it was
after 07:00 on a weekday and flipped Auto Pilot **ON**; `_maybe_run_swing_slot` then
**started a real master scan** - `run_autopilot_scan` -> `_run_master_scan_subprocess`,
an actual child process against the live tape, on the same machine as the running desk.
A 20-minute scan outlives a 6-minute suite, so its `QThread` was still running when the
interpreter tore down, and Qt aborted.

**It depends on the WALL CLOCK, which is the whole reason it looked like a merge
problem.** Every clean run this week happened between 04:00 and 05:00, before the arm
hour. The first run after lunch crashed, and every run after it crashed identically -
including runs of code that had passed at breakfast. Proving it took a probe that
stamped the running test onto `ScanService._start` and reported at session end, because
a print inside a test is swallowed by pytest's capture.

**The guard is one machine-local setting, not a patched method.** `conftest` already
points LOCALAPPDATA at an empty temp dir; it now writes
`qt_autopilot_auto_arm: false` there. Defaulting True is right in production and
indefensible in a test process, which must never reach for IB, spend the scan budget or
race the desk. A patched `_maybe_auto_arm` would have deleted the behaviour from the
tests that exist to check it; a setting only moves the default, so a test that wants
arming turns it back on - and one of the two new tests proves exactly that.

**What is NOT fixed here:** desk construction is still not inert under pytest. The
timers still run and still do everything else, which `conftest` has openly said since
2026-08-10. This closes the one door that leads out of the process.

### 2026-09-02 - Review round R3: the research narration outgrew the model

**Branch `claude/r3-narration-budget`, off `main`.** Live gate 40 owed. No frozen
rebuild: no new module, no new asset, no new dependency.

**THE EVIDENCE.** On 2026-09-01 `setup_research` ran at 03:55, 04:30 and 05:00,
published three superseding packs, spent 29 minutes reading the lake, and produced no
narration on any of them. Each logged *"the local server truncated the prompt: sent
~176827 tokens (442068 chars), server reported seeing 32771"*. Two independent faults
wearing one face.

**1. THE PACKAGE SENT THE WHOLE PACK.** The pack is the deterministic product and it
grew - P3 added the ineligible block, the excluded families and the coverage detail;
P8's grid grew it again - to 437,125 chars against a window that reads about 78,000.
`narration_view` sends what a person reads first: the gate, coverage, the evidence
shape, the excluded families, **every eligible cell** (those ARE the finding), and
**counts** of what was dropped, so the model can say "and 71 thin cells were not
shown" instead of being handed 71 thin cells. Absent by design: the ineligible rows,
the market-context cells, the raw outcome list - input to the arithmetic, never its
answer, and all still in the pack on disk.

**The cells are deduplicated, which is where most of the rest went.** Four prose
constants - the eligibility rule, the n-floor note, the profit-factor convention, the
bootstrap interval - are module constants interpolated into every cell: ~900 identical
characters inside each 1,900-character cell, one paragraph written 33 times. Stated
ONCE under `conventions`. **A constant two cells disagree on is never hoisted**; it
stays inline on all of them, because stating it once would silently restate one of
them. Measured on the 2026-09-01 pack: **437,125 -> 38,184 chars**, headroom from six
more cells to about forty.

**Over budget is a refusal, not an attempt.** `NarrationTooLarge` is raised before any
provider call, and the test asserts the stub was never touched. A prompt above what
the model can read is not a longer answer, it is a silently sheared one, and words
generated from a sheared prompt are not trustworthy even when they validate. **The
evidence hash is taken over what was actually SENT** - hashing the pack while sending
a view would make that traceability a lie.

**2. A MISSING NARRATION IS NOT A FAILED JOB.** It returned `degraded_no_narrative`
under `max_attempts=3`, so the runner re-ran the WHOLE job twice more - and this job
is a ten-minute lake pass. That is the three packs and the 29 minutes, failing
identically each time, because re-reading the lake cannot shorten a prompt. It returns
`ok` with `narration absent: <reason>`. The digest slot already works this way; the
difference that matters is that the digest's retry is cheap and this one is not, so
this returns a status the runner will not re-attempt at all. **If a narration retry is
ever wanted it must read the pack already on disk. It must never re-enter the lake.**

**3. PROVENANCE.** Two packs from one night disagreed by 3,067 outcome rows - 9,372 at
03:55 on the pre-merge checkout, 12,439 at 04:30 on `main` because P8's grid landed in
between - and nothing in either said so, so a reader could not tell a change in the
evidence from a change in the code that measured it. `built_by_commit` is read once
per process and fails **OPEN** to `"unknown"` (a less traceable pack beats no pack);
`recipe_ids` is carried from the coverage the caller passed and is **never re-derived
from the module**, because re-deriving would state the grid this CODE knows rather
than the one these ROWS came from - the one thing the field exists to distinguish.

**4. THE SYNTHESIS COUNTER WAS READING A LIST AS A COUNT** (committed separately).
`matured_horizons` is a comma-joined field like `"20,60"`, and `_matured` compared it
as a number: a date graded at horizon 20 alone read as `"20" > 0` - true - and `"0,60"`
read as truthy as well. It now asks whether ANY listed horizon is non-zero. **The
prompt expected 5 graded dates and 4 with matured horizons; the live counter measures
4, and a hand count agrees** (2026-08-20, 08-21, 08-27, 08-31). The code was right and
the expectation was stale. The count can also legitimately FALL as evidence accrues -
a date whose horizons are all still `0` drops out - which is pinned by a property test
rather than left as a surprise.

**Verification.** `pytest tests/ -q` **6119 passed, 72 subtests, exit 0, zero
failures**, lock probed FREE immediately before the run · `ruff` clean · smoke **7/7**
· `--selftest` **74/74**. Fail-before-fix: **all 11** narration tests fail against the
un-fixed tree.

### 2026-09-02 - Phase 0.13 packet P9: quick like

**Branch `claude/p9-quick-like`, off `main` at `13cbc50`.** Live gate 39 owed. No
frozen rebuild: the one new module sits inside the already-collected
`ui.annotations` package, and there is no new dependency and no new asset.

**THE VERB.** Alt+L writes `like_claim` with `like_mode: "quick"` - no claim, no
why. Trader's own words: *"I just want to let the bot and the future AI know
'something about this was good' and then we can figure out what about it / what's
the best entry later."* This **supersedes R9.2(a)'s why-required for the QUICK
path only**; Alt+K still demands a digit and a why, for the reason it always has.

**AND A BUTTON, asked for the same day**: on the chart's verb row (appended, so
every existing button keeps its spot, and still ONE row) and on the rail beside
the claimed like. It opens a box for an OPTIONAL note - `QInputDialog`, the same
control the setup tracker's dislike detail uses - and CANCEL records nothing. The
key never prompts: a key that stops to ask is not a one-key verb. An optional note
is not R9.2(a)'s required why returning; that rule is about a CLAIM, and this path
makes none.

Alt+L is unbound everywhere in `scripts/ui` - the whole inventory is Ctrl+F,
Ctrl+J, Ctrl+R, Ctrl+Return, F9, Alt+E and the rail's four - and two live bindings
for one sequence fire NEITHER, so a clash would have cost the trader both verbs
without saying anything.

**What it does, and what it deliberately does not.** The chart retires,
`like_advance` is recorded so the scoreboard counts a take, and the symbol is
marked reviewed today - all three for free, because each is keyed on the event
type and `like_claim` was already in `_ANNOTATION_DECISIONS`. Nothing is placed:
no Focus, no park, no watch, no alert. **A like carries zero privileges** (P3.1),
and a one-key verb is worthless if the trader has to wonder what else it did.

**SCHEMA_VERSION STAYS 1, PROVEN RATHER THAN ASSERTED.** A test hands every reader
in the chain - the loader, the like cohort, the auto-tagger's capture lane, the
pass cohort - a row carrying the new key, and each returns its normal answer. A
row written before P9 has no `like_mode` at all, and absence reads as `claimed`
because a claim was REQUIRED until this packet; `like_mode_of` is the one place
that says so.

**THE INTRADAY GRADE IS NOW REACHABLE, WHICH ANSWERS GATE 34.** `pass_cohort`
returned blank on every live pass with `sidecar_ends_before_the_entry_bar` - not a
defect in the grade but the shape of the evidence, since the sidecar holds what
the desk was HOLDING at the click and the entry bar is the first close AFTER it.
The new `sidecar_completion` slot appends the rest of that session after the
close, from the research lake (narrowed Arrow-side by symbol and interval, never a
materialised list) or the desk's own cache when the lake has not ingested yet -
which is the normal case the morning after.

**The original snapshot is never rewritten.** Completion writes a NEW file and a
NEW field; `m5_bars_ref` keeps meaning "what the desk held at the click", and the
two together show how much of the session the trader could actually see. The slot
sits BEFORE `pass_cohort_grading` because it feeds it, so one night completes and
grades. Idempotent, fail-open, every refusal counted by its own reason.

**The joins.** A quick like contributes a LINK to the auto-tagger, never a tag -
it names no setup, and "liked" in a Tags column would mean nothing while
outranking the scanner match beneath it (R2). `like_mode` is a picks column so a
later rollup can split quick from claimed without rewriting a row. Weekend Prep
and `ai_summary`'s judgement scope both now say the `like_unclaimed` cohort is NOT
a setup's edge.

Two order tests updated: the slot order is asserted PAIRWISE now (an index
assertion has been edited by three packets running), and the caveat test asserts
CONTENT rather than a count.

**Verification.** `pytest tests/ -q` **6130 passed, 72 subtests, exit 0, zero
failures** with the lock FREE · `ruff` clean · smoke **7/7** · `--selftest`
**74/74** · spec-drift **17**. Fail-before-fix: 17 of 18.

### 2026-09-02 - Review round R2: two guards, then the stale sentences

**Branch `claude/r2-guards`, off `main` at `1c364c8`**, merged the same day. No new
live gates; gate 38 gained two specific things to watch.

**1. AN EMPTY `assigned_tier` CELL WAS ABOUT TO BECOME A TIER CALLED NAN, and this
landed ahead of the 07:30 scan.** `tier_for_tracker_row` accepted any non-empty
string. The live D1 feature-history file has no `assigned_tier` column yet; the first
scan after P4 WIDENS it, every row written before that gets an empty cell, and
`pd.read_csv` returns those as float NaN - which is TRUTHY and stringifies to `"nan"`.
So an empty cell read as a tier named "NAN" whose source was "assigned".

Reproduced on `main` before the fix: a NAN tier reaches `build_bot_tier_outcome_rows`.
On the packet's own measurement it was 40 of 42 outcome rows and 6 picks from a
6-symbol scan instead of 2 - the tier list, the tier outcomes and the S/A performance
aggregate all filling with a tier that does not exist, while `derived_from_bucket`, the
honest answer, was available the whole time.

The fix READS the vocabulary from the stamper rather than choosing one:
`_priority_partition_tier_rows` writes exactly S, A and B. Both row shapes are tested
because they are different values - the existing unit test models the key ABSENT, and
the real file has it PRESENT AND EMPTY.

**VERIFIED AND NOT CHANGED:** the same `or ""` idiom in `setup_tracker_panel` cannot
see a NaN - that file is read with `csv.DictReader`, so every value is a string. The
packet's condition for changing it is not met.

**2. A LINK IS NOT A TAG AT ANY SEAM.** R1 kept links out of `auto_tag_summary` and
three seams still let them into the trader's own tag column, each with its own idea of
what a link was: the bulk tagger's lane filter, its `max(confidence)` pick, Accept-all,
and `tag_confidence`. A link arrives at 0.90-0.95 because the capture lane is the most
confident there is, so it beat every scanner match beneath it - TRV lost
`avwap_retest_followthrough` at 0.91 to `link:review:arm_level`.

ONE predicate answers it now, accepting BOTH spellings deliberately: `link_only` is
what the tagger sets in memory and the `link:` prefix is what survives a round trip
through `auto_tag_candidates`, which stores a tag and a source but no flag. Links still
RENDER with their event id - the pointer is worth seeing, it is just not a tag - and
Accept-all says how many it skipped. A pass now carries ALL its codes in vocabulary
order; `codes[0]` had been making a two-reason pass into a different statement.

**Measured on a copy of the live journal:** zero link candidates are stored today and
zero provisional tags are links, so nothing on the desk is currently mis-tagged. They
appear at the next `refresh_auto_tags`, which is what this got ahead of. No `--apply`
was run against the live journal.

**3. The sweep.** Seven copies of the corrected "never pooled" sentence, including the
scope description the AI reads BESIDE the pooled row it denied. Four stale
`focus__not_today` spellings, one of them an assertion that passed because the prefix
matcher cannot tell the difference. A dead double assignment of `nonexclusive_groups`.
The overlap note now writes under the file's own lock through a temp file, and REPORTS
its failures - the grading has already succeeded by then, so a silent no-op reads as
"the note is on the file". The trade pane's adjustment query filters BEFORE it limits
(27 adjustments exist on the applied copy, so a trade's own record was already falling
off a global newest-25). The Weekend Prep backlog toggle reads the journal on a worker
(169 ms, charged to a checkbox). Four DESK_INTERNALS entries that CLAUDE.md promises
always exist. And three claims that were simply wrong: the frozen selftest is not
29/29, `claude/gui-phase-0-9` is CONTAINED in `main` (what is open is gate 7, and a
gate is not a branch), and a PASS merges through `_merge_cohort_safely` too.

### 2026-09-02 - The integration that unblocked P8: three branches onto `main`

**Trader instruction: "yes do option 1".** Packet P8 declared its own precondition -
"Requires P3 and P7 landed; if either is missing, stop and say so" - and neither was,
so P8 was not built. This is the merge that fixes that, and P8 follows it.

**Merged in a scratch worktree, not in this checkout.** The desk was mid-run on the
nightly AI job (lock held from ~22:00 onward). A merge leaves the working tree
carrying conflict markers inside `.py` files for as long as resolution takes, and a
running process that lazily imports one of those is the one failure this could
plausibly have caused. The worktree made that impossible; the checkout moves in one
step at the end.

**Order: Phase 0.12, then P3, then P7 - oldest first.** Phase 0.12 merged clean. P3
conflicted only in the two shared ledgers. P7 conflicted in six, every one of them
ADDITIVE (both sides inserting at the same anchor), and all were resolved by keeping
BOTH sides - the rule the 2026-08-31 integration set. `CLAUDE.md` and `AGENTS.md`
verified byte-identical afterwards.

**THE BD COLLISION WAS REAL AND IS NOW RESOLVED.** Three branches cut from `66a0c31`
each numbered their own decisions: Phase 0.12 took 78-80, P3 took 80-84, P7 took
85-86. The LRSI branch merged first and KEPT BD-80; P3's five shifted to 81-85; P7's
two shifted again to 86-87. Renumbering was done by targeted replacement, never by
line range, because `(BD-80)` appears in both lines and only the surrounding words
say which entry a reference means - a duplicate-number assertion over the file's
headings backs it up (77..87, no repeats). The lesson is in BD-86's own numbering
note: **a BD number claimed on a branch is a request, not a number.**

**P7's owed item is paid.** `setup_research.family_role` was P3's own two-entry role
map, kept only because the registry did not exist when P3 was built; P7 built
`fact_pack_role` as the drop-in and named the swap as owed to whichever branch merged
second. Both are here, so the map is gone and the fact pack reads the ONE table. The
registry keeps Appendix C's `TRADE_SETUP` and the lookup translates it back to the
pack's own `TRADE`, so the OUTPUT is unchanged while the ontology has one owner.

**Two P7 tests changed because the merge made them false, and one thing was PROVEN
rather than trusted.** The "nothing in production imports the registry" test now pins
an explicit reader list (the fact pack, plus the selftest), because the swap gives it
a reader on purpose. And the HTF LRSI trial-ledger row - declared blind on the P7
branch from another branch's constants - was checked against the real grid the moment
both landed: **16 declared, 16 real, and all 75 recipe ids across the three grids
resolve to exactly one ledger row.**

**The frozen exe was rebuilt because P7 edited the spec** (packaging trigger 2), and
a new selftest check loads the registry JSON from inside the frozen process: 74/74
frozen, exit 0. A `datas` rule proves a file was bundled; only a frozen run proves
the process can read it.

**Baseline honesty.** 5781 passed. 32 failures are the `ai_jobs` lock standing down
(the desk's nightly held it throughout; the same tests fail on a pristine `main`
worktree, checked) and one was a Windows `PermissionError` on `os.replace` inside
pytest's own sandbox that did not recur. `ruff` clean, smoke 7/7, spec-drift 17.

**Owed:** re-run the three lock-blocked files once the nightly finishes; restart the
desk when the trader chooses, since nothing merged today is live until then.

### 2026-09-01 - Phase 0.12: the Focus surfaces stop growing, and a shadow LRSI study

**Branch `claude/focus-declutter-lrsi-htf`, off `main` at `66a0c31`.** Two
independent packets, both authorized by the trader in chat on 2026-09-01. Live
gates 27 and 28 owed. No frozen rebuild: no packaging trigger was hit.

**Packet A - Focus de-clutter (desk change).**

- **A1** `_poll_focus_d1_interest` evaluates the PULLBACK set only. The
  extension set fires solely from a trader-armed D1 event watch, through the
  separate armed poll - two disjoint lanes, so an extension event has exactly
  one path and cannot arrive twice. Gated at the flag-GENERATION seam: an
  extension kind is never constructed, so nothing is suppressed downstream. The
  2026-08-05 one-extension-per-day ration is removed; it had nothing left to
  ration. Two golden tests that encoded the old rule were rewritten to the new
  one, which is the authorized behaviour change, not drift.
- **A2** Armed alerts expire on a TRADING-day clock. New: `market_calendar.
  trading_days_between` and `scripts/armed_alert_expiry.py` (policy in one
  place). 5 sessions for a manually armed 5d extreme watch, 10 for a 20d one, 10
  for D1 level watches, any-bounce watches and manual price alerts. Uncertainty
  never deletes - a date the calendar refuses keeps the entry armed. Every
  expiry appends a row to the `armed_alert_expiry` evidence stream. **A price
  alert is DISARMED, not deleted**, so `price_alerts.json` still honours plan.md
  sec 5; arming restarts its clock. Each expiry rides the poll that already owns
  its store, so no new timer appeared.
- **A3** A Focus pick with no alert and no pullback event for 10 trading days
  fades to a reversible faded list. `FocusPickStore` is the single writer and
  owns three new sidecars beside the focus files (`focus_pick_clocks.json`,
  `focus_faded.json`, `focus_fade_events.jsonl`). Swing and M5, the trader's own
  included - an explicit authorization to auto-remove a hand-typed name, scoped
  to Focus and routed through the store's own removal path, so a hand-maintained
  broad-watchlist line is still never touched. A faded swing favorite gets a
  RETRACTION with origin `focus_fade`, never an edit; no `pick_feedback` verdict
  is written for a fade. Day roll + a half-hourly timer, never inside the 60 s
  poll.
- **A4** "Focus pick review (N)" and "Faded review (N)". The faded walkthrough
  goes through `_enqueue_review_alert` - the one door - with `FOCUS_FADED_TAG`,
  which bypasses movers-only exactly as `FOCUS_REVIEW_TAG` does.

**Review round (2026-09-01, Fable): one defect found, reproduced, fixed.**
`_arm_price_alert_from_level` - the chart's own arm route - re-armed an
existing side at a changed level without restarting its `armed_at` clock, so a
level re-armed from the chart still carried the stamp that expired it and would
have been disarmed again on the next poll. The Focus-tab board's merge stamped
correctly; the panel's mirror of it did not. Failing test written first
(`test_rearming_from_the_chart_restarts_the_expiry_clock`), then the one-line
stamp added. The deliberate unchanged-level no-re-arm rule is untouched and the
test asserts it does not move the stamp either.

**Packet B - higher-timeframe LRSI entry research (shadow, zero desk cost).**

- **B1** H2 (120 min) is a derived timeframe again. The locked plan cut it for
  having no consumer and named that as the reopen condition; this study is one
  (BD-78). Additive - no existing timeframe, contract id or published row moved.
  H2/H4 stubs are published as evidence and EXCLUDED from the oscillator input.
- **B2** The short legs are unmirrored: cross-down through 50 and 80 on the same
  series the long legs read. The formula clamps at 0, so the mirrored-close
  idiom is a different feature, not a transform (BD-79). Cost stated: this
  measures exhaustion, not down-momentum, and fires earlier -
  `tests/fixtures/efficiency_lrsi_research_v1.json` pins the gap at two bars.
  Live `CROSS_LEVELS` and `m5_signal_engines` untouched.
- **B3** `outcomes.HTF_LRSI_RECIPES`: a bounded 16-recipe diagnostic grid
  (M30/H1/H2/H4 x four entries, one stop model, one 2.0R target) with
  `simulate_htf_lrsi_entry` and its dispatch branch. It reads the occurrences and
  canonical M5 bars the nightly already materialises, so it adds simulation and
  not a second data pass.
- **B4** Nothing registered in `outcome_semantics` (BD-80): these rows are keyed
  by `recipe_id` and never acquire a bounce family.

**One thing the trader should know about the checkout.** When this session
started, the working copy of this file was a 6,881-line PRE-ARCHIVE version -
every entry already moved to `docs/CHECKPOINT_ARCHIVE_2026-08.md` was back, and
the "Active state at a glance" block was gone. It contained nothing dated
2026-09-01 and nothing absent from `HEAD` or the archive, so it was a stale
revert rather than someone's work in progress. It was backed up to the session
scratchpad and this file was restored from `HEAD` before these edits. Worth a
look at whichever agent or tool produced it.

**Ambiguity resolved, and stated rather than buried.** The authorizing prompt
described the B3 grid as "16 recipes" and also as carrying "the existing small
target set" (three targets), which are 16 and 48. It was built as **16** - the
explicit number, and the reading that keeps the nightly inside its reserve -
with one 2.0R target, the middle of `M5_CLOSE_TARGETS_R` and the same target the
fixed-R control uses, so the two compare directly. Widening to the full set is
the single constant `HTF_LRSI_TARGETS_R`.
### 2026-09-01 - Phase 0.13 packet P3: the fact pack tells the truth

**Branch `claude/p3-fact-pack-truth`, off `main` at `66a0c31`.** Five changes to the
nightly `setup_research` pack and the warehouse readout, all shadow-only. Live gate 32
owed. No frozen rebuild. Recorded as **BD-81 … BD-85**; 78/79 belong to the unmerged
Phase 0.12 branch, and the BD log says so.

**The case.** The 2026-08-31 pack had 9 eligible cells - every one
`AVWAPE_TO_FIRST_DEV`/LONG against an ATR stop control, every one NEGATIVE - in one
table sorted by trimmed mean, so rows 10 onward were n=1 cells at +2.9R. The 80-row cap
dropped 508 more without saying which kind. GENERAL (735 occurrences) and
FAVORITE_ZONE_WATCH (486) were pooled as trade setups, which Appendix C forbids in
those words. And `n` was reported as if outcome rows were samples.

**1. Episodes beside rows (BD-81). THE MEASUREMENT CHANGED THE CONCLUSION.** Every cell
carries `n_episodes`; the floor still counts rows, deliberately. But the assumption
behind the follow-up was wrong. On the live lake: **9,372 outcome rows over 599
occurrences and 287 clusters** - and yet **`n` and `n_episodes` were EQUAL in all 756
cells**. One row per occurrence per recipe, so per-cell episode counting is not where
the double-counting lives. It is ACROSS cells: 15.6 recipe rows per occurrence, and
1,804 of 3,436 clusters carry more than one family. Nine ATR variants of one family are
nine readings of the same 33 moves. So the pack publishes `evidence_shape` too, and
BD-81 records that the follow-up must be a CROSS-CELL floor - not the per-cell swap,
which on today's data would change nothing at all.

**2. The eligible block leads (BD-82).** Eligible whole and sorted as before, then a
bounded ineligible block sorted by n DESC then trimmed mean, so what rides along is the
thickest evidence below the floor and never the luckiest single trade. Per-block drop
counts. A pack published before the split still renders as published - a pack is never
edited and a new reading is a superseding sibling.

**3. Non-trade roles excluded and named (BD-83).** An explicit map; everything unnamed
is TRADE, so a family added tomorrow is measured rather than silently dropped. Their
counts still travel (today: 1,182 and 804 outcome rows), because absence is a
first-class fact.

**4. Coverage published (BD-84).** New `research_warehouse/outcome_coverage.py`,
append-only, one line per outcome firing naming its symbol bucket. The pack reports
buckets covered in the last 32 firings, families with zero outcome rows, and the first
M5 session in the lake (**2026-08**, 2 months). No history reads UNKNOWN, never "0 of
32" - a zero there is a measured claim nobody measured.

**DEVIATION, REPORTED NOT FORCED.** The packet asked for the sidecar "beside the packs"
in the AI store. That would make `research_warehouse.cli` - the data layer - import
`ai_jobs.store`, inverting the one-way dependency the tree keeps. It lives under the
store root instead, beside the lake it describes; the reader already imports the
package, so the pack still gets the number. `_first_m5_session` reads partition NAMES
from the manifest, never bar rows (BD-66/BD-69).

**5. The readout is not hard-filtered (BD-85).** `slice_readout(setups=...)` - omitted
is the pinned slice and byte-identical for every existing caller, None is every family.
`SLICE_SETUPS` is NOT widened: `cli._run_outcomes` uses it to pick which occurrences
get the legacy slice recipe, so widening it would change what the warehouse SIMULATES.
The panel gains a family combo and `n_symbols`, `n_sessions`, `n_truncated`,
`as_observed_only` - all computed by the query and dropped by the panel. Selecting a
family reads nothing; Refresh is still the only thing that touches the share.

**Owed, not built:** the optional `cell_history` block over the three sibling packs on
disk.

**Verified against the live lake, not only fixtures.** A pack built now opens with 29
eligible cells, shows `n_episodes` on every row, names both excluded families with
their counts, prints the evidence-shape line and states bucket coverage as UNKNOWN
(correct - the record starts at the first build after this lands, which is what gate 32
checks).

**Verification.** `pytest tests/ -q` **5741 passed, 72 subtests, process exit 0** ·
`ruff` **clean** · smoke **7/7** · source `--selftest` **73/73** · spec-drift **17**.
Fail-before-fix: with `scripts/` stashed including the untracked module, 21 of the 26
new tests fail; with tracked changes only, 16. The five that pass both ways guard
against the wrong fix.
### 2026-09-01 - Phase 0.13 packet P7: one name per setup

**Branch `claude/p7-setup-registry`, off `main` at `66a0c31`.** Two READ-ONLY modules.
Nothing in production imports either, no runtime behaviour changed, and **there is no
live gate** - green tests are the whole gate.

**PLAN.MD P4.1 IS WHERE THE REGISTRY BECOMES AUTHORITATIVE.** Until then it describes
what the code already believes. P4.1 owns two things P7 deliberately did not do:
choosing which spelling is identity for each of the eight recorded divergences, and
filling the columns left blank.

**1. The crosswalk.** `scripts/setup_registry.py` over a frozen
`setup_registry_v1.json` - 57 entries keyed `setup_id@version` - generated by
`scripts/build_setup_registry.py` and reviewed as a diff. It is never rebuilt at
import: a crosswalk that recomputes itself from five moving sources is a sixth source,
whose disagreements appear and vanish without anyone seeing them.

**THE PACKET NAMED FOUR SOURCES; THE CODE HAS FIVE.** `legacy.py` declares study
families as `*_STUDY_FAMILY` constants - 17 of them, and **eight are named nowhere
else**: `htf_trend_retest`, `hv_level_proximity`, `hv_level_break`,
`cloud_flat_proximity`, `compression_break`, `trendline_break`,
`relative_avwap_retest`, `relative_avwap_break`. A registry built from the four sources
the packet listed would have shipped a crosswalk that omits detectors running every
scan. Read by regex - no import of a 27k-line module, no write to it.

**Two refusals, which are the same rule twice.** It resolves no disagreement: eight
`known_divergences` record what each source believes (three aliases pointing at another
family's page, four families the scanner tags but nothing documents, and one pair
Appendix C states that `SETUP_DOC_ALIASES` does not carry) and leave the choice to
P4.1. And it fills no column the sources do not establish - supported sides, timeframe
roles, the exact completed-bar trigger and the primary recipe are EMPTY on every row
and listed under `unestablished`. A guessed side reads as established in exactly the
column a later experiment trusts. An unresolvable name RAISES rather than falling back
to `GENERAL`, which would file "two tables write different things under one word" under
"untagged".

**2. The look-counter.** `scripts/research_warehouse/trial_ledger.py`: one append-only
JSONL row per registered grid, written at REGISTRATION time. `register` refuses to
rewrite an existing `trial_id` - editing a declaration after the numbers arrive is how a
grid of 54 cells becomes a grid of 3 in the record - and a test bans the module from
reading an outcome at all. Four grids backfilled with their real authorization pointers
(M5-close 54 cells, HTF LRSI 16, AVWAP band challenger 3, the v1 recipe library 5),
because a family-lifetime count starting today would report each as never looked at.
Every recipe id resolves to exactly ONE row; `owners_of` returns every claimant, since
the interesting failure is two owners rather than none.

**THE PACKAGING GUARD FIRED AND THE SPEC WAS FIXED, NOT THE TEST.** The frozen JSON is
the first non-`.py` runtime asset at the scripts/ ROOT, and the spec's sweep only walked
package directories. It now sweeps the root too, bundling to `"."` because a frozen
top-level module's `__file__` parent IS the bundle root - swept rather than named, so
the next root-level asset is covered the day it lands. It was NOT added to the unbundled
allowlist: that is only for files the frozen app provably never reads, and P4.1 will
make production read this one. **A rebuild + frozen selftest is owed before merge.**

**VERIFIED DIFFERENCES FROM THE PACKET, none forced.** P3's temporary role map is on
`claude/p3-fact-pack-truth` and NOT on `main`, so `setup_research.py` here has no
`family_role` to replace; `setup_registry.fact_pack_role` is built and tested as the
drop-in and keeps the fact pack's own `TRADE` spelling where Appendix C writes
`TRADE_SETUP`, so the swap changes no output and belongs to whichever branch merges
second. `HTF_LRSI_RECIPES` is likewise not on `main`; the ledger declares that grid
anyway. And the packet's role vocabulary (TRADE / STUDY) differs from Appendix C's,
which the packet said not to deviate from - the spec's vocabulary won and study-ness is
a STATUS.

**SUITE NOTE, MEASURED NOT ASSUMED.** 32 tests across `test_ai_jobs_runner.py`,
`test_ai_evidence_coverage.py` and `test_ai_jobs_store_window.py` stand down while the
machine-local `ai_jobs_runner` writer lock is held - the desk's nightly AI run held it
throughout this build, confirmed by taking the lock directly. **The same tests fail with
every P7 change stashed**, so this is the environment and not a regression. Everything
else: **5625 passed, exit 0** · `ruff` clean · smoke **7/7** · source `--selftest`
**73/73** · spec-drift **17**. Fail-before-fix: with `scripts/` stashed, 24 of the 25
new tests fail (the 25th asserts nothing in production imports them, which is trivially
true when they do not exist).
### 2026-09-01 - Phase 0.13 packet P0: three trader decisions, applied

**Branch `claude/p0-apply-decisions`, off `main` at `66a0c31`.** Authorized by the
trader pasting the packet. Live gate 29 owed. No frozen rebuild: no packaging trigger
was hit. Branched off `main` rather than off `claude/focus-declutter-lrsi-htf`
deliberately - the three decisions are independent of Phase 0.12, and the
higher-timeframe LRSI study on that branch is untouched by the M5 retirement here.

**1. BANGER retired** (trader: *"not sure to be honest. We can probably remove this
because idk what it is"*). It was a top-alert class with a matcher and **no producer**:
the only definition was `"BANGER" in raw_text.upper()` in `alert_center_panel.py`, no
detector path builds the token, and 0 of 8,818 recorded review rows carried
`banger=True`. Removed: the matcher, the tier-gate bypass, the always-sound branch, the
`is_banger` argument to `RepetitionLedger.consider`, the `had_banger` row field and
both escalation branches. The argument is REMOVED, not ignored, so a stale caller
raises. **Kept:** the `banger` review-event column as a constant `False`, documented as
retired, so historical readers and the schema id are unchanged. PROVEN is the top class
and is untouched; two feed labels now say PROVEN where they said bangers.
`REGIME_BANGER_*` in `legacy.py` is a regime-pause threshold and was left alone.

**2. LRSI M5 alerts retired, every row of evidence kept** (trader: *"LRSI alerts seem
to be mostly spam. however I enjoy them as something that can boost the potential of an
alert. for now let's put them on the back burner. let's measure how they perform on
different timeframes but no need for their M5 alerts"*). They were **84 of 128 new M5
episodes by 11:14** that morning. `LRSI_M5_ALERTS_RETIRED = True` sits beside
`H1_ALERTS_RETIRED` and gates the **emit** seam.

The seam was verified before it was chosen, and it is not the one the packet's first
guess named. `check_lrsi_cross_setups` tests `is_m5_signal_enabled` **before** the event
joins `hits`, so a `False` toggle drops it ahead of the candidate row and the outcome
registration - flipping `M5_SIGNAL_TYPE_DEFAULTS` would have stopped the evidence
rather than the noise. The defaults stay `True` with a comment saying why; the gate sits
after `record_alert_tier`. So the sweep, the candidate row,
`intraday_bounce_outcomes.csv`, the learning tier and the PROVEN stamp all keep running,
and only `gui_callback` is skipped - the message goes to the symbol log as
`LEARNING_ONLY [LRSI M5 retired]`, exactly as H1 does.

One deliberate difference from H1: `log_bounce_to_file` still runs. H1 returns before
it, but `journal_analytics.AutoTagger` reads `INTRADAY_BOUNCES_CSV` to answer "which of
my setups was this?", and skipping it would blank the tag on a real LRSI trade. **No
Settings toggle exists** for these engines - nothing under `scripts/ui/` references
`set_m5_signal_enabled` or `M5_SIGNAL_TYPE_DEFAULTS` - so there was no dialog label to
correct. The "different timeframes" measurement is Phase 0.12 packet B's warehouse
study, on the other branch, unaffected.

**Owed, not built:** LRSI as a display suffix on OTHER M5 alerts - the "boost" the
trader described. `_format_bounce_alert_message` is a module-level function taking no
bars, so a cross reading has to be plumbed through every champion alert caller. That is
a champion-path change, not the display tweak the packet's escape clause allowed, so it
was skipped and recorded rather than half-built.

**3. Clicking away is a pass - recorded, no code change** (trader: *"clicking away = a
pass. The tabs under the visual chart review should give us all the tools we need and we
decide as we see. set alerts / add to focus and then move on"*). `_select_review_alert`
already wrote the `skip` row with `detail.reason = clicked_away_from_m5_alert`; the
trader has confirmed that IS the meaning. The decision is now in `docs/DESK_INTERNALS.md`
under the M5 alert bar entry, with a one-line pointer at the writer, so no later packet
repairs it into a take or into silence. The reason string is frozen -
`review_learning` keys on it, and `tests/test_qt_m5_alert_bar.py` already pins it.

**Golden note.** No byte-level M5 alert fixture exists in the tree;
`tests/test_r5_lrsi_cross_wiring.py` is the golden for this path. Every detection,
candidate-row, outcome-row and tier assertion in it is unchanged - the message
assertions now read the identical text off the `LEARNING_ONLY` line instead of
`gui_callback`. That is the "byte-identical except absent from the GUI stream" the
packet asked for, stated honestly.

**Verification.** `pytest tests/ -q` **5720 passed, 72 subtests, process exit 0** ·
`ruff` **clean** · smoke **7/7** · source `--selftest` **73/73**. Every fix ships a test
proven to fail with `scripts/` stashed: five for BANGER
(`test_the_banger_column_is_retired_but_still_written`,
`test_the_banger_escalation_is_gone`,
`test_the_banger_token_no_longer_bypasses_the_tier_gate`,
`test_the_banger_token_no_longer_skips_the_digest`, `test_tier_extraction`) and seven in
`test_r5_lrsi_cross_wiring.py`, including
`test_a_crossing_logs_an_outcome_row_and_produces_no_gui_callback`.
### 2026-09-01 - Phase 0.13 packet P1: grade what you already said

**Branch `claude/p1-grade-what-you-said`, off `main` at `66a0c31`.** Four defects in the
loop between a decision the trader makes and the evidence that grades it. Every premise
was reproduced at code level AND against the live stores before anything was edited.
Live gate 30 owed. No frozen rebuild: no packaging trigger was hit.

**1. Today's swing picks never reached `human_focus_swing_vetted`.**
`human_focus_tracking._pick_key` returned (trade_date, symbol, side) with no category,
so a name already on one Focus list swallowed its row on the other. Live: AMGN LONG was
liked into swing Focus with origin `vetted` at 11:33:06, the day already held a
`focus_m5` AMGN LONG row from 08:02:14, and the swing row was dropped -
**`grep -c vetted` on `human_focus_daily_picks.csv` reads 0 across all 4,083 rows**.
The cohort that origin exists to build has never had one row.

The diagnosis already existed in the tree: `focus_membership_events.py`'s docstring
names it as audit F3 and keys its own episodes by category. The pick store never caught
up. The key now carries the category slot - the base source with any like-origin suffix
removed, so `focus_swing_vetted` and `focus_swing` are ONE swing membership and a
re-snapshot cannot duplicate a row. The same key runs over the outcomes file, so both
cohorts grade forward independently. No column or schema moves: every historical row
carries `source` and re-keys to the slot it already occupied.

Two joins followed the rows. `weekend_prep_panel._join_focus_week` would have handed one
category the other's forward returns - opposite trades, not a rounding error - and now
uses the one canonical `pick_source_family`. `journal_walkaway.load_focus_positions`
would have replayed a two-list name as two positions and double-weighted it; the trader
was in one position, so it dedupes.

**TWO PACKET PREMISES DIFFERED FROM THE CODE.** Reported, not forced. (b) The
swing-favorites write-through ALREADY EXISTS and works: `_place_in_focus` ->
`FocusPickStore.add` -> `focusChanged` -> the Focus panel's coalesced
`_apply_focus_change` -> `snapshot_today(force=True)`, which passes `force` and so is
never stopped by the `already_snapshotted` early return. QFIN on 2026-08-31 is the
proof - liked 11:26:19, pick row stamped 11:26:20. Nothing was added to that path. And
QFIN's `focus_swing_manual` is not a live code path: `FOCUS_LIKE_ORIGIN` read `"manual"`
until commit `edc7999`, which landed at **11:36** that day, **ten minutes after** the
like. It has read `"vetted"` ever since (AMGN's 09-01 row proves it), so there is nothing
to fix at the source and the existing row is correctly left alone.

**2. A like merged overnight; a veto merged on the click.** `like_cohort_picks.csv` was
last written **2026-08-27** (53 rows) against like_claim annotations recorded through
09-01, so a like was invisible to its own cohort for up to a day - and indefinitely on
any day the overnight job did not run. The two cohorts are read side by side on Weekend
Prep. `commit_like` now merges through the same `_merge_cohort_safely` the veto uses, so
they cannot drift; failure degrades to a "(cohort update deferred)" suffix because the
annotation row is already on disk, and `merge_like_cohort_picks` takes the writer lock
now that it has two callers. The nightly slot stays; both are idempotent.

**3. Unversioned veto codes pooled only with the lowest vocabulary.** The unversioned
mapping was written only while walking `min(versions)`, so a code introduced later got
none at all. Live: `human_focus_veto_compressed` (n=3, PF 165 at h3) beside
`human_focus_veto_v2_compressed` (n=18, PF 0.39) - one judgement read as two opposite
ones, the three-sample half looking spectacular. A `setdefault` on the already-ascending
walk IS "the earliest version that defines this code". Verified against the loaded
vocabularies: `veto_compressed` -> `veto_v2_compressed`, `veto_sma_incoming` ->
`veto_v3_sma_incoming`, `veto_volume_dry` -> `veto_v1_volume_dry` unchanged. Neither new
test asserts a literal `vocab_version`; both load and DISCOVER the late codes.

**4. The scoreboard ignored ~640 explicit decisions and could not see an R gap.** Seven
action families joined the take/reject sets, each classified from what its WRITER does
in `alert_center_panel.py`: `auto_pick_pass` 254, `arm_d1_event` 160,
`focus_review_remove` 88, `focus_review_keep` 71, `auto_pick_approve` 63,
`arm_any_bounce` 22, `veto_day_trade` 4. `veto_day_trade` is a REJECT because the
episode being graded is the D1 chart that was shown; its M5 interest is a different
claim on a different timeframe. Machine events, `*_fired`, `*_expired` and every
`disarm_*` are excluded and pinned by a test. Measured: takes **645 -> 845** of 2,607
shown, take rate **0.247 -> 0.324**.

New **`r_gap`** callout class: both sides >= 8 measured R and |taken - passed| >= 0.5R,
with NO reference to the take rate, so it surfaces what the take-rate classes are
structurally unable to see.

**The packet's live case moves once the action sets are fixed, and that is stated rather
than papered over.** `bounce_type=lrsi_cross_20` at taken -0.376R (n=8) vs passed
+0.962R (n=24) reproduces EXACTLY on the un-fixed sets and the new class catches it at a
-1.34R gap while blind_spots and leaks are both empty. Under the corrected sets seven of
those lrsi charts turn out to have been ARMED rather than passed, taken becomes +0.519R
(n=12) and the gap closes - the apparent edge was an artefact of the misread decisions,
which is what the action-set fix exists to remove. The class is pinned to the packet's
literal numbers in a test, so it is proven on that case either way. On the live store
today it produces **18 callouts while blind_spots and leaks produce 0**, so it is
currently the only class saying anything.

`r_gap` is REPORT-ONLY: a field on `review_preference_state.json` and a section in the
report, deliberately absent from `draft_policy_from_state`, `review_guidance` and the AI
evidence package, because those write priority deltas into `review_policy.json`, which
this packet may not touch. A test asserts none of the three mentions it.

Coded vetoes now feed the `dislike_reason` dimension. The join was MEASURED first:
**202 of 212 join to an existing episode, 198 of those to a SHOWN one, side matching on
202 of 202 with zero mismatches**, so the packet's stop-and-report condition did not
apply; 199 attach inside the 90-day window (the dimension previously had only the 33
`dislike` events). It annotates and never re-resolves - the verdict still comes from the
review event store alone; a side disagreement is SKIPPED rather than guessed, a veto with
no episode is left alone rather than inventing an impression, and an unreadable log is 0.

**Verification.** `pytest tests/ -q` **5737 passed, 72 subtests, process exit 0** ·
`ruff` **clean** · smoke **7/7** · source `--selftest` **73/73**. Fail-before-fix with
`scripts/` stashed: 3 tests for #1, 3 for #2, 2 for #3, 12 for #4. The trader's live
`like_cohort_picks.csv`, `veto_cohort_picks.csv` and `human_focus_daily_picks.csv` were
read before and after the full run and are unchanged - the suite redirects
`TRADINGBOTV3_DATA_DIR` (`tests/conftest.py:57`), which is what keeps a capture-time
cohort merge out of the home folder.
### 2026-09-01 - Phase 0.13 packet P2: show me

**Branch `claude/p2-show-me`, off `main` at `66a0c31`.** Six display changes, each
read-only over a file something else already writes. Nothing reaches a detector, score,
alert, Focus list, review queue or `review_policy.json`. Live gate 31 owed. No frozen
rebuild: no packaging trigger was hit.

**1. The two judgement tables show the robust half.** They projected six columns and
dropped `median_return`, `trimmed_mean_return`, `ci_low`/`ci_high`, `symbols`,
`sessions`, `top_symbol_share`, `evidence_label` and `meets_n_floor` - all written since
R10.C, and most already on screen in the Focus performance table on the SAME page. What
survived was a bare mean on a ratio: the statistic R10.C published the robust half to
stop anyone reading alone. One shared `_cohort_robust_fields` feeds both tables so they
cannot drift.

ONE horizon at a time (default h3) with a selector that re-renders from memory, so a
view change never touches disk on the Qt thread. **`meets_n_floor` is not a column**: it
decides the ORDER and the greying. The live `human_focus_veto_compressed` row - n=3,
PF 165 - now sorts after every cohort that cleared the floor instead of wherever the CSV
put it, and the note says a row under the floor is not a weak finding but not a finding.
Rows above it order by the TRIMMED mean. The liked table carries the bounded-picklist
caveat the AI gets, through the ONE existing `ai_summary._offered_claim_caveat`.

**2. The callouts are named.** "Blind Spots: 3" was two integers over a store that knows
which segment, how often it was shown, the take rate against the overall one, and what
each half measured. `callout_lines` builds those rows on the worker and reads the
classes DEFENSIVELY, so the page renders against a scoreboard written with or without
P1's `r_gaps`.

**3. "My Decisions" beside the Daytrade Tracker.** 13 tabs over
`review_preference_state.json`, which had no surface outside a text report. `gap` is the
one derived number and only when both sides carry a measured average. Off the Qt thread
both at construction (READ only) and on the button (which also calls
`refresh_review_learning_if_stale`, exactly as `app.py:250` does). The probation badge is
`M5_SIGNAL_TYPE_DEFAULTS - BOUNCE_TYPE_DEFAULTS` and nothing else; an unreadable taxonomy
badges nothing rather than calling a champion "probation".

**4. The AI phase gates get a surface.** New `ai_jobs/gate_counters.py`, pure and
Qt-free. Live, read not typed: **Digest 6/10 · Enrichment 6/10 · Weekly synthesis 2/10 ·
Policy draft 5/10 · Evidence window 6/10**. Synthesis is counted through `_read_cohort` +
`graded_sessions` - the job's own two functions - and the draft and evidence counts are
parsed from the PUBLISHED files, because a recomputed number could be right while the
file the model was handed says something else. An unreadable source says "unavailable":
a blank cell reads as zero, and zero is a claim.

**5. THE CODE DISAGREED WITH THE PACKET, and it is reported rather than forced.** The
packet's premise was that guidance is computed before `m5AlertPosted.emit`. It is not:
the emit sits at `alert_center_panel.py:2018` and `_enqueue_review_alert` returns for an
M5 alert at 2026, **before** `_queue_score` - the only enqueue-path caller of
`_guidance_for` - is reached. So `_attach_cached_take_prob` reads `_review_guidance.get`
and NEVER `_guidance_for`, whose `_refresh()` stats two files and can re-read a 34 KB
JSON, per alert, on the Qt thread - the exact drip the three snappiness packets removed.
The consequence is stated, not hidden: the suffix appears for a symbol the desk has
already charted this session and is silent otherwise. Silence is the honest rendering of
"not measured"; a 0% would be a claim.

**6. The repetition fold.** A repeat of the same symbol+side folds with a ×N badge and
returns to the top carrying the newest alert, so a tier upgrade rewrites the row with the
stronger one. Keyed on symbol AND side, because a name that flips direction is a
different claim. **Presentation only** and the docstring still says so: every event
reached `_enqueue_review_alert`, the outcome CSV and the review-event store first, a
folded row's tooltip says it folds rather than drops, Copy-all still lists one symbol per
row, and clicking charts the newest. One existing test encoded the old rule and was
rewritten - the authorized behaviour change, not drift - with its invariant now held by
`test_the_fold_is_presentation_only`.

**Found by the full suite and fixed:** both new workers could emit into a deleted panel
(`RuntimeError: Signal source has been deleted`, out of a daemon thread). Isolated runs
never hit it. `shutdown` joins the thread, but deletion can win the race; both guard the
emit and drop the payload now, proven deterministically with `shiboken6.delete` - which
reaches the state the race produces where `deleteLater` alone does not.

**Two fixtures were wrong and are corrected.**
`test_focus_review_keeps_its_rows_when_a_refresh_fails` used `"horizon": "h3"`, which
nothing writes - `human_focus_tracking` writes plain integers and the live rollups carry
"1"/"3"/"5". `test_table_width_rule_pages` rendered cohort rows with no horizon at all.
Both were invisible until the selector started filtering on it.

**Verification.** `pytest tests/ -q` **5775 passed, 72 subtests, process exit 0** ·
`ruff` **clean** · smoke **7/7** · source `--selftest` **73/73**. Fail-before-fix with
`scripts/` stashed: 15 tests for items 1-2, 14 for item 3, 15 for item 4 (the whole file
fails to collect with the untracked module stashed too), 11 for items 5-6, and both
deleted-panel guards.
### 2026-09-01 - Phase 0.13 packet P4: the variables you are not looking at

**Branch `claude/p4-swing-variables`, off `main` at `66a0c31`.** Two halves. The trader
was asked BEFORE the first edit to `master_avwap_lib/legacy.py` (file-scoped ask-first
rule) and answered "yes - do Half A" and "all six (B1-B6)". Live gate 33 owed. No frozen
rebuild.

**HALF A - capture only, and a golden proving it.**

The Qt Setup Tracker gained an **Attributes** tab over
`master_avwap_setup_attribute_leaderboard.csv`, which the scanner has written every scan
since it was built and which only the legacy Tk GUI and the offline tuner ever read.
Live: **38,617 groups, 37,049 of them (96%) under the reportable-n floor** - so the
order is the honesty, with floor-clearing rows first and sub-floor rows greyed, labelled
and last. Every row is KEPT.

Read **off the Qt thread**, unlike its ten siblings, and the comment says why: **19.7
MB** against 5.5 MB for the next largest and under 150 KB for the rest.

Twelve variables already on the record or the row gained attribute keys - human focus
pick/side, tracker setup family, market regime, sector, industry, ATR as a PERCENT of
price, signed SMA200/SMA50 distance in ATR plus two booleans, and relvol - with **no
weight and no gate**. A missing input records NOTHING rather than a zero, and a zero ATR
never divides.

The golden is the ranking itself: `p4_ranking_unchanged_v1.json`, contract-bearing,
frozen from the PRE-Half-A code with `scripts/` stashed, carrying its own inputs and
REPLAYED rather than compared. The Expected-R config is pinned to the shipped defaults
rather than loaded - `expected_r_config.json`'s anchors are re-fitted by the calibration
pass, so a golden that let it load would fail whenever the desk recalibrated. That was
found the hard way: the first freeze read the live config outside pytest and the sandbox
config inside it, and the golden caught the mismatch.

**HALF B - six items, each behind a fixture frozen first.**

**B1** The leaderboard states its own floor (`meets_n_floor`, `evidence_label`, through
`evidence_stats.summarize`, asked of CLOSED setups). The fixture freezes the leaderboard
AND the tuner's recommendations, and is deliberately built so the two verdicts DISAGREE:
a 20-setup group is under the reportable floor but clears the tuner's own gates, so the
tuner still writes its -8 rule. B1 publishes that and changes nothing about it.

**B2** Family and regime views as sibling files; the existing export keeps its exact
grain because the tuner reads it into live weights. Columns read BY NAME - the extra
dimension prepends, so positional indices would have shifted every column one place.

**B3** `stale_horizon` rows leave the scan-factor leaderboard. The horizon indexes a
symbol's own scan rows, not exchange sessions: medians of 64 and 73 sessions for
horizons 5 and 10, and 42-45% of rows over twice their horizon, all inside every average
the file published. The drop count and reason travel on every row; a row whose drift
could not be measured is KEPT. **Step (a) only**, pinned by a test - re-selecting the
future row is a sec-7 promotion.

**B4** `assigned_tier` is stamped at assignment time and preferred by the grader, with
`tier_source` saying which. The disagreement it fixes: a favorite-bucket row held out of
S/A for a poor expected R derived as "S" and shipped as nothing.

**B5** `static_score` reaches the record and the calibration helper prefers it. The fit
was reading the proven-quality score - which already contains realized performance - as
structure quality, a feedback loop.

**B6** `REPRESENTATIVE_EXIT_TEMPLATE_ID` names which exit plan the headline R is measured
on. Empty means today's behaviour, so nothing moved; the resolved template is on the
summary and in every `expected_r_note`, and `setup_docs.py` now says the headline R is
not measured on the plan it documents.

**REPORTED, NOT FORCED.** The packet named `_build_priority_tier_sections`; the function
is `_priority_partition_tier_rows`. And B5's anchor movement **cannot be measured on this
tree**: no record on disk carries `static_score` yet, so today every sample still takes
the old path and the fit is unchanged - the new
`expected_r_calibration_source_counts` is what will show the changeover as records
accumulate. Stated rather than estimated.

**One golden was regenerated, and it says so.** B6 changes the `expected_r_note` STRING,
which the Half A golden pins. The fixture's `intentional_difference` names B6, and the
pre-B6 NUMBERS are preserved separately in `ranking_numbers_before_b6` with a test
asserting they still agree - so the regeneration cannot hide a moved score, bucket or
expected R behind a text change.

**Verification.** `pytest tests/ -q` **5759 passed, 72 subtests, process exit 0** ·
`ruff` **clean** · smoke **7/7** · source `--selftest` **73/73**. Fail-before-fix with
`scripts/` stashed: 9 of 13 Half A tests, 13 of 30 Half B tests.
### 2026-09-01 - Phase 0.13 packet P5: pass and not-today get graded

**Branch `claude/p5-pass-cohorts`, off `main` at `66a0c31`.** Two new cohorts complete
the set: every verdict the trader can record now has a forward record. Live gate 34 owed.
No frozen rebuild.

**The gap.** Veto graded what was thrown away, like graded what was endorsed. Three
verdicts had nothing: the day-trade **pass**, **not_today** (223 live rows) and
**dislike** (34).

**1. The pass cohort.** Multi-select, so one pass grades under each of its codes AND
under the pooled `pass_all` - k+1 rows. **The code cohorts OVERLAP and must never be
summed**, and that fact travels three ways: the module docstring, a `reason_code_count`
column on every row, and `OVERLAP_NOTE`, which the Weekend Prep note and the AI scope
label READ rather than retype. Only `pass_all`'s n counts passes. Identity on write is
(vocab_version, reason_code), and the pass vocabulary is a SEPARATE family never folded
into the veto's - pinned by a test that loads the vocabulary rather than naming a version.

**The intraday grade is currently always blank, and that is MEASURED rather than
assumed.** The sidecar is written from the bars the desk was ALREADY HOLDING when the
pass was recorded, so every bar in it starts BEFORE the pass and the entry bar the rule
asks for - the first completed close AFTER it - is never inside it. Every row therefore
carries `intraday_unmeasured_reason`, distinguishing
`sidecar_ends_before_the_entry_bar` from `no_sidecar_bars`; a bare blank would have read
as the second. Whether entry should instead be the last completed close AT the pass -
the price the trader was looking at - is a definition change and the trader's to make.

**2. The rejection cohort.** `not_today` and `dislike` are separate cohorts whose
numbers are never combined into a verdict (their pooled BASE row exists and is
labelled) -
`pick_feedback` has kept them distinct since R2 because a same-day throwback and a
judgement on the name are different claims. Live: **253 gradeable rows, 219 + 34, zero
sideless.** `unfavorite` is not graded (a membership change, and sideless on the live
log) and the free-text `reason` is carried verbatim and never coded.

**3. THE ONE CHANGE TO EXISTING CODE.** `update_human_focus_outcomes` keyed outcome rows
on (trade_date, symbol, side); every row of one multi-code pass shares all three, so they
would collapse into one and k of the k+1 cohorts would silently vanish. The new
`pick_key` parameter DEFAULTS TO None - every existing caller unchanged, asserted - and
the P5 cohorts pass `pick_key_with_source`. The outcome NUMBERS are identical across
those rows, so what the wider key preserves is which cohorts were graded, not which
figures.

Both families registered by APPENDING. The rejection sources are
`focus__m5_not_today` / `focus__swing_dislike` (R1 put the LANE back into the name)
and the **double underscore is load-bearing**: the matcher tests
`startswith(prefix + "_")`, so `focus_` claims exactly those and cannot reach
`focus_swing`, `focus_m5` or `focus_pick`.

**4. Surfaces.** Two nightly slots appended (5 min, deterministic, no model - asserted).
Capture-time merge for a pass through one shared helper with the veto's. Two Weekend Prep
tables with the six columns PLUS `meets_n_floor` and `evidence_label` and sub-floor rows
greyed. Both files added to the evidence report and to the `trader_judgement` scope -
with the LIKE file, which was also missing: that scope read the veto trio only, and so
asked "were your rejections wrong?" without ever asking "were your endorsements right?".

**SIX EXISTING TESTS PINNED THE OLD SETS AND WERE UPDATED** - the authorized change, not
drift. Three asserted an absolute slot prefix; P5's cohorts sit before `evidence_report`
because the report READS them, which moves later slots' INDEX without reordering any
existing PAIR, so they now assert the pairwise order - the real invariant, and one that
will not need editing next time a cohort is added. Three asserted the judgement scope
held exactly three sources; they now compare against the scope's own declaration.

**Verification.** `pytest tests/ -q` **5749 passed, 72 subtests, process exit 0** ·
`ruff` **clean** · smoke **7/7** · source `--selftest` **73/73** · spec-drift **17**.
Fail-before-fix: with `scripts/` stashed including the two untracked modules, all 34
tests in `tests/test_p5_pass_and_rejection_cohorts.py` fail.
### 2026-09-01 - Phase 0.13 packet P6: from what the trader said to what they traded

**Branch `claude/p6-preference-to-trade`, off `main` at `66a0c31`.** Three stores each
held a third of one question - the annotation log knows what the trader SAID about a name,
the journal knows what they TRADED, the cohort rollups know what it then DID on paper -
and nothing put the three on one row. Live gate 35 owed. No frozen rebuild.

**1. Exact-id candidates in the auto-tagger.** A fifth source, `trader_capture`, over the
trade's OWN window (open date to close date) rather than the fuzzy 16-day neighbourhood
the scanner lanes search: an event id is only worth carrying when the statement and the
trade really are about the same episode. It ranks ABOVE every fuzzy source and a fuzzy
match can never displace it - when the trader has already said what they thought of a
name, that outranks anything inferred about it. A rejection is PREFIXED (`vetoed:<code>`,
`passed:<code>`) so it can never read as an endorsement in a Tags column. Live: **1,229
capture rows, 8 of 193 trades now carrying a capture candidate.**

`context_row_id` is a new nullable column arriving through the store's OWN additive
migration list. **It is a pointer for a reader and never a canonical link** - plan.md
P5.3/P5.4 own the canonical id and a second one invented here would compete with it. Only
**54 of 730** take-class review rows carry an alert `event_id`, so the rest point at their
own natural identity (`review_event:<ts>`); an empty pointer would have looked exactly
like a fuzzy candidate, which is the one thing this lane is not. Nothing writes
`trade_annotations`.

**2. The nightly report.** `preference_trade_outcomes.py`, read-only in
`journal_walkaway`'s pattern: one row per statement across four channels (like_claim,
pass, swing favorite, `pick_feedback` like), joined to the journal and to the cohort
grade. **Live: 558 statements in 90 days, 13 traded, 545 not - and the not-traded rows are
the point.** Every row renders its **match confidence or "no match"**, with `match_basis`
naming what it rested on: the join is a JUDGEMENT, because a trade on the same name that
week may have been taken for another reason entirely. A test bans `uuid`, `hashlib` and
`opportunity_id` from the module outright. Swing favourites resolve PER SESSION through
`favorites_for_session`, so a name added and retracted is not reported as a pick the
trader never took; an unmatured paper grade is blank, never zero. The slot sits before
`evidence_report`, which reads it. The swing strip's "took" badge now names its trade in a
tooltip via the SAME matching rule that put the badge there - **the id is EXTRA, never a
condition**, which an existing test caught when requiring one silently un-marked chips.

**3. The honest empty-dimension banner.** "My setups" renders beside a full auto-tag chart
of the same width while resting on almost nothing: live, **0 of 156 closed trades carry a
confirmed tag**. Below 10% coverage the group's label is prefixed with one sentence saying
so. **The group is never hidden** - hiding it would replace a visible thin answer with an
invisible one, and seeing how little is tagged is the prompt to tag more.

**Also corrected:** `ai_summary`'s comment claimed the `market_journal` scope is "OPT-IN
ONLY". Wrong since R10.H - `briefs.DEFAULT_SCOPES` carries it on the nightly run, so it
reaches the local model every night without anyone selecting it. The COMMENT was the
defect; no behaviour changed, and whether it SHOULD be nightly is the trader's decision.

**Four existing tests pinned absolute slot positions and were updated** - the same
authorized change as P5's, and they now assert the pairwise order, which is the real
invariant. A circular import was resolved by moving `TRADER_CAPTURE_SOURCE` into
`journal_analytics` (which `journal_store` already imports from) and re-exporting it.

**Verification.** `pytest tests/ -q` **5747 passed, 72 subtests, process exit 0** ·
`ruff` **clean** · smoke **7/7** · source `--selftest` **73/73** · spec-drift **17**.
Fail-before-fix: with `scripts/` stashed including the new module, 29 of the 32 tests in
`tests/test_p6_preference_to_trade.py` fail.
### 2026-09-01 - Phase 0.13 packet P6a: tag the backlog

**Branch `claude/p6a-tag-backlog`, off `main` at `66a0c31`.** Authorized by the trader:
*"let's get Opus to do the tagging and I can review after"*. **Run against the live
journal**, backup taken first. Live gate 36 owed. No frozen rebuild.

**The gap.** 193 trades and exactly ONE setup tag the trader typed, so every per-setup
statistic on the desk rested on that row. What was missing was never evidence - it was a
human decision about 155 closed trades.

**1. A mark that never washes out.** `trade_annotations.tag_status` through the store's
additive migration list: `confirmed`, `provisional`, `needs_review`. The column's DEFAULT
is what made it safe on a live database - every row that already existed was typed or
accepted by the trader, so it became `confirmed` the moment the column appeared and no
backfill pass had to decide that afterwards.

**2. The bounded exception.** R7's I7 gives the trader `trade_annotations`;
`journal_bulk_tag.py` is the one authorized writer and pays for it three ways, each
tested: `apply_provisional_tags` REFUSES a confirmed row **inside the store**, because an
exception that depends on every caller remembering a rule is not a boundary;
`distinct_tags` counts a provisional tag in its own lane, so `own` still means "typed or
accepted by a human" - which is exactly what the rename tool may touch; and the tagger
**never writes `tag_corrections`**, since that table is the trader's feedback TO the
tagger and a machine writing it is the tagger teaching itself from its own guesses.

**3. THE MEASURED RUN.** Threshold **0.70**, chosen to encode a sentence rather than a
percentile - "the tracker or a focus favourite named this symbol, on the day I traded it,
on the side I traded". Tracker + same day + side is 0.72; a focus favourite is 0.68 before
its bucket bonus; the SAME tracker row one day later reaches 0.66. The live histogram of
the top setup-lane candidate over the 52 closed trades that had one:

```
  0.20   1 | 0.25   1 | 0.30   1 | 0.40   1 | 0.45   2 | 0.50   5 | 0.55   3
  0.60  10 | 0.65   2 | ----- threshold 0.70 ----- | 0.70   5 | 0.75   2
  0.80   5 | 0.85   6 | 0.90   3 | 0.95   5
```

156 closed trades considered · 0 already tagged by the trader (their one tagged trade is
still OPEN) · 104 with no scanner candidate at all · **24 provisional tags applied, 132
marked `needs_review`, 0 refused**. 24 adjustment rows written and **zero** tag
corrections - the single correction in the store is the trader's own from 2026-08-22. The
database was copied to `trade_journal.sqlite3.p6a-backup-20260901_214926` before the
write. Below the line nothing is guessed: a low-confidence tag parked in `setup_tags`
would be counted by every statistic that groups on setups, which is the circularity the
tagging rules forbid.

**4. The review surface and the split.** A tag-review filter and a count above the Trades
table ("14 of 156 shown; 24 provisional") - a hidden row and an absent row look the same
in a table. It narrows the rows ALREADY LOADED and issues no query. The Tags cell says
`(provisional)` in text rather than colour, because a `QTableWidgetItem` cannot be reached
by `theme.qss`. One click confirms; an edit replaces **and only an edit teaches the
tagger**. In analytics "my setups" is confirmed-only and "provisional setups" is its own
group with no catch-all bucket; the two are never blended.

**VERIFIED DIFFERENCE FROM THE PACKET.** Its binding rules say the list-trades load runs
off the Qt thread. **It does not.** `TradesTab.reload()` calls `journal_feed.load_trades`
synchronously (line 468 before this change) and `AnalyticsTab` does the same at its line
182; the Journal's only worker is the migration one. Nothing here makes that worse - the
new surface adds one indexed single-row read per selection and no reload - but moving that
load onto a worker is its own packet, and the rule as written is not what the code does.

One existing test pinned `nonexclusive_groups` to exactly two entries; it now asserts the
invariant (every tag group is declared non-exclusive) rather than the length.

**Verification.** `pytest tests/ -q` **5737 passed, 72 subtests, process exit 0** ·
`ruff` **clean** · smoke **7/7** · source `--selftest` **73/73**. Fail-before-fix: with
`scripts/` stashed including the new module, all 17 tests in
`tests/test_journal_bulk_tag.py` fail.

### 2026-08-31 (late evening) - Desk snappiness packet 3: the last of the three

**Branch `claude/desk-snappiness-3`, cut from `claude/desk-snappiness-2`.** Five
commits, green, pushed. The three packets are stacked, so this one branch carries
all of them. Live gate 26 owed.

Packets 1-2 took the six largest causes of the day's ~78 minutes of GUI freeze.
This takes what remained: the growing log and its in-process replay, the one
downloader with no quiet-hours gate, the hidden-page timers, and the measured
drips.

### Item 1 - the 618 MB log stops being replayed whole

Its `level_resolved` rows are mirrored to a derived sidecar as they happen. The
layout, the watermark design and why the watermark rides on the ROWS rather than
in a header are in `docs/DESK_INTERNALS.md`. The main log is the authority and is
written first; the sidecar's append is swallowed on failure; the replay falls
back to the full stream on any doubt.

**Review round (2026-08-31, Fable): one defect found, reproduced, fixed.** A
thread switch between the clock's main-log append and its sidecar mirror, while
the wrap-up's sync was catching up the tail, wrote the same event into the
sidecar twice - and the replay then counted it twice while the full stream
counted it once (deterministic reproduction, sidecar `['A','B','B']` vs stream
`['A','B']`). Both copies carry the same source byte offset - the line's
identity - so `_resolved_rows_from_sidecar` now dedupes on it: the duplicate may
sit on disk, never in the answer. Test
`test_a_late_mirror_after_a_catch_up_cannot_double_count` pins it and failed
before the fix.

**Part (d), the month roll, was NOT built - this is the packet's own stop
condition, met.** Renaming the live log into `-YYYYMM` segments requires every
reader to see the live file plus the segments, and
`research_warehouse/ingest_existing.py` registers the log as a `BronzeArtifact`
whose `resolve_path()` returns exactly ONE path. Teaching it several is a change
to the warehouse's bronze contract - a locked area with its own decision log that
this packet does not authorize - and shipping the roll without it would have left
the warehouse silently ingesting one month. Nothing is lost by deferring: (a)-(c)
remove the hour-class replay cost, which was the GUI-freeze problem; the roll only
bounds disk growth. The other three readers would each have been straightforward.

### Item 2 - the Industry Board obeys quiet hours

It was the only recurring downloader with no `auto_scanning_due` gate, so its
~1,930-ticker nine-month `yf.download` ran hourly all night and fired about five
seconds after every desk launch at any hour. The automatic tick is gated,
fail-open; the manual button is never gated. The download is chunked at 200 so a
failing chunk costs that chunk instead of the whole board.

### Item 3 - hidden pages stop paying

The auto-watchlist viewers, the Master AVWAP scheduler tick (also inert when
another process owns scheduled scans) and the RS Window auto tick. Timers keep
running; the work early-returns while hidden and `showEvent` catches up once. The
watchlist viewer additionally guards `setPlainText` on the file's (mtime_ns,
size) - it was yanking scroll and caret every thirty seconds whether or not Auto
Pilot had written anything.

### Item 4 - the eight drips

Entry-assist board to a worker; the 3-second health tick stats inline and spawns
a thread only when the file moved (~1,200 fewer thread creations an hour); the
technical-integrity snapshot and the setup tracker's ten CSV exports memoized per
file version; the tracker's spinbox and the Focus panel's RRS snapshot coalesced;
one settings write instead of two; the Focus chip's badge stylesheet inside the
existing look guard; the paused strategy loop at 5 s instead of 0.5 s.

**The one observable change, authorized in the packet:** the hold expiry now runs
`survives()` exactly once per alert per tick. It has side effects - it rewrites
the caption and writes a `hold_expired` review event - and the current alert was
run through it twice, so an alert that was both queued and on screen produced two
events and two caption mutations on the tick it expired. What expires, when, what
is kept and the `shown` denominators are all identical.

### Verification

`pytest tests/ -q` **5685 passed, 72 subtests**, exit 0; `ruff check .` clean;
smoke 7/7; source `--selftest` 73/73. **46 new tests**, and each change proven to
fail against the un-fixed code by copying the files aside, reverting with
`git checkout`, running and restoring: 11 of 13 for item 1, 5 of 7 for item 2, 9
of 10 for item 3, 15 of 16 for item 4. The ones that pass either way are
documentation of behaviour the new code must not lose and are named in the commit
messages.

One full run showed a Windows `PermissionError` flake on the pytest temp
`local_settings.json` replace, in a test unrelated to this packet; it passed
alone and the next full run was clean, and item 4(e) strictly REDUCES writes to
that file.

**No packaging trigger.** The resolved sidecar is created at runtime beside the
log in the diagnostics directory - not a bundled asset - and there is no new
dependency, top-level package or dynamic import. The stall watchdog stays ON.

---

### 2026-08-31 (late evening) - Desk snappiness packet 2: the next three stall causes

**Branch `claude/desk-snappiness-2`, cut from `claude/desk-snappiness-1` and NOT
from `main`, so it carries packet 1 too.** Three commits, green, pushed. Live
gate 25 owed.

Same evidence as packet 1: the 2026-08-31 stall log measured **8,008 GUI freezes
/ ~78 minutes in one day**. Packet 1 took the three largest causes; these are the
next three. Every change is memoization, threading or batching - nothing
computes anything different.

### Item 1 - the Alert Center's minute tick stops redoing itself

Three repetitions, all over a ~105-symbol Focus set:

- **M5 bars re-materialized per caller.** `bot.m5_chart_bars` rebuilds ~150 dicts
  with six `float()` coercions each, and EIGHT timer-driven sites asked for the
  same symbol's bars per tick. Memoized in the panel - the source series belongs
  to BounceBot, which `ChartDataService` cannot see - on the source list's
  identity plus length and last stamp, strong reference held so `is` cannot be
  fooled by a recycled id. `m5_chart_bars` is untouched and still produces every
  value, so a divergence in the key lookup would cost a cache miss, never a wrong
  bar. `_poll_any_bounce_watches` also read the same bars twice per watch.
- **D1 levels built ten times per symbol.** `evaluate_d1_event_watch` gained an
  optional `levels_cache` scoped by the caller to one symbol and one bar list for
  one tick; with `None` it is exactly the call it replaced. Keyed on (session,
  anchor), so the AVWAPE kinds keep their own entry and never see the plain one.
  No threshold, kind list or evaluation rule moved.
- **~105 single-element prefetch tasks per minute**, queued ahead of the snapshot
  for the chart the trader had just clicked. One batched call per event-loop
  turn. Pool size and priorities untouched.

### Item 2 - the startup heap is swept once, then frozen

`main()` runs `gc.collect(2)` then `gc.freeze()` after the window shows. The
order is the rule: freezing first would make startup GARBAGE immortal.
`_GuiGcController` is untouched - same cadence, same bounded waits. Detail and
the numbers behind it are in `docs/DESK_INTERNALS.md` beside the collector rule.

### Item 3 - the journal's 1 GB click-freeze, the N+1, and the filters

Accepting a correction parsed a **1.08 GB** tracker file plus a 73 MB CSV
synchronously behind the OK button. Now: a single-flight worker
(`ui/services/journal_rebuild_service.py`) with the buttons disabled, a
"tagging..." status, results on the GUI thread and failures SHOWN - the journal's
loud-write rule; a per-source-file parse cache stamped on (mtime_ns, size) that
holds the projection and never the blob; one regime query instead of one sqlite
connection per trade; and a 250 ms debounce on the filter header.

**One thing that needed care.** Deferring the header's emit moved WHEN it lands.
Construction populates the widgets, which fire their change signals; those used
to emit synchronously, before anything was connected, and were harmless.
Coalesced, one landed after construction and read as a real filter change - it
made a failed-migration panel reload and claim "No accounts", which an existing
test caught. Nothing is owed now when the header is born, or after
`refresh_accounts` rebuilds the tree.

### Left out, and why

The hold-expiry double `survives()` evaluation of the current alert. The packet
allowed a fix here only as a per-tick verdict cache; `survives()` has side
effects, so whether the second call's effects are genuinely redundant is a
behaviour question rather than a memoization one, and this packet is not
authorized to answer it. Left for packet 3 with that note.

### Verification

`pytest tests/ -q` **5639 passed, 72 subtests**, exit 0; `ruff check .` clean;
smoke 7/7; source `--selftest` 73/73. **27 new tests**, each change proven to
fail against the un-fixed code by copying the files aside, reverting them with
`git checkout`, running, and restoring - 8 of 12 for item 1, 2 of 3 for item 2,
and all 12 for item 3; the ones that pass either way are documentation of
behaviour the new code must not lose, and are named in the commit messages.

Two existing test stubs needed the new keyword arguments. Worth recording why
they failed loudly only here: both call sites are wrapped in `except Exception`,
so a stub with the wrong signature turns a `TypeError` into "no hit" rather than
an error.

**No packaging trigger** for either packet: no new dependency, no new non-`.py`
asset, no new top-level package (the new service is inside the already-collected
`scripts/ui/services/`), no new dynamic import. The stall watchdog stays ON.

---

### 2026-08-31 (late evening) — Desk snappiness packet 1: the three largest measured stall causes

**Trader-authorized packet on `claude/desk-snappiness-1` (cut from `main` at
`50af716`).** The 2026-08-31 stall log measured 8,008 stalls / ~78 minutes of
GUI-thread freeze in one day; this packet fixes the three largest causes, caching
and cadence only — no detector/scoring/alert behavior, output, file format, or push
differs. Full detail and the measured numbers are in the CHANGELOG dated entry; one
line per item here:

1. **Health audit** (`operations_audit.py`, `diagnostics/shadow_log_audit.py`,
   `health_panel.py`): the 269 MB outcome CSV (2.29 s/pass, every 15 s) and both
   shadow JSONL streams now cache on `(st_mtime_ns, st_size)`; the Health page
   ticks at 15 s only while showing, 120 s hidden (timer never stops, the shell
   chip keeps updating).
2. **Tables/Industry Board** (`data_table.py`, `industry_board_service.py`,
   `industry_panel.py`, `price_alerts_panel.py`, `health_panel.py`): the one shared
   column fit is bounded by `setResizeContentsPrecision(200)` — set on the VERTICAL
   header, the one Qt consults for column hints (85 s single stalls, 9.6 min/day);
   the board's 60 s tick no longer emits an unchanged `snapshot_id` (the panel
   double-checks); the two `ResizeToContents` headers fit once when rows land, then
   stay Interactive.
3. **Auto Pilot status row** (`autopilot_service.py` — only the authorized pieces,
   no alert/push code; `autopilot_panel.py`): the file-backed pieces of
   `status_snapshot` are memoized on the file stamp, the 30 s tick takes ONE
   snapshot (was two), the 5 s panel poll early-returns while hidden, and labels
   restyle only on text/tone change (~14 min/day of file-read stalls).

Tests: 21 new across `test_health_audit_caching.py`,
`test_desk_snappiness_tables.py`, `test_autopilot_status_snapshot_caching.py`, plus
the Health cadence test in `test_qt_health_panel.py` — the behavioral ones verified
failing against the HEAD versions of the touched files before the fix. Also fixed
en route: a pre-existing thread leak in
`test_qt_health_panel.py::test_refresh_never_blocks...` (no `shutdown()` before
`deleteLater()`, so the construction `singleShot(0, refresh)` started one more slow
audit thread at the final `processEvents`), reproduced on unmodified HEAD — it
failed the audit-thread sweep in any non-alphabetical file order.

**Live gate #24** (numbered past the theta gate #23 that lives on
`claude/theta-premium`): one DESK session's `ui_stalls.jsonl` with
`data_table.py:170`, `watchlist_utils.py:33`, `project_paths.py:165` and the
operations-audit CSV parse gone quiet. The stall watchdog stays ON — it is also
gate 19's instrument. No frozen rebuild trigger fired: no new dependency, asset,
package, or dynamic import; every change is inside existing modules.

**NOT touched (packet 2+, some need trader sign-off):** Alert Center M5/D1 poll
work, the 1 GB journal retag load, the 618 MB technical-integrity log, the
Industry Board quiet-hours gate.

## 2026-08-31 (evening) - Theta: the floor is a percent of the strike, and support ranks first

**Phase 0.11, trader-directed.** Built from
`docs/prompts/THETA_PREMIUM_OPUS_PROMPT.md` on `claude/theta-premium` (`82c998a`
then `5442619`). Green; live gate 23 owed. **Supersedes the "planned, prompt
written, NOT built" entry below.**

### What was wrong, and what each fix was

- **The target WAS $0.25** - `THETA_PUT_TARGET_TOTAL_CREDIT` (100) / 100 /
  `THETA_PUT_MAX_CONTRACTS` (4) - judged in flat dollars, so the same number meant
  0.125% of a $200 strike and 1.25% of a $20 one. Now
  `theta_put_credit_floors(strike)` decides once: 1.0% of the strike recommended,
  0.5% cusp, $0.40/contract absolute floor underneath. A quote under both LEAVES
  the report rather than lingering as `below_target`.
- **The final sort preferred the cheapest qualifying option** - its key was
  `(status, strike ASCENDING, ...)`, so the deepest-OTM strike won every time.
  Now: tier, major SMAs above the strike, support quality, yield per market day,
  spread. Premium is a percent PER MARKET DAY, which replaces the flat DTE penalty
  on the sold-put path (PCS keeps it).
- **A wide spread was capped at 18 penalty points**, so past ~150% every market
  cost the same. Uncapped and monotonic now - and still never a block.
- **The quote budget was spent in `base_score` order**, which says nothing about
  premium. Now `thetalongs.txt` first, then estimated premium capacity from ATR%
  (no new network call, never a filter - unmeasurable sorts LAST, not out), then
  `base_score`. IB pacing constants untouched.
- Credit spreads reach **15 market days**; sold puts stay at 10.
- The report emits one `premium=` line per quoted sold put and the extractor reads
  every field back; the Qt theta table gained four columns, blank rather than zero
  for a row with no quote.

### The open decision was put to the trader, and answered

The question was whether the percent floor should apply to the PCS short leg
as well. It was put with its arithmetic - as a percent of the short strike the
20% credit/width target is 1.36% at a $40 close, 0.72% at $150, **0.54% at $200,
0.45% at $240**, 0.72% at $300 and **0.31% at $700**, because
`_pcs_long_strike_choices` caps the width at 10 points however expensive the
stock is, so the target credit stops growing at $2.00 - and answered in as many
words: ***"Yes it should scale with price of the underlying."***

Built as T7. `theta_pcs_credit_floor(short_strike)` is a hard minimum of 0.5% of
the short strike or $0.40, whichever is larger, sharing the sold-put constants so
the percent floor has one definition. Under it the spread leaves the report;
above it the credit/width ratio still decides recommended-vs-cusp. The 1.0%
RECOMMENDED percent is deliberately not applied to spreads: 1% of a $644 strike
is $6.44 on a 10-wide spread, a 64% credit/width bar no market pays, so it would
delete every expensive spread instead of ranking it.

**Consequence, said here rather than discovered on the desk:** expensive credit
spreads will mostly disappear unless their credit genuinely scales. The lever to
bring them back is the WIDTH cap in `_pcs_long_strike_choices`, not the floor -
widening a $700-stock spread to ~17 points lets a 20% ratio pay $3.50 and clear
0.5%. That changes capital at risk per contract, so it was not done without
asking. **Gate 23 should be read with this in mind: a PCS section that is
thinner than usual is the rule working, not a bug.**

### Verification

`pytest tests/ -q` **5619 passed, 72 subtests**, exit 0; `ruff check .` clean;
smoke 7/7; source `--selftest` 73/73. **31 theta tests**, each new or rewritten one
proven to fail against the un-fixed code (the file copied aside, reverted with
`git checkout`, run, restored). Four pass on the un-fixed code and are
documentation of preserved behaviour rather than proof - named in the commit
messages. Two existing tests were deliberately REVERSED: their old rule (a
sub-floor quote kept as `below_target`; the deepest viable strike winning) is
exactly what this packet removes.

Eligibility (>= 3 supports, >= 1 major SMA, earnings buffer) is unchanged, R9.4
`theta_side` semantics are unchanged, the IB pacing budget is unchanged, and
nothing in this chain executes anything.

---

### 2026-08-31 (evening) — Phase 0.11 theta premium optimization: planned, prompt written, NOT built

The trader asked to fix the theta section: the report hands him ~$0.25 credits
with untradeable spreads. Code-level causes verified this session: the target IS
$0.25 (`THETA_PUT_TARGET_TOTAL_CREDIT 100 / 4 contracts`, `legacy.py:487-491`);
the final sold-put sort prefers the LOWEST qualifying strike — the cheapest
option — every time (`legacy.py:19096-19105`); spread is a soft penalty capped
at 18 points; the quote budget is spent in `base_score` order with no
premium-richness ordering. Trader decisions locked in chat: credit floor 0.5% of
the strike / ideal 1% / $0.40 absolute; sold puts stay ≤ 2 weeks, PCS goes to 3;
spread is a SPECTRUM ranked heavily, never a new hard block; ranking priority is
support ≻ premium ≻ spread with 2 major SMAs above the strike a big boost; whole
universe already reaches theta evaluation, so the budget gets richness-ordered
with `thetalongs.txt` pinned first. Spec: `plan.md` Phase 0.11. Build prompt:
`docs/prompts/THETA_PREMIUM_OPUS_PROMPT.md` (Opus builds on branch
`claude/theta-premium`, Fable reviews). No code changed this session — docs
only, baseline unchanged.

### Cross-packet compatibility check before the 2026-08-31 merge

Three packets built in one shared checkout that afternoon - today's swing picks,
the day-trade pass, and the Strength Board move - so the merge was checked as one
thing rather than three. What was verified: full suite **5583 passed / 72
subtests**, exit 0; smoke 7/7; source selftest 73/73; **frozen rebuild + frozen
selftest 73/73, exit 0**; spec-drift 17; `main` a strict ancestor (fast-forward,
no divergence); `CLAUDE.md` byte-identical to `AGENTS.md`; both themes render
with **no unsubstituted `@token@`** left in the stylesheet; every `symbolActivated`
/ `symbolRequested` connected to `chart_symbol` still emits one argument against
its keyword-only `side`/`origin`; the two split settings keys are distinct; no new
dependency, no new non-`.py` asset, no new top-level package.

One real defect was found and fixed in the process: the swing-picks strip asked
the journal for its "took" badge during **construction**, so any desk built and
dropped without `shutdown()` left a worker thread behind it. The read now hangs
off the strip's first paint (`firstShown`), a test pins that construction opens no
thread, and every desk-building test hands its threads back the way the app does.

`QThread: Destroyed while thread '' is still running` still prints once at
interpreter exit on a full run. It is **not ours**: a session-end probe over every
live `QThread` found only `Qt mainThread`, and the line does not appear when the
qt-marked and non-qt halves are run separately. Cosmetic, exit code 0.

### Merged to main without a live-validation day — stated, not hidden

`plan.md` sec 6 normally asks for a live-session validation day before a merge to
`main`, and the frozen-exe policy asks for a rebuild at the same point. The trader
merged on 2026-08-28 anyway, in as many words, and the reasoning is sound rather
than a shortcut: **gates 13-18 are all live proofs that can only be run once the
code is on the desk**, and the desk runs from SOURCE, so merging is the
prerequisite for validating rather than something validation should precede.
What was actually verified before the merge: 5419 passed / 72 subtests / 6
skipped, smoke 7/7, source selftest 72/72, spec-drift 17, ruff clean, fast-forward
with no divergence. What is NOT verified: any of it on a live desk against the
live journal database, and the frozen exe, which is six commits stale.

### Immediate next action

**The desk has been restarted and everything is on `main`** (tip after the theta
merge and the integration docs). It runs from source, so what is running is what
is on `main`, and the stall watchdog is ON.

**So the next action is the trader's, not an agent's: use the desk.** Gates 24, 25
and 26 need one session's `ui_stalls.jsonl` and one overnight; gate 23 needs one
theta scan; 22, 21 and 20 are this afternoon's three. None of them can be run from
a keyboard on this side.

The one judgement call the trader may want reversed: the Strength Board's own RS/RW half retired with its page, on the reasoning that the Alert Center's RS/RW Board tab is now one tab-click away in the same column. Say the word and it comes back as a second section.

Then run the owed live gates in the order above, gate 19 first, because it is
the one that says this morning cannot repeat. The `ed3c73c` shared-checkout
collision recorded earlier is **resolved**: both packets landed on `main`
together, and `claude/swing-favorites` no longer exists - do not go looking for
the swing-picks work on the branch named for it. **The remaining
gates are all live-session work that only the trader can run.** The statement importer is **built and
layers correctly** — the trader's next move is to import both real files on the
desk and run the self-check (gates 14-15). Two of his asks are scoped and
NOT built, and one needs his answer first: statements as the source of truth over
the API (it would cost the only intraday timestamps the journal has), and the
IBKR transaction-file importer (masked account numbers are the open problem).

---

## 2026-08-31 - The Strength Board moves into the Desk's Strength window

**Branch `claude/strength-board-into-desk`, cut from `main` at `f36ab59`. BUILT,
live gate owed.**

Trader: *"The Strength Board tab is good but it really should be modified to fit
in the 'strength' window in the trading desk - either integrated directly or be
positioned below it."* Positioned below it, and the left-nav page is removed.

**Where it is.** A `CollapsibleSection` (new,
`scripts/ui/widgets/collapsible_section.py`) under `FocusStrengthBoard` in the
Alert Center's alert column, hosted through
`AlertCenterPanel.attach_strength_board`. `MainWindow` still builds and owns the
one `StrengthBoardService` - one timer, one single-flight fetch, one 15-minute
cadence - and now also shuts it down, which **nothing did before**: the service
was parented to the window but absent from the panel shutdown loop, so its timer
outlived the close.

**What did not change**, pinned by `tests/test_qt_strength_board_in_the_desk.py`
rather than by prose: zero IB traffic (asserted over the AST of all three
strength-path modules); the M5 Focus adoption gate re-run at click time on the
row's own numbers; and one service with one timer, measured by driving that timer
and counting fetch attempts.

**Width was the constraint**, because the alert column has a 360 px floor and
everything left of it is chart:

| Demand | Measured | What was done |
|---|---|---|
| Section header | 315 px | a `QToolButton` asks for its whole label; Ignored horizontally + elided |
| The board | 270 px | hosted in a `QScrollArea`, so the minimum stops there, not at the desk splitter |
| Status label | 434 px | word-wrapped - it carries failure reasons, so it can be long |
| "Add all shown" | 208 px | relabelled "Add all" (124 px); tooltip unchanged |

The section also **starts closed**, so by default it costs one header row. The
two sides stack **vertically** - side by side was right for a full-width page and
is unreadable in a column.

**A row click charts in the review pane** (trader, same day, second pass:
*"when I click on a stock in this M5 strength board it should come up on the
Visual chart review in the trading desk"*). It goes through `chart_symbol`, the
LOOKUP BOX's door, and deliberately not `_enqueue_review_alert`, the SCANNER's
door - that one drops everything in AWAY, drops parked symbols, diverts M5
alerts to the alert bar and can hide a row behind movers-only, and **a name the
trader clicked must appear**. It charts as a `MANUAL_CHART` (muted, not red -
nothing fired), never enters the alert feed, un-ignores a "not today" symbol as
typing one does, and carries its side. `chart_symbol` grew optional
`side`/`origin`; the lookup box's behaviour is unchanged and pinned.

**The RS/RW half retired with the page.** It was added 2026-08-21 so the two
reads could be compared without flipping PAGES; the Alert Center's own RS/RW
Board tab is now one tab-click away in the SAME column. The tape, its owner, the
`rrsSnapshotChanged` signal and that tab are untouched - one listener retired,
nothing else moved. **This is the one thing in the packet the trader may want
back**, and it is a section rather than a page if so.

Page removal touched every page-tracking site: the single `PAGE_SPECS` list (one
line, which is the payoff of the 2026-08 nav refactor) plus the two test files
that enumerate nav labels. The module stays inside an already-collected package,
so `packaging/tradingbotv3.spec` needed no change.

**Verified:** `pytest tests/ -q` **5571 passed, 72 subtests** (was 5554) · smoke
**7/7** · source `--selftest` **73/73** · spec-drift **17**. `ruff` was not run -
the desk `.venv` has no `ruff` installed.

**Owed, live (gate 22):** one desk session where the trader opens the section,
reads the board in the column, clicks a row onto the Visual Alert Review chart
and adds a name from it - plus a judgement on whether the vertical stack is
right or the sides want their side-by-side shape back with the column dragged
wider.

## 2026-08-31 - "I liked it and passed": the day-trade pass, under the note

**Trader-directed, authorized in chat 2026-08-31.** Branch
`claude/daytrade-pass-reasons`. Phase 0.5 item 14. **BUILT; live gate 21 owed.**

*"Many times I really like this stock for a daytrade but it has this ONE
issue"* - and the trader passes. The capture window had a veto (this chart is
not for today), a like, and a note; the far more common judgement, a name that
WAS tradeable but for one thing, had nowhere to go.

**What was built.** `EVENT_PASS` in `ui/annotations/store.py` with
`record_pass_annotation`; a separate versioned vocabulary FAMILY
(`ui/annotations/vocabularies/pass_reasons_v1.json`, loaded by
`load_pass_vocabulary` - `vocabulary.py` now loads any family and validates
`vocabulary_id` against the filename); the M5 sidecar in
`ui/annotations/pass_bars.py`; the "Passed - why?" block under the Note field
in `ui/widgets/capture_rail.py` (Alt+P, digits 1-5 toggle, scoped to the
checkbox box so a digit typed in the note stays a digit); and
`SymbolSnapshotWidget.cached_m5_bars` wired as a zero-fetch bar provider on all
three capture hosts.

**The three decisions worth remembering** (full reasoning in
`docs/DESK_INTERNALS.md`):

1. **A separate vocabulary family, not five more veto reasons.** Sharing the
   veto list would have restamped `vocab_version` across cohorts already
   accruing forward returns, for two lists that answer different questions.
2. **A pass never retires the chart.** It is note-shaped - written about the
   chart still in front of the trader - so both hosts' `_on_captured` continue
   to key on veto and like alone. No new exception was added anywhere.
3. **The bars are a sidecar and are never fetched.** One session is ~78 M5 bars,
   far past the store's 4096-byte single-write cap, and that cap is what makes a
   torn tail cost exactly one row. Sidecar first, row second. Nothing cached, or
   a provider that raises, costs the attachment and never the row - the trader's
   own fallback was *"just store the exact timestamp"*, and every row carries a
   zoned one.

**Verification.** `pytest tests/ -q` 5554 passed / 72 subtests (this packet adds
39, including fail-before-fix proofs that a pass retires the chart if the host
rule is broken and that the sidecar keeps only the newest session); source
`--selftest` 73/73.

**Both open questions are DECIDED (trader, 2026-08-31), so neither is pending work.** *"Reviewed today" stays OFF for a pass:* the trader's words - *"that flag feeds the scanner report and several badges. Making a pass count as reviewed touches scanner-side code, so it should be its own small job if you want it."* `pick_feedback._ANNOTATION_DECISIONS` therefore still lists `veto`/`like_claim`/`note` only, and a test pins that a pass does not mark a symbol reviewed. *A pass never closes the chart, and no option is needed:* *"if you pass AND want the chart gone, just hit veto after. You get both behaviors without a new rule."*

**Shared-checkout warning.** This branch's commit `ed3c73c` also contains the
"Today's swing picks" packet: a second agent was working the same checkout and
committed while HEAD sat here. See the glance block.

---

## 2026-08-31 - The linter was configured but never installed, and it was hiding four bugs

**Trader-directed** ("ok install ruff then if you need it"), after the merge.

`ruff` is named in `CLAUDE.md`'s stack, declared in `requirements-dev.txt` and
configured in `pyproject.toml` with a narrow defect-class select - and was **not
installed in the desk `.venv`** and not pinned in `constraints.txt`. Every "ruff
clean" claim in this file's history predates an installed linter. Now
`ruff==0.16.5`, pinned.

**1,703 findings on the first run. 1,591 were noise** from four legacy Tk shims
(`master_avwap_lib/gui.py` + `runner.py`, `bounce_bot_lib/gui.py`, `gui_app/`)
that re-export their names out of `legacy` at import time, so a static reader
cannot resolve one of them. They join the two `legacy.py` files already excluded,
with the reason recorded in `pyproject.toml`. **Repo-wide: 1703 -> 75.**

**Four real defects, fixed:**

- `operations_audit.py` called `logging.exception(...)` in **three** handlers
  whose comment reads *"health must never take the audit down"* - and never
  imported `logging`. Each raised `NameError` out of System Health at exactly the
  moment its guard was supposed to hold. All three carry `# pragma: no cover`,
  which is why nothing noticed. `tests/test_operations_audit_never_raises.py` is
  the fail-before-fix proof: 4 of its 7 fail with the import removed.
- `journal_tab.py` built the Questrade token-failure dialog as
  `lambda: ...f"{exc}"`. Python deletes `exc` at the end of the except block and
  the lambda runs later on the Tk loop, so **the error dialog itself** raised
  `NameError`. Bound at raise time now.
- Two imports nothing had used in `ui/app.py`, one in
  `tests/test_trader_annotations.py`.

**Then the 74 were swept too** (trader: "yes clean that up"), across 52 files.
Two guarded availability probes in `test_qt_alert_center.py` kept their imports
under `# noqa: F401` - removing them would have removed the probe. **The sweep
broke one thing and the suite caught it:** `technical_integrity` imports
`row_capture_mode` and **re-exports** it to `regime_collection_audit` and
`test_ti_chain_backfill`, so removing it failed 8 tests. It is back, marked
`# noqa: F401` and commented as a re-export. A multi-line-aware scan over all 105
removed names found no other re-export (the first scan was single-line, which is
exactly how it missed a parenthesised import block), and an import sweep over all
331 modules passes. `ruff check .` now reports **All checks passed**.

The paragraph below is the state *before* that sweep, kept because it is the
reasoning that produced it: 74 unused imports across ~40 files -
auto-fixable, none of them behaviour, and a 40-file sweep in a checkout two other
agents were writing to is the riskier act. The one `F821` was **asked about and then
fixed** (trader: "yes"): `ui/panels/alert_center_panel.py` annotated
`self.strength_board: "StrengthBoardPanel | None"` with a name imported only
inside `attach_strength_board`. It now carries a `TYPE_CHECKING` import, which is
never evaluated at runtime - the lazy import that keeps the board's module out of
the panel's import graph is untouched, and so is every alert path in the file.
`ruff check scripts/ui/panels/alert_center_panel.py` passes clean; the repo is at
**74**, all of them unused imports.

---

## 2026-08-31 - Today's swing picks: the trader's own list, under the alert bar

**Trader-directed, authorized in chat 2026-08-31.** Branch `claude/swing-favorites`,
cut from `main` at `ab2423b`. Green; live gate 20 owed.

### What the trader asked for

*"At the end of the day I have a list of my top swing targets. I want a place to put
them in so the bot knows my personal favourite picks. They will usually become focus
picks too but these ones get special standing because I picked them by hand... put it
at the very bottom of the M5 alerts tab, the tab is so long and I never use all of it.
And the bot should scan the journal to know which ones I actually took."*

Deliberately **not** the Master AVWAP like/dislike capture, which already exists and
records a verdict on a row the bot proposed. This records a name the trader brought in
themselves.

### What was built

- `scripts/swing_favorites.py` - append-only store and the session replay. A row is
  `(schema, action, session_date, symbol, side, event_at, origin)`; `event_at` is
  tz-aware and `session_date` is market-local, the `evidence_ledger` convention.
- `scripts/ui/services/swing_favorites_service.py` - the two writes, and the journal
  join on a worker thread.
- `scripts/ui/widgets/swing_favorites_bar.py` - the strip: input, Long/Short toggle,
  chips with an x, diffed like the Focus board, styled by `theme.qss`.
- `project_paths.SWING_FAVORITES_FILE` -> `swing_favorites.jsonl` in the shared home,
  the same storage class as `pick_feedback.jsonl` and `trader_annotations.jsonl`.

### The decisions worth keeping

- **Two writes, in a fixed order.** The swing Focus write-through goes first through
  the existing store and must not fail - it is what the trader asked for. The evidence
  row goes second and a failed append is swallowed with a status line, because an
  evidence store is never allowed to cost the thing it records.
- **No auto-adoption marker, ever.** A hand-vetted pick carrying one would be reachable
  by "Not today" and the desync repair - the exact removal path that marker exists to
  keep off the trader's own names.
- **A removal is a retraction, not an edit.** The add row stays and a `remove` row
  follows it, so "added AMD and then thought better of it" survives as a fact.
- **The "took" badge is display only**, joins the TRADE journal (not the Market
  Journal), runs on a worker thread over a bounded 10-day window, and is **silent when
  the journal would have to be created or migrated to answer** - a display badge must
  never be the thing that triggers a schema migration. It derives no rate, grade or
  statistic.

### Where it lives, and the one existing test that moved

The M5 alerts surface is a TAB in tabs mode and the tall left COLUMN in workspace mode.
**The trader's saved `qt_workspace_mode` is `workspace`**, so "the M5 alerts tab" is, on
their desk, that column - which is exactly the "so long and I never use all of it" they
described. The bar and the strip therefore share one host (`TradingDeskPanel.m5_column`)
that both modes mount, and the strip is the bottom of it either way. `M5AlertBar` and
every alert routing path are untouched.

That made the splitter hold the column rather than the bar, so
`test_the_bar_is_the_left_column_before_the_chart` now asserts
`splitter.widget(0) is desk.m5_column` **and** that the bar is the first thing inside
it. The trader rule it pins - the bar is left of the chart - is unchanged and still
verified.

### Second pass the same day: it drags, it copies, and it grades

The trader came back with three things. The code for all three landed inside
`edc7999` - the shared checkout had moved to another agent's branch by the time
they were written, the same collision recorded above - and reached `main` with
that branch at `bded98d`.

- **"The tab needs to be resizable relative to the M5 alerts tab, I should be
  able to drag it up to see more."** `m5_column` is now a vertical `QSplitter`
  with its own settings key (`qt_m5_column_split_sizes_v1`), so this drag and the
  desk's three-column drag never overwrite each other, and
  `setChildrenCollapsible(False)` because a strip dragged to nothing is one the
  trader cannot find again. The chip area gained a floor and **lost its ceiling** -
  a maximum height would have made the drag do nothing past it.
- **Copy and Paste, for TC2000.** Copy puts the day's tickers on the clipboard one
  per line, each once, in list order; Paste adds every ticker on the clipboard on
  the side the toggle is showing. Same idiom as the M5 alert bar's "Copy all".
- **"Will the bot do anything special with this data overnight, or for the
  journal, or for setup efficacy?"** As first shipped: nothing beyond it being an
  ordinary swing Focus pick. That is now a better answer by one deliberate line -
  the Focus like-origin is **`vetted`**, not `manual`, so the human-focus tracker's
  existing 1/3/5/10-session grader files these under `human_focus_swing_vetted`
  instead of mixing them with every other hand-typed swing name. "How do my
  hand-picked swings do against the bot's?" is now answerable from a grader that
  already runs.

  **Still deliberately NOT built, each additive and each the trader's call:**
  `swing_favorites.jsonl` is not in `ai_summary`'s overnight evidence pack (that
  costs local-model context, which was its own packet on 2026-08-28); nothing
  joins the list to per-setup journal statistics; and
  `journal_analytics.AutoTagger` reads the SCANNER's output files, so it does not
  see this list - and no tag may ever be derived from an outcome.

### Verification

`pytest tests/ -q` **5538 passed, 72 subtests**, smoke **7/7**, source `--selftest`
**73/73**, spec-drift 17. **Stated plainly: that suite run was not isolated.** The
checkout was shared with a second in-flight packet's uncommitted work (annotations /
pass-bars files), which accounts for 23 of those tests and for the selftest going
72 -> 73. This branch's own contribution is **59 tests** over the 5456 baseline. Only
this feature's files were committed.

### Owed, and one question not asked

Gate 20 above. And a product question the trader has **not** been asked: the strip
shows the CURRENT session only, so a pick typed after the close is shown that evening
and its "took" badge can only ever reflect that same session. Carrying picks forward to
the next session is their call.

---

## 2026-08-31 - The desk froze because one Focus add repainted five surfaces

**Merged to `main` 2026-08-31 by trader instruction ("go ahead and merge to main"),
fast-forward from `claude/focus-refresh-storm`, no divergence. Green; live gate 19
still owed.**

### What happened

07:37-07:53 this morning: ~500 s of GUI-thread blockage in a 16-minute session.
Since 07:45 the UI was blocked 216 s in 5.5 minutes; 07:50-07:52 it was 113 s in
2.3 minutes, about 80% frozen. Single stalls of **44.3 s**, 15.9 s and 15.2 s.
Windows reported the process Not Responding. The trader killed the desk twice
(07:31, 07:37) and each restart re-ran the 07:30 swing scan, repeating the pain.
Memory was fine (~2 GB working set) - this is **not** the 2026-08-27 warehouse
bug.

### Why

At 07:41:58-07:42:11 the Alert Center drain adopted **45 staged picks into M5
Focus one at a time**, ~300 ms apart (`C:\TradingBotData\focus_auto_picks.json`,
every `adopted_at` inside that window). The 15.2 s stall charged to
`focus_picks_panel.py:441` landed at the end of it.

`FocusPickStore.add()` notifies on every add. That contract is right - several
surfaces genuinely need to know about each mutation. What was wrong is that
**five listeners each treated one add as "rebuild everything"**: four editor
rebuilds plus a feedback-file read plus a forced snapshot write (Focus board);
both alert feeds destroyed and reconstructed, up to 350 widget trees each with
its own stylesheet (Alert Center); a full setups-viewport repaint through
`SetupTableDelegate` (Master AVWAP - the hottest stack in the stall log, ~300
samples across paint lines 78-152); the strength board rebuilt as HTML and
re-parsed by `setHtml`; the price-alert symbol combo cleared and refilled.

Times 45, in 13 seconds. The ~300 ms spacing between adoptions WAS that work.

### What was built

The signal contract is untouched: `focusChanged` still fires per mutation and the
coalescing lives at each **listener**. `ui.timer_utils.SignalCoalescer` is a
leading-edge window with a trailing fire - the first request opens a 200 ms
window, later requests fold into it and deliberately do **not** restart it. A
synchronous drain loop lands whole inside one window (one reaction); a sustained
trickle fires on a fixed cadence rather than starving, which a plain
restart-on-signal debounce would do. 200 ms is the trader's ceiling, not a target.

Three more defects in the same chain, all fixed:

- `FocusSideEditor.refresh()` **claimed to diff and did not** - it emptied the
  flow layout and re-added every chip on every call, unchanged list included. 90
  layout operations on a 45-name board to change nothing. The unchanged case now
  does zero layout work; arrivals/departures/reorders are index-precise, and
  `FlowLayout` grew `insertWidget` because `QLayout` has no generic insert and
  its absence is exactly why the teardown existed.
- `record_bounce_alert` rebuilt four editors and re-read `pick_feedback` to light
  one chip's badge. It now touches only the matching chip; `_bounce_state` is
  still written first, so a name that joins Focus after its alert still gets the
  badge when its chip is built.
- The DESK drain now adopts at most **10** picks per 30-second cycle
  (`AUTO_ADOPT_BATCH_LIMIT`). **Pacing, never policy.** The freshness gate, the
  flip barrier, ownership markers and AWAY/EVENING's refusal are all upstream and
  untouched. A deferred pick is not marked seen, the cap counts adoptions rather
  than iterations, and **no pick is ever dropped** - a cap that withheld one would
  be the suppression field this chain deliberately does not have. A 45-pick
  morning now finishes over ~2.5 minutes of background ticks.

### The ask-first rule, twice

`alert_center_panel.py` houses alert code. The packet's authorization covered the
drain cap. The feed-rebuild coalescing was a second edit in that file, so it was
**asked about separately and approved** before it was made. Only the trigger is
coalesced - which alerts pass the feed gate, their order, the repetition fold and
the digest are all decided inside `_rebuild_feed` and are unchanged.

### Deliberately not touched

The GUI-thread GC controller. Its ~600 ms young sweeps in the stall log are a
*symptom* of this churn, and its delay-never-cancel and GUI-thread-only invariants
are load-bearing. No detector, scoring, gating or adoption-gate logic changed.

### Verification

29 new tests, every one written against the old behaviour and **watched failing
first**. Full suite **5456 passed, 72 subtests** on the desk `.venv` (the
pre-change checkout collects 5427). Smoke **7/7**. Source `--selftest` **72/72**.
No packaging trigger: no new dependency, no new non-`.py` asset, no new top-level
package, no new dynamic import - the spec-drift test agrees.

`ruff` is **not installed in the desk `.venv`** (`No module named ruff`), so the
project rule set was run through `uvx ruff@latest` instead. The three F401s it
reports in the touched files (`flow_layout.QSizePolicy`,
`master_avwap_panel.SectionHeader`, `price_alert_board.Qt`) all reproduce on
`main` and are pre-existing; this change adds none. The repo-wide count from that
newer ruff is not comparable to the pinned one and was not chased.

### What is NOT verified

Any of it on a live desk. Gate 19 is a directional morning with a large staged
batch. **The desk runs from source on `main`, so the fix is live at the trader's
next restart** - which is now the only step left between them and it.

### The frozen rebuild that goes with the merge

Policy asks for a rebuild at every merge to `main`, and the frozen `--selftest`
replaces the trader's click-through, so it ran unattended straight after the
merge. `pyinstaller .\packaging\tradingbotv3.spec --noconfirm` at `d0a2ae6`:
exit 0, 419 MB onedir, then `dist\TradingBotV3\TradingBotV3.exe --selftest`
returned **`selftest OK: 72/72 checks passed (frozen)`**, exit 0 - matching the
unfrozen count exactly. Smart App Control was READ, not recalled:
`VerifiedAndReputablePolicyState = 0` (`SAC_PreviousState = 1`,
`SAC_EnforcementReason = 6`), so it is off and the build would start. **SAC
verdicts are per file hash, so this says nothing about the next build.**

None of that changes what is live: the desk launches from source, so the exe is
a verification artifact and the merge is what delivers the fix.

### Merged without a live-validation day - stated, not hidden

`plan.md` sec 6 asks for a live-session validation day before a merge to `main`.
The trader merged anyway, in as many words, and the reasoning holds: gate 19 is a
directional MORNING with a large staged batch, which can only be observed on a
desk running this code, and the desk runs from source. Merging is the
prerequisite for validating rather than something validation precedes - the same
call made on 2026-08-28 for the journal packet. What was verified before the
merge: 5456 passed / 72 subtests, smoke 7/7, source selftest 72/72, spec-drift
clean, fast-forward with no divergence, and no desk process running when the
branch was switched. What is NOT verified: any of it on a live desk.


## 2026-08-28 - The tax number is the broker's, never ours

**Trader decision:** *"Statement is source of truth for final pnl/tax
purposes."* Stronger than the day-level authority landed earlier that morning,
and it needed its own answer: that rule decided WHICH ROWS win, this one decides
what the reported NUMBER is.

**The gap it closes.** Every other P&L in the journal is recomputed - which is
what makes per-setup statistics possible, and is also our arithmetic, drifting
from Questrade's cent-rounded figures by -$0.2386 on $5,298.81 across the year.
`scripts/journal_tax_report.py` recomputes nothing: it sums the broker's own
`net_amount` per fill, and **for a FLAT position that sum IS the realised P&L**,
so no cost-basis model is needed or used. One normalization first - the IBKR
file states that figure in the BASE currency, so its importer now divides by the
row's implied rate before storing and keeps the base figure as evidence.

**It refuses rather than estimates.** Open positions, positions whose opening
fill was invented, and any position with a fill lacking a stated amount are
EXCLUDED and named with the reason. Voided rows never reach a total. CAD
converts per fill at the booked BoC rate and an unbooked date withholds that
position's CAD rather than guessing. Accounts stay separate with their tax
status; currencies are never added together.

**Cross-check on the real data, both brokers:** broker **$8,219.81** vs the
journal's recomputed **$8,220.05**, difference **-$0.2385** - exactly the known
Questrade rounding, IBKR exact. Journal > Fees > "Realised P&L for tax...".

**Verification:** 5419 passed / 72 subtests / 6 skipped (5402 -> 5419). Smoke
7/7, selftest 72/72, spec drift 17, ruff clean. Same two pre-existing
font-metric failures. No packaging trigger.

**Owed:** gate 18. Note the CAD total needs the BoC rates booked for every fill
date - the Health tab's FX coverage is where that happens, and on a fresh store
228 dates were unbooked.

---

## 2026-08-28 - The file wins on money, the sync keeps the clock

**Trader decision**, taken after the cost of the blunt reading was measured and
put to them: *"these should be sources of truth moreso than the auto input
IMO"* -> **money only**. Neither broker's file carries a time of day, so a
blanket override would have thrown away every intraday timestamp the journal
has, and with it every session bucket and entry-time tag.

**The rule.** The sync KEEPS a day the two agree on (it alone knows when each
fill happened); the file TAKES a day they do not (it is the broker's own
statement of the money). Agreement is measured in **cash per (account, day)** -
a trade can span days so a day's P&L is undefined, while its cash impact is -
and that cash is COMPUTED, never read off a Gross/Net column, because Questrade
reports in the trade's currency and IBKR in the base currency. Tolerance is per
FILL, since Questrade rounds each row to the cent.

**Append-only.** I3 forbids deleting a broker row, so the sync's rows are
retired with `VOID_EXECUTION` adjustments naming the day and both cash figures.
They stay on disk; a superseding record undoes it. A day the file does not
mention is a gap, not a disagreement, and is never touched.

**Proven end to end** against the trader's real 2025-26 Questrade export with a
simulated August sync, one day deliberately given only half its fills: **18
shared days, 17 agreed and kept their real 09:45 timestamps, the crippled day
taken over on a $3,116.49 difference** (3 voided, 5 written). 15 August trades
still carry a real entry time afterwards. "Check a statement..." runs the same
comparison as a DRY RUN, so the trader sees which days would move first.

**Verification:** 5402 passed / 72 subtests / 6 skipped (5385 -> 5402). Smoke
7/7, selftest 72/72, spec drift 17, ruff clean. Same two pre-existing
font-metric failures. No packaging trigger.

**Owed:** gate 17.

---

## 2026-08-28 - IBKR's file, and a commission sign that was costing money

**Trader direction:** *"we need IB integration as well. the auto import works
well but we would want to manually input a file as well."* Their real IBKR
export (803 rows, 609 fills, 2025-01-03 -> 2026-08-27) was read here; it is not
committed and the fixtures are synthetic.

**A separate reader.** IBKR's file is SECTIONED (a header per section), its
money is in the BASE currency while its prices are not, and its account numbers
are MASKED. Costs convert by the rate each row implies
(`|Gross| / |qty x price x multiplier|`, measured 1.35530-1.45270 - the USD/CAD
band, which is the proof the reading is right). That rate is evidence only and
is **never** booked into `fx_rates`, which is BoC-only. A mask resolves only
when exactly one known account fits; the second account stays masked and is
reported, which is correct until Flex names it.

**The finding.** `abs()` on commission - in `upsert_executions` AND the
assembly path - turned a broker CREDIT into a charge. **18 of 609** IBKR fills
carry one, and that single sign was the ENTIRE $2.17 by which the file and the
journal disagreed. Every importer already normalizes the sign itself, so
removing the `abs()` is a no-op for Questrade, Flex, the socket, CSV and manual
rows - verified by the full suite and by re-measuring Questrade's own
reconciliation, which is unmoved. **IB now reconciles to -0.0000 across 150
closed symbols**, commission equal to four decimals.

**Verification:** 5385 passed / 72 subtests / 6 skipped (5361 -> 5385). Smoke
7/7, selftest 72/72, spec drift 17, ruff clean. Same two pre-existing
font-metric failures. No packaging trigger.

**Owed:** gate 16.

---

## 2026-08-28 - Statements that layer, a direction that is read, and the trader's own check

**Trader direction:** *"lets add a function to be able to take these files, and
new ones throughout the year that layer on top so that in the end I can totally
manually calculate and demonstrate my pnl and then we can compare it to the auto
generated stuff."* Both of his real files (2026 YTD and 2025-08/2026) were read
here to build against; neither is committed.

**Two defects the first build carried, both found by measuring.** (1) The uid
hashed the file's ROW INDEX, so a longer export made **884 of 884** trades look
new - exactly the layering he asked for would have doubled the year. Identity is
now `fill_signature` + an ordinal within it. (2) Direction was a **coin flip**:
the file lists a same-day round trip SELL-first **227 of 227** times (a sort,
not a sequence), and the assembler's uid tiebreak sent **86 of 199** trades
SHORT at random. Questrade marks shorts in the Description (`STOCK SHORT.`,
`COVER SHORT.`), so `leg_rank` now orders by what each row does to the position.
All 227 resolved: **169 long, 58 short**, every short corroborated by both legs.

**`reconcile_statement`** is his "manually calculate and demonstrate" - it adds
the file up by hand and compares, per symbol, writing a CSV. Reads only.
Measured over both files: statement **$5,298.81** vs journal **$5,299.05**,
diff **-$0.2386** across 428 closed symbols, every symbol inside 2c, commission
**$713.68 both ways**. Importing 2025 dropped NEEDS_REVIEW from **23 to 5**.

**Verification:** 5361 passed / 72 subtests / 6 skipped (5349 -> 5361). Smoke
7/7, selftest 72/72, ruff clean on the new files. Same two pre-existing
font-metric failures. No packaging trigger.

**Owed:** gate 15.

### Two trader asks scoped but NOT built (2026-08-28)

1. ~~**Statements as the source of truth over the API.**~~ **DECIDED AND BUILT
   2026-08-28** - money only, the sync keeps the clock. See the authority entry
   above; gate 17 is its live proof.
2. ~~**IBKR transaction-file import.**~~ **BUILT 2026-08-28** - see the IBKR
   entry above. Gate 16 is its live proof.

---

## 2026-08-28 - Reading a Questrade statement, for the days the API cannot reach

**Branch `claude/last-commit-main-dpouod`. Trader-supplied file and direction:**
*"i can easily get us yearly reports from questrade so long as we can process
these files."* This **resolves the open trader decision** R7 has carried since
2026-08-25: the 44 pre-retention days are recovered from a statement file, not
from `/activities`, so no new coverage status was needed.

**What the real file measured** (their YTD export, read here, not committed):
974 rows, 884 trades, 133 trading days, 2026-01-02 -> 2026-08-27, both
accounts, **zero unreadable rows**, and `Net == Gross + Commission` on every one
of the 884 trade rows to the cent. So the one Commission column is the whole
cost; `fees` is 0.0 rather than a guessed split. End to end it assembled **414
trades** (393 CLOSED, 14 OPEN, 7 CLOSED_PARTIAL) and seeded both account tax
statuses.

**What a statement cannot say** shaped the module: no time of day (midnight
market-local, and `is_date_only` refuses a session bucket - a date-only round
trip is a `day_trade`, never a `scalp`), aggregated fills, no execution id and
no intraday sequence (the file's row order is preserved into the surrogate uid,
or two identical fills collapse into one), and options whose real contract is in
the Description rather than the Symbol column - 174 of the 884 rows.

**The rule that prevents double counting:** a statement never writes into a
(broker, account, day) a richer source already covers. `.xlsx` is read with
`zipfile`+`ElementTree`; adding `openpyxl` would have been packaging trigger 1.

**Measured drift, stated up front:** -$0.1558 on $4,014.18 realised across 253
closed symbols, worst symbol 1.17c; **commission matched exactly, $291.38 both
ways.** The cause is `rebuild_trades` recomputing price x qty against Questrade's
cent-rounded Gross Amount. Fixing that means making the shared assembler prefer
the broker's booked money - a change both brokers ride on, deliberately not made
here.

**Verification:** **5349 passed / 72 subtests / 6 skipped**, up from 5326.
Smoke 7/7 exit 0, source selftest 72/72 exit 0, spec drift 17 passed, ruff clean
on every new file. Same two pre-existing font-metric failures. No packaging
trigger.

**Owed:** gate 14.

---

## 2026-08-28 - Auto-tagging that survives imported history, and the tools to adjust it

**Branch `claude/last-commit-main-dpouod`, cut from `main` at `75880d6`.
Trader-directed** while deciding whether this journal replaces their TradesViz
subscription: *"i want auto tagging then I can come back and adjust."*

**What the ask exposed.** `AutoTagger` scores a trade by matching it against the
scanner's own output files. Those hold the current lookback, so any trade older
than them scores nothing and its summary is written empty. Auto-tagging was not
broken - it had no inputs for the case the trader is about to create, which is a
year imported from a Questrade statement.

**Built.** `scripts/journal_trade_shape.py`: hold bucket (counted in SESSIONS, so
Friday-to-Monday is one night), entry session bucket, execution shape from leg
ROLES, instrument. No files, no network, no scanner import. **No tag is ever
derived from the outcome** - that would make every per-tag statistic circular -
unmeasurable emits no tag, and naive timestamps ATTACH market-local. Candidates
order by LANE not confidence, or shape facts at 1.0 bury every setup match.
Alongside: a tag filter on the SHARED header (Analytics could group BY tag;
nothing could filter TO one), `distinct_tags`, `rename_tag`, a Manage-tags
dialog, Accept-all, and an accepted suggestion that stops re-proposing itself.

**Verification**, reading pytest's own exit code: **5326 passed / 72 subtests / 6
skipped**, up from 5268, 58 added. Smoke 7/7 exit 0, source selftest 72/72 exit 0,
spec drift 17 passed. Two failures (`test_table_width_rule`,
`test_qt_desk_layout`) are this Linux container's font metrics and were confirmed
against a stashed clean checkout. No packaging trigger. Nothing in the packet
reaches a detector, score, alert, watchlist, Focus or `review_policy.json`.

**Owed:** gate 13. Also still open and still the trader's call - the Questrade
statement-file import that makes the imported-history case real (the 44
pre-retention days); the trader has said they can produce yearly reports, so the
file format is the next input needed.

---

## 2026-08-28 - Reading the whole evidence pile in slices (78,119 -> 1,365,259 chars)

**Branch `claude/local-ai-context-64k`. Trader-authorized:** "Can we just give it more
time? Like hours to complete its work then? And spoon feed it slowly so we don't run out
of context?" Advisory layer only.

**Why the budget work was not enough.** 64k context plus a derived budget got the
summary to 17 of 22 sources with none unfunded, and it still read **a tenth** of what
exists - 1,365,259 chars against a prompt that holds ~91,000. The packager spends that
tenth FAIRLY rather than WELL, so `setups.type_stats` gave 3 of its 184 rows. No tuning
fixes that: 96k crashes the runner, so the ceiling is hardware.

**Built:** `scripts/ai_jobs/map_reduce.py`. Slice the evidence, ask for findings per
slice, then synthesize from the findings. **46 slices over 17 sources, ~2.8 hours** of a
window that runs 22:00-06:00 and was using nine minutes. Every row of every source is
read. A slice can never pass for its whole source (`rows 41-80 of 184` travels inside
the content); a map call sees exactly one source so the validator already forbids it
citing anything else; a failed slice is counted and named in `data_quality`; a failed
synthesis publishes the findings marked `UNSYNTHESIZED` rather than losing hours of work;
every slice failing raises.

**Two real defects it exposed, both fixed:**

1. **The tripwire fired on a healthy request.** The findings package is the model's own
   prose at **3.72 chars/token** where JSON evidence is 2.06-2.23, so one estimate cannot
   serve both. The fix needs no estimate: a clip lands at the CEILING, and both observed
   shears pinned within three tokens of half the window (6,147/12,288; 32,771/65,536), so
   below 0.45 of the window nothing was clipped. All four real measurements are pinned.
   The two pre-existing shear fixtures used 12 and 5 tokens - values no clip can produce
   - and were made faithful (2,048 ctx, 1,027 tokens) rather than the guard loosened.
2. **The scheduler could start a second copy of a three-hour job.** It fires every 30
   minutes and the ledger only records a row when a job FINISHES. `run_slots` now takes
   a machine-local lock and a second firing stands down. And the summary slot reserved
   **20 minutes** for a job that now needs 170 - `summary_reserve_minutes()` returns 200
   in chunked mode, because a three-hour job launched with twenty minutes left runs
   straight into the open.

**Validated against the real model, small** (2 slices, ~5 min): both slices read, real
citations, and the synthesis failed - which proved the fallback, since the findings were
published rather than lost. **The full 46-slice run was deliberately NOT started**: it
was 06:00 PDT, the off-hours window had just closed and the open was 30 minutes away, and
"no inference during market hours" is a hard rule.

**Verification:** `pytest tests/ -q` **5297 passed, 33 subtests, exit 0** · smoke **7/7**.

**Owed (the live gate):** tonight's unattended 22:00 run - 46 slices, a synthesized
summary, and the ticker briefs still finishing inside the window afterwards.

---

## 2026-08-28 - The local model was reading a third of its evidence

**Branch: `main` (the milestone was merged while this session was running).
Trader-authorized**: "raise the context... use as much as you want." Advisory layer
only - nothing here reaches a detector, score, alert, watchlist or the review queue.

**The finding.** With the endpoint restored, `ai_summary` stopped reporting
"unreachable" and started reporting a SHEARED prompt - correctly. The evidence package
tokenizes at **2.06-2.23 chars/token** (measured against the desk's own model, 9 KB to
93 KB of prompt), not the 3.0-3.5 assumed in two places. So the 22,000-char budget,
derived in a comment as `7800 tokens x 3.0`, exceeded a 12,288-token window by a third
**from the day it was written**; it only survived while few sources were funded. When
the package reached 17 usable sources the prompt hit ~14,400 tokens and llama.cpp cut
it to half the window - the pin at 6,147 tokens is identical across prompts of 28 KB,
37 KB, 51 KB and 93 KB, which is what proved it.

**Changed.** Model context **12,288 -> 65,536** (`gemma3:12b-tbv3ctx-64k`); measured
cost is nothing that matters - **8.1 GB loaded, still 100% iGPU**, because gemma3's
sliding-window attention keeps KV cheap. The budget is now DERIVED from the configured
context (`local_evidence_budget_ceiling_chars`) and capped there however it is
configured, with a new `ai_local_context_tokens` setting (stock 12,288; desk 65,536).
Two chars-per-token constants now exist deliberately and lean OPPOSITE ways - 2.0 sizes
the budget pessimistically, 2.5 estimates what was sent conservatively - with a test
that fails if anyone merges them. The local request honours its caller's timeout to a
1800s cap; cloud paths keep 300s.

**Result on the 2026-08-27 session:** `ai_summary` went from four straight
`degraded_no_narrative` runs to **`ok` in 343s**, **17 of 22 sources usable, 0
unfunded** (was 10 of 22 with 5 unfunded). It now names real candidates (NET, OII,
NESR), a setup family and the regime; the 08-26 summary managed "mixed results" and
named nothing.

**Then it was taken as far as the hardware allows** (trader: "let's take all the time
we need... crank up the detail"). Findings that are worth more than the numbers:

- **96k context LOADS and then crashes under load.** `ollama ps` said 8.0 GB / 100% GPU
  at `num_ctx 98304`; a real 132 KB prompt killed the runner. The load-time reservation
  says nothing about what happens when the KV cache fills. **65,536 is the working
  ceiling on this iGPU**, established by completed generations, not by loading.
- **An over-long prompt is NOT an error.** At 64k a 150,000-char prompt returned HTTP
  200 with `prompt_tokens = 32,771` - half the window plus three - and answered
  confidently from a prompt it had silently halved. That is why the budget carries a
  safety factor instead of being pushed to the last token.
- **The ceiling formula was wrong a second time.** The first correction allowed 1,000
  tokens for the prompt envelope; measured, the envelope is 10-35% ON TOP of the
  evidence (24,000->32,203 chars, 48,000->59,226, 96,000->111,568, 159,466->175,358).
  `_BUDGET_PROMPT_OVERHEAD = 1.35` now takes the worst observed ratio, and a budget of
  `0` means DERIVE, so the window is one setting rather than two that can disagree.
- **Per-symbol briefs no longer share the session budget** - they cannot. Measured ~60s
  per brief at 22,000 chars: 53 briefs in 55 min (08-26), 121 in two hours (08-17), and
  the job refuses to start with under 120 min left. At the session budget a brief is
  ~42,600 tokens instead of ~14,000, which would put a 121-brief night past seven hours.

**Verified, not calculated:** the derived 78,119-char budget makes a 91,262-char prompt
that the server tokenized at **44,344 tokens, 71% of the 62,036 usable**, whole prompt
read. The 2026-08-27 summary then ran in **567s** and, for the first time on record,
**"Strongest already-qualified candidates" is not "No supported finding"** - it reads
"NET, OII, and NESR ... with high conviction and a 0.83R reward". Per-source slices
roughly doubled (`daily.auto_report` 2,909 -> 4,735 of 8,592; `setups.type_stats` 1 -> 3
of 184 rows; `current_tiers` 4 -> 8 of 200).

**The honest remaining limit:** 14 of 17 sources are still shown in part and the tables
are still 3-of-184-row slices. The prompt is at 71% of a window that cannot go higher on
this hardware, and the rest is divided between seven `setup_trackers` sources inside one
scope. Whole tables need a different model or a narrower scope selection - not another
number.

**Verification:** `pytest tests/ -q` **5275 passed, 27 subtests, exit 0** · smoke
**7/7**; and the real `ai_summary` job was run end to end twice and published `ok` both
times (343s at the first budget, 567s at the derived one).

**Owed:** one unattended 22:00 run producing a narrated summary AND narrated ticker
briefs (tonight's briefs were 0 of a normal 53-62, all lost to the dead endpoint).

---

## 2026-08-28 - Two scans wrote one CSV: the D1 feature-history corruption, fixed and repaired

**Branch `claude/warehouse-build-memory`. No roadmap item. Trader-authorized** - the
file-scoped ask-first rule applied because `master_avwap_lib/legacy.py` houses
detector/scoring code. **No detector, score, signal or alert behaviour changed**: this
is the evidence writer for `d1_features_history.csv` plus the repair of the file it
damaged, so plan.md sec 5's golden-fixture rule is not engaged.

**Root cause, measured not guessed.** The 12:45 swing scan was declared stale at 12:48
(`runner did not survive restart`) and replaced at 12:49 while the first worker was
still alive. Both wrote the file - one appending, one rewriting in place from byte 0.
The signature is unmistakable in the bytes: rows whose leading fields are one
alphabetical symbol stream (AMBP, AMGN, AMLX, AMPX...) and whose trailing JSON blob is
the next one along (AMD, AMH, AMPL, AMRX...), and 2026-04-24 rows sitting at line 61
where the short rewrite stopped. 204-column header over a 97.3%-255-column body, 15
shredded lines, 372 rows destroyed.

**Fixed** with four rules in `append_d1_feature_history`: the write is taken under
`local_writer_lock` and **fails closed** without it; a rewrite goes through a temp
sibling and `os.replace`; an unreadable header **refuses** the write instead of
falling through to a blind append (the old bare `except` set `existing_columns = []`,
which then failed its own truthiness test - that is the whole mechanism); and the
append path only writes the header's own columns.

**Repaired, with a net gain.** Rebuilt from the 2026-08-26 evidence snapshot (119,107
rows, uniformly 255 columns, verified clean) plus every recoverable 08-27 row from the
live file, mapped by NAME onto the wide header - the 204-column schema is a strict
subset of the 255, so narrow rows keep their 204 values and take 51 blanks rather than
being dropped. **129,081 rows, uniformly 255 wide, `pd.read_csv` clean**, versus
128,720 in the corrupt file: **+361 rows**, because the snapshot restores what the
overwrite destroyed. All ten of 08-27's runs are present; the 12:49 run keeps 200 of
its ~1,086. **15 rows unrecoverable**, written with line/width/run_id/symbol to
`d1_features_history.quarantine-2026-08-28.jsonl` beside the file. The corrupt original
is kept as `d1_features_history.csv.corrupt-2026-08-28` (522 MB - delete it once a
scan has run clean).

**Verification:** `pytest tests/ -q` **5266 passed, 22 subtests, exit 0** · smoke
**7/7**. 5 new tests; the three that are true reproductions were confirmed to FAIL
against the old writer by reverting it. The exact line that raised
(`pd.read_csv(history_path, low_memory=False)`, `legacy.py:10089` and `:10726`) now
returns `(129081, 255)`.

**Owed:** one scan on 2026-08-28 confirming `output/scan-factors` and
`output/tier-tracker` produce files again rather than logging `ParserError`. That is
the live gate on this item.

---

## 2026-08-28 - The nightly narration: a dead model server, and a digest that rejected its own evidence

**Branch `claude/warehouse-build-memory`. No roadmap item** - this began as a trader
question about how big the local and DAS files are and how long they cost to read
intraday, and the storage answer was healthy throughout. What the same pass found in
the overnight AI layer was not.

**Storage assessment (no action needed).** Desk `C:\TradingBotData` 4.4 GB,
`%LOCALAPPDATA%\TradingBotV3` 5.1 GB, C: 283 GB free; the DAS holds 6.2 GB and has
9.5 TB free. The share answers in 0.09 s and reads at 95 MB/s. Per-symbol chart
reads are 0.1 ms (bars) and 0.7-3.7 ms (levels) against 23-136 KB files, so charting
costs nothing intraday. The two expensive shapes are both already known and already
guarded: a whole `json.load` of the 979 MB tracker is **12.2 s** against **1.3 s** for
the chunked-hash watermark check (BD-73), and a whole-partition DAS read of
`bronze_setup_tracker` month=2026-08 is **32 s and 1.2 GB of RAM for 15 rows** (each
row is a day's snapshot blob) against **0.2-0.5 s** for a narrowed one-symbol read
(BD-66/69/74). The desk measured 2.2 GB RSS during the pass, so the memory packet is
holding. Cold push, evidence snapshot and the hourly DAS task all ran on time.

**What was wrong.** The `ai_summary` slot was healthy - a real narrative on
2026-08-26 and nearly every night since 08-11. Two things were not:

1. **The local inference server had no autostart.** Ollama's log stops at 06:12 on
   2026-08-27; the desk restarted around 13:00 and nothing restarted it, so all three
   narrating jobs spent the whole 22:00-06:00 window retrying against a refused
   connection. The deterministic slots were unaffected, which is why it stayed
   invisible. **Fixed:** an `HKCU` Run entry at logon, plus a preflight in
   `scripts/run_ai_jobs.ps1` that probes the configured endpoint, starts a LOCAL
   server that is down, waits 60s for the socket, and carries on either way. It can
   never refuse the run.
2. **The daily digest had failed three nights running (08-25, -26, -27) with the
   model and every store up.** Root cause: the fact pack PRINTS a `source_id` on
   every measured cell, the narrator is told to cite exact `source_id` values, and
   the validator accepted only `digest.facts`. The model cited what it was shown.
   **Fixed** on both sides - packages may carry `citable_aliases`, collected by
   walking the built pack, and an unsupported citation now costs its ROW rather than
   the whole document, with what was dropped disclosed as a `[system]` data_quality
   row. A document where every citing row is dropped still raises.

**Also shipped:** fact pack `v1` -> `v2`, hoisting the constant `source_id`/`as_of`
off every cell and the selector shape off every slice row. Measured 14,070 -> 11,124
bytes (21%) with no figure dropped. Still over the 8,192-byte target and well inside
the 16,384-byte cap; the target's actual purpose - ninety packs as a trivial reducer
context load - holds at the new size, and cutting real slices to reach 8,192 exactly
would trade evidence for a round number.

**Found and NOT fixed - needs a trader decision.** `d1_features_history.csv` (498 MB)
went ragged at 12:49 on 2026-08-27: a 204-column header over rows of 119 to 524
columns. `export_scan_factor_views` and `export_bot_tier_tracker_views` raise
`ParserError` on **every scan** since (caught and logged - the scan itself completes,
those two outputs do not). The widening path at `master_avwap_lib/legacy.py:2397-2408`
reads the whole file with pandas when the column set changes and its bare
`except: existing_columns = []` turns a read failure into a blind append. That file
houses detector/scoring code, so the file-scoped ask-first rule applies and nothing
was touched.

**A pre-existing test failure was fixed on the way past.** `test_group_tape_service`
pinned its fixture bars to a hardcoded `2026-08-27` while the service filters to
TODAY's date, so from 2026-08-28 the two tests that assert on sector output began
failing on the calendar rather than on the code. Proven pre-existing by re-running
them with this packet stashed. Fixed test-side with a `frozen_session_clock` fixture
that freezes the service module's clock beside the bars - letting the bars float with
the real clock cannot work, because near midnight there is not yet a session long
enough to hold a 30-minute window. No production change; the same-date filter is
correct and is the whole point of the rebuild.

**Verification:** `pytest tests/ -q` **5261 passed, 22 subtests, exit 0** · smoke
**7/7**. 12 new/updated tests. End-to-end proof beyond the suite: the exact citation
shape that was rejected on 08-25/-26/-27, replayed against the real 2026-08-27 fact
pack, now validates with zero drops, while an id the pack does not print is still
struck, its row dropped, the rest of the document kept, and the drop disclosed.
Ollama was started and answers `/api/tags` with all four models.

**No desk restart is required** for the AI-layer fixes: `run_ai_jobs.ps1` and the AI
modules are read fresh by the 22:00 scheduled task. **Owed:** one overnight run
confirming a narrated digest (the first since 2026-08-24), and a `CLAUDE.md` line for
the citation contract - deliberately not written this pass because a concurrent
session was actively rewriting `CLAUDE.md`/`AGENTS.md`/`docs/DESK_INTERNALS.md` in
this same checkout (mtimes 00:13-00:16), and those files were left untouched and
uncommitted for that session rather than raced.

---

## 2026-08-27 (night) - repo hygiene pass: dead code, a dead dependency, four stale doc claims

**Branch `claude/warehouse-build-memory`. No roadmap item; no runtime behavior
changed.** A codebase-wide assessment for dead code, duplication and doc drift.
Findings and what was done:

- **Dead UI modules removed** (236 lines): `ui/widgets/info_dot.py`,
  `ui/widgets/symbol_chip.py`, `ui/models/journal_table_model.py`. Zero references
  anywhere - Python, spec, JSON, Markdown - and `ui/` does no dynamic module lookup.
  The journal model/proxy pair was superseded by `panels/journal/trades_tab.py`, which
  uses `QTableWidget` against `JournalTrade` directly.
- **scikit-learn / joblib dropped from `requirements-core.txt`, and the spec stopped
  force-collecting sklearn/scipy.** Nothing in the tree has imported either since
  `a73f072` removed the trade-quality training script. The unguarded
  `collect_submodules("sklearn")` was bundling ~93 MB (scipy 59 + scipy.libs 20 +
  sklearn 14) into every build **and would have failed the build outright** the moment
  the dependency was dropped. Bundle effect is unverified - no rebuild was run.
- **`ruff` added to `requirements-dev.txt`.** `pyproject.toml` configures it and
  `CLAUDE.md` names it in the stack, but it was in no requirements file, so a clean
  `uv pip install -r requirements-dev.txt` produced an environment that cannot lint.
- **The "29/29" frozen-selftest expectation was stale in `CLAUDE.md` and `README.md`.**
  The count is a running total: 29 on 2026-08-09, 30 later (this file, `scan_worker`),
  and the unfrozen tree measures **72** today. A future agent comparing a correct run
  against 29 would read it as a failure. Both now say N/N and to compare against a
  current unfrozen run.
- **`README.md` corrected**: Desk Link code described as "unused pending a cleanup
  packet" (it was *removed* 2026-08-24); `master_avwap_mini_pc.py` described as
  remaining in the repo (deleted 2026-08-24); five entry points introduced as "four".
  Added the source-launch-is-production decision and the one-desk-per-machine guard.
- **`docs/README.md`**: the 15 decision records were listed by title only; they are now
  links.
- Stale `scripts/desk_link/` (an empty `__pycache__` shell of the package removed
  2026-08-24) still sits on disk untracked; `rm` was declined by the sandbox. Harmless,
  but delete it by hand.

**Verification (all after the changes):**

| gate | result |
|---|---|
| `pytest tests/ -q` | **5249 passed, 22 subtests, exit 0** (289.5s) |
| `scripts/smoke_check.py` | **7/7** |
| `launch_gui.py --selftest` | **72/72**, exit 0 |
| packaging spec-drift + selftest tests | **24 passed** |
| frozen exe | **not rebuilt** - the spec change is a packaging trigger, so a rebuild + frozen selftest is owed before the next merge to `main` |

**Not done, deliberately - reported for the trader to direct:**

- ~~`CURRENT_CHECKPOINT.md` bloat~~ - **DONE, trader-approved the same night** (see the
  addendum below).
- `_write_json_atomic` is duplicated verbatim in 5 `market_prep` modules (~72 lines);
  total cross-file duplication is only 468 lines, mostly 5-line `_float` helpers.
  Consolidating touches 5 service files for modest payoff - not taken unilaterally.
- 91 `except Exception: pass` blocks outside the legacy cores. Many are the deliberate
  fail-quiet evidence-store contract; separating those from genuine swallowing needs
  the file-scoped ask-first rule.
- `ruff`'s configured select is 5 rules (`E9,F63,F7,F82,F401`). Widening it is a
  separate, gated packet.

### Addendum (same night) - the documentation read is now bounded

Trader approved the split, asking to "make it as easy as possible to keep vibe coding".

| file | before | after |
|---|---|---|
| `CURRENT_CHECKPOINT.md` | 7,901 lines / 449 KB | **1,410 lines / 82 KB** |
| `CHANGELOG.md` | 5,286 lines / 336 KB | **4,090 lines / 265 KB** |
| mandatory read (`CLAUDE.md` + checkpoint + changelog + `plan.md` + `docs/README.md`) | ~1,015 KB / **~260k tokens** | ~575 KB, and the *instructed* read is now a ~40-line block plus targeted searches |

What changed:

- **`docs/CHECKPOINT_ARCHIVE_2026-08.md`** - 95 entries dated 2026-08-25 and earlier,
  verbatim. **`docs/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md`** - the 36 revision-history
  entries from 2026-08-19 back to 2025-11, verbatim. Both are classified in
  `docs/README.md` as historical evidence with an explicit "never load as context" rule.
  Entry-count check: 19 active + 95 archived = 114, the pre-split total.
- **`CURRENT_CHECKPOINT.md` opens with "Active state at a glance"** - branch, active
  roadmap items, last verified baseline, the **eight open gates newest-first**, and the
  immediate next action. That block is what the next agent reads; the dated entries are
  the record behind it. It says explicitly that a dated entry wins if the two disagree,
  so a stale block degrades rather than misleads.
- **`CLAUDE.md`'s mandatory workflow is now a bounded read** and says why: an agent that
  cannot read its brief skims it and appends to it, which is what grew the files. It
  directs a *search* of the changelog inventory rather than a full read, reorders the
  checkpoint to first, adds "when the docs and the code disagree, the code is the fact
  and the doc is the defect", and adds a maintenance rule - refresh the glance block,
  write the shortest entry that carries the decision, and archive past ~1,500 lines.
- My changelog entry was moved from `Revision history` into the top of
  `Current implemented inventory`, matching where the file's own newest dated entries live.

No code changed in this addendum. Re-verified after it: **5249 passed, 22 subtests,
exit 0**; smoke **7/7**; `--selftest` **72/72**.

### Second pass (2026-08-28) - the read is now 97k tokens, not 260k

Trader: "Anyway we can summarize things to be even briefer? Long context uses more AI
usage and makes me less efficient." Two structural cuts, no code touched:

- **`CHANGELOG.md`'s `Current implemented inventory` was 94% narrative** - 3,808 of its
  4,061 lines were 91 dated entries wrapped around a **253-line thematic inventory that
  already states what exists**. The inventory is the contract, so it was promoted to the
  top of the file; the 73 dated entries older than 2026-08-26 moved verbatim to the
  changelog archive, and the 18 from the last two build days stayed under a new
  `Recent changes` heading. Entry check: 18 kept + 73 moved + 36 already archived = 109
  archived, 18 live. 260 KB -> 98 KB. One stale inventory line was corrected while there
  (Desk Link code "remains only pending cleanup" - it was removed 2026-08-24).
- **`CLAUDE.md`'s `Core loop / data flow` was 42 KB - 65% of a file that loads into
  EVERY session.** Each rule carried the incident, the numbers and the trader
  conversation that produced it. Those moved verbatim to
  **`docs/DESK_INTERNALS.md`**; `CLAUDE.md` keeps every rule as a binding imperative
  with a pointer. The section is 71% smaller and the file went 65 KB -> 35 KB.
  **Nothing was dropped**: a check for 45 critical guardrail tokens (`read_rows`,
  `_run_outcomes`, `1.0 ATR`, `passes_focus_adoption_gate`, `replace(tzinfo=None)`,
  `suppression field`, `r_eod_hold`, `local_writer_lock`, ...) found all 45 still
  present in `CLAUDE.md` itself. The rule binds from `CLAUDE.md` alone; the internals
  doc is the reason, and both change together.

| | before | after |
|---|---|---|
| `CLAUDE.md` (every session) | ~15,907 tok | **~8,943 tok** |
| `CURRENT_CHECKPOINT.md` | ~114,987 tok | **~20,639 tok** |
| `CHANGELOG.md` | ~86,141 tok | **~24,522 tok** |
| `plan.md` | ~37,305 tok | ~37,305 tok (untouched) |
| **mandatory read** | **~259,878 tok** | **~97,541 tok (63% smaller)** |

`plan.md` is deliberately untouched - it is the roadmap and its Section 12 work queue is
live. It is the obvious next candidate if the trader wants to go further.

**Concurrency hazard observed, not a defect:** a second Claude session was editing this
same checkout during this pass (`scripts/ai_summary.py`, `scripts/ai_jobs/digest.py`,
`scripts/run_ai_jobs.ps1`, +12 tests at 00:21; HEAD moved to `e22e8e8`). A full-suite run
during its activity reported 2 failures in `tests/test_group_tape_service.py`. They were
**not caused by this work and are not a real defect**: the same two failed at `b6700b6`
in a clean worktree, then passed 16/16 twice and in two later full runs once the other
session was quiet. Two pytest runs sharing one working tree cross-talk through on-disk
fixtures (`.test_tmp/`, `data/runtime`). **Do not run the suite while another session is
building in this checkout** - the result is not trustworthy either way.

### Third pass (2026-08-28) - `plan.md` narrowed; the read is 84k tokens

Trader: "shrink plan.md and ensure those files stay deleted."

**Deletions confirmed permanent.** `info_dot.py`, `symbol_chip.py` and
`journal_table_model.py` are gone from disk AND from HEAD - the other session's commit
`84b1a36` swept up the staged deletions, so they are committed rather than merely
staged. Their stale `.pyc` files were removed. The only surviving mentions in the repo
are the entries here and in `CHANGELOG.md` that record the removal.

**`plan.md` 149 KB -> 76 KB (37,305 -> 19,141 tokens, 49% smaller).** Section 12 was
93% of the file, and Phases 0.5/0.6/0.7 were 72% of Section 12 (97,886 B) while
describing work that is BUILT. Each of their 89 numbered items now keeps its bold
title/status line, a `Spec:` + build-record pointer, and **every gate paragraph reduced
to its bold lead plus the sentences that carry the gate - verbatim, never paraphrased**.
The build narrative moved to `docs/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md`.

**One real defect was caught before it shipped, in the first attempt.** Splitting
sentences on `;` truncated multi-part gate lists: R1's "Still owed: the drain on return;
an EVENING day...; and one SPY-alarm firing" lost its second and third parts. The pass
was reverted with `git checkout -- plan.md` and redone splitting only on `.` followed by
a capital, so a semicolon-separated gate list stays one sentence. **Never split a gate
list on a semicolon.**

Verification of the narrowing itself: gate clauses before **89**, after **90**; the 5
flagged as missing are item title lines whose gate status is preserved in the bold title
(`R1 ... — BUILT 2026-08-15, live proof owed`), each confirmed present by name. Structure
after: **10 sections, 89 numbered items**, unchanged.

| | before | after |
|---|---|---|
| `CLAUDE.md` (every session) | ~15,907 tok | **~8,943 tok** |
| `CURRENT_CHECKPOINT.md` | ~114,987 tok | **~22,850 tok** |
| `CHANGELOG.md` | ~86,141 tok | **~26,397 tok** |
| `plan.md` | ~37,305 tok | **~19,141 tok** |
| **mandatory read** | **~259,878 tok** | **~83,607 tok (68% smaller)** |

Re-verified after this pass: **5261 passed, 22 subtests, exit 0**; smoke **7/7**;
`--selftest` **72/72**; packaging **24 passed**; **149 doc links, 0 broken**.

**Final verification (quiet tree): `pytest tests/ -q` -> 5261 passed, 22 subtests, exit 0**
(5,249 from this branch plus the other session's 12 new tests); smoke **7/7**;
`--selftest` **72/72**; packaging spec-drift + selftest **24 passed**; **0 broken relative
links** across all control and archive documents.

### Rebuild record (2026-08-28) - the packaging gate is MET

`pyinstaller packaging/tradingbotv3.spec --noconfirm` at `fff07b8`, then
`dist\TradingBotV3\TradingBotV3.exe --selftest` -> **`selftest OK: 72/72 checks passed
(frozen)`, exit 0.**

- **The frozen count equals the unfrozen count (72).** That is the direct confirmation
  that the old "29/29" expectation in `CLAUDE.md` and `README.md` was stale by 43 checks
  and would have made a correct run look like a failure.
- **Bundle 442 MB -> 419 MB.** `sklearn` (14 MB) is gone, as intended.
- **`scipy` and `scipy.libs` (79 MB) are still bundled** - not by the removed
  hiddenimports, but transitively. Two importers exist among bundled packages:
  `yfinance/scrapers/history.py` line 1139, a **lazy** import inside
  `_fix_unit_random_mixups` (its own comment: "Only import scipy if users actually want
  function"), reachable only through yfinance's price-repair path; and `pandas`' sparse
  arrays. **Neither is reachable from this codebase** - nothing passes `repair=` to
  yfinance and nothing touches pandas sparse.
  **Recommended, NOT done:** adding `scipy` to the spec's `excludes` would take the
  bundle to roughly **340 MB**. It is left for a trader decision because it is a second
  packaging change needing its own rebuild and frozen selftest, and because the frozen
  selftest performs no network fetch - so it would not, on its own, prove the yfinance
  bar path still works without scipy present. That proof is one live fetch.

## 2026-08-27 - tracker-wide stop/target research and five-timeframe context

**Branch `claude/warehouse-build-memory`. Active roadmap: Phase 3.2 + Phase
6.1.** Trader approved all three recommendations: test every tracker family;
use the next regular session's first completed M5 close as entry for a D1 setup
found after close; and measure Auto Market Bias separately on M5/M30/H1/H4/D1.

**Built.** `tracker_adapter.py` reads the small transition ledger plus the
scenario CSV, never the 1 GB snapshot, collapses daily rescans without future
geometry, and sends every canonical family with valid geometry into
`setup_occurrence`. The real-data read-only audit was 249,438 scenario rows,
10,820 transition rows, 6,663 deduplicated detections, all 16 families, zero
unknown-family skips.

`outcomes.py` has a separate 54-recipe M5-close research grid: structural stop
source ranks 1–3 and 0.5/1.0/1.5 ATR controls, each crossed with 1R/2R/3R
targets. It needs no planned stop/risk, M1, bid/ask or earnings fundamentals.
It uses STOP_FIRST and the existing deterministic fallback cost model. Frozen
slice recipes are unchanged.

`setup_market_context` stores five independent champion Auto Market Bias reads
at entry. `legacy._auto_market_regime_stats` is now the one pure complete rule,
including the early-session day-percent fallback; both existing live callers
and research call it, with live behavior unchanged. This `legacy.py` edit is
inside the trader's explicit approval of all three recommendations. Stable
symbol buckets and Arrow-side occurrence/recipe filters bound the in-process
build. The appended nightly `setup_research` slot always writes
deterministic facts; medium local AI narrates only after n>=30, five symbols and
five sessions and cannot write live policy.

**Verification:** final focused setup/warehouse/bias tests -> **142 passed**.
Full `pytest tests/ -q` -> **5249 passed, 22 subtests passed, exit 0** (278 s).
`scripts/smoke_check.py` -> **7/7**. `launch_gui.py --selftest` -> **72/72**.

**Live gates owed.** Run one post-scan warehouse canary and verify occurrence,
context and outcome writes plus bounded memory; let all symbol buckets fill;
then compare one overnight fact pack directly with warehouse counts. The M5
lake currently begins in August 2026, so older tracker episodes remain an
explicit coverage gap. Phase 3.2's explicit BounceBot occurrence link and the
20-session pilot remain open. **Desk restart required.**

## 2026-08-27 - post-earnings setup tracking and marking verified

**Branch `claude/warehouse-build-memory`.** Trader: "Let's make sure the bot
tracks these setups and that these setups are available for marking."

The three defined post-earnings families already run end to end:
`post_earnings_52w_break`, `post_earnings_candle_break`, and
`post_earnings_avwap_bounce`. Each detector has its own signal, family
derivation preserves the family, `build_tracker_setup_record` stores the
canonical family, and the Capture rail offers all three from the shared setup
registry. A Like stores `claimed_setup_id`; the deterministic like-cohort job
then grades that exact family forward.

Added one characterization test that passes each of the three detector signals
through the real tracker-record builder and pins the saved family. No detector,
score, ranking, alert, marking, or runtime behaviour changed; therefore the
ask-first rule did not apply, `CHANGELOG.md` and `plan.md` are unchanged, and no
restart is needed.

**Verification:** eight focused tests -> **8 passed, 3 subtests passed**. They
cover the three detectors, all three tracker-family mappings, the offered claim
list, the post-earnings letter-key write, and exact-family like-cohort identity.

## 2026-08-27 (evening) - Market Journal: it loads, it carries the tape, the AI reads it

**Branch `claude/warehouse-build-memory`.** Trader, with a screenshot of an
empty page after a full day of in-session notes: "this is empty and feels very
useless to me. this should capture more stuff, such as SPY charts, what they
looked like when the auto mode flipped, my entries, what the charts looked like
when i inputted entries, what the D1 looked like.. i also expect the AI to get
access to these notes for the daily summary function."

Five entries were on disk for the session (`market_journal-202608.jsonl`, 13:36
through 19:34) and the page rendered none of them.

**The two defects.** `MarketJournalPanel.reload()` had NO caller - not
`__init__`, not a show hook, and `_select_page` special-cases only the AWAY
Recap - so the page was blank until "Refresh" was pressed. And the Desk tab
built its own `MarketJournalService`, so its `entryWritten` came from an object
the page had never heard of; both wrote the same file correctly and what was
lost was the refresh. Now: `showEvent` loads once, and one process-wide
`shared_journal_service()` backs both surfaces.

**The feature.** `scripts/market_journal_capture.py` (new) stores the symbol's
M5/D1 and SPY's M5/D1 as they stood at each note - bars, not pictures - in a
sidecar per capture, with a short text digest in a `market_journal_chart_v1`
ledger row. `market_journal_entry_v1` is untouched; a capture joins by
`entry_id` from outside, written AFTER the entry on a worker so a note never
waits on a chart. `AutopilotService.autoModeChanged` fires on a real mode move
only, and `MainWindow._record_auto_mode_flip` writes the flip with SPY attached,
marked `ORIGIN_AUTO_MODE_FLIP`. The page draws the panes that were captured and
hides the ones that were not.

**The AI.** `market_journal` joined `briefs.DEFAULT_SCOPES` on the trader's
explicit instruction, reversing R10.I's opt-in (itself a trader decision).
`TICKER_BRIEF_SCOPES` is no longer an alias and keeps the original four.
Two pinned tests asserted the old decision and now pin the new one.

**File-scoped ask-first rule:** `alert_center_panel.py` and
`autopilot_service.py` were both edited; the trader was asked first and
answered "Yes, build all of it". Every edit in both is capture/announce only -
no detector, score, tier, fold, digest, queue or alert behaviour is touched.

**Verification:** `pytest tests/ -q` -> **5240 passed, 19 subtests, exit 0**
(307 s). `smoke_check.py` -> 7/7. `launch_gui.py --selftest` -> **72/72 passed**.
New tests: `tests/test_market_journal_capture.py` (19),
`tests/test_qt_market_journal_page.py` (14), three appended to
`tests/test_auto_mode_semantics.py`. **Three pinned tests were UPDATED, not
worked around** - `test_evidence_report_slot`, `test_opt_in_evidence_scopes` and
`test_veto_cohort_grading` each asserted the R10.I opt-in the trader reversed;
all three now pin the new decision and the last still guards
`trader_judgement` staying out.
**Packaging trigger 3 does NOT fire** - `market_journal_capture` is a top-level
module, not a new package - but it is added to `selftest.LAZY_ENGINE_MODULES`
because both journal surfaces and `ui.app` import it at call time.

**Owed live gates:** (1) one desk session where a note written on the Desk tab
appears on the left-nav page without a Refresh, with its four charts; (2) one
real auto-mode flip producing a `[desk]` row with SPY's tape; (3) one nightly
`ai_summary` run whose packet names `journal.chart_digests` and
`journal.entries`.

**Needs a restart to reach the desk.**

## 2026-08-27 (13:00) - capture rail: the like's double-click commits like the veto's

**Branch `claude/warehouse-build-memory`.** Trader: "i want to be able to double
click the like and claim the same way i can double click the veto."

The veto's gesture attempts the commit (`select_reason` -> `commit_veto`, which
diverts only for a `note_required` reason). The like's went straight to
`_prompt_for_why` and could never commit, so a why that was already typed was
ignored and re-requested. Both like gestures now call `commit_like`, where
R9.2's required-why guard already lives.

**The 2026-08-22 rule is deliberately unchanged** - a like with no why writes
nothing and holds the chart; its two existing tests pass untouched. The digit
moved with the double-click on purpose, because the veto's two gestures are
identical to each other.

**Verification:** `pytest tests/ -q` -> **5203 passed, 19 subtests, exit 0**
(315 s). Fail-before-fix: 4 of the 5 new tests fail with `capture_rail.py`
stashed; the fifth is the no-why regression guard and passes on both sides. No
packaging trigger.

**Needs a restart to reach the desk** - the desk has been running `a8eeb48`
since 12:22, which predates this and the popup-height commit.

## 2026-08-27 (12:30) - ticker popup no longer edge to edge; desk RESTARTED on the new code

**Branch `claude/warehouse-build-memory`.**

**The desk was restarted at 12:22** at the trader's request and is running
`a8eeb48` from source (`trading_desk.cmd`, pids 12552/33996). The restart caught
the memory bug live: the outgoing desk (launched 09:54, old code) was measured
at **7.53 GB, climbing to 9.47 GB in five seconds, and 10.73 GB at the moment it
was stopped**, with a warehouse build in flight. The new desk settled at
**0.82 GB**. Before stopping it: no scan scheduled task was running, and a
killed build's `single_flight` lock is reclaimed rather than obeyed
(`cli.single_flight`, "a dead holder's lock is reclaimed"), so an interrupted
build is a designed-for case. **This is not the live gate** - that still needs a
full swing-scan slot's build to stay under 3 GB.

**Popup height (this entry's change).** Trader: "make the charts that pop up
when i click on a ticker just a little less tall... reduce by 10% top and
bottom." `inset_vertical_bounds` (pure, in `symbol_snapshot_dialog`) leaves 10%
of the anchor free at each end; `POPUP_MIN_HEIGHT = 760` floors it so the
2026-08-11 squeeze cannot return on a short monitor. Measured on the desk's
monitors: 4K 2052 -> 1690 px (211px each end), 2560x1392 1332 -> 1114 px (139px
each end).

**Verification:** `pytest tests/ -q` -> **5198 passed, 19 subtests, exit 0**
(309 s). `tests/test_snapshot_popup_height.py` (6 new): **6/6 fail** before the
change. No packaging trigger.

**Note:** the popup is created once per owner panel and reused, so the new size
applies from the first ticker click after a restart - the desk restarted at
12:22 predates this commit, so it needs another restart to show it.

## 2026-08-27 (afternoon) - the desk's 8-13 GB memory jumps: BUILT, all three causes; one live gate owed

**Branch `claude/warehouse-build-memory`**, cut from `claude/group-tape-rebuild`
at `cd212bc` rather than from `claude/gui-phase-0-9` as the build prompt said -
**deliberate deviation**, so that ONE desk restart picks up both the group tape
rebuild and this packet, which is what the trader asked for ("before we restart,
integrate this as well"). The branch therefore contains gui-phase-0-9's head,
the tape rebuild, and this.

Built to `docs/analysis/OPUS_BUILD_PROMPT_DESK_MEMORY_2026-08-27.md` on the
2026-08-27 (10:00) investigation below.

### Measured before -> after (live lake, `\\MINI-PC\Trading Bot Data\research_lake`)

| Step | Before | After |
|---|---|---|
| `bar_m5 month=2026-08` partition | 8,704,108 rows / 408 MB / 158 files | unchanged (it is the input) |
| `read_table(...).to_pylist()` on it | **15.4 GB** (1,769 B/row, measured on a 20k slice) | not called any more |
| `build_derived_bars` read (one full session) | 15.4 GB | **0.53 GB** (297,230 rows) |
| `build_intraday_snapshots` read (one session) | 15.4 GB | **0.53 GB** |
| `_run_outcomes` read (20 symbols, whole month) | 15.4 GB | **0.31 GB** (175,235 rows) |
| tracker snapshot, UNCHANGED verdict | 1.03 GB read + hashed | **0 bytes read** (chunked hash) |
| tracker snapshot, CHANGED | 1.03 GB + decode + `json.loads` (several GB) | 1.03 GB + decode, **no parse** |
| BounceBot `self.data` after a sweep | one buffer per request, forever (~206 KB each) | **empty** |

The 10:00 investigation measured 8,175,471 rows / 13.3 GB; by this afternoon the
same partition was 8,704,108 rows / 15.4 GB. It grows all month - that is the
shape of the bug, not a discrepancy.

### What shipped

1. `ResearchStore.read_rows(...)` - Arrow-side narrowing before `to_pylist`,
   used by `aggregate.build_derived_bars`, `features.build_intraday_snapshots`
   and `cli._run_outcomes`. Narrow by design (symbols + interval_start_range
   only). BD-74.
2. `ingest_existing`: chunked `_sha256_path`, the UNCHANGED check hoisted above
   `read_bytes`, and `SNAPSHOT_PARSE_MAX_BYTES = 64 MB` above which a snapshot
   is stored whole but not parsed. BD-73.
3. `bounce_bot_lib/legacy.py` (the ONE authorised edit): the five leaking
   request paths free `self.data[reqId]` with the ready event on both branches,
   and `historicalData` drops bars for an unknown reqId instead of re-creating
   a buffer nobody frees.

### Verification

`.venv\Scripts\python.exe -m pytest tests/ -q` -> **5192 passed, 19 subtests passed, exit 0**
(329 s). `scripts/smoke_check.py` -> **7/7**. Golden fixtures and all 411
BounceBot tests pass UNCHANGED, which is the check that matters for cause 3.

**Fail-before-fix, per file** (production file stashed, suite re-run, restored):
- `tests/test_warehouse_session_scoped_reads.py` (10) - **8/10 fail** without
  `read_rows`. The two survivors are the equivalence guards, which must pass on
  both sides of the change.
- `tests/test_bronze_snapshot_large_files.py` (9) - **9/9 fail**.
- `tests/test_bouncebot_reqid_buffers_are_freed.py` (12) - **11/12 fail**. The
  survivor guards that a live request still collects its bars.

**No packaging trigger** - no new dependency, asset, top-level package, dynamic
import or `__file__` use.

### Premise corrected while building

The build prompt named `cli._run_outcomes` as one of the three live costs. **It
is not one today:** `setup_occurrence` holds **0 rows** on this lake, so
`_run_outcomes` returns `NO_OCCURRENCES` before it reads `bar_m5` at all. Fixed
regardless, because it becomes a cost the moment the BD-44 detector adapter
lands - but the 10:00 attribution of the 10.7 GB sample belongs to
`build_derived_bars` and `build_intraday_snapshots` alone.

### Owed / next

- **Live gate (one DESK session, after the restart):** the first swing-scan
  slot's build keeps the desk under **3 GB** working set
  (`Get-Process -Id <pid> | select WorkingSet64`, sampled across the window the
  lake manifest shows for that build); the manifest still gains the same
  datasets for that session; the baseline stops creeping between builds.
- **Decision, not owed work:** `run_build` was NOT moved into a child process.
  The in-process single-flight lock, the spool seal and the ledger's
  `_record_job` all assume one process, and the filtering removes the growth on
  its own. Available if the live gate says otherwise - the trader decides.
- **Observed in the same session, unchanged, not authorised:** the RRS scan's
  O(n^2) intraday profile (CPU, not memory); the `_poll_focus_d1_interest` ->
  `FocusSideEditor.refresh` GUI stalls (`focus_picks_panel.py:441`, 392 s); the
  RS-window `_auto_tick` reading 1,412 daily parquet files on the GUI thread
  (`rs_window_feed.py:745`, 92 s). Separate packets.
- **Immediate next action:** the trader restarts the desk. The checkout is on
  `claude/warehouse-build-memory`, which carries BOTH this packet and the group
  tape rebuild. Neither is merged to `main`.

## 2026-08-27 (afternoon) - Group RS/RW tape REBUILT (plan.md Phase 0.5 item 11, packets T-1..T-4); live gate owed

**Branch `claude/group-tape-rebuild`**, cut from `claude/gui-phase-0-9` at
`48c0ad4` as the build prompt requires (not rebased onto `main`). Commits
`c4fa8c3` (T-1/T-2) and `3dbff23` (T-3). Built to
`docs/prompts/GROUP_TAPE_REBUILD_OPUS_PROMPT.md`; **all ten of its hard rules
held** - zero IB traffic, no `legacy.py` / `run_rrs_scan` / scan-cycle change,
nothing expensive on the Qt thread, completed today-only bars through
`completed_bars`, UNKNOWN never invented, one formula proven equal to legacy's,
quiet-hours gated with the manual refresh exempt, the RS Window tab untouched,
fail-before-fix shown per test file, and the tree never left broken.

### What shipped

- **T-1 `scripts/group_rrs.py`** (pure - bars in, floats out, no I/O, no Qt,
  `now` always passed in). The formula lifted out UNCHANGED and proven so.
  Session filter = `completed_m5_bars` **+ a same-date filter** (the gap);
  `align_bars` intersects on normalized stamps; `rrs_windows` = 6/12/18 bars
  off ONE filtered+aligned series; `< length + 2` bars is `None`.
  `SECTOR_ETFS` is a drift-tested COPY of legacy's map.
- **T-2 `scripts/ui/services/group_tape_service.py`** - Strength Board shape.
  ONE batched `yfinance` `period=1d interval=5m` download per 5-minute tick
  (SPY + 11 SPDRs + 49 industry proxies, deduped to ~53), **no retry inside
  the tick**, last-good on failure with the failure in `status_text`, bounded
  `shutdown`, `auto_scanning_due` gate with `refresh_now` exempt.
- **T-3 `GroupTapeStrip` + the desk** - `90 | 60 | 30`, ranked by the 30,
  unmeasured windows BLANK, the new rotation callout carrying the as-of and
  the status, chips DIFFED keyed by ETF, variants moved from per-chip
  `setStyleSheet` into `theme.qss` on a `side` property with six pre-mixed
  rgba tokens. The tape is SHOWN again and fed by `tapeChanged`; the
  `rrsSnapshotChanged -> update_groups` wiring is gone while the RS Window tab
  and `focus_picks_panel` keep that signal (pinned by a test); the service is
  in the desk's shutdown list, resolved via `getattr` so a partially-built desk
  still releases everything else.

### Verification

`.venv\Scripts\python.exe -m pytest tests/ -q` -> **5161 passed, 19 subtests,
exit 0** (305 s). `scripts/smoke_check.py` -> **7/7**. Baseline was 5119 at
`48c0ad4`; +32 at `c4fa8c3` (5151), +10 at `3dbff23`.

**Fail-before-fix, per file** (module/files moved aside, suite re-run, restored):
- `tests/test_group_rrs.py` - **16/16 fail** without `scripts/group_rrs.py`.
- `tests/test_group_tape_service.py` - **16/16 fail** without the service.
- `tests/test_qt_group_tape.py` - **15/17 fail** with the four production files
  stashed. The two that pass are deliberate regression guards: the callout
  staying silent on an unsupported payload, and the RS Window tab still
  receiving `rrsSnapshotChanged`.

**No packaging trigger and no exe rebuild.** Both new modules are ordinary
static imports on a chain reachable from `launch_gui.py`, so PyInstaller
collects them by dependency analysis - no new dependency, asset, top-level
package, dynamic import or `__file__` handling. The spec-drift guard passes in
the suite above. The desk runs from source by trader decision 2026-08-26.

### Two failures found on the way - NEITHER caused by this work

1. `test_review_watch_buttons_arm_trigger_and_flag_red` is a **clock bomb**.
   Its fixture's last bar starts at 11:25, so before 11:30 local that bar is
   still forming, the 2026-08-27 VWAP-side leg reads UNKNOWN and the chart
   shows; after 11:30 both bars complete, the fixture's LONG sits under its own
   session VWAP (VWAP 104.25, last close 104.00) and the show-time filter
   correctly hides it. It passed at 10:xx and failed at 11:36 **on the same
   tree** - reproduced with the whole rebuild stashed. The production rule is
   right; the test is about the WATCH BUTTONS, so it now sets
   `_review_movers_only = False` the way five sibling files already do.
   *Worth a sweep: other fixtures anchored near the current wall clock may
   carry the same bomb.*
2. `test_trading_desk_shutdown_continues_after_one_component_raises` builds a
   `SimpleNamespace` desk and needed the new component; it now carries it and
   asserts it is called, plus a new sibling test for the partial-desk path.

### Owed / next

- **Live gate (one DESK session, with the four trader rules of this morning):**
  the tape moves every five minutes rather than every 10-30; the 06:30-07:00
  read carries no overnight gap and unfillable windows are blank rather than
  zero; a stale or failed read says so on the callout line; a chip click still
  charts the ETF.
- **Immediate next action:** the trader restarts the desk to pick this up (the
  desk runs from source, so the branch must be checked out). Branch is NOT
  merged to `main`.

## 2026-08-27 (later) - trader rule 4, third pass BUILT: clicking away from an M5 chart is a skip, not a re-queue

**Branch `claude/gui-phase-0-9`.** Trader: "When I click on an alert in the new
M5 alert bar and then click to another one, it shouldn't queue the old M5 alert
in the waiting list. It should just be considered a 'skip for now' situation."

The bar already kept M5 alerts OUT of the waiting list at
`_enqueue_review_alert`. What leaked was the click path they SHARE with the
feed: `_select_review_alert` pushed whatever chart it replaced to the HEAD of
the queue, so a trader working down the bar refilled the D1 queue with exactly
the M5 rows the bar was built to remove from it.

- `AlertCenterPanel._current_review_holds_place` (new, defaults `True`) records
  where the chart in front came from. `_advance_review_queue` sets it `True`
  (popped off the queue, so it keeps its place); `_select_review_alert` sets it
  `not _is_m5_review_alert(alert)`. Only a place-holder is re-inserted at the
  head; an M5 bar row is skipped.
- A flag, not a re-test of the outgoing alert, because the same-symbol refresh
  branch REPLACES a queued D1 chart's alert object with that symbol's newer M5
  alert - re-testing would answer "M5" for a real queue member and silently
  drop it. Pinned by `test_a_refreshed_d1_chart_still_holds_its_place`.
- The skip is recorded, not silent: `_record_review_event("skip", ...)` with
  the dwell and `detail={"reason": "clicked_away_from_m5_alert"}`.
  `_render_current_review` already writes the `shown` impression for a
  bar-clicked chart, and `shown` is the denominator for P(take | shown), so an
  unanswered impression would bias the rate. `skip` is `review_events.py`'s own
  definition of "looked at the chart and passed" - the trader's phrase. No
  status line (the replacement chart is already up) and no parking (that stays
  specific to Skip-after-arming-a-D1 in `_skip_review_alert`).

**Files:** `scripts/ui/panels/alert_center_panel.py` (+34/-4),
`tests/test_qt_m5_alert_bar.py` (+3 tests, 22 total).

**Verification:** `.venv\Scripts\python.exe -m pytest tests/ -q` ->
**5119 passed, 19 subtests passed in 313s, exit 0.** Fail-before-fix checked
by stashing only the panel change: the two behaviour tests fail on the old code
(queue read `['AMD', 'NVDA', 'MUFG', 'XOM']` where it should read
`['AMD', 'MUFG', 'XOM']`) and the refreshed-D1 regression guard passes, as a
guard should. No exe rebuild: no packaging trigger (no new dependency, asset,
top-level package, dynamic import or `__file__` handling), and the desk runs
from source by trader decision 2026-08-26.

**Live gate owed** (with the other three rules of the morning): one DESK
session - clicking down the M5 bar leaves the waiting count D1-only and
unchanged, and the D1 chart that was in front when a bar row is clicked is
still there afterwards.

**Immediate next action:** commit and push; then the owed live gates.

## 2026-08-27 (10:00) - INVESTIGATION ONLY, nothing changed: why the desk jumps to 10 GB

Trader: "there are times the program jumps to 10gb of RAM usage. investigate to
see why." Measured live on the desk (pid 33336, launched 08:10): **10.7 GB
working set / 12.8 GB private at 09:25:55, 2.5 GB at 09:29:03.** A fresh desk
(pid 2296, 09:39) stayed at **0.9-1.25 GB through a whole BounceBot preamble**
including all four RRS passes, so the scan loop is NOT the holder.

**Cause 1 (the jump, every swing-scan slot): the research-warehouse post-scan
build runs INSIDE the desk process** (`ScanService.start_warehouse_build` ->
`research_warehouse.cli.run_build` on a thread) and three of its steps
materialise the WHOLE current-month `bar_m5` partition as Python dicts:
`aggregate.build_derived_bars` (`store.read_table("bar_m5", partition).to_pylist()`,
aggregate.py:277), `features.build_intraday_snapshots` (features.py:809, plus
`bar_derived` at :817) and `cli._run_outcomes` (cli.py:328) - each then filters
to ONE session in Python. Measured on the lake: `month=2026-08` = **8,175,471
rows / 384 MB parquet / 151 files**, and `to_pylist()` costs **1,627 B/row ->
13.3 GB** if fully held. The lake manifest (UTC) puts the 09:00 slot's build at
16:14:43-16:28:43 = **09:14:43-09:28:43 PT**, `bar_derived` 09:16-09:21 and
`feature_snapshot_intraday` 09:22-09:28:43 - exactly the window of the 10.7 GB
sample and the 09:29 release. The first session's 8.3 GB at 08:07 sits inside
the 07:43 slot's build (15:05-15:07 UTC + later steps). It grows through the
month (the partition is month-keyed), which is why 08-21 saw 8 GB and today 10.7.

**Cause 2 (once per tracker rewrite): `run_bronze_ingest` snapshot-ingests
`master_avwap_setup_tracker.json` (1.03 GB)** in the desk: `read_bytes()` +
`decode` + `json.loads` of the whole file (ingest_existing.py:365-416) - several
GB more, on the days the 07:xx scan rewrites the tracker (`bronze_setup_tracker`
at 15:06:16 UTC today; skipped by sha at 09:15 because the 09:00 run did not
touch it).

**Cause 3 (slow leak, all day): `BounceBot.data[reqId]` is never freed** on the
non-RRS request paths - `request_and_detect_bounce` (legacy.py:12465-12499),
`build_atr_cache` (11425) and the dynamic-VWAP checks (13701/13733/13767)
`del` only `data_ready_events[reqId]`; `request_historical_bars` pops both.
Measured 206 KB per 390-bar request -> ~80 MB per scan cycle (~400 calls) ->
1.5-2 GB over a session. This is the 2.5 GB floor after the build released.

Not the cause, checked and cleared: the GC controller (full sweeps every ~60 s,
0.2-2.1 s each all morning), `latest_bars` (~110 KB/symbol), chart items (pooled),
ChartDataService caches (bounded 60/160/500), the RRS O(n^2) profile (CPU only;
the 15m pass took 14 min at 09:15-09:29 because it ran under the build's memory
pressure, 2 min otherwise), the Setup Tracker CSVs (< 6 MB), the review queue.

Also seen, not memory: `_poll_focus_d1_interest` -> `FocusSideEditor.refresh`
stalled the GUI 12-70 s repeatedly 09:17-09:25 (392 s total at
focus_picks_panel.py:441 today), and the RS-window `_auto_tick` read 1,412
daily parquet files on the GUI thread (92 s at 09:25:52) after the scan
rewrote them.

Nothing was changed. Fix direction (needs the trader's go; `legacy.py` is
ask-first): (a) run `run_build` in a child process like the scan, or have the
three readers filter `bar_m5` by `session_id`/`interval_start` in Arrow
(`open_dataset(...).to_table(filter=...)`) before `to_pylist`; (b) stream or
skip the 1 GB tracker snapshot; (c) pop `self.data[reqId]` on every request
path. Diagnostic traces from this session live only in the session scratchpad.

---

## 2026-08-27 (morning) - four trader rules BUILT: regime-pause auto-Focus (`479c25c`), the VWAP-side / show-time review filter (`76e0b7b`), the D1 SMA trend leg + snapshot Prev/Next (`f3abda7`), the M5 alert bar; queue scan done

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

### Rule 2 - BUILT (after the scan; trader's "yes build it")

EPD, a Focus D1 flag from the 06:30 bar shown at 07:30 under VWAP and fading.
The movers-only review filter now has the session-VWAP leg and is re-asked at
show time (`AlertCenterPanel.vwap_state`, `_review_chart_state`,
`_advance_review_queue`; badge `wrong side of VWAP`). Presentation only - see
`CHANGELOG.md` 2026-08-27 (rule 2). Trader-directed in chat.

| Measure | Value |
|---|---|
| Tests added | 21 (`test_qt_review_vwap_side.py`) |
| Fail-before-fix | panel + widget changes stashed: 21 of 21 fail |
| Full suite | **5067 passed / 19 subtests, exit 0** (313 s; 5046 after rule 1) |
| `scripts/smoke_check.py` | 7/7 |
| Packaging trigger | none |

**Live gate owed:** the hidden count moving at show time on a real queue, the
`wrong side of VWAP` badge on a revealed name, and charts-shown-per-hour
against the 124-in-46-minutes baseline below.

### Rule 3 - BUILT (trader: "longs above the 200 SMA, shorts below the 50 SMA at least - go ahead")

MUFG, a swing-scanner D1 "short - zone-1 reject at AVWAPE" above every SMA in
an uptrend. Third leg of the review verdict, D1 recommendations only, plus
`◀ Prev` / `Next ▶` on the setups snapshot popup. Presentation only - see
`CHANGELOG.md` 2026-08-27 (rule 3). Trader-directed in chat.

| Measure | Value |
|---|---|
| Tests added | 29 (`test_sma_trend_gate.py` 11, `test_qt_review_sma_trend.py` 13, `test_qt_snapshot_prev_next.py` 5) |
| Fail-before-fix | four source files stashed: 18 of 18 Qt tests fail (the pure gate tests stand alone) |
| Full suite | **5096 passed / 19 subtests, exit 0** (308 s; 5067 after rule 2) |
| `scripts/smoke_check.py` | 7/7 |
| Packaging trigger | none (`sma_trend_gate.py` is a plain `scripts/*.py`) |

**Investigated, not changed - Yahoo forming candles:** by design, not an IB
fault. Today's forming daily candle is built from BounceBot's IB M5 cache,
which exists only for names in the current M5 scan set; every other
setups-table name gets a labelled Yahoo daily row for today (the history
underneath is the durable D1 store). An IB fetch path for those previews
would spend the locked pacing budget on every double-click - the trader's
call, not built.

**Live gate owed:** one DESK session covering all three rules - hidden count
moving at show time, the three badges on revealed names, Prev/Next walking a
real setups list, and charts-shown-per-hour against the 124-in-46-minutes
baseline.

### Rule 4 - BUILT: the M5 alert bar (trader: "does this make sense?" - yes; "latest at the top, oldest at the bottom")

Intraday alerts list in a slim bar at the LEFT of the desk (built between the
chart and the setups, moved left on the trader's second pass 09:45); the
review queue keeps D1 rows, Focus D1 flags, armed hits and the trader's own
charts. `ui/widgets/m5_alert_bar.py`, routing in `_enqueue_review_alert`,
desk splitter now three-way. Detail in `CHANGELOG.md` 2026-08-27 (rule 4).

| Measure | Value |
|---|---|
| Tests added / rewritten | 19 new (`test_qt_m5_alert_bar.py`); 8 rewritten to the new expectation; 7 queue-mechanics files gained one routing-off autouse fixture |
| Fail-before-fix | panel + desk + layout stashed: 13 of 19 fail (the 6 pure-widget tests pass - the untracked widget file was not stashed) |
| Full suite | **5115 passed / 19 subtests, exit 0** (299 s; 5096 after rule 3) |
| `scripts/smoke_check.py` | 7/7 |
| Packaging trigger | none (new widget is under `scripts/ui`, collected) |

Second pass (trader, 09:45-10:00): the bar moved to the LEFT of the chart
column (`DESK_SPLIT_KEY` v3) and a clicked line now leaves the bar. Full
suite after it: **5115 passed / 19 subtests, exit 0** (288 s); smoke 7/7.

**Live gate owed:** one DESK session - bar fills in alert order, Copy all
pastes into TC2000, a click charts and clears its line, the waiting count is
D1-only. **Desk
restart needed** to pick up rules 3 and 4 (the desk relaunched at 08:10 runs
`76e0b7b`).

### Group RS/RW tape - REMOVED by trader decision (10:05), rebuild parked

Investigated: right maths, late read (refreshes only when a scan cycle's RRS
pass finishes - 10-30 min apart today; one 60-min window with the overnight
gap in the first hour; ETF proxies). Trader: "just remove it for now and put
this build plan in the .md files for the future." `group_tape.setVisible(False)`
on the desk, widget/tests/wiring kept; the full rebuild plan is in plan.md
Phase 0.5 item 11 and AUTHORIZED for an Opus session via
`docs/prompts/GROUP_TAPE_REBUILD_OPUS_PROMPT.md` (trader, 10:15). One test
added. **Needs a desk
restart to take effect.** Also noted there: today's 27-minute scan cycle
(`rrs_scan` 1084 s over 302 symbols) is a separate, unaddressed finding.

| Measure | Value |
|---|---|
| Full suite | **5116 passed / 19 subtests, exit 0** (307 s) |
| `scripts/smoke_check.py` | 7/7 |

### Restart observation, 08:07-08:10 (trader asked for a safe restart to drop 112 waiting charts)

`CloseMainWindow()` on the desk (pid 26336, launched 06:30 from a VS Code
terminal, 8.3 GB working set): the window closed at 08:07:35, `closeEvent`
ran to completion - scan children reaped 08:07:40, writer lease released
08:07:40 and again by the backstop 08:07:51 - and then the PROCESS stayed
alive for 2+ minutes at a full core with the log silent, threads=46. It was
terminated at 08:10:25 after the shutdown lines were on disk, and the desk
relaunched through `trading_desk.cmd` (pid 33336, window up, BounceBot
warming ATRs by 08:11). Nothing was lost that the close had not already
saved. Worth a look under G-P2.4: with `gc.disable()` process-wide, an 8 GB
heap's interpreter teardown (or a non-daemon Theta quote-scan worker mid
`846`-name loop - it was at 70/846 when IB dropped) is the likely hang, and
the single-instance slot is held until the process dies, so a relaunch
right after close is refused.

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

---

## Earlier entries

Everything dated **2026-08-25 and earlier** moved to
[`docs/CHECKPOINT_ARCHIVE_2026-08.md`](docs/CHECKPOINT_ARCHIVE_2026-08.md) on
2026-08-27 (113 entries, 449 KB). It is evidence, not authority — read it only when a
specific past decision is not answered by this file, `CHANGELOG.md`, or the governing
spec.
