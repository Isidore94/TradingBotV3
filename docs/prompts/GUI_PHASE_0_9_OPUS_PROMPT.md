# GUI redesign implementation, Phase 0.9 — build prompt (Opus)

Paste everything below the line into a fresh Claude Code session in the repo on
the main desk, model set to Opus. **Run it after the AVWAP challenger session has
finished and pushed** — both build in the same checkout, and the desk launches
from that checkout. Paste the handoff back to the Fable session for review.
Authorized: `plan.md` Phase 0.9 (trader, 2026-08-26, "i authorize all
changes"). NOT authorized and not in this prompt: Waves U1–U3, S1 and Snappy P2
of `docs/GUI_REDESIGN_PLAN_2026-08-25.md`.

---

You are building `plan.md` **Phase 0.9 — GUI follow-ons from the 2026-08-26 live
session** in the TradingBotV3 repo on my trading desk. Read `CLAUDE.md` first
and follow its mandatory documentation workflow: `CHANGELOG.md`, `plan.md`
§5–7 and Phase 0.8 + 0.9, `CURRENT_CHECKPOINT.md`, then the governing document
`docs/GUI_REDESIGN_PLAN_2026-08-25.md` — §2 (standing constraints), §3.2 and
§3.4 (the measured session and my live findings), §5.3, §8.3, §8.4, §11.1
(owed fluidity work in measured order), §12 (the table rule), §14 (soak
workflow), §15 decisions 9/10/11/14. Also read
`docs/GUI_FLUIDITY_MEASUREMENT_RUNBOOK.md`. Build only what Phase 0.9 lists.

## Hard rules

1. **Presentation and threading only.** No detector, scorer, alert, queue,
   scheduler, evidence-stream or storage behaviour changes. No read is added or
   removed — only moved off the Qt thread, bounded, or cached.
2. **`scripts/ui/panels/alert_center_panel.py` is fenced.** G-P2.2 touches it.
   Before that edit, show me the exact diff in chat and wait for my "yes". No
   other change in that file.
3. **Fail-before-fix.** Every fix ships with a test shown failing on the
   un-fixed code (stash the fix, run, unstash) — say so per test in the handoff.
4. **A refresh never blanks a populated page**; last-good stays visible with its
   as-of; a failed read is stated; a successful empty read clears.
5. **Every shutdown join is bounded** (`ui/read_worker.join_worker`);
   `tests/test_shutdown_waits_are_bounded.py` enforces it. Copy the single-flight
   `ReadWorker` shape from `weekend_prep_panel.WeekReviewPage` (G-P1.1) for any
   read you move; prefer iterating children over hand-maintained shutdown lists.
6. **Nothing expensive on the Qt thread, and a stylesheet is expensive.** Widget
   variants go in `theme.qss` keyed on object names / dynamic properties, never
   per-widget `setStyleSheet`. Lists diff, never rebuild.
7. **The GC packet is measurement only.** `_GuiGcController`'s scheduling
   (`full_every_ticks`, deadlines, idle thresholds) does not change in this
   prompt. Any proposal goes in the report, not the code.
8. **Never break the tree.** The desk runs `launch_gui.py` from this checkout.
   Commit small and green, push after each commit. Branch:
   `claude/gui-phase-0-9` from the current HEAD (do not rebase onto `main`).
9. Chat to me in very short, simple lines (CLAUDE.md "How to talk to the
   trader"). Depth goes in docs and commit messages.
10. **Stop at each SOAK line below** and tell me to run the desk. Do not start
    the next slice until I say the soak is done.

## Packets, in order

### G-P2.0 — the table width rule (§12), applied through one shell

The rule, verbatim from §12: "Tables stretch to the available width. A table
never hugs the left edge of an otherwise empty page: the widest TEXT column
takes the slack, numeric and badge columns keep their measured width, and the
last section is not the only one that stretches." And: "Long identifiers elide
in the MIDDLE, never at the end, so the distinguishing tail survives
(`human_f…tracking`, not `human_foc…`) … the full value is the tooltip and the
head/tail split is deterministic. An elision that leaves every row reading the
same is a rendering defect."

Facts: the shared shell is `scripts/ui/widgets/data_table.py` (`DataTable`,
68 lines; `fit_columns` at :34–40 caps 80–260 px and stretches only the LAST
section). AWAY Recap and Weekend Prep do **not** use it — they build raw
`QTableWidget`s (`away_recap_panel.py:135–155`, `weekend_prep_panel.py:292–341`).

Build: one helper (module-level in `data_table.py`, e.g. `apply_width_rule(view,
text_columns=...)`) that both `DataTable` and a raw `QTableWidget` call: text
columns `Stretch`, numeric/badge columns `ResizeToContents`-then-fixed, no
last-section-only stretch; plus a middle-elision item delegate used by the
identifier columns, full value as tooltip. Apply it first to Weekend Prep ▸
Focus pick review (the cohort column in all three tables — the page the rule
was learned on, §8.4) and to AWAY Recap's four tables (`Line` takes the slack),
then to every `DataTable` user. Tests: a Qt test that a 1680-wide view gives the
text column the slack and a 2304-wide view gives it more; the elision keeps the
tail and is deterministic; two different long keys never elide to the same
string; tooltips carry the full value.

### G-P2.1 — AWAY Recap as a return surface (§8.3, decision 9)

Facts: `scripts/ui/panels/away_recap_panel.py`; alerts rows come from
`recap["classified_alerts"]` and scanner-status rows arrive with `symbol == ""`
and `side == "WATCH"`; `_ask_for_chart` (:272–289) silently returns on a blank
symbol, which reads as "charting is broken"; the hide-and-count idiom to copy is
the movers-only filter in `alert_center_panel.py` / `alert_chart_review.py:773`.

Build, presentation only, in the recap panel: (a) scanner-status rows hidden
from the alerts table and counted in one line ("2 scanner status messages —
show", one click reveals for the session); (b) a visible `Chart` action per row
plus `Enter` on the selected row and a header line that says so — the existing
`symbolActivated` → snapshot popup is kept, only the invitation is new; (c) a
symbol-less row renders in a distinct style (theme property, not
`setStyleSheet`) with no chart action. The Alert Center's backing list is not
touched; `set_alerts` stays the one reader. Tests: the filter hides and counts
and never deletes; reveal works; `Enter` on a row with a symbol emits
`symbolActivated`; a blank-symbol row emits nothing and shows the distinct
style; the empty state is honest when the recap is absent.

### G-P2.2 — Desk Journal route (§5.3, decision 10) — FENCED, ask first

Facts: the Journal tab is the sixth lower tab of the Trading Desk
(`alert_center_panel.py:796–797`, index 5); its composer is
`self._journal_text` (:3639); panel-scope shortcuts already exist in
`_bind_capture_shortcuts` (:1230–1250, `WidgetWithChildrenShortcut`) and
`_focus_capture_action` (:1252–1255) does "select tab, then focus" — copy that.
The rule in that docstring and in CLAUDE.md: a `QShortcut` bound inside a hidden
tab page never fires, and two live bindings for one sequence are an ambiguous
shortcut and Qt fires NEITHER — so check the sequence you pick is unbound
everywhere (grep every `QShortcut(`/`QKeySequence(` in `scripts/ui`).

Build: one panel-scope shortcut (propose `Ctrl+J`; verify it is free) that
selects the Journal tab and focuses the composer, plus a hint on the tab label
(`Journal  Ctrl+J`). **No second row under the charts. No verb-row verb** unless
I ask for a mouse route. Show me the diff before editing the file. Tests: the
shortcut is registered once at panel scope; firing it selects index
`_journal_tab_index` and gives the composer focus; the tab label carries the
hint; no other binding of the same sequence exists (source-level grep test).

**SOAK 1.** Stop. Tell me to work a normal session with `ui_stall_watchdog` on,
then compare with the runbook command against
`ui_stalls_prefix_baseline_2026-08-26.jsonl`. Record the numbers in
`CURRENT_CHECKPOINT.md` before continuing.

### G-P2.3 — the next fluidity slice, in measured order (§11.1, decision 14)

Take these in this order, one commit each, each with its fail-before-fix test:

1. **`DataTable.fit_columns` bounded measurement** (`data_table.py:35`,
   `resizeColumnsToContents()`, 7.9% / 115 s, worst 23.9 s). Size from a sample
   (first N rows + the widest-known per column) or remembered per-column
   widths; keep the cap; **never re-measure on a refresh whose column set is
   unchanged**. Callers: `master_avwap_panel:474`, `daytrade_tracker_panel:218–219`,
   `move_forensics_panel:165`, `rs_window_panel:437,503`,
   `setup_tracker_panel:453`, `theta_panel:122`. This is a Qt measurement cost,
   not a read — a worker does not fix it. Test: a 5,000-row model measures a
   bounded number of rows (count the delegate/sizeHint calls) and a second
   refresh with the same columns measures none.
2. **Theta refresh — explain the growth FIRST, then fix.**
   `theta_panel.refresh` (:115–126) runs `load_theta_rows()`, `set_rows` →
   `endResetModel` (`theta_table_model.py:72`), `_apply_filters`, `sortByColumn`,
   `fit_columns`, then re-adds the `QFileSystemWatcher` path, all on Qt, on
   every `fileChanged`. Three hourly refreshes cost 3.0 s → 26.6 s → 49.2 s.
   Before changing anything, reproduce the growth in a test or a scratch script
   (candidates: the sort proxy re-sorting per reset; the watcher firing more
   than once per write and the handler re-entering; something accumulating
   across refreshes) and write the cause into the commit message. Then: parse
   on a `ReadWorker`, **diff rows into the model** instead of resetting, apply
   item 1, debounce the watcher. Tests: the cause reproduced and pinned; three
   successive refreshes cost the same order of work; no read on Qt.
3. **`watchlist_utils.read_watchlist_symbols` off Qt** (`scripts/watchlist_utils.py:33`,
   204 stalls / 57 s). Find the Qt-thread caller with the interaction id
   (candidates: `watchlists_panel`, `universe_panel`, `autopilot_service`,
   `strength_board_service`, `weekend_prep_service` when called from a slot);
   move the read behind `ReadWorker` and cache on `(path, mtime, size)` like the
   settings cache. Test: the panel/service path issues no `read_text` on the
   Qt thread.
4. **`project_paths._load_local_settings` `stat()`** (`project_paths.py:165`, 56
   stalls at ~0.5 s each). "A stat that averages half a second is not the
   stat." **Measure before touching**: use the stall records' `interaction_id`
   and sampled stacks to say what held the thread. Report; change the cache
   only if the measurement says the cache is the cost.
5. **The eight `reload()` panels, one whole page at a time, in the measured
   order** — Theta (done by item 2) and Setups (`master_avwap_panel`) first,
   then `setup_tracker_panel`, `industry_panel`, `master_market_prep_panel`,
   `watchlists_panel`, `rs_window_panel`, `universe_panel`. Same `ReadWorker` +
   last-good + bounded-join shape. **In this prompt build only Setups and
   `setup_tracker_panel`**, then stop; the rest are the next slice after a soak.
   While inside each panel, sweep it for the G-P1.6 class (a thread that can
   outlive its panel: `ai_summary_panel`, `daytrade_tracker_panel`,
   `move_forensics_panel`, `universe_panel` are the known candidates — fix the
   one you are in, list the others).

**SOAK 2.** Stop and tell me to run a session; compare; record.

### G-P2.4 — GC measurement packet (decision 11) — MEASUREMENT ONLY

Facts: `_GuiGcController` (`scripts/ui/app.py:787–855`; `gc.disable()` at :888,
2 s tick, full sweep every 30 ticks, deadlines 5 / 90 ticks); no measurement
hooks exist — no `gc.callbacks`, no per-generation timing, no counters. The
sweeps were 17.1% (248 s) of blocked time on 2026-08-26 and the subsystem
behind the 2026-08-21 8 GB / 298 s incident.

Build, without changing when or whether a sweep runs: a `gc.callbacks` hook
that records per sweep `generation`, `duration_ms`, `collected`, `uncollectable`,
`gc.get_count()` before/after and the process RSS, appended to
`%LOCALAPPDATA%\TradingBotV3\diagnostics\gc_sweeps.jsonl` (fail-quiet, bounded);
an opt-in setting (`ui_gc_trace`, default OFF) that, on a full sweep, samples
`gc.get_objects()` by type and by referrer module (top 20) so we learn WHAT
produces the cyclic garbage — sampled, bounded in time, never every sweep; and
a reader (`scripts/ui/gc_trace.py --summary`) that prints sweep cost per
generation, growth per hour, and the top producers. Tests: the hook records a
sweep on an injected collector; the trace is off by default; the reader
summarises a fixture log. Deliverable: a short `docs/analysis/GC_MEASUREMENT_<date>.md`
after one live session — numbers only, plus the scheduling options it
suggests, **none of them built**.

## Not in this prompt

Waves U1–U3, S1, Snappy P2 (`GUI_REDESIGN_PLAN` §11.1 U-items, §13): not
authorized. The remaining six `reload()` panels: next slice. Any
`_GuiGcController` scheduling change: its own ask with the measurement in hand.
`first_paint` / `chart_ready` marks: still owed under G-P1.3, not here.

## Verification before each commit

`.venv\Scripts\python.exe -m pytest tests/ -q` fully green (check pytest's own
exit code; baseline 4902 passed plus whatever the AVWAP session added — read
`CURRENT_CHECKPOINT.md` for the current number) and `scripts\smoke_check.py`
7/7. No packaging trigger is expected (no new dependency, asset, top-level
package or `__file__` change); if you add a non-`.py` asset, run
`tests/test_packaging_spec_drift.py` and say so.

## Handoff (write this, then paste it back to me)

Reconcile the docs per CLAUDE.md: `CHANGELOG.md`, `CURRENT_CHECKPOINT.md`
(active item, branch, verification numbers, each soak's numbers against the
2026-08-26 baseline, what is owed), `plan.md` Phase 0.9 item statuses,
`docs/GUI_REDESIGN_PLAN_2026-08-25.md` §13 build status,
`docs/DESK_TESTING_PLAN.md` if an owed live proof changed, `docs/README.md` if
you add a Markdown file. Then, in the chat, at most fifteen short lines:
commits, test counts, each test proved failing first, the Theta growth cause in
one line, anything not built, and every place you had to ask.
