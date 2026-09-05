# Professional 4K GUI redesign plan

**Status: PROPOSAL FOR TRADER REVIEW — planning only; no runtime change is
authorized by this document.** `plan.md` remains the only roadmap. This proposal
must be accepted and promoted there before implementation.

**Revision 2026-08-26 (evening).** The trader promoted **§11.1 / Wave P1 only**
on 2026-08-26; it is `plan.md` **Phase 0.8** (items G-P1.0 … G-P1.7) and every
code item in it is BUILT on branch `claude/gui-p1-fluidity` (commits listed in
§13). This revision folds in what that build measured and what the trader saw
in the 2026-08-26 live session, so the next UI effort plans against what is
now true rather than the 2026-08-25 snapshot. **Waves U1–U3, S1 and the
experimental Snappy mode (P2) remain PROPOSAL** — nothing in this document
authorizes them, and nothing here is promoted into `plan.md` by being written
here. Sections changed: §1 (two rows), §2 (standing constraints; the arm-bar
"mismatch" corrected), §3.2 (real baseline replaces the 45-minute sample),
§3.4–§3.5 (new: live findings, environment), §5.3, §6.3, §6.7, §8.3, §8.4,
§8.8, §9.1, §11.1, §11.3, §12, §13, §14, §15.

**Target workstation:** one trader, one main 4K monitor, the application using
the full available height and approximately 90% of the monitor width. On the
current Windows display scaling this is a 2304 × 1392 logical-pixel window
(approximately 3456 × 2088 physical pixels), leaving a projected Trading Desk
content area of roughly 2060 × 1308 logical pixels after shell chrome.

**Design goals:** simple, concise, clear, chart-first, keyboard-fast, and honest
about stale or missing data. This is a professional decision-support source of
truth, not a decorative dashboard. Order execution remains permanently out of
scope.

---

## 1. Executive decision

The application already contains most of the information the trader needs. The
primary problem is allocation and hierarchy, not missing features.

| Question | Finding | Recommended decision |
|---|---|---|
| Can the trader see the live decision context quickly? | **Mostly.** The D1/M5 review pair is strong and readable at 4K. Freshness, source, queue state, and chart controls are generally visible. | Keep one chart implementation and make the Veto/Alert workspace the default Trading Desk surface. |
| Is the normal Veto workflow fast to read and use? | **Not yet.** The lower pane always gives 40% of its width to Focus Strength, including while Capture/Veto, Armed, and Journal are open. Capture then scrolls vertically and hides reasons/actions below the fold. | Give Capture, Armed, and quick Journal the full lower width. Show Focus Strength only where it adds value. Make Veto the remembered/default capture mode and fit its full action in one viewport. |
| Can Master AVWAP be used as needed without hurting charts? | **Yes, but the full-width tab is much better than the split drawer.** Setups and most tables remain usable in the split; Watchlist toolbars and RS Window become cramped. | Default to a Veto-first tabbed desk: `Veto / Alerts` first, `Master AVWAP` second. Keep the current simultaneous split as an optional preset. |
| Is the application intuitive outside the desk? | **Uneven.** Strong research tables coexist with giant blank states, raw text dumps, duplicated configuration, and fourteen equally weighted nav items. | Group navigation by daily workflow, review, research, and system. Use a clear empty state and selected-row detail instead of permanent blank regions. |
| Is Market Journal ready to be the trader's thesis record? | **The storage contract is right; the page is not.** It captures text honestly, but the composer and empty lists consume the screen, the promised charts are only a note, and trades are not shown beside thoughts. | Make it a session timeline with a sticky fast composer, one large selected-symbol chart, machine context, and a read-only execution/entry/exit lane. Keep it separate from the Trade Journal. |
| Is the UI currently fluid enough? | **No.** The full pre-fix session of 2026-08-26 measured 3350 stalls, 24 minutes blocked and a 49-second freeze (§3.2). Wave P1 (BUILT, §13) removes the one 12.5% item it was aimed at plus the Health page's churn; the cyclic GC (17.1%) and two Qt table paths (13.3%) are untouched. | Fix measured GUI-thread work first, in order of measured blocked time (§11.1). Then offer an experimental bounded-cache `Snappy` mode that spends the available RAM/CPU on background preparation, never on more work in the Qt thread. |
| Can the trader read the tables? | **No — the most repeated complaint of the 2026-08-26 session.** Columns are pinned narrow while two-thirds of a 4K window sits empty; the one column that distinguishes rows elides to `human_foc…` on every row (§3.4 A). | Make table width a RULE (§12), not a per-page fix: tables take the available width, the widest text column takes the slack, identifiers elide in the middle. |

### Recommended default operating model

1. Open Trading Desk directly into **Veto / Alerts**, with the last reviewed
   chart and the **Capture → Veto** form ready.
2. Keep the D1 and M5 charts large. Between the charts and the lower tab strip,
   retain only the slim queue verb row required by the chart-review contract.
3. Open **Master AVWAP** as the adjacent full-width tab when needed. Use the
   optional split view only when simultaneous monitoring is genuinely useful.
4. Keep a one-line BounceBot/group-tape/status strip visible; do not dedicate a
   nearly empty full-page BounceBot tab to it.
5. Make all lower-pane width contextual. Empty or irrelevant companions should
   never reserve 40% of the desk.

---

## 2. Scope and non-negotiable contracts

This proposal changes presentation and interaction design only. It does not
authorize a detector, scorer, alert, queue, scheduler, evidence, broker, or
storage-contract change.

The implementation must preserve:

- decision support only; no order-entry or execution controls;
- the chart pane owns the Alert Center and stays the largest vertical region;
- the queue verb row advances without an extra click;
- CaptureRail actions remain reachable without a mouse and within the existing
  five-second contract;
- VETO and LIKE move the review queue; NOTE does not;
- LIKE records a setup claim and required reason but does not add Focus or alert
  privileges;
- `Veto D1 — but M5 today` records the ordinary veto first, then asks the one
  Focus owner for the M5 placement;
- Market Journal remains "what the trader thought" and Trade Journal remains
  "what the trader traded"; their stores are not merged;
- journal entries are never backdated; corrections supersede rather than edit
  history;
- every expensive read, parse, calculation, and payload build remains off the
  Qt thread;
- stale, unavailable, partial, unmeasured, and last-good states are stated, not
  rendered as zeros or confidence;
- no duplicate writer, timer, worker, chart implementation, or alert source;
- user-entered watchlist and Focus names remain trader-owned;
- the current local-first storage boundary and DAS separation remain unchanged;
- phone push policy remains AWAY-only except for the two documented wake-alert
  exceptions.

### Standing constraints the Wave P1 build discovered (2026-08-26)

These were each paid for once on the branch. They are invariants for every
later wave and must not be rediscovered:

- **Every shutdown join is bounded.** Use `ui/read_worker.join_worker` (5 s
  default). A bare `worker.wait()` is a hang waiting for a slow day — the DAS
  reader proved it: the trader closed the window on 2026-08-26 and the PROCESS
  OUTLIVED IT. On timeout the worker is disowned and parked, never dropped,
  because dropping the last reference to a running `QThread` destroys its C++
  half mid-run. `tests/test_shutdown_waits_are_bounded.py` is a source-level
  guard and enforces this (`e0f78ae`).
- **A worker that outlives its panel is a crash, not a leak.** A bare
  `threading.Thread` that emits a Qt signal back into its own panel must be
  joined on shutdown; `HealthPanel` did not, and it intermittently segfaulted
  an unrelated Qt test (4 runs in 6) because the fault is an access violation
  that no `except RuntimeError` can catch. **The class was never swept.** The
  same shape exists today in `ai_summary_panel`, `daytrade_tracker_panel`,
  `move_forensics_panel`, `universe_panel`, two module-level threads in
  `app.py` and one in `alert_center_panel.py` (fenced) — each starts a bare
  thread; whether each also emits into its panel after shutdown is exactly
  what the sweep has to establish, panel by panel.
- **Hand-maintained shutdown child lists fall behind.** Three did in one week
  (`WeekendPrepPanel`, `ResearchPanel`, the `MainWindow` list). Prefer
  iterating children over naming them.
- **A refresh never blanks a populated page.** Last-good stays visible; a
  failed read is stated; a SUCCESSFUL empty read still clears (an empty store
  is a fact, a failed read is not).
- **`scripts/ui/panels/alert_center_panel.py` stays fenced.** The trader has
  pre-authorized exactly two changes in it (the quick-journal `symbols`
  attachment; the mover memo at `_measure_mover_state`). Anything else in the
  U waves needs its own ask.
- **`scripts/ui/interaction_trace.py` exists; `first_paint` and `chart_ready`
  are NOT wired.** The stage names are defined and nothing calls `mark()` for
  them — they need the receiving paint path instrumented, not the emit seam.
  Wired today: `page_select`, the Journal inner tab (`tab_select`),
  `model_apply`, `layout`, `chart_request`.

### The arm-bar "mismatch" was a stale operating-context line, not a defect

The 2026-08-25 draft read the root operating context ("the arm bar lives on
the **Armed** tab") and recommended restoring it. The source disagrees for a
reason: the same day's second pass (`4c05de5`, 2026-08-20, CHANGELOG "the
hotbuttons return") put the arm bar **back under the chart on the trader's own
instruction** — "I also need my m5 and D1 alert hotbuttons back on the bottom
of the visual chart… I also need the ability to input a ticker manually as
well". `AlertChartReview(..., dock_arm_bar=True, dock_capture_rail=False)`
is that decision; only the capture rail (84% of the measured height problem)
moved to a tab. The CLAUDE.md/AGENTS.md line describes the FIRST pass and is
stale — a documentation correction for the trader to authorize, not a UI
change. Decision 3 in §15 is therefore reversed: this proposal no longer
recommends moving the arm bar. Any change here is fenced (§2) and would be
reversing an explicit trader instruction, so it is asked, never assumed.

---

## 3. What was tested

### 3.1 Live and projected geometry

The full live application was exercised top-to-bottom, including every
top-level page and representative populated states for every nested tab. The
Windows/Qt scale contract treats 2560 × 1440 logical pixels as this 4K desk
after display scaling. The 90%-width projection therefore used a 2304 × 1392
outer window and a 2060 × 1308 Trading Desk content surface.

Measured in the projected content surface:

| State | Measurement | Consequence |
|---|---:|---|
| Setups hidden | Alert chart/review pane 2028 × 722 | Charts have enough width; vertical chrome is the limiting resource. |
| Setups hidden | Lower row split 1215 / 809 | Focus Strength permanently consumes 809 px even while Capture is active. |
| Setups hidden, Capture | Capture viewport 1195 × 352; content size hint 300 × 759 | Less than half of the vertical capture content fits without scrolling. |
| Setups shown | Desk columns 1184 / 856 | The chart remains usable, but secondary toolbars and board columns become tight. |
| Setups shown, Capture | Capture viewport 679 × 352; Focus Strength 465 px | The primary Veto action is squeezed by a companion that is not needed for the action. |

The current layout tests explicitly cover the 2560 × 1440 logical 4K scale,
chart-leading desk ratios, compact Master table columns, responsive arm controls,
and saved split behavior. The focused planning baseline is **116 tests passed**:

```text
tests/test_qt_desk_layout.py
tests/test_ui_scale.py
tests/test_market_journal.py
tests/test_away_day_recap.py
tests/test_ui_fluidity.py
tests/test_ui_stall_watchdog.py
tests/test_qt_page_specs.py
```

### 3.2 Measured pre-fix baseline (2026-08-26 session; replaces the 45-minute sample)

The 2026-08-25 draft rested on a 45-minute sample of the process started at
21:29 (264 stalls, 117.3 ms median, 205.1 ms p90, 8.45 s worst, 46.0 s
blocked). That sample is kept only as a footnote. The stall watchdog was
already enabled on the desk, and its full log through the morning of
2026-08-26 — **entirely pre-fix code**, `main` at `53b9733` — is archived at
`%LOCALAPPDATA%\TradingBotV3\diagnostics\ui_stalls_prefix_baseline_2026-08-26.jsonl`
(nothing deleted; the 2026-08-21 archive sits beside it). This is the baseline
every Wave P1 soak and every later wave compares against.

| Metric (2026-08-26 rows, `blocked_ms`) | Result |
|---|---:|
| Stalls over 50 ms | **3350** |
| Median / p90 / p99 | 169.8 ms / 617.9 ms / 3771.5 ms |
| Worst single stall | **49.25 s** (`theta_table_model.py:72`, 10:16) |
| Total blocked | **1457.5 s** — about 24 minutes of frozen desk |

**Read the window honestly.** The archive's 2026-08-26 rows run from
**00:00 to 10:25**, not 06:15 to 10:25: the desk was left running overnight
(the same night the process was observed near 1 GB after ~8.5 hours).
1212 of the 3350 stalls (303 s) fell before 06:00 on an unattended desk —
mostly the idle event loop and the GC sweeps — and 138 more (26 s) in
06:00–06:15. The attended 06:15–10:25 window holds **2000 stalls and
1129 s (~19 min) blocked in ~4h10m**, which is the number to quote for "how
the desk felt". No ranking below changes under either window.

**By BLOCKED TIME — the ranking that matters, and a different list from the
ranking by count** (line numbers are at `53b9733`; several have since moved):

| Share | Site (at `53b9733`) | What it is | Status after Wave P1 |
|---:|---|---|---|
| 42.6% (621 s) | `app.py:1029` = `return app.exec()` | The event loop with no Python frame beneath it — Qt layout/paint or C++ work the sampler cannot name. **Uninformative by construction.** | The bucket `interaction_trace` (G-P1.3, `6bd7eef`) exists to resolve: from the next session each record names the click behind it |
| 17.1% (248 s) | `app.py:833` / `:841` = `collector(2)` / `collector(0)` | The cyclic GC sweeps (`_GuiGcController`) | **Untouched, and not authorized** — G-P1.7 records it as a trader decision; live scheduling component, not presentation |
| 12.5% (183 s) | `focus_picks_panel.py:419` = the mover chip update | Mover state re-measured per chip per redraw | **FIXED** (`0f04240` consumer memo, `10a3008` source memo) |
| 7.9% (115 s) | `widgets/data_table.py:35` = `resizeColumnsToContents()` | `DataTable.fit_columns` measures every cell of every row; 44 stalls, worst 23.9 s, first at 03:14 on an unattended desk — so it is driven by automatic refreshes (Setups after a scan, Day Trade Tracker, Move Forensics, RS Window industry table), not only clicks | Untouched |
| 5.4% (79 s) | `models/theta_table_model.py:72` = `endResetModel()` | Three refreshes, **3.0 s → 26.6 s → 49.2 s, roughly hourly and growing each time**. `ThetaPanel.refresh` runs on the report file watcher, parses the report on Qt (`load_theta_rows`), resets the model, sorts, then `fit_columns()`. The growth is itself a finding | Untouched; owns the worst stall of the day |
| 3.9% (57 s) | `watchlist_utils.py:33` = `path.read_text()` | Watchlist file reads on the Qt thread; 204 stalls | Untouched |
| 2.1% (30 s) | `project_paths.py:165` = `LOCAL_SETTINGS_FILE.stat()` | The settings cache's freshness `stat()` — 56 stalls averaging ~0.5 s for a single `stat`, which says the call is cheap and the disk was not, or something upstream holds the thread | Untouched |
| 1.7% (26 s) | `health_panel.py:374` = `selectRow` after render | The Health page's post-render selection restore; 48 stalls, all overnight | Adjacent to the G-P1.4 fix, not measured after it |

Two of these deserve a sentence because the obvious prescription is wrong for
them. `data_table.py:35` and `theta_table_model.py:72` are **not reads**: the
Python frame is a Qt call and the time is Qt measuring or relaying out a
table. Moving the read that precedes them to a worker leaves the cost where it
is. The fix is to bound the measurement — size columns from a sample of rows
or from remembered widths instead of every cell, and diff rows into the model
instead of resetting it — and, for Theta, to explain why each hourly refresh
costs more than the last before touching anything else.

**By COUNT across the whole 2026-08-21..26 log** (which mixes code revisions,
so only the stable names are quoted): `app.exec()` 1736, the Focus chip update
1728, `health_panel.py:147` (the `_fill` cell loop) **973** — the fourth most
frequent culprit of the week, **FIXED** in `49744a7`; it does not appear in
the 2026-08-26 top list by time because it costs whenever the Health page is
open, in small pieces.

Also measured and now fixed, from the 2026-08-25 sample: the 8.45 s Weekend
Prep `WeekReviewPage.reload()` freeze (`d050ee1`) and the 36 repeating Focus
chip stalls (5.93 s; `0f04240`). Still standing from that sample: switching
the Trading Desk Workspace/Tabs mode reparents the large widget trees at about
1.0 s each direction.

The earlier fluidity work materially improved the desk, but the pre-fix
session misses the standing full-session targets (no stall over five seconds,
under about sixty seconds blocked) by a factor of roughly ten on worst-case
and roughly twenty on total. §11.3 states what Wave P1 can and cannot be
expected to do about that.

### 3.3 Testing limitation

This was a realistic layout and interaction review, not a live market-behavior
acceptance. It marks none of the existing Phase 0 live gates complete. A later
implementation needs a real active-session proof because queue pressure,
incoming rows, chart refreshes, and alert sounds cannot be validated by an
offscreen layout alone.

### 3.4 Live-session findings, 2026-08-26 evening (trader-observed)

The trader worked the desk on the Wave P1 build and reported the following in
their own words and screenshots. These are the core of what the large UI
effort must fix; each was checked against source before being recorded here.

**A. Columns are pinned narrow while most of the window is empty.** The single
most repeated complaint, and it recurs on every table page.

- AWAY Recap: the `Line` column of the ranked-swings table truncates to
  `1. FROG …`; every table hugs the left edge with roughly two-thirds of a 4K
  window unused. Source: the four `QTableWidget`s in `away_recap_panel.py` set
  header labels and nothing else — no stretch, no resize mode, default section
  widths.
- Weekend Prep ▸ Focus pick review: the cohort column — the FIRST column of
  the veto-cohort, LIKE-cohort and performance tables, holding values such as
  `human_focus_tracking` and `veto_v3_<reason>` — renders `human_foc…` on
  EVERY row, so rows are mutually indistinguishable. Trader: "the data seems
  useful i just cant read it." Source: `weekend_prep_panel.py` builds those
  tables the same way, and the identifier's distinguishing part is its TAIL.
- Decision: this is a RULE in §12, not a per-page fix. Tables stretch to the
  available width; the widest text column takes the slack; long identifiers
  elide in the MIDDLE so the tail survives.

**B. AWAY Recap is not usable as a return surface.** Trader: "its just hard to
work with... i also cant even check charts from here. kinda useless."

- Charting IS wired — `AwayRecapPanel.symbolActivated` → `MainWindow` → the
  Alert Center snapshot popup, on `itemActivated` (double-click/Enter) of the
  swings and alerts tables — but nothing on the page says so, and a row with
  no symbol looks identical to a row with one. `_ask_for_chart` silently
  returns on a blank symbol, which reads as "charting is broken".
- The day's two alerts were status messages (`Scanning …`, `Learning …`) with a
  blank symbol and side `WATCH` — the shape `BounceService._emit_assist_note`
  and the scanner-status rows produce — so nothing on the page was chartable.
  The page renders the Alert Center's backing list verbatim (`_feed_away_recap`
  hands it over; ground rule 8, one reader), and that list contains rows that
  are about the SCANNER, not about a symbol.
- §8.3 now decides what belongs on the page, how a row is opened, and how a
  symbol-less row is presented.

**C. The Desk Journal is undiscoverable.** The trader could not find it. It is
the sixth and last tab of the Trading Desk lower strip
(`Alerts | D1 Focus | RS/RW Board | Armed | Capture | Journal`, tab index 5)
and is reachable only by clicking that tab. §6.3 assumed it was reachable; the
live evidence says otherwise. Any fix collides with the 2026-08-20 rule that
the review pane carries AT MOST ONE slim row between the charts and the tab
strip — §5.3 and §15 state the cost of each option explicitly rather than
spending that row quietly.

**Confirmed working, recorded so nobody re-investigates them:** the System
Health **Jobs** tab; the Research Warehouse readout's "run the build job first"
message — the lake genuinely has no slice outcomes, so the page is stating a
fact, not failing.

**Seen in passing, out of scope here:** System Health reports
`daily_bars/yahoo: 4/5 attempts failed (80%)` (`operations_audit.py`'s
provider-failure line). That is a data-source finding for the operations
runbooks, not a GUI one; it is recorded so the line is not mistaken for a
rendering defect.

### 3.5 Environment change to record, not to act on

**Smart App Control now reads OFF** on the desk:
`HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy` →
`VerifiedAndReputablePolicyState = 0`, with `SAC_PreviousState = 1` and
`SAC_EnforcementReason = 6` (read 2026-08-26). CLAUDE.md's "Frozen exe rebuild
policy" and the header of `trading_desk.cmd` both still assert SAC is ENFORCED
and cite that as why the desk runs from source. Both launchers still start
source, so nothing is broken — but the stated justification is stale, and
**whether the frozen exe becomes production again is a trader decision**
(§15, decision 12). If it does, the delivery gap CLAUDE.md describes returns
with it: a fix is not on the desk until the exe is rebuilt and its frozen
`--selftest` passes. This document flags it and decides nothing.

---

## 4. Information architecture and navigation

Fourteen equal-weight navigation buttons mix live trading, review, research,
maintenance, and configuration. They all fit vertically at 4K, but fit is not
the same as hierarchy. The trader should be able to locate a page by purpose,
not remember a flat list.

### Proposed navigation

| Group | Always visible | Collapsible/secondary |
|---|---|---|
| **LIVE** | Trading Desk, Focus Picks, Strength Board, Auto Pilot | — |
| **REVIEW** | Market Journal, Trade Journal, AWAY Recap | Chart Review, Weekend Prep |
| **RESEARCH** | Research | A.I. Summary |
| **SYSTEM** | System Health, Settings | Universe / Data Tools |

Rules:

- Rename the left-nav `Journal` label to **Trade Journal**. No store or contract
  changes; this only removes the daily ambiguity with Market Journal.
- Keep Trading Desk first and focused on launch.
- Let the trader pin up to four favorites at the top, but default them to the
  professional workflow rather than requiring setup.
- Collapse group labels, not individual daily pages. A collapsed group retains a
  count/status badge if a child is unhealthy or has pending work.
- Move the Workspace/Tabs presentation choice out of the global top bar and into
  Settings. The daily top bar should contain current page, global search/command,
  connection/health state, and nothing else.
- Add one command palette / page switcher shortcut for pages and actions. It is a
  navigation index, not a new automation surface.
- Preserve direct keyboard shortcuts for the current high-frequency actions.

### Global status strip

Keep a slim bottom status strip, but enforce stable slots:

1. Auto mode;
2. IB/data connection;
3. scan state with last completion and age;
4. current regime/technical state;
5. watchlist/universe freshness;
6. health severity.

Long diagnostic prose should truncate with a tooltip/details action. A status
strip that moves every other field when one message becomes verbose cannot be
scanned by position.

---

## 5. Trading Desk redesign

### 5.1 Recommended default layout

```text
┌──────────────── Trading Desk ──────────────────────────────────────────────┐
│ Group tape / scanner status / setup count                     health badge │
│ [ Veto / Alerts ] [ Master AVWAP ]                    [optional Split View]│
├────────────────────────────────────────────────────────────────────────────┤
│ Symbol · side · setup · freshness · reviewed/moving/armed state            │
│                                                                            │
│                         LARGE D1 CHART                                     │
│                                                                            │
│                         LARGE M5 CHART                                     │
│                                                                            │
│ [Add to Focus] [Skip] [Not today]       queue · hidden · armed summary     │
├────────────────────────────────────────────────────────────────────────────┤
│ [Alerts] [D1 Focus] [RS/RW] [Armed] [Capture] [Quick Journal]              │
│                                                                            │
│ Active tab uses the whole row unless its own companion is useful.          │
└────────────────────────────────────────────────────────────────────────────┘
```

At the 2304 × 1392 target:

- reserve approximately **65–70% of usable vertical space** for chart context,
  including the symbol/header and the two charts;
- give D1 slightly more height than M5, approximately 54/46 within the chart
  pair, with a draggable and remembered split;
- keep the lower tab row approximately 30–35%, with a minimum that allows the
  active action to fit but never steals the majority from charts;
- collapse the left nav when maximum chart width is desired; this is a visible
  user action with a remembered state, not an automatic surprise;
- retain last-good chart content while refreshing and overlay freshness/source
  state rather than replacing charts with blank loading panes.

### 5.2 Veto/Capture — the primary workflow

Current Capture lays out Veto, Like, and Note sections together. It works in a
wide dialog, but in the Alert Center's 60% lower column it wraps and becomes a
759-pixel-high scroll surface inside a 352-pixel viewport.

Target behavior:

- Capture gets **100% of the lower-row width**. Focus Strength collapses while
  Capture is active.
- Inside Capture, use a compact mode selector: **Veto | Like | Note**. The
  shortcut both selects the mode and focuses its first actionable field:
  `Alt+V`, `Alt+K`, `Alt+N`.
- Remember the last mode per machine; default to **Veto** because that is the
  normal desk posture.
- Veto shows every versioned reason at once in two readable columns at this
  width. Keys and labels stay unchanged; no reason is hidden in a dropdown.
- Keep the optional note on the same line or directly below the selected reason.
  A reason that requires a note must visibly change the field to required.
- Keep `Veto — not for today` and `Veto D1 — but M5 today` together at the
  bottom, always visible.
- Like shows the complete offered claim list, keyed in the current stable order,
  plus the required `why` field and commit button. It never changes Focus.
- Note shows one clear composer and save state. NOTE never advances the queue.
- The status line has a fixed reserved height and uses explicit states:
  `READY`, `SAVING`, `SAVED`, `NOT SAVED`.
- Preserve the panel-scoped shortcut rebinding; a hidden tab must never make the
  shortcuts inert.
- Display the selected D1 reference level, side, and current symbol in one
  compact header. Do not repeat explanatory paragraphs in professional mode.

Acceptance: with either Master hidden or shown at the 90%-width target, all Veto
reasons, the note field, both Veto actions, and the save state are visible with
no vertical or horizontal scroll.

### 5.3 Alert Center lower tabs

| Tab | Keep | Change |
|---|---|---|
| **Alerts** | One live row per symbol/side/day, escalation badges, source/tier, quick Focus action. | Use the right-side Focus Strength companion here if it is populated. Let the trader collapse it. Preserve selected row and scroll during updates. |
| **D1 Focus** | Separate untiered stream and unread count. | Full-width list when narrow; optional Strength companion only when it contains relevant rows. Put the short rubric behind help in professional mode. |
| **RS/RW Board** | Same BounceBot snapshot, board selection opens the shared chart. | This is the natural home for the Strength companion; combine their related context rather than duplicating board space elsewhere. |
| **Armed** | Inventory of chart watches, D1 event watches, D1 levels, and price alerts. | The arm bar stays under the chart (trader instruction 2026-08-20, §2) — this tab holds only the inventory. Give the tab full width. Group by current symbol first, then all symbols; expose expiry/source. |
| **Capture** | Append-only Veto/Like/Note recorder and keyboard map. | Full width and compact action modes as §5.2. |
| **Quick Journal** | M5-default free-text capture and Ctrl+Enter. The current chart symbol IS now attached (`db99271`, G-P1.0). | Full width; show the attached symbol as a visible removable chip; show the last three session entries below the composer and link to Market Journal. **Discoverability (§3.4 C):** the trader could not find this tab. Options, each with its cost stated: (a) a keyboard route — one shortcut that selects the tab and focuses the composer, plus a one-line hint on the tab label (`Journal  Ctrl+J`), which costs no row; (b) a `Journal` verb on the existing verb row, which fits the one-row rule only if it displaces nothing and the row does not wrap at 1680 × 954; (c) a second row under the charts — **rejected here**, it breaks the 2026-08-20 rule. Recommended: (a), with (b) only if the trader wants a mouse route. |

Use a per-tab splitter policy rather than one global 60/40 ratio. The policy is
presentation-only and remembered. It must never alter source, queue order, or
alert behavior.

### 5.4 Charts

Keep the shared `SymbolSnapshotWidget`/`CandleChart` path and improve only its
information hierarchy:

- fixed top line: symbol, side, source, as-of age, current price, setup/trigger;
- badges in a predictable order: `REVIEWED`, `MOVING`, `ARMED`, data caveat;
- D1 and M5 titles always include timeframe and last completed bar;
- stale or fallback data uses a banner inside a reserved line, never a modal;
- paint-line controls stay one compact menu; selected level is echoed beside the
  arming/capture context;
- charts retain their scale and crosshair state when the lower tab changes;
- chart loading never clears the last-good chart unless the symbol changes;
- no raw warning or traceback is painted into the chart title.

### 5.5 Master AVWAP tabs

The full-width Master tab is the preferred deep view. Split view remains useful
for brief simultaneous reference but must use responsive toolbars.

| Tab | Review | Redesign decision |
|---|---|---|
| **Setups** | Strong dense ranked table; compact profile protects key industry/expected-R columns. | Preserve. Full-width by default when opened. Add a small selected-row detail band or drawer; do not grow permanent chrome. Keep F9/full-table behavior. |
| **Theta Plays** | Filters and selected-row detail use the space well. | Preserve. Align filter heights/labels with Setups and remember the last filter. |
| **Watchlists → Shared Longs / Shorts** | Lists are readable; toolbar labels collapse to fragments in split view. | Replace fixed horizontal button row with a wrapping row or `Actions` overflow. Never elide an action to two letters. |
| **Watchlists → Master Swing Lists** | Same responsive-toolbar problem. | Same solution; keep ownership/provenance visible before destructive clears. |
| **Watchlists → Auto Lists (bot-owned)** | Read-only ownership is clear. | Preserve; add as-of/source in the header and make the read-only state visually unmistakable. |
| **Industry Board** | Two useful ranked tables fit well at full width. | Preserve. Selection opens the shared chart; keep independent sort state and concise as-of line. |
| **RS Window** | SPY selection chart plus industry/results is conceptually strong but cramped in the split drawer. | Treat as a full-width analytical tab. Use a vertical splitter with a minimum result-table height; collapse prose and keep selection controls on one wrapping row. |

### 5.6 BounceBot presentation

The current top-level BounceBot tab contains very little beyond status and an
entry-assist line, leaving most of the screen blank. The better presentation is:

- persistent one-line status strip in Trading Desk: connection, scan state,
  current mode, active/disabled types, last cycle, and entry-assist state;
- click the strip to open a compact operational drawer with recent activity and
  reconnect;
- move RRS tuning, bounce-type switches, and connection configuration to
  Settings → Automation & Scanners;
- move detailed timing/history to System Health;
- remove the empty top-level Trading Desk tab unless it gains a real monitoring
  purpose.

### 5.7 Standalone Chart Review

Chart Review duplicates much of Alert Center but provides a useful dedicated
keyboard review theater. Keep it only with a distinct job:

- frozen/manual review order, independent of live feed re-ranking;
- D1/M5 charts at maximum height;
- 360–420 logical-pixel right rail so claim/reason labels are never clipped;
- setup drawer becomes a searchable ranked table, not a long alphabetic text
  list that compresses charts;
- same capture semantics and shared chart data service;
- clear `Return to live queue` action.

If the trader does not use the frozen/manual-review distinction, fold this page
into Trading Desk and remove the duplicate nav item. That is a product decision,
not an implementation assumption.

---

## 6. Market Journal redesign

### 6.1 Purpose

Market Journal is the point-in-time record of the trader's market thesis:

- D1 thesis and bias;
- intraday tape observations;
- key levels and invalidation;
- what changed and when;
- the machine context that was available at the time;
- entries and exits that actually occurred beside those thoughts;
- later AI synthesis that helps the trader remain internally consistent.

Trade Journal remains the tax-grade execution record. Market Journal may read
its executions and link to a trade, but it never duplicates, edits, or becomes
the owner of broker data.

### 6.2 Target layout

```text
┌ Session · bias · key levels · invalidation · last saved · AI status ───────┐
├───────────────────────────────┬────────────────────────────────────────────┤
│ STICKY QUICK COMPOSER         │ [symbol chips: up to six for this session]│
│ [D1] [M5] [General] [tags]    │                                            │
│ free text                    │            LARGE SELECTED CHART            │
│ Ctrl+Enter · saved state      │              D1 / M5 toggle                │
├───────────────────────────────┤                                            │
│ SESSION TIMELINE              ├────────────────────────────────────────────┤
│ trader thoughts               │ MACHINE CONTEXT / REGIME / CALENDAR       │
│ thesis changes                ├────────────────────────────────────────────┤
│ read-only entries and exits   │ SESSION EXECUTIONS / ENTRIES / EXITS      │
│ AI synthesis revisions        │ AI SYNTHESIS (collapsed until generated)  │
└───────────────────────────────┴────────────────────────────────────────────┘
```

At the target width, use approximately 58% for composer/timeline and 42% for
the selected chart/context rail. The selected chart is large; up to six symbols
remain one click/number away rather than rendering six undersized live charts at
once. If six simultaneous D1 charts are still a hard trader requirement, offer a
separate `Grid` view using 2 × 3 cards and retain `Focus` view as the default.

### 6.3 Fast capture

- The composer is always visible and never taller than needed.
- Timeframe uses three direct chips: `D1`, `M5`, `General`; M5 stays the
  in-session default, D1 the sit-down-page default.
- Current chart symbol is attached automatically on the Desk quick surface
  (**landed** `db99271`: the chart in front of the trader, when there is one;
  a stale symbol would assert a link they never made) and shown as a removable
  chip (**not yet** — the write carries it, the surface does not show it).
  Free-text symbol recognition may suggest chips but never silently attach
  one.
- The Desk quick surface has to be findable before any of this matters: see
  §3.4 C and §5.3.
- Optional lightweight tags: `bias`, `thesis change`, `level`, `invalidation`,
  `risk`, `review`. Free text remains primary; tags cannot become a required
  form before the journal has earned that complexity.
- `Ctrl+Enter` saves. The input remains visible until the append confirms; a
  failed write says `NOT SAVED` and retains the text.
- Saving produces one timestamped timeline card. No dialogs.
- Addendum/correction uses the existing supersedes contract; the original stays
  on disk and the UI shows the relationship.

### 6.4 Daily thesis header

Provide four optional concise fields above the timeline:

| Field | Purpose |
|---|---|
| Bias | Bullish / bearish / neutral / mixed plus one-line rationale |
| Key levels | The few levels that control the day, not another level database |
| Invalidation | What would make the current thesis wrong |
| What changed | Latest thesis transition, derived from a marked entry or written directly |

These fields should save as ordinary versioned journal entries/metadata through
the one service, not a second mutable store invented by the panel.

### 6.5 Timeline and trade context

The central timeline interleaves visually but not structurally:

- trader journal entries;
- machine regime/context changes;
- calendar events;
- read-only Trade Journal executions, entries, scale-ins/outs, stops, and closes;
- AI synthesis versions.

Every card carries a source icon/label. A trade row links to Trade Journal. A
market-thought row can link to one or more symbols and a selected painted level.
No journal entry is inferred from a trade and no trade annotation is inferred
from prose.

### 6.6 AI synthesis

AI is an optional reader, never the author of record.

- `Generate synthesis` starts from the deterministic session packet: current
  entries, prior thesis, machine context, executions, and source counts.
- Show provider/model, generated time, evidence count, and missing sources.
- Separate `Summary`, `Consistency`, `Thesis changes`, and `Questions for
  tomorrow`; keep the default view concise.
- A generated summary never overwrites trader text. Refresh creates a new
  version.
- Free-text journal evidence remains opt-in under the existing AI policy.
- Local AI can be added later without changing the page contract; provider and
  model configuration live in Settings.

### 6.7 Current implementation gaps to close

- The promised D1 charts are currently a text note, not chart widgets.
- The page uses a large vertical composer over stacked empty lists, which makes
  the session record feel absent even when the feature is available.
- ~~The quick Desk surface writes no current-symbol association despite the
  service already supporting `symbols`.~~ **Closed 2026-08-26** (`db99271`);
  the surface still does not SHOW the attachment.
- The quick Desk surface is the last tab of the lower strip and the trader
  could not find it (§3.4 C).
- No read-only Trade Journal execution lane exists.
- Session reload uses a worker correctly; preserve that.

---

## 7. Trade Journal review

Rename the page label to **Trade Journal** and preserve its tax-grade contract.

| Tab | Current finding | Redesign |
|---|---|---|
| **Trades** | The 50/50 trade table/detail split is useful, but the detail side shows many large empty groups and buries the execution story. | Add a compact selected-trade summary header, then collapsible `Plan & R`, `Executions`, `Tags`, `Notes/Review`, and `Corrections/Audit` sections. Render empty sections as one line. Show legs as a chronological execution timeline. |
| **Calendar** | Month grid occupies a small upper-left region; year heatmap and numeric grid create dead space and nested scrolling. | Responsive two-column month summary + year heatmap. One click filters Trades. No horizontal scrollbar at target width. Use blank for no-trade days and retain the existing zero-centered scale. |
| **Analytics** | Equity curve and grouping bars are useful but key metrics are not prominent; small tables float in large canvases. | Add a KPI strip, a 2 × 2 chart grid, and one breakdown/detail table. Thin samples stay marked. Selected chart/export remains exact. |
| **Health** | Important but overloaded: coverage, raw reconciliation, nightly runs, import history, repair actions, and credentials share one long page. | Separate `Status`, `Coverage`, `Reconciliation`, and `Runs` sections. Put the actionable failure and next step first. Move credentials/configuration to Settings → Data & Connections and deep-link there. |
| **Fees** | A few rows occupy a full tab with large unused space. | Fold into Analytics/Reports unless the trader uses it as a dedicated tax workflow. Keep exact export and per-account/currency separation. |

Shared filters (account/tax group, currency, date range) remain sticky above all
tabs. Filters should show their scope in one compact line and never silently
blend tax groups or currencies.

---

## 8. Review of every other top-level page

### 8.1 Focus Picks

Current four-quadrant Swing/M5 Long/Short layout fits 4K and is understandable.
The main issues are a three-line status header, expensive chip refreshes, and a
duplicated phone-alert table.

Plan:

- retain the four-quadrant ownership model;
- compress snapshot/review/freshness into one status row;
- keep chip badges but resolve their state once per refresh;
- make destructive `Clear all` visually secondary and confirmed;
- use the shared selected-symbol chart popup;
- show operational price alerts through the single Price Alert model instead of
  a second configuration surface;
- preserve the read-only reviewed-today line outside editable watchlist text.

### 8.2 Strength Board

The split M5 board / RS-RW board is useful when populated. Before the first
refresh it is an enormous blank canvas.

Plan:

- show last-good data immediately with as-of age;
- use a useful empty state with source, expected refresh, and one primary
  action;
- keep sortable columns and blank-last behavior;
- selecting a row shows a large shared chart/context pane or popup without a
  refetch;
- remember the M5/RS-RW vertical split;
- standalone page remains the deep board; the Desk companion should not reserve
  width when empty.

### 8.3 AWAY Recap

**What is true now (2026-08-26).** The page draws the day's alerts from the
Alert Center's backing list and can open the shared chart on a row
(`symbolActivated` → the Alert Center snapshot popup). Two of the three
defects the draft listed are closed by `db99271`: the Focus reader call
(`load_focus_map(side)` against a keyword-only signature) raised on every run
and was reported as "Focus lists unavailable" — it now reads the union map
once; and the adoption-gate line called `mover_state(side, None, None, None)`,
which can only return UNKNOWN, and rendered that as a verdict — it now SAYS
the gate was not measured here and points at the surfaces that measure it,
because measuring it would put bars on the Qt thread inside a click. The
third — a live source error included as raw exception text in the summary —
stands. So does everything the trader reported in §3.4 B: the page is not
usable as a return surface.

**Decisions this page needs (recommendations supplied):**

1. *What belongs on the page.* A recap of what the day PRODUCED: ranked
   swings, symbol alerts, staged picks, Focus names that need managing, and
   the write-up. **Scanner status rows do not** — `Scanning …`, `Learning …`,
   entry-assist notes and other blank-symbol `WATCH` rows are about the
   scanner, not about a symbol. Recommended: filter them OUT of the alerts
   table and show them as one count line ("2 scanner status messages — show"),
   the same hide-and-count idiom the movers-only filter uses, so nothing is
   deleted and the backing list stays the single reader's record. This is a
   presentation filter in the recap panel, not a change to the Alert Center's
   list.
2. *How a row is opened.* One visible affordance, not a hidden double-click:
   a `Chart` action per row (button or the row's context) plus `Enter` on the
   selected row, and a header line that says "select a row and press Enter or
   Chart". The existing signal and popup are kept; only the invitation is new.
3. *How a symbol-less row is presented.* It never sits in the symbol table
   looking like a symbol row. If a status row is shown at all (decision 1's
   "show"), it renders in a distinct style with no chart action, so a blank
   `Symbol` cell cannot read as a broken chart.
4. *Width.* Every table on this page is a §12 rule violation today; the
   `Line` column takes the slack.

Target layout (unchanged in intent):

- KPI/status row: session, completeness, ranked swings, alerts, staged, Focus;
- left 40%: ranked swings, alerts, staged picks with one consistent selection;
- right 60%: one large selected D1/M5 chart, trigger/setup context, and Focus
  action;
- bottom or collapsible panel: D1 write-up using Market Journal service;
- incomplete sources render as a short `Partial recap` banner with a details
  expander, never raw Python in the main narrative;
- the adoption-gate line stays honest: "not measured here" until a worker
  measures it from cached bars, never a click-time fetch;
- preserve original day ordering/ranking and the one shared chart owner;
- surface that the process-scoped alert list may be incomplete after a restart
  or midnight rollover.

### 8.4 Weekend Prep

Keep the guided five-step routine and persistent Done/Skipped state. Improve
orientation. **The expensive refreshes are now off Qt** (`d050ee1`, G-P1.1):
`WeekReviewPage` and `FocusReviewPage` read on an owned single-flight
`ReadWorker`, last-good survives a refresh, a failed read is stated, and the
panel's shutdown joins every page with a bounded wait (`e0f78ae`).
`WeekAheadPage` and `DiscoveryPage` were audited clean (they refresh through
service signals), and the walk-away page already had its worker.

| Step/tab | Redesign |
|---|---|
| **Week in review** | Summary cards for takes/skips/rejects/blind spots plus drill-down tables. Worker + last-good: **landed**. |
| **Focus pick review** | One joined pick/outcome table, then collapsible veto/like/performance/verdict views. Show maturity and as-of state clearly. **The cohort column is unreadable today** (§3.4 A: `human_foc…` on every row) — it is the first column of three tables and carries the row's identity in its tail. Apply the §12 width/elision rule here first; it is the page where the rule was learned. |
| **Walk-away** | Results and pending tag proposals side-by-side; weekly is default, all-pending remains an explicit toggle. |
| **Discovery → H1** | Same board contract; show offered/measured/filtered counts and adoption result. |
| **Discovery → D1** | Same layout and clear timeframe-specific gate explanation. |
| **Discovery → Monthly** | Make short-history exclusions prominent and state that the forming month is absent. |
| **Week ahead** | Render the report beside a concise checklist/risk-window summary; keep manual-only execution. |

The left step rail should show status, last refreshed time, and row count. Large
full-width buttons become compact primary actions in the step header. Empty
steps explain what input is absent and what will happen when Refresh is pressed.

### 8.5 Universe

This is a low-frequency data-maintenance tool, not a daily trading destination.

- Move it to Settings → Data & Universe / Advanced Tools or a collapsible System
  nav group.
- Replace the giant plain-text symbol pane with a virtual searchable table and
  counts by source/filter.
- Present pasted-list comparison as a real diff: only-here, only-pasted,
  intersection, invalid.
- Keep merge explicit, previewed, and confirmed; never auto-remove user names.
- Show rebuild source, start/end, last success, and output path.

### 8.6 Auto Pilot

The current page is operationally understandable but gives the ON/OFF button
too much space and mixes status with configuration/maintenance.

- Keep mode/state, next action, last successful scan, current blocker, and
  immediate `Reconnect` / `Run scan now` actions.
- Rename any remaining `Mini PC Mode` language to current Auto Pilot terms.
- Move weekday auto-arm/schedule and scanner configuration to Settings.
- Move rebuild watchlists/universe and durable warming to Advanced Maintenance.
- Keep report generation only if it is a real operator action; otherwise make it
  a diagnostics/export action.
- Collapse the activity log by default, with errors automatically expanded.

### 8.7 A.I. Summary

Move provider, model, and secret management to Settings → AI. The horizontal
scope checkbox run currently clips even at the target viewport.

- Replace the long checkbox row with templates: `Daily desk`, `Market journal`,
  `Trade review`, `Setup research`, and `Custom`.
- Custom scope uses a wrapping grid with counts/caveats.
- Keep **Validated Summary** and **Exact Evidence Preview** as the two result
  tabs; Evidence Preview should be the default before Generate.
- Show source counts, time range, thin-sample/discovery caveats, and output path
  in a fixed metadata band.
- Keep last-good output visible during generation.
- Market-journal synthesis is launched from Market Journal; this page remains
  the broader evidence/research reader.

### 8.8 System Health

The KPI strip and evidence/job/timing split are valuable; the **Jobs** tab was
confirmed working by the trader on 2026-08-26. Since `49744a7` (G-P1.4) the
three tables are written in place rather than rebuilt — the `_fill` cell loop
was the week's fourth most frequent stall culprit (973 records) — with scroll
position held across the update and selection surviving by check id; the
audit thread is now joined on shutdown (G-P1.6). Improve actionability:

- summary column becomes one concise sentence; full evidence stays in detail;
- add `Owner / Next action` and `Last changed` columns;
- severity and freshness sort by default;
- selecting an unhealthy row opens a structured remediation card, with a deep
  link to the relevant Settings section;
- **Evidence** keeps human-readable proof;
- **Jobs** shows last result, next eligible run, attempts, and failure reason;
- **Phase timings** remains the timing table and gains baseline/slowest-stage
  comparison;
- raw paths/logs remain behind `Advanced details`.

---

## 9. Research page — every tab

Research is information-dense and generally the strongest part of the
application. The goal is to preserve dense tables while replacing raw dumps and
unused space with selected-row explanations.

### 9.1 Top-level Research tabs

| Tab | Review and decision |
|---|---|
| **Master AVWAP Market Prep** | Keep the two-sided lists, but make them virtual searchable tables. Convert key report sections to structured cards/tables; keep raw report behind `Exact report`. |
| **Setup Tracker** | Preserve as the primary performance source. Keep metric strip and plain-English summary, but prevent the summary from creating an internal scroll trap. Move wide sample lists into selected-row detail. |
| **Setup Playbook** | Preserve the left setup index and exact mechanics article. Add search/filter and in-document anchors; do not simplify away the actual mechanics. |
| **Move Forensics** | Keep filters and result table. Format numeric values to trading-readable precision. Replace the duplicate raw text panel with a selected-row evidence/explanation pane; retain raw export under Advanced. |
| **Day Trade Tracker** | Preserve metric strip, dimensional tabs, and learning rules. Use selected-row detail for examples/explanations instead of making every row excessively wide. |
| **Ticker Lookup** | Keep one-symbol lookup, but show staged progress, elapsed time, cancel, source-by-source completion, and last-good results. The tested SPY lookup remained at `Looking up SPY…` for more than nine seconds with no progress. |
| **Price Alerts** | Keep the operational alert list but move ntfy server/topic/token and mode policy to Settings → Notifications. Reuse the same operational model in Focus/Alert Center rather than duplicating configuration. |
| **Research Warehouse** | Show enabled/disabled/reachable state, last-known dataset counts, last successful write, spool backlog, and refresh status. Since `49744a7` the readout reads the DAS lake on a single-flight worker (the only read in the desk that leaves the machine, against a share known to drop) and keeps last-good on failure; its join is bounded (`e0f78ae` — this was the reader that kept the process alive after the window closed). The "run the build job first" message is correct: the lake has no slice outcomes yet. What remains is the state/counts header. |

### 9.2 Setup Tracker inner tabs

| Inner tab | Decision |
|---|---|
| **Current Picks** | Preserve ranked picks and selected setup detail. Keep actionable columns pinned; move verbose sample context to detail. |
| **Short-Term 1–2d** | Preserve its distinct mark-to-market question and best-first order. Put the definition in a help tooltip after first use. |
| **Human Picks** | Preserve same-horizon comparison. Make cohort/source and maturity obvious. |
| **Setup Types** | Preserve core n/win/R metrics; selected row opens mechanics and examples. |
| **Last 30 Days** | Preserve NEW/RISING flags and pin behavior; add explicit as-of/window dates. |
| **Playbooks** | Preserve setup × stop/exit ranking; selected row shows exact policy evidence without widening all rows. |
| **Scan Factors** | Preserve factor evidence; group related columns and explain denominator/sample scope in detail. |
| **Tier Performance** | Preserve S/A/B comparison and maturity; keep discovery/thin labels adjacent to n. |
| **Catch Rate** | Preserve opportunity-to-pick accounting; selected detail shows missed winners and why rows were eligible. |

### 9.3 Day Trade Tracker inner tabs

Use the same table shell and selected-row explanation for all nine performance
dimensions:

- **Bounce Types**;
- **Combos**;
- **Time of Day**;
- **Environment**;
- **RRS**;
- **Swing Focus**;
- **Swing Bucket**;
- **Swing Family**;
- **Swing Traits**.

For **Live Alert Rules**, keep it visually distinct as the current policy
readout. It must state whether a row is live, shadow, muted, or discovery and
must not imply a table edit changes a rule unless a real controlled writer
exists.

### 9.4 Ticker Lookup inner tabs

| Inner tab | Decision |
|---|---|
| **Overview** | First partial result; show quote/provenance/freshness and key setup context. |
| **Earnings / Events** | Structured event table with actual vs projected dates clearly distinguished. |
| **News** | Show source/time and a clean unavailable state; no indefinite spinner. |
| **Report** | Exact raw/generated report for audit, secondary to structured tabs. |

---

## 10. Settings redesign and offloads

Settings should own preferences, credentials, schedules, and maintenance—not
live decisions. A workflow surface may show status and link to its setting, but
should not duplicate the setting.

### 10.0 Current Settings tabs

| Current tab | Finding | Disposition |
|---|---|---|
| **General** | Presentation, data path, and durable-store warming occupy only the top of a large page. | Split these controls into Appearance & Workspace, Data & Connections, Performance, and Maintenance. |
| **BounceBot** | Useful connection/tuning/type controls are separated by a large vertical blank region. | Move them into Automation & Scanners with compact sections; retain live status on Trading Desk. |
| **Testing Plan** | The exact operator runbook is valuable, but it is a long document rather than an everyday preference. | Keep it read-only under Maintenance & Advanced / Operations, make it searchable, and allow Health to deep-link to the owed step. |

### 10.1 Proposed Settings categories

| Category | Contents |
|---|---|
| **Appearance & Workspace** | Theme, scale, compact density, explain mode, nav collapse/favorites, default page, default Desk tab, Veto default/remember behavior, optional split view, reset saved layouts. |
| **Data & Connections** | Home folder, IB host/port/client id, Questrade refresh token, IBKR Flex token/query id, data-source status, broker/account labels, research/DAS paths as appropriate. Secrets remain in the existing secret-bearing local settings file and are masked. |
| **Notifications** | ntfy server/topic/token, routine push policy readout, urgent wake-channel test, price-alert monitoring policy, sound preferences. |
| **Automation & Scanners** | Auto-arm/schedule, Auto mode defaults, BounceBot RRS sensitivity/timeframe, bounce-type switches, open-burst digest window, scanner presentation preferences. |
| **AI** | Provider, model, secret, local model endpoint, default evidence templates, opt-in free-text policy, output location. |
| **Performance** | Standard/Snappy mode, cache budget, prewarm toggle, current cache use, clear caches, watchdog enabled, restart requirement, comparison summary. |
| **Maintenance & Advanced** | Universe build/compare/merge, durable-store warm, watchlist rebuild, raw paths/logs, Testing Plan, export diagnostics, layout reset. |

### 10.2 What moves off workflow pages

| Current location | Move | Keep in place |
|---|---|---|
| Journal Health broker drawer | Tokens, ids, host configuration | Coverage/reconcile status and `Repair in Settings` link |
| Research → Price Alerts | ntfy credentials and always-on policy | Live alert inventory, add/remove/rearm/check |
| A.I. Summary | provider/model/key | scope choice, evidence preview, generation |
| Auto Pilot | auto-arm schedule and rebuild maintenance | current state, blocker, reconnect, scan now |
| BounceBot tab | tuning/type switches/configuration | slim live status and recent activity |
| Universe top-level page | build/compare/merge maintenance | optional read-only universe freshness link in Health |
| Settings → Testing Plan | remains under Advanced Operations | health may deep-link to the exact owed step |

### 10.3 Settings interaction rules

- Search finds setting labels and their current category.
- Changed settings show `Apply`/`Restart required` honestly; presentation-only
  changes may apply immediately if they are known safe.
- Dangerous maintenance actions are separated from ordinary toggles and require
  a preview/confirmation.
- Secret fields are masked, never logged, and never echoed in status text.
- A setting shown read-only because another owner controls it identifies that
  owner and provides the correct link.
- Settings never becomes a live dashboard; status summaries are concise and
  link back to the workflow page for detail.

---

## 11. Speed plan

### 11.1 Standard mode: fix known blocking work first

Extra hardware cannot compensate for work performed synchronously on Qt. The
original eight items are kept below with their 2026-08-26 status; what remains
is then **re-ordered by measured blocked time** (§3.2), not by how many IO
calls a panel makes — `theta_panel` is second-to-last by IO count and near the
top by time.

**Status of the original items (Wave P1 = `plan.md` Phase 0.8):**

1. Weekend Prep joins to an owned worker — **BUILT** (`d050ee1`, G-P1.1).
2. Focus mover state resolved once per refresh — **BUILT** (`0f04240` in the
   consumer, `10a3008` at the source under the extended fence authorization,
   G-P1.2). Keyed on the identity of the bars it measured, never a clock; a
   failed measurement is never cached.
3. Instrumentation — **BUILT in part** (`6bd7eef`, G-P1.3): `page_select`,
   the Journal inner tab, `model_apply`, `layout`, `chart_request`, and every
   stall record now carries the interaction id. **Owed:** `first_paint` and
   `chart_ready` (need the receiving paint path instrumented) and the Alert
   Center inner tab (fenced).
4. Hot `QTableWidget` rebuilds — **BUILT for System Health** (`49744a7`,
   G-P1.4). Everything else in this class is owed, and §3.2 says which two
   Qt table paths actually cost the time.
5. Never clear a populated page during a refresh — **now a standing rule**
   (§2) and applied where the wave touched (Weekend Prep, warehouse readout).
6. Stable desk layout; mode change is a Settings action — unchanged, PROPOSAL.
7. `reload()` audit — **audit DONE, one fix landed** (`49744a7`, G-P1.5:
   the warehouse readout). **Eight panels still read on the Qt thread from a
   click or page selection** and are listed in `plan.md` G-P1.5; none was
   touched, because a partial conversion of a page is worse than an honest
   list.
8. Keep the stylesheet/dynamic-property, bounded-GC, settings cache,
   review-event cache, bar-dict cache and warning-rate-limit fixes — kept.
   Added to that list: bounded shutdown joins (`e0f78ae`) and the panel-thread
   join (G-P1.6).

**Remaining work, in order of measured blocked time** (2026-08-26 session,
`blocked_ms`, from §3.2). This is the recommended order for whatever the
trader promotes next; `plan.md` Phase 0.8 already carries items 3–5 as owed
and item 1 as a trader decision, so this list references them and does not
re-plan them:

| # | Share | Site | Recommended treatment | Authorization |
|---|---:|---|---|---|
| 1 | 17.1% (248 s) | Cyclic GC sweeps (`_GuiGcController`) | Not a presentation change: a live scheduling component and the process's ONLY collector. Any work here — sweep cadence, generation thresholds, finding the producers of the cyclic garbage — is its own packet with its own measurement. Recorded as G-P1.7. | **Trader decision; NOT authorized** |
| 2 | 7.9% (115 s) | `DataTable.fit_columns` → `resizeColumnsToContents()` | Bound the measurement: size from a sample of rows (first N + widest-known) or remembered per-column widths, cap as today, and never re-measure on a refresh whose column set is unchanged. Callers: Setups (`master_avwap_panel`, after every scan), Day Trade Tracker, Move Forensics, RS Window industry table. Not a worker problem. | Owed under G-P1.4's class; presentation only |
| 3 | 5.4% (79 s) | Theta refresh (`ThetaPanel.refresh` on the report file watcher → parse on Qt → `endResetModel` → sort → `fit_columns`) | FIRST explain the growth (3.0 s → 26.6 s → 49.2 s across three hourly refreshes) — it is either the table path scaling with something that accumulates, or the sort proxy re-sorting per reset; then parse on a worker, diff rows into the model, and apply item 2. Owns the day's worst stall. | Listed in G-P1.5 (2 IO sites); presentation only |
| 4 | 3.9% (57 s) | `watchlist_utils.read_watchlist_symbols` → `read_text()` on Qt (204 stalls) | Find the Qt-thread caller (candidates: `watchlists_panel`, `universe_panel`, the autopilot/strength services when called from a slot) and move the read behind the same `ReadWorker` shape, cached on `(path, mtime, size)` like the settings cache. | G-P1.5 class |
| 5 | 2.1% (30 s) | `project_paths._load_local_settings` → `stat()` (56 stalls, ~0.5 s each) | A `stat` that averages half a second is not the `stat`. Measure what holds the thread at that moment (interaction id will say) before changing the cache; the obvious "stat less often" fix may address nothing. | G-P1.5 class; measure first |
| 6 | 1.7% (26 s) | `health_panel.py:374` `selectRow` after render | Re-measure on the post-`49744a7` build before touching it; the in-place table write may already have moved it. | G-P1.4 follow-up |
| 7 | — | The eight `reload()` panels by IO count (`setup_tracker_panel` 12, `industry_panel` 6, `master_avwap_panel` 4, `master_market_prep_panel` 3, `theta_panel` 2, `watchlists_panel` 2, `rs_window_panel` 1, `universe_panel` unaudited) | Same `ReadWorker` + last-good + bounded-join + fail-before-fix shape, one page at a time, whole page or not at all. Take them in the order the measured table above puts them (Theta and Setups first), not this count order. | Listed in G-P1.5 |
| 8 | — | The `app.exec()` 42.6% bucket | Not a fix target until the next soak names it: it is whatever Qt work has no Python frame — layout, paint, style repolish, model reset — and from the next session the interaction id says which click owns each record. Read that before spending a line of code. | Measure |
| 9 | — | Panel-thread sweep (§2, G-P1.6's class) | Every panel that starts a bare `threading.Thread` and may emit into itself: join on shutdown, bounded. Candidates listed in §2. A correctness item, not a speed one, but it belongs in the same wave because it is found by the same audit. | Recorded in G-P1.6 as "worth a sweep"; not done |

### 11.2 Experimental `Snappy` mode

Snappy mode spends spare RAM and CPU on bounded background preparation. It does
not increase IB request rates, alter scanner schedules, precompute decisions, or
run expensive work on Qt.

Recommended contract:

- explicit opt-in under Settings → Performance;
- default additional cache budget **2 GB**, selectable 1/2/3 GB, hard maximum
  **3 GB above Standard** until a soak proves more is safe;
- absolute process guard initially **5 GB private bytes**, with eviction before
  the guard and an automatic return to Standard behavior under OS memory
  pressure;
- low-priority UI-preparation pool, initially 2–4 workers and measured before
  raising it; network/IB pacing lanes remain unchanged;
- prewarm only a bounded hot set: current symbol, next review queue items,
  current Focus names, visible Master rows, selected board rows, and recently
  viewed symbols;
- cache immutable chart payloads, bar dictionaries, levels/earnings payloads,
  formatted table rows, and last-good read models—not duplicate QWidget trees;
- size entries by measured bytes and evict LRU by budget, not an unmeasured
  fixed symbol count;
- run prewarm after the current interaction settles and yield when a live scan
  or high-priority chart request needs the pool;
- keep source/as-of metadata with every cache entry; a fast stale answer must
  still say it is stale;
- expose cache used/budget, hit rate, evictions, queue depth, and last pressure
  release in Settings/System Health;
- one-click clear/disable; no restart needed unless the implementation proves
  that worker-pool sizing cannot change safely at runtime.

### 11.3 Performance acceptance targets

| Interaction | Standard target | Snappy target |
|---|---:|---:|
| Warm top-level page switch, click → first stable paint | p95 ≤ 150 ms | p95 ≤ 75 ms |
| Warm inner-tab switch | p95 ≤ 100 ms | p95 ≤ 50 ms |
| Cached symbol selection → first chart paint | ≤ 250 ms | ≤ 100 ms |
| Uncached chart request → visible progress state | ≤ 75 ms | ≤ 50 ms |
| Veto shortcut → focused reason list | ≤ 50 ms | ≤ 50 ms |
| Capture commit → visible saving state | ≤ 50 ms | ≤ 50 ms |
| Desk mode/layout switch if used | ≤ 300 ms | ≤ 200 ms |
| Navigation-triggered stalls over 50 ms | none attributable to file/analysis work | none |
| Full-session hard gate | no stall > 5 s; < 60 s total blocked | same, with a goal of < 30 s |
| Memory after three hours | existing < 2 GB target | Standard use + configured budget; never above guard |

Targets are acceptance criteria, not promises. Capture a Standard and Snappy
trace from the same scripted workflow and one live session before choosing a
default.

**What Wave P1 can be expected to do against the §3.2 baseline — stated so the
soak is read correctly.** Wave P1 removes one measured 12.5% item (the Focus
chip update) plus the Health page's churn (973 records across the week, small
each), and makes the 42.6% `app.exec()` bucket legible for the first time. It
does **not** touch the cyclic GC (17.1%) or the two Qt table paths (13.3%,
including the day's 49 s worst case), and it does not change what the eight
G-P1.5 panels read on a click. **Do not expect the total to halve.** A
post-fix session that shows roughly 12–15% less blocked time, no Focus-chip
culprit, and interaction ids on the event-loop records is Wave P1 working as
built; the worst-case number will not move until item 3 of §11.1 lands, and
the total will not approach the sixty-second gate until items 1–2 do. The
full-session hard gate in the table above therefore remains OWED past Wave P1
by construction, not by shortfall.

---

## 12. Responsive and accessibility rules

The primary target is 2304 × 1392 logical pixels, but the app must remain usable
at the existing 1680 × 954 laptop gate.

- Use logical pixels and current UI scale tokens; never branch on physical 4K
  pixels.
- Toolbars wrap or use an explicit overflow; labels are never clipped to
  ambiguous fragments.
- **Tables stretch to the available width.** A table never hugs the left edge
  of an otherwise empty page: the widest TEXT column takes the slack
  (`Line`, `trigger`, `cohort`, reason, note), numeric and badge columns keep
  their measured width, and the last section is not the only one that
  stretches. This is a rule for every `QTableWidget`/`QTableView` on every
  page, applied through the shared table shell (`DataTable`) and the theme,
  not per panel — the 2026-08-26 session found it on AWAY Recap and Weekend
  Prep and it recurs everywhere (§3.4 A).
- **Long identifiers elide in the MIDDLE**, never at the end, so the
  distinguishing tail survives (`human_f…tracking`, `veto_v3…compressed`,
  not `human_foc…`). Where a column holds a versioned or namespaced key, the
  full value is the tooltip and the head/tail split is deterministic. An
  elision that leaves every row reading the same is a rendering defect.
- Tables pin essential columns and move detail to a selected-row pane before
  adding horizontal scroll.
- **A refresh never blanks a populated page** (§2): last-good stays visible
  with its as-of, a failed read is stated in a reserved line, a successful
  empty read clears.
- Empty states explain source/action and do not vertically center a single line
  in a giant void.
- Compact density reduces padding and row height, not information or font
  legibility.
- Color is secondary to text/icon/state; red/green values retain sign/text.
- Focus order follows the visual order. All primary actions have visible
  shortcuts in the professional preset.
- Splitters remember per page/tab and have `Reset layout` in Settings.
- Automatic pane changes are limited to the documented contextual companion
  policy and never move the trader's current focus.
- No modal dialog for routine review/capture. Confirmations are reserved for
  destructive or ownership-changing actions.

---

## 13. Proposed implementation waves after approval

This is a dependency order for a future `plan.md` promotion, not authorization
to start.

### Wave U1 — primary desk

- Decide the Veto-first tabbed default and standalone Chart Review role.
- ~~Resolve the arm-bar contract/source mismatch under ask-first.~~ Dropped
  2026-08-26: there was no mismatch, the operating-context line is stale (§2).
- Implement contextual lower-row companions.
- Fit Veto/Like/Note modes without scroll at the target.
- Preserve keyboard, queue, and capture semantics with regression tests.
- Add 2304 × 1392 and 1680 × 954 layout screenshots/geometry assertions.

### Wave U2 — Market Journal and review loop

- Build the sticky composer/timeline shell.
- Attach current symbol on quick capture.
- Add selected-symbol shared chart and optional six-symbol grid.
- Add read-only machine-context and Trade Journal execution lanes.
- Add versioned AI synthesis presentation without changing the opt-in policy.
- Redesign AWAY Recap around selected chart + ranked lists.

### Wave U3 — remaining pages and navigation

- Group nav and rename Trade Journal.
- Apply empty-state/selected-detail rules to Focus, Strength, Weekend, Research,
  Auto Pilot, AI Summary, and Health.
- Move low-frequency Universe access into System/Settings.
- Verify every nested tab at both target viewports.

### Wave P1 — measured Standard-mode repairs — **BUILT 2026-08-26** (`plan.md` Phase 0.8, G-P1.0 … G-P1.7)

The one wave the trader promoted. Every code item is built on
`claude/gui-p1-fluidity` (off `main` at `53b9733`); suite 4902 passed /
19 subtests, exit 0; smoke 7/7. Every fix was written fail-before-fix.

| Item | Commit | What landed |
|---|---|---|
| G-P1.0 three verified defects | `db99271` | AWAY Recap never read the Focus lists (`load_focus_map(side)` against a keyword-only signature); its adoption-gate line called `mover_state(side, None, None, None)` and rendered UNKNOWN as a verdict; the Desk quick-journal write dropped `symbols` |
| G-P1.1 Weekend Review worker fix | `d050ee1` | The measured 8.45 s freeze; `WeekReviewPage` and `FocusReviewPage` on an owned single-flight worker; last-good survives a refresh; a failed read is stated |
| G-P1.2 Focus mover-state memo | `0f04240` + `10a3008` | First in the consumer, then at the source in `alert_center_panel.py` under an EXTENDED fence authorization; keyed on bar identity, never a clock |
| G-P1.3 navigation/paint instrumentation | `6bd7eef` | `scripts/ui/interaction_trace.py` + stall-record stamping. **Owed:** `first_paint`, `chart_ready`, the fenced Alert Center inner tab |
| G-P1.4 incremental model/view conversions | `49744a7` | System Health's three tables written in place. **Owed:** the rest of the class, in §11.1's measured order |
| G-P1.5 `reload()` audit | `49744a7` | Audit complete; the warehouse readout moved off Qt. **Owed:** eight panels, listed in `plan.md` |
| G-P1.6 a thread that outlived its panel | `49744a7` | `HealthPanel`'s audit thread joined on shutdown. **Owed:** the sweep of the class (§2) |
| Bounded shutdown joins | `e0f78ae` | `join_worker` (5 s) replaces four bare `wait()`s; source-level guard test |
| G-P1.7 the cyclic GC | — | **NOT started, not authorized**; recorded because the measurement is unambiguous (17.1%) |

**Still OWED for Wave P1 to be called done:** the §11.3 live soak on the
post-fix build with the watchdog enabled, compared against the §3.2 baseline
under the expectation stated in §11.3 — no test run discharges it.

### Phase 0.9 follow-ons (§15 decisions 9, 10, 14) — **G-P2.0…G-P2.2 BUILT 2026-08-27**

`plan.md` Phase 0.9, on `claude/gui-phase-0-9`; suite 5016 passed / 19 subtests,
exit 0; smoke 7/7; 37 tests, every one proved failing on the un-fixed code.

| Item | Commit | What landed |
|---|---|---|
| G-P2.0 the §12 width rule | `1fd9e6e` | `data_table.apply_width_rule` + `apply_width_rule_to_table_widget` + `MiddleElideDelegate`; `fit_columns` routes through it, so every `DataTable` user gets it, and AWAY Recap's four raw tables and Weekend Prep ▸ Focus pick review's five call it directly. Text columns may be named or MEASURED; identifier columns elide in the middle with the full value as the tooltip |
| G-P2.1 AWAY Recap as a return surface | `a5fa6a9` | §8.3 decisions 1–4: scanner status rows hidden and counted (one click reveals for the session, nothing deleted, the Alert Center's list untouched); a `Chart ▸` cell per chartable row plus `Enter` plus a hint line; symbol-less rows muted/italic from a theme token with no chart action |
| G-P2.2 Desk Journal route | `fd76923` | §5.3 option (a): `Ctrl+J` selects the Journal tab and focuses the composer; tab label reads `Journal  Ctrl+J`; panel scope, `WidgetWithChildrenShortcut`; no second row and no verb-row verb. Fenced file — the trader approved the exact diff in chat first |
| G-P2.3 next fluidity slice | — | **NOT started; gated on SOAK 1** |
| G-P2.4 GC measurement | — | **NOT started**; measurement only, no scheduling change authorized |

**One caveat for whoever soaks this:** `measure_column_widths` is still
`resizeColumnsToContents()` — the 7.9% / 115 s site — and G-P2.0 now reaches it
from two more pages. It is deliberately ONE seam, bounded by G-P2.3 item 1. Do
not judge table cost from a soak taken before that lands.

**Still OWED:** SOAK 1 against
`ui_stalls_prefix_baseline_2026-08-26.jsonl`, then G-P2.3 and G-P2.4.

### Waves U1–U3, S1 and P2 — **PROPOSAL, not authorized**

Nothing above promotes them. The new material they must plan against is
§3.4 (the trader's live findings), §2's standing constraints, and §12's table
rule; U2's AWAY Recap item is now §8.3's four decisions, and U1 no longer
carries the arm-bar move (§2).

### Wave P2 — experimental Snappy mode

- Implement bounded caches and low-priority prewarm behind one setting.
- Add memory pressure and cache observability.
- compare Standard/Snappy traces on identical workloads;
- keep experimental until the trader signs off on responsiveness and memory.

### Wave S1 — Settings consolidation

- Build the seven-category Settings shell and search/deep links.
- Move configuration one owner at a time; remove the old duplicate only after
  the new owner is verified.
- Keep operational status/actions on workflow pages.
- Audit secrets, restart behavior, and maintenance confirmations.

---

## 14. Required test matrix for a future build

### Automated layout/interaction

- 2304 × 1392 logical window (90%-width 4K target), standard and compact;
- 2560 × 1392 full-width 4K cross-check;
- 1680 × 954 laptop regression;
- nav expanded/collapsed;
- setups hidden/shown/full-width;
- every Trading Desk lower tab, with and without its companion;
- every Master AVWAP and Watchlists inner tab;
- every Trade Journal, Weekend Prep, Research, Setup Tracker, Day Trade Tracker,
  Ticker Lookup, AI Summary, Health, and Settings tab;
- long labels, maximum reason/claim counts, empty/populated/error/last-good states;
- no clipped primary action, no ambiguous elision, and no unexpected horizontal
  scroll at target width;
- Veto/Like/Note keyboard sequences work while Capture is hidden and visible;
- splitter state survives page changes, restart, scale change, and reset;
- source/age/unmeasured banners retain a fixed readable slot;
- Market Journal failed write retains text and displays `NOT SAVED`;
- entries/exits display read-only and link to the Trade Journal owner;
- settings deep links land on the intended category without changing values;
- **every table at 2304 × 1392 and 2560 × 1392 uses the available width**: no
  page whose widest table is narrower than 60% of its viewport while a text
  column is elided;
- **identifier columns elide in the middle**: an assertion that no two
  distinct cohort/source values render to the same elided string at the
  target width;
- **a refresh never blanks**: a failing worker on a populated Weekend Prep,
  Warehouse and (when built) Setups page leaves the last-good rows and states
  the failure; a successful empty read clears;
- **shutdown is bounded and joins every worker**: the existing source guard
  (`tests/test_shutdown_waits_are_bounded.py`) stays green, and a panel-thread
  sweep test constructs every panel that starts a bare `threading.Thread`,
  calls `shutdown`, and asserts the thread is joined before the widget is
  destroyed — the G-P1.6 crash reproduced only intermittently (4 in 6), so
  the test must assert the join, not the absence of a segfault;
- **AWAY Recap**: a status/blank-symbol alert never renders in the symbol
  table as a chartable row; a symbol row exposes a visible chart action;
- **Desk Journal**: the chosen route (§5.3) reaches the composer from the
  chart with focus, while the review pane still carries exactly one row
  between the charts and the tab strip.

### Performance workflow

Script the same sequence for Standard and Snappy:

1. launch to Veto Desk;
2. review 25 queued symbols with D1/M5 chart changes;
3. veto 10, like 5 with reasons, note 5;
4. open Master Setups, Watchlists, Industry, and RS Window;
5. switch Focus, Strength, Market Journal, AWAY Recap, Weekend Prep, Research,
   Trade Journal, Health, and Settings;
6. select populated rows and open shared charts;
7. leave the desk active for three hours with normal timers;
8. compare stall log, interaction spans, cache metrics, memory, CPU, and Qt
   warnings.

### Manual professional-use scenarios

- premarket: confirm the full thesis, Focus, source freshness, and setup roster
  can be read without hunting;
- active alert burst: keep Capture/Veto open and process the queue entirely by
  keyboard;
- Master consult: move to Master and back without losing queue/chart state;
- AWAY return: identify best candidates, chart them, adjust Focus, and write D1
  context from one page;
- intraday journal: add a thought in under five seconds and see it in the
  session timeline;
- after close: compare thoughts to entries/exits and generate an opt-in summary;
- weekend: complete all five steps without a GUI freeze or unexplained blank
  page;
- failure drill: stale bars, source outage, journal write failure, and bad
  credential each show the correct owner and next action.

---

## 15. Trader decisions required before promotion

Recommendations are supplied so none of these require inventing a design later:

1. **Trading Desk default:** adopt `Veto / Alerts` first and `Master AVWAP`
   second, with simultaneous Workspace split optional. **Recommended: yes.**
2. **Capture presentation:** show one full-width action mode at a time
   (`Veto | Like | Note`), remembering Veto. **Recommended: yes.**
3. **Arm bar:** ~~restore it to Armed above the inventory~~ — **reversed
   2026-08-26** (§2): the arm bar is under the chart because the trader asked
   for the hotbuttons and the ticker box there on 2026-08-20 (second pass);
   only the CLAUDE.md line is stale. **Recommended: keep it under the chart
   and correct the operating-context line; ask only if the trader wants it
   moved.**
4. **Standalone Chart Review:** retain only as a frozen/manual review theater;
   otherwise fold it into Trading Desk. **Recommended: retain initially and
   measure actual use.**
5. **Market Journal charts:** one large selected chart with up to six symbol
   selectors, plus optional 2 × 3 grid. **Recommended: yes; chart size wins over
   six simultaneous small charts.**
6. **Trade Journal Fees:** fold into Analytics/Reports or retain a dedicated tax
   tab. **Recommended: fold unless the dedicated export is used weekly.**
7. **Universe:** move out of primary nav into Settings/System tools.
   **Recommended: yes.**
8. **Snappy budget:** start with +2 GB, selectable through +3 GB, absolute 5 GB
   process guard until soak-tested. **Recommended: yes.**

Added 2026-08-26 from the live session and the Wave P1 build. **All six were
accepted by the trader the same evening ("i authorize all changes") with the
recommended answers**, and 9, 10, 11 and 14 are now `plan.md` Phase 0.9; 12
and 13 were applied to CLAUDE.md/AGENTS.md and `trading_desk.cmd` directly:

9. **AWAY Recap content (§8.3):** hide-and-count scanner status rows, add a
   visible `Chart` action + `Enter`, render symbol-less rows distinctly.
   **Recommended: yes to all three; presentation only, no change to the Alert
   Center's list.**
10. **Desk Journal route (§5.3):** shortcut + tab-label hint (no row cost), or
    a verb-row `Journal` verb (costs space on the one allowed row), never a
    second row. **Recommended: the shortcut; add the verb only if a mouse
    route is wanted.**
11. **Cyclic GC work (G-P1.7):** 17.1% of the pre-fix session's blocked time
    and the subsystem behind the 2026-08-21 8 GB incident. Its own packet or
    not at all; not a presentation change. **Recommended: authorize a
    measurement-first packet (what produces the cyclic garbage; sweep cost per
    generation) before any scheduling change.**
12. **Frozen exe as production (§3.5):** Smart App Control reads OFF; the
    stated reason for the source launch is stale. Keep the source launch as
    production, or return to the exe with its rebuild-before-merge delivery
    gap. **Recommended: keep the source launch until a deliberate rebuild +
    frozen selftest is scheduled; correct CLAUDE.md/AGENTS.md and the
    launcher header either way.**
13. **Operating-context correction:** the CLAUDE.md/AGENTS.md line placing the
    arm bar on the Armed tab describes the superseded first pass (§2).
    **Recommended: correct it to the 2026-08-20 second-pass contract.**
14. **Promotion of the next fluidity slice:** §11.1's measured order — the two
    Qt table paths and the Theta growth first, then the read sites, then the
    eight panels — as the next Phase 0.8 increment. **Recommended: yes, one
    page at a time, each with its fail-before-fix test and a soak between
    slices; the GC (decision 11) separately.**

Decisions 1–8 (the U/S/P2 waves) remain open; until they are accepted and
promoted into `plan.md`, those waves are PROPOSAL. Phase 0.8 (Wave P1, gate:
the live soak) and Phase 0.9 (decisions 9–11 and 14) are the promoted parts of
this document.
