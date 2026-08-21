# Current checkpoint

This file is the frequently refreshed active-work, branch, and verification stamp.

- Implemented inventory and revision history: [`CHANGELOG.md`](CHANGELOG.md)
- Remaining work and gates: [`plan.md`](plan.md)
- Supporting-document roles: [`docs/README.md`](docs/README.md)

---

## 2026-08-20, seventh pass — THE DESK NO LONGER PARSES HISTORY ON A HEALTH TICK

**Branch `testing-week-2026-08-17`.** Trader-authorized Phase-0 GUI
responsiveness repair and R6(c) diagnostic activation. Code is complete and
deterministically green; implementation commit **`d0aebd5`** and its checkpoint
follow-up **`7616499`** were fast-forwarded into the single release candidate on
2026-08-20. The bounded live week begins on the next desk restart.

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
| frozen exe | **not rebuilt** — no dependency, runtime asset, new top-level package, dynamic import or root-path trigger; the new module is an ordinary import inside the already-collected `ui` package |

### Owed, live

Restart the desk, then run the R6(c) bounded diagnostic week. Require no new
Windows `Application Hang` event, no repeated >50 ms culprit at a repaired
seam, and no responsiveness regression while clicking/typing during scan and
wrap-up. Confirm the next Master scan writes the compact active-event file and
that Away reports still verify. Inspect the watchdog log after each session;
do not tune a detector or threshold from this diagnostic.

---

## 2026-08-20, sixth pass — THE VETO COHORT IS GRADED NIGHTLY

**Branch `phase05-integration-blitz`.** Agreed design, built to spec (W1–W5).

### W1 — the function with zero callers now has one

`update_veto_cohort_outcomes` shipped with the cohort packet and was **never
called**. Picks accumulated on every veto commit; nothing graded them.

`ai_jobs/cohorts.py` → slot **`veto_cohort_grading`**, appended fourth
(5-minute reserve, 3 attempts). A slot rather than a step inside
`journal_import`, because the slot is the unit the runner already gives every
job — own ledger row, retry budget, reserve check, failure isolation — and
folding it in would make a grading failure read as a journal failure. Last,
not first: it costs seconds and the briefs must not lose window time to it.
**Deterministic — no model is called**, and a test asserts the provider is
never even consulted.

**Measured on the desk's real data:** 45 picks → 44 graded outcome rows,
0 sideless. `performance_rows: 0`, correctly — every pick is from today, so no
horizon has matured yet.

**Sideless rows are counted and named, never graded.**
`human_focus_tracking._side_label` reads anything that is not "SHORT…" as LONG,
blank included, so handing it one would manufacture a directional claim the
trader never made. Only their presence stages a filtered copy; the healthy path
touches no extra file.

**Idempotence, stated precisely.** A re-run changes exactly one column —
`updated_at` — and nothing measured. Byte-identical is deliberately *not* the
claim: a provenance stamp is supposed to move. Writing the failure test
surfaced the mechanism behind it — a fully matured pick is never recomputed,
which is why patching the outcome computer to raise did not raise.

**The volume defect does not reach these numbers.** Confirmed by inspection:
`human_focus_tracking` contains no reference to volume, AVWAP or bands. The
forward return is close-to-close only.

### W2 — the cohort key carries its vocabulary version

`veto_cohort_source(code, vocab_version)` → `veto_v2_compressed`. An omitted
version keeps the historical `veto_<code>`, which is what lets the 45 rows
already on disk keep grading in the cohort they were filed under — they are
not rewritten.

**Cost recorded, not hidden:** eight of nine v2 reasons are byte-identical to
their v1 entry, so this splits eight cohorts that could have been pooled. Right
way round (pooling stays recoverable from the key; a wrongly pooled cohort is
not), but it halves the sample per reason across the bump.

### W3 — `trader_judgement`, opt-in

Three sources in funding order — performance rollup, outcomes, then the raw
annotation log **last** (the same rule that stopped the setup tracker starving
its own scope). **Not** in `DEFAULT_SCOPES` or `TICKER_BRIEF_SCOPES`. Two
machine-written caveats travel with it as data: Main-swing-only claims, and
"Veto D1 — but M5 today" writing an ordinary veto row.

On demand: `run_ai_jobs.py --scopes trader_judgement`. The override is built
per call, so an opt-in scope cannot leak into the unattended slate by being set
once; unknown names are rejected at the CLI.

### W4 — review-event freshness, and the number that settles it

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
| `pytest tests/ -q` | **3942 passed / 19 subtests, 0 failed**; exit `0xC0000409` (known Qt-teardown crash after the summary). The intermittent `test_stale_d1_tail…` flake did not fire this run |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| frozen exe | **not rebuilt** — `ai_jobs` is in `PACKAGES_NOT_IN_THE_BUNDLE`, so a new module inside it is not a packaging trigger; spec-drift test passes |

### Owed, live

One weekend where the graded cohort is actually read (`--scopes
trader_judgement`) and the trader confirms the reasons ranked against forward
returns are the ones they recognise. Recorded against R8 in `plan.md`. The
weekly synthesis job is **not authorized** — cadence decided, gated on two
weeks of graded rows.

---

## 2026-08-20, fifth pass — THE CHART POPUP WAS UNTYPEABLE BY DESIGN

**Branch `phase05-integration-blitz`.** Trader: "i cant type in the master
avwap charts that I double click on in the notes section."

### One flag

`SymbolSnapshotDialog` set **`Qt.WindowDoesNotAcceptFocus`**. That flag does
not mean "do not steal focus" — it tells the window system the window may
**never hold keyboard focus**, so no widget inside it could receive a
keystroke. The note field, the veto note, the like note and the symbol box were
all dead: clicking in worked, typing did nothing.

**Pre-existing.** The flag has been there since the dialog was written; the
capture rail becoming the product is what made it matter.

### The intent was right, the mechanism was not

A chart popping up must not pull the caret out of a watchlist editor or the
live feed. That is `WA_ShowWithoutActivating`'s job (on Windows it maps to
`SW_SHOWNOACTIVATE`), together with `show()` + `raise_()` and **no**
`activateWindow()` in `show_symbol` — all kept. Those govern what happens when
the popup **appears**. `WindowDoesNotAcceptFocus` governed what could ever
happen afterwards, which is a different question and the wrong answer to it.

The two other users of the flag are correct and untouched: the price-alert
toast has no input and must never take focus, and the satellite window is
retired.

### Why the test asserts a flag and not a keystroke

**The offscreen platform does not enforce OS focus rules.** A test that focuses
the note field and types passes with the flag set *and* unset, so it would
never have caught this. The flag's absence is the contract, so the flag is what
is pinned — verified failing on the pre-change tree.

For the same reason the neighbouring "does not steal focus" test no longer
asserts `editor.hasFocus()`: offscreen has no show-without-activate, so that
assertion measured the test platform rather than the behaviour.

### One thing to confirm on the desk

Focus-stealing is the half that cannot be tested here. If the popup now pulls
the caret when it opens, that is this change and it is one line to revisit —
but `WA_ShowWithoutActivating` is the documented mechanism for exactly this and
is still set.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3902 passed / 19 subtests, 1 failed** — the pre-existing full-suite flake |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |

---

## 2026-08-20, fourth pass — EARNINGS ON THE CHART, AND VETO VOCABULARY v2

**Branch `phase05-integration-blitz`.** All trader-authorized, same day.

### The finding that shaped the earnings work

**The earnings cache holds NO future dates.** Measured on the desk's own
(`earnings_dates_cache.json`, refreshed 2026-08-20): **1,885 symbols, not one
forward date.** So "when does this name report next" is not a lookup — it is a
projection from the symbol's own cadence, and it is labelled `est` everywhere
it appears.

(A first read of that file suggested it was a year stale. It is not — the date
lists are stored **newest-first**, so a tail slice shows the OLDEST entries.
Median newest date is 2026-07-29. No staleness problem exists.)

`scripts/earnings_projection.py` is the math: pure, no I/O, no detector
contact. **Median** gap, not mean — one moved report would drag a mean around.
Gaps outside 40–200 days are dropped before the median (duplicated rows and
cache holes are not a rhythm). Measured cadence across the cache: **91 days**.

### Two things real symbols caught that fixtures would not have

1. **NVDA projected 08/19, one day before the reference.** The first draft
   rolled that forward a whole quarter and reported **November** for a report
   landing that week. `OVERDUE_GRACE_DAYS` (10) now keeps a just-passed
   projection and flags it **"E due"** instead.
2. **`MAX_PROJECTION_DAYS` was dead code** — a projection lands at most one
   cadence past the last report, and `MAX_CADENCE_DAYS` already bounds that at
   200, so a 200-day cap could never fire. Removed rather than left looking
   like a guard.

### Presentation, as the trader chose

- **E on a top ribbon**, dotted connector down to its own candle, never buried
  in price action. A report on a day the chart does not hold gets **no**
  marker — it is never nudged onto a neighbouring candle.
- **Reserved headroom on every symbol**, not only ones with an earnings date:
  otherwise two names at the same price draw at different scales. Without it a
  chart running to the top-right puts its E through the candles that made it —
  pinned by a test.
- **Projection pinned to the viewport's top-right, axis NOT extended.** It sits
  a median **48 sessions** past the last bar, so drawing it in place would cost
  ~40% of candle width to reach a date that is an estimate anyway.
- Built on the chart-data worker beside the levels, so the paint path still
  reads no caches; a failed lookup costs the markers, never the chart.

### Veto vocabulary v2

"S/R cluttered" → **"Compressed"**, as a **NEW code in `veto_reasons_v2.json`,
not a rename.** v1's own description sets that rule: a code is never reused for
a different meaning, because rows already carry it — and "too many levels in
the path" is not "range too tight to work with". v1 stays on disk and stays
loadable; every surviving code keeps its meaning **and its digit**.

Two tests hardcoded `vocab_version == 1` and failed the moment v2 shipped. They
now assert against the loaded vocabulary — the property they were always about
is that a row stamps the list it was written from, not that the number is 1.

### Like + claim

A numbered picklist like the veto, **Main swing only, for now** (trader's
words). A combo hides every option until opened, which is the opposite of the
rail's five-second contract; Alt+K then a digit is now a whole like. The
earnings-cycle, study and playbook groups are unreachable from this rail while
this stands — re-admitting one is adding it to `MAIN_CLAIM_GROUP`.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3899 passed / 19 subtests, 1 failed** — the pre-existing full-suite flake, verified failing identically on the pre-change tree |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| frozen exe | **not rebuilt.** `veto_reasons_v2.json` is a new runtime asset (trigger 2) and `earnings_projection.py` a new lazily-imported module, but the spec-drift test passes — the spec mirrors every non-`.py` file under each first-party tree, and the loose module is collected exactly as `chart_levels` already is from the same call site |

---

## 2026-08-20, third pass — THE PANE WAS SPENDING 1240px ON WHITESPACE

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

~1240px of a 4K screen on whitespace, for ~170px of content — in the state the
desk sits in **whenever the review queue is clear**.

**One-line cause.** The snapshot carries the pane's only expanding stretch, so
HIDING it left Qt with four `Preferred` widgets and a column of slack, which it
split equally. Charted, the same pane was already correct (chart 1212 of
1408px) — which is exactly why this never showed up in a charted screenshot.

**Fix.** An expanding `EmptyState` occupies the chart's slot whenever the chart
is hidden, so a stretch item is always present and the slack collects in one
place that explains how to get a chart. Title / setup line / arm bar pinned to
`Maximum` vertically — a `QLabel` defaults to `Preferred`, i.e. "I will happily
take more".

### The capture rail: 900px column → 379px of columns

Sections now **flow** (the primitive the arm bar already uses): wide hosts put
veto / like / note side by side, the narrow Capture tab still stacks them with
nothing clipped. Symbol and side share one line. The veto list is sized **from
the vocabulary** instead of a hardcoded 190px cap, so all nine reasons are
visible — a surface built for two keystrokes cannot ask for a digit the trader
cannot see. Deliberately NOT a wrapped multi-column list: those labels only fit
in columns by eliding them.

### Capture verbs

- **LIKE now retires the chart**, like a veto — in the Alert Center queue and
  in the Master AVWAP snapshot popup, which already had
  `snapshot_review_advance` for exactly this.
- **NOTE still holds the chart.** It is written ABOUT the thing in front of
  you; a rail that skipped would make every note cost the trader that chart.
- **Hypothetical stop removed** from the rail. The **control only** —
  `ui.annotations.store` still builds and validates `hypo_stop` rows, because
  the stream is append-only evidence and rows already on disk have to stay
  readable. Re-adding it is a layout change, not a migration.

### Still open, deliberately

The **horizontal** split was not touched. The screenshot shows the Setups table
truncating columns ("Diagnostics & Li...", "AVWAP_BAND...") while the alert
column holds an empty chart — but that split is **persisted** (`qt_desk_split_sizes_v2`)
and may be one the trader dragged themselves, so re-weighting it silently would
overwrite their own choice. Needs a decision, not a guess.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3860 passed / 19 subtests, 1 failed** — the pre-existing full-suite flake, verified failing identically on the pre-change tree |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| frozen exe | **not rebuilt** — no packaging trigger hit |

---

## 2026-08-20, second pass — OPEN DEFECT: THE DAILY STORE MIXES TWO VOLUME UNITS

**Branch `phase05-integration-blitz`.** Found while wiring D1 volume bars.
**Not fixed — it needs a trader decision, and the fix is inside detector
input data (ask-first rule).**

### What is wrong

`data/daily_bars/*.parquet` carries volume from two sources with **no unit
normalization between them**. `master_avwap_lib.legacy._normalize_daily_bar_frame`
normalizes column names, dtypes and duplicate dates — and nothing else. IBKR
rows and Yahoo rows are appended into one column as-is.

Proven against a reference, NVDA, same file:

| Session | Daily store | yfinance | Ratio |
|---|---|---|---|
| 2026-05-18 | 934,776 | 146,280,900 | 156× |
| 2026-05-19 | 823,818 | 140,948,200 | 171× |
| 2026-05-20 | 940,980 | 184,201,600 | 196× |
| 2026-05-27 | 167,601,200 | 167,601,200 | **1× (exact)** |
| 2026-06-01 | 212,850,700 | 212,850,700 | **1× (exact)** |

The Yahoo-sourced rows are exact. The IBKR-sourced rows are low by a
**variable** 150–200×, so it is not a clean 100-share-lot conversion and a
constant rescale would be a guess. It alternates in blocks, following whichever
source answered on the day.

**Scale: 338 of 1,949 stored symbols (17.3%)** have a volume series straddling
two magnitudes (p90/p10 > 20×). That is an upper bound — a genuinely spiky name
can trip the same test — but the mechanism is confirmed by reading the code,
not inferred from the statistic.

### Why it matters beyond the chart

`calc_anchored_vwap_bands` is **volume-weighted** over this frame's `volume`
column. A day under-reported 150× contributes ~0.6% of its true weight, so on
an affected symbol the D1 anchored VWAP is effectively computed from the
Yahoo-sourced days alone. Every band consumer — events, zones, tracker
families, scoring history — sits downstream of that.

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

1. Normalize at the writer and **backfill** the store from one source — moves
   the bands, needs golden fixtures first.
2. Normalize at the writer for **new rows only** — stops the bleed, leaves a
   discontinuity mid-history.
3. Drop volume from the IBKR path and take it from Yahoo only — one unit, one
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
full-suite load it can miss it. Left alone deliberately — unlike this morning's
clock fixture, this one needs an investigation of shared state in a
chart/alert-adjacent widget, not a three-line repair.

---

## 2026-08-20, second pass — WHAT THE TRADER ASKED FOR AFTER USING IT

Four changes, all trader-authorized in one message.

1. **Veto retires the chart.** "When I click veto it should just disappear as
   'not for today'." A veto now takes the "Not today" path: recorded, removed
   from today's feed and chart queue, next chart up. LIKE, hypothetical stop
   and note still hold the chart — a note that skipped to the next symbol
   would cost the trader the chart they were writing it about.
2. **"Veto D1 - but M5 today".** "It may be a shit D1 chart but its a good
   daytrade." The rail does not place the name; it emits a REQUEST and the
   panel that owns the Focus store does the placement, same shape as
   BounceBot's desync request, one writer per store. Place first, retire
   second — retiring is what drops the alert object the placement needs. A
   failed placement still retires the chart, because the veto is already on
   disk. **Known limitation, deliberately not papered over:** the veto row is
   an ordinary veto with no new field, so the veto cohort study will count a
   day-traded name as vetoed. Making that queryable is a schema v2 decision.
3. **The arm bar comes back under the chart.** "I also need my m5 and D1 alert
   hotbuttons back on the bottom of the visual chart... I also need the ability
   to input a ticker manually as well." Only the capture rail stays on a tab.
   Measured at this column's 420px: rail 697px, arm bar 131px — sending only
   the rail away keeps 84% of the reclaimed height. `docked_controls` splits
   into `dock_arm_bar` / `dock_capture_rail`. The Armed tab is the
   cross-symbol inventory again; the verb-row armed line switches off with the
   bar docked (its own chips are right there), and the tab keeps its count.
4. **D1 volume bars** — an underlay in the bottom 18% of the price view, not a
   stacked sub-plot, so they cost no chart height. No fetch: the daily store
   already carries volume. **Read the open-defect entry above before trusting
   what they show.**

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3847 passed / 19 subtests, 1 failed** — the failure is the pre-existing full-suite flake named above, which fails identically on the pre-change tree |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| frozen exe | **not rebuilt** — no packaging trigger hit |

Fail-before-feature: all 10 volume tests and the new veto/layout tests were run
against the pre-change tree and failed there.

### A desk restart IS needed

Source-level only, as before.

---

## 2026-08-20, morning — THE CHARTS GET THE PANE. READ THIS FIRST.

**Branch `phase05-integration-blitz`.** Built on the desk's live checkout
(`C:\Users\Aaron\TradingBotV3`) while the desk was running. **Nothing was
restarted and no scheduled task was touched.**

### What the trader asked for, and got

> "I cannot see the charts at all… I am ok with them being tabbed where
> alerts/D1 focus/RSRW board is and clicking into them. But I need to be able
> to see the charts."

The Alert Center review pane stacked title → setup text → charts → a two-row
arm bar → a ~600px capture rail → the verb row, all in the desk's narrow alert
column. The charts — the point of the surface — got whatever was left.

**Now: charts, then one row.** The arm bar moved onto the existing **Armed**
tab, above the inventory it fills. The capture rail became a new scrolled
**Capture** tab. Under the charts there is exactly one row left: the verb row
(Remove/Skip/Not today/Add + queue count), which advances the review queue and
must never cost a click.

Tab strip is now `Alerts | D1 Focus | RS/RW Board | Armed | Capture`.

### Why the arm bar joined "Armed" instead of becoming a sixth tab

"Arm" and "Armed" a millimetre apart on one strip is a misclick waiting to
happen, and the controls and the list they produce are one subject. Arming is
also deliberate enough that a click is fine — unlike the verb row.

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
Qt fires **neither** — the failure mode is the keys going dead with nothing on
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
or watchlist write — this was re-parenting only), the movers-only presentation
filter, the repetition fold, and every line of adoption-gate code.

### Second change: a wake alert the trader can verify

Audit confirmed both EVENING-permitted senders already push at ntfy's maximum
(`price_alert_service._notify` and `AutopilotService._maybe_push_spy_alarm`,
both `priority="urgent"`). The gap was the channel **test**, which went out at
`high` — so "will this break through iOS Sleep Focus" had never been
answerable.

New **Test wake alert (urgent)** button beside Test Push
(`PriceAlertService.test_push(urgent=True)`), sending one urgent push whose
message says what should have happened. **Not a new sender**: nothing
schedules it, only that button calls it, and the phone push policy is
unchanged. `docs/EVENING_MODE_RUNBOOK.md` gains a Sleep breakthrough checklist
— ntfy has no Apple critical-alert entitlement, so urgent priority alone cannot
override Sleep Focus; the device steps are marked to-be-confirmed-on-desk.

### One pre-existing failure repaired

`test_it_reads_the_gates_predicate_over_the_desks_own_bars` was **already red
on this branch before any change here** — it built an 11:00 session while
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
| frozen exe | **deliberately not rebuilt** — no packaging trigger is hit (no new dependency, asset, top-level package or dynamic import). Smart App Control still blocks the built exe on this machine; see the 2026-08-19 midday entry, unchanged and unresolved |

Fail-before-feature: 10 of the 11 new Alert Center capture tests were run
against `d60cbaf` and **all 10 failed**; the eleventh is a deliberate
regression guard on the unchanged recorder path.

### A desk restart IS needed

Source-level only, so **the trader sees none of this until the desk is
restarted.** Nothing is urgent: what changed is where controls sit and one new
test button — nothing about what is detected, recorded, alerted or pushed.
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

## 2026-08-19, evening — MOVERS ONLY IN CHART REVIEW. READ THIS FIRST.

Built on the desk's live checkout (`C:\Users\Aaron\TradingBotV3`, branch
`phase05-integration-blitz`) while the desk was running. **Nothing was restarted
and no scheduled task was touched** — see "restart" below.

### The trader's rule, as recorded

> "A long inside yesterday's range is probably chop. Chart review should only
> show me longs above the previous day's high and shorts below the previous
> day's low. Focus picks that ARE beyond their previous-day extreme should be
> flagged - those are the ones actually moving. Inside-range picks appear only
> when I deliberately review focus picks."

Verbatim as a dated addendum in `docs/M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md`.

### One predicate, not two

`focus_adoption_gate.mover_state(side, price, prev_high, prev_low)` is the
adoption gate's **extreme leg alone** — a thin name over the same
`prev_day_break_state` call — and `focus_adoption_gate_state` now routes its own
extreme leg through it. There is exactly one implementation of "beyond
yesterday's extreme" in the tree, and a test walks the whole input matrix
asserting the two entry points cannot disagree. That is the point: a display
filter with a private copy of the rule would eventually hide a name the machine
had just adopted, and the trader would be reading a queue that disagreed with
their own Focus list.

No session-VWAP leg. The filter asks the weaker question deliberately — the
trader wants to *see* movers, not only the ones the machine would take.

### Where the filter lives, and what it will not do

`AlertCenterPanel._enqueue_review_alert` — the single door into the review
queue, so the D1 Focus feed, the auto-pick drain and the scanner alerts all pass
through it. Default ON.

- Longs and shorts inside yesterday's range: **not queued**.
- **UNKNOWN shows**, tagged `unmeasured`. Missing data is uncertainty; a filter
  that failed closed would blank the review the moment the daily store hiccuped.
- The withheld are counted on a clickable line, `N hidden (inside yesterday's
  range) - show`. One click shows exactly those names and turns the filter off
  **for that session** (day-scoped — tomorrow opens filtered again).
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
poll through a new `focusBreakStatesChanged` signal — **no new timer, no new
market data, no IB traffic**.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3822 passed / 19 subtests, 0 failed, 0 errors**; process exit **`0xC0000409`** (the intermittent Qt-teardown crash, measured through Python's `returncode` — bash shows it as `127`) |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| frozen exe | **not executed** — Smart App Control still blocks the built exe on this machine (see the midday entry). Unchanged and unresolved; no new module was added to the selftest roster, so the expected count stays 56 |

Fail-before-feature: all 28 new tests were run against `e207851` in a throwaway
worktree first and **all 28 failed**; all 28 pass here.

### A desk restart IS needed

The running desk is executing the Python it imported at launch. Everything here
is source-level, so **the trader sees none of it until the desk is restarted.**
Nothing is urgent: the filter changes what is charted, not what is detected,
recorded or alerted, so the running session keeps working exactly as it did.

Cleanest moment: **after tomorrow's close, or before the 07:00 task on
2026-08-20** — the scheduled task launches from source, so letting the desk be
restarted the usual way picks it up with no extra step. To take it now: close the
desk and relaunch via `scripts/launch_gui_auto.ps1` (the task's own path). No task
disarm is needed, because the branch is not changing.

### Owed, live

`docs/DESK_TESTING_PLAN.md` §2.10 — one review session where the trader
confirms the queue shows only movers and the hidden-count line is honest.


---

## 2026-08-19, midday — the desk flipped to this branch

The trader flipped the desk to `phase05-integration-blitz` at 11:08 PT
(mid-session, deliberately — trader's call on a slow tape). The worktree
`..\TradingBotV3-blitz` is removed (it was clean and fully pushed); this main
checkout at `C:\Users\Aaron\TradingBotV3` now holds the branch. Sequence
executed: task disarmed → desk closed by the trader → checkout `198a2bd` →
gates → manual launch via `scripts/launch_gui_auto.ps1` (the task's own path)
→ task re-armed (all three tasks `Ready`). New desk pid 13364, heartbeat
fresh, Auto Pilot resumed ON, slot 11:00 picked up at 11:09:30.

Gates on this checkout, 2026-08-19 ~11:00 PT:

| Check | Result |
|---|---|
| pytest | **3794 passed / 19 subtests, 0 failed**, process exit **0** (the intermittent `0xC0000409` did not occur this run) |
| smoke | **7/7**, exit 0 |
| source selftest | **56/56**, exit 0 |
| frozen rebuild | clean-cache rebuild exit 0 — **but the exe could not be executed; see below** |

**NEW OPEN ITEM — Smart App Control blocks the freshly built exe.** Windows 11
Smart App Control (enforcing, `VerifiedAndReputablePolicyState=1`) refused to
run `dist\TradingBotV3\TradingBotV3.exe` built at ~11:05 ("An Application
Control policy has blocked this file"; CodeIntegrity events 3077/3118 at
11:07). The worktree's byte-different build had run fine at 09:20 the same
morning — SAC verdicts are per-hash cloud reputation, so they can differ
between rebuilds. **The desk is unaffected** (the 07:00 task launches from
source, and this flip was verified with the source selftest), but the frozen
gate cannot be relied on to *execute* on this machine until this is resolved.
Options are a trader decision: code-sign the exe (SAC needs a real signature
with reputation), stop using SAC (WARNING: once turned off it cannot be
re-enabled without reinstalling Windows), or accept that the frozen selftest
may intermittently be blocked and re-run it when the verdict clears. Recorded
here; not resolved.

---

## 2026-08-19 — the gate that could not tell the time.

**Branch: `phase05-integration-blitz`, pushed.** Everything
below happened in `..\TradingBotV3-blitz` (worktree since removed; see the
flip entry above).

### What the first DESK morning actually did

**Zero adoptions. 121 picks refused every 30 seconds from 08:07 onward.**
`focus_auto_picks.json` finished the day with an empty `picks` map, and the
failure logging rotated `trading_bot.log`.

**Root cause — one subtraction, two clocks.** A stored verdict carries two
stamps written by different paths:

| Field | Writer | Awareness |
|---|---|---|
| `gate_bar_end` | the intraday profile's `as_of` (`_intraday_extreme_metrics`) | **always aware** — the provider's own offset when it has one, market-local otherwise |
| `gate_checked_at` | the staging refresh's `datetime.now()` | **naive** |
| the caller's `now` | `AlertCenterPanel` → `datetime.now()` | **naive** |

So `pending_pick_gate_ok`'s wall-clock age check (naive − naive) passed, and its
bar-lag check (naive − aware) raised
`TypeError: can't subtract offset-naive and offset-aware datetimes` — exactly the
line the traceback named. The Alert Center caught it and refused fail-closed,
which is correct behaviour for an unverifiable pick.

**The gate did not judge the picks wrongly. It never ran.** Nothing about the
PDH/VWAP rule was exercised on 2026-08-19, which is why §2.5 and §2.6 of the
testing plan are re-owed **in full** rather than being partly done.

**The fix.** Every datetime the gate compares — the caller's clock, both stored
stamps and the `not_before` flip barrier — is normalized at one seam
(`_gate_moment` → `market_session.normalize_market_local_datetime`), which
ATTACHES market-local to a naive stamp and converts an aware one. Stripping the
offset instead would have ended the crash and kept the outage: an aware 11:05 ET
bar read as naive against an 08:07 PT clock is three hours "ahead of the tape",
so every pick would still have been refused — silently. A test pins that
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
= 5) governs **only** a failed flip re-measurement — the `reverify_pending_picks`
fetch on an AWAY/EVENING → DESK return. The desk was in DESK mode from the start
on 08-19, so no flip happened, no re-verification was owed, and that budget was
never engaged. Correctly.

The 30-second cadence in the log is the **ordinary poll**:
`_poll_auto_pick_pending` rides the Alert Center's 30s `_watch_timer`, and a
refused pick is deliberately not marked seen, so every cycle re-attempts the
whole queue. That is designed — "a stale verdict costs one cycle rather than the
pick" — and it is what makes recovery automatic once the code is fixed rather
than requiring a restart. Two mechanisms were being read as one; nothing
disagreed, so nothing was changed. Recorded in the R2 spec so it is not
re-litigated.

### The strength board, on the trader's two requests

**Sortable columns.** Every heading sorts, with a visible indicator, and clicking
the same heading flips it. Sorting is presentation: it re-orders rows already in
hand and never calls the service, so a header click cannot cost a refetch — the
board's budget stays one batched yfinance pull per 15 minutes and **zero IB
traffic**. Qt's own `setSortingEnabled` is deliberately not used: the last column
holds a per-row cell *widget*, and `QTableWidget` leaves cell widgets behind when
it sorts, so the Add button would end up on its neighbour's row. Owning the order
also puts blank cells last in **both** directions — an unmeasured field is an
absence, not a small number. The default order is unchanged and now stated by the
indicator (longs strength-descending, shorts ascending — strongest for that side
first). Every add still re-runs the adoption gate at click time.

**Charts on selection.** Selecting a row opens that symbol in the desk's existing
snapshot popup — the same one the RS/RW, entry and Industry boards use, owned by
the Alert Center — so it carries the same bot-backed series, painted levels and
CaptureRail. No new chart widget exists anywhere (R4's unification pattern), and
`show_symbol_snapshot` already reuses one dialog per owner, so re-selecting
re-points that window instead of stacking dialogs. Selecting on one side clears
the other; a refresh that keeps the same row selected is not a new chart request;
double-click still works.

**The docked chart is the follow-up option, not this build.** An always-visible
chart inside the board needs a desk-layout decision about what happens to the two
tables' width on that page — a judgement about the trader's screen, not a wiring
problem. The popup reuses a surface the trader already knows.

### Gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3794 passed / 19 subtests, 0 failed, 0 errors**. **The PROCESS exit code is `0xC0000409`, not 0** — see the finding below |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| Source selftest | **56/56**, exit 0 (roster unchanged — no new module is lazily imported) |
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
3. **Neither of today's new test files causes it** — removing either one leaves
   the crash exactly where it was.

**Not "fixed" by editing product code.** `scripts/ui/stall_watchdog.py` is
product code owed R6(c)'s diagnostic week, and making a suite exit cleanly is not
a reason to touch it. The standing rule holds: **quote the summary line AND the
exit code together; neither alone is the truth.** What is new is that the
summary line is now the one that has stayed stable across every run, and the exit
code is the one that moves.

### Live proofs — what today changed

**Re-owed in full** (the 08-19 session proved nothing about them, because the
gate crashed before it could judge anything):

- one adoption actually happening on a DESK day (new §2.5 check: names landing in
  M5 Focus, `focus_auto_picks.json` non-empty);
- one adoption-time refusal with its reason;
- one scoped "Not today" leaving the trader's other entries intact.

**Newly owed:**

- §2.7a — the board's sorting and chart-on-selection, on real rows.

**Unchanged and still owed:** the strength board's TC2000-character check, the
EVENING stop, the SPY wake alarm, and everything in R3–R8's ledger from the
08-18 report below.

### Putting this build on the desk

Same sequence as the 08-18 report below, with one number changed — the frozen
selftest count is **56/56**, unchanged from yesterday, because nothing added a
lazily-imported module today:

1. **Disarm the scheduled task first.**
2. Close the desk app, then in `C:\Users\Aaron\TradingBotV3`:
   `git fetch origin` → `git checkout phase05-integration-blitz`.
3. `.venv\Scripts\python.exe -m pytest tests/ -q` (expect the figure above, exit
   0) and `.venv\Scripts\python.exe scripts/smoke_check.py` (7/7).
4. `.venv\Scripts\pyinstaller.exe .\packaging\tradingbotv3.spec --noconfirm`,
   then `dist\TradingBotV3\TradingBotV3.exe --selftest` — expect
   `selftest OK: 56/56 checks passed (frozen)`, exit 0.
5. Launch once by hand; confirm the desk opens on Main and the Focus tab looks
   normal.
6. **Re-arm the scheduled task.**

**On the next DESK morning, check the thing that failed:** within a poll cycle or
two of the first staged picks, names should appear in M5 Focus and the status
line should say `N auto pick(s) added to M5 Focus for today`. If the gate ever
fails again you should see **one** traceback and **one**
`Focus gate check unavailable for N staged pick(s) this cycle` line per 30
seconds — not a flood. A flood is a regression worth reporting on its own.


---

## The 2026-08-18 integration blitz report (still current for everything below)

**Branch: `phase05-integration-blitz`, pushed. The desk is untouched.** The main
checkout at `C:\Users\Aaron\TradingBotV3` is still on
`phase05-r2-focus-gating-strength-board`, its `dist/` was never rebuilt from
this work, and the 07:00 scheduled task will run exactly what it ran yesterday.
Everything below happened in a linked worktree at `..\TradingBotV3-blitz`.

### The first thing to know: most of the redirect was already built

The redirect asked for R3, R4, R5, R6, R7 and R8. When the branch audit ran, R3
through R8 were **already built** on `testing-week-2026-08-17` — 30-plus commits
from 2026-08-16/17 that the desk branch's own checkpoint had not caught up with.
So the blitz branch was cut from that lineage and the four newer R2 commits were
merged into it, rather than rebuilding landed work (which `CLAUDE.md` forbids).

That merge is itself a deliverable: **one branch now carries testing-week, R1,
R1.1, R2 (including the 08-18 defect fixes), R3, R4, R5, R6, R7 and R8.** Before
today, the desk branch and the release candidate had diverged.

### What was actually built today

| Packet | State before today | What landed |
|---|---|---|
| **R5 §3.2** | pure logic not written | Confluence engine (HA reversal + SMI turn + LRSI cross within 4 completed bars), **M5 Focus symbols only**, wired, **default OFF** |
| **R5 §3.3** | pure logic not written | First-candle ORB flow: candidate mark, post-pullback new-extreme break, informational LRSI recross — three separately toggleable types, all **default OFF** |
| **R5 §4** | not started | `AnyBounceWatch`: one armed request per symbol/side over nine levels, own store, Alert Center owns it, fires once naming the level that held then disarms; **Any bounce** button on the arm bar |
| **R5 §8.3** | decided, not built | `prev_avwape` carried onto the zone-arms entry as a top-level key, golden fixture first, fixture passes unchanged after the edit |
| **R6(b)** | decided + narrowed | Read-only JSONL-ledger audit inside the existing footprint check; the stale `~106 MB` comment removed. R6 is now fully closed |
| **R7 visuals** | deferred | Analytics per-group bar charts with honest n counts + a CSV of exactly what is charted; Calendar pyqtgraph year heatmap centred on zero |
| **R8 joins** | retained future scope | Week review folds the week's RS/RW extremes per symbol; Focus review joins picks WITH their outcomes, one row per pick |
| **R4 held items** | held ask-first | Focus Picks reviewed-today marker built as a line BESIDE the editors; `review_host` for the boards declined on the record |
| **WISHLIST** | 20+ candidates | One was buildable and is built (external chart deep link). Every other item has one blocking trader question, written down in `docs/WISHLIST_OPEN_QUESTIONS.md` |

### Every autonomous decision I made where a spec was ambiguous

1. **R5 §8.2 said do not wire §3.2/§3.3 until a desk session measures §3.1.** The
   redirect is that decision's own first reopen trigger ("the trader
   overrides"), so I wired them — and kept the substance by shipping **all four
   new alert types OFF** and writing both engines as **stateless** functions
   over the session's completed bars. §8.2's objection was a dormant state
   machine waking mid-session with contents nobody exercised; a function that
   recomputes from bars has nothing to carry. **What the desk session now
   decides is which toggles earn a default-on, not whether the code exists.**
2. **The ORB candidate mark does not seed the bounce outcome tracker.** Only the
   re-break does. Measuring an engine against events it never claimed were
   entries would corrupt the evidence the promotion ladder reads.
3. **The confluence is Focus-scoped at the sweep**, intersecting the watchlist
   with the human focus sets — the trader's framing was "on names I'm watching",
   and a perfect chart on a non-Focus name is silence.
4. **The any-bounce watch reuses `detect_zone_arm_triggers`' two-bar idiom**
   rather than inventing a bounce rule, so "bounce" means one thing system-wide.
   Its tolerance for a chart-armed watch (no scan measurement available) is
   0.15% of the level — deliberately small, and a named constant.
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

**R3 §4.3.5, the same-slot volume-thrust normalization.** The trader deferred it
explicitly on 2026-08-16 with a reason that today's redirect does not touch: the
D1 scoring seam has no intraday slot series, the faithful TC2000 baseline would
need a 5-minute fetch across ~1,100 symbols (a data-budget and contract change),
and the zero-fetch session-elapsed proration was offered and REJECTED because
real volume is U-shaped. Reopening it needs a fresh decision about the data
seam, not a permission — the blanket ask-first approval removes the asking, not
the missing judgment. The 18-point thrust bonus therefore keeps its full-day
baseline as a known, accepted pre-close gap, characterized by
`tests/fixtures/r3_swing_quality_v1.json`.

### Wishlist: built vs stubbed-with-a-question

**Built (1):** deep-link a symbol into an external charting tool.

**Stubbed with the blocking question stated (12), in
`docs/WISHLIST_OPEN_QUESTIONS.md`:** voice dictation (local vs cloud speech, and
what happens to a bad transcription); chart line-density presets (blocked on
P1.2's clutter budget — a desk-evidence decision, not a preference); read-only
mobile/web dashboard (who may read it, from where); self-hosted ntfy (is the
operational burden worth it); macOS scheduled jobs (will a Mac ever be the
unattended host); broader strength-board universe (explicitly gated on the R2
board proving itself); and the six research/data captures, which share one rule —
each needs a **registered consumer** before capture is justified.

Nothing on that list was implemented, and nothing was promoted into `plan.md`
except the one item that was built.

### The gate figures

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3760 passed / 19 subtests, 0 failed, 0 errors, exit 0 (2026-08-18 21:56 PT)** |
| `scripts/smoke_check.py` | **7/7**, exit 0 |
| Source selftest | **56/56** (55 → 56: `external_chart_links` joined the roster) |
| Clean-cache frozen rebuild + frozen selftest | **`selftest OK: 56/56 checks passed (frozen)`, exit 0. `build/` AND `dist/` deleted first and built from the worktree, so the desk's own `dist/` was never touched; exe mtime 22:02 postdates the commit at 22:00** |

**The teardown crash is gone.** The `0xC0000409` native crash at interpreter
shutdown that the testing-week checkpoint says must be quoted alongside the
summary line **did not occur in any run today** — every full-suite run on this
branch exited 0. I did not fix it and cannot claim it is fixed; I can only
report that it stopped reproducing on this tree. If it returns, quote the
summary line and the exit code together, as that entry says.

### Live proofs now owed — the full ledger

**Nothing below has been observed.** UNKNOWN is a result and `plan.md` sec 6
requires recording it as one.

Inherited (8, unchanged except where the 08-17/08-18 AWAY sessions closed one):

- **R1 (3 open):** an EVENING day that stops after its early block; one SPY ±1%
  alarm; the AWAY→DESK **drain on return** (the trader never flipped back). The
  quiet boot PASSED 2026-08-16; AWAY staging-without-adoption PASSED both days.
- **R2 (3 open):** one adoption-time refusal; one scoped "Not today" that leaves
  other entries intact; one strength-board session matching the TC2000 scan's
  character. The eviction proof PASSED 2026-08-18.
- **R3 (3):** the `would_demote` shadow week **before any row moves**; the
  one-week 12:45-vs-close and STABLE-vs-PREVIEW churn comparison; the first
  real-data curation cycle.
- **R4:** the whole §8 exit gate.
- **R6:** the stall-watchdog diagnostic week.
- **R7:** the trader-present finale — dry-run migration report, live migration,
  full backfill, ≥10-trade statement audit, one clean reconciliation week, ≥5
  consecutive nightly ledger entries.
- **R8:** one real weekend run (does not wait for Monday).

Added by today's work:

- **R5, per engine:** one desk session confirming the **LRSI cross** volume is
  sane; then the same for the **confluence**; then for the **first-candle ORB**.
  Each session also decides whether that engine's toggle should default on.
- **R5 §4:** one observed any-bounce firing that names the level that held, and
  one re-arm after it.
- **R5 §8.3:** one scan whose zone-arms file actually carries `prev_avwape` for a
  symbol with a prior anchor (deterministic tests prove the shape, not the feed).
- **R7 visuals:** one look at the Analytics group chart and the year heatmap on
  real data, to confirm the thin-sample labels and the blank (untraded) days read
  the way you expect.
- **R8 joins:** one weekend where the focus-review table shows a pick whose
  horizons are still maturing, so the blank-not-zero rule is seen rather than
  trusted.

### Known weak spots — where I would look first if something misbehaves

1. **The four new alert types are OFF.** If you turn one on and the feed floods,
   that is the volume question §7 was written to ask — turn it back off and tell
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
   the chart looks emptier than the table, read the note under it — it says how
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
   then `dist\TradingBotV3\TradingBotV3.exe --selftest` — expect
   `selftest OK: 56/56 checks passed (frozen)`, exit 0.
5. Launch once by hand and confirm the desk opens on Main, Auto mode reads what
   you left it at, and the Focus tab looks normal.
6. **Re-arm the scheduled task**: `Enable-ScheduledTask`.

To back out at any point: `git checkout phase05-r2-focus-gating-strength-board`,
rebuild, re-arm. Nothing in this branch writes to `C:\TradingBotData` differently
from the branch you are on now, and no store schema changed — the new
`any_bounce_watches.json` is created on first use and its absence is normal.

### What I did NOT do

- **No merge to `main`.** As instructed.
- **No live migration or backfill** (R7's trader-present finale) — built and
  tested against fixtures only, still behind their manual actions.
- **No threshold tuning** from any single session, and no live proof claimed
  from a deterministic test.
- **No changes to the desk's scheduled tasks, settings, or `C:\TradingBotData`.**


---

## THE WEEKEND OF 2026-08-15/16 — history, superseded by the report above

**The two stray remote branches are known and deliberately NOT merged
(trader decision, 2026-08-17).** A branch audit that day found exactly two refs
carrying commits absent from the release candidate, and both were ruled out:

- `scoring-flagging-evidence-guardrails` — one commit from 2026-08-03,
  "Tighten setup flags and add evidence boosts" (13 files, 704 insertions,
  `master_avwap` setup scoring/flagging plus a golden fixture). **Ignored by
  explicit trader decision.** It predates the consolidation, has never run
  alongside any of R1-R8, and merging a scoring change into a release candidate
  awaiting live validation would make the validation unreadable - a behaviour
  change could no longer be attributed. The branch is left in place, not
  deleted: ignoring is not discarding, and the work is still reachable if it is
  ever wanted.
- `claude/trading-system-review-e0p8ll` — one doc commit from 2026-08-09,
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

**Branch renamed 2026-08-17: `phase05-r8-weekend-prep` → `testing-week-2026-08-17`.**
Same commits, same SHAs, nothing merged or rebased — only the name moved, and the
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
  └─ testing-week → R1 → R1.1 → R2 (8d25c92) → R7 → R8 → [R3, R4, R5 this weekend]
       = testing-week-2026-08-17          ← the single release candidate
```

**What got built, packet by packet.** 64 commits since the R2 tip; 14 of them in
the final session.

| Packet | State | One-line summary |
|---|---|---|
| **R1 + R1.1** | BUILT, 4 live proofs owed | OFF/DESK/AWAY/EVENING matrix, one fail-open quiet-hours gate over every automatic starter, EVENING SPY wake alarm |
| **R2** | BUILT, 4 live proofs owed | PDH+VWAP Focus adoption gate at build/refresh/adoption, provenance sidecar, scoped "Not today", M5 strength board |
| **R3** | **CLOSED 2026-08-16** | Shadow-only `would_demote` classifier, relvol + daytrade annotation, reviewed-today badge, 12:45 preview slot, post-close tracker write, STABLE+PREVIEW, structured dislike codes. **§4.3.5 volume-thrust deferred by trader decision** |
| **R4** | **BUILT 2026-08-16** | CaptureRail on every chart surface, painted armed alerts, forming-bar honesty fix, reviewed-today markers, labeled Like→Focus, feed repetition + open-burst digest |
| **R5** | **§2 + §5 + §3.1 built; §3.2/§3.3 behind a LIVE gate** | Three pure indicator modules, the one shared completed-bars rule, and the **LRSI cross engine wired live 2026-08-17** with its own `M5_SIGNAL_TAG` family and toggle map. The packaging trigger fired and was discharged: frozen count 51 → 55 |
| **R6** | **(a) BUILT, (b) DECIDED + narrowed, (c) already existed** | AI batch layer now has a System Health row; rotation declined on measurement (see the R6(b) decision row); the stall watchdog was already built and owes only its diagnostic week. Item (d) already resolved into R7 |
| **R7** | BUILT, trader-present finale owed | Tax-grade journal from both brokers, rebuilt Journal tab |
| **R8** | BUILT, one weekend run owed | Guided weekend prep routine, H1/D1/M1 strength boards |

**Gates on the current tip** — all exit 0, all re-run this weekend:

| Check | Result |
|---|---|
| `pytest tests/ -q` | **3638 passed / 19 subtests, 0 failed, 0 errors** (2026-08-18 11:19 PT, quiet-hours window OPEN, hermetic). **Process exit code is `0xC0000409`, not 0** - a native Qt-teardown crash after the summary prints, attributed to importing `ui.stall_watchdog`; see the entry above. The 3638/exit-0 row previously dated 2026-08-18 was written while its confirmation run was in flight: that run did land with 3638 passed, but its exit-code line never wrote, and the code was almost certainly already this crash. **Do not quote exit 0 for this suite until the teardown crash is resolved.** |
| `scripts/smoke_check.py` | **7/7** |
| clean-cache frozen rebuild + selftest | **`selftest OK: 55/55 checks passed (frozen)`** (2026-08-17; 51 → 55 on the R5 roster growth) |

The frozen rebuild deleted `build/` **and** `dist/` first and ran from the
worktree, so the desk's own `dist/` was never touched. Exe mtime **22:08:48**
postdates the last code commit — provenance stated on its face, because a past
round shipped an exe built 21 seconds *before* its tip and an external review
correctly refused it.

**The count moved 49 → 51, and that movement is the point.** The first rebuild
of the evening was taken at `6d81492`, then `7d97904` added `completed_bars.py`
and made `weekend_strength` reach it through a **function-level import** — the
exact shape PyInstaller can follow today and a refactor can quietly break, whose
failure mode is a bundle that starts fine and dies the first time a weekend
board filters a forming bar. `completed_bars` and `alert_repetition` were added
to `selftest.LAZY_ENGINE_MODULES` so the frozen run *proves* they import instead
of inferring it, and the count moving from 49 to 51 is what shows the rebuild
was real rather than a cached reuse.

`indicators.*` is deliberately **not** in the roster: it has no importer
anywhere and is listed in `PACKAGES_NOT_IN_THE_BUNDLE`, so the frozen exe
genuinely does not contain it. When R5's wiring gives it a real importer, that
entry is removed and its modules are added to the roster **in the same commit** —
the two lists must never contradict each other.

**Nothing live was touched all weekend.** No broker call, no journal write, no
desk-branch switch, no `main` push. The desk kept running
`phase05-r2-focus-gating-strength-board` from the main checkout throughout.

**Six trader decisions were taken and are recorded, not re-litigable:**

1. R3 §4.3.5 volume-thrust normalization — **deferred** (no intraday seam; a
   flat-profile proration was offered and rejected).
2. R4 open-burst digest window — **30 minutes**, zero disables.
3. R4 like-to-Focus — **one click**, no reason prompt.
4. R4 escalation list — **exhaustive at three**: higher tier, first BANGER,
   first PROVEN.
5. R5 confluence scope — **M5 Focus members only**.
6. R5 ORB candidate surface — **Alert Center annotation**, not a board lane.

Plus delegated to Fable and recorded (the trader may override any of them):
R5 gets a **new `M5_SIGNAL_TAG` family**, no tier bypass, foldable (spec §8.1);
**R5 §7 holds the WIRING of §3.2/§3.3, not their pure logic** — no wiring into
the live M5 loop even default-OFF until §3.1's desk session, pure
correlator/ORB-classifier code with fixtures may land now, and note nothing in
the UI can flip `m5_signal_toggles` anyway (spec §8.2, 2026-08-17); the
**prior-anchor AVWAP line is carried as an optional top-level `prev_avwape`
key** on the existing zone-arms entry — never a `trigger_levels` arm, absent
when no prior anchor, golden fixture over `build_d1_zone_arms` first, and the
value already exists at `runner.py:747` so no new band computation and no
`master_avwap_lib/legacy.py` edit at all (spec §8.3, 2026-08-17).

**The three held-ask-first items are TRIAGED 2026-08-17 (Fable, delegated) —
none needs a trader question:**

- the Focus Picks reviewed-today marker — **technical, decided: decoration
  only**, never in the document text, save-path byte-identity pinned by test;
  the only path back to the trader is if decoration proves impossible (R4 spec
  header note);
- R4 §2.2's `review_host` for the boards — **CLOSED, no build**: auto-advance
  on a re-ranking board advances to the wrong symbol; reopen only on a trader
  ask for a frozen review-queue mode (R4 spec header note);
- the completed-bars migration — **verified NOT a live bug**: every checked
  site strips the offset only *after* `get_market_local_now()` has converted
  to market time, so naive market-local compares against naive market-local
  and the answers are correct today. Migration stays opportunistic hygiene —
  it rides along with the next authorized `legacy.py` wiring edit behind an
  old-vs-new equivalence pin, and never opens that ask-first file on its own
  (R5 spec §5 note).

**→ Next session: see RESUME HERE in the table below.**

---

## Active work — read this before choosing a task

There may be only one active build item unless `plan.md` explicitly identifies an
elapsed evidence lane that can run in parallel.

| Field | Current value |
|---|---|
| Roadmap phase | **Phase 0.5 — R5 in progress, R6 and the review-deferral completions still to come.** R3 CLOSED and R4 BUILT 2026-08-16. R1 + R1.1 + R2 + R3 + R4 + R7 + R8 built; every live gate remains owed |
| **Active packet** | **R5 M5 signal engines** (`docs/M5_SIGNAL_ENGINES_PLAN.md`). **§2's three pure indicator modules and §5's shared completed-bars helper are BUILT and green.** The lane question that blocked the wiring is **ANSWERED** — spec §8.1: one new `M5_SIGNAL_TAG` family, main feed, **no tier-gate bypass**, not loud by default where the spec does not say, and **not** privileged against R4 §6.3 (foldable and digest-eligible). Per-engine identity rides `bounce_type`, not the tag. §9 carries build state and the packaging rules |
| **RESUME HERE** | **R5 §3.1 (LRSI cross) is WIRED, green and frozen-verified as of 2026-08-17 — see below. The next two engines are blocked on a LIVE gate, not on build effort.** **1. §7's per-engine desk session**: the confluence (§3.2) and first-candle ORB (§3.3) engines wire ONLY after one desk session confirms the LRSI cross's alert volume is sane. Do not wire them from deterministic tests. **2. R5 §4's any-bounce watch** is not behind that gate and can build next, but its prior-anchor AVWAP line is an **ask-first** edit to `master_avwap_lib/legacy.py` (D1 scan output) — ask before touching it. **3. R6 — (a) BUILT 2026-08-17, (c) was already built, (b) DECIDED 2026-08-17 and narrowed to tests/docs.** Rotation is declined on measurement; do **not** re-propose it without a reopen trigger from the decision row below. What R6(b) still owes: **(1) the replay characterization fixture is BUILT 2026-08-17** — `tests/fixtures/technical_integrity_replay_v1.json` + `tests/test_technical_integrity_replay.py`, 18 tests, mutation-proven (session filter removed → 7 fail; provenance strip removed → 3 fail), `scripts/technical_integrity.py` untouched. **(2) the read-only JSONL-ledger audit** via the existing footprint check; **(3) the stale-size comment fix** in `operations_audit.py` (~106 MB was a mid-July docstring, never a measurement — the audit measures live). The fixture and the audit touch tests and docs only; ask-first still binds any `technical_integrity.py` edit. Both stale sizes are resolved: measured **370 MB / 318,040 rows / 25 sessions** on 2026-08-17. **4. The review-deferral completions**: R8 Week-in-Review RRS-extremes join, R8 Focus Review joins (**join picks WITH their outcomes, not as separate rows**), Analytics per-setup/per-account charts with honest n counts and a CSV under each, and the Calendar pyqtgraph year heatmap. **Leave true USD conversion deferred** — the FX table books CAD only |
| **R4 close-out (2026-08-16)** | **BUILT, live proofs owed.** §6.1 armed-alert survival; CaptureRail in the snapshot popup and Alert Center pane (so the RS/RW and Industry boards, which had no capture at all, now inherit it); Alert Center LIKE as capture-not-placement; armed price alerts + D1 level watches painted as a read-only `GROUP_ALERTS` family on the worker; the Yahoo forming-bar early print suppressed 15 min after the open and labeled when drawn; the reviewed-today marker on snapshot/Alert pane/RS-RW/Industry; the labeled `☆ Like → M5 Focus` verb; and one feed row per symbol+side+day with a three-item escalation list and a 30-minute open-burst digest. Three trader confirmations recorded in the spec's §6.4. **Held ask-first:** the Focus Picks marker (editable watchlist *text*, not a table) and §2.2's `review_host` for the boards. **Owed:** the whole §8 exit gate, all live |
| **R3 close-out (2026-08-16)** | **DETERMINISTIC WORK COMPLETE.** The classifier stays shadow-only — `would_demote` stamps, nothing moves, hides or reorders a live row. Built: relvol + `daytrade_candidate` annotation, reviewed-today badge from recorded decisions only, the 12:45 PT preview slot, actual-close ownership of the single scheduled tracker write, STABLE+PREVIEW with `bar_status` stamps, and structured dislike codes counted as `review_learning`'s `dislike_reason` dimension. **§4.3.5 volume-thrust normalization is DEFERRED by explicit trader decision** — the D1 seam has no intraday slot series, a per-symbol 5-min fetch was refused as a data-budget/contract change, and a session-elapsed proration was offered and rejected because real volume is U-shaped. **Owed, live only:** the `would_demote` shadow week (required before any row moves), the one-week 12:45-vs-close and STABLE-vs-PREVIEW churn comparison, and the first real-data curation cycle |
| **Working branch** | **`testing-week-2026-08-17`** — since the 2026-08-15 consolidation this is **THE single release candidate**, carrying testing-week + R1 + R1.1 + R2 + R7 + R8 and every review-repair pass. It was cut from the R7 tip `4420bbf`, which was cut from the R2 tip `8d25c92`; the one R2 commit made after that cut (`fc4bcaf`) is now **merged in**, so `phase05-r2-focus-gating-strength-board` is a proven ancestor (`git merge-base --is-ancestor` = 0). Built in a linked worktree at `..\TradingBotV3-r8`. **The main `C:\Users\Aaron\TradingBotV3` checkout stays on the R2 branch, because the desk's scheduled task runs the desk from it.** Run tests with the main repo's venv python and the worktree as cwd |
| Desk branch | **`phase05-r2-focus-gating-strength-board`** at `fc4bcaf` — what the desk runs and what Monday's live proofs are observed against. It is kept **only until the Monday merge**; do not switch, rename or delete it before the scheduled task is disarmed |
| Scope | R5 §6 fences its files: `bounce_bot_lib/legacy.py`, `chart_watch.py`, `master_avwap_lib/d1_zone_arms.py`, `master_avwap_lib/legacy.py` (prior-anchor output), `alert_center_panel.py`, `bounce_service.py`. Edits outside the files the active packet's spec names are **ask-first**, fixtures first on anything detector/scoring/alert adjacent — the recovered-rule detour proved that pattern works. **Never edit `scripts/strength_scan.py`** |
| State | **3590 passed / 19 subtests, exit 0; smoke 7/7, exit 0; clean-cache frozen rebuild + `selftest OK: 55/55 checks passed (frozen)`, exit 0** (2026-08-17). **The packaging trigger finally fired and moved the count 51 → 55**, which is the outcome the stale-build rule demands: R5 §3.1 gave `indicators` its first real importer, so it left `PACKAGES_NOT_IN_THE_BUNDLE`, entered the spec's `FIRST_PARTY_PACKAGES`, and four modules joined the selftest roster — all in one commit, the two lists still disjoint. `build/` **and** `dist/` were deleted first and the build ran from the worktree, so the desk's own `dist/` was never touched; exe mtime 18:59:35 postdates the commit at 18:57:23. Main desk checkout and live runtime untouched; no live broker call, no live journal write |
| Next action | See **RESUME HERE** above. Do not claim any live proof from deterministic tests |
| Do not start yet | **Phases 1–7 remain NOT authorized.** Do not run R7's live migration/backfill before Monday's validation day passes; do not claim any live proof from deterministic tests |
| **Owed live gates — the full ledger** | Nothing below has been observed. UNKNOWN is a result and `plan.md` sec 6 requires recording it as one. **R1 (4):** a ~21:00 boot that starts nothing; an EVENING day that stops after its early block; an AWAY session staging-not-adopting with a clean post-flip drain; one SPY ±1% alarm. **R2 (4):** one staged pick evicted on a VWAP/PDH fallback; one adoption-time refusal; one scoped "Not today" leaving other entries intact; one strength-board session matching the TC2000 scan's character (re-measure the fetch during market hours). **R3 (3):** the `would_demote` shadow week **before any row moves**; the one-week 12:45-vs-close and STABLE-vs-PREVIEW churn comparison; the first real-data curation cycle. **R5 (1, and it blocks build work):** one desk session confirming the LRSI cross engine's alert volume is sane — §7's gate is per engine, so §3.2's confluence and §3.3's first-candle ORB stay unwired until it runs. **R4:** the whole §8 exit gate, including the two-direction Not-today/armed-watch check. **R6:** the stall-watchdog diagnostic week (the code was already built; only the week is owed). **R7:** the trader-present finale — dry-run review, migration, full backfill, ≥10-trade statement spot-audit, one clean reconciliation week, ≥5 consecutive nightly ledger entries. **R8:** one real weekend run (does **not** wait for Monday — read-only, starts nothing until a button is pressed) |
| **Live sessions 2026-08-17 + 2026-08-18 (merged in from the desk branch)** | Both days ran AWAY open-to-close on `c69b69c`. **R2 eviction PASSED** (four timestamped `Focus gate evicted N staged pick(s)` lines with per-symbol reasons), **R1 quiet boot PASSED with a note** (the `IB: connected` at 22:06:41 is `BouncePanel`'s launch auto-connect, not an Auto Pilot start), **R1 AWAY discipline HALF-PROVEN** (no DESK flip ever happened), and the other five proofs stay **UNKNOWN**. Two defects found and fixed on the desk branch and now merged here: an open report file aborting a whole swing scan (`_write_text_atomic` PermissionError), and one odd yfinance frame blanking the universe rebuild |
| Doc-only addendum (2026-08-15, late) | Phase 0.5 gained packets **R7 (journal reliability + UX)** and **R8 (Weekend Prep)**: specs written, WISHLIST/plan.md/docs README reconciled (incl. the P3.3 nightly-journal-pull promotion into R7 and the P5.4 narrowing). **Markdown-only — the release candidate, gates, and baseline above are unchanged** |
| **R6(b) decision (2026-08-17, delegated)** | Rotation of `technical_integrity_events.jsonl` is **declined for now** — measured 370 MB / 2.2 s boot re-parse, session-filtered replay makes closed sessions inert, and in-place rotation would break the warehouse ingest (SHA + line-offset) watermark; retention stays owned by the locked warehouse plan's after-verified-ingest cleanup, to be built as forward-only per-session segments with the monolith frozen. R6(b) narrows to the replay characterization fixture + read-only ledger audit. Recorded in `plan.md` item 6(b). **Markdown-only — the release candidate, gates, and baseline above are unchanged** |
| **R7 redirect (2026-08-15, second of the day)** | The trader explicitly authorized **R7 code to start now**, ahead of the P0.7 merge: branch **`phase05-r7-journal-reliability-ux` cut from the R2 tip** — same redirect pattern as R1/R2, recorded in `plan.md` Phase 0.5 preamble and the R7 spec header. Rationale: R7/R8 touch journal/weekend surfaces, not the scanning/alerting/Focus path Monday's proofs cover. **The desk keeps running the R2 branch via the scheduled task until the validation day passes — do not switch the desk branch without disarming that task.** R1/R2's eight live proofs remain owed and are inherited by the eventual stack merge. R7's own trader-present steps (live DB migration, full backfill) must NOT run on the desk before Monday's validation passes |
| **R3–R6 weekend redirect (2026-08-15)** | The trader explicitly authorized the remaining packets on this consolidated branch: *"integrate the rest — build R3 through R6 on the consolidated branch."* Build order is R3, R4, R5, R6, with per-packet governance and full-suite pushes. The redirect authorizes code; it does not discharge R3's shadow week or R6's watchdog week |

## R7 build progress — `docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md` §9

Each step is its own green commit, pushed. A step is not done until
`pytest tests/ -q` passes **by its own exit code**.

| §9 step | State | Evidence |
|---|---|---|
| 0 Characterization fixture | **DONE** | `tests/fixtures/journal_rebuild_trades_v1.json` + `tests/test_journal_characterization.py`; 2931 passed, exit 0 |
| 1 Hygiene (A10, B5, A4) | **DONE** | `tests/test_journal_import_hygiene.py` (34 tests); 2965 passed, exit 0 |
| 2 v3 migration + uid migration | **DONE** | `scripts/journal_migrate.py` + `tests/test_journal_migration.py` (26 tests); 2991 passed / smoke 7/7, exit 0 |
| 3 Group-key normalization | **DONE** | `scripts/journal_identity.py` + `tests/test_journal_identity.py` (34 tests); 3025 passed / smoke 7/7, exit 0. Golden regenerated with a note: 10 trades → 9 |
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
| **R8 §9 1-12** | **ALL DONE** | See the repaired R8 release candidate below: 3354 passed, smoke 7/7, frozen 49/49 |
| 14 Governance close-out | **DONE** | Frozen rebuild + `selftest OK: 45/45 (frozen)`, exit 0; CHANGELOG, `docs/README.md`, `WISHLIST.md`, `plan.md`, this file and `docs/DESK_TESTING_PLAN.md` reconciled |
| Pre-flight fix pass (2026-08-16) | **DONE** | Five trader spot-check findings closed; focused journal gate 145 passed; full suite 3375 passed / 19 subtests, all exit 0; live journal/brokers untouched |

### The R8 finale — one weekend, and it does not wait for Monday

Read-only against the trader's data, and the tab starts nothing until a button
is pressed. Spec §10: the desk boots on a weekend with the tab present and no
network activity until a press (log-verified); zero IB traffic across the
routine; H1/D1/M1 each refreshed with its wall clock recorded in the spec's §11;
the monthly board spot-checked for the absence of a current-month bar; one real
Adopt verified in Focus swing, `swinglongs.txt`, the membership file and
`pick_feedback.jsonl` with `origin="weekend_prep"`, and **nothing removed
anywhere**; one auto-tag confirm and one correction; a walk-away windowed to the
reviewed week; the week-ahead rendering only on its button press; the app closed
mid-routine and reopened with progress restored; and the trader confirming board
character per timeframe — until that, §5's filters are approved but not proven.

**R7 and R8 are built and their adversarial-review repairs are complete.** The
earlier R3–R6 hold is superseded by the trader's 2026-08-15 weekend redirect
recorded in the active-work table; R3 is now active. R8's §5 discovery filters
are trader-approved as proposed; the live weekend run still has to prove their
board character.

**The golden fixture is the packet's spine.** It freezes what `rebuild_trades`
does today, six known defects included, and it is regenerated only by
`tests/journal_characterization.py` with the change written into the fixture's
`intentional_difference` field in the same commit. It was verified to fail: a
trial `CLOSED_PARTIAL` status change turned three assertions red, and was
reverted.

**Step 1 finding — the ibapi timestamp gap is latent, not live.** The old parser
did not understand ibapi **10.x**'s `"20260804 09:31:00 US/Eastern"` execution
time and answered `pacific_now()` for it, which would have stamped every socket
fill with the import time. The desk is unaffected today: `constraints.txt` pins
**`ibapi==9.81.1.post1`**, whose `"20260804  09:31:00"` form the old parser did
read. So this is a defect that fires on an ibapi upgrade, not one already in the
live journal — recorded that way rather than as a live data-corruption finding.
Verified by running the pre-fix module directly against both spellings.

**Step 2 changed the golden once, on the record.** Schema v3 adds five columns
to every trade row (`net_pnl_cad`, `fx_rate`, `fx_rate_date`,
`reconcile_status`, `anchor_execution_uid`), all NULL or empty until steps 4, 8
and 9 populate them. **No assembled value moved**: legs, opportunity events and
the summary are byte-identical and every shared trade column matches, verified
column by column before regenerating. The note is in the fixture's
`intentional_difference` field, and the generator now **refuses to write a
changed golden without one**.

**Step 4's narrowing — APPROVED by the trader 2026-08-15, closed.** §5 fix 4
says a missing-opening-fill produces a `SYNTHETIC_OPEN` leg + `NEEDS_REVIEW`.
Built as: **only the unambiguous case is flagged** — a fill that closes more
than the journal knows is open, where the leftover is proof an opening fill is
missing. A plain sell with no position is *genuinely* ambiguous (a real short
entry, or a sale of shares bought before the import window), and nothing in the
execution distinguishes them; flagging every short would make the review queue
noise. That other half is caught by §9 step 9's reconciliation, where the broker
reporting flat against a journal that says short is the proof this step cannot
have. **This is a decided narrowing, not an open item** — do not re-litigate it
or "restore" the broader reading.

**The live journal DB has not been touched.** Everything above ran against
fixture and temporary databases. `journal_migrate.py` defaults to a dry run
against a throwaway copy, and a test asserts the live file is byte-identical
afterwards and that no backup is taken (because nothing changed). The real
migration is a trader-present step and waits for Monday.

### Broker credentials — DONE and live-verified (trader, 2026-08-15)

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

### Tax status — partly decided (trader, 2026-08-15)

For the §9 step 11 labeling UI. The migration seeds `tax_status` from
`account_type` and never overwrites a `trader`-sourced value (I7):

| Account | Status |
|---|---|
| Questrade TFSA **51830546** | `TAX_FREE` |
| Questrade Margin **29347316** | `TAXABLE` |
| IBKR **U4867396** | `TAX_FREE` — TFSA, **currently unfunded and deliberately kept** |
| IBKR **U5102524** | `TAXABLE` — margin |

**All four confirmed by the trader 2026-08-15** and recorded in
`journal_migrate.TRADER_CONFIRMED_TAX_STATUS` as `tax_status_source='trader'`,
because a statement from the person who opened the account is a different kind
of fact from an inference off an account-type string — and only one of them may
never be overwritten (I7).

**U4867396 stays labeled while unfunded.** A zero balance is not zero history,
and an account that drops out of the tax grouping is an account whose past
trades quietly stop being counted.

An account nobody has decided about still stays blank and lands in the account
tree's own "Unlabeled" group. A guessed tax status is a wrong number in a tax
record.

**Deferred out of step 3, deliberately — one spec conflict.** Spec §5 fix 3 puts
"the manual-execution dialog gains real broker/account pickers" in this step,
but that dialog exists **only in the legacy Tk tab** (`scripts/journal_tab.py`),
which spec §7 says stays untouched — and the Qt panel has no manual-entry dialog
at all yet. The data layer already accepts a real broker/account
(`manual_execution_from_fields` honours them), so the missing half is purely
UI and belongs to the Qt Trades tab in **step 11**. Recorded rather than
silently skipped.

### Suite instability seen during R7 — READ BEFORE MONDAY

Two events, neither in a file R7 touches, both recorded because the merge gate
has **no rerun-until-green carve-out** and a 6am reader needs to know these
exist before deciding what a red run means.

| When | What | Reproduced? |
|---|---|---|
| During step 3 | One full run exited **3** — a crash, not a test failure | No. Next run green |
| During step 4 | `tests/test_desk_link_control.py::test_set_auto_mode_intent_round_trip_from_controller` **failed** | No. **1 failure in 10 full-suite runs** on this branch; 3/3 in isolation |

What is known: the Desk Link test drives a **real loopback TCP server** and
polls `_pump_until` against a **20-second wall-clock deadline**. Twenty seconds
is not an ordinary scheduling miss, which makes "just load" an unsatisfying
explanation — something stalled. `tests/conftest.py` already names the likely
family: leaked `bounce_bot_lib.legacy.run_strategy` worker threads that outlive
their tests, and its own honest verdict that "12/12 is a real improvement over
8/10 but it is not a proof of thread safety".

What is **not** known: whether R7 makes it more likely. R7 adds 123 tests and
~17s of runtime, which is more load on a load-sensitive test, so "R7 is
innocent" is a plausible claim and not a proven one. One full run at the R2 tip
was green — one run is not evidence of absence. No R7 file touches Qt, sockets,
or Desk Link.

**Context for Monday's gate decision, not a licence to ignore a failure:**
`tests/test_desk_link_control.py` guards **Desk Link, retired 2026-08-08**
(`CHANGELOG.md`) and kept in-repo only pending the P1.5 cleanup. Nothing the
desk runs today depends on it. That is worth knowing when weighing whether a
red run blocks the merge — it is *not* a reason to re-run until green, and the
flake stays **unattributed**.

**Do not treat either event as a known-flaky exemption on Monday.** P1.1 owns
suite hermeticity; if this recurs, it is worth a bounded investigation before
the merge rather than a re-run.

**Packaging, checked at the step-10 boundary.** The five new top-level modules
(`journal_identity`, `journal_migrate`, `journal_coverage`, `journal_fx`,
`journal_reconcile`) are **modules, not packages**, and every one is statically
reachable from the frozen entry point:
`ui/services/journal_import_service.py` → `journal_runner` → `journal_coverage`
/ `journal_fx` / `journal_reconcile` → `journal_store` → `journal_migrate` /
`journal_identity`. The spec-drift test passes and the **source** selftest
reports 31/31. **No packaging trigger fired**: no new third-party dependency
(`journal_fx` uses `requests`, already pinned), no new non-`.py` runtime asset,
no new top-level package, no dynamic string import, no `__file__`/`ROOT_DIR`
change. The **frozen** rebuild + frozen selftest are still owed before the merge
— CLAUDE.md requires them regardless of triggers, and they are the gate that has
historically caught what the suite could not.

**`ai_jobs` still is not in the frozen bundle**, and the new slot does not change
that: `default_slots()` imports `journal_runner` lazily inside the function, so
the roster/selftest disjointness rule is untouched.

### The R7 finale — trader-present, and all of it after Monday

Nothing below has happened. The build is complete; this is the part that needs
the trader and real data, in this order:

| # | Step | Note |
|---|---|---|
| 1 | **Read the migration dry-run report** — `python scripts/journal_migrate.py` (dry run is the default; it copies the DB to a temp file and leaves the live one byte-identical) | Look at the duplicate collapses and the annotation-orphan count before anything is applied |
| 2 | **Apply the migration** — open Journal and explicitly click **Prepare Journal database** | Runs backup, migration, and rebuild in a background worker. The tabs stay disabled and the status stays visible until it succeeds; this is when the four confirmed tax statuses land |
| 3 | **Full backfill** — Journal ▸ Health ▸ backfill, or `journal_runner --backfill-days 365` | Flex caps at 365 days; older history needs the one-time Flex file import (spec §8) |
| 4 | **Spot-audit ≥10 trades against statements**, then reconcile trade counts and commissions to **one monthly statement per broker, to the cent** | This is the gate that decides whether the journal is tax-grade |
| 5 | **One clean reconciliation week** on both brokers | Every mismatch fixed upstream or explained by an adjustment record |
| 6 | **≥5 consecutive nightly `journal_import` ledger entries** with coverage advancing and at least one observed self-heal | |

**Questrade env-var cleanup**, if `QUESTRADE_REFRESH_TOKEN` is still set: local
settings win, but the env var is a first-boot seed only and a stale copy can be
mistaken for the live rotating token. The Health tab warns when it sees one.

**Nothing in R7's build touched the live journal database.** Every test ran
against fixture and temporary stores; `journal_migrate.py` defaults to a dry run
against a throwaway copy, and a test asserts the live file is byte-identical
afterwards.

## Live sessions 2026-08-17 and 2026-08-18 — what they proved, and two defects

Both days ran **AWAY from open to close**. That is the single fact that shapes
everything below: AWAY exercises staging, eviction, silent alert queueing and the
hourly phone reports, and it exercises **none** of adoption, "Not today", the
strength board, EVENING or the SPY alarm. Those did not fail — their triggering
conditions never occurred, so they stay **UNKNOWN**, which `plan.md` sec 6 counts
as a result.

### Proof results

| Proof | Result | Evidence |
|---|---|---|
| R2 eviction | **PASS** | `Focus gate evicted N staged pick(s)` in `trading_bot.log` / `trading_bot.log.1` on **2026-08-18 at 10:31, 11:40, 12:11 and 12:48**, each with per-symbol reasons — e.g. `Focus gate evicted 6 staged long pick(s): BMRN (not above yesterday's high and not above session VWAP), COO (not above session VWAP), DLTR (not above session VWAP), HLT (not above session VWAP), PAYC (not above session VWAP), SBH (not above session VWAP)`. Refusals at candidate build (`Focus gate refused N long candidate(s)`) appear in the same file and hour |
| R1 quiet boot | **PASS, with a note** (see below) | The 2026-08-16 22:06 launch logged `Auto Pilot is ON from saved state, but nothing starts yet - weekend - quiet hours until the next session` (autopilot.log 22:06:38) and nothing automatic ran until `Automatic work resumed - inside the 06:00-14:00 automatic-work window` at 2026-08-17 06:00:11 |
| R1 AWAY discipline | **HALF-PROVEN** | Two full sessions staged without adopting, and every hourly `Hourly Away swing report verified for HH:00` line is present on both days. **The flip-back-to-DESK half never ran** — the trader never flipped, so the R2.2 post-flip re-measurement is untested |
| R2 adoption refusal | **UNKNOWN** | AWAY never adopts, so `Focus gate refused N staged pick(s) at adoption` cannot appear |
| R2 scoped "Not today" | **UNKNOWN** | Needs an auto-adopted M5 entry; AWAY produced none |
| R2 strength board | **UNKNOWN** | Never opened during a session |
| R1 EVENING stop | **UNKNOWN** | No EVENING day ran |
| R1 SPY wake alarm | **UNKNOWN** | No EVENING day ran |

**No UNKNOWN above was upgraded.** A green suite does not move any of them, and
none may be written as `LIVE_VALIDATED` in `CHANGELOG.md` without its own
preserved evidence.

### The quiet-boot note: `IB: connected` at 22:06:41 — what it actually was

autopilot.log shows `IB: retrying` and then `IB: connected` at 2026-08-16
22:06:41, three seconds after the quiet-hours refusal. Answered from the code:

**It is neither of the two candidates.** It is not an Auto Pilot BounceBot start
— finding #1 of the R1 review has **not** regressed — and it is not a standalone
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
  show it immediately *before* the same IB status pair — the contrast is the
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
  the window gate never enabled it — the same behaviour recorded for the
  2026-08-10 21:19 restart further down this file.

**Recorded, not fixed — the trader decides.** The desk connects to IB on every
launch regardless of hour. That is arguably right (a connection is cheap, and the
trader may want live charts at 22:00) but it contradicts the *wording* of the R1
quiet-hours proof row, which said "no IB connect". That row is now written
against what the build does. Making the panel's launch connect obey quiet hours
is a one-line change at `bounce_panel.py:280` plus a test; it is an R1 behaviour
change, so it waits for direction rather than riding along with a defect repair.

### Defect 1 — a reader holding a report open killed three whole swing scans

**Symptom.** `Swing scan for slot HH:MM FAILED: Master AVWAP scan process exited
with code 1.` on 2026-08-17 at 07:30 and 10:00, and 2026-08-18 at 12:00 (a
tracker-write slot), while neighbouring slots the same days succeeded.

**Root cause, with evidence.** All three run manifests record `"status":
"failed"`, `"error": "PermissionError(13, 'Access is denied')"`, and a phase list
ending at `output/signals` — the next phase, `output/reports`, never completed.
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
text** on the change (`:163` → `ui/services/market_prep_feed.py:90-96`); and
Windows' `open()` does not grant FILE_SHARE_DELETE, so the `os.replace` landing
milliseconds later is denied. Reproduced directly on this desk: a plain read
handle on a destination file makes `os.replace` raise the identical
`[WinError 5] Access is denied`.

**Cost.** The whole scan died — tracker, reports, feature history, scan factors
and state — because one 60 KB report file was being read for a millisecond.

**Fix** (`c69b69c`; the trader approved the `legacy.py` edit before it was made):

1. `_write_text_atomic` and `_write_dataframe_csv_atomic` now replace through
   `_replace_with_retry` — ten attempts a tenth of a second apart. Same doctrine
   as `project_paths.SafeRotatingFileHandler`, which already tolerates a locked
   log file on rollover. A lock that outlives the budget still raises: a report
   that cannot be published must never be reported as published.
2. `ui/services/scan_service.py` lifts the child's own final exception line onto
   the **first** line of the `RuntimeError`, bounded to 240 characters, because
   `_on_scan_failed` writes only `detail.splitlines()[0]` to `autopilot.log`
   (`autopilot_service.py:1144`). The next occurrence reads `... exited with code
   1. PermissionError: [WinError 5] Access is denied: ...` instead of sending the
   reader to the run manifests and a log that may have rotated. No change to
   `autopilot_service.py` was needed — putting the cause on the first line was
   enough.

**Tests** (`tests/test_atomic_publish_under_reader_lock.py`,
`tests/test_scan_service_marker.py`): nine new, every one verified to fail
against the unfixed code, including a Windows-only reproduction that holds a real
read handle on the destination while the write runs.

**Not fixed, deliberately:** the panel still re-reads the report on every JSON
change, so the race can still *start*; the writer now survives it. Removing the
trigger as well is a UI change outside this pass.

### Defect 2 — one odd yfinance frame aborted the universe rebuild

**Symptom.** `Universe rebuild failed: "['datetime'] not in index"` (autopilot.log
2026-08-17 06:00:16). It self-healed on the ~60-minute retry — the universe was
rebuilt at 13:00 the same day — so the visible cost was a stale universe for one
session, which is exactly why it needed a test rather than a watch.

**Root cause.** `scripts/universe_builder.py:329` (pre-fix), the column selection
that ends `fetch_price_history`'s per-symbol loop. yfinance normally names the
daily index `Date`, so `reset_index()` yields a `Date` column the rename turns
into `datetime`; that chunk arrived with an **unnamed** index instead,
`reset_index()` produced `index`, and the selection raised pandas'
`KeyError: "['datetime'] not in index"` — the exact message the log carries. One
malformed sub-frame aborted the entire rebuild, while every other per-symbol
fault in that loop is skipped. The upstream response itself is not recoverable:
`trading_bot.log` has since rotated past 08-17.

**Fix** (`0d355b1`): the date axis is resolved by name (`Date` / `Datetime` /
`index` / `level_0`) and then by dtype; a frame with no usable date column is
skipped and counted rather than fatal, bounded to five warnings plus one total.

**And a floor under that fail-soft.** `build_universe` wrote
`universe_all/longs/shorts` unconditionally, so a fetch outage that priced
nothing would have overwritten a good universe with an empty file. `plan.md`
sec 5 — *a failed publish never destroys the last verified report* — so an empty
screen now raises; the caller already logs and retries in ~60 minutes, and the
previous universe stays authoritative until a rebuild succeeds. Trader approved.

**Tests** (`tests/test_universe_builder.py`): five new. The offending frame shape
was verified to fail against the unfixed code with the identical `KeyError`.

### Release candidate — 2026-08-18

Code changed, so this is a **new** release candidate and all three gates were
re-run against it.

| Check | Result | When |
|---|---|---|
| pytest | **2935 passed / 19 subtests**, exit 0 | 2026-08-18, on `c69b69c` |
| smoke | **7/7**, exit 0 | 2026-08-18, on `c69b69c` |
| frozen rebuild + selftest | **`selftest OK: 31/31 checks passed (frozen)`**, exit 0 | 2026-08-18, on `c69b69c` |

2921 → 2935 is the fourteen new tests; no test was weakened or removed. The
frozen count stays 31: neither fix added a dependency, asset, package or dynamic
import, and the spec-drift test passes.

**Provenance, on its face:** last code commit `c69b69c` at **19:36:04**; last
commit of any kind `f2141c5` (Markdown only) at **20:00:01**;
`dist\TradingBotV3\TradingBotV3.exe` mtime **20:02:47** — the executable
postdates both. `build/` and `dist/` were **deleted before each of the two
builds**, so no cached module could have been reused.

The second build was not ceremony: `docs/DESK_TESTING_PLAN.md` is a **bundled
runtime asset** (the 31st selftest check exists because of it), and the doc pass
changed it after the first build. Rebuilding keeps the packaged Settings ▸
Testing Plan page from rendering a superseded runbook. Both builds returned
`selftest OK: 31/31 checks passed (frozen)`, exit 0; the bundled copy at
`dist/TradingBotV3/_internal/docs/DESK_TESTING_PLAN.md` was confirmed to carry
the 2026-08-18 text.

### Next action — one DESK day and one EVENING night

Neither needs code. Both are written up for a human reader in
`docs/DESK_TESTING_PLAN.md`.

| Day | What it closes |
|---|---|
| One **DESK** session | R2 adoption refusal, scoped "Not today", the strength board's first real look — and the second half of AWAY discipline if the trader spends part of the day in AWAY and flips back |
| One **EVENING** night | EVENING stop (the early block runs, then each refused hourly slot is named once) and the SPY wake alarm |

The SPY alarm does not need a real ±1% day: set `push_evening_spy_alarm_pct` low
for one night to force it, confirm one urgent push with repeats no sooner than
five minutes and silence after flipping out of EVENING, then **restore the
setting** — a forgotten low threshold wakes the trader on an ordinary move.

## Merge safeguards — read before Monday

### Repaired R7/R8 release candidate — code tip `dd201cd`

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

The adversarial review closed all A1–A19 and B1–B14 findings. Weekend board
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

### ~~MERGE NOTE — expected conflict in this file~~ — ABSORBED 2026-08-15

The conflict is gone: `phase05-r2-focus-gating-strength-board` was merged into
this branch on 2026-08-15 rather than left for Monday morning. Monday's merge is
now **one** merge, not three, and it has no known conflict.

`fc4bcaf` (the R2 frozen-gate re-verification) was the only R2 commit outside
R7/R8 ancestry. It touches `CURRENT_CHECKPOINT.md` only; git auto-merged it
without a conflict, and **both** the R2 clean-cache re-verification note and the
R7/R8 sections are present above. The merge's whole contribution to this branch
is **7 inserted Markdown lines in this file** — verified with
`git diff --stat b154b8a HEAD`, no `.py` and no test touched.

**Gates re-run after the merge, in `..\TradingBotV3-r8`:**

| Check | Result |
|---|---|
| pytest | **3354 passed / 19 subtests**, exit 0 |
| smoke | **7/7**, exit 0 |
| frozen rebuild + selftest | **not re-run, deliberately** — the merge added Markdown only, so the frozen gate recorded on code tip `dd201cd` (**49/49 `(frozen)`**, exit 0, from a wiped `build/` and `dist/`) still describes this tree's code exactly |

### Branch consolidation — 2026-08-15

The repository was reduced to **three** branches so Monday has one thing to
merge. Everything deleted was proven fully contained first
(`git merge-base --is-ancestor <branch> testing-week-2026-08-17`); no commit was
lost, and every named rollback SHA is still reachable from this branch.

| Branch | Fate |
|---|---|
| `main` | kept — trunk, tip untouched at `7d85a27` |
| `phase05-r2-focus-gating-strength-board` | kept — the desk branch, until Monday |
| `testing-week-2026-08-17` | kept — **the** consolidated release candidate |
| `testing-week-2026-08-10`, `phase05-r1-auto-modes-quiet-hours`, `phase05-r7-journal-reliability-ux`, `testing`, `chart-review-workspace`, `chart-perf-c`, `integration-test`, `durability-catchup`, `local-ai-phase-0`, `local-ai-phase-1`, `repair-packet-2` | deleted, local and (where it still existed) on `origin` — all contained |

Worktrees `..\TradingBotV3-r7`, `..\TBV3-testing` and `..\TBV3-chart-review`
were confirmed clean and removed. `..\TradingBotV3-r8` and the main checkout are
the only two that remain.

**Three remote-only branches were deliberately NOT deleted** — each still holds
one commit that is in neither `main` nor this branch, so the trader decides:
`origin/scoring-flagging-evidence-guardrails` (`47a3e97` "Tighten setup flags and
add evidence boosts" — the only one of the three carrying code),
`origin/claude/trading-system-review-e0p8ll` (`18c9c93`) and
`origin/claude/wishlist-integration-analysis-2ixvy0` (`671ee57`). Two further
remote branches, `origin/claude/testing-production-blockers-oek3aj` and
`origin/claude/ticker-briefs-hardening-imcm8r`, **are** proven contained but
their deletion was refused by the tooling; they are safe to delete from the
GitHub UI at any time.

### R7 release candidate — `fe4fe73`

**`phase05-r7-journal-reliability-ux` at `fe4fe73` is a named release candidate**,
verified by all three gates on the tree that produced it:

| Check | Result | Command |
|---|---|---|
| pytest | **3203 passed / 19 subtests**, exit 0 | `.venv\Scripts\python.exe -m pytest tests/ -q` |
| smoke | **7/7**, exit 0 | `.venv\Scripts\python.exe scripts/smoke_check.py` |
| frozen rebuild + selftest | **`selftest OK: 45/45 checks passed (frozen)`**, exit 0 | `pyinstaller .\packaging\tradingbotv3.spec --noconfirm` then `dist\TradingBotV3\TradingBotV3.exe --selftest` |

**Rollback for R7 alone: `3339dd9`** — the step-10 tip, the last commit before
the Journal UI was rebuilt. Everything earlier in the stack keeps its own
rollback points in the table further down; the R7 branch is a strict superset of
the R2 tip `8d25c92`, so backing R7 out entirely is a checkout of that.

**The frozen build was made from the worktree** (`..\TradingBotV3-r7`), so its
`dist/` is the worktree's and the desk's own `dist/` — the R2 release candidate
it has been running — was never touched.

#### What the frozen run caught, and it is not nothing

Three rebuilds were needed. The first two reported **31/31**, the pre-existing
roster, passing with R7 code in the bundle. Extending
`selftest.LAZY_ENGINE_MODULES` by fourteen journal modules **did not change the
frozen count** until `build/` was deleted — a PyInstaller rebuild had silently
reused the cached module. That is exactly the failure shape that let "frozen
selftest 30/30" be recorded three times during R1/R2 for runs that never
happened. **Treat a frozen count that does not move after a roster change as a
stale build, not as a passing gate.** The clean rebuild reports 45/45, and `ui`
collects **117** submodules against 109 before — the new `ui/panels/journal/`
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
re-run and this whole section is updated — a stale line here is worse than
none.**

| Check | Result | When |
|---|---|---|
| pytest | **2921 passed / 19 subtests**, exit 0 | 2026-08-15, after R2.3 |
| smoke | **7/7**, exit 0 | 2026-08-15, after R2.3 |
| frozen rebuild + selftest | **`selftest OK: 31/31 checks passed (frozen)`**, exit 0 | 2026-08-15, after R2.3 |

**Provenance, on its face:** last code commit `90ba0d4` at **13:11:19 PT**;
`dist\TradingBotV3\TradingBotV3.exe` mtime **13:13:54 PT** — the executable
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
gates were re-run against it — including the frozen rebuild, even though no
packaging trigger applied. The frozen one is never optional: it is the gate that
caught the `ai_jobs` roster clash and the `-c` scan-spawn defect when the suite
could not.

### Rollback points

**Read this first: the branch names below no longer exist.** The 2026-08-15
consolidation deleted them, but every SHA is still reachable from
`testing-week-2026-08-17`, so each remains a plain `git checkout <sha>` — a
detached checkout, not a revert. Nothing here got harder to roll back; the names
just stopped being branch heads.

| Point | SHA | What it is |
|---|---|---|
| Pre-everything | **`7d85a27`** | `main`. Last known-good merged trunk |
| Pre-R1 | **`e18757e`** | Former tip of `testing-week-2026-08-10`. The build that ran the desk before any Phase 0.5 work |
| Pre-R2 | `4389961` | Former tip of R1+R1.1, if only R2 needs backing out |
| Pre-R7 | `8d25c92` | The R2 tip R7 was cut from, if R7+R8 need backing out but R1/R2 do not |
| Pre-R8 | `4420bbf` | Former tip of `phase05-r7-journal-reliability-ux`, if only R8 needs backing out |
| Desk build | `fc4bcaf` | Tip of `phase05-r2-focus-gating-strength-board` — the build the desk runs and Monday's proofs are observed against. Still a live branch until the merge |

Ancestry is a single line with one merge at the end —
`main` → `testing-week` → R1 → R2 (`8d25c92`) → R7 → R8, then `fc4bcaf` merged
in — so every row above is an ancestor of the consolidated tip.

**The rolled-back build reports `selftest OK: 30/30`, not 31/31, and that is
correct** — the 31st check is the one bundling `docs/DESK_TESTING_PLAN.md`, which
did not exist at `e18757e`. `docs/DESK_TESTING_PLAN.md` §3.4 now says so in plain
language, because a 6am reader watching the count drop would otherwise read a
successful rollback as a broken one.

### Rollback drill — EXECUTED 2026-08-15

Run once, unattended, with no desk process running:

| Step | Result |
|---|---|
| Disarm `TradingBotV3 0700 Launch` | `Ready` → `Disabled` |
| Check out the pre-R1 rollback SHA `e18757e` | clean, no conflicts |
| Verify the rolled-back build starts | `selftest OK: 30/30 checks passed`, exit 0 (30 not 31: the testing-plan check did not exist at that SHA — the count moving is *correct*) |
| Return to the release candidate | back at `bf1ab89`, `selftest OK: 31/31` |
| Re-arm the launch task | `Disabled` → `Ready` |

All three TradingBotV3 tasks confirmed `Ready` afterwards (`0700 Launch`,
`AI Jobs`, `Push cold data to DAS`).

**What the drill did NOT prove:** a full GUI launch. The selftest is the
designed proxy — it imports every lazily-loaded engine and loads every
`__file__`-relative asset with no window and no network — but it is not a
double-click. If the trader wants that certainty before Monday, one manual
launch at `e18757e` is the missing step; the mechanical path around it is
proven.

**The order matters and is the point:** disarm first. The launch task starts
the desk from source, so checking out another SHA while it is armed can have
the task launch a half-swapped tree.

### Live proofs are UNKNOWN until observed

Nothing in the tables below has been run on a live session. They are
**UNKNOWN**, and UNKNOWN is a result — `plan.md` sec 6 requires recording it as
such. A green test suite does not upgrade any of them, and none may be written
as PASS in `CHANGELOG.md` without preserved real-session evidence.

## Monday sequence — 2026-08-17

Do these in order. **Nothing merges until (a) and (b) both pass.**

**The trader can read all of this on the desk**: Settings ▸ Testing Plan renders
`docs/DESK_TESTING_PLAN.md`, a plain-language version of the same sequence. That
file restates the proofs below for a human reader and **must be updated in the
same pass whenever they change**.

### (a) Run the live proofs on THIS build, during the real session

Both packets' proof tables are below — four for R1, four for R2. They are written
against the finished build, not against what either packet did mid-flight; the
AWAY proof in particular changed when R2 landed.

Two are already actionable outside the session: the R1 quiet-boot proof (a ~21:00
launch, which the trader is running the evening of 2026-08-15) and the R2 "Not
today" proof (needs an auto-adopted M5 entry, so it needs a session first).

Record every result, including UNKNOWNs, without rewriting the outcome
(`plan.md` sec 6).

### (b) Run the plan.md sec 6 first-session checklist

`docs/FIRST_SESSION_CHECKLIST.md`, which already carries the four R1 rows added
2026-08-15. It has **no R2 rows** — use the R2 proof table below alongside it
rather than assuming the checklist covers this build.

### (c) Only if both pass: P0.7 merges **one** branch into `main`

The 2026-08-15 consolidation replaced the three-branch ladder with a single
merge. There is no order to get wrong and no known conflict:

```
testing-week-2026-08-17  ->  main
```

That one branch carries testing-week + R1 + R1.1 + R2 + R7 + R8 and every
review-repair pass. The old ladder (`testing-week` → R1 → R2, each merged
separately) is gone along with those branch names; per-packet rollback is
preserved by SHA in the rollback-points table instead.

**Then, in this order — the desk is not switched until the gates pass on `main`:**

| # | Step | Note |
|---|---|---|
| 1 | Merge `testing-week-2026-08-17` into `main` | one merge, no expected conflict |
| 2 | Re-run **all** gates on `main`, including a **clean-cache** frozen rebuild | delete `build/` **and** `dist/` first — R7's close-out proved a rebuild silently reuses cached modules, and a frozen count that does not move after a roster change is a stale build, not a pass |
| 3 | Disarm `TradingBotV3 0700 Launch` | **before** touching the checkout — the task starts the desk from source and can launch a half-swapped tree |
| 4 | Switch the desk checkout to `main` | this is when `phase05-r2-focus-gating-strength-board` stops being needed |
| 5 | Re-arm `TradingBotV3 0700 Launch` | confirm all three tasks read `Ready` (`0700 Launch`, `AI Jobs`, `Push cold data to DAS`) |

**Gates to re-run at merge time, on `main` after the merge:**

| Gate | Command | Expected |
|---|---|---|
| Full suite | `.venv\Scripts\python.exe -m pytest tests/ -q` | **3370 passed / 19 subtests**, exit 0 — check pytest's own exit code, not a piped tail |
| Smoke | `.venv\Scripts\python.exe scripts/smoke_check.py` | **7/7**, exit 0 |
| Frozen rebuild | delete `build/` and `dist/`, then `.venv\Scripts\pyinstaller.exe .\packaging\tradingbotv3.spec --noconfirm` | exit 0, ~4 min unattended. **Required on `main` regardless of triggers** |
| Frozen selftest | `dist\TradingBotV3\TradingBotV3.exe --selftest` | **49/49**, exit 0, output ending `(frozen)` |

**R7 and R8's own live gates come AFTER this merge, not before it.** Nothing in
step (a) or (b) exercises them, and none of them is a merge blocker:

- **R8** — one real weekend run (spec §10). This one does not have to wait for
  Monday at all: it is read-only against the trader's data and starts nothing
  until a button is pressed.
- **R7** — the trader-present sequence in "The R7 finale" above, in order: read
  the migration dry-run, click **Prepare Journal database**, full backfill,
  the ≥10-trade statement spot-audit, one clean reconciliation week, and ≥5
  consecutive nightly `journal_import` entries. **None of it may start before
  Monday's validation day passes.**

**Subsequently authorized by the trader's 2026-08-15 weekend redirect:** R3,
R4, R5 and R6 now build on this consolidated branch in that order; the
active-work table owns their current state. Phases 1–7 remain open and are not
authorized this session.

**Is a packaging trigger pending? No — but rebuild anyway.** Checked all five
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
> predated all three packets. **Resolved the same day — see the frozen rebuild
> below.** Re-run it at merge time only if code lands after that rebuild: this is
> the gate that has historically caught what the suite could not, finding the
> `ai_jobs` roster clash on 2026-08-09 and the `-c` scan-spawn defect on
> 2026-08-13.

### Frozen rebuild and REAL frozen selftest — 2026-08-15

Five rebuilds, all green. The first was the run three packets of notes had
mislabeled; the second was forced by the testing-plan asset; the third was the
R2.1 release candidate `bf1ab89`; the fourth was the R2.2 tip — built 21
seconds before its final commit, which the external review correctly refused as
provenance; the fifth is the current R2.3 candidate, built after `90ba0d4`.

| # | Time | Result |
|---|---|---|
| 1 | 09:58 | `selftest OK: **30/30** checks passed **(frozen)**`, exit 0 |
| 2 | 10:27 | `selftest OK: **31/31** checks passed **(frozen)**`, exit 0 |
| 3 | 11:0x | `selftest OK: **31/31** checks passed **(frozen)**`, exit 0 — on `bf1ab89` |
| 4 | 13:0x | `selftest OK: **31/31** checks passed **(frozen)**`, exit 0 — superseded: exe predated its tip by 21 s |
| 5 | 13:13 | `selftest OK: **31/31** checks passed **(frozen)**`, exit 0 — **current, after code commit `90ba0d4` (13:11:19)** |

Rebuilds 4 and 5 were run **without a packaging trigger**, because a code commit
makes a new release candidate and CLAUDE.md requires a rebuild before merging to
`main`. The count is unchanged at 31, which is the expected result: neither R2.2
nor R2.3 added a dependency, asset, package or dynamic import.

**31, not 30, and that is the point.** The Testing Plan tab renders
`docs/DESK_TESTING_PLAN.md`, a runtime asset that lives **outside `scripts/`**.
The spec's package-asset sweep only mirrors files inside `FIRST_PARTY_PACKAGES`,
and `test_packaging_spec_drift.py` only walks `scripts/` — so **neither would
have noticed it going missing**, and the frozen desk would have shipped showing
"plan file not found" on the one page the trader opens when nothing else is
behaving. Three things now guard it: an explicit `datas` rule with a hard
`SystemExit` if the file is absent at build time, a new selftest asset check
(the 31st), and a test asserting the spec rule still exists. Confirmed present
in the bundle at `dist/TradingBotV3/_internal/docs/DESK_TESTING_PLAN.md`.

That trigger is trigger 2 in the CLAUDE.md list ("new non-`.py` runtime asset"),
plus trigger 5 (`__file__`-relative resolution — the view resolves through
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
modules are in the bundle and import cleanly under it — which is what this run
was needed to prove and no packaging-trigger analysis could.

The desk was running from source, so nothing had to be closed — and no desk
process was running at all for rebuild 4. `dist/` and `build/` are gitignored, so
this is verification only and never a commit artifact.

**This satisfies the frozen gate for the current tree** (rebuild 4, on the R2.2
tip). Re-run it at merge time only if code lands after that.

### ~~Known blocker for the merge gate~~ — FIXED 2026-08-15

`tests/test_warehouse_seal.py::test_stale_staged_files_are_quarantined_not_deleted`
no longer fails intermittently, and the merge gate has **no rerun-until-green
carve-out**. Any test failure on Monday is a real failure.

It was never flakiness. `reconcile` compared `st_mtime > cutoff` where
`cutoff = utc_now() - grace`, and Windows' system clock ticks about every
15.6 ms while NTFS stamps mtimes far more finely — so `utc_now()` could round
BELOW the mtime of a file written microseconds earlier, and that file read as
"from the future" and was never quarantined. The earlier "timing-sensitive
under suite load" note was wrong: load was never the variable, and it
reproduced in isolation at 3 failures in 6 runs.

Fixed in `store.py` with a 50 ms clock-granularity slack (trader-approved
before the edit; recorded as a warehouse build decision). Verified by 20
consecutive passes of the previously flaky test plus a new deterministic
reproducer that writes and reconciles back to back 25 times.

### R2.2 review pass — 2026-08-15 (four items from the final external review)

Four items, each its own green commit, plus one refinement of item 1 found while
reviewing it. Two changed behaviour, one is documentation with a test that keeps
it honest, one reconciled the desk runbook.

| # | What | Where |
|---|---|---|
| 1 | **The flip drain is explicitly locked.** The AWAY/EVENING → DESK flip records its own moment; adoption refuses any verdict stamped before it (`pending_pick_gate_ok(..., not_before=)`). A failed re-verification now retries every 60 s, five times, instead of falling through to the ordinary stored-verdict drain — the 2-bar lag bound is defense in depth, no longer the only lock. Giving up after five is safe because the barrier holds and the 30-minute staging refresh stamps post-flip verdicts. A follow-up commit closed the DESK → AWAY → DESK mid-flight case: an attempt remembers which flip it answers, so a newer return is owed its own measurement rather than inheriting one whose bars predate it | `alert_center_panel.py`, `autopilot_core.py`, spec §11.1 |
| 2 | **One 14:00 boundary.** `auto_scanning_due` used an inclusive datetime endpoint, `_auto_work_due`'s fallback used `hour < 14`; at 14:00:00.000000 they disagreed. Both now call `within_auto_scanning_window` over `auto_quiet_hours_fallback_window`, inclusive at both ends. Test pins the exact microsecond at both call sites and was verified to fail against the old spelling | `autopilot_core.py`, `autopilot_service.py`, R1 spec §4 |
| 3 | **The two-bar tolerance is recorded as an accepted exposure**, with its backstop named: BounceBot's four-close triple-VWAP invalidation plus the desync repair removes a bad adoption within ~4 completed bars. A test pins both constants so the documented bound cannot quietly stop being true. No behaviour changed | `autopilot_core.py` comment, spec §11.2 |
| 4 | **The runbook stopped contradicting this file.** It claimed 31/31 at 09:58 where this file says 30/30 — the checkpoint was right, provable from the build: the only selftest change since `e18757e` is the testing-plan asset check added at 10:38, so the runbook was claiming its own bundling was verified before the file existed. Also removed its stale "known flaky test, just re-run it" carve-out and added the rollback section with the 30/30 explanation | `docs/DESK_TESTING_PLAN.md` |

**Not done, and deliberately:** item 3 offered `max_bar_lag = 1` as an
alternative. The trader's note left that as their call, so the accepted-exposure
documentation was built as written and the constant is unchanged. Switching it
later is a one-line change plus the golden-fixture update.

### R2 live proofs — one PASSED 2026-08-18, three still owed

From `docs/M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md` §8. **Eviction PASSED on
2026-08-18** with the log lines quoted in the 2026-08-18 section above; the other
three need a DESK day, which AWAY could not provide:

| Proof | What to look for |
|---|---|
| Eviction — **PASSED 2026-08-18** | One staged pick evicted for falling back through VWAP or the previous-day extreme: `Focus gate evicted N staged long pick(s): SYM (not above session VWAP)` in the Auto Pilot log. Silent on the desk by design — the log is the record |
| Adoption refusal | One pick refused at adoption, in `trading_bot.log`: `Focus gate refused N staged pick(s) at adoption`. A verdict older than 45 min reads `gate check is NN min old` |
| Scoped "Not today" | On an auto-adopted M5 entry the button reads `✕ Not today - drop pick` and removes only that entry; the trader's own picks, the swing list and the other side are all still there afterwards. On a name the trader typed the button keeps its old feed-only wording and Focus is untouched |
| Strength board | A board session the trader confirms matches the TC2000 scan's character (~20–40/side). **Re-measure the fetch during market hours** — §10's 27.6 s was taken on a Saturday and is a floor, not a worst case. Decide the RVOL column then; it is specified but deliberately not built |

**Deferred deliberately:** RVOL for the surviving ~20–40 rows a side. Specified
in §9, not built — the trader decides on the first live board session whether
they miss it, and the fetch is cheap only at survivor scale.

### R1 live proofs — one PASSED, one HALF-PROVEN, two owed

**Quiet hours PASSED** on the 2026-08-16 22:06 boot (with the `IB: connected`
note resolved in the 2026-08-18 section above). **AWAY discipline is
HALF-PROVEN**: staging without adoption held for two full sessions, but the
flip back to DESK never happened. EVENING stop and the SPY alarm need an
EVENING night. Each is one observation on the desk:

| Proof | What to look for |
|---|---|
| Quiet hours — **PASSED 2026-08-16** | Launch at ~21:00 on a weekday with Auto left ON. `autopilot.log` says `Auto Pilot is ON from saved state, but nothing starts yet`; **no `Starting BounceBot` line**, no universe rebuild, no self-arm. A manual scan from the same desk still runs. **`IB: connected` on its own is expected and is not a failure** — `BouncePanel` connects on every launch at any hour (`bounce_panel.py:280`), outside Auto Pilot; the Auto Pilot start is the announced one |
| EVENING stop | An EVENING day: the open+30 slot and the 07:00/07:15/07:30 checks run, then one `Evening mode: swing slot(s) … not run` line per refused hourly slot and no further scan. The after-close wrap-up still fires |
| AWAY discipline — **HALF-PROVEN 2026-08-17/18** | An AWAY session: picks do not reach `longs.txt`/`shorts.txt`, alerts arrive silently while the feed and D1 badge fill, and the flip back to DESK adopts **only picks re-measured since the flip** — R2 changed this proof and R2.2 tightened it, so anything staged hours ago and no longer qualifying is refused rather than adopted. If the re-check itself fails, the status line says `Retrying in 60s` and **nothing adopts** — that is also a pass |
| SPY wake alarm | One real ±1% EVENING day, or force it by setting `push_evening_spy_alarm_pct` low: an urgent push, a repeat no sooner than 5 minutes, and silence after flipping out of EVENING |

**~~Known limitation, deliberate~~ — CLOSED by R2 (2026-08-15).** The
AWAY/EVENING→DESK drain no longer adopts an un-revalidated backlog: every staged
pick carries a gate verdict from the most recent 30-minute refresh, and adoption
refuses anything failing, missing, or older than 45 minutes. The AWAY live proof
below is written against that behaviour, not the R1 behaviour it replaced.

### R1 build review — 2026-08-15 (independent five-dimension review; findings code-verified)

**All five findings are FIXED as of the R1.1 pass below.** The list is kept
because the defects are the useful record, not the fact that they closed.

Overall: the architecture is right, fail-open holds at every consumer, the manual
carve-outs are real, the alarm's dedupe/day-roll/restart mechanics are solid, the
shared-scan parity claim is proven against the base commit, no existing test was
weakened, and CLAUDE.md/AGENTS.md are byte-identical. But an **R1.1 fix pass is
required before the live proofs are attempted and before R2 stacks on top** —
the following were verified against the code, not just claimed:

1. **BLOCKER — the boot gate is defeated by the tick.** `_tick` calls
   `self._ensure_bot_running()` ungated (`autopilot_service.py:450`), so a 21:00
   boot with Auto left ON logs "nothing starts yet" and then connects BounceBot
   to IB 30 seconds later. Live proof #1 above will fail as written; every doc
   stating "no IB connect until the window opens" currently describes behavior
   the code does not have. The suite stayed green because the boot test stops
   the timer before a tick can run — the fix needs a test that runs a tick.
2. **BLOCKER — the EVENING SPY alarm fires on YESTERDAY's move pre-open.**
   `_maybe_push_spy_alarm` (`autopilot_service.py:1869-1872`) trusts
   `_spy_session_bars(cached_only=True)` with no bar-date check, and its only
   session gate is the quiet window, which opens 30 minutes before the open. On
   any EVENING morning after a ±1% day, ~7 false urgent wake-ups fire on stale
   data before the first new-session bar (all night if quiet hours are disabled).
   Fix at the data read: refuse a series whose last bar predates `now.date()`.
   Every alarm test stubs `_spy_session_bars`; add one with stale-dated bars.
3. **IMPORTANT — a post-14:00 relaunch silently cancels the after-close
   wrap-up.** The quiet refusal in `_maybe_run_swing_slot`
   (`autopilot_service.py:953-955`) returns before any slot resolution, so slots
   still pending after 14:00 (crash or sleep before the close slot — a 4h39m
   sleep happened on this desk 2026-08-11) stay pending forever and
   `after_close_wrapup_due` never fires that day. Same rationale as the EVENING
   marked-done decision; apply it on the post-window side.
4. **IMPORTANT — EVENING picks still adopt into M5 Focus immediately.**
   `_poll_auto_pick_pending` refuses only AWAY
   (`alert_center_panel.py:1612`); the spec §1/§3.3, CLAUDE.md matrix, EVENING
   runbook, and CHANGELOG all state EVENING stages until the DESK flip. Make the
   code match the documented rule.
5. **IMPORTANT — the legacy Tk GUI dies at construction.** `gui.py:1040` still
   calls `get_shared_watchlist_paths`, which the removal deleted from
   `legacy.py`'s import block; `gui.py` acquires its globals from `legacy`, so
   construction raises NameError. One-line import fix. Invisible to the suite
   (tests import but never construct) and to the import-only frozen selftest.

### R1.1 repair pass — 2026-08-15 (all five findings closed)

| # | Fix | Proof |
|---|---|---|
| 1 | Quiet hours moved **into** `_ensure_bot_running`, the one place automation starts the bot; `force=True` is the manual carve-out and `force_reconnect` passes it | `test_the_tick_cannot_undo_the_boot_refusal` runs a real tick with the clock frozen to a weekday 21:00; `test_the_reconnect_button_starts_the_bot_at_any_hour` |
| 2 | The alarm refuses a SPY series whose last bar predates the day being asked about — stale cache is not a move | `test_yesterdays_cached_move_never_wakes_the_trader` (and the same +3% once today's tape prints it still fires) |
| 3 | `_resolve_slots_after_window` marks still-pending slots done once the window closes, so the after-close wrap-up survives a crash or a long sleep. Before the window opens nothing is resolved | `test_slots_left_pending_past_the_window_are_resolved` |
| 4 | `_poll_auto_pick_pending` refuses `("AWAY", "EVENING")`; EVENING also stops beeping, closing the spec §1 alert cell | `test_away_and_evening_refuse_to_adopt_staged_picks`, `test_evening_queues_alerts_without_a_sound` |
| 5 | `gui.py` uses `LONGS_FILE, SHORTS_FILE` instead of the deleted helper | New `tests/test_module_globals_resolve.py` statically resolves every global four never-constructed legacy modules read — verified to fail on the un-fixed file before the fix went back in |

Hardening taken in the same pass: NaN threshold guard on the alarm; the
quiet-window ⊇ sweep-window containment is now **structural** (`auto_scanning_window`
widens itself to contain `bouncebot_scan_window`, so two independent settings keys
cannot contradict each other); `autopilot_auto_arm_due` takes `quiet_hours` and the
arm test pins it, so a desk with quiet hours disabled no longer turns that test red;
`MainWindow._self_heal_universe`'s gate and the D1-feed beep site now have coverage;
the Qt tests **skip** instead of silently passing without PySide6; the false
"an early close moves this window" docstring claim is corrected (no early-close
modelling exists anywhere — pre-existing, and fail-open since the window is only
ever too long).

**Baseline after R1.1: 2785 passed / 19 subtests / smoke 7/7 / source selftest
30/30**, all exit 0. (Recorded at the time as "frozen"; it was the source run —
`launch_gui.py --selftest`, whose output carries no `(frozen)` suffix.)

Still owed, recorded not fixed: a corrupt `local_settings.json` silently re-homes
the store to `%LOCALAPPDATA%` (wants one loud stderr line plus atomic settings
writes); and the spec §1 EVENING **sweep** cell is now explicitly unresolved in
that spec's new §9 rather than silently unbuilt — the recommendation there is to
leave the sweep running, and the trader decides before the EVENING live proof is
recorded as passed.

Original hardening list from the review, for reference: NaN threshold
bypasses the alarm's threshold test (guard `threshold != threshold` like
`day_pct`); the quiet-window⊇sweep-window containment is enforced nowhere at
runtime (two independent settings keys; clamp or log the contradiction);
`test_autopilot_auto_arm_due_daily_hands_off_rules` reads the machine-local
`qt_auto_quiet_hours` setting and goes red on any desk that disables quiet hours
(pin `quiet_hours=True`); `MainWindow._self_heal_universe`'s gate and the D1-feed
beep site have zero coverage; five Qt tests silently pass (not skip) without
PySide6; the spec §1 matrix retains two EVENING cells (sweep "then quiet",
alerts "queue") the build never implemented and §8 never settled — reconcile or
build; a corrupt `local_settings.json` still silently re-homes the store to
`%LOCALAPPDATA%` (one loud stderr line + atomic settings writes); the
"early close moves this window" docstring/CHANGELOG claim is false —
`get_market_session_window` hardcodes regular hours (pre-existing, fail-open).

### Previous packet — ticker-briefs hardening (TB-0..TB-6)

| Field | Value |
|---|---|
| State | **Integrated and green on `testing-week-2026-08-10`**. **Live proof still owed: the 2026-08-12 22:00 window.** The 08-11 night proved TB-0, broke on TB-3, and exposed a task time limit that defeated its own concurrency guard plus 4h39m of machine sleep |
| Side item landed | **Snapshot popup opens at desk height** (2026-08-11) — UI geometry only |
| Side item landed | **Phone push policy + two richer pushes** (2026-08-11) — AWAY became the only pushing mode; R1 has since added EVENING's SPY alarm as the second exception |

A newly arriving AI resumes the active packet if it is unfinished. If it is complete,
it performs the stated next action. It does not select a different roadmap item
without explicit trader direction.

## Planning pass — 2026-08-15 (documentation only)

**Superseded the same day**: the trader then directed R1 to be built, and it was.
See the active-work table above. This section is kept for the recon findings it
records, which are still the current understanding.

The trader promoted the 2026-08-14 `WISHLIST.md` entries and directed a build
foundation for the next implementer. Recorded in this pass:

- **`plan.md` Phase 0.5 (R1–R6)** inserted with the trader's ranked order
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
  `docs/SWING_QUALITY_AND_FEEDBACK_PLAN.md` §4. No fix is built.
- Verification: Markdown-only pass — link resolution, `git diff --check`,
  control-document consistency. The recorded automated baseline (2738 passed /
  19 subtests / smoke 7/7 / source selftest 30/30) is **unchanged**.
- Housekeeping note: untracked `desk_report.xml` at the repo root is generated
  pytest JUnit output from the 2026-08-09 desk gate — left untracked; P1.5 owns
  gitignoring desk JUnit artifacts.

The active build item above (P0 live gates) is unchanged; Phase 0.5 code starts
only after P0.7 merges.

## Branch

Three branches exist, and that is the whole list (2026-08-15 consolidation):

| Branch | Tip | Role |
|---|---|---|
| `main` | `7d85a27` | trunk. Tip untouched; nothing is merged into it until Monday |
| `phase05-r2-focus-gating-strength-board` | `fc4bcaf` | **the desk branch** — what the scheduled task runs and what Monday's live proofs are observed against. Retired at merge step 4 |
| `testing-week-2026-08-17` | consolidated tip | **the release candidate** — testing-week + R1 + R1.1 + R2 + R7 + R8 + all review repairs. Worked in `..\TradingBotV3-r8` |

- State: **nothing merged to `main`; no PR recorded.**
- The consolidated branch is a strict superset of the desk branch (proven with
  `git merge-base --is-ancestor`), so the desk's source-run scheduled tasks are
  unaffected by the merge itself. The standing rule still holds: **disarm the
  scheduled task before switching branches on the desk.**
- Merge only after a `plan.md` Section 6 day passes — see the Monday sequence
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

- `07395a0` — Chart Review Setups column defaults hidden and can be restored.
- `bfc8850` — a late-opened alert receives current bars.
- `4907b6f` — a published best-swing report can notify the phone.
- `1f41af1` — the swing push stays quiet when no readable setups exist.
- documentation consolidation: `CHANGELOG.md` for implemented history, `plan.md` for
  remaining work, `docs/README.md` for classification, and the renamed
  `CURRENT_CHECKPOINT.md` for active state;
- mandatory AI read/update workflow in `CLAUDE.md`/`AGENTS.md`, phase-gated roadmap
  ordering, and the new non-authoritative `WISHLIST.md`.

The documentation packet does not change the recorded automated baseline. Markdown
verification consists of link resolution, `git diff --check`, control-document
consistency, and confirmation that tracked edits remain Markdown-only.

## Re-baseline and desk configuration — 2026-08-10 (evening)

**P0.1 is satisfied for the four post-gate commits above.** Full Windows run on the
working tree:

| Check | Result |
|---|---|
| pytest | **2647 passed, 7 subtests passed**, exit 0 (109s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | not re-run — no packaging-trigger change since 29/29 |

Re-run after the decision-0015 documentation/comment pass: **2647 passed**,
**smoke 7/7**, unchanged. That pass edited Markdown, docstrings, comments, and two
user-facing strings only; no behavior, path, or test changed.

**Current baseline after the local-AI evidence-budget packet:**

| Check | Result |
|---|---|
| pytest | **2659 passed, 19 subtests passed**, exit 0 (104s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | not re-run — no packaging trigger (no new package, no new runtime asset, no new dependency) |

Twelve new tests cover the budget resolver and its fallbacks, the cloud ceiling
staying untouched, the derivation itself (worst-case retry prompt must fit the
context left after generation), the truncation tripwire firing/staying silent, and
ledger usage recording.

**Current baseline after the BounceBot scan-window packet (2026-08-10, late):**

| Check | Result |
|---|---|
| pytest | **2672 passed, 19 subtests passed**, exit 0 (104s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | not re-run — no packaging trigger (no new package, no new runtime asset, no new dependency) |

Thirteen new tests in `tests/test_bouncebot_scan_window.py` cover the window bounds,
the overnight and weekend refusals, the settings escape hatch and margin fallbacks,
and the four service transitions that matter: the close pauses a running sweep, an
after-hours start pauses on its first tick without needing a boundary crossing, a
manual resume survives subsequent ticks, and a broken session lookup changes nothing.

**Why this packet exists.** The trader reported the bot "running all night prompting
the API constantly". Reading the artifacts found two independent causes, and the loud
one was not the AI layer:

1. **BounceBot swept all night** — Auto Pilot's 30-second tick re-enabled scanning
   with no clock check, and `trading_bot.log` showed ~830-900 metric lines/hour for
   147 symbols, about eight full sweeps an hour, continuing hours past the close with
   IB answering `HMDS data farm connection is broken` and RRS timeouts. **Fixed here.**
2. **`ticker_briefs` retried all night** — see the open question below. **Not fixed.**

No metered API was involved in either: every unattended AI call is hardcoded
`provider="local"` against Ollama on localhost. OpenAI and Anthropic are reached only
from GUI buttons.

### Resolved — overnight AI job cadence (armed and built 2026-08-11)

The ticker-briefs hardening packet was **armed by the trader on 2026-08-11** after the
first overnight run and is **built** on this branch. The question below is kept because
its premises were partly wrong, and the correction is the useful part.

**What the first repaired night (2026-08-10/11) actually showed.** `ticker_briefs`
completed **all 95 symbols in 5,962 s — ~63 s/call**, not the ~4.75 min/call recorded
below. There was no window overrun. Instead **every one of the 95 briefs was
content-free**: the base evidence package was budgeted to the local ceiling *before*
the per-symbol projection, so the per-symbol-rich sources were unfunded at 0 chars
(`setups.current_tracker` 95,806 chars, `setups.current_tiers` 77,124,
`setups.bounce_learning` 17,995, `market.industry_intraday_rs` 17,833) and the funded
tables were sheared to about one row. MRVL's brief reads **"1 of 19 requested source(s)
usable"**, the one being its own watchlist membership. That is TB-0, and it was the
defect worth an hour and a half of GPU time to fix.

**Built:** TB-0 project-then-budget; TB-1 per-ticker failure isolation with an honest
partial morning file (`Briefed N of M. Failed: …` in the header); TB-2 deterministic
membership-only skip; TB-3 resumable completion keyed by
`(session_date, symbol, evidence_hash)`; TB-4 a three-attempt per-session cap with an
identical-error early stop. `run_daily_summary` is untouched, so the two jobs now run
**separate five-session clocks**: `ai_summary`'s continues, `ticker_briefs`' restarts
at zero.

**Live proof owed — the next 22:00 window.** In the morning check: coverage counts
above one usable source per brief, statements citing real evidence, a morning-file
header stating the outcome, at most three `ticker_briefs` ledger rows for the session
(with a `terminal: true` row if it stopped early), and exactly one artifact set per
symbol under `ai_store/briefs/<year>/<session>/tickers/<symbol>/`.

**~~Known defect, reported not yet fixed (2026-08-11 evening review).~~ FIXED
2026-08-12 — and it fired live first.** TB-3's cross-firing reuse could never
trigger on the desk: the projected package's `evidence_hash` covers `generated_at`
and every source's read stamp, so identical evidence hashed differently on every
firing. On the night of 2026-08-11 a second runner instance restarted from symbol 1
and re-briefed 25 symbols, leaving 25 duplicate artifact sets on the DAS. The
manifest now carries a `resume_key` over stable fields only (symbol, session,
memberships, source ids + content); `evidence_hash` keeps its whole-package meaning
for artifact identity. Manifest schema `v1` → `v2`; a row without a `resume_key` is
regenerated, never reused.

**Queued, not built (trader-approved 2026-08-11):** the **nightly journal pull** —
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
  each — about 111 minutes of local inference that produced nothing.** A per-session
  attempt cap (2-3) would keep the self-heal for transient faults (NAS asleep,
  endpoint down) and end the grind.
- **`ticker_briefs` cannot finish as scoped.** It calls the model once per unique
  Focus/watchlist symbol — **95 today** — and publishes the morning file only after
  every one succeeds. At the observed ~4.75 min per call that is **~7.5 hours against
  an 8-hour window**, while the slot reserves only 120 minutes. It needs a symbol cap,
  incremental publication, or both.

Neither is fixed. Deferred deliberately: the 22:00 window on 2026-08-10 is the first
run with the repaired `gemma3:12b-tbv3ctx` model and is the live proof the AI-jobs
repair is owed, so the night was left alone rather than changed hours before it.

**Contingency drafted (2026-08-10, late):** the repair plan for both defects — plus
per-ticker failure isolation with an honest partial morning file, a deterministic
membership-only skip, and resumable per-symbol completion — is fully specified as the
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
| frozen self-test | not re-run — no packaging trigger (no new package, no new runtime asset, no new dependency) |

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
| frozen self-test | not re-run — no packaging trigger |

**Current baseline after the phone-push policy packet (2026-08-11):**

| Check | Result |
|---|---|
| pytest | **2720 passed, 19 subtests passed**, exit 0 (119s) |
| smoke | **7/7**, exit 0 |
| frozen self-test | **29/29**, exit 0 — rebuilt at the trader's request, not by a packaging trigger |

Thirty-three new tests across `tests/test_away_push_roster_and_d1.py` (roster
membership, bucket-spelling collapse, the honest trim marker, and the D1 push
formatting/capping) and `tests/test_away_push_gating.py` (the AWAY-only gate on both
pushes, once-per-hour cadence, a failed send keeping its events, the kill switch, the
Alert Center classifier, and the panel signal firing on both D1 routing paths). Two
existing tests were updated rather than worked around: the Desk Link reclaim push now
declares AWAY (with a new sibling test proving it stays quiet in DESK), and the day-roll
test asserts yesterday's unsent D1 events are cleared.

**Live proof owed:** the next AWAY session — a swing push whose roster matches the
Setup Tracker's Favorite + High Conviction rows, a D1 push naming only events from that
hour, and silence on the swing/D1 channels while the desk sits in DESK or EVENING.

**Trader-verified on the phone, 2026-08-11 20:0x.** One real push built from the live
feed (593 rows, `data_date` 2026-08-11, source `focus`) delivered `ok: True`: five ranked
HC longs plus the full roster — HC 12 long / 7 short, FAV 30 long / 6 short, 55 names,
nothing trimmed. The D1 push is NOT yet proven: its queue only fills from live alerts in
the running desk.

**Documentation close-out (2026-08-11, Markdown only).** The push policy is now stated
where an operator or an arriving AI will actually meet it: `CLAUDE.md`/`AGENTS.md` core
loop (with the rule that a new ntfy sender must gate on AWAY or justify itself),
`docs/AWAY_SCANNER_RUNBOOK.md`, `docs/EVENING_MODE_RUNBOOK.md`, a `docs/FIRST_SESSION_CHECKLIST.md`
row, and `plan.md` P0.3. No file was added, removed, or reclassified, so `docs/README.md`
is unchanged; `WISHLIST.md` is untouched (no trader-directed idea moved). The recorded
baseline above still stands — this pass changed no code, path, or test.

### Desk rebuilt and relaunched onto the push-policy build — 2026-08-11 20:15

The frozen exe was the running desk (pid 35676, started 19:02); the python desk pid 32620
named earlier in this file was already gone. Rebuilt at the trader's request rather than
on a packaging trigger: graceful `CloseMainWindow`, `pyinstaller … --noconfirm` exit 0,
**frozen self-test 29/29 exit 0**, relaunch. **Running pid is now 2552** (started
20:15:20), heartbeat fresh at the 30-second cadence from 20:16:05. `dist/` is gitignored,
so the rebuild is verification only and no commit artifact.

### Desk restarted onto the scan-window build — 2026-08-10 21:19

The desk was closed gracefully (`CloseMainWindow`, so `closeEvent` ran its panel
shutdowns and released the writer lease) and relaunched through
`scripts/launch_gui_auto.ps1`, the same path the 06:00 task uses. **Running pid is now
32620** (started 21:19:22); it supersedes pid 17984 named below. Auto Pilot resumed ON
from saved state and BounceBot started and connected to IB as before.

Verified on the live desk immediately after:

- `bouncebot_scan_window` resolves to **06:00-13:30** from the real machine settings,
  with the verdict `False` at 21:20 and `True` at 09:45.
- **Zero `Metrics ->` sweep lines in `trading_bot.log` after the restart**, watched to
  fifteen minutes — the previous build would have run two full sweeps in that time.
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

1. **Designated writer was unset** — `autopilot_today.txt` had not published since
   2026-07-30, so the whole 2026-08-10 session produced no phone digest and no swing
   push. Fixed with `writer_role.py --designate-self` (NucBox_K8_Plus). The desk was
   restarted at 19:37 local to pick it up (pid 17984 then; superseded by the 21:19
   restart above — the designation is a saved setting and survives both), and
   `writer_role.py` now resolves `designated_writer / may publish True`, exit 0.
   **Not yet proven end to end:** `hourly_away_report_slot_due` returns nothing once
   the hour is past the session close, so no publish was due at restart time.
   `writer_health.json` consequently still carries its pre-fix 15:18 payload — that
   file is rewritten on a *publish attempt*, not at startup, so a stale copy here is
   expected and is **not** evidence the fix failed.
2. **`research_store_dir` was unset** — the warehouse was fully disabled and captured
   nothing. Now `\\MINI-PC\Trading Bot Data\research_lake`, layout created, and the
   restarted desk is the first process to run with it enabled. Capture is proven by
   the next scan writing under the lake, not by configuration alone.
3. **ntfy was already configured and works** — verified by test push (`ok: True`) at
   both `default` and `urgent` priority. Delivery to the iPhone banner/sound is an
   iOS-side setting and is **not yet confirmed by the trader**.

**AI jobs repaired 2026-08-10 (evening).** The task now exits 0 when run through the
scheduler. Details in `CHANGELOG.md`; the live proof is the 22:00 window tonight, and
`%LOCALAPPDATA%\TradingBotV3\logs\ai_jobs-<date>.log` will now carry any failure.
Two AI-layer caveats remain unproven and must be checked against tomorrow's ledger:

- ~~Context smaller than the evidence cap~~ — **closed the same evening.** Local
  calls now cap evidence at `ai_local_evidence_budget_chars` (22,000) and a
  truncation tripwire fails loudly if the server still sees less than was sent. The
  cloud ceiling is untouched.
- ~~The large tier cannot load~~ — **accepted and designed around.** The local large
  tier is retired (plan sec 2); policy drafts and retros belong to the frontier
  model. Revisit triggers recorded: Ollama Vulkan allocator work, ROCm on gfx1103,
  or more RAM.
- **Phase 2 design packet is PROPOSED, not approved.** `docs/LOCAL_AI_AUTOMATION_PLAN.md`
  sec 6.4a. Its six open questions need trader answers before any digest code is
  written — question 1 ("what counts as winning": R at scenario close, MFE/MAE, or
  both) is a trading judgement and is the one the whole fact pack hangs on.

### What the next session must confirm

Four fixes are configured and unit-verified but have **not** completed a live cycle.
None could be proven on the evening of 2026-08-10; all resolve by 09:00 on 08-11:

| Fix | Proof to look for | When |
|---|---|---|
| Designated writer | `autopilot_today.txt.meta.json` names `NucBox_K8_Plus` with a current `verified_at` — it still names the retired `DESKTOP-IABHR62` at 2026-07-30 | 07:00 publish |
| Swing phone push | an ntfy notification carrying numbered swings | 09:00 (push start hour) |
| Research warehouse | new files appearing under the lake root | first scan |
| AI jobs | `ai_jobs-20260811.log` records a completed `ai_summary` / `ticker_briefs` | 22:00-06:00 window |
| BounceBot scan window | **Requires a desk restart first** — the running pid predates the change. Then: one "scanning resumed" line at 06:00, one "scanning paused" at 13:30, and no symbol sweep in `trading_bot.log` after it | 06:00 and 13:30 |

If the 07:00 publish does not happen, read `writer_health.json` first: it will then be
fresh, and its `reason` names the exact gate that refused.

Still open on the desk, not blocking the week:

- `technical_integrity_events.jsonl` is ~247 MB and is never pruned (~10 MB/session).
- Off-site backup: cloud sync was the only off-site Class A copy (decision 0015).
- ~~One flaky test~~ **FIXED 2026-08-15.** `test_stale_staged_files_are_quarantined_not_deleted` was never flaky: `reconcile` compared a file's mtime against a coarser system clock, so a file written microseconds earlier read as "from the future". Both earlier notes here were wrong - it was not "observed once" (3 in 6) and not load-
  related (it reproduced in isolation). See the merge-safeguards section above.

## URGENT — the frozen desk cannot scan (found and fixed 2026-08-13)

The desk switched to `dist\TradingBotV3\TradingBotV3.exe` as its daily driver on
2026-08-12. The frozen build spawned its scan child as `sys.executable -c <code>`,
which under PyInstaller means `TradingBotV3.exe -c …` — rejected by the app's own
argument parser, exit 2, one second after each slot fired. **Every Master AVWAP D1
swing scan failed from 2026-08-12 07:30 through 2026-08-13 09:00.** Last success:
2026-08-11 13:23:59, 622 setup rows.

Nothing else broke, which is why it went unnoticed: BounceBot, the 07:00 open scan,
Auto Pilot and the away report all run in-process. The visible cost was one layer
away — the overnight AI read 11 stale D1 sources.

**Code fix is committed and green** (`scripts/scan_worker.py`,
`scan_service.scan_worker_command`, `launch_gui --run-scan`, `selftest` roster,
`tests/test_scan_worker_spawn.py`), and the desk was **rebuilt 2026-08-13 11:00:25**
after the trader closed it:

| Check | Result |
|---|---|
| pytest | **2738 passed, 19 subtests**, exit 0 |
| smoke | **7/7**, exit 0 |
| frozen selftest | **30/30**, exit 0 — was 29/29; `scan_worker` is the added check |
| frozen `--run-scan` dispatch | **verified** — a deliberately malformed payload now fails inside `scan_worker.parse_payload`, where the old build answered `TradingBotV3.exe: error: unrecognized arguments: -c …` |

**Still owed: one real slot on the desk** — `Swing scan for slot HH:MM finished at …
(N setup rows)` in `autopilot.log`. Nothing before that proves a full scan runs
end to end under the frozen build; the checks above prove only that the child
starts and reaches the scanner. Until then the fallback is running from source
(`scripts/launch_gui_auto.ps1`), where the `-c` form is correct.

Also owed once a slot passes: the D1 sources have been stale since 2026-08-11
13:23:59, so tonight's AI window is the first that can read fresh evidence. A brief
that still cites truncation after a good scan day means something else is wrong.

## What the 2026-08-11 window measured, and what was repaired — 2026-08-12

The packet's owed live proof ran and is **partial**. Ledger and manifest evidence:

| | Result |
|---|---|
| `ai_summary` | **ok at 22:02:53**, first attempt, ~170 s, 10 usable sources — against six degraded rounds the night before |
| `ticker_briefs` | **no completion row.** 126 briefs / 101 unique symbols of 182, 0 failures, 22:04:33 → 01:20:08, killed mid-batch |
| `ai_morning_brief.txt` | **never published** — still the 2026-08-10 file, because publication happened only after the loop |
| TB-0 | **Confirmed.** MDB's real brief: 7 of 19 usable, 0 unfunded (08-10 was 4 of 19 with 5 unfunded) |
| TB-1 / TB-2 / TB-4 | Not exercised — 0 failures, and every membership-only name sits past list position 100 |
| TB-3 | **Proven broken**, 25 symbols with two rows and two distinct `evidence_hash` values |

Three defects and one machine fault, all now addressed except the last:

1. **TB-5 — roster noise.** 96.2% of everything sent to the model (307,630 of
   319,687 chars) was ticker name-dumps matched line-wise; median symbol-specific
   content 42 chars; only 18 of 166 symbols had a real scan line. Fixed by a
   residue test, not a ticker count. Measured effect: **166 model calls → 49**.
2. **TB-3** — see the repaired entry above.
3. **TB-6 — publication only after the loop.** Now republished after every resolved
   symbol, with an explicit in-progress note; the market-session block still
   suppresses publication outright.
4. **`ExecutionTimeLimit` was `PT2H` against an 8-hour window** — it terminated the
   22:00 run's parent at 00:00, freeing `IgnoreNew` so the 00:00 repetition started a
   second runner while the first instance's Python child kept going. The manifest
   shows the two interleaving one-for-one from 00:01:54. Now `PT8H` in
   `scripts/register_ai_jobs_task.ps1` **and applied to the live desk task**.
5. **Machine sleep — trader-owned, not code.** 60 Modern Standby transitions during
   the window, **4h39m asleep**, including an unbroken 01:39:42 → 05:57:09 that
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

- **P0.1:** ~~run the complete Windows automated gate~~ — **done 2026-08-10**
  (2647 passed / smoke 7/7), and **re-run 2026-08-15 on the R1 branch**
  (2773 passed / 19 subtests / smoke 7/7 / source selftest 30/30, all exit 0).
  Re-run again before merge if further code lands.
- **P0.2–P0.4:** run the single-main session checklist, Away/ntfy validation, and
  observability rollover.
- **P0.5:** run the durability mid-session restart/backfill drill.
- **P0.6:** start Local-AI's five-session clock and the warehouse broker/live/pilot
  sequence.
- **P0.7:** merge only after the live-validation day and applicable rechecks pass.
  **One** branch now queues for `main` — `testing-week-2026-08-17`, which carries
  testing-week, R1, R1.1, R2, R7 and R8 together after the 2026-08-15
  consolidation. The gates to re-run, the clean-cache rebuild rule, and the
  desk-switch order are in the Monday sequence at the top of this file.

Do not add historical detail here. When a change lands, update `CHANGELOG.md`; when a
gate remains, update `plan.md`.
