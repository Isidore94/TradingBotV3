# TradingBotV3 implemented history

Last reconciled: **2026-08-18** from the working copy of
`phase05-integration-blitz` (cut from `testing-week-2026-08-17`, which carries
testing-week + R1 + R1.1 + R2 + R3 + R4 + R5 + R6 + R7 + R8, with the four later
`phase05-r2-focus-gating-strength-board` commits merged in on 2026-08-18)

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
