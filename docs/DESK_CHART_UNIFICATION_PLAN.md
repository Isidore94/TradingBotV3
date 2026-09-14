# Desk chart unification — packet R4

Status: **BUILT 2026-08-16 — LIVE PROOFS OWED**, for `plan.md` Phase 0.5 **R4**.
Authorized by the trader on 2026-08-15; built on `testing-week-2026-08-17` under
the 2026-08-15 weekend redirect, after R3 closed.

Sections 1–5 and 6.1–6.3 are built and green. Two items are **explicitly held
for a trader decision** rather than silently skipped:

- the **Focus Picks** reviewed-today marker (§5) — that surface is editable
  watchlist text, not a table; see the §5 note. **Triaged 2026-08-17 (Fable,
  trader-delegated; the trader may override): a technical decision, taken —
  build it as decoration only.** The marker renders as a per-line editor
  decoration (extra-selection highlight or gutter mark with a tooltip),
  derived from the same recorded-decision streams as every other marked
  surface, and **never enters the document text** — nothing new is ever
  serialized to the watchlist files, making the never-corrupt-user-names
  invariant structural rather than careful. A test must pin save-path
  byte-identity with markers active. If the editors cannot be decorated
  without touching the text path, the fallback is NOT a `● ` prefix — it is
  returning to the trader with that one question;
- **§2.2's `review_host`** for the RS/RW and Industry boards — `watch_host` is
  wired on both, and §2.1's CaptureRail now gives every chart surface
  veto/like/note/hypothetical-stop, which is what §2 was for. `review_host`
  additionally carries the setups table's *advance-to-the-next-row* flow, which
  is table-specific and has no meaning on a ranked board. Recorded as answered
  by a better mechanism rather than built as written. **Triaged 2026-08-17
  (Fable, trader-delegated): CLOSED, no build.** Auto-advance on a board that
  re-ranks under the reviewer would advance to the wrong symbol — the flow is
  affirmatively not wanted there, not merely unbuilt. Reopen only if the
  trader asks for a review-queue mode (ordering frozen at review start).

The §8 exit gate is unrun: all of it is live observation.

Trader intent (2026-08-14): *"anytime I bring up a chart from master avwap setups
or the RS/RW board or anywhere it would be nice if it had all the functions of
chart review. it would also be nice if it made it very obvious I have already
checked that chart today"* — plus armed price alerts visible on charts, and the
early-morning D1 gap question.

## 1. Answered outright by recon (no build work)

- **Labeled Y axis already exists**: `CandleChart` draws labeled price ticks in
  both log and linear mode via its custom `PriceAxis`
  (`scripts/ui/widgets/candle_chart.py:151-183, 338-343`). The trader's instinct
  was right that something is off about early-morning D1 charts — but it is the
  data, not the axis (section 3).

## 2. Chart Review's capture everywhere

Current state: the veto/like/note/hypothetical-stop toolbar is one self-contained,
already-portable widget — `CaptureRail` (`scripts/ui/widgets/capture_rail.py`,
`set_context(symbol=…)` at 260, annotation-only by documented design reversal;
writes `trader_annotations.jsonl` via `ui.annotations.store`). It is instantiated
in exactly one host (`scripts/ui/panels/chart_review_panel.py:234`). The snapshot
popup (`SymbolSnapshotDialog`) and the Alert Center pane (`AlertChartReview`) each
carry different partial control sets; the RS/RW board and Industry panels pass **no
review_host at all** (`rs_window_panel.py:278-294`, `industry_panel.py:186-215`),
so they lack even the Dislike button.

Design:

1. Embed a `CaptureRail` in `SymbolSnapshotDialog` (layout room exists — it is a
   reused floating QDialog) and in `AlertChartReview`'s pane, wired exactly as
   `chart_review_panel.py:384-415` does (`set_context` on every symbol change,
   `d1LevelSelected` connection).

   **Amended 2026-08-20 (trader, with screenshot): where the rail sits is the
   HOST's decision, not `AlertChartReview`'s.** "Layout room exists" held for
   the floating dialog and the Chart Review workspace; it did not hold in the
   desk's alert column, where title → setup text → charts → two arm rows → a
   ~600px rail → the verb row left the charts unreadable ("I cannot see the
   charts at all… I am ok with them being tabbed where alerts/D1 focus/RSRW
   board is and clicking into them"). `AlertChartReview` takes
   `docked_controls` (default `True` — the historical single-column stack, and
   what the snapshot dialog and workspace keep); the Alert Center passes
   `False` and hosts `arm_bar` on its existing **Armed** tab and `capture_rail`
   on a new **Capture** tab. Under the charts there is now exactly one row: the
   verb row, which advances the review queue and must never cost a click.

   The rail's five-second contract survives the move rather than being traded
   away for it: a `QShortcut` bound inside a hidden tab page never fires, so
   Alt+V / Alt+K / Alt+S / Alt+N are rebound at **panel** scope
   (`WidgetWithChildrenShortcut`), each raising the Capture tab before handing
   off to the rail's own handler, and the rail's copies are switched off for
   that host (`bind_action_shortcuts=False`) because two live bindings for one
   sequence is an ambiguous shortcut and Qt fires neither. The handler list
   comes from `CaptureRail.action_shortcuts()`, so it is a rebinding, not a
   second copy. Enter-to-commit is untouched.

   Armed state stays legible with the Armed tab closed: the tab title carries
   a count and the review pane's verb row carries an always-visible armed line
   (`AlertChartReview.armed_summary`, fed by `armedSummaryChanged`).
2. Give the RS/RW board and Industry panel the same `review_host`/`watch_host`
   wiring Master AVWAP already has, so Dislike/D1-Focus/watch controls appear
   there too.
3. Alert Center gains **"I like the stock"**: the CaptureRail like+claim control on
   the same pane (writes one annotation row, same dataset as Chart Review). The
   existing **Add to Focus Picks** verb stays the explicit placement action —
   capture stays analysis-only per `docs/CHART_REVIEW_WORKSPACE_PLAN.md` §7; LIKE
   never places membership, and packet R3's badge reads both streams.
4. The dislike flow everywhere offers the structured veto vocabulary (packet R3
   §3.4 owns that change; this packet gives it the surfaces).

## 3. Early-morning D1 gap honesty

Found mechanism (recon 2026-08-15): the forming D1 preview candle is built from IB
RTH M5 bars when an M5 cache exists (accurate), but **falls back to a Yahoo
`yf.download(interval="1d")` "today" row taken verbatim as OHLC** whenever no M5
cache exists yet (`symbol_snapshot_dialog.py:36-80, 451-546` calling
`fetch_daily_bars_from_yahoo`, `master_avwap_lib/legacy.py:15001-15051`) — a thin
pre-market/early print both mis-states the gap and **drives the Y autoscale**
(`candle_chart.py:413-442`; painted levels are excluded from autoscale, so they are
not the cause). IBKR is never consulted on this specific path.

Design: prefer building the forming bar from an IB M5 fetch when the desk is
connected; when only the Yahoo daily row is available, label the preview candle's
source visibly on the snapshot (the provenance plumbing exists in
`chart_review_panel.provenance_state()`; the dialog does not surface it today) and
suppress the Yahoo-sourced forming candle for the first N minutes after
`session_has_opened()` (default 15, setting-tunable) rather than painting a thin
print as a real bar. Missing data renders as absence with a caveat, never as a
confident candle.

## 4. Armed alerts painted on charts

Current state: armed `PriceAlertService` entries and armed D1 level/event watches
render only as text chips in `ArmBar`, never on the chart; the levels payload has
no alert family (`chart_levels.py:36-81`).

Design: a new `GROUP_ALERTS` family in `chart_levels.build_d1_levels`, built from
`price_alerts.load_price_alerts()` plus the symbol's `D1LevelWatch`/`D1EventWatch`
entries, threaded through the `ChartDataService` worker like every other family
(never the paint path), drawn by `CandleChart.set_levels` with stable ids and the
existing item-pool discipline. `PaintLinesButton` derives its menu from
`LEVEL_GROUPS`, so the show/hide toggle is free. Strictly read-only display: the
single-writer rule on `price_alerts.json` is untouched, and clicking a painted
alert line selects it (existing `levelSelected` path) — arming still goes through
the one existing writer flow.

## 5. "Already checked today" badge

Trader decision 2026-08-15: checked = **recorded decisions only** (✕/★/veto/like/
note) — no view tracking, zero new capture. Render packet R3 §3.3's decided-today
set as a prominent badge on the snapshot header and a row marker on every table
that opens charts (setups, RS/RW, Industry, Focus, Alert Center). Presentation
only; resets at the market-date boundary.

**BUILT 2026-08-16**, on the setups table (R3), the snapshot popup header, the
Alert Center review pane, the RS/RW window and the Industry board. The marker
rides display text only: sort roles and the row payload are untouched, so it can
never become a ranking and a chart-open handler never receives a `● AAPL`.

**Focus Picks is deliberately NOT marked, and this is an ask-first item, not an
omission.** The Focus Picks panel is not a table — it is a pair of editable
plain-text watchlist editors whose *text content is the watchlist itself*.
Injecting a `● ` prefix into those editors would put the marker inside data that
gets written back to `swinglongs.txt`/`longs.txt`, which risks the hard
invariant that user-entered watchlist names are never corrupted or auto-removed.
Marking that surface needs a design decision this spec does not make — a
separate non-editable status column, a per-line gutter, or a decision to leave
it unmarked because the trader already owns every name there. **Ask the trader
before building it.**

## 6. Recovered Alert Center quality contract (2026-08-14)

Historical source: `docs/archive/ALERT_CENTER_QUALITY_PACKET.md`, recovered from commit
`671ee57` on 2026-08-16 and classified as historical evidence. Packet R2 later
absorbed its auto-pick provenance, scoped M5-side removal, persistent decline,
and **`not_today`, not trader dislike** outcomes. Do not rebuild those under a
second design. The following trader outcomes were not otherwise absorbed and
now belong to R4.

### 6.1 "Not today" never cancels a trader's alarms

Trader wording, 2026-08-14:

> "'Not today' should still trigger on the alerts I set."

This is a general rule for every dismissal path. Dismissing a symbol for the day
must never disarm or defer a trader-armed chart watch, D1 event watch, armed D1
level alert, or price alert. Only the explicit disarm toggles may cancel them.
Their hits still enter the feed and sound. Alerts tagged `CHART_WATCH_TAG` are
the routing identity for chart-watch, armed-level, and D1-event-watch hits and
therefore bypass the ignored-symbol feed filter. Focus-derived automatic D1
interest (`FOCUS_D1_EVENT_TAG`) is not trader-armed and correctly continues to
lapse with Focus membership.

The trader-authorized 2026-08-16 repair scope for the fenced
`alert_center_panel.py` names all four edits: (1) `_ignore_alert_symbol` stops
deleting `_chart_watches`; (2) `add_alert` exempts `CHART_WATCH_TAG` from the
ignored-symbol return; (3) `_poll_d1_level_watches` stops deferring an ignored
symbol; and (4) `_poll_d1_event_watches` stops deferring an ignored symbol. The
second pair was explicitly approved after recon found the extra suppressors.
Producer tracing confirmed both persistent poll stores are trader-armed only:
their entries come from the UI arm APIs or those APIs' persisted files. Automatic
Focus interest is separate in `_poll_focus_d1_interest`, creates only transient
evaluator objects, retains its ignored-symbol guard, and emits
`FOCUS_D1_EVENT_TAG`; it therefore continues to lapse with Focus membership.
Deterministic tests must traverse both seams: ignored trader-armed level/event
hits feed + sound, while ignored automatic Focus D1 interest does neither. Any
additional fenced-file change requires another ask-first approval.

**Built 2026-08-16 as the authorized R4 quick win.** The two-direction seam is
pinned in `tests/test_qt_alert_center.py`; 80 focused Alert Center/arm/watch tests
and the full 3377-test suite plus 19 subtests pass. Live proof still owed: dismiss
a symbol with a real trader-armed watch, observe its hit in the feed and sound,
and confirm the same symbol's automatic Focus-derived D1 interest stays absent.

### 6.2 Make explicit Focus placement readable on the Alert screen

Trader wording, 2026-08-14:

> "If I like a stock I can add it to m5 focus picks. Then I get flagged on
> pullbacks."

The existing feed-row favorite action has the intended membership semantics but
only a star glyph. R4 promotes it to a labeled action such as **Like → M5 Focus**
or **Like → Swing Focus**, with the lit state retaining the remove-from-Focus
affordance. The chart pane already has an explicit labeled Add-to-Focus control;
keep it. This must not blur into R4's separate CaptureRail LIKE: CaptureRail LIKE
is analysis-only and never writes Focus membership.

### 6.3 Repetition control is presentation, not weaker detection

Trader wording, 2026-08-14:

> "I don't want to be constantly seeing the same stocks over and over ... less
> spam and more quality ... I basically don't want to see the same ticker over
> and over again. It def finds bangers though."

The R4 display-only outcome for the main Alerts feed is:

1. One live row per symbol + side + market day. A repeat updates that row in
   place, retains first-seen time, shows a repeat-count badge, and does not
   re-sound or re-float unless it escalates.
2. Escalation means a strictly higher best tier, first BANGER, or first PROVEN.
   Focus-privileged names and trader-armed hits always surface and sound; they
   are never silently folded into a stale row.
3. During the first configurable N minutes after the open (historical proposed
   default 30; zero disables), ordinary alerts group into one ranked digest row
   per scan cycle. BANGER, PROVEN, Focus-privileged, trader-armed, entry-assist,
   and ready-D1 output remain immediate. Digest contents stay reachable rather
   than being discarded.
4. This changes no detector, score, evidence stream, History, AWAY push, or
   `review_policy.json`. It adds no suppression field and is superseded by the
   future P5.1 typed-delivery challenger once that manifest passes.

### 6.4 The three §6.2/§6.3 confirmation gates — ANSWERED by the trader 2026-08-16

The historical gates were put to the trader before any §6.2/§6.3 code was
written. All three are now decisions, not open questions:

| Gate | Trader's answer |
|---|---|
| Open-burst digest window default | **30 minutes**, the historical proposal unchanged. Settings-tunable via `alert_open_digest_minutes`; **0 disables** the digest entirely |
| Optional Enter-to-skip reason on a **Focus** like | **No prompt.** Liking stays one click. The structured vocabulary earns its keep on the dislike side, which R3 just wired; a prompt on every like would tax the cheap action to collect data nothing yet asks a question of. **Still true for the Focus like (the ★ verb).** It does NOT extend to the **CaptureRail** LIKE, which §2's line above keeps distinct: the trader reversed that one on 2026-08-22 ("if I like a chart I should always be prompted with why") and its why is now required — R9.2, `docs/CHART_REVIEW_WORKSPACE_PLAN.md` §7 |
| Is the escalation list exhaustive? | **Yes — those three**: a strictly higher best tier, the first BANGER, the first PROVEN. Conservative by design, and safe because Focus-privileged and trader-armed hits bypass folding entirely under §6.3.2, so nothing the trader armed can be quieted by this |

These answers bind the build. Changing any of them later is a fresh trader
decision, not an implementation detail.

## 7. Fenced files, invariants, tests

Ask-first at edit time: `scripts/chart_levels.py` (shares detector-adjacent state),
`scripts/chart_watch.py`, `scripts/price_alerts.py`,
`scripts/ui/panels/alert_center_panel.py`, `scripts/master_avwap_lib/legacy.py`
(only if the forming-bar fetch path is touched there). Display/capture surfaces
(`candle_chart.py`, `symbol_snapshot_dialog.py`, `capture_rail.py`, panels) are
UI-side but reviewed with the same care. Invariants: capture is analysis-only —
nothing here may mute, suppress, score, gate, or alert; forming bars stay labeled
preview; one writer for `price_alerts.json`; worker-thread levels build, never the
paint path.

Tests: levels-family construction (alerts appear/disappear with the stores),
forming-bar source selection + suppression window + provenance label, CaptureRail
context wiring per host (Qt tests), badge derivation from decision stores, and a
guard that no capture path writes Focus/watchlist membership.

## 8. Exit gate

All entry points open a chart with capture + watch controls + painted armed alerts;
one desk morning confirms the forming-bar caveat replaces the inflated-gap
rendering; the trader records a dislike from the RS/RW board and sees the badge
appear everywhere that symbol renders that day. Section 6.1 additionally requires
one live ignored-symbol armed-watch hit that feeds/sounds while automatic Focus D1
interest for that ignored symbol remains absent.

## History and the viewport (packet WS-CH, 2026-09-13)

The trader's sentence was *"200 candles is not enough"* (WISHLIST 10H). The
answer separates two questions R4 had treated as one: how many bars a payload
**holds** and how many the chart **opens on**.

**D1.** `chart_snapshot.D1_HISTORY_SESSIONS` (1,000, roughly four NYSE years) is
how far back a daily payload reaches; `D1_DEFAULT_SESSIONS` (90) stays the
opening window. The durable parquet store already held years — `load_d1_bars`
has always read the FULL history and `build_d1_snapshot` has always computed
indicators over it before slicing — so the only thing that was ever short was
the slice. This costs one longer slice of bars already in memory and **no
provider request whatsoever**. The payload is a target capped by what the store
holds: it carries `oldest_available` (the oldest bar drawn) and
`history_truncated` (there is more behind it), and a symbol with 300 stored
sessions reports 300 and `False`, because there is nothing left to pan to and
saying otherwise invites the trader to drag at a wall.

`CandleChart.set_data(..., initial_view_sessions=N)` frames the tail while the
widget holds every bar, so **panning left reveals the older bars with no
request of any kind**. The y-range is taken from the visible window — a
four-year scale flattens today's candles into a line — while the log/linear
decision still asks every bar, because a bar off the left edge is one pan away
and flipping the scale mid-drag is worse than opening linear. Downsampling
(`setClipToView` + auto peak) was already in place for exactly this and stops
being a no-op here. Measured offscreen at 1,000 candles with 14 overlays:
`set_data` ~24-31 ms, `grab()` ~15-22 ms.

Both hosts are the same widget. The centre Visual Alert Review pane
(`AlertChartReview` → `SymbolSnapshotWidget(compact=True)`) opens on 90; Chart
Review keeps its own `CHART_REVIEW_D1_SESSIONS` (520) opening window; both now
hold 1,000 behind it.

**Levels do not follow the payload.** `chart_levels.build_d1_levels` gained
`price_range_bars` and `ChartDataService` hands it the INITIAL VISIBLE window,
so `horizontal_levels`' price filter behaves exactly as it did before the
history grew: a 2021 store level is not admitted to a chart opened on 2026, and
the per-bucket clutter budget is not spent on lines nobody can see. **Panning
left does not recompute levels** — the payload is fixed at build time and the
paint path reads no caches (Milestone 8 stands).

**The provider request did not grow.** `SymbolSnapshotWidget._start_d1_backfill`
still sizes its catch-up off the host's `d1_sessions` (260 calendar days
compact, 754 for Chart Review), not off the history target. That path is a
repair for a stale symbol, not a history import; the store is filled by the scan
pipeline. A test caps it at 800 calendar days.

**M5 in bounded chunks.** The intraday chart opens on today's two sessions and a
**Load older** button on the M5 legend row adds two more, through the same
in-memory `bot.m5_chart_bars(max_sessions=n)` read the chart already used —
never a fetch, and the pan handler deliberately has no trigger in it, because a
pan that fetches is a fetch on the paint path. Ten sessions per symbol per desk
session is the ceiling. The chunks overlap by construction, so the merge cuts at
the fresh chunk's first bar rather than unioning: no bar appears twice, and a
bot whose cache has since shrunk cannot take history off a chart that has it.
Older bars arrive on the LEFT, so the view is preserved by CANDLE IDENTITY
(`CandleChart.visible_bar_span` / `restore_bar_span`), never by index range. A
raising provider costs the older bars and never the chart: the extra sessions
roll back, the drawn bars stay, and the button reads `older bars unavailable`.

**The oldest date is in the strip, not a popup.** `provenance_state` appends
`D1 back to <date>`, with `(more behind)` when the store holds more. That is the
only provenance strip the desk has (Chart Review's); the centre pane has none to
add to.

**H1/H4 (item 3) was not built, because the desk draws neither.**
`bounce_bot_lib/legacy.py._closed_h1_bars` aggregates completed H1 bars and
`master_avwap_lib/legacy.py:28235 resample_intraday_bars_to_4h` resamples H4
from that H1 history; both feed the HTF study, and neither reaches a chart, a
widget or a payload — every `set_data` call in `scripts/ui/` passes `"d1"` or
`"m5"`. The packet's own instruction applies: stop at D1/M5. A 500-bar H1/H4
target is only meaningful once an H1/H4 chart exists.

**Symbol ownership (CH-SYM, trader-authorized repair 2026-09-14).** Retained
history belongs to one symbol. Switching names clears the previous D1/M5
snapshots and chart state before any pending read, capture, quick-fill or
history merge can use them. A cached snapshot for the new name may render
immediately. An empty or raising provider for the new name must never borrow
the previous name's prices; same-symbol refreshes still retain older history.
This repairs the WS-CH merge seam, not a price-jump filter: real gaps stay real.

Tests: `tests/test_ws_ch_chart_history.py` over a 1,300-session golden fixture;
`tests/test_chart_symbol_isolation.py` covers symbol switching, pending actions,
empty/raising data reads and cached-symbol reuse.

## The two held-back items — resolved 2026-08-18

R4 recorded two items as held under the ask-first rule rather than skipped. The
2026-08-18 integration redirect pre-satisfies that rule, so both were revisited.

**The Focus Picks reviewed-today marker — BUILT, but not where R4 proposed it.**
The hold was correct and its reason still stands: those editors hold editable
watchlist *text* that is synced back to the shared watchlists, so a marker
injected into a row is one careless save away from becoming a symbol name — and
"user-entered watchlist names are never auto-removed" would be one bad round-trip
from being violated in the other direction. The marker is therefore a read-only
LINE beside the editors ("Reviewed today, in Focus: …"), which answers the same
question and cannot reach a byte the sync writes. Unreadable evidence renders
nothing rather than claiming nothing was reviewed, and a reviewed name that is
not in Focus is counted rather than listed. `tests/test_qt_focus_reviewed_today.py`
pins all of it, including that the editors' own text is untouched.

**§2.2's `review_host` for the boards — still declined, and now on the record as
a decision rather than a hold.** Its remaining half is the setups table's
advance-to-next-row flow. A ranked board has no "next row" in that sense: the
order is a score, not a queue, and advancing through it would either fight the
ranking or invent a queue the trader never asked for. §2.1's CaptureRail already
delivered what §2 was for — capture on every chart-opening surface. Reopening
this needs a trader statement that they want a queue over the boards, which is a
workflow decision, not a wiring one.


## Chart Review retires as a PAGE, not as a surface (packet WS-WL, 2026-09-13)

WISHLIST 10G puts one Watchlist on the Trading Desk and retires the standalone **Chart
Review** and **Focus Picks** nav entries. This plan's subject — the capture rail, the
provenance line, the movers-only filter, the D1 gap honesty, the armed-alert painting —
is untouched. `ChartReviewPanel` is still constructed by `MainWindow`, still imported by
`tests/test_chart_review_workspace.py` and `tests/test_r4_capture_surfaces.py`, and is
still the reference implementation the Alert Center's own rail was built against. What
went is the left-nav row.

The two things the page could do that nothing else could, and where they live now:

* **Open a symbol that is on a list** → `WatchlistTabPanel.chart_selected()`, which
  routes through the Alert Center's `chart_symbol` — the desk's ONE door (trader,
  2026-09-03: every ticker click on the Trading Desk lands on the centre chart). It is a
  MANUAL look: `MANUAL_CHART_TAG`, no place in the waiting list, never a re-queue and
  never a skip count.
* **Open a symbol that is on NO list** (`ChartReviewPanel.open_symbol`, the lookup box —
  the one thing the page could do that selecting a row cannot) →
  `WatchlistTabPanel.chart_lookup()`, behind the tab's **Chart only** button and the same
  `Ctrl+L` box. Read-only exactly as it was: the name goes in the machine-local recents
  (`ui/services/symbol_lookup.RecentLookups`, the SAME store and file the page used) and
  onto the chart, and into no watchlist, no Focus list and no CandidateRegistry. The
  page's recents CHIP STRIP became a completer on the add box — same memory, no second
  widget row in a height-conscious column.
* **`Ctrl+L`** (focus the lookup box) → the same sequence, bound ONCE at the Watchlist
  tab's own scope with `WidgetWithChildrenShortcut`, focusing the tab's add box. It is
  not bound at window scope, deliberately: two bindings for one sequence fire neither,
  and a `QShortcut` in a hidden tab never fires at all.

`Alt+E` (the page's setups toggle) is not carried over — the desk has its own always-on
setups toggle and F9, which is what the trader uses.

**The thing that bit us here, recorded because it will bite the next tab too.** Raising a
tab does not put the keyboard inside it: after `setCurrentWidget` the focus widget is the
tab BAR, which is a child of the `QTabWidget` and NOT of the page. `correctWidgetContext`
then refuses a `WidgetWithChildrenShortcut` bound on the page, and the shortcut silently
does nothing. Measured in this worktree with offscreen Qt: with focus on the tab bar the
binding never fires; with focus on any widget INSIDE the page it fires every time. So
raising the Watchlist tab both reveals its column (a tab in a hidden column cannot be
read, and a hidden widget's shortcut never matches) and moves focus into the panel.
