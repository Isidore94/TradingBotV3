# Desk chart unification — packet R4

Status: **ACTIVE specification** for `plan.md` Phase 0.5 **R4**. Authorized by the
trader on 2026-08-15. Builds after R1–R3 unless a sub-item is pulled forward as a
quick win (each sub-item is independent).

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

## 6. Recovered Alert Center quality contract (2026-08-14)

Historical source: `docs/ALERT_CENTER_QUALITY_PACKET.md`, recovered from commit
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
| Optional Enter-to-skip reason on a Focus like | **No prompt.** Liking stays one click. The structured vocabulary earns its keep on the dislike side, which R3 just wired; a prompt on every like would tax the cheap action to collect data nothing yet asks a question of |
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
