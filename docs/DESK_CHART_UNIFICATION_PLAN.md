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

## 6. Fenced files, invariants, tests

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

## 7. Exit gate

All entry points open a chart with capture + watch controls + painted armed alerts;
one desk morning confirms the forming-bar caveat replaces the inflated-gap
rendering; the trader records a dislike from the RS/RW board and sees the badge
appear everywhere that symbol renders that day.
