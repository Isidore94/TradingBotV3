# M5 signal engines — packet R5

Status: **ACTIVE specification** for `plan.md` Phase 0.5 **R5**. Authorized by the
trader on 2026-08-15; ranked **last** among the feature packets (trader's ordering),
so R1–R4 land first. Every engine here is new detector work: pure builders may land
with hand-written unit tests, but **wiring any of them into the live M5 alert path
requires a golden characterization fixture first** (decision 0009; the
`tests/test_d1_zone_arms.py` precedent states this split explicitly), and every
detector-hosting file is ask-first at edit time.

Trader decision recorded 2026-08-15: in the any-bounce aggregator, "previous AVWAP"
means the prior earnings-anchor's **VWAP line itself** — which is tracked nowhere
today and needs new D1 scan output — in addition to the prior 1st-dev band that
already exists.

## 1. Current state (recon 2026-08-15)

- **No Heikin-Ashi, no SMI, no TC2000-style LRSI code exists anywhere.** The one
  module named `laguerre_rsi` (`scripts/indicators/laguerre_rsi.py`) is a
  different Ehlers fractal-energy algorithm, has **zero importers**, and is
  deliberately excluded from the frozen bundle
  (`tests/test_packaging_spec_drift.py:73`). Its value here is as the style
  precedent: pure module, Config/Result dataclasses, completed-only arrays, its own
  `FEATURE_VERSION`, and a contract-bearing golden fixture
  (`tests/fixtures/laguerre_rsi_v1.json`).
- New-alert-type plumbing is mature: a `BOUNCE_TYPE_DEFAULTS`/`BOUNCE_TYPE_LABELS`
  entry, a dedicated `emit_*` method in the per-symbol M5 loop, and a
  `gui_callback(message, tag)` → `BounceAlert.from_callback_many` → Alert Center
  route (`scripts/bounce_bot_lib/legacy.py:243-292`;
  `scripts/ui/services/bounce_service.py:938-978`; `scripts/ui/models/bounce.py`).
- The nearest shape for "arm several levels, fire on the first completed-bar hit"
  is the zone-arm pair `build_d1_zone_arms`/`detect_zone_arm_triggers`
  (`scripts/master_avwap_lib/d1_zone_arms.py:107-411`), and the nearest persistent
  re-armable watch is `D1LevelWatch`/`D1EventWatch`
  (`scripts/chart_watch.py:559-1055`) — currently **single-kind per watch**.
- H1 needs no new fetch: `_closed_h1_bars` aggregates completed H1 candles from the
  cached M5 array in the loop (`legacy.py:2185-2224`); an H1 15EMA already exists
  for mid-earnings focus levels (`legacy.py:180-183`).
- D1 AVWAP band values are already available inside the M5 loop via the zone-arms
  file written by the D1 scan (`legacy.py:4073-4089`); the **prior anchor's AVWAP
  line is not** (only its 1st-dev band, `PREV_UPPER_1`/`PREV_LOWER_1`).
- The existing ORB detectors (`check_orb_break_setups`, `legacy.py:6208-6342`) are
  the 30–60-min **delayed** opening-range break — a different setup from the
  trader's gap-up-first-candle-HOD ask.
- The completed-bars guard is re-implemented ad hoc at each detector call site
  (same idiom at `legacy.py:4384-4386` and `4533-4535`); there is no shared
  helper.

## 2. New pure indicator modules (`scripts/indicators/`)

All completed-bars-only, offline-deterministic, hand-computed fixture tests at
land time, golden contract fixtures before live wiring. Note: the first real
importer of `scripts/indicators` changes packaging — update
`tests/test_packaging_spec_drift.py`'s allowlist, the spec's `collect_submodules`,
and the frozen selftest roster in the same change, then rebuild per the frozen
policy.

### 2.1 SMI (stochastic momentum index, TC2000 parity)

Trader's TC2000 source (recorded verbatim in WISHLIST history, commit `994f575`):
`XUP(XAVG(XAVG(C - (MAXH5 + MINL5)/2, 5), 20) / XAVG(XAVG(MAXH5 - MINL5, 5), 20),
XAVG(…, 6))` with the numerator EMA > 0. Implementation: 5-bar high/low range
midpoint distance, double-EMA smoothing (5 then 20), normalized by the
double-EMA-smoothed range, with a 6-EMA signal line. Signal of interest: SM1 < SM2
with both below 0, then SM1 crosses above SM2.

### 2.2 TC2000-style LRSI (efficiency oscillator)

Trader's source: 4-bar sum of `ABS(C >= EMA9.prev) * (EMA9 − EMA9.prev)` over the
4-bar sum of `ABS(EMA9 diffs)` × 100 — an efficiency-ratio-style oscillator over
EMA9 changes, range 0–100. **Name it distinctly** (e.g. `efficiency_lrsi.py`) —
it must not be confused with the unrelated Ehlers `laguerre_rsi.py`. Crossing
states: up through 20 (strongest), up through 50 (fine).

### 2.3 Heikin-Ashi transform + reversal classifier

Standard HA OHLC transform plus a per-bar color/reversal classification (style
precedent: the H1 candle-color loop, `legacy.py:2185-2197`). Output: bar color
series and reversal events (first HA candle against the prior run's direction).

## 3. New M5 alert types (BounceBot taxonomy)

1. **LRSI cross** — its own toggleable alert type: fires on a completed M5 bar
   crossing up through 20, or up through 50 (each side inverted for shorts).
2. **HA + SMI + LRSI combo** ("strongest") — HA reversal with an SMI cross
   (2.1's signal) and an LRSI cross occurring **within 3–4 completed M5 candles of
   each other** (window setting-tunable). New correlator tracking each signal's
   most recent firing bar per symbol; fires once per confluence. **M5 Focus
   symbols only**, per the trader's framing.
3. **First-candle ORB candidate** — a gap-up name whose first completed M5 candle
   prints the session HOD marks the symbol an ORB candidate (tag/annotation, not
   yet an entry alert); then an LRSI pullback (below 50/20) arms two follow-ups:
   an alert on a new session HOD, and an informational alert on LRSI crossing back
   above 50. Distinct from — and coexisting with — the delayed-ORB detectors.
   Shorts inverted (gap-down, first-candle LOD).

Each type: TYPE constant + `BOUNCE_TYPE_LABELS` entry + `emit_*` in the per-symbol
loop + toggle, following the `orb_breakout`/`ema8_grind` precedent (these shipped
as live alert types, not shadow challengers; the champion SPY-pause and D1-wick
paths are untouched, which is what the champion invariant actually protects — state
this in the packet's ask-first request). Alert Center routing: reuse an existing
tag family vs a new lane is decided at build time with the trader.

## 4. The any-bounce watch

One armed request per symbol/side: *"tell me when this name bounces off any of my
levels."* Level set (trader 2026-08-14 + 2026-08-15): D1 1st-dev band, current
AVWAP, **previous AVWAP line** (new — see below), previous 1st-dev band, D1 15EMA,
D1 21EMA, session M5 15/21EMA, and the H1 15EMA.

Design:

- **New `AnyBounceWatch`** dataclass in `scripts/chart_watch.py`, parallel to
  `D1EventWatch` but carrying a **set of kinds**, persisted in its own JSON store
  (new file beside `D1_EVENT_WATCHES_FILE`), owned by the same Alert Center panel
  store owner — one writer, `PriceAlertService` untouched (it is a different,
  simpler above/below system).
- Armed from a single **Any bounce** button on the chart surfaces (packet R4 gives
  every chart the button row) and the Focus board.
- Evaluation each completed M5 bar reuses the two-bar bounce idiom from
  `detect_zone_arm_triggers`; D1 levels come from the zone-arms file, session EMAs
  from the loop's own series, H1 15EMA from `_closed_h1_bars` aggregation. Fires
  once naming the level that held, then disarms; one click re-arms (the trader's
  stated workflow: "if I still dislike it when that alert fires then I can set it
  again").
- **Prior-anchor AVWAP line**: extend the D1 scan's zone-arm/level output to carry
  the previous earnings-anchor's AVWAP value alongside the existing
  `PREV_UPPER_1`/`PREV_LOWER_1` (computed with the same frozen-sigma
  `calc_anchored_vwap_bands` call over the prior anchor — the formula itself is
  untouched). This is the one D1-side change in the packet.

## 5. Shared completed-bars helper

Before adding four new detectors that would each re-implement the ad hoc
`cutoff`-filter idiom, extract one shared `completed_m5_bars(df, now)` helper and
use it in the new engines (existing call sites migrate opportunistically, not in
this packet — no behavior change to shipped detectors without fixtures).

## 6. Fenced files, invariants, tests

Ask-first at edit time: `scripts/bounce_bot_lib/legacy.py`,
`scripts/chart_watch.py`, `scripts/master_avwap_lib/d1_zone_arms.py`,
`scripts/master_avwap_lib/legacy.py` (prior-anchor output),
`scripts/ui/panels/alert_center_panel.py`, `scripts/ui/services/bounce_service.py`.
Invariants: completed bars only (every trigger); champions untouched; one writer
per store; a forming H1/M5 bar is preview; missing zone-arm data for a symbol means
that level silently absent from the watch, never a fabricated level.

Tests: per-indicator hand-computed fixtures, then golden contract fixtures at
wiring time (`tests/conftest.py` FixtureContract); correlator window tests
(signals 1/3/4/5 bars apart); AnyBounceWatch persistence/re-arm/one-fire tests
(pattern: existing chart-watch tests); ORB first-candle classification incl.
no-gap and open-missing-data refusals; packaging drift + frozen selftest updates
for the `indicators` package.

## 7. Exit gate

Pure modules green with fixtures; live wiring lands engine-by-engine (LRSI cross
first — it is the simplest and the trader wants it standalone anyway), each behind
its toggle, each with a desk session confirming alert volume is sane before the
next engine wires in. The frozen exe rebuild + 30-check selftest runs when the
`indicators` package first enters the bundle.

## 8. Open questions

- Alert Center lane: same `d1_flag` family or a new tag for LRSI/combo/ORB alerts
  (trader preference at build time).
- Combo signal scope: literally only current M5 Focus members, or Focus + the
  day's watchlists? (Default specced: Focus only.)
- ORB candidate tagging surface: Alert Center annotation vs a small ORB lane on
  the strength board (packet R2's board is a natural home).
