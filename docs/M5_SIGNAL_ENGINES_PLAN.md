# M5 signal engines — packet R5

Status: **IN PROGRESS** for `plan.md` Phase 0.5 **R5**. Authorized by the trader on
2026-08-15; ranked **last** among the feature packets (trader's ordering), and
R1–R4 have now landed. **§2's pure indicator modules are BUILT (2026-08-16); all
wiring is unbuilt and BLOCKED on the §8 Alert Center lane question — see §9.** Every engine here is new detector work: pure builders may land
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

Put to the trader on 2026-08-16, before any wiring was written.

| Question | Status |
|---|---|
| **Alert Center lane** — same `d1_flag` family, or a new tag for LRSI/combo/ORB alerts | **ANSWERED 2026-08-16 (Fable, trader-delegated; the trader may override): a NEW TAG FAMILY.** Full decision and its consequences in §8.1 below |
| **Combo signal scope** | **ANSWERED 2026-08-16: M5 Focus members only** — as specced, and as the trader originally framed it. A three-signal confluence run over the full watchlists would fire on names the trader never chose, which is the trash-volume problem R3 exists to reduce. Focus membership is the trader's explicit statement that they care about the name |
| **ORB candidate tagging surface** | **ANSWERED 2026-08-16: Alert Center annotation**, not a strength-board lane. The candidate is marked where its two follow-up alerts will fire, so the candidate and its follow-ups read as one story in one place, and the mark expires with the day like every other day-scoped annotation |

### 8.1 The lane decision — `M5_SIGNAL_TAG`, 2026-08-16

Decided by Fable under trader delegation; **the trader may override**. Recorded
here before any wiring, as the decision itself requires.

**One new tag family, `M5_SIGNAL_TAG`, for all three engines. `d1_flag` is not
reused.** Two structural reasons:

1. **Measurability.** §7's per-engine desk session *is* the promotion gate, and
   folding three unproven detectors into the champion D1 family makes "is this
   engine noisy?" unanswerable exactly when it has to be answered. The project
   rule is adopt-only-what-was-measured; reusing `d1_flag` forfeits the
   measurement.
2. **Semantics.** These are M5 intraday signals. `d1_flag` routes toward D1
   surfaces and carries champion privileges these engines have not earned.
   "Champions stay untouched" includes not diluting what their tag means.

Per-engine identity rides **`bounce_type`**, not the tag: each engine gets its
own `BOUNCE_TYPE_DEFAULTS`/`BOUNCE_TYPE_LABELS` entry, so the feed and History
can count each one separately. That plus the per-engine toggle satisfies §7's
measurement requirement without three families' worth of plumbing.

| Aspect | Decision |
|---|---|
| Feed routing | **Main feed.** The D1 Focus feed stays reserved for favorite/high-conviction D1 transitions |
| Tier gate | **No bypass.** These pass `alert_passes_min_tier` like ordinary alerts. The existing bypasses (`CHART_WATCH_TAG`, entry-assist, BANGER, PROVEN) are earned or trader-armed; these are neither, yet |
| Loudness | Follow §3/§4 where they state it. **Where unstated, default NOT loud for each engine's first desk session**, and revisit per engine once its volume is observed. Do not invent loudness the spec does not name |
| R4 §6.3 fold policy | **Not privileged — foldable and digest-eligible.** An unproven engine must be foldable, or it can spam precisely what the repetition ledger was built to stop |

**One correction to the reasoning, and it matters.** The decision notes that
confluence alerts, firing only on M5 Focus members, "inherit Focus-privilege
through membership, not through the tag". The mechanism is right — privilege
rides the *symbol*, never the tag, so no special-casing is needed. But the gate
is **stricter than membership**: `_alert_has_focus_privilege` is Focus
membership **AND** an open prev-day break on the alert's own side
(`alert_center_panel.py:1286-1300`; "A Focus long still inside yesterday's range
is ordinary: it competes on tier like any other name").

So a confluence alert on a Focus member that has *not* broken yesterday's
extreme on its side is **foldable and tier-gated**, exactly like any other
ordinary alert. That is more conservative than the decision assumed, and it
points the same way — do not "fix" it, and do not write "Focus member ⇒ never
folded" anywhere, because that is not what the code does.

## 9. Build state (2026-08-17)

**§3.1, the LRSI cross engine, is WIRED and green.** Built in this order:

1. `scripts/m5_signal_engines.py` — the pure seam between the indicator maths
   and the 11k-line detector. Bars in, events out; no clock, no I/O, no
   BounceBot import. It owns the three rules the call site has historically got
   wrong: completed bars only; the indicator warms **across** sessions while the
   event belongs to **one** (the `_evaluate_ema8_grind` precedent); and shorts
   mirror by negating price rather than inverting the test — the oscillator
   clamps at zero, so a falling name reads LOW, never negative, and there is no
   downward crossing that means the same thing. `latest_lrsi_cross` fires only
   on the most recently completed bar, so one crossing is one alert instead of
   one per scan cycle.
2. The detector wiring: `check_lrsi_cross_setups` beside the ORB and 8-EMA
   grind sweeps, on both the fast-lane and full-cycle hooks.

**The lane is built as §8.1 decided**: `M5_SIGNAL_TAG = "m5_signal"`, defined
once in `m5_signal_engines` because the detector cannot import from the UI and
both sides must agree. It replaces the `"green"`/`"red"` colour BounceBot passes
as the callback's second argument — safe because direction has always come from
the feedback block, and that is now asserted rather than assumed. The alert is
not D1, holds no chart-watch or entry-assist bypass, and passes the tier gate
like any other.

**The toggle map is separate on purpose.** `BOUNCE_LEARNING_TYPE_KEYS` is
derived from `BOUNCE_TYPE_DEFAULTS`, so adding engines on probation to that dict
would widen what the learning path treats as an established bounce type — a
scoring change smuggled in as a feature. `M5_SIGNAL_TYPE_DEFAULTS` is its own
map, and a taxonomy pin in `tests/test_r5_lrsi_cross_wiring.py` fails if anyone
ever tidies the two together.

**The packaging trigger fired and was discharged in the same commit**, exactly
as the allowlist entry promised: `indicators` left `PACKAGES_NOT_IN_THE_BUNDLE`,
joined the spec's `FIRST_PARTY_PACKAGES`, and `m5_signal_engines` plus the three
indicator modules joined `selftest.LAZY_ENGINE_MODULES`. The clean-cache rebuild
(`build/` **and** `dist/` deleted, run from the worktree) moved the frozen count
**51 → 55** — the movement is the proof it was not a cached reuse. Exe mtime
18:59:35 postdates the commit at 18:57:23. `laguerre_rsi` is deliberately absent
from the roster: collecting the package sweeps it in, but the selftest asserts
what is *reachable*, and it still has no importer.

Gate: **3590 passed / 19 subtests, exit 0; smoke 7/7, exit 0;
`selftest OK: 55/55 checks passed (frozen)`, exit 0.**

**What §3.1 has NOT earned.** §7's gate is per engine and it is live: the
confluence (§3.2) and first-candle ORB (§3.3) engines wire only after a desk
session confirms this engine's alert volume is sane. No such session has run.
§4's any-bounce watch is not behind that gate, but its prior-anchor AVWAP line
is an ask-first edit to the D1 scan output.

## 9.1 Prior build state (2026-08-16)

**§2 pure indicator modules: BUILT.** `scripts/indicators/smi.py`,
`efficiency_lrsi.py` and `heikin_ashi.py`, with 42 hand-computed tests in
`tests/test_indicators_r5.py`. Nothing imports them, so **no packaging trigger
has fired** — they stand exactly where `laguerre_rsi.py` does, which the
spec-drift allowlist already covers.

Two fixture defects were caught on the first run and are worth remembering,
because both are the failure mode R3's Amendment was written about:

- the SMI separate-smoothing test — the parity detail the formula most invites
  getting wrong — **passed vacuously on a clean linear ramp**, because a linear
  ramp has a constant 5-bar range, so the denominator is constant and dividing
  early or late agree exactly. Rebuilt on an expanding-and-contracting swing;
- the LRSI "already above the level" test asserted a crossing index that cannot
  exist, since a series with no measurable prior bar has no crossing to report.

**Everything else in this packet is unbuilt.** The lane question that blocked it
is **answered** (§8.1: a new `M5_SIGNAL_TAG` family, no tier bypass, foldable).
One thing still binds the wiring:

1. **§5's shared completed-bars helper must be reconciled, not re-invented.**
   `weekend_strength.completed_bars` already holds a correct intraday
   definition — `bar_start + bar_minutes <= now`, normalizing a tz-aware stamp
   with `astimezone(...)`. BounceBot's ad hoc idiom (`legacy.py:4384-4386` and
   `4533-4535`) uses `replace(tzinfo=None)`, which is the wrong spelling for a
   tz-aware stamp. The R5 helper must be **one** definition shared with
   `weekend_strength` and the strength board, on `astimezone`. It was
   deliberately **not** built ahead of its consumers: an unused abstraction
   would have been reconciled by guesswork, and §5 itself says existing call
   sites migrate opportunistically, never as a silent behavior change to a
   shipped detector.

**The packaging trigger fires the moment `scripts/indicators` gains its first
real importer**, which is §3's wiring. In that same change: the spec's
`collect_submodules` list, `tests/test_packaging_spec_drift.py`'s allowlist, the
`selftest.LAZY_ENGINE_MODULES` roster, and a frozen rebuild with **`build/` AND
`dist/` deleted first** — a count that does not move after a roster change is a
stale build, not a pass. Mind the disjointness rule: a package listed in
`PACKAGES_NOT_IN_THE_BUNDLE` may not also appear in the selftest roster.
