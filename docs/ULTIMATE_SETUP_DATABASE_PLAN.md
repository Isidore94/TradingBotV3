# TradingBotV3 Ultimate Setup Intelligence Database

## Status and authority

This document is the **locked, implementation-ready engineering plan** for the
long-term research-data architecture that determines which setup and trade style
is working best in the current market context. It supersedes the 2026-08-03
review draft: every formerly open question is now a decision in Section 23, and
an AI coding agent implements this plan top to bottom by following Section 19's
Phase 0-8 build order. Section 15 governs analysis, not build order.

It is subordinate to the root [`plan.md`](../plan.md) and enters the roadmap as
trader-directed Section 12 item **13a** (insertion text in Appendix E). It does
not authorize a detector, score, ranking, alert, or production-policy change.
Phases 0-8 are shadow-only additive evidence capture with read-only consumers —
zero champion influence, hence no golden fixtures are required by the capture
phases themselves. Promotion of anything remains governed by plan.md Section 7.

The plan incorporates every Markdown file under `docs/` as of 2026-08-03
(Appendix A maps each source document to its carried constraint).

## 1. Desired outcome

Build a durable, self-hosted research system that can answer, at any historical
or live decision time:

> Given the market, sector, industry, symbol, structure, important levels,
> timeframe alignment, liquidity, catalyst state, and available entry quality,
> which **trade style** has the best supported edge now—and how uncertain is
> that conclusion?

The answer is not merely a ticker or setup label. A trade style is the versioned
tuple:

```text
(setup family,
 side,
 structural timeframe,
 context timeframe(s),
 trigger timeframe,
 management timeframe,
 session scope (RTH/ETH/overnight),
 entry method,
 invalidation/stop method,
 target/management method,
 execution/fill model,
 cost model,
 portfolio/risk-policy version,
 expected holding horizon)
```

The system separately returns the best-supported:

- swing style;
- intraday quick-payoff style;
- intraday session-hold style;
- current opportunities matching each style;
- evidence against the conclusion;
- an honest abstention (`THIN_CONTEXT_EVIDENCE`, `INSUFFICIENT_CONTEXT_DATA`,
  or `NO_SUPPORTED_STYLE` — the Section 16.3 codes).

It must distinguish four questions that are currently too easy to mix:

1. **Pattern quality:** did the market structure predict a favorable move?
2. **Trade-policy quality:** did this entry, stop, and management recipe convert
   that move into attractive R?
3. **Selection quality:** did the bot or trader choose the right opportunities
   from what was available?
4. **Execution quality:** did the actual trade follow the plan and capture the
   available edge?

The objective is not to maximize trade count, win rate, or alert volume. It is
to improve point-in-time, out-of-sample opportunity selection while preserving
actionability and controlling downside.

## 2. Non-negotiable inherited rules

The database and every consumer must preserve these rules:

- TradingBotV3 remains decision-support only. No order execution is added.
- Existing live detectors and alerts remain champions until their challengers
  pass the documented promotion ladder.
- No detector or scoring change occurs without golden characterization fixtures
  first and reviewed intentional-difference fixtures second.
- `calc_anchored_vwap_bands` retains its frozen running-deviation sigma formula.
  Any alternative formula is a separately named research feature and can never
  silently replace the champion series.
- Only completed bars confirm a setup or advance a state. Forming bars may be
  captured as preview observations but are never confirmation evidence.
- Every timestamp is timezone-aware and tied to an exchange session.
- Point-in-time research uses only information whose applicable **as-observed or
  declared-market-basis** timestamp was no later than the decision time.
- Missing, stale, partial, conflicting, or late data is uncertainty, never
  confirmation.
- Opportunity, setup, anchor, level, attempt, trigger, alert, impression, trade,
  and outcome identities remain stable and distinct.
- Selected, rejected, quiet, never-triggered, passed, missed, and zero-trade
  cases are retained. The database must contain the denominator, not only wins
  and alerts.
- User-created watchlist names, horizontal levels, trendlines, and notes are
  historical facts. Automation cannot silently erase them.
- One named component owns each ingestion job, mutable ledger, and published
  snapshot.
- A failed write, compaction, migration, or publish never destroys the last
  verified artifact.
- The shared Drive folder remains for compact operational exports. It is not a
  multi-machine live research database.
- AI may explain evidence and propose research. It cannot mutate live scores,
  thresholds, detectors, watchlists, modes, alerts, or promotion state.
- `review_policy.json` remains annotation/ranking-only and gains no suppression
  field.
- The system may conclude that no setup is sufficiently supported today.

## 3. Core design principles

### 3.1 Preserve facts before interpretations

Raw observations, normalized bars, level geometry, feature values, setup
evaluations, ranking decisions, user actions, and outcomes are different facts.
They live in separate versioned datasets so a later formula change does not
rewrite history.

### 3.2 Store primitives; declare combinations

Do not materialize every possible EMA × SMA × AVWAP × band × level × timeframe
combination. Store reusable continuous primitives and atomic interactions, then
define setup hypotheses and trade recipes through a versioned registry.
Materialize only frequently queried, registered experiments.

### 3.3 Separate context, trigger, and management

A weekly leader, a 2nd-deviation power hold, or a bullish market regime may be
context. A completed M15 reclaim or M5 retest may be the trigger. The stop,
partials, trail, and expiry are a trade policy. The schema must not collapse
these into one ambiguous family name.

### 3.4 Continuous measurements before bins

Persist distance, slope, penetration, recovery, width, volume, age, streak, and
relative-strength values continuously. Human-friendly buckets are versioned
projections. This permits better thresholds later without refetching history.

### 3.5 One observation can support many correlated diagnostics

One setup occurrence may be evaluated under several time horizons or trade
recipes, but it remains one independent market episode. Alternative outcomes
are correlated diagnostics, not extra trades or extra sample size.

### 3.6 "Best now" must include uncertainty and abstention

Every result reports independent episodes, distinct sessions and symbols,
missingness, confidence interval, evidence freshness, and its evidence tier
(Section 15.8). Thin exact matches shrink toward broader priors. The engine
abstains with `THIN_CONTEXT_EVIDENCE` when the current context cell has too few
matured episodes to support a claim.

### 3.7 The research corpus may be large; authority must stay simple

Aaron retains large files on a DAS. The plan favors preservation, replayability,
and immutable evidence over aggressive deletion. This does not justify multiple
writers, tiny-file sprawl, unchecked corruption, or a DAS with no independent
backup.

## 4. Information layers

The target system has seven layers:

```text
Bronze  Immutable source observations and ingestion manifests
  ↓
Silver  Normalized bars, sessions, events, anchors, levels, and geometry
  ↓
Feature Point-in-time continuous features and atomic interactions
  ↓
Setup   Versioned hypotheses, eligibility, occurrences, and lifecycles
  ↓
Style   Candidate × entry/stop/management simulations and outcomes
  ↓
Gold    Research marts, conditional expectancy, ranking, and dashboards
  ↓
Control Experiments, evidence freezes, promotions, policies, and rollbacks
```

Operational GUI/phone artifacts are small, immutable projections from a saved
canonical snapshot. They do not independently recompute research results.

## 5. Market data, capture policy, and IB pacing

### 5.1 Provider limits — the binding constraint

Disk is not the constraint; IBKR pacing is. All numbers below are **published
conservative floors, not measured capacity**:

| Constraint | Published value | Note |
|---|---|---|
| IBKR historical requests | ~60 req / 10 min (≈6/min) | Soft-throttled; this installation demonstrably sustains far more (see 5.2) |
| IBKR identical-request cooldown | 15 s | Dedupe identical (symbol, timeframe, window) requests |
| IBKR market-data lines | ~100 concurrent | Shared by champion subscriptions + capture streams |
| IBKR pacing errors | 162 / 366 | Capture yields instantly; tagged, never counted against the champion circuit breaker |
| TWS auto-restart | ~23:45 ET nightly | Overnight jobs must be idempotent and reconnect-tolerant across it |
| yfinance 1m window | ≈7 days | Cannot recover premarket M1 history later |
| yfinance 5m-30m window | ≈60 days | Basis of the one-time M5 seed |
| yfinance 1h window | ≈730 days | |
| yfinance throttle | Unofficial ban risk | Seed is trickled over several nights, chunked with backoff — never one bulk scrape |

The architecture (tee-first, priority classes, cohort mechanisms, aggregation
rules, ETH scope, the H2 cut) is fixed from these floors. Every numeric
allocation (capture req/min, M1 cap, weekly-vs-nightly sweep, stream ceiling)
starts at the floor and expands only on pilot measurement (Section 5.6).

### 5.2 Capture policy — resolution × cohort × acquisition mode

**Capture-by-interception first.** Phase-3 capture tees provider responses the
master scan (D1, full universe) and BounceBot ("5 D/5 mins" M5, watchlist
cohort) already fetch into bronze at **zero added IB cost**; net-new request
volume begins only after the tee is proven. A one-time yfinance 60-day
full-universe M5 seed (provider=YAHOO, capture_mode=BACKFILL) is trickled over
several nights.

| Resolution | Cohort | Mechanism | Cadence | Session scope | capture_mode | Provider cost |
|---|---|---|---|---|---|---|
| Quote/last | Armed watches (champion) | Existing services, pass-through | Live | RTH per production | LIVE | 0 added (champion) |
| M1 live | Focus ≤40 (hard ceiling 60) | `reqRealTimeBars` 5-sec stream, post-slice | Continuous | ETH 04:00-20:00 | LIVE | Lines, not hist requests |
| M1 backfill | Focus + fixed exploration, ~150 syms (→300 on measured headroom) | Nightly historical, `useRTH=0` | Nightly | ETH | BACKFILL | ~150 req ≈ 25 min at floor |
| M1 deep history | Focus only, 1-yr depth | Multi-night trickle, lowest priority | As headroom allows | ETH | BACKFILL | Residual budget only |
| M5 tee (slice) | Watchlist union ≤250 + fixed 30 exploration | Tee of BounceBot "5 D/5 mins" fetches | Every production cycle | Production scope; nightly backfill adds ETH | LIVE/DELAYED | **0 added** |
| M5 full universe | ~1,539 symbols | Weekly Saturday "1 W" sweep + nightly active-cohort gap backfill | Weekly floor; nightly-universe upgrade expected on measured headroom | ETH | BACKFILL | 1,539 req ≈ 4.3 h at floor |
| M5 seed | Full universe, 60 days | One-time yfinance, trickled | Once | Provider scope | BACKFILL (provider=YAHOO) | 0 IB |
| M15/M30/H1(/H4) | Everything M5 covers | Deterministic derivation from M5 | EOD build job | RTH-only aggregates v1 | (computed) | 0 |
| H2 | — | **CUT** (no consumer anywhere in `scripts/`) | — | — | — | 0 |
| D1 | Full universe | Existing master-scan store/tee (champion path) | Per scan | Session close | LIVE/DELAYED | 0 added |
| W1 | Full universe | Derived from canonical D1 (exchange week) | EOD build job | — | (computed) | 0 (saves ~1,539 req/refresh) |
| Exploration cohorts | Fixed 50 + rotating 100/night (post-slice; seeded RNG logged) | BACKFILL only — never live | Nightly | ETH | BACKFILL | Inside the nightly budget |

Intraday absence for non-streamed cohorts is recorded as
`NOT_COLLECTED_BY_POLICY`, never `MISSING`. "Every completed interval" cadence
applies only to the streamed Focus cohort. Exploration cohorts are BACKFILL-only
by design: backfilled OHLCV is identical evidence for discovery, and
capture_mode segregates the evidence class — removing champion-selection bias
without spending live lines. Exploration symbols are **never added to the
champion scan cohort** (Section 19.6 R3): they are captured via the tee when a
symbol happens to be scanned anyway, otherwise via the nightly BACKFILL budget.

Exploration cohort reconciliation: the slice uses the fixed 30-symbol list
(`exploration_cohort.txt`); post-slice the fixed set expands to 50 (a superset
of the 30) and the rotating 100/night joins. The M1 nightly-backfill cohort of
~150 = Focus ≤40 + fixed exploration 50 + ~60 headroom for recently active
symbols and trickle.

**The arithmetic behind the table:**

- **Floor vs measured.** Published floor ≈ 6 req/min. Measured: the full-universe
  ~1,539-symbol D1 scan completes in ≈28.5 min network-bound ⇒ ≈54 req/min
  sustained, ~9x the floor, while BounceBot concurrently re-polls up to ~200
  watchlist symbols. Floors are safe starting allocations, not capacity truth;
  the pilot measures the real ceiling.
- **Overnight window.** 20:00-02:00 ET minus the ~23:45 TWS restart ≈ 5.5 usable
  hours ⇒ ~1,980 requests at floor. Nightly plan at floor: M1 backfill 150 +
  active-cohort M5 gap backfill ~200 (initial value — pilot confirms) + sentinel
  parity 20 + trickle remainder — comfortably inside budget even at the floor.
- **Weekly sweep.** 1,539 "1 W" M5 requests ≈ 4.3 h at floor — fits Saturday.
  Nightly full-universe M5 fits only if measured headroom ≈2x floor confirms;
  that upgrade is expected, not exceptional.
- **Polling infeasibility.** Live M5 polling for N symbols = N req per 5 min
  against a 30-req/5-min floor ⇒ ~30 symbols saturate the entire budget and
  collide head-on with champion fetches. Hence streaming (lines) for live
  capture; per-interval historical polling for live capture is rejected.
- **Streams.** Focus ≤40 lines of ~100 leaves ≥60 lines for champion
  subscriptions (initial split — pilot confirms the real line cap).
- **M1 is never required for M15/M30.** 15 and 30 are exact multiples of 5; M5
  derives both losslessly. M1 exists for entry-timing fidelity on the Focus
  cohort, not for aggregation.

### 5.3 Shared IB pacer — champions are never metered

One process-wide arbiter. Champion traffic (master scan, BounceBot, armed
watches) is pass-through and observed-only: counted, never delayed, never queued
— metering champions would change live champion timing without golden fixtures,
violating plan.md Section 5. Capture traffic runs in a token bucket initialized
to the published floor minus observed champion consumption (during RTH
effectively ~10-15 req/10 min for capture; initial value — pilot confirms),
yields instantly to any champion activity and on error 162/366, and grows only
via pilot-measured headroom.

A capture-caused pacing error must NEVER count against the champion fetch
boundary's Yahoo-only circuit breaker
(`scripts/master_avwap_lib/legacy.py:1653-1662, 2272-2292`) — capture errors are
tagged at the request layer and excluded from `_IBKR_HISTORICAL_FAILURE_COUNT`.

Client-ID allocation (asserted at connect; reconciled with the plan.md M1
dual-scheduler fix): retire colliding **1003**; **1010** capture streamer;
**1011** nightly backfill; **1020-1029** reserved for future mini-PC bundle
production. Overnight jobs are idempotent and reconnect-tolerant across the TWS
auto-restart.

### 5.4 Aggregation contracts and the per-bar record

Every derived bar carries an `aggregation_contract_id` defining session-open
alignment, timezone/DST rules, RTH/ETH segmentation, half days, constituent
expectations, and final partial-bucket policy. The v1 RTH contract
(09:30-16:00 ET, session-anchored):

| Derived frame | Bars per full session | Rule |
|---|---|---|
| M15 | 26 | 3 consecutive M5 constituents, session-anchored |
| M30 | 13 | 6 consecutive M5 constituents, session-anchored |
| H1 | 7 (6 full + one 30-min stub) | Session-anchored; boundaries match IB native `useRTH=1` H1 so sentinel derived-vs-native parity is checkable; the 15:30-16:00 stub is flagged with its true duration and never compared with full bars as equivalent |
| H4 (tier-2) | 2 (one 4-h + one 2.5-h stub) | Same stub semantics |
| W1 | 1 per exchange week | Derived from canonical D1; completes at the week's final session close; short weeks flagged; provider-native W1 is a validation variant only |

Half-day sessions use the half-day variant of the same contract. `NO_TRADE`,
`HALTED`, `NOT_LISTED`, `OUTSIDE_SESSION`, and provider-missing intervals remain
distinct. Provider-native D1 bars are canonical; intraday-derived D1 is a
validation variant, never a silent replacement. Derived aggregates are RTH-only
in v1 (session-scope column kept, so ETH variants are additive
`aggregation_contract_id`s later); raw M1/M5 capture is ETH-inclusive
(04:00-20:00 ET) and session-tagged PRE/RTH/POST from day one, because premarket
extremes are first-class levels in the trader's workflow and ETH history not
captured forward is permanently lost.

Every bar records: symbol, exchange session ID and phase, timeframe, interval
start/end, OHLCV plus optional provider VWAP/trade-count, completed/preview
state, raw-vs-adjusted semantics, provider and fallback classification, quality
state, content hash, schema version, run ID, and the five point-in-time columns
of Section 9.3 (`event_at`, `observed_at`, `computed_at` on derived records,
`capture_mode`, revision chain).

At any decision time, every higher timeframe joins only to its last completed
and available bar. A forming higher-timeframe value may be stored under a
separate preview feature ID and contributes zero to confirmation.

Cohort membership is recorded point-in-time: in the slice, daily watchlist/focus
/exploration snapshots in `universe_membership_daily` (Section 7); post-slice,
the forward-declared `collection_universe_membership` adds resolution-scoped
cohorts, selector versions, and assignment provenance for streamed and rotating
cohorts. `NOT_COLLECTED_BY_POLICY` is always distinct from `MISSING`,
`NO_RESPONSE`, and `TIMED_OUT`.

### 5.5 Timeframe roles

Every setup definition declares roles rather than assuming D1/M5:

- **Regime timeframe:** broad market backdrop, often D1/W1 or a session episode.
- **Structural timeframe:** where the thesis and major risk levels live.
- **Context timeframe:** one or more frames that qualify or conflict with it.
- **Trigger timeframe:** where the executable completed-bar event occurs.
- **Management timeframe:** the cadence used for stop, trail, and expiry rules.

This supports, for example:

```text
D1 earnings-AVWAP thesis
  + H4/H1 rising structure
  + M30 controlled pullback
  + M15 reclaim
  + M5 retest entry
```

without treating each rescan as a new setup.

**M15/M30 are the trader's explicitly requested new ground** and lead the
research queue (Section 19.4, LD-23). **H1 is confirm-only context** since
2026-07-17: H1 alerts were retired to learning-only as the worst quick producers
in the tracker; `h1_riding_15ema` is a PROVEN-eligible trait confirming D1
picks, never a trigger family. H1 trigger studies rank below M15/M30.

### 5.6 The 20-session pilot — confirmation and measurement, never design

The pilot is an engineering confirmation, shakedown, and measurement phase. It
is never a design phase and never a schema-design phase. Capture design,
cohorts, and cadences are fixed by this document and adjusted only if
measurements contradict assumptions. Pilot sessions are engineering validation,
never efficacy evidence. Checklist:

1. Tee adds zero provider requests (request-count assertion around a full scan).
2. Measured sustained req/min without error 162 while champions run.
3. Actual concurrent market-data line cap (measured with the post-slice Focus
   streaming milestone if no streaming prototype exists during the slice).
4. Stream-vs-poll M5 parity on a 5-symbol overlap (same staging as item 3).
5. Idempotent resume across the ~23:45 TWS restart (no duplicate, no hole).
6. Measured bytes/row by dataset (feeds the Section 8.3 sizing revision).
7. DAS/backup health: seal, manifest, Class A/B copies, scripted restore check.
8. Optional non-blocking data-rights confirmation question to Aaron (LD-14).

**Compute budget (anti-over-engineering statement):** nightly incremental
feature pass ~2-5 min; full 1-yr M5 feature recompute ~7-26 min wall on 5
workers of the i5-8600K. No distributed compute; no second machine for feature
builds.
## 6. Technical feature universe

The following is a research universe, not an authorization to score every
combination. Each family requires a registered definition, point-in-time tests,
and correlation-aware analysis.

### 6.1 Moving averages — three tiers

The grid is tiered by actual use, not symmetry. Every tier-1 entry has a named
consumer in `SETUPS_MAJOR`/`SETUPS_TEST` or shipped code; everything cut has
none.

| Tier | Series | Status |
|---|---|---|
| **Tier-1 (wired first)** | M5 EMA8/15/21; H1 EMA10/15 + SMA20; H4 EMA15; D1 EMA8/15/21 + SMA50/100/200; W1 EMA8/15 + SMA50/100 — the 17 trader-actual series — **plus** the six new M15/M30 EMA8/15/21 | Bars captured/derived from Phases 3-4. Phase 5 computes exactly the frozen Section 7.1 snapshot columns (D1 grid, M5 EMAs, M15/M30 EMAs, AVWAP block); the H1/H4/W1 tier-1 series activate post-slice via a `feature_set_version` bump and additive columns — at latest before milestone M-D begins |
| **Tier-2 (cheap-capture research margin)** | M15/M30 EMA10/50 + SMA20/50; H1 EMA8/21/50 + SMA50; D1 EMA50 + SMA20; M5 SMA20/50 | Captured continuously, studied only through registered experiments |
| **CUT (no feature capture)** | All H2 series; intraday SMA100/200; H4 SMAs; W1 SMA20/200 | Remain recomputable later — canonical bars are still archived per Section 5 |

The chart-template inventory is a **confirm-or-amend step** on this default
(Section 23 confirmation register), never a blocker.

For each tier-1/tier-2 period/timeframe pair, persist:

- level value and calculation version;
- price distance in dollars, percent, ATR, and local band-width units;
- slope over registered lookbacks, slope acceleration, and slope percentile;
- price-side state and consecutive completed-bar residence;
- ordering/stack state among related averages;
- separation and compression/expansion percentile;
- approach direction and velocity;
- wick tag, close tag, pierce, rejection, reclaim, cross, hold, acceptance,
  failure, first retest, later retest, and role reversal;
- touch count, time since last touch/break, and bars since reclaim;
- volume, relative strength, and candle-quality evidence on the interaction.

Feature identity includes close/typical-price input, EMA seed and warm-up,
raw/adjusted basis, RTH/ETH scope, minimum history, and exact aggregation
contract.

Near-identical periods are correlated siblings, not independent votes. They
share an evidence-family contribution cap in any future ranking. H1 series are
confirm-only context (Section 5.5).

### 6.2 Event/manual AVWAP and deviation bands

Champion anchors remain current and previous earnings anchors with the frozen
running-deviation bands at ±1/2/3 sigma.

**The 1σ band is the PRIMARY AVWAP entry structure.** `SETUPS_MAJOR` names
AVWAPE→UPPER_1 "the favorite zone — the measured best-edge zone," with band-2
partial and band-3 runner as management. The tier-1 favorite-zone feature block:

- `favorite_zone_coord` = (close − AVWAPE) / (UPPER_1 − AVWAPE), mirrored for
  shorts;
- `favorite_zone_residence_bars`;
- `first_dev_touch_order`;
- `band1_rejection_strength`;
- `second_band_streak`.

2σ/3σ bands persist as target/management levels and as the 2nd-deviation
power-hold **context episode** — not equal-priority entry structures. Zone
residency alone remains a WATCH state per Section 6.9: these are features
feeding completed-bar triggers, never triggers themselves.

**Anchor registry scope (LD-09):**

| Tier | Anchors |
|---|---|
| Slice | Current/previous earnings only, sourced from `earnings_avwap_anchors.csv` + calendar history |
| Tier-1, immediately post-slice | Post-earnings-candle anchor (flagship play); manual anchors |
| Tier-2, later | Gap/catalyst bar; confirmed swing pivot; period opens (W/M/Q/Y) |
| CUT | 52-week-event, breakout-bar, volume-thrust anchors (no current consumer; additive registry entries if a registered study ever names one) |

HOD/LOD live-reanchored variants stay in the session-VWAP namespace (6.3).

Every anchor records its source bar/event, why it qualified, when it became
knowable, effective range, revision chain, expiry, base-bar resolution,
provider, formula version, and creator. A pivot anchor becomes available when
the pivot is confirmed—not retroactively at the pivot bar.

The anchor registry publishes a semantic dictionary for current earnings,
previous earnings, pre-earnings, post-earnings, and earnings-candle anchors **as
implemented in code**. These names are not interchangeable. The Pre/Post-
Earnings AVWAPE anchor semantics (candle before the gap vs post-earnings — one
code family, one canonical ID) is an explicit dictionary item on the Section 23
confirmation register. Feature identity includes anchor-bar inclusion/exclusion,
price basis, raw/adjusted inputs, volume/session scope, base resolution, and
frozen sigma version.

For each AVWAP and band, persist: value, width, width slope and percentile;
continuous band coordinate and named zone; zone residence and excursions;
distance/age from anchor; first/second/nth approach and touch; penetration
depth, close recovery, rejection strength, and follow-through; cross/chase,
break-hold, break-retest, bounce, failed break/reclaim, compression-expansion,
power hold, and extreme-move retest interactions; confluence with MAs,
horizontals, trendlines, and independently sourced anchors; nearest opposing
obstacle and available reward/risk.

An AVWAP computed from D1 bars and one computed from M1/M5 bars are distinct
features: aggregation changes typical-price and running-deviation paths, so base
resolution is always part of feature identity.

### 6.3 Session VWAP families

Model reset-based and live-reanchored intraday VWAP separately from persistent
event/manual AVWAP. **Tier-1 session-VWAP definitions freeze at exactly the
current three production algorithms** — standard, dynamic, and EOD, plus the
HOD/LOD live-reanchored variants already in this namespace — **with ±1σ bands
only**, matching current intraday code. Additional sigma tiers (±2/3σ intraday)
enter only through the feature registry as named research features, never as
silent extensions of migrated champions.

A `session_vwap_definition` records reset boundary or re-anchor event, RTH/ETH
scope, price basis, base resolution, volume eligibility, algorithm, band-sigma
method, and formula version. Each newly known HOD/LOD extreme creates an
immutable revision.

The session-open event-AVWAP label is an alias of standard session VWAP whenever
the frozen definitions are mathematically identical; it never creates a second
feature or confluence vote. Two VWAP labels derived from the same observations
are correlated provenance, not independent confluence. A genuinely persistent
event/manual AVWAP and a reset-based session VWAP never share an ID.

### 6.4 Horizontal levels

Horizontal levels are first-class, versioned entities—not disposable chart
decorations.

| Family | Examples |
|---|---|
| Session | prior close/open, PDH/PDL, overnight/premarket high/low, opening-range 5/15/30/60, HOD/LOD |
| Calendar | prior week/month/quarter/year OHLC, period opens, 52-week extremes |
| Rolling | 5-day high/low, configurable N-bar and 10-candle high/low |
| Catalyst | earnings-candle high/low, gap edges/midpoint, catalyst-bar extremes, high-volume-bar extremes |
| Structure | confirmed swing pivots, consolidation/range boundaries, multi-touch S/R, breakout shelf, failed-break level |
| Statistical | AVWAP/bands, VWAP/bands, moving averages, volatility envelopes |
| Human | trader-entered line, zone, planned entry, invalidation, obstacle, or target |
| Reference | round-number and explicitly configured psychological levels |

**Every plan concept maps to an existing module to migrate or wrap — reimplement
nothing:**

| Existing system | Role here |
|---|---|
| HV S/R level stores (~1,539 symbols × 5 yr as of 2026-07, refreshed every scan) | Primary horizontal-store source, wrapped as bronze |
| `d1_level_feed` (SMA50/100/200 + trendlines + horizontal stores at 30-min resolution) | D1 major-level source for `level_state_daily` |
| Shipped blocking-penalty logic (0.5-ATR entry-path window, −10 max, evidence-promoted 2026-07-06) | Interaction-feature precedent; its inputs become level features |
| `technical_integrity_events.jsonl` | Declared a bronze source (enables its pending retention cleanup after verified ingestion) |
| `master_avwap_bucket_state.py` | Zone/band state source |
| `bounce_bot_lib` session/rolling levels (PDH/PDL, 10-candle, ORB) | Intraday level source |

No missing level family was found in review; the families above are complete
against the trader's actual usage.

Each level definition stores: `level_id`, family, subtype, symbol, source
timeframe; exact price or zone bounds and adjustment space; original source
bar/event/pivots; algorithm/parameters/version or human creator; `created_at`,
`known_at`, validity interval, expiry, invalidation; strength components, touch
count, role history; revision/supersession chain; and user notes without
conflating them with setup confirmation.

Support/resistance role is an as-of episode, not an immutable definition field.
Statistical levels reference their source MA/VWAP/AVWAP identity rather than
duplicating a copied level with fake independent provenance. Generated
multi-touch levels become historically available only after the required touches
complete. Manual lines exist for research only from creation time. Editing
creates a revision; deletion closes the validity interval but preserves history.
A corporate action never silently moves history.

For every approach/interaction, capture: ATR/percent/dollar distance and
approach velocity; wick penetration and close displacement; volume and relative
volume at touch; first versus repeated test and time since prior test; hold,
rejection, close break, acceptance, retest, false break, reclaim, role reversal;
next opposing level and space between clusters; and whether a last-price alert
fired separately from a completed-bar event.

Every interaction identifies `level_source_timeframe` and
`observation_timeframe`, plus a distinct trigger timeframe when applicable.
Touch order is scoped by level lifecycle, observation timeframe, and an explicit
reset rule so M5 touches cannot contaminate "first M15 touch" research. The
observation is typed `QUOTE_CROSS`, `BAR_HIGH_LOW_TOUCH`,
`COMPLETED_CLOSE_BREAK`, `ACCEPTANCE`, or `RETEST_HOLD`; these are not
interchangeable evidence, including for Post-Earnings Candle Break.

### 6.5 Trendlines

Create a versioned trendline registry for both human and algorithmic lines.
Each line stores: `trendline_id`, symbol, direction, source timeframe/type; two
or more stable pivot IDs, their anchor prices, and when every pivot became
knowable; chart scale (`LINEAR`/`LOG`) and price space; fitting method,
equation, projection domain (`SEGMENT`/`RAY`/`INFINITE`), ATR-normalized slope,
fit residual, touch count; creation time, validity interval, invalidation rule,
revision chain; corporate-action transformation rule; projected value at every
eligible completed-bar timestamp; and the same interaction events as horizontal
levels. A line fitted with future pivots cannot be projected backward into the
research set. Trendline interactions carry the same source-timeframe,
observation-timeframe, touch-order, and reset semantics as horizontal levels.

**Capture path (LD-08):** the slice bronze-ingests the existing
`d1_level_watches.json`, `d1_event_watches.json`, `alert_chart_watches.json`,
and `price_alerts.json` daily from creation time onward — trader geometry is
already persisted with arm timestamps, satisfying known-at semantics with zero
new UI. The pyqtgraph draw/edit/delete capture surface + bulk-entry form is a
named **post-slice deliverable (old WS4; carried in Section 19.4)**, required
before any manual-trendline research family activates; until then algorithmic trendlines
remain research-only and manual-trendline datasets stay empty rather than
blocking.

### 6.6 Confluence and distinct provenance

At each decision time, cluster nearby active levels using a versioned ATR- or
volatility-normalized distance. Store a `level_cluster_id`, members, sources,
timeframes, spread, strongest member, nearest opposing cluster, and their
`distinct_provenance_family` values. Different provenance does not prove
statistical independence; redundancy is estimated on training data only.

Examples of distinct-provenance confluence:

- D1 earnings AVWAP + H1 SMA20 + prior-week high;
- W1 EMA15 + D1 SMA50 + human support zone;
- M30 opening-range edge + M5 VWAP retest.

EMA15 and EMA21 on the same timeframe, or the same level emitted by two
adapters, are correlated evidence and must not receive two full votes.

### 6.7 Price action, volume, and participation

Store continuous, point-in-time features for:

- candle range/body/wicks, close location, gap, inside/outside bars, and
  displacement in ATR;
- range and volatility compression/expansion over registered lookbacks;
- realized volatility and ATR level/percentile;
- volume, relative volume, dollar volume, volume trend, thrust, dry-up,
  participation proxies, and volume on level interaction;
- liquidity, spread/quote quality when available, price, and optionability
  metadata used by the universe;
- opening drive, controlled pullback, impulse/retrace proportions, and time to
  resumption;
- HOD/LOD grind persistence and new-extreme frequency;
- extension from structure and anti-chase state;
- `session_structure_gate` state (the existing BounceBot chop veto, migrated —
  never reinvented);
- `pullback_count_in_current_leg`, so the trader's ONE-simple-retest philosophy
  is expressible as measured touch-order/pullback-count edge.

### 6.8 Market, sector, industry, catalyst, and clock context

Store both continuous context values and versioned regime labels. **The six
shipped production context systems are tier-1 and are migrated, not
reinvented:**

- `rvol_tc2000` + `rvol_gate_pass` (≥1.0 gates ALL M5 bounce alerts in
  production; sub-1.0 rows are learning-only but always recorded — the complete
  denominator);
- `rs_rw_vs_spy` anchored to the day's OPENING regime (`rvol.py`);
- `group_rs_debiased` (per-alert group RS de-biasing, 2026-07-22);
- `market_internals_v1` (VXX/HYG/TLT/RSP/MAGS + measured-negative-environment
  flag; evidence: 49% of alerts fired into a measured-negative environment);
- `session_structure_gate` (6.7);
- legacy SPY champion state and shadow `market_state` output, stored separately.

Plus: SPY pullback/rebound episode and exact aligned interval; sector/industry
membership as known on the decision date; market/sector trend, volatility, gap,
breadth, and participation; earnings timing, earnings gap/candle, catalogued
catalysts, days since event; day of week, month/quarter boundary, time since
open, time-of-day bucket, session phase, shortened-session status; operating
mode (DESK/AWAY/EVENING) as presentation context only, never a reason to omit an
observed opportunity.

Regime labels are calculated using only data available at the time. Store the
underlying continuous vector so a later regime definition can be replayed
without rewriting the original snapshot.

### 6.9 Existing setup ontology that must survive migration

The warehouse must reproduce these existing distinctions before it invents new
ones:

- Swing: AVWAPE-to-1st-Dev Favorite, AVWAP Retest Followthrough, AVWAP
  Breakout, AVWAP Band Bounce, Extreme Move Retest, SMA50/100/200 Breakout and
  Retest, and TOP Weekly Leader with its separate daily trigger.
- Earnings cycle: Post-Earnings Candle Break, Post-Earnings 52-week Break,
  Post-Earnings AVWAPE Bounce, and Mid-Earnings EMA15/EMA21/1st-Dev Retests
  following a 2nd-deviation-zone episode.
- Intraday: standard/dynamic/EOD VWAP and ±1-sigma bands, VWAP/EOD confluence,
  impulse retest, EMA8/15/21, rolling 10-candle levels, PDH/PDL, H1 EMA10 and
  color-state patterns, regime-pause RS/RW, 30-minute-plus ORB breaks, and
  EMA8 HOD/LOD grinds.
- Research/study: Weekly 8EMA Hold and Retest, H1/H4 EMA15 Rejection, 1st- and
  2nd-Dev Breakout controls, Volume Thrust, 2nd-Dev Power Hold, Quiet Pullback
  Resume, Golden Pullback plus Volume, Post-Earnings Volume Break, and every
  current playbook/forensics family with its original definition version.

`Favorite Zone Watch`, weekly leadership, power-hold residence, and similar
states remain context/watch conditions unless a separately defined completed-
bar trigger occurs. Move-Forensics lift remains association, not trade edge.
"Banger" remains excluded from schemas until Aaron supplies a precise definition
(LD-27); the migration must not infer one.

## 7. Canonical datasets and identities

Avoid one enormous flat CSV. Use typed Parquet datasets with stable schemas and
purpose-built materialized views.

### 7.1 First-increment schemas (frozen now)

The first buildable increment is exactly **13 tables plus two JSONL ledgers**.
The typed pyarrow definitions in `scripts/research_warehouse/schemas.py` are the
single source of truth; the listings below are their normative documentation.

Conventions: all enums are strings (widening never rewrites files); `symbol` is
the natural key until the first real rename adds a `symbol_alias` table;
timestamps are UTC with explicit timezone; every record carries the five PIT
columns where applicable (`event_at`, `observed_at`, `computed_at` on derived
records, `capture_mode`, `revision_id`/`supersedes_revision_id`) plus
`schema_version` and `run_id` — repeated below only where their meaning is
dataset-specific. Partition spec (locked): one file per (dataset, timeframe,
month); M1 additionally 8 symbol-hash buckets; D1/W1 and small reference
datasets per (dataset, year).

**`trading_session`** — partition: (year)

- `session_id`: string — calendar + date key, e.g. `XNYS-2026-08-03`.
- `exchange_calendar`: string — calendar identity and version.
- `session_date`: date — exchange-local trading date.
- `rth_open_at` / `rth_close_at`: timestamp — regular-session boundaries.
- `eth_open_at` / `eth_close_at`: timestamp — 04:00/20:00 ET as applicable.
- `is_half_day`: bool — drives the half-day aggregation variant.
- `expected_m5_bars_rth`: int32 — coverage denominator for Health tile 3.
- `expected_m1_bars_rth`: int32 — same, M1 (post-slice consumer).
- `calendar_version`: string — bumped when the calendar source revises.

**`bar_m5`** — partition: (month); grain: symbol × interval_start × provider ×
revision

- `symbol`: string — displayed symbol (natural key).
- `interval_start` / `interval_end`: timestamp — completed-interval boundaries.
- `session_id`: string — FK to `trading_session`.
- `session_phase`: string — PRE | RTH | POST.
- `open` / `high` / `low` / `close`: float64 — raw (unadjusted) prices.
- `volume`: int64 — share volume as provided.
- `vwap`: float64 (nullable) — provider bar VWAP when supplied.
- `trade_count`: int32 (nullable) — when supplied.
- `provider`: string — IBKR | YAHOO; never blended without a transition record.
- `is_complete`: bool — completed bars only confirm; forming is preview.
- `quality`: string — slice subset per Section 9.4.
- `event_at`: timestamp — ≡ interval_end (market fact time).
- `observed_at`: timestamp — when this installation received the bar.
- `capture_mode`: string — LIVE | DELAYED | BACKFILL | RECONSTRUCTED.
- `source_hash`: string — content hash of the source response row.

**`bar_d1`** — partition: (year); grain: symbol × session × provider × revision

- `symbol`: string; `session_id`: string; `session_date`: date.
- `open` / `high` / `low` / `close`: float64 — raw prices, never silently
  adjusted.
- `volume`: int64 — as provided (the ×100 round-lot bug is a sentinel check,
  not a rewrite).
- `adjustment_version`: string (nullable) — corporate-action view applied, if
  any.
- `corporate_action_id`: string (nullable) — FK when an action affects this
  bar's view.
- `provider`: string; `quality`: string; `is_complete`: bool.
- `event_at` / `observed_at` / `capture_mode` / revision chain — per convention.

**`bar_derived`** — partition: (timeframe, month); grain: symbol × timeframe ×
interval_start × aggregation_contract_id

- `symbol`: string; `timeframe`: string — M15 | M30 | H1 | H4 | W1.
- `aggregation_contract_id`: string — the Section 5.4 contract that produced it.
- `interval_start` / `interval_end`: timestamp; `session_id`: string.
- `open` / `high` / `low` / `close`: float64; `volume`: int64.
- `is_stub`: bool; `stub_duration_min`: int32 (nullable) — end-of-session stubs
  carry their true duration and are never compared with full bars as equivalent.
- `constituent_count` / `constituent_expected`: int32 — completeness evidence.
- `is_complete`: bool; `quality`: string.
- `event_at`: timestamp (≡ interval_end); `computed_at`: timestamp;
  `input_capture_mode_worst`: string — worst constituent mode.

Phase 4 populates M15/M30/H1 (from M5) and W1 (from canonical D1); H4
derivation activates post-slice with the H4 feature series.

**`universe_membership_daily`** — partition: (year); grain: session_date ×
list_name × symbol

- `session_date`: date — snapshot day.
- `list_name`: string — universe_all | longs | shorts | autolongs | autoshorts |
  swinglongs | shortswings | focus | exploration_fixed.
- `symbol`: string.
- `rank_in_list`: int32 (nullable) — file order where meaningful.
- `inclusion_reason`: string (nullable) — auto-populate score band, manual, etc.
- `snapshot_at`: timestamp — first-capture time (≡ observed_at); never
  backfilled.

**`anchor_instance`** — partition: (year); bitemporal (a revisable reference
dataset, Section 9.5)

- `anchor_instance_id`: string — deterministic hash(symbol, anchor_type,
  anchor_bar_date, formula_version).
- `symbol`: string.
- `anchor_type`: string — slice: EARNINGS_CURRENT | EARNINGS_PREVIOUS (additive
  later per LD-09).
- `anchor_bar_date`: date — the anchor bar's session date.
- `catalyst_event_id`: string (nullable) — FK to the sourcing earnings event.
- `price_basis`: string — as implemented in code (semantic-dictionary item).
- `anchor_bar_included`: bool — inclusion/exclusion is part of feature identity.
- `formula_version`: string — running-deviation σ variant; never swapped
  (Section 2 invariant).
- `source`: string — earnings_avwap_anchors.csv | calendar_history.
- `valid_from` / `valid_to`: timestamp — market-validity interval.
- `system_from` / `system_to`: timestamp — knowledge interval (bitemporal).

**`level_state_daily`** — partition: (year); grain: symbol × level_id ×
session_date

- `symbol`: string; `session_date`: date.
- `level_id`: string — stable identity from the source store.
- `level_family`: string — SESSION (PDH/PDL, prior close) | HORIZONTAL_STORE |
  MA_LEVEL | TRENDLINE | WATCH_JSON. (TRENDLINE rows store the `d1_level_feed`
  projected value for that session; full trendline geometry arrives with the
  post-slice `trendline_definition`/`trendline_snapshot` datasets.)
- `level_price`: float64; `zone_low` / `zone_high`: float64 (nullable).
- `source_timeframe`: string; `source_store`: string — hv_level_store |
  d1_level_feed | d1_level_watches.json | price_alerts.json | computed.
- `strength_score`: float64 (nullable); `touch_count`: int32 (nullable).
- `is_active`: bool; `definition_version`: string.
- `known_at`: timestamp — when the level became knowable (≡ observed_at for
  ingested geometry).

**`feature_snapshot_daily`** — partition: (year); grain: symbol × session_date ×
feature_set_version

- `symbol`: string; `session_date`: date; `feature_set_version`: string.
- `close`: float64; `atr14`: float64.
- `avwape_value`, `avwape_upper_1/2/3`, `avwape_lower_1/2/3`: float64 (nullable)
  — champion bands, parity-tested.
- `favorite_zone_coord`: float64 (nullable) — per Section 6.2, mirrored for
  shorts.
- `favorite_zone_residence_bars`: int32 (nullable); `first_dev_touch_order`:
  int32 (nullable).
- `band1_rejection_strength`: float64 (nullable); `second_band_streak`: int32
  (nullable).
- `ema8` / `ema15` / `ema21` / `sma50` / `sma100` / `sma200`: float64 — tier-1
  D1 grid.
- `dist_sma50_atr` / `dist_sma100_atr` / `dist_sma200_atr`: float64 — distances
  in ATR.
- `spy_regime_state`: string — champion state (shadow market_state stored
  separately).
- `input_manifest_hash`: string — hash of the input file set (reproducibility).
- `computed_at`: timestamp; `event_at`: timestamp (session close);
  `capture_mode` of worst input: string.

**`feature_snapshot_intraday`** — partition: (month); grain: symbol × M5
interval_start × feature_set_version (cohort only)

- `symbol`: string; `interval_start`: timestamp; `session_id`: string;
  `session_phase`: string.
- `feature_set_version`: string.
- `session_vwap` / `session_vwap_upper_1` / `session_vwap_lower_1`: float64 —
  with `vwap_algorithm`: string (STANDARD | DYNAMIC | EOD), the three frozen
  production algorithms.
- `ema8_m5` / `ema15_m5` / `ema21_m5`: float64.
- `ema8_m15` / `ema15_m15` / `ema21_m15`: float64 (nullable) — from Phase-4
  derived M15 bars, at the enclosing M15 boundary.
- `ema8_m30` / `ema15_m30` / `ema21_m30`: float64 (nullable) — same, M30.
- `rvol_tc2000`: float64; `rvol_gate_pass`: bool — the production ≥1.0 gate;
  sub-1.0 rows retained as learning-only denominator.
- `rs_rw_vs_spy`: float64 — anchored to the day's opening regime.
- `group_rs_debiased`: float64 (nullable).
- `market_internals_negative`: bool — VXX/HYG/TLT/RSP/MAGS measured-negative
  flag.
- `session_structure_gate`: string — existing BounceBot chop-veto state,
  migrated.
- `pullback_count_in_current_leg`: int32 (nullable) — one-simple-retest
  measurability.
- `dist_pdh_atr` / `dist_pdl_atr`: float64 (nullable).
- `computed_at` / `observed_at` / `capture_mode`: per convention.

**`setup_occurrence`** — partition: (year); grain: one thesis occurrence

- `occurrence_id`: string — deterministic hash(symbol, canonical_setup_id,
  side, structural_timeframe, anchor_instance_id or episode-window start);
  rescans update, never append.
- `symbol`: string; `canonical_setup_id`: string — verbatim from
  `setup_tagging.py`; display labels live in Appendix C only.
- `side`: string — LONG | SHORT (distinct identities).
- `structural_timeframe` / `trigger_timeframe`: string.
- `anchor_instance_id`: string (nullable); `dependency_cluster_id`: string —
  episode identity for evidence floors.
- `status`: string — detector lifecycle state as reported (never re-detected).
- `trigger_at`: timestamp (nullable); `trigger_bar_interval_start`: timestamp
  (nullable).
- `entry_price_ref` / `stop_price_ref`: float64 (nullable) — detector-reported
  geometry.
- `detector_version`: string; `first_detected_run_id` / `last_updated_run_id`:
  string.
- `tags`: string — free text (the "banger" attachment point, LD-27).
- `event_at` / `observed_at` / `computed_at` / revision chain: per convention.

**`outcome_path`** — partition: (year); grain: occurrence × recipe ×
outcome_definition

- `occurrence_id`: string; `recipe_id`: string; `outcome_definition_id`: string
  (`house_default_v1`).
- `analysis_unit`: string — exactly one of OPPORTUNITY | ATTEMPT |
  MARKET_EPISODE.
- `entry_at`: timestamp (nullable); `entry_price`: float64 (nullable).
- `stop_price`: float64 (nullable); `stop_distance`: float64 — the R
  denominator.
- `r_at_15m` / `r_at_30m` / `r_at_60m` / `r_at_120m` / `r_at_eod`: float64
  (nullable) — intraday checkpoints (`r_at_60m` ≡ quick_r; `r_at_eod` ≡
  entry_r).
- `r_at_s1` / `r_at_s2` / `r_at_s3` / `r_at_s5` / `r_at_s10` / `r_at_s18`:
  float64 (nullable) — swing checkpoints (the freshly frozen superset of
  Section 16).
- `mfe_r` / `mae_r`: float64 (nullable); `time_to_mfe_min`: int32 (nullable).
- `first_hit`: string (nullable) — STOP | TARGET | NEITHER; `first_hit_at`:
  timestamp (nullable).
- `path_resolution`: string — EXACT | LOWER_TIMEFRAME | AMBIGUOUS.
- `r_lower_bound` / `r_upper_bound`: float64 (nullable) — STOP_FIRST primary /
  TARGET_FIRST bound.
- `gross_r` / `net_r`: float64 (nullable); `cost_model_id`: string.
- `result_state`: string — slice subset per Section 14.2; MATURED is derived
  (`maturity_at <= as_of`), never stored.
- `maturity_at`: timestamp; `censor_reason`: string (nullable).
- `computed_at`: timestamp; `input_capture_mode_worst`: string — worst input
  mode, drives the exclusion filter.

**`scan_coverage`** — partition: (month); grain: risk_set × symbol

- `risk_set_id`: string — one per scheduled scan run.
- `scheduled_at`: timestamp; `run_kind`: string — master_scan |
  bouncebot_cycle | autopilot_slot.
- `symbol`: string.
- `scan_status`: string — NOT_ASSIGNED | REQUESTED | NO_RESPONSE |
  PARTIAL_DATA | TIMED_OUT | EVALUATED_INELIGIBLE | EVALUATED_ELIGIBLE.
- `provider`: string; `bar_source`: string — tracked per scan as production
  already does.
- `family_status_map`: string — compact JSON map {canonical_setup_id:
  NOT_APPLICABLE | STRUCTURE_ABSENT | ELIGIBLE | INELIGIBLE(reason) |
  TRIGGERED}.
- `observed_at`: timestamp.

**`collection_gap`** — partition: (month); grain: symbol × timeframe × gap
interval

- `symbol`: string; `timeframe`: string.
- `gap_start` / `gap_end`: timestamp; `expected_bars`: int32.
- `reason`: string — quality-state subset incl. NOT_COLLECTED_BY_POLICY (never
  MISSING for policy absence).
- `detected_at`: timestamp; `resolved_at`: timestamp (nullable).
- `resolution`: string (nullable) — BACKFILLED | PERMANENT | POLICY.

**`manifest_log.jsonl`** (append-only ledger — the read authority, Section 8.3).
One JSON line per action:

- `manifest_seq`: int — monotonically increasing.
- `action`: string — PUBLISH | COMPACT | RETIRE | IMPORT | QUARANTINE.
- `dataset` / `partition`: string; `file_path`: string (lake-relative).
- `sha256`: string; `row_count`: int; `min_ts` / `max_ts`: timestamp.
- `supersedes`: [string] (nullable) — for COMPACT, the part files simultaneously
  retired in this same line (the atomic switch).
- `git_commit`: string — definitions provenance; `job_id`: string — job-ledger
  link.
- `written_at`: timestamp.

**`imported_bundles.jsonl`** mirrors this shape with `bundle_hash`,
`source_machine`, and `accepted_at` (idempotency key = content hash; empty in
the slice).

### 7.2 Forward-declared datasets

The remaining datasets are defined when their owning phase or post-slice
milestone begins; their grains are declared now so identities stay stable:

| Dataset | Independent grain and purpose |
|---|---|
| `instrument_master` | One stable instrument identity/version; symbol aliases, exchange, currency (bitemporal) |
| `collection_universe_membership` | Instrument × resolution/session scope × effective interval, cohort, selector, assignment provenance |
| `provider_observation` | Raw request/response observation with availability and hash |
| `raw_bar` / `normalized_bar` | Provider vs canonical bar separation when multi-provider normalization begins |
| `corporate_action` | Split/dividend/action as known at a point in time (bitemporal) |
| `catalyst_event` | Earnings/other event, source, event time, known-at, revisions (bitemporal) |
| `anchor_definition` | Versioned anchor hypothesis and calculation contract |
| `level_definition` / `level_snapshot` / `level_interaction` | Full level lifecycle, per-timestamp values, typed interactions |
| `trendline_definition` / `trendline_snapshot` | Versioned geometry and projected values |
| `feature_definition` | Semantic feature version, inputs, formula, parameters, null policy |
| `context_snapshot` / `context_episode` | Market/sector/industry context and durable episodes |
| `setup_definition` / `strategy_recipe` | Versioned theses and trade policies (registry-backed) |
| `risk_set` / `evaluation_slot` / `candidate_eligibility` | Full denominator lattice past `scan_coverage` (post-slice milestone) |
| `market_episode` / `dependency_cluster` | Outcome-blind underlying move/reset cluster |
| `opportunity_lifecycle` / `attempt` / `trigger_event` | Discovery→ready/failed/rearm/expired; one try per attempt; completed-bar evaluations |
| `ranking_snapshot` / `alert_event` / `delivery_event` | Eligible cohort, component scores, deliveries |
| `impression` / `review_action` | What was shown and how the trader resolved it |
| `trade` / `fill` / `management_event` | Actual imported execution records |
| `outcome_definition` / `outcome_result` | Frozen outcome contracts and declared analysis units |
| `experiment_definition` / `experiment_run` / `hypothesis_registry` | Registered questions, runs, trial ledger |
| `evidence_snapshot` / `promotion_decision` / `data_manifest` | Immutable reviewed metrics, promotions, partition health |

### 7.3 Identity graph

Extend—do not replace—the canonical identity graph already specified by the GUI
learning plan. Add outcome-blind episode/dependency identity above correlated
setup variants:

```text
instrument_id
  ├─ universe_membership_id
  ├─ collection_universe_membership_id
  ├─ anchor_instance_id
  ├─ level_id / trendline_id
  └─ market_episode_id / dependency_cluster_id
       └─ candidate_id
          └─ setup_occurrence_id
            └─ opportunity_id + thesis_version
                 └─ lifecycle_id
                      └─ attempt_id
                           ├─ trigger_id
                           ├─ ranking eligibility_id
                           ├─ alert_event_id -> delivery_id(s)
                           ├─ impression_id -> action_id(s)
                           └─ outcome_id(s) by recipe/outcome definition
```

Identity rules:

- Swing and M5 theses for the same symbol are distinct.
- Long and short theses are distinct.
- Setup, anchor, structural timeframe, or material plan changes create a linked
  successor opportunity rather than mutating history.
- A rescan updates a snapshot and never creates an extra occurrence or outcome.
  Phase 6 implements this with the **deterministic occurrence key stated in
  `schemas.py`**, so hourly rescans update rather than append (the tracker
  episode-dedup lesson).
- A failed attempt followed by re-arm creates a new `attempt_id` under the same
  lifecycle.
- A repeated delivery creates another `delivery_id`, not another alert or
  sample.
- Manual and model-originated levels retain their separate identities even when
  clustered at the same price.
- Alternative recipe/horizon results share the same occurrence and cannot be
  summed as independent samples.
- Simultaneous EMA/SMA/AVWAP/band/horizontal/trendline variants attached to one
  underlying move are multiple hypotheses in one dependency cluster, not extra
  independent episodes.
- A re-arm creates a new attempt but becomes a new independent episode only when
  a predeclared outcome-blind washout/reset rule passes.
- Every outcome definition declares exactly one `analysis_unit` (`OPPORTUNITY`,
  `ATTEMPT`, or `MARKET_EPISODE`); it never says "occurrence/attempt."

Phase 1 publishes a small ERD with primary/foreign keys, cardinalities,
deterministic ID algorithms, occurrence start/end/dedup rules, corrections, and
supersession behavior. Every evidence report includes `n_rows`, `n_attempts`,
`n_market_episodes`, `n_sessions`, `n_symbols`, and method-derived
`n_effective`.
## 8. Storage architecture on the DAS

### 8.1 The store

An **immutable Parquet/Zstd lake on the DAS with a pyarrow-only write path** —
forever. DuckDB is a deferred, optional, **read-only** query engine over
manifest-resolved Parquet file lists (Phase 7); any `.duckdb` file is a
disposable machine-local cache, never shared, never authoritative; there is no
persistent generation catalog. (DuckDB's own concurrency documentation motivates
this posture: single-writer in-process model, extra caution on shared/network
storage — https://duckdb.org/docs/stable/connect/concurrency.)

Decision record 0014 (skeleton in Appendix E) scopes the lake as a **new
append-only storage class**. Decision 0005 stays fully in force for operational
mutable data: watchlists, reports, JSONL evidence logs, and every live surface
stay in the Drive home folder / `%LOCALAPPDATA%` exactly as today. No mutable
database file ever lives in the Drive folder or on the DAS.

### 8.2 Directory contract

```text
<research_store_dir>/
  _incoming/            # part-<uuid>.parquet staged writes (same NTFS volume)
  _quarantine/          # per-symbol/per-partition dirty tails, surfaced in Health
  _retired/<yyyymmdd>/  # compaction-superseded files; GC-purged after 30 days
  bronze/  silver/  gold/
  definitions/          # Git-authoritative copies; manifest lines record git hashes
  manifest_log.jsonl    # append-only read authority
  imported_bundles.jsonl

<machine_local_state>/research_spool/
```

Generated data stays outside Git. Git holds schemas, definitions, migrations,
golden fixtures, and small sanitized examples. Reviewed definitions in Git are
authoritative; the lake retains immutable released copies and hashes so an old
evidence freeze remains reproducible.

### 8.3 Seal protocol, read consistency, partitioning, capacity

**4-step seal protocol** (the entire write path):

1. Write `part-<uuid>.parquet` to `<lake>/_incoming/` (same NTFS volume).
2. Hash + validate (SHA-256, row count, min/max timestamps).
3. `os.replace()` into the final partition path.
4. Append one JSON line to `manifest_log.jsonl`.

**Read-consistency rule:** `manifest_log.jsonl` is the read authority, not the
directory tree. Publishes are atomic per file. Compaction appends ONE manifest
record that simultaneously registers the replacement file and marks its source
part files retired; canned queries and the research query API resolve the live
file set from `manifest_log.jsonl` at query start and pass an explicit file list
to the reader (`pyarrow.dataset`, later DuckDB `read_parquet`) — a consistent
snapshot per query, since files are immutable and nothing is deleted inside the
30-day `_retired` window. Physical moves to `_retired/<yyyymmdd>/` are garbage
collection only and may lag; direct ad-hoc globbing of lake directories can
double-count during a compaction GC window and **is not a supported read path**.
This is a few dozen lines of manifest reading, not a catalog system. The
manifest append is the atomic switch and `_retired/` is the rollback window.
Logical row/key-count reconciliation runs only for compaction, never per
publish. Bronze raw and evidence-frozen files are never compaction inputs.

**Partial-publish semantics** (tracker-blackout precedent, week of 2026-07-13):
quarantine at per-symbol/per-partition granularity, publish the clean remainder,
surface the quarantine count in Health; wholesale veto only on manifest
corruption, never on a bounded dirty tail. Malformed/conflicting records go to
`_quarantine/`, never silently discarded.

**Partition spec (fixed outright, no tuning task):** one file per (dataset,
timeframe, month); M1 additionally 8 symbol-hash buckets; D1/W1 and small
reference datasets per (dataset, year).

**Capacity.** Assumptions: Parquet + Zstd; ~50 B/row fully-loaded bar rows,
~60 B/row worst case (initial values — pilot confirms bytes/row); 252
sessions/yr; RTH M5 = 78 bars/session; ETH M5 effective ~120; universe ~1,539
symbols (current inventory, not a spec constant); tee cohort ~280; M1 cohort
150 generous / 300 worst:

| Dataset class | Volume assumption | GB/yr generous | GB/yr worst |
|---|---|---|---|
| `bar_m5` full universe, ETH | 46-75 M rows | 2.3 | 4.5 |
| `bar_m1` cohort, ETH | 26-73 M rows | 1.3 | 4.4 |
| Derived M15/M30/H1/H4 (RTH) | ~19 M rows | 0.9 | 1.5 |
| `bar_d1` + derived W1 | 1,539 × ~304 rows | 0.04 | 0.05 |
| Bronze tee / provider observations | ≈1× M5 archive | 2.5 | 6.0 |
| `feature_snapshot_daily` | 388 k rows × ~1 KB | 0.4 | 0.8 |
| `feature_snapshot_intraday` (cohort) | 5.5 M rows × ~0.4-1 KB | 2.2 | 6.0 |
| Occurrences, outcomes, coverage, gaps, ledgers | `scan_coverage` ~11 k rows/day dominates | 1.0 | 2.0 |
| Wrapped legacy bronze (tracker, integrity events, ledgers) | ~1-2 GB one-time + growth | 1.0 | 2.0 |
| Manifests, definitions, evidence freezes (Class A) | Megabytes | <0.1 | <0.1 |
| **Total** | | **~12-15** | **≤40** |

Disk is not the constraint — IB pacing is. Raw and evidence-frozen data are
retained indefinitely; the retention policy is revisited only if the lake
exceeds **250 GB** (≥6 years at worst case).

### 8.4 Ownership, jobs, and multi-machine behavior

**No new process, daemon, or service.** Live spooling is one GUI-owned
`ResearchSpoolWriter`; seal/aggregate/feature/outcome work runs as one
post-scan/EOD CLI build job (`python -m scripts.research_warehouse.cli build`)
registered in the existing job ledger and runnable manually. The warehouse is
fully disabled when `research_store_dir` is unset.

- **Single-flight rule:** the build job takes a job-ledger lock; a manual
  invocation during a scheduled build refuses with a clear message.
- **Spool rollover contract:** the CLI seals only spool segments the GUI writer
  has closed; the writer rolls its active segment on size/time or roll request —
  writer and sealer never touch the same file.
- **The main desktop is the ONLY lake writer for the entire initial program
  (Phases 0-8).** No leases or fencing for the lake: decision 0006 governs
  Drive-shared exports with two candidate writers; the DAS has one.
- **Mini-PC: deferred, design recorded now.** Excluded from Phases 0-8; never
  runs streaming capture; gated on the plan.md M1 dual-scheduler/client-1003
  reconciliation. Deferred mechanism: hash-named immutable bundle zips in a
  Drive-synced `research_inbox/`; the importer validates hashes, seals via the
  4-step protocol, records the hash in `imported_bundles.jsonl`, and moves the
  zip to `imported/` (the move IS the acknowledgment); idempotent by content
  hash. Away-day policy: Focus capture is recorded `NOT_COLLECTED_BY_POLICY`;
  the mini's scan fetches tee into a local bundle; the desktop's nightly
  backfill supplies the day's bars as capture_mode=BACKFILL.
- **DAS unavailable:** spool to `%LOCALAPPDATA%\TradingBotV3\research_spool`,
  cap 5 GB / 7 days; shedding order: M1-exploration extras → non-focus M1 → ETH
  bars; D1/M5 capture and operational champions are never shed; gaps recorded,
  evidence never deleted; Health goes red. Continuity policy is one sentence:
  acceptable loss ≤1 session of raw capture plus all derived data; recovery =
  re-run backfill and rebuild.

The Away-runbook vs First-Session-checklist publisher-role wording conflict is a
**Phase 0 documentation cleanup task** (edit both runbooks to name the
session-scoped `scan_owner`), not a gate — the single-writer lake never depends
on cross-machine ownership.

### 8.5 Backup and recovery — three classes

A DAS/RAID is capacity and availability, not backup.

- **Class A — irreplaceable-small** (manual geometry, journal, review events,
  decisions, definitions, evidence freezes, manifests): nightly robocopy to the
  backup disk AND mirrored into the existing Drive home folder (off-site for
  free).
- **Class B — the lake:** nightly incremental robocopy (append-only copy, never
  /MIR-style deletion propagation) to a second physical disk.
- **Class C — derived:** never backed up; rebuilt from A+B.

Verification: scripted restore check at slice exit and after any storage-code
change; then a semiannual 15-minute spot restore (one month-partition, hash
check, one canned query), logged in one line. Restores go to a new root and
re-point the manifest; never restore destructively in place.

### 8.6 Security, privacy, and data rights

- Keep API keys, broker credentials, Desk Link secrets, and encryption keys out
  of the lake, manifests, diagnostics, and Git.
- Restrict filesystem and query access to the trader's machines; satellites
  receive only the data their UI requires.
- Keep fills, account identifiers, free-text journal notes, and screenshots in
  access-controlled local partitions; redact compact Drive and AI exports.
- AI evidence packages remain explicit opt-in, show the exact selected sources,
  and cite immutable IDs/hashes. A research-store path never implies permission
  to upload its contents.
- **Data rights (one line, no licensing-review gate):** the archive is
  personal-use only — raw provider data is never redistributed and archive
  contents are never bulk-uploaded to external/AI services.

## 9. Ingestion and data-quality contract

### 9.1 Provider boundary and sentinel parity

Use app-owned repository interfaces for daily bars, intraday bars, quotes,
corporate events, and read-only execution imports. Persist normalized app-owned
records, never IBKR/Yahoo SDK objects. IBKR remains primary and Yahoo fallback.
Every observation and downstream feature records its actual source; never blend
sources without a provider-transition record.

Provider parity is the **minimal sentinel job only**: 20 fixed symbols
including SPY and BF.B; nightly D1 close within 0.1%; volume ×100 within 5%
(precedent: the 2026-07-20 round-lot RVOL bug,
`scripts/bounce_bot_lib/legacy.py:564-569`); M5 mismatch <2% on sentinel
overlap; always drop yfinance's forming last bar; corporate-action days flagged
`ADJUSTMENT_WINDOW` and exempt; disagreements emit `PROVIDER_CONFLICT` rows.

### 9.2 Collection cadence

| Evidence | Cadence |
|---|---|
| M1/M5 streamed (Focus cohort, post-slice) | Every completed interval with explicit missing-bar rows |
| M5 tee (watchlist cohort) | Tee cadence (production cycle) with nightly gap reconciliation |
| All other bar cohorts | Nightly/weekly backfill per Section 5.2, with `NOT_COLLECTED_BY_POLICY` rows intraday |
| M15/M30/H1/H4/W1 | Derived in the EOD build job |
| D1 | Existing store/tee at each scan; canonical at session close |
| Setup eligibility / `scan_coverage` | Every scheduled scan |
| User last-price crossings | Current service cadence, stored as operational crossing evidence |
| Early persistence | Open+30, +45, and +60 snapshots where relevant |
| Intraday outcomes | +15/+30/+60/+120 minutes and EOD, with near-close truncation |
| Swing outcomes | +1/+2/+3/+5/+10/+18 sessions and final policy resolution |
| Health/coverage | Open, midday, close, and teardown |

Report cadence never defines collection cadence. DESK, AWAY, and EVENING retain
the same observed-candidate evidence even when presentation behavior differs.

### 9.3 Point-in-time availability — five columns, two bases

Every record carries **five point-in-time columns**:

```text
event_at                      when the market fact occurred
observed_at                   when this installation received it
computed_at                   derived records only
capture_mode                  LIVE | DELAYED | BACKFILL | RECONSTRUCTED
revision_id / supersedes_revision_id
```

Availability is a **per-experiment declaration, never a per-row column**. Two
bases exist:

- **`AS_OBSERVED`** — mandatory for coverage, denominator, queue-exposure,
  latency, live-shadow, and all promotion evidence. A source row is available at
  `observed_at`, a derived row at `computed_at`; only capture_mode ∈ {LIVE,
  DELAYED} qualifies; one filter applied everywhere excludes
  BACKFILL/RECONSTRUCTED from these claims.
- **`MARKET`** — permitted only for declared reconstructed market backtests:
  available at `event_at` plus one conservative lag per (provider, timeframe)
  from a single named, versioned, registry-owned lag table cited by the
  experiment manifest; BACKFILL admissible, RECONSTRUCTED only when explicitly
  declared.

The experiment derives availability from its declared basis; callers can never
supply it. Latency evidence = `observed_at − event_at` on LIVE rows.
`pit_eligible` is a derived predicate of (capture_mode, declared basis).

**Why two bases:** on this installation `observed_at − event_at` is routinely
tens of minutes on scanned symbols (28.5-min full scans, 61-min hourly scans,
IBKR pacing), so binding as-observed claims to `event_at` would itself be
look-ahead leakage into promotion evidence. And forward-observed capture makes
actual and simulated availability identical by construction, so the dual
per-row clock apparatus carries zero information on live rows; if a
PIT_RECONSTRUCTED backtest phase is ever registered, a modeled simulated clock
is added as nullable columns — no rewrite, no PIT invariant weakened.

### 9.4 Quality states

Typed states, not blank values. The slice implements the reachable subset:

```text
COMPLETE | PARTIAL | MISSING | PROVIDER_FALLBACK | NOT_COLLECTED_BY_POLICY |
TIMED_OUT | NO_RESPONSE | HALTED | OUTSIDE_SESSION
```

The full target vocabulary adds: `PREVIEW`, `STALE`, `LATE_ARRIVAL`,
`MISSED_SNAPSHOT`, `TRUNCATED`, `PROVIDER_CONFLICT`, `NO_TRADE`, `NOT_LISTED`,
`INVALID_DATA`, `QUARANTINED`. Enums are strings, so widening is free.

Research queries default to completed, eligible observations whose availability
under the experiment's declared basis is no later than the simulated decision
time. A late-arriving bar cannot be made retroactively available simply because
its market timestamp is earlier.

### 9.5 Corporate actions and revisions

Retain raw and adjusted prices with an explicit adjustment version and the time
the corporate action became known. Never overwrite old partitions after a split,
symbol change, earnings correction, or provider revision — append a revision and
choose the appropriate view per experiment.

Bitemporal `valid_from/valid_to` + `system_from/system_to` intervals exist
**only** on the revisable reference datasets: `corporate_action`,
`catalyst_event`, `anchor_instance` (and anchor definitions), level/trendline
definitions, `instrument_master`, and the forward-declared
`collection_universe_membership` (effective-interval grain).
`universe_membership_daily` is an append-only daily snapshot (LD-05) and
carries none. All other datasets are append-only with revision chains;
the superseding row's `observed_at` is the knowledge time — preserving
as-recorded vs corrected experiment views without universal bitemporal columns.

## 10. Feature registry and calculation graph

Every feature definition includes:

- stable feature ID, semantic version, family, units, and description;
- input datasets/fields and exact timeframe roles;
- parameters, minimum history, session rules, and adjustment semantics;
- completed/preview eligibility;
- null/missing/stale behavior;
- formula/code/config hash and dependency versions;
- long/short transformation or explicit asymmetry;
- independence/correlation family;
- owner, status, and deprecation/supersession link.

Use a deterministic dependency graph:

```text
bars/events
  -> anchors/levels/geometry
  -> continuous technical/context features
  -> atomic interactions and episodes
  -> setup eligibility and state
  -> strategy simulations and outcomes
  -> research aggregates and rankings
```

For performance, store coherent typed feature families in wide Parquet tables
and keep definitions/lineage in the registry. Avoid a single generic EAV table
with billions of `(feature_name, value)` rows, and avoid one permanently growing
flat row with every experimental column.

Every calculation publishes coverage and exclusion reasons. Re-running the same
feature version and source snapshot must produce identical keys and values. The
dependency graph is statically/auditably separated so a feature, context, setup,
eligibility, or prediction node cannot read outcome/result tables. Availability
is recomputed from dependency `observed_at`/`computed_at`, never trusted from
the feature implementation.

## 11. Setup-definition and experiment language

Create a declarative, versioned setup registry. A definition contains:

- thesis and failure mode;
- detector symmetry: mirrored logic or an explicitly side-specific detector;
- outcome-side support: actual long paths, actual short paths, or explicitly
  unsupported sides;
- structural/context/trigger/management timeframe roles;
- anchor and required level families;
- context predicates;
- atomic trigger sequence and completed-bar requirements;
- mandatory hard gates versus supporting features;
- state machine, reset, re-arm, attempt limit, expiry, and no-chase rules;
- candidate universe and point-in-time membership rule;
- primary strategy recipe and allowed diagnostic recipes;
- feature, regime, cost, and outcome versions;
- research/challenger/champion status and rollback target.

Example definition shape:

```yaml
setup_id: d1_earnings_avwap_m30_m15_retest_v1
detector_symmetry: mirrored_logic
outcome_side_support: [LONG_ACTUAL_PATHS, SHORT_ACTUAL_PATHS]
structure:
  timeframe: D1
  anchor: current_earnings_avwap_v1
context:
  - H1.ema15_slope_atr > 0
  - M30.controlled_pullback = true
trigger:
  timeframe: M15
  sequence: [close_reclaim, retest_hold]
hard_gates: [complete_data, room_to_obstacle, not_extended]
primary_recipe: structural_retest_house_v1
status: research
```

The stored definition—not ad hoc notebook code—determines what was tested.
Parameter searches declare their grid and trial-ledger entry before outcomes are
inspected.

## 12. Strategy-recipe library

The same setup must be evaluated under explicit trade-style recipes rather than
baking one management policy into the setup name. The concrete v1 recipes
(`swing_house_v1`, `intraday_bounce_v1`, the two shared controls, and the
registered ATR-stop diagnostic) are defined once in Section 19.3.

### 12.1 Entry methods

- signal close;
- next regular-session open;
- first observable armed quote cross or bar-high/low penetration through a
  trigger price, including a stop-entry recipe with deterministic gap-through
  fill and slippage rules;
- first completed retest hold;
- later/nth retest, recorded separately;
- limit/zone fill with deterministic gap and no-fill semantics;
- opening-range break after the configured time;
- first lower-timeframe trigger inside a higher-timeframe level interaction.

Every entry simulation stores `signal_known_at` and `entry_eligible_at`. A
signal computed from a completed close is executable no earlier than the next
eligible quote/bar/open unless the recipe explicitly models a precommitted
market-on-close order. A same-close fill can never be assumed merely because the
close created the signal. Quote-observed entries retain the quote and timestamp
provenance. When only OHLC bars exist, the first-cross time is interval-censored
within the bar and receives the declared conservative fill; it is never assigned
a precise intrabar order. A later completed-close-confirmed break is a separate
recipe, not another fill interpretation for the same recipe. `QUOTE_CROSS` may
alert or fill only a pre-armed recipe whose eligibility came from completed-bar
context; it never creates a setup `trigger_event` or advances lifecycle
confirmation. `BAR_HIGH_LOW_TOUCH` becomes confirmation only when that bar is
completed and available. Any anticipatory quote-first style is separately
labeled research evidence and cannot replace a completed-bar champion. The
Post-Earnings Candle Break seed family must include distinct
first-cross/stop-entry and completed-close variants under these boundaries.

### 12.2 Invalidation and stop methods

- house structural level with one- or two-close failure;
- one band beyond the bounced level;
- opposing earnings-candle extreme;
- signal/retest-bar extreme plus registered ATR buffer;
- nearest valid structural/horizontal/trendline level;
- fixed price, percent, ATR, or volatility-distance control policy;
- no intrabar stop for a level-close thesis, if that is the declared policy.

Under-bar tick stops and close-based structural invalidation are different trade
styles. One poor stop policy must not make a valid pattern appear invalid. Store
`invalidation_observed_at` and `exit_eligible_at`; one/two-close failure is
executable only after the confirming close is known. Preserve full intrabar MAE
and gap tails because a close-based thesis stop is not bounded intraday risk. An
optional catastrophe stop is a separately named recipe component.

### 12.3 Targets and management

- fixed 1R/2R/3R controls;
- next AVWAP/deviation band;
- next independent obstacle/level cluster;
- house band-2 partial, band-3 runner, band-1 trail;
- MA, AVWAP, trendline, chandelier, or prior-bar trail;
- scale-out variants with explicit fractions;
- time stops by minutes, EOD, or 1/2/3/5/10/18 sessions;
- trend-leader runner with no artificial band-3 cap.

### 12.4 Combination control

Do not run the full Cartesian product. Maintain:

- a small control recipe set shared by all setups;
- one setup-specific primary recipe declared before validation;
- a bounded set of diagnostic alternatives chosen for a written reason;
- a record of every attempted recipe, including failures;
- correlation labels so alternative recipes do not inflate sample counts.

Canonical short simulations always use actual short-side OHLC paths. Synthetic
mirrored prices survive only as a labeled legacy-parity diagnostic —
**short-side playbook rows are a pre-declared approved intentional difference in
the migration parity gate** (Section 19.5): mirrored legacy vs actual-path
warehouse, reconciled via the labeled diagnostic rather than exact parity.
Actionability and outcome confidence include point-in-time shortability/borrow
availability, HTB/locate and borrow-cost evidence when available, SSR,
halts/LULD, liquidity, and gap risk. Missing historical borrow data is
uncertainty, never an assumption that the short was freely tradable.
## 13. Complete denominator and exposure funnel

Capture every rung independently:

```text
Point-in-time universe member
  -> scheduled risk-set/evaluation assignment
  -> requested/returned/not returned
  -> data eligible
  -> relevant level/structure exists
  -> level approached
  -> atomic interaction occurred
  -> setup eligible
  -> setup triggered
  -> hard gates passed/failed
  -> ranked/quiet/rejected
  -> surfaced/not surfaced
  -> reviewed/unseen
  -> taken/passed/missed/late
  -> filled/not filled
  -> managed/closed
  -> standardized and actual outcomes
```

For each transition store the reason, source snapshot, availability time, and
policy version. Unevaluated is never labeled rejected or "no setup."

**Right-sized grain (LD-21):** `risk_set` = one row per scheduled scan run; one
evaluation record per (`risk_set_id`, symbol) — ~11k rows/day — carrying scan
statuses plus a compact per-family status map `{canonical_setup_id:
NOT_APPLICABLE | STRUCTURE_ABSENT | ELIGIBLE | INELIGIBLE(reason) |
TRIGGERED}`. Full (symbol, family) eligibility rows are created only past the
"relevant structure exists" rung (vs ~290k pre-created slots/day as originally
drafted — ~4% of the row volume with the funnel reconciliation fully
preserved). Metrics reconcile risk set → assigned → returned → data complete →
eligible → triggered → surfaced → matured.

The slice implements only the light `scan_coverage` table (Section 7.1); the
full eligibility lattice is a post-slice milestone (Section 19.4) — the slice
table is a **down-payment on this section, not its replacement**.

**Exploration cohort:** kept — it is load-bearing against champion-conditioning
bias — but acquired BACKFILL-only with seeded RNG logged, and reported as
"exploration cohort, unweighted." No stratum weights, positivity, or balance
diagnostics (Appendix D); adaptive acquisition still cannot claim full-universe
expectancy. Deterministic Focus/model cohorts support cohort-conditional claims
only; Current Edge displays the population each estimate applies to.

Current review data that folds by `(trade_date, symbol)` or treats arming a
watch as a take remains exploratory. Preference cannot influence production
ordering until Swing/M5, side, thesis, impression, and action identities are
repaired (post-slice milestone, Section 19.4).

Manual levels/trendlines are observed only on charts Aaron selected; their
research estimand is conditional on exposure to the chart. Store the exposed
cohort and a point-in-time matched-control set. Claim universal geometry edge
only when a reproducible algorithmic definition provides the non-drawn
denominator; human-selection lift remains association without a predeclared
causal design.

## 14. Outcome engine

### 14.1 Two primary outcome classes

1. **Standardized opportunity outcome:** what the setup did under a frozen
   hypothetical recipe.
2. **Actual execution outcome:** what Aaron entered, sized, managed, and exited.

Never substitute actual P&L for setup quality or standardized R for execution
quality.

Keep four terms mechanically distinct, with the **continuity mapping to the
shipped production quantities**:

- `planned_reward_risk_r` — geometric target distance divided by planned risk;
  ≡ the production `entry_r`;
- `model_expected_net_r` — frozen out-of-sample predicted expectancy net of the
  declared cost model; in v1 this is the shrunk grouped-table mean net R;
- `standardized_realized_r` — realized result under the standardized recipe;
  its +60-minute checkpoint ≡ the production `quick_r`;
- `actual_execution_r` — realized result from imported fills and management.

`production_r = 0.5·quick_r + 0.5·entry_r` remains unchanged and authoritative
for live alert ranking until promotion gates pass. Never label planned
reward/risk as Expected R or train/calibrate it as expectancy.

### 14.2 Required outcome contract — `house_default_v1`

Every `outcome_definition_id` freezes: declared analysis unit and decision
timestamp; trigger and completed-bar identity; entry, fill, gap, no-fill,
slippage, commission, and liquidity assumptions; stop/invalidation,
close-failure, target, scale, trail, and expiry rules; same-bar ordering;
RTH/ETH eligibility; MFE/MAE and first-hit calculation; censoring and
missingness rules.

The v1 contract `house_default_v1`:

- `net_r = gross_r − 2×(commission_per_share + half_spread)/stop_distance_$`;
- commission $0.0035/share (IBKR tiered);
- `half_spread` = observed NBBO at signal, fallback `max($0.01, 2bp×price)`;
- +1 half_spread slippage on stop/market entries;
- same-bar ambiguity: **STOP_FIRST primary** (matches the tracker's existing
  stop-first fix), TARGET_FIRST stored as `r_upper_bound`;
- maturity: EOD intraday; `min(+18 sessions, stop/target/expiry)` swing;
- every deviation is a new `outcome_definition_id`.

OHLC bars cannot reveal path order when stop and target both occur in one bar.
Store `path_resolution = EXACT | LOWER_TIMEFRAME | AMBIGUOUS`, `r_lower_bound`,
`r_upper_bound`, `fill_quality`, `cost_model_id`, `maturity_at`, and
`censor_reason`. Use finer data only when provenance-compatible; otherwise
retain bounds, use the preregistered conservative primary estimate, and report
sensitivity. Never silently drop ambiguous/no-fill/missing cases.

Result states implemented in the slice:

```text
NO_TRIGGER | OPEN | STOPPED | TARGETED | EXPIRED | TRUNCATED | CENSORED |
AMBIGUOUS_BAR
```

The full target list adds `NO_FILL`, `MISSING`, `INVALID_DATA`. `MATURED` is a
derived predicate (`maturity_at <= as_of`), never a stored state. Report
full-risk-set, trigger-conditional, and fill-conditional estimands separately.
An unresolved label cannot enter training or a current posterior until its
`maturity_at` is no later than that prediction's evidence cutoff.

### 14.3 Outcome paths

Retain the path, not just one terminal R:

- side-adjusted return and R at +15/+30/+60/+120 minutes and EOD;
- side-adjusted return and R at +1/+2/+3/+5/+10/+18 sessions;
- MFE/MAE, time to MFE/MAE, and maximum favorable/adverse close;
- first target/stop/obstacle hit and timestamp;
- gap-through-stop/target behavior;
- remaining R and extension at alert/review/entry;
- opportunity-window expiry and whether the move occurred before visibility.

Tracker and playbook results currently use materially different entries and
stops. They remain incomparable until each carries an explicit recipe/outcome ID
— resolved by Phase 6's recipe IDs and the Section 19.5 parity gate.

## 15. Research and statistical framework

The apparatus in this section activates when registered technical-variation
research (Section 19.4, old Workstream 8) begins; no gate in this section
applies before the First Vertical Slice completes and its 20 forward-observed
sessions exist. **This section governs analysis, not build order.** Everything
demoted at solo-trader scale lives in Appendix D with explicit activation
triggers.

### 15.1 Registered research question and the trial ledger

Every experiment predeclares: thesis and failure mode; primary metric and
tolerated degradation metrics; independent observation unit; universe, side,
timeframes, feature/recipe versions, and exclusions; training, validation,
final-test, and live-shadow windows; matched control/baseline; parameter grid;
minimum evidence per Section 15.8; costs, missing-data rule, and
promotion/abandonment criteria.

The global hypothesis registry records:

```text
hypothesis_id
specification_id
role = DISCOVERY | VALIDATION | CONFIRMATORY
holdout_exposure_count
outcome_first_opened_at
n_variants_examined        (the trial ledger — family-lifetime count)
```

**The trial ledger replaces formal multiplicity machinery:** every family
records `n_variants_examined` (family-lifetime count; splitting the search into
several experiment files never resets it); claims headline "best of k"; the
widening rule is k>10 ⇒ 99% interval on holdout AND beat the family-median
holdout. Once a final holdout is inspected it is spent and becomes validation
evidence; a new confirmatory claim needs untouched forward evidence. Scheduled
repeated reviews use the fixed monthly cadence (LD-13) — the cadence itself is
the repeated-look control. The registry retains losing and inconclusive studies
so the same hypothesis is not repeatedly rediscovered and selectively reported.

### 15.2 Baselines and controls

Retain `baseline_every5` and add matched controls by **coarse stratification**
on four covariates: month, time-of-day bucket, liquidity tercile, and side ×
broad market direction. Every matched set freezes `matched_set_id`, its
covariates, fixed seed, and control-reuse count. No calipers, propensity
weights, or balance diagnostics (Appendix D).

Compare a new family against both its matched control and the current setup
portfolio. Lift from Move Forensics is association only and never becomes a
score until a tradeable, forward-tested setup clears the full ladder.
Human-selection comparisons remain associational unless exposure was randomized
or a defensible causal design was registered in advance.

### 15.3 Validation design

- Chronological walk-forward splits; never random row splits.
- **Purge: 18 sessions (swing; = the documented maximum time-stop horizon) /
  1 session (intraday) around every boundary; no extra embargo.**
- **Expanding walk-forward with 3-calendar-month test blocks; ≥2 completed folds
  before any OOS claim.**
- Freeze training, validation, and final-test partitions before inspection.
- Every `dependency_cluster_id`, market/catalyst episode, correlated re-arm
  chain, and overlapping outcome interval lies wholly inside one partition.
- Fit feature selection, preprocessing, imputation, matching, shrinkage, and
  hyperparameters only inside the applicable training/inner-validation window.
- Test both sides explicitly; a mirrored detector is not evidence of mirrored
  expectancy.
- Test multiple regimes, open/midday/late periods, and material volatility
  states.
- Require portfolio-level incremental value after overlap/correlation with
  champion setups.

**Evidence floors replace power calculations** — the tier table in Section 15.8
is normative. Floors apply to untouched out-of-sample market episodes
(`dependency_cluster_id` = one market move; rescans/variants/re-arms never add
episodes), never raw rows, and are never pooled across side, style, recipe,
`outcome_definition_id`, or regime. Thin cells remain exploratory. Rare families
cap at ADVISORY permanently — no manual override; manual review does not
substitute for statistical efficacy evidence.

### 15.4 Estimation — rungs 1-2, locked

- **Rung 1: grouped tables.** Wilson 95% intervals for rates; percentile
  cluster-bootstrap 90% intervals for mean R (B = 2,000). Bootstrap blocks:
  intraday = trading session (all symbols in a session resample together);
  swing = market_episode/dependency_cluster (all symbols' overlapping swing
  episodes inside one market episode resample as a single block; symbol may be a
  nested secondary dimension for diagnostics, but symbol×market_episode cells
  are never resampled as independent blocks — that breaks the preserved
  cross-sectional dependence and understates variance exactly when swing
  families fire during market-wide legs).
- **Rung 2: two-level empirical-Bayes shrinkage**, cell→family (k=15) and
  family→global (k=30).
- **Climb to rung 3** (regularized regression/additive models) requires ≥300
  episodes + a monotone 3-bin effect + stability across ≥2 walk-forward folds,
  per family, never globally. Rungs 4-5 (trees, survival models) are deferred ≥2
  years (Appendix D).

Every estimate is versioned, replayable, explainable at the opportunity level,
and compared with a trivial baseline. Complexity is not itself progress.

### 15.5 Recent versus durable edge — two estimates

Exactly two estimates per cell, plus one flag:

- all-history shrunk estimate;
- rolling recent estimate (60 sessions intraday / 12 months swing);
- one **divergence flag** when the recent interval excludes the durable point
  estimate.

Regime is a grouping column, not a posterior. Divergence may trigger abstention
or a challenger proposal; it never silently retunes a live champion. Decay
half-lives and change-point models are deferred (Appendix D). Recent adaptation
uses only outcomes matured before the prediction cutoff and updates at fixed
review times.

### 15.6 Prediction ledger and abstention

- **Prediction ledger:** every published estimate is persisted before its
  outcome matures — the irreversible prerequisite for any later honesty audit.
- **Quarterly decile reliability check** on matured ledger entries.
- **Abstention by support count:** fewer than 8 matured episodes in the current
  context cell ⇒ `THIN_CONTEXT_EVIDENCE`.

The full calibration/OOD contract (Brier/slope/intercept, OOD distance,
calibration-age limits) activates on rung-3 climb (Appendix D).

### 15.7 Promotion estimator

Promotion freezes one primary estimand and one primary benefit, then compares
challenger with champion on identical ranking snapshots and risk sets. Require:

- 90% cluster-bootstrap CI lower bound > 0 on the predeclared primary benefit
  (99% if trial-ledger k > 10);
- three named point-estimate guardrails, all reported, none pooled:
  missed-winner rate, p10 downside R, and alert latency;
- the PROMOTION-ELIGIBLE evidence floor of Section 15.8, including live-shadow
  coverage;
- no dominant symbol/session/catalyst concentration;
- frozen model, features, splits, costs, outcomes, and data hashes;
- no tuning during shadow/canary.

plan.md Section 7 still governs promotion itself — this section supplies the
evidence method, not the authority. A short canary validates safety, mechanics,
parity, and rollback; it cannot establish efficacy until the powered
matured-outcome floor also passes.

### 15.8 Acceptance-manifest template

Every registered family claim fills this manifest.

**Identity block:**

```text
family_id / canonical_setup_id(s):
side:                LONG | SHORT          (never pooled)
style_bucket:        SWING | INTRADAY_QUICK | INTRADAY_SESSION   (never pooled)
recipe_id:                                  (never pooled)
outcome_definition_id:                      (never pooled)
availability_basis:  AS_OBSERVED (mandatory for promotion evidence) |
                     MARKET (declared reconstructed backtests only; cite lag-table version)
capture_mode filter: LIVE/DELAYED only for coverage, latency, queue-exposure,
                     live-shadow, and all promotion claims
episode unit:        dependency_cluster_id — one market move
```

**Evidence tiers (fixed floors):**

| Tier | Independent episodes | Distinct symbols | Distinct sessions | Regime coverage | Display/consequence |
|---|---|---|---|---|---|
| EXPLORATORY | <30 | — | — | — | Display-only, greyed; no advisory text |
| ADVISORY | ≥30 | ≥10 | ≥20 | ≥2 regime states with ≥8 episodes each | May annotate; never ranks live surfaces |
| PROMOTION-ELIGIBLE | ≥100 untouched OOS (of which ≥40 live-shadow) | ≥20, no symbol >20% of episodes | ≥60 | ≥2 regime states with ≥20 episodes each | May enter plan.md sec 7 promotion review |

**Methods block:** intervals, blocks, shrinkage, and recency per 15.4-15.5;
splits and purge per 15.3.

**Multiplicity block:**

```text
n_variants_examined (k):   (family-lifetime; splitting files never resets it)
headline convention:       "best of k" stated on every claim
widening rule:             k > 10  =>  99% interval on holdout AND beats family-median holdout
holdout status:            untouched | spent (a spent holdout is validation evidence only)
```

**Promotion block** (PROMOTION-ELIGIBLE families only): per 15.7.

**Calibration/abstention block:** per 15.6.

## 16. "Best trade style now" engine (v1)

Sections 16-17 are **gated behind plan.md Section 12 items 14-18** (the
canonical opportunity/ranking pipeline), except the minimal Phase-7 Research-tab
table readout. The engine begins as a research/shadow consumer of the current
point-in-time feature snapshot and remains advisory until promotion gates pass.

### 16.1 Style buckets — seeded with the trader's playbooks

Bucket membership is a seeded hypothesis (registered per step 3 below),
revisable from evidence, not a data conclusion. The objectives are frozen:

- **Swing** — standardized house-recipe R at the frozen checkpoint set
  +1/+2/+3/+5/+10/+18 sessions (a fresh outcome-contract superset anchored to
  existing artifacts: tracker short_horizon R@1/R@2, the 10-session forensics
  horizon, and the 18-session house time stop). Seed playbooks: AVWAPE→1σ
  Favorite, AVWAP Retest Followthrough, D1 Band Bounce, SMA50/100/200 Retest,
  Post-Earnings Candle Break, Extreme Move Retest, TOP Weekly Leader.
- **Intraday Quick** — `quick_r` at the 60-minute milestone. Seed playbooks: M5
  band bounce across the three session-VWAP algorithms, EMA8/15/21 bounce,
  PDH/PDL + 10-candle, delayed 5-min ORB 30m+.
- **Intraday Session** — `entry_r` at EOD. Seed playbooks: 8EMA HOD/LOD grind,
  VWAP session-hold/impulse-retest, ORB continuation.

`production_r = 0.5·quick_r + 0.5·entry_r` remains the untouched live alert
ranking; the engine publishes per-bucket ranks BESIDE it, shadow-advisory only.

### 16.2 Pipeline

1. Verify current data health, completed-through times, and source coverage.
2. Build the current market/sector/industry/symbol context vector.
3. Enumerate locked, registered champion setup/style models plus frozen,
   preregistered challengers eligible for that context, carrying explicit
   authority/status; never select the luckiest visible Setup Matrix cell on
   demand.
4. Apply hard structural, liquidity, freshness, risk, and actionability gates —
   including the production RVOL ≥1.0 gate for intraday styles exactly as
   production applies it, retaining sub-1.0 candidates as learning-only
   denominator rows.
5. Retrieve global, family, side, timeframe, and regime evidence.
6. Estimate context-conditioned outcomes with rungs 1-2 shrinkage (15.4).
7. Penalize staleness, provider dependence, excessive uncertainty, extension,
   and overlap with other current ideas.
8. Rank separately by the three bucket objectives.
9. Match current opportunities to supported styles.
10. Publish an immutable advisory snapshot or abstain.

### 16.3 Output contract — v1

Per style bucket and matching opportunity:

- family/side/version and timeframe roles;
- evidence tier (15.8), n_episodes / n_symbols / n_sessions;
- shrunk mean net R ± 90% CI; shrunk win rate ± 95% Wilson;
- empirical realized-R p25/p50/p75 (this IS the outcome spread — no predictive
  model in v1);
- median time-to-payoff; median MFE/MAE;
- last evidence date and current-regime support count;
- all-history vs rolling-recent estimates and the divergence flag;
- entry condition, invalidation, nearest obstacle, and remaining R;
- a counter-evidence line (why this might fail; what's missing);
- Personal Fit as a visibly separate annotation;
- result/abstention codes: `SUPPORTED_STYLE_WITH_CANDIDATE`,
  `SUPPORTED_STYLE_NO_CURRENT_CANDIDATE`, `NO_ACTIONABLE_CANDIDATE`,
  `INSUFFICIENT_CONTEXT_DATA`, `THIN_CONTEXT_EVIDENCE`, `STALE_EVIDENCE`,
  `NO_SUPPORTED_STYLE`.

Do not force quick and EOD evidence into one scalar — the production_r precedent
proves the divergence matters. Publish **two side-by-side rankings**
(quick_r-ranked vs EOD-ranked) instead of a Pareto frontier.

Deferred to the model era (Appendix D): modeled success probability, predictive
distributions, epistemic intervals, OOD distance, top-K/portfolio utility, and
calibration displays.

Every snapshot records `prediction_as_of`, per-timeframe `completed_through`,
`valid_until`, capture-universe coverage, `evidence_trained_through`,
`labels_matured_through`, model/estimate versions, and the manifest_log
position/hash it was computed from.

### 16.4 Objective and personal fit

Canonical ranking order remains:

1. data health and hard risk gates;
2. lifecycle/actionability stage;
3. objective setup and trade-style quality;
4. remaining reward/risk and no-chase;
5. evidence confidence;
6. concentration/correlation;
7. Personal Fit tie-break inside comparable quality/stage bands;
8. separate delivery policy.

Personal Fit may annotate or reorder comparably qualified items. It cannot
change eligibility, hard gates, lifecycle stage, objective expectancy,
monitoring cadence, sound, suppression, or execution.

## 17. Research and trader-facing tools

All tools below are **post-slice and gated behind plan.md Section 12 items
14-18**; the only near-term deliverable is the minimal Phase-7 Research-tab
table readout. That readout is raw canned-query results only — counts, mean R,
and checkpoint values for the two slice setups; shrinkage, intervals, evidence
tiers, and the full Section 16.3 contract arrive with milestone M-E.

- **Current Edge dashboard** — best-supported Swing / Intraday Quick / Intraday
  Session styles now, matching opportunities, deteriorating-style flags,
  evidence tier and freshness, and an honest no-edge state.
- **Setup Matrix** — pivot across family, side, timeframes, level family,
  recipe, regime, catalyst state, and time of day; every cell shows n, distinct
  sessions/symbols/episodes, net R with interval, baseline edge, and tier; small
  cells visible but greyed exploratory.
- **Level Edge Lab** — first-touch vs later-touch, confluence, and
  confirmation-delay questions (Appendix B holds the seed queries).
- **Multi-Timeframe Map** — completed-bar state on M5/M15/M30/H1/H4/D1/W1,
  active levels and clusters, current/next trigger, conflicts, freshness;
  forming frames visibly Preview.
- **Strategy Recipe Comparator** — the same independent occurrences under
  declared policies; correlation explicit, so users cannot mistake 20 recipes on
  20 occurrences for 400 trades.
- **Replay and audit** — reconstruct the chart and decision state at any
  historical `as_of`: bars then available, geometry then known, eligibility,
  blockers, score/rank, alert/impression/action, with the subsequent path hidden
  until released.
- **Research queue and promotion workspace** — idea, hypothesis, registered
  grid, trial ledger, latest run, evidence freeze, status, next gate, rollback.
  AI suggestions enter here; they never edit production configuration.

## 18. Health, observability, and operations

System Health gains exactly **six tiles**:

1. DAS mount / free-GB / 30-day growth;
2. backup age + last spot-restore date;
3. expected-vs-observed bar coverage per (resolution, cohort), worst-5 symbols;
4. inbox/spool backlog + oldest age + quarantine count;
5. last seal/import result;
6. manifest-integrity count (live files not in `manifest_log.jsonl` must be 0,
   counting `_retired/` as expected GC lag).

Experiment reproducibility surfaces in the research workspace, not System
Health. A green storage/collection audit means the mechanics worked. It does not
mean a setup is predictive or promoted.
## 19. Implementation plan

This section is the build order. An AI coding agent implements Phases 0-8 top to
bottom; each phase is committable, testable, and leaves `main` green. Rough
total effort: ~16 agent-sessions.

### 19.0 Already-exists inventory — "Reimplement: nothing"

| Existing artifact | Disposition |
|---|---|
| Per-symbol D1/H1 Parquet stores | Reuse-as-is; wrapped reads feed `bar_d1` |
| HV level stores (~1,539 symbols × 5 yr) + `d1_level_feed` | Reuse-as-is; snapshot into `level_state_daily` |
| `earnings_avwap_anchors.csv` + calendar history | Wrap-as-bronze; feeds `anchor_instance` |
| `calc_anchored_vwap_bands` | Reused as the computation, parity-tested to 1e-9 — never reimplemented |
| Universe builder outputs + watchlist files | Daily snapshot into `universe_membership_daily` |
| Setup tracker JSONL + scenario CSVs (~676 MB) | Wrap-as-bronze; legacy IDs/watermarks preserved |
| Bounce ledgers / day-trade tracker outputs | Wrap-as-bronze |
| `alert_review_events.jsonl` | Wrap-as-bronze (Class A) |
| Regime snapshots / `spy_state_shadow.jsonl` / RS artifacts | Wrap-as-bronze |
| `technical_integrity_events.jsonl` (~108 MB) | Wrap-as-bronze — enables its pending retention cleanup after verified ingestion |
| Run manifests / `job_ledger.jsonl` / `heartbeat.json` | Wrap-as-bronze; feeds `scan_coverage` joins |
| Trader geometry/watch JSONs (`d1_level_watches`, `d1_event_watches`, `alert_chart_watches`, `price_alerts`) | Daily bronze ingest from creation time onward (Class A) |
| Atomic-publish helpers, `scripts/market_session.py` | Atomic-publish pattern reused for the seal protocol; the writer-lease helper remains Drive-export-only (decision 0006) — never used on the lake |
| `scripts/master_avwap_lib/setup_tagging.py`, `scripts/bounce_bot_lib/`, `scripts/setup_docs.py` | Canonical setup-ID source (there is no `scripts/setups/` package) |

### 19.1 Module map

New package `scripts/research_warehouse/`:

| Module | Responsibility |
|---|---|
| `config.py` | `research_store_dir` setting + `TRADINGBOTV3_RESEARCH_DIR` env override + `warehouse_enabled()` no-op guard |
| `schemas.py` | Single source of truth: typed pyarrow schemas for Section 7.1, incl. the deterministic occurrence key |
| `store.py` | 4-step seal protocol, quarantine, retirement/GC |
| `manifest.py` | `manifest_log.jsonl` append/resolve; manifest-resolved read API |
| `spool.py` | `ResearchSpoolWriter`, rollover contract, cap/shedding |
| `ingest_existing.py` | Bronze wraps of the 19.0 inventory |
| `bar_archive.py` | M5 tee, nightly/weekly backfill jobs, yfinance seed, IB pacer integration |
| `aggregate.py` | M5→M15/M30/H1(/H4), D1→W1 under the Section 5.4 contracts |
| `features.py` | Tier-1 feature snapshots (daily + intraday) |
| `occurrences.py` | Detector-output ingestion; deterministic occurrence key; rescan-updates rule |
| `outcomes.py` | Recipe simulation, checkpoint grid, `house_default_v1` |
| `backup.py` | 3-class robocopy jobs + restore check |
| `cli.py` | `build`, `status`, `restore-check`; job-ledger registration; single-flight lock |

Plus: `scripts/ui/services/warehouse_service.py` (job-ledger registration,
Settings field, six Health tiles) and `docs/decisions/0014-das-research-lake.md`
(Appendix E skeleton). Tests as `tests/test_warehouse_*.py` per existing pytest
conventions.

### 19.2 Build order — Phases 0-8

| Phase | Deliverable | Exit criterion | Est. sessions |
|---|---|---|---|
| **0** | Decision record 0014 committed; `config.py`; plan.md item 13a insertion (Appendix E text); runbook publisher-role doc cleanup; confirmation register sent to Aaron | 0014 committed as PROPOSED and approval requested; config no-op verified by `test_warehouse_config.py`. Phases 1+ proceed while approval is pending | 0.5 |
| **1** | Store core: seal protocol, manifest, quarantine, retirement | Crash-matrix tests green incl. the tracker-incident quarantine regression | 2 |
| **2** | Bronze wrap of existing artifacts + daily universe/geometry snapshots (incl. the four watch/level JSONs) | Every 19.0 artifact whose disposition is wrap-as-bronze or daily ingest is ingested with hashes (reuse-as-is rows are wrapped reads only); re-run is a no-op | 1.5 |
| **3** | M5 tee archive + `scan_coverage`/`collection_gap` | Tee proven zero-added-requests; coverage rows reconcile against run manifests | 1.5 |
| **3b** | Shared IB pacer + client-ID allocation (Section 5.3); nightly ETH-inclusive M5/M1 backfill jobs; weekly universe sweep; yfinance 60-day seed | Pacer tests green (`test_warehouse_pacer.py`); champion traffic proven pass-through; backfill idempotent across the TWS restart; seed resumable | 1.5 |
| **4** | `trading_session` + M5→M15/M30/H1 aggregation into `bar_derived` + W1 from canonical D1 (H4 activates post-slice) | DST/half-day/stub tests green; sentinel derived-vs-native H1 parity | 2 |
| **5** | Feature snapshots — exactly the frozen Section 7.1 columns (D1 grid, M5 + M15/M30 EMAs, AVWAP block, context) | `calc_anchored_vwap_bands` parity 1e-9 on golden fixtures; determinism test green | 2 |
| **6** | Occurrences + outcomes for the two slice setups under the 19.3 recipe mapping | Rescan-updates-never-appends proven; `house_default_v1` arithmetic tests green | 2.5 |
| **7** | Read path: `pyarrow.dataset` canned queries; optional DuckDB (read-only, requirements-dev + constraints pin, cp314 win_amd64 wheel verified first); minimal Research-tab table readout (raw canned-query results only, per Section 17) | Manifest-resolved reads consistent under concurrent compaction | 1.5 |
| **8** | Backup/restore + six Health tiles; 20-session forward run (the pilot of Section 5.6) | Pilot checklist complete (items 3-4 staged per 5.6); scripted restore check passed | 1 |

DuckDB is deferred to Phase 7 deliberately: `pyarrow.dataset` (pyarrow 22.0.0 /
pandas 2.3.3, already pinned) covers all slice queries, keeping an unapproved
dependency off the critical path per decision 0012. The write path is
pyarrow-only forever. If no cp314 wheel materializes, stay on pyarrow reads.

### 19.3 First vertical slice — pinned

One end-to-end forward-observed slice: tee capture → seal → derive → features →
occurrences → outcomes → read path → backup/restore, over 20 forward RTH
sessions, shadow-only, main desktop only, zero champion influence. These
sessions are engineering validation only — never efficacy evidence.

**Cohort:**

- **Intraday (M5 tee):** daily union of `longs.txt`, `shorts.txt`,
  auto-populated longs/shorts, and Focus lists, capped at 250 symbols,
  snapshotted into `universe_membership_daily` at first capture each session;
  PLUS the fixed 30-symbol exploration list committed at
  `scripts/research_warehouse/exploration_cohort.txt`. Exploration M5 bars are
  acquired via the tee when a symbol happens to be scanned anyway, otherwise via
  the nightly BACKFILL budget (Phase 3b) — **never by adding symbols to the
  champion scan cohort** (Section 19.6 R3), and never a second independent IB
  requester.
- **Daily (D1):** full screened universe via the existing per-symbol D1 store
  (wrapped as bronze; no new fetches).
- **Sessions:** the tee inherits production RTH scope; nightly ETH-inclusive M5
  backfill begins at Phase 3b — inside the slice — per LD-03, so no slice-window
  ETH history is lost.

**Datasets:** exactly the Section 7.1 thirteen tables plus the two JSONL
ledgers (`imported_bundles.jsonl` stays empty in the slice).

**The two setups** (canonical IDs verbatim from
`scripts/master_avwap_lib/setup_tagging.py`; display labels live in Appendix C
only):

- **`AVWAPE_TO_FIRST_DEV`** (long) — display label "AVWAPE to 1st Dev Favorite".
- **`POST_EARNINGS_CANDLE_BREAK`** (short side).

Occurrences are recorded from detector output — the warehouse never re-detects.
`occurrences.py` implements Section 7.3's identity rule via the deterministic
occurrence key in `schemas.py`.

**The recipes:**

- **`swing_house_v1`** — signal-close entry; structural-level stop with
  2-daily-close failure (1 close for post-earnings families); band-bounce stop
  one band beyond; 50% partial at band 2, trail to band 1, run to band 3;
  18-session time stop.
- **`intraday_bounce_v1`** — completed M5 bounce-bar close entry; production
  per-bounce-type stop; outcomes at the 60-minute quick_r milestone and EOD
  entry_r.
- **Recipe-to-setup mapping (normative for Phase 6):** `swing_house_v1` is the
  primary recipe for both slice setups (both are D1 swing families from
  master-scan detector output). `intraday_bounce_v1` is evaluated only on
  occurrences with a linked BounceBot M5 bounce event from the Phase-2 wrapped
  bounce ledgers (join: symbol + session + bounce-bar time window; the bounce
  event supplies the bounce bar and bounce_type). When no linked bounce event
  exists, no `intraday_bounce_v1` row is produced — the warehouse never
  re-detects. Intraday checkpoint columns under swing recipes populate whenever
  M5 bars cover the entry session, else stay null.
- **Controls (both setups):** `control_fixed_1r2r_v1` (fixed 1R stop / 2R
  target) and `control_time_only_v1` (time-only exit).
- **Registered diagnostic:** signal-bar-extreme + 0.25×ATR(M5,14) stop — a
  diagnostic, not the primary, so tracker parity and existing evidence carry
  over unchanged.
- **Cost/ambiguity:** `outcome_definition_id = house_default_v1` (Section 14.2).

**Paths:** spool = `LOCAL_SETTINGS_DIR/research_spool` (i.e.
`%LOCALAPPDATA%\TradingBotV3\research_spool`), cap 5 GB / 7 days, shedding per
Section 8.4. Lake = `research_store_dir` (Settings; `TRADINGBOTV3_RESEARCH_DIR`
override), location following the `shared_data_dir` precedent in
`scripts/project_paths.py`; never inside the Drive folder; layout per
Section 8.2.

**Config surface:** `research_store_dir` (unset ⇒ warehouse fully disabled via
`warehouse_enabled()`); env override; Settings field + six Health tiles via
`scripts/ui/services/warehouse_service.py`; CLI `build`/`status`/`restore-check`
registered in the existing job ledger with the single-flight lock and the spool
rollover contract.

**Exit gate:** 20 forward RTH sessions captured + aggregation and
`calc_anchored_vwap_bands` parity green + scripted restore verification passed.

**pytest checklist (`tests/test_warehouse_*.py`):**

| Test module | Must prove |
|---|---|
| `test_warehouse_config.py` | Unset `research_store_dir` ⇒ total no-op; env override wins; paths never in Drive |
| `test_warehouse_seal.py` | 4-step seal; crash mid-write leaves artifacts only in `_incoming/`; crash between rename and manifest append reconciled at startup |
| `test_warehouse_manifest.py` | Manifest-resolved reads; query concurrent with compaction returns pre- or post-compaction row set, never a double count; live files not in manifest = 0 |
| `test_warehouse_quarantine.py` | Tracker-incident regression: dirty tail quarantined per symbol/partition, clean remainder publishes, Health count surfaces; wholesale veto only on manifest corruption |
| `test_warehouse_retire.py` | NTFS sharing violation on retiring an open file ⇒ skip and retry next run, fails safe |
| `test_warehouse_spool.py` | Rollover contract; 5 GB/7-day cap; shedding order; D1/M5 never shed |
| `test_warehouse_import.py` | Importer re-run is a byte-identical no-op (idempotent by content hash) |
| `test_warehouse_aggregate.py` | M5→M15/M30/H1 session-anchored rules; DST, half-day, stub-bar duration flags; boundary parity with IB native `useRTH=1` H1 on sentinels |
| `test_warehouse_avwap_parity.py` | `calc_anchored_vwap_bands` parity to 1e-9 on golden fixtures (never reimplemented) |
| `test_warehouse_occurrence.py` | Deterministic occurrence key; hourly rescan updates, never appends; long/short and swing/M5 identities distinct |
| `test_warehouse_outcomes.py` | STOP_FIRST primary + `r_upper_bound`; `house_default_v1` net_r arithmetic; MATURED as derived predicate; enum subsets only |
| `test_warehouse_pit.py` | capture_mode exclusion filter (BACKFILL/RECONSTRUCTED out of coverage/latency/live-shadow); AS_OBSERVED vs MARKET basis; lag-table version pinning |
| `test_warehouse_pacer.py` | Champions pass-through unmetered; capture yields on 162/366; capture errors excluded from `_IBKR_HISTORICAL_FAILURE_COUNT`; client-ID assertion |
| `test_warehouse_build_job.py` | Single-flight lock; sleep/wake resumes or fails cleanly with the job ledger recording truth; TWS-restart resume idempotent |
| `test_warehouse_restore.py` | Full one-month restore verifies hashes + one canned query |

### 19.4 Post-slice milestones

These carry forward — not drop — the unbuilt remainder of the original
workstreams. Only root `plan.md` may assign implementation order or status.

| Milestone | Scope (former workstream) | Exit gate |
|---|---|---|
| **M-A Complete denominator** | Full eligibility lattice past `scan_coverage`: risk sets, evaluation slots, candidate eligibility with rejection reasons (old WS5) | Repeated scans never inflate n; missed-winner analysis has a complete eligible denominator |
| **M-B Recipe simulator** | Full deterministic candidate × recipe simulator + combination control (old WS6) | Every reported R names its exact policy; one occurrence cannot masquerade as many trades |
| **M-C Legacy parity & cutover** | Setup Tracker / Day Trade Tracker / playbook / Move Forensics / Technical Integrity projections reproduce on frozen data; source-by-source cutover per 19.5 (old WS7) | Warehouse can replace research reads without changing live decisions; legacy projection restorable by configuration; short-mirror intentional difference labeled |
| **M-D Registered variation research** | Old WS8. **The first registered study family is M15/M30 structure/trigger value on existing theses** (M15 acceptance + M5 retest vs first-cross entries; M30 EMA15/21 controlled pullback as swing-entry context) before any other grid research; H1 trigger studies rank below M15/M30. Then: AVWAP anchor/band interactions, horizontal/trendline interactions, MTF alignment, confluence redundancy, recipe comparator | Nominations come from untouched validation/test evidence, never the best in-sample slice |
| **M-E Conditional expectancy + style challenger** | Old WS9: rungs 1-2 service, recency estimates, abstention, per-bucket rankings, immutable `current_edge_snapshot_v1` | Grouped tables reproduce from pinned inputs; prediction ledger populated; decile check runs; abstention fires on thin cells; advisory only |
| **M-F Research UI** | Old WS10: Current Edge, Setup Matrix, Level Edge Lab, MTF Map, Recipe Comparator, replay, research queue; bounded AI evidence packages | Aaron can reach every claim's underlying occurrence and recreate its point-in-time chart |
| **M-G Live shadow + review identity** | Old WS11: complete live-session collection; repaired impression/action identity and outcome joins; daily audits | plan.md live-shadow floors + the 15.8 evidence-tier template pass (not the deferred calibration limits); no unpromoted field affects production |
| **M-H One-family canary + promotion** | Old WS12: one narrow family, golden intentional-difference fixtures, advisory → opt-in soft alert → canary; 15.7 evidence on identical risk sets; surveillance manifest (matured-outcome cadence, drift/downside limits, immediate safety rollback, expiry/review date, no retuning from the monitoring window) | Only the approved family/version influences the bounded production surface |
| **M-I Continuous expansion** | Old WS13: add families one at a time, re-running data-health, trial-ledger, portfolio-overlap, canary, and rollback gates each time | Never promote an unrestricted model that can silently redefine its features or training window |

Also post-slice: the manual-geometry capture surface (Section 6.5, old WS4
deliverable), Focus streaming (`reqRealTimeBars`, Section 5.2), H4 derivation,
and the H1/H4/W1 tier-1 feature series (Section 6.1).

Crosswalk to root plan.md milestones (dependency proposals only — root plan.md
always wins):

| Root dependency | Work enabled here |
|---|---|
| Milestone 2 storage migration/authority | DAS lake and local-vs-shared classification (decision 0014) |
| Milestone 3 golden/replay harness | Characterization and migration parity |
| Milestone 4 provider repository | Normalized acquisition and deterministic aggregation |
| Milestone 5 point-in-time repair | Five-column PIT contract, anchors, identities, labels |
| Milestone 6 canonical authority | Complete risk sets/candidates and one lifecycle (M-A) |
| Milestone 7 canonical opportunity/ranking | Current-style challenger and portfolio ranking (M-E) |
| Milestone 9 journal/learning | Impressions, actual execution, Personal Fit linkage (M-G) |

### 19.5 Migration and cutover

Principle: legacy artifacts are wrapped as bronze sources, never rewritten,
never re-owned. The legacy writer keeps writing; the warehouse ingests beside
it.

Per-artifact sequence (tracker first):

1. **Freeze-copy + hash.** Copy the legacy artifact to `_incoming/`, SHA-256 it,
   seal into bronze via the 4-step protocol; the manifest line records the
   source path and git commit. Original file untouched.
2. **Shadow ingest.** The nightly build job re-ingests deltas using legacy
   IDs/watermarks; importer re-run is a no-op.
3. **Parity checks.** Row and key counts match; spot-check N=50 records
   field-by-field; tracker/playbook projections reproduce on frozen data (M-C
   gate). Approved intentional differences are labeled, never silently absorbed
   — including the short-mirror difference (Section 12.4).
4. **Research-reader switch.** One artifact at a time, by configuration:
   research consumers read the warehouse projection; the legacy writer keeps
   writing; live/champion surfaces untouched.
5. **Projection ownership (separately promoted, post-slice).** Only after parity
   has held across a live-session validation day does a projection ever become
   the serving copy — never two independent production writers at any point.

Parity gates: bars (derived-vs-native H1 sentinel; D1 close 0.1% / volume ×100
within 5% / M5 <2% on the 20-symbol sentinel set; corporate-action days
`ADJUSTMENT_WINDOW`-exempt); AVWAP (1e-9 golden fixtures); levels (sources named:
HV stores, `d1_level_feed`, blocking-penalty inputs,
`master_avwap_bucket_state.py`, bounce_bot_lib session/rolling levels);
tracker/playbook (M-C projection parity with the labeled short-mirror
difference).

Rollback: reader rollback = flip the per-artifact config back to the legacy
reader, effective immediately, no data movement. Lake rollback =
`manifest_log.jsonl` is the atomic switch; `_retired/<yyyymmdd>/` holds
superseded files 30 days; restoring a prior state is re-pointing the manifest,
never rewriting files. Legacy files are never deleted by any migration step; the
`technical_integrity` retention cleanup happens only after its bronze ingestion
passes parity and a restore check. A failed migration step leaves artifacts only
in `_incoming/`/`_quarantine/`.

### 19.6 Risk register

| # | Risk | Class | Mitigation |
|---|---|---|---|
| R1 | Capture-caused IB pacing errors trip the champion fetch boundary's Yahoo-only circuit breaker, silently degrading live scans to Yahoo (BF.B/LC blackout precedent) | Correctness, live-impact | Request-layer tagging excludes capture errors from `_IBKR_HISTORICAL_FAILURE_COUNT`; pacer yields instantly on 162/366; `test_warehouse_pacer.py` asserts isolation |
| R2 | Client-ID collision re-creates the 1003 dual-scheduler failure (silent Yahoo fallback on overlap) | Correctness | Fixed allocation table (1010/1011/1020-1029; 1003 retired) asserted at connect; mini-PC excluded until plan.md M1 reconciliation |
| R3 | The tee changes champion timing or adds requests, violating plan.md sec 5 without golden fixtures | Live-impact | Champions pass-through unmetered; tee observes in-memory responses only; pilot item 1 proves zero added requests; no capture code inside champion fetch paths |
| R4 | A bounded dirty tail vetoes a whole publish, blacking out capture for days (tracker week-of-07-13 precedent) | Delivery | Per-symbol/per-partition quarantine with clean-remainder publish; wholesale veto only on manifest corruption; Health quarantine count; regression test pinned to the incident |
| R5 | DAS unavailable or wedged (DriveFS-style silent unmount precedent) loses a session of capture | Delivery | Local spool 5 GB/7 days with declared shedding order; D1/M5 never shed; Health red; continuity policy accepts ≤1 session loss with backfill recovery |
| R6 | Windows file locking: compaction retirement or backup hits an NTFS sharing violation on an open file | Correctness | Reads are manifest-resolved (never directory globs), so a lingering file is harmless; retirement skips and retries next run; robocopy is append-only incremental, never /MIR |
| R7 | Silent provider blend or bad fallback data contaminates evidence (round-lot ×100 RVOL bug precedent) | Correctness | Provider recorded per row; sentinel parity job (Section 9.1); `PROVIDER_CONFLICT` rows; yfinance forming last bar always dropped |
| R8 | Look-ahead leakage: BACKFILL rows enter coverage/latency/live-shadow/promotion evidence | Correctness | Single capture_mode filter applied everywhere; AS_OBSERVED basis mandatory for those claims; `test_warehouse_pit.py`; MARKET basis requires a cited versioned lag table |
| R9 | Episode inflation: rescans/variants/re-arms counted as independent samples (tracker episode-dedup lesson) | Correctness | Deterministic occurrence key (rescan updates, never appends); `dependency_cluster_id` as the episode unit; evidence floors count episodes only |
| R10 | Scope creep back toward enterprise machinery (generations, leases, drill calendars, FDR ceremony) burns agent-sessions without evidence value | Delivery | Locked decision log with explicit reopen triggers; six-tile Health cap; Appendix D activation triggers; Section 22 anti-goal wording |
| R11 | yfinance seed throttling/ban mid-scrape leaves a biased partial 60-day archive | Delivery | Trickle over several nights, chunked with backoff; per-symbol completion ledger; gaps recorded as `collection_gap` rows, resumable and idempotent |
| R12 | Python 3.14 wheel gaps (DuckDB cp314 win_amd64) stall the read path | Delivery | Write path pyarrow-only forever; Phases 1-6 read via `pyarrow.dataset`; DuckDB deferred to Phase 7 behind an install-verification precondition; fallback = stay on pyarrow reads |
## 20. Verification strategy

### Data and time

- DST, half-day, holiday, extended-hours, and exchange-session fixtures.
- Completed versus forming M15/M30/H1/H4 joins at boundary times, including
  legitimate final-session stubs.
- Late arrivals, provider revisions, missing constituents, and source switches.
- Corporate-action and symbol-change point-in-time replays.
- Universe membership and catalyst `known_at` tests.
- AS_OBSERVED vs MARKET basis tests; capture_mode exclusion-filter tests;
  lag-table version pinning tests.
- Static dependency audit proving feature/setup/prediction code cannot read
  outcome tables.

### Levels and geometry

- Frozen AVWAP sigma parity at every existing consumer.
- Anchor confirmation and revision timing.
- First/second/nth touch identity.
- Manual level/trendline creation, edit, deletion, and historical replay.
- Pivot-confirmation lag and no backward-projected trendline evidence.
- Confluence dedupe and correlated-family caps.

### Identity and denominators

- Swing versus M5 and long versus short remain distinct.
- Multiple setups/anchors on one symbol remain distinct.
- Rescans never create extra occurrences, attempts, impressions, or outcomes.
- Re-arm and material thesis revisions create the correct linked IDs.
- Every aggregate reconciles to risk set, assigned, returned, data-complete,
  eligible, rejected, surfaced, and matured counts.

### Outcomes and statistics

- Entry/fill/gap/stop/target/expiry/cost semantics.
- Same-bar ambiguity and censoring.
- Alternative recipes/horizons share one dependency-clustered market episode.
- Purged chronological splits and immutable holdouts.
- Matched-control determinism, trial-ledger/holdout discipline, and
  cluster-aware intervals.
- Reproducible evidence from frozen manifests and code/config hashes.

### Storage and operations — the eight scenarios

1. Crash mid-write leaves artifacts only in `_incoming/`.
2. Crash between rename and manifest append is reconciled at startup.
3. A query concurrent with compaction returns either the pre- or
   post-compaction row set via the manifest-resolved file list, never a double
   count.
4. Importer re-run is a no-op.
5. DAS unmounted → spool + red Health.
6. File-in-use retirement fails safe (NTFS sharing violation on retiring an open
   Parquet file — skip and retry next run; harmless because reads are
   manifest-resolved).
7. Windows sleep/wake during the EOD build job resumes or fails cleanly with the
   job ledger recording the truth.
8. Full one-month restore verifies hashes + one canned query.

No GUI render path performs provider or large warehouse reads.

### Promotion

- Research features contribute exactly zero to champion score/alerts.
- One switch restores the prior champion without code revert or evidence loss.
- Every promoted value has a cited fixture, evidence snapshot, approval,
  effective time, expiry/review date, and rollback target.

## 21. Success metrics

### Corpus integrity

- At least 99.9% expected completed-bar coverage within each declared
  (resolution, cohort, acquisition-mode) scope, with every non-collected or
  missing interval explicit.
- Zero unexplained duplicate canonical keys.
- Zero silent provider blends, backward-known levels, or future feature inputs.
- 100% `manifest_log.jsonl` coverage for sealed partitions and evidence freezes.
- Successful rebuild and the scripted/semiannual restore checks pass (loss
  policy: ≤1 session of raw capture plus derived data).

### Research integrity

- 100% experiments registered with immutable split and outcome definitions.
- 100% claims linked to market-episode/dependency-cluster IDs and source
  manifests.
- Complete reporting of sessions, symbols, episodes, missingness, uncertainty,
  costs, and the trial ledger (`n_variants_examined`).
- Losing/inconclusive experiments retained.
- No production contribution from research-only fields.

### Opportunity quality

Monitoring metrics — tracked with intervals monthly; formal champion-relative
testing only at promotion:

- Ready precision, precision@1/@3, planned remaining reward/risk, model expected
  net R, MFE/MAE, missed-winner rate, false-confirmation rate, time-to-payoff.
- Stability across validated sides, regimes, time periods, providers, and
  liquidity/volatility buckets.
- A new setup/style demonstrates incremental portfolio expectancy after overlap
  with current champions.
- The system abstains when current context is unsupported.

### Trader usefulness

- Aaron can see what is working for Swing, Intraday Quick, and Intraday Session
  separately.
- Every recommendation explains entry, invalidation, obstacle, planned
  reward/risk, model expected net R, timeframe alignment, evidence strength,
  counter-evidence, and freshness.
- Passed, missed, late, and actual execution results can be compared fairly.
- Any historical recommendation can be replayed exactly as it looked at the
  time.

## 22. Explicit anti-goals and traps

- No single giant mutable CSV or Drive-synchronized database.
- No arbitrary Cartesian explosion of every indicator combination.
- No silent change to AVWAP sigma math.
- No future-selected anchors, pivots, trendlines, universes, or regime labels.
- No mixing M1-derived and provider-native higher-timeframe bars without
  identity.
- No symbol/date-only setup identity.
- No treating repeated scans, horizons, or recipes as independent samples.
- No optimizing the final holdout after results are inspected.
- No promoting the luckiest cell from thousands of trials.
- No raw win-rate ranking without costs, baseline, downside, and uncertainty.
- No conflating setup quality, trader preference, and execution performance.
- No mode-driven missing rows: EVENING/DESK/AWAY presentation differences are
  recorded separately from opportunity observation.
- No last-price crossing treated as a completed-bar setup confirmation.
- No AI-created opportunity IDs, factual levels, score changes, or automatic
  promotions.
- No process ever rewrites a sealed lake file in place; only the build/import
  job writes the lake; supported reads resolve their file set from
  `manifest_log.jsonl`; DuckDB opens Parquet read-only; any `.duckdb` file is a
  disposable local cache, never shared, never authoritative.
- No assumption that RAID/DAS is a backup.
- No deletion of failed experiments or historical manual chart geometry.
- No claim that "best now" exists when evidence is stale, sparse, unstable, or
  outside the validated context.

## 23. Decision log (LOCKED)

Every formerly open question is decided. Format: decision → rationale → reopens
if. Agents do not re-litigate these; numbers live in their owning sections and
are pointed to, not repeated.

- **LD-01 Ownership and machines** (Section 8.4). Main desktop is the sole lake
  writer; GUI-owned spool writer + one post-scan/EOD CLI build job; no daemon,
  no leases; mini-PC excluded from Phases 0-8 with the drop-folder bundle design
  recorded for later. *Rationale:* one machine/one writer trivially satisfies
  the one-owner invariant; building cross-machine import atop a known scheduler
  collision imports the failure the plan prevents. *Reopens if:* plan.md M1
  client-ID reconciliation lands AND a concrete away-day capture need exists
  that nightly backfill cannot cover.
- **LD-02 Base intraday archive** (Section 5.2). M5 base; M15/M30/H1 derived;
  H2 cut; H4 tier-2; tee-first; weekly universe sweep + nightly active-cohort
  backfill; M1 ≤150→300; Focus streams ≤40. *Rationale:* IB pacing is the
  binding constraint, not disk; M1 adds zero fidelity for M15/M30. *Reopens if:*
  pilot measurements contradict the floors (changes allocations, never the
  architecture).
- **LD-03 Extended hours** (Section 5.4). Raw M1/M5 backfill capture is
  ETH-inclusive from the first backfill job onward (Phase 3b, inside the
  slice); the tee inherits production RTH scope; derived aggregates RTH-only in
  v1. *Rationale:* `useRTH=0` is free, premarket extremes are first-class
  trader levels, and uncaptured ETH history is permanently lost. *Reopens if:*
  never for raw capture; ETH aggregates are additive contracts later.
- **LD-04 Store** (Sections 8.1-8.3). Immutable Parquet/Zstd + 4-step seal +
  manifest-log read authority; pyarrow-only writes; DuckDB read-only and
  deferred to Phase 7 (requirements-dev pin, cp314 wheel verified first);
  decision record 0014. *Rationale:* pyarrow/pandas already pinned; a catalog
  artifact protects against concurrency that cannot occur. *Reopens if:* no
  cp314 wheel by Phase 7 (stay on pyarrow), or a second concurrent writer is
  genuinely required.
- **LD-05 Universe versioning** (Section 7.1). Daily list snapshots at first
  capture; never backfilled from today's files. *Reopens if:* intraday watchlist
  mutation demonstrably misattributes cohort membership (add an intraday
  cadence; never rewrite history).
- **LD-06 Style buckets and v1 output** (Section 16). Three buckets with frozen
  objectives (+1..+18-session checkpoints / quick_r@60m / entry_r@EOD),
  playbook-seeded; empirical v1 output; two side-by-side rankings; production_r
  untouched. *Rationale:* matches the measured quick-vs-EOD rank divergence
  (corr ≈0.33); every v1 field computable on day one. *Reopens if:* rung-3
  activation unlocks the deferred model contract.
- **LD-07 Recipes, ambiguity, cost model** (Sections 14.2, 19.3).
  `swing_house_v1` + `intraday_bounce_v1` + two controls + registered ATR
  diagnostic; STOP_FIRST primary; `house_default_v1` costs. *Rationale:*
  transcribes the documented house exits and existing tracker methodology, so
  parity is free. *Reopens if:* a measured NBBO source proves systematically
  unavailable at signal time (fallback becomes primary under a new
  `outcome_definition_id`).
- **LD-08 Human geometry** (Section 6.5). Slice ingests the four existing
  watch/level JSONs daily; the drawing capture surface is a named post-slice
  deliverable. *Reopens if:* never; timing may only move earlier.
- **LD-09 Anchor scope** (Section 6.2). Slice = current/previous earnings;
  tier-1 post-slice = post-earnings-candle + manual; tier-2 = gap/catalyst,
  confirmed pivot, period opens; cut = 52-week/breakout/volume-thrust anchors.
  *Reopens if:* a registered study names a cut anchor (additive registry entry).
- **LD-10 Retention** (Section 8.3). Indefinite; review trigger 250 GB.
  *Rationale:* ≤40 GB/yr worst case; the tracker blackout says bias toward
  writing. *Reopens if:* the lake crosses 250 GB or the DAS shrinks.
- **LD-11 Backup** (Section 8.5). 3-class policy; scripted restore check at
  slice exit + semiannual spot restore; no drill calendar, no RPO/RTO matrix.
  *Reopens if:* a restore check fails, or Class A outgrows Drive sync (~1 GB).
- **LD-12 Continuity, spool, shedding** (Section 8.4). Loss policy one sentence
  (≤1 session raw + derived); spool 5 GB/7 days; shedding order fixed; D1/M5
  and champions never shed. *Reopens if:* a measured DAS-outage day shows 5 GB
  covers <~3 sessions.
- **LD-13 Review cadence** (Section 15.1). Weekly automated evidence-health
  report; monthly research review as the only holdout/mart inspection point;
  the fixed cadence IS the repeated-look control. *Reopens if:* more than one
  person inspects holdouts, or rung-3 activates.
- **LD-14 Data rights** (Section 8.6). Personal-use one-line policy; no
  licensing-review gate; one optional non-blocking confirmation question.
  *Reopens if:* any plan proposes sharing/selling/bulk-exporting archive
  contents, or provider terms change materially.
- **LD-15 PIT columns and bitemporal scope** (Sections 9.3, 9.5). Five columns;
  two per-experiment bases; bitemporal intervals only on the revisable
  reference datasets enumerated in Section 9.5 (including `anchor_instance`).
  *Reopens if:* a PIT_RECONSTRUCTED backtest phase is registered (nullable
  modeled-clock columns; rewrites nothing).
- **LD-16 Multiplicity** (Section 15.1). Trial ledger + "best of k" + widening
  rule + holdout discipline; FDR/gatekeeping/FWER/sequential methods deferred.
  *Reopens if:* a family exceeds ~100 variants, or a second researcher joins.
- **LD-17 Estimation and floors** (Sections 15.3-15.4). Rungs 1-2 only;
  per-family rung-3 climb condition; evidence-tier floors replace power
  calculations. *Reopens if:* the climb condition is met for a family.
- **LD-18 Validation numbers** (Section 15.3). Purge 18 sessions swing / 1
  intraday; expanding walk-forward, 3-month test blocks, ≥2 folds;
  session/market-episode bootstrap blocks. *Reopens if:* the house time stop
  changes, or a registered outcome contract exceeds 18 sessions.
- **LD-19 Recency** (Section 15.5). Two estimates + divergence flag; regime is
  a grouping column. *Reopens if:* rung-3 climb for a family.
- **LD-20 Calibration and abstention** (Section 15.6). Prediction ledger +
  quarterly decile check + support-count `THIN_CONTEXT_EVIDENCE`. *Reopens if:*
  rung-3 climb.
- **LD-21 Denominator grain** (Section 13). Light per-(risk_set, symbol) grain
  with per-family status map; slice ships `scan_coverage` only; exploration
  cohort BACKFILL-only, unweighted. *Reopens if:* milestone M-A begins (extends
  the grain; never rewrites `scan_coverage`).
- **LD-22 Grid reduction** (Sections 6.1-6.3). Tier-1 = 17 trader-actual MA
  series + six new M15/M30; session VWAPs frozen at the three production
  algorithms ±1σ; the 1σ favorite zone is the primary AVWAP structure; cut list
  explicit. *Reopens if:* Aaron's confirm-or-amend names a missed series, or a
  registered study nominates a tier-2 cell.
- **LD-23 M15/M30 research priority** (Section 19.4). The first registered
  study family is M15/M30 structure/trigger value; H1 trigger studies rank
  below. *Reopens if:* Aaron redirects the queue (trader-directed by
  definition).
- **LD-24 Canonical W1** (Section 5.4). W1 derived from canonical D1;
  provider-native W1 is a validation variant. *Reopens if:* sentinel parity
  shows persistent derived-vs-native divergence beyond corporate-action
  windows.
- **LD-25 Pilot role** (Section 5.6). Confirmation + measurement, never design.
  *Reopens if:* never — the pilot's measurements are the designed adjustment
  path.
- **LD-26 plan.md slotting** (Appendix E). Trader-directed Section 12 item 13a,
  scoped to Phases 0-8, shadow-only, no golden fixtures required because no
  champion behavior changes. *Reopens if:* plan.md Section 12 is reordered by
  Aaron.
- **LD-27 "Banger"** (Section 6.9). Excluded from schemas; free-text `tags`
  attachment point; standing non-blocking question. *Reopens if:* Aaron supplies
  a precise definition.
- **LD-28 Storage failure scenarios** (Section 20). NTFS file-in-use retirement
  and sleep/wake added; clock-skew/lease/two-machine-collision tests removed as
  unreachable. *Reopens if:* cross-machine import activates (re-adds exactly the
  importer idempotency tests already specified).

### Confirmation register for Aaron (non-blocking; confirm-or-amend; none blocks Phase 0-8 code)

1. Confirm the tier-1 MA grid (Section 6.1) matches your chart templates.
2. Supply a "banger" definition, or leave the `tags` column free-text.
3. Confirm the Pre/Post-Earnings AVWAPE anchor semantics (candle before the gap
   vs post-earnings — one code family, one canonical ID).
4. Optional data-rights comfort check (personal-use archive, LD-14).
5. Confirm the fixed 30-symbol exploration list contents.
6. Confirm the Class-A backup set includes everything you consider
   irreplaceable.

## 24. Definition of done

This program is complete when:

1. The DAS research corpus can be rebuilt from immutable, checksummed evidence
   via `manifest_log.jsonl` and the 3-class backup.
2. Every bar, anchor, level, trendline, feature, setup, recipe, and outcome is
   versioned and point-in-time reproducible.
3. M5/M15/M30/H1/H4/D1/W1 states align without forming-bar confirmation.
4. The complete opportunity denominator is retained, including quiet/rejected
   and zero-opportunity cases.
5. Current production setup/tracker outputs reproduce from the warehouse.
6. Setup quality, trade-policy quality, trader selection, and actual execution
   can be evaluated independently and together.
7. Technical variations are tested under immutable walk-forward splits with
   controls, uncertainty, and trial-ledger/holdout safeguards.
8. The Current Edge engine can identify supported Swing, Intraday Quick, and
   Intraday Session styles—or abstain—with cited evidence.
9. Any historical recommendation can be replayed with the exact data and chart
   geometry available at the time.
10. The GUI, satellite, Auto/Away report, and AI evidence package consume the
    same verified snapshot.
11. No research field affects a live score or alert before golden fixtures,
    shadow evidence, canary gates, Aaron approval, and tested rollback.
12. Storage loss, corruption, migration, backup, and restore checks pass
    without destroying the last verified corpus.
13. The six Health tiles are live and the semiannual spot restore is logged.
14. The application remains decision-support only and never executes orders.
## Appendix A — Documentation traceability

| Source | Constraint carried into this plan |
|---|---|
| `docs/AWAY_SCANNER_RUNBOOK.md` | Single designated writer, truthful freshness, atomic last-good publication, explicit takeover and Drive limitations |
| `docs/BROKER_ADAPTERS.md` | App-owned provider interfaces, IBKR primary/Yahoo fallback, source provenance, read-only execution imports |
| `docs/decisions/0001-decision-support-only-no-order-execution.md` | Permanent decision-support boundary |
| `docs/decisions/0002-champion-challenger-shadow-promotion-ladder.md` | Research/shadow/advisory/canary/promotion separation and rollback |
| `docs/decisions/0003-ibkr-primary-yahoo-fallback-market-data.md` | Provider hierarchy and visible fallback identity |
| `docs/decisions/0004-pyside6-consumer-ui-tk-legacy-during-migration.md` | Headless/core data services with Qt as consumer; no warehouse logic in widgets |
| `docs/decisions/0005-cloud-synced-home-folder-file-storage.md` | Plain-file/Drive rule for operational data — stays fully in force; decision 0014 (Appendix E) scopes the DAS lake as a separate append-only storage class |
| `docs/decisions/0006-writer-lease-fencing-for-shared-exports.md` | One owner, fencing, fail-closed ambiguity for Drive-shared exports; the lake has exactly one writer, so lease/fencing applies only to the Drive channel |
| `docs/decisions/0007-completed-bars-only-for-state-transitions.md` | Completed bars confirm; forming bars are Preview |
| `docs/decisions/0008-frozen-anchored-vwap-sigma-formula.md` | Running-deviation AVWAP sigma remains frozen and versioned |
| `docs/decisions/0009-golden-fixtures-before-detector-changes.md` | Characterization before any scoring/detector change |
| `docs/decisions/0010-ai-in-the-loop-review-policy-annotation-only.md` | Preference AI ranks/annotates only and never suppresses |
| `docs/decisions/0011-one-way-evidence-grounded-ai-advisory.md` | AI output cites immutable evidence and has no mutation path |
| `docs/decisions/0012-layered-requirements-with-constraints-pin.md` | Core/headless dependency placement and reproducible pins |
| `docs/decisions/0013-plan-md-authority-hierarchy.md` | Root `plan.md` remains authoritative |
| `docs/EVENING_MODE_RUNBOOK.md` | Same discovery semantics across modes; preserve open+30/+45/+60 persistence and zero-recommendation truth |
| `docs/FIRST_SESSION_CHECKLIST.md` | Session validation, artifacts, clock discipline, multi-machine drills, and the writer-role wording to clean up (Phase 0) |
| `docs/FOCUS_PRICE_ALERTS_PROPOSAL.md` | Versioned user levels, one-fire-per-arm lifecycle, sticky delivery, last-price crossing separate from detector confirmation |
| `docs/MACOS_SETUP.md` | Configurable cross-platform paths, per-machine local state, no Windows-only path assumptions in core storage |
| `docs/MULTI_MACHINE_DESK_PROPOSAL.md` | Engine/data ownership remains on main; satellites consume relay snapshots and send acknowledged intents only |
| `docs/REGIME_INFRASTRUCTURE_PHASE1_RUNBOOK.md` | Scheduled point-in-time regime snapshots, explicit proxies/missing snapshots, no retroactive reconstruction |
| `docs/REVIEW_LEARNING_LOOP.md` | Impression/action/outcome loop, shrinkage, annotation-only preference policy, and known identity limitations |
| `docs/SETUPS_MAJOR.md` | Current swing/intraday ontology, house exits, favorite-zone doctrine, major levels, and production semantics |
| `docs/SETUPS_TEST.md` | Forward/backfill/reverse research harnesses, controls, point-in-time caveats, and promotion discipline |
| `docs/SHIP_READINESS.md` | Internal tool remains the priority; platform/dependency changes stay layered and operationally supportable |

## Appendix B — Example high-value research queries

**First-quarter registered research order** (LD-23 — the M15/M30 studies lead,
before any other grid research; the numbers reference the list below): 1) B3
M15 acceptance + M5 retest vs first cross; 2) B6 post-earnings short style M15
vs M30; 3) B4 band-bounce stop comparison; 4) B9 fast +1R vs EOD divergence by
family; 5) B2 first vs later touches. B8 is queued next once enough matured
regime-tagged outcomes exist; B1/B5/B7/B10 wait for warehouse maturity.

1. Which long setup/recipe performs best when SPY is in a completed pullback,
   the stock holds D1 earnings AVWAP, M30 EMA15 is rising, and M5 reclaims PDH?
2. Does the first M15 touch of H1 EMA21 after a D1 2nd-deviation power-hold
   episode outperform the second and third touches?
3. Does waiting for M15 acceptance plus M5 retest improve expectancy enough to
   offset worse remaining R versus the first close through a D1 trendline?
4. Which stop works best for AVWAP band bounces: one band beyond, signal-bar
   extreme, structural close failure, or fixed ATR—and in which regimes?
5. Do D1 SMA50 retests improve when a prior-week high and human horizontal zone
   form a distinct-provenance confluence cluster?
6. Which short post-earnings style performs best in weekly-weak names: earnings-
   candle break, previous-AVWAPE bounce, or volume break, and on M15 versus M30?
7. Are EMA15 and EMA21 genuinely different edges on each timeframe, or
   correlated proxies whose apparent winner changes by sample?
8. Which current setup families deteriorate first when volatility expands,
   sector participation weakens, or market state turns choppy?
9. Which setups produce fast +1R but poor EOD outcomes, and which need longer
   management to realize their edge?
10. After controlling for setup, regime, and queue exposure, where does Aaron's
    selection improve the bot, and where is execution—not selection—the leak?

## Appendix C — Seed setup ontology requirements

Before migration, generate this registry from code and have Aaron review it.
Canonical-ID sources: `scripts/master_avwap_lib/setup_tagging.py`,
`scripts/master_avwap_lib/setups/`, `scripts/bounce_bot_lib/`,
`scripts/setup_playbook_study.py`; descriptions in `docs/SETUPS_MAJOR.md` and
`docs/SETUPS_TEST.md`. Required columns are `canonical_setup_id`, aliases,
parent/variant, role (`TRADE_SETUP`, `CONTEXT`, `WATCH_STATE`, `CONTROL`,
`FALLBACK`), status, supported side, structural/context/trigger timeframes,
exact completed-bar trigger, primary recipe, exclusivity group, detector/config
version, and current weight authority. Prose evidence strings are not weight
authority.

Vertical-slice display-label mapping: `AVWAPE_TO_FIRST_DEV` ↔ "AVWAPE to 1st
Dev Favorite" (labels are display-only; `setup_occurrence` stores canonical IDs
only).

| Seed family/group | Role/status to preserve | Identity rule |
|---|---|---|
| AVWAPE to 1st Dev Favorite (`AVWAPE_TO_FIRST_DEV`) | Production trade setup | Parent favorite thesis; completed trigger distinct from zone residency |
| AVWAP Retest Followthrough | Production trade variant | Retest-hold entry; fold/compare with parent without double-counting one move |
| AVWAP Breakout | Production momentum variant | Cross/chase is distinct from later retest |
| AVWAP Band Bounce | Production trade setup | Band and touch order in identity; stop recipe is separate |
| Extreme Move Retest | Production trade setup | Displacement episode plus first controlled retest |
| SMA50/100/200 Breakout and Retest | Production family with period variants | Reclaim/watch and confirmed retest are separate states |
| TOP Weekly Leader | Context/basket plus linked daily trigger | Weekly pattern alone is not the entry |
| Favorite Zone Watch | Watch state | Never counted as a triggered trade setup |
| General/Untagged | Diagnostic fallback | Must not become a pooled "setup" edge |
| Post-Earnings Candle Break | Production, evidence accruing | Mutually exclusive with the 52-week variant for one trigger |
| Post-Earnings 52-week Break | Production family | Separate extreme-break thesis and exclusivity group |
| Post-Earnings AVWAPE Bounce | Production family with side asymmetry | Preserve long confirm-only/weak evidence and short hypothesis separately. **Canonical ID is `POST_EARNINGS_AVWAP_BOUNCE`** (`setup_tagging.py` `_FAMILY_TAGS`); the AVWAPE-spelled tag and the "Pre-Earnings AVWAPE Reject" label are aliases resolved by `_TAG_ALIASES` (verified 2026-07-31) — one code family, one canonical ID; short-after-gap-down / long-mirror-reclaim doctrine annotated on the side-asymmetry note; anchor semantics is the Section 6.2 dictionary item on Aaron's confirmation register |
| Mid-Earnings EMA15 Retest | Production family | Requires the prior 2nd-deviation-zone episode |
| Mid-Earnings EMA21 Retest | Production sibling | Correlated with EMA15; explicit family wins if both fire |
| Mid-Earnings 1st-Dev Retest | Production sibling | Deepest retest; same episode dependence cluster |
| 2nd-Dev Power Hold | Context episode plus long-only research trade thesis | Two linked IDs; `mid_earnings_above_2nd_stdev` is an alias, not another sample |
| Standard/Dynamic/EOD VWAP families | Production intraday groups | Exact algorithm, band, confluence, and impulse-retest variants retained |
| EMA8/15/21 bounce | Production intraday siblings | Period/timeframe/touch count explicit; correlated-family cap |
| Rolling 10-candle and PDH/PDL | Production intraday levels | Exact rolling/session definition and interaction trigger retained |
| H1 EMA10 / blue-after-red / green-to-yellow | Production intraday HTF families | H1 state and lower-timeframe delivery remain distinct |
| `h1_riding_15ema` | CONTEXT / confirm-only, PROVEN-eligible trait | Never a triggered trade setup |
| Regime-pause RS/RW | Production/shadow status from released config | Tied to exact market episode and aligned RS window |
| ORB breakout/breakdown | Production intraday family | Opening-range definition and earliest eligible time versioned |
| EMA8 HOD/LOD grind | Production intraday family | Persistence episode and new-extreme trigger separated |
| `baseline_every5` | CONTROL | Anchors every playbook comparison; never tradable |
| Post-Earnings Gap-Hold (`post_earnings_gap_hold3`) | Research/playbook family (from `scripts/setup_playbook_study.py`) | Separate ID and exclusivity group from the raw 52w break; **NO 52-week condition in its identity** (the detector has none — re-adding a 52w gate would recreate the exact spurious gating the trader flagged); weekly-strong is a measured evidence segment (+0.54R, n=73, evidence pointer only), never part of identity |
| Weekly 8EMA Hold and Retest | Study context/basket | Completed-week streak has no look-ahead; daily trigger separate |
| H1/H4 EMA15 Rejection | Study family | H1 and H4 are distinct variants with session aggregation IDs |
| 1st-Dev Breakout | Study family | Explicit cross variant compared with retest |
| 2nd-Dev Breakout | Chase/control study | Must not alias power-hold residence |
| Volume Thrust | Research family | Volume baseline, directional move, and AVWAP side versioned |
| Quiet Pullback Resume | Research family | Countertrend sequence plus resumption trigger |
| Golden Pullback plus Volume | Research family | Forensics-derived association awaiting forward edge |
| Post-Earnings Volume Break | Research family | Both sides retained; known short asymmetry reported, not assumed |

`PROVEN` is an evidence qualification on a segment, not a new setup occurrence.
Mute semantics remain setup-identity-specific and degrade to CAUTION rather than
suppression. "Banger" remains unresolved until Aaron supplies a precise
definition; the migration must not infer one (LD-27).

## Appendix D — Deferred statistical and operational apparatus

Each entry: what it is, why deferred at solo-trader n, and its activation
trigger. Until a trigger fires, none of this is built.

- **FDR families & multiple-testing correction.** Formal false-discovery
  control over large trial counts; at 5-30 variants/family the trial ledger and
  holdout discipline do the work. *Activate if a family's lifetime trial ledger
  exceeds ~100 variants (matching LD-16), or a second researcher joins.*
- **Hierarchical gatekeeping / FWER allocation.** Familywise-error budgeting
  across simultaneous claims. *Activate if simultaneous multi-family promotion
  is ever attempted.*
- **Sequential-error methods.** Alpha-spending for continuous monitoring; the
  fixed monthly cadence is the current control. *Activate if review cadence
  stops being the fixed monthly control.*
- **Power/design-effect calculations.** Replaced by the fixed evidence floors of
  Section 15.8. *Activate on rung-3 climb.*
- **Estimation rungs 3-5** (regularized regression/additive models; trees;
  survival/competing-risk). *Rung 3 activates per family at ≥300 episodes +
  monotone 3-bin effect + 2-fold stability; rungs 4-5 ≥2 years out.*
- **Full calibration/OOD contract** (Brier/log loss, slope/intercept,
  reliability error, OOD distance, calibration-age limits). *Activate on rung-3
  climb.*
- **Regime posterior/change-point models, decay half-lives.** *Activate when
  regime-conditioned support counts clear ADVISORY floors.*
- **Propensity calipers/weights/balance diagnostics; exploration design
  weights.** *Activate if full-universe expectancy claims from adaptive
  acquisition are ever needed.*
- **Dual availability clocks / modeled simulated availability
  (PIT_RECONSTRUCTED).** *Activate if a reconstructed-history backtest phase is
  approved; added as nullable columns, no rewrite.*
- **Predictive-model output contract** (success probability, predictive
  distributions, epistemic intervals, top-K/portfolio utility). *Activate with
  rung ≥3 models.*
- **Mini-PC drop-folder import** (design recorded in Section 8.4). *Activate
  after the plan.md M1 client-ID/dual-scheduler reconciliation.*

## Appendix E — Governance attachments

### E.1 Decision record 0014 skeleton (`docs/decisions/0014-das-research-lake.md`)

```markdown
# 0014 — DAS research lake as a new append-only storage class

Status: PROPOSED (Phase 0 confirms; drafted by the plan revision)
Date: (fill at commit)
Relates to: 0005 (home-folder/mutable-state policy), 0012 (dependency pinning),
            plan.md sec 12 item 13a

## Decision
1. A new storage class exists: the research lake — immutable Parquet/Zstd files
   on the trader-owned DAS at `research_store_dir`, written only by the main
   desktop's build/import job via the 4-step seal protocol, read via
   manifest_log.jsonl.
2. Decision 0005 remains FULLY IN FORCE for operational mutable data:
   watchlists, reports, JSONL evidence logs, and every live surface stay in the
   Drive home folder / %LOCALAPPDATA% exactly as today. Nothing operational
   moves to the DAS.
3. The Drive home folder additionally carries: (a) nightly Class A mirrors
   (irreplaceable-small research artifacts), and (b) the future research_inbox/
   bundle channel — always whole immutable files, never a live database.
4. No mutable database file ever lives in the Drive folder or on the DAS. Any
   .duckdb file is a disposable machine-local cache, never shared, never
   authoritative.

## Consequences
- One new config key (research_store_dir; TRADINGBOTV3_RESEARCH_DIR override);
  warehouse fully disabled when unset.
- Backup: 3-class policy per plan sec 8.5; Class B lake copies live on a second
  physical disk, never in Drive.
- The lake is out of scope for decision 0005's single-file atomic-publish rules;
  its integrity contract is the seal protocol + manifest instead.

## Rejected alternatives
- Extending 0005 to cover the lake (mutable-publish semantics are wrong for an
  append-only archive).
- A shared DuckDB/SQLite database as the store (Windows file locking,
  single-writer concurrency model, Drive sync hazards).
- Hosting the lake inside the Drive home folder (sync latency, DriveFS wedge
  precedent 2026-07-17, quota).
```

### E.2 plan.md Section 12 item 13a insertion text

To be inserted by Aaron (or Phase 0 with Aaron's approval) after item 13, under
the established trader-directed 7a/7b convention:

> **13a. (Trader-directed) Research warehouse Phases 0-8 —
> `docs/ULTIMATE_SETUP_DATABASE_PLAN.md`.** Shadow-only additive evidence
> capture with read-only consumers: DAS Parquet lake, M5 tee archive, derived
> M15/M30/H1, tier-1 feature snapshots, two-setup occurrence/outcome slice, and
> the 20-session pilot. Zero detector/score/ranking/alert influence, hence no
> golden fixtures required (no champion behavior changes). Main desktop is the
> sole lake writer; the mini-PC is excluded until the client-ID/dual-scheduler
> reconciliation (M1) lands. The plan's Sections 16-17 (best-style engine and
> research UI) and post-slice milestones remain gated behind items 14-18 and
> the Section 7 promotion ladder.
