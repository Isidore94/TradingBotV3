# TradingBotV3 Ultimate Setup Intelligence Database

## Status and authority

This document is a **review draft** for Aaron and Fabel Ultracode. It proposes
the long-term research-data architecture needed to determine which setup and
trade style is working best in the current market context.

It is subordinate to the root [`plan.md`](../plan.md). It does not authorize a
detector, score, ranking, alert, or production-policy change. Implementation
must follow the master roadmap's ordered queue, golden-fixture requirement,
point-in-time repair, champion/challenger ladder, live-validation gates, and
rollback rules.

The proposal incorporates every Markdown file under `docs/` as of 2026-08-03.
Appendix A maps each source document to the constraint carried into this plan.

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

The system should separately return the best-supported:

- swing style;
- intraday quick-payoff style;
- intraday session-hold style;
- current opportunities matching each style;
- evidence against the conclusion;
- an honest `INSUFFICIENT_EVIDENCE` or `NO_QUALIFIED_STYLE` result.

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
- Point-in-time research uses only information whose applicable frozen
  as-observed or simulated-availability timestamp was no later than the decision
  time.
- Missing, stale, partial, conflicting, or late data is uncertainty, never
  confirmation.
- Opportunity, setup, anchor, level, attempt, trigger, alert, impression, trade,
  and outcome identities remain stable and distinct.
- Selected, rejected, quiet, never-triggered, passed, missed, and zero-trade
  cases are retained. The database must contain the denominator, not only wins
  and alerts.
- User-created watchlist names, horizontal levels, trendlines, and notes are
  historical facts. Automation cannot silently erase them.
- One named component owns each ingestion job, catalog, compaction job, mutable
  ledger, and published snapshot.
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
They must live in separate versioned datasets so a later formula change does not
rewrite history.

### 3.2 Store primitives; declare combinations

Do not materialize every possible EMA × SMA × AVWAP × band × level × timeframe
combination as a new column or setup. Store reusable continuous primitives and
atomic interactions, then define setup hypotheses and trade recipes through a
versioned registry. Materialize only frequently queried, registered experiments.

### 3.3 Separate context, trigger, and management

A weekly leader, a 2nd-deviation power hold, or a bullish market regime may be
context. A completed M15 reclaim or M5 retest may be the trigger. The stop,
partials, trail, and expiry are a trade policy. The schema must not collapse
these into one ambiguous family name.

### 3.4 Continuous measurements before bins

Persist distance, slope, penetration, recovery, width, volume, age, streak, and
relative-strength values continuously. Human-friendly buckets are versioned
projections. This permits better thresholds later without refetching history or
pretending an old bin definition was timeless.

### 3.5 One observation can support many correlated diagnostics

One setup occurrence may be evaluated under several time horizons or trade
recipes, but it remains one independent market episode. Alternative outcomes
are correlated diagnostics, not extra trades or extra sample size.

### 3.6 “Best now” must include uncertainty and abstention

Every result reports independent episodes, distinct sessions and symbols,
effective sample size, missingness, confidence interval, evidence freshness,
regime similarity, and out-of-sample status. Thin exact matches shrink toward
broader priors. The engine abstains when support is inadequate.

### 3.7 The research corpus may be large; authority must stay simple

Aaron is willing to retain large files on a DAS. The plan therefore favors
preservation, replayability, and immutable evidence over aggressive deletion.
This does not justify multiple writers, tiny-file sprawl, unchecked corruption,
or a DAS with no independent backup.

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

## 5. Market-data and timeframe coverage

### 5.1 Canonical resolutions

The desired resolution set is:

| Resolution | Primary use | Completion rule |
|---|---|---|
| Quote/last price | User-armed crossing delivery and latency evidence | Operational event only; never a completed-bar setup confirmation |
| M1 | Highest-fidelity intraday archive and deterministic aggregation | Complete exchange minute; store gaps explicitly |
| M5 | Current BounceBot champion, entry timing, and lifecycle transitions | Five complete constituent minutes or a verified provider M5 bar |
| M15 | Intermediate trigger and pullback structure | Session-aligned and complete under the aggregation contract |
| M30 | Opening structure, slower confirmation, and trade management | Session-aligned and complete under the aggregation contract |
| H1 | Existing hourly structure and slower trigger families | Session-aligned; never generic wall-clock resampling |
| H2/H4 | Higher-timeframe trend, pullback, rejection, and compression | Versioned session-aligned aggregation |
| D1 | Swing structure, earnings AVWAP, daily MAs, gaps, and outcomes | Canonical exchange-session close |
| W1 | Leader/laggard structure and slow regime | Only the most recently completed exchange week |

Recommended default: use a resolution-scoped capture policy rather than promise
M1 for the full screen. Capture M1 for Focus/watch/current candidates plus a
fixed and rotating random exploration cohort, M5 for a broader research cohort,
and D1 for the full point-in-time screened universe. Retain honest lower-
resolution coverage when M1 is unavailable. Never invent M1 history from M5
bars. Independently sourced bars remain a separate provider variant rather than
being silently mixed with derived bars.

Create `collection_universe_membership` with instrument, resolution, RTH/ETH
scope, effective interval, inclusion reason, selector version, assignment
probability where known, and cohort type (`FOCUS`, `MODEL`, `FIXED_EXPLORATION`,
`ROTATING_EXPLORATION`, `FULL_SCREEN`). Distinguish
`NOT_COLLECTED_BY_POLICY` from `MISSING`, `NO_RESPONSE`, and `TIMED_OUT`. A
fixed/random exploration cohort is required so intraday discovery is not
conditioned only on champion picks.

Before freezing scope, run a 20-session provider-throughput and data-rights
pilot measuring request budgets, live subscription coverage, backfill limits,
latency, gaps, and bytes per row. Live-observed M1, provider backfill, and
reconstructed history are separate evidence classes. Coverage targets apply
only to the declared collection cohort and acquisition mode.

Every bar records:

- stable instrument ID and displayed symbol;
- exchange, currency, calendar, session ID, and regular/extended-hours state;
- timeframe, interval start, interval end, and timezone;
- market event time, provider availability time, ingestion time, and revision
  time;
- OHLCV and optional trade/count/quote-quality fields when supplied;
- completed/preview state and constituent coverage;
- raw versus adjusted semantics and corporate-action version;
- provider, request ID, fallback/proxy classification, and source contract;
- quality flags, staleness, gap reason, content hash, schema version, and run ID.

Every derived bar carries an `aggregation_contract_id` defining session-open
alignment, timezone/DST rules, RTH/ETH segmentation, half days, constituent
expectations, and final partial-bucket policy. A 6.5-hour RTH session does not
divide evenly into H1/H2/H4; legitimate end-of-session stubs are identified and
never compared with full-duration bars as if equivalent. `NO_TRADE`, `HALTED`,
`NOT_LISTED`, `OUTSIDE_SESSION`, and provider-missing intervals remain distinct.
Provider-native D1/W1 bars are canonical source variants; optional intraday-
derived D1/W1 bars are validation variants, not silent replacements.

At any decision time, every higher timeframe (M15/M30/H1/H2/H4/D1/W1) joins only
to its last completed and available bar. A forming higher-timeframe value may be
stored under a separate preview feature ID and must contribute zero to
confirmation.

### 5.2 Timeframe roles

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

## 6. Technical feature universe

The following is a research universe, not an authorization to score every
combination. Each family requires a registered definition, point-in-time tests,
and correlation-aware analysis.

### 6.1 Moving averages

Preserve the currently meaningful series:

- M5 EMA8/15/21;
- H1 EMA10/15 and SMA20;
- H4 EMA15;
- D1 EMA8/15/21 and SMA50/100/200;
- W1 EMA8/15 and SMA50/100.

Candidate research grid:

| Timeframe | EMA candidates | SMA candidates |
|---|---|---|
| M5 | 8, 15, 21 | 20, 50 |
| M15 | 8, 10, 15, 21, 50 | 20, 50, 100 |
| M30 | 8, 10, 15, 21, 50 | 20, 50, 100 |
| H1 | 8, 10, 15, 21, 50 | 20, 50, 100, 200 |
| H2/H4 | 10, 15, 21, 50 | 20, 50, 100, 200 |
| D1 | 8, 15, 21, 50 | 20, 50, 100, 200 |
| W1 | 8, 15, 21 | 20, 50, 100, 200 |

Before this grid freezes, inventory every actual chart template, timeframe,
period, price input, session setting, and warm-up convention Aaron uses. Keep a
broad inexpensive capture grid separate from the bounded preregistered test
grid. Do not add or omit a 200-period intraday average merely by guesswork.

For each approved period/timeframe pair, persist:

- level value and calculation version;
- price distance in dollars, percent, ATR, and local band-width units;
- slope over multiple registered lookbacks, slope acceleration, and slope
  percentile;
- price-side state and consecutive completed-bar residence;
- ordering/stack state among related averages;
- separation and compression/expansion percentile;
- approach direction and velocity;
- wick tag, close tag, pierce, rejection, reclaim, cross, hold, acceptance,
  failure, first retest, later retest, and role reversal;
- touch count, time since last touch/break, and bars since reclaim;
- volume, relative strength, and candle-quality evidence on the interaction.

Feature identity also includes close/typical-price input, EMA seed and warm-up,
raw/adjusted basis, RTH/ETH scope, minimum history, and exact aggregation
contract. H2 and H4 always have separate IDs.

Near-identical periods are correlated siblings, not independent votes. They
share an evidence-family contribution cap in any future ranking.

### 6.2 Event/manual AVWAP and deviation bands

Champion anchors remain current and previous earnings anchors with the frozen
running-deviation bands at ±1/2/3 sigma.

Create an anchor registry so research can safely study:

- current/prior earnings;
- post-earnings candle;
- week, month, quarter, and year open; session-reset/open variants belong to the
  session VWAP registry below;
- gap or catalyst bar;
- confirmed swing high/low;
- 52-week high/low event;
- breakout/breakdown bar;
- high-volume thrust or displacement bar;
- trader-created manual anchors.

Every anchor records its source bar/event, why it qualified, when it became
knowable, effective range, revision chain, expiry, base-bar resolution,
provider, formula version, and creator. A pivot anchor becomes available when
the pivot is confirmed—not retroactively at the pivot bar.

The anchor registry must publish a semantic dictionary for current earnings,
previous earnings, pre-earnings, post-earnings, earnings-candle, and other
event anchors as implemented in code. These names are not interchangeable.
Feature identity includes anchor-bar inclusion/exclusion, price basis
(OHLC4/HLC3/etc.), raw/adjusted inputs, volume/session scope, base resolution,
and frozen sigma version.

For each AVWAP and ±1/2/3 sigma band, persist:

- value, width, width slope/acceleration, and historical width percentile;
- continuous band coordinate and named zone;
- zone residence/streak and excursions beyond each band;
- distance/age from anchor and number of sessions/bars since anchor;
- first/second/nth approach and touch;
- penetration depth, close recovery, rejection strength, and follow-through;
- cross/chase, break-hold, break-retest, bounce, failed break/reclaim,
  compression-expansion, power hold, and extreme-move retest interactions;
- confluence with MAs, horizontals, trendlines, and other independently sourced
  AVWAP anchors;
- nearest opposing obstacle and available reward/risk.

An AVWAP computed from D1 bars and one computed from M1/M5 bars are distinct
features. Aggregation changes typical-price and running-deviation paths, so the
base resolution must always be part of the feature identity.

### 6.3 Session VWAP families

Model reset-based and live-reanchored intraday VWAP separately from persistent
event/manual AVWAP. A `session_vwap_definition` records reset boundary or
re-anchor event, RTH/ETH scope, price basis, base resolution, volume eligibility,
standard/dynamic/EOD/live-reanchored algorithm, band-sigma method, and formula
version. Seed definitions reproduce:

- standard session VWAP and ±1-sigma bands;
- dynamic VWAP and ±1-sigma bands;
- EOD VWAP and ±1-sigma bands;
- VWAP/EOD confluence and impulse-retest interactions;
- HOD/LOD live-reanchored intraday variants as their exact current code defines
  them, with each newly known extreme creating an immutable revision.

The session-open event-AVWAP label is an alias of standard session VWAP whenever
the frozen definitions are mathematically identical; it never creates a second
feature or confluence vote. HOD/LOD, dynamic, and EOD variants have exactly one
owning definition in this session/intraday namespace and are referenced, not
redefined, elsewhere. Two VWAP labels derived from the same observations are
correlated provenance, not independent confluence. A genuinely persistent
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
| Structure | confirmed swing pivots, consolidation/range boundaries, multi-touch S/R, breakout shelf, failed-break level, cloud flat |
| Statistical | AVWAP/bands, VWAP/bands, moving averages, volatility envelopes |
| Human | trader-entered line, zone, planned entry, invalidation, obstacle, or target |
| Reference | round-number and explicitly configured psychological levels |

Each definition stores:

- `level_id`, family, subtype, symbol, and source timeframe;
- exact price or zone bounds, displayed chart price, raw/split-adjusted/total-
  return adjustment space, and the conversion version used between them;
- original source bar/event/pivots and source snapshot;
- algorithm, parameters, code/config version, or human creator;
- creation chart timeframe, session template, and chart scale where relevant;
- `created_at`, `known_at`, `valid_from`, `valid_to`, expiry, and invalidation;
- strength components, touch count, role history, and current active state;
- revision/supersession chain and reason;
- user notes and watch/arm relationships without conflating them with setup
  confirmation.

Support/resistance or long/short relevance is an as-of role episode/snapshot,
not an immutable definition field: the same price may reverse role. Statistical
levels reference their source MA/VWAP/AVWAP identity rather than duplicating a
copied level with fake independent provenance.

Generated multi-touch levels become historically available only after the
required touches are complete. Manual lines exist for research only from the
time the trader created them. Editing creates a new revision; deletion closes
the validity interval but preserves the history. A split or other corporate
action never silently moves history: the original displayed coordinate remains,
and any transformed coordinate is a new, linked revision under a declared
corporate-action rule.

For every approach/interaction, capture:

- ATR/percent/dollar distance and approach velocity;
- wick penetration and close displacement;
- volume and relative volume at touch;
- first versus repeated test and time since prior test;
- hold, rejection, close break, acceptance, retest, false break, reclaim, and
  role reversal;
- next opposing level and space between level clusters;
- whether a last-price alert fired separately from a completed-bar event.

Every interaction identifies both `level_source_timeframe` and
`observation_timeframe`, plus a distinct trigger timeframe when applicable.
Touch order is scoped by level lifecycle, observation timeframe, and an explicit
reset rule so M5 touches cannot contaminate “first M15 touch” research.
Also type the observation as `QUOTE_CROSS`, `BAR_HIGH_LOW_TOUCH`,
`COMPLETED_CLOSE_BREAK`, `ACCEPTANCE`, or `RETEST_HOLD`; these are not
interchangeable evidence, including for Post-Earnings Candle Break.

### 6.5 Trendlines

Create a versioned trendline registry for both human and algorithmic lines.
Each line stores:

- `trendline_id`, symbol, direction, source timeframe, and source type;
- two or more stable pivot/bar IDs, their displayed anchor prices, and when every
  pivot became knowable;
- chart scale (`LINEAR` or `LOG`), raw/split-adjusted/total-return price space,
  session template, and the source chart timeframe;
- fitting method, equation/coordinates, projection domain (`SEGMENT`, `RAY`, or
  `INFINITE`), ATR-normalized slope, fit residual, touch count, and quality
  components;
- creation time, validity interval, invalidation rule, and revision chain;
- corporate-action transformation/revision rule;
- projected value at every eligible completed-bar timestamp;
- approach, wick tag, close hold/reject, close break, acceptance, break-retest,
  failed break/reclaim, and role-reversal events.

Algorithmic variants may include two-pivot, multi-pivot robust-fit, channel, and
log-price lines, but each is a separately versioned hypothesis. A line fitted
with future pivots cannot be projected backward into the research set.

Trendline interactions carry the same source-timeframe, observation-timeframe,
touch-order, and reset semantics as horizontal levels.

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

EMA15 and EMA21 on the same timeframe, or the same level emitted by two adapters,
are correlated evidence and must not receive two full votes.

### 6.7 Price action, volume, and participation

Store continuous, point-in-time features for:

- candle range/body/wicks, close location, gap, inside/outside bars, and
  displacement in ATR;
- range and volatility compression/expansion over registered lookbacks;
- realized volatility and ATR level/percentile;
- volume, relative volume, dollar volume, volume trend, thrust, dry-up,
  sell/buy participation proxies, and volume on level interaction;
- liquidity, spread/quote quality when available, price, and optionability
  metadata used by the universe;
- opening drive, controlled pullback, impulse/retrace proportions, and time to
  resumption;
- HOD/LOD grind persistence and new-extreme frequency;
- extension from structure and anti-chase state.

### 6.8 Market, sector, industry, catalyst, and clock context

Store both continuous context values and versioned regime labels:

- legacy SPY champion state and shadow market-state output separately;
- SPY pullback/rebound episode and exact aligned interval;
- stock-versus-SPY, sector-versus-SPY, and stock-versus-industry RS/RW;
- sector/industry membership as known on the decision date;
- broad-index context such as SPY plus any deliberately approved QQQ/IWM or
  breadth proxies, each with explicit provenance;
- market/sector trend, volatility, gap, breadth, participation, and integrity;
- earnings timing, earnings gap/candle, other catalogued catalyst, and days
  since event;
- day of week, month/quarter boundary, time since open, time-of-day bucket,
  session phase, and shortened-session status;
- operating mode (DESK/AWAY/EVENING) as presentation context only, never as a
  reason to omit an observed opportunity.

Regime labels must be calculated using only data available at the time. Store
the underlying continuous vector so a later regime definition can be replayed
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
The undefined operational meaning of “banger” and any unconfirmed historical
weight descriptions must be resolved with Aaron rather than guessed during
migration.

## 7. Canonical datasets and identities

Avoid one enormous flat CSV. Use typed Parquet datasets with stable schemas and
purpose-built materialized views.

| Dataset | Independent grain and purpose |
|---|---|
| `instrument_master` | One stable instrument identity/version; symbol aliases, exchange, currency |
| `trading_session` | One exchange session/calendar version with boundaries and expected bars |
| `universe_membership` | Instrument × universe version × effective interval and inclusion reason |
| `collection_universe_membership` | Instrument × resolution/session scope × effective interval, cohort, selector, and assignment probability |
| `provider_observation` | Raw request/response or source observation with availability and hash |
| `raw_bar` | Provider instrument × timeframe × interval × revision |
| `normalized_bar` | Canonical instrument × timeframe × interval × source-selection/normalization generation × revision |
| `corporate_action` | Split/dividend/action as known at a point in time |
| `catalyst_event` | Earnings/other event, source, event time, known-at, and revisions |
| `anchor_definition` | Versioned anchor hypothesis and calculation contract |
| `anchor_instance` | Instrument × anchor definition × source event/bar × revision |
| `level_definition` | Stable generated/manual level identity and lifecycle |
| `level_snapshot` | Level × completed decision timestamp; value, slope, distance, freshness |
| `trendline_definition` | Stable geometry source, pivots, fit method, and revision chain |
| `trendline_snapshot` | Projected line value at one eligible completed timestamp |
| `level_interaction` | One typed approach/touch/break/retest/failure event |
| `feature_definition` | Semantic feature version, inputs, formula, parameters, null policy |
| `instrument_feature_snapshot` | Instrument × as-of × feature-set version, independent of a setup thesis |
| `opportunity_feature_snapshot` | Opportunity/thesis version × as-of × feature-set version |
| `context_snapshot` | Market/sector/industry/symbol context at one decision time |
| `context_episode` | Durable regime/pullback/compression/residence episode |
| `setup_definition` | Versioned thesis, context, trigger, state machine, failure mode |
| `strategy_recipe` | Versioned entry, stop, target, management, cost, and expiry policy |
| `risk_set` / `evaluation_slot` | Scheduled decision cohort and every setup assignment, including not-assigned/timeout/data-failure states |
| `candidate_eligibility` | Evaluation slot × universe membership × setup version × anchor instance × run/as-of, including rejection reason |
| `market_episode` / `dependency_cluster` | Outcome-blind underlying move/reset cluster linking correlated setup/timeframe/anchor variants |
| `setup_occurrence` | One thesis occurrence linked to its dependency cluster; repeated scans update it |
| `opportunity_lifecycle` | Discovery through ready/failed/rearm/expired, linked revisions |
| `attempt` | One try within a lifecycle; re-arm creates another attempt |
| `trigger_event` | One completed-bar detector/state transition evaluation |
| `ranking_snapshot` | Exact eligible cohort, component scores, order, exclusions, and policy versions |
| `alert_event` / `delivery_event` | Material state transition versus each presentation/push/relay delivery |
| `impression` / `review_action` | What was shown and how the trader resolved it |
| `trade` / `fill` / `management_event` | Actual imported execution and process record |
| `outcome_definition` | Frozen standardized or actual outcome contract |
| `outcome_path` | Attempt × recipe × outcome definition path, excursion points, first-hit ordering, coverage |
| `outcome_result` | One declared analysis unit × outcome definition, linked to its dependency cluster |
| `experiment_definition` | Registered question, data freeze, split, metrics, controls, and multiplicity family |
| `experiment_run` | Code/config/data hashes, outputs, failures, and reproducibility manifest |
| `hypothesis_registry` | Research generation, hypothesis/specification, multiplicity family, role, and holdout exposure history |
| `evidence_snapshot` | Immutable reviewed metrics and cited source partitions |
| `promotion_decision` | Status, approval, effective time, prior champion, rollback target |
| `data_manifest` | Partition schema/hash/coverage/owner/health and lineage |

### 7.1 Identity graph

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
- A failed attempt followed by re-arm creates a new `attempt_id` under the same
  lifecycle.
- A repeated delivery creates another `delivery_id`, not another alert or sample.
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
  `ATTEMPT`, or `MARKET_EPISODE`); it never says “occurrence/attempt.”

Workstream 1 must publish a small ERD with primary/foreign keys, cardinalities,
deterministic ID algorithms, occurrence start/end/dedup rules, corrections, and
supersession behavior. Every evidence report includes `n_rows`, `n_attempts`,
`n_market_episodes`, `n_sessions`, `n_symbols`, and method-derived `n_effective`.

## 8. Storage architecture on the DAS

### 8.1 Recommended target

Use an immutable Parquet/Arrow lake on the DAS, Zstandard-compressed, with:

- append-only JSONL/Parquet event ledgers;
- content-addressed manifests and checksums;
- a rebuildable DuckDB catalog/query index;
- a machine-local hot cache and write spool;
- compact atomic snapshots for the GUI and Desk Link;
- a separate verified backup target.

DuckDB is a query engine and rebuildable catalog, not the sole authoritative
copy. This target requires a new architecture decision that explicitly limits
or supersedes decision 0005 for the research warehouse. Until that decision is
approved, do not introduce a mutable database file into the shared home folder.
The single-writer rule also follows DuckDB's documented in-process concurrency
model and its warning to use extra caution with database files on shared/network
storage: [DuckDB concurrency](https://duckdb.org/docs/stable/connect/concurrency).

### 8.2 Proposed directory contract

```text
<research_store>/
  manifests/
    ingestion/
    partitions/
    evidence_freezes/
    snapshots/
      snapshot.<generation>.json
    CURRENT
  bronze/
    provider_observations/
    bars/
    corporate_actions/
    catalysts/
  silver/
    normalized_bars/
    sessions/
    universes/
    anchors/
    levels/
    trendlines/
    interactions/
    contexts/
  gold/
    candidates/
    lifecycles/
    rankings/
    outcomes/
    research_marts/
    current_edge_snapshots/
  control/
    feature_definitions/
    setup_definitions/
    strategy_recipes/
    experiment_registry/
    promotion_records/
  catalog/
    catalog.<generation>.duckdb
  quarantine/
  rollback_generations/

<machine_local_state>/
  research_active_spool/
```

Generated data stays outside Git. Git holds schemas, definitions, migrations,
golden fixtures, and small sanitized examples. Reviewed definitions in Git are
authoritative; the warehouse retains immutable released copies and hashes so an
old evidence freeze remains reproducible.

### 8.3 Partitioning and compaction

- Partition large bar datasets by schema version, timeframe, provider,
  year/month, and a stable symbol bucket; avoid one file per tiny event.
- Use a measured pilot to tune file and row-group sizes. DuckDB's current
  performance guidance suggests starting around 100,000–1,000,000 rows per row
  group and moderate files of roughly 100 MB–10 GB, then benchmarking the real
  filter/join workload: [DuckDB file-format performance](https://duckdb.org/docs/current/guides/performance/file_formats).
- Write active-session data to one owned spool, then seal immutable partitions
  at rollover with row counts, min/max timestamps, coverage, and SHA-256.
- Compact only sealed, replaceable derived generations. Write a replacement
  beside the source, validate exact row/key/hash reconciliation, atomically
  promote the manifest, and retain the old generation through a rollback
  window. Bronze raw and evidence-frozen files are never compaction inputs;
  they remain immutable and indefinitely retained under the approved policy.
- Quarantine malformed/conflicting records; never silently discard them.
- Derived indexes are disposable and rebuildable. Raw and evidence-frozen
  partitions are immutable.
- Preserve raw evidence indefinitely by default, subject to capacity health.
  Retention changes require an explicit policy and verified backup.

Use a generation-based commit protocol:

1. Write immutable data files under a new generation and hash every file.
2. Build a root snapshot manifest containing schema versions, logical row/key
   hashes, file hashes, coverage, and `parent_generation`.
3. Build `catalog.<generation>.duckdb` beside—not over—the active catalog.
4. Validate referential integrity and deterministic logical-row reconciliation.
5. Atomically replace one tiny `CURRENT` pointer with the new generation ID.
6. Make every query pin a generation for its entire lifetime.
7. Retain prior generations through the rollback window.

Never replace or mutate a DuckDB file that Windows readers may still have open.
Compaction compares canonical logical rows/keys, not only Parquet byte hashes,
because a valid re-encoding changes file bytes.

Capacity planning should use measured bytes per row after a 20-session pilot.
The row-count formula is explicit: regular-hours M1 is approximately 98,280
rows per symbol-year before extended hours; multiply by actual point-in-time
universe membership, fields, revisions, and replication. Health should forecast
30/90/365-day capacity rather than relying on a guessed compression ratio.

### 8.4 Ownership and multi-machine behavior

One physical warehouse service on the main-hosted DAS owns normalization,
manifests, catalogs, and the generation pointer. Separately, a session-scoped
`scan_owner` is explicitly and mutually exclusively assigned to the main or
mini-PC. The mini may run its existing scheduled scan when assigned, but it
submits immutable bundles through the warehouse import boundary and never writes
the DAS/catalog. Satellite control changes decision rights, not scan or warehouse
ownership.

- Satellites query a main/warehouse-owned service or consume immutable snapshots;
  they do not open the catalog for writes.
- The mini-PC never opens the DAS/catalog for writes. When assigned scan
  ownership, it creates immutable installation-scoped acquisition bundles
  locally and submits them through an acknowledged, idempotent warehouse import
  boundary.
- If the DAS is unavailable, the main uses a bounded machine-local pending spool
  with an explicit degraded state. It must not silently create a second
  authoritative warehouse. Spool-full behavior is predeclared: reduce optional
  cohorts, keep operational champions alive, and record gaps rather than delete
  unimported evidence.
- Writer leases and fencing protect publications but are not distributed database
  locking. Ambiguity fails closed.

| Work | Sole owner | Other machines |
|---|---|---|
| Live TWS collection/scans | Session-scoped main or mini `scan_owner` | All non-owners remain scan/TWS inactive |
| Machine-local acquisition spool | Producing installation | Immutable until acknowledged import |
| Warehouse import/normalization | Warehouse service | Submit immutable bundles only |
| Feature/outcome build | Warehouse service/job owner | Read pinned generation |
| Compaction/catalog generation | Warehouse service/job owner | Read prior/current pinned generation |
| Backup/restore | Named backup job/operator | No in-place restore by clients |
| GUI/Desk Link/phone projection | Existing named publisher | Consume verified snapshot |

The Away runbook and First Session checklist currently describe different
standing publisher roles on Away days. Resolve that conflict through one
session-scoped role configuration and update both documents before the warehouse
depends on cross-machine ownership.

### 8.5 Backup and recovery

A DAS/RAID is capacity and availability, not backup. Require:

- immutable filesystem snapshots where supported;
- a second physical backup target with a verified volume identity, outside the
  DAS research root;
- an off-site copy for manifests, definitions, irreplaceable raw evidence, and
  promotion records;
- checksummed incremental backups;
- a catalog-rebuild command;
- monthly sampled restore verification and quarterly full restore drills;
- recorded recovery-point and recovery-time results;
- disk-health, free-space, growth, backup-age, and restore-age status in System
  Health.

Define RPO/RTO per data class: live spool, irreplaceable raw observations,
rebuildable derived partitions, reviewed definitions, and promotion records.
Restores go to a new root, verify manifests and logical rows, rebuild the catalog,
then switch the generation pointer; never restore destructively in place. The
20-session pilot must also verify market-data retention/licensing terms for the
proposed archive and backups.

### 8.6 Security and privacy

- Keep API keys, broker credentials, Desk Link secrets, and encryption keys out
  of the warehouse, manifests, diagnostics, and Git.
- Restrict filesystem and query-service access to the trader's machines and
  named service accounts; satellites receive only the data their UI requires.
- Prefer DAS/filesystem encryption where it does not compromise tested recovery,
  and include keys in the secure recovery plan rather than ordinary backups.
- Keep fills, account identifiers, free-text journal notes, and screenshots in
  access-controlled local partitions; redact compact Drive and AI exports.
- AI evidence packages remain explicit opt-in, show the exact selected sources,
  and cite immutable IDs/hashes. A research-store path never implies permission
  to upload its contents.

## 9. Ingestion and data-quality contract

### 9.1 Provider boundary

Use app-owned repository interfaces for daily bars, intraday bars, quotes,
corporate events, and read-only execution imports. Persist normalized app-owned
records, never IBKR/Yahoo SDK objects.

IBKR remains primary and Yahoo fallback. Every observation and downstream
feature records its actual source. Never blend sources without a provider-
transition record. Retain enough source evidence to test whether an apparent
edge depends on IBKR, Yahoo, or mixed fallback data.

### 9.2 Collection cadence

| Evidence | Minimum target cadence |
|---|---|
| M1/M5 bars | Every completed eligible interval with explicit missing-bar rows |
| M15/M30/H1/H2/H4 | On full deterministic completion from constituent bars |
| D1/W1 | Canonical session/week close |
| Setup eligibility | Every scheduled scan and every material input revision |
| Lifecycle/level interaction | Every eligible completed trigger bar |
| User last-price crossing | Current service cadence, stored as operational crossing evidence |
| Early persistence | Open+30, +45, and +60 snapshots where relevant |
| Frozen regime snapshots | Every predeclared scheduled target; absence becomes `MISSED_SNAPSHOT` |
| Intraday outcomes | +15/+30/+60/+120 minutes and EOD, with near-close truncation |
| Swing outcomes | +1/+2/+3/+5/+10/+18 sessions and final policy resolution |
| Health/coverage | Open, midmorning, midday, late day, close, and teardown |

Report cadence never defines collection cadence. DESK, AWAY, and EVENING retain
the same observed-candidate evidence even when presentation behavior differs.

### 9.3 Universal point-in-time availability

Every source and derived record carries, where applicable:

```text
event_at
source_published_at
provider_received_at
observed_at
ingested_at
first_seen_at
computed_at
actual_decision_available_at
simulated_information_available_at
availability_model_id
valid_from / valid_to
system_from / system_to
revision_id / supersedes_revision_id
capture_mode = LIVE | DELAYED | BACKFILL | RECONSTRUCTED
max_input_actual_available_at
max_input_simulated_available_at
pit_eligible
pit_fidelity_grade
pit_exclusion_reason
```

Use two explicit availability clocks:

- `simulated_information_available_at`: when a market fact would ordinarily have
  become knowable under the frozen provider/calendar/publication contract; used
  only by permitted `PIT_RECONSTRUCTED` market backtests and tied to
  `availability_model_id`.
- `actual_decision_available_at`: when this installation actually observed,
  ingested, computed, and could publish the fact; required for as-observed replay,
  exposure, latency, queue, and live-shadow evidence. It is immutable and never
  earlier than `first_seen_at`.

Each experiment freezes `availability_view` plus allowed PIT fidelity grades.
The experiment compiler binds `evidence_available_at` to exactly one of the two
clocks and derives it from the maximum dependency time plus the applicable
measured or modeled computation/publication latency; callers cannot supply it.
Every replay joins on `evidence_available_at <= simulated_as_of`. A bar fetched
today for 2024 may participate in a declared reconstructed market backtest, but
cannot prove 2024 installation coverage, denominator, queue exposure, latency,
or live-shadow eligibility. Provider revisions create new vintages, and every
experiment also freezes as-recorded versus corrected view.

### 9.4 Quality states

Use typed states, not blank values:

- `COMPLETE`
- `PREVIEW`
- `PARTIAL`
- `STALE`
- `LATE_ARRIVAL`
- `MISSING`
- `MISSED_SNAPSHOT`
- `TRUNCATED`
- `PROVIDER_FALLBACK`
- `PROVIDER_CONFLICT`
- `NOT_COLLECTED_BY_POLICY`
- `NO_RESPONSE`
- `TIMED_OUT`
- `NO_TRADE`
- `HALTED`
- `NOT_LISTED`
- `OUTSIDE_SESSION`
- `INVALID_DATA`
- `QUARANTINED`

Research queries default to completed, eligible observations whose selected
`evidence_available_at <= simulated_decision_time`. A late-arriving bar cannot be made
retroactively available simply because its market timestamp is earlier.

### 9.5 Corporate actions and revisions

Retain raw and adjusted prices with an explicit adjustment version and the time
the corporate action became known. Never overwrite old partitions after a split,
symbol change, earnings correction, or provider revision. Append a revision and
choose the appropriate view for each experiment.

Use bitemporal semantics wherever facts can change: `valid_from/valid_to`
describe when the fact applied in the market or model, while
`recorded_at/superseded_at` describe when the system learned and revised it.
Historical simulation filters on both market validity and knowledge time.

## 10. Feature registry and calculation graph

Every feature definition includes:

- stable feature ID, semantic version, family, units, and description;
- input datasets/fields and exact timeframe roles;
- parameters, minimum history, session rules, and adjustment semantics;
- completed/preview eligibility;
- null/missing/stale behavior;
- formula/code/config hash and dependency versions;
- `known_at`/availability-lag rule;
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
feature version and source snapshot must produce identical keys and values.
The dependency graph is statically/auditably separated so a feature, context,
setup, eligibility, or prediction node cannot read outcome/result tables.
Both maximum-input availability clocks and lineage are recomputed from dependencies rather
than trusted from the feature implementation.

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
Parameter searches declare their grid and multiplicity family before outcomes
are inspected.

## 12. Strategy-recipe library

The same setup must be evaluated under explicit trade-style recipes rather than
baking one management policy into the setup name.

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

Every entry simulation stores `signal_known_at` and `entry_eligible_at`. A signal
computed from a completed close is executable no earlier than the next eligible
quote/bar/open unless the recipe explicitly models a precommitted market-on-close
order. A same-close fill can never be assumed merely because the close created
the signal. Quote-observed entries retain the quote and timestamp provenance.
When only OHLC bars exist, the first-cross time is interval-censored within the
bar and receives the declared conservative fill; it is never assigned a precise
intrabar order. A later completed-close-confirmed break is a separate recipe,
not another fill interpretation for the same recipe. `QUOTE_CROSS` may alert or
fill only a pre-armed recipe whose eligibility came from completed-bar context;
it never creates a setup `trigger_event` or advances lifecycle confirmation.
`BAR_HIGH_LOW_TOUCH` becomes confirmation only when that bar is completed and
available. Any anticipatory quote-first style is separately labeled research
evidence and cannot replace a completed-bar champion. The Post-Earnings Candle
Break seed family must include distinct first-cross/stop-entry and completed-
close variants under these boundaries.

### 12.2 Invalidation and stop methods

- house structural level with one- or two-close failure;
- one band beyond the bounced level;
- opposing earnings-candle extreme;
- signal/retest-bar extreme plus registered ATR buffer;
- nearest valid structural/horizontal/trendline level;
- fixed price, percent, ATR, or volatility-distance control policy;
- no intrabar stop for a level-close thesis, if that is the declared policy.

Under-bar tick stops and close-based structural invalidation are different trade
styles. One poor stop policy must not make a valid pattern appear invalid.
Store `invalidation_observed_at` and `exit_eligible_at`; one/two-close failure is
executable only after the confirming close is known. Preserve full intrabar MAE
and gap tails because a close-based thesis stop is not bounded intraday risk.
An optional catastrophe stop is a separately named recipe component.

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
mirrored prices survive only as a labeled legacy-parity diagnostic. Actionability
and outcome confidence include point-in-time shortability/borrow availability,
HTB/locate and borrow-cost evidence when available, SSR, halts/LULD, liquidity,
and gap risk. Missing historical borrow data is uncertainty, never an assumption
that the short was freely tradable.

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
policy version. This permits measurement of:

- setup precision and missed winners;
- level-approach-to-trigger conversion;
- trigger-to-Ready conversion;
- bot selection versus full eligible universe;
- human selection lift after controlling for exposure and context;
- queue-position and review-latency bias;
- execution/adherence separately from opportunity quality.

Each scheduled research decision creates a `risk_set_id` and one
`evaluation_slot_id` for every instrument × currently registered in-scope setup
before scan ordering or data acquisition. It stores scheduled time, universe and
collection memberships, setup version, frozen assignment policy/probability,
assignment time/order, and:

```text
NOT_ASSIGNED | REQUESTED | NO_RESPONSE | PARTIAL_DATA | TIMED_OUT |
EVALUATED_INELIGIBLE | EVALUATED_ELIGIBLE
```

Metrics reconcile risk set → assigned → returned → data complete → eligible →
triggered → surfaced → matured. Unevaluated is never labeled rejected or “no
setup.”

Every experiment declares its target population. Deterministic Focus/model
cohorts support cohort-conditional claims only. Broader inference from randomized
exploration uses frozen design/stratum weights, positivity/common-support and
balance diagnostics; any learned coverage/capture weighting is fit inside the
training fold. Adaptive acquisition with unknown assignment probability cannot
claim full-universe expectancy. Current Edge displays the population to which
each estimate applies.

Current review data that folds by `(trade_date, symbol)` or treats arming a watch
as a take remains exploratory. Preference cannot influence production ordering
until Swing/M5, side, thesis, impression, and action identities are repaired.

Manual levels/trendlines are observed only on charts Aaron selected. Their
research estimand is therefore conditional on exposure to the chart and geometry.
Store the exposed cohort and a point-in-time matched-control set. Claim universal
geometry edge only when a reproducible algorithmic definition provides the
non-drawn denominator; human-selection lift remains association without a
predeclared causal/randomized design.

## 14. Outcome engine

### 14.1 Two primary outcome classes

1. **Standardized opportunity outcome:** what the setup did under a frozen
   hypothetical recipe.
2. **Actual execution outcome:** what Aaron entered, sized, managed, and exited.

Never substitute actual P&L for setup quality or standardized R for execution
quality.

Keep four terms mechanically distinct:

- `planned_reward_risk_r`: geometric target distance divided by planned risk;
- `model_expected_net_r`: frozen out-of-sample predicted expectancy net of the
  declared cost model;
- `standardized_realized_r`: realized result under the standardized recipe;
- `actual_execution_r`: realized result from imported fills and management.

Never label planned reward/risk as Expected R or train/calibrate it as expectancy.

### 14.2 Required outcome contract

Every `outcome_definition_id` freezes:

- declared analysis unit, linked opportunity/attempt IDs, and decision timestamp;
- trigger and completed-bar identity;
- entry, fill, gap, no-fill, slippage, commission, and liquidity assumptions;
- stop/invalidation, close-failure, target, scale, trail, and expiry rules;
- same-bar stop/target ordering;
- regular/extended-hours eligibility;
- MFE/MAE and first-hit calculation;
- censoring, missingness, and independent sampling rules.

OHLC bars cannot reveal path order when stop and target both occur. Store
`path_resolution = EXACT | LOWER_TIMEFRAME | AMBIGUOUS`, `r_lower_bound`,
`r_upper_bound`, `primary_ambiguity_policy`, `fill_quality`, `cost_model_id`,
`maturity_at`, and `censor_reason`. Use finer data only when provenance-compatible;
otherwise retain bounds, use a preregistered conservative primary estimate, and
report sensitivity. Never silently drop ambiguous/no-fill/missing cases.

Required result states include:

- `NO_TRIGGER`
- `NO_FILL`
- `OPEN`
- `MATURED`
- `STOPPED`
- `TARGETED`
- `EXPIRED`
- `CENSORED`
- `TRUNCATED`
- `MISSING`
- `AMBIGUOUS_BAR`
- `INVALID_DATA`

Report full-risk-set, trigger-conditional, and fill-conditional estimands
separately. An unresolved label cannot enter training or a current posterior
until its `maturity_at` is no later than that prediction's evidence cutoff.

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
stops. They remain incomparable until each carries an explicit recipe/outcome ID.

## 15. Research and statistical framework

### 15.1 Registered research question

Every experiment predeclares:

- thesis and failure mode;
- primary metric and tolerated degradation metrics;
- independent observation unit;
- universe, side, timeframes, feature/recipe versions, and exclusions;
- training, validation, final test, and live-shadow windows;
- matched control/baseline;
- parameter grid and multiple-testing family;
- minimum distinct episodes, symbols, sessions, and regime coverage;
- costs, missing-data rule, and promotion/abandonment criteria.

The global hypothesis registry also records:

```text
research_generation_id
hypothesis_id
specification_id
multiplicity_family_id
role = DISCOVERY | VALIDATION | CONFIRMATORY
holdout_exposure_count
outcome_first_opened_at
```

One multiplicity family spans all neighboring indicator periods, timeframes,
thresholds, sides, context slices, recipes, horizons, and reruns considered in
that research generation. Splitting the same search into several experiment
files cannot reset multiplicity. Once a final holdout is inspected, it is spent
and becomes validation evidence; a new confirmatory claim needs untouched
forward evidence. Scheduled repeated reviews use a fixed cadence or a declared
sequential-error method.

The registry retains losing and inconclusive studies so the same hypothesis is
not repeatedly rediscovered and selectively reported.

### 15.2 Baselines and controls

Retain `baseline_every5` and add matched controls by:

- date and time of day;
- symbol liquidity, price, and volatility;
- side and broad market direction;
- regime and catalyst proximity;
- distance to comparable levels;
- opportunity availability and provider coverage.

Compare a new family against both its matched control and the current setup
portfolio. Lift from Move Forensics is association only and never becomes a
score until a tradeable, forward-tested setup clears the full ladder.

Every matched set freezes `matched_set_id`, point-in-time covariates, caliper,
weights, fixed seed, control-reuse count, and balance diagnostics. Human-selection
comparisons remain associational unless exposure/assignment was randomized or a
defensible causal design was registered in advance.

### 15.3 Validation design

- Use chronological walk-forward splits; never random row splits.
- Purge/embargo overlapping label windows around split boundaries.
- Freeze training, validation, and final-test partitions before inspection.
- Keep every `dependency_cluster_id`, market/catalyst episode, correlated re-arm
  chain, and overlapping outcome interval wholly inside one training,
  calibration, validation, or final-test partition.
- Fit feature selection, preprocessing, imputation, matching, shrinkage,
  hyperparameters, and calibration only inside the applicable training/inner-
  validation window.
- Freeze a primary dependency cluster, secondary cluster dimensions, block/
  bootstrap method and length, purge/embargo span, control-reuse rule, and
  effective-n estimator/version.
- Default to chronological outer splits purged by the maximum label window and
  session/catalyst/market-episode block inference that preserves cross-sectional
  dependence; handle symbol dependence by registered multiway clustering or
  grouped resampling.
- Report bootstrap confidence intervals and downside/tail distributions.
- Correct large grids for multiple testing or false-discovery rate.
- Measure provider, universe, earnings, and outcome-join coverage.
- Test both sides explicitly; a mirrored detector is not evidence of mirrored
  expectancy.
- Test multiple regimes, open/midday/late periods, and material volatility
  states.
- Require portfolio-level incremental value after overlap/correlation with
  champion setups.

Every acceptance manifest uses the maximum of all applicable root-plan floors,
setup-specific floors, and a power/design-effect calculation for the predeclared
minimum useful effect. Floors apply to untouched out-of-sample market episodes,
not raw rows or all-history totals, and cannot be pooled across unsupported side,
style, recipe, outcome definition, or regime. Thin cells remain exploratory.
Rare families remain Research/Advisory or narrow their declared scope; manual
review does not substitute for statistical efficacy evidence.

### 15.4 Estimation ladder

Prefer the simplest method that earns out-of-sample value:

1. Deterministic grouped tables with honest intervals.
2. Hierarchical/empirical-Bayes shrinkage across related periods, timeframes,
   sides, and contexts.
3. Regularized regression or additive models for continuous effects.
4. Tree-based challenger for nonlinear interaction discovery.
5. Survival/competing-risk models for time-to-target versus time-to-stop.

Every model is versioned, replayable, calibrated, explainable at the opportunity
level, and compared with a trivial baseline. Complexity is not itself progress.

### 15.5 Recent versus durable edge

Store multiple estimates rather than one reactive score:

- all-history or durable-regime prior;
- rolling recent window;
- exponentially decayed estimate with declared half-life;
- current-regime posterior;
- change-point/deterioration flag;
- evidence freshness and stability across walk-forward folds.

The exact context estimate shrinks toward setup-family and global priors when
thin. Three recent outcomes must not overpower a durable sample merely because
the feature grid is highly specific.

Recent adaptation uses only outcomes matured before the prediction cutoff,
updates at fixed review times, and applies hysteresis/minimum dwell. It may cause
abstention or create a new challenger proposal; it never silently retunes a live
champion.

### 15.6 Calibration and regime-model contract

Persist every out-of-sample prediction before its outcome matures. Acceptance
manifests predeclare:

- binary calibration: Brier score/log loss, intercept, slope, and reliability
  error;
- R-distribution calibration: bias, a declared proper distribution score, and
  empirical interval coverage;
- calibration-fit window separate from evaluation/test;
- maximum calibration age and minimum calibrated effective n;
- abstention limits for OOD distance, interval width, data coverage, staleness,
  and walk-forward instability.

Each regime snapshot records model/version, fit cutoff, training freeze,
preprocessing versions, posterior probabilities, entropy, and OOD distance.
Regime boundaries, scaling, clustering, decay, and change-point parameters are
trained inside the applicable fold and frozen before shadow use.

Each acceptance manifest freezes numerical or champion-relative calibration
limits, their uncertainty intervals, minimum effective calibration sample,
applicable subgroups, and the failure action. Failed or underpowered calibration
forces abstention or continued Advisory status even when rank metrics look good.

### 15.7 Promotion estimator

Promotion freezes one primary estimand and one primary benefit, then compares
challenger with champion on identical ranking snapshots and risk sets. Require:

- a confirmatory, cluster-aware confidence interval adjusted for the frozen
  multiplicity family—or produced by a preregistered hierarchical gatekeeping
  procedure—whose lower bound demonstrates the predeclared benefit; discovery-
  stage FDR alone never qualifies;
- simultaneous non-inferiority bounds for capture, downside, remaining
  opportunity, and latency, with frozen confidence level, familywise-error
  allocation, and metric direction;
- sufficient untouched out-of-sample episode/session/symbol/regime and live-
  shadow coverage after design-effect adjustment;
- no dominant symbol/session/catalyst concentration;
- frozen model, features, splits, costs, outcomes, multiplicity family, and data
  hashes;
- no tuning during shadow/canary.

A short canary may validate safety, mechanics, parity, and rollback. It cannot
establish efficacy until its powered matured-outcome floor also passes.

## 16. “Best trade style now” engine

This engine begins as a research/shadow consumer of the current point-in-time
feature snapshot.

### 16.1 Pipeline

1. Verify current data health, completed-through times, and source coverage.
2. Build the current market/sector/industry/symbol context vector.
3. Enumerate locked, registered champion setup/style models plus frozen,
   preregistered challengers eligible for that context, carrying explicit
   authority/status; never select the luckiest visible Setup Matrix cell on
   demand.
4. Apply hard structural, liquidity, freshness, risk, and actionability gates.
5. Retrieve global, family, side, timeframe, and regime evidence priors.
6. Estimate context-conditioned outcome distributions with hierarchical
   shrinkage and uncertainty.
7. Penalize staleness, provider dependence, tail risk, excessive uncertainty,
   extension, portfolio correlation, finite opportunity slots/risk budget,
   simultaneous alerts, and existing read-only position exposure.
8. Rank separately by swing, quick intraday, and session-hold objectives.
9. Match current opportunities to supported styles.
10. Publish an immutable advisory snapshot or abstain.

### 16.2 Output contract

For each style and matching opportunity show:

- setup/style/version and all timeframe roles;
- `predictive_standardized_r_distribution`, including predictive median,
  quantiles, downside, and tail metrics;
- `success_event_definition_id` and its `success_probability`;
- `model_expected_net_r` as the predictive mean plus its distinct
  `model_expected_net_r_interval` for epistemic estimation uncertainty;
- MFE/MAE and expected time-to-payoff profile;
- independent episodes, distinct symbols, sessions, effective n, and coverage;
- calibration, stability, and last evidence date;
- current-regime estimate beside the all-regime prior;
- current-context similarity and out-of-distribution warning;
- entry condition, invalidation, nearest obstacle, target, and remaining R;
- atomic interactions and distinct provenance families that contributed;
- counter-evidence, blockers, missing data, and why the result might fail;
- overlap/concentration with other current ideas;
- Personal Fit as a visibly separate annotation.

Predictive outcome spread and epistemic uncertainty are different objects.
Risk penalties use predictive downside/tail measures; evidence-confidence and
abstention use the uncertainty around the estimated mean and calibration. The UI
must not relabel a mean interval as an outcome range or a planned target/risk
ratio as either quantity.

Every snapshot also records `prediction_as_of`, per-timeframe
`completed_through`, `valid_until`, capture-universe coverage,
`evidence_trained_through`, `labels_matured_through`, model/calibration versions,
and the pinned warehouse generation. Use explicit abstention/result codes:

- `SUPPORTED_STYLE_WITH_CANDIDATE`
- `SUPPORTED_STYLE_NO_CURRENT_CANDIDATE`
- `NO_ACTIONABLE_CANDIDATE`
- `INSUFFICIENT_CONTEXT_DATA`
- `OUT_OF_DISTRIBUTION`
- `STALE_EVIDENCE`
- `NO_SUPPORTED_STYLE`

Do not force quick and EOD evidence into one scalar: existing BounceBot evidence
shows those rankings can diverge materially. Present a small Pareto frontier when
styles trade model expected net R, success probability, downside, and time-to-payoff differently.
Evaluate both single-opportunity quality and top-K/current-portfolio utility under
the declared slot, risk, side, sector, and correlation constraints.

### 16.3 Objective and personal fit

Canonical ranking order remains:

1. data health and hard risk gates;
2. lifecycle/actionability stage;
3. objective setup and trade-style quality;
4. remaining reward/risk and no-chase;
5. evidence confidence;
6. concentration/correlation;
7. Personal Fit tie-break inside comparable quality/stage bands;
8. separate delivery policy.

Personal Fit may annotate or reorder comparably qualified items. It cannot change
eligibility, hard gates, lifecycle stage, objective expectancy, monitoring cadence,
sound, suppression, or execution.

## 17. Research and trader-facing tools

### 17.1 Current Edge dashboard

Show:

- best-supported Swing, Intraday Quick, and Intraday Session styles now;
- matching current opportunities;
- deteriorating styles and change-point warnings;
- evidence maturity, coverage, freshness, and current-regime similarity;
- an honest no-edge state.

### 17.2 Setup Matrix

Pivot and drill down across:

- setup family and side;
- structural/context/trigger timeframe;
- level family and interaction type;
- entry/stop/management recipe;
- market/sector/industry regime;
- earnings/catalyst state;
- time of day and holding horizon.

Every cell shows n, distinct sessions/symbols/market episodes, model expected
net R, standardized realized R, median R, win rate, MFE/MAE, interval, baseline
edge, stability, missingness, and status. Small cells remain visible but clearly
exploratory.

### 17.3 Level Edge Lab

Answer questions such as:

- Does the first M15 tag of a rising H1 EMA21 outperform later tags?
- Does an M5 reclaim at D1 earnings AVWAP work better when M30 EMA15 is rising?
- Are D1 SMA50 retests better when also near a prior-week high or trendline?
- How much does edge decay after the second or third test?
- Does a close break, acceptance, or retest retain enough remaining R to justify
  the additional confirmation delay?

### 17.4 Multi-Timeframe Map

For one symbol/opportunity, display completed-bar state on M5/M15/M30/H1/H2/H4/D1/W1,
active levels and clusters, current/next trigger, conflicts, freshness, and exact
source snapshots. Forming frames are visibly Preview.

### 17.5 Strategy Recipe Comparator

Compare the same independent occurrences under declared entry, stop, target,
trail, and horizon policies. Make correlation explicit and prevent users from
mistaking 20 recipes on 20 occurrences for 400 trades.

### 17.6 Replay and audit

Reconstruct the chart and decision state at any historical `as_of`, including:

- bars then available;
- anchors, horizontal levels, and trendlines then known;
- feature/context versions;
- setup eligibility and blockers;
- score/rank and competing cohort;
- alert/impression/action;
- subsequent path and outcomes hidden until replay is released.

### 17.7 Research queue and promotion workspace

Track idea, owner, hypothesis, registered grid, data readiness, latest run,
evidence freeze, review notes, status, next gate, prior champion, and rollback.
Fabel/AI suggestions enter here; they do not edit production configuration.

## 18. Health, observability, and operations

Extend System Health with:

- authoritative writer, installation, machine, process, and lease state;
- DAS mount and latency status;
- expected versus observed bars by resolution/session;
- late, missing, duplicate, conflict, revision, and fallback rates;
- ingestion, normalization, feature, outcome, and snapshot latency;
- active spool/quarantine backlog and oldest pending item;
- partition seal, manifest, checksum, and catalog-rebuild status;
- experiment reproducibility failures and outcome-join coverage;
- disk use and 30/90/365-day growth forecast;
- last backup, last checksum verification, and last successful restore drill;
- current/next owned jobs and clean shutdown state.

A green storage/collection audit means the mechanics worked. It does not mean a
setup is predictive or promoted.

## 19. Proposed Milestone-10 workstreams

These are dependency proposals for root `plan.md` Milestone 10, not a second
authoritative execution queue. Only root `plan.md` may assign implementation
order or status. Its Section 12 and earlier milestones always win.

Crosswalk:

| Root dependency | Work enabled here |
|---|---|
| Milestone 2 storage migration/authority | DAS ownership and local-vs-shared classification |
| Milestone 3 golden/replay harness | Characterization, corrected expectations, and migration parity |
| Milestone 4 provider repository | Normalized acquisition and deterministic aggregation |
| Milestone 5 point-in-time repair | Bitemporal features, anchors, identities, labels, and research validity |
| Milestone 6 canonical authority | Complete risk sets/candidates and one lifecycle |
| Milestone 7 canonical opportunity/ranking | Current-style challenger and portfolio ranking |
| Milestone 9 journal/learning | Impressions, actual execution, Personal Fit, and review linkage |

No workstream changes a live champion merely because its code is complete.

### First vertical slice

Before attempting the full program, prove one end-to-end slice over 20 forward-
observed sessions:

- RTH only;
- one declared M1/M5 capture cohort plus canonical D1;
- derived M15/M30/H1 with exact session contracts;
- current earnings AVWAP and frozen bands, EMA15/21, and PDH/PDL;
- one existing long and one existing short setup;
- one primary swing and one primary intraday recipe;
- risk-set assignment through PIT features, eligibility, outcome path, evidence
  freeze, query, backup, and verified restore.

Defer H2/H4 expansion, broad anchor grids, algorithmic trendlines, tree/survival
models, and the full UI until this slice reconciles and restores cleanly. The
slice must preserve the final IDs and manifests so it is not throwaway code.
These 20 sessions validate engineering, collection, replay, and recovery only;
they are never efficacy or promotion evidence by themselves.

### Workstream 0 — Review, decisions, and scope freeze

Deliverables:

- Fabel Ultracode architecture/statistical review of this document;
- Aaron decisions from Section 23;
- complete inventory of current data files, writers, readers, schemas, sizes,
  retention, and data-quality defects;
- a new decision record for DAS research storage and the role of Parquet/DuckDB;
- resolution of the Away desktop/mini-PC writer-role documentation conflict;
- exact initial universe, sessions, resolutions, and retention scope;
- capacity pilot plan and privacy/security boundary;
- approved RPO/RTO by data class, maximum pending-spool bytes and age, and the
  exact optional-cohort shedding order before degraded-operation tests begin.

Exit gate: approved architecture and no conflict with root `plan.md` or existing
decision records.

### Workstream 1 — Identity, time, schema, and golden baseline

Deliverables:

- canonical IDs and timezone/session contracts;
- primary/foreign-key ERD, cardinalities, deterministic ID/correction rules,
  risk-set/episode semantics, and cutover ownership matrix;
- schema registry, feature registry, setup registry, recipe registry, and
  outcome-definition registry;
- golden characterization of current D1/M5 bars, setups, levels, tracker rows,
  outcomes, ranking, and alerts;
- corrected-expectation fixtures for known point-in-time and identity defects;
- small sanitized end-to-end replay fixture.

Exit gate: current behavior is reproducible offline and intentional differences
are reviewable.

### Workstream 2 — DAS lake, manifest, and recovery foundation

Deliverables:

- configurable research-store path and capability check;
- single-writer ingestion service and local pending spool;
- immutable partitions, checksums, manifests, quarantine, and idempotency;
- generation manifests, logical-row reconciliation, pinned queries, and atomic
  `CURRENT` pointer;
- rebuildable catalog and query API;
- compaction, backup, restore, and capacity-health jobs;
- failure tests for DAS loss, partial writes, corruption, restart, and duplicate
  ingestion.

Exit gate: no evidence loss across controlled failures; a clean rebuild from
raw partitions succeeds.

### Workstream 3 — Canonical bars and multi-timeframe aggregation

Deliverables:

- provider-owned raw/normalized bar contracts;
- tiered M1/M5 collection, provider-native canonical D1/W1, and deterministic
  M15/M30/H1/H2/H4 plus optional validation-only derived D1/W1 variants;
- exchange calendar, extended-hours, DST, half-day, missing constituent, and
  revision handling;
- provider parity and provenance reports;
- live coverage shadow with no detector influence.

Exit gate: every timeframe replays deterministically and no forming/late bar can
confirm a state.

### Workstream 4 — Anchor, level, horizontal, and trendline registry

Deliverables:

- versioned anchors and champion AVWAP/band parity;
- moving-average level snapshots;
- structured horizontal-level lifecycle;
- human-level and human-trendline capture with known-at/revision semantics;
- algorithmic pivot/trendline research definitions;
- atomic level-interaction engine and confluence clusters;
- replay charts proving what geometry existed at each decision time.

Exit gate: all current levels reproduce exactly; no hindsight line or pivot can
leak backward.

### Workstream 5 — Complete eligibility and opportunity denominator

Deliverables:

- point-in-time universes and memberships;
- scheduled risk sets/evaluation slots plus candidate eligibility, including
  not-assigned, timeout, data-failure, and rejection reasons;
- stable setup occurrence/opportunity/lifecycle/attempt identities;
- quiet, rejected, never-triggered, not-surfaced, and zero-opportunity rows;
- coverage reconciliation from universe through alert/review.

Exit gate: repeated scans no longer inflate n, and missed-winner analysis has a
complete eligible denominator.

### Workstream 6 — Strategy recipes and unified outcome paths

Deliverables:

- primary control and setup-specific recipes;
- deterministic candidate × recipe simulator;
- intraday and swing path outcomes, MFE/MAE, first-hit, costs, gaps, censoring,
  ambiguity, and no-fill states;
- tracker/playbook mapping to explicit recipe IDs;
- standardized-versus-actual outcome separation.

Exit gate: every reported R names the exact policy that produced it and one
occurrence cannot masquerade as many independent trades.

### Workstream 7 — Reproduce current research and production evidence

Deliverables:

- warehouse projections matching Setup Tracker, Day Trade Tracker, playbook,
  Move Forensics, Technical Integrity, and current setup scoring on frozen data;
- exact parity or approved intentional differences;
- source-by-source cutover ledger: legacy champion write → immutable shadow
  ingest with legacy IDs/watermarks → parity → research-reader switch → separately
  promoted projection ownership; never two independent production writers;
- authority/reader/rollback state for every migrated artifact;
- coverage and survivorship reports;
- evidence links from every aggregate to source occurrences.

Exit gate: the new warehouse can replace research reads without changing live
decisions and can restore the legacy projection by configuration.

### Workstream 8 — Registered technical-variation research

Deliverables:

- bounded MA/timeframe grid;
- AVWAP anchor/band interaction studies;
- horizontal and trendline interaction studies;
- multi-timeframe alignment/conflict studies;
- confluence provenance and empirical-redundancy analysis;
- entry/stop/management comparator;
- walk-forward, matched-control, multiple-testing, and portfolio-overlap reports.

Exit gate: promising variations are nominated from untouched validation/test
evidence, not the best in-sample slice.

### Workstream 9 — Conditional expectancy and current-style challenger

Deliverables:

- hierarchical/shrunk conditional expectancy service;
- durable/recent/current-regime estimates and change detection;
- uncertainty, calibration, out-of-distribution, and abstention behavior;
- separate Swing, Intraday Quick, and Intraday Session style rankings;
- immutable `current_edge_snapshot_v1` with full provenance;
- champion comparison and replay shadow.

Exit gate: every result is reproducible, calibrated, and honest under thin or
missing evidence; it remains advisory.

### Workstream 10 — Research UI and evidence packages

Deliverables:

- Current Edge dashboard;
- Setup Matrix, Level Edge Lab, Multi-Timeframe Map, Recipe Comparator, replay,
  and research/promotion queue;
- deterministic plain-English explanations;
- bounded AI evidence package containing exact source IDs and hashes;
- System Health coverage and storage views.

Exit gate: Aaron can reach every claim's underlying occurrence and recreate its
point-in-time chart without trusting an AI summary.

### Workstream 11 — Live shadow, review identity, and calibration

Deliverables:

- substantially complete live-session collection across sides, regimes, and
  time periods;
- repaired review impression/action identity and outcome joins;
- daily candidate/eligibility/ranking audits;
- trader review of disagreements, missed winners, late confirmations, and
  no-trade sessions;
- frozen ranking acceptance manifest and rollback switch.

Exit gate: existing root-plan live-shadow floors plus setup-specific registered
floors pass. No unpromoted field affects production.

### Workstream 12 — One-family canary and controlled promotion

Deliverables:

- select one narrow setup/style family with incremental portfolio value;
- golden intentional-difference fixtures;
- bounded advisory, opt-in soft alert, and canary stages;
- paired cluster-aware primary-benefit and simultaneous non-inferiority evidence
  on identical champion/challenger risk sets;
- canary safety/mechanics gate followed by the powered matured-outcome efficacy
  gate, with no tuning between them;
- explicit Aaron approval, effective record, prior champion, and tested rollback;
- post-promotion monitoring and expiry/review date.

The promotion record freezes a surveillance manifest: matured-outcome review
cadence; drift, calibration, downside, and control limits; fixed-review or
sequential-error method; minimum evidence for deterioration; immediate safety
rollback triggers; expiry/revalidation rules; and a prohibition on retuning the
active champion from the same monitoring window.

Exit gate: only the approved family/version influences the bounded production
surface; every other variation remains research/shadow.

### Workstream 13 — Continuous expansion

Add families one at a time. Re-run data-health, multiple-testing, portfolio-
overlap, live-canary, and rollback gates for each. Never promote an unrestricted
model that can silently redefine its feature set or training window.

## 20. Verification strategy

### Data and time

- DST, half-day, holiday, extended-hours, and exchange-session fixtures.
- Completed versus forming M15/M30/H1/H2/H4 joins at boundary times, including
  legitimate final-session stubs.
- Late arrivals, provider revisions, missing constituents, and source switches.
- Corporate-action and symbol-change point-in-time replays.
- Universe membership and catalyst `known_at` tests.
- LIVE versus BACKFILL/RECONSTRUCTED fidelity and mechanically derived actual/
  simulated availability-clock tests.
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
- Matched-control determinism, multiple-testing correction, and cluster-aware
  intervals.
- Reproducible evidence from frozen manifests and code/config hashes.

### Storage and operations

- Idempotent ingest and retry.
- Crash during active write, partition seal, compaction, catalog rebuild, and
  snapshot publication.
- DAS loss/reconnect, bounded spool, corruption/quarantine, and full restore.
- Two-machine collision, clock skew, sleep/wake, role handoff, and fail-closed
  ownership.
- No GUI render path performs provider or large warehouse reads.

### Promotion

- Research features contribute exactly zero to champion score/alerts.
- One switch restores the prior champion without code revert or evidence loss.
- Every promoted value has a cited fixture, evidence snapshot, approval, effective
  time, expiry/review date, and rollback target.

## 21. Success metrics

### Corpus integrity

- At least 99.9% expected completed-bar coverage within each declared collection
  cohort/provider/acquisition-mode scope, with every non-collected or missing
  interval explicit. Scope expansion waits for the throughput pilot.
- Zero unexplained duplicate canonical keys.
- Zero silent provider blends, backward-known levels, or future feature inputs.
- 100% manifest/checksum coverage for sealed partitions and evidence freezes.
- Successful rebuild and sampled restore within declared recovery targets.

### Research integrity

- 100% experiments registered with immutable split and outcome definitions.
- 100% claims linked to market-episode/dependency-cluster IDs and source
  manifests.
- Complete reporting of sessions, symbols, episodes, missingness, uncertainty,
  costs, and multiple-testing family.
- Losing/inconclusive experiments retained.
- No production contribution from research-only fields.

### Opportunity quality

- Ready precision, precision@1/@3, planned remaining reward/risk, model expected
  net R, MFE/MAE, missed-winner
  rate, false-confirmation rate, and time-to-payoff improve or meet predeclared
  non-inferiority limits.
- Performance remains stable across validated sides, regimes, time periods,
  providers, and meaningful liquidity/volatility buckets.
- A new setup/style demonstrates incremental portfolio expectancy after overlap
  with current champions.
- The system abstains when current context is unsupported.

### Trader usefulness

- Aaron can see what is working for Swing, Intraday Quick, and Intraday Session
  separately.
- Every recommendation explains entry, invalidation, obstacle, planned reward/
  risk, model expected net R,
  timeframe alignment, evidence strength, counter-evidence, and freshness.
- Passed, missed, late, and actual execution results can be compared fairly.
- Any historical recommendation can be replayed exactly as it looked at the time.

## 22. Explicit anti-goals and traps

- No single giant mutable CSV or Drive-synchronized database.
- No arbitrary Cartesian explosion of every indicator combination.
- No silent change to AVWAP sigma math.
- No future-selected anchors, pivots, trendlines, universes, or regime labels.
- No mixing M1-derived and provider-native higher-timeframe bars without identity.
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
- No process mutates an active catalog. Only the warehouse builder may write a
  new unpublished `catalog.<generation>.duckdb`; application clients and other
  machines open published, pinned generations read-only.
- No assumption that RAID/DAS is a backup.
- No deletion of failed experiments or historical manual chart geometry.
- No claim that “best now” exists when evidence is stale, sparse, unstable, or
  outside the validated context.

## 23. Decisions for Aaron and Fabel Ultracode

Recommended defaults are shown first. These decisions should be resolved in
Workstream 0, before schemas freeze.

1. **Ownership:** one fixed, main-hosted warehouse service owns the DAS. A
   mutually exclusive session-scoped `scan_owner` is assigned to either main or
   mini-PC. Whichever machine scans writes its local immutable acquisition data
   and submits it through the same acknowledged warehouse import boundary; only
   the warehouse service writes new DAS generations.
2. **Base intraday archive:** M1 for Focus/model plus fixed/rotating exploration
   cohorts, M5 for a broader declared cohort, and D1 for the full screened
   universe, subject to the 20-session provider pilot.
3. **Sessions:** store regular and extended hours distinctly; production setup
   definitions declare which may be consumed.
4. **Store:** immutable Parquet/Zstd plus rebuildable DuckDB catalog; no mutable
   research DB in Drive.
5. **Universe:** version the full screened universe plus all manual/watch/focus
   symbols, including rejected names; separately version resolution-scoped
   collection cohorts and never backfill membership from today's universe.
6. **Primary objectives:** rank Swing, Intraday Quick, and Intraday Session
   separately; default utility is model expected net R subject to hard downside,
   actionability, and liquidity limits.
7. **Primary recipes:** choose one standardized swing and one standardized
   intraday recipe before comparing setups; alternatives remain correlated
   diagnostics.
8. **Human geometry:** capture manual horizontal levels and trendlines as
   versioned events from creation time onward.
9. **Experimental anchors:** begin with earnings, period-open, gap/catalyst,
   confirmed-pivot, breakout, volume-thrust, HOD/LOD, and manual anchors; add
   others only through the registry.
10. **Retention:** preserve raw and evidence-frozen data indefinitely while
    capacity permits; compact losslessly and monitor growth.
11. **Backup:** second physical target plus off-site manifests/definitions and
    scheduled restore drills.
12. **Operational continuity:** freeze RPO/RTO by data class, maximum pending-
    spool bytes and age, and the exact optional-cohort shedding order before
    Workstream 2 sizing and failure tests.
13. **Review cadence:** weekly evidence-health review, monthly research review,
    and explicit per-family promotion meetings rather than continuous auto-tuning.
14. **Data rights:** approve provider retention/backfill/backup terms before the
    archive leaves pilot scope.

### Questions the Fabel review should answer

- Is Parquet plus a rebuildable DuckDB catalog the right Windows/macOS/DAS design,
  or is another embedded/lakehouse layout safer for the projected write/query load?
- What M1/M5 cohort sizes and exploration-assignment policy are sustainable under
  actual provider budgets without embedding champion-selection bias?
- Which datasets should be event ledgers versus immutable snapshots or materialized
  marts?
- Is the proposed identity graph sufficient for simultaneous setups, anchors,
  timeframes, re-arms, and actual fills?
- Where can point-in-time leakage still enter anchors, corporate actions, manual
  geometry, resampling, universe membership, or regime conditioning?
- Which parts of the moving-average/AVWAP/level grid should be reduced before data
  collection, and which continuous primitives make the rest unnecessary?
- Which primary swing/intraday outcome recipes best separate pattern quality from
  stop/management quality?
- Are hierarchical shrinkage, walk-forward validation, clustered uncertainty, and
  multiple-testing controls sufficient for the expected dependency structure?
- What minimum evidence floors should apply to common versus rare setup families?
- How should the current-style engine balance durable edge, recent deterioration,
  regime similarity, remaining R, and uncertainty without overreacting?
- What migration sequence can reproduce every existing tracker before switching
  any research reader?
- What is the smallest useful first vertical slice that preserves the final schema?
- Which operational failure or recovery scenarios are still missing?

The requested Fabel deliverable should include: a written critique, explicit
accepted/rejected design decisions, a revised data model, a recommended first
vertical slice, a dependency-ordered implementation plan, statistical acceptance
manifests, a migration/rollback plan, and a risk register.

## 24. Definition of done

This program is complete when:

1. The DAS research corpus can be rebuilt from immutable, checksummed evidence.
2. Every bar, anchor, level, trendline, feature, setup, recipe, and outcome is
   versioned and point-in-time reproducible.
3. M5/M15/M30/H1/H2/H4/D1/W1 states align without forming-bar confirmation.
4. The complete opportunity denominator is retained, including quiet/rejected
   and zero-opportunity cases.
5. Current production setup/tracker outputs reproduce from the warehouse.
6. Setup quality, trade-policy quality, trader selection, and actual execution
   can be evaluated independently and together.
7. Technical variations are tested under immutable walk-forward splits with
   controls, uncertainty, and multiple-testing safeguards.
8. The Current Edge engine can identify supported Swing, Intraday Quick, and
   Intraday Session styles—or abstain—with cited evidence.
9. Any historical recommendation can be replayed with the exact data and chart
   geometry available at the time.
10. The GUI, satellite, Auto/Away report, and AI evidence package consume the
    same verified snapshot.
11. No research field affects a live score or alert before golden fixtures,
    shadow evidence, canary gates, Aaron approval, and tested rollback.
12. Storage loss, corruption, writer collision, migration, backup, and restore
    drills pass without destroying the last verified corpus.
13. The application remains decision-support only and never executes orders.

## Appendix A — Documentation traceability

| Source | Constraint carried into this plan |
|---|---|
| `docs/AWAY_SCANNER_RUNBOOK.md` | Single designated writer, truthful freshness, atomic last-good publication, explicit takeover and Drive limitations |
| `docs/BROKER_ADAPTERS.md` | App-owned provider interfaces, IBKR primary/Yahoo fallback, source provenance, read-only execution imports |
| `docs/decisions/0001-decision-support-only-no-order-execution.md` | Permanent decision-support boundary |
| `docs/decisions/0002-champion-challenger-shadow-promotion-ladder.md` | Research/shadow/advisory/canary/promotion separation and rollback |
| `docs/decisions/0003-ibkr-primary-yahoo-fallback-market-data.md` | Provider hierarchy and visible fallback identity |
| `docs/decisions/0004-pyside6-consumer-ui-tk-legacy-during-migration.md` | Headless/core data services with Qt as consumer; no warehouse logic in widgets |
| `docs/decisions/0005-cloud-synced-home-folder-file-storage.md` | Existing plain-file/Drive rule; a DAS lake requires an explicit new decision rather than silent replacement |
| `docs/decisions/0006-writer-lease-fencing-for-shared-exports.md` | One owner, fencing, fail-closed ambiguity, and atomic verified exports |
| `docs/decisions/0007-completed-bars-only-for-state-transitions.md` | Completed bars confirm; forming bars are Preview |
| `docs/decisions/0008-frozen-anchored-vwap-sigma-formula.md` | Running-deviation AVWAP sigma remains frozen and versioned |
| `docs/decisions/0009-golden-fixtures-before-detector-changes.md` | Characterization before any scoring/detector change |
| `docs/decisions/0010-ai-in-the-loop-review-policy-annotation-only.md` | Preference AI ranks/annotates only and never suppresses |
| `docs/decisions/0011-one-way-evidence-grounded-ai-advisory.md` | AI output cites immutable evidence and has no mutation path |
| `docs/decisions/0012-layered-requirements-with-constraints-pin.md` | Core/headless dependency placement and reproducible pins |
| `docs/decisions/0013-plan-md-authority-hierarchy.md` | Root `plan.md` remains authoritative |
| `docs/EVENING_MODE_RUNBOOK.md` | Same discovery semantics across modes; preserve open+30/+45/+60 persistence and zero-recommendation truth |
| `docs/FIRST_SESSION_CHECKLIST.md` | Session validation, artifacts, clock discipline, multi-machine drills, and writer-role conflict to resolve |
| `docs/FOCUS_PRICE_ALERTS_PROPOSAL.md` | Versioned user levels, one-fire-per-arm lifecycle, sticky delivery, last-price crossing separate from detector confirmation |
| `docs/MACOS_SETUP.md` | Configurable cross-platform paths, per-machine local state, no Windows-only path assumptions in core storage |
| `docs/MULTI_MACHINE_DESK_PROPOSAL.md` | Engine/data ownership remains on main; satellites consume relay snapshots and send acknowledged intents only |
| `docs/REGIME_INFRASTRUCTURE_PHASE1_RUNBOOK.md` | Scheduled point-in-time regime snapshots, explicit proxies/missing snapshots, no retroactive reconstruction |
| `docs/REVIEW_LEARNING_LOOP.md` | Impression/action/outcome loop, shrinkage, annotation-only preference policy, and known identity limitations |
| `docs/SETUPS_MAJOR.md` | Current swing/intraday ontology, house exits, Expected-R behavior, major levels, and production semantics |
| `docs/SETUPS_TEST.md` | Forward/backfill/reverse research harnesses, controls, point-in-time caveats, and promotion discipline |
| `docs/SHIP_READINESS.md` | Internal tool remains the priority; platform/dependency changes stay layered and operationally supportable |

## Appendix B — Example high-value research queries

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
7. Are EMA15 and EMA21 genuinely different edges on each timeframe, or correlated
   proxies whose apparent winner changes by sample?
8. Which current setup families deteriorate first when volatility expands, sector
   participation weakens, or market state turns choppy?
9. Which setups produce fast +1R but poor EOD outcomes, and which need longer
   management to realize their edge?
10. After controlling for setup, regime, and queue exposure, where does Aaron's
    selection improve the bot, and where is execution—not selection—the leak?

## Appendix C — Seed setup ontology requirements

Before migration, generate this registry from code and have Aaron review it.
Required columns are `canonical_setup_id`, aliases, parent/variant, role
(`TRADE_SETUP`, `CONTEXT`, `WATCH_STATE`, `CONTROL`, `FALLBACK`), status,
supported side, structural/context/trigger timeframes, exact completed-bar
trigger, primary recipe, exclusivity group, detector/config version, and current
weight authority. Prose evidence strings are not weight authority.

| Seed family/group | Role/status to preserve | Identity rule |
|---|---|---|
| AVWAPE to 1st Dev Favorite | Production trade setup | Parent favorite thesis; completed trigger distinct from zone residency |
| AVWAP Retest Followthrough | Production trade variant | Retest-hold entry; fold/compare with parent without double-counting one move |
| AVWAP Breakout | Production momentum variant | Cross/chase is distinct from later retest |
| AVWAP Band Bounce | Production trade setup | Band and touch order in identity; stop recipe is separate |
| Extreme Move Retest | Production trade setup | Displacement episode plus first controlled retest |
| SMA50/100/200 Breakout and Retest | Production family with period variants | Reclaim/watch and confirmed retest are separate states |
| TOP Weekly Leader | Context/basket plus linked daily trigger | Weekly pattern alone is not the entry |
| Favorite Zone Watch | Watch state | Never counted as a triggered trade setup |
| General/Untagged | Diagnostic fallback | Must not become a pooled “setup” edge |
| Post-Earnings Candle Break | Production, evidence accruing | Mutually exclusive with the 52-week variant for one trigger |
| Post-Earnings 52-week Break | Production family | Separate extreme-break thesis and exclusivity group |
| Post-Earnings AVWAPE Bounce | Production family with side asymmetry | Preserve long confirm-only/weak evidence and short hypothesis separately |
| Mid-Earnings EMA15 Retest | Production family | Requires the prior 2nd-deviation-zone episode |
| Mid-Earnings EMA21 Retest | Production sibling | Correlated with EMA15; explicit family wins if both fire |
| Mid-Earnings 1st-Dev Retest | Production sibling | Deepest retest; same episode dependence cluster |
| 2nd-Dev Power Hold | Context episode plus long-only research trade thesis | Two linked IDs; `mid_earnings_above_2nd_stdev` is an alias, not another sample |
| Standard/Dynamic/EOD VWAP families | Production intraday groups | Exact algorithm, band, confluence, and impulse-retest variants retained |
| EMA8/15/21 bounce | Production intraday siblings | Period/timeframe/touch count explicit; correlated-family cap |
| Rolling 10-candle and PDH/PDL | Production intraday levels | Exact rolling/session definition and interaction trigger retained |
| H1 EMA10 / blue-after-red / green-to-yellow | Production intraday HTF families | H1 state and lower-timeframe delivery remain distinct |
| Regime-pause RS/RW | Production/shadow status from released config | Tied to exact market episode and aligned RS window |
| ORB breakout/breakdown | Production intraday family | Opening-range definition and earliest eligible time versioned |
| EMA8 HOD/LOD grind | Production intraday family | Persistence episode and new-extreme trigger separated |
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
suppression. “Banger” remains unresolved until Aaron supplies a precise definition;
the migration must not infer one.
