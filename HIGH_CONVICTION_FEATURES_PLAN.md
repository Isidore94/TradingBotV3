# High-Conviction Trade Discovery Feature Plan

## Purpose

This document proposes end-user features that improve the quality, timing, and explainability of the trades the bot recommends most strongly.

The flagship change is a new **Greatness Monitor** for D1 Focus candidates. The current system is good at discovering stocks that are close to becoming excellent trades, but it treats an intraday crossing of an armed level as if the stock has already confirmed. A five-minute wick can therefore create a loud alert, while the later close, hold, or successful retest that truly confirms the setup may be missed.

The desired product behavior is different:

1. The slower Master AVWAP/D1 process discovers the thesis and important levels.
2. The bot creates a small, persistent confirmation plan for that ticker.
3. A lightweight intraday process watches the plan on every completed bar.
4. The ticker moves through visible stages as it clears levels and proves itself.
5. The user receives a high-priority alert only when the complete trade is genuinely ready.

This makes D1 Focus a developing-opportunity system instead of a periodic list of stocks that might be interesting.

The document complements [`plan.md`](plan.md). That file focuses mainly on architecture, correctness, performance, and maintainability. This file focuses on the end-user features that should produce better trade suggestions.

---

## Product Goal

The bot should answer four questions at all times:

- **What are the best trades right now?**
- **Which stocks are one or two confirmations away from becoming the best trades?**
- **Exactly what must happen next?**
- **What changed when a candidate became ready, failed, or was replaced?**

The strongest recommendation should not mean “many scanners mentioned this ticker.” It should mean:

- the daily thesis is strong;
- the intraday structure has confirmed;
- the stock is outperforming the correct benchmarks;
- volume and participation support the move;
- the next obstacle leaves enough reward;
- the entry is still actionable rather than extended;
- the market and sector context are compatible;
- the data is current and complete;
- no hard risk block is active.

This is a decision-support bot, not an execution bot. It should provide the evidence, timing, invalidation, and scenario plan needed for the user to make the execution decision.

---

## 1. Flagship Feature: The Greatness Monitor

### 1.1 What it is

The Greatness Monitor is a persistent intraday confirmation engine for promising D1 candidates.

Instead of rebuilding the entire opinion of a ticker each hour, the engine maintains two layers:

#### Slow structural layer

Calculated by Master AVWAP/D1 scans and revised only when material inputs change:

- direction and setup family;
- daily quality and conviction;
- relevant AVWAP anchors;
- daily moving averages and trendlines;
- major breakout, reclaim, or breakdown levels;
- prerequisite levels that must clear in order;
- invalidation levels;
- next obstacles and targets;
- catalyst and event context;
- setup expiration policy.

#### Fast confirmation layer

Updated incrementally on each completed intraday bar:

- distance to the next required level;
- whether the level was touched, wicked, closed through, or held;
- break-and-retest status;
- relative strength or weakness during the current market move;
- volume expansion and relative volume;
- VWAP, opening-range, and short-term trend alignment;
- entry extension and remaining reward-to-risk;
- setup stage, confidence, and failure state.

The slow layer defines what greatness would look like. The fast layer determines whether greatness is actually arriving.

### 1.2 Candidate lifecycle

Every candidate should have a typed state rather than being either “on the watchlist” or “alerted.”

```text
DISCOVERED
    -> DEVELOPING
    -> NEAR_TRIGGER
    -> TESTING_LEVEL
    -> CONFIRMING
    -> READY
    -> ACTIVE / EXTENDED / EXPIRED

Any live state may also move to:
    FAILED
    REARMED
    INVALIDATED
```

Suggested meanings:

| Stage | Meaning | User treatment |
|---|---|---|
| `DISCOVERED` | Strong D1 idea exists, but it is not close enough for active attention. | Quiet backlog. |
| `DEVELOPING` | Structure is improving or prerequisite levels are approaching. | Visible on the board. |
| `NEAR_TRIGGER` | Candidate is within a configurable ATR/percentage distance or has only one prerequisite left. | Promote to fast monitoring. |
| `TESTING_LEVEL` | Price is interacting with the next important level. | Update mini chart; optional soft heads-up. |
| `CONFIRMING` | Initial break/reclaim occurred and the system is waiting for a close, hold, retest, volume, or RS confirmation. | Prominent but not yet a “banger.” |
| `READY` | All required gates pass and an actionable entry window remains. | Loud high-conviction alert. |
| `ACTIVE` | Setup confirmed and remains within its intended entry/management window. | Track without repeatedly alerting. |
| `EXTENDED` | Thesis worked, but price is too far from the planned entry or invalidation. | Mark “correct idea, no chase.” |
| `FAILED` | A level test failed without invalidating the full D1 thesis. | Show failure reason; permit re-arming. |
| `REARMED` | The setup reset constructively and is eligible for a new confirmation attempt. | Return to monitoring with attempt count. |
| `INVALIDATED` | The structural thesis or hard invalidation failed. | Remove from active suggestions. |
| `EXPIRED` | The opportunity aged out or a catalyst/session boundary made it irrelevant. | Archive for review. |

State changes must be driven by completed bars and typed evidence. A UI refresh, hourly scan, or application restart must not accidentally reset the progression.

### 1.3 Multi-level “greatness ladder”

Many D1 Focus candidates need to clear one or two levels before they become excellent. The UI should represent that explicitly.

Example long plan:

```text
NVDA — LONG — Stage 4 of 6

[x] Reclaimed earnings AVWAP at 128.40
[x] Held VWAP on SPY pullback
[>] Testing prior-day high at 129.65
[ ] Needs a completed M5 close above 129.65
[ ] Needs hold/retest with RS still positive
[ ] Final quality check: room to 132.10, not extended
```

Example short plan:

```text
TSLA — SHORT — Stage 3 of 6

[x] Rejected daily supply at 248.20
[x] Lost intraday VWAP
[>] Testing prior-day low at 243.80
[ ] Needs a completed M5 close below 243.80
[ ] Needs weak bounce/retest while SPY stabilizes
[ ] Final quality check: clean downside room and acceptable extension
```

Each candidate should expose:

- levels already cleared;
- the current blocker;
- the next required confirmation;
- the final activation condition;
- invalidation;
- next obstacle/target;
- whether it is still actionable.

This transforms “watch this stock” into a usable if/then trade plan.

### 1.4 A cross is evidence, not confirmation

The existing intraday D1 trigger logic can fire when the latest five-minute high reaches a long level, or the latest low reaches a short level. That is useful as a `TESTING_LEVEL` event, but it is not sufficient for `READY`.

The new engine should distinguish:

- approached level;
- wick through level;
- close through level;
- second close/acceptance;
- retest begun;
- retest held;
- retest failed;
- false break recovered;
- fully invalidated.

Confirmation policies should be selectable by setup family. Examples:

- **Breakout:** completed close above the level, sufficient volume, then either acceptance or a successful retest.
- **Reclaim:** close back above the level, remain above for a minimum period, and hold the first controlled pullback.
- **Rejection short:** test from below, rejection candle, close away from the level, continued relative weakness.
- **Pullback entry:** constructive contraction into support, no structural violation, then reversal/hold evidence.
- **Opening-range continuation:** range break plus participation and market/sector agreement.

The system should never require every possible confirmation. It should require the small, setup-specific set that historical testing shows improves expectancy without making alerts too late.

### 1.5 Failure and re-arming

A failed first attempt should not consume the trigger for the entire day.

Recommended behavior:

1. A wick through a level marks `TESTING_LEVEL`.
2. A close back on the wrong side marks `FAILED_ATTEMPT`, not “alert completed.”
3. The system records the attempt, excursion, volume, and failure reason.
4. If the candidate rebuilds correctly, it moves to `REARMED`.
5. A later genuine close/hold can still produce a `READY` alert.

Re-arming should have controls:

- maximum attempts per session;
- minimum reset distance or time;
- no re-arm after structural invalidation;
- lower confidence after repeated failed breaks unless the setup specifically benefits from liquidity sweeps;
- explicit handling for undercut-and-reclaim and failed-breakout reversal setups.

Alert deduplication must use lifecycle transition identifiers, not only ticker, level, or message text.

---

## 2. D1 Development Radar Interface

### 2.1 Mini-chart grid

Create a grid of small live charts for the best developing candidates. The initial version can generalize the existing pyqtgraph candlestick implementation used by the SPY M5 chart into a reusable `SymbolMiniChart`.

Each mini chart should show only decision-relevant information:

- M5 candles, with optional M1/M15 switching;
- current price and session change;
- intraday VWAP;
- short-term EMA structure;
- opening range;
- prerequisite D1/AVWAP/SMA/trendline levels;
- prior-day high/low and premarket high/low when relevant;
- current trigger zone rather than an unrealistically precise single price;
- planned invalidation;
- next obstacle/first target;
- event markers for test, close, retest, confirmation, and failure;
- SPY/sector pullback shading or a compact relative-strength trace.

The chart should answer “what is this stock doing at the important level?” without forcing the user to open an external chart for every name.

### 2.2 Candidate card

Each chart sits inside a concise candidate card:

```text
NVDA  LONG                         CONFIRMING  84/100
Favorite setup | Post-earnings AVWAP reclaim

2 of 3 structural levels cleared
Now: testing PDH 129.65
Needs: M5 close + hold while RS remains positive

RS vs SPY pullback: +1.2 ATR       RVOL: 1.8
Entry quality: Good                Expected R: 2.6
Next obstacle: 132.10              Invalidation: 128.85
Data: complete through 10:35 ET
```

The card must clearly separate:

- **Why it is interesting** — the structural thesis;
- **Why it is not ready yet** — missing conditions;
- **What will make it ready** — the next observable event;
- **What would kill it** — invalidation;
- **Whether the user can still act** — extension and freshness.

### 2.3 Board organization

Recommended sections:

1. **Top 3 Ready Now** — candidates that pass every hard gate and are still actionable.
2. **Confirming** — currently interacting with the final level or retest.
3. **One Step Away** — one prerequisite remains.
4. **Developing** — strong daily ideas not yet close enough.
5. **Failed/Rearming** — useful context without contaminating top suggestions.
6. **Expired/Extended** — recently relevant names retained briefly for learning and review.

Recommended sorting within a section:

1. actionable readiness;
2. change in readiness over the last one to three bars;
3. setup expectancy and D1 strength;
4. expected reward-to-risk;
5. market/sector alignment;
6. freshness and data completeness.

The user should be able to:

- pin a ticker;
- mute a ticker for the session;
- pass on a setup with a reason;
- mark a setup as especially relevant;
- filter longs, shorts, day trades, or swing trades;
- filter by setup family;
- open the full setup detail;
- copy the symbol or trade plan;
- inspect the evidence timeline.

Pinned names receive visibility, not an automatic score bonus. Personal interest must not silently override objective quality.

### 2.4 Incremental chart updates

Mini charts must reuse the same market-data snapshot as the scanners and alerts. Rendering a card must not trigger a new provider request.

Recommended behavior:

- build the initial chart from the shared bar cache;
- append or replace only the latest bar;
- update only changed overlays;
- throttle repaint frequency separately from calculation frequency;
- render charts only for visible cards;
- pause animation/painting when the panel is hidden;
- keep the confirmation state active even if the panel is closed.

This allows a large watch universe without making the GUI or data provider the bottleneck.

---

## 3. Dedicated Intraday Priority Lane

### 3.1 Why it is needed

D1 watchlist names currently enter the general BounceBot symbol universe, but they are not automatically included in its highest-priority symbol list. Background names can be refreshed less often. That is acceptable for broad discovery, but not for a stock currently testing the final confirmation level.

The Greatness Monitor needs its own adaptive priority lane.

### 3.2 Suggested cadence

| Candidate class | Monitoring cadence | Work performed |
|---|---:|---|
| `TESTING_LEVEL` / `CONFIRMING` | Every completed M5 bar; optional lightweight M1 checks | Update live evidence and state. |
| `NEAR_TRIGGER` | Every completed M5 bar | Distance, RS, volume, extension, level interaction. |
| `DEVELOPING` | Every 10–15 minutes | Proximity and thesis health. |
| `DISCOVERED` | Hourly or on material market move | Determine whether promotion is justified. |
| `INVALIDATED` / `EXPIRED` | No live polling | Retain only for journal/outcome tracking. |

This is not a full Master AVWAP scan on every bar. It is a small calculation using cached bars and a precomputed confirmation plan.

### 3.3 Promotion rules

A candidate should enter the fast lane if any of these is true:

- it is within a configured ATR/percentage distance of its next level;
- only one prerequisite remains;
- it has touched or crossed the trigger zone;
- its readiness increased sharply;
- SPY or its sector entered the exact pullback/rebound regime relevant to the setup;
- the user pinned it;
- an opening-range, VWAP, or volume event made confirmation imminent.

It should leave the fast lane if:

- invalidated or expired;
- too extended to enter;
- moved far away from the trigger without constructive structure;
- data is unavailable for too long;
- replaced by higher-value candidates under a configurable capacity limit.

### 3.4 Capacity and provider safety

Start with a bounded fast lane, such as 20–40 names. Capacity should adapt to provider budgets and current application health.

When capacity is full, use an explicit priority calculation rather than first-come-first-served behavior. The UI should indicate if a candidate is being monitored at a reduced cadence.

---

## 4. High-Conviction Readiness Model

### 4.1 Separate the components

Avoid hiding all logic inside one unexplained score. Present four sub-scores plus hard gates.

#### Structural quality

- D1 setup strength;
- setup-family historical expectancy;
- AVWAP/daily level quality;
- trend and compression structure;
- catalyst recency and relevance;
- liquidity and tradability.

#### Live confirmation

- required levels cleared;
- close/acceptance/retest status;
- volume participation;
- intraday VWAP and EMA structure;
- relative strength/weakness behavior;
- quality of the trigger candle and follow-through.

#### Context alignment

- SPY regime;
- sector/industry regime;
- long/short breadth;
- time of day;
- volatility environment;
- relevant scheduled event risk.

#### Entry quality

- distance from trigger and invalidation;
- extension from VWAP/EMA/AVWAP;
- remaining room to the next obstacle;
- realistic expected reward-to-risk;
- spread, volume, and slippage risk;
- whether the opportunity window is still open.

### 4.2 Hard “Banger Ready” gate

A candidate should not enter **Top 3 Ready Now** merely because it has a high blended score. It must pass hard conditions:

- current D1 thesis remains valid;
- all required prerequisite levels are cleared or held;
- the setup-specific final confirmation occurred on completed data;
- RS/RW and volume rules pass when required by the setup;
- minimum reward-to-risk remains available;
- candidate is not excessively extended;
- no stale/incomplete-data block is active;
- no catalyst, halt, liquidity, spread, or market-context hard block is active;
- long and short direction are internally consistent across the plan.

Scores rank candidates that have passed. Scores should not cancel hard failures.

### 4.3 Avoid correlated evidence inflation

Several detectors can be different descriptions of the same price event. For example, a VWAP reclaim, a short-term EMA reclaim, and a momentum score may all rise because of one green candle.

The ranking system should group evidence into independent families:

- structure;
- participation;
- relative performance;
- market context;
- tradeability;
- entry timing.

Cap the contribution from each family. “Five signals agree” should only be persuasive when the signals add genuinely different information.

### 4.4 Confidence and evidence quality

Display confidence separately from readiness.

- **Readiness** answers: how close is the trade to activation?
- **Confidence** answers: how trustworthy is the available evidence?

Confidence should fall when:

- bars are incomplete or stale;
- benchmark/sector data is missing;
- the setup has a small historical sample;
- an inferred level lacks reliable metadata;
- market data sources disagree;
- the candidate entered through a legacy path missing required fields.

This prevents a precise-looking score from disguising uncertain inputs.

---

## 5. SPY Pullback Relative-Strength Confirmation

This should be one of the highest-priority additions because it directly identifies leaders and laggards when the market creates a useful test.

### 5.1 Long-side behavior

On a strong SPY day, detect controlled SPY pullback episodes and rank stocks by how they behave during the pullback.

High-quality long evidence includes:

- stock holds flat or advances while SPY pulls back;
- stock gives back materially less than SPY on an ATR/beta-adjusted basis;
- stock holds VWAP, opening-range high, or the armed trigger while SPY tests VWAP/EMA/support;
- sell volume contracts during the stock’s pullback;
- stock is among the first to make a new local high when SPY stabilizes;
- its sector also holds or rebounds, confirming the move is not isolated noise.

The best alert is not simply “this ticker’s RS line is high.” It is:

> SPY pulled back for three M5 bars; NVDA held above its breakout and VWAP, outperformed SPY by 0.9 beta-adjusted ATR, and is now breaking its pullback high with volume.

### 5.2 Short-side inversion

Mirror the logic for shorts:

- on a weak SPY day, identify stocks that cannot bounce when SPY rebounds;
- identify stocks that remain below VWAP/trigger levels while SPY recovers;
- measure underperformance on an ATR/beta-adjusted basis;
- prefer renewed downside expansion as SPY rolls back over;
- require clean downside room and borrow/liquidity checks where available.

Also support countertrend context carefully:

- a stock making new lows while SPY is strong can be exceptional relative weakness;
- a stock making new highs while SPY is weak can be exceptional relative strength;
- these should require stronger sector/liquidity confirmation because idiosyncratic news or bad data can create false extremes.

### 5.3 Pullback episode model

Define market episodes instead of comparing arbitrary snapshots:

1. Detect SPY impulse.
2. Mark the local impulse high/low.
3. Detect a controlled pullback or rebound.
4. Measure each candidate during that same interval.
5. Detect SPY stabilization/resumption.
6. Measure which stocks lead the resumption.

Features should include:

- raw and beta-adjusted return difference;
- ATR-normalized drawdown;
- percentage of SPY pullback retraced by the stock;
- VWAP/EMA/trigger preservation;
- stock and sector volume pattern;
- response lag after SPY resumes;
- whether stock structure improved while the benchmark weakened.

Persist the episode ID and window so every score and alert can be reproduced later.

### 5.4 User interface

Add a compact context line to each candidate:

```text
SPY test: PASS — held breakout during 10:15–10:30 pullback
Sector test: PASS — semis outperformed SPY by 0.4 ATR
Resumption: EARLY LEADER — broke pullback high one bar before SPY
```

The mini chart can shade the SPY pullback interval and display a small normalized stock-versus-SPY trace below price.

---

## 6. Sector and Industry Confirmation Chain

A ticker should be evaluated against both SPY and the benchmark that best explains its behavior.

Suggested hierarchy:

```text
Market -> Sector ETF -> Industry/group peers -> Stock
```

Useful features:

- sector relative strength versus SPY;
- stock relative strength versus sector;
- peer breadth and confirmation;
- sector ETF location versus VWAP/opening range/key D1 levels;
- whether the stock is a leader, follower, or isolated outlier;
- whether multiple peers are triggering the same setup simultaneously.

Do not automatically reject an isolated leader. Label the difference:

- **Chain confirmed:** market, sector, and stock align.
- **Stock-led:** stock leads a neutral sector; potentially powerful but needs catalyst/volume support.
- **Context conflict:** stock setup is long while sector and market are deteriorating.

The user should see the chain in one line rather than opening three extra charts.

---

## 7. Participation and Volume Quality

Raw relative volume is not enough. The system should describe whether participation supports the exact stage of the setup.

Add:

- time-of-day adjusted RVOL;
- volume on the break versus recent M5 baseline;
- contraction during pullback/retest;
- expansion on resumption;
- up/down volume imbalance;
- dollar-volume and spread checks;
- premarket participation versus regular-session confirmation;
- abnormal print/outlier handling;
- volume relative to the same time window on prior sessions.

Setup-specific examples:

- breakout: expanding break volume plus follow-through;
- pullback: contracting countertrend volume, then expanding reversal volume;
- reclaim: meaningful participation through the level, not a thin drift;
- rejection short: failure volume and inability to reclaim on lighter bounce volume.

The card should say `Volume: CONFIRMED`, `Volume: NEUTRAL`, or `Volume: FAILED`, with a short reason.

---

## 8. Entry Quality and Anti-Chase Protection

A great thesis can produce a bad alert if it arrives after the actionable entry window.

Every candidate should calculate:

- planned entry zone;
- current distance from entry;
- distance to logical invalidation;
- distance to next obstacle and targets;
- extension from VWAP, EMA, trigger, and ATR-normalized mean;
- spread/slippage-adjusted expected R;
- whether a pullback entry is still plausible;
- time since confirmation.

Use explicit states:

- `EARLY` — setup is close but not confirmed;
- `ACTIONABLE` — confirmation passed and price remains in the entry zone;
- `LATE` — some opportunity remains but quality is degraded;
- `NO_CHASE` — thesis worked, but the current entry is unacceptable;
- `RESET_WATCH` — wait for a new base/retest before reconsidering.

A `NO_CHASE` result should never appear as the highest recommendation, even if the stock is making a spectacular move.

---

## 9. Top Opportunities Command Center

Add a single decision surface that merges the best candidates from D1 Focus, Setup Tracker, RS/RW, opening-range, post-earnings, and other scanners.

The Command Center should not merge raw alerts. It should merge canonical opportunity records so the same ticker/setup appears once with combined evidence.

### 9.1 Top 3 Now

Keep this intentionally small. For each candidate, show:

- ticker and direction;
- setup family;
- readiness and confidence;
- entry zone and invalidation;
- first obstacle/target and expected R;
- D1, live, SPY, sector, and volume verdicts;
- what changed on the last bar;
- as-of time and data health;
- mini chart.

If fewer than three trades meet the standard, show fewer than three. The product should never fill a quota with mediocre trades.

### 9.2 “What changed?” feed

Show material transitions, not every scanner refresh:

```text
10:35 NVDA moved CONFIRMING -> READY
      Closed above PDH, held earnings AVWAP during SPY pullback,
      RVOL expanded 1.4 -> 1.9, expected R remains 2.6.

10:40 AMD moved READY -> NO_CHASE
      Price is now 1.1 ATR above the entry zone.

10:45 TSLA moved TESTING -> FAILED
      Wicked through PDL but closed back above; short thesis remains armed.
```

This makes the system feel alive without producing alert fatigue.

### 9.3 Honest empty state

On weak or directionless days, the best output may be:

> No high-conviction trade is ready. Two candidates are one confirmation away.

That is more valuable than elevating low-quality setups.

---

## 10. Alert Ladder and Notification Design

Use alert severity that matches the lifecycle.

| Tier | Example event | Delivery |
|---|---|---|
| Board only | Developing candidate improves slightly. | Silent card update. |
| Heads-up | One level away or within trigger proximity. | Soft visual notification. |
| Testing | Price begins interacting with final level. | Optional sound/user-configurable. |
| Confirming | Close occurred; waiting for hold/retest. | Prominent visual update. |
| Ready | All hard gates pass and entry is actionable. | Loud D1 Focus/Top Opportunity alert. |
| Failure | Test failed but thesis remains. | Quiet state change. |
| Invalidation | Structural thesis ended. | Clear removal notification if pinned. |
| No chase | Setup succeeded but entry is late. | Informational; never a loud entry alert. |

Each alert should include:

- the transition, such as `CONFIRMING -> READY`;
- exact evidence that changed;
- remaining entry room;
- invalidation and next obstacle;
- timestamp and bar interval;
- a compact chart snapshot or direct navigation to the candidate card.

Users should be able to configure sounds and delivery per tier, setup family, long/short side, session, and Auto/Away mode.

---

## 11. Auto/Away Mode Opportunity Report

Auto Mode should use the exact same opportunity snapshots and readiness gates as the desktop GUI.

### 11.1 Text report

The Google Drive text output should prioritize decisions:

```text
AS OF 10:35 ET — DATA COMPLETE THROUGH 10:35 M5

READY NOW
1. NVDA LONG — 84 readiness / High confidence
   Earnings AVWAP reclaim + PDH break
   Confirmed: M5 close, SPY pullback hold, RVOL expansion
   Entry 129.70–130.05 | Invalidation 128.85 | First obstacle 132.10
   Expected R 2.6 | Not extended

ONE STEP AWAY
1. MU LONG — needs M5 close above 143.20; currently testing
2. TSLA SHORT — needs weak retest of 243.80 from below

FAILED / NO CHASE
AMD — valid thesis, but entry is now extended
```

### 11.2 Mini-chart contact sheet

Optionally generate a single image containing charts for the top ready and near-ready candidates. Upload it alongside the text report.

The contact sheet should include:

- symbol, direction, stage, readiness;
- last 20–40 M5 bars;
- current/prerequisite levels;
- entry, invalidation, and first obstacle;
- SPY pullback shading;
- last transition time.

Generate it from existing cached bars. It must not start a second market-data pipeline.

### 11.3 Update policy

Update Away outputs on material changes rather than on a blind timer only:

- candidate becomes ready;
- top-ranked opportunity changes;
- candidate fails or becomes extended;
- new SPY pullback episode reveals a leader/laggard;
- data-health state changes;
- periodic heartbeat if nothing changed.

Use atomic file replacement so Google Drive never uploads a partially written report.

---

## 12. Additional High-Value Setup Features

The following features should be added after the core lifecycle and readiness model exist. They will be more reliable when they use the same canonical candidate, evidence, and outcome records.

### 12.1 Compression-to-expansion countdown

Detect high-quality daily/intraday compression near a meaningful level and monitor the transition toward expansion.

Evidence:

- decreasing ATR/range;
- tightening closes;
- declining pullback volume;
- repeated support/pressure absorption;
- sector/market alignment;
- room beyond the compression boundary.

The output should say which boundary matters and what confirms expansion.

### 12.2 Post-earnings constructive pullback

Monitor strong earnings gaps after the initial excitement:

- hold above earnings AVWAP;
- controlled multi-day or intraday retracement;
- declining countertrend volume;
- higher low or reclaim trigger;
- relative strength during market weakness;
- sufficient room before the next daily obstacle.

This is often more actionable than alerting only on the original earnings move.

### 12.3 Failed breakout and undercut/reclaim

Explicitly model failed moves instead of treating them only as invalidations.

- breakout failure that traps longs and confirms a short;
- breakdown failure that traps shorts and confirms a long;
- undercut of a key low followed by rapid reclaim;
- sweep of a high followed by rejection and inability to recover.

Require a transition and follow-through, not merely a wick.

### 12.4 Opening drive to first pullback

Identify strong opening moves, but delay the best alert until the first controlled pullback proves support.

- opening drive quality;
- relative volume and range;
- stock versus SPY behavior during the first pullback;
- hold of opening-range high/VWAP/fast EMA;
- resumption trigger;
- anti-chase gate.

### 12.5 Trend-day leader/laggard continuation

When the market establishes a credible trend-day regime, continuously rank:

- leaders holding high and re-accelerating on SPY pullbacks;
- laggards failing to bounce on SPY rebounds;
- stocks with sector confirmation and clean continuation levels.

This should share the SPY episode model rather than create a separate RS calculation.

### 12.6 Multi-timeframe alignment matrix

Show the evidence across D1, H1, M15, and M5:

| Timeframe | Trend | Level relationship | Setup role |
|---|---|---|---|
| D1 | Up | Above earnings AVWAP | Structural thesis |
| H1 | Up | Tight below PDH | Compression |
| M15 | Up | Holding VWAP | Context confirmation |
| M5 | Triggering | Closing through PDH | Entry timing |

Avoid requiring every timeframe to point in the same direction. The matrix should explain the role of each timeframe and identify genuine conflicts.

### 12.7 Catalyst-aware setup quality

Distinguish useful catalysts from dangerous event proximity:

- earnings recency and gap behavior;
- guidance/analyst/news catalyst when reliable data exists;
- scheduled earnings, Fed, CPI, or company-event risk;
- sympathy/group moves;
- stale headlines versus new information.

Treat uncertain or missing catalyst data as uncertainty, not proof that no catalyst exists.

### 12.8 Opportunity expiry and replacement

Every plan needs a lifetime.

Expire or downgrade when:

- the relevant session window passes;
- too many failed tests occur;
- price drifts too far from the setup structure;
- D1 thesis version changes;
- reward-to-risk collapses;
- market regime changes materially;
- event risk makes the original plan obsolete.

When a top candidate is replaced, tell the user why the replacement is now better.

### 12.9 Scenario plans

For the best candidates, generate concise branches:

```text
IF NVDA closes above 129.65 and holds the first retest,
THEN long setup becomes actionable in 129.70–130.05.

IF it wicks above but closes below,
THEN record a failed attempt and wait for a rebuild.

IF it loses 128.85,
THEN invalidate the intraday long thesis.
```

This is more helpful than a static score and remains useful even if the user disagrees with the final ranking.

---

## 13. Personalization Without Corrupting the Model

The bot should learn the user’s preferences while keeping objective performance separate.

Capture lightweight feedback:

- took;
- watched but passed;
- missed;
- too late;
- bad setup;
- wrong context;
- liked setup, disliked entry;
- alert useful but not traded.

Maintain separate values:

- model-estimated setup quality;
- user relevance/preferences;
- historical user execution fit.

For example, the user may prefer high-liquidity post-earnings names and dislike thin small caps. That should improve ordering for the user, but it should not rewrite the measured expectancy of the underlying setup.

Add a personal lane for manually focused names. These names receive continuous monitoring and full evidence, but they should still have honest readiness and risk labels.

---

## 14. Journal and Learning Loop

The journal should automatically reconstruct what the bot knew at every important moment.

### 14.1 Record every transition

For each candidate, persist:

- discovery time and original thesis;
- thesis/plan version;
- prerequisite levels and their sources;
- every stage transition;
- market, sector, RS, volume, and risk evidence at the transition;
- alert delivery and user response;
- best favorable/adverse excursion after each stage;
- whether the setup became actionable later after an initial failure;
- whether the user received the alert while it was still actionable.

### 14.2 Measure the funnel

Analyze outcomes at every stage, not only `READY`:

```text
Discovered -> Near Trigger -> Testing -> Confirming -> Ready -> Follow-through
```

Questions the bot should be able to answer:

- Which D1 setup families most often reach `READY`?
- Which prerequisite levels add useful selectivity?
- Does requiring a retest improve expectancy or merely make alerts late?
- Which RS behavior during SPY pullbacks predicts follow-through?
- How often do wick-only crossings fail?
- When does re-arming rescue a valid setup?
- Which confirmation gates reduce false alerts most efficiently?
- How much opportunity remains when the user receives the alert?

### 14.3 Primary quality metrics

Track:

- precision and expectancy of `READY` alerts;
- precision@1 and precision@3 for the Command Center;
- false-confirmation rate;
- actionable-at-alert rate;
- median remaining expected R at alert time;
- missed-opportunity rate;
- readiness calibration by score band;
- time from final confirmation to notification;
- top-candidate churn;
- percentage of sessions where the honest output is “no trade”;
- outcome coverage and data completeness.

Do not optimize only for alert win rate. A system can manufacture a high win rate by alerting too late or using tiny targets. Include reward, adverse excursion, opportunity capture, and timeliness.

### 14.4 Counterfactual research

Store enough evidence to compare candidate policies offline:

- wick versus close confirmation;
- one close versus two closes;
- immediate break entry versus retest;
- RS required versus optional;
- different extension thresholds;
- different market-regime gates.

Any learned change should run in shadow mode first and be compared to the current policy before it affects live ranking.

---

## 15. Canonical Data Model

Introduce a versioned `DevelopmentCandidate` or `OpportunityDevelopment` record.

Suggested conceptual fields:

```text
identity
  candidate_id
  ticker
  side
  setup_family
  session_date
  thesis_version

structural_thesis
  source_scan
  discovered_at
  quality_bucket
  reasons
  anchors
  validity_window

confirmation_plan
  ordered_steps[]
  trigger_zones[]
  confirmation_policy
  invalidation
  obstacles[]
  targets[]
  rearm_policy

live_state
  stage
  stage_entered_at
  current_step
  attempt_count
  readiness
  confidence
  distance_to_next_level
  actionable_state

evidence
  structural
  level_interaction
  relative_strength
  sector
  volume
  market_regime
  entry_quality
  data_health

provenance
  market_snapshot_id
  completed_bar_time
  benchmark_episode_id
  ruleset_version
  feature_version
```

Every field used for a recommendation should have a source, as-of time, and validity policy.

### 15.1 Ordered confirmation steps

A confirmation step should support:

- level or zone;
- long/short comparison direction;
- `touch`, `close`, `accept`, `retest`, or `reject` condition;
- required timeframe;
- number of completed bars;
- volume/RS/context predicates;
- whether the step is mandatory or supporting;
- expiration and reset behavior;
- evidence captured when passed/failed.

This is more expressive than a flat list of trigger levels and lets each setup family define its own path to readiness.

---

## 16. Integration With Current Code

The following mappings are based on the current implementation and identify where the behavior should evolve.

### 16.1 Master AVWAP/D1 discovery

Current area:

- `scripts/master_avwap_lib/legacy.py`
- `_build_d1_watchlist_trigger_levels`
- `update_master_avwap_d1_watchlist`
- `build_master_avwap_d1_upgrade_alert_payload`

Recommended evolution:

- keep the existing trigger-level builder as a compatibility source;
- translate trigger levels into ordered `ConfirmationPlan` steps;
- add prerequisite relationships, confirmation type, next obstacle, invalidation, and re-arm policy;
- persist thesis and plan versions;
- do not discard live development history simply because a subsequent scan does not repeat the exact same row;
- distinguish “no longer active in this scan” from “structurally invalidated.”

### 16.2 Live trigger detection

Current area:

- `scripts/bounce_bot_lib/legacy.py`
- `_find_master_avwap_intraday_trigger_events`
- `emit_master_avwap_intraday_trigger_flags`

Recommended evolution:

- replace the single wick-cross decision with a state update over completed bars;
- emit typed events such as `LEVEL_TOUCHED`, `CLOSED_THROUGH`, `RETEST_HELD`, `READY`, and `FAILED_ATTEMPT`;
- allow valid re-arming;
- deduplicate by candidate, plan version, attempt, and transition;
- apply the same D1 quality, risk, expected-R, and data-health gates to live upgrades;
- prevent a weak first wick from suppressing a later real confirmation.

### 16.3 Priority symbols

Current area:

- `get_scan_symbol_set`
- `get_priority_scan_symbols`
- background refresh cadence in BounceBot.

Recommended evolution:

- include `NEAR_TRIGGER`, `TESTING_LEVEL`, and `CONFIRMING` candidates in a dedicated priority lane;
- use stage/proximity to allocate cadence;
- separate small confirmation updates from expensive full detector passes;
- expose monitoring cadence and last complete bar to the UI.

### 16.4 Alert Center

Current area:

- `scripts/ui/panels/alert_center_panel.py`
- current text-based D1 Focus filtering for bucket/trigger alerts.

Recommended evolution:

- retain a chronological transition feed;
- add the D1 Development Radar card/grid view;
- reserve loud D1 Focus alerts for `READY` or user-configured stages;
- display “why not yet,” remaining levels, freshness, and actionability;
- stop relying on message-prefix parsing for important routing decisions.

### 16.5 Setup detail and charts

Current area:

- `scripts/ui/widgets/setup_detail_view.py`
- `scripts/ui/widgets/spy_m5_chart.py`

Recommended evolution:

- generalize the SPY candlestick widget into a reusable symbol mini chart;
- let Setup Detail subscribe to live opportunity snapshots;
- invalidate or version cached levels when a new thesis/scan is published;
- show the confirmation timeline and current plan on the chart;
- use the same snapshot that produced the alert.

### 16.6 Auto Mode and Away reports

Recommended evolution:

- publish one ranked opportunity snapshot consumed by both GUI and Auto Mode;
- make report generation a projection of that snapshot, not a separate ranking path;
- add material-change publishing and atomic writes;
- optionally generate a mini-chart contact sheet from cached bars.

---

## 17. Delivery Phases

### Phase 0 — Correctness and data prerequisites

Implement the relevant foundations in `plan.md`:

- canonical timestamps and completed-bar rules;
- shared market snapshots;
- data freshness/coverage states;
- typed direction and setup identifiers;
- centralized quality/risk gates;
- atomic snapshot publication.

Exit condition: the same ticker, bar, and level produce the same evidence in scanners, UI, alerts, and reports.

### Phase 1 — Candidate lifecycle and compatibility adapter

- define `DevelopmentCandidate`, `ConfirmationPlan`, stages, and transition events;
- convert current D1 trigger levels into simple one-step plans;
- persist state across cycles and restarts;
- retain current UI while running the new lifecycle in shadow mode;
- compare legacy wick alerts with staged outcomes.

Exit condition: every current D1 watch candidate has a persistent, explainable live state.

### Phase 2 — Dedicated priority lane and multi-level plans

- promote near-trigger candidates to fast monitoring;
- support ordered prerequisites;
- implement close/hold/retest/failure/re-arm transitions;
- add SPY pullback episode detection and RS evidence;
- enforce completed-bar, freshness, and actionability gates.

Exit condition: a weak wick cannot create or suppress a later genuine high-conviction alert.

### Phase 3 — D1 Development Radar and mini charts

- generalize the chart widget;
- build candidate cards and board sections;
- show levels, stages, reasons, and transition timeline;
- add pin/mute/pass actions;
- ensure charts render from shared cached data.

Exit condition: the user can understand the top candidates and their next required event without opening another scanner.

### Phase 4 — Command Center and alert ladder

- merge canonical opportunities across scanners;
- implement Top 3 Ready Now;
- add independent evidence-family scoring and hard gates;
- deliver stage-aware alerts and “what changed” explanations;
- implement anti-chase protection.

Exit condition: only fully qualified, actionable candidates reach the loudest recommendation tier.

### Phase 5 — Auto/Away parity

- project the same ranked snapshot into the text report;
- generate optional mini-chart contact sheet;
- publish on material changes plus heartbeat;
- verify atomic Google Drive-visible outputs.

Exit condition: Away mode and desktop mode show the same top candidates, evidence, and timestamps.

### Phase 6 — Learning and advanced setups

- measure the full candidate funnel;
- add outcome/replay tooling;
- calibrate confirmation policies per setup family;
- implement the advanced setup features in Section 12;
- personalize ordering without altering objective performance statistics.

Exit condition: policy improvements are supported by out-of-sample evidence and can be rolled back by version.

---

## 18. Required Tests

### Lifecycle tests

- a wick through a long level produces `TESTING_LEVEL`, not `READY`;
- a wick through a short level behaves as the exact directional mirror;
- a completed close through a required level advances the correct step;
- a required retest must actually hold before `READY`;
- a multi-level plan shows correct progress after each prerequisite;
- an invalidation prevents later readiness;
- an ordinary failed attempt can re-arm under policy;
- a later genuine confirmation alerts after an earlier failed wick;
- maximum-attempt and expiry rules are enforced;
- application restart restores the same stage and attempt count;
- an hourly scan does not reset valid intraday progress;
- a material thesis version change migrates or retires state safely.

### Data and timing tests

- incomplete bars cannot confirm a completed-bar rule;
- stale data blocks `READY` and displays the reason;
- out-of-order or duplicate bars do not duplicate transitions;
- SPY and stock evidence use aligned intervals;
- benchmark episode IDs and as-of times are persisted;
- missing sector data lowers confidence rather than inventing confirmation.

### Ranking and risk tests

- hard failures cannot be offset by a high score;
- an extended candidate cannot rank in Top 3 Ready Now;
- expected-R rules use the next real obstacle and logical invalidation;
- correlated indicators cannot inflate several independent evidence families;
- fewer than three qualified candidates yields an honest smaller list;
- long and short gates behave symmetrically.

### Alert tests

- exactly one loud alert is emitted for a specific `READY` transition;
- a failed/re-armed second attempt can create its own later `READY` transition;
- board-only changes do not make loud sounds;
- alert content includes completed-bar time, entry status, invalidation, and evidence;
- message formatting changes do not affect routing or deduplication.

### UI and performance tests

- mini charts use cached/shared bars and make no provider request during paint;
- only visible charts repaint at the high cadence;
- hidden panels do not stop monitoring;
- current level/plan versions replace stale Setup Detail cache entries;
- GUI remains responsive while the maximum fast-lane candidate count updates;
- Auto report and GUI render the same opportunity snapshot.

### Replay tests

Create deterministic replay cases for:

- wick failure followed by later successful break;
- breakout and clean retest;
- breakout that becomes immediately extended;
- long leader holding during a strong-day SPY pullback;
- short laggard failing during a weak-day SPY bounce;
- sector-confirmed move versus isolated false move;
- changing market regime that invalidates a formerly good setup;
- no-trade session with zero candidates meeting the hard gate.

---

## 19. Acceptance Criteria

The feature set is successful when:

1. D1 candidates near important levels are evaluated within one completed intraday bar, without waiting for the next full hourly rebuild.
2. The UI visibly shows levels cleared, current blocker, next confirmation, invalidation, and actionability.
3. A wick alone cannot create a `READY` alert unless a particular setup explicitly defines that behavior.
4. A failed first attempt does not suppress a later valid confirmation.
5. Top recommendations pass structure, confirmation, context, reward/risk, extension, and data-health gates.
6. SPY pullback/resumption behavior is part of the evidence for leader longs and laggard shorts.
7. The system can recommend fewer than three trades, including zero, without lowering its quality standard.
8. Desktop, Alert Center, Setup Detail, Auto Mode, and Google Drive outputs use the same opportunity snapshot.
9. Every high-priority alert explains what changed and why the trade is actionable now.
10. Historical replay can reproduce the state and ranking that existed at alert time.
11. Precision, expectancy, timeliness, false-confirmation rate, and remaining opportunity are measured for every ruleset version.
12. Mini charts and live monitoring do not cause duplicate downloads or materially degrade GUI responsiveness.

---

## 20. Recommended First Release

The smallest release that would materially improve the current D1 Focus experience is:

1. Convert each armed D1 trigger into a persistent candidate with stages.
2. Promote near-trigger D1 names into a dedicated every-M5-bar priority lane.
3. Treat a wick as `TESTING`, a completed close as preliminary confirmation, and a hold/retest as setup-specific final confirmation.
4. Permit failure and re-arming instead of suppressing the trigger for the day.
5. Add SPY pullback RS/RW and basic time-adjusted volume confirmation.
6. Add extension and minimum reward-to-risk hard gates.
7. Build a D1 Development Radar with six to twelve mini-chart cards.
8. Reserve the loudest alert for an actionable `READY` transition.
9. Put the same Ready/One-Step-Away list into Auto Mode’s Google Drive report.
10. Log the entire funnel for replay and later calibration.

That first release would directly address the current weakness: D1 Focus would stop saying only that a stock crossed an interesting level and begin saying that the stock has cleared the required structure, proved itself intraday, and is still offering a trade worth considering.
