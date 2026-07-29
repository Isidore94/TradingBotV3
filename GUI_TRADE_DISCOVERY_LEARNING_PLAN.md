# TradingBotV3 — Trade Discovery, Focus, Tracking, and Learning GUI Plan

**Status:** Plan-only product and implementation addendum

**Audit base:** `main` at `41eabc4`, 2026-07-28

**Scope:** GUI, read models, identity, tracking, learning, and alert presentation
**Authority:** Subordinate to `plan.md`, especially Sections 5–7 and the Section 12 execution order

This document does not authorize detector changes, shadow-engine promotion, automatic execution, automatic watchlist removal, or automatic AI tuning. It defines how the existing pieces should become one coherent decision-support platform.

## 1. Desired outcome

The platform should support one continuous loop:

```text
Universe discovery
  -> qualified opportunity
  -> ranked attention
  -> Focus monitoring
  -> material alert
  -> track / watch / plan / enter / pass / later decision
  -> execution or explicit no-trade
  -> objective and execution outcome
  -> evidence review
  -> advisory alert policy
```

The trader should be able to answer these questions without reconstructing information across tabs:

1. What are the best qualified trades now?
2. Why are they best?
3. What is missing before each one becomes actionable?
4. Which Focus names need attention now?
5. What changed since the last scan or completed bar?
6. Which setups are objectively working for swings and day trades?
7. Which setups fit the trader’s preferences and execution style?
8. Which good trades are being missed, and which weak trades are being over-selected?
9. Why did an alert interrupt the trader?
10. Can every recommendation, decision, and result be reconstructed later?

## 2. Executive recommendation

The next major GUI improvement should be a unified **Attention System**, not another independent scanner, tracker, or alert tab.

The system should have four connected working surfaces:

1. **Today / Command Center** — the best qualified opportunities by lifecycle stage.
2. **Focus Workbench** — the trader’s Swing and M5 names, ordered by what needs attention.
3. **Review and Alert Inbox** — one chart-first queue with explicit delivery lanes and decision capture.
4. **Learning Center** — objective edge, personal preference, execution quality, and policy audit.

The most important implementation rule is to keep three quantities separate everywhere:

| Quantity | Meaning | Allowed effect |
|---|---|---|
| **Objective Quality** | Measured setup quality after hard eligibility and data-health gates | Determines whether an opportunity deserves consideration |
| **Actionability Now** | Lifecycle stage, trigger state, remaining reward, extension, freshness, and current blocker | Determines urgency and alert severity |
| **Personal Fit** | Revealed preference and historical fit with the trader’s execution style | May reorder only inside a declared same-stage and Objective Quality band; never changes quality, gates, cadence, budgets, delivery lane, or sound |

Do not collapse these into one unexplained “AI score.”

## 3. What already exists and should be preserved

The current build already contains much of the required foundation. The plan should extend it rather than rebuild it.

| Existing capability | Keep |
|---|---|
| Chart-first Trading Desk with Alert Center on the left and Master Setups on the right | Preserve as the main intraday workspace |
| Compact sortable Master AVWAP table, bucket filters, Expected R, D1/M5 snapshots, setup details, Focus and dislike actions | Reuse as a primary discovery input |
| Chart-by-chart setup review and Space-key advancement | Extend with complete impression/pass tracking and stable opportunity identity |
| Swing and M5 Focus Picks with independent long/short files | Preserve the user-facing category split and safe watchlist injection behavior |
| Visual Alert Center review queue, tier filters, D1 Focus, RS/RW board, chart watches, persistent D1 levels, and armed inventory | Keep and reorganize into typed delivery lanes |
| Always-visible sector/industry tape and cache-only snapshot charts | Reuse in every opportunity dossier |
| Setup Tracker, plain-English “What’s Working,” Day Trade Tracker, Move Forensics, and human-vs-bot summaries | Surface through a Learning Center and link them to current opportunities |
| Journal schema v2 with imported Taken/Closed events and structured trade review | Extend into the complete discovery-to-outcome lifecycle |
| Review decision log, preference scoreboard, AI-curated policy, queue guidance, blind spots/leaks, and watch conversion | Keep advisory; improve identity, semantics, visibility, versioning, and audit |
| CandidateRegistry with user-protection, source leases, typed transitions, and shadow adoption | Continue the roadmap’s dual-write and promotion sequence |
| Legacy SPY pause and D1 wick alert champions | Preserve until their challengers pass the declared evidence gates |

`SOL_PROGRESS.md` is older than the current July 28 build, so implementation planning must use current source and Git history in addition to that checkpoint file.

## 4. P0 correctness findings

These issues must be fixed before stronger personalization or learned alert ordering is trusted.

### 4.1 Identity is too coarse

Several current paths use a ticker as if it were an opportunity:

- Alert review episodes are folded by `(trade_date, symbol)`.
- The visual review queue replaces queued rows for the same symbol.
- Dwell tracking and guidance caching are symbol-based.
- Setup chart review locates the first visible row for a symbol.
- Human Focus snapshots key rows by `(trade_date, symbol, side)`.

This can merge:

- a Swing thesis and an M5 thesis;
- a long and short thesis;
- two setup families or anchors;
- two attempts after a failure/re-arm;
- separate alerts at different times;
- an Alert Center decision and a Master Setups decision.

The current audit found six long symbols simultaneously present in both Swing and M5 Focus. The current daily Focus snapshot can retain only the category first recorded for a same-day symbol/side, so those memberships cannot be graded independently.

**Required correction:** propagate canonical candidate, opportunity, lifecycle, transition, impression, attempt, watch, trade, and outcome identities before using the data to alter live priority materially.

### 4.2 “Take probability” currently means engagement probability

The current review aggregation treats these as takes:

- add to Focus;
- favorite;
- cross-focus toggle;
- arm chart watch;
- arm price level.

Arming a watch usually means “not ready; notify me later,” not “I would enter this trade.” Until event semantics are separated, the GUI should label this metric **engagement probability**, not take probability.

The event model must distinguish:

```text
SHOWN
  -> TRACKED / FAVORITED
  -> WATCHED / PLANNED
  -> READY
  -> ENTERED

or

SHOWN
  -> PASS_NOT_READY
  -> PASS_POOR_QUALITY
  -> PASS_PERSONAL_FIT
  -> LATER
  -> MISSED
  -> INVALIDATED
```

### 4.3 M5 Focus grading is not day-trade grading

Swing and M5 Focus cohorts currently use daily closes and 1/3/5/10-session forward returns. That is useful selection research for swings, but it does not establish day-trade expectancy.

M5 evidence needs:

- decision and trigger timestamps;
- completed-M5-bar identity;
- entry, stop, target, and risk per share;
- EOD and close R;
- MFE and MAE in R;
- time to trigger and time in trade;
- alert/event linkage;
- took, watched, passed, missed, or never-triggered status.

Swing evidence should retain versioned 1/3/5/10-session returns and setup-scenario R. The two horizons must not share one outcome definition.

### 4.4 Setup review lacks a complete denominator

The Master Setups table records favorite/dislike decisions, but it does not record a chart impression, dwell time, or a structured pass when the trader advances without liking or disliking.

That means:

- take rates are unavailable for setup-table segments;
- unreviewed and deliberately passed charts cannot be distinguished;
- likes and dislikes are not a representative denominator;
- preference learning is vulnerable to selection bias.

Every chart review should create an impression ID, and every advance should resolve it as Track, Watch, Plan, Enter, Pass, Later, or `SKIP_UNRESOLVED`.

### 4.5 Existing evidence must not silently auto-promote

The historical setup/tracker program still has open point-in-time and setup-identity work in `plan.md` Milestone 5. The current tracker also has an `apply_changes=True` scoring-tuner path.

Before new GUI learning is connected to live alerts:

- snapshot the current champion configuration and capture golden characterization before changing tuner execution;
- then route new learned/tuner mutations into proposal/shadow outputs behind rollback;
- preserve existing champion behavior;
- produce immutable train/validation/test evidence;
- add intentional-diff fixtures;
- require an explicit promotion record and rollback.

This plan must not expand the existing auto-tuning path.

### 4.6 Data capture has not reached the evidence floor

At the audit timestamp:

- `alert_review_events.jsonl` had not yet been created in the shared store;
- the preference state, report, and active/draft policy files were therefore absent;
- `pick_feedback.jsonl` contained 102 rows: 101 likes and one dislike.

This is expected before restarting onto the July 28 review-learning build, but it means there is not yet enough balanced behavior data to tune anything. First collect approximately two to three weeks of normal sessions and clear the declared per-segment sample floors.

These observations were read at **2026-07-28 16:34 PDT** from `C:\Users\aaron\My Drive\Trading\TradingBot`; Appendix A records the source file metadata. They are an audit snapshot, not a permanent product fact.

All evidence gathered before the v2 identity and action-semantic parity gate is **Exploratory / Non-Promotable**. The promotion-evidence clock starts only after the corrected schemas produce stable, independently attributable episodes.

The current preference guidance can actively reorder the review queue after only four shown/outcome samples. Until identity is corrected and the declared policy gates pass, run preference-derived ordering in shadow or force FIFO compatibility. Guidance may annotate the chart, but it must not change the active queue order.

## 5. Target information architecture

Avoid a risky navigation rewrite at the start. Evolve the existing pages in place, then simplify navigation after their data contracts are stable.

### 5.1 Near-term navigation

- **Trading Desk** becomes **Today** once the Command Center exists.
- **Focus Picks** becomes **Focus Workbench**.
- **Research** becomes **Research & Learning** and gains preference/policy views.
- **Journal**, **Universe**, **Auto Pilot**, **A.I. Summary**, **Health**, and **Settings** remain.

### 5.2 Later consolidated navigation

```text
Today
Focus
Learn
Journal
Universe
System
```

`System` may contain Auto Pilot, A.I. Summary, Health, and Settings after the current pages are stable. This is presentation consolidation only; it must not merge their runtime owners.

## 6. Today / Opportunity Command Center

The Today page should answer “what deserves attention now?” from one canonical snapshot.

### 6.1 Default layout

```text
┌──────────────── Market / regime / freshness / data health ────────────────┐
│ Ready Now: 0–3 honest best opportunities                                  │
├───────────────────────┬───────────────────────────────────────────────────┤
│ Chart-first review    │ Focus + current opportunity queue                 │
│ D1/M5 + levels        │ Confirming / One Step Away / Developing           │
│ next blocker          │ reason ranked / last meaningful change            │
├───────────────────────┴───────────────────────────────────────────────────┤
│ New universe discoveries | Invalidated / No Chase / Stale | Data issues   │
└────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Lifecycle lanes

Use three primary groups so the page can be scanned in seconds:

1. **Act Now** — user-armed hits and completed-bar Ready opportunities.
2. **Monitor** — Confirming, One Step Away, Testing Level, Developing, and new discoveries.
3. **Resolve / Review** — materially changed, Failed/Rearming, No Chase, Invalidated, Expired, Stale, and Data Issues.

The detailed lifecycle states remain visible as chips, filters, and saved views. They are not nine competing top-level columns.

Show zero, one, or two Ready candidates when fewer than three qualify. Never backfill with lower-quality names to make the page look active.

### 6.3 Opportunity card contract

Keep the collapsed card compact. It should show:

- symbol, side, horizon, setup family, thesis, and lifecycle stage;
- Objective Quality, Actionability Now, and Personal Fit separately;
- exact next confirmation and current blocker;
- entry zone, invalidation, obstacle, target, and Expected R;
- extension/no-chase state and the last material change;
- Focus/armed state;
- complete-through timestamp and data-health state.

The detail drawer should show:

- market, sector, industry, and stock alignment;
- volume/participation state;
- setup evidence: sample count, sessions, recent edge, and evidence maturity;
- source/provenance;
- alert profile and delivery history;
- missing inputs and snapshot ID;
- what changed on the last material transition.

### 6.4 Opportunity rank versus attention order

Preserve two inspectable orders:

- **Canonical opportunity rank** — hard-gate completeness, lifecycle stage, Objective Quality, remaining Expected R/anti-chase, evidence confidence, concentration, and stable identity. Armed/Focus state and Personal Fit do not change this rank.
- **Attention order** — the trader-facing work queue. It starts from the canonical rank, pins fired user-armed conditions to the front, surfaces material Focus changes, and may use Personal Fit only as a tie-break inside a declared same-stage/quality band.

The default attention order is:

1. User-armed condition that fired.
2. Completed-bar Ready and actionable.
3. Confirming / Testing / One Step Away.
4. Focus names with a material state change.
5. Other qualified developing opportunities.
6. New universe discoveries.

Within the same stage/attention class, preserve:

1. hard-gate completeness;
2. objective quality;
3. remaining Expected R and anti-chase state;
4. evidence confidence;
5. concentration/diversification;
6. Personal Fit as a tie-breaker;
7. stable deterministic identity.

The explanation drawer must show both rank values and why the attention order differs. An arm, Focus membership, or preference signal may not change canonical opportunity rank.

### 6.5 Opportunity dossier

Add a persistent search/command box to the app shell. Searching a ticker first opens a symbol overview, but the trader must explicitly select a side, horizon, and thesis before mutating state. The selected opportunity dossier contains:

- D1/M5 charts and levels;
- every distinct current side/horizon/setup thesis;
- Focus memberships and origins;
- current lifecycle, blockers, and recent transitions;
- armed watches and alert profile;
- setup and day-trade evidence;
- alert/review history;
- linked journal trades and outcomes;
- freshness and source health.

The dossier must consume cached snapshots and perform no provider request during render. Until canonical opportunity identity lands, it must show ambiguous matches as separate, explicitly provisional rows rather than merging them by symbol.

## 7. Focus Workbench

This is the highest-value low-risk GUI improvement. Focus Picks should become a sortable monitoring cockpit while retaining the current safe write behavior.

### 7.1 Default lanes

- **Armed / Ready** — a user-armed hit or genuinely actionable completed-bar Ready state.
- **Changed / Decision Needed** — a new material change, including a newly invalidated thesis, until acknowledged.
- **Waiting** — a valid thesis with a named next condition or blocker.
- **Developing / New** — newly focused or still building.
- **Invalid / No Chase / Stale / Data Missing** — terminal or unhealthy states retained for review, never silently removed.
- **Archived by User** — recoverable history.

Apply that precedence in order so each opportunity occupies exactly one lane. After a new invalidation is acknowledged, it moves from Changed / Decision Needed to the terminal Invalid lane.

### 7.2 Required columns

The compact default must fit the existing Focus viewport without depending on a horizontal scrollbar:

| Default column | Purpose |
|---|---|
| Symbol / Side | The trade direction, not only the ticker |
| Horizon | Swing or M5; both can exist independently |
| Stage | Current lifecycle state, or Unknown |
| Next | Exact next confirmation or blocker when known |
| Exp R | Remaining Expected R when valid |
| Changed / Alert | Last material change and armed state |
| Freshness | Complete-through time and health |

The expandable detail drawer and optional column chooser expose:

- setup family, anchor, and thesis version;
- distance to trigger/level;
- Objective Quality and evidence confidence;
- Personal Fit with sample/maturity indicator;
- market, sector, and industry RS/RW;
- source/provenance;
- alert profile and armed condition;
- complete ranking explanation.

### 7.3 Saved views

- Swing — Needs Attention
- Swing — Waiting
- M5 — Needs Attention
- M5 — Waiting
- Both Horizons
- Changed Since Last Scan
- Unreviewed
- Armed
- Stale / Missing Data
- Long / Short
- Setup Family
- Sector / Industry concentration

### 7.4 Common row actions

Use the same actions from Master Setups, Universe, RS/RW, Alerts, and Focus:

- open D1/M5 chart;
- add/remove Swing Focus;
- add/remove M5 Focus;
- arm named chart condition;
- arm persistent level;
- mark Plan Entry;
- record Entered only through an explicit action or broker-linked execution;
- Pass with structured reason;
- Later / snooze until event or time;
- archive manually;
- undo the last mutation;
- open the opportunity dossier.

### 7.5 Manual-name protection

- Automation may mark a manual Focus name Stale, Invalidated, or Needs Review.
- Automation may never remove it.
- Detector evaluation and freshness cadence remain champion-defined unless a separately tested runtime policy is explicitly promoted.
- Attention and sound budgets may change presentation only; they do not authorize reduced scanning.
- “Clear All” must require confirmation and offer undo.
- Removing one horizon or thesis must not remove every category, side, or opportunity for the symbol.

## 8. Review Queue and alert decision capture

The chart-first review flow should remain the primary way to process alerts and setup candidates.

### 8.1 Queue drawer

Replace the opaque “N waiting” label with an expandable queue showing:

- queue position and age;
- symbol, side, horizon, setup, and stage;
- why it is ahead of the next item;
- objective contribution;
- urgency/actionability contribution;
- preference contribution;
- armed/focus state;
- freshness/data warnings.

User-armed hits remain fixed at the front.

### 8.2 Review actions

Keep one compact primary action row with explicit, non-overlapping semantics:

- **Track / Focus**
- **Watch / Arm**
- **Plan Entry**
- **Entered / Took Trade**
- **Pass**
- **Later**

`Entered` must come from an explicit trader action or a broker-linked execution. Planning an entry, adding to Focus, or arming a watch must never be inferred as entry.

Put destructive or scope-sensitive actions in an overflow menu:

- remove this opportunity today;
- remove this horizon/thesis from Focus;
- remove this symbol today;
- restore the prior state.

Every removal must display its exact scope, require confirmation when it can affect more than one membership, and offer undo. **Not Relevant** is a structured Pass reason, not a separate primary verdict.

Pass reasons should use structured chips plus optional text:

- not ready;
- not relevant to this workflow;
- poor chart/structure;
- no room / poor R;
- stale or missing data;
- too extended / chased;
- market or sector conflict;
- earnings/event risk;
- liquidity;
- duplicate thesis;
- not my style;
- already exposed / concentration;
- wrong side;
- other.

This separates objective defects from personal preference and execution constraints.

Advancing to another chart without a verdict records `SKIP_UNRESOLVED`, not Pass. The end-of-queue workflow should make unresolved impressions easy to revisit.

### 8.3 Complete impression model

Record:

- eligible for queue;
- admitted to queue;
- shown;
- shown position;
- `eligibility_id` and `ranking_snapshot_id`;
- ranking components at impression time;
- preference-policy and delivery-policy versions;
- dwell time;
- every action;
- final resolution;
- not shown because of attention budget;
- outcome at the versioned horizon.

Without eligible-but-not-shown records, the system cannot measure whether a new ordering policy causes good opportunities to disappear from view.

Write at most one eligibility record for an `opportunity_id + attempt_id + policy_version + ranking_snapshot_id`. A GUI refresh may update delivery state, but it must not create a second independent denominator.

### 8.4 Review ergonomics

- Space: next unresolved chart.
- `F`: add to matching Focus horizon.
- `W`: watch/arm.
- `P`: pass and open reason chips.
- `L`: later.
- Backspace: previous chart.
- Ctrl+Z: undo last review mutation.
- End-of-queue summary: taken, focused, watched, passed, deferred, unresolved.

Keyboard bindings must avoid conflicts with chart typing and text fields.

## 9. Anti-spam Alert Center

The system should reduce interruption, not discard evidence.

### 9.1 Delivery lanes

| Lane | Default behavior |
|---|---|
| **Immediate / Loud** | User-armed hit, completed-bar actionable Ready, critical invalidation on a Focus name |
| **Heads-Up** | Testing, Confirming, or One Step Away material transition; one notification per typed transition |
| **Focus Changes** | Every meaningful Focus transition remains visible; sound profile is configurable |
| **Developing Board** | Silent state updates and previews |
| **Research / Data** | Research observations, health, partial/stale warnings |
| **History** | Every retained transition and decision |

### 9.2 Focus alert profiles

Every material Focus transition must remain visible in **Focus Changes** and **History**. This does not mean every scan event stays at the front of the active queue or produces a sound.

“Material Focus transition” is a fixed typed contract:

- lifecycle-stage or Actionability Now change;
- entry/invalidation/target/no-chase plan change;
- armed-watch arm, fire, disarm, or expiry;
- data-health/freshness state change;
- setup/thesis-version change;
- explicit Focus add, scoped remove, restore, or archive.

Preference or delivery policy may not redefine materiality. Every genuine typed event remains discoverable in Focus Changes/History; only duplicate render, refresh, or retry deliveries may collapse.

Before the delivery-policy canary passes, preserve only controls already supported by the champion. Existing global mute/volume behavior may remain, but do not add a new selective sound policy.

After the canonical lifecycle/alert-transition gates and Phase 8 delivery manifest pass, support these per-opportunity profiles:

- **Ready + invalidation only** — recommended post-gate default;
- **All material transitions**;
- **Armed conditions only**;
- **Visual only**.

An explicit user-armed hit always bypasses digest delay/cooldown and remains at the front; whether it sounds follows the trader’s explicit profile.

### 9.3 Deduplication and burst handling

- Deduplicate by `opportunity_id + attempt_id + transition_type + completed_bar_id`.
- A repeated render, scan, or message text must not create a second alert.
- A genuine escalation bypasses the prior cooldown.
- Update one live opportunity card instead of appending repeated status messages.
- Group correlated sector/thesis bursts under the best representative while keeping alternatives inspectable.
- Never drop events from History.
- A notification budget routes lower-urgency items to the queue/digest; it does not delete them.

### 9.4 Safety fixes

- Split **Clear Feed History** from **Disarm Session Watches**.
- Require confirmation before bulk disarm or bulk Focus removal.
- Preserve persistent D1 levels unless explicitly disarmed.
- Show “Why this sounded” on every loud alert.
- Show current policy version and evidence age.

### 9.5 Attention controls

Add a compact alert-control strip showing:

- active champion/delivery mode;
- loud alerts used this session;
- queued/digest count;
- armed-watch count;
- next digest time;
- noncritical snooze state.

Before the delivery canary, this strip is observational and exposes only champion-supported controls. After the Phase 8 gate, the trader may quiet noncritical Heads-Up events, choose a Focus sound profile, or snooze noncritical sound for a bounded period. Snooze never disarms a watch, changes monitoring cadence, hides History, or delays a fired user-armed hit from the front of the queue. Any learned budget change is previewed as a separate delivery-policy proposal.

## 10. Learning Center

The current objective trackers and preference artifacts should become one evidence workspace.

### 10.1 Four separate evidence questions

1. **Objective setup edge** — What follows through under standardized rules?
2. **Preference** — What does the trader choose to review, focus, watch, or enter?
3. **Execution fit** — Which opportunities does the trader execute well?
4. **Alert utility** — Which notifications lead to useful action without being late or noisy?

Do not use one of these as a substitute for another.

### 10.2 Swing and Day Trade tabs

**Swing**

- setup family, side, bucket, setup tag, and Expected-R band;
- 1/3/5/10-session side-adjusted returns;
- standardized scenario R;
- closed R, target/stop rate, MFE/MAE when valid;
- recent versus baseline edge;
- focus origin and human-vs-bot lift.

**Day Trade**

- bounce type/combo, side, time bucket, regime, RRS, RVOL, Focus state;
- event-linked entry/stop/target;
- close/EOD R, MFE/MAE R, 1R/2R hit and stop rate;
- notification latency and remaining R at alert;
- watch-to-fire and fire-to-entry conversion;
- actual execution versus standardized hypothetical outcome.

### 10.3 Default dashboards

- **What’s Working Now** — qualified leaders with horizon, n, sessions, freshness, and confidence.
- **Preference vs Edge matrix**:

  | | Low objective edge | High objective edge |
  |---|---|---|
  | **High preference** | Potential leak | Best fit |
  | **Low preference** | Correctly ignored | Blind spot |

- **Selection funnel** — discovered → reviewed → focused → armed → ready → entered → followed through.
- **Human vs Bot** — matched cohorts, same horizon and point-in-time evidence.
- **Alert Quality** — loud alerts, duplicate rate, action rate, Ready precision, remaining R, missed winners.
- **Data Health** — event coverage, independent episode count, outcome join rate, malformed rows, missing fields, stale artifacts, policy age.

### 10.4 Evidence display rules

Every claim must show:

- independent sample count;
- session count;
- date range;
- horizon and outcome definition;
- baseline/control;
- mean and median;
- dispersion or confidence band;
- missingness;
- recent versus long-run result;
- setup/rule/outcome version;
- in-sample, validation, test, shadow, or live status.

Thin evidence may be shown but must be labeled Early. It may not drive a production policy.

### 10.5 Click-through behavior

Clicking a “What’s Working” result should open:

1. the underlying cohort;
2. the exact outcome definition;
3. representative winners and losers;
4. current qualified opportunities that match it;
5. any active policy rule using it.

### 10.6 Research and promotion discipline

Until point-in-time repair and v2 identity/action parity are complete, place a persistent banner on every result:

> **RESEARCH / NOT POINT-IN-TIME VALIDATED — cannot affect production ranking or alert delivery**

Before any result can support a policy:

- freeze the feature, configuration, train, validation, and untouched test windows before examining test outcomes;
- correct for multiple comparisons and retain every tried, rejected, and control cohort;
- retain eligible-but-rejected and no-trigger opportunities;
- define the independent sampling unit and prevent rescans or correlated variants from inflating `n`;
- report missingness, delayed labels, censoring, expiry, and never-triggered cases explicitly;
- use a predeclared primary outcome per horizon;
- treat alternative targets, stops, and holding periods from the same opportunity as correlated scenario analyses, not independent wins;
- require shadow and canary evidence collected only after v2 parity; pre-parity data remains exploratory and non-promotable.

## 11. Journal and lifecycle review

The journal should become the durable bridge between recommendation quality and execution quality.

### 11.1 Universal timeline

```text
Discovered
  -> Developing
  -> Testing / Confirming
  -> Ready / No Chase / Invalidated
  -> Alerted
  -> Focused / Watched / Passed / Missed / Entered
  -> Managed
  -> Closed
  -> Reviewed
```

### 11.2 Required links

- candidate, opportunity, lifecycle, attempt, transition, alert, review, trade, and outcome IDs;
- setup/thesis/rule version;
- source snapshot and as-of time;
- planned entry, stop, target, obstacle, and risk;
- actual fills and exits;
- remaining R at notification and at entry;
- MFE, MAE, realized R, and time in trade;
- automatic and user market environment;
- Focus origin and state;
- chart/screenshot references;
- structured decision and review reasons.

### 11.3 End-of-day review inbox

Automatically assemble:

- broker trades without an opportunity link;
- Ready alerts without a user resolution;
- watched conditions that fired;
- Focus names that became Ready but were not reviewed;
- passed opportunities with matured outcomes;
- trades missing setup tags, planned risk, or review.

The trader should resolve these with a short, structured workflow rather than reconstructing the day manually.

### 11.4 Separate grades

Each reviewed trade should have independent grades for:

- setup validity;
- entry quality;
- size/risk discipline;
- stop discipline;
- exit discipline;
- process adherence;
- outcome.

A profitable rule break is not a good-process trade. A valid setup that stops out is not an invalid setup.

## 12. Universe discovery and replenishment

The Universe page should become a candidate intake funnel, not only a raw list builder.

### 12.1 Suggested lane

Show newly discovered candidates with:

- first seen;
- source and source priority;
- why the candidate is new;
- side, horizon, setup/thesis when known;
- current stage;
- lease/expiry;
- objective quality and data health;
- sector/industry concentration;
- last material change;
- promotion or drop reason.

### 12.2 Actions

- accept into Swing Focus;
- accept into M5 Focus;
- chart;
- watch;
- pass with reason;
- leave as universe-only;
- compare with a similar current Focus name.

Suggested names must not silently become human Focus picks. Automated registry leases may expire, but user-entered names remain protected.

### 12.3 Discovery measurement

Measure:

- new candidates per session;
- percent reaching Testing, Confirming, and Ready;
- time from discovery to Ready;
- Ready precision and remaining R;
- percent promoted to Focus;
- user-versus-model selection lift;
- missed winners among rejected/control candidates;
- incremental value after sector and setup correlation.

Do not reward a discovery source simply for generating more candidates.

## 13. Canonical data contracts

Use two publication patterns deliberately:

- event and lifecycle records are append-only, idempotent ledgers;
- GUI read models are immutable snapshots published atomically with a manifest, checksum, and last-known-good fallback.

The GUI consumes these contracts; it does not infer a second source of truth.

### 13.1 Identity graph and cardinality

| Entity | Meaning and parent relationship |
|---|---|
| `candidate_id` | One registry candidate for symbol + side with provenance and source memberships |
| `membership_id` | One candidate membership with source, horizon/scope, owner, lease, and user-protection state; Swing and M5 memberships remain distinct |
| `setup_id` | A versioned setup definition, family, anchor, detector/configuration, and rule version |
| `opportunity_id` | One candidate × horizon × setup × anchor × thesis version |
| `lifecycle_id` | The durable lifecycle for that opportunity; a material thesis revision creates a linked successor instead of overwriting history |
| `attempt_id` | One try within a lifecycle; failure followed by re-arm creates a new attempt |
| `trigger_id` | One completed-bar trigger evaluation/event within an attempt |
| `alert_event_id` | One typed material transition worth surfacing; retries do not create another alert event |
| `delivery_id` | One delivery of an alert event to a surface/channel; one alert may have several deliveries |
| `ranking_snapshot_id` / `eligibility_id` | The exact ranked cohort and one opportunity’s eligibility record at one decision time |
| `impression_id` / `action_id` | A visible presentation and each explicit trader action resolving or modifying it |
| `watch_id` | One explicit arm; disarm/re-arm produces a linked new watch, never a silent overwrite |
| `trade_id` | One actual trade, potentially containing multiple fills, linked only after deterministic or reviewed reconciliation |
| `outcome_id` | One opportunity/attempt × `outcome_definition_id`; rescans and scenario variants do not create extra independent samples |

Identity rules:

- A rescan or GUI refresh updates a snapshot; it does not create a candidate, opportunity, attempt, impression, or outcome.
- Adding/removing one membership affects only its exact source/horizon scope; it cannot erase another membership for the same candidate.
- Changing side, horizon, setup family, anchor, or a material thesis/plan creates a new opportunity version with `parent_opportunity_id`.
- Rearming after failure creates a new `attempt_id` under the same lifecycle.
- Re-rendering or retrying a message may create a new `delivery_id`, but not a new `alert_event_id`.
- A trade may reference one primary opportunity/attempt after reconciliation; ambiguity goes to an inbox and is never guessed.
- Every child record carries its parent IDs, schema version, source version, `machine_id`, `as_of`, timezone, and idempotency key.

### 13.2 Opportunity snapshot

Required fields:

- `candidate_id`
- `setup_id`
- `opportunity_id`
- `lifecycle_id`
- `attempt_id`
- `snapshot_id` and `ranking_snapshot_id`
- symbol, side, horizon, setup family, anchor, thesis version
- discovery sources and memberships
- lifecycle stage and transition history
- hard-gate results and blocker codes
- Objective Quality components
- Actionability Now components
- Personal Fit components and evidence maturity
- entry, invalidation, obstacle, target, Expected R
- market/sector/industry/stock evidence
- freshness, coverage, provenance, snapshot ID, complete-through time
- Focus, armed-watch, alert-profile, and journal-link state

Use the canonical Opportunity model required by `plan.md`; do not create a GUI-only competing identity.

### 13.3 Review event v2

Required fields:

- `impression_id`
- `review_event_id`
- `action_id`
- `opportunity_id`, `lifecycle_id`, `attempt_id`, `trigger_id`, `alert_event_id`
- `ranking_snapshot_id` and `eligibility_id`
- `surface`, queue eligibility, position, and reason ranked
- shown time and dwell
- action class: Track, Watch, Plan, Enter, Pass, Later, Remove, Restore
- structured reason class and optional text
- old/new state for toggles, `supersedes_action_id` for undo/restore, and an explicit final-resolution event
- objective/actionability/personal components at decision time
- preference-policy version, delivery-policy version, and evidence snapshot ID
- data health and as-of time

Retain v1 reads during migration and dual-write v1/v2 until parity is proven. The two versions must use separate artifacts or one idempotent event envelope with versioned projections. A reader must consume exactly one projection, never count both as two decisions.

### 13.4 Outcome definition and envelope

Keep two outcome types:

- **Standardized opportunity outcome** — what the setup did under a versioned hypothetical entry/stop/exit policy.
- **Actual execution outcome** — what the trader entered, sized, managed, and exited.

Every `outcome_definition_id` must freeze:

- side, horizon, decision time, completed-bar ID, timezone, exchange session, and observation window;
- point-in-time planned entry, invalidation, stop, targets, obstacles, and plan version;
- trigger rule, fill assumption, gap behavior, costs, and slippage;
- same-bar stop/target ordering rule;
- management/exit rule and expiry;
- explicit `NO_TRIGGER`, `MATURED`, `CENSORED`, `MISSING`, and `INVALID_DATA` states;
- MFE/MAE calculation and the independent sampling unit.

Choose one primary standardized Swing scenario before evaluation. Alternative holding periods or target/stop variants are correlated diagnostics, not independent observations. Never substitute actual P&L for setup quality or hypothetical R for execution quality.

### 13.5 Alert transition

Required fields:

- typed transition identity;
- opportunity and attempt identity;
- from/to stage;
- completed bar ID;
- severity lane;
- delivery reason;
- dedupe key;
- objective/actionability/personal components;
- sound decision and rule that caused it;
- source snapshot and freshness.

### 13.6 Separate preference and delivery policies

The **preference policy** may annotate and reorder only within a predeclared comparable lifecycle stage and Objective Quality band. It may not change:

- eligibility or hard gates;
- lifecycle stage;
- Objective Quality;
- alert severity or sound;
- notification budgets;
- scanning/monitoring cadence;
- suppression, arming, Focus membership, or execution.

The **delivery policy** controls queue lane, sound, digest routing, cooldown, and burst handling. It is a separate champion/challenger artifact with its own replay, live-shadow, canary, approval, expiry, and rollback record. Preference evidence may be an inspected input to a proposal, but it may not silently mutate delivery.

Both policy documents need:

- policy ID and version;
- policy type;
- author/reviewer;
- created, effective, and expiry times;
- cited evidence snapshot IDs;
- sample/session floors;
- aggregate priority cap;
- affected stages/horizons;
- expected queue/loud-alert impact;
- shadow comparison;
- approval status;
- prior policy and rollback target.

No suppression, auto-arm, auto-remove, or auto-execute field should be added.

### 13.7 Storage, publication, and ownership

Do not expand the shared Google Drive directory into a multi-machine live database before the supervised storage migration in `plan.md` Milestone 2.

Target operating model:

- in the current deployment mode, the primary desktop is the sole authoritative live writer while the mini-PC remains Auto Pilot OFF;
- every mutable stream has one named owner, `machine_id`, writer lease, and job-ledger record;
- authoritative mutable state lives in supervised local storage;
- shared-drive publication is an immutable, versioned export/snapshot for readers and recovery, not concurrent append from two machines;
- ledgers use stable event IDs and idempotency keys so retries are harmless;
- snapshots publish by write-new, validate, checksum, atomic replace, and retain-last-good;
- failed publication never destroys the prior verified snapshot;
- v1/v2 dual writes use separate files or a single canonical envelope so readers cannot double-count;
- every GUI surface reports source snapshot, owner, complete-through time, and health.

This is not a permanent desktop-only assumption. Any future mini-PC participation requires an explicit mutually exclusive lease/handoff, stale-owner fencing, and the same crash/restart/recovery drills before it may write.

## 14. Safe data-to-alert ranking

The alert presentation pipeline should be:

1. Data health and hard risk gates.
2. Lifecycle stage.
3. Objective setup quality.
4. Remaining reward, actionability, and anti-chase.
5. Evidence confidence.
6. Concentration/correlation handling.
7. Personal Fit tie-break within a comparable quality/stage band.
8. Delivery lane and sound profile.

Steps 1–6 produce canonical opportunity rank. Step 7 modifies only the trader-facing attention order inside a comparable cohort. Step 8 is a separate delivery decision and must not feed back into eligibility, stage, Objective Quality, or canonical rank.

The current guidance formula may remain an interim advisory annotation, but it should not become the canonical rank. Until the Phase 3 identity/parity gate and later ranking promotion pass, it must not reorder the active review queue. Its component contributions must be visible, correlated rules must have an aggregate cap, and every proposed change must be shadow-compared.

Personal Fit may reorder only opportunities inside an explicit comparable-stage and quality-band cohort. It cannot move an item across a hard gate, upgrade Developing to Ready, alter monitoring cadence, or determine sound.

## 15. Ordered implementation plan

This order is intentionally constrained by `plan.md` Section 12.

### Phase 0 — Capture readiness and baseline

**Goal:** Observe the current champion accurately and prevent exploratory learning from changing it.

Tasks:

1. Verify and record the full-suite and smoke-check baseline.
2. Restart the GUI onto a build containing `c45d965` or later.
3. Verify creation, schema, growth, recovery, and backup of `alert_review_events.jsonl`.
4. Show review-log, scoreboard, outcome-join, policy, writer, and snapshot health in System Health.
5. Capture current screenshots, queue ordering, alert and sound counts, Focus size, duplicate rate, missed-winner review, latency, and decision time.
6. After saving a fixed characterization replay of current queue behavior, force preference guidance to annotation-only/FIFO compatibility until the later policy gates pass.
7. Snapshot the active setup-scoring configuration/hash and characterize exactly when the existing `apply_changes=True` tuner runs. Do not invoke, redirect, or promote a new tuner result before Phase 1 golden characterization.
8. Label all pre-v2 evidence Exploratory / Non-Promotable.

Exit gate:

- event capture is durable and observable;
- failures are visible rather than silently swallowed;
- current ranking, detector, watch, Focus, and delivery champions are reproducible;
- no ranking or detector tuning is inferred from the cold-start sample.

### Phase 1 — Runtime foundation, fixtures, and a read-only Focus quick win

**Goal:** Complete the roadmap’s immediate reliability work and improve scanning ergonomics without adding a mutable authority.

Tasks:

1. Run the live-session checklist, two-machine drill, failure drills, shutdown drill, and publication/readback drill from `plan.md`.
2. Complete the Health page and daily audit package.
3. Add golden/benchmark fixtures for current detector, scoring, queue, alert, and sound behavior, including:
   - same symbol in Swing and M5;
   - same symbol on opposite sides;
   - two setup families/anchors;
   - failure and re-arm attempts;
   - multiple alerts in one session;
   - setup-table plus Alert Center decisions.
4. After scoring behavior is characterized, route future tuner proposals to a shadow artifact behind a tested compatibility/rollback switch and prove the frozen champion score output is unchanged.
5. Add a transitional v2 sidecar/envelope for impressions and collision diagnostics only if it is machine-local, diagnostic-only, single-writer, and excluded from shared-drive synchronization and production readers. Otherwise defer it to Phase 2.
6. Transitional keys are explicitly non-authoritative and may not be used to rank, merge, or mutate production state.
7. Build a presentation-only Focus table and opportunity dossier from current cached sources.
8. Display missing stage, thesis, entry, invalidation, Expected R, or Personal Fit as **Unknown**. Do not infer Ready or fill gaps in the GUI.
9. Leave existing Focus actions routed to their current owners; add no new writer in this phase.

Exit gate:

- baseline golden results are locked before behavior changes;
- failure and ownership drills pass;
- the provisional Focus view can help find a name without changing Focus, scoring, scans, alerts, or sound;
- transitional dual writes cannot be double-counted and are clearly labeled non-authoritative.

### Phase 2 — Supervised authority and point-in-time prerequisites

**Goal:** Complete the roadmap work that must precede authoritative Opportunity, learning, or alert mutations.

Tasks:

1. Perform the supervised storage/secrets migration with an explicit owner matrix, machine IDs, writer leases, recovery drill, and immutable shared exports.
2. Introduce the provider repository only behind golden parity tests.
3. Repair tracker identity, moving-level look-ahead, same-day contamination, adaptive backfill, score ordering, factor horizons, and survivorship with intentional-diff fixtures.
4. Freeze research configuration/train windows and preserve untouched validation/test windows, rejected/control candidates, no-trigger cases, missingness, and censoring.
5. Add typed membership identity/scope so Swing Focus, M5 Focus, automated sources, and user ownership cannot overwrite one another.
6. Complete CandidateRegistry live-writer adoption through dual-write parity and rollback rehearsal.
7. Integrate canonical SPY/RS evidence only at the roadmap-approved advisory/shadow level.
8. Decouple Greatness into its dedicated priority lane while legacy D1 wick alerts remain champion.
9. Prove completed-bar, timezone, stale-data, and source-alignment invariants end to end.

Exit gate:

- under the current deployment mode, the desktop is the supervised single live writer and the mini-PC cannot contend for authority;
- one authoritative candidate lifecycle exists and manual names remain protected;
- historical research is point-in-time reproducible;
- all provider and writer cutovers have parity evidence and a tested rollback;
- SPY state and Greatness still have zero unpromoted production contribution.

### Phase 3 — Stable identity, event semantics, and Focus Workbench v1

**Goal:** Make Swing, M5, side, setup, attempt, review, watch, and outcome attribution correct, then enable scoped Focus operations.

Tasks:

1. Implement the Section 13 identity graph in the domain/event layer, not as GUI-generated ticker fingerprints.
2. Propagate IDs through setup rows, alerts, Focus memberships, watches, review events, journal events, trades, and outcomes.
3. Change queue, dwell, cache, and episode keys from symbol to opportunity/attempt identity.
4. Split Track, Watch, Plan, Enter, Pass, Later, Remove, and Restore semantics.
5. Correct Swing versus M5 outcome definitions and add no-trigger/maturity states.
6. Add impression, eligible-not-shown, dwell, final-resolution, and reason capture.
7. Dual-write v1/v2 in separate projections, publish a parity report, and switch each reader independently.
8. Upgrade the Focus table to scoped registry-backed mutations, explicit confirmation, and undo.
9. Add deterministic Focus lanes, sortable compact columns, saved views, chart click, and opportunity dossier.
10. Surface the champion sound state and “Why this sounded”; introduce no new selective sound/delivery mode.
11. Split Clear History from Disarm Watches.

Before the canonical Opportunity snapshot lands, any unavailable lifecycle stage, Expected R, or Personal Fit remains **Unknown**.

Exit gate:

- no cross-horizon, cross-side, cross-family, cross-attempt, or cross-surface collisions;
- repeated scans do not create independent samples;
- every review has a resolvable impression and exact action meaning;
- v2 parity passes before its promotion-evidence clock starts;
- no automated process removes a manual name or changes detector/scanner scoring.

### Phase 4 — Learning Center v1

**Goal:** Make evidence understandable and auditable without letting the page tune production.

Tasks:

1. Add Objective, Preference, Execution, and Alert Utility views.
2. Separate Swing and Day Trade outcome definitions.
3. Add sample/session/freshness/confidence badges.
4. Add blind spot, leak, watch conversion, human-vs-bot, and data-health dashboards.
5. Make “What’s Working” claims clickable to cohorts and matching current candidates.
6. Add active/draft policy inspection and rank-impact preview.
7. Show frozen configuration/train/validation/test windows, multiple-testing controls, rejected/control cohorts, no-trigger rate, missingness, and censoring.
8. Keep policy authoring AI-reviewed, user-approved, and proposal-only.

Affected historical tracker outputs remain labeled **RESEARCH / NOT POINT-IN-TIME VALIDATED** until their exact point-in-time, identity, and outcome gates pass. The Learning Center may expose them; it may not silently turn them into production ranking or delivery evidence.

Exit gate:

- no evidence claim is shown without its horizon, n, sessions, freshness, and source;
- hypothetical and actual outcomes are visually distinct;
- the page has no direct live detector, ranking, Focus, watch, or delivery mutation control;
- every promotable cohort traces to an immutable post-v2 evidence snapshot.

### Phase 5 — Canonical Opportunity and ranking challenger

**Goal:** Build one opportunity truth and prove it in replay/live shadow before it affects production.

Tasks:

1. Build the versioned canonical Opportunity snapshot from authoritative registry, setup, market, and alert inputs.
2. Implement hard gates, Objective Quality, Actionability Now, Expected R, anti-chase, evidence confidence, and concentration controls as independently inspectable components.
3. Generate immutable eligibility and ranking snapshots in replay shadow, then live shadow.
4. Implement and evaluate the full Greatness/readiness gate stack in pure tests, replay shadow, and live shadow. Any actual Greatness promotion is a separate `plan.md` Section 7 evidence-and-approval event; unpromoted SPY state, Greatness, Technical Integrity, and advisory RS fields remain informational.
5. Prove by fixture and manifest that every unpromoted input has exactly zero contribution to production eligibility, score, stage, order, sound, and delivery.
6. Add a single champion/challenger selector that restores the prior champion snapshot without a code revert.
7. Compare candidate coverage, stages, ranks, Ready precision, remaining R, misses, latency, and zero-trade behavior.

Exit gate:

- canonical snapshots are deterministic, point-in-time, immutable, and reconstructable;
- replay and live-shadow disagreements are explained;
- the rollback switch is exercised;
- no challenger output has production effect.

### Phase 6 — Advisory Command Center and Review Queue v2

**Goal:** Let the trader inspect the challenger and review faster while champion routing remains intact.

Tasks:

1. Render the three Command Center groups and honest Ready count from the challenger snapshot, clearly labeled Advisory.
2. Add compact opportunity cards, deterministic sorting, explanation drawer, freshness, and data warnings.
3. Add the queue drawer, previous/next/undo, compact primary actions, structured Pass/Later reasons, and scoped overflow removals.
4. Show champion and challenger order/delivery side by side; keep active order FIFO/champion compatible.
5. Show Objective Quality, Actionability Now, and Personal Fit contributions separately.
6. Build the typed alert ladder and separate delivery-policy challenger in advisory/shadow mode; do not change active routing or sound.
7. Manually audit representative wins, losses, disagreements, stale states, no-trade sessions, both sides, and each horizon.

Exit gate:

- the trader can find the Focus name or qualified opportunity needing action in seconds;
- every card explains why, why not yet, what next, and what invalidates it;
- active champion eligibility, order, and delivery remain unchanged;
- weak Personal Fit cannot upgrade stage or leapfrog a stronger quality band.

### Mandatory quantitative canary contracts

Ranking projection and alert delivery require separate versioned acceptance manifests. They may share calendar sessions, but Ready outcomes, fired watches, and loud deliveries cannot be pooled into one evidence count. These initial minimums may be changed only before an evidence window begins and never after results are inspected. Any higher engine-specific floor in `plan.md` still applies.

**Ranking manifest**

| Gate | Initial requirement |
|---|---|
| Shadow coverage | At least 15 substantially complete sessions, 50 independent ranking-eligible opportunity-attempts, and 20 independent matured canonical Ready attempts |
| Regime/scope | Three materially different regimes plus open/midday/late coverage; both sides for side-symmetric behavior or promotion limited to the validated side |
| Identity/data | 100% valid parent IDs for displayed events, zero malformed promoted rows, and at least 95% outcome joins among matured attempts |
| Primary benefit | Predeclare one: at least +5 points in Ready precision or precision@3, or at least +0.10R median remaining Expected R, versus champion under the same snapshots/outcome definition |
| Quality/capture | Ready precision and predeclared precision@K no more than five percentage points below champion; no more than one additional missed Ready attempt and no more than a five-point Ready-capture decline |
| Remaining opportunity | Median remaining Expected R no worse than champion by more than 0.10R under the same outcome definition |
| Latency | Publish-to-visible p95 at or below two seconds and no more than 250 ms slower than champion on the primary desktop |
| Cross-surface parity | 100% agreement on snapshot ID, opportunity ID, stage, canonical rank, and complete-through time |
| Canary window | At least five substantially complete sessions after the ranking shadow gate, with daily manual review |

Report confidence intervals and the exact numerator/denominator for every row. If the sample cannot distinguish the proposed behavior from the allowed degradation, continue shadow collection; do not treat “not significant” as proof of equivalence.

**Delivery manifest**

| Gate | Initial requirement |
|---|---|
| Shadow coverage | At least 15 substantially complete sessions, 10 distinct fired user-armed watches, and 30 champion loud-alert events |
| Armed protection | 100% fired-watch delivery, zero missed/misdirected armed hits, and no delivery later than the declared champion latency tolerance |
| Anti-spam | At least 30% fewer duplicate/non-material loud deliveries when the champion baseline has at least 20; otherwise zero duplicates and no loud-count increase |
| Ready visibility | No more than one additional missed Ready attempt and no more than a five-point Ready-capture decline |
| Retention/cadence | 100% typed events retained in History; detector evaluation and freshness cadence exactly match champion |
| Latency | Alert-event-to-visible p95 at or below two seconds and no more than 250 ms slower than champion |
| Canary window | At least five substantially complete sessions after the delivery shadow gate, with daily manual review |
| Rollback | One tested switch restores champion delivery without code revert, lost watches, or lost History |

Immediate rollback triggers:

- any missed or misdirected user-armed hit;
- a side/horizon/thesis identity collision;
- production influence from an unpromoted field;
- loss of History, Focus membership, a watch, or the last verified snapshot;
- cross-surface parity below 100%;
- any ranking, Ready-capture, sound, retention, cadence, or latency measure outside its applicable manifest;
- a second live writer or unresolved writer-lease conflict.

### Phase 7 — Canonical production projection and Auto/Away parity

**Goal:** Promote only the proven canonical snapshot projection and make every surface consume it.

Tasks:

1. Freeze the ranking manifest before the outcome window.
2. Canary the canonical projection on a bounded GUI surface while retaining one-switch champion rollback.
3. Promote canonical identity/snapshot projection only after parity gates pass; promote the challenger rank/order only if its separate primary-benefit and non-inferiority rows also pass.
4. Only after the bounded GUI canary passes the applicable ranking manifest, project the same saved snapshot into Desk, Focus, Alert Center, Auto/Away, reports, and AI summaries.
5. Let Alert Center consume the promoted opportunity snapshot while its champion delivery order, lane, cooldown, and sound remain unchanged until Phase 8.
6. Make every surface display snapshot ID, source health, complete-through time, and challenger/champion status.
7. Run parity, freshness, restart, Away publish, failure, and rollback drills.
8. If the challenger-rank benefit is not proven, keep champion order and continue shadow collection even if the canonical snapshot itself is promoted.

Exit gate:

- all surfaces consume one verified canonical snapshot;
- canonical projection parity and its applicable canary gates pass with no immediate rollback trigger;
- challenger rank/order is active only if its primary-benefit and non-inferiority rows pass;
- zero-trade states remain honest;
- unpromoted detector fields remain advisory only.

### Phase 8 — Alert ladder and anti-spam delivery canary

**Goal:** Reduce interruptions while preserving the best opportunity capture.

Tasks:

1. Use the prebuilt delivery-policy challenger, kept separate from preference ranking.
2. Add typed Immediate, Heads-Up, Focus Changes, Developing, Research, and History lanes.
3. Deduplicate by alert event, opportunity, attempt, transition type, and completed bar.
4. Add post-gate per-opportunity sound profiles and material-transition cooldowns.
5. Add burst grouping and notification-budget routing without dropping History.
6. Shadow-log champion versus challenger delivery, sound, duplicates, latency, Ready/armed capture, queued winners, and misses.
7. Freeze and pass the delivery manifest, then canary behind one delivery-policy switch.
8. Keep Personal Fit unable to change severity, sound, budget, eligibility, or hard gates.

Exit gate:

- duplicate and non-material loud alerts meet the predeclared reduction target;
- Ready/armed capture and latency pass the manifest;
- every hidden-from-interruption event remains in board/history;
- every material Focus transition remains in Focus Changes/History;
- the delivery champion can be restored immediately.

### Phase 9 — Full lifecycle and journal linkage

**Goal:** Reconstruct every important opportunity and separate selection from execution.

Tasks:

1. Write discovery, stage, alert, review, Focus, watch, entry, management, close, and review events to the lifecycle ledger.
2. Link broker fills by deterministic match plus a reconciliation inbox for ambiguity.
3. Compute standardized and actual outcomes separately.
4. Add planned versus actual risk, remaining R, MFE/MAE, realized R, and adherence.
5. Build after-close review and missing-link queues.
6. Export reproducible daily and weekly learning packages.

Exit gate:

- every high-priority recommendation can be reconstructed;
- passed and missed opportunities are measured;
- actual fills are not silently attached to the wrong thesis.

### Phase 10 — Universe intake and controlled personalization

**Goal:** Let the platform replenish good ideas and adapt attention safely.

Tasks:

1. Build the Suggested candidate lane with provenance and lifecycle.
2. Measure discovery-source conversion and incremental value.
3. Let preference annotate or reorder only similarly qualified suggestions.
4. Have AI draft a versioned preference policy from cited evidence.
5. Preview affected opportunities, precision@K, cohort coverage, and missed winners.
6. If delivery changes are desired, draft a separate delivery-policy proposal and run its independent ladder.
7. Require user approval, shadow/canary, expiry, and rollback.
8. Promote setup families one at a time through the roadmap ladder.

Exit gate:

- new names can flow from Universe to Focus without spam;
- the bot learns attention preferences without changing objective truth;
- no policy can auto-suppress, auto-arm, auto-remove, reduce monitoring cadence, or execute.

## 16. Verification strategy

### 16.1 Identity and data tests

- Same symbol in Swing and M5 remains two independent focus/outcome episodes.
- Same symbol long and short remains distinct.
- Two setup families/anchors remain distinct.
- Re-arm creates a new attempt under the same lifecycle.
- A material thesis revision creates a linked successor opportunity.
- Repeated delivery creates a delivery record, not a new alert event or outcome.
- Repeated scans update one opportunity rather than creating new samples.
- Setup and Alert Center reviews cannot overwrite each other.
- Every action and outcome references a valid identity and version.
- v1/v2 dual writes cannot be counted twice.
- Only one machine/writer can own a mutable stream.
- Snapshot failure preserves the last verified snapshot.

### 16.2 GUI tests

- Focus table sorting is numeric and stable.
- Saved filters persist presentation only.
- Focus rows open the correct chart/thesis.
- Queue previous/next/undo works.
- Clear History does not disarm watches.
- Bulk removal requires confirmation.
- Every material Focus transition remains in Focus Changes and History.
- Honest empty lanes remain empty.
- Fresh/Preview/Stale/Partial/Failed are not color-only.
- Keyboard actions do not fire while typing into a text field.
- Plan Entry and Entered produce different events and labels.
- Navigating without a verdict records `SKIP_UNRESOLVED`.

### 16.3 Alert golden tests

- One typed transition produces one alert.
- A repeated scan produces no duplicate.
- An escalation produces a new alert.
- An armed hit remains first and loud.
- A Developing event remains board-only.
- A Ready event is loud only when completed-bar and actionable.
- A Focus invalidation surfaces clearly.
- A quiet policy never deletes History.
- Preference policy cannot alter sound, severity, budget, or eligibility.
- An unpromoted shadow field has zero production contribution.
- Champion delivery is restored by one switch without a code revert.

### 16.4 Research tests

- Point-in-time bars, levels, membership, regime, and feature versions.
- Independent episode counts rather than repeated scans.
- Standardized versus actual outcome separation.
- Costs and missing data handled explicitly.
- Same-bar stop/target ordering and no-trigger/expiry states are deterministic.
- Train/validation/test splits are immutable.
- Multiple-testing adjustments and rejected/control cohorts are retained.
- Every policy cites the exact evidence snapshot.
- Correlated policy contributions are capped.

### 16.5 Repository gates

Before each implementation commit:

```powershell
.venv\Scripts\python.exe -m pytest tests/ -q
.venv\Scripts\python.exe scripts/smoke_check.py
```

Add offscreen Qt tests, golden alert-routing fixtures, identity-collision fixtures, and live-session validation where the change affects runtime behavior.

Implement from `Sol3` or a branch from it, keep commits small and green, and push each verified checkpoint. Merge to `main` only after the required live-session validation day passes; never leave the user’s runnable tree between schema writers/readers.

## 17. Success metrics

Metrics used for promotion must share a predeclared `outcome_definition_id`, horizon, cohort rule, and ranking-snapshot schedule. Do not compare numbers produced by different holding periods or fill assumptions as though they were the same metric.

### Promotion metric definitions

| Metric | Definition |
|---|---|
| **Ready precision** | Among independent, matured attempts that entered canonical Ready, the fraction whose primary standardized scenario reached its success condition before invalidation/stop/expiry |
| **Ready capture** | Canonical Ready attempts shown on the intended active surface before their action window expired ÷ all canonical Ready attempts |
| **Precision@K** | At each predeclared ranking snapshot, the fraction of the top K distinct matured opportunities that succeeded under the primary standardized outcome; repeated refreshes are not new samples |
| **Remaining Expected R** | Side-adjusted distance from point-in-time alert price to the versioned primary target ÷ distance to the point-in-time invalidation/stop; invalid or missing denominators are reported, not imputed |
| **Missed winner rate** | Eligible matured opportunities not shown before their action window that later met the primary success condition ÷ eligible matured opportunities not shown |
| **Duplicate loud rate** | Loud deliveries after the first for the same typed `alert_event_id` without a genuine escalation ÷ all loud deliveries |
| **User-armed hit delivery** | Distinct fired `watch_id` events visibly delivered within the declared latency bound ÷ all distinct fired watches |
| **Outcome join rate** | Matured independent opportunity-attempts with a valid versioned outcome ÷ all matured independent opportunity-attempts |

### Attention and usability

- Time from opening the app to identifying the best qualified opportunity.
- Time to review the top five or ten charts.
- Percent of Focus names with a visible next condition.
- Percent of actionable cards with complete freshness and provenance.
- Number of clicks needed to chart, Focus, arm, pass, and journal a name.

### Alert quality

- Loud alerts per session.
- Duplicate alert rate.
- Ready precision, precision@1, and precision@3.
- Remaining Expected R at alert.
- Alert-to-action conversion by action type.
- Missed-winner rate among quiet/queued items.
- User-armed hit delivery rate and latency.

### Learning quality

- Independent reviewed episodes.
- Resolved impression rate.
- Outcome join rate.
- Swing and M5 attribution completeness.
- Blind spot/leak sample maturity.
- Policy lift versus champion in shadow.
- Policy expiry and rollback success.

### Selection and execution

- Universe → Ready conversion.
- Focus → Ready conversion.
- Human selection lift after controlling for setup, side, and regime.
- Standardized opportunity R versus actual execution R.
- Planned-risk coverage.
- MFE/MAE and realized-R coverage.
- Process-adherence rate.

The platform should optimize qualified opportunity capture and decision quality, not maximize trade count or take rate.

## 18. Priority backlog

This backlog does not override the Phase 0–10 gates. Presentation-only work may proceed in parallel; reader cutovers, new writers, and policy influence may not move ahead of their prerequisites.

### P0 — Must precede stronger learning

1. Verify the baseline, review-event capture, and annotation-only/FIFO compatibility.
2. Add writer, review-data, outcome, snapshot, and policy health visibility.
3. Complete live/two-machine/failure drills and golden/replay fixtures.
4. Complete supervised storage ownership and provider parity.
5. Repair point-in-time research defects and freeze evaluation windows.
6. Complete CandidateRegistry writer adoption and rollback.
7. Freeze new learned/tuner mutations into proposal/shadow paths.

### P1 — Largest trader benefit after its gate

8. Presentation-only Focus table and opportunity dossier with honest Unknown fields.
9. Stable cross-horizon/cross-side/cross-thesis identity and v2 parity.
10. Split Track, Watch, Plan, Enter, Pass, Later, Remove, and Restore semantics.
11. Correct M5 day-trade outcomes and setup-chart impression denominators.
12. Registry-backed Focus lanes, scoped actions, confirmation, and undo.
13. Queue drawer, why-ranked explanation, previous/undo, and decision reasons.
14. Split Clear History from Disarm Watches.
15. Learning Center data-health and “What’s Working” drill-down, research-only.

### P2 — Requires canonical foundation

16. Canonical Opportunity snapshot and ranking challenger.
17. Complete Greatness/readiness gates without bypassing promotion floors.
18. Advisory Today Command Center.
19. Quantitative ranking canary and Desk/Focus/Alert/Away parity.
20. Separate alert-delivery challenger, burst grouping, and sound profiles.
21. Full lifecycle/journal linkage.

### P3 — Evidence-controlled adaptation

22. Universe Suggested lane and discovery funnel.
23. Separate preference/delivery policy preview and audit history.
24. Shadow comparison and limited delivery-policy canary.
25. Setup-family promotion one at a time.

## 19. Explicit non-goals

- Order placement or automated execution.
- AI-created suppression rules.
- Automatic arming of watches or levels.
- Automatic removal of user-entered names.
- Automation reducing detector evaluation or Focus monitoring cadence.
- Concurrent mutation of shared-drive ledgers by multiple machines.
- Personal preference changing objective expectancy.
- Personal preference overriding hard gates, stale data, invalidation, or no-chase.
- Automatic promotion of `market_state`, `greatness_monitor`, Technical Integrity, or advisory industry RS/RW.
- Changing the existing AVWAP sigma calculation.
- Using M5 Focus daily returns as day-trade expectancy.
- Treating a Day Trade Tracker segment or forensics correlation as a standalone setup.
- Filling Top 3 or Ready lanes with weaker names.
- Tuning and evaluating on the same outcomes.
- Creating independent calculations for Desk, Focus, Alerts, Auto/Away, or AI summaries.

## 20. Definition of done

This program is complete when:

1. A ticker can carry distinct Swing, M5, side, setup, and attempt identities without collision.
2. The trader can identify the best qualified opportunities from one screen.
3. Focus is a prioritized monitoring workbench, not a static chip wall.
4. Every top opportunity explains why, why not yet, what next, and what kills it.
5. Loud alerts are rare, typed, deduplicated, current, and actionable.
6. Every material Focus transition remains in Focus Changes/History without requiring every low-value scan event to sound or lead the active queue.
7. Every reviewed chart has a measurable impression and resolution.
8. Swing and day-trade outcomes use correct, separate definitions.
9. Objective quality, Actionability Now, and Personal Fit remain visibly separate.
10. Every evidence claim shows sample size, horizon, freshness, confidence, and provenance.
11. The journal links discovery, decisions, fills/no-trades, outcomes, and reviews.
12. Passed, missed, invalidated, and never-ready opportunities are retained for research.
13. Desktop, Focus, Alert Center, Auto/Away, and reports consume the same canonical snapshot.
14. AI policy changes are cited, versioned, previewed, approved, shadow-tested, reversible, and non-suppressing.
15. Manual names are never automatically removed.
16. Champion detectors and shadow promotion gates remain intact.
17. The full test suite and smoke checks are green.
18. The platform remains decision-support only.

## 21. Likely implementation map

Keep domain contracts and event ownership outside Qt panels. Panels should render cached read models and call narrow services.

| Responsibility | Likely existing touchpoints |
|---|---|
| Candidate ownership and identity | `scripts/candidate_registry.py`, `scripts/focus_picks.py`, `scripts/ui/services/focus_service.py` |
| Focus evidence and outcomes | `scripts/human_focus_tracking.py`, `scripts/ui/services/human_focus_tracker_feed.py` |
| Review events, learning, and guidance | `scripts/review_events.py`, `scripts/review_learning.py`, `scripts/review_guidance.py`, `scripts/review_policy.py` |
| Alert review state and UI | `scripts/alert_review_state.py`, `scripts/ui/panels/alert_center_panel.py` |
| Setup discovery and scoring | `scripts/master_avwap_lib/`, `scripts/ui/panels/master_avwap_panel.py`; isolate the tuner path in `scripts/master_avwap_lib/legacy.py` |
| Focus GUI | `scripts/ui/panels/focus_picks_panel.py` |
| Journal lifecycle and reconciliation | `scripts/journal_store.py`, `scripts/journal_importers.py`, `scripts/ui/panels/journal_panel.py` |
| Writer/storage reliability | `scripts/writer_lease.py`, `scripts/job_ledger.py`, `scripts/master_avwap_lib/storage.py` |
| Health and validation | `scripts/ui/panels/health_panel.py`, `scripts/smoke_check.py`, existing focused tests under `tests/` |

Prefer new pure modules for Opportunity identity, immutable snapshots, ranking explanations, delivery policy, and outcome definitions rather than adding more calculation logic directly to large panel files.

## Appendix A — Audit snapshot metadata

**Observed:** `2026-07-28T16:34:07-07:00`

**Shared root:** `C:\Users\aaron\My Drive\Trading\TradingBot`

| Relative path | Modified (PDT) | Bytes | SHA-256 |
|---|---:|---:|---|
| `pick_feedback.jsonl` | 2026-07-28 09:27:36 | 29,038 | `5AD5FC88DEAF45368DDB705803FFC0E99F74CD5CCD7A8E13E30FFD32563EBE1F` |
| `data\runtime\human_focus_daily_picks.csv` | 2026-07-28 09:27:35 | 19,260 | `2B8A2067A88CFD9F65CFBA5931E34B52458BD0FDA5B35763B90842B0FC23656E` |
| `data\runtime\human_focus_outcomes.csv` | 2026-07-28 13:08:14 | 48,761 | `EE70A3D1A2100FA65F784DC29C3846A0ACBD7B7394A58FAF3FADA35B9AB935EC` |
| `focus_swing_longs.txt` | 2026-07-28 09:27:35 | 96 | `4EA1F01CA95EC6EB09D62502D0CB4E66E024A9975AF640A42ABF8D6A47D4378B` |
| `focus_swing_shorts.txt` | 2026-07-28 08:50:38 | 20 | `00CE3EFC40BA222A3E7553676BD61DF6F96D839EC56BF74B3FA8439620DF9F03` |
| `focus_longs.txt` | 2026-07-28 09:18:15 | 110 | `7B4726891D5559DD63A80F9E315323A33F6C4B0F931C60360EE53E3CE6C95948` |
| `focus_shorts.txt` | 2026-07-28 08:21:14 | 27 | `308FDCADFDC00BC0EF192BB838814F52A16AAA286FF17EAC2770E698EDBFEFA2` |

At that snapshot, Focus contained 17 Swing longs, four Swing shorts, 20 M5 longs, and five M5 shorts; six long symbols were in both horizons. `pick_feedback.jsonl` contained 102 decisions: 101 likes and one dislike.

`alert_review_events.jsonl`, `review_preference_state.json`, `output\review_learning_report.txt`, `review_policy.json`, and `review_policy_draft.json` were not present at the observation time. Their absence was expected before restarting onto the review-learning build and must not be treated as a permanent state.
