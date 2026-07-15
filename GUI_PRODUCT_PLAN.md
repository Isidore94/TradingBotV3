# TradingBotV3 GUI, Decision Support, Journal, and AI Product Plan

Status: discussion draft

Audit date: 2026-07-14

Scope: product and implementation planning only; this document changes no runtime behavior.

## 1. Relationship to the master roadmap

This document is a product-focused addendum to plan.md. It does not replace plan.md, reorder its Section 12 execution list, or relax any evidence gate.

The governing rules remain:

- Legacy SPY pause detection and D1 wick alerts remain the champions.
- market_state and greatness_monitor remain shadow-only until the evidence and promotion requirements in plan.md pass.
- Detector, scoring, routing, or ranking behavior changes require golden-result fixtures first.
- The current calc_anchored_vwap_bands sigma formula remains unchanged.
- User-entered watchlists are never automatically removed.
- Completed bars drive state transitions; forming bars may only be labeled previews.
- TradingBotV3 remains decision-support software. It will not place orders.

The product objective is:

> Find the best supportable trades for today, explain them clearly, record what the user and bots did, and turn the resulting evidence into better research for tomorrow.

Production ranking must remain deterministic and testable. AI may explain, summarize, compare, and propose research questions; it must not silently change production rules, promote a shadow engine, invent a candidate, or execute a trade.

## 2. Executive product decisions

The recommended direction is an incremental consolidation into five user-facing areas:

    Today
      Command Center
      Master AVWAP opportunities
      Bounce Timing / RS-RW
      Industry Board
      Alert Center

    Research
      Setup Tracker
      Day Trade Tracker
      Move Forensics
      Playbooks and evidence

    Journal
      Decisions, executions, outcomes, screenshots, and review

    AI Summary
      User-selected evidence packages and provider-generated reports

    System
      Auto Pilot, Health, data freshness, provider connections, and Settings

This does not require a risky navigation rewrite at the start. Existing panels can first be made consistent, then moved behind the simpler information architecture after their data contracts are stable.

The two core loops are:

1. Today: observe -> rank -> explain -> alert -> decide.
2. Tomorrow: link -> record -> measure -> review -> propose a test.

The system must never confuse a research observation with an executable setup, an AI opinion with measured evidence, or a forming-bar preview with a completed-bar decision.

## 3. Current-state audit: preserve what works and fix the real gaps

| Area | What exists now | Actual gap | Product decision |
|---|---|---|---|
| Industry Board | A manual yfinance refresh, daily 5/20/65-session group rankings, and CSV-backed sector/industry context | No owned hourly schedule, no reliable freshness state, no numeric column sorting, and no intraday composite participation in RS/RW | Create one shared snapshot service, numeric sorting, hourly refresh, and completed-M5 composite RS/RW |
| RS Window | Automatic five-minute stock mover refresh and existing joins to Industry Board context | Industry values are context only; composite industries are not ranked through the same aligned intraday window | Add industry-vs-SPY and stock-vs-primary-industry calculations as explicit fields, initially advisory |
| Master AVWAP | Favorite and High Conviction collections and a focus export | The same opportunity can be emitted in both collections, while the UI identity includes the bucket and therefore preserves the duplicate | Render one opportunity once, with one primary lane and all applicable badges |
| Bounce market environment | An automatic SPY regime path and a user override path | The dropdown defaults to a concrete bullish state and mirrors the detected regime, so bot belief and user choice are conflated | Default to Auto source plus User override = N/A; display bot belief separately |
| Entry Assist | Automatic pause/window processing, a cached one-minute board refresh, and 30-minute mover tools | The main UI still presents manual window controls, hiding the fact that normal operation is automatic | Make automatic behavior the primary surface; retain strongest, weakest, and movers as useful on-demand views |
| D1 Focus | Final bucket-upgrade events are already restricted to Favorite/High Conviction targets | The panel also accepts an upgrade-trigger event that may represent a developing condition rather than a final target-bucket upgrade | Preserve the final gate; remove non-final trigger crossings from D1 Focus or place them in a quiet Developing research lane |
| Setup Tracker | A detail view for some current-pick rows plus a statistics-heavy summary | Not every row opens an explanation; terminology assumes an experienced user; the summary can elevate non-actionable research | Add deterministic novice explanations to every row and a plain-English evidence summary |
| Day Trade Tracker | Segment tables, R statistics, and proven/muted state | No row-linked explanation and a segment can look like a standalone trade | Explain it as a condition modifier and link it to a valid setup/trigger |
| Move Forensics | Pattern leaderboard and raw reports | No row-linked interpretation; correlations can be mistaken for executable strategies | Explain evidence, limitations, and the next validation step; only show execution rules when a documented setup exists |
| Journal | Local journal storage, trade import, notes/tags, and basic analytics | Weak linkage from discovery through decision, execution, MFE/MAE, and outcome | Add stable opportunity and event linkage, reason codes, screenshots, and review metrics |
| Auto/Away Drive report | A verified, hourly, phone-oriented autopilot_today.txt report | The first candidate sections are day-trade lists and bot picks; Top Swing Picks appears later | Keep the safety/freshness header first, then make swings the first and most prominent candidate section in every human-facing Auto/Away Drive report |
| AI | OpenAI-based Market Prep paths already exist | The implementations are duplicated, there is no provider-neutral report tab, no Anthropic path, and no common evidence manifest | Consolidate behind one provider interface and build AI Summary on normalized, user-selected evidence packages |
| Health | A System Health page and runtime diagnostics already exist | New industry and AI jobs would not yet expose freshness, ownership, or last-result state | Extend the existing page; do not build a second health surface |

## 4. Product principles

### 4.1 One fact, one owner

Each data product has one runtime owner and one canonical snapshot. GUI pages, Auto Pilot, Away mode, reports, and AI packages consume that same snapshot instead of launching independent scans.

### 4.2 One opportunity, one identity

A bucket, badge, alert type, or watchlist is a view of an opportunity, not its identity. A real second thesis, side, anchor, or setup may be a second opportunity; a second label is not.

### 4.3 Evidence before eloquence

Plain-English summaries are generated deterministically first. An AI rewrite may improve readability later, but it cannot change the underlying measurements and must cite the local evidence rows that support each conclusion.

### 4.4 Honest empty states

If nothing meets the quality gate, the correct answer is "No qualified setup yet." The GUI must not fill a Top 3, D1 Focus, or What's Working card with weaker names simply to avoid an empty panel.

### 4.5 Progressive disclosure

Explain Mode should be on by default. The first layer answers what, why, entry condition, invalidation, and evidence quality. Advanced calculations remain available without dominating the novice view.

## 5. Detailed feature plan

### 5.1 Industry Board: refresh, sorting, and RS/RW participation

#### User experience

The Industry Board should:

- Refresh once at startup when no valid snapshot exists or the last successful snapshot is more than 60 minutes old.
- Refresh every 60 minutes while the relevant scanning runtime is active.
- Offer a manual Refresh Now action that joins or coalesces with an in-flight refresh instead of starting a second job.
- Display Last attempted, Last successful, Data complete through, Snapshot ID, and Fresh/Stale/Failed status.
- Preserve the last good snapshot when the provider fails.
- Support numeric header sorting for every score, rank, return, member count, and freshness column.
- Default to blended RS strongest-first, with one-click Strongest and Weakest presets.
- Persist the user's sort column, direction, filters, and sector/industry view.
- Make pinned groups visible without silently defeating the requested numerical sort.
- Clearly distinguish a tradable sector ETF from an analytical industry composite.

The refresh cadence should be session-aware: premarket, regular session, and a final after-close snapshot by default. There is no benefit in repeatedly downloading unchanged daily data all night. A forced manual refresh remains available.

#### Runtime ownership

Create a shared IndustryBoardService rather than adding an isolated GUI timer:

- The desktop runtime owns the service while the desktop is the active scanner.
- Auto Pilot and the GUI subscribe to the same immutable IndustrySnapshot.
- The mini-PC must not become a second writer while the desktop scanner owns the job.
- A single-flight lock prevents overlapping scheduled and manual runs.
- Writes are atomic: build a complete temporary snapshot, validate it, then replace the prior good snapshot.
- Job start, completion, failure, duration, provider result, and output hash are recorded in the existing job ledger/run-manifest system.
- The GUI never performs provider I/O on the paint thread.

#### Calculation design

Keep four concepts explicit:

1. Market-relative: stock versus SPY over aligned completed bars.
2. Industry strength: the primary industry composite versus SPY.
3. Stock-within-industry: stock versus its primary industry composite.
4. Theme context: additional industry/theme memberships shown as context, not silently cherry-picked as the strongest membership.

The current daily 5/20/65-session structural board remains useful. Add a separate intraday series built from aligned, completed M5 bars for RS/RW timing:

- Composite each industry from eligible member returns using the documented robust aggregation method.
- Enforce minimum member coverage and expose coverage in the UI.
- Align the stock, SPY, and industry timestamps before calculating relative moves.
- Treat the forming M5 bar as Preview and exclude it from state transitions and final ranks.
- Show stock move, stock-vs-SPY excess, industry-vs-SPY excess, and stock-vs-industry excess.
- Rank industry leaders and laggards in the RS/RW board alongside stock movers.
- Use a declared primary industry for scoring. Show all additional memberships as labeled context.

Release this in two steps:

- Advisory release: calculate, display, log, and replay the new fields without changing live gates or production scores.
- Ranking release: allow the canonical ranking engine to consume the fields only after golden fixtures, replay comparisons, missing-data tests, and the plan.md promotion requirements pass.

#### Acceptance criteria

- Clicking a numeric column produces true numeric sorting, including negative values and missing values.
- A fake-clock test proves startup refresh, 60-minute scheduling, and single-flight behavior.
- A provider failure leaves the prior snapshot readable and visibly stale.
- Industry Board, RS Window, Auto Pilot, and reports show the same Snapshot ID.
- An aligned-bar fixture proves that forming bars cannot change the final industry state.
- Insufficient member coverage produces Unknown/Insufficient Coverage, not a confident zero.

### 5.2 Master AVWAP: remove duplicate opportunities

The immediate correction belongs in the focus read model and GUI, not in AVWAP math.

Define an opportunity identity from:

- symbol;
- side;
- setup family/thesis;
- anchor or structural reference where it distinguishes a genuinely separate trade;
- session or lifecycle identifier where necessary.

Then:

- Render the opportunity once.
- Select one primary lane using explicit precedence, normally High Conviction before Favorite before Study.
- Preserve secondary classifications as badges, for example High Conviction / Favorite.
- Count unique opportunities and unique symbols separately.
- Copy/export/alert by opportunity identity so the same idea cannot be sent twice.
- Preserve two rows when the same ticker truly has different sides, anchors, or setup theses.

The canonical Opportunity model in plan.md should eventually supply a stable opportunity_id. Until then, use a conservative legacy fingerprint and log collisions so the temporary key can be audited.

Golden fixtures must cover:

- one legacy payload present in both Favorite and High Conviction -> one displayed opportunity;
- same symbol and side but two valid setup families -> two opportunities;
- same symbol with long and short theses -> two opportunities;
- repeated refresh of the same lifecycle -> no duplicate alert or count inflation.

### 5.3 BounceBot: Auto is the default; User environment defaults to N/A

The GUI must separate two independent concepts:

- Global Auto Pilot mode: Off, Desk, or Away.
- Bounce market-environment source: Auto or User Override.

Recommended control:

    Bot sees: Bullish Strong     Source: AUTO
    User override: N/A - follow Auto

Behavior:

- A new trading session starts with User override = N/A and Source = AUTO.
- The detected environment is read-only and never copied into the user dropdown.
- Selecting Bullish Strong, Bullish Weak, Bearish Strong, Bearish Weak, or Chop immediately changes Source to USER and uses that environment.
- Selecting N/A immediately clears the override and resumes the automatic environment.
- An override may survive an application restart within the same market session, but expires to N/A at the next session boundary.
- Every select, change, and clear action produces an append-only annotation.

Each annotation should contain:

- event ID, timestamp, timezone, and market-session ID;
- old and new user selections;
- automatic environment at the moment of selection;
- effective source and effective environment;
- SPY data complete-through timestamp;
- the deterministic environment feature snapshot;
- optional reason chips such as trend, breadth, volatility, news, or other;
- optional free-text rationale;
- application, configuration, and detector versions;
- later-linked outcomes and the duration of user/auto disagreement.

The rationale prompt should be lightweight and optional so the mode change is not blocked. Capturing the surrounding deterministic context is mandatory. If persistence fails, the GUI must show an unsaved annotation warning and retry locally; it must never pretend the event was recorded.

Future AI may compare user and auto calls in an offline report. It may not train or update the live detector automatically. Any proposed rule change returns to fixtures, replay, shadow evidence, and explicit promotion.

Acceptance criteria:

- A clean-session test starts in Auto with User override N/A.
- The read-only Bot sees value can change without changing the dropdown.
- Selecting a user environment changes effective behavior and creates exactly one annotation.
- Returning to N/A restores Auto and creates a clear annotation.
- A next-session test expires the override without deleting its history.

### 5.4 Entry Assist: make the automatic behavior obvious

The existing automatic engine should become the normal product experience. Rename the visible feature to something clearer, such as Bounce Timing / RS-RW, while retaining legacy internal names until code migration is safe.

In Auto:

- Champion SPY pause logic automatically decides when a pullback/bounce observation window opens and closes.
- Candidate moves are measured automatically through the window.
- Rankings and the board refresh automatically.
- The user does not need to press Pullback Window or Bounce Window.
- No action places an order.

The primary status card should use plain states:

- Waiting: SPY has not produced a completed-bar pause condition.
- Tracking: a window is active; show start time, completed-through time, SPY move, and candidate coverage.
- Ranked: the completed window has results; show leaders, laggards, and why.
- Stale/Insufficient data: say exactly which feed or coverage is missing.

Keep the useful user-requested tools visible:

- Strongest over the last 30 minutes.
- Weakest over the last 30 minutes.
- Movers over the last 30 minutes.
- Refresh results.

Move manual pullback/bounce window controls into Advanced / Replay Diagnostics. They are useful for testing, but their placement should not imply that routine Auto operation needs manual intervention.

Automatic window results should live in the board. Only material typed transitions should enter the Alert Center, preventing repeated list dumps from overwhelming actionable alerts.

### 5.5 D1 Focus: only final best-of-the-best upgrades

The existing final bucket-upgrade gate is valuable and should be preserved.

The exact D1 Focus contract should be:

- A row enters only after the completed scan classifies an opportunity into Favorite or High Conviction and confirms that this is a real lifecycle upgrade.
- A trigger crossing, armed condition, study candidate, or "nearly qualified" row is not a D1 Focus item.
- A repeated scan of the same upgraded lifecycle does not create a second item.
- If the opportunity later loses qualification, history remains in the journal but the live lane reflects current state.
- The Master AVWAP dedupe rule applies before display and alerting.

Recommended treatment of existing upgrade-trigger events:

- Hide them from D1 Focus by default.
- If they retain research value, place them in a quiet Developing / One Step Away view with no implication that they are final selections.

This routing change requires a golden fixture because it changes what the user sees as actionable, even though it does not alter the underlying detector.

### 5.6 Row-click novice explanations

Use the existing explain-mode state and detail view as a foundation. Explain Mode should default on, and every meaningful row in Setup Tracker, Day Trade Tracker, and Move Forensics should open a common detail pane by mouse or keyboard.

Every explanation should answer:

1. What is this row?
2. Is it an executable setup, a context modifier, or a research correlation?
3. What market conditions does it apply to?
4. What must happen before an entry is valid?
5. What would a sample entry sequence look like?
6. Where is the idea wrong, and how is invalidation determined?
7. How would targets and management work?
8. What do R, win rate, expectancy, lift, sample count, and confidence mean here?
9. Why does the bot currently like, avoid, prove, mute, or merely study it?
10. How fresh is the evidence, and what source produced it?

Panel-specific truthfulness matters:

- Setup Tracker may show exact current levels only when they come from the current scan and are labeled with an as-of time.
- Day Trade Tracker rows are condition modifiers. They must be tied to an eligible setup/trigger and must never look like standalone buy/sell instructions.
- Move Forensics rows are observations until promoted through research validation. If no executable playbook exists, the explanation must explicitly say so and identify the next test rather than inventing an entry and stop.

The first implementation should be deterministic and template-driven. AI can later offer "Rewrite this more simply," but it must not be required for a row click and may not alter levels or evidence.

Acceptance criteria:

- Every data table exposes a selection-to-detail path.
- The same metric has the same novice definition everywhere.
- Missing execution rules are reported as missing, never synthesized.
- Exact levels carry source and freshness labels.
- Color is never the only carrier of strength, weakness, or risk.

### 5.7 Plain-English "What's Working Now"

Build this before depending on an external AI provider. It should be a deterministic projection of the tracker and journal evidence.

Recommended summary blocks:

- Best supported right now: the strongest measured setup/condition with horizon, side, regime, average R/expectancy, sample count, and freshness.
- Matching today: current qualified opportunities that match that evidence, if any.
- What is not working: muted or deteriorating patterns with the same evidence labels.
- Too early to know: attractive-looking rows below the versioned evidence floor.
- Human versus bot: where user environment annotations or took/passed decisions differed, with no reward for either side until outcomes are measured.
- Next action: wait condition, invalidation, review task, or "no qualified trade."

Every sentence should link back to the rows that support it. The summary must not select a tiny lucky sample as "best." Existing versioned learning thresholds should govern Proven, Developing, Muted, and Insufficient Evidence rather than adding an untracked second set of thresholds.

Illustrative language:

> Long VWAP-bounce examples after 10:30 have the best support in bullish-strong sessions: +0.62R average across 24 closed examples. One current candidate matches the context, but it still needs the documented completed-bar trigger.

That sentence is only an example of style, not a claim about current bot data.

Place a compact version in Today/Command Center and a full version at the top of Research. The AI Summary tab may consume the same normalized evidence later.

### 5.8 AI Summary tab and provider connections

#### Product behavior

Add an AI Summary tab with:

- report type;
- date/session range;
- source checkboxes;
- provider and model selection;
- exact payload preview;
- estimated size/cost guard;
- Run, Cancel, Save, Compare, and Export Evidence Package actions;
- links from every claim to its local evidence.

Initial selectable sources:

- Daily Market Prep report.
- Auto market-condition scanner, including automatic regime and user annotations.
- Master AVWAP current/final opportunities.
- RS/RW windows and Industry Board snapshots.
- Setup Tracker and playbooks.
- Day Trade Tracker.
- Move Forensics.
- Journal entries, imported executions, notes, and outcomes.
- Pick feedback and took/passed/missed decisions.
- Runtime Health and data-quality summary.

Initial report types:

- Morning brief.
- Best supported trades today.
- Auto market-condition scanner review.
- All setup trackers review.
- Daily after-close review.
- Journal coach review.
- Weekly learning review.
- Compare user environment calls with automatic calls.

#### Connection model

"Attach ChatGPT or Claude" should mean an explicit provider connection:

- OpenAI API through the Responses API.
- Anthropic API through the Messages API.
- Optional Export Evidence Package mode for manually pasting a prompt into a consumer ChatGPT or Claude session without storing an API key.

Do not automate a browser session or assume a consumer subscription provides API access.

Consolidate the existing Market Prep OpenAI code behind one provider-neutral interface instead of adding a third client. A conceptual contract is:

    summarize(request) -> validated AiSummary

Adapters:

- OpenAIProvider.
- AnthropicProvider.
- ExportOnlyProvider.
- A deterministic local summary provider for tests and offline fallback.

The OpenAI adapter should use the Responses API and Structured Outputs. OpenAI's official documentation recommends the Responses API for new text-generation work and documents schema-constrained output:

- [OpenAI text generation and Responses API](https://developers.openai.com/api/docs/guides/text?api-mode=responses)
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

The Anthropic adapter should use the Messages API and a strict schema/tool contract where appropriate:

- [Anthropic Messages API](https://platform.claude.com/docs/en/api/messages)
- [Anthropic tool use and strict schemas](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview)

Models must be configurable and recorded by exact model ID. Do not bake a marketing alias into research history. Prompt changes, schema changes, and model changes require versioning and evaluation.

#### Security and privacy

- Store provider secrets in Windows Credential Manager or read them from environment variables.
- Never write API keys to the repository, ordinary JSON settings, diagnostics, report manifests, Google Drive, or prompts.
- Show only connection status and the key source in Settings.
- Offer Test Connection without logging the secret.
- Default all journal/free-text sources to opt-in.
- Redact account identifiers and unrelated personal data.
- Preview the exact outbound evidence package before the first run and whenever source scope changes materially.
- Default to manual runs; scheduled after-close/weekly reports are a later opt-in.
- Apply time, token, and cost caps and make cancellation real.

Anthropic's official authentication guidance also recommends environment variables or a secret-management system:

- [Anthropic authentication guidance](https://platform.claude.com/docs/en/manage-claude/authentication)

OpenAI likewise documents API keys as secrets that should be loaded from an environment variable or key-management service:

- [OpenAI API authentication and key safety](https://platform.openai.com/docs/api-reference/introduction)

#### Evidence package and output contract

Every run receives an immutable input manifest:

- report ID and generation time;
- selected sources, filters, date range, and as-of timestamps;
- source file/snapshot IDs and hashes;
- provider, exact model, prompt version, and output-schema version;
- maximum token/cost budget;
- user question, if any.

The validated output schema should include:

- headline and scope;
- key observations;
- best-supported patterns with evidence references;
- current opportunities using only supplied opportunity IDs;
- risks, counter-evidence, and missing data;
- questions for the user;
- suggested research or replay tasks;
- explicit uncertainty.

Store the manifest, validated output, raw provider metadata needed for debugging, usage/cost when available, and failure/refusal/incomplete state. Never treat malformed free text as a valid report.

AI restrictions:

- It cannot create an opportunity ID.
- It cannot modify watchlists, scores, thresholds, detector state, mode, or alerts.
- It cannot place or recommend an order as an automated action.
- It cannot claim a pattern is Proven without the supplied evidence state.
- It cannot hide missing/stale data.
- Suggestions enter a research queue and require human approval plus the normal fixture/replay/evidence path.

#### Provider validation

Before release:

- Contract-test both adapters against recorded responses.
- Test strict-schema success, refusal, timeout, rate limit, invalid schema, and incomplete output.
- Run a fixed golden evidence set through supported providers and compare factual extraction.
- Reject or visibly flag evidence references that do not exist.
- Record latency, cost, schema success, and unsupported-claim rate.
- Ensure an unavailable provider never blocks scanning, journaling, or deterministic summaries.

### 5.9 Journal and the learning loop

The journal should become the durable link between "best trade today" and "better evidence tomorrow."

Create a universal lifecycle:

    discovery
      -> rank/gate transition
      -> alert
      -> user decision
      -> execution import or explicit no-trade
      -> management events
      -> MFE/MAE and outcome
      -> review and research label

Required links:

- opportunity_id and lifecycle_id;
- scanner/setup version and source snapshot;
- market, sector, industry, and stock context at decision time;
- user environment annotation and automatic environment;
- took, passed, missed, late, invalidated, duplicate, or not useful reason;
- planned entry, invalidation, targets, and risk when supplied;
- imported fills linked idempotently;
- MFE, MAE, realized R, and time in trade;
- cached chart/screenshot references;
- rule adherence separated from P&L;
- free-text notes separated from structured reason codes.

Important analysis rules:

- A profitable rule break is not evidence that the process was good.
- A valid setup stopped out is not evidence that the setup was invalid.
- User preference, objective setup quality, and execution fit remain separate dimensions.
- Passed and missed opportunities matter; analyzing only filled trades creates selection bias.
- Imports and reruns must be idempotent.

The AI Summary tab reads this lifecycle. It does not become the source of truth for it.

### 5.10 Auto/Away Google Drive reports: swing-first

The hourly Auto/Away report is primarily consumed as a quick phone briefing. Its candidate hierarchy should match the user's priority: swings first, intraday opportunities second.

Scope:

- Apply this contract to autopilot_today.txt and every future human-readable report or summary automatically published to Google Drive by Desk/Away Auto.
- Keep the current scheduling boundary: Away remains the hourly publisher. This ordering requirement does not by itself authorize Desk mode to start hourly Drive writes.
- Do not merge or reorder dedicated machine-readable files such as longs.txt, autolongs.txt, swinglongs.txt, JSON, CSV, or tracker state merely to change presentation. Existing consumers rely on those separate schemas.
- When a future combined machine-readable artifact is needed, version its schema explicitly rather than changing an existing file in place.

Recommended report order:

1. Safety header: generated time, freshness warning, mode, IB connection, regime, data status, and health.
2. Swing Opportunities: the first candidate section and the visual headline of the report.
3. Day Trade Opportunities: long and short intraday lists.
4. Bot Picks and current alerts.
5. Schedule, tracker status, and activity/operations detail.

The swing section should:

- Lead with finalized High Conviction/Favorite swing upgrades when present.
- Then preserve the canonical swing feed's existing deterministic order; do not invent a Drive-only score.
- Show side, bucket/badges, expected R when valid, setup/thesis, data complete-through time, and a TradingView paste line split by long/short when useful.
- Use the same opportunity dedupe contract as the GUI.
- Label a distinct intraday thesis separately when the same symbol genuinely has both a swing and a day-trade opportunity.
- Say "No qualified current-session swing opportunity" when empty.
- Say "Awaiting today's first completed swing scan" when current-session data does not yet exist; never elevate yesterday's swing rows as if they were current.

The safety/freshness header remains above swings because stale Auto output must never look tradable. "Swing-first" means the first and dominant opportunity content, not hiding whether the report is stale.

Acceptance criteria:

- A golden text fixture proves Swing Opportunities appears before every day-trade or bot-pick candidate section.
- High Conviction/Favorite finalized swing upgrades appear first without changing their underlying score.
- An empty or not-yet-scanned swing state still occupies the first candidate section with an honest message.
- The report preserves the verified atomic publish, metadata, archive, and last-good recovery behavior already implemented.
- Any manually requested Desk report, the scheduled Away report, and any future candidate-bearing AI Drive summary use the same report-section ordering contract.

## 6. Cross-cutting improvements

### 6.1 Command Center

After the canonical Opportunity model is ready, build a Today summary with:

- Ready Now;
- Confirming;
- Waiting / One Step Away;
- invalidated or stale;
- the best evidence-matched opportunity, if one exists;
- market, sector, industry, stock RS, entry, invalidation, targets, R, freshness, and next event.

Desktop, Desk Auto, Away Auto, alerts, and reports must project the same canonical snapshot.

### 6.2 Alert Center

- Deduplicate by typed lifecycle transition, not message text.
- Separate Actionable, Developing, Research, Data/Health, and History.
- Show why it fired, what changed, and what would invalidate it.
- Make acknowledgement, snooze, mute, and journal actions explicit and reversible.
- Prevent repeated refreshes from generating repeated alerts.

### 6.3 Data freshness everywhere

Every actionable card should expose:

- source snapshot ID;
- completed-through timestamp;
- local display timezone;
- Fresh, Preview, Stale, Partial, or Failed;
- missing dependency when partial.

Do not silently mix timestamps from different sources.

### 6.4 Health extensions

Extend the existing Health page with:

- Industry Board owner, last attempt/success, next due, duration, coverage, and snapshot ID.
- RS/RW last completed M5 bar and candidate coverage.
- AI provider connection state without secrets.
- Last AI summary result, duration, schema state, cost, and source coverage.
- Journal linkage/import failures.
- Background job collision/single-flight state.

### 6.5 Usability and accessibility

- Use consistent names for environment, mode, bucket, status, and evidence.
- Keep Explain Mode on by default and remember the user's choice.
- Support keyboard table selection and detail opening.
- Never encode long/short, strong/weak, or fresh/stale by color alone.
- Provide copyable plain-English explanations with timestamps.
- Preserve layout and filters without persisting stale market state as a user decision.

### 6.6 Performance and reliability

- Provider fetches and heavy summaries run off the GUI thread.
- Views consume cached immutable snapshots.
- Background work supports cancellation, bounded retries, and single-flight ownership.
- Last-good data remains readable during failure.
- Scheduled work is visible in manifests and the job ledger.
- No new feature may create a second competing market-data loop.

## 7. Ordered implementation plan

This sequence is subordinate to plan.md Section 12.

### Phase 0 - finish the current safety gates

1. Complete the current live-session validation checklist and preserve the legacy baseline.
2. Confirm Health and lifecycle diagnostics during a real session.
3. Add or extend golden fixtures before any change to D1 routing, RS scoring, detector behavior, or ranking.
4. Record current screenshots and payload fixtures for the Industry Board, Master focus duplicates, Bounce controls, Entry Assist, D1 Focus, Auto/Away Drive report, and all tracker tables.

Exit: current behavior is reproducible, failures are attributable, and new product work has comparison fixtures.

### Phase 1 - trustworthy GUI foundations

1. Build IndustryBoardService, atomic snapshots, freshness state, and hourly single-flight scheduling.
2. Add real numeric sorting, strongest/weakest presets, and last-good failure behavior.
3. Add snapshot IDs/freshness to Industry Board and current RS joins.
4. Fix Master focus read-model deduplication without changing AVWAP scoring.
5. Make swings the first candidate section in all human-facing Auto/Away Google Drive reports.
6. Extend Health for industry freshness and ownership.

Exit: no duplicate focus opportunity, no mystery-stale board, all consumers share the same industry snapshot, and phone reports lead with current swing opportunities.

### Phase 2 - clarify automatic and actionable behavior

1. Separate Bounce Bot sees from User override.
2. Default each session to Auto plus N/A and add append-only annotations.
3. Make Entry Assist automatic status primary; move manual windows to Advanced.
4. Keep Strongest, Weakest, and Movers 30m prominent.
5. Enforce final-upgrade-only D1 Focus routing after its golden fixture is approved.

Exit: a novice can tell what is automatic, what the bot believes, what the user overrode, and which D1 rows are truly final selections.

### Phase 3 - explanations and deterministic summaries

1. Define one metric glossary and typed ExplanationContext.
2. Wire every Setup Tracker, Day Trade Tracker, and Move Forensics row to the detail pane.
3. Implement setup/context/research-specific explanation templates.
4. Build deterministic What's Working Now and evidence links.
5. Add GUI/offscreen tests and accessibility checks.

Exit: every row explains itself truthfully and the GUI can summarize evidence without an internet connection or LLM.

### Phase 4 - canonical industry RS/RW and Opportunity data

1. Follow the canonical SPY/RS dependency work in plan.md.
2. Define primary-industry and additional-theme provenance.
3. Build aligned completed-M5 industry composites with coverage rules.
4. Release industry and stock-within-industry fields in advisory/shadow form.
5. Replay and compare missing-data, overlap, strong/weak, and session-boundary cases.
6. Integrate the fields into canonical ranking only after gates pass.
7. Replace the temporary Master fingerprint with canonical opportunity_id.

Exit: Industry Board genuinely participates in canonical calculations without double-counting, cherry-picking membership, or changing live behavior without evidence.

### Phase 5 - provider and AI Summary foundation

1. Land the storage/secrets abstraction already required by plan.md.
2. Consolidate existing OpenAI Market Prep paths behind AiProvider.
3. Add OpenAI, Anthropic, export-only, and deterministic test adapters.
4. Create normalized evidence packages and immutable run manifests.
5. Build the AI Summary tab with manual source selection, payload preview, strict output validation, and local evidence links.
6. Add provider contract tests and a fixed evaluation set.

Exit: either provider can produce a schema-valid, evidence-linked report without gaining authority over production state.

### Phase 6 - complete the all-in-one learning loop

1. Link opportunity discovery, alerts, user choices, executions, outcomes, and screenshots.
2. Add passed/missed/invalidated evidence, not only filled trades.
3. Build the canonical Today Command Center.
4. Add after-close and weekly AI summaries as explicit opt-in jobs.
5. Add comparison reports for auto versus user environment calls and bot versus user selection/execution.

Exit: a trade idea can be followed from discovery to outcome and reviewed tomorrow without manual reconstruction.

### Phase 7 - evidence-driven refinements

1. Measure product and model performance over multiple sessions.
2. Promote only changes that meet their declared evidence floor.
3. Route AI suggestions into explicit experiments with fixtures and replay.
4. Continue the shadow-engine promotion ladder exactly as plan.md defines.

Exit: improvements are attributable, reversible, and evidence-backed.

## 8. Test and verification strategy

### Unit and contract tests

- Numeric sort roles, missing values, negative values, and stable tiebreakers.
- Industry schedule, fake clock, session calendar, single-flight, retry, atomic replace, and last-good fallback.
- Composite member coverage, timestamp alignment, completed/forming bar behavior, primary membership, and overlap.
- Master opportunity fingerprint and dedupe cases.
- Auto/N/A/user environment state transitions, restart/session expiry, and annotation persistence.
- Entry Assist automatic state projection and manual diagnostics isolation.
- D1 final bucket upgrade versus non-final trigger routing.
- Explanation templates, metric glossary, missing-level behavior, and evidence links.
- Deterministic summary evidence floors and no-qualified-result behavior.
- Auto/Away report-section ordering, swing dedupe, empty/current/stale swing states, and preservation of verified publishing metadata.
- AI provider schema, refusal, incomplete, timeout, cancellation, usage, redaction, and invalid evidence references.
- Journal lifecycle idempotence and opportunity/execution linkage.

### GUI integration tests

- Clickable and keyboard-selectable rows open the correct explanation.
- Sorting changes the visible order without mutating source data.
- Stale/failed snapshots are obvious.
- Bot environment changes do not alter the N/A user control.
- Selecting and clearing an override updates status and history once.
- D1 Focus contains only final target-bucket upgrades.
- AI source preview matches the outbound package.
- Provider failure cannot freeze or disable scanning.

### Replay and golden tests

- Preserve existing detector and score outputs for all presentation-only phases.
- Compare canonical industry calculations across normal, missing, late, overlapping, half-day, and session-boundary data.
- Snapshot the same opportunity across lifecycle transitions and prove alert dedupe.
- Snapshot the Auto/Away phone report and prove the safety header remains first while swing opportunities precede all intraday candidate sections.
- Use fixed evidence packages to evaluate AI factual consistency across prompt/model versions.

### Repository gates

Before every implementation commit:

- Run the full pytest suite and check pytest's actual exit code.
- Run scripts/smoke_check.py and require all deterministic checks to pass.
- Keep commits small and green and push each completed commit.

## 9. Product success measures

Trust and reliability:

- Zero duplicate Master opportunities caused only by bucket membership.
- Industry snapshot freshness within the promised cadence during active hours.
- Zero overlapping industry writers.
- Percentage of actionable cards with complete freshness and source metadata.
- Background job and provider failure recovery rate.

Decision quality:

- Time from opening the app to understanding the best qualified opportunity.
- Percentage of D1 Focus rows that end in Favorite/High Conviction.
- Percentage of alerts linked to one canonical opportunity lifecycle.
- Qualified-opportunity precision, measured without forcing a minimum daily count.

Comprehension:

- Percentage of tracker rows with a complete novice explanation.
- Percentage of explanation views with valid source/as-of data.
- Reduction in manual Entry Assist window actions during Auto operation.
- Percentage of human-facing Auto/Away Drive reports that satisfy the swing-first ordering contract.

Learning loop:

- Percentage of decisions linked to an outcome or explicit no-trade reason.
- MFE/MAE and realized-R coverage.
- Passed/missed opportunity capture rate.
- Ability to compare auto versus user environment calls on matched evidence.

AI quality:

- Strict-schema success rate.
- Evidence-reference validity rate.
- Unsupported-claim rate.
- Provider latency and cost per report.
- Percentage of reports that remain reproducible from their saved input manifest.

## 10. Explicit non-goals

- Order placement or automated execution.
- AI-driven live score, threshold, watchlist, alert, or mode changes.
- Automatic promotion of market_state or greatness_monitor.
- Replacement of the current AVWAP sigma calculation.
- Automatic removal of user-entered watchlist names.
- Filling recommendation panels with low-quality names to meet a quota.
- Treating a forensics correlation or Day Trade segment as a standalone setup.
- Maintaining independent desktop, Auto Pilot, Away, and report calculations.

## 11. Decisions to confirm during discussion

Recommended defaults are shown first.

1. User market override lifetime

   Recommendation: retain across an app restart within the same session, then expire to N/A at the next market-session boundary.

2. D1 upgrade-trigger events

   Recommendation: remove them from D1 Focus. Keep them only in a quiet Developing research view if they remain useful.

3. Industry membership

   Recommendation: declare one primary industry for stock-within-industry scoring and display additional themes as context. Do not silently use whichever group currently ranks best.

4. Industry refresh hours

   Recommendation: hourly through premarket and the regular session, plus one after-close snapshot; manual refresh at any time.

5. AI connection

   Recommendation: support OpenAI API, Anthropic API, and export-only mode. Do not automate consumer browser sessions.

6. AI scheduling

   Recommendation: manual runs first. Add opt-in after-close and weekly schedules only after evidence packages and provider validation are stable.

7. Novice explanations

   Recommendation: Explain Mode on by default, with advanced calculations collapsed but always available.

8. Developing opportunities

   Recommendation: keep them in Research, not D1 Focus or the main Ready Now lane.

## 12. Definition of done for this product program

This program is complete when:

- Industry rankings refresh predictably, sort correctly, and participate in canonical aligned RS/RW with visible provenance.
- Master AVWAP shows one card per real opportunity.
- Bounce starts every session in Auto with an unambiguous N/A user override and durable user annotations.
- Entry Assist requires no manual window action during Auto operation.
- D1 Focus contains only final Favorite/High Conviction upgrades.
- Auto/Away Google Drive reports retain their safety header and show swing opportunities before day-trade candidates.
- Every tracker/forensics row gives a truthful novice explanation.
- What's Working Now is understandable, evidence-linked, and honest about insufficient data.
- OpenAI or Anthropic can be connected safely to a provider-neutral AI Summary tab, while export-only mode remains available.
- Every AI report is reproducible from an immutable evidence package and cannot mutate production state.
- Opportunity discovery, user decisions, executions/no-trades, outcomes, and reviews form one queryable lifecycle.
- Desktop, Auto Pilot, Away, alerts, journal, deterministic summaries, and AI summaries all agree on the same canonical facts.
