# Professional 4K GUI redesign plan

**Status: PROPOSAL FOR TRADER REVIEW — planning only; no runtime change is
authorized by this document.** `plan.md` remains the only roadmap. This proposal
must be accepted and promoted there before implementation.

**Target workstation:** one trader, one main 4K monitor, the application using
the full available height and approximately 90% of the monitor width. On the
current Windows display scaling this is a 2304 × 1392 logical-pixel window
(approximately 3456 × 2088 physical pixels), leaving a projected Trading Desk
content area of roughly 2060 × 1308 logical pixels after shell chrome.

**Design goals:** simple, concise, clear, chart-first, keyboard-fast, and honest
about stale or missing data. This is a professional decision-support source of
truth, not a decorative dashboard. Order execution remains permanently out of
scope.

---

## 1. Executive decision

The application already contains most of the information the trader needs. The
primary problem is allocation and hierarchy, not missing features.

| Question | Finding | Recommended decision |
|---|---|---|
| Can the trader see the live decision context quickly? | **Mostly.** The D1/M5 review pair is strong and readable at 4K. Freshness, source, queue state, and chart controls are generally visible. | Keep one chart implementation and make the Veto/Alert workspace the default Trading Desk surface. |
| Is the normal Veto workflow fast to read and use? | **Not yet.** The lower pane always gives 40% of its width to Focus Strength, including while Capture/Veto, Armed, and Journal are open. Capture then scrolls vertically and hides reasons/actions below the fold. | Give Capture, Armed, and quick Journal the full lower width. Show Focus Strength only where it adds value. Make Veto the remembered/default capture mode and fit its full action in one viewport. |
| Can Master AVWAP be used as needed without hurting charts? | **Yes, but the full-width tab is much better than the split drawer.** Setups and most tables remain usable in the split; Watchlist toolbars and RS Window become cramped. | Default to a Veto-first tabbed desk: `Veto / Alerts` first, `Master AVWAP` second. Keep the current simultaneous split as an optional preset. |
| Is the application intuitive outside the desk? | **Uneven.** Strong research tables coexist with giant blank states, raw text dumps, duplicated configuration, and fourteen equally weighted nav items. | Group navigation by daily workflow, review, research, and system. Use a clear empty state and selected-row detail instead of permanent blank regions. |
| Is Market Journal ready to be the trader's thesis record? | **The storage contract is right; the page is not.** It captures text honestly, but the composer and empty lists consume the screen, the promised charts are only a note, and trades are not shown beside thoughts. | Make it a session timeline with a sticky fast composer, one large selected-symbol chart, machine context, and a read-only execution/entry/exit lane. Keep it separate from the Trade Journal. |
| Is the UI currently fluid enough? | **No.** The latest live session measured 264 stalls in about 45 minutes, 46 seconds blocked, and an 8.45-second freeze. | Fix measured GUI-thread work first. Then offer an experimental bounded-cache `Snappy` mode that spends the available RAM/CPU on background preparation, never on more work in the Qt thread. |

### Recommended default operating model

1. Open Trading Desk directly into **Veto / Alerts**, with the last reviewed
   chart and the **Capture → Veto** form ready.
2. Keep the D1 and M5 charts large. Between the charts and the lower tab strip,
   retain only the slim queue verb row required by the chart-review contract.
3. Open **Master AVWAP** as the adjacent full-width tab when needed. Use the
   optional split view only when simultaneous monitoring is genuinely useful.
4. Keep a one-line BounceBot/group-tape/status strip visible; do not dedicate a
   nearly empty full-page BounceBot tab to it.
5. Make all lower-pane width contextual. Empty or irrelevant companions should
   never reserve 40% of the desk.

---

## 2. Scope and non-negotiable contracts

This proposal changes presentation and interaction design only. It does not
authorize a detector, scorer, alert, queue, scheduler, evidence, broker, or
storage-contract change.

The implementation must preserve:

- decision support only; no order-entry or execution controls;
- the chart pane owns the Alert Center and stays the largest vertical region;
- the queue verb row advances without an extra click;
- CaptureRail actions remain reachable without a mouse and within the existing
  five-second contract;
- VETO and LIKE move the review queue; NOTE does not;
- LIKE records a setup claim and required reason but does not add Focus or alert
  privileges;
- `Veto D1 — but M5 today` records the ordinary veto first, then asks the one
  Focus owner for the M5 placement;
- Market Journal remains "what the trader thought" and Trade Journal remains
  "what the trader traded"; their stores are not merged;
- journal entries are never backdated; corrections supersede rather than edit
  history;
- every expensive read, parse, calculation, and payload build remains off the
  Qt thread;
- stale, unavailable, partial, unmeasured, and last-good states are stated, not
  rendered as zeros or confidence;
- no duplicate writer, timer, worker, chart implementation, or alert source;
- user-entered watchlist and Focus names remain trader-owned;
- the current local-first storage boundary and DAS separation remain unchanged;
- phone push policy remains AWAY-only except for the two documented wake-alert
  exceptions.

### Contract/source mismatch to resolve before implementation

The root operating context says the arm bar belongs on the **Armed** tab above
the inventory, leaving only the verb row beneath the charts. The running source
and live 4K screen currently render the arm bar under the charts
(`AlertChartReview(..., dock_arm_bar=True)`). The redesign recommendation is to
restore the documented contract because it increases chart height and groups
controls with the inventory they create. This touches the fenced Alert Center
path and requires an explicit ask-first confirmation when promoted.

---

## 3. What was tested

### 3.1 Live and projected geometry

The full live application was exercised top-to-bottom, including every
top-level page and representative populated states for every nested tab. The
Windows/Qt scale contract treats 2560 × 1440 logical pixels as this 4K desk
after display scaling. The 90%-width projection therefore used a 2304 × 1392
outer window and a 2060 × 1308 Trading Desk content surface.

Measured in the projected content surface:

| State | Measurement | Consequence |
|---|---:|---|
| Setups hidden | Alert chart/review pane 2028 × 722 | Charts have enough width; vertical chrome is the limiting resource. |
| Setups hidden | Lower row split 1215 / 809 | Focus Strength permanently consumes 809 px even while Capture is active. |
| Setups hidden, Capture | Capture viewport 1195 × 352; content size hint 300 × 759 | Less than half of the vertical capture content fits without scrolling. |
| Setups shown | Desk columns 1184 / 856 | The chart remains usable, but secondary toolbars and board columns become tight. |
| Setups shown, Capture | Capture viewport 679 × 352; Focus Strength 465 px | The primary Veto action is squeezed by a companion that is not needed for the action. |

The current layout tests explicitly cover the 2560 × 1440 logical 4K scale,
chart-leading desk ratios, compact Master table columns, responsive arm controls,
and saved split behavior. The focused planning baseline is **116 tests passed**:

```text
tests/test_qt_desk_layout.py
tests/test_ui_scale.py
tests/test_market_journal.py
tests/test_away_day_recap.py
tests/test_ui_fluidity.py
tests/test_ui_stall_watchdog.py
tests/test_qt_page_specs.py
```

### 3.2 Current-session fluidity evidence

The live stall log was filtered to the GUI process started at 21:29 on
2026-08-25:

| Metric | Result |
|---|---:|
| Stalls over 50 ms | 264 |
| Median / p90 | 117.3 ms / 205.1 ms |
| Worst | 8445.6 ms |
| Total blocked | 46.0 s in about 45 minutes |
| Working set / private bytes during review | about 0.80 GB / 1.31 GB |

Measured causes include:

- **8.45 s:** Weekend Prep `WeekReviewPage.reload()` synchronously read and
  joined review-learning/outcome CSVs on the GUI thread;
- **36 repeating stalls, 5.93 s total:** Focus chips repeatedly resolved mover
  state per symbol/editor during `_refresh_all`;
- **about 1.0 s each direction:** changing Trading Desk Workspace/Tabs mode
  reparents and relays out the large widget trees;
- frequent event-loop-only samples indicate expensive Qt layout/paint work where
  the Python sampler cannot name a deeper frame;
- Health/table population still creates cells imperatively and appears in the
  stall record.

The earlier fluidity work materially improved the desk, but the active run still
misses the standing full-session targets of no stall over five seconds and less
than about sixty seconds blocked.

### 3.3 Testing limitation

This was a realistic layout and interaction review, not a live market-behavior
acceptance. It marks none of the existing Phase 0 live gates complete. A later
implementation needs a real active-session proof because queue pressure,
incoming rows, chart refreshes, and alert sounds cannot be validated by an
offscreen layout alone.

---

## 4. Information architecture and navigation

Fourteen equal-weight navigation buttons mix live trading, review, research,
maintenance, and configuration. They all fit vertically at 4K, but fit is not
the same as hierarchy. The trader should be able to locate a page by purpose,
not remember a flat list.

### Proposed navigation

| Group | Always visible | Collapsible/secondary |
|---|---|---|
| **LIVE** | Trading Desk, Focus Picks, Strength Board, Auto Pilot | — |
| **REVIEW** | Market Journal, Trade Journal, AWAY Recap | Chart Review, Weekend Prep |
| **RESEARCH** | Research | A.I. Summary |
| **SYSTEM** | System Health, Settings | Universe / Data Tools |

Rules:

- Rename the left-nav `Journal` label to **Trade Journal**. No store or contract
  changes; this only removes the daily ambiguity with Market Journal.
- Keep Trading Desk first and focused on launch.
- Let the trader pin up to four favorites at the top, but default them to the
  professional workflow rather than requiring setup.
- Collapse group labels, not individual daily pages. A collapsed group retains a
  count/status badge if a child is unhealthy or has pending work.
- Move the Workspace/Tabs presentation choice out of the global top bar and into
  Settings. The daily top bar should contain current page, global search/command,
  connection/health state, and nothing else.
- Add one command palette / page switcher shortcut for pages and actions. It is a
  navigation index, not a new automation surface.
- Preserve direct keyboard shortcuts for the current high-frequency actions.

### Global status strip

Keep a slim bottom status strip, but enforce stable slots:

1. Auto mode;
2. IB/data connection;
3. scan state with last completion and age;
4. current regime/technical state;
5. watchlist/universe freshness;
6. health severity.

Long diagnostic prose should truncate with a tooltip/details action. A status
strip that moves every other field when one message becomes verbose cannot be
scanned by position.

---

## 5. Trading Desk redesign

### 5.1 Recommended default layout

```text
┌──────────────── Trading Desk ──────────────────────────────────────────────┐
│ Group tape / scanner status / setup count                     health badge │
│ [ Veto / Alerts ] [ Master AVWAP ]                    [optional Split View]│
├────────────────────────────────────────────────────────────────────────────┤
│ Symbol · side · setup · freshness · reviewed/moving/armed state            │
│                                                                            │
│                         LARGE D1 CHART                                     │
│                                                                            │
│                         LARGE M5 CHART                                     │
│                                                                            │
│ [Add to Focus] [Skip] [Not today]       queue · hidden · armed summary     │
├────────────────────────────────────────────────────────────────────────────┤
│ [Alerts] [D1 Focus] [RS/RW] [Armed] [Capture] [Quick Journal]              │
│                                                                            │
│ Active tab uses the whole row unless its own companion is useful.          │
└────────────────────────────────────────────────────────────────────────────┘
```

At the 2304 × 1392 target:

- reserve approximately **65–70% of usable vertical space** for chart context,
  including the symbol/header and the two charts;
- give D1 slightly more height than M5, approximately 54/46 within the chart
  pair, with a draggable and remembered split;
- keep the lower tab row approximately 30–35%, with a minimum that allows the
  active action to fit but never steals the majority from charts;
- collapse the left nav when maximum chart width is desired; this is a visible
  user action with a remembered state, not an automatic surprise;
- retain last-good chart content while refreshing and overlay freshness/source
  state rather than replacing charts with blank loading panes.

### 5.2 Veto/Capture — the primary workflow

Current Capture lays out Veto, Like, and Note sections together. It works in a
wide dialog, but in the Alert Center's 60% lower column it wraps and becomes a
759-pixel-high scroll surface inside a 352-pixel viewport.

Target behavior:

- Capture gets **100% of the lower-row width**. Focus Strength collapses while
  Capture is active.
- Inside Capture, use a compact mode selector: **Veto | Like | Note**. The
  shortcut both selects the mode and focuses its first actionable field:
  `Alt+V`, `Alt+K`, `Alt+N`.
- Remember the last mode per machine; default to **Veto** because that is the
  normal desk posture.
- Veto shows every versioned reason at once in two readable columns at this
  width. Keys and labels stay unchanged; no reason is hidden in a dropdown.
- Keep the optional note on the same line or directly below the selected reason.
  A reason that requires a note must visibly change the field to required.
- Keep `Veto — not for today` and `Veto D1 — but M5 today` together at the
  bottom, always visible.
- Like shows the complete offered claim list, keyed in the current stable order,
  plus the required `why` field and commit button. It never changes Focus.
- Note shows one clear composer and save state. NOTE never advances the queue.
- The status line has a fixed reserved height and uses explicit states:
  `READY`, `SAVING`, `SAVED`, `NOT SAVED`.
- Preserve the panel-scoped shortcut rebinding; a hidden tab must never make the
  shortcuts inert.
- Display the selected D1 reference level, side, and current symbol in one
  compact header. Do not repeat explanatory paragraphs in professional mode.

Acceptance: with either Master hidden or shown at the 90%-width target, all Veto
reasons, the note field, both Veto actions, and the save state are visible with
no vertical or horizontal scroll.

### 5.3 Alert Center lower tabs

| Tab | Keep | Change |
|---|---|---|
| **Alerts** | One live row per symbol/side/day, escalation badges, source/tier, quick Focus action. | Use the right-side Focus Strength companion here if it is populated. Let the trader collapse it. Preserve selected row and scroll during updates. |
| **D1 Focus** | Separate untiered stream and unread count. | Full-width list when narrow; optional Strength companion only when it contains relevant rows. Put the short rubric behind help in professional mode. |
| **RS/RW Board** | Same BounceBot snapshot, board selection opens the shared chart. | This is the natural home for the Strength companion; combine their related context rather than duplicating board space elsewhere. |
| **Armed** | Inventory of chart watches, D1 event watches, D1 levels, and price alerts. | Restore the documented arm bar above this inventory. Give the tab full width. Group by current symbol first, then all symbols; expose expiry/source. |
| **Capture** | Append-only Veto/Like/Note recorder and keyboard map. | Full width and compact action modes as §5.2. |
| **Quick Journal** | M5-default free-text capture and Ctrl+Enter. | Full width; attach the current chart symbol by default with a visible remove control; show the last three session entries below the composer and link to Market Journal. |

Use a per-tab splitter policy rather than one global 60/40 ratio. The policy is
presentation-only and remembered. It must never alter source, queue order, or
alert behavior.

### 5.4 Charts

Keep the shared `SymbolSnapshotWidget`/`CandleChart` path and improve only its
information hierarchy:

- fixed top line: symbol, side, source, as-of age, current price, setup/trigger;
- badges in a predictable order: `REVIEWED`, `MOVING`, `ARMED`, data caveat;
- D1 and M5 titles always include timeframe and last completed bar;
- stale or fallback data uses a banner inside a reserved line, never a modal;
- paint-line controls stay one compact menu; selected level is echoed beside the
  arming/capture context;
- charts retain their scale and crosshair state when the lower tab changes;
- chart loading never clears the last-good chart unless the symbol changes;
- no raw warning or traceback is painted into the chart title.

### 5.5 Master AVWAP tabs

The full-width Master tab is the preferred deep view. Split view remains useful
for brief simultaneous reference but must use responsive toolbars.

| Tab | Review | Redesign decision |
|---|---|---|
| **Setups** | Strong dense ranked table; compact profile protects key industry/expected-R columns. | Preserve. Full-width by default when opened. Add a small selected-row detail band or drawer; do not grow permanent chrome. Keep F9/full-table behavior. |
| **Theta Plays** | Filters and selected-row detail use the space well. | Preserve. Align filter heights/labels with Setups and remember the last filter. |
| **Watchlists → Shared Longs / Shorts** | Lists are readable; toolbar labels collapse to fragments in split view. | Replace fixed horizontal button row with a wrapping row or `Actions` overflow. Never elide an action to two letters. |
| **Watchlists → Master Swing Lists** | Same responsive-toolbar problem. | Same solution; keep ownership/provenance visible before destructive clears. |
| **Watchlists → Auto Lists (bot-owned)** | Read-only ownership is clear. | Preserve; add as-of/source in the header and make the read-only state visually unmistakable. |
| **Industry Board** | Two useful ranked tables fit well at full width. | Preserve. Selection opens the shared chart; keep independent sort state and concise as-of line. |
| **RS Window** | SPY selection chart plus industry/results is conceptually strong but cramped in the split drawer. | Treat as a full-width analytical tab. Use a vertical splitter with a minimum result-table height; collapse prose and keep selection controls on one wrapping row. |

### 5.6 BounceBot presentation

The current top-level BounceBot tab contains very little beyond status and an
entry-assist line, leaving most of the screen blank. The better presentation is:

- persistent one-line status strip in Trading Desk: connection, scan state,
  current mode, active/disabled types, last cycle, and entry-assist state;
- click the strip to open a compact operational drawer with recent activity and
  reconnect;
- move RRS tuning, bounce-type switches, and connection configuration to
  Settings → Automation & Scanners;
- move detailed timing/history to System Health;
- remove the empty top-level Trading Desk tab unless it gains a real monitoring
  purpose.

### 5.7 Standalone Chart Review

Chart Review duplicates much of Alert Center but provides a useful dedicated
keyboard review theater. Keep it only with a distinct job:

- frozen/manual review order, independent of live feed re-ranking;
- D1/M5 charts at maximum height;
- 360–420 logical-pixel right rail so claim/reason labels are never clipped;
- setup drawer becomes a searchable ranked table, not a long alphabetic text
  list that compresses charts;
- same capture semantics and shared chart data service;
- clear `Return to live queue` action.

If the trader does not use the frozen/manual-review distinction, fold this page
into Trading Desk and remove the duplicate nav item. That is a product decision,
not an implementation assumption.

---

## 6. Market Journal redesign

### 6.1 Purpose

Market Journal is the point-in-time record of the trader's market thesis:

- D1 thesis and bias;
- intraday tape observations;
- key levels and invalidation;
- what changed and when;
- the machine context that was available at the time;
- entries and exits that actually occurred beside those thoughts;
- later AI synthesis that helps the trader remain internally consistent.

Trade Journal remains the tax-grade execution record. Market Journal may read
its executions and link to a trade, but it never duplicates, edits, or becomes
the owner of broker data.

### 6.2 Target layout

```text
┌ Session · bias · key levels · invalidation · last saved · AI status ───────┐
├───────────────────────────────┬────────────────────────────────────────────┤
│ STICKY QUICK COMPOSER         │ [symbol chips: up to six for this session]│
│ [D1] [M5] [General] [tags]    │                                            │
│ free text                    │            LARGE SELECTED CHART            │
│ Ctrl+Enter · saved state      │              D1 / M5 toggle                │
├───────────────────────────────┤                                            │
│ SESSION TIMELINE              ├────────────────────────────────────────────┤
│ trader thoughts               │ MACHINE CONTEXT / REGIME / CALENDAR       │
│ thesis changes                ├────────────────────────────────────────────┤
│ read-only entries and exits   │ SESSION EXECUTIONS / ENTRIES / EXITS      │
│ AI synthesis revisions        │ AI SYNTHESIS (collapsed until generated)  │
└───────────────────────────────┴────────────────────────────────────────────┘
```

At the target width, use approximately 58% for composer/timeline and 42% for
the selected chart/context rail. The selected chart is large; up to six symbols
remain one click/number away rather than rendering six undersized live charts at
once. If six simultaneous D1 charts are still a hard trader requirement, offer a
separate `Grid` view using 2 × 3 cards and retain `Focus` view as the default.

### 6.3 Fast capture

- The composer is always visible and never taller than needed.
- Timeframe uses three direct chips: `D1`, `M5`, `General`; M5 stays the
  in-session default, D1 the sit-down-page default.
- Current chart symbol is attached automatically on the Desk quick surface and
  shown as a removable chip. Free-text symbol recognition may suggest chips but
  never silently attach one.
- Optional lightweight tags: `bias`, `thesis change`, `level`, `invalidation`,
  `risk`, `review`. Free text remains primary; tags cannot become a required
  form before the journal has earned that complexity.
- `Ctrl+Enter` saves. The input remains visible until the append confirms; a
  failed write says `NOT SAVED` and retains the text.
- Saving produces one timestamped timeline card. No dialogs.
- Addendum/correction uses the existing supersedes contract; the original stays
  on disk and the UI shows the relationship.

### 6.4 Daily thesis header

Provide four optional concise fields above the timeline:

| Field | Purpose |
|---|---|
| Bias | Bullish / bearish / neutral / mixed plus one-line rationale |
| Key levels | The few levels that control the day, not another level database |
| Invalidation | What would make the current thesis wrong |
| What changed | Latest thesis transition, derived from a marked entry or written directly |

These fields should save as ordinary versioned journal entries/metadata through
the one service, not a second mutable store invented by the panel.

### 6.5 Timeline and trade context

The central timeline interleaves visually but not structurally:

- trader journal entries;
- machine regime/context changes;
- calendar events;
- read-only Trade Journal executions, entries, scale-ins/outs, stops, and closes;
- AI synthesis versions.

Every card carries a source icon/label. A trade row links to Trade Journal. A
market-thought row can link to one or more symbols and a selected painted level.
No journal entry is inferred from a trade and no trade annotation is inferred
from prose.

### 6.6 AI synthesis

AI is an optional reader, never the author of record.

- `Generate synthesis` starts from the deterministic session packet: current
  entries, prior thesis, machine context, executions, and source counts.
- Show provider/model, generated time, evidence count, and missing sources.
- Separate `Summary`, `Consistency`, `Thesis changes`, and `Questions for
  tomorrow`; keep the default view concise.
- A generated summary never overwrites trader text. Refresh creates a new
  version.
- Free-text journal evidence remains opt-in under the existing AI policy.
- Local AI can be added later without changing the page contract; provider and
  model configuration live in Settings.

### 6.7 Current implementation gaps to close

- The promised D1 charts are currently a text note, not chart widgets.
- The page uses a large vertical composer over stacked empty lists, which makes
  the session record feel absent even when the feature is available.
- The quick Desk surface writes no current-symbol association despite the
  service already supporting `symbols`.
- No read-only Trade Journal execution lane exists.
- Session reload uses a worker correctly; preserve that.

---

## 7. Trade Journal review

Rename the page label to **Trade Journal** and preserve its tax-grade contract.

| Tab | Current finding | Redesign |
|---|---|---|
| **Trades** | The 50/50 trade table/detail split is useful, but the detail side shows many large empty groups and buries the execution story. | Add a compact selected-trade summary header, then collapsible `Plan & R`, `Executions`, `Tags`, `Notes/Review`, and `Corrections/Audit` sections. Render empty sections as one line. Show legs as a chronological execution timeline. |
| **Calendar** | Month grid occupies a small upper-left region; year heatmap and numeric grid create dead space and nested scrolling. | Responsive two-column month summary + year heatmap. One click filters Trades. No horizontal scrollbar at target width. Use blank for no-trade days and retain the existing zero-centered scale. |
| **Analytics** | Equity curve and grouping bars are useful but key metrics are not prominent; small tables float in large canvases. | Add a KPI strip, a 2 × 2 chart grid, and one breakdown/detail table. Thin samples stay marked. Selected chart/export remains exact. |
| **Health** | Important but overloaded: coverage, raw reconciliation, nightly runs, import history, repair actions, and credentials share one long page. | Separate `Status`, `Coverage`, `Reconciliation`, and `Runs` sections. Put the actionable failure and next step first. Move credentials/configuration to Settings → Data & Connections and deep-link there. |
| **Fees** | A few rows occupy a full tab with large unused space. | Fold into Analytics/Reports unless the trader uses it as a dedicated tax workflow. Keep exact export and per-account/currency separation. |

Shared filters (account/tax group, currency, date range) remain sticky above all
tabs. Filters should show their scope in one compact line and never silently
blend tax groups or currencies.

---

## 8. Review of every other top-level page

### 8.1 Focus Picks

Current four-quadrant Swing/M5 Long/Short layout fits 4K and is understandable.
The main issues are a three-line status header, expensive chip refreshes, and a
duplicated phone-alert table.

Plan:

- retain the four-quadrant ownership model;
- compress snapshot/review/freshness into one status row;
- keep chip badges but resolve their state once per refresh;
- make destructive `Clear all` visually secondary and confirmed;
- use the shared selected-symbol chart popup;
- show operational price alerts through the single Price Alert model instead of
  a second configuration surface;
- preserve the read-only reviewed-today line outside editable watchlist text.

### 8.2 Strength Board

The split M5 board / RS-RW board is useful when populated. Before the first
refresh it is an enormous blank canvas.

Plan:

- show last-good data immediately with as-of age;
- use a useful empty state with source, expected refresh, and one primary
  action;
- keep sortable columns and blank-last behavior;
- selecting a row shows a large shared chart/context pane or popup without a
  refetch;
- remember the M5/RS-RW vertical split;
- standalone page remains the deep board; the Desk companion should not reserve
  width when empty.

### 8.3 AWAY Recap

The page now draws alerts and can open shared charts, but all tables stack down
the left/top and leave most horizontal space unused. A live source error is
currently included as raw exception text in the main summary, and the Focus
reader call is incompatible with the current `load_focus_map` signature.

Target layout:

- KPI/status row: session, completeness, ranked swings, alerts, staged, Focus;
- left 40%: ranked swings, alerts, staged picks with one consistent selection;
- right 60%: one large selected D1/M5 chart, trigger/setup context, and Focus
  action;
- bottom or collapsible panel: D1 write-up using Market Journal service;
- incomplete sources render as a short `Partial recap` banner with a details
  expander, never raw Python in the main narrative;
- preserve original day ordering/ranking and the one shared chart owner;
- surface that the process-scoped alert list may be incomplete after a restart
  or midnight rollover.

### 8.4 Weekend Prep

Keep the guided five-step routine and persistent Done/Skipped state. Improve
orientation and move every expensive refresh off Qt.

| Step/tab | Redesign |
|---|---|
| **Week in review** | Summary cards for takes/skips/rejects/blind spots plus drill-down tables. Refresh on a worker and retain last-good content. |
| **Focus pick review** | One joined pick/outcome table, then collapsible veto/like/performance/verdict views. Show maturity and as-of state clearly. |
| **Walk-away** | Results and pending tag proposals side-by-side; weekly is default, all-pending remains an explicit toggle. |
| **Discovery → H1** | Same board contract; show offered/measured/filtered counts and adoption result. |
| **Discovery → D1** | Same layout and clear timeframe-specific gate explanation. |
| **Discovery → Monthly** | Make short-history exclusions prominent and state that the forming month is absent. |
| **Week ahead** | Render the report beside a concise checklist/risk-window summary; keep manual-only execution. |

The left step rail should show status, last refreshed time, and row count. Large
full-width buttons become compact primary actions in the step header. Empty
steps explain what input is absent and what will happen when Refresh is pressed.

### 8.5 Universe

This is a low-frequency data-maintenance tool, not a daily trading destination.

- Move it to Settings → Data & Universe / Advanced Tools or a collapsible System
  nav group.
- Replace the giant plain-text symbol pane with a virtual searchable table and
  counts by source/filter.
- Present pasted-list comparison as a real diff: only-here, only-pasted,
  intersection, invalid.
- Keep merge explicit, previewed, and confirmed; never auto-remove user names.
- Show rebuild source, start/end, last success, and output path.

### 8.6 Auto Pilot

The current page is operationally understandable but gives the ON/OFF button
too much space and mixes status with configuration/maintenance.

- Keep mode/state, next action, last successful scan, current blocker, and
  immediate `Reconnect` / `Run scan now` actions.
- Rename any remaining `Mini PC Mode` language to current Auto Pilot terms.
- Move weekday auto-arm/schedule and scanner configuration to Settings.
- Move rebuild watchlists/universe and durable warming to Advanced Maintenance.
- Keep report generation only if it is a real operator action; otherwise make it
  a diagnostics/export action.
- Collapse the activity log by default, with errors automatically expanded.

### 8.7 A.I. Summary

Move provider, model, and secret management to Settings → AI. The horizontal
scope checkbox run currently clips even at the target viewport.

- Replace the long checkbox row with templates: `Daily desk`, `Market journal`,
  `Trade review`, `Setup research`, and `Custom`.
- Custom scope uses a wrapping grid with counts/caveats.
- Keep **Validated Summary** and **Exact Evidence Preview** as the two result
  tabs; Evidence Preview should be the default before Generate.
- Show source counts, time range, thin-sample/discovery caveats, and output path
  in a fixed metadata band.
- Keep last-good output visible during generation.
- Market-journal synthesis is launched from Market Journal; this page remains
  the broader evidence/research reader.

### 8.8 System Health

The KPI strip and evidence/job/timing split are valuable. Improve actionability:

- summary column becomes one concise sentence; full evidence stays in detail;
- add `Owner / Next action` and `Last changed` columns;
- severity and freshness sort by default;
- selecting an unhealthy row opens a structured remediation card, with a deep
  link to the relevant Settings section;
- **Evidence** keeps human-readable proof;
- **Jobs** shows last result, next eligible run, attempts, and failure reason;
- **Phase timings** remains the timing table and gains baseline/slowest-stage
  comparison;
- raw paths/logs remain behind `Advanced details`.

---

## 9. Research page — every tab

Research is information-dense and generally the strongest part of the
application. The goal is to preserve dense tables while replacing raw dumps and
unused space with selected-row explanations.

### 9.1 Top-level Research tabs

| Tab | Review and decision |
|---|---|
| **Master AVWAP Market Prep** | Keep the two-sided lists, but make them virtual searchable tables. Convert key report sections to structured cards/tables; keep raw report behind `Exact report`. |
| **Setup Tracker** | Preserve as the primary performance source. Keep metric strip and plain-English summary, but prevent the summary from creating an internal scroll trap. Move wide sample lists into selected-row detail. |
| **Setup Playbook** | Preserve the left setup index and exact mechanics article. Add search/filter and in-document anchors; do not simplify away the actual mechanics. |
| **Move Forensics** | Keep filters and result table. Format numeric values to trading-readable precision. Replace the duplicate raw text panel with a selected-row evidence/explanation pane; retain raw export under Advanced. |
| **Day Trade Tracker** | Preserve metric strip, dimensional tabs, and learning rules. Use selected-row detail for examples/explanations instead of making every row excessively wide. |
| **Ticker Lookup** | Keep one-symbol lookup, but show staged progress, elapsed time, cancel, source-by-source completion, and last-good results. The tested SPY lookup remained at `Looking up SPY…` for more than nine seconds with no progress. |
| **Price Alerts** | Keep the operational alert list but move ntfy server/topic/token and mode policy to Settings → Notifications. Reuse the same operational model in Focus/Alert Center rather than duplicating configuration. |
| **Research Warehouse** | Show enabled/disabled/reachable state, last-known dataset counts, last successful write, spool backlog, and refresh status. A blank page is not an adequate disabled/unavailable state. |

### 9.2 Setup Tracker inner tabs

| Inner tab | Decision |
|---|---|
| **Current Picks** | Preserve ranked picks and selected setup detail. Keep actionable columns pinned; move verbose sample context to detail. |
| **Short-Term 1–2d** | Preserve its distinct mark-to-market question and best-first order. Put the definition in a help tooltip after first use. |
| **Human Picks** | Preserve same-horizon comparison. Make cohort/source and maturity obvious. |
| **Setup Types** | Preserve core n/win/R metrics; selected row opens mechanics and examples. |
| **Last 30 Days** | Preserve NEW/RISING flags and pin behavior; add explicit as-of/window dates. |
| **Playbooks** | Preserve setup × stop/exit ranking; selected row shows exact policy evidence without widening all rows. |
| **Scan Factors** | Preserve factor evidence; group related columns and explain denominator/sample scope in detail. |
| **Tier Performance** | Preserve S/A/B comparison and maturity; keep discovery/thin labels adjacent to n. |
| **Catch Rate** | Preserve opportunity-to-pick accounting; selected detail shows missed winners and why rows were eligible. |

### 9.3 Day Trade Tracker inner tabs

Use the same table shell and selected-row explanation for all nine performance
dimensions:

- **Bounce Types**;
- **Combos**;
- **Time of Day**;
- **Environment**;
- **RRS**;
- **Swing Focus**;
- **Swing Bucket**;
- **Swing Family**;
- **Swing Traits**.

For **Live Alert Rules**, keep it visually distinct as the current policy
readout. It must state whether a row is live, shadow, muted, or discovery and
must not imply a table edit changes a rule unless a real controlled writer
exists.

### 9.4 Ticker Lookup inner tabs

| Inner tab | Decision |
|---|---|
| **Overview** | First partial result; show quote/provenance/freshness and key setup context. |
| **Earnings / Events** | Structured event table with actual vs projected dates clearly distinguished. |
| **News** | Show source/time and a clean unavailable state; no indefinite spinner. |
| **Report** | Exact raw/generated report for audit, secondary to structured tabs. |

---

## 10. Settings redesign and offloads

Settings should own preferences, credentials, schedules, and maintenance—not
live decisions. A workflow surface may show status and link to its setting, but
should not duplicate the setting.

### 10.0 Current Settings tabs

| Current tab | Finding | Disposition |
|---|---|---|
| **General** | Presentation, data path, and durable-store warming occupy only the top of a large page. | Split these controls into Appearance & Workspace, Data & Connections, Performance, and Maintenance. |
| **BounceBot** | Useful connection/tuning/type controls are separated by a large vertical blank region. | Move them into Automation & Scanners with compact sections; retain live status on Trading Desk. |
| **Testing Plan** | The exact operator runbook is valuable, but it is a long document rather than an everyday preference. | Keep it read-only under Maintenance & Advanced / Operations, make it searchable, and allow Health to deep-link to the owed step. |

### 10.1 Proposed Settings categories

| Category | Contents |
|---|---|
| **Appearance & Workspace** | Theme, scale, compact density, explain mode, nav collapse/favorites, default page, default Desk tab, Veto default/remember behavior, optional split view, reset saved layouts. |
| **Data & Connections** | Home folder, IB host/port/client id, Questrade refresh token, IBKR Flex token/query id, data-source status, broker/account labels, research/DAS paths as appropriate. Secrets remain in the existing secret-bearing local settings file and are masked. |
| **Notifications** | ntfy server/topic/token, routine push policy readout, urgent wake-channel test, price-alert monitoring policy, sound preferences. |
| **Automation & Scanners** | Auto-arm/schedule, Auto mode defaults, BounceBot RRS sensitivity/timeframe, bounce-type switches, open-burst digest window, scanner presentation preferences. |
| **AI** | Provider, model, secret, local model endpoint, default evidence templates, opt-in free-text policy, output location. |
| **Performance** | Standard/Snappy mode, cache budget, prewarm toggle, current cache use, clear caches, watchdog enabled, restart requirement, comparison summary. |
| **Maintenance & Advanced** | Universe build/compare/merge, durable-store warm, watchlist rebuild, raw paths/logs, Testing Plan, export diagnostics, layout reset. |

### 10.2 What moves off workflow pages

| Current location | Move | Keep in place |
|---|---|---|
| Journal Health broker drawer | Tokens, ids, host configuration | Coverage/reconcile status and `Repair in Settings` link |
| Research → Price Alerts | ntfy credentials and always-on policy | Live alert inventory, add/remove/rearm/check |
| A.I. Summary | provider/model/key | scope choice, evidence preview, generation |
| Auto Pilot | auto-arm schedule and rebuild maintenance | current state, blocker, reconnect, scan now |
| BounceBot tab | tuning/type switches/configuration | slim live status and recent activity |
| Universe top-level page | build/compare/merge maintenance | optional read-only universe freshness link in Health |
| Settings → Testing Plan | remains under Advanced Operations | health may deep-link to the exact owed step |

### 10.3 Settings interaction rules

- Search finds setting labels and their current category.
- Changed settings show `Apply`/`Restart required` honestly; presentation-only
  changes may apply immediately if they are known safe.
- Dangerous maintenance actions are separated from ordinary toggles and require
  a preview/confirmation.
- Secret fields are masked, never logged, and never echoed in status text.
- A setting shown read-only because another owner controls it identifies that
  owner and provides the correct link.
- Settings never becomes a live dashboard; status summaries are concise and
  link back to the workflow page for detail.

---

## 11. Speed plan

### 11.1 Standard mode: fix known blocking work first

Extra hardware cannot compensate for work performed synchronously on Qt. These
repairs precede the experimental mode:

1. Move Weekend Prep week-review joins and every similar file aggregation to an
   owned worker. Keep last-good output visible and make the Refresh action
   single-flight.
2. Resolve Focus mover state once per `(symbol, side, source-series identity)`
   per refresh and hand the result map to all four editors. Do not remeasure it
   per chip.
3. Instrument page select, inner-tab select, model apply, layout, first paint,
   chart-request, and chart-ready. The stall watchdog should name an interaction
   id so event-loop-only samples can be tied to the click that caused them.
4. Convert hot `QTableWidget` rebuilds to model/view or incremental diffs,
   starting with Health and any live lists still creating every cell on refresh.
5. Never clear a populated page while a refresh is running. Apply one immutable
   result on Qt after all parsing/formatting completes off-thread.
6. Keep page widgets alive, but avoid reparents/rebuilds during ordinary
   navigation. The selected Desk layout should be stable for the session;
   changing the mode is a Settings action, not a daily top-bar toggle.
7. Audit every `reload()` reachable from a click or page selection. A function
   reading CSV/JSONL/SQLite, building large Python rows, or running analysis must
   not execute on Qt even if it is "manual".
8. Keep the current stylesheet/dynamic-property, bounded-GC, settings cache,
   review-event cache, bar-dict cache, and warning-rate-limit fixes.

### 11.2 Experimental `Snappy` mode

Snappy mode spends spare RAM and CPU on bounded background preparation. It does
not increase IB request rates, alter scanner schedules, precompute decisions, or
run expensive work on Qt.

Recommended contract:

- explicit opt-in under Settings → Performance;
- default additional cache budget **2 GB**, selectable 1/2/3 GB, hard maximum
  **3 GB above Standard** until a soak proves more is safe;
- absolute process guard initially **5 GB private bytes**, with eviction before
  the guard and an automatic return to Standard behavior under OS memory
  pressure;
- low-priority UI-preparation pool, initially 2–4 workers and measured before
  raising it; network/IB pacing lanes remain unchanged;
- prewarm only a bounded hot set: current symbol, next review queue items,
  current Focus names, visible Master rows, selected board rows, and recently
  viewed symbols;
- cache immutable chart payloads, bar dictionaries, levels/earnings payloads,
  formatted table rows, and last-good read models—not duplicate QWidget trees;
- size entries by measured bytes and evict LRU by budget, not an unmeasured
  fixed symbol count;
- run prewarm after the current interaction settles and yield when a live scan
  or high-priority chart request needs the pool;
- keep source/as-of metadata with every cache entry; a fast stale answer must
  still say it is stale;
- expose cache used/budget, hit rate, evictions, queue depth, and last pressure
  release in Settings/System Health;
- one-click clear/disable; no restart needed unless the implementation proves
  that worker-pool sizing cannot change safely at runtime.

### 11.3 Performance acceptance targets

| Interaction | Standard target | Snappy target |
|---|---:|---:|
| Warm top-level page switch, click → first stable paint | p95 ≤ 150 ms | p95 ≤ 75 ms |
| Warm inner-tab switch | p95 ≤ 100 ms | p95 ≤ 50 ms |
| Cached symbol selection → first chart paint | ≤ 250 ms | ≤ 100 ms |
| Uncached chart request → visible progress state | ≤ 75 ms | ≤ 50 ms |
| Veto shortcut → focused reason list | ≤ 50 ms | ≤ 50 ms |
| Capture commit → visible saving state | ≤ 50 ms | ≤ 50 ms |
| Desk mode/layout switch if used | ≤ 300 ms | ≤ 200 ms |
| Navigation-triggered stalls over 50 ms | none attributable to file/analysis work | none |
| Full-session hard gate | no stall > 5 s; < 60 s total blocked | same, with a goal of < 30 s |
| Memory after three hours | existing < 2 GB target | Standard use + configured budget; never above guard |

Targets are acceptance criteria, not promises. Capture a Standard and Snappy
trace from the same scripted workflow and one live session before choosing a
default.

---

## 12. Responsive and accessibility rules

The primary target is 2304 × 1392 logical pixels, but the app must remain usable
at the existing 1680 × 954 laptop gate.

- Use logical pixels and current UI scale tokens; never branch on physical 4K
  pixels.
- Toolbars wrap or use an explicit overflow; labels are never clipped to
  ambiguous fragments.
- Tables pin essential columns and move detail to a selected-row pane before
  adding horizontal scroll.
- Empty states explain source/action and do not vertically center a single line
  in a giant void.
- Compact density reduces padding and row height, not information or font
  legibility.
- Color is secondary to text/icon/state; red/green values retain sign/text.
- Focus order follows the visual order. All primary actions have visible
  shortcuts in the professional preset.
- Splitters remember per page/tab and have `Reset layout` in Settings.
- Automatic pane changes are limited to the documented contextual companion
  policy and never move the trader's current focus.
- No modal dialog for routine review/capture. Confirmations are reserved for
  destructive or ownership-changing actions.

---

## 13. Proposed implementation waves after approval

This is a dependency order for a future `plan.md` promotion, not authorization
to start.

### Wave U1 — primary desk

- Decide the Veto-first tabbed default and standalone Chart Review role.
- Resolve the arm-bar contract/source mismatch under ask-first.
- Implement contextual lower-row companions.
- Fit Veto/Like/Note modes without scroll at the target.
- Preserve keyboard, queue, and capture semantics with regression tests.
- Add 2304 × 1392 and 1680 × 954 layout screenshots/geometry assertions.

### Wave U2 — Market Journal and review loop

- Build the sticky composer/timeline shell.
- Attach current symbol on quick capture.
- Add selected-symbol shared chart and optional six-symbol grid.
- Add read-only machine-context and Trade Journal execution lanes.
- Add versioned AI synthesis presentation without changing the opt-in policy.
- Redesign AWAY Recap around selected chart + ranked lists.

### Wave U3 — remaining pages and navigation

- Group nav and rename Trade Journal.
- Apply empty-state/selected-detail rules to Focus, Strength, Weekend, Research,
  Auto Pilot, AI Summary, and Health.
- Move low-frequency Universe access into System/Settings.
- Verify every nested tab at both target viewports.

### Wave P1 — measured Standard-mode repairs

- Weekend Review worker fix.
- Focus mover-state batch/cache fix.
- navigation/paint instrumentation.
- incremental model/view conversions in measured cost order.
- run the fluidity workflow and a live-session soak.

### Wave P2 — experimental Snappy mode

- Implement bounded caches and low-priority prewarm behind one setting.
- Add memory pressure and cache observability.
- compare Standard/Snappy traces on identical workloads;
- keep experimental until the trader signs off on responsiveness and memory.

### Wave S1 — Settings consolidation

- Build the seven-category Settings shell and search/deep links.
- Move configuration one owner at a time; remove the old duplicate only after
  the new owner is verified.
- Keep operational status/actions on workflow pages.
- Audit secrets, restart behavior, and maintenance confirmations.

---

## 14. Required test matrix for a future build

### Automated layout/interaction

- 2304 × 1392 logical window (90%-width 4K target), standard and compact;
- 2560 × 1392 full-width 4K cross-check;
- 1680 × 954 laptop regression;
- nav expanded/collapsed;
- setups hidden/shown/full-width;
- every Trading Desk lower tab, with and without its companion;
- every Master AVWAP and Watchlists inner tab;
- every Trade Journal, Weekend Prep, Research, Setup Tracker, Day Trade Tracker,
  Ticker Lookup, AI Summary, Health, and Settings tab;
- long labels, maximum reason/claim counts, empty/populated/error/last-good states;
- no clipped primary action, no ambiguous elision, and no unexpected horizontal
  scroll at target width;
- Veto/Like/Note keyboard sequences work while Capture is hidden and visible;
- splitter state survives page changes, restart, scale change, and reset;
- source/age/unmeasured banners retain a fixed readable slot;
- Market Journal failed write retains text and displays `NOT SAVED`;
- entries/exits display read-only and link to the Trade Journal owner;
- settings deep links land on the intended category without changing values.

### Performance workflow

Script the same sequence for Standard and Snappy:

1. launch to Veto Desk;
2. review 25 queued symbols with D1/M5 chart changes;
3. veto 10, like 5 with reasons, note 5;
4. open Master Setups, Watchlists, Industry, and RS Window;
5. switch Focus, Strength, Market Journal, AWAY Recap, Weekend Prep, Research,
   Trade Journal, Health, and Settings;
6. select populated rows and open shared charts;
7. leave the desk active for three hours with normal timers;
8. compare stall log, interaction spans, cache metrics, memory, CPU, and Qt
   warnings.

### Manual professional-use scenarios

- premarket: confirm the full thesis, Focus, source freshness, and setup roster
  can be read without hunting;
- active alert burst: keep Capture/Veto open and process the queue entirely by
  keyboard;
- Master consult: move to Master and back without losing queue/chart state;
- AWAY return: identify best candidates, chart them, adjust Focus, and write D1
  context from one page;
- intraday journal: add a thought in under five seconds and see it in the
  session timeline;
- after close: compare thoughts to entries/exits and generate an opt-in summary;
- weekend: complete all five steps without a GUI freeze or unexplained blank
  page;
- failure drill: stale bars, source outage, journal write failure, and bad
  credential each show the correct owner and next action.

---

## 15. Trader decisions required before promotion

Recommendations are supplied so none of these require inventing a design later:

1. **Trading Desk default:** adopt `Veto / Alerts` first and `Master AVWAP`
   second, with simultaneous Workspace split optional. **Recommended: yes.**
2. **Capture presentation:** show one full-width action mode at a time
   (`Veto | Like | Note`), remembering Veto. **Recommended: yes.**
3. **Arm bar:** restore it to Armed above the inventory, matching the documented
   one-row-under-charts contract. **Recommended: yes; ask-first required.**
4. **Standalone Chart Review:** retain only as a frozen/manual review theater;
   otherwise fold it into Trading Desk. **Recommended: retain initially and
   measure actual use.**
5. **Market Journal charts:** one large selected chart with up to six symbol
   selectors, plus optional 2 × 3 grid. **Recommended: yes; chart size wins over
   six simultaneous small charts.**
6. **Trade Journal Fees:** fold into Analytics/Reports or retain a dedicated tax
   tab. **Recommended: fold unless the dedicated export is used weekly.**
7. **Universe:** move out of primary nav into Settings/System tools.
   **Recommended: yes.**
8. **Snappy budget:** start with +2 GB, selectable through +3 GB, absolute 5 GB
   process guard until soak-tested. **Recommended: yes.**

Until those decisions are accepted and this proposal is promoted into
`plan.md`, the repository's active build and live-validation sequence is
unchanged.
