# GUI design language

Status: visual direction approved by the trader, 2026-09-26. Implementation
and the Windows font-startup fix are integrated with current main. The trader
explicitly requested today's work be merged on 2026-09-27. Owner: the trader.
Restart and detector/scoring/alert behavior changes are separate authorizations.

Sketch source: `design/gui-sketch.html`; standalone preview: `design/gui-preview.html`.
The four clickable views are Desk, Day Review, Journal and Research. Full-size
preview checked at 2560x1440 logical pixels (4K at 150% scaling), plus the normal
preview width. The running desk reports 96 DPI (100% scale); it was windowed
at 1656x1019 during the probe. Full-screen 4K live validation is still owed.
The first Journal implementation has five detail sections, a selected-trade
header and a clear empty state. Editors stay alive across section switches;
their existing saves and one-click tag confirmation are retained. The existing
60/40 opening splitter ratio and user drag behavior remain unchanged in this
first pass (the proposed 45/55 ratio below is still a design option).
Day Review now keeps its session controls outside the page scroll and offers
named section jumps. Its guided-review action also stays in reach while scrolling.
The guided walk hides those controls while active. Journal refresh preserves the
selected trade and detail section; an in-memory filter keeps drafts while that
trade remains visible. The combined area checks pass (459 tests), followed by
119 final-area checks and 28 Weekend checks using the rendered theme; Ruff is
clean. These runs overlap and are not a full-suite count. Journal and Day
Review have dark/light viewport checks at 1920, 2560 and 3840 pixels wide, using
test data. Research checks the full shell, custom-date fit and non-overlapping
sections at those widths. These are geometry checks, not a live 4K session.
Research now has grouped local navigation to all 12 existing destinations, with
Results first. Its result/evidence split opens near 65/35; band summaries,
Looking back and the measured report stay on the same scrollable page. The
population, horizon, window and environment controls keep their existing meaning.
Full-suite/live/performance checks are tracked in STATUS.md. No performance
improvement is claimed from layout tests.

A 2026-09-27 offscreen spot benchmark used identical staged data for Journal,
Day Review and Research at 3840x2160, two samples per operation. Page-show call
times stayed near baseline (about 4-5 ms for Journal/Day Review and 17-18 ms for
Research). Some construction/tab timings increased; background-read deadlines
occurred in both versions. This is diagnostic evidence, not performance signoff.
Exclude `weekend_prep` from this benchmark: its existing refresh workload starts
weekly Market Prep, including external AI and source-relative cache/output writes.
That side effect occurred in the baseline run; the comparison run excluded it.

The supporting-page pass is built: A.I. Summary now pairs a setup pane
with its existing result/evidence tabs; key settings fold away without losing
typed input. System Health pairs checks with evidence and wraps long audit
metadata. General Settings groups Appearance, Risk and Mentor, and Storage.
Auto Pilot pairs activity with staged picks below its status/manual controls.
Universe retains its list/compare split with clearer build-filter wording.
Weekend Prep keeps step titles and completion controls fixed while long bodies
scroll; Focus Review retains its full-height split. The verdict belongs to Week
Review and tag coverage stays beside the tagging work.
Shared headers use transparent surfaces; tabs, splitter handles and keyboard
focus have static theme cues. Compact Desk filters now wrap above the chart;
the longs-off banner has its own wrapped line. Capture and Journal remain clear
in the drawer tab bar. The daily Watchlist and Manage lists editor have distinct
names. The same controls, filter meanings and saved splitter keys are retained.
Native windows passed empty/populated geometry checks at 1920, 2560 and 3840
pixels, at 100% scale; 43 focused Desk checks passed after the final test setup
was corrected for an already-expanded classic window. Live validation remains.
Theme application compares the rendered stylesheet first, so an unchanged
appearance does not repolish every Qt widget when another setting changes.

Validation: complete file-isolated coverage passed on the integrated tree:
13,446 tests passed, 14 skipped and 72 subtests passed across all 924 collected
files, with clean process exits. The outer run used eight pytest workers;
each file ran in a fresh process. Five files were rechecked after import-path
setup and fixture-cleanup corrections. Source and frozen selftests both passed
105/105; smoke 7/7; Ruff clean. Earlier long combined-process runs hit Qt worker
crashes; fresh-process coverage does not claim to fix that harness issue.
Live 4K usability and resource measurements remain in docs/GATES.md. The trader
explicitly requested merge before that next session; no live performance signoff
is claimed.

Windows startup uses Qt's GDI font engine by default, configured before
QApplication. Explicit QT_QPA_PLATFORM overrides are preserved. Native scratch
construction dropped from over 270 seconds blocked in DirectWrite font loading
to 9.8-10.9 seconds with GDI; actual next-launch timing remains a live check.

## Aim

Make the next useful fact easy to see. Keep every stored detail reachable.
The desk should feel calm, precise and fast even as new features arrive.
Beauty comes from type, alignment, spacing and clear ownership of information.
The target is a 9/10 experience; that is a design goal, not a measured result.

## What this assessment covers

Source baseline: main at `1051c1ce`. Read the current navigation, theme, Trading
Desk, Day Review, Journal, Research and the supporting page controls. This is a
source-based assessment with a live browse of Desk, Day Review, Journal, Research
and Weekend Prep. Startup took several minutes before a window was available.
Research and Weekend Prep visibly compressed multiple sections into clipped strips;
Journal's plan editor dominated the inspector even before a trade was selected.
Day Review was observed loading; that state is not proof its source data is absent.
Supporting pages below have source review only unless noted. The sketches use
illustrative data, not portfolio results. The limited benchmark above does not
establish live performance acceptance.

- The Desk already has useful density, chart context and saved splitters. Refine
  its hierarchy rather than replacing it with large dashboard tiles.
- Research has 12 peer tabs and a long explanatory paragraph above them
  (`scripts/ui/panels/research_panel.py`). Builder tools and trader results need
  much clearer separation without moving required facts out of trader pages.
- Day Review has a glance strip and guided walk already. Its main page also stacks
  report, story, calls, charts, misses, trades, ideas and plan
  (`scripts/ui/panels/day_review_panel.py`). Make the walk prominent and provide
  named jumps through the existing full record.
- Journal already has a table/detail split and shared account/currency/date
  filters. The detail pane stacks plan, legs, tags, suggestions, notes, AI,
  structured review and corrections (`scripts/ui/panels/journal/trades_tab.py`).
  Give each editing job its own clear section and save feedback.
- A shared light/dark palette exists in `scripts/ui/theme.py`. Extend it instead
  of inventing a second theme. Keep the established chart-line colors.
- The source navigation has no page named Fable. Its identity is unverified;
  do not rename A.I. Summary or redesign a presumed Fable page on that guess.

## Core principles

1. **Answer, work, evidence.** Each page has one purpose, a short answer/status,
   a main work area and a consistent route to exact evidence.
2. **One obvious next action.** Give one action visual priority per work area.
   Secondary actions remain labeled and available. Avoid repeated review prompts.
3. **Density where it helps.** Use tables for comparisons and prose for meaning.
   No oversized metric cards that push useful rows off the screen.
4. **Reveal detail without losing place.** Row selection opens a stable inspector.
   Back restores filters, selection, scroll and splitter positions.
5. **No lost information.** Compact columns, expandable sections, named page jumps
   and complete tables expose every existing field and action. No silent row cap.
6. **Truth stays attached.** Keep currency, population, timeframe, sample count,
   freshness and missing-data reasons beside their numbers. Unknown is not zero.
7. **Stable meaning.** A color, label, interaction or unit means the same thing
   across pages. An AI suggestion is visibly separate from the trader's words.
8. **Fast by construction.** Native Qt widgets, bounded worker reads, cached charts
   and incremental paints. No decorative GPU work, animation loops or web shell.

## Full-screen 4K is the primary workspace

Trader clarification: one 3840x2160 panel, dedicated to the desk. Design for that
whole workspace first. Physical pixels are not Qt logical pixels: at 150% Windows
scaling the workspace is about 2560x1440 logical pixels; at 200% about 1920x1080.
The live window currently reports 96 DPI (100%); retain the other scale checks
for later display changes. Never shrink a 3840-wide bitmap mock
and call that readable UI.

Use the width for simultaneous context. Desk: opportunity/alert lane, dominant
chart, selected-name evidence/Movers lane. Journal: trade list and a roomy selected
trade workspace. Research: local navigation, full-height result table, evidence
inspector. Day Review: retain the two-column story/words structure with named jumps.
Avoid making every page three columns merely because there is space.

Keep actions near their content so the mouse does not travel across the panel.
Allow the chart and tables to grow; cap prose at roughly 65-85 characters within
its column, without capping the whole panel or restoring the old narrow reader
bug. Essential work stays in the main window; no workflow depends on a second
monitor or a floating dialog. Preserve draggable, saved pane ratios.

For the Desk start near 25/50/25, then honor saved preferences. Show more rows on
4K, not more competing headings. Journal uses roughly 45/55. Research uses a
180-220 logical-pixel local navigator, with remaining width roughly 65/35 for
results/evidence. These are initial sizing proposals, not fixed pixel layouts.

## App shell and navigation

Keep all ten current destinations and their identities. Group the destinations:

| Group | Pages | Purpose |
| --- | --- | --- |
| Trade | Trading Desk | Find, inspect and track opportunities |
| Review | Day Review, Journal, Weekend Prep | Learn, keep records, prepare |
| Explore | Research, Universe, A.I. Summary | Examine evidence and coverage |
| System | Auto Pilot, System Health, Settings | Control operation and diagnose |

The live desk has a compact top bar: Desk, Journal, Day Review, Research and More.
Keep that familiar shape; use the grouping above inside More rather than adding
a permanent app-wide sidebar. Retain currently hidden/configured destinations
without exposing extra pages solely for symmetry. Research gets a local navigator.
Built: More groups Review, Explore and System, with a fallback for future pages.
Empty groups disappear when their pages are hidden. The More button names the
active secondary page and mirrors its review badge. Main tabs and More accept
keyboard focus. Twenty navigation/compact-layout checks pass; selection still
routes through the window's existing owner.
Page header: title,
scope/date, freshness and one primary action. Keep page filters below the header.
Show a compact global mode/connection strip; detailed health belongs in Health.
Critical stale-data or failed-save messages remain visible at the affected work.

Use page identity rather than array position for navigation. Preserve all existing
cross-page links, especially Desk -> Setup Tracker, Journal -> Watchlist Positions,
Day Review -> a selected trade, and overnight summary -> the selected session.
Counts must describe their scope: `3 trades need review`, not an unexplained `3`.

## Visual tokens

Proposed values are logical pixels and need DPI validation in Qt before adoption.

| Token | Proposal |
| --- | --- |
| Canvas / surface / raised | Existing dark #0F1216 / #171B21 / #1E242C |
| Light appearance | Existing light tokens, with the same hierarchy |
| Primary / secondary text | Existing #E6EAF0 / #9AA4B2 dark tokens |
| Essential small text | Secondary token or stronger; never the dim muted token |
| Selection | Existing accent-soft background with a thin accent edge |
| Accent | Existing blue for selection/navigation; color is not decoration |
| Semantic colors | Existing long, short, caution, study and rejected-today meanings |
| Type | Segoe UI; 14 body, 12 metadata, 16 section, 22 page heading |
| Numbers | Tabular figures; right aligned; stable precision per column |
| Spacing | 4, 8, 12, 16, 24; page inset 16, section gap 24 |
| Corners / borders | 4-6 radius, 1px separators; no nested box outlines |
| Row height | 30 compact / 36 comfortable; text size stays readable |
| Focus | Visible keyboard focus on every action, distinct from selected row |

Use restrained surfaces. A panel boundary should separate jobs, not wrap every
label. Default to quiet rows, aligned columns and short section headings. Profit,
loss, side and warning retain written labels/signs, so color never carries meaning
alone. Preserve the existing rejection marker and chart palette exactly.

## Shared interaction patterns

- **Scope bar:** account, currency and date remain visible when needed. Research
  population controls are explicit. Applying a filter updates its shown count.
- **Table + inspector:** single click selects; Enter opens detail; Escape returns
  to the prior context unless an unsaved edit needs a decision. Double-click is
  a shortcut, never the only way to discover an action.
- **Columns:** a compact default, a full field picker, saved widths/order and a
  reset. Hidden columns remain inspectable. Long text is selectable and copyable.
- **Evidence:** show source time, population, horizon and method with the result;
  expand to exact contributing rows. Keep observational/shadow labels in view.
- **Editing:** show dirty, saving, saved or failed next to the editor. Never claim
  success early. Preserve typed text on failure and make journal failures loud.
- **Empty states:** distinguish no rows, no matches, loading, stale last-good data,
  unavailable source and failed read. Give one relevant route to recover.
- **Scroll:** Day Review keeps one page scrollbar and section jumps. Data workspaces
  may use a table scrollbar and an inspector scrollbar; avoid scroll areas within
  scroll areas. Headers and current scope stay easy to find.
- **Size:** use saved splitters. At narrow widths, offer list/detail navigation
  instead of crushing columns. Validate 1366x768 and 1920x1080 at 100/150% DPI,
  plus full-screen 3840x2160 at the trader's actual Windows/Qt scale as the primary
  acceptance case. Never solve overflow with tiny text.

## Each page

### Trading Desk — what deserves a closer look now?

Keep the existing setup/watchlist workflows, alert streams and chart behavior.
Top: compact mode, freshness and market-context strip. Main: opportunity table,
the selected name's existing chart, and contextual details. Preserve the user's
splitter/layout choice. The new hierarchy should fit inside existing Desk modes.

Use fewer competing borders and button styles. Keep Watchlist (daily workflow)
distinct from Manage lists (list management). Keep Setups, Theta Plays, Industry
Board and RS Window reachable. M5 alerts, D1 alerts, Movers and Mentor retain
their own identities and services. No new prioritization or suppression logic.
The existing Working Lately line remains visible and links to full evidence.
Selecting a setup must not add it to Focus; adoption stays an explicit action.
The trader approved layout-only edits inside `trading_desk.py` and
`alert_center_panel.py` on 2026-09-27. Alert logic, rankings, timers, service
ownership and saved panel sizes remain outside this change.

### Day Review — what happened, and what should I learn?

Use the current label Day Review; the user also calls it day recap. Lead with the
selected session, final/provisional state, the existing glance strip and one
`Review my day` action. `Show` remains a secondary presentation option.

Below: section jumps `Overview`, `My calls`, `My trades`, `Picks and passes`,
`Ideas`, `My plan`.
These are anchors into the same complete page, not five disconnected stores.
Overview pairs the story and relevant chart with the trader's exact words.
Preserve the saved 55/45 story/words split and the single-scroll design. Reduce
duplicate calls to start the walk. Existing detail/report-card disclosure stays.

The review walk remains a focused sequence. Completing an edit uses its existing
save path. Full history, external forecast, theses, picks/passes, walkaway rows,
plan and night digest remain reachable through named sections. Links open the
exact trade or symbol/session and restore the user's place on return.

### Journal — one accurate trade record, one review at a time

Keep account/currency/date scope above Trades, Calendar, Analytics, Health and
Fees. Keep the current eight trade columns as the compact baseline. Make the
tag-review filter obvious and retain its exact existing meanings.

Give the selected trade a strong header: symbol, side, dates, record status,
P&L/currency and R if measured. Divide its inspector into `Review`, `Plan`,
`Executions`, `Notes and tags`, and `Corrections`. Each section keeps its real save
boundary; do not introduce a universal Save that can partly succeed unnoticed.
AI tags stay provisional until the trader accepts them. Bulk confirmation stays
explicit with its current safeguards. Missing opening fills stay visibly excluded
from totals. Fees, FX, tax treatment and account provenance remain accessible.

Calendar opens a dated trade selection; Analytics opens the rows behind a result;
Health explains record gaps; Fees retains its detailed ledger. Positions still
links to the single Watchlist Positions view instead of duplicating ownership.

### Research — ask a question, then inspect its evidence

Keep Results as the initial destination. Replace the overflowing peer-tab rail
with a small grouped local navigator, not a new dashboard of summary cards:

| Group | Existing destinations, all retained |
| --- | --- |
| Results | Results |
| Setups | Setup Tracker, Setup Playbook, Setup keys |
| Studies | Move Forensics, Day Trade Tracker, Long lab, Retest entry |
| Tools | Master AVWAP Market Prep, Ticker Lookup, Price Alerts |
| Data | Research Warehouse |

Results shows explicit Bot setups / My trades and Swing / Day trading controls;
four populations never pool. The selected population determines the headline:
swing win rate with n and Wilson lower bound; day-trade held-run score with its
coverage. Bot windows retain their own session scope; the personal trade window
uses closed_at. Exact definition and evidence remain beside the number.

Replace the long permanent introductory paragraph with a short `Results for the
trader; studies are research` note and a full help disclosure. Research reads the
same Working Lately snapshot as the Desk. No new metric, ranking or promotion.
Every study shows question, scope, data coverage, result table and method in that
order. Label shadow output next to the result, not only on an About page.

### Weekend Prep — finish a useful weekly routine

Lead with the current week and existing ritual progress. Give the active step
the main area; retain direct access to the full material. Group week review,
tag checks, strongest/weakest boards, walkaway and week-ahead prep by purpose.
Keep explicit Mark done / Skip this week. Expensive refreshes state their scope;
Refresh everything is secondary. Adoption into swing Focus stays explicit.
Built: titles and Mark done / Skip remain outside each step's body scroll. Focus
Review keeps its existing full-height splitter and 75/25 opening ratio. The week verdict
lives within Week Review; Tag Week owns its coverage line. All six steps retain
their current status, refresh, selection and action paths.

### Universe — what is covered, and where did it come from?

Show selected universe, count, as-of time and source status above the table.
Keep source breakdown and exclusions in an inspector. Separate inspection from
Build and Merge actions. Pasted comparison leads to a visible diff before the
existing merge action. Copy tickers remains easy. Never auto-remove typed names.

### Auto Pilot — what is running, and what happens next?

Lead with real mode, connection, current job and next scheduled work. Below: the
existing recent activity and staged picks side by side, with the staged table
using the available height. Keep Add to Focus explicit. Put scan,
rebuild and write-report controls in a labeled manual-actions section. Show a
failed publish with the last good report still available. No timer/job duplication.

### A.I. Summary — what did the evidence support?

Keep Validated Summary and Exact Evidence Preview paired. Show generation time,
evidence scope and validation state above the text. Keep Build Evidence Preview
and Generate Advisory Summary explicit. Provider/model and evidence scopes live
in a scrollable left setup pane; the result reader takes the remaining width.
Key management folds open within setup; its values stay masked. Link to Day Review for the actual
session. Missing narration never replaces deterministic facts with a made-up story.

### System Health — what needs attention?

Lead with actionable failed/degraded/stale items, then the services table. Select
a service to inspect Evidence, Jobs and Phase timings beside the list. The full
description folds under `What is checked`; audit metadata stays visible and wraps.
Keep healthy services
available but visually quiet. Explain what failed and when the last success was.
Do not equate skipped, failed and degraded. No new background probes for styling.

### Settings — change one thing with confidence

Keep General, BounceBot and Testing Plan. Group General by appearance, risk,
storage and Mentor: Appearance on the left, Risk and Mentor above Storage on the
right. Show units, current values and a short description beside
controls. Separate connection actions and store-warming from presentation settings.
Use existing save semantics and clear feedback. Testing Plan remains complete.
Detector-related settings require a separately authorized implementation scope.

## Performance contract

The HTML sketches are separate design artifacts. Runtime changes must preserve:

- Native PySide6/QWidget rendering; no WebEngine replacement, blur, animated
  gradients, backdrop effects, decorative canvas or continuously running animation.
- Shared `theme.qss` variants, applied outside hot refresh paths. No per-row
  stylesheet construction. Avoid a QWidget per cell in large tables.
- Existing model/view diff updates and stable row identity. Keep selection and
  scroll through refreshes. Never rebuild the whole table for a changed value.
- Worker-owned disk/database reads and result shaping. Visible UI consumes a
  bounded payload. Reuse ChartDataService and completed cached bars.
- One existing owner per service/timer/job. A hidden page stops presentation work
  when safe, not data collection or scheduled work it happens to own.
- Lazy detail/chart creation with correct shutdown; latest request wins during
  navigation, stale results do not overwrite the selected session.

Before and after each runtime phase, measure on the same machine, same stored
dataset and same display scale. Record warm navigation, selection latency,
Qt-thread stalls, idle CPU/GPU, memory and service counts. Targets: warm page or
row response under 100 ms at p95; show a loading state promptly for cold reads;
no new periodic GPU activity; no sustained idle CPU increase beyond baseline
noise. These are acceptance targets, not claims that this sketch meets them.
Repeat the existing desk_perf_report comparison during the agreed live session.

## How features fit as the app grows

For each new feature, name its user question, home page, primary/secondary/detail
level, source of truth, loading/empty/error states and owner of expensive work.
Add it to an existing section first. A new top-level page needs a distinct daily
job; a new tab needs a distinct task, not merely another data file.
Every new displayed field must have a full-detail route and a copy/export route
where the existing feature supports one. No fact should require opening logs.

Update this file's page map when a feature moves. Keep an action inventory during
implementation: old control -> new control -> test/deep link. Check off every
existing action, not only the first-screen actions, before removing old UI.

## Delivery order and acceptance

1. Review these sketches. Agree on hierarchy, density and which view feels best.
2. Shared typography/spacing/header/table components and Journal inspector.
3. Research navigation and Results, preserving snapshot semantics and links.
4. Day Review hierarchy/anchors, reusing the current walk and one-page record.
5. Desk polish, then the supporting pages using the same components.

Expected runtime areas: `scripts/ui/theme.py`, `theme.qss`, `app.py`, common
widgets and the named panel modules. Inspect each scope before editing; any
detector/scoring/alert-bearing file still requires ask-first authorization and
the required golden fixtures. This design does not override docs/RULES.md.

Validation for implementation: Qt navigation/selection/filter and save-failure
tests, existing page-spec and panel tests, light/dark and DPI visual checks,
performance comparison and a full action inventory. Full suite with offscreen Qt
and required worker limits plus Ruff before a commit; obey night windows. Frozen
selftest before merge. Live-session validation remains required for performance
signoff; the trader requested this delivery before that session. No restart is
implied by approval of a sketch or merge.

A successful design lets the trader find the next task within five seconds,
reach a result's source rows within two clicks, review a trade without losing the
list, and return to a session with context intact. Test these tasks with the
trader before calling the GUI a 9/10.
