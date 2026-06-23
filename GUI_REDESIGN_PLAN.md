# GUI Redesign Plan — TradingBotV3 (Windows desktop)

> Status: **FEATURE-COMPLETE (Phases 0–5 + most of 6).** The full PySide6 app in
> `scripts/ui/` ships the Trading Desk (Master AVWAP setups surface with a
> chip/score-bar table, BounceBot live panel, Theta Plays, Watchlists),
> the **Journal** (store-backed trade table, filters, analytics KPIs, and a
> setup-tags/notes annotation editor), **Research** (Master AVWAP Market Prep +
> a threaded **Ticker Lookup**), and **Settings** (theme, density, Explain
> toggle, and a durable-store warming button). Remaining: optional shipping
> polish only — PyInstaller `.exe` packaging, an Explain-mode legend/InfoDots
> pass, and retiring the legacy Tk GUI once it's been exercised on the mini PC.
> **Bot/trading logic is untouched** — this is a presentation-layer redesign
> over the existing engines.

---

## 1. Goal & audience

Make the app **beautiful, legible, and clear** for a real end consumer — usable
by **both new and experienced traders** — on **Windows only**.

The bots already work; the problem is the GUI. Today the entire main GUI is
**Tkinter** (`scripts/gui_app/app.py` and friends) and the outputs are walls of
text in scrolled-text widgets. That cannot be made "beautiful" to modern
consumer standards. We will replace the presentation layer with a modern,
themed desktop UI and turn text dumps into structured, color-coded, scannable
views.

### Design principles
- **Legible first.** Every output a trader reads should be a table, card, chip,
  or KPI — never a raw text dump. Strong typographic hierarchy, generous spacing.
- **Works for new *and* experienced.** New: plain-language labels, a legend /
  "What is this?" affordances, an optional **Explain mode** with inline tooltips.
  Experienced: dense sortable tables, fast filters, keyboard shortcuts, one-click
  copy lists. Same data, two reading densities.
- **Beautiful and calm.** A cohesive dark trading theme, semantic color
  (green = long/strong, red = short/weak, amber = caution), restrained accent,
  consistent components. No visual noise.
- **Windows-native feel.** Segoe UI font, native window chrome, crisp on HiDPI.
- **Front-and-center = active trading.** Master AVWAP scan + BounceBot live are
  the headline. Market Prep and Ticker Lookup move "to the side." Journal is its
  own tab.

---

## 2. Framework decision

**Adopt PySide6 (Qt 6 for Python).** Rationale:
- Tkinter cannot deliver a modern, beautiful UI; Qt can (QSS stylesheets,
  proper widgets, HiDPI, animations, real tables/charts).
- **PySide6** is the official Qt for Python, **LGPL** (cleaner for shipping a
  consumer app than PyQt5's GPL/commercial). Qt6 is current and HiDPI-clean.
- The repo already ships a Qt tool (`TickerMover.py`, currently PyQt5), so Qt is
  not new to the codebase; it can be folded into the new app later.

> Alternative if the team prefers: **PyQt6** is a near drop-in (import names
> differ). Do **not** stay on Tkinter, and do **not** mix Tk and Qt in one
> window. If PyQt6 is chosen, keep the same architecture below.

### New packages (Windows)
Add to `requirements.txt` (and a `requirements-gui.txt` if we want to keep the
headless/mini-PC install lean):
- `PySide6` — the UI framework.
- `qtawesome` — Material/FontAwesome icon fonts (crisp, themeable icons).
- `pyqtgraph` — fast embedded charts (sparklines, mini price/level charts).
- `qtpy` *(optional)* — thin abstraction so PyQt6/PySide6 are swappable.
- Packaging (dev-only): `pyinstaller` for the Windows `.exe`.

Fonts: ship/sanction **Segoe UI** (Windows default) for UI text and a monospace
(**Cascadia Mono** / **Consolas**, Windows-native) for numeric/price columns so
digits align. Optionally bundle **Inter** for a more branded look.

---

## 3. Information architecture (the layout the user asked for)

Two top-level reading modes, toggled in the title bar, persisted per machine
(replaces today's "full / simple" prompt):

- **Workspace mode (default, big screens):** a single combined **Trading Desk**
  with Master AVWAP and BounceBot visible together (resizable splitter), so the
  active trading scripts are front and center.
- **Tabs mode (small screens / less space):** the same content as tabs
  (`Master AVWAP` | `BounceBot`) so nothing is cramped.

### Top-level shell
A persistent **left nav rail** (icons + labels, collapsible) — not a row of
flat tabs — with these destinations:

1. **Trading Desk** ★ (default) — the combined active-trading workspace.
2. **Journal** — its own destination (existing journal functionality, restyled).
3. **Research** (secondary) — houses **Market Prep** and **Ticker Lookup**
   "to the side," as sub-tabs or a right-side drawer. Explicitly **not**
   front-and-center.
4. **Settings** — home folder, shared-data/Drive paths, tracker storage, theme,
   Explain mode, IB connection defaults, durable-store warming button.

A slim **global status bar** (bottom) always shows: IB/TWS connection state,
scan running/idle + last-run time, active watchlist, and the durable-store /
data-dir in use.

### Trading Desk (the centerpiece)
A combined workspace with three regions (splitters; collapsible):
- **Left control rail (per active script):** run/stop, connection, watchlist
  selector, filters (min score, min supports, max days to earnings, LONG/SHORT),
  market environment / RRS sensitivity / timeframe (BounceBot).
- **Center — the setups surface:** the redesigned, legible output (Section 5).
  Master AVWAP setups table/cards on top, BounceBot live alert feed below (or
  side-by-side in workspace mode; stacked tabs in tabs mode).
- **Right info drawer (collapsible):** detail for the selected setup — playbook,
  stop / profit-take, supports stack (incl. the new HV green/red levels &
  relative-AVWAP studies), theta play, factor impact, scenario outcomes. This
  replaces the many separate "Details" text boxes in today's AVWAP GUI.

Watchlist editing lives in a dockable panel / drawer (shared longs/shorts +
master swing lists), not consuming primary space.

---

## 4. Visual design system

Define once in `scripts/ui/theme.py` + a `theme.qss` (Qt stylesheet) and reuse
everywhere. No ad-hoc colors in panels.

### Color tokens (dark theme baseline)
- Surfaces: `bg/app #0F1216`, `bg/panel #171B21`, `bg/elevated #1E242C`,
  `border #2A313B`.
- Text: `text/primary #E6EAF0`, `text/secondary #9AA4B2`, `text/muted #6B7480`.
- Accent (brand/interactive): `#4C8DFF` (calm blue).
- Semantic: `long/strong #2ECC71`, `short/weak #FF5C5C`, `caution #F5A623`,
  `info #4C8DFF`, `neutral #6B7480`.
- Setup buckets: favorite = accent/gold, near-favorite = blue, study =
  muted/violet — consistent chips everywhere.
- Provide a **light theme** token set too (Settings toggle); build against tokens
  so it's free.

### Typography
- UI: Segoe UI 10–11pt; headings 13–18pt semibold; section labels 11pt
  uppercase tracked, `text/secondary`.
- Numbers/prices: monospace, tabular figures, right-aligned in tables.

### Spacing & shape
- 4px spacing scale (4/8/12/16/24/32). Card radius 10px, inputs 8px. Comfortable
  default density with a **Compact** toggle (experienced traders) that tightens
  row height and padding.

### Reusable components (`scripts/ui/widgets/`)
- `Badge`/`Chip` (side LONG/SHORT, bucket, tags, status).
- `KpiTile` (e.g., # favorites, avg score, IB status).
- `DataTable` (sortable, filterable, color-coded cells, sticky header, copyable).
- `SetupCard` (compact + expanded) for setups.
- `AlertFeedItem` for BounceBot live alerts.
- `Sparkline` / `MiniLevelChart` (pyqtgraph) for price vs key levels.
- `SectionHeader`, `Toolbar`, `EmptyState`, `Toast`/notification.
- `InfoDot` — hover for plain-language explanation (powers **Explain mode**).

---

## 5. Output legibility overhaul (the heart of this redesign)

The bots write structured data (priority rows, theta rows, study rows, tracker
stats) to files and return `run_result` dicts. Stop dumping text — render models.

### Master AVWAP — Setups surface
A primary **DataTable** (toggle to card grid) of ranked setups. Suggested columns
(reuse existing row fields — no new bot logic):
- **Symbol** (mono) · **Side** chip · **Score** (with a subtle bar) ·
  **Bucket** chip (Favorite / Near / Study) · **Setup tags** (chips, truncated) ·
  **Key level / entry** · **Supports** (count + green/red HV indicator) ·
  **Theta** (best play summary: strike @ credit, DTE) · **Expected R / rank** ·
  **Days to earnings**.
- Color-code Score and Side; favorites pinned/﻿highlighted; study rows visually
  de-emphasized (they don't affect scoring — label them "Study").
- Sortable by any column; quick filters mirror existing controls (min score,
  min supports, side, max days to earnings). Row click → right info drawer.
- Keep the existing **copy lists** (Longs/Shorts/Favorites/Active/Ranked) as
  one-click toolbar buttons.

Secondary, in the right drawer or sub-tabs (don't crowd the main table):
- **Favorite Setups / Near Favorite Zones / Score-Ranked** as table filters,
  not separate text boxes.
- **S/A Tier picks, Factor impact, Setup-type performance, Scenario outcomes,
  Best stop/profit playbooks** → small tables / KPI tiles in the drawer.

### BounceBot — Live alert feed
A reverse-chronological **alert feed** of `AlertFeedItem`s: time · symbol · side
chip · trigger (e.g., VWAP reclaim, EMA15 retest) · timeframe · RRS context,
color-coded, newest highlighted. Controls (start/stop, connection, market
environment, RRS sensitivity, timeframe, filters) in the left rail. A compact
**connection/health** strip up top. "D1 Master AVWAP events" as a small side list.

### For new traders specifically
- **Explain mode** (Settings toggle, default ON for first run): `InfoDot`s and
  short captions translate jargon ("Favorite setup = highest-conviction; AVWAP =
  anchored volume-weighted average price; green rvol line = high-volume support").
- A **legend** for chips/colors, reachable from any surface.
- Sensible empty states ("Run a scan to see setups") instead of blank boxes.

### For experienced traders
- Compact density toggle, full sortable tables, keyboard shortcuts (run scan,
  focus filter, copy lists, next/prev setup), and the raw copy lists they rely on.

---

## 6. Technical architecture

```
scripts/ui/                      # new PySide6 app (presentation only)
  app.py                         # QApplication, MainWindow, nav rail, mode toggle
  theme.py  theme.qss            # design tokens + stylesheet
  state.py                       # app/session state, settings (per-machine)
  services/
    scan_service.py              # wraps run_master / run_master_with_shared_watchlists in QThread
    bounce_service.py            # wraps run_bot_with_gui in QThread
    data_feed.py                 # parse run_result + report files -> view models; file watchers
  models/                        # SetupRow, ThetaPlay, BounceAlert, TrackerStat (dataclasses/QAbstractTableModel)
  widgets/                       # Badge, DataTable, SetupCard, KpiTile, AlertFeedItem, Sparkline, InfoDot ...
  panels/
    trading_desk.py              # combined workspace + tabs-mode variant
    master_avwap_panel.py
    bounce_panel.py
    journal_panel.py
    research_panel.py            # market prep + ticker lookup (secondary)
    settings_panel.py
  resources/                     # icons, fonts
```

### Rules of engagement
- **Bot logic is untouched.** The UI imports and calls the existing entry points
  (`master_avwap.run_master`, `run_master_with_shared_watchlists`,
  `bounce_bot.run_bot_with_gui`, journal APIs) and reads the existing report
  files / `run_result`. If a value isn't exposed, prefer a small read-only
  accessor over moving logic into the UI.
- **Threading:** long work (scans, bounce loop, IB) runs in `QThread`/worker
  objects; UI updates only via **Qt signals/slots** (never touch widgets off the
  GUI thread). Replace tkinter `after()` polling with `QTimer` + a `QFileSystemWatcher`
  on the report/output files.
- **Old GUI stays during migration.** Keep `gui_app/app.py` working; add the new
  app behind a launcher flag (`--ui qt` / `--ui tk`, default `tk` until Phase 3,
  then flip to `qt`). Only retire the Tk GUI once parity is reached.
- **Reuse `gui_output.py` data**, but render it as models, not text.
- **Windows packaging:** PyInstaller spec producing a single windowed `.exe`
  (no console), app icon, HiDPI manifest. Document `pip install -r requirements.txt`
  + `pyinstaller ui.spec`.

---

## 7. Phased roadmap (each phase ships & runs)

Implement in order; after each phase the app launches and is demoable. Keep PRs
small so they can be reviewed/uploaded between phases.

- **Phase 0 — Foundation.** Add PySide6/qtawesome/pyqtgraph. Create `scripts/ui/`
  with `app.py`, the **theme system** (tokens + QSS), nav rail shell, the
  Workspace/Tabs mode toggle (persisted), global status bar, and 2–3 core widgets
  (Badge, SectionHeader, EmptyState). Launch via `python -m scripts.ui.app` or a
  `--ui qt` flag. *Acceptance:* themed empty shell opens, nav + mode toggle work.

- **Phase 1 — Master AVWAP setups surface (highest value).** `scan_service` runs
  a scan in a QThread; `data_feed` turns `run_result`/report files into
  `SetupRow` models; render the `DataTable` with color-coded columns, filters,
  copy-list buttons, and the right **info drawer** for the selected setup.
  *Acceptance:* run a scan from the UI, see a beautiful sortable setups table +
  details, copy lists work.

- **Phase 2 — BounceBot live panel.** `bounce_service` + live **alert feed**,
  connection/health strip, controls. *Acceptance:* start BounceBot, alerts stream
  into the feed, color-coded.

- **Phase 3 — Combined Trading Desk + responsive modes.** Compose Master AVWAP +
  BounceBot into the workspace (splitters) and the tabs-mode variant; dockable
  watchlist editor (shared longs/shorts + master swing). Flip default launcher to
  the Qt UI. *Acceptance:* both scripts usable together front-and-center; tabs
  mode works on a small window; watchlists editable.

- **Phase 4 — Journal tab.** Port journal functionality into a restyled panel
  (tables/filters), its own nav destination. *Acceptance:* journal parity.

- **Phase 5 — Research (Market Prep + Ticker Lookup) to the side.** Move these
  into the secondary Research destination/drawer. *Acceptance:* available but not
  competing with the Trading Desk.

- **Phase 6 — Polish & ship.** Explain mode + tooltips + legend, light/dark
  toggle, Compact density, keyboard shortcuts, onboarding for first run, settings
  (Drive paths, durable-store warming button), PyInstaller Windows build, icon.
  Retire the Tk GUI. *Acceptance:* installable `.exe`, first-run feels guided,
  experienced flow is fast.

---

## 8. Non-goals / guardrails
- No changes to scanning, scoring, theta, levels, tracker, or data-layer logic.
- No web/mobile; **Windows desktop only**.
- Don't try to migrate everything in one PR — follow the phases.
- Market Prep & Ticker Lookup stay functional but **secondary**.
- Keep the headless/mini-PC scan runnable without the GUI deps (guard imports so
  `master_avwap_mini_pc.py` doesn't require PySide6).

## 9. Open decisions for the owner (pick before/at Phase 0)
1. **PySide6 (recommended) vs PyQt6** for the new UI.
2. **Charts in v1?** (pyqtgraph sparklines/mini level charts) or defer to a later
   pass to keep Phase 1 lean.
3. **Default theme** (dark recommended) and whether to bundle **Inter** or stay
   100% Segoe UI.
4. **Light mode** in v1 or Phase 6.

---

### Appendix A — current state (for the implementer)
- Main GUI: **Tkinter** — `scripts/gui_app/app.py` (`ConsolidatedTradingGUI`),
  `gui_app/bounce_panel.py`, `master_avwap_lib/gui.py` (~3.3k lines),
  `journal_tab.py`, `market_prep_gui/`, `market_prep_tab.py`,
  `gui_output.py` (consolidated text snapshot), `gui_text_highlighter.py`.
- Only Qt today: `scripts/TickerMover.py` (PyQt5, standalone) — fold into the new
  app eventually or leave standalone.
- Entry points: `scripts/gui.py` → `gui_app.app`; mini-PC GUI launcher in
  `master_avwap_mini_pc.py` (`launch_gui_app`).
- Bot entry points the UI should call (do not reimplement):
  `master_avwap.run_master`, `run_master_with_shared_watchlists` (returns
  `run_result` with `tracked_rows`, `theta_put_rows`, study rows, etc.),
  `bounce_bot.run_bot_with_gui`, journal APIs in `journal_tab` / `journal_store`.
- Report files to render as models (see `project_paths.py`): priority setups,
  theta puts, stdev range, event tickers, D1 upgrade alerts, tracker CSVs.
