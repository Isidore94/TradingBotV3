# Focus Picks + Human Setup Tracker Plan

original user input: ok one big change id like to make is a way to combine master_avwap and the trading_bot is by making a tab maybe on the left and below the upper left tab. id like a place to input stocks I really care about and will be highlighted/flagged when theyre ANY type of bounce whatsoever or when theyre RS if theyre a long and RW if theyre a short. we can have 2 boxes one for longs one for shorts. i envision this as a place each day I can input my favourite tickers/ including at EOD and the bot tracks them very closely for me. Additionally we can add a button in master_avwap setup to quickly add a ticker to the list based on its "side". also, lets move the recent bounce alerts tab ot hte left switch it wiht the relatifve strength board. finally, the D1 focus alerts should only flag stocks that go from near or nothing to favourite or high conviction stocks. i only want the best of the best in there. plan this out in GUI_redesign_plan.md (delete everythign in there first as all of that stuff has been implemented). start by layout out the objectives as you understand them first then map out the plan. one last point, on first boot in a new day the program should auto populate the stocks in the new favouritre watchlist that we are creating for longs and shorts into a seperate not as visible watchlist that can appear in the market prep tab in research and basicallyt setup tracker will track these setups seperately because these are basically setups I am handpicking as A or S tier picks so they should be resepted heavily and we should follow their success rate/WR/profit factor to see if human input makes the performance of setups better. setup tracker will need a slight revamp to take this into account.

---

> **For implementers (CODEX + Opus).** This is the next build phase after the Qt
> GUI redesign (now shipped). Read **§1 Objectives** and **§2 What already exists**
> first — several pieces the user wants are already in the backend and must be
> *reused, not rebuilt*. The phase is split into **shared/engine work** (runs in
> `run_master` / the bot, so the headless work mini-PC participates) and **GUI
> work** (Qt, presentation only). Keep that split: tracking + snapshots + alert
> gating live in the engine; the GUI displays and edits. Do not mutate raw
> Master AVWAP scores in v1 (see §11). Commit/push after each numbered step in §10.

---

## 1. Objectives

Combine Master AVWAP and BounceBot around a **trader-curated daily focus list** so
the bot watches the user's handpicked names closely, and so we can **measure
whether human discretion beats the bot's own picks**.

1. A dedicated **Focus Picks** place to enter the tickers the user cares about each
   day (incl. EOD), split into **Focus Longs** and **Focus Shorts**.
2. Treat focus names as high-priority for both Master AVWAP and BounceBot
   (highlight + always-on alerting), **without** silently boosting scores in v1.
3. **Flag** focus names when:
   - they trigger any bounce of their **own direction** — Focus Long → any long
     bounce, Focus Short → any short bounce — bypassing the visible bounce-type
     filters (but not the opposite direction), and
   - their RRS aligns with their side: **Focus Long → RS**, **Focus Short → RW**.
4. One-click **Add Focus** from a Master AVWAP setup row, routed by the row's side.
5. **Reorder** the BounceBot board: Recent Bounce Alerts moves left, swapping with
   the Relative Strength board.
6. **D1 Focus Alerts = best-of-the-best only:** emit only when a symbol *upgrades*
   into Favorite or High Conviction (from near/nothing). No generic D1/stdev noise.
7. **Daily snapshot + human-pick tracking:** on the first scan of a new market day,
   snapshot the focus lists into a dated cohort, show it as a secondary table in
   Research → Market Prep, and have the Setup Tracker measure WR / avg return /
   profit factor of human picks vs the bot's own S/A picks.

Plus four smaller asks folded in (§9): Journal Questrade key input + pull button;
clarify or drop the tracker's Tier Performance / Catch Rate tabs; a root launcher;
and surfacing an **SMA track** setup type alongside the existing Stdev track.

---

## 2. What already exists (REUSE — do not rebuild)

Verified in the codebase; the plan leans on these:

| Capability | Where | How to use |
|---|---|---|
| Watchlist editor (chips-ish text, add/paste/copy/dedupe, autosave) | `scripts/ui/panels/watchlists_panel.py` (`WatchlistEditorPanel`) | Base the Focus Picks panels on this; add per-symbol chip + `X`. |
| Shared longs/shorts loading into the scan | `master_avwap_lib/legacy.py::resolve_master_scan_watchlist_paths` / `load_tickers_from_paths`; `runner.py:~241` | Focus symbols injected into `longs.txt`/`shorts.txt` are already scanned — no scan change needed. |
| **Questrade trade import** (OAuth refresh-token flow) | `scripts/journal_importers.py::QuestradeImporter` (+ `journal_runner.run_journal_import_for_date`) | Journal item is **UI only**: capture the refresh token + a "Pull new trades" button that calls the existing importer. |
| **SMA-breakout retest tracking** | `legacy.py::analyze_sma_breakout_setup` (3526), `SMA_BREAKOUT_TRACKING_FAMILY="sma_breakout_retest_tracking"`, `_priority_should_track_sma_breakout` | "SMA track" largely exists; the work is **surfacing** it as a first-class bucket next to `stdev_retest_tracking` (label map at `legacy.py:~19954`). |
| **D1 upgrade-alert pipeline** | `legacy.py::build_master_avwap_d1_upgrade_alert_payload` (21709), `write_master_avwap_d1_upgrade_alert_outputs` (21837), `MASTER_AVWAP_D1_UPGRADE_ALERTS_FILE`, `MASTER_AVWAP_D1_WATCHLIST_FILE` | Gate this to fire only on real bucket upgrades; don't build a new alert path. |
| **Focus-bounce bypass** in the bounce engine | `bounce_bot_lib/legacy.py::_emit_master_avwap_focus_bounce_alert` (3071); `bounce_type_enabled(...)` honors `include_disabled_bounce_types` (5559) | Reuse this "emit regardless of enabled type" path for human focus symbols. |
| **Separate tracker cohort** pattern (records that never pollute the bot aggregates) | tracker `control_setups` namespace + `select_tracker_control_rows` / `_prune_control_setups` (see [[tracker-methodology-fixes]]) | Model the human-pick cohort the same way: its own namespace, its own export, never folded into `setups` aggregates. |
| Setup Tracker Qt panel w/ tabs | `scripts/ui/panels/setup_tracker_panel.py` (Current Picks, Setup Types, Playbooks, Scan Factors, Tier Performance, Catch Rate) | Add a **Human Picks** tab here; address the Tier/Catch-rate clarity here. |
| RRS board widget | `scripts/ui/widgets/rrs_snapshot.py` (consumes the snapshot payload: `results`/`results_sector`/...) | Add focus highlighting; cross-reference focus set vs `RS`/`RW` rows. |
| Bounce alert feed + service signals | `ui/panels/bounce_panel.py`, `ui/services/bounce_service.py` (`alertReceived`, `rrsSnapshotChanged`) | Subscribe the Focus Picks tab to these for per-symbol live state. |
| Root launcher | `TradingBotV3_GUI.cmd` (exists, untracked) | Commit + document; optionally add a Python entry shim. |

> **Two known code gotchas** for every engine-side change below:
> 1. `apply_final_priority_buckets` (and `_classify_priority_bucket`) is **defined
>    twice** in `legacy.py` (the later def shadows the earlier). Any change to
>    classification/marking must be applied to **both** copies.
> 2. Tracking changes must run in **both** `runner.run_master` and the legacy
>    backfill chain so historical reprocessing stays consistent.

---

## 3. Architecture & data flow

**Shared/engine layer (runs in `run_master` and the bot — so the work mini-PC
participates headlessly):**
- Reads `focus_longs.txt` / `focus_shorts.txt` from the shared home.
- Daily focus **snapshot** + **forward-return** tracking (the human-pick cohort).
- Marks tracker records `human_focus_pick=True` when a setup's `(symbol, side)` is
  in the focus list.
- Maintains `master_avwap_bucket_state.json` and gates D1 upgrade alerts.
- Loads the focus set into the bounce engine for always-on focus alerting.

**GUI layer (Qt, presentation + editing only):**
- `FocusPickStore` (new service) — single source of truth for the focus lists in
  the running app; read/write the txt files, manage shared-watchlist injection
  membership, emit a `focusChanged` Qt signal.
- One `FocusPickStore` instance is created by `TradingDeskPanel` and **injected**
  into: `FocusPicksPanel`, `MasterAvwapPanel` (Add Focus + table highlight),
  `BouncePanel`/`AlertFeedItem` (highlight + state), `RrsSnapshotWidget` (highlight).
- The Focus Picks tab subscribes to `BounceService.alertReceived` and
  `rrsSnapshotChanged` to show each focus symbol's latest bounce/RRS state.

**Important:** the editable focus lists and the engine-side tracking communicate
**only through the shared txt files** (and the runtime JSON/CSV the engine writes).
The GUI never has to call the engine directly for tracking; it edits the lists, the
next scan picks them up. This keeps home (GUI) and work (headless) consistent.

---

## 4. Focus Picks UI

### 4.1 Placement & layout
- New **`Focus Picks`** tab in `MasterAvwapWorkspace` (`ui/panels/trading_desk.py`),
  inserted immediately **after `Setups`** (before Theta Plays / Watchlists).
- Two side-by-side editable panels: **Focus Longs** | **Focus Shorts**
  (`QSplitter`, mirror `WatchlistEditorArea`).

### 4.2 Per-side controls & behavior
- Symbols render as **chips** with a per-chip `X` (fast individual removal) — new
  small widget `ui/widgets/symbol_chip.py` (reuse `Badge` styling).
- Buttons per side: `Add`, `Paste`, `Copy`, `Clear All`, `Refresh`.
- Edits **autosave** (debounced, like `WatchlistEditorPanel._save_timer`).
- **Shared-watchlist sync (via `FocusPickStore`):**
  - Adding to Focus Longs → also append to `longs.txt` **iff** not already present;
    record the injection in `focus_pick_membership.json`.
  - Adding to Focus Shorts → same against `shorts.txt`.
  - Removing a focus symbol → remove the focus flag, and remove from the shared
    list **only if** `focus_pick_membership.json` proves Focus Picks injected it
    (never delete a name the user maintains independently in the broad list).
- Status line: save path + today's snapshot status (snapshotted / not yet) +
  `Snapshot Today` button (§7).
- Per-symbol live state column (populated from bounce/RRS signals): last bounce
  (time + type), current RRS state (`RS`/`RW`/`–`), and a side-alignment check
  (Long+RS or Short+RW = green; misaligned = muted).

### 4.3 Master AVWAP integration
- Row-level **`Add Focus`** action in the setups table (`master_avwap_panel.py` +
  the `SetupTableDelegate` / a context-menu action on `DataTable`). Routes by
  `row.side`: LONG → Focus Longs, SHORT → Focus Shorts. Duplicate add = no-op +
  status update.
- **Focus marker** in the setups table for symbols already in Focus Picks: a small
  star/dot drawn by `SetupTableDelegate` in the Symbol cell (reads `FocusPickStore`
  membership). Add an `is_focus_pick` passthrough on the `SetupRow` view model or
  query the store at paint time.
- **No score mutation** (v1). Focus is a human overlay + tracking cohort, not a
  hidden scoring change.

---

## 5. BounceBot layout + focus alerting

### 5.1 Panel reorder (`ui/panels/bounce_panel.py`)
Current split order is `Live Alert Feed | D1 Master AVWAP Events | RRS`. Reorder to:
1. **Recent Bounce Alerts** (leftmost)
2. **D1 Focus Alerts** (renamed/retargeted D1 events, §6)
3. **Relative Strength (RRS) board**

### 5.2 Always-on focus alerting (engine change — `bounce_bot_lib/legacy.py`)
- Load the focus set into the bounce engine (extend `run_bot_with_gui` to accept a
  focus provider, or have the engine read `focus_longs.txt`/`focus_shorts.txt` +
  re-read on change).
- For a focus symbol, emit a **direction-matching** bounce regardless of the
  per-type enable toggles — **Focus Longs only on long-direction bounces, Focus
  Shorts only on short-direction bounces** (opposite-direction bounces are not
  forced). Reuse the existing `include_disabled_bounce_types` path used by
  `_emit_master_avwap_focus_bounce_alert`, scoped by side. Still respect **global
  scanning on/off**.
- Tag focus alerts in the callback payload (`feedback.is_focus_pick=True`,
  `feedback.focus_side`) so the GUI can highlight without re-deriving membership.

### 5.3 GUI highlight (`ui/widgets/alert_feed_item.py`)
- Focus alerts get a distinct accent (gold left-stripe / focus badge), consistent
  with the setups-table favorite styling.

### 5.4 RRS focus flags (`ui/widgets/rrs_snapshot.py` + Focus Picks tab)
- On each `rrsSnapshotChanged`, cross-reference focus set against the parsed
  `RrsRow`s: **Focus Long present as RS** or **Focus Short present as RW** = an
  aligned flag. Surface as a badge on the RRS board for focus names, and feed the
  per-symbol state column in the Focus Picks tab.

---

## 6. D1 Focus Alerts — upgrades only

Goal: the D1 panel shows only **transitions into** `favorite_setup` or
`high_conviction`. Everything else (generic D1 band/stdev events, favorite→favorite,
*→near, etc.) is suppressed.

### 6.1 Bucket-state store (engine)
- New `data/runtime/master_avwap_bucket_state.json`:
  `{ "SYMBOL|SIDE": { "bucket": "...", "scan_date": "YYYY-MM-DD", "updated_at": ... } }`.
- Written after each Master AVWAP scan, once final buckets are known
  (`runner.run_master`, **after** `apply_final_priority_buckets`).

### 6.2 Transition gate
Emit a D1 Focus upgrade event only for **prev → curr** where `curr ∈ {favorite,
high_conviction}` and `prev ∈ {missing, unbucketed, near_favorite_zone}` (or
`favorite → high_conviction`). Suppress favorite→favorite, hc→hc, *→near, and any
non-bucket D1/stdev event.

### 6.3 Wiring
- Compute transitions where final buckets are produced, then feed only the
  upgrade events into `build_master_avwap_d1_upgrade_alert_payload` /
  `write_master_avwap_d1_upgrade_alert_outputs` (i.e., constrain what those
  already-existing functions emit). The bounce side that displays them
  (`emit_master_avwap_d1_flags`) then naturally shows only upgrades.
- Keep the prior bucket from `master_avwap_bucket_state.json`; update state at the
  end of the scan. Handle first-run (no state file) as "missing" for everyone.

---

## 7. Daily snapshot + human-pick cohort (engine, with GUI views)

### 7.1 Snapshot (runs in `run_master`, market-local date)
- On the **first scan of a new market-local trading day** (`market_session.py`
  helpers for the market date), copy the current Focus Longs/Shorts into a dated
  cohort row set; do **not** clear the editable lists.
- Idempotency via `data/runtime/human_focus_snapshot_state.json`
  (`{"last_snapshot_market_date": "..."}`). Manual `Snapshot Today` button merges
  safely (no duplicate rows for the same `trade_date+symbol+side`).
- `human_focus_daily_picks.csv` fields: `trade_date, symbol, side, source,
  snapshotted_at, active_at_snapshot` with `source=focus_pick`.

### 7.2 Forward-return tracking (runs in `run_master`, reuses durable daily bars)
- For each pick whose `trade_date` is within ~10 sessions of today and not yet
  fully matured, compute **side-adjusted forward returns** from the durable daily
  bar store (entry = snapshot-day close):
  - long: `(close_tN - entry) / entry`; short: `(entry - close_tN) / entry`.
  - horizons **1 / 3 / 5 / 10 sessions**.
- Write per-pick outcomes to `human_focus_outcomes.csv`; aggregate to
  `human_focus_performance.csv` (count, WR = share with positive side return at a
  chosen horizon, avg side return, profit factor when enough closed samples).
- Keep this **separate** from the bot scenario machinery (it's plain forward
  returns, which is exactly what the user asked for). Treat it as its own cohort,
  like `control_setups` — never folded into the bot tracker aggregates.

### 7.3 Tracker record marking
- When a scan produces a setup whose `(symbol, side)` is in the current focus list,
  set `human_focus_pick=True` on the tracker record and the priority/feature rows.
  (Apply in both `apply_final_priority_buckets` copies / both run paths.)

### 7.4 GUI views
- Research → **Market Prep**: a secondary, less-prominent **"Today's human picks"**
  table reading `human_focus_daily_picks.csv` (with live RRS/bounce state if cheap).
- Research → **Setup Tracker → Human Picks** tab (§8).

---

## 8. Setup Tracker revamp (Human Picks)

Add a **Human Picks** tab to `ui/panels/setup_tracker_panel.py` that reads
`human_focus_performance.csv` / `human_focus_outcomes.csv` and answers:

- Are handpicked longs/shorts **outperforming bot S/A picks**? (side-by-side WR /
  avg side return / profit factor, **same horizons + same side-return metric** for
  both cohorts so the comparison is apples-to-apples).
- Which human-picked **setup families** are working (join human picks that also
  became bot setups, grouped by family)?
- Does human discretion **improve or hurt** ranking (compare human-flagged setups
  vs non-flagged within the bot tracker, using `human_focus_pick`)?

To make the comparison fair, compute the **same 1/3/5/10-session side returns** for
the bot S/A tier picks (from existing tracker entry dates + durable bars) and show
both cohorts in one table. Columns: cohort, n, WR, avg side return per horizon,
profit factor, and a delta vs bot S/A. **Headline the 5- and 10-session horizons**
(the owner's chosen comparison numbers); keep 1/3 as supporting columns.

> **Design principle (measure before boosting).** v1 only *measures*. If human
> picks demonstrably beat bot S/A over a meaningful sample, a v2 can fold that into
> Expected-R (e.g., a `human_focus` prior or a regime-style weight). Do not boost
> scores until the data says so. See [[expected-r-ranking-system]].

### 8.x Tracker clarity (folded-in ask)
Tier Performance / Catch Rate tabs are unclear. Either (a) add a one-line
`SectionHeader` subtitle + per-column tooltips explaining what each measures
(Tier Performance = realized outcome by S/A/B tier; Catch Rate = how often a tier's
picks actually triggered/were catchable), or (b) if they remain low-signal, drop
them. Decide with the owner; default to (a) clarify.

---

## 9. Folded-in smaller asks

- **Journal — Questrade (UI only).** Add to `ui/panels/journal_panel.py`: a
  collapsible "Broker sync" area with a **Questrade refresh-token** field (note:
  Questrade rotates the refresh token, so the user re-pastes it ~weekly) saved via
  `save_local_setting(QUESTRADE_REFRESH_TOKEN_SETTING, ...)`, and a **Pull new
  trades** button that runs `journal_runner.run_journal_import_for_date` for the
  recent window on a worker thread (reuse the `ScanService`/`WarmingService`
  threading pattern), then refreshes the trade table. Backend already exists
  (`QuestradeImporter`).
- **Launcher.** Commit `TradingBotV3_GUI.cmd` and document it in the README;
  optionally add a thin root `launch_gui.py` that calls `scripts/gui.py --ui qt`.
- **SMA track.** Surface the existing `sma_breakout_retest_tracking` family as a
  first-class "SMA track" bucket parallel to `stdev_retest_tracking`: ensure it has
  a label (`legacy.py:~19954` label map → `"sma-track"`), appears in the setups
  table bucket filter / drawer, and is tracked like the stdev-track bucket. Confirm
  `analyze_sma_breakout_setup` already emits the retest-completion (15EMA retest)
  the user describes; if the retest-completion transition isn't recorded, add it.

---

## 10. Implementation steps (sequenced; commit/push after each)

Order chosen so each step ships independently and later steps depend on earlier
storage/services.

1. **[DONE]** **Storage + `FocusPickStore` (engine + shared).** Path constants in
   `project_paths.py` (focus txt + the six runtime files in § Storage).
   `scripts/focus_picks.py::FocusPickStore` — plain Python (no Qt) so engine + GUI
   share it: read/add/`add_many`(paste)/remove/clear/dedupe, observer listeners,
   `load_focus_map()` read helper, and shared-watchlist injection with
   `focus_pick_membership.json` (removal only un-injects what Focus Picks added;
   never deletes an independently maintained broad-list symbol). Tests:
   `tests/test_focus_picks.py` (10, green). *Note: autosave/Qt `focusChanged` signal
   arrives with the GUI wrapper in Step 2.*
2. **[DONE]** **Focus Picks tab (GUI).** `ui/services/focus_service.py::FocusService`
   (Qt adapter over the store; `focusChanged` signal + autosave-on-edit),
   `ui/widgets/flow_layout.py` (wrapping chip layout), `ui/widgets/symbol_chip.py`
   (ticker chip w/ × remove, side-toned), `ui/panels/focus_picks_panel.py`
   (Focus Longs | Focus Shorts editors: add/paste/copy/clear-all, chip flow).
   Wired into `MasterAvwapWorkspace` as the tab right after Setups; `FocusService`
   created in `TradingDeskPanel` and injected. Tests: `tests/test_qt_focus_panel.py`
   (3). *Per-symbol live RRS/bounce state column arrives with Step 4.*
3. **[DONE]** **Master AVWAP Add-to-Focus + table marker (GUI).** `DataTable` gained
   per-row context-menu actions (`add_row_action`); `MasterAvwapPanel` (now takes
   the injected `FocusService`) adds **"Add to Focus Picks"** which routes by the
   row's side (LONG→Focus Longs, SHORT→Focus Shorts). `SetupTableDelegate` draws a
   gold ★ in the Symbol cell for focus names (`set_focus_lookup`), and repaints on
   `focusChanged`. Tests in `tests/test_qt_focus_panel.py`.
4. **BounceBot reorder + focus alerting.**
   - **[DONE] 4a (GUI):** Recent Bounce Alerts is now leftmost (Recent Bounce Alerts
     | D1 Focus Alerts | RRS). Focus bounce alerts get a gold ★ FOCUS badge + left
     stripe (`AlertFeedItem(is_focus=...)`, cross-referenced client-side against the
     `FocusService`). RRS board stars focus-**aligned** names (focus long shown as
     RS / focus short shown as RW) via `RrsSnapshotWidget.set_focus_service`.
     `FocusService` injected into `BouncePanel`. Tests in `test_qt_focus_panel.py`.
   - **4b (engine, pending):** in `bounce_bot_lib`, emit a focus symbol's
     **direction-matching** bounce regardless of the per-type enable toggles (reuse
     the `include_disabled_bounce_types` path used by
     `_emit_master_avwap_focus_bounce_alert`), scoped by side. Tag the callback
     payload `feedback.is_focus_pick` / `focus_side`. (Riskier; needs IB to verify.)
5. **D1 Focus Alerts.**
   - **[DONE] 5a (engine):** `scripts/master_avwap_bucket_state.py` (pure, tested) —
     `is_bucket_upgrade` transition rule, `load/save_bucket_state`,
     `compute_bucket_upgrades`, `record_scan_bucket_upgrades`. Wired into
     `runner.run_master` after final buckets: diffs vs
     `master_avwap_bucket_state.json`, persists new state, exposes
     `run_result["bucket_upgrades"]` (only genuine upgrades into Favorite/High
     Conviction). Tests: `tests/test_bucket_state.py`.
   - **5b (display, pending):** retarget the D1 Focus Alerts panel to show only those
     upgrade events (consume `bucket_upgrades` / write them where the bounce-side
     `emit_master_avwap_d1_flags` display reads) and drop the generic D1/stdev noise.
6. **Daily snapshot + forward-return tracking (engine) + Market Prep view (GUI).**
7. **Setup Tracker Human Picks tab + comparison (GUI)**; Tier/Catch-rate clarity.
8. **Folded asks:** Journal Questrade UI; launcher commit/doc; SMA-track surfacing.
9. **Keep this doc current:** remove completed bullets as they merge; leave
   remaining work visible for the next agent.

---

## 11. Design principles / guardrails

- **No hidden score mutation in v1.** Focus Picks highlight, alert, and track; they
  do not change Master AVWAP scores. (Revisit only after §8 shows an edge.)
- **Engine owns tracking; GUI owns display.** So the headless work mini-PC produces
  the same snapshots/outcomes the home GUI shows.
- **Fail-open / non-destructive.** Missing focus files → empty lists (no errors).
  Never delete a user's broad-watchlist symbol unless membership proves Focus Picks
  injected it. Snapshots are idempotent per market day.
- **Reuse the `control_setups` separation** so human-pick records never contaminate
  the bot's Expected-R / calibration aggregates.
- Apply every engine change to **both** `apply_final_priority_buckets` copies and
  **both** run paths (run_master + backfill).

---

## 12. Storage (paths)

Shared home (synced — both machines share the curated lists):
- `focus_longs.txt`, `focus_shorts.txt`

Runtime (shared home `data/runtime/`):
- `focus_pick_membership.json` — which shared-watchlist entries Focus Picks injected
- `master_avwap_bucket_state.json` — last bucket per `symbol|side` (D1 gate)
- `human_focus_snapshot_state.json` — last snapshot market date (idempotency)
- `human_focus_daily_picks.csv` — dated cohort
- `human_focus_outcomes.csv` — per-pick forward side-returns (1/3/5/10)
- `human_focus_performance.csv` — aggregated WR / avg / profit factor + vs bot S/A

(Diagnostic app logs remain **local** per the logging fix — unrelated to these.)

---

## 13. Test plan

- **FocusPickStore:** parse/dedupe/add/remove/paste/clear-all; shared sync (long→
  `longs.txt`, short→`shorts.txt`); removal does **not** delete an independently
  present shared symbol; membership json correctness.
- **Snapshot:** first scan of a new market day creates exactly one cohort; same-day
  re-run does not duplicate; manual snapshot merges safely.
- **Forward returns:** side-adjusted math (long vs short), correct horizon mapping,
  matures at 10 sessions; aggregation WR/avg/PF.
- **D1 gate:** near→favorite emits; missing→high-conviction emits; favorite→favorite
  and hc→hc and *→near do **not**; stdev-only events do not.
- **Master AVWAP:** row action adds to correct side; focus marker renders.
- **BounceBot:** focus symbol fires on a disabled bounce type; non-focus respects
  filters; focus alert highlighted; RRS flags Focus Long=RS / Focus Short=RW.
- **Tracker marking:** `human_focus_pick=True` set when a setup matches the focus
  list (both run paths).
- **Qt smoke:** Focus Picks tab loads + chips delete; BounceBot panel order correct;
  Human Picks tab renders; Journal Questrade field saves + pull button threads.

---

## 14. Resolved decisions (owner)

1. **Snapshot trigger:** **both** first GUI boot **and** first `run_master` of the
   day — whichever runs first performs the snapshot; the other no-ops via
   `human_focus_snapshot_state.json` (idempotent per market-local day).
2. **Headline horizons:** **5- and 10-session** side returns headline the human-vs-
   bot comparison; 1/3 stay as supporting columns.
3. **Tier Performance / Catch Rate:** **clarify** — add a `SectionHeader` subtitle +
   per-column tooltips explaining what each measures (do not remove).
4. **Focus alerting (side-specific):** Focus **Longs** alert only on **long**
   bounces and flag when **RS**; Focus **Shorts** alert only on **short** bounces
   and flag when **RW**. Opposite-direction bounces are not forced.
