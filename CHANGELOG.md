# TradingBotV3 implemented history

Last reconciled: **2026-08-27** on `claude/gui-phase-0-9`, at the four trader
rules of that morning (regime-pause auto-Focus `479c25c`, the VWAP-side /
show-time review filter `76e0b7b`, the D1 SMA trend leg + snapshot Prev/Next
`f3abda7`, the M5 alert bar `41963de`/`39c3ef7` and its click-away skip, then
the group tape removed and REBUILT, then the desk-memory packet, both on
`claude/warehouse-build-memory`)
after Phase 0.9's first three packets - the table width rule, the AWAY Recap return
surface and the Desk Journal keyboard route. The same branch also carries Phase
0.10's AVWAP band challenger and its review fixes (two sessions shared one
checkout on 2026-08-26; see `CURRENT_CHECKPOINT.md`).

Authoritative for: **what exists and the historical sequence of revisions**

Remaining work: [`plan.md`](plan.md)

This is a curated product history, not a raw commit dump. It reconciles the former
status sections in `plan.md`, the accumulated `CURRENT_CHECKPOINT.md` ledger, the GUI
plans, warehouse plans/reviews, dated handoffs, and Git history. Exact current test
counts remain in `CURRENT_CHECKPOINT.md`.

The labels retain their strict meanings: `IMPLEMENTED` means code exists, `GREEN`
means deterministic tests pass, `LIVE_VALIDATED` requires real-session evidence,
and `PROMOTED` requires an explicit champion decision. A feature can be implemented
and green while its live or promotion gate remains open in `plan.md`.

## Current implemented inventory

**This is the contract: what exists, by area. Search it before building anything so you
do not rebuild landed work.** It is deliberately short. The dated entries under
`Recent changes` below cover the last two build days; everything older is in
[`docs/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md`](docs/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md),
which is evidence and must not be loaded as context.

### Application, runtime, and data ownership

- **Win rate leads every trader-facing SWING surface** (V3, decision 0016 answer
  3). `scripts/swing_headline.py` is the one implementation: win rate first, `n`
  and a **Wilson lower bound** beside it, mean R beside that and never instead of
  it. **Sorting is by the LOWER BOUND** - the raw rate puts a 100%-on-three cell
  above a 62%-on-ninety every time. It reads the TRACKER'S OWN `win` verdict
  rather than re-deriving one, and the average carries its unit, because a column
  headed "Avg R" showing a percent is a number that lies.
  `setup_docs.family_record_sentence` renders one line per family AT READ TIME,
  at ONE declared horizon (`evidence_stats.SWING_HORIZON_SESSIONS`, 5 - the same
  one the AWAY digest ranks on), from ONE pass over the tracker.
  **PARTIAL, and what is wired is named** (R4 B3): the AWAY digest ranking, both
  setup-doc renderers, the Master AVWAP setups table's **Family Win %** column,
  the Setup Tracker's **Last 30 Days** tab, and all four Weekend Prep cohort
  tables - each SORTING by the Wilson lower bound. **STILL OWED: the Setup
  Tracker's Setup Types tab**, because `master_avwap_setup_type_stats.csv` has no
  win column and the outcomes file cannot be joined at that table's grain (184
  rows over 71 (side, bucket, family, zone) groups). **ONE Wilson z**:
  `swing_headline.WILSON_Z` (1.96). `expected_r`'s 1.28 is a parameter of the
  proven-quality score in a fenced scoring file and no trader-facing surface may
  reach for it.
- **MFE after a held level leads every DAY-TRADE surface** (V3 item 2, WIRED by
  R4 A9/A10). The Day Trade Tracker leads with **Held 30m** and Held x Ran and
  opens sorted by the second; the tier statistics stay beside them. **One
  formula**: the panel joins `held_run_score.dimension_summaries` and computes
  nothing, which is why the column may finally say "30m" - V3 shipped a SECOND
  formula under the same key (`1 - stop_rate` x `avg_mfe_r`, the aggregator's own
  window, every row rather than the held ones). **The join is an equality since
  R4 fix round 1**: this module spells its segments the aggregator's way - the
  champion's own `time_bucket_for` (the private copy compared wall-clock hours
  against Eastern cutoffs on a Pacific desk), an episode counted under EACH of
  its bounce types, and the combination `+`-joined. Live: `bounce_type` 36/36,
  `bounce_combo` 58/59, `time_bucket` 10/10, `market_environment` 10/10, against
  28/36, 0/59, 2/10 and 10/10 before. The four `master_avwap_*` tabs read BLANK
  because the outcome log does not carry them; `rrs_alignment` is reachable and
  simply not derived yet, and `UNDERIVED_DIMENSIONS` says so rather than filing
  it under "cannot".
  The M5 alert row carries "held NN% / ran N.NR" through `alert_cell` +
  `alert_suffix`, silent below the floor, attached as a dict read from an index
  built once per session on a worker. `d1_setup_present` had no caller at all and
  is now fed from `master_avwap_tracker_scoring_snapshot.json`.
- **"Lately" is ONE number, counted in trading sessions** (V3 item 3).
  `evidence_stats.LATELY_SESSIONS` (20) and `lately_window()`, which walks the
  exchange calendar: twenty calendar days is fourteen sessions in a normal month
  and twelve across a holiday week.
- **One annotation writer, and every row carries its screen** (V3 item 4,
  completed by R4 A5). Exactly one module outside the store calls the raw writer,
  and the capture rail's VETO path stamps `surface` as its LIKE path already did.
  **All five declared surfaces now have a writer**: the Master AVWAP star/cross,
  the review pane's "Not today", the bare rail, the Focus chip's right-click
  Like / Not today, and the M5 alert row's right-click quick like. The two
  chart-review HOSTS call `set_scan_context(surface=SURFACE_CHART_REVIEW)`; the
  override existed from P10 B1 and no host ever called it, so every verdict
  passed on a review chart filed as `rail`. The Focus panel's "Not today" writes
  the row FIRST and then asks for the scoped removal that refuses a name the
  trader typed. The note box saves on **Enter** and newlines on Shift+Enter
  through one helper, `ui/widgets/note_prompt.py` (R4 A6).
- **The Research tab is the builder's surface** (V3 item 5). The nightly fact
  pack's headline gets one line on Weekend Prep's verdict card; the full panel
  stays in Research, which now says so on the page.
- **Weekend Prep has ONE Refresh and a verdict card** (V2 item 2, decision 0016
  answer 10; finished by R4 A13/A14/A18). The click starts each page's own reader
  and returns - measured under 50 ms with the reads stubbed at the WORKER
  boundary - and the five per-page buttons left the layout, as did **Discovery's
  six per-table ones**, which now have a real `reload` (it had none, so one
  Refresh counted the step and built nothing). `week_trades` moved off the Qt
  thread: it was 775 ms of the click. Every table carries the ten-row floor
  through one constant, `TABLE_TEN_ROWS_PX`. The card's take rate READS `shown`
  and `overall_take_rate` off the state - it used to add `takes + skips +
  rejects`, and the state has never published the last two, so it printed
  "100% of 94" where the truth was 30% of 318. The card is a PURE builder
  (`scripts/weekend_verdict.py`): take rate, blind spots and leaks BY NAME, the
  best liked claim and weakest veto reason at h3, the week's net and win rate
  (**confirmed tags only**), the tag-review count. Every measured line carries its
  n; a cohort under n=5 is named thin and never ranked; a missing input says so
  rather than printing a zero. The RS/RW prose is retired - it duplicated a live
  board with a Saturday snapshot - and the log scans are kept UNCALLED with
  docstrings that say so.
- **"Tag this week" is a weekend step** (V2 item 2e, corrected by R4 A15). The
  week's provisional and needs_review trades, confirm-all-shown and
  confirm-selected through `JournalStore.confirm_tags`, ten visible rows, read AND
  written on a worker. A confirmed row is never listed again, and a failed write
  is reported LOUDLY. **A row with no tag is SKIPPED and counted**: `confirm_tags`
  only flips the lane, so confirming a blank leaves the nightly tagger re-flagging
  it `needs_review` every night forever. "Edit tag..." is the path those rows have
  - the trader's own wording through `correct_auto_tag`, then confirmed.
- **The tagger runs every night** (V2, decision 0016 answer 10). `journal_auto_tag`
  is a deterministic slot inserted SECOND, right after `journal_import` — the
  second and last sanctioned exception to this list's append-only rule. It applies
  P6a's plan at 0.70, never touches a confirmed row, and **fails LOUDLY**: the
  journal is the one store on this desk that may not fail quietly. The Journal nav
  button reads "Journal (N to review)", counted off-thread from `showEvent`.
- **The Market Journal capture is one box and one Enter** (V2, answer 11; the
  LEFT-NAV PAGE too since R4 A16, which V2 never touched). The picker and the
  button leave the surface; **nothing leaves the schema**. The page is a DATED
  newest-first list across every session. The entry is dated to the SESSION IT IS
  ABOUT — today while today trades, the last session that traded otherwise — and
  **the roll is the session's OPEN, not midnight in New York** (R4 A17): a Pacific
  note at 21:00 was filing against tomorrow. `written_after_the_session` is still
  COMPUTED, and measured against the session's **CLOSE** rather than its date.
- **The unused surfaces are HIDDEN, never removed** (V2, answer 7). One setting,
  default OFF, hides the Alerts / D1 Focus / Armed tabs and the Universe page.
  `setTabVisible`, so no index shifts; every timer stays visibility-gated; and a
  test proves every rail shortcut is panel-scoped, bound once, and not owned
  inside a hidden tab — a QShortcut in a hidden tab never fires, and two bindings
  for one sequence fire NEITHER.
- **The Strength Board IS the trader's TC2000 scan** (V1, 2026-09-02, decision
  0016 answer 9; corrected by R4 A7/A8). Relative volume is
  `AVG(V / mean(V at the same bar offset over the prior 15 sessions), 12)` -
  **SESSION-RELATIVE**, which is what answer 9 asks for and calls "the time-of-day
  relative volume". V1 shipped a flat positional stride on a TC2000-parity
  argument; one 39-bar early close shifts every offset past it, and on a series
  whose volume is a pure function of the time of day - where the answer must be
  exactly 1.0000 - it read 1.2949. A prior session that never reached bar k
  CONTRIBUTES NOTHING rather than a zero. Blank and never zero under fifteen
  prior sessions. Plus the $5, D1 200 SMA, D1 100 SMA and M5 15 EMA floors, each
  a NAMED boolean carrying the sentence that failed; the D1 pair reads a **`2y`**
  download with **today's forming bar dropped**. The fetch period is `1mo`
  because the RVOL needs sixteen sessions of bars. The universe is
  `universe_all.txt` PLUS the four watchlists. **A row that misses a
  filter is GREYED with its reason, never dropped**, behind a default-on "TC2000
  parity" toggle. The D1 SMAs come from a second batched daily download; still zero
  IB traffic. Golden `tc2000_parity_v1` pins strength and RVOL for **seven** symbols
  against a SECOND hand implementation - AAA-EEE are clean sessions on which both
  readings agree, which is exactly why they could not catch the defect, and R4
  added `FFF` (one early close) and `GGG` (one missing bar).
- **`strength_scan.py`'s fence is narrowed, not lifted.** The R8 spec froze the
  module whole; the trader authorized this change naming the file, so the test now
  pins the seven FORMULA functions byte-identical to the R8 baseline instead.
- **One window, two sections, RS/RW first** (decision 0016 answer 7). The RS/RW
  board left the Alert Center's tab stack for the strength column, above the M5
  Strength section, in a scroll area - hosted bare its minimum took the column's
  floor from 190 px to 452, past the alert column's whole 360 px budget.
- **The AWAY digest ranks swing picks by the tracker's record, not by the
  bucket** (V1 item 3, built R4 A11; decision 0016 answer 8: *"the best pick is
  often in the near bucket, not the favourite bucket, so the cream is not being
  sent."*) The order is the **Wilson lower bound** on the setup family's realized
  win rate, read from `master_avwap_tier_outcomes.csv`'s own `win` column inside
  `lately_window()`, with expected R as the tiebreak; an ungraded family sorts
  BELOW every graded one rather than at zero. The bucket is PRINTED and never
  ranked on, and the near cap is applied AFTER the ranking, so what is hidden is
  the weakest near rows and never the best one. `render_away_report` stays a pure
  renderer - the read is the caller's. AWAY is still the only routine pusher.
- **`held_run_score` measures whether the level held and then how far it ran**
  (V1 item 2, decision 0016 answer 4): P(the level MEASURED held inside 30 minutes) x
  trimmed-mean MFE_R of the held ones, per (bounce_type, time_bucket, environment,
  d1_alignment), over the shared `lately_window`. **Since packet Q1 (2026-09-04)** every
  episode carries a measurement state - `measured_held` / `measured_broken` / `pending` /
  `unmeasured` (reasons `no_follow_up`, `window_not_reached`, `break_time_unknown`) - and
  only the first is held; `hold_rate` is held / MEASURED and every cell carries
  `n_measured`, `n_broken`, `n_pending`, `n_unmeasured` and `coverage` (the Daytrade
  Tracker's **Measured** column, `35 / 41`). The D1 dimension keeps the setup's SIDE
  (`aligned` / `opposed` / `none` / `unknown`; `d1_setup_present` is aligned only; a missing
  snapshot is UNKNOWN, never False; basis `same_session_retrospective`). The window is
  `evidence_stats.lately_window` and `window_report` names the missing sessions on the
  tracker's status line. **A SECOND score** - the champion tier,
  the mutes and the PROVEN stamp are untouched and a test pins that the champion
  never imports it. The row suffix is BLANK below the floor, never a number in
  brackets. **Its surfaces landed with R4 A9/A10** - the Daytrade Tracker column
  and sort, and the M5 alert row's suffix. The priority/ordering switch is V4;
  the seam is `daytrade_tracker_panel._by_headline`.


- PySide6 Trading Desk launched by `launch_gui.py`; the legacy Tk compatibility
  path was removed on 2026-09-03 (F2).
- Main-desk single-process ownership, bounded BounceBot startup/shutdown, generation
  guards, child-process reaping, runtime heartbeat, durable job ledger, typed retry
  budgets, stale-run marking, and a hardened single-instance launch guard that also
  sees the frozen executable.
- User-selected shared home folder for operational text/JSONL/CSV artifacts;
  machine-local settings, caches, and diagnostics under LocalAppData; a separate
  research-lake storage class outside that home folder.
- **No cloud sync (2026-08-10, decision 0015).** Google Drive/OneDrive were removed
  from the system entirely. `C:\TradingBotData` keeps its path and role as a plain
  local folder; the DAS file server `\\MINI-PC\Trading Bot Data` is the durable
  tier, holding the research lake, the AI store, and hourly cold-pushed subtrees.
  Documentation-only change: no path, behavior, or test changed.
- Designated-writer authority, local kernel exclusion, fenced writer lease, atomic
  publication, readback verification, last-good preservation, and bounded archives.
- **`focusChanged` is coalesced at every listener** (`ui.timer_utils.SignalCoalescer`,
  200 ms leading-edge window, trailing fire). The store still emits once per mutation;
  the Focus board, the Alert Center feed, the setups-table repaint, the strength board
  and the price-alert combo each react once per BURST. The DESK auto-adoption drain
  additionally adopts at most `AUTO_ADOPT_BATCH_LIMIT` (10) staged picks per 30-second
  cycle - pacing only, nothing withheld, no pick dropped.
- Main desk is the sole always-on scanner. The former mini-PC scanner and Desk Link
  satellite topology were `RETIRED` 2026-08-08 and their code was **removed 2026-08-24**
  (P1.5): no `desk_link` package, no `ui/satellite.py`, no `master_avwap_mini_pc.py`, no
  `--satellite`/`--desk-role` flags.

### Scanning, candidates, and decision support

- Master AVWAP D1 swing scanning with earnings anchors, current/previous AVWAP
  families, running-deviation bands, focus buckets, Expected-R ranking, study tags,
  theta candidates, tracker history, and durable daily-bar storage.
- **Theta premium rules, 2026-08-31 (Phase 0.11, trader-directed).** Sold-put credit
  is judged as a PERCENT OF THE STRIKE - recommended at >= 1.0%, cusp at >= 0.5%,
  with a $0.40/contract absolute floor - and a quote under both floors leaves the
  report instead of showing as `below_target`. The old bar was literally $0.25
  ($100 / 4 contracts), which is 0.125% of a $200 strike and 1.25% of a $20 one.
  Ranking priority is support (major SMAs above the strike, 2+ a large boost, then
  the covered stack) -> yield per market day -> spread; the strike-ascending sort
  key that always preferred the cheapest qualifying option is gone, and the spread
  penalty is monotonic and uncapped but never a block. Credit spreads reach 15
  market days (sold puts stay at 10). The IB quote budget - unchanged at 240
  quotes / 360 s - is spent `thetalongs.txt` first, then estimated premium
  capacity (ATR%-based, no new network call), then `base_score`; nothing is
  dropped and the support-only fallback still covers the tail. The report and the
  Qt theta panel carry credit % of strike, yield per week, spread %, credit source
  and the SMA-above-strike count. Credit spreads carry the same rule: above the
  20% credit/width target the ratio still decides the tier, but the credit must
  also clear 0.5% of the short strike (or $0.40), because the width is capped at
  10 points however expensive the stock is and the ratio therefore stops scaling.
- BounceBot completed-M5 detection with session VWAP/bands, EMA and prior-day
  levels, relative strength/weakness, regime-aware candidate discovery, tiering,
  alerts, outcome tracking, and the day-scoped M5 Focus path.
- BounceBot's sweep runs only inside the session window (open-30m to close+30m by
  default, weekdays); outside it Auto Pilot pauses scanning and holds the IB
  connection open. A manual resume survives until the next boundary.
- CandidateRegistry foundation with provenance, source leases, transitions, atomic
  versioned persistence, and partial shadow adoption. Full authority remains open.
- Industry Board with one single-flight owner, hourly refresh, atomic last-good
  snapshot, numeric sorting, freshness/Health integration, and advisory aligned
  industry-vs-SPY plus stock-vs-primary-industry fields. Since 2026-08-31 the 60 s
  check tick emits `snapshotChanged` only when `snapshot_id` moved, so an unchanged
  board is never re-read or re-measured (snappiness packet 1 item 2).
- **Snappiness packet 2, 2026-08-31.** The Alert Center's minute tick materializes
  each symbol's M5 bars once per series rather than once per caller (eight
  timer-driven sites asked), builds each symbol's D1 reference levels once per
  tick rather than once per event kind, and issues ONE batched chart prefetch per
  event-loop turn rather than ~105 single-symbol tasks. The GUI-thread collector
  sweeps the startup heap once at launch and then `gc.freeze()`s it out of every
  later sweep. The journal's retag runs on a single-flight worker with the
  buttons disabled and failures shown, its scanner-file parses are cached per
  file version, `list_trades` resolves every trade's regime in one query instead
  of one connection per trade, and the filter header debounces at 250 ms.
- M5 Strength Board (`strength_scan` + one `StrengthBoardService` owner, 15-minute
  single-flight refresh on the quiet-hours window, last-good on failure): batched
  yfinance over `universe_all.txt`, **zero IB traffic**, every column click-to-sort
  with blanks last, a row select charting through the desk's one snapshot popup, and
  every add re-running the M5 Focus adoption gate at click time with the refusal
  reason named. **Since 2026-09-04 its TC2000 parity rows also join M5 Focus by
  themselves** (packet T1.4, trader: *"I want all shorts and longs on the RS/RW
  board TC2000 to bne auto added to the M5 focus picks"*):
  `_auto_adopt_strength_board` runs on `boardChanged` and once at attach, over
  rows with an EMPTY `failed_floors` only, re-running the one adoption gate on
  each row's own numbers (UNKNOWN fails), skipping any symbol in
  `_ignored_symbols` so a "Not today" survives the next refresh, **DESK only**,
  writing through the STORE plus `mark_auto_adopted` and **never
  `FocusService.add`** - a machine placement is not a trader like. It never
  removes, never re-marks an existing entry, and writes one
  `strength_board_auto_focus` review event per refresh. **Since 2026-08-31 it is a collapsible section under the Desk's
  Strength window rather than a left-nav page** (trader request): starting closed so
  it costs the charts nothing, sides stacked vertically for the column, its own
  RS/RW half retired to the Alert Center's RS/RW Board tab (one tab-click away in
  the same column), and a row click charting into the **Visual Alert Review pane**
  through `chart_symbol` rather than opening the snapshot popup. **Since 2026-09-03
  every ticker click on the Trading Desk does the same** (trader: *"the main tab
  should always be centralized with the main chart"*): the Alert Center's RS/RW,
  entry and Focus-strength boards and the feed ticker-name click always chart in
  the pane; the setups column's four panels (setups table, RS Window, Industry
  Board, Watchlists) do so through a `set_chart_sink` the desk sets in workspace
  mode and clears in tabs mode. **A board chart holds NO place in the waiting list**
  (packet T1.3, 2026-09-04, trader: *"once i look and click off, its done"*):
  `_is_manual_chart_look` is an exact `MANUAL_CHART_TAG` test, a look is never
  re-queued and never skip-counted - it was never a shown alert, so it belongs in
  no P(take | shown) denominator - while the M5-alert-bar `skip` with
  `clicked_away_from_m5_alert` and the dequeued-D1 return-to-head rule are both
  untouched. The popup remains the door for a board on another
  page (`show_board_symbol`, the AWAY Recap) and for a standalone panel.
- Auto-populate rules for both regimes, previous-day-extreme gating and DESK
  adoption into M5 Focus. A Focus pick's AUTOMATIC D1 alerts are the pullback set
  only (2026-09-01); the extension set fires solely from a trader-armed D1 event
  watch, through the separate armed poll, so an extension event has exactly one
  path. Supersedes the earlier one-extension-per-name-per-day ration.
- Armed alerts expire on the TRADING-day clock (`scripts/armed_alert_expiry.py`):
  5 sessions for a manually armed 5d extreme watch, 10 for a 20d one, 10 for D1
  level watches, any-bounce watches and manual price alerts. Uncertainty never
  deletes; every expiry appends a row; a price alert is disarmed, not deleted;
  arming restarts the clock. No new timer - each expiry rides the poll that
  already owns its store.
- A Focus pick with no alert and no pullback event for 10 trading days FADES to a
  reversible faded list (`focus_pick_clocks.json`, `focus_faded.json`,
  `focus_fade_events.jsonl`), swing and M5, the trader's own included by explicit
  2026-09-01 authorization. Activity resets the clock; restore gives a fresh one;
  discard leaves the evidence. A faded swing favorite gets a RETRACTION row, never
  an edit, and no `pick_feedback` verdict is written for a fade.
- The strength board's buttons carry their counts - "Focus pick review (N)" and
  "Faded review (N)" - and the faded walkthrough charts through the one review
  door with `FOCUS_FADED_TAG`, which bypasses movers-only.
- Focus privileges begin only beyond the previous session's directional extreme;
  missing prior-day data grants nothing.
- D1 Focus routes final Favorite/High Conviction upgrades while developing trigger
  evidence remains research-only. Legacy D1 champion alerts are unchanged.

### Charts, review, alerts, and phone surfaces

- Chart-first review flow, current forming D1 preview, D1/M5 shared snapshot widget,
  log scale, crosshair/OHLCV readout, source/age strip, fallback warning, cache
  invalidation, background loading, prewarming, and stall watchdog. **The
  watchdog's record cap is per HOUR** (F1, 2026-09-03): 2,000 an hour, session
  total untouched. A per-session cap was spent overnight on an idle desk and the
  log went blind at 06:03 on the morning the trader reported the desk unusable.
- Chart Review workspace with lookup for any symbol, hidden-by-default Setups drawer,
  keyboard-first LIKE/veto/note/setup-claim capture, versioned veto vocabulary,
  append-only `trader_annotations.jsonl`, and isolated forward veto cohorts.
- Day-trade **pass** capture under the Note section of the capture rail (2026-08-31):
  multi-select reasons from a separate versioned `pass_reasons` vocabulary family,
  the same free-text note, and — only when the desk already holds them — one session
  of the symbol's M5 bars in a sidecar keyed by the annotation id. A pass writes one
  row and retires nothing; a capture click never fetches.
- Painted D1 S/R, previous-day H/L, projected trendline, SMA/EMA/AVWAP groups,
  machine-local visibility preferences, stable level IDs, click selection, and
  click-to-arm routed through the one `PriceAlertService` writer.
- Chart Review annotations cannot add Focus/watchlist membership or price alerts;
  LIKE records judgement only.
- Visual Alert Center and review queue, chart-armed watches, persistent History,
  structured review decisions, review scoreboard, and annotation-only/FIFO policy
  gate.
- **PROVEN is the top alert class, and since 2026-09-01 it is the only one.**
  BANGER was retired by trader decision ("We can probably remove this because idk
  what it is"): its only definition was a literal `"BANGER" in raw_text` match in
  the Alert Center, nothing in the tree ever emitted the token, and 0 of 8,818
  recorded review rows carried it. The matcher, the tier-gate bypass, the
  always-sound branch and both repetition escalations are gone; `is_banger` is
  REMOVED from `RepetitionLedger.consider` rather than ignored, so a stale caller
  is a loud error. The `banger` column stays in the review-event row as a constant
  `False` so historical readers and the row shape are unchanged. The
  `REGIME_BANGER_*` constants in `bounce_bot_lib/legacy.py` are regime-pause
  thresholds - a different thing - and are untouched.
- **The LRSI M5 alerts are retired and every row of their evidence is kept**
  (trader, 2026-09-01: "LRSI alerts seem to be mostly spam ... no need for their
  M5 alerts"; they were 84 of 128 new M5 episodes by 11:14 that morning).
  `LRSI_M5_ALERTS_RETIRED` gates the EMIT seam in `_emit_lrsi_cross_alert`, the
  same shape as `H1_ALERTS_RETIRED`: the sweep, the candidate row,
  `_register_bounce_outcome` (`intraday_bounce_outcomes.csv`), the learning tier
  and the PROVEN stamp all still run, and only `gui_callback` is skipped. **The
  detection toggles stay `True` on purpose** - `is_m5_signal_enabled` is tested
  before the event joins `hits`, so flipping them would stop the evidence rather
  than the noise. Unlike H1's, this retirement still calls `log_bounce_to_file`,
  because `journal_analytics.AutoTagger` reads `INTRADAY_BOUNCES_CSV` to name a
  trade's setup. No Settings toggle exists for these engines. The higher-timeframe
  LRSI warehouse study is the measurement the trader asked for and is untouched.
- **A click away from an M5 chart IS a pass** (trader decision 2026-09-01,
  confirming the 2026-08-27 mechanic): `_select_review_alert` writes a `skip` row
  with `detail.reason = clicked_away_from_m5_alert`, and that string is frozen
  because `review_learning` keys on it. What the trader wanted from the chart they
  take with the tabs under it - arm an alert, add to Focus - before moving on.
- **A human-focus pick is identified by its CATEGORY as well as its name**
  (2026-09-01). `human_focus_tracking._pick_key` returns
  (trade_date, symbol, side, category slot), so one name on both the swing and
  the M5 list gets one row per list and grades in both cohorts. The slot strips
  the like-origin suffix, so a re-snapshot under a newly-recorded origin adds
  nothing. Before this, whichever list was snapshotted second was silently
  discarded and `human_focus_swing_vetted` had zero rows in the whole file. The
  weekend-prep pick/outcome join uses the same canonical
  `pick_source_family`; `journal_walkaway` replays ONE position per
  (date, symbol, side) because the trader was in one.
- **A like merges into its cohort on the click, exactly as a veto does**
  (2026-09-01). `commit_like` and `commit_veto` share one
  `_merge_cohort_safely`, so they cannot drift; failure degrades to a
  "(cohort update deferred)" status and the next merge recovers, because the
  annotation row is already on disk. The nightly slot stays and both merges are
  idempotent. `merge_like_cohort_picks` now takes the writer lock the veto merge
  always took.
- **A pre-versioning veto pools with the version that INTRODUCED its code**, not
  with the lowest version overall (2026-09-01). A code added in a later
  vocabulary used to get no unversioned mapping at all, so its pre-versioning
  picks graded alone forever. Pooling still happens only in
  `_rebuild_pooled_performance`; rows are never rewritten.
- **The review scoreboard grades every explicit decision, and carries a third
  callout class** (2026-09-01). Seven action families joined the take/reject
  sets - `auto_pick_approve`, `focus_review_keep`, `arm_d1_event`,
  `arm_any_bounce` as takes; `auto_pick_pass`, `focus_review_remove`,
  `veto_day_trade` as rejects - about 640 decisions previously scored as
  silence. Machine events and disarms are deliberately excluded and pinned by a
  test. The new **`r_gap`** class fires on |taken.r_avg - passed.r_avg| >= 0.5R
  with >= 8 measured R per side and NO reference to the take rate, so it sees
  what the take-rate classes structurally cannot. It is report-only: it never
  reaches `review_policy.json`, `review_guidance` or the AI evidence package.
  Chart Review's coded vetoes now feed the `dislike_reason` dimension through a
  measured (session_date, symbol, side) join - 202 of 212, zero side
  mismatches - annotating only, never re-resolving an episode.
- **Weekend Prep's two judgement tables show the robust half** (2026-09-01):
  median, trimmed mean, symbols, sessions, top-symbol share, block CI and the
  evidence label, all written since R10.C and previously dropped. ONE horizon at
  a time (default h3) with a selector that re-renders from memory. `meets_n_floor`
  is not a column - it decides the ORDER and the greying, so a cohort under the
  floor sorts after every cohort above it and rows above it order by the TRIMMED
  mean. The liked table carries the same bounded-picklist caveat the AI gets,
  through the one `ai_summary._offered_claim_caveat`.
- **The week page names its callouts** instead of counting them: segment,
  dimension, shown, take rate, and what each half measured. It reads the classes
  defensively, so a scoreboard written with or without P1's `r_gaps` renders.
- **"My Decisions" sits beside the Daytrade Tracker** (2026-09-01): one tab per
  scoreboard dimension over `review_preference_state.json`, columns shown / takes
  / take rate / taken R (n) / passed R (n) / gap, badged `probation` by set
  membership in `M5_SIGNAL_TYPE_DEFAULTS - BOUNCE_TYPE_DEFAULTS`. Read on a
  daemon thread; the button also calls `refresh_review_learning_if_stale` exactly
  as `app.py` does, while construction only READS.
- **The five AI phase gates have a surface** (`ai_jobs/gate_counters.py`,
  2026-09-01): digest, enrichment, weekly synthesis, policy draft and evidence
  window, on one strip on the A.I. Summary page with each gate's own statement as
  the tooltip. Every number is READ from the source that owns it - the synthesis
  count through the same two functions the job uses, the draft and evidence counts
  parsed from the PUBLISHED files. An unreadable source says "unavailable", never
  zero.
- **The M5 alert bar shows the take rate and folds repeats** (2026-09-01). A row
  ends "take 28%" when the Alert Center already has guidance CACHED for that
  symbol, and is silent otherwise - never a 0%. A repeat of the same symbol+side
  folds into its row with a ×N badge and returns to the top carrying the newest
  alert; the other side of the same name is a different row. **Presentation only**:
  every event reached the review-queue door, the outcome CSV and the review-event
  store first, the folded row's tooltip says so, and Copy-all still lists one
  symbol per row.
- **Every verdict the trader can record now has a forward record** (P5,
  2026-09-01). Veto and like already did; the day-trade **pass**, **not_today**
  and **dislike** did not. Two new trios - `pass_cohort_*` and
  `rejection_cohort_*` - graded by the ONE existing
  `update_human_focus_outcomes`, summarised through `evidence_stats`, registered
  in `COHORT_BASE_BY_SOURCE_PREFIX` by APPENDING, with two nightly slots appended
  to `default_slots()`.
- **A pass grades in k+1 cohorts and they must never be summed.** A day-trade
  pass is multi-select, so it is written into one cohort per reason code AND into
  the pooled `pass_all`; only `pass_all`'s n counts passes. The overlap travels
  in the module docstring, in a `reason_code_count` column on every row, and in
  `OVERLAP_NOTE`, which the Weekend Prep note and the AI scope label read rather
  than retype. **The pass vocabulary is a separate family and is never folded
  into the veto's.**
- **A pass also carries a same-session grade when the desk held bars**: entry at
  the first completed M5 close AFTER the pass, stop at the session extreme on the
  pass side, target 2R, stop-first. When it cannot be computed the columns are
  BLANK and `intraday_unmeasured_reason` says which absence it is.
- **`not_today` and `dislike` are separate cohorts and their numbers are never combined into a verdict** (corrected R1: the family's pooled BASE row does exist and is labelled where it is shown) - a
  same-day throwback and a judgement on the name are different claims.
  `unfavorite` is not graded (a membership change, not a verdict, and sideless on
  the live log), and the free-text `reason` is carried verbatim and never coded.
- **`update_human_focus_outcomes` takes an optional `pick_key`**, defaulting to
  the existing identity so every caller is unchanged. A MULTI-SOURCE cohort - one
  where the same name on the same date legitimately grades under several sources -
  passes `pick_key_with_source`; without it a multi-code pass would collapse to
  one outcome row and k of its k+1 cohorts would vanish.
- Main-only price-level polling with cross-up/cross-down, one fire per arm, urgent
  ntfy push, persistent main-desk presentation, and manual re-arm.
- Auto modes OFF/DESK/AWAY/EVENING, honest global status, EVENING early scan and
  briefing, and one verified `autopilot_today.txt` with safety/freshness first,
  numbered best swings, intraday candidates, and condensed operations.
- **The daily pick scorecard runs on ONE owned worker** (`autopilot-scorecard`, packet
  Q5, 2026-09-04): the tick and the wrap-up decide, the read streams today's rows through
  `autopilot_core.read_scorecard_inputs` (never a materialised year), every group is scored
  before any row is appended, `picks_scored_at` is written only on SUCCESS, a failure keeps
  the last-good line and counts toward `SCORECARD_MAX_ATTEMPTS` (3, then
  `picks_scoring_failed_at` for the day), and a missing file is the one empty answer while
  any other `OSError` raises for retry.
- The double-click symbol snapshot popup opens at desk height (2026-08-11): its size
  is taken from the hosting window's frame, or the screen's available area when the
  window is not yet measurable, never smaller than the former fixed 1180x760, and is
  centered on the desk window and clamped inside the screen. Opening geometry only —
  a trader resize survives subsequent double-clicks.
- On 2026-08-10, best swings gained an ntfy report notification; it stays quiet when
  the generated swing section contains no readable setups. Late-opened alerts now
  receive current bars, and the Chart Review Setups column defaults hidden with a
  visible restore control.
- Phone push policy, 2026-08-11 (trader rule): **AWAY is the only mode that pushes**,
  and the Research/Focus price alerts are the single deliberate exception — they keep
  their own always-on urgent channel, unchanged. The EVENING morning-briefing push and
  the retired Desk Link control-reclaim push are now silent outside AWAY; both still
  announce on the desk. The hourly swing push carries the **full favorite and
  high-conviction roster** under the ranked picks, built from the whole current feed
  rather than the top-ten slice, side-split, with `near` excluded and an explicit
  "did not fit" marker if the message ever exceeds the ntfy size ceiling; a roster with
  no ranked picks still sends. A **second hourly push names every stock that fired a D1
  level or event alert since the previous one** (armed D1 levels, D1 event watches,
  Focus D1 flags, and the scanner's ready D1 focus alerts), new-since-last-push rather
  than cumulative, silent on an empty hour, and cleared only on a delivered push so an
  ntfy failure never eats the events. The Alert Center classifies (it owns the D1
  routing rules) and Auto Pilot aggregates and gates, so the phone and the D1 Focus
  feed cannot disagree. Machine-local kill switches: `push_away_swings`,
  `push_away_d1_events`. **Extended 2026-08-14 (packet R1):** EVENING's SPY ±1%
  wake alarm is the *second* deliberate exception — urgent, repeating every five
  minutes while the move holds, stopping on the flip out of EVENING, kill switch
  `push_evening_spy_alarm`.
- Auto-mode matrix, 2026-08-14 (packet R1): discovery is identical in every mode;
  what differs is who is present to act. DESK adopts staged picks immediately;
  AWAY stages and never adopts and queues alerts silently (only the sound is
  suppressed); EVENING runs its early block and then stops scanning entirely,
  staging picks for the wake-up flip; OFF is the only mode that still
  self-applies. Quiet hours confine every automatic starter to weekdays,
  06:00–14:00 local; manual buttons are never gated.
- **Today's swing picks**, 2026-08-31 (trader-directed): a strip at the bottom of the
  M5 alerts column where the trader types or pastes their own end-of-day swing
  targets with a Long/Short toggle. Two writes per add — the swing Focus
  write-through through the existing store, as the TRADER's entry with **no
  auto-adoption marker**, and an append-only row in `swing_favorites.jsonl`
  (`project_paths.SWING_FAVORITES_FILE`). A removal appends a RETRACTION row and
  drops the Focus entry; nothing is ever rewritten, and prior sessions stay in the
  store. A "took" badge marks a pick whose symbol has a TRADE-journal trade opened
  on or after the pick date — display only, joined on a worker thread over a
  bounded 10-day window, silent when the journal would have to be migrated to
  answer. The strip and the alert bar share a **draggable** vertical split with
  its own settings key, no collapse, and a chip area with a floor and no ceiling;
  **Copy** puts the day's tickers on the clipboard one per line for TC2000 and
  **Paste** adds a TC2000 list on the selected side. The Focus like-origin is
  **`vetted`**, so the picks grade as their own `human_focus_swing_vetted`
  sub-cohort in the existing 1/3/5/10-session human-focus tracker rather than
  mixing with every other hand-typed swing name. Diffed like the Focus board,
  styled by `theme.qss`, no phone push, and nothing in the chain reaches a
  detector, score, alert, watchlist ranking or `review_policy.json`.

- One Master AVWAP scan action, 2026-08-15 (packet R1). The Shared/Local pair read
  the identical two watchlist files, so `use_shared_watchlists` and the menu choice
  it drove were removed across thirteen files. Cloud-drive *store discovery* went
  with it (decision 0015 amendment); the mount-presence guard stays.

### Journal, explanations, and learning

- **R7/R8 adversarial release-candidate repair (2026-08-15).** Every verified
  A1–A19 and B1–B14 finding was closed before handoff. The repair normalizes
  broker-ledger casing and Flex dates; preserves shared Focus wiring and exact
  suggestion-row identity; scopes reconciliation clears to reachable brokers;
  bounds shutdown; migrates every execution leg and gives fills stable
  identities; makes coverage, quarantine, currency, FX ordering, token
  precedence, weekly identity, exit-window, empty-last-good, and OCC handling
  fail honestly; and restores the journal's missing pull/gap/retry controls,
  grouped tags and filters, reversible undo, atomic exports, and truthful labels.
  Expensive journal work now runs in a worker and re-renders from captured
  structured results without re-querying; migration starts only after an
  explicit **Prepare Journal database** click and remains visibly gated in the
  background. Weekend rollover, timezone conversion, failed-discovery state,
  Flex reuse, single-fetch boards, board persistence, and failure signaling are
  likewise pinned by regression tests. Account tax labels moved out of source
  into machine-local settings. No live journal database or broker was touched.

  Scope reconciliation is explicit: true non-USD-to-USD conversion, the
  Calendar year heatmap, additional Analytics charts, Weekend RRS-strength
  joins, and Weekend Focus performance/pick-feedback/veto joins remain deferred
  in their governing specs. They are not represented as shipped behavior. The
  repaired code tip is `dd201cd`; deterministic baseline is 3354 passed / 19
  subtests, smoke 7/7, frozen selftest 49/49, all exit 0. Live gates remain owed.

- **Weekend Prep (R8, 2026-08-15).** A guided five-step weekend routine with
  persisted progress: week in review, focus-pick review, week-windowed walk-away
  with the weekly auto-tag review, strength discovery on H1/D1/Monthly using the
  M5 formula through the fenced `strength_scan` functions, and the week-ahead
  prep from the `market_prep` weekly engine. Manual refresh only, zero IB
  traffic, adds-only adoption into swing Focus.

- **Tax-grade journal (R7, 2026-08-15).** Stable `BROKER:account:exec_id`
  execution identity; one security-type vocabulary across both brokers; anchored
  `trade_id` with an annotation re-key pass and `trade_aliases`;
  `CLOSED_PARTIAL` and a `SYNTHETIC_OPEN` marker instead of a fabricated inverse
  position; append-only `trade_adjustments` corrections re-applied at every
  rebuild; an `import_coverage` ledger with a bounded nightly self-heal; IBKR
  Flex as the primary history source including OptionEAE, OpenPositions and
  CashTransactions; Questrade activities and a trade-day cross-check; Bank of
  Canada FX booked once per (date, currency); reconciliation against both
  brokers' reported positions with trader-confirmed force-closes; a nightly
  `journal_import` slot at the front of the `ai_jobs` slate; and a five-tab
  Journal (Trades, Calendar, Analytics, Health, Fees) over one shared
  tax-grouped header.

- **Broker-stated tax report (2026-08-28).** `scripts/journal_tax_report.py`
  reports realised P&L for a year by summing the broker's own `net_amount` per
  fill — never recomputed. Open positions, positions with an invented opening
  fill, and fills with no stated amount are excluded and named. CAD per fill at
  the booked BoC rate. Journal > Fees > "Realised P&L for tax...", with a CSV.
- **File authority over the live sync (2026-08-28).** `scripts/journal_file_authority.py`
  compares a broker file against the sync per `(account, day)` on computed signed
  cash. The sync keeps a day they agree on, so its trade times survive; the file
  takes a day they do not, retiring the sync's rows with append-only
  `VOID_EXECUTION` adjustments. Runs as a dry run behind "Check a statement...".
- **IBKR transaction-file import (2026-08-28).** `scripts/journal_ib_transactions.py`
  reads IBKR's sectioned csv: per-section headers, costs converted from the base
  currency by the rate each row implies, masked account numbers unmasked only
  when exactly one known account fits, assignments treated as fills, options
  already OCC. One Health-tab button serves both brokers and reads the broker
  from the file's contents. Commission now carries a SIGN through the store and
  the assembler, so a broker credit stays a credit.
- **Statement layering, direction and self-check (2026-08-28).** Statement
  identity is `fill_signature` + an ordinal within it, so a later, longer export
  layers instead of doubling; long vs short is read from Questrade's own
  `STOCK SHORT.` / `COVER SHORT.` description marking rather than from row
  order; and `reconcile_statement` adds a file up by hand and compares it to the
  assembled trades, per symbol, writing a CSV. Journal > Health >
  "Check a statement...".
- **Broker statement import (2026-08-28).** `scripts/journal_statement_import.py`
  reads a Questrade activity export (.xlsx via `zipfile`+`ElementTree`, no new
  dependency; also .csv) and writes executions, cash rows and account tax status
  for the days the executions endpoint's retention horizon can no longer reach.
  One commission column taken as the whole cost; options resolved from the
  Description into OCC symbols with a 100 multiplier; timestamps at midnight
  market-local so a date-only row is never given a session; and a statement
  never writes into a (broker, account, day) a richer source already covers.
  Reachable from Journal > Health > "Import statement file...".
- **One name per setup: the frozen registry** (P7, 2026-09-01). `scripts/setup_registry.py`
  over `scripts/setup_registry_v1.json` - 57 entries keyed `setup_id@version`, joining the
  FIVE places that name a setup: `_FAMILY_TAGS` (the canonical warehouse id), `setup_docs`,
  the playbook study, the claim picklist, and `legacy.py`'s `*_STUDY_FAMILY` constants
  (eight families are named ONLY there). Regenerated by
  `scripts/build_setup_registry.py --write` and reviewed as a DIFF; never rebuilt at
  import. It **resolves no disagreement** - eight `known_divergences` record what each
  source believes - and **fills no column its sources do not establish**, so supported
  sides and timeframe roles are deliberately blank. An unknown name RAISES rather than
  defaulting to GENERAL. **Not authoritative until `plan.md P4.1`**; its only readers are
  the fact pack's role lookup and the selftest's asset check.
- **The look-counter** (P7). `scripts/research_warehouse/trial_ledger.py` writes one
  append-only JSONL row per registered grid at
  `<store root>/_diagnostics/trial_ledger.jsonl`, at REGISTRATION time and never
  rewritten - `register` refuses a `trial_id` already on file, and every row carries
  `registered_at`. Written by `cli.run_build` beside the coverage line. Five grids
  declared, including the four that predate it.
- **The first setup-parameter grid** (P8, 2026-09-02). `SETUP_ENTRY_TIMING_RECIPES`: 12
  cells over `AVWAPE_TO_FIRST_DEV` LONG (840 occurrences, 622 clusters) asking whether an
  entry that WAITS for confirmation beats the next session's first completed M5 close.
  Four entry moments x three targets, **one structural stop (`current_anchor:1`) and one
  exit machine** - the control delegates to `simulate_m5_close_opportunity` unchanged, so
  it reproduces the `m5close_current_anchor1_*` rows by construction, and the three
  challengers use the SAME function through one optional `entry_selector`. Every recipe is
  `is_diagnostic=True`, the twelve are correlated diagnostics of ONE episode, and the
  declared 20-session window means no cell is read for a verdict before it closes.
- **Two-lane journal auto-tagging (2026-08-28).** `scripts/journal_trade_shape.py`
  derives hold bucket, entry session bucket, execution shape and instrument from a
  trade's own timestamps and legs, so history imported from outside the scanner's
  lookback is tagged rather than blank; `AutoTagger`'s setup lane still leads both
  the stored summary and the candidate list, ordered by lane rather than confidence.
  No tag is ever derived from the outcome. Around it: a tag filter on the shared
  Journal header, `distinct_tags` counting the trader's lane separately from the
  machine's, `rename_tag` (rename or retire across every trade, trader-typed tags
  only), a Manage-tags dialog, Accept-all, and an accepted suggestion that stops
  re-proposing itself.
- **A fifth auto-tag lane that is not a guess** (P6, 2026-09-01). `trader_capture`
  offers what the trader ALREADY SAID about the symbol - a veto, a like_claim, a
  pass or a take-class review decision - when the statement falls inside THE
  TRADE'S OWN WINDOW (open date to close date), never the fuzzy neighbourhood the
  scanner lanes search. It ranks ABOVE every fuzzy source and a fuzzy match can
  never displace it. A rejection is PREFIXED (`vetoed:` / `passed:`) so it can
  never read as an endorsement in a Tags column. Each candidate carries
  `context_row_id`, **a pointer for a reader and never a canonical link** - plan.md
  P5.3/P5.4 own the canonical id. Nothing here writes `trade_annotations`, and no
  tag is derived from an outcome.
- **"What I said, what I did, what happened"** (P6, 2026-09-01,
  `scripts/preference_trade_outcomes.py`, nightly deterministic slot + a Weekend
  Prep table). One row per statement across four channels - like_claim, pass, swing
  favorite, `pick_feedback` like - joined to the journal and to the cohort paper
  grade. **Every row renders its match confidence or says "no match"**, with
  `match_basis` naming what the match rested on; the join is a JUDGEMENT, because
  a trade on the same name that week may have been taken for another reason.
  Read-only, mints no identifier, and an unmatured paper grade is blank rather than
  zero. The swing strip's "took" badge now names its trade in a tooltip through the
  SAME matching rule that put the badge there - the id is EXTRA and never a
  condition for the mark.
- **A dimension resting on almost nothing says so** (P6, 2026-09-01). Below 10%
  confirmed-tag coverage the journal's "My setups" group is prefixed with one
  sentence naming the coverage. **The group is never hidden**: hiding it would
  replace a visible thin answer with an invisible one, and seeing how little is
  tagged is the prompt to tag more.
- **Every like and every dislike, from every screen, writes ONE annotation row**
  (P10, 2026-09-02). Trader: a star in Master AVWAP setups and a like in chart
  review are the SAME thing - one bucket, graded together, and the screen is a
  COLUMN (`surface`: `master_avwap_setups` / `chart_review` / `focus_panel` /
  `m5_alert_bar` / `rail`), never a second cohort. One writer,
  `ui/annotations/verdicts.py`. The review event, the `pick_feedback` row and the
  Focus removal are all unchanged and still happen; the annotation row is the
  ADDITION and its failure is swallowed. **An UNCODED veto is legal** and carries
  no `vocab_version` - a version on a row that cites no vocabulary would file it
  in a pool it was never part of - and grades as `veto_uncoded`, never pooled
  with a coded cohort. Those rows were previously SKIPPED, so "Not today", the
  desk's most-used dismissal, had no forward record at all.
- **The note is a SECOND row and the click goes first** (P10 A2). Joined by
  `supersedes`, never an edit. If the box came first, Escape would mean the click
  never happened - which is exactly the case the trader named. The dialog is
  **MODELESS** (`open()`, not `getMultiLineText`): a nested event loop would sit
  between the click and the queue advancing, and in a headless test it never
  returns at all. It opens only where no quick button was used.
- **A verdict on a scanner row records which search found it** (P10 B1):
  `scan_date`, `tracker_setup_id`, `canonical_setup_id` (P7's registry),
  `priority_bucket`, `score`, `expected_r`, all copied from a row the desk was
  ALREADY showing. **A capture click never fetches**; a bare lookup stamps
  nothing, because absent is a real answer and `""` is not.
- **A like links to a warehouse occurrence, and absence is a row** (P10 B2,
  BD-90). `bronze_like_occurrence_link`: basis `exact_family` / `any_family` /
  `none`, window ONE session back and FIVE forward (the trader's own range),
  `candidates_in_window` beside it. A like with no occurrence is written with
  basis `none` - dropping them would report on the subset the scanner happened to
  find. `queries.occurrence_features` finally builds the round-1 audit's item 6:
  the latest snapshot on or before the trigger, and never a later REVISION of the
  right session.
- **The after-like grid is registered, bounded and shadow** (P10 C, BD-92/93):
  `after_like_entry_grid_v1`, 20 cells (5 day offsets x 4 entries), ONE stop and
  ONE target so a winning cell cannot have won on either, floors counted on the
  LIKE EPISODE, a 20-session window fixed at registration. Rows are keyed by the
  like episode rather than the occurrence - two likes on one occurrence would
  otherwise collide on `outcome_path`'s grain. **The unlinked bucket is a COUNT**:
  the declared stop needs the occurrence's anchor, and a substitute stop would end
  the grid's one-stop model.
- **A VETO and a CLAIMED like retire the chart; a QUICK like and a NOTE never do**
  (packets T1 and T2, 2026-09-04, trader: *"i still need time to enter alerts"* and,
  second pass, *"double clicking that box should advance the chart"*). A capture-rail veto has its own
  verb - `AlertChartReview.vetoRetireRequested` -> `_retire_after_veto` - and
  writes ONE coded row with **no note box and no uncoded second row**; the
  "✕ Not today" BUTTON keeps `removeTodayRequested` and is unchanged (uncoded
  row, box, advance), and the day-trade veto retires through the box-free verb
  after its Focus placement. Both retirements are ONE body with a flag
  (`_retire_review_alert`), so the auto-pick / faded / Focus-review branches
  cannot drift. A QUICK like is reported through `likeRecorded` -> `_after_like`, which
  records the review event and moves nothing; a CLAIMED like goes
  `likeAdvanceRequested` -> `_advance_after_like` -> `_advance_review_queue`
  (packet T2), and an advance is NOT a retirement - no park, no Focus drop, no
  sweep of the symbol's other queued alerts, no placement. `like_mode_of` picks
  the route and absence reads as claimed. **Both record the event named
  `like_advance`, through one helper (`_record_like_advance`)**, because
  `review_learning.TAKE_ACTIONS` keys on that string.
- **`note_vocabulary_audit`** (P10 A4): a deterministic nightly slot listing the
  day's notes beside the vocabulary that exists. It proposes no code and adds
  none - a vocabulary code is permanent and never reused.
- **A LIKE has two modes** (P9, 2026-09-02). **Alt+L** writes a QUICK like -
  `like_mode: "quick"`, no claimed setup, no why - and **Alt+K** the claimed one,
  which since packet T2 (2026-09-04) needs only the CLAIM - the why is optional on
  every like path now, a double-click on a setup is the whole gesture, and
  `_prompt_for_why` is deleted. A quick like LEAVES THE CHART UP (2026-09-04, packet
  T1.2; it retired until then, and a CLAIMED like advances again since T2), records
  `like_advance` and marks the symbol reviewed exactly as a claimed one does, and
  **places nothing**: a like carries zero privileges (plan.md P3.1). It grades
  under `like_unclaimed`, saves the M5 sidecar on an M5 chart through the writer
  Pass uses, and contributes a **LINK** to the auto-tagger rather than a tag,
  because it names no setup. `like_mode` is ADDITIVE - schema stays 1, proven
  against every reader - and its absence reads as `claimed`.
- **A capture sidecar is completed after the close** (P9, `sidecar_completion`
  nightly slot). The snapshot holds what the desk had AT the click, so the
  intraday grade's entry bar - the first completed close AFTER it - was never in
  it and every live pass graded blank. The slot appends the rest of the session
  from the research lake (narrowed Arrow-side, never materialised) or the desk
  cache, into a **NEW file and a NEW field**; the original reference still means
  what it always meant and is never rewritten. **This makes the intraday grade
  reachable, which answers gate 34's open definition question without changing
  the definition.**
- **The backlog is tagged, provisionally, and the mark is permanent** (P6a,
  2026-09-01). `trade_annotations.tag_status` carries `confirmed` (the trader's),
  `provisional` (machine-applied, awaiting review) or `needs_review` (the tagger
  looked and would not guess - **no tag at all**). Existing rows became
  `confirmed` through the column's DEFAULT, so nothing had to decide that after
  the fact. `scripts/journal_bulk_tag.py` is **the single authorized exception to
  I7**: dry run by default, idempotent, refuses a confirmed row inside the STORE
  rather than in the caller, never promotes a shape tag, never writes
  `tag_corrections`, and appends an inert `APPLY_PROVISIONAL_TAG` adjustment
  naming the candidate behind every tag it applies. Threshold **0.70**, chosen to
  encode a sentence - "the tracker or a focus favourite named this symbol, on the
  day I traded it, on the side I traded" - not a percentile.
- **"My setups" counts only what the trader confirmed** (P6a). `provisional
  setups` is its own analytics group beside it with no catch-all bucket, the two
  are **never blended**, and the chart says which is which. In the Trades tab a
  tag-review filter narrows the rows ALREADY LOADED (no query - `reload()` is that
  tab's expensive half and runs on the Qt thread) and counts what it hid; the Tags
  cell says `(provisional)` in text, because a `QTableWidgetItem` cannot be
  reached by `theme.qss`. One click confirms; an edit replaces. **Only an edit
  teaches the tagger** - agreeing with a guess would raise that guess's own
  confidence forever.
- Journal schema v2 with append-only opportunity lifecycle events, idempotent broker
  Taken/Closed imports, structured reviews, free-form notes, tags, and analytics.
- Deterministic novice explanations across Setup Tracker, Day Trade Tracker, and
  Move Forensics, plus an evidence-floor-aware “What’s Working” summary.
- Review events partitioned by installation, merged/deduplicated by readers, capture
  audits, preference scoreboard, AI-curated `review_policy.json`, and a permanent
  no-suppression boundary.
- Technical Integrity research hierarchy with point-in-time predictions/outcomes,
  break pressure, calibration report, and no detector/watchlist/alert influence.
- Regime infrastructure evidence for SPY baseline, breadth, Technical Integrity
  follow-ups, and audit tooling. The evidence remains exploratory/non-promotable.

### AI and automation

- Provider-neutral A.I. Summary workspace for OpenAI and Anthropic, explicit evidence
  selection, bounded preview, credential-manager storage, structured/source
  validation, immutable evidence packages, and export-only results.
- Config-gated local OpenAI-compatible provider through Ollama, default off; small and
  medium model tiers verified on the Ryzen main desk with no market-hours inference.
  The local large tier is `RETIRED` (2026-08-10): 27B-class models no longer load
  beside the running desk on the 780M, so its jobs belong to the frontier model. Local
  calls are capped to the tier's context window and fail loudly on server-side prompt
  truncation.
- Separate off-hours `ai_jobs` process and scheduled task, job-ledger integration,
  deterministic evidence coverage, daily advisory summary, per-ticker briefs, full
  artifacts in `ai_store`, and bounded atomic `ai_morning_brief.txt` publication.
- Per-ticker briefs project each symbol out of a full-size base package and then
  ration the projection to the local context window; ticker-roster and bare-name
  lines are discarded as non-evidence, each symbol resolves independently, a symbol
  with no evidence beyond watchlist membership is answered without a model call,
  completions resume on a read-stamp-independent evidence key, the morning file is
  republished after every resolved symbol, and the slot spends at most three attempts
  a session.
- Local-AI Phase 0 is complete. Phase 1 implementation is complete; its five-session
  unattended live gate remains in `plan.md`.

### Durability and catch-up

- Repeating 06:00 Pacific weekday launch task through the session, protected by the
  existing single-instance guard.
- Master AVWAP tracker staleness catch-up from completed prior-session D1 data with
  explicit `data_session` vintage and no automatic scoring-tuner/prior-refit side
  effects.
- Technical Integrity follow-up and breadth-ledger deterministic backfill with
  bounded retries, explicit `capture_mode`, honest gap rows, and live/backfill audit
  separation.
- Frozen snapshots, never-started predictions, and other Tier-C evidence remain
  intentionally non-reconstructed.

### Research warehouse

- Phase 0: research-lake decision record, configuration, home-folder-path refusal, layout,
  and disabled-by-default no-op behavior.
- Phase 1: immutable Parquet store, four-step seal, append-only manifest authority,
  13 frozen schemas, quarantine, compaction, retirement, and crash reconciliation.
- Phase 2: idempotent bronze wraps, daily universe/level snapshots, and completed D1
  projection with source hashes and watermarks.
- Phase 3/3b: zero-extra-request M5 tee, coverage/gap rows, capped spool, capture-only
  pacer, IB backfill transport, nightly/weekly backfill, and trickled yfinance seed.
  Since 2026-09-03 (BD-96) the tee de-duplicates BEFORE any per-bar work against a
  persisted per-symbol high-water mark (`tee_high_water.json` beside the spool,
  never reset by a clock), the seal de-duplicates at the dataset grain and counts
  the drop (`SealResult.rows_deduplicated`), and `research_warehouse.cli dedupe`
  (dry run; `--apply` rewrites) repairs a partition through
  `ResearchStore.dedupe_partition`, a COMPACT-shaped rewrite that keeps the earliest
  `observed_at` and writes `rows_dropped` on the manifest line.
- Phase 4: versioned XNYS sessions and deterministic M15/M30/H1/W1 aggregation.
- Phase 5: point-in-time daily/intraday feature snapshots and anchor instances using
  champion calculations, including AVWAP parity at 1e-9. **Anchor instances come from
  `earnings_avwap_anchors.csv` (bronze wrap) and nothing else; since 2026-09-04 the D1
  scan appends every symbol's cached current and previous earnings anchor to that CSV
  (`runner.bridge_earnings_anchor_caches_to_csv`, append-only, de-duplicated) - before
  that the CSV held 14 hand-imported rows and the swing bands were 99% null.**
- Phase 6: deterministic occurrence/revision/episode identity and versioned swing and
  intraday outcome simulation with costs, ambiguity bounds, partials, time stops,
  slippage, and open/truncated states.
- Phase 7: manifest-resolved read path and read-only Research panel; DuckDB remains
  optional and pyarrow can answer every slice.
- Phase 8: three-class backups, restore check, single-flight build/status CLI, job
  ledger, and six Health tiles.
- Defect passes repaired outcome supersession, management bounds, feature windows,
  per-bar backfill dedupe, pacing clocks, gap semantics, session identity, compaction
  reads, every job invoker, live tee wiring, and off-GUI-thread spool I/O.
- Phases 0–8 are code-complete on the testing-week branch. The broker check,
  confirmation items, and 20-session pilot remain open.
- **The model is narrated a VIEW of the fact pack, never the pack** (R3, 2026-09-02).
  The pack is the deterministic product and it outgrew the window - 437,125 chars
  against ~78,000 readable - so the server sheared it silently three nights running.
  `narration_view` sends the gate, coverage, evidence shape, excluded families, EVERY
  eligible cell and COUNTS of what was dropped; the ineligible rows, context cells and
  raw outcomes stay on disk. The four prose constants each cell repeats verbatim are
  stated ONCE under `conventions` (a constant two cells disagree on is never hoisted).
  437,125 -> 38,184 chars. Over budget **raises before any provider call**, and the
  evidence hash is over WHAT WAS SENT. A missing narration returns **`ok`**, never a
  retryable status: this job is a ten-minute lake pass, and re-reading the lake cannot
  shorten a prompt. Every pack carries `built_by_commit` (fails open to `"unknown"`)
  and the `recipe_ids` its rows came from, **never re-derived from the module**.
- **The nightly fact pack states its own evidence shape** (2026-09-01, BD-81…85).
  Every cell reports `n_episodes` beside `n`, and the pack reports `evidence_shape` -
  rows, occurrences, episodes and rows-per-occurrence - because the correlation the ERD
  warns about is ACROSS cells, not inside one (measured: `n` == `n_episodes` in all 756
  cells, while 9,372 rows rest on 599 occurrences and 287 clusters). **The eligibility
  floor still counts ROWS**; moving it is a cross-cell change and its own packet.
- **The pack leads with what cleared the floor.** Two blocks - eligible whole, then a
  bounded ineligible block ordered by n DESC - with drop counts per block, so a
  single-trade cell can never sit above the answer.
- **Non-trade families are excluded and reported.** `GENERAL` = FALLBACK and
  `FAVORITE_ZONE_WATCH` = WATCH_STATE per Appendix C; anything unnamed is a TRADE setup.
  Their counts still publish, because absence is a first-class fact. Packet P7's setup
  registry replaces the map.
- **Outcome bucket coverage is recorded per firing**
  (`research_warehouse/outcome_coverage.py`, append-only under the store root), so a
  pack can say "not measured yet" rather than implying "measured and flat". No history
  reads UNKNOWN, never zero.
- **`slice_readout` can read every family** (`setups=None`) while `SLICE_SETUPS` stays
  the pinned Phase-6 slice - it also decides what the warehouse SIMULATES. The Research
  readout panel gained a family filter and the `n_symbols` / `n_sessions` /
  `n_truncated` / `as_observed_only` columns the query always computed.
- **The attribute leaderboard has a desk surface** (2026-09-01, P4 A1): an
  **Attributes** tab on the Setup Tracker over the ~190-attribute export the scanner has
  always written, floor-clearing rows first and sub-floor rows greyed, labelled and last.
  Read OFF the Qt thread - alone among that page's exports - because it is 19.7 MB.
- **Twelve swing variables are recorded and none is weighted** (P4 A2): human focus
  pick/side, tracker setup family, market regime, sector, industry, ATR as a PERCENT of
  price (beside the dollar bucket, never replacing it), signed SMA200/SMA50 distance in
  ATR with two booleans, and relvol. A contract-bearing golden frozen from the pre-change
  code proves the priority score, bucket and expected R are unchanged.
- **The attribute leaderboard states its sample floor** (P4 B1) through
  `evidence_stats.summarize`, asked of CLOSED setups, with every row kept. The offline
  tuner's own gates still decide what may influence scoring.
- **It can also be read by family and by regime** (P4 B2) as sibling files; the export
  the tuner reads keeps its exact grain.
- **Scan-factor rows whose horizon is fiction are dropped and counted** (P4 B3). The
  horizon indexes a symbol's own scan rows, not exchange sessions. A row whose drift
  could not be measured is KEPT. Re-selecting the future row remains a sec-7 promotion.
- **The tier tracker grades the tier that SHIPPED** (P4 B4): `assigned_tier` is stamped
  after the expected-R demote, the de-dupe and the best-swing merge, and `tier_source`
  says whether a row was graded by the decision or by the old bucket derivation.
- **Expected-R calibration reads structure points** (P4 B5), not the proven-quality score
  that already contains realized performance.
- **The headline per-setup R names its exit template** (P4 B6).
  `REPRESENTATIVE_EXIT_TEMPLATE_ID` defaults to today's behaviour, and `setup_docs.py`
  now says the headline R is not measured on the house plan it documents.
- **The post-scan build runs in a CHILD PROCESS, never a desk thread** (F1,
  2026-09-03, BD-95). `ScanService.start_warehouse_build` spawns
  `research_warehouse.cli build --run-id <id>` at BELOW_NORMAL priority (frozen: the
  app's own `--warehouse-build` flag, since a frozen `sys.executable` cannot take
  `-m`), registers it for the shutdown reap, and waits on it from one thread that
  blocks on the child's pipe. It was a `qt-warehouse-build` THREAD, which held the
  GIL in **82.7%** of py-spy samples while the GUI thread got 2.3% - an unusable
  desk, for a 27-57 minute build, four times a session, all inside RTH. LD-01
  specified a CLI build job; in-process was the deviation. The parent-side
  `warehouse_enabled()` gate, the one-build-at-a-time rule and
  `wait_for_warehouse_build` are unchanged, and a reaped child is safe because
  `single_flight` reclaims a dead holder's lock. The child is OWNED (reaped at
  shutdown, present in `owned_scan_process_snapshot`) but is **not a scan
  child**: `owned_scan_process_count` gates whether a new scan may start, and a
  half-hour build must never be the reason a scheduled scan is refused -
  `owned_build_process_count()` answers for builds.
- **The XNYS exchange calendar is memoized** (F1). `holidays`, `half_days` and the
  session builder behind `trading_session` are `lru_cache`d - **84%** of that build
  thread's samples were recomputing them once per M5 bar per occurrence. 20,000
  `session_for` calls: 0.25 s -> 0.0114 s. The cache sits behind
  `trading_session` positionally because `lru_cache` keys on the call SHAPE, and the
  returned holiday dicts are shared and must never be mutated.

### Testing, packaging, and platform

- **`ruff` is installed and actually runs** (2026-08-31, trader-directed). It was
  declared in `requirements-dev.txt` and configured in `pyproject.toml`, but was
  absent from the `.venv` and unpinned in `constraints.txt`, so the configured
  lint had never been executed against this tree - every "ruff clean" claim in the
  history predates an installed linter. Pinned `ruff==0.16.5`. `extend-exclude`
  now also covers the legacy Tk shims (`master_avwap_lib/gui.py` + `runner.py`,
  `bounce_bot_lib/gui.py`, `gui_app/`), which re-export their names out of
  `legacy` at import time and so report 1,591 "undefined" names a static reader
  cannot resolve - noise that buried five real ones. Repo-wide: **1703 → 75**.
  All five real undefined names were fixed, and the 74 remaining unused imports
  were swept the same day: **`ruff check .` reports `All checks passed`.**

- Broad pytest suite, deterministic smoke check, pytest markers, narrow Ruff gates,
  layered requirements with constraints, and Windows/macOS path handling.
- Provider telemetry at IBKR/Yahoo/Nasdaq boundaries with completeness contracts and
  honest UNKNOWN until measured.
- PyInstaller onedir spec, Qt runtime hook, asset/package drift test, lazy-engine
  `--selftest`, and a permanent guard preventing self-test from demanding packages
  deliberately excluded from the bundle.
- The first Windows frozen run found and closed an `ai_jobs` bundle-roster conflict.
  **The frozen self-test count is a RUNNING TOTAL that grows as checks are added**, so
  it is not restated here: it was 29 on 2026-08-09 and 74 on 2026-09-02. Compare a
  frozen run against the CURRENT unfrozen count on the same tree, never against a
  number recalled from a document - which is what this line used to invite.
- macOS launcher, CloudStorage Drive discovery, Keychain credentials, and machine-
  local path normalization.

### Shadow research: higher-timeframe LRSI entries

- `H2` (120 min) is a derived timeframe again (BD-78) because the LRSI study is
  the consumer the locked plan's cut asked for. RTH is 6.5 h, so H2/H4 end each
  session with a stub: published as evidence, excluded from the oscillator input.
- `outcomes.HTF_LRSI_RECIPES` is a bounded 16-recipe diagnostic grid - M30/H1/H2/H4
  x {cross-up 50, cross-up 20, cross-down 50, cross-down 80}, one stop model (the
  signal bar's extreme + 0.25 ATR on the same timeframe) and one 2.0R target.
  `simulate_htf_lrsi_entry` builds the rolling multi-session series through the
  warehouse's own aggregation contract from canonical M5 - never a second bar
  source - and enters on a completed derived bar close at or after the setup
  became known.
- Long and short legs read the SAME unmirrored series (BD-79): the efficiency
  formula clamps at 0, so the mirrored-close idiom the live M5 engines use is a
  different feature, not a transform. `RESEARCH_CROSS_LEVELS` is additive; the
  live `CROSS_LEVELS` and every `m5_signal_engines` behaviour are unchanged.
- Nothing here is registered in `outcome_semantics` (BD-80) - these are warehouse
  `outcome_path` rows keyed by `recipe_id` and never acquire a bounce family.
  Shadow only: no detector, score, alert, Focus list or review queue is reachable.

### Shadow challengers

- Side-symmetric SPY market-state/pullback engine runs beside the legacy pause
  detector, emits replayable evidence, and cannot affect candidates, alerts, or rank.
- Greatness Monitor persists ordered touch/wick/close/acceptance/retest/failure/re-arm
  transitions beside legacy D1 alerts and cannot alter the champion path.
- Champion-invariance tests prove enabled, failing, or poisoned shadow engines leave
  production SPY/D1 results unchanged.

Neither challenger is promoted. Their remaining evidence gates are in `plan.md`.

## Recent changes (2026-08-26 onward)

### 2026-09-04 - Q1: `held_run_score` says what it measured

Process review findings 1 and 2. Live: 979 of 8,161 recent episodes read `held=True`
with the thirty-minute question never answered (2 registered-only, 977 never reaching 30
minutes); 8 of 2,646 "D1 present" episodes were the OPPOSITE side of the swing setup.
Now: a measurement state per episode (only `measured_held` is held; a stop first seen
past the window with no earlier row bracketing it is `break_time_unknown`, never held);
`hold_rate` = held / measured with the five counts and `coverage` on every cell and a
**Measured** column on the Daytrade Tracker; the D1 join keeps the side (aligned /
opposed / none / unknown, basis retrospective - the snapshot carries no time of day) and
only ALIGNED carries the privilege; the window is `evidence_stats.lately_window` with its
gaps on the status line. `alert_suffix` text unchanged. Owed, ask-first (`legacy.py`): a
first-break-time column and the sweep autorun default. Live gate #60.

### 2026-09-04 - Q5: the pick scorecard leaves the Qt thread

Process review, performance: `ui_stalls.jsonl` recorded 15,739 ms at 13:00:44 PT in
`_score_todays_picks`, which materialised both runtime CSVs (335 MB + 308 MB) on the
calling thread and wrote `picks_scored_at` before scoring. Now one owned daemon worker
(`autopilot-scorecard`; the wrap-up worker calls the body inline through the same guard),
streamed today-only reads (`autopilot_core.read_scorecard_inputs`), append after every
group scored, success-only `picks_scored_at`, last-good line kept on failure, three
attempts then `picks_scoring_failed_at`. Measured read-only on the live files: the old
materialise 5.66 s, the streamed pass 5.40 s (the parse dominates; the win is the thread,
not the seconds) - 7,933 candidates and 12,030 outcome rows for the day kept out of
324,605 + 100,506. Live gate #64.


### 2026-09-04 - Project process review and evidence-note corrections

Source-backed advisory review in `docs/analysis/PROJECT_PROCESS_REVIEW_2026-09-04.md`.
Corrected the historical zero-like-link claim (`match_basis`, not `basis`), the
no-target-guaranteed-loss claim, and stale built-state descriptions. Recorded
held/overlap measurement gaps, unsupported AI position language and measured
scorecard/scan costs. No runtime change, new authorization or gate completion;
the 6608-pass recorded baseline is unchanged and was not rerun.

### 2026-09-04 - Earnings-anchor bridge: the scan feeds the anchors CSV the warehouse reads

`swing_house_v1` graded 0/257 because `feature_snapshot_daily` had AVWAP bands on 234 of
27,579 rows, because `anchor_instance` had 14 rows, because `earnings_avwap_anchors.csv`
- the ONLY source `cli.anchors_from_bronze` reads - held 14 hand-imported rows while the
scan's current/previous earnings anchors lived only in two JSON caches
(`docs/SWING_SIMULATOR_INVESTIGATION_2026-09-04.md`).

- `master_avwap_lib/runner.py`: `build_earnings_anchor_bridge_candidates` and
  `bridge_earnings_anchor_caches_to_csv`, called once in `_run_master_impl` right after
  the two cache saves. One `EarningsGapAnchorCandidate` per (ticker, ISO anchor date)
  across both caches through the existing, previously uncalled `append_anchor_candidates`
  (append-only, de-duplicated on ticker + anchor_date, new rows at the END). `side` is
  watchlist membership; gap/price/volume/cap are empty or zero; `source` is
  `scanner_earnings_cache`. A failure is logged and never fails the scan.
- Shadow-only additive: nothing live reads the CSV (`run_anchor_watchlist_scan` is its
  only other reader and has no caller). `anchors_from_bronze`, `build_anchor_instances`,
  every detector, score, alert and Focus path unchanged.
- `tests/test_earnings_anchor_bridge.py` (11). Live gate #59.

### 2026-09-04 - Packet T2: a claimed like is one double-click, and it advances

Trader, after reading the T1 tree on the desk: *"pretty close. for the 'like and
claim' part of the capture tab, a double click of any of the setups there should be
sufficient. I shouldnt have to type anything below that box. and then double clicking
that box should advance the chart."*

- **A claimed like needs no why.** `CaptureRail.commit_like` refused an empty why and
  refocused the field; it now records whatever is there, empty included, and
  `_prompt_for_why` is deleted (nothing else called it). A whitespace-only why strips
  to nothing and the row carries no `note`. The placeholder reads **"why
  (optional)"**. The digit, the double-click, Enter and the button all commit at once.
  This supersedes R9.2(a)'s required why for the CLAIMED path; P9 had already
  superseded it for the quick one. The only refusal left is "no setup picked".
- **A claimed like ADVANCES the chart; a quick like still does not.**
  `AlertChartReview._on_captured` reads the row's mode through
  `ui.annotations.store.like_mode_of` (absence reads as claimed, the P9 rule) and
  fires the new **`likeAdvanceRequested`** for a claimed like or the existing
  `likeRecorded` for a quick one. The panel answers with `_advance_after_like`:
  record, status line, `_advance_review_queue()`.
- **An advance is not a retirement.** `_ignored_symbols` is untouched, so the name
  keeps alerting and keeps reaching the hourly D1 phone push; no auto-adopted Focus
  pick is dropped; the symbol's other queued alerts keep their places; nothing is
  placed. R9.2(b)'s measured harm (40 of 52 likes parking their own symbol) stays
  fixed.
- **One recorder, two callers.** `_record_like_advance` writes the `like_advance`
  review event for both handlers, so the quick and claimed paths cannot drift. The
  name is historical - `review_learning.TAKE_ACTIONS` keys on the exact string.
- **Tests:** `tests/test_t1_capture_and_board_like.py` and
  `tests/test_qt_alert_capture.py` - every test that pinned the required why or "the
  chart stays" for a claimed like was rewritten to the new rule and named in its
  docstring; nothing was deleted. Proven red on the un-fixed tree: **19 failed, 72
  passed**; green after: **91 passed**.
- **Docs:** `CLAUDE.md`/`AGENTS.md` (byte-identical), `docs/DESK_INTERNALS.md`
  (second-pass section under the T1 entry), `docs/CHART_REVIEW_WORKSPACE_PLAN.md`
  (R9.2(a) marked superseded for the claimed like), `plan.md` gate #58.

### 2026-09-04 - Packet T1: a veto with no box, a like that stays, a board click that queues nothing, the TC2000 board on M5 Focus

**Branch `claude/t1-capture-and-board`**, tester-first (48 tests committed red),
trader-authorized the same day. Trader, verbatim: *"when i double tap something in
the capture window (either veto or like+claim) i shouldnt get a pop up note
box ... the 'like' button in the visual chart review should NOT advance the char
to the next page because i still need time to enter alerts etc. not today can
continue to go to the next chart with a pop up note box."* and *"I want all shorts
and longs on the RS/RW board TC2000 to bne auto added to the M5 focus picks.
additionally when I click on ANYTHING from the RS/RW board it should not make a
queue of picks if I click on more nor should it add to the 'waiting' list. once i
look and click off, its done."*

- **T1.1 - one veto click is one veto row.** It used to be two rows and a dialog:
  the rail wrote the CODED row, the pane forwarded it as the "✕ Not today"
  button's `removeTodayRequested`, and the panel then wrote an UNCODED row through
  `verdicts.record_not_today` and opened `open_note_prompt`. The rail's veto now
  emits `vetoRetireRequested` and the panel answers with `_retire_after_veto`,
  which shares one body with the button's verb (`_retire_review_alert(...,
  write_not_today_annotation=)`) and differs only in that flag. The
  `remove_today` review event, the auto-pick drop, the parking and the advance are
  all unchanged; the button is unchanged.
- **The day-trade veto too** (lead ruling 2026-09-04, amending the packet, which
  had called it untouched): `_veto_but_day_trade` ended in the same method and so
  carried the same second row and box. It now retires through
  `_retire_after_veto`, still placing on M5 Focus FIRST, and a failed placement
  still retires.
- **T1.2 - a like never advances.** `likeAdvanceRequested` -> `likeRecorded`,
  `_advance_after_like` -> `_after_like`, and `_advance_review_queue` is not
  called. **The review event keeps the name `like_advance`** - historical, because
  `review_learning.TAKE_ACTIONS` keys on that exact string - and now means "liked;
  the symbol keeps alerting and the chart stays". Covers the claimed like, Alt+L
  and the chart's "♥ Like" button, whose optional P9 note box is untouched.
- **T1.3 - a board look queues nothing.** A `MANUAL_CHART_TAG` chart holds no place
  and, when replaced, is neither re-inserted nor given a `skip` - a look is not a
  shown alert. Five board clicks now leave the pane reading "queue clear"; they
  used to leave four names waiting. The M5-alert-bar `skip` with
  `clicked_away_from_m5_alert` and the dequeued-D1 return-to-head rule are both
  proven untouched.
- **T1.4 - the TC2000 board's parity rows auto-join M5 Focus.** On `boardChanged`
  and once at attach; empty `failed_floors` only; the one adoption gate re-run on
  the row's own numbers (a fourth call site, never a second definition);
  `_ignored_symbols` skipped; DESK only; through `store.add` +
  `mark_auto_adopted`, never `FocusService.add`; never removes; idempotent; one
  `strength_board_auto_focus` review event per refresh carrying `side_counts`,
  `adopted`, `refused` and `as_of`. That event's `symbol` is the constant
  `M5_STRENGTH_BOARD` - `record_review_event` refuses an empty symbol, the event is
  about the board rather than a name, and the underscore makes the value
  unrepresentable as a ticker under `SYMBOL_RE`.

**Fix round 1** (reviewer NO-GO, one blocker + five advisories, all addressed).

- **BLOCKER - the auto-join undid a removal.** `_ignored_symbols` only holds names
  the "Not today" verb parked, so the Focus-review walkthrough, the Focus list's
  remove button, the cross-focus toggle and the Master AVWAP unfavorite were all
  put back by the next refresh - **and the name was re-injected into `longs.txt`
  with it**. Fixed in the STORE, so every door counts without touching any of
  them: `FocusPickStore` records `(symbol, side, category, session_date)` under an
  ADDITIVE `declined` key in `focus_auto_picks.json` on `remove`,
  `remove_everywhere`, `clear` and the fade; `declined_today` answers for today
  only, `_load_declined` prunes older rows, adding the name back by hand clears
  it, and the day roll clears the declines with the markers. The auto-join skips a
  declined name and names it in `refused` as "(you took it off today)".
- **Adds are batched**, one `add_many` per side (60 names one at a time measured
  781 ms on the Qt thread); the marker stays per name.
- **The review event counts `already_auto` / `already_trader_owned`** through
  `store.is_auto_adopted` - counts only, never a re-mark.
- **Pinned and documented, not changed:** looking at a name that was WAITING takes
  it out of the waiting list and it does not come back (that IS "once i look and
  click off, its done"); and **every adopted name is injected into the shared
  `longs.txt` / `shorts.txt`**, as every Focus add always has been, so the
  auto-join grows BounceBot's intraday scan input (live that day: longs.txt 29,
  shorts.txt 50, 33 + 32 store-injected m5 entries).
- `test_the_two_verbs_share_one_body_rather_than_two_branch_ladders` proved only
  that two differently-named methods existed - true of two copied ladders too. It
  now patches `_retire_review_alert` and shows both verbs reach it with the right
  flag.

Full suite with the nightly AI lock probed FREE and nothing deselected: **6577
passed, 1 skipped, 72 subtests, exit 0** (+43). `ruff` clean, smoke 7/7, source
`--selftest` 74/74. No packaging trigger. **Live gate #58.**

### 2026-09-04 (early) - The last two projects started: forced outcome recompute (BD-98) and the tracker record store (decision 0017)

**On `main`, lead-built, trader-authorized** ("go ahead and start these last 2 projects").

- **Outcome recompute, BD-98.** `outcomes.build_outcomes(..., force=True)`
  re-simulates terminal rows (the nightly's "idempotent by knowledge" rule skips
  them, which is right until the inputs turn out to have been wrong);
  `_run_outcomes(..., bucket=, force=)` takes an explicit bucket; and
  `research_warehouse.cli recompute-outcomes [--buckets a-b] [--time-budget-minutes N]
  [--apply]` walks every bucket with force, ONE lock per bucket so the nightly
  build slots in between, recording a coverage firing per bucket
  (`run_id=outcomes_recompute-bNN`). A re-simulation that reproduces the stored
  result writes nothing; only a changed result supersedes. Pinned by
  `tests/test_warehouse_recompute_outcomes.py` (4). **Started 07:00 PT 2026-09-04**
  against the live lake with a 340-minute budget (one lock per bucket, so the
  day's post-scan builds interleave) (6,850 occurrences over 1,715
  symbols, 32 buckets); finished 07:53 PT: 32/32 buckets, 134,502 rows superseded, 3,803 unchanged, no
  errors - gate #56 met in full.
- **Tracker record store, F3 step 1, decision 0017.** `scripts/tracker_store.py`:
  SQLite (`master_avwap_setup_tracker.sqlite` beside the JSON, WAL), one row per
  tracker record with a content hash, so a save rewrites only what changed and a
  read can narrow by section / symbol / scan date. `save_setup_tracker_payload`
  mirrors the SAME payload after `save_json` (`mirror_payload`, behind the
  `tracker_storage_shadow` setting, default ON, never allowed to cost the save).
  No reader moves; the JSON stays authoritative. `python scripts/tracker_store.py
  verify|mirror|counts` measures parity. Pinned by `tests/test_tracker_store.py`
  (7: exact round trip, changed-only rewrite, narrowed reads, every difference
  named, the hook never raises and honours the setting, the scanner's save
  mirrors after the JSON write and survives a mirror failure, the CLI).
- **Tests**: +11. Full suite in the checkpoint.

### 2026-09-03 (night) - The rest of the assessment packets: S2 instrumentation, S4 cadence, F2 removal, the rebuild tool

**On `main`, lead-built, trader-authorized** ("go ahead and implement the rest").

- **S2 - the M5 cycle, instrumented, not trimmed** (`bounce_bot_lib/legacy.py`,
  the detector file, under the trader's blanket authorization). The preamble log
  line now carries one clock mark per RRS run (`rrs_scan_5m/15m/1h/gui`) and one
  per engine sweep (`engine_orb_break`, `engine_ema8_grind`, `engine_lrsi_cross`,
  `engine_confluence`, `engine_orb_first_candle`, `engine_h1_color`) instead of two
  folded stages. Every sweep still runs, in the same order, with the same guards:
  nothing detects differently. The trim waits for the first uncontended morning
  after the restart - every cycle number so far was taken under a held lock.
- **S4 - desk-day scan cadence** (`autopilot_core.get_autopilot_swing_slots(...,
  cadence=)`, `desk_scan_cadence()` reading the `desk_scan_cadence` local setting,
  default `reduced`): DESK days run four scans - open+60, open+210 (13:00 ET), the
  R3 near-close preview and the close slot that owns the tracker write - instead
  of six. AWAY and EVENING keep the hourly ladder the phone digest reads. The
  manual Setups-page scheduler follows the same setting. `hourly` restores the
  old schedule on the next day roll.
- **F2 - the Tk stack is gone**: `scripts/gui.py`, `gui_app/` (7 files),
  `market_prep_gui/` (4), `market_prep_tab.py`, `journal_tab.py`,
  `master_avwap_lib/gui.py`, `bounce_bot_lib/gui.py`, `bounce_bot_lib/alerts.py`
  (three Tk-only re-exports), `TickerMover.py` and `tests/test_market_prep_tab_helpers.py`
  - 19 files. The two legacy cores lose their `tkinter` imports and their Tk
  tails; the scanner CLI (`master_avwap.py --once` / `--loop`) moved into
  `master_avwap_lib/runner.py` without its `--gui` branch; BounceBot's
  `--use_gui` prints a notice and runs console mode. `PyQt5` left
  `requirements-gui.txt` and `constraints.txt` (the spec keeps its exclude as a
  guard). `pyproject.toml`'s lint excludes shrink by three; the packaging
  allowlist by two. `tests/test_module_globals_resolve.py` now guards the two
  runner shims and learned that a `def` inside an `if` binds a name (it was
  blind to that). Decision 0004 carries an amendment.
- **The rebuild tool** (`ResearchStore.retire_partition`, `cli rebuild-month`,
  BD-97): retire a month's `bar_derived` + `feature_snapshot_intraday` partitions
  by one RETIRE line each and recompute them session by session from the repaired
  `bar_m5`. Dry run by default. Pinned by `tests/test_warehouse_rebuild_month.py`
  (5), which reproduces the pollution (`constituent_count` 6 for 3 bars) and the
  repair.
- **Applied 22:28-22:45 PT with the trader's explicit permission** (an earlier
  attempt had been refused by the session's permission classifier): the desk
  restarted onto this code (1.7% of a core, was 101%), `dedupe --apply` dropped
  10,530,916 `bar_m5` rows, `rebuild-month --apply` recomputed August (21
  sessions, 250 files retired) and September (4 sessions, 44 files), and the
  498 MB `.corrupt` copy was deleted. Rebuilt M15 bars: 15 of 605,909 August rows
  over-count (0.002%, was every row), 99.9% COMPLETE. Outcomes for those months
  stay owed.
- **Corrected from the assessment**: `evidence_snapshots/` already has retention
  (`ops/evidence_snapshot.prune`, 7 daily / 4 weekly / 12 monthly, run by
  `snapshot_to_das.ps1`); the 5.8 GB is inside that policy. Rotation of
  `technical_integrity_events.jsonl` was DECLINED on 2026-08-17 (R6(b)) until the
  warehouse's verified ingest of it passes - that ingest now runs nightly
  (`bronze_technical_integrity_events`), so the trigger has fired and the
  segment scheme is owed as its own packet, not done here.
- **Tests**: +6 (rebuild 5, reduced cadence 1), -1 file; the removed Tk tests
  replaced by one that asserts the stack is gone. Full suite in the checkpoint.

### 2026-09-03 (evening) - The research tee burned a core; the lake was 85% duplicates; a thread gauge

**On `main`, lead-built, trader-authorized** ("go ahead and implement all packets"
on the evening desk assessment). Measured on the running desk (pid 18548, `f903ca4`)
at 21:05 PT: 101% of one core, 26,540 of the process's 29,909 CPU-seconds on
`warehouse-m5-tee`, 91% of GIL samples in `capture_m5_tee`, the GUI thread in 0 of
362. Detail and the rules in `docs/DESK_INTERNALS.md` ("The research tee burned a
core") and BD-96.

- **`research_warehouse/bar_archive.py`**: `capture_m5_tee` runs in two passes -
  identity (timestamp, forming, high-water / `seen`) first, prices + hash + session
  tag for survivors only. New `high_water=` argument (per-symbol newest
  `interval_start`, advanced in place); a symbol whose newest bar is behind its mark
  is `symbols_unchanged` and never walked. Session lookups cached per session date;
  `_market_session_module` memoized (its `Path.resolve()` ran per bar).
- **`ui/services/warehouse_service.py`**: `WarehouseTeeCapture` replaces the
  UTC-date-keyed `seen` set with the persisted mark (`tee_high_water.json` in the
  spool dir, atomic replace, 14-day per-symbol retention, unreadable = empty +
  warning). A restart resumes; a UTC midnight re-spools nothing.
- **`research_warehouse/spool.py`**: `seal_spool` drops rows whose grain key is
  already live in the target partition or already sealed in the same call, and
  counts them (`SealResult.rows_deduplicated`). Superseding datasets exempt.
- **`research_warehouse/store.py`**: `duplicate_rows` (dry run) and
  `dedupe_partition` (COMPACT-shaped rewrite, earliest `observed_at` kept, inputs
  retired, `rows_dropped` + `dedupe_grain` on the line; refuses non-compactable and
  superseding datasets). **`research_warehouse/cli.py`**: `dedupe` subcommand, dry
  run unless `--apply`, apply under the build's single-flight lock.
- **`ui/thread_cpu_gauge.py`** (new, always on from `app.main`): per-thread CPU
  time once a minute from the OS (`GetThreadTimes` / `/proc`), one record per tick
  in `diagnostics/thread_cpu.jsonl`, a WARNING naming any non-GUI thread above 50%
  of a core. CLI summary: `python scripts/ui/thread_cpu_gauge.py`.
- **Lake state at the time of writing (dry run, read-only)**: `bar_m5
  month=2026-08` 12,015,283 rows / 1,816,970 keys (10,198,313 to drop);
  `month=2026-09` 541,444 / 208,841 (332,603). `bar_d1`, `bar_derived`,
  `feature_snapshot_intraday`: 0 grain duplicates, but the derived/feature rows for
  those months were computed from the duplicated M5 rows and need a rebuild.
- **Tests** (+16): `test_warehouse_tee.py` (dedupe-first does zero hashes/session
  lookups for seen bars; high-water skips an unchanged symbol without walking it;
  unreadable stays unreadable), `test_qt_warehouse_tee.py` (UTC midnight re-spools
  nothing; the mark survives a restart; an unreadable mark file starts empty),
  `test_warehouse_spool.py` (seal drops and counts lake/in-seal duplicates;
  superseding datasets untouched), `test_warehouse_dedupe.py` (5: dry run writes
  nothing, earliest observation kept + inputs retired + idempotent, refusals, CLI
  dry-run vs apply, inert without a store), `test_ui_thread_cpu_gauge.py` (3: hot
  thread named and the GUI thread never, a real spinning thread measured from the
  OS, summary).

### 2026-09-03 - Every ticker click on the Trading Desk charts in the centre pane

**On `main`, lead-built.** Trader: *"when i click on a ticker anywhere while on
the trading desk tab, i want the chart to come up on the visual chart review chart
we have in the center of that tab. right now i click things in the auto RS/RW board
or the master avwap setups board and it does a pop up. thats fine on other tabs,
but the main tab should always be centralized with the main chart."* The rule the
M5 Strength Board got on 2026-08-31 now covers every click surface on the desk.

- **Inside the Alert Center** (`alert_center_panel.py`): the RS/RW board, the entry
  board and the Focus-strength board connect to a new `_chart_board_symbol`, which
  is `chart_symbol` with a named origin - the lookup box's door, never
  `_enqueue_review_alert`. The feed's ticker-name click (`_show_symbol_snapshot`)
  now does what a row click does (`_show_alert_detail`): the REAL alert reaches the
  pane with its trigger, not a manual chart of the same name. The popup opener
  `_show_board_symbol_snapshot` is unchanged and still behind `show_board_symbol`,
  the AWAY Recap's door.
- **The setups column** (`master_avwap_panel.py`, `rs_window_panel.py`,
  `industry_panel.py`, `watchlists_panel.py`): each panel gains `set_chart_sink(sink)`
  and its opener calls the sink - `sink(symbol, side=..., origin=...)` - before it
  would build a popup. `TradingDeskPanel.set_mode` points all four at
  `alert_center.chart_symbol` in workspace mode and at `None` in tabs mode, where
  the pane is on a different sub-tab and a chart there would be unseen. `None` is
  the constructor default, so every existing standalone-panel test is untouched.
  The setups table's Space / Prev / Next walk goes through the same opener, so on
  the desk it steps the CENTRE chart one row at a time.
- **Tests**: `tests/test_qt_desk_ticker_clicks_chart_center.py` (14) pins each
  surface charting in the pane with no popup, the manual-chart tag and origin, the
  tabs-mode fallback and the way back, the AWAY door still popping, and a sinkless
  panel still popping. `test_qt_alert_center.py`'s board-click test was asserting
  the popup and now asserts the pane.
- **Not changed**: the snapshot popup itself, `chart_symbol`, `_enqueue_review_alert`,
  movers-only, the arm bar, any store. No packaging trigger.

### 2026-09-03 - Packet F1: the desk freeze, and what was actually holding the GIL

**Branch `claude/f1-desk-freeze`.** Authorized by the trader at ~09:00 PT (*"the
program has been freezing and has been basically unusable all morning ... fix
it"*) after the lead measured the running desk (pid 11612, old `main` tip
`93732ef`). Three items, each with a test PROVEN to fail on the un-fixed file.
The measurements and the reasoning are in `docs/DESK_INTERNALS.md`; the warehouse
decision is BD-95.

**What was measured.** `uvx py-spy record --gil`: the `qt-warehouse-build` thread
held the GIL in **82.7%** of samples, `MainThread` got **2.3%**, and WM_NULL pings
to the desk window from outside the process hung **100-606 ms** every few seconds.
**84%** of that thread was inside `research_warehouse/exchange_calendar.py`,
recomputing holidays once per M5 bar per occurrence. `manifest_log.jsonl`: the
outcomes stage ran **27-57 minutes after every scan**, four scans a day, all in
RTH. `ui_stalls.jsonl` stopped at **06:03:35** with its per-session cap spent
overnight (1,614 records between midnight and 06:03), so the morning in question
has no stall evidence at all.

**1 - the calendar is memoized.** `holidays`, `half_days` and the session builder
are `lru_cache`d; 20,000 `session_for` calls went 0.25 s -> 0.0114 s (21x). The
cache is behind `trading_session` positionally, because `lru_cache` keys on the
call shape and every in-module caller passes `calendar=` as a keyword.

**2 - the build left the process.** `start_warehouse_build` spawns
`research_warehouse.cli build` at BELOW_NORMAL priority (`getattr`-read flags, so
macOS still launches), `launch_gui` answers `--warehouse-build <run_id>` beside
`--run-scan`, and `_run_warehouse_build` is deleted. A CPU-bound Python thread
holds the GIL by construction - no priority or timer trick returns the GUI thread,
which is why nothing smaller was attempted. The child is owned and reaped but
deliberately NOT counted as a scan child, because that count decides whether the
next scan may start. The three tests in `test_qt_warehouse_tee.py` that pinned
the in-process mechanism ask the same four questions of the child.

**3 - the stall watchdog's cap rolls hourly.** `MAX_RECORDS_PER_HOUR = 2000` with
a counter beside the session total; a runaway loop is still bounded at 48k a day,
and a quiet night can no longer spend the trading morning's budget.

No packaging trigger: no new dependency, no new asset, and `research_warehouse` is
already a collected package. **Live gate #53.**

### 2026-09-03 - Round R4 Part B: the surfaces the packets promised

**Branch `claude/v3-keep-it-honest`, `main` (with Part A) merged into it first.**
Eight items (B1-B8), each with a test PROVEN to fail on the un-fixed file. Two are
the lead's additions (B7, B8).

**B1 - the newest fact pack is the LAST one written.** Both Weekend Prep readers
used `sorted(root.rglob("*.json"))[-1]`, an ASCII sort in which `.` (0x2E) is
below `1` (0x31), so `2026-09-01.1.json` sorts BEFORE `2026-09-01.json` and the
last name is the pack every re-run superseded. Live that day: the reader took the
47-cell original, whose older shape carries no `eligible_policies` list, and the
verdict card printed "no cell has cleared the evidence floor yet" while the `.2`
pack had 33 that had. `setup_research.latest_pack_path` / `pack_sort_key` undo
exactly what `_superseding` did, in the module that owns the naming, with the
ordinal parsed as an INTEGER (a tenth re-run sorts after a ninth) and the session
stem first (a re-run of yesterday never outranks today).
`weekend_verdict.research_line` additionally falls back to `policies` filtered on
`stats.eligible` when `eligible_policies` is absent - on the live pack those two
lists are the same 33 of 73 cells, so the fallback is exact rather than an
estimate.

**B2 - a setup doc's record is ONE horizon, ONE read, and it reaches the screen.**
`_read_family_outcomes` pooled every horizon; the tracker grades each scan row at
1, 3, 5 and 10 sessions, so one decision was counted up to four times
(`avwap_band_bounce`: n=1797 pooled, n=329 at the horizon). The rate barely moves;
the Wilson LOWER BOUND does, in the flattering direction, unevenly across
families, which changes the order. The horizon value moved to
`evidence_stats.SWING_HORIZON_SESSIONS` (5) and
`autopilot_core.SWING_DIGEST_HORIZON_SESSIONS` re-exports it, so the docs and the
AWAY digest rank on one number - the top three families by bound read
0.585 / 0.543 / 0.522 on both. `stale_horizon == True` rows are dropped, the rule
the digest and the scan-factor leaderboard already apply. The CSV is read ONCE
into `{family: rows}`, memoised on (path, mtime, window, horizon): 24 full passes
became one, measured 0.16 s. And nothing had called `family_record_sentence` at
all - `SetupDocsPanel` now builds every sentence on a `_RecordWorker` QThread and
renders each click from the cache, so the selection handler opens no file.

**B3 - `swing_headline` has production callers, and there is ONE Wilson.** PARTIAL
and the docs say so. Wired: the Master AVWAP setups table's appended
**Family Win %** column (records built on `_FamilyRecordWorker`; the header says
FAMILY because it is a statistic about the family, not about that symbol today),
the Setup Tracker's **Last 30 Days** tab, and all four Weekend Prep cohort tables
(veto, like, pass, rejection) - every one of them now SORTING by the Wilson lower
bound. The veto/like view had been ordered by the trimmed mean; the pass and
rejection tables had no order at all. `swing_headline.headline_from_rate` is what
made three of those possible: those stores keep a rate and a count, and
`round(rate * n)` recovers the integer pair exactly, because each file writes the
rate as `wins / n`. STILL OWED: the Setup Tracker's **Setup Types** tab, for a
measured reason - `master_avwap_setup_type_stats.csv` has no win column at all
(`target_hit_rate` and `stop_rate` are different questions) and the outcomes file
cannot be joined at that table's grain, its 184 rows collapsing to 71
(side, bucket, family, zone) groups so one rate would repeat across up to six rows
and read as each row's own. **ONE z**: `swing_headline.WILSON_Z` (1.96).
`expected_r`'s 1.28 is a parameter of the proven-quality score in a fenced scoring
file and no trader-facing surface may reach for it - a test asserts it.

**B4 - the Daytrade Tracker says which number is which.** The champion tier is a
COLUMN at last (the header comment had promised it since V3): PROVEN / MUTED /
active joined from the bounce learning state on the same key the headline uses,
BLANK for a segment it never saw - live 4 / 2 / 185 / 104 of 295 rows. The
aggregator's verdict is headed **"Verdict (edge score)"**, because it is computed
from average R and sat unlabelled three columns from a headline computed from
something else. The **My Decisions** tabs carry Held 30m and Held x Ran through
the same `apply_held_and_ran`; those rows name no side, so `held_run_score`
gained `ALL_DIRECTIONS` - a pooled cell accumulated FROM THE EPISODES in the same
loop, never an average of the long and short cells, which would be a mean of
trimmed means and a second formula in that file again.

**B5 - a day-trade pass writes the screen it came from.** `commit_pass` needed the
sidecar writer, so it called `record_pass_annotation` directly and skipped
`_record` - and with it the two lines that stamp `surface` and the scan context.
Every pass landed with neither while the veto, the like and the note beside it
carried both. `_record` gained one keyword, `writer`, which is the only thing that
path actually needed to differ on. The V3 item-4 test that guarded this asserted
on the SOURCE TEXT of `_record`, which is true of a method a verb never calls;
it is replaced by five behavioural tests, one per real click handler, each reading
the written row off disk.

**B6 - one "lately", counted in sessions, and a flat is neither.**
`review_learning.DEFAULT_WINDOW_DAYS = 90` was a calendar-day literal on the very
window CLAUDE.md names as reading `LATELY_SESSIONS` (the blind-spot and leak
callouts are cut on it); it is now `DEFAULT_WINDOW_SESSIONS = LATELY_SESSIONS`
with the cutoff walking the exchange calendar. Weekend Prep's `window_days=7`
became `evidence_stats.WEEK_SESSIONS` (5) - it printed "Week of \<Mon\> to \<Fri\>"
over the last 7 calendar days, so a holiday week measured four sessions and still
called itself a week. The state key, report header, CLI flag, System Health audit
and Daytrade Tracker status line all say **sessions**, and a literal-scan test
fails if a `window_days` comes back. Separately,
`swing_headline.headline_from_outcomes` counted `close_r == 0.0` as a LOSS; a
scratch now counts in a third bucket, out of `n` (it has no answer to the win/loss
question) and in `avg_r` (it is a measured outcome).

**B7 (test-only) - the journal panel fixture cannot expire again.** Six tests went
red at midnight with no commit near them: the header opens on the `30d` preset and
the fixture's AAPL round trip was pinned to 2026-08-03, one day outside it. The
dates anchor on the Monday two weeks back, and a guard test asserts them against
`journal_feed.date_range_bounds("30d")` rather than a re-spelled 30.

**B8 (docs) - four tabs fill, five blank.** plan.md's Phase 0.14 table and the
CHANGELOG's Part A entry still carried the retired "six of nine" claim that fix
round 1 superseded.

### 2026-09-02 - Round R4 Part A: fix what review round 3 found

**Branch `claude/r4-fixes`, off `main`.** Eighteen items (A1-A18) across P10, V1
and V2, each with a test PROVEN to fail on the un-fixed file. Nothing here is a
new feature: every item is a claim the docs made that the code did not keep.

**P10 (A1-A6).** `after_like_block` read `summary["eligible"]` off
`evidence_stats.summarize`, which never sets that key, so every cell of the
after-like grid reported ineligible however large - a 60-episode, 60-symbol,
28-session cell showed nothing. One helper, `_meets_eligibility_floors`, now owns
the rule for both blocks. `trial_ledger.backfill` ran BELOW `_run_outcomes` in
`cli.run_build`, so the row declaring the grid was appended one step AFTER the
outcomes it governs; moved above, asserted on recorded call order.
`simulate_after_like_rows` shares one `series_cache` across twenty cells that look
at different windows, and `_entry_from_derived` keyed it without the window - so
an offset>=1 cell was served offset 0's longer derived series and whether a cell
was MEASURABLE depended on which sibling ran first (13 M30 bars alone, 39 after a
sibling, against a 21-bar EMA floor). `like_links.link_rows_for_bronze` had no
production caller while the ERD, this file and gate 42 all said
`bronze_like_occurrence_link` is written nightly; `_run_after_like_pass` publishes
it now, month-keyed, skipping by record hash what the partition already holds.
`SURFACE_FOCUS_PANEL` and `SURFACE_M5_ALERT_BAR` were constants with no writer and
now have one each; the two chart-review hosts call the `surface` override that had
existed since P10 B1 with no caller. And both note boxes save on **Enter**
(Shift+Enter newlines) through `ui/widgets/note_prompt.py` - the plain-text mode
that makes them multi-line also handed Return to the editor, so the only way to
save was the mouse.

**V1 (A7-A12).** The Strength Board's relative volume is SESSION-RELATIVE, which
is what decision 0016 answer 9 asks for; the positional stride V1 shipped reads
1.2949 on a series whose honest answer is exactly 1.0000. The D1 SMA floors drop
today's forming bar and read `2y` rather than `1y`. `autopilot_core._frame_rows`
passes a missing volume through as None instead of a measured 0.0.
`d1_setup_present` had NO caller anywhere - 346 live segments read False - and is
fed from the scanner's own 19 MB snapshot, never the 1.1 GB tracker. The Daytrade
Tracker's second held/ran formula is deleted and the module's own answer joined in
its place; the M5 alert row gained the suffix `segment_index` was built for. The
AWAY digest ranks ACROSS the buckets by the tracker's realized win rate (Wilson
lower bound, expected R as tiebreak) with the near cap applied AFTER the ranking.

**V2 (A13-A18).** The verdict card's take rate read `takes + skips + rejects` and
the state publishes neither of the last two: "100% of 94 shown" where the truth
was 30% of 318. `week_trades` moved off the Qt thread (775 ms of the one Refresh
click) and `DiscoveryPage` gained the `reload` it never had, losing its six
per-table buttons. "Confirm all shown" no longer confirms a blank - which the
nightly tagger re-flagged every night forever - and the page gained the per-row
"Edit tag...". The Market Journal left-nav page is one box, one Enter and a dated
newest-first list. `session_date_for` rolls at the OPEN rather than at midnight
in New York, and `written_after_the_session` is measured against the CLOSE. Every
Weekend Prep table carries a ten-row floor through one constant.

**Deviations from the packet, reported rather than forced.** `capture_rail`'s
`commit_veto` already stamped `surface` and `scan_context` - V3 item 4 closed that
seam before this packet was written. The `alert_center_panel` note dialog was
already asynchronous; what was wrong in both comments was "MODELESS"
(`QDialog.open()` is window-modal) and "DEFERRED" (the call is the handler's last
statement). **FOUR of the Daytrade Tracker's nine tabs fill Held and Held x Ran -
Bounce Types, Combos, Time of Day, Environment - and five read BLANK** (corrected
by fix round 1, which made `held_run_score` spell its segments the aggregator's
way; before that join was an equality, six read blank). The five split two ways in
`UNDERIVED_DIMENSIONS`: the four `master_avwap_*` Swing tabs are dimensions
`intraday_bounce_outcomes.csv` cannot be asked for at all, and `rrs_alignment` is
reachable and simply not derived yet. That is the honest consequence of deleting
the second formula, not a regression.

### 2026-09-02 - Phase 0.14 packet V3: keep it honest

**Branch `claude/v3-keep-it-honest`, off `main` - MERGED to `main` the same evening
(fast-forward, trader-directed).** Live gate #50 owed. P10, V1 and V2 were merged first.

Six items, all six built, and the shape of all of them is the same: a number the
trader reads has to mean one thing on every screen, and it has to say what it
rests on.

**Win rate leads swings, MFE-after-a-held-level leads day trades**, one
implementation each, with `n`, a Wilson lower bound and a floor flag; mean R and
the tier statistics stay beside them and are never replaced. **"Lately" is one
constant** counted in trading sessions. **The rail's veto seam is closed** - it
wrote without a `surface` while its like path wrote with one. **The Research tab
says it is the builder's surface**, and the one number the trader needs from it
now has a line on Weekend Prep.

**Measured against the packet:** it asks for exactly five annotation entry points.
Three are wired, because those are the screens that carry a like or dislike
gesture; the Focus panel's "Not today" IS the chart-review one and the M5 bar's
click-away is deliberately a review event. The test records that rather than
inventing a gesture so a count comes out at five.

**Verification.** `pytest tests/ -q` **6310 passed, 72 subtests, process exit 0,
zero failures**, lock probed FREE immediately before the run - `ruff` clean -
smoke **7/7** - source `--selftest` **74/74**.

### 2026-09-02 - Phase 0.14 packet V2: the loop closes (items 1, 4 and 5)

**Branch `claude/v2-loop-closes`, off `main`.** Live gates #46-#48 owed. V1 was
merged to `main` first, as the packet required.

**Built:** the nightly auto-tagger and its review badge; the Market Journal
capture as one box and one Enter, dated to the session it is about; and the
default-off switch that hides the four surfaces the trader never opens without
removing any of them.

**Also built (second run): item 2's (a), (b), (c) and (e)** - one Refresh for the
whole tab, the verdict card, the retired RS/RW prose, and the "Tag this week"
step that completes the tagging loop item 1 started.

**Not built: item 3, and part of 2's (c).** The AWAY Recap is still the
forward-looking digest with no outcomes and no charts; Weekend Prep still shows
its takes and watch conversion as text rather than a table. plan.md's Phase 0.14
entry records exactly what each owes.

**One defect of my own, found and fixed here.** Item 1's badge started its reader
in `__init__`; that thread opened the journal while another test was still
monkeypatching the journal's module globals, and it made an unrelated journal
test fail from a hundred tests away - green alone, red in the suite. It starts
from `showEvent` now and is joined in `closeEvent`.

**Verification.** `pytest tests/ -q` **6222 passed, 72 subtests, process exit 0,
zero failures**, lock probed FREE immediately before the run · `ruff` clean ·
smoke **7/7** · source `--selftest` **74/74**.

### 2026-09-02 - Phase 0.14 packet V1: names first (items 1 and 2)

**Branch `claude/v1-names-first`, off `main`.** Live gates #44-#45 owed. Decision
0016 landed on `main` first, as the packet required.

**Item 1 is complete.** The board runs the trader's own scan now: their relative
volume, their floors, their universe, and their filters as a DISPLAY filter - rows
that miss are greyed and say why, behind a default-on parity toggle. The RS/RW
board moved in above it: one window, two sections. The golden's expected values
come from a second hand implementation, because a golden generated by the code it
checks pins that code's mistakes.

**Item 2's score is complete and none of its three surfaces are.**
`held_run_score` is built, tested and shadow; the Daytrade Tracker column, the M5
alert-bar suffix and the ordering switch are not wired.

**Items 3 (the phone digest's near-bucket picks) and 4 (the "Working lately"
section and the priority switch) are NOT BUILT.** plan.md's Phase 0.14 entry
records exactly what each still owes.

**Verification.** `pytest tests/ -q` **6174 passed, 72 subtests, process exit 0,
zero failures**, lock probed FREE immediately before the run · `ruff` clean ·
smoke **7/7** · source `--selftest` **74/74**.
### 2026-09-02 - Phase 0.13 packet P10: what happens after I like it

**Branch `claude/p10-after-the-like`, off `main`.** Live gates #41-#43 owed.

Trader: *"the veto and like+claim tabs are just quicker ways to make a note for a
stock ... sometimes I may not want to write a note but the fact I clicked like
should be processed by the bot eventually"*, and *"anytime I like a D1 it should
be treated with respect ... if I like a stock one day it may not be for 3-5 days
later that the best entry is."*

**What was true before, measured.** Three writers, one of them graded. The Master
AVWAP star and X wrote a review event and reached NO graded cohort - so the most
considered judgement the trader makes all day left no forward record, while the
same opinion two panels away did. "Not today" wrote a `pick_feedback` verdict
whose reason is the hardcoded string `"not today"`. Only the rail's like wrote a
`trader_annotations` row.

**Part A** gives every screen one writer, a `surface` column, an optional note as
a superseding row, and `veto_uncoded` for a dismissal with no code. Plus a
deterministic `note_vocabulary_audit` slot that lists recurring uncoded words and
coins nothing.

**Part B** stamps the scanner row onto the click, links each like to a warehouse
occurrence with a stated basis, and joins an occurrence to the feature snapshot of
its own session - point in time, refusing a later revision as firmly as a later
day.

**Part C** registers `after_like_entry_grid_v1` BEFORE any outcome exists and
simulates it on P8's machinery, with the offset restricting where the entry
selector may look rather than what the simulator sees. Parity with P8's control is
pinned field-for-field. The readout is a pack block, a Weekend Prep table of
ELIGIBLE cells only, and the eligible cells in R3's narration view.

**Three differences from the packet, each measured rather than assumed:** the
bronze namespace rather than a "new frozen schema" (the slice datasets are
frozen); `setup_context_fields` does NOT already collect B1's six fields; and the
unlinked bucket is a COUNT rather than graded cells, because the declared stop
needs the occurrence's anchor.

**One defect found and fixed while building.** The first note dialogs were
`QInputDialog.getMultiLineText`, which runs a nested event loop - every existing
test that clicks a star or a "Not today" HUNG rather than failed. They are
modeless now, which is also what A2 asks for.

**Verification.** `pytest tests/ -q` **6206 passed, 72 subtests, process exit 0,
zero failures**, with the `ai_jobs_runner` lock probed FREE immediately before the
run · `ruff` clean · smoke **7/7** · source `--selftest` **74/74**. No packaging
trigger: every new module sits inside an already-collected package.

### 2026-09-02 - The merge, and the test run that started a real scan

`main` takes P9 (quick like) and R3 (the narration budget). Three documentation
conflicts, each resolved by keeping BOTH entries.

**The merged tree printed 6,145 passed and then killed its own process** -
`QThread: Destroyed while thread '' is still running`, exit `0xC0000409`. Each branch
alone had exited 0, so it read as an interaction between them. It was not.

Five test files build a real `MainWindow` with a live `AutopilotService`, and nothing
shuts them down. A later `processEvents()` let a surviving timer tick; `_maybe_auto_arm`
saw it was after 07:00 on a weekday, flipped Auto Pilot ON, and `_maybe_run_swing_slot`
**started a real master scan** - a child process against the live tape, on the machine
running the desk. A 20-minute scan outlives a 6-minute suite, so its thread was still
running at teardown.

**It depends on the wall clock.** Every clean run this week was between 04:00 and 05:00,
before the arm hour; the first run after lunch crashed, and so did every run after it,
including code that had passed at breakfast.

The guard is a machine-local setting, not a patched method: `conftest` writes
`qt_autopilot_auto_arm: false` into the temp LOCALAPPDATA it already isolates. Stubbing
`_maybe_auto_arm` would have deleted the behaviour from the tests that check it; a
setting only moves the default, and a test that wants arming turns it back on. Both new
tests fail without the guard.

Desk construction is still **not** inert under pytest - the timers still run. This
closes the one door that leads out of the process.

**Verification.** `pytest tests/ -q` **6147 passed, 72 subtests, process exit 0** ·
`ruff` clean · smoke **7/7** · `--selftest` **74/74**.

### 2026-09-02 - Review round R3: the research narration outgrew the model

**Branch `claude/r3-narration-budget`, off `main`.** Live gate #40 owed.

On 2026-09-01 `setup_research` ran three times - 03:55, 04:30, 05:00 - published
three superseding packs, spent 29 minutes reading the lake and produced **no
narration at all**. Every attempt logged the same line: *"the local server truncated
the prompt: sent ~176827 tokens (442068 chars), server reported seeing 32771"*. Two
independent faults behind one symptom.

**The package sent the whole pack.** P3 added the ineligible block, the excluded
families and the coverage detail; P8's grid grew it again. `narration_view` sends what
a person reads first instead - gate, coverage, evidence shape, excluded families,
**every eligible cell**, and **counts** of what was dropped, so the model can say "and
71 thin cells were not shown" rather than being handed 71 thin cells. The cells are
deduplicated too: four prose constants are interpolated into every one of them, ~900
identical chars inside each 1,900-char cell, so they are stated **once** under
`conventions`. **A constant two cells disagree on is never hoisted** - it stays inline
on all of them, because stating it once would silently restate one. **437,125 ->
38,184 chars**; headroom went from six more cells to about forty. Over budget now
**raises before any provider call**: a sheared prompt is not a shorter answer, it is
an untrustworthy one. The hash is over what was actually **sent**.

**A missing narration is not a failed job.** `degraded_no_narrative` under
`max_attempts=3` re-ran a **ten-minute lake pass** twice more to fail identically -
that is where the three packs and the 29 minutes came from. It returns `ok` with
`narration absent: <reason>`. If a narration retry is ever wanted it must read the
pack on disk; **it must never re-enter the lake.**

**Provenance.** Two packs from one night disagreed by 3,067 outcome rows - 9,372 on
the pre-merge checkout, 12,439 on `main` after P8 landed - and neither said why.
`built_by_commit` (once per process, fails **open**) and `recipe_ids` now travel with
the pack and into the view.

**And the synthesis counter was reading a LIST as a COUNT** (item 4, committed
separately). `matured_horizons` is a comma-joined field like `"20,60"`; `_matured`
compared it as a number, so a date graded at horizon 20 alone read as `"20" > 0` -
true - while `"0,60"` read as truthy too. It now asks whether ANY listed horizon is
non-zero. The live counter measures **4** graded dates, matching a hand count
(2026-08-20, 08-21, 08-27, 08-31). **The prompt expected 5; the code was right and
the expectation was stale** - and the count can legitimately FALL as evidence
accrues, which is now pinned by a property test.

**Verification.** `pytest tests/ -q` **6119 passed, 72 subtests, process exit 0, zero
failures**, with the `ai_jobs_runner` lock FREE and re-checked immediately before the
run · `ruff` clean · smoke **7/7** · source `--selftest` **74/74**. Fail-before-fix:
**all 11** new narration tests fail against the un-fixed tree, and the two synthesis
tests fail against theirs. No packaging trigger.
### 2026-09-02 - Phase 0.13 packet P9: quick like

**Branch `claude/p9-quick-like`, off `main` at `13cbc50`.** Live gate #39 owed.

Trader: *"anytime I like and claim a setup or like a day trade setup I just want to
let the bot and the future AI know 'something about this was good' and then we can
figure out what about it / what's the best entry later."*

**Alt+L** - unbound everywhere else in `scripts/ui`, and two live bindings for one
sequence fire NEITHER, so a clash would have cost both verbs silently. **A BUTTON
too**, on the chart's verb row and on the rail (trader follow-up the same day),
opening a box for an OPTIONAL note through the same `QInputDialog` the setup
tracker's dislike detail uses. The KEY never prompts and the BUTTON always does;
cancel records nothing. It writes
`like_claim` with `like_mode: "quick"`, no claim and no why, **superseding
R9.2(a)'s why-required for that path only**. Everything a claimed like does to the
review it does too - retire, `like_advance`, reviewed-today - and NONE of that
needed code, because all three are keyed on the event type. Everything a like has
never done it still does not: no Focus, no park, no watch, no alert.

**The schema stays at 1 and that is proven**, not asserted: a test hands the
loader, the like cohort, the auto-tagger's capture lane and the pass cohort a row
carrying `like_mode` and each returns its normal answer. Absence reads as
`claimed`, because a claim was REQUIRED until this packet.

**On an M5 chart it saves the bars**, through the writer Pass already uses -
generalised in name only, so `m5_bars_ref` and the sidecar directory are unchanged
and no reader forks.

**And the intraday grade became reachable.** `pass_cohort` returned blank on every
live pass with `sidecar_ends_before_the_entry_bar` - the sidecar holds what the
desk had AT the click, so the first close AFTER it was never inside. The new
`sidecar_completion` slot appends the rest of the session after the close, into a
NEW file and a NEW field, leaving the original snapshot byte-identical. **Gate
34's open definition question is answered without changing the definition.**

A quick like contributes a LINK to the auto-tagger, never a tag (R2's rule, since
it names no setup); `like_mode` is a picks column so a later rollup can split the
two without rewriting a row; Weekend Prep and the AI's judgement scope both say
the unclaimed cohort is not a setup's edge.

**Verification.** `pytest tests/ -q` **6122 passed, 72 subtests, exit 0, zero
failures**, with the `ai_jobs_runner` lock FREE · `ruff` clean · smoke **7/7** ·
source `--selftest` **74/74** · spec-drift **17**. Fail-before-fix: 17 of the 18
new tests fail against `main`.

### 2026-09-02 - Review round R2: two guards, then the stale sentences

**1. An empty `assigned_tier` cell was about to become a tier called NAN**, and this
landed ahead of the 07:30 scan. The live feature-history file has no such column; the
first scan after P4 widens it, and `pd.read_csv` reads every older row's empty cell
back as a float NaN - TRUTHY, and `str(nan)` is `"nan"`. Reproduced on `main`: a NAN
tier reaches the outcome rows, and on the packet's measurement 40 of 42 of them.
`tier_for_tracker_row` now accepts only the vocabulary the stamper writes (S, A, B).
Both row shapes are tested, because "key absent" and "key present and empty" are
different values and only the second one broke.

**2. A link is not a tag at any seam.** R1 covered `auto_tag_summary`; the bulk lane,
the bulk `max(confidence)` pick, Accept-all and `tag_confidence` each let one through.
A link arrives at 0.90-0.95, so it beat every scanner match beneath it - TRV lost
`avwap_retest_followthrough` at 0.91. ONE predicate now answers it, accepting both the
in-memory flag and the `link:` prefix that survives the store. Links still render with
their event id. A pass carries ALL its codes in vocabulary order.

**3. The sweep**: seven copies of the corrected "never pooled" sentence, four stale
`focus__not_today` spellings, a dead double assignment, an unlocked in-place CSV
rewrite made atomic and made to REPORT its failures, a globally-capped adjustment query
made trade-scoped, a 169 ms journal read moved off the Qt thread, four DESK_INTERNALS
entries, and three wrong claims (the frozen selftest is not 29/29,
`claude/gui-phase-0-9` IS contained in `main`, and a pass merges through
`_merge_cohort_safely` too).

**Verification.** `pytest tests/ -q` **6104 passed, 72 subtests, exit 0, zero failures** with the
`ai_jobs_runner` lock free · `ruff` clean · smoke **7/7** · source `--selftest`
**74/74**. No packaging trigger.

### 2026-09-02 - Review round R1: the blockers, then the whole of Phase 0.13 onto `main`

**Eleven blockers across five packets, each reproduced before it was fixed, then eight
merges.** With Phase 0.12, P3 and P7 already in, every Phase 0.13 packet is now on
`main`.

**P4 - three values that were computed and then thrown away.** The stale-horizon
coverage line was built onto every leaderboard row and dropped by
`pd.DataFrame(rows, columns=...)`. The tier that actually shipped was stamped where the
grader could never read it, so `tier_for_tracker_row` fell through to the bucket
derivation on every row forever - and `tier_source` was dropped by one column list and
left as a dead local in the other. And the attribute leaderboard looked its baseline up
by POSITION in a key that `extra_group_fields` PREPENDS to, so every edge in the
by-family and by-regime views shipped blank. A blank edge reads as "no edge".

**P5 - the cohort name had to be right the first time.** Rows are never rewritten and
`rejection_cohort_source` dropped the category, so both verdicts filed under the verdict
alone: `not_today` is recorded on intraday picks (223 rows) and `dislike` on swing names
(34), and a cohort called "not_today" claims a record it does not have. Fixed BEFORE the
slot's first nightly run; nothing had been graded, and nothing was rewritten. Also: the
"never pooled" note sat above a pooled base row, which is now LABELLED rather than
hidden; and the capture-time PASS merge had no test at all.

**P6 - a link is not a tag, and a blank R is not an R.** A chart housekeeping click was
minting `took:<action>` on 676 of 730 review rows and, ranked first, spending a slot of
a four-slot Tags column on it - EYPT and SMPL lost `avwape_to_1stdev` to one. The
coverage note was computed and never rendered, and its arithmetic summed BUCKETS of a
non-exclusive group, measuring 24 tagged trades of 156 as 40% and suppressing itself.
And `journal_r` read a key that exists nowhere in `scripts/`, with a fixture that
invented it.

**P7 - a declaration with no date.** Every ledger row now carries `registered_at`,
stamped by the ledger and not by the caller, with backfilled rows carrying the date
their work was authorized.

**P8 - a gate nothing could satisfy.** Gate 37 asked for a ledger row and nothing in
production wrote one; `cli.run_build` registers them now, beside the coverage line.
`assert ... or True` became a real assertion. And BD-88's claim that the derived series
were memoised was false - 2.06 s per occurrence, ~0.8 s of it rebuilding - so the cache
is real now and the entry is corrected rather than quietly made true.

**The merges.** Ledger conflicts kept both sides throughout. The code conflicts were all
ADDITIVE and were resolved by hand: P1's like-cohort half beside P5's pass-cohort half
in `capture_rail`; P2's, P5's and P6's tables in `weekend_prep_panel`; P5's two cohort
slots and P6's preference slot in `runner.py`, in the order the data requires; both
journal migrations, in order; and P6a as the ONE owner of the tag-lane helpers, with
P6's temporary copy dropped.

**Two premises stopped being true when branches met, which is the point of merging them
together.** P5 asserted the default pick key collapses a multi-code pass into one row -
P1 had widened that key for a different reason and it no longer does. And
`test_warehouse_restore` pins the build's step list, which P8's wiring extended.

**Verification.** 6053 passed · `ruff` clean · smoke 7/7 · source `--selftest` 74/74 ·
no exe rebuild required (no packaging trigger; P7's was already rebuilt). The 32
`ai_jobs` tests that stand down under the writer lock are **not** counted as a baseline:
the nightly run held it from 22:00 through this round.

### 2026-09-02 - Phase 0.13 packet P8: the first setup-parameter grid

**Branch `claude/p8-param-grid`, off `main` AFTER the morning's integration** - the
packet declared Phase 0.12, P3 and P7 as preconditions and refused to be built without
them. Live gate #37 owed. Shadow only.

**One setup, one stop, four entry moments.** `AVWAPE_TO_FIRST_DEV` LONG (the
registry's `avwape_to_first_dev@1`), 840 occurrences over 622 clusters. Twelve cells:
`m5_first_close` (control), `m15_acceptance_close`, `m5_retest_trigger`,
`m30_ema15_21_pullback`, each at 1R / 2R / 3R, with the stop fixed at
`current_anchor:1` and the exit machine, time stop and checkpoints identical
throughout. **A grid that also varied the stop could not answer the question it
declared**, because a winning cell might have won on the stop.

**The control is the code it challenges.** `m5_first_close` delegates to the existing
`simulate_m5_close_opportunity` with the existing rank-1 selector, so its rows
reproduce the `m5close_current_anchor1_*` rows by construction; the three challengers
call the same function through one new optional `entry_selector`. **The golden fixture
was pinned from `outcomes.py` as `main` had it** - imported through `git show` into a
temp package - so it pins code that had never heard of P8.

**Each confirmation entry is defined by what it refuses**: a completed M15 CLOSE
beyond the trigger (not a wick), an M5 bar that TAGS the level and still closes
holding it, an M30 bar with the EMAs in trend order whose extreme reaches the band and
whose close is still beyond it. All read the warehouse's own derived bars, stubs
excluded, eligible only STRICTLY after the trigger - a derived bar ending at the
trigger instant is the signal bar. Unmeasurable produces NO row.

**The trial-ledger row was written before any outcome was inspected**, with status
`collecting` (new: the declared 20-session window's clock is running) and with the
failure mode named in advance - **a waiting entry can look better purely because it
SKIPS the episodes that went straight down**, so the control's rows-per-cluster is the
denominator to read first.

**Measured and reported:** the packet's "22 of 61 like claims" is right for the setup
but splits **11 LONG / 11 SHORT**, and `avwap_breakout` LONG carries 15. This is the
most-claimed SETUP, not the most-claimed long - still the right first grid, since it
has the deepest evidence, but not for that reason.

Recorded as **BD-88** and **BD-89**. **Verification.** `pytest tests/ -q` **5800
passed**, with 32 `ai_jobs` tests standing down while the desk's nightly held the
machine-local writer lock (the same tests fail on a pristine `main` - checked) ·
`ruff` clean · smoke **7/7** · source `--selftest` **74/74**. Fail-before-fix: 17 of
the 18 new tests fail with `scripts/` and the fixture stashed.

### 2026-09-02 - Three branches onto `main`, so packet P8 has ground to stand on

**Trader instruction: "yes do option 1".** P8 declared its own precondition - "Requires
P3 and P7 landed" - and neither was, so P8 was not built; this merge is what unblocks
it. Merged oldest first: **Phase 0.12 A+B** (clean), **Phase 0.13 P3** (two ledger
conflicts), **Phase 0.13 P7** (six, all additive). Every conflict was resolved by
keeping BOTH sides' dated entries, and the Active state block was rewritten once at the
end from what is actually true.

Done in a SCRATCH WORKTREE rather than the live checkout: the desk was mid-run on the
nightly AI job, and a working tree carrying conflict markers inside `.py` files is the
one state a running process must never see.

**THE BD COLLISION IS RESOLVED.** Three branches off one commit each numbered their own
decisions - 78-80, 80-84, 85-86. Phase 0.12 kept BD-80, P3 shifted to **81-85**, P7 to
**86-87**; headings are now 77..87 with no repeats, asserted. Renumbered by targeted
replacement, never by line range, because `(BD-80)` appears in both lines and only the
surrounding words say which is which.

**P7's owed swap is paid**: `setup_research.family_role` no longer carries its own
two-entry role map and reads the registry instead. Output unchanged (`fact_pack_role`
translates Appendix C's `TRADE_SETUP` back to the pack's `TRADE`); the ontology now has
one owner. **And P7's blind declaration checked out**: the HTF LRSI trial-ledger row,
written from another branch's constants, matches the real grid exactly - 16 declared,
16 real, all 75 recipe ids owned by exactly one row.

**Frozen exe rebuilt** (P7 edited the packaging spec - trigger 2): 420 MB,
`selftest OK: 74/74 (frozen)`, exit 0. The new 74th check loads the registry JSON from
inside the frozen process, because a `datas` rule proves a file was bundled and only a
frozen run proves the process can read it.

**Verification.** 5781 passed. 33 failures, none a regression: 32 are the `ai_jobs`
tests standing down while the desk's nightly holds the machine-local writer lock (the
same tests fail on a pristine `main` worktree - checked, not assumed), and one was a
Windows `PermissionError` on `os.replace` inside pytest's own sandbox that did not
recur. `ruff` clean · smoke **7/7** · source `--selftest` **74/74** · spec-drift **17**.

### Older entries

Entries from **2026-09-01 back to 2026-08-26** (56 entries) moved to
[`docs/CHANGELOG_ARCHIVE_2026-08-26_2026-09-01.md`](docs/CHANGELOG_ARCHIVE_2026-08-26_2026-09-01.md)
on 2026-09-03 (F1 docs packet). Recent changes holds the last two build days.

## Revision history

Entries from **2026-09-01 back to 2026-08-26** moved to
[`docs/CHANGELOG_ARCHIVE_2026-08-26_2026-09-01.md`](docs/CHANGELOG_ARCHIVE_2026-08-26_2026-09-01.md)
on 2026-09-03 (56 entries); entries from **2026-08-19 back to the initial system in 2025-11** moved to
[`docs/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md`](docs/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md)
on 2026-08-27 (36 entries). Newer revisions are dated entries at the top of
`Current implemented inventory` above. The archive is evidence, not authority — read it
only when the history of a specific change is not answered here or by the governing spec.

## Retired or superseded implementations

- Desk Link satellite relay/control and the separate mini-PC scanner role are retired
  as of 2026-08-08. The code remains pending a scoped cleanup.
- H1 alerts were retired; H1 now confirms D1 tracker picks.
- The old DESK approval queue for auto-populate was superseded by direct day-scoped
  M5 Focus adoption.
- The legacy shared review-event ledger is read-only; per-installation shards are the
  current writer path.
- The legacy Tk UI, its shims, the Tk journal/market-prep tabs, `TickerMover.py` and
  `PyQt5` were REMOVED on 2026-09-03 (assessment packet F2). `scripts/ui` is the only UI.
- Historical plans and handoffs listed as such in `docs/README.md` are evidence, not
  current execution authority.
