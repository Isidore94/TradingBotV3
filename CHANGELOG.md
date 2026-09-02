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

- PySide6 Trading Desk launched by `launch_gui.py`, with the legacy Tk UI retained
  as a compatibility path.
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
  reason named. **Since 2026-08-31 it is a collapsible section under the Desk's
  Strength window rather than a left-nav page** (trader request): starting closed so
  it costs the charts nothing, sides stacked vertically for the column, its own
  RS/RW half retired to the Alert Center's RS/RW Board tab (one tab-click away in
  the same column), and a row click charting into the **Visual Alert Review pane**
  through `chart_symbol` rather than opening the snapshot popup.
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
  invalidation, background loading, prewarming, and stall watchdog.
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
- Phase 4: versioned XNYS sessions and deterministic M15/M30/H1/W1 aggregation.
- Phase 5: point-in-time daily/intraday feature snapshots and anchor instances using
  champion calculations, including AVWAP parity at 1e-9.
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

### 2026-09-01 - Phase 0.13 packet P3: the fact pack tells the truth

**Branch `claude/p3-fact-pack-truth`, off `main` at `66a0c31`.** Five changes to the
nightly `setup_research` pack and the warehouse readout. Shadow-only throughout:
nothing here reaches a detector, score, alert, Focus list or watchlist. Live gate #32
owed. Recorded as **BD-81 … BD-85** in `docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md`
(78/79 are taken by the unmerged Phase 0.12 branch).

**The case.** The 2026-08-31 pack had 9 eligible cells - every one
`AVWAPE_TO_FIRST_DEV`/LONG against an ATR stop control, every one NEGATIVE - printed in
a single table sorted by trimmed mean, so rows 10 onward were n=1 cells reading +2.9R.
The 80-row cap then dropped 508 more without saying which kind. It pooled GENERAL (735
occurrences) and FAVORITE_ZONE_WATCH (486) as trade setups, which Appendix C forbids in
those words. And it reported `n` as if outcome rows were samples.

**1. Episodes beside rows (BD-81), and the measurement changed the conclusion.** Every
cell now carries `n_episodes`. The floor still counts ROWS on purpose - moving it
changes what the model may narrate and is its own packet. But the assumption behind
that follow-up was wrong: on the live lake, 9,372 outcome rows rest on 599 occurrences
and 287 clusters, and yet **`n` and `n_episodes` were EQUAL in all 756 cells**. One row
per occurrence per recipe, so the per-cell count is not where the double-counting is.
The correlation is ACROSS cells - 15.6 recipe rows per occurrence, 1,804 of 3,436
clusters carrying more than one family - so the pack now also publishes
`evidence_shape` (rows / occurrences / episodes / rows-per-occurrence), which is the
denominator a reader comparing cells actually needs. BD-81 records that the follow-up
must be a cross-cell floor, not the per-cell swap first assumed.

**2. The eligible block leads (BD-82).** Two blocks: eligible whole and sorted as
before, then a bounded ineligible block sorted by n DESC then trimmed mean - the shape
the context-cell path already used - so what rides along is the thickest evidence below
the floor, never the luckiest single trade. Drops are counted per block. A pack
published before the split still renders as its author published it.

**3. Non-trade roles excluded and named (BD-83).** A small explicit role map -
`GENERAL` = FALLBACK, `FAVORITE_ZONE_WATCH` = WATCH_STATE, everything else TRADE -
keeps them out of every policy and context cell and publishes their counts, because an
absent family with no explanation reads as one with no data. Packet P7's registry
replaces the map.

**4. Coverage is published (BD-84).** New `research_warehouse/outcome_coverage.py`:
append-only, one line per outcome firing naming the symbol bucket it covered. The pack
reports buckets covered in the last 32 firings, families with occurrences but zero
outcome rows, and the first M5 session in the lake - so "not measured yet" reads
differently from "measured and flat". No history reads UNKNOWN, never "0 of 32".

**Deviation, reported not forced:** the packet asked for the sidecar "beside the packs"
in the AI store; that would make `research_warehouse.cli` import `ai_jobs.store`,
inverting the tree's one-way dependency. It lives under the store root instead. The
reader already imports the package, so the pack still gets the number.

**5. The readout is not hard-filtered (BD-85).** `slice_readout(setups=...)`: omitted
means the pinned slice (every existing caller byte-identical), None means every family.
`SLICE_SETUPS` is NOT widened - `cli._run_outcomes` uses it to choose which occurrences
get the legacy slice recipe, so widening it would change what the warehouse SIMULATES.
The panel gains a family combo and the four columns the query always computed and the
panel dropped: `n_symbols`, `n_sessions`, `n_truncated`, `as_observed_only`. Choosing a
family reads nothing; Refresh stays the only thing that touches the share.

**Owed, not built:** the optional `cell_history` block over the sibling packs on disk.

**Verification.** `pytest tests/ -q` **5741 passed, 72 subtests, process exit 0** ·
`ruff` clean · smoke **7/7** · source `--selftest` **73/73** · spec-drift 17 passed. No
frozen rebuild: `research_warehouse` is an existing collected package and there is no
new dependency and no new non-`.py` asset.

### 2026-09-01 - Phase 0.13 packet P7: one name per setup

**Branch `claude/p7-setup-registry`, off `main` at `66a0c31`.** Two READ-ONLY
modules; nothing in production imports either, and every row of both names the packet
that makes it authoritative (`plan.md P4.1`). No runtime behaviour changed, so no live
gate beyond green tests.

**1. The crosswalk.** `scripts/setup_registry.py` over a frozen
`setup_registry_v1.json` - 57 entries keyed `setup_id@version`, joining every spelling
each source uses to one canonical warehouse id, generated by
`scripts/build_setup_registry.py` and reviewed as a diff rather than recomputed at
import.

**THE PACKET NAMED FOUR SOURCES; THE CODE HAS FIVE.** `legacy.py` declares study
families as `*_STUDY_FAMILY` constants - 17 of them, and **eight are named nowhere
else** (`htf_trend_retest`, `hv_level_proximity`, `hv_level_break`,
`cloud_flat_proximity`, `compression_break`, `trendline_break`,
`relative_avwap_retest`, `relative_avwap_break`). Read by regex, without importing a
27k-line module and without writing to it.

Roles are Appendix C's vocabulary assigned from Appendix C's own table: `GENERAL` and
`none_of_these` FALLBACK, the three watch families WATCH_STATE, `baseline_every5`
CONTROL, everything else TRADE_SETUP. Status comes from `setup_docs`' own group text.

**Two refusals.** It does not resolve a disagreement - **eight `known_divergences`**
record what each source believes and leave the choice to P4.1. And it does not fill a
column the sources do not establish: supported sides, timeframe roles, the exact
trigger and the primary recipe are EMPTY on every row and listed under
`unestablished`, because a guessed side reads as established in exactly the column a
later experiment trusts. An unresolvable name RAISES rather than defaulting to
`GENERAL`.

**2. The look-counter.** `scripts/research_warehouse/trial_ledger.py` - one
append-only JSONL row per registered grid at
`<store root>/_diagnostics/trial_ledger.jsonl`, written at REGISTRATION time.
`register` refuses to rewrite an existing `trial_id`, and nothing in the module can
read an outcome. Four grids backfilled with their real authorization pointers -
M5-close (54 cells), HTF LRSI (16), AVWAP band challenger (3), the v1 recipe library
(5) - so the family-lifetime count does not start at zero for families already looked
at. Every recipe id resolves to exactly ONE row.

**The packaging guard fired and the SPEC was fixed, not the test.** The frozen JSON is
a new non-`.py` runtime asset at the scripts/ ROOT, which the spec's package-only
asset sweep could not see. The spec now sweeps the scripts root too, bundling to `"."`
because a frozen top-level module's `__file__` parent IS the bundle root. It was not
added to the unbundled allowlist: that is for files the frozen app provably never
reads, and P4.1 will make production read this one.

**Verified differences from the packet, none forced.** P3's role map is not on `main`
(it lives on `claude/p3-fact-pack-truth`), so the swap is owed to whichever branch
merges second; `setup_registry.fact_pack_role` is built and tested as the drop-in and
keeps the fact pack's own `TRADE` spelling. `HTF_LRSI_RECIPES` is likewise not on
`main`, so the ledger declares that grid from its own branch's constants and the
membership test iterates whatever `outcomes` exposes. And the packet's role vocabulary
(TRADE/STUDY) differs from Appendix C's (TRADE_SETUP/CONTEXT/WATCH_STATE/CONTROL/
FALLBACK, with study-ness a STATUS); the spec's won.

Recorded as **BD-86** and **BD-87** (written as 85/86 on the branch; the 2026-09-02
merge shifted them, because P3's own five shifted first - see BD-86's numbering note).

**Verification.** `pytest tests/ -q` **5625 passed, exit 0** with three files
deselected: 32 tests in `test_ai_jobs_runner.py`, `test_ai_evidence_coverage.py` and
`test_ai_jobs_store_window.py` stand down while the desk's nightly AI run holds the
machine-local `ai_jobs_runner` writer lock, which it did throughout this build.
**Verified environmental, not a regression - the same tests fail with every P7 change
stashed.** `ruff` clean · smoke **7/7** · source `--selftest` **73/73** · spec-drift
**17**. Fail-before-fix: with `scripts/` stashed, 24 of the 25 new tests fail.
### 2026-09-01 - Phase 0.13 packet P0: three trader decisions applied

**Branch `claude/p0-apply-decisions`, off `main` at `66a0c31`.** Three decisions the
trader gave in chat on 2026-09-01, each quoted in the code beside what it changed.
Live gate #29 owed.

**1. BANGER is retired.** Trader: *"not sure to be honest. We can probably remove
this because idk what it is."* It was a top-alert class with a matcher and no
producer. The only definition was `"BANGER" in raw_text.upper()` in
`alert_center_panel.py`; no detector path builds the token (the regime-pause sweep is
deliberately untiered and stamps none), and 0 of 8,818 recorded review rows carried
`banger=True` (`docs/analysis/EVIDENCE_AUDIT_2026-08-22.md`, row D8b). It granted
three privileges nothing could reach: a tier-gate bypass, an always-sound, and a
repetition escalation. All three are gone, along with `is_banger_alert`, the
`is_banger` argument to `RepetitionLedger.consider` and the `had_banger` row field.
The argument is REMOVED rather than ignored, so a caller still passing it raises
instead of silently doing nothing. The `banger` column survives in the review-event
row as a documented constant `False`, so every reader of the historical rows keeps
working and the schema id does not move. PROVEN stays the top class, untouched, and
the two feed labels now say so ("S tier / PROVEN only", "Sound on S/A + PROVEN").

**2. The LRSI M5 alerts are silenced; the evidence is not.** Trader: *"LRSI alerts
seem to be mostly spam. however I enjoy them as something that can boost the potential
of an alert. for now let's put them on the back burner. let's measure how they perform
on different timeframes but no need for their M5 alerts."* Measured basis: the two
LRSI levels were 84 of 128 new M5 episodes by 11:14 that morning - 66% of the
session's alert volume from one engine. New `LRSI_M5_ALERTS_RETIRED = True` beside
`H1_ALERTS_RETIRED`, applied at the EMIT seam.

The seam matters and was verified before it was chosen. `check_lrsi_cross_setups`
tests `is_m5_signal_enabled` **before** the event joins `hits`, so a `False` toggle
drops the event ahead of the candidate row and the outcome registration: flipping the
defaults would have stopped the very rows the trader asked to keep. The defaults stay
`True` with a comment saying why, and the gate sits after `record_alert_tier` instead
- so the sweep, the candidate row, `intraday_bounce_outcomes.csv`, the learning tier
and the PROVEN stamp all keep running and only `gui_callback` is skipped. The message
goes to the symbol log as `LEARNING_ONLY [LRSI M5 retired]`, exactly as H1 does.

One deliberate difference from H1: `log_bounce_to_file` still runs. H1's retirement
returns before it, but `journal_analytics.AutoTagger` reads `INTRADAY_BOUNCES_CSV` to
answer "which of my setups was this?", and skipping it would blank the tag on a real
LRSI trade. No Settings toggle exists for these engines - nothing in `scripts/ui/`
references `set_m5_signal_enabled` or `M5_SIGNAL_TYPE_DEFAULTS` - so there was no
dialog label to correct. The "different timeframes" measurement is the Phase 0.12
packet B warehouse study (`outcomes.HTF_LRSI_RECIPES`, M30/H1/H2/H4), which this flag
does not touch and which lives on a different branch.

**Owed, not built:** LRSI as a display suffix on OTHER M5 alerts - the "boost" the
trader described. `_format_bounce_alert_message` is a module-level function that takes
no bars, so feeding it a cross reading means plumbing bars through every champion
alert caller. That is a change to the champion alert path, not the display tweak the
packet's escape clause allowed, so it was skipped and is recorded here.

**3. Clicking away is a pass - recorded, not changed.** Trader: *"clicking away = a
pass. The tabs under the visual chart review should give us all the tools we need and
we decide as we see. set alerts / add to focus and then move on."* `_select_review_alert`
already wrote a `skip` row with `detail.reason = clicked_away_from_m5_alert`; the
trader has now confirmed that IS the intended meaning. No code behaviour changed. The
decision is in `docs/DESK_INTERNALS.md` under the M5 alert bar entry with a one-line
pointer at the writer, so no later packet repairs it into a take or into silence, and
the reason string is frozen because `review_learning` keys on it.

**Verification.** `pytest tests/ -q` **5720 passed, 72 subtests, process exit 0**
(desk `.venv`) · `ruff` clean · smoke **7/7** · source `--selftest` **73/73**. Every
fix ships a test proven to fail with `scripts/` stashed: five for BANGER, seven in
`test_r5_lrsi_cross_wiring.py` for LRSI. No frozen rebuild - no packaging trigger was
hit (no new dependency, no new non-`.py` runtime asset, no new top-level `scripts/`
package).
### 2026-09-01 - Phase 0.13 packet P1: grade what you already said

**Branch `claude/p1-grade-what-you-said`, off `main` at `66a0c31`.** Four defects in
the loop between a decision the trader makes and the evidence that grades it. Every
premise was reproduced at code level and against the live stores before anything was
edited. Live gate #30 owed.

**1. Today's swing picks never reached `human_focus_swing_vetted`.** `_pick_key`
returned (trade_date, symbol, side) with no category, so a name already on one Focus
list swallowed its row on the other. Live: AMGN LONG was liked into swing Focus with
origin `vetted` at 11:33:06 on 2026-09-01, the day already held a `focus_m5` AMGN LONG
row from 08:02:14, and the swing row was dropped - `grep -c vetted` on
`human_focus_daily_picks.csv` reads **0 across all 4,083 rows**. The cohort that origin
exists to build has never had a single row.

The diagnosis already existed: `focus_membership_events.py`'s docstring names it as
audit F3 and keys its own episodes by category. The pick store is what never caught up.
The key now carries the category slot - the base source with any like-origin suffix
removed, so `focus_swing_vetted` and `focus_swing` are one swing membership and a
re-snapshot cannot duplicate a row. The same key runs over the outcomes file, so the
two cohorts grade forward independently. No column or schema moves: every historical
row carries `source` and re-keys to the slot it already occupied.

Two joins had to follow the rows. `weekend_prep_panel._join_focus_week` would have
handed one category the other's forward returns - opposite trades, not a rounding
error - and now uses the one canonical `pick_source_family`.
`journal_walkaway.load_focus_positions` would have replayed a two-list name as two
positions and double-weighted it in every aggregate; the trader was in one position,
so it dedupes and leaves the cohort question to the tracker.

**Two packet premises differed from the code and are reported, not forced.** The
swing-favorites write-through ALREADY EXISTS and works: `_place_in_focus` ->
`FocusPickStore.add` -> `focusChanged` -> the Focus panel's coalesced
`_apply_focus_change` -> `snapshot_today(force=True)`, which passes `force` and is
never stopped by the `already_snapshotted` early return. QFIN on 2026-08-31 proves it -
liked 11:26:19, pick row stamped 11:26:20. And QFIN's `focus_swing_manual` is not a
live code path: `FOCUS_LIKE_ORIGIN` read `"manual"` until commit `edc7999`, which
landed at 11:36 that day - ten minutes AFTER the like. Nothing to fix at the source,
and the existing row is correctly left alone.

**2. A like merged overnight; a veto merged on the click.** `like_cohort_picks.csv` was
last written **2026-08-27** (53 rows) against like_claim annotations recorded through
09-01, so a like was invisible to its own cohort for up to a day - and indefinitely on
any day the overnight job did not run. The two cohorts are read side by side on Weekend
Prep, so a difference between them has to come from the data. `commit_like` now merges
through the same `_merge_cohort_safely` the veto uses, failure swallowed the same way
(the annotation row is already on disk), and `merge_like_cohort_picks` takes the writer
lock now that it has two callers. The nightly slot stays; both are idempotent.

**3. Unversioned veto codes pooled only with the lowest vocabulary.**
`_canonical_cohort_map` mapped the unversioned form of a reason only while walking
`min(versions)`, so a code introduced later got no unversioned mapping at all. Live:
`human_focus_veto_compressed` (n=3, PF 165 at h3) sat beside
`human_focus_veto_v2_compressed` (n=18, PF 0.39) - one judgement read as two opposite
ones, the three-sample half looking spectacular. A `setdefault` on the already-ascending
version walk IS "the earliest version that defines this code" and stays right for
anything added later. Verified against the loaded vocabularies: `veto_compressed` ->
`veto_v2_compressed`, `veto_sma_incoming` -> `veto_v3_sma_incoming`, `veto_volume_dry`
-> `veto_v1_volume_dry` unchanged. No literal `vocab_version` is asserted in either new
test - both load the vocabularies and DISCOVER which codes arrive late.

**4. The scoreboard ignored ~640 explicit decisions and could not see an R gap.**

Seven action families joined the take/reject sets, each classified from what its WRITER
does in `alert_center_panel.py` rather than from its name, and each documented in the
module comment the way `like_advance` is. Live counts: `auto_pick_pass` 254,
`arm_d1_event` 160, `focus_review_remove` 88, `focus_review_keep` 71,
`auto_pick_approve` 63, `arm_any_bounce` 22, `veto_day_trade` 4. `veto_day_trade` is a
REJECT because the episode being graded is the D1 chart that was shown; the M5 interest
is a different claim on a different timeframe. Machine events, `*_fired`, `*_expired`
and every `disarm_*` are deliberately excluded and pinned by a test. Measured effect:
takes **645 -> 845** of 2,607 shown, overall take rate **0.247 -> 0.324**.

The new **`r_gap`** class fires when both sides carry >= 8 measured R and the averages
differ by >= 0.5R, with no reference to the take rate - so it can only surface what the
other two are structurally unable to see.

**The packet's live case moves once the action sets are fixed, and that is reported
rather than papered over.** `bounce_type=lrsi_cross_20` at taken -0.376R (n=8) vs
passed +0.962R (n=24) reproduces EXACTLY on the un-fixed sets, and the new class
catches it at a -1.34R gap while blind_spots and leaks are both empty. Under the
corrected sets it no longer reads that way: seven of those lrsi charts turn out to have
been ARMED rather than passed, so taken becomes +0.519R (n=12) and the gap closes. The
apparent edge was an artefact of the misread decisions - which is what the action-set
fix exists to remove. The class is pinned to the packet's literal numbers in a test, so
its behaviour is proven on that case either way. On the live store today it produces
**18 callouts while blind_spots and leaks produce 0**, so it is currently the only class
saying anything at all.

`r_gap` is REPORT-ONLY by design: it is a field on `review_preference_state.json` and a
section in the rendered report, and it is deliberately not wired into
`draft_policy_from_state`, `review_guidance` or the AI evidence package - those write
priority deltas into `review_policy.json`, which this packet may not touch. A test
asserts none of the three so much as mentions it.

Chart Review's coded vetoes now feed the `dislike_reason` dimension. The join was
MEASURED before it was built: **202 of 212 vetoes join to an existing episode, 198 of
those to a SHOWN one, and the side matches on 202 of 202 with zero mismatches**, so the
packet's stop-and-report condition does not apply; 199 attach inside the 90-day window.
It annotates and never re-resolves - the verdict still comes from the review event store
alone. A veto whose side disagrees is SKIPPED rather than guessed, one with no episode
is left alone rather than inventing an impression, and an unreadable log is 0 rather
than an error.

**Verification.** `pytest tests/ -q` **5737 passed, 72 subtests, process exit 0** (desk
`.venv`) · `ruff` clean · smoke **7/7** · source `--selftest` **73/73**. Every fix ships
tests proven to fail with `scripts/` stashed: 3 for #1, 3 for #2, 2 for #3, 12 for #4.
No frozen rebuild - no packaging trigger was hit. The trader's live evidence files were
checked before and after and are byte-unchanged (the suite redirects
`TRADINGBOTV3_DATA_DIR`, `tests/conftest.py:57`).
### 2026-09-01 - Phase 0.13 packet P2: show me

**Branch `claude/p2-show-me`, off `main` at `66a0c31`.** Six display changes, each
read-only over a file something else already writes. Nothing reaches a detector,
score, alert, Focus list, review queue or `review_policy.json`. Live gate #31 owed.

**1. The two judgement tables show the robust half.** `_read_veto_cohort` and
`_read_like_cohort` projected six columns and dropped `median_return`,
`trimmed_mean_return`, `ci_low`/`ci_high`, `symbols`, `sessions`, `top_symbol_share`,
`evidence_label` and `meets_n_floor` - every one written by `human_focus_tracking`
since R10.C, and most already on screen in the Focus performance table on the SAME
page. What survived was a bare mean on a ratio, which is exactly the statistic R10.C
published the robust half to stop anyone reading alone.

All of them render now, through one shared `_cohort_robust_fields` so the two tables
cannot drift. The view is ONE horizon with a selector (default h3) that re-renders from
memory - a view change never touches disk on the Qt thread. `meets_n_floor` is
deliberately not a column: it decides the ORDER and the greying, so the live
`human_focus_veto_compressed` row (n=3, PF 165) sorts after every cohort that cleared
the floor instead of wherever the CSV put it, and the note says a row below the floor is
not a weak finding but not a finding. Rows above it order by the TRIMMED mean. The
liked table carries the bounded-picklist caveat the AI gets on every package, through
the one existing `ai_summary._offered_claim_caveat`.

**2. The callouts are named.** `_build_summary_text` printed "Blind Spots: 3" over a
store that has always known which segment, how often it was shown, and what each half
measured. `callout_lines` builds those rows on the worker, from the state file only, and
reads the classes DEFENSIVELY so the page works against a scoreboard written with or
without P1's `r_gaps`.

**3. "My Decisions" beside the Daytrade Tracker.** `review_preference_state.json` had no
surface outside a text report. One tab per dimension (13), same shape as the tracker
tabs, ordered by how often the segment was shown. `gap` is the one derived number and it
is computed only when both sides carry a measured average. Off the Qt thread both times;
construction READS while only the button rebuilds. The probation badge is set membership
over `M5_SIGNAL_TYPE_DEFAULTS - BOUNCE_TYPE_DEFAULTS` - no threshold, no second list.

**4. The AI phase gates get a surface.** New `ai_jobs/gate_counters.py`, pure and
Qt-free. Live on this desk, read not typed: **Digest 6/10 · Enrichment 6/10 · Weekly
synthesis 2/10 · Policy draft 5/10 · Evidence window 6/10**. The synthesis count goes
through `_read_cohort` + `graded_sessions`, the two functions `run_weekly_synthesis`
uses, because a second counting rule could disagree with the document it reports on; the
draft and evidence counts are parsed from the PUBLISHED files rather than recomputed. An
unreadable source says "unavailable" with its reason - a blank cell reads as zero, and
zero is a claim.

**5. The take-rate suffix. THE CODE DISAGREED WITH THE PACKET AND IS REPORTED, NOT
FORCED.** The packet's premise was that guidance is computed before
`m5AlertPosted.emit`. It is not: the emit is at `alert_center_panel.py:2018` and
`_enqueue_review_alert` returns for an M5 alert at 2026, before `_queue_score` - the only
enqueue-path caller of `_guidance_for` - is reached. So `_attach_cached_take_prob` reads
`_review_guidance.get` and NEVER `_guidance_for`, whose `_refresh()` stats two files and
can re-read a 34 KB JSON per alert on the Qt thread. The consequence is stated rather
than hidden: the suffix appears for a symbol the desk has already charted this session
and is silent otherwise, which is the honest rendering of "not measured".

**6. The repetition fold.** The main feed has folded repeats since 2026-08-16; the bar,
which is narrower and read faster, drew one line per alert. A repeat of the same
symbol+side now folds with a ×N badge and returns to the top carrying the newest alert -
so a tier upgrade rewrites the row with the stronger one. Keyed on symbol AND side,
because a name that flips direction is a different claim and folding those would hide the
flip. **Presentation only**, and the bar's docstring still says so: every event reached
`_enqueue_review_alert`, the outcome CSV and the review-event store first, and a folded
row's tooltip says it folds rather than drops. One existing test encoded the old
one-row-per-alert rule and was rewritten to the new one - the authorized behaviour
change, not drift - with the invariant it protected now held by
`test_the_fold_is_presentation_only`.

**Found by the suite, and fixed:** both new workers could emit into a deleted panel
(`RuntimeError: Signal source has been deleted` out of a daemon thread). `shutdown`
joins, but deletion can win the race; both guard the emit now and drop the payload,
proven deterministically with `shiboken6.delete`.

**Two fixtures were wrong and are corrected, not worked around.**
`test_focus_review_keeps_its_rows_when_a_refresh_fails` used `"horizon": "h3"`, which
nothing writes - `human_focus_tracking` writes plain integers and the live rollups carry
"1"/"3"/"5". `test_table_width_rule_pages` rendered cohort rows with no `horizon` at all.
Both were invisible until the selector started filtering on it.

**Verification.** `pytest tests/ -q` **5775 passed, 72 subtests, process exit 0** (desk
`.venv`) · `ruff` clean · smoke **7/7** · source `--selftest` **73/73** · spec-drift
green. No frozen rebuild: `ai_jobs` is an existing collected package, and there is no new
dependency and no new non-`.py` asset.
### 2026-09-01 - Phase 0.13 packet P4: the variables you are not looking at

**Branch `claude/p4-swing-variables`, off `main` at `66a0c31`.** Two halves, both
authorized by the trader before the first edit to `master_avwap_lib/legacy.py` (the
file-scoped ask-first rule): Half A capture-only, Half B all six items. Live gate #33
owed.

**HALF A - see the evidence, add the variables.**

The scanner has written `master_avwap_setup_attribute_leaderboard.csv` every scan since
it was built - ~190 attributes x side x bucket, each with its own edge - and its only
readers were the legacy Tk GUI and the offline tuner. It is now an **Attributes** tab on
the Qt Setup Tracker. Live: **38,617 groups, 37,049 of them (96%) under the
reportable-n floor**, which is why the order is the honesty - floor-clearing rows first,
sub-floor rows greyed and last with a "below floor (<30)" label, and every row KEPT.

Read **off the Qt thread**, unlike its ten siblings: 19.7 MB against 5.5 MB for the next
largest and under 150 KB for the rest. `TrackerTableModel` gained one opt-in row flag
that mutes a row ahead of every other colour rule.

Twelve variables that were already on the record or the row now have attribute keys:
human focus pick/side, tracker setup family, market regime label, sector, industry, ATR
as a PERCENT of price, signed SMA200/SMA50 distance in ATR with two booleans, and
relvol. **No weights, no gates.** The percent-of-price ATR sits BESIDE the dollar bucket
(a $2 ATR is quiet on a $400 stock and violent on a $12 one), and the SMA geometry is
the trader's rule 3 recorded as D1 evidence so the swing record can agree or disagree
with the Alert Center's filter.

The golden is the RANKING ITSELF (`p4_ranking_unchanged_v1.json`, contract-bearing,
frozen from the pre-Half-A code with `scripts/` stashed, replayed rather than compared).
Score, static score, proven-quality score, bucket and expected R are unchanged. A second
structural guard asserts none of the twelve keys appears in the scoring functions at all.

**HALF B - grade what shipped.** Each item behind its own fixture, frozen first.

- **B1** The attribute leaderboard states its own sample floor. Only numeric bucketing
  had one; categorical, bool and list rows shipped at setup_count=1 with full edges.
  `meets_n_floor` and `evidence_label` now travel on every row, through
  `evidence_stats.summarize`, asked of CLOSED setups. The fixture freezes the leaderboard
  AND the tuner's recommendations, and is built so the two verdicts DISAGREE - a 20-setup
  group is under the reportable floor but clears the tuner's own gates, and the tuner
  still writes its rule. B1 publishes that and changes nothing about it.
- **B2** The leaderboard can be read by family and by regime, as separate sibling files.
  The existing export keeps its exact grain because the tuner reads it into live weights.
  Columns are read BY NAME now - the extra dimension prepends, so positional indices
  would have shifted every column one place.
- **B3** Fictional horizons leave the scan-factor leaderboard. `future_idx = idx +
  horizon` indexes a symbol's own scan rows, not exchange sessions: live medians are
  horizon 5 -> 64 sessions, horizon 10 -> 73, with 42-45% of rows over twice their
  horizon. `stale_horizon` was computed since R10.D and never filtered on. Dropped now,
  with the count and reason on every row. A row whose drift could not be measured is
  KEPT. **Step (a) only** - the future row is still selected the same way, and a test
  pins that.
- **B4** The tier tracker grades the tier that SHIPPED. `assigned_tier` is stamped at
  assignment time, after the expected-R demote, the de-dupe and the best-swing merge;
  the grader prefers it and reports `tier_source`. The bucket derivation stays for the
  months of rows without the column.
- **B5** Calibration reads structure points. The record stored only the overwritten
  proven-quality score, so the expected-R fit was reading realized performance as
  structure quality - a feedback loop. `static_score` is on the record now and the helper
  prefers it, with a counter reporting how much of each path a run used.
- **B6** The representative exit template is NAMED.
  `REPRESENTATIVE_EXIT_TEMPLATE_ID` defaults to today's behaviour so nothing moves; the
  resolved template travels on the summary and is printed in every `expected_r_note`,
  and `setup_docs.py` now says the headline R is not measured on the plan it documents.

**Two things reported rather than forced.** The packet named
`_build_priority_tier_sections`; the function is `_priority_partition_tier_rows`. And
B5's anchor movement cannot be measured on this tree - no record on disk carries
`static_score` yet, so every sample still takes the old path and the fit is unchanged
today; the new counter is what will show the changeover.

**Verification.** `pytest tests/ -q` **5759 passed, 72 subtests, process exit 0** ·
`ruff` clean · smoke **7/7** · source `--selftest` **73/73**.
### 2026-09-01 - Phase 0.13 packet P5: pass and not-today get graded

**Branch `claude/p5-pass-cohorts`, off `main` at `66a0c31`.** Two new cohorts,
completing the set: every verdict the trader can record now has a forward record.
Live gate #34 owed.

**The gap.** The veto cohort has graded what was thrown away since it shipped and the
like cohort what was endorsed. Three verdicts had nothing: the day-trade **pass**
("I really like this stock for a daytrade but it has this ONE issue", 2026-08-31),
**not_today** (223 rows on the live log) and **dislike** (34 rows, carrying the most
information-dense free text the trader writes).

**1. `ui/annotations/pass_cohort.py`.** A pass is MULTI-SELECT, so it grades under each
of its reason codes AND under a pooled `pass_all` - k+1 rows. The code cohorts therefore
OVERLAP AND MUST NEVER BE SUMMED, and that fact travels three ways: the module
docstring, a `reason_code_count` column on every row, and `OVERLAP_NOTE`, which the
Weekend Prep note and the AI scope label both READ rather than retype. Identity on write
is (vocab_version, reason_code); no version returns the historical unversioned form so a
row already on disk keeps grading where it was filed.

The intraday grade: entry at the first completed M5 close AFTER the pass - never a bar
that had not finished - stop at the session extreme on the pass side up to entry, target
2R, STOP FIRST on a bar touching both.

**MEASURED AND REPORTED:** on the live desk that grade is currently always blank, and
the reason is structural. The sidecar is written from the bars the desk was ALREADY
HOLDING when the pass was recorded, so every bar in it starts BEFORE the pass and the
entry bar the rule asks for is never inside it. Rather than an ambiguous blank, every row
carries `intraday_unmeasured_reason` - `sidecar_ends_before_the_entry_bar`, which is a
different fact from `no_sidecar_bars`. Whether entry should instead be the last completed
close AT the pass is a definition change and the trader's to make.

**2. `scripts/rejection_cohort.py`.** `not_today` and `dislike` are separate cohorts and
never combined into a verdict; the pooled base row is labelled rather than hidden (R1). Each source names its lane - `focus__m5_not_today`, `focus__swing_dislike`. Live: 253 gradeable rows, 219 + 34, zero sideless. `unfavorite` is NOT
graded - a membership change rather than a verdict, and sideless on the live log - and
the free-text `reason` is carried verbatim and never coded, because the whole value of
those 34 dislikes is the sentence.

**3. The one change to existing code.** `update_human_focus_outcomes` keyed outcome rows
on (trade_date, symbol, side), and every row of one multi-code pass shares all three, so
they would collapse into one and k of the k+1 cohorts would vanish. A new `pick_key`
parameter DEFAULTS TO None - every existing caller unchanged - and the two P5 cohorts
pass `pick_key_with_source`. The outcome numbers are identical across those rows, so what
the wider key preserves is which cohorts were graded, not which figures.

The rejection sources are `focus__m5_not_today` / `focus__swing_dislike` - R1 put the
LANE back into the name - and the DOUBLE underscore
is load-bearing: the prefix matcher tests `startswith(prefix + "_")`, so `focus_` claims
exactly those and cannot reach `focus_swing`, `focus_m5` or `focus_pick`. Pinned by a
test.

**4. Surfaces and wiring.** Two nightly slots appended (5-minute reserve, deterministic,
no model - asserted). Capture-time merge for a pass, mirroring the veto's, through one
shared helper. Two Weekend Prep tables showing the six columns PLUS `meets_n_floor` and
`evidence_label`, sub-floor rows greyed. Both performance files added to the evidence
report and to `ai_summary`'s `trader_judgement` scope - along with the LIKE file, since
that scope read the veto trio only and so asked "were your rejections wrong?" without
ever asking "were your endorsements right?".

**Six existing tests pinned the old sets and were updated - the authorized change, not
drift.** Three asserted an absolute slot prefix: P5's cohorts sit before
`evidence_report` because the report READS them, which moves later slots' INDEX without
reordering any existing PAIR, so they now assert the pairwise order - the actual
invariant, and one that will not need editing next time a cohort is added. Three asserted
the judgement scope held exactly three sources; they now compare against the scope's own
declaration.

**Verification.** `pytest tests/ -q` **5749 passed, 72 subtests, process exit 0** ·
`ruff` clean · smoke **7/7** · source `--selftest` **73/73** · spec-drift 17 passed.

### 2026-09-01 - Phase 0.13 packet P6: from what the trader said to what they traded

**Branch `claude/p6-preference-to-trade`, off `main` at `66a0c31`.** Three stores each
held a third of one question and nothing put the three on one row. Live gate #35 owed.

**1. Exact-id candidates in the auto-tagger.** A fifth source, `trader_capture`: for the
trade's symbol, any veto / like_claim / pass or take-class review event whose session
falls inside THE TRADE'S OWN WINDOW - open date to close date, not the fuzzy 16-day
neighbourhood the scanner lanes search, because an event id is only worth carrying when
the statement and the trade really are about the same episode. It ranks above every fuzzy
source and a fuzzy match can never displace it. A like_claim contributes its claimed
setup id; a veto contributes `vetoed:<code>` and a pass `passed:<code>`, **prefixed so a
rejection can never read as an endorsement**. Live: **1,229 capture rows, and 8 of 193
trades now carry a capture candidate.**

The candidate carries `context_row_id`, a new nullable column arriving through the store's
OWN additive migration list rather than an in-place edit. It is **a pointer for a reader,
never a canonical link** - plan.md P5.3/P5.4 own the canonical id and a second one
invented here would compete with it. Only **54 of 730** take-class review rows carry an
alert `event_id`, so the rest point at their own natural identity
(`review_event:<ts>`); an empty pointer would look exactly like a fuzzy candidate.
Nothing writes `trade_annotations` - the tagger suggests, the trader accepts.

**2. The nightly report.** `preference_trade_outcomes.py`, following `journal_walkaway`'s
read-only pattern: one row per statement across four channels, joined to the journal and
to the cohort grade. **Live on this desk: 558 statements in 90 days, 13 traded, 545 not -
and the not-traded rows are the point.** Every row renders its **match confidence or "no
match"**, with `match_basis` naming what the match rested on (same-session 0.9, in-window
0.7, side-unknown 0.5, opposite side 0.35). Nothing mints an identifier and a test bans
`uuid` / `hashlib` / `opportunity_id` from the module outright. Swing favourites are
resolved PER SESSION through the store's own `favorites_for_session`, so a name added and
retracted is not reported as a pick the trader never took; an unmatured paper grade is
blank, never zero. Registered as a deterministic slot BEFORE `evidence_report`, which it
feeds, and surfaced as a Weekend Prep table.

**3. The honest empty-dimension banner.** "My setups" renders beside a full auto-tag chart
of the same width while resting on almost nothing - live, **0 of 156 closed trades carry a
confirmed tag**. Below 10% coverage the group's label is prefixed with one sentence saying
so, through the same refusal-message mechanism `resolve_pnl_key` uses. The group is never
hidden.

**Also:** `ai_summary`'s comment said the `market_journal` scope is "OPT-IN ONLY". That has
been wrong since R10.H - `briefs.DEFAULT_SCOPES` carries it on the nightly run - so the
COMMENT was the defect and is corrected. No behaviour changed; whether it should be
nightly is the trader's decision.

**Four existing tests pinned absolute slot positions and were updated** - the same
authorized change as P5's. They now assert the pairwise order, which is the real
invariant. A circular import had to be resolved: `journal_store` already imports from
`journal_analytics`, so `TRADER_CAPTURE_SOURCE` is defined there and re-exported, keeping
the dependency one-way.

**Verification.** `pytest tests/ -q` **5747 passed, 72 subtests, process exit 0** ·
`ruff` clean · smoke **7/7** · source `--selftest` **73/73** · spec-drift 17 passed.
Fail-before-fix: with `scripts/` stashed including the new module, 29 of the 32 tests in
`tests/test_p6_preference_to_trade.py` fail.
### 2026-09-01 - Phase 0.13 packet P6a: tag the backlog

**Branch `claude/p6a-tag-backlog`, off `main` at `66a0c31`.** Authorized by the trader:
*"let's get Opus to do the tagging and I can review after"*. Live gate #36 owed.

**The gap.** 193 trades, and exactly ONE carried a setup tag the trader typed. Every
per-setup statistic on the desk rested on that row. The missing thing was never evidence -
it was a human decision about 155 closed trades.

**1. A permanent mark.** `trade_annotations.tag_status`, through the store's own additive
migration list. The column's DEFAULT is what made it safe on the live database: every
existing row was typed or accepted by the trader, so it became `confirmed` the moment the
column appeared. Three states, the third being `needs_review` - the tagger saying it
looked and would not guess, carrying no tag.

**2. `scripts/journal_bulk_tag.py`.** Dry run by default, idempotent, and bounded: the
refusal to overwrite a confirmed row lives in `JournalStore.apply_provisional_tags` rather
than in the caller, because an exception that depends on every caller remembering a rule
is not a boundary. It never promotes a shape tag (`midday` is a fact about the clock at
confidence 1.0 and would outrank every scanner match while answering a different
question), and it **never writes `tag_corrections`** - that table is the trader's feedback
TO the tagger, and a machine writing it is the tagger teaching itself.

**THE MEASURED RUN, 2026-09-01.** Threshold **0.70**. Live histogram of the top
setup-lane candidate over the 52 closed trades that had one:

```
  0.20   1 | 0.25   1 | 0.30   1 | 0.40   1 | 0.45   2 | 0.50   5 | 0.55   3
  0.60  10 | 0.65   2 | ----- threshold 0.70 ----- | 0.70   5 | 0.75   2
  0.80   5 | 0.85   6 | 0.90   3 | 0.95   5
```

156 closed trades considered, 0 already tagged by the trader (their one tagged trade is
still open), 104 with no scanner candidate at all. **Applied 24 provisional tags, marked
132 needs_review, refused 0**, and wrote 24 adjustment rows and **zero** tag corrections -
the one correction in the store is the trader's own from 2026-08-22. The journal was
copied to `trade_journal.sqlite3.p6a-backup-20260901_214926` first.

The threshold encodes a sentence rather than a percentile: tracker + same day + same side
is 0.72 and a focus favourite is 0.68 before its bucket bonus, while the SAME tracker row
one day later reaches 0.66. Everything under the line still gets looked at - it gets a
marker instead of a guess.

**3. The review surface.** A tag-review filter and a count above the Trades table, because
a hidden row and an absent row look the same in a table. It narrows the rows ALREADY
LOADED and issues no query. The Tags cell says `(provisional)` in text rather than colour -
a `QTableWidgetItem` cannot be reached by `theme.qss`, and a brush here would be the one
place in the desk painting outside the theme. One click confirms; an edit replaces and
teaches, and **only an edit teaches**.

**4. Analytics.** "my setups" groups on CONFIRMED tags only; "provisional setups" is its
own group with no catch-all bucket. Never blended, and the chart says so.

**VERIFIED DIFFERENCE FROM THE PACKET.** Its binding rules say the list-trades load runs
off the Qt thread. **It does not.** `TradesTab.reload()` calls `journal_feed.load_trades`
synchronously (line 468 before this change) and `AnalyticsTab` does the same (line 182);
the Journal's only worker is the migration one. Nothing here makes that worse - the new
surface adds one indexed single-row read per selection and no reload - but moving that
load to a worker is its own packet.

One existing test pinned `nonexclusive_groups` to exactly two entries and now asserts the
invariant rather than the length.

**Verification.** `pytest tests/ -q` **5737 passed, 72 subtests, process exit 0** · `ruff`
clean · smoke **7/7** · source `--selftest` **73/73**. Fail-before-fix: with `scripts/`
stashed including the new module, all 17 tests in `tests/test_journal_bulk_tag.py` fail.

Dated entries for the two most recent build days, newest first. Older dated entries
move to the archive; the durable statement of what they built is in the inventory above.


### 2026-09-01 — Focus de-clutter, and a shadow study of higher-timeframe LRSI entries

Phase 0.12, authorized by the trader in chat that day. Two independent packets:
one changes the desk, one adds a research lane with zero desk cost.

**Packet A - the Focus surfaces stop growing without bound.**

- **A Focus pick's automatic D1 alerts are PULLBACKS only.** The extension set -
  new 5d/20d extreme, SMA break, AVWAPE and 1σ break - no longer fires by itself;
  it is what filled the feed with "still going" news about names the trader had
  already seen. The trader arms the ones they want, and `_poll_d1_event_watches`
  stays the single path that fires one, so an extension event cannot arrive twice.
  The gate is at the flag-GENERATION seam: an extension kind is never constructed
  in `_poll_focus_d1_interest`, so nothing has to be suppressed downstream. The
  2026-08-05 one-extension-per-day ration is gone - it had nothing left to ration,
  and a filter that can never fire reads to the next agent like a live rule.
- **An arm now has a life, measured in SESSIONS.** 5 trading days for a manually
  armed 5d extreme watch, 10 for a 20d one, 10 for D1 level watches, any-bounce
  watches and manual price alerts. `market_calendar.trading_days_between` is new
  and is the clock - weekday arithmetic counts Thanksgiving and brings a Friday
  arm due on the wrong Friday. Uncertainty never deletes: a date the calendar
  refuses keeps the entry armed. Every expiry appends a row naming store, symbol,
  kind, `armed_at` and `expired_at`. A price alert is DISARMED rather than
  deleted, keeping its levels, note and history, so `price_alerts.json` still
  honours "user-entered names are never automatically removed"; arming restarts
  its clock. Expiry rides the poll that already owns each store - no new timer.
- **A quiet Focus pick fades, reversibly.** Ten trading days with no alert and no
  pullback event and the pick moves to a faded list; a fired Focus D1 flag, an
  armed-watch hit or the trader's own "★ keep" resets the clock. It covers swing
  and M5 picks including the trader's own - an explicit authorization to
  auto-remove a hand-typed name, scoped to Focus and routed through the store's
  own removal path so a hand-maintained watchlist line is untouched. Nothing is
  deleted: "★ Restore to Focus" gives a FRESH ten sessions, "✕ Discard" clears the
  list and leaves the evidence. A faded swing favorite appends a RETRACTION with
  origin `focus_fade`, never an edit, and no `pick_feedback` verdict is written -
  a fade is the desk noticing silence, not the trader passing a verdict, and every
  verdict in that file feeds a graded surface. `FocusPickStore` is the single
  writer; the check runs on the day roll plus a half-hourly timer, never inside
  the 60 s poll's per-symbol loop.
- **The buttons say how many.** "Review ▶" became "Focus pick review (N)", with
  "Faded review (N)" beside it. Both counts repaint through the board's existing
  `SignalCoalescer`, so a burst of Focus mutations is one render.

**Packet B - is there anything in a higher-timeframe LRSI entry? Shadow only.**

- **H2 is a derived timeframe again.** The locked plan cut it for having no
  consumer and named that as the reopen condition; this study is one (BD-78).
  Additive: no existing timeframe, contract id or published row changes. RTH is
  6.5 h, so H2 and H4 end each session with a stub - published as evidence,
  excluded from the oscillator's input, because an EMA fed a 30-minute bar inside
  an H2 series measures a duration that changes with the time of day.
- **The short legs are unmirrored, and that is a decision with a stated cost**
  (BD-79). The efficiency formula clamps at 0, so a perfectly efficient DOWN move
  and a motionless one both read 0 - the mirrored-close idiom and `cross_down` are
  different features, not a transform of one. All four legs read the SAME series
  so the grid answers one question; the cost is that the short legs measure
  EXHAUSTION rather than down-momentum and fire earlier.
  `tests/fixtures/efficiency_lrsi_research_v1.json` pins the gap as a number: the
  unmirrored down-cross at bar 27, the mirrored up-cross at bar 29.
- **A bounded 16-recipe diagnostic grid**, never a Cartesian search: M30/H1/H2/H4
  x four entries, one stop model (the signal bar's extreme + 0.25 ATR on the SAME
  timeframe - an M5 ATR under an H4 entry would size risk off a bar the recipe
  never looks at) and one 2.0R target. It reads the occurrences and canonical M5
  bars the nightly has already materialised, so it adds simulation work and not a
  second data pass.
- **Nothing is registered in `outcome_semantics`** (BD-80): these rows are keyed by
  `recipe_id` and never acquire a bounce family, so they never reach `claim_kind`.
  Registering the `lrsi_cross_80` that registry's docstring names as a
  hypothetical would assert a claim kind for a family with no producer.

Live `CROSS_LEVELS` and every `m5_signal_engines` behaviour are unchanged, and no
Packet B output reaches a detector, score, alert, Focus list or review queue.

### 2026-08-31 — Desk snappiness packet 3: the log, the downloader, the hidden pages, the drips

The last of the three. Packets 1-2 took the six largest causes of the day's
~78 minutes of GUI freeze; this one takes what remained.

- **The 618 MB technical-integrity log stops being replayed whole.** Its
  `level_resolved` rows are mirrored to a derived sidecar as they happen, with
  the source byte offset on every line so the file stays append-only and a
  catch-up streams only the tail. The main log is untouched - same rows, same
  path - and the replay falls back to the full stream on any doubt. Layout and
  reasoning in `docs/DESK_INTERNALS.md`. The **month roll was not built**: it
  would need the research warehouse's one-path `BronzeArtifact` contract to
  accept segments, which is a locked area this packet does not authorize.
  **Review-round fix (2026-08-31):** a thread switch between the clock's
  main-log append and its sidecar mirror, during a concurrent sync catch-up,
  could mirror one event twice and make the replay count it twice (reproduced
  deterministically). The reader now dedupes on the source byte offset each
  line already carries; the duplicate may sit on disk, never in the answer.
- **The Industry Board obeys quiet hours.** It was the only recurring
  downloader without an `auto_scanning_due` gate, so its ~1,930-ticker
  nine-month `yf.download` ran hourly all night and fired five seconds after
  every launch. The automatic tick is gated, fail-open; the manual button never
  is. The download is chunked at 200 with per-chunk failure isolation.
- **Three hidden pages stop paying for their timers** - the auto-watchlist
  viewers, the Master AVWAP scheduler tick and the RS Window auto tick. The
  timers keep running; the work early-returns while hidden and `showEvent`
  catches up once. The watchlist viewer also guards its `setPlainText` on the
  file's (mtime_ns, size), so reading a list no longer means being yanked to the
  top every thirty seconds.
- **Eight drips**: the entry-assist board moved to a worker; the 3-second health
  tick stats inline and spawns a thread only when the file moved (~1,200 fewer
  thread creations an hour); the technical-integrity snapshot and the setup
  tracker's ten CSV exports are memoized per file version; the tracker's spinbox
  and the Focus panel's RRS snapshot are coalesced; the Alert Center writes its
  preferences once instead of twice; the Focus chip's badge stylesheet moved
  inside the existing look guard; the hold expiry evaluates each alert exactly
  once per tick (one `hold_expired` event, not two - authorized); and the paused
  strategy loop waits 5 s instead of 0.5 s.

**53 new tests**, 50 of them proven to fail against the un-fixed code. No
packaging trigger: the sidecar is created at runtime in the diagnostics
directory, not bundled.

### 2026-08-31 — Desk snappiness packet 2: the next three measured stall causes

Packet 1 took the three largest. These are the next three, ranked by benefit,
against the same evidence: 8,008 GUI freezes / ~78 minutes in one day.
Memoization, threading and batching only - no detector, gate, alert, score or
statistic computes anything different, and no output, file format or push
changed.

**The Alert Center's minute tick stops redoing itself.** Three repetitions, all
over a ~105-symbol Focus set:

- **M5 bars were re-materialized per caller.** `bot.m5_chart_bars` rebuilds ~150
  dicts with six `float()` coercions each, and EIGHT timer-driven sites asked
  for the same symbol's bars per tick. Memoized in the panel - the source series
  belongs to BounceBot, which `ChartDataService` cannot see - keyed on the
  source list's identity plus its length and last stamp, with a strong reference
  held so `is` cannot be fooled by a recycled id. A replaced series is caught by
  identity, an in-place append by the stamp. `m5_chart_bars` is untouched and
  still produces every value. `_poll_any_bounce_watches` also read the same
  bars twice per watch; it reads once.
- **D1 reference levels were built ten times per symbol.** `d1_event_levels`
  sorts ~490 bars and builds 5d/20d extremes, three SMAs, an EMA15 recursion and
  the AVWAP band series; `_poll_focus_d1_interest` re-entered it once per kind
  with identical arguments. `evaluate_d1_event_watch` gained an optional
  `levels_cache` the caller owns and scopes to one symbol and one bar list for
  one tick - with `None` it is exactly the call it replaced, which is what makes
  the fast path behaviour-identical by construction. Keyed on (session, anchor),
  so the AVWAPE kinds keep their own entry.
- **~105 single-element prefetch tasks per minute** queued ahead of the snapshot
  for the chart the trader had just clicked, in a 2-thread pool. Symbols are
  queued and issued as ONE call on the next event-loop turn. Pool size and task
  priorities untouched.

**The startup heap is swept once, then frozen.** 6.5 of that day's ~78 minutes
of freeze were collections - gen-0 ~300 ms, full ~770 ms - and all cyclic
collection runs on the GUI thread by design, so every sweep is a freeze. Most of
what it walked can never be garbage: the widget tree, the theme, every import.
`main()` runs one `gc.collect(2)` then `gc.freeze()` after the window shows. The
order is the rule - freezing first would make startup GARBAGE immortal.
`_GuiGcController`'s cadence, idle waits and bounded deadlines are untouched.

**The journal's worst per-click costs.** Accepting a correction ran
`rebuild_trades()` -> `refresh_auto_tags()` -> a `json.loads` of the 1.08 GB
setup-tracker file plus a 73 MB CSV, synchronously behind the OK button.

- The retag runs on a worker through the new
  `ui/services/journal_rebuild_service.py`: single-flight (a second request is
  refused, not queued), buttons disabled with a "tagging..." state, results back
  on the GUI thread, and a failure shown rather than swallowed - the journal's
  loud-write rule. Both journal tabs share the service, so results are routed by
  token.
- `load_context_rows` caches the PROJECTED rows per source file on
  (`st_mtime_ns`, `st_size`); a missing or unstampable file is deliberately not
  cached so a later one is picked up. No new dependency - no `ijson`, which
  would be a packaging trigger.
- `list_trades` opened a fresh sqlite connection PER TRADE for the regime.
  `get_regimes_for_dates` reads the one-row-per-day table once and answers the
  same two questions in Python; `get_regime_for_date` is now that method with a
  list of one, so they cannot drift.
- The filter header debounces at 250 ms through `SignalCoalescer`.

**27 new tests**, 29 of the 31 assertions-of-change proven to fail against the
un-fixed code; the rest are documentation of behaviour that must not be lost and
say so. Not done, and named: the hold-expiry double `survives()` evaluation -
the packet allowed a per-tick verdict cache only, and `survives()` has side
effects, so whether the second call's effects are redundant is a behaviour
question rather than a memoization one. Left for packet 3.

No packaging trigger: no new dependency, no new non-`.py` asset, no new
top-level package (the new service is inside the already-collected
`scripts/ui/services/`), no new dynamic import.

### 2026-08-31 — Desk snappiness packet 1: the three largest measured stall causes

A read-only performance audit of the desk's own stall log
(`ui_stalls.jsonl`, 2026-08-31: 8,008 stalls, ~78 minutes of GUI-thread freeze in
one day) named three causes; the trader authorized exactly these fixes. All three
are caching/cadence changes only — no detector, scoring, alert, output, file
format, or push behavior differs, and every cached result is byte-identical to
the uncached parse for unchanged inputs. The `(st_mtime_ns, st_size)` stamp
template is `review_events.load_review_events`; both stamps because an append
inside one filesystem timestamp tick still moves the byte count.

**Item 1 — the Health audit stops re-parsing unchanged evidence.**
`_outcome_claim_coverage_check` re-parsed the 269 MB, 294k-row
`intraday_bounce_outcomes.csv` on every 15 s audit pass (measured 2.29 s each),
and the two shadow checks re-streamed both shadow JSONLs with no mtime guard.
The CSV check now caches its finished check dict on the file stamp
(`operations_audit.py`), and `shadow_log_audit.scan_shadow_log` caches its scan
per (profile, path, stamp, market_date, reconcile date) — `now` deliberately
outside the key, so a future-stamped row in an unchanged file stays counted
until the file moves. Error results are never cached. `HealthPanel` keeps its
15 s cadence only while the page is showing; hidden it ticks at 120 s — the
timer never stops, so the shell's status chip keeps updating, and returning to
the page refreshes immediately. Tests: `tests/test_health_audit_caching.py`
(parse-once, append-invalidates, error-not-cached, market-date-in-key) plus a
cadence test in `tests/test_qt_health_panel.py`; all verified failing pre-fix.

**Item 2 — column auto-sizing is bounded, and the Industry Board stops its
no-op refresh.** `measure_column_widths` (`data_table.py`) ran
`resizeColumnsToContents` unbounded — 9.6 minutes and single stalls of 85 s on
2026-08-31. It now applies `setResizeContentsPrecision(200)` first; Qt gotcha:
`sizeHintForColumn` walks rows bounded by the VERTICAL header's precision, so
that header is the one that caps this call (the horizontal gets the same cap
for any section still in ResizeToContents mode). Tables under 200 rows measure
identically. `IndustryBoardService.refresh_if_due` skips the `snapshotChanged`
emit when `snapshot_id` is unchanged (the due-check still runs), and
`IndustryPanel._on_snapshot_changed` early-returns on an id it already rendered
(belt-and-braces). The two headers stuck in `ResizeToContents` —
`price_alerts_panel.py` and `health_panel.py`'s `_table` — are now Interactive
with one bounded fit when rows land. Known small tradeoff, accepted by the
packet: the Industry Board's reviewed-today badges now refresh when the CSVs
move (or on manual Reload) rather than every 60 s. Tests:
`tests/test_desk_snappiness_tables.py`.

**Item 3 — the Auto Pilot status row stops re-reading files on the GUI
thread.** `status_snapshot()` did 2 watchlist + 2 auto-watchlist reads and 2
state-JSON parses per call, driven by a 5 s panel timer with no visibility
check and called twice back-to-back in the 30 s tick — most of the 10 minutes
charged to `watchlist_utils.py:33` and 3.9 minutes to `project_paths.py:165`.
Each file-backed piece is memoized on the file stamp
(`_memoized_file_read` in `autopilot_service.py`; values copied out), the tick
computes one snapshot and reuses it for the heartbeat and the emit, the
panel's 5 s slot early-returns while hidden (timer keeps running; the
service's 30 s `statusChanged` still lands), and `_apply_status` restyles a
label only when its text/tone changed (five unconditional `setStyleSheet`
calls per refresh before). Only these authorized pieces of
`autopilot_service.py` were touched — no alert or push code. Tests:
`tests/test_autopilot_status_snapshot_caching.py`; 5 of 7 verified failing
pre-fix (the two invalidation tests pass either way, by design).

**What should now go quiet in `ui_stalls.jsonl`** (the next desk session is
the proof): `data_table.py:170`, `watchlist_utils.py:33`,
`project_paths.py:165`, and the operations-audit CSV parse. The stall watchdog
stays ON — it is the before/after instrument and still owed for lockup gate
#19. Also fixed en route: `test_refresh_never_blocks_the_gui_thread...` leaked
a 0.5 s audit thread (construction's `singleShot(0, refresh)` firing after
`deleteLater` without `shutdown`), which failed the process-wide thread sweep
in any non-alphabetical file order.

### 2026-08-31 — The Strength Board moves into the Desk's Strength window

Trader: *"The Strength Board tab is good but it really should be modified to fit in
the 'strength' window in the trading desk — either integrated directly or be
positioned below it."* Positioned below it, and the left-nav page is removed.

**Where it is.** A `CollapsibleSection` (new, `ui/widgets/collapsible_section.py`)
under `FocusStrengthBoard` in the Alert Center's alert column, hosted through
`AlertCenterPanel.attach_strength_board`. `MainWindow` still builds and owns the one
`StrengthBoardService` — one timer, one single-flight fetch, one 15-minute cadence —
and now also shuts it down, which nothing did before: the service was parented to the
window but absent from the panel shutdown loop, so its timer outlived the close.

**What did not change**, pinned by `tests/test_qt_strength_board_in_the_desk.py`
rather than by prose: zero IB traffic (asserted over the AST of all three
strength-path modules, so an `ibapi` import or an `EClient`/`reqHistoricalData` name
fails it); the adoption gate re-run at click time on the row's own numbers; and one
service with one timer, measured by driving that timer and counting fetch attempts.

**Width was the constraint.** The alert column has a 360 px floor and everything left
of it is chart, so the section must never be why the charts get narrower. Four
measurements shaped it: the section header demanded 315 px (a `QToolButton` asks for
its whole label) and is now Ignored horizontally with elided text; the board demands
270 px and is hosted in a `QScrollArea` so that minimum stops there instead of
reaching the desk splitter; the status label demanded 434 px and now wraps, because
it carries failure reasons and can be long; and "Add all shown" (208 px) became "Add
all" (124 px) with the tooltip unchanged. The section also **starts closed**, so by
default it costs one header row. The two sides stack **vertically** — side by side
was right for a full-width page and unreadable in a column.

**A row click charts in the review pane, not a popup.** Same day, second pass:
*"when I click on a stock in this M5 strength board it should come up on the Visual
chart review in the trading desk."* The popup was the right answer while the board
was a page elsewhere; sharing a column with the pane, it is a window in the way. The
click goes through `chart_symbol` — the **lookup box's** door — and deliberately not
through `_enqueue_review_alert`, the **scanner's** door, which would have been wrong
four ways for a click: it drops everything in AWAY, drops parked symbols, diverts M5
alerts to the alert bar instead of the chart, and can hide a row behind movers-only.
A name the trader clicked must appear. It charts as a `MANUAL_CHART` (muted, not red
— nothing fired, the trader was looking), never enters the alert feed, un-ignores a
"not today" symbol exactly as typing one does, and carries its side, because a short
charted as a plain `WATCH` reads as the wrong thesis. `symbolActivated` is now
`(symbol, side)`; `chart_symbol` grew optional `side`/`origin` whose defaults
reproduce the lookup box exactly.

**The RS/RW half retired with the page.** It was added 2026-08-21 so the two reads
could be compared without flipping **pages**; the Alert Center's own RS/RW Board tab
is now one tab-click away in the **same column**, so keeping it would have been two
views of one payload six inches apart. The tape, its owner, the `rrsSnapshotChanged`
signal and that tab are untouched — one listener retired, nothing else moved. The arm
bar did not move either, and a test asserts it stays put when the section opens.

Page removal touched every site that tracks pages: the single `PAGE_SPECS` list (the
structure the 2026-08 nav bug forced into existence, which is why this is one line
rather than three) and the two test files that enumerate nav labels. The module stays
in the tree inside an already-collected package, so `packaging/tradingbotv3.spec`
needed no change and the spec-drift test stays green at 17.

Verified: `pytest tests/ -q` **5571 passed, 72 subtests** (was 5554) · smoke **7/7** ·
source `--selftest` **73/73** · spec-drift **17**. Live gate owed: one desk session
where the trader opens the section, reads the board in the column, clicks a row onto
the review chart and adds a name from it — plus a judgement on the vertical stack.

### 2026-08-31 - "I liked it and passed": the day-trade pass, under the note

**Trader-directed, authorized in chat 2026-08-31.** Branch
`claude/daytrade-pass-reasons`.

*"Many times I really like this stock for a daytrade but it has this ONE issue"*
- and the trader passes. The capture window could record a veto (this chart is
not for today), a like, or a note; it had nowhere to put the far more common
judgement of a name that WAS tradeable but for one thing. It does now.

**A new decision kind, not a wider veto.** `EVENT_PASS` (`"pass"`) joins the
schema-v1 event types, and its reasons come from a NEW vocabulary family,
`ui/annotations/vocabularies/pass_reasons_v1.json`, carrying the five reasons
the trader listed in their own words: Poor market conditions, Low rvol,
LRSI/SMI incongruency, Incoming Horizontal, Other incoming S/R. Extending the
veto vocabulary instead would have restamped `vocab_version` across veto
cohorts that are already accruing forward returns, for two lists that answer
different questions. `ui/annotations/vocabulary.py` now loads any family
(`load_vocabulary` / `load_pass_vocabulary`, `available_versions`) with the
identical fail-closed validation the veto list has always had, plus one new
check: a file must declare the `vocabulary_id` its filename claims.

**A pass never retires the chart.** It is note-shaped: written about the chart
the trader is still reading. Both hosts' `_on_captured` key on veto and like
alone, so nothing had to learn a new exception.

**Several reasons at once, in vocabulary order.** Ticking is multi-select by
instruction. `_clean_pass_codes` dedupes and reorders into vocabulary order, so
two passes citing the same two reasons compare equal a year from now. The note
field is the one already in that section - a pass is a note with the reason
ticked - and it stays optional.

**The chart rides along when the desk already has it, and is never fetched.**
`ui/annotations/pass_bars.py` writes one session of cached M5 bars to
`trader_annotation_bars/<event_id>.json` and the row carries `m5_bars_ref`,
`m5_bar_count` and the first/last bar stamps. A sidecar rather than an inline
array because one session is ~78 bars and far past the store's 4096-byte
single-write cap, and that cap is what keeps a torn tail costing exactly one
row. Sidecar first, row second, so a reference in the stream always has a file
behind it. The bars come from a host-supplied provider reading what the pane
already DREW (`CaptureRail.set_m5_bars_provider` <- `SymbolSnapshotWidget.cached_m5_bars`),
wired on all three capture hosts; nothing cached, or a provider that raises,
costs the attachment and never the row - the trader's own fallback was *"just
store the exact timestamp"*, and every row carries a zoned one.

**Keyboard.** Alt+P focuses the tick list and 1-5 toggle, scoped to the box
that holds only the checkboxes so a digit typed into the note above stays a
digit. `action_shortcuts()` gained the pair, so the Alert Center's panel-scope
rebinding picked it up without a second list; the rail still binds nothing its
host owns.

**Boundaries held.** Analysis-only evidence: no mute, no suppression field, no
score, no gate, no alert, no watchlist or Focus write. Deliberately NOT
changed, and DECIDED rather than pending (trader, 2026-08-31):
`pick_feedback._ANNOTATION_DECISIONS` still lists `veto`/`like_claim`/`note`,
so a pass does NOT mark a symbol "Reviewed today" - *"that flag feeds the
scanner report and several badges. Making a pass count as reviewed touches
scanner-side code, so it should be its own small job if you want it."* A test
pins it. The other question closed the same way: a pass never retires the chart
and needs no option - *"if you pass AND want the chart gone, just hit veto
after. You get both behaviors without a new rule."*

Verified: `pytest tests/ -q` 5554 passed / 72 subtests; source `--selftest`
73/73 (the pass vocabulary is its own bundled-asset check).

### 2026-08-31 — Theta: the floor is a percent of the strike, and support ranks first

**Trader-directed, Phase 0.11**, built from `docs/prompts/THETA_PREMIUM_OPUS_PROMPT.md`
on `claude/theta-premium`. The report was handing over ~$0.25 credits with
untradeable spreads. Four causes, all fixed:

- **The target WAS $0.25.** `THETA_PUT_TARGET_TOTAL_CREDIT` (100) / 100 /
  `THETA_PUT_MAX_CONTRACTS` (4). Flat dollars: 0.125% of a $200 strike, 1.25% of a
  $20 one. Now `theta_put_credit_floors(strike)` is the single decider - 1.0% of
  the strike recommended, 0.5% cusp, $0.40/contract absolute floor underneath - and
  a quote under both LEAVES the report. Trader: *"on a 200 dollar stock I'd want at
  least 1 dollar, ideally 2."* The $100/4-contract framing survives as display info.
- **The final sort preferred the cheapest option.** Its key was
  `(status, strike ASCENDING, ...)`, so the deepest-OTM qualifying strike won every
  time. Now: tier -> major SMAs above the strike -> support quality -> yield per
  market day -> spread. Trader: *"#1 priority is still areas of support."* Premium
  is a percent per market day, which replaces the flat DTE penalty on this path
  (PCS keeps it) and stops punishing a longer expiry that pays for the wait.
- **A wide spread was a soft penalty capped at 18 points**, so past ~150% every
  market cost the same and a catastrophic spread stopped sinking. The penalty is
  now monotonic and uncapped - and still never a block, because *"spreads are a
  spectrum ... within reason"*. An unmeasurable spread is treated as wide, not free.
- **The quote budget was spent in `base_score` order**, a trend ranking that says
  nothing about premium, so dead-vol names burned quotes rich ones never got. Now
  `thetalongs.txt` first (the trader's own list, R9.4), then estimated premium
  capacity from ATR% - no new network call, never a filter, unmeasurable sorts last
  rather than out - then `base_score`. IB pacing constants are untouched.

Also: credit spreads reach **15 market days** (*"3 weeks for credit spread"*);
sold puts stay at 10. The report emits one machine-readable `premium=` line per
quoted sold put (credit %, yield/week, spread %, source, SMA-above-strike with the
2+ boost marked), the extractor reads every field back, and the Qt theta table
gained the four matching columns - blank, never zero, for a row with no quote.

**26 tests**, 14 new plus rewrites, each proven to fail against the un-fixed code;
four of them are documentation of preserved behaviour and say so. Two existing
tests were deliberately reversed - their old rule (a sub-floor quote kept as
`below_target`, the deepest viable strike winning) is exactly what this packet
removes. Eligibility (>= 3 supports, >= 1 major SMA, earnings buffer) and R9.4
`theta_side` semantics are unchanged, and nothing here executes anything.

**And then the spreads too.** The one open question - whether the floor should
apply to the PCS short leg - was put to the trader with its arithmetic and
answered: *"Yes it should scale with price of the underlying."* The credit/width
ratio does not scale because the width is capped at 10 points at any price, so
the 20% target credit stops growing at $2.00: 1.36% of a $37 short strike, 0.31%
of a $644 one. `theta_pcs_credit_floor` is now a hard minimum of 0.5% of the
short strike (or $0.40), with the ratio still deciding the tier above it. The
1.0% recommended percent is deliberately not applied to spreads - it would be a
64% credit/width bar on an expensive name, which deletes rather than ranks.
Expensive spreads will mostly disappear unless their credit really scales; the
lever to bring them back is the WIDTH cap, not the floor, and that changes
capital at risk per contract so it was not touched.

### 2026-08-31 — The linter was configured but never installed, and it was hiding four bugs

**Trader-directed** ("ok install ruff then if you need it"), after the merge.

`ruff` was named in `CLAUDE.md`'s stack, declared in `requirements-dev.txt` and
configured in `pyproject.toml` with a deliberately narrow defect-class select
(`E9`, `F63`, `F7`, `F82`, `F401`) - and was **not installed in the desk `.venv`**
and not pinned in `constraints.txt`. The first run it has ever had here returned
1,703 findings.

- **1,591 of them are noise from four legacy Tk shims.** `master_avwap_lib/gui.py`,
  `master_avwap_lib/runner.py`, `bounce_bot_lib/gui.py` and the `gui_app/` package
  pull their names out of `legacy` at import time, so a static reader cannot see a
  single one. They join the two `legacy.py` files already in `extend-exclude`,
  with the reason written down. That leaves **75**.
- **Four real defects, all fixed, all with fail-before-fix proofs where testable:**
  - `operations_audit.py` called `logging.exception(...)` in **three**
    `except Exception:` handlers whose comment reads *"health must never take the
    audit down"* — and never imported `logging`, so each one raised `NameError`
    out of System Health at exactly the moment its guard was supposed to hold.
    All three carry `# pragma: no cover`, which is why nothing noticed.
    `tests/test_operations_audit_never_raises.py` drives each into an exception
    and asserts UNKNOWN or an empty note; 4 of its 7 fail without the import.
  - `journal_tab.py` built its Questrade token-failure dialog as
    `lambda: ...f"{exc}"`. Python deletes `exc` at the end of the except block and
    the lambda runs later on the Tk loop, so the **error dialog itself** raised
    `NameError`. The message is now bound at raise time.
  - Two imports nothing had used in `ui/app.py` (`pathlib.Path`, `QtCore.QProcess`)
    and one in `tests/test_trader_annotations.py`.
- **The one `F821` in an alert file was asked about, then fixed** (trader: "yes").
  `ui/panels/alert_center_panel.py` annotated
  `self.strength_board: "StrengthBoardPanel | None"` with a name imported only
  inside `attach_strength_board`; it now carries a `TYPE_CHECKING` import, which
  is never evaluated at runtime. The lazy import that keeps the board's module
  out of the panel's import graph is untouched, and so is every alert path.
- **The 74 unused imports were then swept too** (trader: "yes clean that up"),
  across 52 files. Two guarded availability probes in `test_qt_alert_center.py`
  kept their imports under `# noqa: F401` with the reason - removing them would
  have removed the probe. The sweep broke one thing and the suite caught it:
  `technical_integrity` imports `row_capture_mode` and **re-exports** it to
  `regime_collection_audit` and `test_ti_chain_backfill`, so removing it failed 8
  tests; it is back, marked `# noqa: F401` and commented as a re-export. A
  multi-line-aware scan over all 105 removed names found no other re-export, and
  an import sweep over all 331 modules passes. **`ruff check .` now reports
  `All checks passed`.**


**Trader-directed, authorized in chat 2026-08-31.** Branch `claude/swing-favorites`.

*"At the end of the day I have a list of my top swing targets. I want a place to
put them in so the bot knows my personal favourite picks. They will usually
become focus picks too but these ones get special standing because I picked them
by hand... put it at the very bottom of the M5 alerts tab, the tab is so long and
I never use all of it. And the bot should scan the journal to know which ones I
actually took."*

Deliberately **not** the Master AVWAP like/dislike capture, which already exists
and records a verdict on a row the bot proposed. This records a name the trader
brought in themselves.

- **New:** `scripts/swing_favorites.py` (plain-Python append-only store and the
  session replay), `scripts/ui/services/swing_favorites_service.py` (the two
  writes plus the journal join on a worker thread),
  `scripts/ui/widgets/swing_favorites_bar.py` (the strip), and
  `project_paths.SWING_FAVORITES_FILE` → `swing_favorites.jsonl` in the shared
  home, the same storage class as `pick_feedback.jsonl` and
  `trader_annotations.jsonl`.
- **Two writes per add, in a fixed order.** The swing Focus write-through goes
  first, through the existing store, and must not fail — it is the thing the
  trader asked for. The evidence row goes second and a failed append is swallowed
  with a status line, because an evidence store is never allowed to cost the thing
  it records. Nothing in the chain calls `mark_auto_adopted`: a hand-vetted pick
  carrying an auto marker would be reachable by "Not today" and the desync repair,
  which is the removal path that marker exists to keep off the trader's own names.
- **A removal is a retraction.** The add row stays; a `remove` row follows it. The
  live list is a replay of one session in file order, so a re-add returns to the
  end, where the trader just put it. Prior sessions are untouched.
- **The "took" badge is display only** and joins against the TRADE journal, not the
  Market Journal. Symbol match, "opened on or after the pick date", bounded to a
  10-day window, run on a worker thread, and **silent when the journal would have
  to be created or migrated to answer** — a display badge must never be the thing
  that triggers a schema migration. No rate, no grade, no statistic.
- **Where it lives.** The M5 alerts surface is a tab in tabs mode and the tall left
  column in workspace mode, and the trader's saved `qt_workspace_mode` is
  `workspace` — so the alert bar and the strip now share one host
  (`TradingDeskPanel.m5_column`) that both modes mount, and the strip is the bottom
  of it either way. `M5AlertBar` and every alert routing path are unchanged. The
  pinned "bar is left of the chart" test now asserts the column holds the bar first.
- **The split is the trader's to drag** (same day, second pass: *"the tab needs to
  be resizable relative to the M5 alerts tab, I should be able to drag it up to see
  more"*). `m5_column` is a vertical `QSplitter` with its own settings key
  (`qt_m5_column_split_sizes_v1`), so this drag and the desk's three-column drag
  never overwrite each other, and `setChildrenCollapsible(False)` because a strip
  dragged to nothing is one the trader cannot find again. The chip area gained a
  floor and **lost its ceiling** — a maximum height would have made the drag do
  nothing past it.
- **Copy / Paste, the TC2000 seam.** Copy puts the day's tickers on the clipboard
  one per line, each once, in list order; Paste adds every ticker on the clipboard
  on the side the toggle is showing. Same idiom as the M5 alert bar's "Copy all".
- **They grade as their own cohort.** The Focus like-origin is `vetted`, not
  `manual`, so the human-focus tracker's existing 1/3/5/10-session grader files
  them under `human_focus_swing_vetted` instead of mixing them with every other
  hand-typed swing name — which is what makes "how do my hand-picked swings do
  against the bot's?" answerable at all. Deliberately NOT built: the file is not in
  `ai_summary`'s overnight evidence pack and nothing joins it to per-setup journal
  statistics. Both are additive and unasked.
- Chips are diffed, never rebuilt, and every variant (side colour, the "took" mark)
  is a dynamic property answered by `theme.qss` — no per-widget stylesheet.
- **69 new tests** (`tests/test_swing_favorites.py`,
  `tests/test_qt_swing_favorites.py`). No phone push; nothing reaches a detector,
  score, alert, watchlist ranking or `review_policy.json`.
- **Known limit, stated:** the strip shows the CURRENT session's list, so a pick
  typed after the close is shown that evening and the "took" badge for it can only
  ever reflect that same session. Carrying a pick forward to the next session is a
  product decision the trader has not been asked yet.

### 2026-08-31 — One Focus add must not repaint the desk five times over

`IMPLEMENTED`, `GREEN`, **live gate owed**. Branch `claude/focus-refresh-storm`.

**The measurement.** 07:37-07:53 on 2026-08-31: ~500 s of GUI-thread blockage in a
16-minute session. Since 07:45 the UI was blocked 216 s in 5.5 min; 07:50-07:52 it
was 113 s in 2.3 min (~80% frozen). Single stalls of 44.3 s, 15.9 s and 15.2 s;
Windows reported Not Responding and the trader killed the desk twice, each restart
re-running the 07:30 swing scan. Memory was fine (~2 GB WS) — this is not the
2026-08-27 warehouse bug.

**The cause.** At 07:41:58-07:42:11 the Alert Center drain adopted **45 staged picks
into M5 Focus one at a time**, ~300 ms apart (`focus_auto_picks.json`, all
`adopted_at` in that window). `FocusPickStore.add()` notifies per add — correctly,
several surfaces depend on it — but every one of five listeners treated a single add
as "rebuild everything":

| Listener | What one add cost |
|---|---|
| `FocusPicksPanel._on_focus_changed` | 4 full editor rebuilds + a `pick_feedback` read + a forced snapshot WRITE |
| `AlertCenterPanel._rebuild_feed` | both feeds destroyed and reconstructed — up to 350 widget trees, each with its own stylesheet |
| `MasterAvwapPanel` | a full setups-viewport repaint through `SetupTableDelegate` (the hottest stack in the stall log, ~300 samples across paint lines 78-152) |
| `FocusStrengthBoard._render` | the whole board rebuilt as HTML and re-parsed by `setHtml` |
| `PriceAlertBoard._refresh_symbol_choices` | the symbol combo cleared and refilled |

Times 45, in 13 seconds. The 15.2 s stall charged to `focus_picks_panel.py:441`
landed at the end of that burst.

**The fix, and where it does NOT live.** `focusChanged` still fires once per store
mutation — other listeners rely on it, and moving the coalescing into the store
would change a contract to fix a symptom. The coalescing lives at each LISTENER.

`ui.timer_utils.SignalCoalescer` is a **leading-edge window with a trailing fire**:
the first request opens a 200 ms window, later requests fold into it and
deliberately **do not restart** it. A synchronous drain loop therefore lands whole
inside one window and produces exactly one reaction; a sustained trickle fires on a
fixed cadence instead of being starved, which is what a plain restart-on-signal
debounce would do. A reaction that raises cannot leave the coalescer armed. 200 ms
is the trader's ceiling, not a target — a hand-typed ticker still appears instantly.

**Three more defects in the same chain.**

- `FocusSideEditor.refresh()` documented itself as a diff and still **emptied the
  flow layout and re-added every chip on every call**, even when nothing had
  changed — 90 layout operations on a 45-name board to change nothing. The
  unchanged case now performs **zero** layout work and only hands each chip its
  state; arrivals, departures and reorders are index-precise. `FlowLayout` grew
  `insertWidget`, because `QLayout` has no generic insert and its absence is why
  the teardown existed.
- `record_bounce_alert` lit ONE chip's badge by rebuilding four editors and
  re-reading the feedback file. It now updates only the matching chip.
  `_bounce_state` is still written first, so a name that joins Focus after its
  alert picks the badge up when its chip is built.
- The DESK drain adopts at most **`AUTO_ADOPT_BATCH_LIMIT` = 10** picks per
  30-second cycle (trader-approved 2026-08-31: *"cap the auto-adopt batch and slow
  the redraws"*). **Pacing, never policy** — the freshness gate, the flip barrier,
  ownership markers and AWAY/EVENING's refusal are untouched and upstream of it. A
  deferred pick is **not** marked seen, so the next cycle finds it exactly as this
  one did; the cap counts adoptions rather than iterations, so a day the gate
  refuses most of the queue still adopts a full batch of what qualifies. **No pick
  is ever dropped** — a cap that withheld one would be the suppression field this
  chain deliberately does not have. A 45-pick morning now finishes in ~2.5 minutes
  of background ticks instead of 13 seconds of frozen desk.

The `alert_center_panel.py` feed-rebuild coalescing was approved separately under
the file-scoped ask-first rule; only the TRIGGER is coalesced, and which alerts
pass the feed gate, their order, the repetition fold and the digest are all decided
inside `_rebuild_feed` and are unchanged.

**Deliberately not touched.** The GUI-thread GC controller (`app.py`
`GcController` / `install_gui_thread_gc`): its ~600 ms young sweeps were a
*symptom* of this churn, and its delay-never-cancel and GUI-thread-only invariants
are load-bearing. No detector, scoring, gating or adoption-gate logic changed.

**Verification.** 29 new tests, every one written to fail against the old behaviour
first and watched failing. Full suite **5456 passed, 72 subtests** (29 of those
tests are new; the pre-change local count was 5427), smoke **7/7**, source
`--selftest` **72/72**. No packaging trigger: no new dependency, no new asset, no
new top-level package, no new dynamic import.

**Live gate owed:** one DESK session on a directional morning where the drain
stages a large batch — the desk stays responsive, every staged pick reaches M5
Focus across successive ticks, and `ui_stalls.jsonl` no longer charges seconds to
`focus_picks_panel.py` or `setup_delegate.py`.


### 2026-08-28 — The tax number is the broker's, never ours

`IMPLEMENTED`, `GREEN`, **live gate owed**. Trader decision: *"Statement is
source of truth for final pnl/tax purposes"* — a stronger rule than the day-level
authority landed earlier, and one that needed its own answer.

Every other P&L in the journal is **recomputed** (average-cost matching, price ×
quantity), which is what makes per-setup statistics possible and is also
arithmetic of our own: it drifts from Questrade's cent-rounded figures by
**−$0.2386 on $5,298.81** across the trader's year. Immaterial for deciding what
to trade; not the number for a return.

`journal_tax_report` recomputes nothing. It sums `raw_executions.net_amount` —
the broker's own statement of each fill's cash — and for a **flat** position that
sum *is* the realised P&L, so no cost-basis model is needed or used. One
normalisation was required first: the IBKR file states that figure in the base
currency, so the importer now divides by the row's implied rate before storing,
keeping the base figure as evidence. `net_amount` means one thing store-wide or
the sum is meaningless.

**What it refuses to report is the point.** Open positions, positions whose
opening fill was invented (`SYNTHETIC_OPEN`), and any position with a fill
lacking a stated amount are excluded — and named, with the reason, so the trader
knows which file would fix them. Voided executions never reach a total. CAD
converts per fill at the booked BoC rate; an unbooked date withholds that
position's CAD rather than guessing. Accounts stay separate with their tax
status, currencies are never added, and a position spanning the year end is
reported whole.

Cross-check on the trader's real data across both brokers: broker **$8,219.81**
vs the journal's recomputed **$8,220.05**, difference **−$0.2385** — precisely the
known Questrade rounding, with IBKR exact.

Verification: **5419 passed / 72 subtests / 6 skipped**, 5402 → 5419. Smoke 7/7,
selftest 72/72, spec drift 17, ruff clean.

### 2026-08-28 — The file wins on money, the sync keeps the clock

`IMPLEMENTED`, `GREEN`, **live gate owed**. Trader decision, taken after the cost
of the blunt version was measured: *"these should be sources of truth moreso than
the auto input IMO"* → **money only**.

Neither broker's downloadable file carries a time of day. Letting a file take over
every day it covers would have discarded the only intraday timestamps the journal
has — every session bucket and every entry-time tag built on them. So the rule is
split by what each source is actually good for: the **sync keeps** a day the two
agree on, the **file takes** a day they do not.

Agreement is measured in **cash, per (account, day)** — a trade can span days, so
a day's P&L is not even defined, while its cash impact is. That cash is COMPUTED
(`sign × qty × price × multiplier − commission − fees`) rather than read off the
file's Gross/Net column, because Questrade reports those in the trade's currency
and IBKR in the base currency, so the columns are not comparable to each other.
Tolerance is **per fill**, not flat: Questrade rounds each row to the cent, so a
busy day drifts more than a quiet one.

Taking a day over is **append-only** — I3 forbids deleting a broker row, so the
sync's executions are retired with `VOID_EXECUTION` adjustments naming the day,
both cash figures and the difference. They stay on disk and a superseding record
undoes it. A day the file does not mention is a gap, not a disagreement, and is
never touched.

Proven against the trader's real 2025–26 export with a simulated August sync, one
day deliberately given half its fills: **18 shared days, 17 agreed and kept their
real timestamps, the crippled day taken over on a $3,116.49 difference** (3 voided,
5 written), 15 August trades still carrying a real entry time.

Verification: **5402 passed / 72 subtests / 6 skipped**, 5385 → 5402. Smoke 7/7,
selftest 72/72, spec drift 17, ruff clean.

### 2026-08-28 — IBKR's file, and the commission sign that was costing money

`IMPLEMENTED`, `GREEN`, **live gate owed**. Trader direction: *"we need IB
integration as well. the auto import works well but we would want to manually
input a file as well."*

**A separate reader, because three things differ and each produces a plausible
wrong number.** IBKR ships a SECTIONED csv — a header per section — so a plain
`DictReader` misaligns every table after the first. Its `Price` is USD while
`Gross`/`Net` are CAD, so a passed-through row computes a USD gross and subtracts
a CAD commission; costs are converted by the rate each row implies,
`|Gross| / |qty × price × multiplier|`, which ran **1.35530–1.45270** across 608
rows — the USD/CAD band, and the check that the reading is right. That rate is
recorded as evidence and **never booked into `fx_rates`**, which is BoC-only: a
broker's internal rate is not a tax rate. The option multiplier sits inside that
denominator; without it the rate comes out 100× too large.

**Account numbers arrive masked** (`U***2524`). A mask cannot be an identity —
the same account through Flex carries its full number — so it is unmasked only
when exactly one known account fits, the filename is another candidate rather
than an override, and an unresolved mask is reported rather than guessed.

**The commission sign.** `upsert_executions` and the assembly path used to
`abs()` commission and fees. Every importer already normalises a charge to a
positive cost, so removing it is a no-op for Questrade, Flex, the socket, CSV and
manual rows — but **18 of 609** IBKR fills carry a commission CREDIT, and `abs()`
turned each rebate into a charge, overstating cost by twice the credit. That one
sign was the **entire** $2.17 by which the IB file and the journal disagreed.
With it fixed IB reconciles to **−0.0000 across 150 closed symbols**, commission
equal to four decimals — exact, where Questrade is off by cents, because IB
writes full-precision amounts and Questrade rounds. Questrade's reconciliation
was re-measured and is unmoved.

One Health-tab button serves both brokers and reads the broker from the file's
contents, never its name.

Verification: **5385 passed / 72 subtests / 6 skipped**, 5361 → 5385. Smoke 7/7,
selftest 72/72, spec drift 17, ruff clean on the new files.

### 2026-08-28 — Statements that layer, a direction that is read rather than guessed, and the trader's own check

`IMPLEMENTED`, `GREEN`, **live gate owed**. Trader direction: *"lets add a
function to be able to take these files, and new ones throughout the year that
layer on top so that in the end I can totally manually calculate and demonstrate
my pnl and then we can compare it to the auto generated stuff."*

**Two defects the first statement build carried, both found by measuring rather
than reviewing.**

*The uid was positional.* It hashed the file's row index, so a January-to-December
export — the same January trades at different row positions — made **884 of 884**
real trades look new. Identity is now `fill_signature` plus an ordinal counted
within that signature. Proven on the trader's two real files: all 884 of the 2026
file recognised inside the 2025–26 file, and re-importing either in any order
leaves 1,516 executions and 202 cash rows unchanged.

*Direction was a coin flip.* A statement has no clock and lists a same-day round
trip SELL-first **227 times out of 227** — a sort, not a sequence — so the
assembler's uid tiebreak decided long vs short at random: **86 of 199**. Questrade
says it in the Description instead (`STOCK SHORT.`, `COVER SHORT.`), so `leg_rank`
orders each row by what it does to the position. That resolved all 227 — **169
long, 58 short** — and all 58 carried both markings.

**`reconcile_statement`** is the trader's own proof and writes nothing: for a
symbol whose quantities net to zero across the file, the sum of its Net Amount
column IS the realised P&L, and that is compared to what `rebuild_trades`
assembled. Open positions are excluded, not zeroed. Measured across both files:
statement **$5,298.81** vs journal **$5,299.05**, difference **−$0.2386** over 428
closed symbols, every symbol inside two cents, **commission exact at $713.68 both
ways**. It does not prove the parse — both sides read the same one.

Importing the 2025 file dropped NEEDS_REVIEW trades from **23 to 5**. Three days
hold both a short and a long in one symbol and are named rather than silently
blended.

Verification: **5361 passed / 72 subtests / 6 skipped**, 5349 → 5361. Smoke 7/7,
selftest 72/72, ruff clean on the new files, same two pre-existing font failures.

### 2026-08-28 — Reading a Questrade statement, for the days the API cannot reach

`IMPLEMENTED`, `GREEN`, **live gate owed** (the trader's own YTD file, imported
on the desk against their real journal). Trader-supplied file and direction:
*"i can easily get us yearly reports from questrade so long as we can process
these files."*

**Why.** The executions endpoint stops at a retention horizon — 2026-06-10 on
this desk. That is why 44 of the 45 `activities report trades…` days can never
be repaired by retrying, and it is the open trader decision R7 has carried since
2026-08-25. The portal's activity export does not stop there.

**What the real file measured**, and what every decision below follows from: 974
rows, 884 trades, 133 trading days, 2026-01-02 → 2026-08-27, both accounts, zero
unreadable rows, and `Net == Gross + Commission` on **every one of the 884 trade
rows to the cent**. So the single Commission column is the complete cost; `fees`
is written 0.0 rather than inventing a split the file does not contain.

**What a statement cannot say** shapes the module. No time of day (every row is
"12:00:00 AM"), so executions are written at midnight market-local and
`journal_trade_shape.is_date_only` refuses to name a session — a date-only round
trip is a `day_trade`, never a `scalp`. Fills are aggregated (some descriptions
say "AVG PRICE"). No execution id and no intraday sequence, so the statement's
own row order is preserved and carried into the surrogate uid; without it two
identical fills on one day hash to one uid and half the position vanishes.
Options carry a Questrade internal id in the Symbol column and the real contract
in the Description — parsed into an OCC symbol, which is what keeps the 100
multiplier and stops option P&L being understated a hundredfold.

**The rule that prevents double counting:** a statement never writes into a
(broker, account, day) that a richer source already covers. The two sources give
one fill different uids, so the upsert cannot see the duplicate; the day is
refused and the count reported. `.xlsx` is read with `zipfile` + `ElementTree`
rather than adding `openpyxl`, which would be packaging trigger 1.

**Measured drift, stated rather than found later.** `rebuild_trades` recomputes
gross P&L from price × quantity while Questrade books Gross Amount to the cent:
**−$0.1558 on $4,014.18 realised across 253 closed symbols**, worst symbol 1.2¢,
and commission matching exactly at $291.38 both ways. Making the assembler prefer
the broker's booked money is a change to the engine both brokers share and was
deliberately not made here.

Verification: **5349 passed / 72 subtests / 6 skipped**, 5326 → 5349, 23 added;
the same two pre-existing font-metric failures. Smoke 7/7 exit 0, source selftest
72/72 exit 0, spec drift 17 passed, ruff clean. No packaging trigger. No detector,
score, alert, watchlist, Focus or `review_policy.json` path is touched.

### 2026-08-28 — Auto-tagging that works on imported history, and the tools to adjust it

`IMPLEMENTED`, `GREEN`, **live gate owed** (one desk session tagging real trades).
Trader-directed, evaluating whether this journal can replace their TradesViz
subscription: *"i want auto tagging then I can come back and adjust."*

**The defect the ask exposed.** `AutoTagger` scores a trade by matching it against
the scanner's own output files. Those files hold the current lookback, so every
trade older than them scores nothing — `suggest_for_trade` returns `[]`, the
summary is written empty, and a year pulled from a broker statement arrives as one
undifferentiated untagged block. Auto-tagging was not broken; it had no inputs for
the case the trader was about to create.

**`scripts/journal_trade_shape.py`** is the second lane: hold bucket (counted in
SESSIONS, so a Friday-to-Monday hold is one night), entry session bucket, execution
shape from leg ROLES, and instrument — all from the trade's own row, no files, no
network, no scanner import. Three rules keep the tags safe to average: no tag is
ever derived from the OUTCOME (a `winners` bucket would post a 100% win rate and
explain nothing), anything unmeasurable emits NO tag, and a naive timestamp gets
market-local ATTACHED rather than an aware one stripped. Candidates order by LANE,
never confidence — shape tags carry 1.0 and would otherwise bury every setup match.

**Adjusting, which is the other half of the ask.** A tag filter on the SHARED
header, so one tag narrows the calendar, the equity curve and the fee totals too;
Analytics could already group BY tag and nothing could filter TO one. `distinct_tags`
counts the trader's lane apart from the machine's. `rename_tag` rewrites or retires
a tag across every trade that carries it — `setup_tags` only, because a derived tag
is re-computed on every refresh and the Manage-tags dialog refuses one rather than
accept a rename the next rebuild would undo. Accepting a suggestion now drops that
SUGGESTION from the queue: the 2026-08-24 reasoning that a tagged trade may still
deserve a second tag is unchanged, but a confirmed trade no longer re-proposes what
it was confirmed with — the mechanism behind 220 proposals against one annotation.

Verification, reading pytest's own exit code: **5326 passed / 72 subtests / 6
skipped**, 5268 → 5326, 58 added; two pre-existing failures are this Linux
container's font metrics and reproduce on a clean checkout. Smoke 7/7 exit 0,
source selftest 72/72 exit 0, spec drift 17 passed. No packaging trigger — the new
file is a module under `scripts/`, not a package, reached by a static import. No
detector, score, alert, watchlist, Focus or `review_policy.json` path is touched.

### 2026-08-28 — Reading the whole evidence pile in slices: 78,119 chars → 1,365,259

`IMPLEMENTED`, `GREEN`, **live gate owed** (tonight's 22:00 window is the first
unattended run). Trader-authorized: *"Can we just give it more time? Like hours to
complete its work then? And spoon feed it slowly so we don't run out of context?"*
Advisory layer only.

**The problem the budget work could not solve.** Raising the context to 64k and deriving
the budget took the summary from 10 of 22 sources to 17 of 22 with none unfunded, and
it still read **one tenth** of what exists: 1,365,259 characters of session evidence
against a prompt that can hold ~91,000. The packager spends that tenth *fairly* rather
than *well* — every source gets a share, so `setups.type_stats` contributed **3 of its
184 rows** and `setups.playbooks` 2 of 200. No further tuning fixes that; 96k crashes
the runner, so the ceiling is hardware.

**The trade.** `scripts/ai_jobs/map_reduce.py` cuts the evidence into slices that fit
comfortably, asks the model for findings from each, then hands back only the findings
and asks it to synthesize. **Every row of every source is read** — 46 slices over 17
sources, ~2.8 hours of a window that runs 22:00–06:00 and was using nine minutes of it.

What the module is careful about, each of which is a test:

- **A slice never passes for its whole source.** Every chunk carries `rows 41-80 of 184`
  in the content the model reads, and the package note tells it to describe only what is
  in front of it. Tables split by ROW (half a row is not evidence); text by window.
- **Citations stay real.** A map call is handed a package containing exactly one source,
  so the existing validator already forbids citing anything else. The synthesis gets
  `citable_aliases` for the ids that actually appear in the findings — so it can name
  the store a statement came from, and nothing that was not read.
- **A failed slice is counted and named** in the published `data_quality`, because a
  document synthesized from 44 of 46 slices is not the same document as one from 46.
- **A failed synthesis does not throw away hours of map work.** The findings are already
  validated and already cite real stores, so they are published *unsynthesized* — and
  the executive line says `UNSYNTHESIZED` in capitals, because a raw pile presented as a
  review would be the more dishonest of the two failures. Proven in the live validation
  run, where the synthesis pass failed and the findings survived.
- **Every slice failing raises** rather than publishing an empty review.

**Two things this exposed, both fixed:**

1. **The truncation tripwire fired on a healthy request.** The findings package is the
   model's own prose and tokenizes at **3.72 chars/token**, where dense JSON evidence
   measures 2.06–2.23 — so an estimate calibrated for one is wrong for the other, and an
   8,325-char package estimated at 3,330 tokens against a truthful 2,235 was called
   sheared. The fix is the half of the check that needs no estimate: **truncation means
   the server clipped to its context, and a clip lands at the ceiling.** Both observed
   shears pinned within three tokens of half the window (6,147 of 12,288; 32,771 of
   65,536), so a prompt evaluated below `TRUNCATION_CLIP_FLOOR_RATIO = 0.45` of the
   window was not clipped, whatever the estimate says. All four real measurements are
   pinned as a regression test. The two pre-existing shear fixtures used 12 and 5
   tokens — values no clip of any context can produce — and were made faithful to the
   historical failure (2,048 context, 1,027 tokens) rather than the guard being loosened.
2. **The scheduled task could start a second copy of a three-hour job.** It fires every
   30 minutes for eight hours, and the ledger only records a row when a job *finishes* —
   harmless while every slot took minutes. `run_slots` now takes a machine-local lock
   (`local_writer_lock`, the same primitive the feature-history writer uses) and a second
   firing stands down cleanly. `local_writer_lock` reports "someone holds it" and "this
   box has no primitive" as the *same* exception and they want opposite answers, so they
   are told apart by the module's own sentence — with a test asserting that sentence
   still exists in `local_writer_lock.py`, so a rewording breaks a test rather than the
   guard. And the summary slot's window reservation, **20 minutes**, was the reservation
   for a job that now takes 170: `summary_reserve_minutes()` returns 200 in chunked mode,
   because a three-hour job launched with twenty minutes left runs into the open.

Off by default (`ai_local_map_reduce`), on for this desk. Tests: 20 in the new
`tests/test_ai_map_reduce.py`, plus the tripwire regression cases.


### 2026-08-28 — The local model was reading a third of its evidence: context 12k → 64k, budget derived

`IMPLEMENTED`, `GREEN`. Trader-authorized ("raise the context... use as much as you
want"). Advisory layer only; nothing here can reach a detector, score, alert, watchlist
or the review queue.

**What the review found.** With the endpoint back up, `ai_summary` stopped saying
"unreachable" and started saying the prompt had been **sheared** — and it had.
Measured against the desk's own model over prompts from 9 KB to 93 KB, the evidence
package tokenizes at **2.06–2.23 chars/token**, not the 3.0–3.5 the code assumed
in two separate places. The consequences compounded:

- the 22,000-char budget was derived in a comment as `7800 tokens × 3.0 chars = 23400`;
  at the real rate 7,800 tokens is ~16,400 chars, so **the default exceeded a
  12,288-token window by about a third from the day it was written**. It survived only
  while few sources were funded;
- on 2026-08-27 the package reached 17 usable sources, the prompt reached ~14,400
  tokens, and llama.cpp sheared it to half the window (6,147 tokens — the pin is
  visible as a constant across prompts of 28 KB, 37 KB, 51 KB and 93 KB);
- the tripwire caught it, but by a **2.7% margin**, because the same wrong constant
  understates the estimate it compares against.

**What changed.**

1. **The desk's model context went 12,288 → 65,536** (`gemma3:12b-tbv3ctx-64k`, built
   from the saved definition of the old tag with one parameter changed). Measured cost:
   **none worth counting — 8.1 GB loaded, still 100% on the iGPU**, because gemma3's
   sliding-window attention keeps the KV cache cheap. The rollback Modelfile is kept at
   `C:\TradingBotData\_tools\ollama\gemma3-12b-tbv3ctx.BEFORE-2026-08-28.Modelfile`;
   the old tag is untouched, so reverting is one settings change.
2. **The budget is now DERIVED, not remembered.**
   `local_evidence_budget_ceiling_chars()` subtracts generation and scaffold from the
   configured context, converts at the worst measured rate, and leaves retry headroom;
   `local_evidence_budget_chars()` can never return more than that however the setting
   is configured. A budget bigger than the model can read does not produce a bigger
   summary, it produces a silently sheared one — capping here means the packager
   degrades the way it was designed to instead. New setting `ai_local_context_tokens`
   (stock 12,288; the desk is set to 65,536) is what the ceiling is computed from.
3. **Two chars-per-token constants, deliberately different and never to be merged.**
   `_BUDGET_CHARS_PER_TOKEN = 2.0` sizes the budget and is pessimistic (small ratio →
   small budget); `_ESTIMATED_CHARS_PER_TOKEN = 2.5` (was 3.5) estimates what was sent
   and is conservative the other way (large ratio → small estimate → no false alarm).
   A test asserts they lean opposite ways, because merging them reintroduces the shear.
4. **The local request honours its caller's timeout** up to
   `LOCAL_REQUEST_TIMEOUT_CAP_SECONDS = 1800`; the cloud paths keep their 300s clamp. A
   hosted API silent for five minutes has failed; a local 12B is still working — at
   ~118 tok/s evaluating the prompt, the nightly package needs minutes before the first
   output token exists.

**Result, measured on the 2026-08-27 session.** `ai_summary` went from four consecutive
`degraded_no_narrative` runs to **`ok` in 343s**, with **17 of 22 sources usable and
zero unfunded** (it was 10 of 22 with 5 unfunded). The narrative now names real
candidates (NET, OII, NESR), a setup family (`bounce_combo`) and the regime, where the
2026-08-26 one managed "mixed results" and named nothing.

**Then the budget was taken as far as the hardware actually allows** (trader:
"let's take all the time we need... crank up the detail"). Four separate limiters
stack on this path — `MAX_ROWS`, `MAX_SOURCE_CHARS`, the per-scope weights and the
within-scope fair share — and the binding one for almost every source was the share of
the total budget, so the budget is where the work went.

- **96k context loads and then CRASHES under load.** `ollama ps` reported 8.0 GB and
  "100% GPU" at `num_ctx 98304`, and 128k refused outright — but a real 132 KB prompt
  killed the runner with `wsarecv: An existing connection was forcibly closed`. The
  reservation at load time says nothing about what happens when the KV cache actually
  fills. **65,536 is the working ceiling on this iGPU**, established by completing
  generations rather than by loading.
- **An over-long prompt is not an error.** At 64k, a 150,000-char prompt returned
  HTTP 200 and `prompt_tokens = 32,771` — exactly half the window plus three. It
  answered confidently from a prompt it had silently cut in half. That is the whole
  reason the tripwire exists, and it is why the budget is sized with a safety factor
  rather than pushed to the last token.
- **The ceiling formula was wrong a second time and is now measured.** The first
  correction allowed 1,000 tokens for the prompt envelope; the real overhead is
  **10–35%** on top of the evidence the budget counts (measured 24,000→32,203 chars,
  48,000→59,226, 96,000→111,568, 159,466→175,358). `_BUDGET_PROMPT_OVERHEAD = 1.35`
  takes the worst observed ratio. Setting the budget to `0` now means **derive it**,
  so raising the model's window is one setting and not two that can disagree.
- **Verified end to end, not calculated:** the derived 78,119-char budget produces a
  91,262-char prompt that the server tokenized at **44,344 tokens — 71% of the 62,036
  usable**, whole prompt read, no shear.
- **A per-symbol brief no longer shares the session's budget.** It cannot: measured
  ~60s per brief at 22,000 chars, so a normal night is 53 briefs in 55 minutes
  (2026-08-26) or 121 in two hours (2026-08-17), and the job already refuses to start
  with under 120 minutes left. The same package at the session budget is ~42,600 tokens
  instead of ~14,000 — three times the time per brief, which would put a 53-brief night
  past three hours and a 121-brief night past seven. `evidence_budget_for(...,
  per_item=True)` keeps briefs at the value every healthy night ran at, capped by the
  same context ceiling.

**What the extra evidence actually bought.** The 2026-08-27 summary ran in 567s and,
for the first time in any run on record, **"Strongest already-qualified candidates" is
not "No supported finding"** — it reads *"NET, OII, and NESR are highlighted as top long
candidates with high conviction and a 0.83R reward."* Per-source slices roughly doubled:
`daily.auto_report` 2,909 → 4,735 of 8,592 chars, `setups.type_stats` 1 → 3 of 184 rows,
`setups.current_tiers` 4 → 8 of 200, `setups.short_horizon` 5 → 8 of 26.

**Still thin, and this is the honest limit rather than a dial left unturned:** 14 of 17
sources are still shown in part, and the tabular ones are still 3-of-184-row slices. The
prompt is at 71% of a context window that cannot go higher on this hardware, and the
remaining share is divided between seven `setup_trackers` sources competing inside one
scope. Getting whole tables in needs a different model or a narrower scope selection,
not another number.

Tests: 9 new/updated in `tests/test_local_ai_provider.py`. The load-bearing one is
**`test_the_derived_budget_produces_a_prompt_that_fits_the_context`**, which asserts the
invariant this file got wrong twice — across three context sizes, the derived budget's
prompt must fit the window at the pessimistic tokenization rate. Beside it: the budget
cap, the ceiling scaling with context, a configured budget being honoured under the
ceiling, the per-item budget being smaller than the session one and capped by the same
ceiling, the cloud budget being untouched by any local setting, the two chars-per-token
constants leaning opposite ways, and the local timeout surviving where a cloud one would
clamp. The pre-existing derivation test now reads its inputs from the module instead of
re-typing `12288` and `3.0` — which is how it agreed with a wrong number for weeks.


### 2026-08-28 — Two scans wrote one CSV: the D1 feature-history corruption, fixed and repaired

`IMPLEMENTED`, `GREEN`. Trader-authorized (the file-scoped ask-first rule applied;
`master_avwap_lib/legacy.py` houses detector/scoring code). No detector, score, signal
or alert behaviour changed — this is the evidence WRITER for `d1_features_history.csv`
and the repair of the file it damaged, so plan.md sec 5's golden-fixture rule is not
engaged.

**What happened.** On 2026-08-27 the 12:45 swing scan was declared stale at 12:48
(`runner did not survive restart`) and a replacement started at **12:49** while the
first worker was demonstrably still alive. Both wrote the feature history. One
appended at the end; one rewrote it in place from byte 0. The result was a 498 MB CSV
with a **204-column header over a body that is 97.3% 255 columns**, **15 shredded
lines** — two alphabetical symbol streams interleaved into single rows, the leading
fields from one record and the trailing JSON blob from the next — and **372 rows of
real history destroyed** where the short rewrite overwrote the top of the file. From
that moment `export_scan_factor_views` and `export_bot_tier_tracker_views` raised
`ParserError` on **every scan**; both are caught and logged, so the scan went on
reporting success while two of its outputs silently stopped.

**Four rules, now enforced in `append_d1_feature_history`:**

1. **One writer at a time**, through `local_writer_lock` (the machine's real
   cross-process primitive — named mutex plus byte-range lock, released by the kernel
   on a hard kill) keyed by `lock_key_for_path`. Two overlapping scans is not an
   exotic state here: the stale-runner replacement path produces it by design. If no
   lock primitive is available the write is **skipped**, not attempted — without
   exclusion this is exactly the write that caused the damage.
2. **A rewrite is atomic** — temp sibling plus `os.replace`, never an in-place
   `to_csv` over a 498 MB file. That long half-written window is what the other
   process appended into.
3. **An unreadable header refuses the write.** The old `except: existing_columns = []`
   then failed its own truthiness test and fell through to a **blind append**, turning
   a transient read failure into permanent corruption. Losing one run's rows is
   recoverable; an unparseable record is not. The schema-change branch refuses the
   same way and leaves the file exactly as it was.
4. **The append path only ever appends the header's own columns**, so rows cannot
   stop lining up with their header.

**The file was repaired, and the repair recovered more than it lost.** Rebuilt from
the 2026-08-26 evidence snapshot (119,107 rows, uniformly 255 columns, verified clean)
plus every recoverable 2026-08-27 row from the live file, mapped onto the wide header
**by name** — the 204-column schema is a strict subset of the 255, so the narrow rows
carry their 204 real values and 51 blanks rather than being dropped. Result:
**129,081 rows, uniformly 255 columns, `pd.read_csv` clean**, against 128,720 in the
corrupt file — a **net gain of 361 rows**, because the snapshot restores what the
overwrite destroyed. All ten of 2026-08-27's runs are present; the 12:49 run survives
with 200 of its ~1,086 rows. **15 rows were unrecoverable** and are written to
`d1_features_history.quarantine-2026-08-28.jsonl` beside the file with their line
number, width, run_id and symbol — counted and named, never silently dropped. The
corrupt original is kept as `d1_features_history.csv.corrupt-2026-08-28`.

Tests: 5 new in `tests/test_master_avwap_setups.py`. Three are true reproductions and
**fail against the old writer** (proven by reverting it): the unreadable-header refusal,
the lock acquisition, and no-lock-means-no-write. Two are standing guards — the atomic
rewrite leaves no temp file, and every written row matches the header width.


### 2026-08-28 — The nightly narration: one bad citation no longer costs the night

`IMPLEMENTED`, `GREEN`. Three fixes to the local-AI layer, from a trader question
about file sizes that turned into an audit of what the overnight run actually
produced. The AI summary was healthy — a real narrative on 2026-08-26 and on
nearly every night since 08-11 — but the **daily digest had failed three nights
running (08-25, -26, -27)** while the model and every store were up, and the night
of 08-27 lost all three narrating jobs to a dead inference server.

**1. The local model server had no autostart.** Ollama is a user-session tray app
started by hand. Its log stops at 06:12 on 2026-08-27; the desk restarted around
13:00 and nothing brought it back, so `ai_summary`, `ticker_briefs` and
`daily_digest` spent the entire 22:00–06:00 window retrying against a refused
connection on `127.0.0.1:11434`. The deterministic slots (`veto_cohort_grading`,
`like_cohort_grading`, `evidence_report`, `journal_import`) were unaffected, which
is why it went unnoticed until the summaries were read. An `HKCU\...\Run` entry now
starts it at logon, and `scripts/run_ai_jobs.ps1` gained a **preflight**: probe the
configured endpoint, start `ollama serve` if it is a LOCAL endpoint that is down,
wait up to 60s for the socket, and **carry on either way**. It never refuses the
run — `degraded_no_narrative` is a designed state, the fact packs and the counting
jobs need no model, and a preflight that could block the night would be worse than
the problem it fixes. A remote endpoint is left alone; it belongs to whoever runs it.

**2. An unsupported citation now costs its ROW, not the document**
(`validate_ai_summary`, trader decision). One bad `evidence_refs` entry used to
raise, discarding every supported statement beside it; with two model attempts and
a three-attempt session cap, one predictable 12B slip cost a whole night. Nothing
is loosened about what may be PUBLISHED — an invalid ref is still struck out, a row
left citing nothing is still discarded — but what was dropped is now recorded
through a `dropped` sink, carried on the result as `citation_drops`, and disclosed
as a `[system]` row in the published `data_quality` section, because a document
quietly missing two of its four findings reads exactly like a thin evidence night.
If EVERY citing row is dropped the document still raises: a summary supported by
nothing is not a degraded summary. Shape and value errors still raise, unchanged —
a malformed document is the provider failing to answer, a different fault from a
model that answered and mis-attributed one line.

**3. The digest's own fact pack was the thing making it fail.** The narrator is
instructed to cite exact `source_id` values and is handed a document in which every
measured cell PRINTS one — `outcomes.intraday_finals`, `review.alert_review_events`,
`ops.ai_job_ledger` — while the validator knew only `digest.facts`. The model cited
what it was shown; that is the instruction working, not a hallucination. Packages
may now carry `citable_aliases`, and `digest.provenance_ids` **walks the built pack**
to collect them (listed by hand, a block added later would be shown and yet
forbidden). `usable_source_ids` honours aliases only when the package already has a
usable source, so an alias can never conjure citability out of an empty package.

**4. Fact pack v2 — the pointer hoist.** `daily_digest_facts_v1` → `v2`. The
2026-08-27 pack rendered at **14,070 bytes against an 8,192-byte target**, 72% of it
the outcomes block, and one `source_id` plus one `as_of` were printed 21 times for
21 cells that all shared them. `_hoist_block_pointer` lifts those two fields to
their block when *every* measured cell agrees (a block that mixes two stores keeps
them per cell, because then hoisting would state something false), and the slice
rows trade fourteen near-identical selectors for one `selector_template` that
rebuilds any row's exact selector from its own `env_key` and `side`. Measured:
**14,070 → 11,124 bytes, 21% smaller, not one figure dropped** and D2 untouched —
every value still carries its `n`. It is a new schema NAME because the SHAPE
changed; v1 packs on disk stay v1 and stay readable, and `clean_digest_sessions`
counts by session rather than by schema, so the Phase 2 collection window is
unaffected. The old sizing comment claimed 16 slices land "near the 8 KB target";
it is replaced with the measured number and the note that the figure the target
exists to protect — ninety packs as a trivial reducer context load — holds
comfortably at the post-hoist size. Cutting real slices to reach 8,192 exactly
would trade evidence for a round number.

Also fixed on the way past: `test_group_tape_service` pinned its fixture bars to a
hardcoded `2026-08-27` while the service filters to TODAY's date, so from 2026-08-28
its two sector-output assertions failed on the calendar rather than on the code
(proven pre-existing by re-running with this packet stashed). A `frozen_session_clock`
fixture freezes the service module's clock beside the bars; no production change, and
the same-date filter it exercises is correct.

Tests: 11 new (`tests/test_ai_summary.py`, `tests/test_ai_digest.py`), covering the
row-drop, the kept-good-refs case, the all-dropped raise, the `[system]` disclosure,
alias citability and its empty-package guard, the pack walk, the mixed-store
non-hoist, and exact selector reconstruction. `test_validation_rejects_hallucinated_
evidence_reference` was updated to the new contract; it is the regression pin for
the old behaviour and it failed before this change.

**Also found, NOT fixed — needs a trader decision.** `d1_features_history.csv`
(498 MB) went ragged at 12:49 on 2026-08-27: a 204-column header over rows of 119
to 524 columns. `export_scan_factor_views` and `export_bot_tier_tracker_views` now
raise `ParserError` on **every scan** (caught and logged; the scan continues, those
two outputs do not). The widening path in `master_avwap_lib/legacy.py:2397-2408`
reads the whole 498 MB file with pandas when the column set changes, and its
`except: existing_columns = []` degrades a read failure into a blind append. That
file houses detector/scoring code, so the file-scoped ask-first rule applies.


### 2026-08-27 (night) — repo hygiene: dead code, a dead dependency, stale doc claims

`IMPLEMENTED`, no runtime behavior change; frozen rebuild owed. A codebase-wide
assessment for dead code, duplication and documentation drift.

Removed: `ui/widgets/info_dot.py`, `ui/widgets/symbol_chip.py` and
`ui/models/journal_table_model.py` (236 lines) — zero references in Python, the
PyInstaller spec, JSON or Markdown, and `ui/` performs no dynamic module lookup. The
journal model/proxy pair was superseded by `panels/journal/trades_tab.py`, which builds
a `QTableWidget` over `JournalTrade` directly.

Dependencies: `scikit-learn` and `joblib` dropped from `requirements-core.txt`, and
`packaging/tradingbotv3.spec` no longer force-collects sklearn/scipy submodules. Nothing
has imported either since `a73f072` removed the trade-quality training script; the
collection was pulling ~93 MB into every bundle, and because
`collect_submodules("sklearn")` was unguarded it would have failed the build outright
once the dependency was dropped. `ruff` added to `requirements-dev.txt`: `pyproject.toml`
configures it and `CLAUDE.md` names it in the stack, but it was declared nowhere, so a
clean dev install could not run the configured lint.

Documentation corrections: the frozen-selftest expectation of `29/29` in `CLAUDE.md` and
`README.md` was stale — the count is a running total (29 on 2026-08-09, 30 later, 72
unfrozen today), so an agent comparing a correct run against 29 would read it as a
failure; both now say N/N and direct the reader to a current unfrozen run. `README.md`
also described the Desk Link code as "unused pending a cleanup packet" and
`master_avwap_mini_pc.py` as still present — both were removed 2026-08-24 — and
introduced five documentation entry points as "four"; the source-launch-is-production
decision and the one-desk-per-machine guard were added. `docs/README.md` now links the
15 decision records instead of listing them by title.

Documentation size (trader-approved the same night, "make it as easy as possible to
keep vibe coding"): `CURRENT_CHECKPOINT.md` had reached 7,901 lines / 449 KB across 113
dated entries and was a second changelog rather than a checkpoint, which made the
mandatory documentation read in `CLAUDE.md` about **260k tokens before any change could
be proposed** — an instruction no agent can follow, so it skims and appends, which is
what grew the file. Entries dated 2026-08-25 and earlier moved verbatim to
[`docs/CHECKPOINT_ARCHIVE_2026-08.md`](docs/CHECKPOINT_ARCHIVE_2026-08.md) (95 entries)
and the revision history from 2026-08-19 back to 2025-11 moved to
[`docs/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md`](docs/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md)
(36 entries); both are classified in `docs/README.md` as historical evidence that must
never be loaded as context. `CURRENT_CHECKPOINT.md` (now 1,431 lines / 82 KB) opens with
an **"Active state at a glance"** block carrying the branch, active roadmap items, last
verified baseline, the eight open live gates and the next action, and states that a
dated entry wins if the two disagree. `CLAUDE.md`'s mandatory workflow became a bounded
read — checkpoint glance block first, `plan.md` §5–7 plus the active phase, a *search*
of the changelog inventory rather than a full read — plus a standing rule to refresh the
glance block, keep entries short, and archive past ~1,500 lines. `CHANGELOG.md` is
unchanged as the authority on what exists; only its closed history moved.

Second pass, 2026-08-28 (trader: "can we summarize things to be even briefer?"):
`CHANGELOG.md`'s `Current implemented inventory` was 94% narrative — 3,808 of 4,061 lines
were dated entries wrapped around a **253-line thematic inventory that already states
what exists**. That inventory is the contract and was promoted to the top of the file;
the 73 dated entries older than 2026-08-26 moved verbatim to the archive and the 18 from
the last two build days remain under `Recent changes` (260 KB → 98 KB). `CLAUDE.md`'s
`Core loop / data flow` section — 42 KB, 65% of a file that loads into *every* session —
had each rule carrying the incident, measurements and trader conversation that produced
it; those moved verbatim to **[`docs/DESK_INTERNALS.md`](docs/DESK_INTERNALS.md)** while
`CLAUDE.md` keeps every rule as a binding imperative with a pointer (section 71% smaller,
file 65 KB → 35 KB). A check for 45 critical guardrail tokens confirmed all 45 survive in
`CLAUDE.md` itself; the rules bind from `CLAUDE.md` alone and both files change together.
The mandatory documentation read fell from **~260k tokens to ~97.5k (63% smaller)**.
`plan.md` was deliberately left untouched.

Third pass, 2026-08-28: `plan.md` narrowed **149 KB → 76 KB** (37,305 → 19,141 tokens).
Section 12 was 93% of the file and its Phases 0.5/0.6/0.7 were 72% of that section while
describing work already BUILT. Each of their 89 numbered items keeps its title/status
line, a spec and build-record pointer, and every gate reduced to its bold lead plus the
sentences carrying the gate — **verbatim, never paraphrased**; the build narrative moved
to [`docs/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md`](docs/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md).
A first attempt was reverted because splitting sentences on `;` truncated multi-part gate
lists (R1 lost two of its three owed proofs); the redone pass splits only on a period
before a capital, and a clause-level check reports 89 gate clauses before and 90 after,
with the 5 flagged items confirmed present by name in their bold titles. Structure is
unchanged at 10 sections and 89 items. The three dead UI modules removed in the first
pass are confirmed gone from disk and from HEAD, their stale bytecode deleted.

The mandatory documentation read across `CLAUDE.md`, `CURRENT_CHECKPOINT.md`,
`CHANGELOG.md`, `plan.md` and `docs/README.md` is now **~83,607 tokens, down from
~259,878 — 68% smaller** — with no rule, gate, or inventory statement lost.

Packaging gate MET 2026-08-28: rebuilt at `fff07b8` and the frozen self-test reports
`selftest OK: 72/72 checks passed (frozen)`, exit 0 — the frozen count equals the
unfrozen count, confirming the retired "29/29" expectation was stale by 43 checks. The
bundle fell **442 MB → 419 MB** with `sklearn` gone. `scipy` (79 MB) remains, pulled
transitively by a lazy import in `yfinance`'s price-repair path and by pandas sparse
arrays; neither is reachable from this codebase, so excluding it would reach roughly
340 MB — recorded as a recommendation, not done, because it needs its own rebuild plus a
live bar fetch to prove the yfinance path.

Verified after the changes: `pytest tests/ -q` **5261 passed, 22 subtests, exit 0**;
`smoke_check.py` **7/7**; `launch_gui.py --selftest` **72/72**; packaging spec-drift and
selftest suites **24 passed**. The spec edit is a packaging trigger, so a rebuild and a
frozen selftest are owed before the next merge to `main`.


### 2026-08-27 - Every tracker family enters bounded stop/target research

**IMPLEMENTED / GREEN; shadow accumulation and live canary owed.** The trader
authorized three linked research choices: evaluate every markable tracker
family, treat the next regular session's first completed M5 close as the entry
for a D1 setup found after close, and attach Auto Market Bias separately on
M5/M30/H1/H4/D1.

`research_warehouse/tracker_adapter.py` now streams the scenario CSV and reads
the small transition ledger to create warehouse occurrences. It deliberately
never opens the 1 GB tracker snapshot. Daily rescans collapse by
symbol/side/family/anchor and keep first-seen geometry, preventing a later scan
from leaking future information. A read-only real-data audit covered all 16
registered families: 249,438 scenario rows plus 10,820 transition rows became
6,663 deduplicated detections with zero unknown-family skips.

`outcomes.M5_CLOSE_RECIPES` is a separate 54-recipe discovery grid, leaving the
frozen slice recipes unchanged. Structural stop-source ranks 1–3 and ATR stops
0.5/1.0/1.5 are each crossed with 1R/2R/3R targets. The engine uses M5 only,
STOP_FIRST ambiguity and the existing fallback cost model; it needs no
trader-planned stop/risk and reads no bid/ask or earnings fundamentals.

`setup_market_context` stores five point-in-time champion bias readings for each
entry. The complete live decision is now one pure helper, including its
early-session day-percent fallback; the two live callers were refactored onto
it with no behavior change, and research calls that same helper. M30/H1/H4
derive from completed SPY M5 bars, D1 sees only prior complete daily bars, and
truly absent input stays `unknown`. The outcome build uses stable symbol
buckets plus Arrow-side occurrence/recipe filters so the expanded grid cannot
rematerialize the whole research history inside the desk.

The appended nightly `setup_research` slot always writes deterministic JSON and
Markdown. Medium local AI may only explain the bounded facts after a cell has
n>=30, five symbols and five entry sessions; below that floor it is not called.
No output reaches a detector, score, alert, Focus, watchlist, stop, target or
order. The durable M5 archive currently begins in August 2026, so older tracker
episodes honestly remain uncovered until backfill exists.

### 2026-08-27 - The Market Journal loads, carries the tape, and reaches the nightly AI

**IMPLEMENTED / GREEN; live gate owed.** Trader, after a full session of
in-session notes: "this is empty and feels very useless to me. this should
capture more stuff, such as SPY charts, what they looked like when the auto
mode flipped, my entries, what the charts looked like when i inputted entries,
what the D1 looked like.. i also expect the AI to get access to these notes for
the daily summary function."

Five entries were on disk for 2026-08-27 and the page showed none of them. Two
defects sat behind that, and the rest of the report was a missing feature.

**The page never loaded.** `MarketJournalPanel.reload()` had no caller at all -
not in `__init__`, not on show, and `_select_page` only special-cases the AWAY
Recap. The page was blank until "Refresh" was pressed, which reads as an empty
journal. It now loads the first time it is shown, and only then: the desk builds
every left-nav panel at startup and most are never opened.

**There were two services, not one.** `_build_journal_tab` constructed its own
`MarketJournalService`, so the desk tab's `entryWritten` was emitted by an
object the left-nav page had never heard of. Both wrote the same file correctly
(the ledger append is atomic per line) - what was lost was the refresh. One
process-wide `shared_journal_service()` now backs both surfaces, which is what
the R10.H docstring had claimed since it was written.

**Every entry now stores the tape it was written against** -
`scripts/market_journal_capture.py`, new. Bars, never pictures: a PNG cannot be
re-ranged, measured, or read by the nightly AI layer. A capture holds the
symbol's M5 and D1 and SPY's M5 and D1 as they stood at the moment of the note,
in two stores on purpose - a **sidecar** JSON per capture for the bar windows
(tens of KB, only the page reads it) and a **ledger row**
(`market_journal_chart_v1`, stream `market_journal_charts`) carrying a short
text `digest`: where price sat against its session range, session VWAP, the
prior session's extremes, the 20/50/200-day averages and RVOL. The raw window
would starve every other source in an AI packet; the digest says the same thing
in a few hundred characters.

`market_journal_entry_v1` is **untouched** - a capture joins by `entry_id` from
the outside, which is what lets it be written AFTER the entry, on a worker,
without a note ever waiting on a chart. A capture that fails leaves an entry
that is honestly chartless; an entry that was never written is a lost thought,
and those are not the same cost. Every bar list is a CACHE read
(`AlertCenterPanel.journal_chart_bars`, new and public) - nothing fetches.

**Auto-mode flips write their own row.** `AutopilotService.autoModeChanged`
(previous, current) fires only when `auto_mode` actually moves - a profile
change while Auto is OFF is not a flip - and `MainWindow._record_auto_mode_flip`
writes a Market Journal entry with SPY's M5 and D1 attached. The row carries
`ORIGIN_AUTO_MODE_FLIP` and `market_journal.is_machine_entry` reads it back, so
the page can mark it `[desk]`: the journal is one timeline, and a reader
counting "what did you think?" must never count a sentence nobody thought.

**The page draws the capture.** Selecting an entry loads its sidecar on a worker
and draws up to four panes; a pane with nothing stored is HIDDEN rather than
drawn empty, and a stored bar whose stamp will not parse is dropped, counted and
named (the axis formats every stamp with `strftime`, so one string takes the
chart down rather than degrading it).

**The nightly AI reads the journal now.** `market_journal` joins
`briefs.DEFAULT_SCOPES` on the trader's explicit instruction, reversing the
R10.I opt-in - which was itself a recorded trader decision, and the same trader
is the only thing that could reverse it. Its sources keep the funding rule
(distilled first, free text last): evidence report, day context, **chart
digests**, then the entries. `TICKER_BRIEF_SCOPES` stops being an alias for
`DEFAULT_SCOPES` and keeps the original four - a session-level journal entry in
a per-symbol packet is the TB-0/TB-5 failure mode.

Two pinned tests changed rather than being worked around: both asserted the
opt-in that the trader has now reversed, and they now pin the new decision.

### 2026-08-27 - Double-click on a claim commits the like, the way it does on a veto

**IMPLEMENTED / GREEN.** Trader: "i want to be able to double click the like and
claim the same way i can double click the veto."

The two gestures had drifted apart. `select_reason` (veto digit) and the reason
list's `itemActivated` (veto double-click) both call `commit_veto`, which
diverts to the note field only when that reason's `note_required` is unmet. The
like's `select_setup` and `_claim_picked` went straight to `_prompt_for_why`
and could never commit - so a trader who had ALREADY typed the why was sent
back to a field they had just filled in.

- Both like gestures now call `commit_like`, which is where R9.2's required-why
  guard already lives. That is the veto's exact shape: the gesture ATTEMPTS the
  commit; the rule enforces itself inside the commit rather than by refusing to
  reach it.
- **The 2026-08-22 rule is untouched** ("if I like a chart I should always be
  prompted with why"). A like with no why still writes nothing, still holds the
  chart, and still moves focus to the why with the same message - its two
  existing tests pass unchanged, and a new one pins the double-click case of
  it. The only new capability is: why typed, then the gesture commits.
- The digit changed with the double-click deliberately. The veto's digit and
  double-click are identical to each other, and leaving the like's digit
  nagging while its double-click committed would make the rail internally
  inconsistent in a way the veto is not.
- The stale docstring that claimed "double-click and Enter commit it exactly as
  they do a veto" is now true, and says what "exactly as a veto" means.

Nothing else in the rail moves: the LIKE still retires the chart the way it
did, still writes only `trader_annotations.jsonl` + the like cohort, and still
adds nothing to Focus or any watchlist.

Tests: `tests/test_qt_alert_capture.py` +5 (59 in the file) - the double-click
committing with a why, the double-click still refusing without one, the digit
committing with a why, the why field cleared after a commit so the next chart
cannot silently inherit the previous chart's reasoning, and both lists'
activation routed through their commit. Fail-before-fix: 4 of the 5 fail on the
old wiring; the fifth is the no-why regression guard, which must pass on both
sides. Full suite **5203 passed, 19 subtests, exit 0**.

### 2026-08-27 - The ticker popup opens 10% short of the screen, top and bottom

**IMPLEMENTED / GREEN.** Trader: "make the charts that pop up when i click on a
ticker just a little less tall. i dont want them edge to edge on the screen just
reduce by 10% top and bottom."

This is the 2026-08-11 sizing ask corrected, not reverted. That one fixed the
opposite problem - the popup opened at a fixed 1180x760 whatever the monitor,
squeezing both charts into about half the vertical space - by taking the height
from the hosting desk window, or the screen's available area, minus a 60px/40px
title-bar allowance. On this desk's monitors that is edge to edge.

- `symbol_snapshot_dialog.inset_vertical_bounds(anchor_top, anchor_height)` is
  a new PURE helper: it leaves `POPUP_VERTICAL_INSET` (0.10) of the anchor free
  at the top and again at the bottom, so the popup opens at 80% of whatever it
  is anchored to. The gaps come from CENTRING the final height inside the
  anchor rather than from adding the inset to the top, so they stay equal even
  when the floor below overrides the inset.
- `POPUP_MIN_HEIGHT` (760) is a floor the inset cannot go under - both charts
  carry a 120px minimum and a squeezed popup is exactly what 2026-08-11 fixed -
  and it never pushes the popup off the top of the screen to honour itself.
- The anchor is chosen exactly as before (hosting window frame if visible, else
  the screen's available area). The proportional inset replaces the old 60/40
  allowances, which it dwarfs.
- Measured on the desk's three monitors: 4K panels go 2052 -> **1690 px** with
  211px free at each end; the 2560x1392 goes 1332 -> **1114 px** with 139px at
  each end. 82-84% of the old height.

One dialog class and one factory (`show_symbol_snapshot`) serve every ticker
click - Alert Center, Industry, Master AVWAP, and through them the Strength
Board and the group tape - so this is one change for all of them. It sets only
the OPENING size; a trader resize afterwards is still kept, because the dialog
is created once per owner panel and reused.

Tests: `tests/test_snapshot_popup_height.py` (6) - the tenth at each end, equal
gaps across four anchor sizes, the constant, the floor winning on a short
screen, the floor not pushing the popup off-screen, and the dialog actually
routing through the shared helper. Fail-before-fix: 6/6. Full suite **5198
passed, 19 subtests, exit 0**.

### 2026-08-27 - The desk's 8-13 GB memory jumps: three causes, all fixed

**IMPLEMENTED / GREEN. One live gate owed.** plan.md Phase 0.9 item 6, built to
`docs/analysis/OPUS_BUILD_PROMPT_DESK_MEMORY_2026-08-27.md` on the 2026-08-27
(10:00) investigation. The trader: "there are times the program jumps to 10gb
of RAM usage."

**Cause 1 - the post-scan warehouse build materialised a whole MONTH of M5 bars
to use one session.** Three steps did
`store.read_table("bar_m5", "month=YYYY-MM").to_pylist()` and then filtered in
Python. Partitions are month-keyed, so the cost grew all month. Measured on the
live lake this afternoon: `silver/bar_m5/month=2026-08` = **8,704,108 rows /
408 MB parquet / 158 files**, and `to_pylist` costs **1,769 B/row = 15.4 GB**
held whole - against a largest single session of 588,778 rows, 6.8% of it.
(The 10:00 investigation measured 8,175,471 rows / 13.3 GB; the month has grown
since, which is the point.)

- New `ResearchStore.read_rows(dataset, partition, *, columns, symbols,
  interval_start_range)` pushes the predicate into
  `Dataset.to_table(filter=...)`, so Arrow drops rows before any Python object
  exists. Deliberately NOT a free-form filter argument: only the two predicates
  the callers replaced are offered, so nobody can express something subtly
  different from the Python test it stands in for. `symbols` matches exactly,
  no case folding, because `symbol in wanted` did.
- `aggregate.build_derived_bars` and `features.build_intraday_snapshots` narrow
  to the session window (and to named symbols); `cli._run_outcomes` narrows by
  SYMBOL ONLY - deliberately no date filter, because the outcome walk runs
  forward over a horizon that crosses sessions, which is why
  `_m5_partitions_for` already widens to the trigger's month plus the next
  (BD-66/BD-69). `build_intraday_snapshots` applies the symbol filter only when
  symbols were named, because otherwise its cohort is derived from the bars
  present in the session.
- **Measured after, same lake:** a full session read **0.53 GB** (297,230
  rows), a 20-symbol outcome read **0.31 GB** (175,235 rows). 15.4 GB -> 0.53
  GB, ~29x.
- Equivalence is asserted against a longhand REFERENCE implementation of the
  old read (read the month, filter in Python) and compared as published ROWS,
  not counts - a filter that shifted a session boundary by one bar would keep
  the count and change the answer.

**Cause 2 - the 1.03 GB tracker snapshot was read whole to decide it was
unchanged.** `master_avwap_setup_tracker.json` measured **1,026,057,028 bytes**.
`ingest_artifact` did `read_bytes()` and hashed the bytes BEFORE consulting the
watermark, so every bronze ingest allocated 1.03 GB - including the ones that
immediately answered UNCHANGED - and a changed file then ran `json.loads` over
the decoded text, several GB more.

- `_sha256_path` hashes in 1 MB chunks and the UNCHANGED check is hoisted above
  `read_bytes`, so an unchanged snapshot now costs no allocation at all.
- A SNAPSHOT over `SNAPSHOT_PARSE_MAX_BYTES` (64 MB) is stored in FULL but not
  parsed; `_looks_like_json` (first/last non-space characters) drives the
  quality flag instead. **This loses nothing measurable for the artifact that
  triggers it:** `setup_tracker` declares neither `event_keys` nor `id_keys`,
  so `_parse_event_at` returns None on its first line and `_first_value`
  returns "" without reading the payload - parsed or not. The parse influenced
  exactly one column, and a test asserts the parsed and skipped rows come out
  identical. Residual stated rather than hidden: a CHANGED snapshot still costs
  ~size bytes plus a same-size `str`, because `payload_text` must be a string
  for the publish path, which this packet did not touch.
- BD-73 records the threshold and its reopen trigger: if `setup_tracker` ever
  gains those key tuples the skip WOULD empty real columns, and a fixture
  assertion fails loudly rather than silently.

**Cause 3 - BounceBot never freed its IB bar buffers.** `self.data[reqId]` held
every historical reply; only the RRS and contract-bars paths popped it. Five
others (`build_atr_cache`, `request_and_detect_bounce`, and the three
`check_*vwap*_touches`) deleted the ready event and left the bars: **206 KB per
390-bar request, ~400 requests a scan cycle, 1.5-2 GB over a session**, held
until the process exits. That is why the desk settled at 2.5 GB rather than 1
GB once a build released.

- All five now free the buffer with the event, on the success AND timeout
  branches. `request_and_detect_bounce` - the hottest path - pops at the read.
- `historicalData` no longer auto-creates a buffer for an unknown reqId. Every
  request path creates its buffer before issuing the request, so an unknown
  reqId can only be a straggler; auto-creating one meant a timed-out request
  leaked AFTER the fact, and a bar racing the requester's own pop appended to
  the very list the caller was reading. Both are closed.
- **The trader authorised this one `legacy.py` edit and nothing else in that
  file.** It was verified LIKE a detector change even though it is not one: a
  repo-wide sweep confirmed each reqId is read exactly once, by the function
  that created it, with `self.data` never iterated, persisted or touched
  outside the class - and the golden fixtures plus all 411 BounceBot tests pass
  unchanged.

**Premise corrected while building** (reproduce, do not inherit): the build
prompt listed `cli._run_outcomes` as one of the three live costs. It is not one
today - `setup_occurrence` holds **0 rows** on this lake, so `_run_outcomes`
returns `NO_OCCURRENCES` before it ever reads `bar_m5`. It was fixed anyway,
because it becomes a cost the moment the BD-44 detector adapter lands.

**No packaging trigger** (no new dependency, asset, top-level package, dynamic
import or `__file__` use).

Tests: `tests/test_warehouse_session_scoped_reads.py` 10,
`tests/test_bronze_snapshot_large_files.py` 9,
`tests/test_bouncebot_reqid_buffers_are_freed.py` 12. Fail-before-fix per file:
8/10 (two are the equivalence guards, which must pass on both sides), 9/9, and
11/12 (the survivor guards that a live request still collects its bars). Full
suite **5192 passed, 19 subtests passed, exit 0**; smoke 7/7.

### 2026-08-27 - Group RS/RW tape rebuilt: its own five-minute clock, 90 | 60 | 30 minutes, today's bars only

**IMPLEMENTED / GREEN. Live gate owed** (one DESK session). plan.md Phase 0.5
item 11, built through `docs/prompts/GROUP_TAPE_REBUILD_OPUS_PROMPT.md`
(packets T-1..T-4), which the trader authorized the same morning after deciding
to hide the old tape rather than delete it.

The complaint was "often times the sectors and industry RS/RW thing at the top
is totally wrong and doesn't reflect what is actually strong over the last
30-60-90 minutes". The investigation found the maths RIGHT and the clock wrong,
so the formula is lifted out unchanged and given a new clock, a new source and
three windows.

- `scripts/group_rrs.py` (pure: bars in, floats out, no I/O, no Qt, `now`
  always passed in). `wilder_atr_last` + `real_relative_strength` reproduce
  `legacy`'s, including the two details a re-derivation gets wrong - the ATR
  seeds on the first `length` true ranges and smooths over ALL the rest, so it
  depends on the whole series and not its tail, and a non-positive ATR is
  `None` rather than 0, which is what stops the division producing an infinity.
  `session_bars` = `completed_bars.completed_m5_bars` AND a same-date filter,
  which is what stops a window reaching over the overnight gap; `align_bars`
  intersects the two series on normalized stamps, so an ETF that halted for a
  bar cannot have its move measured over a longer span than SPY's and read as
  strength, and an ET-stamped ETF still meets a UTC-stamped SPY. `rrs_windows`
  = 6/12/18 bars = 30/60/90 minutes off ONE filtered+aligned series, so the
  three numbers are guaranteed to describe the same bars. A window without
  `length + 2` bars is `None`. `SECTOR_ETFS` is a COPY of `legacy`'s map, not
  an import - the tape must survive BounceBot being off and must not drag a
  14k-line detector module onto a worker thread - and a drift test pins the
  copy.
- `scripts/ui/services/group_tape_service.py`, the Strength Board's shape: one
  `QTimer`, single-flight worker, last-good on failure, `status_text`, bounded
  `shutdown`. **ONE batched `yfinance` download per tick** (SPY + the 11 SPDRs
  + the 49 industry proxies, deduped to ~53 symbols), `period=1d interval=5m`,
  **no retry inside the tick** - Yahoo rate-limits bursts, and the next tick is
  the retry. **Zero IB traffic and no `legacy.py` change**, so the locked
  pacing budget is untouched. Quiet-hours gated on `auto_scanning_due`,
  fail-open; `refresh_now` never gated. A missing or unreadable industry map
  means SECTORS ONLY, said in the status line - two thirds of the chips
  disappearing silently would read as "nothing is moving". No completed SPY
  bars for today is said out loud rather than rendered as an empty strip.
- `GroupTapeStrip`: `SPARK_TIMEFRAMES = ("90", "60", "30")`, ranked by the 30.
  An unmeasured window draws NOTHING - a zero-height bar on the zero line is
  indistinguishable from "exactly in line with SPY", which is a claim - and the
  tooltip names which windows are still filling. `rotation_callout` is now "up
  on 30 while still down on 90" and its mirror, and the callout line carries
  the payload's as-of plus the service's `status_text`, so a stale or failed
  read is visible rather than silent. **Chips diff**, keyed by ETF: reused,
  re-labelled and re-ordered instead of destroyed and re-created, and the
  variants moved from a per-chip f-string `setStyleSheet` into `theme.qss`
  keyed on a `side` dynamic property with six pre-mixed rgba tokens in
  `theme._derived_tokens`. The old path was 34 CSS parses plus 34 widget
  constructions every payload, on the GUI thread - the exact shape the
  2026-08-21 fluidity pass measured. `GroupChip` sets `WA_StyledBackground`,
  which a widget carrying its own stylesheet got for free.
- `TradingDeskPanel`: the tape is VISIBLE again and fed by
  `tapeChanged`/`statusChanged`; the `rrsSnapshotChanged -> update_groups`
  wiring is gone. **The RS Window tab and `focus_picks_panel` still receive
  `rrsSnapshotChanged` unchanged** - it answers a different question (who led
  over the selected window at scan time) - and a test pins that both wirings
  coexist. The service is shut down in the desk's shutdown list, and that list
  now resolves it the way it already resolved `price_alert_service`: naming it
  inline made a missing attribute raise while the component list was being
  BUILT, before the fan-out loop ran, so a desk whose `__init__` died partway
  would have released nothing instead of one thing.

**Deliberately NOT built** (from the prompt's own "not in this prompt"):
industry = median member return instead of the ETF proxy (needs member bars -
an IB-budget question), any change to the 27-minute scan cycle, and anything in
`legacy.py`.

**No packaging trigger**: `scripts/group_rrs.py` and the new service are
ordinary static imports on a chain reachable from `launch_gui.py`, so
PyInstaller collects them by dependency analysis - no new dependency, asset,
top-level package or dynamic import. The spec-drift guard passes.

Two failures were found on the way and fixed; **neither was caused by this
work**. `test_review_watch_buttons_arm_trigger_and_flag_red` was a CLOCK BOMB:
its fixture's last bar starts at 11:25, so before 11:30 local that bar was
still forming, the 2026-08-27 VWAP-side leg read UNKNOWN and the chart showed -
after 11:30 both bars complete, the fixture's long sits under its own session
VWAP and the filter correctly hid it. It passed at 10:xx and failed at 11:36 on
the same tree. The production behaviour is right; the test is about the watch
buttons, so it now switches the show-time filter off the way five sibling files
already do. `test_trading_desk_shutdown_continues_after_one_component_raises`
needed the new component on its `SimpleNamespace` desk.

Tests: `tests/test_group_rrs.py` 16, `tests/test_group_tape_service.py` 16,
`tests/test_qt_group_tape.py` rewritten to 17, plus one new partial-desk
shutdown test. Fail-before-fix shown per file: 16/16, 16/16, and 15/17 (the two
survivors are the deliberate regression guards - the silent callout, and the RS
Window tab still receiving `rrsSnapshotChanged`). Full suite **5161 passed, 19
subtests, exit 0** (305 s); smoke 7/7.

### 2026-08-27 - Clicking away from an M5 chart is a skip, not a re-queue (trader rule 4, third pass)

**IMPLEMENTED / GREEN.** Trader: "When I click on an alert in the new M5 alert
bar and then click to another one, it shouldn't queue the old M5 alert in the
waiting list. It should just be considered a 'skip for now' situation." The
bar took M5 alerts OUT of the waiting list, but `_select_review_alert` - the
shared feed-row/bar click path - still pushed whatever chart it replaced to
the HEAD of that list, so a trader working down the bar refilled the D1 queue
with the M5 rows the bar was built to keep out of it.

- `AlertCenterPanel._current_review_holds_place` (new, defaults `True`)
  records where the chart in front CAME FROM, which is the thing the decision
  actually turns on. `_advance_review_queue` sets it `True` (that chart was
  popped off the waiting list and keeps its place); `_select_review_alert`
  sets it to `not _is_m5_review_alert(alert)` (a clicked D1 row / armed hit
  holds a place, an M5 bar row does not). On the next click, a chart that
  holds a place is re-inserted at the head exactly as before and one that
  does not is skipped.
- Why a flag rather than re-testing the outgoing alert: the refresh path
  (`_enqueue_review_alert`, same-symbol branch) REPLACES a queued D1 chart's
  alert object with that symbol's newer M5 alert, so `_is_m5_review_alert`
  asked about the outgoing object would answer "M5" for a chart that really
  is holding a D1 queue slot, and clicking away would silently drop it.
  Pinned by `test_a_refreshed_d1_chart_still_holds_its_place`.
- The skip is RECORDED, not silent: `_record_review_event("skip", ...)` with
  the dwell and `detail={"reason": "clicked_away_from_m5_alert"}`. The
  impression was already written - `_render_current_review` emits `shown` for
  a bar-clicked chart like any other - and `shown` is the denominator for
  P(take | shown), so leaving the click-away unanswered would have stranded
  an impression with no verb and biased the rate. `skip` is that stream's
  existing definition of "looked at the chart and passed"
  (`scripts/review_events.py`), which is the trader's own phrase. No status
  line: the replacement chart is already up, so a message would be noise.
- Unchanged, deliberately: the routing at `_enqueue_review_alert` still
  records nothing, the M5 bar is still not a queue, the feed and History
  still keep every clicked-away row, and no parking happens here (that stays
  specific to Skip-after-arming-a-D1 in `_skip_review_alert`).

Tests: `tests/test_qt_m5_alert_bar.py` +3 (22 total, whole suite 5119 passed):
the second bar click skips rather than queues and writes exactly one `skip`
with its reason; a queued D1 chart still returns to the head of the queue when
a bar row is clicked, while the M5 that replaced it does not; and the
refreshed-D1 regression guard. With the panel change stashed the first two
fail on the old behaviour (`['AMD', 'NVDA', 'MUFG', 'XOM']` where the queue
should read `['AMD', 'MUFG', 'XOM']`) and the third passes, as a guard should.

### 2026-08-27 - Group RS/RW tape removed from the desk (trader decision); rebuild plan parked in plan.md

**IMPLEMENTED / GREEN (a hide).** "Often times the sectors and industry RS/RW
thing at the top is totally wrong and doesn't reflect what is actually strong
over the last 30-60-90 minutes." Investigated: the formula
(`real_relative_strength`, ATR-normalized) is right - an independent Yahoo
recompute at 09:55 ranked the same window the same way - but the tape refreshes
only when a scan cycle's RRS pass finishes (10-30 min apart that day, frozen in
between, once 31 minutes late on a flip), its one intraday number is a
60-minute window that carries the overnight gap for the first hour, and
"industry" is one of 49 ETF proxies for 136 industries. Trader: "just remove
it for now and put this build plan in the .md files for the future."

- `TradingDeskPanel`: `group_tape.setVisible(False)`. Hidden, not deleted -
  the widget, `tests/test_qt_group_tape.py`, the `rrsSnapshotChanged` wiring
  and the `tape_host` mount point all stay, so the rebuild drops into place.
  Nothing upstream changed; the RS Window tab still reads the scan payload.
- The rebuild (a 5-minute Yahoo-batched 30|60|90 tape off today's bars, zero
  IB, no `legacy.py` change) is written out under plan.md Phase 0.5 item 11
  and, later the same morning, authorized for an Opus build session:
  `docs/prompts/GROUP_TAPE_REBUILD_OPUS_PROMPT.md` (packets T-1..T-4, hard
  rules: zero IB, no `legacy.py`, completed today-only bars, UNKNOWN never
  invented, parity test against `real_relative_strength`).
- Test added: the tape is hidden on the desk and still wired.

### 2026-08-27 - Intraday alerts are a list beside the chart, not a queue in front of it (trader rule 4: the M5 alert bar)

**IMPLEMENTED / GREEN.** "A lot of my charts to review are M5 charts. If I can
instead just get a list I can copy and paste into TC2000 that would be
faster... a little sidebar in between the master AVWAP setups and the chart...
the ticker and the alert type (new HOD, VWAP bounce etc) and I can choose what
to look at. Then we can totally purge M5 alerts from the waiting list and keep
those for D1 alerts." Ordering, when asked: "latest at the top, the oldest at
the bottom."

- `scripts/ui/widgets/m5_alert_bar.py` - `M5AlertBar`: one line per alert
  (`07:09  ▲ SYMBOL  type`), newest on top, side-coloured through an item
  foreground role (no per-widget stylesheet, no rebuild). `Copy all` puts the
  tickers on the clipboard one per line, each once, in bar order - a TC2000
  paste; `Clear all` empties the bar ON SCREEN. A click charts the alert
  and takes its line away (trader: "after I click on an alert it should go
  away") - the feed and History still have it.
  Bounded at 400 rows (a session produced 72 in its first 46 minutes).
- `AlertCenterPanel._is_m5_review_alert` + routing in `_enqueue_review_alert`
  - the one door into the queue, AFTER the AWAY-recap branch and the parked
  check, so everything upstream (the backing list, the feed, History, the
  evidence streams, the AWAY recap) is untouched. An ordinary intraday alert
  is emitted on `m5AlertPosted` and never queued; a D1 row, a Focus D1 flag,
  a chart-watch hit, a price alert the trader armed, an auto-pick proposal, a
  typed symbol and a deliberate Focus review keep their chart. The chart in
  front still refreshes from its own symbol's new M5 alert. `chart_alert()`
  is the public click path (same as a feed-row click). `m5AlertsDayRolled`
  clears the bar with the other day-scoped state.
- `TradingDeskPanel` - the bar is the LEFT column of the desk splitter
  (`m5_alert_bar | alert_center | master_workspace`, stretch 0/3/2, floor
  `px(150)`), an "M5 alerts" tab in tabs mode, rescued across mode switches.
  It was built between the chart and the setups and moved to the left the
  same morning at the trader's second pass ("move it to the left of the
  visual chart"); `DESK_SPLIT_KEY` bumped to `..._v3` so the middle-bar split
  saved that morning is not replayed onto the new order.
  `desk_layout.DESK_SPLIT_*` are three weights - the bar's share comes out of
  the setups side, so the chart column keeps its lead.
- Consequences, stated: the regime-pause hold EXPIRY (2026-08-21) and the
  movers/VWAP/SMA legs now act on D1 rows and the trader's own charts; an
  intraday row never reaches them. A counter-trend regime-pause row lists in
  the bar. EVENING's "queue the trader wakes up to" is now the bar plus the
  D1 queue. AWAY is unchanged: recap, and nothing posted to the bar.

Tests: `tests/test_qt_m5_alert_bar.py` (19: order, row text, copy dedupes
newest-first, clear, click, bound; routing for every kind that stays and the
kinds that go; nothing recorded; AWAY untouched; the chart in front refreshes;
the day roll; the bar between the two columns and wired both ways). With the
panel, desk and layout changes stashed, 13 of the 19 fail (the six pure-widget
tests pass: the untracked widget file was not stashed). Seven queue-mechanics
files (`test_qt_alert_center`, `test_movers_only_review`,
`test_qt_review_vwap_side`, `test_qt_regime_pause_expiry`, `test_qt_arm_dock`,
`test_review_events`, `test_review_guidance`) gained one autouse
fixture that switches the routing off - they test what the QUEUE does with a
row, and a D1 fixture would drag the D1 feed into every assertion; the
routing itself is owned by the new file. `test_away_day_recap` (2) and
`test_qt_regime_pause_auto_focus` (6) were rewritten to the new expectation.

### 2026-08-27 - D1 recommendations against their trend are hidden; the setups popup walks with Prev/Next (trader rule 3)

**IMPLEMENTED / GREEN.** The chart it came out of: MUFG, a swing-scanner D1
row "(short) zone1 reject at AVWAPE", sitting above its SMA50, SMA100 and
SMA200 in a clean uptrend. The scanner's own feature file called MUFG a LONG
setup and carries a `directional_sma_stack_aligned` flag; the short alert
never read it. Trader: "longs should be above the 200 SMA and shorts below
the 50 SMA at least."

**The rule:** a D1 long charts only above its SMA200, a D1 short only below
its SMA50 - the D1 recommendations (`is_d1` rows and `focus_d1_event` flags)
and nothing intraday. It is the THIRD leg of the one review verdict
(`_review_chart_state`), so it is asked at queue time and again at show time,
hides and counts on the same button ("N hidden (inside yesterday's range /
wrong side of VWAP or SMA) - show"), and a revealed name is badged
`wrong side of SMA`. UNKNOWN shows.

- `scripts/sma_trend_gate.py` - the decision, pure: `sma_trend_state(side,
  price, sma50, sma200)` (a long needs `> sma200`, a short `< sma50`, the
  other average is not consulted - "at least"), and `trend_levels(d1_bars,
  today=)` off COMPLETED daily closes: a bar marked `preview`, or dated today
  while today trades, is left out, because an average that moves every tick
  must never be the thing that hides a chart. Fewer than 200 closes is no
  SMA200 (`strength_scan.sma` refuses "as many as we have").
- `AlertCenterPanel.sma_trend_state(symbol, side)` - averages off the local
  daily store (`_d1_bars_for`), price off the last completed M5 bar when the
  bot has one and the last daily bar otherwise; memoized on both series'
  identity; any failure is UNKNOWN. No fetch, no IB traffic.
- Detector untouched: the scanner still writes the row and its evidence;
  this decides only whether the chart occupies the pane.

**Prev / Next on the snapshot popup** (same request): `SymbolSnapshotDialog`
gains `◀ Prev` / `Next ▶` beside `✕ Dislike`, visible only in a review walk
(a typed lookup has no list). They route through the setups panel
(`snapshot_review_previous` / `snapshot_review_advance`, both on the existing
`_open_next_symbol_snapshot`, now `step=±1`, wrapping at either edge) and
record nothing - Space on the table is unchanged.

**Investigated, not changed - "a lot of these candles are from Yahoo despite
the API being up":** the daily HISTORY is the durable D1 store. Only today's
FORMING candle is at issue, and it is built from BounceBot's cached IB M5
bars - which exist only for names in the current M5 scan set (the
watchlists, Focus, auto lists). For any setups-table name outside that set
(FTRE: "No cached M5 bars - not in the current scan set") the popup fetches a
Yahoo daily row for today as the preview and labels it exactly so
(`SymbolSnapshotWidget._request_snapshots`: `ibkr-cache` when M5 bars exist,
else `yfinance-fallback`). IB is up; there is simply no IB fetch path for a
forming candle on a name the bot is not scanning, and adding one would spend
the locked IB pacing budget on every double-click - a design decision for the
trader, recorded in `CURRENT_CHECKPOINT.md`.

Tests: `tests/test_sma_trend_gate.py` (11), `tests/test_qt_review_sma_trend.py`
(13: D1 short over its 50 hidden, Focus D1 long under its 200 hidden, an M5
alert is not asked, UNKNOWN shows, the button and badge, hides-never-deletes,
show-time withholding, the measurement over real bars - MUFG's shape, the M5
close as price, short history is UNKNOWN), `tests/test_qt_snapshot_prev_next.py`
(5: visible only in a walk, next/previous wrap, records nothing, side travels).
With the four source files stashed, the two Qt files fail together.

### 2026-08-27 - Chart review hides the wrong side of VWAP, and checks at show time (trader rule 2)

**IMPLEMENTED / GREEN.** The chart it came out of: EPD, a Focus D1 flag
("New 5-day high", M5 bar 06:30) that reached the review pane at 07:30 sitting
under session VWAP and fading - "a stock like this really is just wasting my
time." Two defects in the movers-only filter of 2026-08-19: it had only the
prev-day-extreme leg, and it was measured when a row was QUEUED, not when the
chart was SHOWN, so a queue 74 deep served hour-old verdicts.

**The rule:** a long charts only above session VWAP, a short only below it,
and the filter is asked again the moment a chart is about to show. Hidden
names are counted on the same button ("N hidden (inside yesterday's range /
wrong side of VWAP) - show") and one click reveals them for the session. Same
exemptions as before: a deliberate Focus review and an armed chart-watch hit
always show.

- `AlertCenterPanel.vwap_state(symbol, side)` - the adoption gate's own VWAP
  leg, `focus_adoption_gate.session_vwap_state`, fed by
  `regime_pause_hold.session_levels` over the cached M5 series (session VWAP
  from `chart_snapshot.session_vwap_series` on completed bars; never
  BounceBot's dynamic/EOD VWAP). Memoized on the bar-series identity like
  `_measure_mover_state`; a sideless row is UNKNOWN. No fetch, no IB traffic.
- `_review_chart_state(alert)` - both legs, one answer: CLOSED when EITHER leg
  is verified against the name, UNKNOWN when nothing is verified against it
  and something could not be measured (SHOWS, tagged), OPEN otherwise. This is
  deliberately not the gate's ordering ("could not measure" before "failed"):
  the gate explains an eviction, the filter decides a display, and one
  measured reason to hide is enough.
- `_enqueue_review_alert` reads `_review_chart_state` at queue time (was the
  extreme leg alone); `_advance_review_queue` reads it again at show time and
  withholds a candidate that has since gone wrong, walking on to the next.
  The revealed-for-the-session flag switches both checks off together.
- The review badge gains `wrong side of VWAP` for a revealed name the VWAP
  leg hid; `MOVING` now means extreme verified AND VWAP not verified against.
- Unchanged: it hides, never deletes; nothing reaches the review-learning
  stream, `review_policy.json`, any store or watchlist; the chart in front of
  the trader is not re-judged while they look at it; the Focus chip's own
  `MOVING` flag still reads the extreme leg alone.

Tests: `tests/test_qt_review_vwap_side.py` (21: the leg, the badge, the
button, the exemptions, no evidence written; show-time withholding, reveal,
the session-scoped switch-off, an armed hit never withheld; the measurement
over real bars - under/over VWAP, no bars / no volume / no side is UNKNOWN,
an unreadable read is UNKNOWN, the memo is keyed on the bars). All 21 fail
with the panel and widget changes stashed. `test_movers_only_review.py` is
untouched and still green.

### 2026-08-27 - With-trend regime-pause rows auto-join M5 Focus (trader rule, same morning)

**IMPLEMENTED / GREEN.** "I've been doing nothing but managing the bot all
morning. There are too many trades." Measured from `alert_review_events` for
the session's first 46 minutes (06:33-07:19): **124 charts shown** - one every
22 seconds - 40 skipped, 60 "Not today", and at 07:09 the pane read
**23 hidden / 74 waiting**. Between 07:09 and 07:18 the trader reviewed all 21
"holding highs" rows the regime-pause watch produced on a `bullish_weak` open
and put **twelve of them on M5 Focus by hand**, one click each.

**The rule (trader, 2026-08-27):** a swing LONG holding its highs on a bullish
day, or a swing SHORT pressing its lows on a bearish day, is added to M5 Focus
by the machine and never occupies the review chart - the decision is made. The
mirror cases (counter-trend rows) and a day with no directional read stay on
the queue exactly as before.

- `scripts/regime_pause_focus.py` - the whole decision, pure: `day_bias(env)`
  collapses `bullish_weak`/`bullish_strong` to one family, `focus_side_for(env,
  side)` names the Focus side or `None`. Reads nothing, no clock.
- `AlertCenterPanel._auto_focus_regime_pause` - called from `add_alert` AFTER
  the backing list insert and AFTER `is_focus` is measured, so the feed row is
  presented exactly as before (no new beep, no fold change); only the
  `_enqueue_review_alert` call is skipped when the row is resolved. The day
  label is `resolve_discovery_env(bot live env, load_opening_environment())` -
  the ONE definition discovery already uses - via `_regime_pause_day_env`.
- Writes through the STORE (not `FocusService.add`, which would log a "like"),
  stamps the auto-pick marker only when `add()` actually added - a trader's
  unmarked Focus entry keeps its owner AND its chart - and records a
  `regime_pause_auto_focus` row (`env`, `focus_side`, `outcome` in
  `adopted | already_auto | already_trader_owned`). "Not today" and the desync
  repair can therefore reach what it placed (packet R2 provenance).
- **DESK only**, like auto-pick adoption (R1 matrix). Any failure falls open
  onto the old path: the row is queued, never lost.
- Not built, on purpose: no eviction when the name stops holding (the queue's
  15-minute rule is a queue rule; the Focus entry stays until the trader or the
  desync repair says otherwise), and no change to the detector, the sweep, the
  hold measurement or the counter-trend rows.

Tests: `tests/test_regime_pause_focus.py` (18, the two-case rule and every
refusal) and `tests/test_qt_regime_pause_auto_focus.py` (12, through
`add_alert` on the real panel: placement + marker + skipped chart; counter-trend
still charts; blank/neutral admits nothing; the trader's own entry is not
relabelled; a repeat resolves as `already_auto`; AWAY/EVENING/OFF never place;
a store failure queues the row; an ordinary alert is untouched). With the panel
change stashed, 3 of the 12 fail on the assertion and 9 pass - the 9 are the
"stays on the queue" cases, which hold either way.

**Scan of what else fills the queue (same 46 minutes, 124 shown):** D1 rows
67 (`d1_flag_long/short` 41 from the Master AVWAP D1 scanner, `focus_d1_event`
26 - and the Focus list that feeds those had just received **69 machine-adopted
auto picks** at 07:09: 20 "Bullish-day weakness", 13 "RS vs SPY", 36 PDH/PDL
breaks - which raised 102 `focus_d1_flag` rows on 95 names); M5 `lrsi_cross_20`
/`lrsi_cross_50` 25; regime-pause 21; armed chart watches 11. The other primary
chart type is therefore the **D1 flag** (54% of everything shown), with the
LRSI cross (20%) second. Nothing was changed for either - that is the trader's
next call, recorded in `CURRENT_CHECKPOINT.md`.

### 2026-08-27 - Phase 0.9 G-P2.0..G-P2.2: the three presentation follow-ons from the 2026-08-26 live session

**IMPLEMENTED / GREEN.** `plan.md` Phase 0.9 items 1-3, from
`docs/GUI_REDESIGN_PLAN_2026-08-25.md` §15 decisions 9, 10 and 14, all
presentation only: no detector, scorer, alert, queue, scheduler, evidence-stream
or storage behaviour changed, and no read was added or removed.

**The table width rule now has one implementation** (`1fd9e6e`, G-P2.0). §12's
rule - "the widest TEXT column takes the slack, numeric and badge columns keep
their measured width, and the last section is not the only one that stretches" -
lives in `scripts/ui/widgets/data_table.py` as module-level `apply_width_rule`,
with `apply_width_rule_to_table_widget` for raw `QTableWidget`s. Two of the three
pages the rule was learned on do not use `DataTable` at all, so a rule applied
only through the shell would have missed them. `DataTable.fit_columns` routes
through it, so every existing `DataTable` user gets the rule with no per-panel
edit; AWAY Recap's four tables and Weekend Prep - Focus pick review's five call
it directly. A caller may name its text columns; one that does not gets a
MEASURED answer (a column whose every non-empty sampled value parses as a number
is numeric; of what is left the widest stretches, ties to the lowest index), so
the rule cannot fall behind the way a hand-maintained per-panel list would.
Identifier columns get `MiddleElideDelegate`: middle elision through
`QStyleOptionViewItem.textElideMode`, per item, with the full value as the
tooltip - `human_f...tracking`, never `human_foc...`, because the identity is in
the tail and an elision that leaves every row reading the same is a rendering
defect. `measure_column_widths` is the one seam every caller measures through;
it is `resizeColumnsToContents()` today, unchanged in cost, and it is the 7.9% /
115 s site of the 2026-08-26 measurement, so G-P2.3 item 1 bounds it in exactly
one place.

**AWAY Recap is a return surface** (`a5fa6a9`, G-P2.1; §8.3, decision 9).
Charting was wired the whole time and nothing on the page said so, and the day's
only two alerts were scanner status messages with a blank symbol, so the trader's
verdict was "i also cant even check charts from here. kinda useless." Scanner
status rows - the blank-symbol test, because a row with no symbol cannot be
charted whatever its side says - are hidden from the alerts table and COUNTED in
one line, revealed for the session by one click; nothing is deleted, nothing is
muted, and the Alert Center's backing list is untouched with `set_alerts` still
its one reader here. Every chartable row carries a visible `Chart` cell (a plain
item, never a cell widget: a widget per row is the shape the 2026-08-21 fluidity
pass spent a day removing, and an AWAY day can produce hundreds of alerts),
`Enter` on the selected row opens it through an event filter rather than Qt's
per-platform `itemActivated`, and a hint line says so. A symbol-less row renders
muted and italic from the `text_muted` THEME TOKEN - not a per-widget stylesheet,
and Qt style sheets do not reach view items at all - and offers no chart action
by either route.

**The Desk Journal has a keyboard route** (`fd76923`, G-P2.2; §5.3 option (a),
decision 10). The trader could not find the sixth lower tab. `Ctrl+J` selects it
and focuses the composer; the tab label reads `Journal  Ctrl+J`. Bound at PANEL
scope with `WidgetWithChildrenShortcut`, copying `_bind_capture_shortcuts`,
because a `QShortcut` bound inside a hidden tab page never fires - and the
Journal page is hidden exactly when the trader reaches for it. `Ctrl+J` was
verified unbound across `scripts/ui` first (the whole inventory is Ctrl+R,
Ctrl+F, F9, Ctrl+Return and Alt+V/K/S/N), because two live bindings for one
sequence is an ambiguous shortcut and Qt fires NEITHER, silently; a source-level
test now fails if a second binding ever appears. No second row under the charts
and no verb-row verb: the 2026-08-20 one-row rule holds, and a mouse route stays
the trader's to ask for. `alert_center_panel.py` is fenced and the trader
approved this exact diff in chat before the edit.

**Verification.** 5016 passed / 19 subtests, exit 0; smoke 7/7. 37 new tests, and
every one was proved failing on the un-fixed code by stashing the source file and
re-running. No packaging trigger: no new dependency, asset, top-level package or
`__file__` change.

**Owed:** the §11.3 soak against
`ui_stalls_prefix_baseline_2026-08-26.jsonl` before G-P2.3 starts, and G-P2.3 /
G-P2.4 themselves.

### 2026-08-26 night - Phase 0.10 review fixes: the shadow cannot cost the save, and the fence is no longer a hand-maintained list

**IMPLEMENTED / GREEN.** Fable's review of `002f2a3..292e335` returned GO with
two fixes owed before B-4 and one trader decision recorded. All three landed in
`ac9a952` on `claude/gui-phase-0-9`.

**The shadow export is guarded.** `export_setup_tracker_views` wrote the
band-variant CSV as its last statement with no guard, and
`update_setup_tracker_from_scan` runs `save_setup_tracker_payload` AFTER it - so
one malformed setup dict reaching `build_band_variant_stats_rows` would have
aborted the day's tracker save. That is the evidence store costing the thing it
records, which R10 forbids everywhere else in this codebase. The `try/except` +
`logging.warning` wraps the SHADOW write only: every champion export above it is
already on disk by then, and a champion export that fails must still fail
loudly - asserted as its own test.

**The fence is guarded at source.** Seven readers filter on
`_is_band_variant_scenario`, and three of those were found by the parity fixture
rather than by reading the code - so an eighth would not be found by reading
either. `tests/test_band_variant_fence_guard.py` walks the AST of `legacy.py`
and requires every scenario-iteration site to mention the fence inside its
enclosing function or to be named in `ALLOWED_UNFENCED` with its reason. Two
entries, both readers that MUST see the shadow: the stop rebuild on replay
(`_extract_tracker_stop_candidates_from_setup`, which sorts by label so
`VARIANT_*` still lands last) and sealed-record compaction
(`_compact_tracker_setup_record`, which strips the shadow's per-bar event log
exactly as it strips the champion's).

The detector is deliberately wider than the spelling the fence was written
against - `setup["scenarios"].values()`, `.get("scenarios", {}).values()` and a
local `working_scenarios.values()` all count - because a guard that only
recognizes today's spelling is passed by tomorrow's. It finds nine readers where
the narrow `(setup.get("scenarios") or {}).values()` pattern finds six. Proved
against real code rather than a mutation: pointed at `5613eec:legacy.py`, the
tree as it stood before the fence, it reports six unfenced readers. Four
companion tests keep the guard itself honest. It does NOT claim that mentioning
the helper means it was used correctly - a name in a function is not a proof
about its logic, and the parity fixture remains what proves the values did not
move.

**The shadow crosses the four BASELINE exit templates only** (trader decision,
2026-08-26). `_is_band_variant_stop` is the candidate-side twin of
`_is_band_variant_scenario`, kept beside it so the two spellings of "is this the
shadow" cannot drift apart, and `_build_tracker_scenarios` skips experimental
templates for such a stop. The champion is untouched and still crosses all six;
the experimental templates are a comparison framework for the CHAMPION's stops,
and a challenger inside them would be two variables at once. Re-measured:
**9,982 -> 6,524 bytes per new setup** (474 anchor blocks + 6,050 for four
variant scenarios), so **~144 MB -> ~89.5 MB, 15% -> 9.4%** at the live
14,386-setup / 950.2 MB scale, forward only - and 5,739 bytes once sealed-record
compaction strips the event logs. All four baseline templates remain, so the
stats table's per-template pairing is still possible.

Verification: **4995 passed / 19 subtests, exit 0** at `ac9a952`, and **5010
passed, exit 0** on the tip `714f717` once Phase 0.9's `a5fa6a9` (committed by a
concurrent session while this work ran) landed beneath it; smoke 7/7. Eleven
tests added, every one proved failing first. **Owed, unchanged**: T4's three
criteria, >= 20 sessions of forward accrual before T3 counts, and B-4 - which
these two fixes were the gate on.

### 2026-08-26 - AVWAP band challenger: a second formula, computed beside the champion and unable to reach it

**IMPLEMENTED / GREEN. Every T4 gate is OWED and no test discharges one.**
`plan.md` Phase 0.10, governing spec `docs/AVWAP_BAND_VARIANT_STUDY.md`, build
prompt `docs/prompts/AVWAP_BAND_CHALLENGER_OPUS_PROMPT.md`. Branch
`claude/avwap-band-challenger` off `claude/gui-p1-fluidity` at `88a34b7`.

`calc_anchored_vwap_bands` is untouched and frozen (decision 0008). Nothing in
this packet reaches a detector, score, rank, tier, alert, zone arm, Focus list,
review queue or `review_policy.json`.

**B-0 - the formula, replicated** (`002f2a3`).
`scripts/indicators/avwap_band_variants.py`
(`avwap_bands_oneoption_bb20_v1`): an anchored HLC/3 volume-weighted centre with
a 20-close **population** Bollinger sigma as its half-width - the form the trader
pinned on 2026-08-26 against OneOption / Option Stalker Pro. The two halves know
nothing about each other, so the sigma window deliberately reaches back BEFORE
the anchor; that is why the band is already wide on the anchor bar where the
champion's is exactly zero. `indicators/` shape throughout: completed bars in,
immutable aligned tuples out, no I/O, no pandas import, `None` below the
lookback - never padded, never 0.0, never a shorter window.

`tests/fixtures/avwap_band_variant_oneoption_v1.json` freezes OKTA
2026-04-01..06-05 from the durable store **through
`_normalize_daily_bar_frame`**, not an ad-hoc threshold, because that store's
OKTA volumes are mixed-unit. Neither golden row is affected and the fixture says
why. The expectations are the trader's hover readings, not this repo's output:
centre to 0.2% relative (126.78 here against 126.565 there - a consolidated-vs-IB
volume-feed gap), sigma to +/-0.02 absolute (18.039 vs 18.035).

Two discriminators are pinned as arithmetic: the champion's sigma is 0.0 on a
one-bar anchor where OneOption read 10.28, and the killed sample-OHLC form
predicts an upper of 138.09 on 2026-06-02 where the trader read 144.60. The
killed form lives in the TEST, not the module - no live code carries a formula
the study already killed. An AST test forbids the module from importing
`master_avwap_lib` at all.

**B-1 - the hover-comparison table** (`13505d1`).
`scripts/avwap_band_variant_fit.py SYMBOL ANCHOR_DATE [--lookback 20]` prints
both formulas per session since an anchor. Offline; writes nothing without
`--csv`, and then only into `OUTPUT_DIR/reports/`. The champion publishes only
its final bar, so its column comes from calling the frozen function once per
session on a truncated frame - a call, never an edit. An unmeasurable cell prints
EMPTY, because the champion's sigma really is 0.00 on the anchor bar and the two
must not look alike. Live read on OKTA reproduces the study's S2 column exactly.

**B-2 - the tracker shadow** (`5613eec` fixture, `603333b` code).
Golden fixture FIRST: `tracker_record_band_variant_parity_v1` was frozen on the
champion's code BEFORE either fenced file was touched, and it earned its keep.
`runner.build_anchor_band_variant_meta` computes the challenger from the same
frame and anchor index; `current_anchor_variant` / `previous_anchor_variant`
ride `symbol_entry` and the setup record; `_find_tracker_stop_candidates`
appends one `VARIANT_<protective>` candidate LAST with the champion's own
`close_failure_limit`; `master_avwap_band_variant_stats.csv` is written in the
existing export pass and read by a "Band Variant" tab on the Setup Tracker page.

**Appending after the champion's candidates was necessary and NOT sufficient**,
and the prompt assumed it was. `representative_total_r` is picked by label and
did not move - but `_summarize_tracker_setup_outcome` averages `total_r` across
every tradeable non-experimental scenario, and that average reaches
`build_tracker_setup_type_rows` -> `apply_tracker_setup_type_adjustments` ->
`row["score"]`. Measured on the frozen fixture before the fence: `avg_total_r`
-0.0790 -> -0.0755, `tradeable_scenario_count` 8 -> 12, eight summary values in
all, plus `daily_marks[1].scenario_events` 10 -> 15, the short's `setup_status`
CLOSED -> OPEN, and 12 -> 18 rows in the scenario and stats CSVs.
**Trader-authorized 2026-08-26**, `_is_band_variant_scenario` now fences seven
readers. The shadow is still graded - `_evaluate_tracker_scenario_bar` runs for
it exactly as before - its events simply stay off the champion's mark. The last
three fence sites were found by the fixture rather than by reading the code.

Two findings worth more than the code. The challenger's sigma is 1.339 where the
champion's is 0.586 seven sessions after an anchor (2.3x), which is why the
trader's screenshots looked better early. And **"the wider band is stopped out
less often by construction" is only true when entry sits INSIDE the band**: the
fixture's short is entered above both upper bands, the wider sigma pushes the
upper band UP toward entry, and the challenger's stop lands 0.159 away where the
champion's is 0.971 - six times TIGHTER, from the wider formula. T1 and T3 may
not assume a direction.

**Tracker JSON growth, measured**: 9,982 bytes per NEW setup (474 for the two
anchor blocks, 9,508 for six variant scenarios with their event lists) against a
live file of 950.2 MB holding 14,386 setups - about 144 MB, ~15%, if every setup
carried it. It accrues forward only; existing records do not grow until rebuilt.
The study estimated "a few hundred bytes per setup" and was ~30x low.

**B-3 - the D1 overlay, default OFF** (`3abf61d`).
`chart_levels.avwap_variant_levels` builds six sloped lines in the
`avwap_variant` group on the ChartDataService worker, never on the paint path,
anchored on the date the snapshot already resolved so the two lines on one chart
differ for one reason rather than two. **The paint-lines preference file had no
way to express a default-OFF group** - every group defaults ON there on purpose -
so `chart_levels.GROUPS_HIDDEN_BY_DEFAULT` names the exceptions and
`PaintLinesPrefs` gained a `shown_groups` list, letting both defaults live in one
file: an older preference file keeps the group off, the trader's own hidden
groups survive a rewrite beside it, and an unreadable file falls back to the
defaults rather than to "show everything". Four existing paint-lines tests now
assert the amended rule instead of the blanket one.

`indicators.avwap_band_variants` joined `selftest.LAZY_ENGINE_MODULES` - the
first lazy import of it from a path a frozen run can reach. `indicators` was
already in the spec's `collect_submodules`, so no spec edit was needed, verified
rather than assumed.

Verification across the packet: 4968 passed / 19 subtests, exit 0 (baseline
4902); smoke 7/7; `launch_gui.py --selftest` 71/71; spec-drift 17 passed.
**Owed**: T4's three criteria, including >= 20 sessions of forward accrual with
>= 40 finalized setups before T3 counts, and the B-4 backfills (T1 level quality,
T2 playbook re-run) which are the next packet and are NOT started.

### 2026-08-26 - GUI fluidity Wave P1: the desk stops reading its stores on the click

**IMPLEMENTED / GREEN. The live soak is OWED and no test discharges it.**
`plan.md` Phase 0.8, promoted by the trader from
`docs/GUI_REDESIGN_PLAN_2026-08-25.md` §11.1 on 2026-08-26. Presentation and
threading only: no detector, scorer, alert, queue, scheduler, evidence stream or
store changed behavior, and no read was added or removed - only moved.

**Three verified defects, each reproduced at source before it was touched.** The
AWAY Recap called `focus_picks.load_focus_map(side)` against a keyword-only
signature: TypeError on every run, absorbed by a fail-quiet `except` and shown to
the trader as "Focus lists unavailable". The page had therefore NEVER read the
Focus lists, and no amount of the files being present could have changed that.
The same page's adoption-gate line called `mover_state(side, None, None, None)`
against `(side, price, prev_high, prev_low)` - with nothing to compare it could
only return UNKNOWN, which was then rendered as a gate verdict for the symbol. It
now says the gate was not measured here and why, and points at the surfaces that
do measure it; UNKNOWN stays UNKNOWN. And the Desk quick-journal write (Ctrl+Enter,
the one used mid-session with a chart up) dropped the `symbols` field
`MarketJournalService.write_entry` has accepted since R10.H, so the entries most
likely to be about one name were the ones stored with no name.

The shared shape of the first two is worth keeping: **a `try/except` written to
keep a page from crashing had been absorbing a programming error and reporting it
as missing data.** Fail-quiet is right for a store that might not be there; it is
wrong for a call that can never work.

**Weekend Prep now reads on a worker.** `WeekReviewPage.reload` ran
`build_review_learning_state(window_days=7)` plus two RS log scans, and
`FocusReviewPage.reload` ran five CSV/JSONL reads and built five tables of cells,
all inside the click that selected the page - the worst measured stall on the
desk, 8.45 s frozen. `WalkawayPage` in the same file already owned a QThread, so
one `_ReadWorker` now serves all three. Deliberately NOT copied from it: that
page blanks its body to "Running walk-away..." while refreshing. **Clearing a
populated page to announce a refresh destroys the only copy of what it knew**,
most damagingly when the refresh then fails, so the new pages keep last-good
visible and put "refreshing" and any stated error in their own slot. On
`FocusReviewPage` a failed refresh keeps every row: the graded cohorts are the
whole forward record of the trader's own vetoes and likes. That page previously
had no error handling at all - a bad CSV propagated out of the click. Panel
shutdown, which named `walkaway` while that was the only threaded page, now joins
every page.

**The Focus board measures each mover state once per poll, not once per redraw**
(36 repeating stalls, 5.93 s). `_refresh_all` fires on things unrelated to
previous-day extremes - a BounceBot alert, an RS/RW snapshot, a side edit - and
each walked every chip through `AlertCenterPanel.mover_state`, reading the D1/M5
series per symbol per side. Memoized per (symbol, side), discarded by
`refresh_mover_flags`, which is not an arbitrary expiry but the signal that a
newer measurement exists. A FAILED measurement is never cached: a flag is
decoration over a measurement, and one transient miss must not switch it off
until the next poll.

**A stall record now says which click it belongs to.** `scripts/ui/interaction_trace.py`
plus stamping in `StallWatchdog._write`. The watchdog names the frame that held
the GUI thread, which is the wrong question whenever the modal frame sits inside
Qt's own event dispatch and names no application code. The trace is read from the
sampler thread and therefore holds **no lock** - live state is one module-level
tuple replaced whole, because a lock in a diagnostic could stall the thread it
exists to measure. It owns no timer and no thread, and a test PARSES the module
and fails on any call to sleep/wait/start/join/Thread/Timer/QTimer - the
`ScanCycleClock` rule, for the same reason. An empty interaction id means an
idle-desk stall, which is a fact about the stall rather than a gap. Wired at page
select, the Journal inner tab and the chart request.

**Fence discipline.** `alert_center_panel.py` is fenced under the file-scoped
ask-first rule; the trader pre-authorized only the quick-journal symbols
attachment, and the diff there is six added lines and no deletion. The mover memo
was implemented in the consumer rather than at its natural point inside that
file.

**The mover memo moved to its source.** The trader extended the fence
authorization, so the memo now lives in `AlertCenterPanel._measure_mover_state`
as well - the review queue asks the same question once per alert and now gets
the same answer for free. The design came from a measurement rather than a
guess: per (symbol, side), m5 materialization is 0.049 ms and everything after
it is 0.186 ms, so **79% of the cost is memo-able after materialization**. That
is why the key is the identity of the bars measured - session date plus the
length and last timestamp of both series - and not a clock. `mover_state` feeds
the movers-only review filter, which decides what the trader SEES; a time-based
cache would let a name that has just broken yesterday's high stay hidden until
it lapsed. A new bar is a new key.

**System Health stopped rebuilding itself**, and the warehouse readout stopped
reading a network share on the Qt thread. `_fill` and the checks table built a
fresh `QTableWidgetItem` per cell of three tables every 15 seconds - which is
also where the scroll position went, so a trader reading the bottom of the jobs
list was pulled back to the top mid-read, on a timer, with nothing on screen to
explain it. `WarehouseReadoutPanel.refresh` called `ResearchStore.open()` and
`slice_readout()` inline against the DAS lake; that share is known to drop, and
an SMB read against a dropped share blocks until it times out. It was the only
read in the whole audit that leaves the machine. It also blanked its table on
every failure path - an unreadable lake is not an empty lake - and now keeps
last-good on failure while still clearing on a successful empty read.

**The `reload()` audit is complete, and most of what it found is still owed.**
Fourteen panels have a reload/refresh plus file IO; eight own no worker at all.
One was fixed (above); `WeekAheadPage` and `DiscoveryPage` audited clean.
The other eight are named in `plan.md` under G-P1.5, `setup_tracker_panel`
first. Nothing was half-converted: a partial page is worse than an honest list.

**A latent crash the audit found (G-P1.6).** Adding a second
HealthPanel-constructing test file made an unrelated Qt test segfault two files
later - 4 runs in 6. `HealthPanel.shutdown` stopped the panel's timer and left
its audit thread running; that thread emits a Qt signal back into the panel, so
it could fire into a freed C++ object - an **access violation**, which the
`except RuntimeError` at the emit cannot catch because it is not a Python
exception. Reproduced at the committed HEAD with all work stashed, so it
pre-dates this wave. `shutdown` now joins the thread and a `_closing` flag stops
a refresh queued before shutdown (construction uses `singleShot(0, ...)`) from
starting a fresh one after it. **The class is not closed:** any panel that
starts a bare `threading.Thread` and emits a Qt signal back into itself has the
same defect.

Three shutdown lists in this wave named their threaded children by hand and had
each fallen behind: `WeekendPrepPanel` (named only `walkaway`), `ResearchPanel`
(missed the readout), and the `MainWindow` list the readout sits under. Two were
fixed by naming the missing child; the weekend prep one now iterates its pages.

**Every shutdown join is bounded (`e0f78ae`).** Found in live use, not by a
test: the trader closed the window on 2026-08-26, it "froze for a few seconds",
and the PROCESS OUTLIVED THE WINDOW. Four shutdown paths joined their reader
with a bare `worker.wait()`, which has no upper bound - two from this wave
(weekend prep, warehouse readout), two older (journal panel, weekend prep
service) - and the warehouse reader is on the DAS, the one read in the desk
that can block for minutes when the share is unwell, which is exactly when a
trader gives up and closes the app. `ui/read_worker.join_worker` (5 s default)
replaces all four. On timeout the worker is DISOWNED AND PARKED in a
module-level list rather than dropped, because dropping the last Python
reference to a running `QThread` destroys its C++ half mid-run - a crash, not a
leak; these are reads with no side effects and the process is leaving anyway.
`tests/test_shutdown_waits_are_bounded.py` is a source-level guard: a bare
`.wait()` on a shutdown path fails the suite. Tests 4897 -> 4902.

**The proposal is reconciled to the build (docs only, 2026-08-26 evening).**
`docs/GUI_REDESIGN_PLAN_2026-08-25.md` now records Wave P1 as BUILT with
commit ids, replaces its 45-minute fluidity sample with the archived full
pre-fix session (3350 stalls / 1457.5 s; by blocked time, not count), states
what Wave P1 can and cannot be expected to change against it, re-orders the
owed fluidity work by measured time (the two Qt table paths and the growing
Theta refresh first - those are Qt measurement costs, not reads, so a worker
does not fix them), folds in the trader's 2026-08-26 live findings (narrow
columns on every table page; AWAY Recap unusable as a return surface; the Desk
Journal undiscoverable) as a table-width RULE plus page decisions, adds the
build's standing constraints (bounded joins, panel threads, child lists,
never-blank refresh, the fence, the unwired paint marks), and records that
Smart App Control now reads OFF. **One premise of the 2026-08-25 draft was
refuted at source:** its "arm bar contract/source mismatch" - the arm bar is
under the chart by the trader's 2026-08-20 second-pass instruction
(`4c05de5`, "the hotbuttons return"), so the CLAUDE.md/AGENTS.md line placing
it on the Armed tab is the stale one. Flagged for the trader; not edited.
Waves U1-U3, S1 and Snappy P2 remain PROPOSAL. **The trader then authorized
all changes (same evening):** CLAUDE.md/AGENTS.md now say the arm bar is under
the chart, that SAC reads OFF and the source launch stays production by trader
decision, and carry a new rule that chat messages to the trader are written
very simply; `trading_desk.cmd`'s header matches; `plan.md` gained Phase 0.9
(table width rule, AWAY Recap return surface, Desk Journal route, the next
fluidity slice in measured order, a GC MEASUREMENT packet with no scheduling
change). Nothing in Phase 0.9 is built.

**AVWAP band challenger planned, replicated and authorized (same evening, docs
only).** The trader compared their anchored-VWAP bands with OneOption / Option
Stalker Pro's, which are wide from the anchor bar. A one-evening study
(`docs/AVWAP_BAND_VARIANT_STUDY.md`) replicated the vendor's band from three
OKTA hover readings: `AVWAP(HLC/3) ± k · stdev(close, 20, population)` - the
textbook Bollinger σ laid on an anchored HLC/3 centre, no anchor memory
(the anchored sample-OHLC form predicted 138.09 on 2026-06-02; the reading was
144.60). `plan.md` gained **Phase 0.10** (module + fixture, fit script, tracker
shadow stops + stats + panel section, D1 overlay off by default; backfills
after review) with the Opus build prompt at
`docs/prompts/AVWAP_BAND_CHALLENGER_OPUS_PROMPT.md`. Nothing is built; the
champion σ stays frozen (decision 0008) and any promotion would be an
additional level family, never a swap.

Tests 4844 -> 4902, exit 0 (4897 at `49744a7`, 4902 at `e0f78ae`); smoke 7/7.
No packaging trigger. **Owed:** the eight
panels under G-P1.5, the bare-thread sweep under G-P1.6, the
`first_paint`/`chart_ready` marks (which need the receiving paint path
instrumented rather than the emit seam), and **the §11.3 live soak, which is the
trader's to run and which no test discharges.**

### 2026-08-26 - the Phase 0.5 work is on `main`, and the branch chain is retired

**Three weeks of Phase 0.5 development became the trunk.** From 2026-08-04 the work
ran on a nested chain of branches rather than on `main`, because the trader was
running unmerged branch code in production through a scheduled task
(`docs/CHECKPOINT_REVIEW_2026-08-08.md`). The chain ended at
`testing-week-2026-08-24`, which contained every commit of its predecessors, and
`main` was a **strict ancestor** of it. The consolidation was therefore a
fast-forward: 354 commits, 480 files, no conflict, and no merge resolution
performed. `git merge-base --is-ancestor` proved the relationship before the merge
rather than after it.

**The code state on `main` is byte-identical to the state that was verified.** The
only non-`main` content added beside the fast-forward is Markdown, so the
4844-passed/19-subtest baseline recorded for `ed277a7` describes `main` exactly. It
was **not** re-run for this merge: the container this consolidation ran in has no
project virtualenv and Python 3.11 against a project floor of 3.12, so a run there
would have proved nothing. That is a stated limit of this entry, not a claim of
green.

**One unlanded document was brought in.** `claude/trade-analysis-opus-prompt`, a
single additive commit from 2026-08-22, contributes
`docs/prompts/TRADE_ANALYSIS_OPUS_ULTRACODE_PROMPT.md` - the Opus trade-analysis
prompt carrying the scoreboard read, the earliness audit and the AEP DT case. Its
context list still told the reader to load `SOL_PROGRESS.md`, which this repository
deleted when `CHANGELOG.md` and `CURRENT_CHECKPOINT.md` took over that role; the
reference now names the pair. The prompt is classified in `docs/README.md` and is
authorization for nothing.

**Three branches are cleared for deletion, and what they were is written down.** New
`docs/BRANCH_HISTORY.md` records every branch in the chain with its commit count,
date range, tip SHA and disposition, so deleting a merged branch never destroys the
only account of what it held. `claude/ticker-briefs-hardening-imcm8r` (94 commits),
`phase05-r2-focus-gating-strength-board` (150) and `phase05-integration-blitz` (308)
each hold no commit that is not on `main`, proved with
`git merge-base --is-ancestor` against `226fbac`. **The deletion itself did not
happen and is owed to the desk:** the cloud session's GitHub credential pushes but
refuses ref deletion with `HTTP 403`, with no proxy policy denial recorded, and the
GitHub MCP surface has no delete-branch counterpart to `create_branch`. The three
commands are in `docs/BRANCH_HISTORY.md`. `testing-week-2026-08-24` is **kept** -
the active GUI-optimization work continues on it.

**The Alert Center quality packet remains unmerged, by decision.**
`claude/alert-center-quality-packet-5btu3w` (8 commits, tip `57fcf47`, 2026-08-18)
builds the alert-delivery measurement surface `GUI_TRADE_DISCOVERY_LEARNING_PLAN.md`
sec 10.3/17 specify but never built: `scripts/alert_quality.py`,
`scripts/alert_delivery_events.py`, a delivery-capture emit in
`scripts/ui/panels/alert_center_panel.py`, a System Health surface, and its tests.
Two things block it and both are recorded rather than guessed at - it **edits alert
code**, so the file-scoped ask-first rule governs the merge itself; and it adds its
own `docs/ALERT_CENTER_QUALITY_PACKET.md` at the same path where `main` already
carries the *different* historical P1.6 packet recovered from `671ee57`, so a
content merge would silently destroy one of the two. Nothing on `main` depends on
it. No alert behavior changed in this consolidation.

## Revision history

Entries from **2026-08-19 back to the initial system in 2025-11** moved to
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
- The legacy Tk UI remains only for migration compatibility and is not the product
  direction.
- Historical plans and handoffs listed as such in `docs/README.md` are evidence, not
  current execution authority.
