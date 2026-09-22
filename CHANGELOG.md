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

**2026-09-14 local test preparation:** the trader authorized advancing the local sweep checkout to the independently accepted repair integration 6753f9fd plus status documentation. Repair behavior is unchanged from the reviewed build; the independent full suite passed 7968 tests with five skipped (exit 0), ruff was clean, smoke 7/7 and selftest 81/81. Live gates and the main-merge decision remain open; no restart or data repair was performed.

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
`Recent changes` below cover the last two build days; everything older is in the dated
archives named under `Revision history`, the newest being
[`docs/archive/CHANGELOG_ARCHIVE_2026-09-19_2026-09-20.md`](docs/archive/CHANGELOG_ARCHIVE_2026-09-19_2026-09-20.md).
They are evidence and must not be loaded as context.

- **Manual-only Pullback alerts (2026-09-17, working tree).** Claims and Focus names no longer
  pre-arm Pullback alerts; any old automatic row is removed. The chart button is the only arm path,
  and it skips the unnecessary M5-cache read that could stall the desk.

- **Claimed-like current rating (2026-09-17, working tree).** A claimed-only
  D1 row now overlays the latest Master AVWAP scan's existing score inputs for
  its Points rating. Its original claim snapshot stays immutable, and an
  unmeasured symbol remains honest rather than receiving invented data. The
  scan, its outputs, detector, alerts, watchlists and claim store are unchanged.

- **Pullback retest LRSI gate (2026-09-17, working tree).** M15 and M30 SMA retests now fire only
  with a same-timeframe LRSI 80 reversal on the retest bar or either of the two before it. A hold
  alone writes no Pullback alert, feed row, review row or phone push. The reclaim and later-LRSI
  triggers are unchanged; gate #126 remains owed.

### Application, runtime, and data ownership

- **Day Review note markers (TJ-3, 2026-09-19, branch `claude/tj3-note-markers`, merged
  `72647104` into `lead/p033-integration2`).** The trader's own words are drawn on the
  session tape. `scripts/day_review_markers.py` is a PURE builder (no Qt, no store, no
  clock): `placement_for` puts a marker on a bar ONLY when the stamp fell inside that bar's
  own span (`start <= stamp < start + width`, the width read off the tape as the smallest
  positive gap; a daily tape matches by market-local DATE), comparing through `astimezone`
  and never a stripped zone - the journal is UTC, the durable tape is Pacific-local. A stamp
  past the end of the tape is **never clamped onto the last candle**: it is `index: None`,
  `placement: "after_tape"`, KEPT so the page can SAY how many there are (62 of 216 live
  trade legs, 29%, fill after 13:00 Pacific); a stamp in a hole the tape does not draw is
  `between_bars`, the same way; a pre-open, unreadable or absent stamp yields NO marker
  rather than bar zero. `placement_counts` counts on the worker, `name_charts` carries the
  counts per name and `spy_marker_placements` the benchmark's, and the captions read
  `2 marks after the tape - not drawn.` (or say there is no tape at all). `benchmark_markers`
  builds the SPY chart (market notes, Mentor reads placed at `responded_at`, the pasted
  forecast and EVERY trade of the day labelled with its symbol); `symbol_markers` builds one
  name's decisions, claims and trades; `name_charts` pairs each acted-on name with the tape
  that holds it. Eleven `MARKER_KINDS` - the ten plan names plus `prediction`;
  `dislike`/`not_today` draw beside `veto` and `swing_favorite` with `like`; a machine row
  never gets a marker; each trade leg carries its own `marker_id` (`<trade_id>:in` / `:out`)
  while `ref_id` stays the page's selector. `ui/widgets/candle_chart.py` gains ONE ADDITIVE
  overlay family - `NoteMarkers` plus pooled `TextItem` glyphs (hidden, never destroyed),
  `set_note_markers` / `note_marker_count` / `note_marker_position` / `note_marker_at`, and
  `markerClicked(ref_id)` emitted IN ADDITION to the existing signals; with no markers set
  nothing is built, so the Alert Center's charts pay nothing (proven byte-for-byte, PNG
  hash included, at base and tip). `day_review_service.read_day` builds `spy_markers`,
  `name_charts` and `spy_marker_placements` ON THE WORKER at the end of the one read from
  rows it had already read, and the page only pushes them: the SPY chart always, a walk-away
  row click opening that name in ONE reused `CandleChart` beside the tables, a marker click
  selecting that note in "What you said". The page has no D1 toggle and this packet added
  none. Live gate #147 owed. Long form: DESK_INTERNALS "TJ-3".
- **Day Review walk-away v2 (TJ-11, 2026-09-19, branch `claude/tj11-walkaway-v2`, merged
  `a89ec7d5`).** `scripts/real_miss.py` holds `REAL_MISS_V1` (`RUN_ATR` 1.0, `ADVERSE_ATR`
  0.5, one pure `verdict()`; the adverse extreme taken first inside a bar; a missing ATR or
  no completed bar is `unmeasured:<reason>`; import-light so TJ-15's nightly slot calls the
  same function). `market_calendar.decision_session` answers which exchange session a
  decision JUDGED - the session whose New York date the stamp falls on when that date is a
  session day, else the most recent PRIOR session, so an after-close, weekend or holiday
  call maps BACK to the session before it (**TJ-11F, trader 2026-09-19**, reversing this
  packet's original forward rule); `next_session` is its own helper, both on memoized
  holidays. `walkaway_day` gains
  the D1 ruler (daily bars from the decision session's CLOSE over 1/3/5 exchange sessions,
  `pending <date>` until the horizon closes, point-in-time ATR(14)), against-you-first and
  at-the-close in percent and ATR, the row's real-miss verdict and reason code, a fifth
  `earlier_calls` population (the previous five sessions' D1 calls measured to the selected
  session's close, most-ran first), the skill line (liked/claimed vs rejected vs untouched
  names of the SAME scan, `n` + `measured` + `pending` + the ONE Wilson interval by side
  and, where `n` allows, by family, overlaps stated, both the session and `LATELY_SESSIONS`
  windows rendered), one deterministic sentence per table, instrument-aware trade rows
  (option or held past five sessions -> `not judged here`, assignment read off the legs or
  `unmeasured`), and one money line carrying its `n`. **A pooled rate counts a name only
  when its horizon has CLOSED** - an early run inside an open one stays on its row and out
  of the fraction, and open names print as `pending P`. `ui/annotations/store` writes the
  ADDITIVE `decision_session` beside an UNCHANGED `session_date`, stamped
  `decision_session_rule: "judged_session_v2"` (TJ-11F); a stored session WITHOUT that
  marker was written under the struck forward rule and every reader recomputes it from the
  row's own stamp, and no row is ever rewritten or backfilled.
  `load_annotations(..., by_decision_session=True)` is TJ-11's opt-in read; every other
  reader on the desk is byte-identical to base, pinned for `pick_feedback`,
  `review_learning`'s join, the three cohort graders and `daily_recap_reader._decisions`.
  Day Review shows the fifth table, six new columns with full headers, a sentence above
  each table and both skill lines above the grid; the daily-bar read is bounded by three
  named SIZE caps (400 symbols, 150 earlier, 150 untouched, 60-bar tails) in name order and
  never by a result, and the render suspends `ResizeToContents` while filling. Shadow and
  report only: nothing here reaches a detector, score, alert, watchlist, Focus, the review
  queue or `review_policy.json`. Gate #156 remains owed.
- **Day Review instant walk-away (TJ-2B, 2026-09-18, merged into local `main`
  `d3ae3aff`).** `walkaway_day` is a pure projection of the one Day Review worker payload: separate decision times survive source duplicates, claims replay their history, and missing bars or horizons state what was not measured. Claim history resolves with each reader's staged home, so a restart reads the same durable source. The four populations do not alter a detector, score, alert, store, or live desk state; gate #153 remains owed.

- **Day Review durable session bars (TJ-2A, 2026-09-18, merged into local `main`
  `4dd99034`).** Closed sessions fetch decided names plus four
  benchmarks in batched Yahoo M5 requests, store completed regular-hours bars in
  `DAY_REVIEW_DIR/bars/<session>.parquet`, and show the durable SPY tape on a past
  Day Review. Index then bars work stays off the Qt thread; a missing past tape has a
  single-flight worker backfill. Gate #152 remains owed on a real closed session.

- **Overnight AI repair (2026-09-17, merged).**
  `master_avwap_lib.runner.record_theta_picks` is a module-level lazy forwarding
  seam, so a cold `theta_pick_tracker` import cannot cycle through the compatibility
  package while scan callers can still monkeypatch the recorder. The local provider
  keeps its full closed schema and retries exactly once with `json_object` only for
  an explicit HTTP 400 grammar parse/initialization failure; all returned text is
  still validated against the original contract, including `maxLength`. No theta
  score, recorder identity, stage order, output identity, evidence semantics, journal
  selection, timeout, or live store changes. Gate #144 is owed.

- **Phase 0.32 Packet 3 next-test proposal (2026-09-16, merged).**
  `scripts/research_proposal.py` is a validated advisory-only proposal/memo seam: source report
  id/hash and every cited cell must match code facts; unsafe fields/actions, unknown cells/recipes,
  invented numbers and observed-looking proposed thresholds refuse; the closed proposal/status,
  changed-condition and citation schemas also prevent a displayed proposal from claiming
  confirmation or authority. Immutable history and the
  current JSON/Markdown memo live only under the existing AI-store briefs namespace; a failed
  current/memo write restores their prior pair. A model-free current-report refresh updates only
  deterministic trial progress while retaining the immutable proposal source/history. A runner-owned
  session marker spends the one model call before inference, and an absent/empty code-owned recipe allowlist refuses. Compact selection
  is by coverage, repeated questions, unresolved comparisons, readiness and active tests, never
  observed movement or R; it states `narrated K of N` and retains the full cited-cell facts.
  The owned low-priority warehouse outcome pass now publishes the additive versioned
  `entry_quality_window` flat P8 fixed-window dataset from the same completed M5 bars and selectors,
  independently of `outcome_path`; Measured Report reads that session/month-scoped dataset in memory
  into deterministic entry-quality cells (or an honest unknown). `setup_research` is ready only for
  measured cells. Its declared `30_trading_minutes` primary window is selected only by identity
  before the export reads results, while sorted available windows travel with the report/facts.
  Daily Recap > Review and Setup Tracker's compact Next-test route share the same
  published display object, with no tracker ranking, Qt I/O or model call; Review offers
  clipboard-only **Copy test brief**.
  No proposal registers/runs/amends a trial or touches a
  detector, score, alert, ranking, watchlist, Focus, journal, policy or live store. Gates #142-143 owed.
  Final integration verification passed 8,523 tests with 6 skips and 72 subtests, ruff,
  smoke 7/7, source selftest 87/87 and frozen selftest 87/87.

- **Phase 0.32 Packet 1 forward entry quality (2026-09-15, merged 2026-09-16).**
  `scripts/entry_quality.py` adds `entry_quality_forward_v1`, a pure research-only view over
  caller-supplied completed M5/D1 bars. It measures named fixed elapsed/session endpoints with
  coverage and explicit unavailable/pending/partial/no-trigger/invalid/missing states; retains
  gross MFE/MAE/close movement after a hypothetical stop; separates percent, entry-frozen ATR
  and valid-risk R; excludes confirmation-bar wicks; labels reconstructed knowledge; and keeps
  daily touch order ambiguous. It has no store, detector, score, alert, ranking, journal or
  policy effect, and the old `outcome_path` meanings are unchanged. Packet 2 connects bounded
  research-reader denominators; Packet 3 publishes grounded next-test facts from the separate
  `entry_quality_window` dataset. Live gate #140 is owed.

- **Phase 0.32 Packet 2 fair entry comparison (2026-09-15, merged 2026-09-16).**
  `scripts/entry_comparison.py` is a pure reader over supplied Packet 1 measures and the existing
  read-only trial ledger. It normalizes P8's four declared entry variants and existing M5
  occurrences without multiplying exit recipes, retains no-trigger/missing attempts and full
  opportunity denominators, distinguishes shared-triggered movement from all-opportunity coverage,
  keeps scanner/liked/vetoed/trade/unreviewed populations separate, and reports clustered
  uncertainty, distribution and outlier sensitivity. Repeat scans retain opportunity coverage but
  contribute one stable dependency-cluster distribution/pair sample; public adapters refuse a
  recipe or forward entry variant outside the declared ledger-owned axis and retain Packet 1's
  entry rule/version; review comparisons also require matching coverage and entry convention.
  Declaration freezing is in-memory only;
  recipe authorization and family-lifetime multiplicity resolve through `trial_ledger`, so no trial
  is registered, run or amended. It never names a winner below declared floors or for an immature
  trial. No store, detector, score, alert, ranking, journal or policy changes. Live gate #141 is owed.

- **Phase 0.31 remaining WISHLIST integration (2026-09-15, H4/LRSI excluded).**
  `BounceService` owns the unchanged M5 scanner through one below-normal spawned child and
  restarts it after a crash; callbacks cross a bounded queue and commands a pipe. Review
  learning uses `opportunity_identity_v1` (session, symbol, side, timeframe, thesis), reports
  the old-to-new restatement, and leaves ambiguous old veto annotations unmatched; journal
  reviews stamp that same canonical identity while the ordering gate stays annotation-only.
  `trendline_break_retest_v1` is a trader-only, frozen-line, three-completed-D1-bar watch.
  Market Journal writes preserve the subject session plus the write session, and the nightly
  narration stage adds a strict, source-linked explanation of deterministic story packs while
  preserving the last verified file on failure. Trade Mentor saves raw morning words first,
  prepares a validated off-Qt local-AI draft, keeps existing values, shows one grounded coaching
  question, and opens in a resize-persistent 900x820 popup. The frozen bundle now includes only
  the reachable `ai_jobs.market_story_narration` module, pins PySide6 during analysis, and fences
  foreign Codex PDF/image DLLs while restoring Python's own SSL pair. Live gates #134-#139 are owed.

**Daily Recap (WISHLIST 10F + 5F, packet WS-DR, 2026-09-13).** The nav entry `AWAY Recap` is now **Daily Recap** and is offered in EVERY Auto mode. `scripts/daily_recap_reader.py` reads one session from the DURABLE stores - its whole input is a session, a lookback, a clock and a set of PATHS (`RecapSources`, twelve stores), never `center._alerts`, so a restart or a midnight roll cannot lose the record and the same session reads the same in another interpreter. It declares per-source coverage (`rows` / `oldest` / `newest` / `unavailable_reason`) and four frozen views: `worked_today` (the session's M5 outcome rows; `mfe_pct` / `eod_move_pct` are read ALREADY side-adjusted from a store whose side column is `direction`), `recent_swings` (the 1/2/3 lookback is one control - the window is counted on the exchange calendar and the horizon reported is the same number; next-session close, first next-session favorable move and the selected end are three columns and an immature horizon is `pending`, never zero), `my_decisions` (likes quick and claimed, swing favorites, passes, vetoes, not-today and the M5 click-away as separate facts at the `(trade_date, symbol, side, category slot)` grain plus the verdict; repeated clicks link with an occurrence count and credit from the first; `unfavorite` never graded; a retraction removes a favorite; one pass with two codes is ONE decision; WS-5B's report supplies journal R and P&L joined on the CHANNEL) and `rejected_that_worked` (refusals whose side-adjusted later path was favorable, with the trader's own reason and the ADVERSE move beside it). A decision's credit starts at its own timestamp - measured forward from the last completed bar where a pass sidecar makes the timing knowable, `unavailable` where nothing recorded WHEN, never the day's number and never 0.0; `pick_feedback` / `alert_review_events` timestamps are NAIVE on the desk and the reader ATTACHES the desk zone, never strips an aware side. `scripts/ui/panels/daily_recap_panel.py` draws it as four TABS (stacked, the fourth runs past the bottom at 1640x980), each with its cohort/window/n/pending sentence and a sort control limited to the declared measures; the read is on a worker and the page is usable before it returns; a row click goes through `show_board_symbol`, so a recap row is a board look and takes no place in the waiting list (`alert_center_panel.py` untouched; the decision time travels in the row's Time column and tooltip because the chart has no marker seam). The AWAY staged-pick block moved onto the page unchanged (stage, never adopt; the R2 gate shown at click time, never enforced). `away_recap.build_recap`, `away_recap_panel.py` (still constructed and still fed on page select) and `autopilot_today.txt` are untouched, so the phone digest is unchanged; no push was added; `AWAY_RECAP_PAGE_TITLE` is kept as an alias of `DAILY_RECAP_PAGE_TITLE`. `scripts/selftest.py` lists `daily_recap_reader` (imported by name on the worker), selftest 80 -> 81. Tests: `tests/test_ws_dr_daily_recap.py` (the tester's board-door test records only symbol-bearing alerts by lead fix - the scanner's own symbol-less `Scanning paused.` row reaches that door and is discarded on its first line; the builder's added test measures the waiting list itself). Follow-up noted, not built: `desk_bench.py` benches `away_recap` only. **Automatic read at 12:00 Pacific (trader-directed, 2026-09-14):** `scripts/daily_recap_schedule.py` is the PURE decision (`due_session(now, auto_time, last_fired_session)` -> today's ISO date or `None`; due from the configured wall-clock time until midnight on an exchange session, once per session; `parse_auto_time` / `auto_time_from_settings` read `local_settings.json` `daily_recap_auto_time`, default `"12:00"`, `""`/`"off"` disabling and a mistyped value disabling rather than guessing; `next_fire_at` is a label only; Pacific is `America/Los_Angeles`, DST-aware). `DailyRecapPanel` owns one `QTimer` (`AUTO_POLL_INTERVAL_MS` 60 s) started by `MainWindow.showEvent` beside the Mentor's, never in the constructor; `poll_auto_read` -> `show_session(today)` refills the picker, selects today and calls `reload`; a desk started after the hour reads today on its first tick. The noon read is PROVISIONAL by the reader's own labelling (the close is 13:00 Pacific) and the next read of the session - page select or Refresh - is the closed one; `reload` now calls `_refresh_session_picker`, which rebuilds the list only when the newest completed session moved, keeping the selection by DATE, so today loses its "provisional" label after the close without a restart. No scan, fetch, push or write, so it runs in every Auto mode and is outside `auto_scanning_due` (`docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md` amendment 2026-09-14). `clock` and `auto_time_reader` are injectable. Tests: `tests/test_daily_recap_auto_populate.py` (both DST regimes, weekend/holiday, late start, once-per-session, the relabel after the close, the `showEvent` seam); the WS-DR picker test is now clock-aware (it only passed before 13:00 Pacific).
- **Daily Recap repair (DR-REPAIR, 2026-09-15).** `daily_recap_reader` streams the append-only M5 log (coverage still counts every update), retains the latest nonblank event state while keeping blank ids distinct, and displays one whole best measured event per stock/side. Annotation timeframe is preserved: M5 decisions read the reduced M5 state and D1 decisions read their matching session-horizon result, with missing unavailable and immature pending; `Rejected, and it worked` uses the same source split. Pending swings now render with an explicit state and dashes. The reader supplies the compact factual summary; the panel only formats it. `daily_recap_schedule.post_close_due_session` schedules exactly one additional worker read after the exchange-owned regular or early close. No store write, detector, score, alert, queue, Focus, watchlist, or policy behavior changed. Tests: `test_daily_recap_repair.py` plus the reconciled existing Daily Recap contracts. Live gate #132 owed.
- **Workspace memory (2026-09-12, WISHLIST 11):** root `MEMORY.md` is a routing index
  only; `memory/` holds provenance-tagged detail (`people/`, `projects/`, `decisions/`,
  dated notes, prunable `context/`); rules in `CLAUDE.md` "Workspace memory", role
  paragraphs in `.claude/agents/*.md` and `.codex/agents/*.toml`. Recall only, never
  authority; adapted from JumpStarter M1 (`664e083`). Verification gate #93 owed.
- **Codex agent operations (amended 2026-09-15):** removed the project Astra model pin
  at the trader's request; session selection and user defaults choose the lead model.
  Luna remains the default for unspecified helpers; recon uses Luna, builder/tester/reviewer use Terra.
  `docs/AGENT_TEAM.md` owns routing, escalation, isolation and final lead acceptance.

- **Win rate leads every trader-facing SWING surface** (V3, decision 0016 answer
  3). `scripts/swing_headline.py` is the one implementation: win rate first, `n`
  and a **Wilson lower bound** beside it, mean R beside that and never instead of
  it. **Sorting is by the LOWER BOUND** - the raw rate puts a 100%-on-three cell
  above a 62%-on-ninety every time. It reads the TRACKER'S OWN verdict rather
  than re-deriving one, and the average carries its unit, because a column headed
  "Avg R" showing a percent is a number that lies. **That verdict is a
  FAVORABLE-DIRECTION flag, not a win** (ST1, 2026-09-06): the tier outcomes
  `win` column is `side_return_pct > 0`, the sign of a close-to-close percent
  move measured at a SCAN-ROW offset, so `outcome_kind` now declares it
  (`favorable_direction_scanrow_v1`, additive, an empty cell reading as v1),
  `Headline.outcome_kind` decides the words, and every surface fed by that file
  says **favorable** - `62% favorable (>=52%, n=90)`, the setups table's **Family
  favorable %**, the setup-doc sentence and the AWAY digest's ranking line. Only
  a `trade_r` headline says "win rate". Decision 0016 answer 3 is unchanged; the
  number is unchanged; the claim it makes is now the one that was measured.
  `setup_docs.family_record_sentence` renders one line per family AT READ TIME,
  at ONE declared horizon (`evidence_stats.SWING_HORIZON_SESSIONS`, 5 - the same
  one the AWAY digest ranks on), from ONE pass over the tracker.
  **WIRED** (R4 B3, completed by ST2): the AWAY digest ranking, both
  setup-doc renderers, the Master AVWAP setups table's **Family Win %** column,
  the Setup Tracker's **Last 30 Days** tab, and all four Weekend Prep cohort
  tables - each SORTING by the Wilson lower bound. **The Setup Types tab now
  counts its own wins at its own grain** (ST2.2, 2026-09-06): the seam V3 left
  owed, because `master_avwap_setup_type_stats.csv` had no win column and
  `master_avwap_tier_outcomes.csv` cannot be joined at that table's grain (184
  rows over 71 (side, bucket, family, zone) groups, so one joined rate would
  repeat across up to six rows and read as each row's own). **ONE Wilson z**:
  `swing_headline.WILSON_Z` (1.96). `expected_r`'s 1.28 is a parameter of the
  proven-quality score in a fenced scoring file and no trader-facing surface may
  reach for it. **ONE eligible-row reader** (ST1): `scripts/swing_evidence.py`
  declares the policy (`SwingOutcomePolicy`: outcome kind, horizon, knowledge
  basis, maturity rule, window, missingness) and `read_eligible_rows` applies it
  for `setup_docs`, `autopilot_core` and `build_bot_tier_performance_rows` -
  which now DROPS explicit `stale_horizon` rows like the other two, so one file
  no longer gives two answers. The read RECONCILES (eligible + pending +
  sum(excluded) == source rows, every exclusion named) and `describe(...)` puts
  that on the surface in one line, horizon stated in its own unit ("5 scan rows",
  never "5 sessions").
- **Exact exchange-session horizons, versioned beside v1** (ST1 item 2,
  2026-09-06). `scripts/master_avwap_lib/session_horizon_outcomes.py` asks the
  same question of the EXCHANGE CALENDAR: the entry session's close against the
  close ON the N-th session after it, holidays skipped, `sessions_spanned ==
  horizon_sessions` by construction. A missing target bar is
  `no_bar_for_target_session` and NEVER the next bar or a later scan; a target
  past `last_completed_session` is `immature` and lands in `pending`, never in
  the rate. **A REPEAT and a COLLAPSE are counted separately, under their own
  names.** `_scan_factor_row_id` is `symbol:scan_date:run_id` and the desk ran 15
  scans on 2026-08-31, so a `(symbol, scan_date)` key reported 475,492
  "duplicates" over the live history where the truly repeated ids number 75 (300
  at four horizons): `dropped_duplicates` is now that true repeat count only.
  The measurement is the same number for every scan that day - entry-session
  close to target-session close - so the build keeps ONE row per
  `(symbol, side, scan_date, horizon)`, the session's LAST scan row exactly as
  v1 chooses it (so `observation_id` joins **1:1**), and both the row and the
  builder report `collapsed_same_session`, how many scan rows stand behind it.
  **The build is a declared ROLLING WINDOW** of `BUILD_WINDOW_SESSIONS` (30
  sessions, 1.5x the widest window any reader uses), with what falls outside
  counted in `excluded['outside_build_window']`. Measured through the export path
  on the live history (146,367 scan rows): **91,116 rows / 5.3 s / 25.6 MB**,
  against 110,308 / 5.7 s / 30.9 MB collapsed but unbounded and 458,336 / 13.4 s
  / 127.5 MB before the collapse - 91,880 scan rows folded, 300 true duplicates.
  Written to `master_avwap_session_horizon_outcomes.csv` in the same export
  pass, from the daily frames the scan already holds
  (`closes_from_daily_frames`) - it never fetches, and the write is guarded so it
  can never cost the v1 exports or the tracker save. **Shadow: `POLICY_SESSION_V2`
  has no production caller**, and v1 keeps every row, column and value (pinned by
  `tests/fixtures/st1_tier_outcomes_golden.csv`, generated from the pre-fix code).
- **A reconstructed tier never validates a shipped one** (ST1 item 4).
  `swing_evidence.tier_split` counts `tier_source`, every
  `build_bot_tier_performance_rows` row carries `n_assigned_tier` /
  `n_derived_tier`, and `assigned_only=True` restricts a cell to the tier that
  shipped. Measured on the live file 2026-09-06: of the 2,642 horizon-5 rows in
  the last 20 sessions, **2,642 are `derived_from_bucket` and 0 are `assigned`** -
  the 341 assigned rows in the whole file are horizon 1 from 2026-09-02/03, so
  every recent S/A cell today is entirely reconstructed labels.
  reach for it.
- **The tracker exports INTEGER counts at each table's own grain, and the
  weighted rate keeps its own name** (ST2.1/ST2.2, 2026-09-06).
  `build_recent_tracker_setup_family_rows` and `build_tracker_setup_type_rows`
  each write `n_wins` / `n_losses` / `n_flats` (representative closed R exactly
  0) / `n_unmeasured` (closed and unreadable) / `n_pending`, plus - on the
  recent rows - `n_observations` (pre-dedupe), `n_episodes` (post-dedupe, the
  true name of today's `tracked_setups`), `n_symbols`, `n_entry_sessions`,
  `win_rate_closed_unweighted`, `latest_measured_session` and `outcome_kind` =
  `trade_r_representative_exit`. **A count is never rebuilt from a rate**:
  `win_rate_closed` is a RECENCY-WEIGHTED mean and
  `swing_headline.headline_from_rate` used to recover `round(rate * n)` from it,
  which printed `25% (n=4)` on a family that went 2-2. That function stays for
  its legitimate callers - the ones whose stored rate IS `wins / n`, now named
  in its docstring - and the tracker's readers use `headline_from_counts`. A row
  from an export that predates the columns reads `counts not exported yet`,
  never a reconstructed number - and `working_lately.counted_pair` is the ONE
  reader of these columns, strict in both directions, so an exported integer 0
  is a count and only a missing or blank cell is "not exported".
  **The new columns sit at the END of each SHIPPED header**, after
  `namespace`/`status` and after the rank columns. `win_rate_closed`,
  `ranking_score`, `score_delta` and every pre-existing column keep their values
  (goldens). `build_tracker_short_horizon_rows` carries the same counts plus its
  own `latest_measured_session`.
- **Which observation of a thesis becomes the graded episode is a NAMED
  policy, and since 2026-09-06 the default is the repaired one** (ST4, then ST7
  and decision 0019). `scripts/master_avwap_lib/selection_policy.py` owns both:
  `SELECTION_CLOSED_FIRST_V1` (`closed_first_v1`) is what shipped until
  2026-09-06 and stays selectable BY NAME - prefer a record that has CLOSED, then
  the earliest scan date, which reads the OUTCOME to pick the entry, so a later
  rescan that happened to close beats the earlier open row a trader could have
  taken. `SELECTION_FIRST_ACTIONABLE_V2` (`first_actionable_v2`) is
  `DEFAULT_SELECTION_POLICY` since 2026-09-06: an episode is
  `(symbol, side, anchor_date, setup_family,
  attempt_index)`, attempt 1 is the EARLIEST scan row, and a later row opens
  attempt k+1 **only** when the previous attempt's representative scenario
  closed strictly before it (`REENTRY_RULE_V2`) - a rescan of a live attempt is
  one more OBSERVATION of the same episode, and an open attempt stays
  `pending`. `assign_attempts` / `select_episode_rows` are pure and the v1 body
  moved into them verbatim, so there is ONE implementation of the shipped rule
  and it cannot drift from what it is measured against. Under v2
  `_representative_scenario` uses the DECLARED
  `REPRESENTATIVE_EXIT_TEMPLATE_ID_V2` (`full_band2`, the first baseline
  template - the one dict order always practically returned, so declaring it
  reads no outcome) instead of `matching[0]`, and
  `_summarize_tracker_setup_outcome` leaves an open representative at
  `representative_closed_r = None` / `representative_status "pending"` instead
  of substituting `avg_closed_r`. **There is NO scalar exit field on a
  scenario**: the recorded exit is the `trade_date` of the LAST entry of
  `events`, read by `_scenario_recorded_exit_date` and gated on the CLOSED
  status so a partial's leg is never mistaken for an exit;
  `build_recent_tracker_setup_family_rows` stamps it as
  `representative_exit_date` and that is the attempt rule's only input.
  `as_of_session` replays a build (scans after the cutoff excluded; a scenario
  closing after it reads as still running, from its own recorded date, bars
  never re-walked). Every family row carries `selection_policy`,
  `as_of_session`, `n_excluded` and `excluded_reasons`, whose token grain is
  load-bearing: a bare `reason=N` counts records never admitted and sums into
  `n_excluded`, an `_in_population` token counts counted episodes a decision
  deliberately KEPT. The 2026-09-06 decisions are NAMED, never reopened -
  `untradeable` is (c), `expired_unmeasured_in_population` is (b), and the M5
  side (a) is another file this work does not touch. **The switch was the
  trader's** (2026-09-06: *"Yes a trade not yet completed should say pending. A
  second entry after a first close is its own trade yes."*), recorded as decision
  0019 and made the default by packet ST7. v1 is still reproducible byte for byte
  (`tests/fixtures/st4_family_rows_golden.csv`, pinned from `main` before the code
  existed, now read with `selection_policy=SELECTION_CLOSED_FIRST_V1`) and the new
  default is pinned by `tests/fixtures/st7_family_rows_v2_default_golden.csv`,
  frozen on `main` at `68762909` through the explicit v2 keyword before the flip.
  Nothing here promotes a setup or reaches a detector, alert, watchlist, Focus,
  the review queue or `review_policy.json`.
- **The v1-vs-v2 comparison is a frozen artifact that decides nothing** (ST4.5,
  2026-09-06). `scripts/tracker_selection_compare.py` runs
  `build_recent_tracker_setup_family_rows` twice on a COPY at one
  `as_of_session` and writes `selection_comparison_<stamp>.json/.csv`: per
  (side, bucket, family) `n_observations`, and v1/v2 `n_episodes`, `n_pending`,
  `n_wins`, `n_losses`, the UNWEIGHTED win rate, the ONE Wilson lower bound
  (`swing_headline.WILSON_Z`), mean R, `changed`, the rank under each policy and
  the rank move, with a `README` block stating the decision is not taken and
  naming decisions (a)(b)(c). It prints `project_paths.DATA_DIR`, refuses when
  that resolves under `C:\TradingBotData`, refuses a `--tracker` or `--out`
  under the resolved live home or that literal root, and never overwrites a
  stamped output.
- **ONE declared leader, and the banner reads it** (ST2.3, 2026-09-06).
  `scripts/working_lately.py` is pure (no Qt, no file I/O) and owns the
  decision: `select_leader(rows, *, kind, last_completed_session, previous=None,
  min_n=MIN_REPORTABLE_N)` orders eligible rows by the Wilson lower bound on the
  INTEGER counts and returns one of four states - `leader`, `no_clear_leader`,
  `last_reliable_reading`, `no_evidence` - each naming the gate that closed.
  Eligible is `namespace == "live"` AND at or above `min_n` AND fresh within
  `LEADER_FRESHNESS_SESSIONS` (2) EXCHANGE sessions; the crown needs
  `LEADER_MARGIN_LB` (0.05) of clear air over the runner-up. Both numbers were
  declared 2026-09-06 before any forward evaluation and are not tuned to make a
  winner appear. **A study NEVER leads**, whatever its R, and is counted in
  `coverage["studies_excluded"]`; a NEW/RISING pin is a novelty badge and is not
  an input. When nothing is eligible the coverage carries a `discovery_leader`
  the banner prints as `discovery only` - never as a leader, and never beside
  one. The banner used to pick `max(avg_closed_r)` across BOTH namespaces while
  the table under it ranked by the bound, so the two named different families on
  one screen and a three-example study could be crowned. **Every leader surface
  on the page reads this one function** - the banner AND the Summary card's
  plain-English block, which had the same defect three lines higher up. **The
  floor is judged BEFORE freshness**, and freshness is measured on the ENTRY
  session with the rule stated in words (`FRESHNESS_SENTENCE`).
- **ONE Working-lately snapshot, and four surfaces print its id** (ST6,
  2026-09-06; plan.md Phase 0.14 V1 item 4 and V2 item 3). `working_lately`
  gained `EvidenceCell` / `EvidenceSnapshot` / `build_snapshot`, and
  `build_snapshot` is PURE - no file, no thread, no clock that decides anything;
  the caller reads and hands the rows in. **`snapshot_id` is a sha1 over the
  SORTED cell tuples, the declared policy lines and `as_of` and nothing else** -
  not `built_at`, not a source's mtime, not the verdicts - so a half-hourly tick
  cannot move it and the events file stays a record of the evidence rather than
  a log of the timer. Three kinds (`swing_trade_r` from ST2's recent rows,
  `swing_favorable` from ST1's `read_eligible_rows`, `daytrade_held_run` from
  `held_run_score.dimension_summaries`, SIDED cells only - the pooled `all` row
  is the same episodes under a second name) get three verdicts and are **never
  pooled**: `pool_cells` RAISES across kind, side or outcome kind and names the
  axis. Each cell states its side/family, population, outcome version, knowledge
  basis, horizon IN ITS OWN UNIT, window, latest measured session, maturity and
  coverage counts, name/day concentration, statistic and uncertainty.
  **Dependence is answered by REFUSING, not by a new interval**: a cell whose
  top symbol or top session supplies MORE than `CONCENTRATION_LIMIT` (0.5) of
  its own sample cannot lead (`concentrated`), and the multiple-testing exposure
  is PRINTED - `observational leader among K cells` - rather than corrected
  away. `LEADER_PERSISTENCE_SNAPSHOTS` (2) holds a NEW leader until it has led
  in two snapshots with a DISTINCT `as_of`; both numbers were declared
  2026-09-06 before any forward evaluation. Nothing is called proven.
- **The snapshot is built off the Qt thread and persisted small** (ST6.3).
  `scripts/ui/services/working_lately_service.py` is owned by `MainWindow`
  (four surfaces read it), reads the three sources on ONE worker and writes
  `%LOCALAPPDATA%\TradingBotV3\working_lately\snapshot_latest.json`
  (temp-and-rename) plus an append-only `leader_change_events.jsonl`. The GUI
  slot emits and does nothing else. Four triggers - the first `showEvent`, the
  day roll, `scan_service.finished` and a 30-minute timer - fold through ONE
  `SignalCoalescer`, and a build in flight is single-flight. An event is
  `{ts, kind, prior_leader, new_leader, prior_snapshot_id, new_snapshot_id,
  cause, as_of}`, written when the leader NAME moved or the STATE moved with a
  name on one side, deduplicated on
  `(kind, prior_leader, new_leader, new_snapshot_id)` so a restart replays
  nothing. **Cause precedence is REFUSAL-FIRST**: `lost_coverage`,
  `corrected_data`, `window_rollover`, `new_outcomes` - `as_of` moves on nearly
  every build, so checking the rollover first would make the two interesting
  causes unreachable. A `lost_coverage` event carries the SAME name in both
  slots; a session bucket that merely EMPTIED while the source's total held
  steady is a window moving forward, not a correction.
- **Held x Ran carries its identities, and stays name-selection evidence**
  (ST6.2). `held_run_score.Segment` gained `symbols_of_held` /
  `sessions_of_held`, appended in the same breath as each MFE value (parallel is
  the contract - `session_block_bootstrap` refuses when the lists differ in
  length), and `summary()` passes them to `evidence_stats.summarize`. Before
  this the call carried the VALUES ALONE, so `concentration.by_symbol`,
  `concentration.by_session` and the session-block `bootstrap` came back
  UNMEASURED for every held x ran cell the desk has ever shown - the day-trade
  headline had no way to say it was one name six times. The summary also gained
  `n_floor`, `n_symbols`, `n_sessions` and `latest_session`. The cell's
  `statistic_name` is `held_run_score (P(held 30m) x trimmed MFE_R)` and a test
  asserts the dataclass has NO P&L field. `evidence_stats._concentration` now
  rounds a share to ten places rather than four: the share is a DECISION input
  here (compared against a declared 0.5 and hashed into a snapshot id) and a
  display rounding inside a decision is how a cell sitting on the limit lands on
  the wrong side of it.
- **The priority switch is BUILT, and it only reorders** (ST6.5). The
  `local_settings` key `prioritise_working_lately`, default OFF, READ AT SORT
  TIME and never at write time. ON, the M5 bar, the WAITING review list (sorted
  where `_advance_review_queue` picks the next chart, never where a row is
  written) and the Master AVWAP setups table are stably re-ordered by the
  snapshot's own verdict order, ties keeping arrival order and the report's own
  ranking as the secondary key. The tier gate, movers-only and the repetition
  fold are untouched; no row is hidden, parked, muted or dropped; the
  identical-visible-rows test CLAUDE.md owed WITH the switch exists and checks
  the fold counts and the hidden set byte-for-byte both ways.
- **The setups table can be ranked by a POINT system, and it only reorders**
  (trader, 2026-09-08; `scripts/setup_points.py`, pure). Four named parts, each
  from a field the scan already writes on a focus row or the family record the
  panel already injects: `setup` (0-50: the family's Wilson lower bound x 40 +
  expected R clamped to +-1 x 10; an ungraded family scores 0 and says so),
  `sr` (-20..+10: a clean path starts at +10 and loses points per HV level
  blocking / nearby, per cloud level, for a trendline in play, for an MA inside
  1 ATR AHEAD of price, for the nearest level inside 0.5 ATR), `rs` (-15..+15:
  vs SPY / sector / industry, +-5 each, sign-flipped for a SHORT), `bounce`
  (+15 today, +8 by name). The `Points` column (appended LAST, compact width 58,
  tooltip = the four parts and why) shows the total on every row; the
  **Points** checkbox on the setups strip persists `rank_setups_by_points`
  (default OFF, read AT SORT TIME) and, ON, re-orders the favourite /
  near-favourite / high-conviction rows by total with ties keeping arrival
  order and every other row after them in its own order. Applied AFTER the
  Working-lately order, from the rows AS THEY ARRIVED, re-applied when the
  family record lands. `legacy.py` untouched; nothing hidden, written or
  scored. Tests: `tests/test_setup_points.py`.
- **The point system is GRADED, and it corrects itself only on the trader's
  word** (trader, 2026-09-08 *"do higher ranked setups perform better? ... a way
  for the system to correct itself"*; `scripts/setup_points_evidence.py`).
  Every report refresh, the panel's `_PointsEvidenceWorker` (off the Qt thread,
  default-store panels only) appends one row per ranked-bucket setup per scan
  date to `SETUP_POINTS_LOG_FILE` (`setup_points_log.jsonl`, append-only,
  de-duplicated on `(scan_date, symbol, side)`: the RAW four parts, the shown
  total, the multipliers used), joins the whole log to the tracker's outcome
  rows through the ONE reader (`swing_evidence.read_eligible_rows`,
  `POLICY_SCANROW_V1`, the declared 5-session horizon, window = the log's own
  dates) on the same key, and reads TERCILES by total - each with `n`, win rate
  and the Wilson lower bound; the headline is the LIFT (top third minus bottom
  third), refused as "not enough per third yet" under `MIN_REPORTABLE_N`, and
  every part gets its own top-half-minus-bottom-half lift (a constant part has
  none). The grade sentence sits on the setups status row and in the Points
  tooltip. The correction: `propose_weights` writes `SETUP_POINTS_WEIGHTS_FILE`
  (`setup_points_weights.json`: one multiplier per part, `1 + 2 x lift` clamped
  to [0.5, 1.5], proposed only when BOTH halves hold the floor, else 1.0 with
  the reason, plus the grade payload). The desk APPLIES it only when the
  `... > Points: learned weights` menu switch (`setup_points_learned_weights`,
  default OFF) is on - `setup_points.active_weights` reads the file, never
  recomputes - and the tooltip names every multiplier in force. Shadow only:
  nothing reaches a detector, a score, an alert, a watchlist or the tracker.
  Tests: `tests/test_setup_points_evidence.py`.
- **The desk surface, and the AWAY Recap** (ST6.4/ST6.6). A one-line **Working
  lately** strip sits at the TOP of the M5 alerts column - mounted INSIDE
  `M5AlertBar` rather than as a third child of the saved two-pane splitter -
  with every cell in its tooltip, and clicks through to the Setup Tracker. The
  tracker banner renders the SHARED payload and labels its own CSV pass `panel
  read`; the Summary card three lines above it takes the same verdict. Weekend
  Prep prints the line as its HEAD line and only when a snapshot exists (the
  card's eight-line cap holds). The AWAY Recap leads its summary with the line
  and that session's leader changes WITH THEIR CAUSE, lists every ranked swing
  row (there was no top-five cap here to remove - the cap is the PHONE
  digest's), and carries the `alert_cell` + held x ran suffix the M5 row already
  showed in an eighth column inserted BEFORE the chart affordance.
  `autopilot_today.txt` gains `== WORKING LATELY ==` in its EXISTING body:
  **no new push**, AWAY-only routine output, already inside the rule, and an
  absent snapshot is an ABSENT SECTION.

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
- **The Research tab is the builder's surface, EXCEPT its Results page** (V3
  item 5; decision 0016 answer 7 AMENDED 2026-09-06). The nightly fact pack's
  headline gets one line on Weekend Prep's verdict card; the full panel stays in
  Research, which says so on the page - and the page now says it of the other
  eight tabs only. The second sentence still binds: nothing the trader must see
  may live only in Research.
- **Research > Results is the trader-readable full readout** (packet G5,
  2026-09-07; `scripts/research_results.py` + `ui/panels/research_results_panel.py`).
  FIRST tab and the one Research opens on; the other eight keep their order.
  Four populations that are never pooled - **Bot setups | My trades** x **Swing |
  Day trading** - over **Recent 20 sessions | All history | Custom**, the choice
  remembered under `local_settings` key `research_results_selection` and
  defaulting to bot / swing / recent (trader, decision 3 of 2026-09-06). Bot
  cells come ONLY from ST6's snapshot, read through
  `working_lately.cells_from_payload` because `snapshot["cells"]` is compacted,
  and Bot x Swing shows `swing_trade_r` and `swing_favorable` as TWO labelled
  sections that never pool. **The page computes no statistic**: `band_cells`
  borrows `meets_floor` / `n_floor`, `EvidenceCell.concentrated`,
  `working_lately.rank_basis` and `evidence_stats.lately_window`, every number a
  row carries is the cell's own object (pinned by an identity test), and
  `rate * n` appears nowhere (pinned by a source-level guard). Stronger lately is
  the eligible cells in the snapshot's own order, top three; Weaker is the three
  lowest by statistic **among the ones Stronger did not take**, so no cell can
  stand under both headings; Not enough evidence is every ineligible cell with
  its own reason (below the floor with `n_graded`/`n_floor`, concentrated with
  the share, not measured, pending); a study is in no band and is listed under
  its own label. The kind's `LeaderVerdict` state and reason are printed
  verbatim above the bands with `observational leader among K cells`. **My
  trades** splits CLOSED trades by holding period through
  `journal_trade_shape.is_date_only` - a broker row is `unknown timing` under
  BOTH horizons and assigned to neither - reports fees beside net, counts R only
  where the trader's own `planned_risk` is present, names an instrument and
  never a direction, and prints *"no confirmed tags yet - nothing here names a
  setup"* rather than a leaderboard of machine guesses. Both reads are on a
  `ReadWorker`; the snapshot arrives through `set_working_lately_snapshot`
  (`app.py` connects `snapshotChanged` to `ResearchPanel`) or off disk on the
  worker; the detail pane is G4's identity-aware one (a control change takes it
  down, a refresh re-opens the same row from the NEW numbers). No "Chart it":
  `EvidenceCell` carries no example symbols and G5 adds no new read.
  **The window control applies where it can and is disabled where it cannot**
  (fix round): My trades filters its closed trades by `closed_at` through
  `research_results.in_window` (inclusive both ends, the turned-away count
  reported as `n_outside_window`, a trade with no `closed_at` counted OUT of a
  bounded window), and on Bot setups the three buttons are DISABLED because
  each cell was measured over the window its own aggregator walked -
  `window_applies` / `window_sentence` state that from `window_sessions` and
  `as_of` and never a date range. The freshness line names each source by its
  `rows` (`working_lately` stores `path ""` / `mtime null`) and says
  `NO_SOURCES` when none was recorded; a band card reads `N of M shown` for the
  lines it actually printed; money names its currency and a refused total
  prints `resolve_pnl_key`'s own reason; `untagged` is never a row under the
  Confirmed-tag header; a study row is muted like an ineligible one; the
  section text is ONE short verdict line per kind (state and reason, verbatim)
  with the full leader line in the tooltip, and every running-text label is
  capped at G3's reading measure.
- **Weekend Prep has ONE Refresh and a verdict card** (V2 item 2, decision 0016
  answer 10; finished by R4 A13/A14/A18). The click starts each page's own reader
  and returns - measured under 50 ms with the reads stubbed at the WORKER
  boundary - and the five per-page buttons left the layout, as did **Discovery's
  six per-table ones**, which now have a real `reload` (it had none, so one
  Refresh counted the step and built nothing). `week_trades` moved off the Qt
  thread: it was 775 ms of the click. Every table carries the ten-row floor
  through one constant, `TABLE_TEN_ROWS_PX` - **except the nine on Focus
  Review**, which took it off in G1 (below). The card's take rate READS `shown`
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
- **Weekend Prep's Focus Review page is ONE table behind a view selector, with
  a detail pane** (G1, 2026-09-06). Nine `QTableWidget`s each carrying the 260 px
  ten-row floor is 2,340 px of minimum height in the ~2,050 px a 2160 screen
  gives the page - the overlap the trader reported is arithmetic, and the panel
  could not be shown at 2160 at all (page 2,824 px, panel 2,934 px). The nine
  are a `QStackedWidget` behind nine exclusive checkable `QToolButton`s in a
  `QButtonGroup` - **Week's picks, Picks graded, Vetoes, Likes, After-like,
  Passes, Not-today, Said vs did, Said at the time** - with the horizon combo on
  the selector row, VISIBLE only on Vetoes and Likes. The floors come off THIS
  page only (`_ten_row_table` and `TABLE_TEN_ROWS_PX` unchanged for the other
  five) and the ONE VISIBLE table takes the height, so decision 0016 answer 10 is
  kept on the VIEWPORT at 1440 rather than on a minimum height. Beside the stack,
  behind a `QSplitter(Horizontal)` at 3:1, a read-only `QTextBrowser` shows every
  column of the selected row as `header: value` with the long text IN FULL (the
  cell elides at paint time; the pane does not) and clears to EMPTY on a view
  change - never a placeholder. Each view carries a POPULATION SENTENCE saying
  what a ROW is, and no count the render does not already have. **The read did
  not move**: `_read_everything` is still one pass over all nine stores on the
  page's worker, `_on_focus_ready` still fills all nine tables on every render,
  and selecting a view is `setCurrentIndex` plus a cleared pane - no file, no
  worker, no re-render. The chosen view is remembered for the session and
  survives a refresh. **The pane never outlives the read it describes**: it is
  filled from `itemSelectionChanged`, and a render that KEEPS the row count
  leaves the row selected without re-emitting, so every render pass ends in
  `_refresh_detail_pane` - `_on_focus_ready` AND `_on_cohort_horizon_changed`,
  the horizon being the second door onto the same staleness. It re-reads the
  visible view's selected row from the NEW cells and empties when the new
  render could not carry the selection; a render that SHRINKS the table drops
  the selection and Qt re-emits by itself. Clicking the button of the view
  ALREADY shown is a NO-OP (an exclusive checkable button still emits
  `clicked` when checked, and the only thing that click could change is the
  row the trader is reading). A note names a VIEW, never a position - "the
  Vetoes view", "the Picks graded view", because nothing is above anything in
  a stack. Both new widgets are styled by object name in `theme.qss`
  (`QToolButton#WeekendViewButton`, `QTextBrowser#WeekendRowDetail`); the page
  sets no stylesheet. Layout lane: no number, no read, no sort key and no write
  changed, `apply_width_rule_to_table_widget` calls are untouched, and the
  verdict card is still uncapped (gate #49).
- **The Journal's Trades table and Weekend Prep's Tag Week tables name their
  text column** (G2a, 2026-09-07). Both used Qt's default resize mode with no
  `apply_width_rule_to_table_widget` call anywhere in either file, so option
  symbols and tag lists clipped on a 3,456 px desk while most of the row sat
  blank. Journal Trades' `_populate_table` (the tab's ONE render seam) now ends
  with the rule naming `Tags` (`TRADES_COLUMNS.index("Tags")`, a new module
  constant replacing the header literal) as the text column that takes the
  slack and `Symbol` as the middle-elide column; Weekend Prep's
  `TagWeekPage._render` and `_render_missing_risk` do the same for `self.table`
  (`TAG_WEEK_COLUMNS.index("Tag")` / `.index("Symbol")` — the tuple carries SIX
  columns, `Week` having been added after the packet was written) and
  `self.risk_table` (`MISSING_RISK_COLUMNS.index("Tag")` /
  `.index("Symbol")` — `Tag` is that table's only free-text column; it carries
  no description/reason column). Every index is looked up by NAME so a column
  inserted tomorrow moves the lookup rather than silently breaking it; the risk
  table's call runs BEFORE its empty-rows early return so a zero-row render
  still names `Tag`. `_populate_table`'s existing NEEDS_REVIEW tooltip is
  untouched — the rule's own tooltip write only fires where none exists.
  Layout lane only: no number, sort key, read or write moved, and the Trades
  splitter opening at 39/61 instead of its declared 3:2 is a separate,
  unfixed defect (a later Journal packet).
- **Every Setup Tracker tab, the Desk Setups table and the AWAY Recap tables
  name their text column too, so the §12 rule's MEASURED path has no
  trader-facing caller left** (G2b, 2026-09-07). The measured answer is
  content-dependent and content moves: a populated Catch Rate gave the width to
  `sample_caught_winners` while the MISSED winners the tab exists for clipped;
  an EMPTY Controls or Studies tab measured its HEADERS, so `Win % (low)`
  stretched and `Family` sat at its floor; and Human Picks handed `cohort` most
  of a 4K window and pushed its ten measurements off the right.
  `SetupTrackerPanel._make_table` takes `text_key` / `elide_keys` /
  `stretch_last` and resolves every index BY KEY through `_column_index`, which
  RAISES on an unknown name (a literal index is a defect waiting for the next
  column insert - ST2 and M5 each added one this month); the two `fit_columns()`
  call sites are unchanged, so the attribute leaderboard's worker slot picks the
  declaration up as well. **Human Picks is the one table that wants the slack
  left EMPTY**: `apply_width_rule` gains `stretch_last: bool = True`, and with it
  False and no text column named it suppresses the MEASURED auto-pick as well as
  `stretchLastSection`, because leaving the classify path on would hand `cohort`
  the slack by the back door; a NAMED text column still stretches, and every
  existing caller is byte-identical under the default. `DataTable.set_width_rule`
  carries the flag. On the DESK setups table the FULL profile names `setup_tags`
  and the elision is installed as `_KeyLevelElideDelegate`, which inherits
  `MiddleElideDelegate` AND `SetupTableDelegate` — **a per-column delegate
  REPLACES the view's own**, so the rule's plain `elide_columns` delegate would
  have left `key_level` alone without the alternating background, favorite tint,
  selection fill and hairline separator its own row still draws; the setups
  delegate wins `paint`/`sizeHint`, `MiddleElideDelegate` supplies the
  full-value tooltip, and the ONE override is `_text`, because the base
  hard-codes `ElideRight` and a key level's tail is its anchor and retest date.
  The COMPACT profile takes the column delegate off again and is otherwise
  untouched (`COMPACT_COLUMN_WIDTHS`, `_fit_compact_columns` and F9 unchanged,
  pinned by a golden). AWAY Recap middle-elides `Line`, `Trigger` and
  `Cell / held x ran` — whose `held x ran` suffix is exactly what an end elision
  loses — and names `Symbol` on the Focus table. Layout lane only: no number,
  sort key, read or write moved, the rule still runs once per fill after the
  fill, and a golden pins every tracker tab's rendered cells and row order.
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
- **The full thought is READABLE, and the list shows an excerpt** (G3, 2026-09-06).
  The page put the WHOLE text of an entry into a one-line `QListWidgetItem`, which
  the narrow list elided, and selecting it repainted the capture charts and nothing
  else — so a 1,200-character thought was shown NOWHERE. The list label now carries
  `_excerpt` (first line, 90 characters, `…` only when there IS more), and the right
  half is a vertical splitter with a READER above the charts (2 to 3): `thought_meta`
  (session, timeframe, `written HH:MM UTC±HH:MM` converted with `astimezone` to
  `pass_bars.desk_zone()` (G3b, 2026-09-07 — every live row stores `created_at` in UTC,
  and printing the zone the stamp CARRIES read `written 13:36 UTC` for a note typed at
  06:36 Pacific) — a naive stamp is never guessed at and still says "no zone recorded" —
  `[desk]`, the after-the-session marker, the symbols) over `thought_view`, a read-only
  `QTextBrowser` filled with `setPlainText` so the trader's own `<` is never markup.
  **The words carry a MEASURE and the list a readable column**: the reader is capped at
  `_reader_measure` (`averageCharWidth() × 100` or `theme.px(1200)`, the smaller, floored at
  `theme.px(240)`) and left-aligned with the slack on the right — 96 characters a line under
  the desk's theme, not the 400 the full pane gave — and the `lower` splitter OPENS at a
  third of the page, because stretch shares a resize and never the first layout.
  **The reader is filled at the HEAD of `_on_entry_selected`**, synchronously, keyed
  by `entry_id` (`_selected_entry_id` → `_entry_for_id`, G3b item 2 — a row-index
  lookup only lined up with `self._entries` because every row was an entry; a future
  header/grouping row would desync the two silently), before the no-capture guard
  returns and before `_CaptureWorker` is constructed: an entry with no capture is
  still readable, and one entry's words can never appear under another's selection.
  The empty-session branch clears it. The composer opens at four text lines
  (`setSizes`, measured 228 → 106 px at 3456 × 2160) and still drags taller. The
  reader's pixel cap is recomputed on a scale change too
  (`refresh_reader_measure()`, called from `MainWindow._apply_scaled_metrics`, G3b
  item 3 — it used to be computed once, in `__init__`, and never again).
  **Layout only**: the store, `entry_id`, `created_at`, `session_date`,
  `written_after_the_session` and the Desk tab are untouched, and the dated
  newest-first contract and the two-space separator both stay.
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
  **Superseded 2026-09-07: the two sections are gone and the window is ONE flat
  scrolling page** (`ui/widgets/strength_page.py`; Recent changes below) - the
  same reads, one under another, each sized to its own content.
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

- **Compression is measured, shown and calibrated before it is tuned (packet PCT-3, trader
  2026-09-15, `claude/pullback-compression-2026-09-15`).** `summarize_anchor_compression` has
  always computed a 0-3 `compression_score` and three ATR ratios and discarded them;
  `legacy.compression_copy_through` now publishes all four plus `compression_rule_version =
  "anchor_compression_v1"` on the priority row, the `ai_state` symbol entry, the feature row and
  `build_tracker_setup_record` - at BOTH row builders, because the live desk scan runs
  `runner._run_master_impl`'s loop and not `legacy._evaluate_priority_snapshot_for_date` (the
  first build missed that; `tests/test_pct3_runner_publishes_compression.py` drives the real
  runner in a child process with its own data dir and reads the ai_state and the feature CSV off
  disk). The eight new fields join `runner`'s `feature_columns` allowlist (the first scan on this
  build widens the 667 MB `d1_features_history.csv` once, about two minutes). Nothing is stamped
  as measured when it was not: both copy-through helpers answer `None` and omit the rule version
  when a ratio is missing, and `compression_break_recent` travels with its rule version or not at
  all. Pure `scripts/compression_chip.py` is the reader (string-safe against `bool("False")`);
  `SetupTableDelegate` paints an amber `caution` `compressed` pill after the bucket and wrong-side
  chips with one tooltip line - display only, hiding and re-ordering nothing, moving no score. The
  36 MB `ai_state` parse runs on `master_avwap_panel._AiStateCompressionWorker` through
  `ai_state_levels.warm_cache()` behind an mtime+size key (one parse, one coalesced refresh, the
  worker freed on `finished`); the Qt-thread `data_feed.merge_compression_from_ai_state` fills from
  `cached_symbol_compression()` and never opens a file. New read-only
  `scripts/compression_calibration.py` (`cd scripts && python -m compression_calibration --since
  YYYY-MM-DD [--out DIR] [--live]`) counts every compression veto as `joined` / `untracked` /
  `pending`, reads each session's ACTIVE tracker population (entry date to
  `last_replayed_session`, a blank one collapsing to the entry day and saying so), recomputes a
  missing anchor measure through `summarize_anchor_compression` itself, prints per measure `n =
  vetoed / rest`, both medians and a rank-sum AUC over seven measures plus today's
  `compression_flag` hit rate, writes one CSV, refuses the live stores without `--live`, never
  reads a bar after the session, and streams the 1.26 GB tracker at 0.14 GB peak (the SQLite
  mirror is not read, decision 0017). `legacy.evaluate_compression_break_v1` reuses
  `assess_compression_break_context` plus a >= 1.0 ATR-20 bar-range clause
  (`compression_break_recent` / `compression_break_v1_note` / `compression_break_v1`), and
  `setup_tagging` adds `COMPRESSION_BREAK` LAST under the six-tag cap. First live-copy reading:
  213 vetoes (140 joined / 50 untracked / 23 pending), flag hit rate 0.18 with 2,631 false
  positives, every measure's AUC 0.34-0.49. Tests: `tests/test_pct3_compression.py`,
  `_merge.py`, `_runner_publishes_compression.py`, `_calibration_accounting.py`; long form
  DESK_INTERNALS "PCT-3 - compression is measured before it is tuned".
- **A trendline break is a frozen tag and event (packet PCT-2, trader 2026-09-15).** The existing
  `TRENDLINE_BREAK` tag is pinned, not rebuilt. `find_directional_trendline_candidate` now carries
  a stable line id plus both endpoint dates/prices. The saved D1 upgrade report appends exactly one
  `Trendline break` row per `(symbol, side, break_date)` while its champion rows remain byte-identical.
  `trendline_break` is an explicit D1 extension kind, never a Focus pullback: arming reads the compact
  saved report once and persists the complete line plus its offset-aware `generated_at`; incomplete,
  stale-format or timezone-less evidence refuses. Evaluation uses completed D1 closes only, never a
  wick or forming bar, never substitutes a redraw, and collapses duplicate persisted watches to one
  save and one fire. Tests: `tests/test_pct2_trendline_break.py` with a nonempty pre-change golden;
  long form DESK_INTERNALS "PCT-2 - a trendline break is a tag and an event".
- **The M5 window is fetched whole once a day, then extended (WS-SN2, WISHLIST item 4,
  2026-09-13, sweep branch).** `BounceBot.request_and_detect_bounce`
  (`scripts/bounce_bot_lib/legacy.py`) asked IB for `durationStr="5 D"` of 5-minute bars for
  every symbol on every cycle - ~390 bars a symbol, ~120 MB and ~230,000 rows a 25-minute cycle
  on the interpreter lock the GUI needs. It now fetches the whole window ONCE per symbol per
  market-local day, keeps the raw rows (`_sn2_bar_windows`), and every later cycle asks only for
  the bars since the last completed one (`"<n> S"` reaching `SN2_DELTA_MARGIN_SECONDS` past the
  gap so the overlap bar returns) and merges them: measured 342 bars on the first cycle and 7
  per cycle after (49x fewer). The frame the detectors read is the frame a fresh `5 D` fetch
  would have produced - same rows, order, dtypes and index (the merged raw rows go through the
  unchanged `pd.DataFrame(all_bars)`; the `len(all_bars) < 10` guard judges the MERGED result) -
  pinned by `tests/fixtures/ws_sn2_fresh_frames.json`, recorded on the pre-SN2 code at
  b8bdec24. The window is refetched WHOLE on the market-local day roll, on an overlap bar whose
  dt or close moved (`_sn2_same_bar`), on a delta that never reached the kept last bar or that
  carries a session the window does not have, on a gap of a day or more, and on unreadable or
  out-of-order rows. **A bar that was still forming when served never enters the kept window**
  (`completed_bars.is_completed_bar`), so a preview price can never be merged into a later frame
  as final; the packet's literal alternative - refetch the whole window on a forming tail - is
  one constant away (`SN2_FORMING_TAIL_FORCES_REFETCH`, False by lead decision because IB's
  `endDateTime=""` ALWAYS serves a forming last bar, so that rule would refetch every cycle and
  SN2 would save nothing; the tester's assertion was corrected to the shipped invariant).
  `_prune_latest_bars_for_cycle(scanned_symbols=...)` bounds the cache by the scanned set and
  `_sn2_log_cycle_fetch` writes one line per cycle naming full windows against deltas. Steady
  cost ~90-100 MB of kept rows in place of the same amount churned every cycle. Nothing below
  the fetch changed. Tests: `tests/test_ws_sn2_incremental_bars.py`,
  `tests/test_ws_sn2_incremental_bars_builder.py`.
- **Last scan, latest input bar and shown report are three clocks (WS-10A, WISHLIST 10A,
  2026-09-13, sweep branch).** Every Master AVWAP scan writes `master_avwap_scan_manifest.json`
  and one append-only line to `master_avwap_scan_manifest_history.jsonl`
  (`scripts/master_avwap_lib/scan_manifest.py`, `project_paths` constants, temp + rename, shared
  home), called by `runner.run_master` on BOTH the success and the failure branch (the
  `legacy.py` diff is ZERO). It records when the scan ran (`started_at` / `finished_at`, aware
  market-local), how fresh its INPUTS were (`latest_input_bar_session` + `preview_bar_used`,
  asked ONCE of WS-FC1's `daily_bar_cache.last_completed_session`, so a forming bar is a
  labelled preview and never moves the session), and what it published (`outputs`:
  `priority_setups` rows, `theta_puts` = put + PCS rows, `d1_watchlist` symbols), plus
  `universe_size` / `symbols_fetched`, per-source counts and WS-FC1's two drop counters.
  `status` is `ok` only when the scan reached its whole universe, `partial` when it RETURNED
  having fetched fewer, `failed` when it raised - and a failed scan touches no output file, so
  the last good report keeps its bytes and its mtime. `scan_manifest.freshness_line` builds one
  sentence (`Scan ok 12:31 - inputs through Thu 09-10 (D1 complete) - shown: 12:31 report`,
  `Scan FAILED 13:02 - showing 12:31 report (stale)`, `Scan partial 12:31 (940 of 1097
  symbols) - ...`, `... inputs: today preview ...`, `Scan: not recorded yet - no report`,
  `inputs: no completed session recorded`), rendering each stamp in the offset it CARRIES; the
  Setups status row shows it via `_ScanFreshnessWorker` off the Qt thread on the existing
  `refresh_from_reports` path (two `stat` calls decide whether either file moved; nothing parsed
  on paint), and System Health prints the same string as check `master_scan_freshness` (failed
  = unhealthy, partial = degraded, no manifest = unknown, never green; its two paths are named
  parameters of `build_operations_audit` so a sandbox audit resolves nothing to the shared
  home). One dated copy per publishing scan lands in
  `%LOCALAPPDATA%\TradingBotV3\diagnostics\scan_reports\<YYYY-MM-DD>_<HHMM>.json` with
  `{symbol, side, bucket}` rows, capped at `REPORT_COPY_RETENTION_SESSIONS` (30) counted as the
  30 most recent DISTINCT dates; a failed scan gets no copy. `cd scripts && python -m
  master_avwap_lib.scan_replay --symbol X --session YYYY-MM-DD` replays one session read-only,
  prints `DATA_DIR` first, one line per checkpoint (open `<11:00`, midday `[11:00, 15:00)`,
  final hour `[15:00, 16:00)`, close `>= 16:00`, in EXCHANGE time - the Pacific desk's stamps
  are converted so a 09:58 PT snapshot files under midday, the builder's reading), and names
  first eligibility, first publication and the delay in minutes - `no recorded snapshot` /
  `delay unmeasured` where nothing was kept, never a reconstruction. Tests:
  `tests/test_ws_10a_scan_freshness.py` (20), `tests/test_ws_10a_scan_freshness_builder.py`
  (7). Rules: DESK_INTERNALS "10A - last scan, latest bar and shown report are three clocks";
  fields: `docs/BROKER_ADAPTERS.md`.
- **The theta picks are recorded and graded (WS-TH, WISHLIST item 6, 2026-09-13, sweep
  branch).** `scripts/theta_pick_tracker.py` records one row per `(symbol, scan_date,
  play_type)` in `theta_picks.jsonl` (`project_paths.THETA_PICKS_FILE`) from the RUNNER right
  after `write_theta_put_report` - the scan's own output pass, never `legacy.py`'s tracker save
  (`legacy.py` READ, zero bytes changed). A key already present is not rewritten; a repeat
  appearance is its own row and keeps `first_seen_scan_date`; a failed append or a malformed
  row loses the row, never the scan. The row carries the support set as the scan built it
  (SMA_50/100/200, never SMA_20, which `_is_valid_theta_support_entry` drops) with `held` read
  off the LEVEL (`level <= close`) and never off `distance_atr` (clamped to 0.0 for a level up
  to 0.05 ATR overhead); the report's own rank; both scores (`score` = the option's
  `rank_score`, `base_score` = the support score); and the sold strike either way (`strike` for
  a sold put, `short_strike` / `long_strike` for a PCS, never a NULL strike on a spread).
  `grade_theta_picks` writes `master_avwap_theta_outcomes.csv`
  (`MASTER_AVWAP_THETA_OUTCOMES_FILE`) beside the tier outcomes through temp-and-rename: held
  above the sold strike at the EXACT 5th / 10th / 20th exchange session, `held_at_expiry` only
  once the expiry session is complete, `mae_atr` and `first_support_broken` from the session
  LOWS (which is why `closes_for` hands back `{date: {close, low, high}}`), only a support
  HOLDING on the scan date can break; a session the calendar has not reached is `pending`
  (`target_session_not_complete`), never a break; a pick with no option quote is `unmeasured`,
  never a loss; `rs_flag` is `not_measured` (the theta score has no RS term). The overnight
  `theta_pick_grading` slot (`scripts/ai_jobs/theta_grading.py`) is deterministic, calls no
  model, is idempotent, and is APPENDED at the END of the deterministic stage directly after
  `daily_digest` - both slot-order pins gained the one name in that position, nothing crossed a
  stage. `theta_readout` builds support-combo x play-type cells with the hold rate FIRST, `n`
  (first appearances; `repeat_days` beside it, never summed in) and the ONE Wilson bound,
  sorted by the bound, floored on `MIN_REPORTABLE_N`; the tercile grade line reuses
  `setup_points_evidence.Cell`'s wording and refuses under 30 per third. The Setup Tracker gains
  a **Theta** tab (sentence above, grade line below, built once on the read worker). Shadow
  only: nothing reaches the theta scan, score or report, a detector, an alert, a watchlist,
  Focus, the review queue or `review_policy.json`. Tests: `tests/test_ws_th_theta_tracker.py`
  (27), `tests/test_ws_th_theta_grading_slot.py` (5). Long form: DESK_INTERNALS "TH - the theta
  picks are graded, never changed".
- **One RRS pass per cycle (WS-SN3, WISHLIST item 4, 2026-09-13, sweep branch).**
  `BounceBot.run_rrs_scan` walks the universe ONCE per scan cycle and produces the 5m, 15m and
  1h payloads together, where it used to be entered four times - once per timeframe and once
  more for whichever the GUI had selected - each entry re-walking the universe and rebuilding
  every symbol's O(n^2) `_build_intraday_rrs_profile` from the same 5-minute bars (275 s of CPU
  per cycle on 2026-09-08, on the interpreter lock the GUI needs). Everything
  timeframe-independent is measured once; aggregation, alignment and RRS run per timeframe
  inside the walk. `RRS_CYCLE_TIMEFRAME_KEYS` is `("5m", "15m", "1h")` and a 30m GUI selection
  joins the same walk. `rrs_payload_for(timeframe_key)` is the new seam and `latest_rrs_payload`
  IS the GUI timeframe's entry (the same object), so one `rrs_snapshot` reaches
  `rrsSnapshotChanged` per cycle; `_intraday_rrs_profile_for_cycle` caches the profile on the
  SYMBOL's last bar dt (SPY gaining a bar the symbol did not print cannot move it), today's rows
  only, pruned to the scanned universe. Reference-ETF bars are bucketed once per (ETF, timeframe)
  per cycle and the industry map is read from memory instead of re-read per symbol per pass. No
  formula, threshold, universe, ETF alignment or output field changed: the payloads are
  BYTE-IDENTICAL to the four-pass recording `tests/fixtures/ws_sn3_rrs_four_pass.json` made on
  the pre-SN3 code at 204f4640 (never regenerate it; its contract metadata was added without
  touching the recording). Deliberately kept: `_record_environment_focus_history` still runs
  four times a cycle because its `hit_count` feeds `bouncebot.*_hit_count`, a scoring input;
  dropping the now-duplicate fourth call is an ask-first for the trader with golden fixtures
  first. Write-only diagnostics (`rrs_strength_scan.csv`, `rrs_group_strength.csv`, the industry
  map's `seen_count`) lose their duplicate blocks. Tests: `tests/test_ws_sn3_one_rrs_pass.py`,
  `tests/test_ws_sn3_fixture_contract.py`.
- **A forming bar never reaches the daily-bar cache (WS-FC1, WISHLIST item 2, 2026-09-12, sweep
  branch).** `scripts/master_avwap_lib/daily_bar_cache.py` holds the rule both CSV writers now
  route through (`legacy._write_cached_daily_bar_frame` and
  `_seed_daily_bar_cache_from_durable`; the `legacy.py` diff is 19 lines at the writer seam the
  trader's FC1 prompt authorized): a row whose session is not complete in EXCHANGE time
  (`astimezone`, inclusive of the 16:00 ET close - `market_calendar` models no early closes, so a
  half day waits until 16:00, safe never early) is `forming_dropped`, a completed row breaking
  `low <= open, close <= high` is `invalid_dropped`, forming wins over invalid so
  `kept + forming + invalid == fetched`; the filter is vectorised and an internal failure refuses
  NOTHING (WARNING) because a cache that stops updating is worse than one counted row. The run
  manifest ALWAYS carries `daily_bars_forming_dropped` / `daily_bars_invalid_dropped` (even at
  zero, on the failure path too); one INFO line per scan names both, one DEBUG line per dropped
  row. The in-process frame cache holds the FILTERED frame; `fetch_daily_bars`' RETURN value is
  unchanged, so the scan's D1 indicators see what they saw. Repair: `cd scripts && python -m
  master_avwap_lib.daily_bar_cache repair [--apply] [--cache-dir]`, dry run by default, prints
  `DATA_DIR` then the cache dir, refuses a target under `C:\TradingBotData`, removes only an
  invalid or forming LAST row, refetches that session through `fetch_daily_bars_from_yahoo`
  (window widened to reach an old session), temp + rename. Dry run on a COPY of the live cache
  (2026-09-12): 1,988 files, 66 end in an impossible candle (2026-09-11 x55 - an ONGOING defect,
  not the trader's older 100), 64 get a completed replacement, MCW and TERN return no Yahoo data
  and lose the row; **`--apply` on the live cache is the trader's action and was never run**.
  What the bad rows touched (copy of the tracker mirror): 443 records had the bad date inside
  their replay window (270 setups / 169 studies / 4 controls), 443 had band levels recomputed
  from it, 441 were marked on it, ZERO had a fill booked on it (`gap_aware_v2` refuses an
  invalid bar); the next persisted tracker write rebuilds every record, so no tracker repair is
  owed. Two seams reported, NOT edited (ask-first, outside the yes):
  `legacy._persist_durable_daily_bars` still writes the UNFILTERED frame to the durable Parquet
  mirror, and `fetch_daily_bars`' return still carries today's forming bar to the D1 indicators.
  Tests: `tests/test_ws_fc1_daily_bar_cache.py` (tester, 17), `tests/test_ws_fc1_daily_bar_cache_builder.py`.
- **A watchlist edit is a dated event, never a verdict (WS-5D, 2026-09-12, sweep branch).**
  `scripts/watchlist_intent_events.py` (schema `watchlist_intent_event_v1`,
  `WATCHLIST_INTENT_EVENTS_FILE` in the shared home) appends one row per symbol that joins or
  leaves `longs.txt` / `shorts.txt` / `swinglongs.txt` / `shortswings.txt`, carrying the
  OBSERVATION time (aware, market-local), the list, side and horizon, an optional reason that
  is never prompted for, and a `source` that keeps the trader's typing (`trader_edit` /
  `trader_paste`) distinct from the Focus store's injection (`machine_inject` /
  `machine_uninject`) and from a difference merely SEEN at load time after an edit outside
  the app (`observed_external`, stamped at the load, never back-dated).
  `WatchlistEditorPanel._write_symbols` writes the FILE first and appends after, so a failed
  append shows `(intent not recorded)` and costs nothing; a sort and an unchanged save append
  nothing; a re-add is a new `add`. A list the stream cannot reconstruct gets ONE small
  `baseline_recorded` row (comma-joined symbols, count, digest) and no invented adds.
  Membership means interest - never a setup claim, a position or a prediction - and a
  `remove` is not a dislike. Known gap: `autopilot_core`'s auto-populate is a third machine
  writer and is unlabelled (its adds surface as `observed_external`). Read with
  `read_events()` or `python -m watchlist_intent_events tail --list longs`; nothing consumes
  it yet and it reaches no detector, score, alert, Focus list, scanner or
  `review_policy.json`. Tests: `tests/test_ws_5d_watchlist_intent.py`.
- **The day-trade watchlists are wiped after the close (trader 2026-09-15, decision 0020).**
  `scripts/daytrade_watchlist_reset.py`: `longs.txt` / `shorts.txt` are the intraday (M5) lists
  and nothing on them outlives its session. The rule is stateless - `reset_due(now, written_at)`
  names the last completed exchange session (`market_calendar.last_completed_session`) when the
  file holds names and its mtime is at or before that session's 16:00 ET close, and `None` when
  the file was written after it (a name typed in the evening is tomorrow's and survives to
  tomorrow's close). `apply_reset` empties a due list through `autopilot_core.write_watchlist_file`
  (atomic, designated-writer gated) FIRST and then appends one WS-5D `remove` row per name with
  the new source `session_reset` (`watchlist_intent_events.SOURCE_SESSION_RESET`, writer
  `daytrade_watchlist_reset`), so the Watchlist tab's next load reconciles to an empty list and
  invents no `observed_external` removals; a refused write records nothing; a failed append
  costs the evidence, never the wipe; an emptied file's mtime is after the close, so a session is
  never wiped twice. `AutopilotService._maybe_reset_daytrade_watchlists` runs on every 30-second
  tick in every Auto mode, right after `_roll_day_state` - before the weekend short-circuit (a
  Saturday start owes Friday's wipe) and before the open scan - and after a wipe forgets
  `autopilot_written` and logs one line naming each list, its count and its session. The
  `local_settings` switch `daytrade_watchlists_reset` (default ON) turns it off; the CLI
  `python -m daytrade_watchlist_reset` is a dry run unless `--apply`. The swing lists, the
  auto lists (their own day-roll clear), the Focus store and its injection membership are
  untouched (a Focus pick is still scanned through the fast lane; its later un-injection finds
  the name gone and records nothing); BounceBot re-reads the files every cycle and is untouched.
  Tests: `tests/test_daytrade_watchlist_reset.py` (26). Long form: `docs/DESK_INTERNALS.md` "DTR".
- **The M5 scanner breathes and scans the trader's picks first (SN5/SN6, 2026-09-08).**
  `BounceBot._breathe` waits `SYMBOL_BREATH_SECONDS` (0.02 s) on the stop event after each
  symbol in the fast lane and both sweep loops - pacing only, never `time.sleep`, nothing
  produced changes. `BounceBot._fast_lane_order` scans the trader's own Focus names before
  the auto-adopted ones (`focus_picks.load_auto_pick_symbols`, today's markers only, a
  failed read = every name is the trader's); the set is unchanged. Tests:
  `tests/test_sn5_sn6_scanner_breath_and_fast_lane_order.py`.
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
  `strength_board_auto_focus` review event per refresh. **Since 2026-08-31 it lives under the Desk's
  Strength window rather than a left-nav page** (trader request), and **since
  2026-09-07 at the foot of that window's ONE flat page** (`ui/widgets/strength_page.py`:
  no sections, no tabs, every block sized to its content, its two tables sized to
  their rows and capped at `FIT_ROWS_CAP`): sides stacked vertically for the
  column, its own RS/RW half retired (the Alert Center's RS/RW read is a block on
  the same page), and a row click charting into the **Visual Alert Review pane**
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

- **Visual Alert Review colour key (2026-09-17, trader-directed).** Alert rows now carry a
  display-only `alertTone` property: personal D1 level alerts are red, Focus D1 alerts green,
  standard D1 alerts blue, and Pullback fires amber (M15), purple (M30) or cyan (H1). Persistent
  Pullback rows remain in the D1 feed for their existing routing, while a second badge names the
  measured source timeframe. No alert emission, routing, sound, queue, capture, score, detector,
  Focus or store behaviour changed. Tests: `tests/test_qt_feed_like_label.py`.

- **Claimed D1 picks reset in Master AVWAP Setups each market day (2026-09-17, trader-directed).**
  The table now projects only active `claimed_picks.jsonl` rows whose `session_date` is the current
  market session. Old claims remain append-only evidence and remain active for their existing fade,
  grading, Pullback auto-arm and repeat-review paths. Nothing is deleted, expired early or removed
  from any other reader. Tests: `tests/test_d1c_claimed_picks_panel.py`.

- **A claimed D1 like is a pick, and the chart is done (packet D1C-A, trader 2026-09-14, branch `claude/d1c-claimed-picks-build`, reviewed GO).** A CLAIMED like whose horizon resolves to `d1` writes one row into the append-only `claimed_picks.jsonl` (`scripts/claimed_picks.py`, `project_paths.CLAIMED_PICKS_FILE`, identity `(symbol, side, claimed_setup_id)`, a duplicate appending nothing, the fade `focus_picks.FADE_TRADING_DAYS` trading days on `market_calendar.trading_days_between` - due on the ELEVENTH session by the packet's "more than" rule, one later than the Focus fade - with a raising calendar expiring nothing) and that row appears once in the Master AVWAP setups table through the pure `ui/services/claimed_setup_rows.merge_claims`: labelling a matching scan row in place with `My liked trade`, or becoming a new row with a blank Score, its `known_at_claim` measurements and the Points notes naming what was not measured. The horizon is resolved once by `claimed_picks.claim_horizon` from the alert (`is_d1` -> d1; the panel's M5-review flag -> m5; else the setup's registry group, which today has no day-trade group, so d1 or unknown) and never from the capture rail; the rail's stale timeframe is fixed at its own seam - `AlertChartReview.set_alert` passes `bounce.capture_timeframe(alert.timeframe)` ("5m"/"M5"/"5" -> M5, anything else including blank -> D1), so a typed D1 look after an M5 chart no longer stamps M5 and a real `"5m"` alert now gets its M5 sidecar. The pick is SAVED before the chart is retired: `claimPlaced` -> `AlertCenterPanel._place_claimed_d1` -> `_retire_claimed_review` (a separate method from the parking verb: no `remove_today`, no `_parked_symbols`, no Focus drop; one `like_advance` as before), and a failed write keeps the chart, fires `likeRecorded`, and the rail shows `NOT PLACED - claimed_picks.jsonl could not be written; chart kept` through `CaptureRail.set_capture_status` (a listener's line outranks the verb's own for that commit). While a claim is active the same `(symbol, side)` `is_d1` scan alert stays out of the review queue at the one door (`_enqueue_review_alert`, after the parked check, before the M5 branch, chart-watch exempt, an mtime-keyed `_active_claim_keys` cache, the skip counted and stated on the pane); M5 alerts, chart-watches, detection, the feed, the evidence streams and the phone push are untouched and nothing reaches `review_policy.json`. `SetupRow.bucket_keys` and the focus feed's fold make a row answerable to every bucket it belongs to; the setups strip is five independently checkable chips FAV / HC / Near / Liked / All (`qt_setups_bucket_chips`, a one-time migration of `qt_setups_bucket_filter`; the proxy tests `row.bucket_keys & selected`); `setup_points.RANKED_BUCKETS` gains `claimed_like`; a claimed row's context menu offers `Drop my claim` (`DataTable.add_row_action(visible=)`, shown only where `bucket_keys` carries `claimed_like`). A claim writes nothing to Focus or a watchlist (lead decision; the trader may overrule); `docs/CHART_REVIEW_WORKSPACE_PLAN.md` section 7 records the narrow supersession. Tests: `tests/test_d1c_claimed_picks_store.py`, `_route.py`, `_queue.py`, `_panel.py` (73 red from the tester, ten added by the builder). Long form: `docs/DESK_INTERNALS.md` "D1C".
- **M5 on the left, D1 on the right (packet D1C-L, trader 2026-09-14, branch `claude/d1c-desk-sides-build`, reviewed GO).** The Trading Desk's left column is the M5 alert bar ALONE (`m5_column` stays a one-child vertical splitter so every mount, rescue and floor keeps its seam; the ST6.4 Working-lately line is still the first thing inside the bar) and a new `d1_column` vertical splitter holds the Master AVWAP workspace over the swing favorites strip, non-collapsible, the setups taking the stretch and opening at 6:1 (`D1_COLUMN_SPLIT_KEY = "qt_d1_column_split_sizes_v1"`; `M5_COLUMN_SPLIT_KEY` retired in place, never written, the old value untouched). The strip's 2026-08-31 place at the bottom of the M5 list is SUPERSEDED by the same trader; both writes, the `vetted` like-origin, the retraction row, the "took" badge, the day-roll re-derive and every action and signal are untouched. Workspace mode mounts `d1_column` as the third column and tabs mode's "Master AVWAP" tab holds it, so `set_setups_visible`, the open-hidden state, F9, `_setups_restore_sizes`, `_apply_column_floors` and the `_detach_mode_panels` rescue act on the COLUMN, and the strip hides and shows with the setups. Two consequences of the move were repaired in the same change (both builder-added tests pass on the base; they guard the move, not prior bugs): `show_watchlist`'s tabs-mode branch now raises `d1_column`, and the tabs-mode visibility calls come after `addTab` so a parentless column is never shown as a top-level window whose `showEvent` fired the strip's one-shot `firstShown` and emptied the chips. Files: `scripts/ui/panels/trading_desk.py`; tests `tests/test_d1c_desk_sides.py` plus one re-pointed assertion each in `tests/test_st6_service_and_surfaces.py`, `tests/test_qt_m5_alert_bar.py`, `tests/test_qt_desk_layout.py` and the re-pointed `TestWhereItLives` in `tests/test_qt_swing_favorites.py`. Long form: `docs/DESK_INTERNALS.md` "D1C-L".
- **Claimed D1 picks are graded beside the tracker's own populations without a second pipeline (packet D1C-B, trader 2026-09-14, branch `claude/d1c-claim-grading-build`, reviewed GO at ba234f1a).** `scripts/claimed_pick_evidence.py` is one pure reader over four existing stores - the like cohort's picks and outcomes (`ui/annotations/like_cohort.py` -> `human_focus_tracking`, graded nightly by `ai_jobs.cohorts.run_like_cohort_grading`), `master_avwap_tier_outcomes.csv` through the ONE reader `swing_evidence.read_eligible_rows(..., POLICY_SCANROW_V1)` (the `all` window passes an explicit wide `window=`; `end=` alone only moves the right edge), and `claimed_picks.jsonl` - and writes none of them; `load_inputs` is the only file read, `build_comparison` is pure, `render_text` is the ONE renderer printed by `python -m claimed_pick_evidence [--window lately|all] [--as-of]` and rendered as the `My claims` tab on Research > Setup Tracker (`CLAIM_POPULATION_COLUMNS`, `CLAIM_SETUP_COLUMNS`, read on the panel's existing ReadWorker, never on the Qt thread). THE TWO CLOCKS ARE NEVER POOLED: the like cohort measures 5 exchange sessions from the claim day's close, the tracker 5 scan rows; an overlap is NAMED (`of which N also FAV that day`, `tracker_sightings` over the RAW tier rows - the scan's own fact on the claim's session, any horizon, any eligibility - because the graded sample has no horizon-5 row for the newest sessions) and never summed; the tier CSV is parsed once per version through the panel's `_load_csv_rows_cached` (`load_inputs(read_csv=)`); a quick like is excluded and counted once; repeated clicks are one trade (`dropped_duplicates`); HC with no `high_conviction` outcome rows reads `unmeasured: the tracker records favorite_setup / near_favorite_zone only (0 HC rows)` and never 0%; tracker-side pending reads `0 (mature by construction)` and unmeasured `-`, the read-level exclusions printed once. Every statistic is a count or `swing_headline.wilson_lower_bound`, the sort is the bound, the leader names no setup below `evidence_stats.MIN_REPORTABLE_N`; `WINDOW_RECENT` rather than a `*LATELY*` name because `tests/test_r4b_one_lately_window.py` reserves that for `evidence_stats`. Nothing scores, ranks, gates, alerts or promotes; nothing is imported from the journal. Tests: `tests/test_d1c_claim_grading_reader.py`, `_tab.py`, `_cli.py` (41 red from the tester), `_render.py` (6 from the builder). Long form: `docs/DESK_INTERNALS.md` "D1C-B".

**One Watchlist on the Trading Desk, and the two pages it replaced (WISHLIST 10G, packet WS-WL, 2026-09-13).** `scripts/watchlist_views.py` is the one PURE reading - `build_watchlist_rows` takes the four plain lists, the Focus store, today's swing favorites, the journal's open trades, the WS-5D intent stream, the M5 board, today's decisions, the armed alerts and the per-broker last sync and returns `WatchRow`s keyed `(symbol, side)`; it opens no store, writes nothing and reads no clock it was not handed, and `filter_rows` gives the five views (`My watchlist | M5/TC2000 | Swing favorites | Open positions | All`) as FILTERS over that one row set. **Presence on a shared list is not authorship**: `FocusPickStore.add` injects into `longs.txt` / `shorts.txt`, so `manual` is the WS-5D stream's answer (the newest `add` names its writer; `machine_inject` is not the trader) and, where the stream never saw the pair, whether a LIVE Focus pick explains the injection; `first_seen` is the earliest add the stream can VOUCH for and an `observed_external` add leaves it blank. Rows sort by symbol then side, never by source - WS-10B's `adoption` rides along as a label. `positions` is a TUPLE (four accounts; folding two would hide one or invent a sum), `horizons` a frozenset, an option's `exposure` NULL ("not measured"), and a seventh source `alert` exists so an armed price alert on a name that is on no list still has a row. A position is a **read-only projection**: stale when the last VERIFIED import run (`import_runs.status='OK'`) is older than the previous session's close - shown greyed, never removed - a CLOSED position leaves only after a verified refresh, and nothing here arms an alert. `ui/services/watchlist_tab_service.py` is the one owner (the StrengthBoardService pattern, owned by `TradingDeskPanel`, aliased on `MainWindow`): the Qt thread freezes the Focus store into a `FocusSnapshot` (it is a WRITER) and gathers the cheap stores, the worker named `watchlist-tab` publishes those rows and only then reads the Journal - and it **never creates or migrates** the journal, refusing while `store_needs_preparation()` is true (found by the full suite: `JournalStore` builds the schema on construction). `ui/panels/watchlist_tab.py` routes every verb to the owner it already had (`WatchlistEditorPanel` for a manual add/paste/removal, `FocusService.remove_everywhere`, a swing-favorite RETRACTION row, `FocusPickStore.restore_faded`, `PriceAlertService.save_entries`); a position row has no Remove, a paste reports duplicates and never doubles a name, Disarm keeps the entry (A2), and removing the trader's own name leaves a Focus injection on the same list to ITS owner. The **Chart Review** and **Focus Picks** nav pages are retired - both panel classes stay and are still constructed, every old action is inventoried in the WL DESK_INTERNALS entry (Chart Review's Alt+E setups drawer is the desk's own Setups tab) - with `Ctrl+L` rebound once at the tab's scope (a tab raised with `setCurrentWidget` leaves focus on the tab BAR, so raising also moves focus into the page), and the Journal's "Positions on the Watchlist" button is a nav call to the Positions view. `project_paths` gained `FOCUS_SWING_LONGS_FILE` / `FOCUS_SWING_SHORTS_FILE`. Open for the trader: the tab strip now reads **Watchlist** (this) beside **Watchlists** (the raw file editors), and the desk still opens with the setups column hidden. Tests: `tests/test_ws_wl_watchlist_tab.py` (the manual-removal test expects the Focus injection to remain, by lead fix).
**H1 retester watch (WISHLIST 10C step 1, packet WS-10C, 2026-09-13).** The arm bar carries an **H1 retester** button (`chart_watch.WATCH_KINDS["h1_ema_bounce"]`), the one watch kind on that surface that is NOT session-scoped. The rule is frozen and versioned as `h1_ema_bounce_v1` in `scripts/indicators/h1_ema_bounce.py`: completed, session-aligned H1 bars aggregated from the desk's cached M5 dicts by a pure COPY of the intraday engine's `_closed_h1_bars` (no `bounce_bot_lib` import - it drags ibapi and ~1,050 modules, and its own takes IbBar objects), a touch within 0.25 ATR of the 15-EMA, a reclaim on the LAST completed bar by 0.10 ATR inside three bars with the EMA sloping the trade's way over five, invalidation on a close 1 ATR through the line, `ambiguous` for a same-candle touch-and-reclaim, a 45-bar warm-up and 24 h staleness both answering "not measured", an invalid candle skipped and counted. `ChartWatch` gained persisted `watch_id` and `reason` (absent reads blank); `PERSISTENT_WATCH_KINDS` is the one name for "not session-scoped", read by `load_chart_watches` AND `watch_is_stale` (the 30 s M5 poll would otherwise delete the watch a minute after the restart it must survive), with a 10-TRADING-day life through `armed_alert_expiry`; `armed_at` stays naive in the store (its convention) and the rows that leave it are aware. `_poll_d1_event_watches` evaluates the kind at its head (before its empty-list return), fires ONE event carrying every measured reason and both bar times, then disarms; the event is an ARMED event on the D1 feed (`CHART_WATCH_TAG`, timeframe D1), never a detector alert and never on the M5 list, and it reaches the phone through `PriceAlertService.notify_armed_watch(*, watch_id, title, message)` - the existing armed price-alert sender, in every mode (the recorded push exception, `docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md`), de-duplicated by watch id, pushed BEFORE the alert is drawn. When the cached M5 window is short of the warm-up (the desk's `m5_chart_bars` is one 5-session RTH window, ~35 completed H1 bars; `MASTER_AVWAP_INTRADAY_BARS_DIR` has never been written), `scripts/h1_history.py` fetches that ARMED symbol's hourly bars through yfinance on its own daemon thread - LEAD RULING 2026-09-13, the trader may overrule; the alternative is a wider M5 window for armed symbols in `bounce_bot_lib` (ask-first) or a `v2` rule on less history - zero IB traffic, the cache primary, at most one fetch per completed H1 bar, completed bars only, `astimezone` never a stripped offset; the Armed inventory names the source (`H1 from cache` / `H1 from yfinance`) or says `not measured (N of 45 H1 bars[, yfinance unavailable])` through a new optional `watch_note` hook on `ArmedWatchList.set_watches`. A watch is entry timing, never a claim: it grades nothing and reaches no detector, score, tier, watchlist, Focus list, review queue or `review_policy.json`; `H1_ALERTS_RETIRED` keeps its four mentions; `chart_watch.ANY_BOUNCE_KINDS["h1_ema15"]` is older prior art with a different rule, untouched. Rule sheet: `docs/M5_SIGNAL_ENGINES_PLAN.md` section 10. Tests: `tests/test_ws_10c_h1_retester.py` (26; the arm-bar test enables the bar for a symbol first, by lead fix) and `tests/test_ws_10c_h1_retester_builder.py` (18).

**H1 backup history stays FRESH (repair RV-H1-HISTORY, review blocker B1, 2026-09-13).** Three corrections to the WS-10C yfinance H1 fallback, none touching the frozen rule sheet `h1_ema_bounce_v1`. (1) The need is measured on the PRIMARY series, never on the one that was chosen: `alert_center_panel._h1_bars_for_watch` asked for a refresh only when the CHOSEN series was short of the 45-bar warm-up, so once the fallback held 45 bars nothing asked again while the desk's own window stayed at ~35 - the watch was judged on ageing bars until the rule's 24 h `STALE_AFTER` answered "not measured" for good; `chart_watch.h1_bars_for_watch` returns the yfinance series only when the primary is short, so that source IS the short answer and the panel asks on it (`source == H1_SOURCE_YFINANCE or len(bars) < WARMUP_BARS`), no second aggregation pass on the Qt thread. (2) The refresh cadence is a completed SESSION-ALIGNED bucket, not the wall-clock hour: `H1HistoryCache.request` keys its refusal on `h1_history.last_completed_h1_bucket` (open-relative buckets 06:30, 07:30 ... 12:30 market-local, the last 30 minutes long and closed at the bell, through `market_session`'s open/close helpers); the clock-hour key refetched twice inside one bucket and every hour all evening. A fetched bar is admitted through `h1_history.h1_bucket_end`, still via the ONE `completed_bars.is_completed_bar` but over the bucket's own span, so the two series agree at the close; zones stay CONVERTED with `astimezone`, an aware `now` included. Known limit (reviewer advisory): `market_session` is not calendar-aware, so a weekend day still allows ~7 refetch attempts for a short-primary armed symbol (old code: 24 a day, every day); no surface number is wrong. (3) A refresh that fails after a success keeps the bars and says they STOPPED: the armed health cell reads `H1 from yfinance (stale - last refresh failed)` through `H1HistoryCache.last_refresh_failed`, retried at the next completed bucket; `unavailable` keeps its meaning (nothing was EVER fetched) and its `not measured (N of 45 H1 bars, yfinance unavailable)` string. Tests: `tests/test_rv_h1_history_refresh.py`, `tests/test_rv_h1_history_staleness.py` (six written red by the tester, proven red again on the restored files by the builder and the reviewer); one corrected FIXTURE in `tests/test_ws_10c_h1_retester_builder.py` (`test_the_fallback_fetches_at_most_once_per_completed_hour` crosses a bucket at +46 min instead of +30; its intra-bucket property now lives in `test_two_asks_inside_one_session_bucket_are_one_request_not_two`).

**A new arm never fires on an old bounce (repair RV-H1-ARM-TIME, review blocker B2, 2026-09-13).** `h1_ema_bounce_v1` anchors its verdict at the LAST completed bar and knows nothing about arm times, so a series that already held a finished reclaim fired the instant the trader armed (the review's reproduction: confirm bar 11:30-12:30, `armed_at` 13:30, `new_alerts 1 watches_left 0`). `chart_watch` now fences the EVENT, not the series: warm-up keeps every bar, and the rule's `confirm_bar_dt` (the reclaim bar for a confirmation, the closing-through bar for an invalidation) is eligible only when its END is strictly after `armed_at` - the existing armed-watch convention (`_evaluate_extreme`: `_bar_end(bar) <= armed_at` is pre-arm), inclusive on the pre-arm side, the bar end from `h1_history.h1_bucket_end` so the short 12:30 bucket ends at the bell; a candle FORMING when the button was pressed is post-arm once it completes. A pre-arm confirmation or invalidation comes back from `evaluate_h1_bars` as `H1_PRE_ARM_REASON = "pre_arm"` (a `chart_watch`-level verdict, never an indicator reason), so `_poll_h1_bounce_watches` leaves the watch armed and writes no `watch_fired` / `watch_invalidated` row, no push, no alert; while a pre-arm closing-through bar sits inside the rule's age window the rule keeps saying `invalidated` and the watch simply waits. The comparison ATTACHES the desk's market-local zone to a naive stamp and keeps an aware one as the instant it is (`_comparable_moments`, the `autopilot_core._gate_moment` pattern; never `chart_watch._naive`, which strips - an arm written three hours west would otherwise read three hours EARLIER and turn a pre-arm arm into a post-arm one). `armed_at` round-trips through `chart_watches.json` unchanged, so a restart is not a second chance; a disarm + re-arm is a new `watch_id` with a new `armed_at`. **The fence fails CLOSED** (lead ruling on the reviewer's advisory): an event whose bar or arm cannot be dated is not the trader's (`h1_event_is_post_arm` returns False; unreachable today, pinned by `tests/test_rv_h1_arm_time_fails_closed.py`). Tests: `tests/test_rv_h1_arm_time_fence.py` (14); six WS-10C tests had their `armed_at` FIXTURE pinned before the golden bounce through `pin_armed_before_the_golden_bounce`, no assertion changed, and the reviewer proved the pins neutral (the old code still passes 72/72 with them).

**The armed-watch phone push leaves the Qt thread (repair RV-H1-PHONE-WORKER, review blocker B3, 2026-09-13).** `PriceAlertService.notify_armed_watch` called `push_notify.send_push` inline while its caller was the GUI poll, and `push_notify`'s HTTP timeout is 10 s, so a slow ntfy endpoint held the desk for up to ten seconds per fire (the review's reproduction: `send_push_on_qt_thread [True] call_seconds 0.202`). Dispatch is now synchronous and delivery is not: on the Qt thread the method makes only the cheap decisions - the engine check and the watch-id de-duplication, where the id joins `_announced_watch_ids` BEFORE the dispatch (under a lock shared with the thread list) so a repeat in the same tick is refused without waiting for the first send - hands the send to a one-shot daemon thread the service OWNS and tracks (`armed-watch-push`, the `check_now` pattern; an armed watch fires once and disarms, so a standing consumer would idle for days), and returns `{"ok": True, "queued": True, "watch_id": ...}`. `ok` now means "accepted for delivery by the one armed sender"; the refusals are unchanged. The outcome comes back the way `_notify` already reports one (`_last_push_error`, the `ARMED WATCH ...` log line, `statusChanged`); a transport that raises is logged on the worker, never lost. `shutdown()` joins what is in flight against ONE budget, `ARMED_PUSH_SHUTDOWN_WAIT_SECONDS = 2.0`, then returns; the threads are daemons; and because a worker's `statusChanged` is QUEUED to the GUI thread by Qt, the joining thread emits the final snapshot itself when something was pending. The feed row is drawn while the phone is still answering; no `auto_mode` gate is added - DESK, AWAY, EVENING and OFF all deliver through the one armed sender (`docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md` amendment corrected in place). Tests: `tests/test_rv_h1_phone_worker_delivery.py` (10; nine written red by the tester, proven red again by the builder and the reviewer; no test can reach ntfy).
- **The wrong side of the AVWAPE is shown and never hidden (WS-WS, WISHLIST item 9,
  2026-09-13, sweep branch).** `scripts/avwape_side.py` is the one rule, pure and shared:
  `wrong_side(side, close, avwape)` is True for a LONG under the current anchor or a SHORT over
  it, with a tolerance of 0 (a close exactly on the line is the right side) and `None` whenever
  the side or a number is missing, because an unknown is never "wrong". The scan rows the desk
  reads carry neither price (`current_close` / `current_avwape` are tracker `feature_snapshot`
  fields, `legacy.py:5132-5133`; 0 of 435 live feed rows carry them), so `read_row` prefers the
  two numbers when a caller has them and otherwise reads `current_band_zone` (top level or
  `setup_candidate.trigger`, never `favorite_zone`), and `tooltip_text` never prints a price it
  did not read (`LONG below AVWAPE 412.50 (close 409.10)` when priced, `LONG below AVWAPE (band
  zone LOWER_1 to VWAP)` from a zone). `ui/widgets/setup_delegate.py` paints a `wrong side` chip
  in the `caution` token after the bucket chip (`_chip` returns its rect and takes `after=`;
  `sizeHint` asks for the width because `fit_columns` measures the delegate; below
  `_MIN_CHIP_WIDTH` - the compact profile's 96 px bucket cell - the chip is not drawn and the
  tooltip still carries it), and `autopilot_core.render_away_report` appends ` [wrong side]`
  after the symbol and counts them under the list (`N wrong side of the anchor (shown, never
  hidden ...)`); a raising reader costs the tag, never the digest. On a copy of the 2026-09-11
  feed: 342 right, 93 wrong, 0 unreadable. Display only: no detector, score, gate, alert,
  watchlist, Focus or `review_policy.json` change, nothing re-ordered or filtered - hiding a
  wrong-side row, the previous anchor, and the other surfaces (M5 list, chart review, Focus)
  stay the trader's open questions. Tests: `tests/test_ws_ws_wrong_side.py` (58).
- **CH-SYM symbol ownership repair (2026-09-14; loaded into the local sweep checkout).** A chart switch clears the previous
  symbol's retained D1/M5 snapshots and chart state before reading the next symbol.
  The WS-CH history merge therefore keeps older bars only for the same symbol;
  missing new-symbol M5 data cannot become a foreign D1 preview or an old quick-fill
  price. Cached snapshots for the selected symbol still render immediately.
  Tests: `tests/test_chart_symbol_isolation.py`; acceptance and delivery state are
  recorded in `CURRENT_CHECKPOINT.md`. No detector, score or provider change.
- **The chart's bars and the chart's view are two different numbers (WS-CH, WISHLIST 10H,
  2026-09-13, sweep branch).** `chart_snapshot.D1_HISTORY_SESSIONS` (1,000, about four NYSE
  years) is how far back a daily payload REACHES and `D1_DEFAULT_SESSIONS` (90) is how many bars
  it OPENS on; the durable parquet store always held the years and `build_d1_snapshot` always
  computed indicators over the full history before slicing, so this costs one longer slice of
  bars already in memory and NO provider request. The payload carries `oldest_available` (the
  oldest bar drawn) and `history_truncated` (the store holds more), capped by what the store has.
  `CandleChart.set_data(..., initial_view_sessions=N)` holds every bar and frames the tail, so
  panning left reveals the older ones with no request; the y-range comes from the VISIBLE window
  while the log/linear decision still asks every bar. `chart_levels.build_d1_levels` gained
  `price_range_bars` and `ChartDataService` passes the INITIAL VISIBLE window, so
  `horizontal_levels`' price filter and clutter budget behave exactly as before and panning does
  not recompute levels (lead ruling). `SymbolSnapshotWidget._start_d1_backfill` is untouched by
  design - it still sizes its stale-store catch-up off the host's `d1_sessions` (260 / 754
  calendar days), so one chart click never asks a provider for four years. On M5 a **Load
  older** button on the legend row adds two sessions through the same in-memory
  `bot.m5_chart_bars(max_sessions=n)` read, capped at ten per symbol per desk session; the
  chunks overlap so the merge CUTS at the fresh chunk's first bar, the view is preserved by
  CANDLE identity (`visible_bar_span` / `restore_bar_span`) because older bars arrive on the
  left, a stale symbol's result is dropped, and a raising provider costs the older bars and never
  the chart (`older bars unavailable`). The pan-left trigger was deliberately not wired: a pan
  that fetches is a fetch on the paint path. `provenance_state` prints `D1 back to <date>` with
  `(more behind)`. H1/H4 were NOT built - the desk draws neither (the H4 resampler is
  `resample_intraday_bars_to_4h` at `legacy.py:28235`, read only). A consequence: the shadow
  AVWAP band challenger lines now draw for anchors older than 90 sessions, correctly anchored
  (display only). Measured: 1,000 candles + 14 overlays cost `set_data` 24-31 ms and a paint
  15-22 ms; one built snapshot is 701 KB against 75 KB at 90 sessions, so
  `chart_data_service._LAST_SNAPSHOT_CAP` (60) means ~41 MB per chart service - documented, cap
  unchanged, the trader may lower it. Tests: `tests/test_ws_ch_chart_history.py` over the
  1,300-session golden `tests/fixtures/ws_ch_chart_history_v1.json`.
- **The alert feed diffs itself instead of rebuilding (WS-SN4, WISHLIST item 4, 2026-09-13,
  sweep branch).** `ui/panels/alert_center_panel.py` states what the feed should look like ONCE,
  in `_feed_target_rows`, and both paths read it: `_sync_feed` reconciles the rows on screen
  against that target - destroying what is gone, inserting what is missing, restyling what
  changed and leaving every other row the SAME widget at the same position - and
  `_rebuild_feed` builds every row from it. A veto (`_ignore_alert_symbol`) and the coalesced
  `focusChanged` refresh call the diff; the rebuild is kept for the whole-feed decisions (the
  minimum-tier switch, the day's Clear, `_unpin_d1_focus` - the obvious fourth diff case for a
  follow-up). Measured offscreen on 250 M5 + 100 D1 rows: a veto 220.1 ms -> 8.5 ms, a
  coalesced focus flush 223.7 ms -> 6.2 ms, neither constructing a row widget (live 2026-09-08:
  4.0-4.1 s and 24.2 s). The parity exposed two rebuild defects, both fixed: it was not
  fold-aware (a repeated name drew one row per entry at the newest position; the target keeps
  ONE row per (symbol, side) at the OLDEST qualifying entry's position) and it dropped every ×N
  badge (re-stamped from the read-only `RepetitionLedger.repeat_counts()` - `consider` is a
  DECISION and is never re-called for a redraw). The open-burst digest is a day-scoped registry
  (`_digested_keys` + `_refresh_open_digest_row`) so a veto inside the burst redraws that one
  row. `AlertFeedItem.apply_focus_state` re-dresses a single row for a Focus change (star
  `focusOn`, gold frame, ★ badge; unpolish/polish on that widget alone; no-op when unchanged).
  Nothing is gated, scored, folded or withheld; the backing lists, the ledger, the review queue
  and every evidence stream are written before it. The tester's parity assertion counted the
  test's own explicit rebuild; the lead moved the read before it (intent kept). Tests:
  `tests/test_ws_sn4_feed_diff.py`, `tests/test_ws_sn4_feed_diff_builder.py`;
  `test_focus_refresh_coalescing` now pins "one reaction, not a rebuild".
- **The board's picks reach the scan, and every row says why not (WS-10B, WISHLIST 10B,
  2026-09-12, sweep branch).** The DESK chain was traced and is PINNED end to end by
  `tests/test_ws_10b_board_to_scan.py`: board publication -> rows with an empty `failed_floors`
  -> the ONE adoption gate -> `FocusPickStore.add_many` -> the `focus_auto_picks.json` marker ->
  `_inject_into_shared` (appends only when absent, so a refresh appends nothing and a
  trader-typed line is never touched) -> `longs.txt` / `shorts.txt` -> `BounceBot.get_scan_symbol_set`,
  rebuilt from those files every cycle, so an adopted name is scanned on the NEXT cycle without
  a restart. **AWAY was the broken link** - it neither adopted nor staged - and now STAGES the
  eligible rows through the queue's existing owner (`autopilot_core.stage_auto_populate_candidates`:
  one lock, one file, the per-side cap, a name already listed or decided today skipped) with
  `gate_bar_end` left EMPTY so `pending_pick_gate_ok` refuses until the DESK flip's
  re-verification measures it; EVENING and OFF do nothing and nothing new polls. Every board row
  carries an `adoption` verdict written where `_auto_adopt_strength_board` decides - `adopted` /
  `already_in_focus` / `staged (AWAY)` / `not today` / `declined today` / `mode EVENING` /
  `mode OFF` / `not adopted: floor <what it missed>` / `not adopted: <the gate's reason
  verbatim>` - rendered as the LAST column of both side tables, `Scan`: text only, no colour
  vote, NOT sortable (a scan list re-ordered by how the machine answered is not the trader's
  ranking); a row with no verdict is BLANK. One INFO line per refresh: `Strength board: N rows,
  A adopted, S staged, R not adopted (reasons: ...)`. Scanner inclusion and Focus adoption stay
  DISTINCT: nothing scans a row the gate refused. `FocusPickStore.shared_watchlist_path()` is
  the accessor the staging call uses. Two private helpers (`_stage_strength_board_picks`,
  `_publish_strength_board_adoption`) serve only that one function in the alert file.
- **A chart opened from the setups table cycles through the table, and a veto for the day
  hides the row (SC, trader 2026-09-15).** `AlertCenterPanel.chart_symbol` gained
  `next_pick=`: a caller's own "what comes after this chart" (a callable returning True when it
  charted something), stored as `_manual_next_pick`, CONSUMED by `_advance_review_queue`
  instead of the waiting list - so a veto (`_retire_after_veto` -> `_ignore_alert_symbol`), a
  claimed like (`_retire_claimed_review`) or the Next verb on a chart that came from the
  setups table charts that table's next row - and dropped in `_select_review_alert` the
  moment any non-manual chart takes the pane (a lookup-box chart passes none and clears it);
  a callback that raises logs and falls back to the queue; the waiting list is never touched by
  the walk. `MasterAvwapPanel._chart_row_on_desk` passes one for every row it charts
  (`SETUPS_CHART_ORIGIN`), and `_chart_next_pick` finds the row after `(symbol, side)` in the
  proxy's VISIBLE order (by identity first; a row the hide filter already removed is answered
  by the row that took its place), skips the same symbol and any symbol rejected today,
  moves the table's selection with it, and says `End of the setups list - nothing after X`
  when it runs out. The Alert Center emits `reviewDecisionRecorded` after a rail veto and a
  placed claim; the desk connects it to `MasterAvwapPanel.refresh_decisions` (the WS-SX
  coalesced refresh), so the ✕ mark and the hide filter follow the chart's verdict at once.
  **The hide:** `pick_feedback.HIDDEN_REJECT_KINDS` (`veto`, `dislike`, `not_today`,
  `remove_today` - the swing-side verdicts; a day-trade `pass` and an M5 click-away are
  verdicts on another population and hide nothing), `DayDecisions.rejected_symbols()`,
  `SetupFilterProxyModel.set_filters(rejected_symbols=, show_rejected=)` +
  `hidden_rejected()`, fed from the same `_on_day_decisions_ready` snapshot that paints the
  ✕; the strip's `Show vetoed (N)` box (`qt_setups_show_vetoed`, default OFF) restores the
  rows in their original order. Per SYMBOL, like the ✕ mark. Presentation only: nothing is
  deleted, re-ordered or written; the scan, the Setup Tracker's save pass and every evidence
  row never read the filter, so a vetoed name is still tracked. WS-SX's "a decision moves
  nothing" clause is superseded for these kinds; `tests/test_ws_sx_star_x.py`'s moves-nothing
  test became `test_a_veto_hides_its_row_and_show_vetoed_brings_it_back`. Tests:
  `tests/test_setups_cycle_and_veto_hide.py` (16). Long form: `docs/DESK_INTERNALS.md` "SC".
- **The setups table's two mark columns state the day's decisions (WS-SX, WISHLIST item 8,
  2026-09-12, sweep branch).** `scripts/pick_feedback.py` `decisions_today` / `DayDecisions`,
  `scripts/ui/widgets/setup_delegate.py` `set_decision_lookup`,
  `scripts/ui/panels/master_avwap_panel.py`, `scripts/ui/theme.py` token `reject_today` (both
  themes). The star is filled for a name in Focus OR liked today (quick or claimed; an absent
  `like_mode` reads claimed), and the X is painted bright red for a name vetoed, disliked,
  passed on, removed-for-today or clicked away from its M5 alert today, each with a tooltip
  (`helpEvent`) naming the decision and its time; a name both liked and vetoed shows both
  marks. The decision snapshot is a WIDER read than `reviewed_symbols_today` - the day-trade
  pass annotation and the M5 click-away `skip` row (`clicked_away_from_m5_alert`) are in it and
  are not in that badge, whose answer is unchanged - and both come from one cached, mtime-keyed
  parse; `unfavorite` is in neither. It is rebuilt on a worker thread on every capture verb,
  `set_rows`, `showEvent` and the day roll, repainted through the panel's existing 200 ms
  `SignalCoalescer` (now created unconditionally), never read inside `paint`, and it hides,
  re-orders, mutes and writes nothing. Test environment: `tests/conftest.py` registers one
  symbol font (seguisym.ttf) ONLY for `_PIXEL_GLYPH_MODULES`, because an offscreen process's
  font database is empty until qtawesome's icon fonts load and then U+2605/U+2606 draw nothing.
  Known, out of scope: `test_qt_desk_layout.py`'s compact-profile test overflows at 1400 px
  when run after a MainWindow with real fonts. Tests: `tests/test_ws_sx_star_x.py` (14).
- **The AWAY digest ranks swing picks by points when the trader's Points switch is on
  (WS-PT4, 2026-09-12, sweep branch).** `autopilot_core.swing_pick_projection` is the ONE
  projection of a digest pick (`AutopilotService._write_report_locked` calls it) and now
  carries the scan row, `d1_vs_sector`, `d1_vs_industry` and `bucket_key` beside the display
  fields; `autopilot_core.order_swing_picks` reads `setup_points.rank_enabled()` AT SORT TIME
  and, when on, orders the favourite / near / high-conviction rows by `setup_points.rank_order`
  over `swing_pick_points(...)` totals (the same `setup_points.score_row` the setups table
  uses - one scorer, two callers), every other row after them in arrival order; off is the
  identity. `RANKED_BUCKETS` is matched on the bucket KEY, never the display label. The
  `Ranked on:` line ends `| order: Wilson bound` or `| order: points (switch on)`; the near
  cap is applied after ranking as before; the bucket is printed, never ranked on; an ungraded
  family scores its setup part 0 with the note and is never dropped. Known, not built: the
  hourly phone push `build_swing_push` iterates the picks in arrival order and never followed
  either order. Tests: `tests/test_ws_pt4_digest_points.py` (golden
  `tests/fixtures/ws_pt4_away_digest_switch_off.txt` pinned by the pre-change code).
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
- **The digest gate has TWO measured halves, and enrichment waits for both**
  (packet Q4, 2026-09-04). `clean_digest_sessions` returns the length of the run
  of CONSECUTIVE clean exchange sessions ending at the newest pack - walked
  through `market_calendar.previous_session`, never weekday arithmetic - where
  clean is `is_session` plus an EMPTY `unavailable`, the pack's own failure
  record, which its own summary already calls INCOMPLETE. A non-session pack
  neither counts nor breaks, and `first_gap_session` names where the run stopped;
  `sessions_collected` keeps the pre-Q4 distinct count for existing readers. It
  counted DISTINCT packs until now, so ten scattered across a month read as a met
  window. The second half is a FILE - `digest_audit_approval.json` beside the
  packs, written ONLY by `python -m ai_jobs.digest approve-audit --pack <date> …`,
  which refuses fewer than three packs and any date with no pack. **No nightly
  job may write it** (a test walks the runner's source): a runner that approves
  its own evidence has asserted, not audited. `gate_met = window_met and
  audit_recorded`, and **`journal_enrichment` now refuses until both are true** -
  no model, nothing written, ledger row `refused: audit not recorded`.
  `review_policy_draft` and `setup_research` keep their own separate gates.
- **The nightly slate runs in three stages** (decision 0018, 2026-09-04):
  every deterministic slot, then `ai_summary` + `ticker_briefs` as a unit, then
  the model-gated slots. The original narration pair held up to 2½ h of reserve ahead of
  every deterministic slot, a slot that cannot fit its reserve records SKIPPED,
  the 2026-09-01 run took six hours - and no deterministic slot reads either
  narration slot's OUTPUT. Relative order inside each stage, every reserve and
  every retry budget unchanged. The rule is now "a later phase appends inside its
  stage and never reorders across stages", and the order is pinned once as
  `EXPECTED_SLOT_ORDER` in `tests/test_ai_jobs_runner.py`; Phase 0.31 appends
  `market_story_narration` at the end of that same stage.
- **`entry_index.json` is the compact, deterministic handoff** (Q4.4). Written
  beside the packs at the end of `run_daily_digest` with a temp-and-rename; a
  failure is logged and NEVER fails the digest. Sessions in the `LATELY_SESSIONS`
  window with pack path, version count, `superseded`, clean flag, failures and
  coverage; `changes_vs_prior_window` by FLOOR STATUS only, never by ranking an
  immature cell; FOUR sections never merged - `intraday_held_run` (MFE/MAE only,
  because `close_r` is the RESULT), `swing_win_rates` and `journal_execution`
  both EMPTY BY CONSTRUCTION with the reason stated, `preference_observations`
  classified by `review_learning`'s own TAKE/REJECT sets; `pending_experiments`
  from the trial ledger with their frozen windows, listed and UNRANKED; and an
  `open_questions_for_a_ticker_brief` that stays empty because a brief opens only
  for a STATED question. `read_entry_index` exists for the readers to come and
  **nothing consumes it yet**.
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

- **The Mentor tells an EXIT from an ENTRY (TJ-9E, 2026-09-21, branch
  `claude/tj9e-exit-notes`, merged into `lead/p033-integration2` `7230b30d`).** An ENTRY is
  unchanged - the same four material fields, the same four answer states, the same day-time
  `trade_mentor_ai` draft. An EXIT is any trade with a closing fill in the reviewed session,
  whenever it opened, and it is ONE forced free-text box ("Why did you exit? What did you
  feel? What were you watching?") with two small answer-state buttons and no dropdown; the
  row survives an EMPTY `missing`, so a swing closed yesterday is asked about at all for the
  first time (116 of 180 live closed trades exit on a day they did not open). A scale-out
  inside one session is ONE ask however many fills; closing legs on two dates are two asks on
  two mornings. The words are an append-only `EXIT_NOTE_RAW` row written BEFORE anything
  reads them - a blank is refused, a store that cannot append RAISES, a second note appends
  and never rewrites - and provenance is ruled by the EXIT's session, never the first fill's,
  a date-only exit never `same_session`. The nightly stage-2 slot `exit_note_fields`
  (`scripts/ai_jobs/exit_note_fields.py`, between `week_review_narration` and
  `ticker_briefs`) reads each waiting note once on the local MEDIUM model and DRAFTS three
  fields: `why`, ONE code from the new closed versioned vocabulary `scripts/exit_reasons.py`
  + `ui/annotations/vocabularies/exit_reasons_v1.json`; `felt`, up to
  `trader_state_tags.MAX_STATE_TAGS` codes from TJ-7's OWN feelings list through ITS loader;
  and `watching`, up to `MAX_WATCHING` (3) EXACT quotes. Every value carries a character span
  that must reproduce its quote. The request is built from an ALLOW-LIST holding the note's
  words, the symbol, the side and the two code lists and NOTHING else - no price, no P&L, no
  R, no fill, no later bar - `verify_reply` re-checks every bound itself (key sets at three
  levels, both vocabularies, both counts, a repeated value, the span), one bad value rejects
  the WHOLE reply and leaves the prior file byte-identical, nothing waiting means NO model
  call, there is one call per note with the window re-asked before each call after the first,
  and at most `EXIT_NOTES_PER_NIGHT` (20) notes a night, oldest first, the rest counted and
  said. A reading is `provisional` until the trader's own Confirm or Correct, which is the
  ONE writer of an `EXIT_NOTE_FIELDS` row (a corrected value carries `span: []`, `quote: ""`
  and `source: "correct"`); its identity is `check.exit_key` = `<trade_id>@<exit session>`,
  so a trade that exited twice (13 live) is read and confirmed twice and the writer refuses a
  note from the other session. A waiting reading has ONE home - the draft row in the
  questions area, never a second copy in the trade section - and is offered on its OWN clock
  for `EXIT_DRAFT_OFFER_SESSIONS` (5) exchange sessions by the REGISTRY's budget of three
  (`exit_draft_review`, AWAY prompting nothing, the rest said and carried), never against
  words the trader has since REWRITTEN, never confirmed by age; `Rewrite` is the one door to
  a second note. An exit with a note or an answer state is ANSWERED and never asked again,
  and `unexplained_exit_count` (separate from `unlabelled_trade_count`) makes the 09:00 check
  RIDE on a later delivered slot. Readers: Day Review's trades section carries `exit_note`
  and `exit_fields` on every row, and `day_report_card.exit_note_counts` prints "exits
  explained K of N, C of those confirmed by you" in INTEGERS only (`unmeasured` when nobody
  opened the notes). Also in this work: `journal_store` now NAMES the three trade statuses it
  writes (`TRADE_STATUS_CLOSED`, `TRADE_STATUS_CLOSED_PARTIAL`, `TRADE_STATUS_OPEN`) and the
  Mentor imports them instead of its own copy, which spelled the half-exited state
  `PARTIALLY_CLOSED` and hid all seven live `CLOSED_PARTIAL` trades from the 09:00 check.
  Nothing here reaches a detector, score, alert, watchlist, Focus, the review queue or
  `review_policy.json`. Long form: DESK_INTERNALS "TJ-9E". Gates #176-#180 owed.
- **An answered trade is stored at once and never asked again (the per-trade Mentor Save,
  2026-09-21, branch `claude/mentor-stores-answers-2026-09-21` `0caf2bba`, merged into
  `lead/p033-integration2` `182f3e08`).** From the trader's own report on TJ-9's first live
  morning. The 09:00 trade check's forced Save gate is PER TRADE: `_open_fields` asks what is
  still open on ONE trade, each block carries `Save this trade`, the bottom Save files every
  answered trade through `save_trade_check(only=)`, `_answered_trades` and
  `_drop_trade_block` take a filed trade off the card while a half-answered one keeps its
  exact widgets. `_field_answer` reads words typed beside a blank dropdown as an answer
  (`not_supplied`, with what it was) and the trade's ONE raw note as the answer to every
  field still open on it, the note saved verbatim as `RECALLED_RAW` first;
  `MainWindow._mentor_answered` finds a trade's answer by `trade_id`, so a `once` question
  does not return when the position closes and `trade_date` moves. TJ-9E's forced exit box is
  one more FIELD of its trade inside that gate (`_open_fields` asks `_exit_is_open`, the one
  predicate) and filing a trade writes its exit note FIRST. Still forced: a trade cannot be
  half filed. Merged on the trader's instruction to combine the day's work, with NO reviewer
  round of its own - a review is OWED (plan.md). Gate #181 owed.
- **A mood is a field the desk REPORTS, and nothing acts on it (TJ-7, 2026-09-20, branch
  `claude/tj7-mood-fields`, merged into `lead/p033-integration2` `b63db7af`, lead guard
  amendment `ae7c06c7`, follow-up merge `0a0a0be4`).** `market_journal` carries ONE
  additive row key `mood` (schema `trader_mood_v1`): a 1-5 `score` off `MOOD_SCALE`, at
  most two `state_tags`, and a `process` block (`followed_plan` yes/partly/no or nothing,
  a `note` of at most 200 characters). It is `{}` on a row nobody clicked, ABSENT on every
  row written before the packet and NEVER a default — all three absences read `None` — and
  `market_journal.mood_of` and the point-in-time `mood_at` are its ONLY readers.
  `MoodFieldError` is LOUD at `build_entry` and the same question is asked AGAIN at
  `is_publishable`, the gate every write passes through: a score off `MOOD_SCALE`, a bool,
  a tag outside the versioned closed vocabulary, more tags than the cap, an unknown plan
  answer, a process note over 200 characters REFUSED and never truncated, and a mood on a
  MACHINE-written row refused through `is_machine_entry`'s own rule
  (`MACHINE_MOOD_REFUSAL`). The vocabulary is
  `ui/annotations/vocabularies/state_tags_v1.json` with its own loader
  `scripts/trader_state_tags.py`, which OWNS the cap (`MAX_STATE_TAGS`) and the picklist on
  `ai_jobs.observation_tags`' loader rules: the declared version equals the number in the
  FILENAME, a duplicate code raises, an older version reads back by number, a code is never
  renamed or reused, and no test asserts a literal `vocab_version`. ONE strip
  (`scripts/ui/widgets/mood_strip.py`) serves BOTH surfaces — the Trade Mentor's `day_close`
  question hook and the desk's journal tab — optional, with nothing ever pre-selected and
  every click takeable back: a second click on the selected face CLEARS it (the faces are
  deliberately NOT an exclusive `QButtonGroup`, because Qt will not un-check the checked
  member of one and a misclick would otherwise have no way home), after which the strip is
  indistinguishable from one nobody touched. It never greys TJ-9's Save, a mood clicked with
  no plan answer is STILL FILED, it rides the `day_close` kind (`mentor_questions.KIND_DAY_CLOSE`
  — no new registry kind, so the budget of three does not move), and
  `TradeMentorCard.set_questions` became a MERGE for the WHOLE questions box, so a
  half-clicked answer survives the same questions being offered again. The day pack gains a
  `mood` section whose citable ids are minted through the ONE seam
  `day_review_pack.mood_source_id` — shared with the Day Review payload, so one row has one
  id wherever its section is built — each item PROJECTED through the load-bearing
  `MOOD_ITEM_FIELDS`, in time order, inside `allowed_source_ids`, with a clock-free
  `inputs_hash` that MOVES when a mood moves; `MOOD_EMPTY_STATEMENT` / `mood_statement` are
  the ONE formatter, and Day Review prints ONE mood line off the ONE payload the worker
  builds. `week_review_narration.EVIDENCE_KEYS` and `improvement_ideas.EVIDENCE_KEYS` each
  gained exactly ONE name, `mood`, and neither gained a `MEASURABLES` entry — a kept
  `process` idea can never be checked against how the trader felt.
  `market_read_grades.context_for(..., mood_entries=)` takes only the mood recorded AT OR
  BEFORE the read's stamp, else `unmeasured`. **The tagger never sees a mood**, and nothing
  here reaches a detector, score, alert, watchlist, Focus, the review queue,
  `review_learning`, a cohort grader or `review_policy.json`; no outcome selects, ranks or
  pre-fills one. TJ-7 adds NO nightly slot, so the NINE slot-order pins do not move, and
  source selftest stays **97/97** (89 lazily-imported engine modules plus 8 asset checks,
  verified in `scripts/selftest.py`). Tests: `tests/test_tj7_journal_fields.py`,
  `tests/test_tj7_state_tags_vocabulary.py`, `tests/test_tj7_mood_strip_is_optional.py`,
  `tests/test_tj7_day_pack_and_narration.py`, `tests/test_tj7_evidence_lists.py`,
  `tests/test_tj7_point_in_time_and_no_outcome.py`,
  `tests/test_tj7_reported_never_acted_on.py`, `tests/test_tj7_day_review_mood_line.py`,
  `tests/test_tj7_review1_followups.py`, `tests/tj7_support.py`. Rule: DESK_INTERNALS
  "TJ-7". Gates #151 and #175 owed.
- **The read grader, the prediction ledger and the congruence lines (TJ-10, 2026-09-20,
  branch `claude/tj10-read-grader`, merged into `lead/p033-integration2` `57b44ca9`).**
  `scripts/market_read_grades.py` grades what the trader said the market would do against
  what it did, with no model anywhere: a CLICK (`market_journal.prediction_of`) is the read
  and the words are never consulted for a row that has one; an extraction
  (`market_thesis.extract_thesis`) is labelled `extracted` and is never pooled with a click
  (`PoolingError`); `no_view` and an `unstated` note are recorded and never graded.
  `rest_of_day` anchors on the OPEN of the first completed M5 bar that starts AFTER the
  stamp and ends at the last completed bar at or before `session_close` (half days
  included); `next_5_sessions` anchors on the DECISION session's own daily close
  (`market_calendar.decision_session`) - `None` while that session is still trading, never
  a forming close - reports 1 and 3 sessions and is JUDGED at 5. Verdicts are
  `right / wrong / flat / pending <date> / unmeasured:<reason>`; the flat band is
  `FLAT_BAND_ATR` 0.25 of the point-in-time daily ATR(14), edge inclusive, stamped
  `flat_band_rule: "atr_0.25_v1"` on every row, with `flat` IN the accuracy denominator.
  A verdict may only ever move UP (`verdict_rank`: measured > pending > unmeasured-for-data
  > final), so an `unmeasured` result never supersedes a `pending` row, an open read is
  revisited on every later run, a measured verdict is final and a re-grade appends a NEW row
  carrying the original context; `daily_bars_for_symbol` / `session_bars_for` /
  `atr_for_session` are the ONE loader set the page and the night share. Every gradable
  grade carries its point-in-time `trade_mentor_context_v2` snapshot built through the ONE
  internals builder (`trade_mentor_context.internals_at`); `append_grades` refuses a blank
  context and refuses a gradable CLICKED grade carrying the named absence
  `CONTEXT_UNMEASURED` (`ContextMissingError`), in which case the grade is held back and the
  payload row says `grader_gap: "context_unbuildable: ..."`. `grader_gap` is now on every
  row (never set for `not_a_call`) and nothing reads it yet. The store is append-only JSONL
  at `project_paths.DAY_REVIEW_READS_DIR`. `congruence_lines` prints the three D1 lines
  (`CONGRUENCE_KINDS`: the read against the desk's D1 label, the session's like/claim side
  mix and the bias of its fills, read from the LEGS) plus `CONGRUENCE_M5_KIND`
  (`m5_picks_side_mix`, the rest-of-day read against that session's M5 likes, `not today`
  counted apart, D1 picks never entering it); each names its source in plain words
  (`your call: up (clicked 10:02)` vs `we read your note as up`, `select_read` preferring a
  click and refusing contradictory extracted notes), its timeframe, its `n`, its source ids
  and its missing side; a side mix under `MIN_REPORTABLE_N` reads `VERDICT_TOO_FEW`; a tie
  names no side; printed, never pushed, never acted on. `baseline_reads` scores `always_up`,
  `same_as_the_last_hour` and `with_the_d1_environment` out of each row's own context (feed
  it GRADE rows). `market_thesis.EXTRACTOR_VERSION` is `market_thesis_vocab_v2` (whole-token
  trend words, `rejecting`/`rejected` scoped: 18 -> 20 of the 49 live notes read as a
  direction, none reverses). The Day Review payload gains `reads` and `congruence`, both
  built on the worker inside the one read; `DayReviewService.build_reads_for` is the named
  post-close seam and the only writer; the page shows a verdict chip per graded entry
  (`verdict_chips`, `theme.qss` `#VerdictChip[verdict=…]`) and the congruence lines under
  the story, and computes nothing. Shadow to every detector, score, alert and watchlist.
  Live gate #155 owed. Long form: DESK_INTERNALS "TJ-10".
- **One Day Review page, no machine rows, a per-session index (TJ-1, 2026-09-17,
  branch `claude/tj1-day-review`).** `Day Review`
  (`scripts/ui/panels/day_review_panel.py` + `scripts/ui/services/day_review_service.py`)
  takes the Market Journal's nav slot and does the Daily Recap's job: one session at a
  time, one `read_day` payload read on ONE `QThread`, six sections that paint from that
  payload and nothing else, ZERO `CandleChart`s at construction and ONE built on the first
  payload that carries SPY bars. The picker is the Daily Recap's (15 completed sessions plus
  Today marked provisional) and `daily_recap_schedule`'s noon and post-close functions still
  make the decision; the post-close tick builds that session's index once through
  `DayReviewService.build_index_for`. An auto-mode flip is now ONE Auto Pilot log line
  (public `AutopilotService.log`, forwarding to the one `_log` writer) and writes no journal
  row and captures nothing, and `market_journal.is_machine_entry` is the ONE filter, applied
  in `MarketJournalService.entries_about`, `market_story.build_daily_story` and
  `market_story_rollups._stories_from_journal` - the 34-of-77 rows already on disk are
  hidden, never deleted. `scripts/day_review_index.py` + `project_paths.DAY_REVIEW_DIR`
  store, per session, `sessions/<date>/outcomes.json` holding exactly the `_Store`s
  `read_session` would have built (its rows, the FULL-FILE coverage, the append counts) for
  every store over a megabyte - the 476 MB intraday log, the 31 MB horizon CSV, the 14 MB
  tier CSV and the 1.0 MB human-focus CSV - while the six small ones stay LIVE so a veto, a
  note, a favorite or a staged pick from a minute ago is on the page; revival is ALL OR
  NOTHING, anything that is not an index of that session and window streams, and
  `day_review_index.is_stale` is the one staleness rule (pending only; stale once the session
  it waited for has closed, always stale for a session that had not closed when it was
  built, and stale when a stored `(size, mtime)` stamp of the indexed stores says they were
  REWRITTEN - shrunk, or changed at the same size, as a warehouse recompute does. GROWTH is
  read instead of assumed: the M5 scanner appends to the intraday log all day, so the
  appended TAIL is parsed and the scope is `(session, symbol)` - the selected session and its
  window count for any name, a target session weeks forward only for the names the index's
  own observations reference. An append it does not reference leaves the 22 MB body alone and
  records the new stamp in a `stamp.json` beside it: 28 ms to decide, 502 ms to open. One it
  does reference rebuilds on the worker, 11.4 s, with the page still showing what it had -
  and on the live store that set is ~1,198 names, so the rule keeps the page warm after the
  close and outside trading hours rather than during a session). The index is written only for a CLOSED session, only
  when its content changed, and the folder is pruned to the newest 40; the post-close build
  runs on a worker and the SPY bars are read on the Qt thread and handed into the read. `_d1_horizon_row` and `_outcome_for` became dict lookups built once per read
  (keyed by session and symbol, never by side), which is what took an indexed read from
  2,217 ms to **820 ms settle p50** warm against the retired page's 13,500-16,500 ms across runs; the
  post-close build runs on a worker (the slot returns in 0.2 ms) and a cold open, with no
  index yet, paints after 12.5 s while it builds. "Paste
  daily forecast..." files a brief against `target_session` through
  `MarketJournalService.import_daily_forecast` (a second paste supersedes the first;
  `market_thesis.record_forecast` records `target_session` beside the kept `target_week`;
  `import_weekly_forecast` is a deprecated alias for one release), and
  `scripts/forecast_brief.py` reads it by heading match only - no model, nothing fetched,
  anything unsaid left empty. The Daily Recap's Review tab is now the **Measured report**
  section on Research > Results (same cells, same `report_id`, same `review_cells` parity
  seam, same `ReadWorker`, `entryQualityProposalChanged` moved with it) and its Staged picks
  table and verb are on the **Auto Pilot** page (same `focusAddRequested`, the R2 gate still
  SHOWN and never enforced). `market_journal_panel.py` and `daily_recap_panel.py` stay on
  disk, unregistered and no longer constructed, until TJ-8. **Since TJ-1L (2026-09-18) the
  page reads in TWO COLUMNS** (trader: *"there's a lot of empty space horizontally that's not
  being efficiently used"*): one horizontal `QSplitter` named `DayReviewColumns` at 55/45,
  restored at construction and saved per machine on the drag through `ui.panels.desk_layout`
  (`qt_day_review_columns_v1`), with *What happened* / *Open theses* / the SPY pane (320 px
  floor, taking the column's slack) LEFT and the entries list over the reader in their own
  60/40 splitter (`qt_day_review_said_split_v1`) with *New entry* and the 3-line external
  forecast RIGHT; *Walk-away* is a 2 x 2 grid of equal columns, the real table top-left and
  TJ-2's three populations as small titled frames; *What you traded* sits beside *Ideas from
  the desk's AI*. Every table on the page measures its columns (`ResizeToContents`, set once
  at construction) and stretches the last section, which is what stopped "Against me first %"
  printing as "ainst me first". Presentation only - no reader, service, store, worker or
  signal moved, and the page is still ONE `QScrollArea` with ONE chart built on first need.
  Gate #145 owed. Long form:
  DESK_INTERNALS "Day Review - one page, no machine rows, a per-session index".

- **The Market Journal tells the session's story and challenges the thesis in it (WS-10D,
  WISHLIST 10D / 10K, 2026-09-13, sweep branch).** `scripts/market_story.py` builds a
  `DailyStory` in three kinds that never blur: `trader_said` (the day's entries verbatim in
  `created_at` order, each carrying `written_after_the_session` and `predicts_this_session`, so
  the same sentence typed at 11:00 and at 21:00 Pacific is a prediction and a description),
  `measured` (completed daily bars for SPY / QQQ / IWM / VXX / TLT / USO, each cell carrying
  `bars_through`, `bars_used` and a named rule version, an absent series reading `unmeasured`
  with a reason), and `ai_said`, ALWAYS EMPTY here - code computes, the model explains in a
  later packet. No note means an empty `trader_said` and a sentence saying so. Which session an
  entry belongs to is ONE function, `market_journal.session_of_entry`. Phase 0.31 preserves
  the explicit subject in `session_date`, records the actual write day in
  `written_session_date`, and retains the created-at fallback for legacy rows.
  `scripts/market_thesis.py` reads a note with a versioned vocabulary into claim / horizon (in
  exchange SESSIONS) / stance / condition / invalidation / benchmarks, every field carrying a
  span that reproduces it exactly, `unstated` carrying none; a later note links only inside the
  horizon and only on the same benchmark (same stance SUPPORTS, a reversal CONTRADICTS, a
  stance-less mention MENTIONS). Rows live in `market_theses.jsonl`
  (`project_paths.MARKET_THESES_FILE`), append-only, keyed on `entry_id` + `extractor_version`;
  a trader edit is a NEW superseding row and the journal entry is never touched. An imported
  weekly forecast is `origin=external_forecast` plus a `kind=forecast` sidecar whose unsupplied
  creation time stays `unknown`; `active_theses` never returns one. The Market Journal page
  gains the Story pane ("You said" / "External forecast" / "The market did" / "Sources", the
  sources clickable), the Active theses list with one or two grounded questions and an
  interpretation box, and "Paste weekly forecast..." - all on the panel's existing worker.
  `scripts/market_story_rollups.py` is the LAST deterministic nightly slot: each week belongs
  to the month of its Thursday, a month adds its uncovered days, every pack names covered /
  expected / missing sessions, carries open theses forward, and is rebuilt only when its
  `inputs_hash` changed; both slot-order pins gained the name in that position. Shadow only.
  Advisory: a panel refresh reads the journal ledger three times on the worker (cheap today).
  Tests: `tests/test_ws_10d_market_story.py` (one contradictory ordering assertion corrected by
  the lead), `tests/test_ws_10d_story_links.py`.
- **The said-vs-did report has both halves (WS-5B, WISHLIST 5B, 2026-09-13, sweep branch).**
  `scripts/preference_trade_outcomes.py` now collects every explicit REFUSAL beside the
  endorsements: `annotation:veto` (detail `<code> (v<vocab_version>)`, an uncoded veto reads
  `uncoded`), `annotation:pass` (unchanged detail), `pick_feedback:dislike`,
  `pick_feedback:not_today` and `review_event:m5_click_away` (a new `events_path` kwarg);
  `unfavorite` is absent by decision. Three columns at the END of `COLUMNS` - `like_mode`
  (`quick` / `claimed` for `annotation:like_claim` rows, an absent field reading `claimed`;
  empty for the pick-feedback and favorite likes, which were neither the key nor the dialog),
  `verdict_family` (`endorse` / `reject`), `match_state` (`matched` / `window_open` /
  `no_match_after_window` / `journal_unavailable`; `matching_unavailable` reserved, no path
  emits it) - and `schema` bumps to `preference_trade_outcomes_v2`; the first 19 columns stay
  byte-identical (golden). `trade_level_summary` gains `n_statements_by_family` and still keys
  money by `trade_id`; `match_trade`'s side logic is untouched (P6's opposite-side match at
  0.35 stands, the verdicts stay separate rows, the money counts once); a sideless coincidence
  is `symbol+window_side_unknown` 0.50; `match_state` agrees with
  `ai_summary.preference_to_trade_section` by construction (same empty-`match_basis` rule, same
  `statement_window_end`); an unreadable journal publishes the statements with empty
  `match_basis`, `journal_unavailable` and a `degraded` slot status instead of `skipped` with no
  file. Weekend Prep's Focus Review gains a TENTH view, "Said no" (`preference_rejection_table`
  / `preference_rejection_note`), filled by the same read pass, `Match state` visible, counts
  never pooled across families. Two ST5 tests widened their fixture to name the fourth store
  (a test naming three leaks the fourth from the shared pytest home). No free text reaches a
  model (`statement_detail` is not a `PREFERENCE_EXAMPLE_COLUMNS`). Tests:
  `tests/test_ws_5b_preference_symmetric.py` (17).
- **A fourth auto-tagging lane: the trader's own Market Journal notes (WS-10E, WISHLIST 10E,
  2026-09-13, sweep branch).** `AutoTagger` now reads the Market Journal. For a CLOSED trade,
  entries whose `symbols` carry the trade's symbol and whose ACTUAL write time (`created_at`;
  the subject and write day are stored separately since Phase 0.31) fall inside the trade's
  own window - open to close, widened by one trading session before the open - are candidates;
  a date-only broker fill has no intraday window and is `unmeasured`. A tag is emitted only for
  an explicit claim: `setup_docs.SETUP_DOCS` compiled to whole-token phrase patterns from each
  family's key and label, single-token phrases dropped so `general` cannot tag a trade. The
  candidate carries `match_basis = note:<entry_id>` and the span quoted verbatim
  (`auto_tag_candidates.match_basis` / `match_span`, additive in `NEW_COLUMNS_V3`). A note whose
  own words state the opposite side never matches; a side-silent note matches either. Order
  stays by LANE - capture, note, scanner, shape (`journal_analytics._lane_rank` and the mirrored
  SQL in `JournalStore.list_auto_tag_candidates`) - with confidence 0.88 / 0.84 placing the note
  between the two, so `journal_bulk_tag` writes it under the same 0.70 threshold and
  `apply_provisional_tags`' refusal to overwrite a confirmed tag is untouched. The lane's
  verdict is stored per trade in the new `note_lane_verdicts` table (its own table, LEFT JOINed
  into `list_trades`; never a column on the golden-pinned `trades`) and printed by
  `journal_analytics.format_note_lane_line` in three shapes: the claim, `no explicit claim in N
  candidate note(s)`, `unmeasured (date-only fill)`. The Journal's Trades detail shows it above
  the overnight AI row; Weekend Prep's Tag Week gains a `From` column marking rows whose waiting
  tag came from a note. The advisory enrichment package carries `trader_notes` and
  `deterministic_note_lane` so the model cites `note:<id>`. Nothing in this chain reads an
  outcome field (grep-guard). Tests: `tests/test_ws_10e_note_tags.py` (13). Docs: DESK_INTERNALS
  "The four auto-tagging lanes"; `docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md`. The CLAUDE.md /
  AGENTS.md "three lanes" rule line is rewritten in the sweep's docs pass.
- **Trade Mentor pop-up and hidden market context (WS-TM follow-up, trader 2026-09-14).**
  On `codex/mentor-popup-context`, the chart host owns one reusable modeless pop-up,
  with Submit, Read unchanged, Skip and draft preservation. It takes no chart height;
  the arm bar stays in place. `scripts/trade_mentor_context.py` builds shallow,
  completed-bar measurements for the trader's exact 17 symbols: 30-minute M5 change,
  position against session VWAP, five-session D1 change and position against SMA20.
  Unknown and stale readings carry reasons rather than zero. The context service
  owns bounded background collection on prompt opening, uses available local caches
  and batches missing coverage through Yahoo, with hourly M5 and completed-session
  D1 reuse. Submit never waits for data. The snapshot stays under `mentor.context`
  beside the original text, including on Read unchanged; a late worker cannot
  change a saved note. `ai_summary` compacts only this attachment in `journal.entries`
  for the existing AI budget; there are no new model calls or raw candle arrays.
  Independent review is GO; 130 focused tests and the 7980-test full suite passed, with natural exit 0. Checkpoint gate #110 remains a live check.
- **Trade Mentor: a prompt is a slot, an answer is a dated row (WS-TM, WISHLIST 10J steps 1-2,
  2026-09-13, sweep branch).** `scripts/trade_mentor_schedule.py` (pure) builds the day's
  `MentorSlot`s - whole hours from `FIRST_HOUR` 7 Pacific (`America/Los_Angeles`, DST-aware)
  to before the close, `D1_HOURS` (8, 12) and `TRADES_HOUR` 10 combined into one slot each,
  `post_close` when the slot sits past an early close read from `market_early_close.session_close`
  (`market_calendar.session_close` is 16:00 ET even on a half day), `expires_at = scheduled_at
  + 1 h`, nothing on a weekend or holiday; `slot_id` = session + wall time + kind.
  `ui/services/trade_mentor_service.py` is the ONE scheduler (one 60 s `QTimer`, owned by
  `MainWindow`, started in `showEvent`): slot records in `trade_mentor_slots.json` (delivered /
  answered / skipped with `away` / `paused` / `locked` / `idle` / `expired` / `not_present`),
  expiry at the head of the poll, `promptDue` once per service instance per slot, a restart
  re-shows an unanswered unexpired card without moving `delivered_at`, a missed hour is
  recorded and never re-asked, AWAY / paused / idle beyond `IDLE_GRACE_MINUTES` (20) skip
  (`scripts/user_presence.py`: `GetLastInputInfo`, None off Windows reads PRESENT;
  `session_locked` is an injected callable, no lock hook yet). `ui/widgets/trade_mentor_card.py`
  initially docked under the chart; the 2026-09-14 follow-up moves it to a pop-up,
  `WA_ShowWithoutActivating`, Ctrl+Enter scoped to its boxes; Submit writes the RAW text first
  through `market_journal_service.write_entry(origin="trade_mentor", mentor=..., reaffirms=...)`
  - `build_entry` / `write_entry` grew those two kwargs (present and empty on every other entry)
  because no metadata field existed; "Read unchanged" writes a NEW row referencing the previous
  read; drafts live in `trade_mentor_drafts.json`, never a read. The 10:00 check
  (`scripts/trade_mentor_trade_check.py`) reads the previous exchange session's trades from
  `journal_store.JournalStore` + `journal_coverage` (NOT the market journal), asks only the
  missing material fields with the four answer states distinct (`not supplied` / `no fixed
  target` / `not remembered` / `not applicable`; no stop is never 0), stores the answers as
  RECALLED `opportunity_events` annotation rows stamped with the actual write time (no schema
  migration, `planned_stop` never written), capped at three trades / five minutes with the rest
  counted; missing broker coverage says `journal not ready`. Settings: the checkbox (default
  OFF, `qt_trade_mentor_enabled`), the DST sentence, `next prompt hh:mm`, `Pause today`; "Give a
  read" on the chart host at all times. Independent of the scanner's Auto setting; no phone
  push; nothing in AWAY. Defect fixed alongside: `market_journal_service.write_entry` built the
  entry from the caller's `now` but appended without it, so an entry written with an explicit
  clock was filed under one date and stamped with another (no production caller passed `now`
  before). Steps 3 (AI form filling) and 4 (coaching) are NOT built. Tests:
  `tests/test_ws_tm_trade_mentor.py` (36 + 3), one added in `tests/test_market_journal.py`;
  `test_qt_alert_capture`'s "nothing under the charts" pin now names the hidden card after the
  arm bar. Selftest 75 -> 80 (five reach checks).
- **The Mentor card asks two different questions (TJ-14A item 1, 2026-09-19).** **What I see**
  is the words (optional; a clicks-only answer's `text` stays `""`, never a synthesised
  sentence) and **What I expect** is a forced click stored as `mentor.prediction`
  (`mentor_prediction_v1`: direction, horizon, confidence, optional `because`), with
  `market_journal.prediction_of` the ONE accessor - `None` for all four older live row
  vintages, so an extracted stance is never pooled with a clicked one - and `prediction_line`
  the ONE wording for a screen. Horizons are `rest_of_day` (every card) and `next_5_sessions`
  (the 08:00 and 12:00 D1 cards); directions `up / down / chop / no_view` for the day and
  `up / down / range / no_view` for the five sessions; `no_view` is complete and hides `How
  sure`, which is otherwise forced. The gate is inside `submit()` and `read_unchanged()` as
  well as on the buttons, an `m5_d1` card always writes both rows, drafts keep unsaved clicks,
  `entry_id` salts a WORDLESS row so two of them cannot share an identity, and `is_publishable`
  is relaxed for a clicked row only. **A row's `timeframe` and its prediction's `horizon`
  always agree** (`HORIZON_FOR_TIMEFRAME`), enforced at the writer - `build_entry` raises
  `PredictionTimeframeError` and `is_publishable` refuses - so `Read unchanged` reaffirms PER
  TIMEFRAME from the latest read of each (the host supplies `{"M5": row, "D1": row}`, never
  `rows[-1]`) and is unavailable, saying which timeframe is missing, rather than substituting
  one; it now returns `{"ok", "entries": [...]}`. TJ-9's forced trade-check section is
  unchanged and keeps its own gate. Tests: `tests/test_tj14a_mentor_prediction.py`,
  `tests/test_tj14a_fix_round.py`, and 6 WS-TM functions gained clicks. Rule: DESK_INTERNALS
  "TM" (2026-09-19 addendum). Gate #159's first clause.
- **`trade_mentor_context_v2` and the internals strip (TJ-14A item 6, 2026-09-19).** 18 symbols
  (`XLRE` added alphabetically; every existing index, SPY at 6 included, unchanged), each with
  the day's change against the prior session's close, its place in the day's range and both
  prior-session sides, plus a `derived` block - breadth, fear (with a divergence flag), rates,
  oil, sector leaders and laggards on the day AND over 30 minutes, offense vs defense, and
  sectors above VWAP as a count with its denominator - each line naming its readings and
  `unmeasured` naming the input that is actually missing. Completed bars only; v1 rows stay
  readable and `compact_for_ai` projects both vintages, now carrying `common.internals`, ONE
  scalar that survives `ai_summary._bounded`'s six-level depth cut. `internals_at` is a pure
  rebuild for any moment on the SAME builder as the live card; the thin loader
  `internals_bars_at` reads M5 from the durable Day Review tape and D1 from the scanner's daily
  cache, falling back to daily bars built from the prior sessions' own tape (`_TAPE_D1_LOOKBACK`
  3, no network) for the names that cache has never held - RSP, USO and TLT, which the scan
  universe never fetches - so a rebuild measures breadth, rates and oil while the five-session
  and SMA20 facts stay `unmeasured` with the reason; it has no production caller until TJ-10 /
  TJ-16. The card SHOWS the block as `MentorInternalsStrip`, worded by the pure
  `internals_lines`, styled in `theme.qss` by object name, built from the context the service
  already delivered (no second fetch). `day_review_bars.decided_symbols` adds the symbols to
  the one batched post-close download - one yfinance call per FIFTY symbols, fixed base 18
  instead of 4, so a session with more than 32 other decided names needs a second call; no D1
  leg, zero IB. Shadow of nothing: no detector, score, alert, watchlist, Focus, review queue or
  `review_policy.json` is reachable from it. Rule: DESK_INTERNALS "TM" (2026-09-19 addendum).
- **The Mentor question registry, the budget of three and the day's fills (TJ-14B items 2-5,
  2026-09-20, branch `claude/tj14b-mentor-questions`, merged `161e905c` into
  `lead/p033-integration2`).** `scripts/mentor_questions.py` is PURE (no store, no Qt, no
  clock - every lane arrives in the `state` mapping the caller built) and is the card's SINGLE
  description of every question it may carry: nine kinds, each naming its trigger (a measured
  gap, never a clock alone), its click options, the store it `writes`, its `cadence` /
  `expiry`, its `priority` and - the point of the registry - the `consumer` that reads its
  answer plus the `answer_key` that consumer reads. `consumer_report()` resolves the dotted
  consumer and walks its SOURCE with `ast` to ask whether the key is used in CODE (a subscript,
  a call argument, a comparison, a keyword value); a comment, a docstring and a bare string
  statement are all refused, and a probe that CALLED the consumer would pass for `json.dumps`.
  **A question is asked only when its answer has a reader:** four kinds are registered in full
  and carry `dormant_until` naming the packet that builds it - `trade_origin` and
  `open_position_check` (TJ-12), `grader_gap` (TJ-10), `quick_like_followup` (TJ-14C).
  `pending()` never returns a dormant kind on a live card, `consumer_report` reports it as
  `dormant` with the packet named, and NO shim reader was written. **Budget:** beyond TJ-9's
  forced `trade_label` section and TJ-14A's forced prediction rows (`budgeted=False`), a card
  asks at most `BUDGET` = 3 questions by `priority` then kind; the rest are COUNTED on the card
  and CARRIED to the next one, never dropped, never a fourth, and AWAY asks nothing at all.
  `Stop asking this` (`STOP_ASKING`, the fifth option beside the four `ANSWER_STATES`) retires
  ONE subject through `TradeMentorService.stop_asking` - the single writer - persisted in
  `trade_mentor_slots.json` under `PERSISTENT_DATA_DIR`, so it survives a restart and a machine
  cache wipe, and it is never stored as an ANSWER. **The day's fills:** `pre_card_pull` is the
  ONE owner of the desk's day-time Questrade attempts (`PULLS_PER_DAY_CAP` 3,
  `PULL_FAILURES_PER_DAY_CAP` 2, `PRE_CARD_PULL_DAYS` 2, one tally keyed on the day and reset
  by the next, a corrupt tally read as empty and rewritten clean) and ONE card starts AT MOST
  ONE import: TJ-9's three-day morning catch-up goes FIRST when it is owed, outside the cap,
  with `last_retry` stamped only when an import actually STARTED, and otherwise the two-day
  pre-card pull runs on one of the day's three RESERVED, SPACED cards (`pull_slot_ids`: the
  09:00 card, the middle card and the LAST card, read off the session's real slot list, so an
  early close forfeits the middle attempt and never rolls it earlier). It calls
  `JournalImportService` (its own `QThread`, Questrade only), never refreshes a token itself,
  never runs in AWAY and never raises; the card never waits, and a fill a late pull lands is
  asked about on the NEXT card. IBKR has no day leg and the card says so.
  **Same-session fills:** `trade_mentor_trade_check.build_task` lists today's unlabelled fills
  BESIDE the reviewed session's and is not gated on import coverage, every trade block naming
  its own session; the card MERGES each delivered slot's fresh task (a row already on the card
  keeps its exact widgets and values, a new one is added after them, a row no longer owed is
  dropped, the heading and freshness line are always rewritten, the Save gate recomputed over
  all rows); and `save_answers` writes `label_provenance` from `trade_origin.label_provenance`
  with the BOOLEAN `recalled_after_session` that matches, instead of the constant `True` every
  row used to carry - a DATE-ONLY first fill is never `same_session` and says why in
  `label_provenance_reason`. **The AI question:** `market_story_narration.NARRATION_JSON_SCHEMA`
  gains the OPTIONAL `mentor_question_options` (array, `maxItems` 4, string items, `maxLength`
  60) and VALIDATES it - a five-option night is rejected WHOLE (`degraded_no_narrative`) and the
  prior verified file stands byte-identical - so the overnight question becomes ONE click a day
  whose answer is a dated Market Journal row instead of the `One thing to test: ...` line
  printed on every card forever with no options and no answer; the legacy line is hidden
  whenever the click is offered. `selftest.LAZY_ENGINE_MODULES` gains `mentor_questions`
  (source selftest 92/92 -> 93/93), because the registry is imported lazily inside the Qt slot
  whose guard would swallow a missing module silently. Shadow of nothing: no detector, score,
  gate, alert, watchlist, Focus, review queue or `review_policy.json` is reachable from it.
  Tests: `tests/test_tj14b_*.py`. Rule: DESK_INTERNALS "TJ-14B". Gate #159's TJ-14B clauses
  and the new gate #163.
- **The Journal's Trades splitter opens at its declared 3:2 and the tag-review row no longer
  eats the tab (WS-J1, WISHLIST item 1 leftover, 2026-09-13, sweep branch).** Two layout defects
  in `scripts/ui/panels/journal/trades_tab.py`: the `QSplitter` declared `setStretchFactor` 3:2
  but never `setSizes`, so the opening split came from the children's size hints (measured
  1347 / 2105, the trader's 39/61) - `showEvent` / `resizeEvent` now call `_apply_splitter_ratio`
  until the trader's own drag (`splitterMoved`) sets `_splitter_user_sized` and the tab stands
  down for the desk session, nothing persisted; and `tag_filter_note` (an empty label) kept Qt's
  default `Preferred` vertical policy as the row's only growable item, so the ROW absorbed ~960 px
  of a 2,160 px tab above an empty table (the blank band) - pinned `(Preferred, Fixed)`. Layout
  only: no number, sort, read or write moved (golden headers/row-count pinned). Tests:
  `tests/test_ws_j1_journal_splitter.py` (6).
- **The Weekend Prep verdict card's two cohort lines read NUMBERS** (WS-5A,
  2026-09-12; `scripts/weekend_verdict.py` + `scripts/ui/panels/weekend_prep_panel.py`).
  Neither line had ever printed a cohort: `best_cohort_line` read `avg_r_h3` /
  `n_h3`, columns nothing writes, off rows whose return was already the table's
  formatted `+1.23%` - so both said "nothing with enough behind it yet" against
  115 graded veto and 129 graded like rows, and a percent was about to be printed
  as R. `_cohort_numeric_fields` types the row (`horizon_sessions: int`, `n: int`,
  `avg_side_return_pct: float | None`, a blank staying `None`), `_cohort_cell_text`
  formats at the display edge so the two tables read exactly as before, and
  `_cohort_view` compares the horizon as an int. The card's `CARD_HORIZON` is the
  integer 3 SESSIONS, the pooled `ALL` side never leads a ranking of reasons, and
  BOTH lines take the HIGHEST side-adjusted return: "Likes that work: ... over 3
  sessions (n=..)" and "Rejections worth another look: ... side-adjusted (n=..)" -
  `min()` named the rejection that was right. Thin says "(best n was N against a
  floor of F)" and empty says "no like|veto cohorts measured yet"; the floor stays
  the card's own `MIN_COHORT_N` (5). Tests: `tests/test_ws_5a_weekend_verdict.py`.
- **`unresolved` means UNMEASURED** (M2, 2026-09-05). `scripts/outcome_semantics.py`
  gained a second half beside `claim_kind`: `terminal_kind(row)` returns
  `measured_eod` / `measured_swept` / `unmeasured` / `open` from the outcome row's
  `status` plus `context_json.finalization`. The after-close sweep needs no bars, so
  every row it wrote was `unresolved` by construction - **3,607 of the 4,251
  `unresolved` rows in the twenty sessions to 2026-09-05 carry a measured basis**
  (`last_measured_bar` / `stop_hit_from_prior_measurement`), and their R was already
  readable under `setup_scoreboard.exit_policy_r`. The writer now says which
  (`swept_measured` when a prior measurement was used, `unresolved` ONLY when nothing
  was measured, `eod_complete` unchanged) through the one decision,
  `status_for_finalization_basis`; the value is ADDITIVE, the CSV header is unchanged,
  `schema_version` stays 4 and **no historical row is rewritten** - `terminal_kind`
  reads them. The sweep logs the four-way split (`outcome_sweep_log_line`), the
  Daytrade Tracker status line and the AWAY digest print
  `format_terminal_coverage`'s one sentence, and `held_run_score.Episode` carries
  `terminal_kind` off the read it already does. **The champion aggregator
  (`_latest_bounce_outcome_rows`) is deliberately unchanged** - it takes
  `eod_complete` rows ONLY, so no tier, mute or PROVEN stamp moved; whether it
  SHOULD count swept-measured rows is a scoring question and stays ask-first.
  `setup_scoreboard` never read the status column at all. Live at merge:
  measured_eod 3,820, measured_swept 3,607, unmeasured 644, open 90 of 8,161 events.
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
- **Questrade fills know what they are, and a sold put is a sale (TJ-9Q, 2026-09-19/20;
  merged into `lead/p033-integration2` `86c64f96`, not on `main`, stored rows NOT yet
  moved).** `v1/accounts/{id}/executions` states no `securityType` and no `symbolType` —
  226 of 226 recorded payloads carry exactly 20 keys and neither — so every Questrade fill
  in the journal is `security_type = 'UNKNOWN'` and every option fill was priced with a
  contract multiplier of one; and `journal_importers.normalize_side` did not know `STO`,
  `BTC` or `COV`, so they were stored as the broker spelled them and
  `journal_store._signed_quantity` read them as buys. The trader's three sold-put
  positions (four STO fills) opened LONG and have sat OPEN with `quantity_closed = 0`
  since June, and the one put they bought was recorded at 1/100th of its loss.
  `journal_importers.classify_questrade_security_type` now reads the broker's own symbol
  and side words (both must agree; a payload that states a type is believed over both; a
  disagreement, an unreadable symbol or an absent side stays `UNKNOWN`; nothing is read
  from a ticker's length) and `normalize_side` maps `STO`→SELL, `BTC`→BUY, `COV`→BUY (the
  old map kept by name as `normalize_side_pre_tj9q`). The stored SYMBOL never moves:
  `canonical_option_symbol` rewrites a symbol as soon as the type is `OPT` and the symbol
  is half of `journal_identity.group_key`. **The seams are gated, not the functions:**
  `QUESTRADE_INSTRUMENT_FROM_SYMBOL` ships `False`, the effective value is read at call
  time from `local_settings["questrade_instrument_from_symbol"]`, and while it is off
  `QuestradeImporter._append_normalized` and `journal_statement_import._execution_from_row`
  store exactly what they stored before (raw side word, `UNKNOWN`, multiplier 1.0,
  `STO`/`BTC`/`Cov` CSV rows still dropped) — 29 Questrade positions are open under the old
  convention and one journal may never hold both. `journal_file_authority._BUY_SIDES` gains
  `COV` (the set held `COVER`), correcting a comparison in which each of the trader's 25
  covers counted as cash coming IN; that one is ungated, has no automatic caller (a
  trader-initiated statement import or "Check a statement..." only), and makes the file
  AGREE with the sync on the 14 (account, day) pairs it moves. Stored rows move only
  through `scripts/journal_reclassify.py` (dry run by DEFAULT, reads copies, opens the
  target database only under `--apply`), which backs up byte-exact first, moves
  `security_type` + `side` + `multiplier` together through the one additive
  `JournalStore.reclassify_executions` (Questrade rows only), rebuilds, verifies, restores
  the backup if anything is wrong, re-keys every `trade_annotations` row by largest
  execution overlap and REFUSES a tie rather than guessing, carries the machine's
  trade-keyed rows by the same rule (`ai_trade_enrichment` carried or its position
  refused; `note_lane_verdicts` carried or dropped under `refresh_auto_tags`' own rule for
  that derived table; `opportunity_events` never rewritten, carried by a `trade_aliases`
  row with unresolvable references counted and named), prints the broker's own `totalCost`
  cash beside the recomputed P&L while saying `net_amount` is ABSENT for Questrade, lists
  every (account, day) whose file-authority cash moves, and writes the `local_settings`
  key LAST — **and only when nothing was refused and no open `UNKNOWN` Questrade position
  remains**, otherwise the rows that moved stay moved and correct, the switch stays OFF,
  the blockers are named and it exits 4. Exit codes: 0 done · 1 verify failed and restored
  · 2 would not start · 3 busy (the desk slot or the `ai_jobs_runner` lock) · 4 moved,
  switch stayed off, re-runnable; an interrupted `--apply` is repaired by running it
  again; `--apply` refuses a database under `C:\TradingBotData` or the DAS without
  `--i-am-the-trader`. **Running `--apply` on the live journal is the trader's own act.**
  Measured on copies of the live journal, 2026-09-19/20: 226 fills move, 0 refused,
  exactly FOUR positions change and no equity trade, total recomputed P&L 4,737.45 →
  4,184.25 (AAOI +241.03, BE -666.99, QBTS -57.96, QQQ -81.98), all 616 `execution_uid`s
  and every broker-stated amount byte-identical, 185 annotations in and out with
  `label_provenance` still empty on all 185, `ai_trade_enrichment` 24 of 24, 94
  `note_lane_verdicts` carried and 4 dropped, and `journal_tax_report.build_tax_report`
  byte-identical for 2026 (76 positions, CAD 5,426.33) and 2025 (24, CAD 1,308.97). Tests:
  `tests/test_tj9q_questrade_sides.py`, `tests/test_tj9q_builder_additions.py`,
  `tests/test_tj9q_money_guards.py`, `tests/test_tj9q_reclassify_cli.py`,
  `tests/test_tj9q_review_round.py`. Rule: DESK_INTERNALS "TJ-9Q".
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
  condition for the mark. **The window is SESSIONS and the counts are BY TRADE**
  (ST5, 2026-09-06): `TRADE_WINDOW_SESSIONS` (10) walked through
  `market_calendar` by `statement_window_end` replaces `TRADE_WINDOW_DAYS`
  (kept one release as an alias), so a statement on Friday 2026-09-04 reaches
  2026-09-21 rather than 2026-09-14 and a Labor Day week no longer throws five
  sessions away; a calendar that refuses falls back to the OLD, strictly
  NARROWER arithmetic, because uncertainty may not invent a match. Confidence
  labels are unchanged. `trade_level_summary` sums P&L ONCE per `trade_id` over
  a file that stays one row per statement (live: 13 `traded=yes` rows over 10
  distinct trades), `summary_note` prints both denominators, and
  `run_preference_trade_outcomes` carries `n_statements_matched` /
  `n_trades_matched` out of the slot.
- **Ownership is not market bias** (ST5.3, 2026-09-06,
  `scripts/journal_exposure.py`). `classify_exposure(trade)` ->
  `Exposure(instrument, ownership_direction, market_bias, structure,
  certainty)`. `trades.direction` is the sign of the opening quantity and
  nothing else; read as a market view it makes the live journal a bearish trader
  with a bullish record (53 of 89 option trades SHORT, 39 of those winners -
  they are sold puts). **A LONG option is never a bullish setup**: a bought put
  is `bearish`, a sold put `bullish_or_neutral`, a sold call
  `bearish_or_neutral`, stock follows `direction`, and `UNKNOWN` / `BAG` /
  `CASH` stay `unknown`. **A `trade_legs` row is a FILL**, so `multi_leg` means
  more than one distinct option CONTRACT among the legs - two fills of one
  contract is not a structure; the contract is read from the OCC `trades.symbol`
  and from `raw_executions.raw_json["option"]`, which is why
  `JournalStore.list_trade_legs` gained ONE column (`e.raw_json`). The store has
  no sibling seam for a spread, so a second option trade on the same underlying,
  expiry and session is `partial_of_spread_candidate` - never
  `partial_of_spread` - and lands in the uncertain population rather than
  claiming half a spread.
- **Three populations by STATUS, and uncertainty is a LABEL across them**
  (ST5.4, 2026-09-06, `journal_analytics.personal_evidence_summary`, additive on
  `build_analytics_summary` as `personal_evidence`). `complete` (CLOSED) /
  `partly_closed` (CLOSED_PARTIAL) / `open_exposure` (everything else) PARTITION
  every trade - each with n, winners, CAD and USD P&L, the `market_bias` split
  with `unknown` printed as its own bucket, and `n_uncertain` beside it. The
  cross-cutting `uncertain` block (UNKNOWN instrument, BAG, CASH, `multi_leg`,
  `partial_of_spread_candidate`) lists its members WITH the status each is
  counted under and **pools no money at all**, because its members span three
  statuses. **Checking uncertainty first made it a fourth bucket that ate the
  other three**: measured on a copy of the live journal, `uncertain` came out
  n=120 holding 84 CLOSED, all 7 CLOSED_PARTIAL and 29 of 32 OPEN trades, so
  `partly_closed` read n=0 and one pooled figure summed realized results with
  open positions' unrealized marks and counted those marks as WINNERS.
  **An open position has NO result**: `net_pnl` and `winners` are both `None`,
  never zero, and its size travels as `notional` (61,662 live). An EMPTY bucket
  reports `None` too - a net of 0.00 says "measured and it came to nothing".
  **No personal setup is called best without confirmed tags at the floor**:
  `best_setup` is `None` until CONFIRMED tags reach
  `evidence_stats.MIN_REPORTABLE_N` (30 against 1 live); above it the winner
  ranks on `swing_headline`'s Wilson lower bound. **Both tag lanes share ONE
  denominator, closed OR PARTLY CLOSED** - the live journal's single confirmed
  tag sits on a CLOSED_PARTIAL trade, and counting confirmed over CLOSED while
  counting provisional over everything made the headline say "No confirmed setup
  tags" about a journal that holds one. The coverage line -
  `Confirmed tags: 1 of 172 closed or partly closed trades. Provisional awaiting
  review: 26. Planned risk recorded: 0 of 172.` - and the headline `1 confirmed
  setup tag - under the n=30 floor (...) - no personal setup can be called best`
  reach the Journal's Analytics tab and Weekend Prep through that one helper;
  the "No confirmed" wording is reserved for a true zero.
- **The whole tag backlog reaches the review screen, and a missing plan is a
  worklist** (ST5.5, 2026-09-06, `ui/panels/weekend_prep_panel.py`).
  `_read_week_tag_rows(bounds, *, store=None, path=None)` gained an injection
  seam and lists EVERY `provisional` trade rather than the current week's (26
  waiting on 2026-09-06 against a page scoped to one week - gate #36's
  obstacle); `needs_review` stays week-scoped because those 145 rows carry no
  proposal and would bury the ones that do, and the week's rows sort first with
  a `Week` column naming the population. The new "Missing planned risk" table
  lists closed trades with `planned_risk` null (0 of 204 live), newest first, on
  the ten-row floor and capped at the newest `MISSING_RISK_ROWS_SHOWN` (50) with
  `showing 50 of 165` printed, and a row only REFERS: `openTradeRequested` ->
  `JournalPanel.show_trade` -> `TradesTab.select_trade`, the tab where
  `save_risk_fields` already lives. **Nothing here writes `planned_risk` and
  nothing computes one from an outcome**; a test spies `save_risk_fields` into a
  raise and asserts every reader leaves it uncalled. The coverage sentence sits
  UNDER the verdict card, never inside it - the card is five to eight lines by
  the trader's own request.
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
- **That read is TIMEZONE-AWARE, and a failed read names itself** (N1,
  2026-09-05). A sidecar's `dt` is naive DESK-local wall time - the live SHW row
  opens at `2026-09-01T06:30:00`, the RTH open on a Pacific desk, while
  `created_at` on the same file carries `-07:00`. `pass_bars.desk_zone()` is the
  ONE named seam for that zone (a configured `market_local_timezone` wins,
  otherwise the platform is asked PER MOMENT so DST is right on both sides of a
  transition - Windows has no IANA key and `market_session` falls back to an
  offset frozen at "now"), re-exported by `sidecar_completion` so both modules
  attach the same thing. A naive moment is ATTACHED and an aware one is never
  stripped; `_session_close` is 16:00 **market-local** rather than 16:00 in
  whatever zone the bar was read in (it was 16:00 Pacific, three hours past the
  real close); and `_serialisable_bar` writes the offset from now on, so a new
  sidecar needs no convention - `sidecar_schema_version` stays **1**, because an
  offset on a stamp that already had to be parsed is additive. The two lake
  failures are SPLIT: `ResearchStore.open()` refusing is the only thing that
  means `research_store_unreachable`, and a read that faults returns
  `lake_read_failed: <ExceptionClass>`. Naive bounds against
  `bar_m5.interval_start` (`timestamp[us, tz=UTC]`) raise `ArrowInvalid`, which
  the blanket `except` had called unreachable every night since 2026-09-02 while
  the store answered **60 rows** to the same window with aware bounds - so gate
  \#39's blocker was a timezone, not a share.
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
- **A detail pane never outlives the context that opened it** (G4, 2026-09-06).
  `ResearchExplanationView` and `SetupDetailView` each carry `shown_identity` and
  an OVERRIDE of `clear()` that empties, HIDES and forgets - `QTextEdit.clear()`
  alone leaves an empty pane standing. On the Day-trade Tracker the identity is
  `(kind, dimension, direction, segment)` read from the ROW DICT, never the
  display text; either tab strip changing clears the pane, and a data revision
  (`_on_refresh_finished`, `_on_held_run_loaded`) looks that identity up in the
  model that now holds the tab's rows and redraws from the **new** row dict, or
  clears when the revision dropped the segment. Display only: no model, sort,
  read or number changes, one dict lookup per revision on the Qt thread.
  **BOTH Research detail panes now carry the rule** (G4b, 2026-09-07): the Setup
  Tracker's `detail_view` clears on any move of its fourteen-tab strip, and the
  END of `refresh()` re-shows the open row from the NEW row dict or takes the
  pane down. **The visibility question is asked FIRST** - the trader reaches the
  hidden state by moving tabs, so a re-show keyed on a match alone would pop an
  explanation open under someone who had closed it. The match is one linear scan
  (`row_at`, no dict copied per row) of the CURRENT tab's model, found through
  `_detail_tables` by asking which tab widget owns the table, never by tab
  position, and re-shown through that tab's own `show_*` call. `shown_identity`
  is coarse on this page - a Setup Types row has no `dimension` and no `symbol` -
  so the scan widens with `DETAIL_WIDENING_KEYS` (`favorite_zone`,
  `priority_bucket`), normalised so a row carrying neither compares equal on both
  sides; a pair differing only in `retest_label` still collides and falls to the
  first such row in the model's own order. The dead `_on_family_row_clicked` was
  removed (`SetupDetailView.show_family` is untouched). Display only on this
  panel too: a golden pins all fourteen tabs' render across the change.
- Review events partitioned by installation, merged/deduplicated by readers, capture
  audits, preference scoreboard, AI-curated `review_policy.json`, and a permanent
  no-suppression boundary.
- Technical Integrity research hierarchy with point-in-time predictions/outcomes,
  break pressure, calibration report, and no detector/watchlist/alert influence.
- Regime infrastructure evidence for SPY baseline, breadth, Technical Integrity
  follow-ups, and audit tooling. The evidence remains exploratory/non-promotable.

### AI and automation

- **Trustworthy overnight output (WS-AI1, WISHLIST 10K step 1 + 5C, 2026-09-12, sweep
  branch).** The per-trade journal enrichment has its OWN validated contract,
  `ai_jobs.enrichment.ENRICHMENT_JSON_SCHEMA` (`summary`, `tags`, `confidence`, `sources`,
  `unknowns`, closed), sent down the SAME provider path: `ai_summary.request_ai_summary` takes
  `schema` / `schema_name` / `prompt_version` with the session summary's schema as the default
  (every existing caller's payload byte-identical), and `ai_summary.validate_structured_output`
  validates a caller-supplied contract. `_proposed_tags` / `_summary_text` are the ONE
  extraction seam and read exactly that schema's keys - the previous reuse of
  `AI_SUMMARY_JSON_SCHEMA` forbade every key they read, so six trades over 2026-09-09..11
  carried a blank `ai_trade_enrichment` row while the ledger said `ok`. A row now says what it
  is: `ai_trade_enrichment` gains `status` (`enriched` / `abstained` / `failed`), `reason` (the
  model's own `unknowns`, or the error class), `confidence` and `supersedes_row_id`
  (`NEW_COLUMNS_V3`, additive, idempotent), and the slot reports `enriched A, abstained B,
  failed C of N` with `STATUS_OK` only for `A + B == N, C == 0`. `enrichment.is_legacy_blank`
  (blank summary AND blank tags AND no status) is what `_trades_for_session` refuses to read as
  done, so the blank rows are repairable; the repair APPENDS a row naming the one it replaces
  and never rewrites. Every published nightly summary names its completion in one top-level
  word - `map_reduce.completion_word`: `synthesized` / `partial` / `unsynthesized_fallback` /
  `failed` - and `briefs.run_daily_summary` publishes `STATUS_OK` only for `synthesized`, still
  publishing the document and naming the synthesis error verbatim; the word reaches the ledger
  row through the runner's `extra` and the System Health AI row prints it. Two defects in that
  reader were fixed with it: `operations_audit._ai_jobs_check` counted `degraded` where the
  ledger constant is `degraded_no_narrative` (no AI job could ever show as degraded), and read
  `ts` / `timestamp` where `ledger.record` writes `started_at` / `finished_at` (every AI row read
  as undated). The advice has a reader: `ui/services/journal_feed.latest_ai_enrichment` (newest
  row nothing supersedes) rendered by `TradesTab._show_trade`, marked advisory, an abstained or
  failed row shown rather than hidden. A new package scope `preference_to_trade` carries a
  derived, bounded section over ST5's `preference_trade_outcomes.csv`
  (`ai_summary.preference_to_trade_section`): three grains kept apart, coverage derived from the
  report's own `match_basis` plus the 10-SESSION window (`journal_unavailable` / `window_open` /
  `no_match_after_window`, summing to the unmatched count), at most 20 examples selected by
  `(session_date, report row order)` descending - no result column enters that key;
  `REPORT_FILE` resolved at call time. **Lead decision 2026-09-12 (trader may overrule):** the
  scope joins `briefs.DEFAULT_SCOPES` (the nightly slate is six), because 5C says the summary
  is fed "into the existing AI package". No timeout was raised and no model changed; the
  runner's stage order is unchanged and `weekly_synthesis` stays optional. Tests:
  `tests/test_ws_ai1_enrichment_status.py`. Rule: DESK_INTERNALS "AI1 - an enrichment row is
  never blank on success"; contract: `docs/LOCAL_AI_AUTOMATION_PLAN.md` Phase 3.
- **One shared measured report (WS-RP, WISHLIST 10K steps 2 + 5, 2026-09-15).**
  `scripts/measured_report.py` is a pure, versioned five-answer contract: broker money once
  per trade, side-adjusted opportunity and adverse movement, target speed in trading minutes,
  EOD, and independent observation/follow-through session controls. Every cell carries its
  population, window, reference clock, policy, source paths and measured/pending/unknown
  state; its stable report id changes only when evidence matures. The deterministic nightly
  `measured_report` slot publishes versioned JSON/Markdown siblings and indexes the newest
  report without failing the night. Daily Recap's fifth **Review** tab reads that published
  report off its worker and a manual Copy/Export emits a capped 32 KiB brief, payload and
  manifest with the existing `ai_summary._ESTIMATED_CHARS_PER_TOKEN` estimate. No report path
  calls a model, uploads, changes a live decision or writes a live store. Tests:
  `tests/test_ws_rp_shared_report.py` (44 behavior checks plus two unchanged N3 guards) and
  `tests/test_ai_jobs_runner.py` (deterministic-slot order). Gate #133.
- **`read_grades_mature` - the matured read is closed by arithmetic (TJ-10, 2026-09-20).**
  `scripts/ai_jobs/read_grades_mature.py` re-measures every OPEN market read and appends the
  new verdict. Deterministic: `uses_model=False`, `max_attempts=3`, `reserve_minutes=2.0`,
  registered directly after `theta_pick_grading` and before `miss_contrast` /
  `market_story_rollups` / `measured_report` so it stays inside `_deterministic_stage` and
  runs on the weeknight, Saturday AND Sunday slates. A missing reads directory, an
  unreadable ledger or a missing daily store each give an `ok` row with a reason in under a
  second and never raise; idempotent across the task's 30-minute re-firings; a night that
  cannot measure writes NOTHING, because an `unmeasured` result may not supersede a
  `pending` row. The only file it writes is `DAY_REVIEW_READS_DIR/<session>.jsonl`. Tests:
  `tests/test_tj10_read_grades_slot.py`, `tests/test_ai_jobs_runner.py`
  (`EXPECTED_SLOT_ORDER`). Slot position: decision 0018 addendum 2026-09-20.
- **`miss_contrast` - what the misses had in common (TJ-15, 2026-09-19).**
  `scripts/evidence_contrast.py` is pure: per feature two counts, two medians and ONE rank
  statistic through `compression_calibration.auc` (called, never re-derived), rank key
  `abs(auc-0.5)` then feature NAME with no group size and no R statistic in it;
  `MIN_REPORTABLE_N` gates a group's RATE and `MIN_CONTRAST_SIDE_N` (10 a side,
  `MIN_REPORTABLE_N` across both) gates a FEATURE, a thinner one named in `thin_features`
  with both counts and no AUC, `compared` counting the ranked only; `rate()` is the ONE
  Wilson over CLOSED horizons with `pending` in neither half. `scripts/ai_jobs/miss_contrast.py`
  is the deterministic Stage 1 slot (`uses_model=False`, `max_attempts=3`,
  `reserve_minutes=5.0`), registered after the cohort graders and `theta_pick_grading` and
  BEFORE the `market_story_rollups` + `measured_report` pair that closes the stage, so it
  stays on the weeknight, Saturday and Sunday slates (`_STAGE_ONE_LAST_SLOT` untouched);
  `EXPECTED_SLOT_ORDER` gains one name. It judges **D1 decisions only**, counting every
  other timeframe in `excluded_by_timeframe` and a blank one in `no_timeframe`; per veto
  reason CODE (pooled across vocabulary versions) and once per other verdict it contrasts
  real misses against correct rejections on the LAST D1 scan at or before the decision's
  stamp, from its own session or the ONE before (`MAX_SCAN_AGE_SESSIONS = 1`, ages counted
  per group), streaming the ~709 MB `d1_features_history.csv` by session. A group is named
  only with a reportable rate AND a ranked feature. One JSON pack per session beside the
  digest in `store.digests_dir()` (`miss_contrast-<session>.json`, temp-and-rename onto a
  superseding sibling), `read_latest` the reader - a file read, to be called on a worker. An
  unreadable `session_date` refuses with `failed` and writes nothing; every other missing
  input is a recorded reason on an `ok` row. Shadow evidence: nothing reaches a detector,
  score, alert, watchlist, Focus, the review queue or `review_policy.json`. Tests:
  `tests/test_tj15_evidence_contrast.py`, `tests/test_tj15_point_in_time.py`,
  `tests/test_tj15_miss_contrast_slot.py`, `tests/test_tj15_miss_contrast_builder.py`,
  `tests/test_tj15_fix_round.py`. Rule: DESK_INTERNALS "TJ-15"; slot position: decision 0018
  amendment 2026-09-19. Gate #160 owed.
- **What leads to a good call - the prediction ledger beside its baselines, the
  right-against-wrong contrast, and a tagger that never sees an outcome (TJ-16,
  2026-09-20).** Three modules and two nightly slots, all REPORTED evidence: nothing here
  reaches a detector, score, alert, watchlist, Focus, the review queue or
  `review_policy.json`.
  `scripts/prediction_ledger.py` is the read ledger's reader. `read_ledger(sessions, root=,
  source=)` returns the CURRENT grade row per read (`market_read_grades.current_grades`, so
  a matured re-grade counts once at its current verdict); `build_readout(rows)` raises
  `market_read_grades.PoolingError` on a clicked/extracted mix, keeps `rest_of_day` and
  `next_5_sessions` apart, and prints every accuracy cell beside what `always_up`,
  `same_as_the_last_hour` and `with_the_d1_environment` scored on the SAME stamps. A
  baseline has no stored verdict, so it is measured NOW through
  `market_read_grades._verdict_for` - the grader's own band, CALLED through its module
  attribute, so a superseding band moves every baseline while the stored rows stay where
  they were; a baseline with no direction (`compressed`) is `unmeasured`, never wrong.
  Calibration by `How sure` sets `high_beats_low` and says "High did not beat Low" when it
  did not, and `None` while either bucket is under `MIN_REPORTABLE_N`. `horizon_of` is
  public - ONE mapping, no cross-module private reach. An empty ledger reads `no clicked
  calls yet` with `rate: None`, never 0.0.
  `scripts/ai_jobs/prediction_contrast.py` is the deterministic Stage 1 slot
  `prediction_contrast` (`uses_model=False`, `max_attempts=3`, `reserve_minutes=5.0`),
  registered DIRECTLY after `miss_contrast` and still above the `market_story_rollups` +
  `measured_report` pair that closes the stage. Right against wrong per point-in-time
  context field through TJ-15's one `evidence_contrast.contrast`, in two populations
  printed apart (`lately` = `LATELY_SESSIONS` exchange sessions ending at the pack's
  session; `all` = every session on disk), per horizon, plus a `by_hour` and a
  `by_environment` table with `n`, the Wilson bounds and `reportable` on every cell. A
  `flat` reading is counted on the horizon and is in NEITHER contrast group; a numeric
  field keeps its own name, a categorical one becomes one feature per value worth 1.0 / 0.0
  on a MEASURED row and NOTHING on an `unmeasured` one; clicked reads are the population
  and extracted stances are counted in `excluded_by_source`. The contrast is called with a
  `top` covering every feature over the FEATURE floor, so nothing is hidden and nothing is
  selected by a result; the bounded view is `tendencies(pack, limit=3)`, ordered by `n`
  descending then name - a SIZE rule (gate #43) - and a cell under `MIN_REPORTABLE_N` is
  never offered. A `tags` block names the tag features considered (`entries_tagged`,
  `reads_matched`, `codes`, `features`, `entries_written_after`). `read_latest` is the
  reader; packs are superseding siblings beside the digest in `store.digests_dir()`.
  `scripts/ai_jobs/observation_tags.py` plus the vocabulary asset
  `scripts/ui/annotations/vocabularies/observation_tags_v1.json` (11 codes) are the Stage 2
  slot `observation_tags` (local MEDIUM model, `uses_model=True`, `max_attempts=3`,
  `reserve_minutes=15.0`), after `ai_summary` and before `ticker_briefs`. It codes
  `mentor.observation` and `mentor.prediction.because` from a closed versioned vocabulary,
  each code carrying the exact span that must reproduce its quote. The payload is BUILT
  from the two texts and the picklist, so it structurally contains no verdict, grade,
  price or bar. `verify_reply` believes an answer only if it clears BOTH halves - the
  GROUNDING (every span reproduces its quote, every code is in the vocabulary, every
  `note_id` was offered) and the SHAPE (no more than `MAX_TAGS` = 60 rows, no key outside
  `REPLY_KEYS` / `TAG_KEYS`, no byte-identical duplicate row) - because the schema's own
  `maxItems` and `additionalProperties` are a grammar hint to the provider and are
  re-checked here. Any failure rejects the WHOLE reply and the last verified file stays
  byte-identical. The session's notes are read through `EvidenceLedger.read(start=session,
  end=session)`, windowed at the stream and never filtered after the fact. Each stored tag
  carries the entry's computed `written_after_the_session` and the file header counts
  `entries_written_after`: hindsight is LABELLED, never dropped, never re-weighted, and the
  label never reaches the prompt. The vocabulary has its OWN loader (`load_vocabulary()`
  reads the highest `observation_tags_v*.json` and refuses a file whose declared
  `vocab_version` disagrees with its FILENAME; the shared `ui/annotations/vocabulary.py`
  `_parse` is veto-shaped and serves the capture rail). The codes become `tag:<code>`
  context features for the NEXT contrast run - never the same one, the tagger being stage 2
  and the contrast stage 1.
  Slot order after this packet: `... read_grades_mature, miss_contrast, prediction_contrast,
  market_story_rollups, measured_report, ai_summary, observation_tags, ticker_briefs,
  market_story_narration, ...`. `your_reads`, `tendencies` and `read_latest` ship as READERS
  with **no page yet** - Day Review's `Your reads` line is TJ-12's and the Week Review
  tables are TJ-5's. Tests: `tests/test_tj16_prediction_ledger.py`,
  `tests/test_tj16_prediction_contrast_slot.py`, `tests/test_tj16_observation_tags.py`,
  `tests/test_tj16_tagger_bounds.py`, `tests/tj16_support.py`. Rule: DESK_INTERNALS "TJ-16";
  slot positions: decision 0018 addendum 2026-09-20. Gates #164-#167 owed.
- **The day pack, the overnight day story that narrates only measured rows, and the rolling
  D1 view (TJ-4, 2026-09-20).** Shadow evidence and one trader-facing page area: nothing
  here reaches a detector, score, alert, watchlist, Focus, the review queue or
  `review_policy.json`.
  `scripts/day_review_pack.py` builds ONE hash-stable pack per session at
  `DAY_REVIEW_DIR / "sessions" / <date> / "pack.json"` - twelve `SECTIONS` (`trader_said`,
  `forecast`, `environment`, `measured`, `internals`, `walkaway`, `skill`, `reads`,
  `congruence`, `trades`, plus the `report_card` and `mood` hooks, present and empty).
  The builder is PURE: it opens no store and has no clock of its own, the caller hands it
  the day it already read, and `inputs_hash` covers the INPUTS while the clock lives in
  `built_at`, outside the hash - so a second build hours later is byte-identical. Every item
  names a `source_id` that ONE `_Minter` guarantees is unique inside that pack (the nth
  claimant gets `<base>#n`, nothing dropped and nothing raised), because ids are derived
  from what they point at and a reader keyed on a colliding id would silently take the LAST
  row. An observation and a prediction from one Mentor card are SEPARATE items, an empty
  observation emits no item, and a machine row or the pasted forecast never reaches
  `trader_said`. `walkaway.top` carries the three rows that ran furthest after the decision
  - an ORDER, never a ranking that decides anything - each with an added `population` key.
  The pack sits INSIDE `day_review_index._prune`'s 40-session delete path, which is safe
  only because it is rebuildable; the hash-stability test is what proves it, and the write
  seam says so in a comment.
  `scripts/ai_jobs/day_review_narration.py` is the Stage 2 slot `day_review_narration`
  (local medium model, `uses_model=True`, `reserve_minutes=10.0`, `max_attempts=3`),
  registered after `ai_summary` and before `observation_tags` and `ticker_briefs`. It writes
  `DAY_REVIEW_DIR / "narration" / <date>.json` (`day_review_narration_v1`) and the ONE
  rolling `DAY_REVIEW_DIR / "d1_view.json"` (`d1_view_narration_v1`), built from the
  trader's D1 PREDICTION clicks and D1 notes of the last `evidence_stats.LATELY_SESSIONS`
  sessions - never an M5 item, and `market_thesis.current_theses` is not read. Both
  artifacts carry their own `inputs_hash`, so an unchanged input costs no call, and they
  have two independent verdicts: a rejected day story never costs the rolling view its
  prior file. **Six rules reject an output WHOLE** - not trimmed, not partly kept, the last
  verified file byte-identical and the ledger row `degraded_no_narrative`: a
  `were_you_right[].verdict` that does not EQUAL the measured verdict of the `reads` row its
  `evidence_id` names; a claim on a read row the pack does not carry, or a second claim on
  one read; an `observation` graded as a call; a `chased_against_news` other than `unknown`
  with no pasted forecast; an empty headline; and any closed-schema or bound breach. **The
  bounds come FROM THE PACK** - claims <= its reads, one claim per read, sources <= its
  citable ids, theses <= the D1 evidence - and the schema handed to the model carries those
  numbers; `MAX_GRADED_CLAIMS` (64), `MAX_SOURCES` (512) and `MAX_OPEN_THESES` (32) survive
  only as absolute render ceilings. A story that grades fewer reads than the session held
  records the count and the page says `graded K of N reads`, a size statement with no result
  in it.
  **The night SWEEPS what a daytime Redo queued.** After its own story and the rolling view
  it re-narrates up to `REDO_SWEEP_LIMIT` (3) marked sessions, oldest first (a marker
  overrides that session's unchanged-hash skip), spending that budget only on sessions it
  actually NARRATES: a marker with no pack costs nothing, keeps its place and is NAMED
  (`MAX_NAMED_UNBUILT` 5, then `+N more`), and no marker is ever retired by age. A marker is
  cleared only after a good run; a swept session's failure keeps its marker and never
  degrades the night's own status; a night whose own pack is missing still drains the queue.
  `day_review_pack.validated_session` fails CLOSED - exactly `YYYY-MM-DD`, a real exchange
  session, already closed - so `".."` and `"2026-09-18-extra"` can no longer write a marker
  outside a real day's folder, and `queued_sessions` ignores a folder that is not a session
  date. `reserve_minutes` buys the FIRST call only: the slot re-asks `ai_jobs.window` itself
  for `SWEEP_CALL_MINUTES` (9.0, the per-call timeout) before every call after that, and a
  window that has closed stops the run CLEANLY with every remaining marker kept and said.
  The clock is ONE injectable seam - `now` plus the run's own monotonic elapsed - and
  nothing sleeps.
  **The page** (`ui/services/day_review_service.py`, `ui/panels/day_review_panel.py`) reads
  the two verified files on its ONE worker as `day_story` and `d1_view` in the ONE payload,
  prints the headline over the deterministic facts, and its **Redo story** button BUILDS
  that session's pack first on its own off-Qt `_RedoPackWorker` - then writes the
  `redo_requested` marker outside the night window (no process at all) or starts one
  below-normal child process inside it. A build that produces nothing says `No pack could be
  built for <date> - nothing queued`. **One redo is in flight at a time**: the button is
  grey from the click and released by `_release_redo` on EVERY ending (queued, launched, no
  pack, refused, a launcher that raised, a build that raised), so three impatient clicks are
  one build, and a session switch mid-build neither frees the button early nor strands it.
  `scripts/run_ai_jobs.py --session YYYY-MM-DD` is accepted ONLY with
  `--slot day_review_narration` (anything else is a parser error, exit 2, nothing run),
  validates through that same `validated_session` (fails closed, exit 2), and reaches that
  one slot through an additive `run_slots(session_override=...)`; every ledger row that run
  writes - the window refusal included - is keyed to the session the slot WORKED on.
  `selftest.LAZY_ENGINE_MODULES` gains `day_review_pack` and `ai_jobs.day_review_narration`:
  source selftest **93/93 -> 95/95**. Tests: `tests/test_tj4_day_pack.py`,
  `tests/test_tj4_narration_slot.py`, `tests/test_tj4_d1_view.py`,
  `tests/test_tj4_day_review_page.py`, `tests/test_tj4_redo_sweep.py`,
  `tests/test_tj4_review2_fixes.py`, `tests/test_tj4_redo_one_at_a_time.py`,
  `tests/test_tj4_redo_session_cli.py`, `tests/tj4_support.py`. Rule: DESK_INTERNALS "TJ-4";
  slot position: decision 0018 addendum 2026-09-20. Gate #148 owed.
- **The Day Review report card - six lines that measure nothing and say what they could not
  see (TJ-12, 2026-09-20).** One trader-facing page area and two Mentor question kinds woken:
  nothing here reaches a detector, score, alert, watchlist, Focus, the review queue or
  `review_policy.json`.
  `scripts/day_report_card.py` is PURE - no store, no Qt, no clock of its own - and
  `build(day_inputs) -> ReportCard` returns SIX lines in the fixed order `LINE_KEYS`
  (`did_well`, `missed`, `your_reads`, `congruence`, `process`, `how_fresh`), each
  `{key, text, n, measured, target}` and each resolving its click through `LINE_TARGETS`.
  **It computes NO new statistic**: every number is READ from the owner that measured it -
  `walkaway_day`'s `REAL_MISS_V1` verdicts, its sentences and its skill cells' own `low`;
  `prediction_ledger.your_reads`' integers; `market_read_grades.congruence_lines`;
  `trade_origin.planned_state` with the trade row's own `label_provenance`. The best setup
  family is chosen by the ONE Wilson **lower bound**, never the rate, and none under
  `MIN_REPORTABLE_N` is named; a line whose input the desk does not have SAYS so and prints no
  number, and `n - measured` never enters a rate. The DAY lines deliberately carry no
  `rate_lb`; only `week(sessions)` computes one, from POOLED counts once per session (never
  the mean of two days' rates), and it names its floor - that re-cut is TJ-5's week /
  four-week / month strip. **Every line is guarded on its own** (`_guarded` /
  `_unreadable_line`): one owner raising costs its own sentence (`could not be read: ...`,
  `measured_ok: False`, `measured` staying an integer 0) and the other five still say what
  they measured. `did_well`'s `n` is the `liked_not_traded` table alone - the table its click
  opens - with claimed D1 picks said separately.
  `how_fresh` names when the story was written, how far the VERIFIED fills reach
  (`trade_mentor_trade_check.fills_current_to`; an absence is not a date), which session the
  reads were graded through, and any overnight slot the ledger says went wrong. **A ledger
  `status` is a vocabulary `ai_jobs/ledger.py` OWNS and this module re-spells none of it**:
  `STATUS_OK` and `ATTEMPT_STATUSES` are IMPORTED, `skipped` / `manual_test` / a `correction`
  row decide nothing, `failed` and `degraded` are named SEPARATELY, and the **LAST deciding
  row wins**, so a slot that failed and recovered is not named and one that ran and then
  failed is. The ledger is read TAIL-ONLY through an EXPLICIT path (`LEDGER_TAIL_ROWS` 500,
  `ai_jobs.ledger.recent_rows` for a small file) and past `LEDGER_TAIL_BYTES` (256 KB) the
  tail is SEEKED from the end inside `day_report_card`; the file is opened exactly once
  either way and `ai_jobs/ledger.py` is unchanged for its other callers. The tail says what it
  could NOT see, so a session at or beyond its oldest row is `night_status: "unknown"` with
  **no counts at all**, a covered session with no rows is said as `no_rows`, "none reported
  trouble" is said only over at least one slot actually read, and a MISSING AI store is
  `night status unknown` and **creates nothing** (`ledger_path()` defaults to `create=True`).
  **The Process line never reads an unread store as an answer.** `trade_origin.planned_state`
  answers `unplanned` whenever no lane row precedes the first fill, so an UNREAD lane is
  indistinguishable from "nothing was said". While any of `ORIGIN_LANES` sits outside
  `DESK_ORIGIN_LANES_READ` - ONE constant both lane builders build from - the line counts
  `planned` and `no claim or like before the fill`, never a bare `unplanned`, and NAMES what
  it could not look at, carrying `lanes_read` / `lanes_unread` so TJ-5 and the pack say the
  same; declare every lane read and the plain wording returns on its own. The reader for the
  two unread lanes is the follow-up **TJ-12F** (plan.md 12.4).
  The card is built on the Day Review WORKER inside the ONE payload (TJ-1):
  `day_review_service.PAYLOAD_KEYS` gains `report_card`, `empty_payload` carries it present
  and falsy, and `read_day` fills it LAST inside its own guard - a failed card costs the card,
  never the day. The page FORMATS and never calls `build` or `how_fresh`:
  `day_review_panel.report_card_section` is a third row of the page's own column between
  `provisional_note` and `columns`, six fixed line widgets built once and only re-texted,
  styled in `ui/theme.qss` by object name. `day_review_pack.build_pack(..., report_card=None)`
  keeps TJ-4's empty default and, GIVEN a card, mints one `source_id` per line from the pack's
  ONE `_Minter` inside the hashed body. **`build_pack_for` hands it the session's card since the
  follow-up `c37d5673` (merged `ccecf320`)** through ONE seam, `day_report_card.pack_card` /
  `PACK_LINE_KEYS`: the five DAY lines go in and `How fresh` stays OUT, because it describes the
  machine's night, its text moves with every ledger row, and inside the hashed body it would
  re-narrate the same session night after night.
  **Two Mentor question kinds WOKEN** (TJ-14B -> TJ-12): `trade_origin` and
  `open_position_check` carried `dormant_until="TJ-12"` because this card is their named
  reader, so `dormant_until` is now `""` on those two and their `consumer` strings name
  `day_report_card.process_line` / `.long_hold_lines`; `grader_gap` (TJ-10) and
  `quick_like_followup` (TJ-14C) stay dormant. `MainWindow._mentor_origin_lanes` really fills
  the `decisions` and `claims` lanes, bounded to the two sessions a question can be about;
  `ORIGIN_OPTIONS` gains `a_focus_pick` and `ORIGIN_PROMPT_CAVEAT` leads with the same words
  the Process line uses, so the trader can answer what the desk cannot yet read.
  `selftest.LAZY_ENGINE_MODULES` gains `day_report_card` (both the worker and the page import
  it inside a guard that swallows an `ImportError`): source selftest **95/95 -> 96/96**.
  Tests: `tests/test_tj12_report_card_lines.py`,
  `tests/test_tj12_report_card_calls_the_owners.py`, `tests/test_tj12_how_fresh.py`,
  `tests/test_tj12_week_recut.py`, `tests/test_tj12_day_review_page.py`,
  `tests/test_tj12_pack_hook.py`, `tests/test_tj12_mentor_wake.py`,
  `tests/test_tj12_review1_fixes.py`, `tests/test_tj12_review2_tail_window.py`,
  `tests/tj12_support.py`. Rule: DESK_INTERNALS "TJ-12". Gates #157, #168-#170 owed.
- **Week Review - the week the trader opens Weekend Prep to read (TJ-5, 2026-09-20).**
  Weekend Prep's FIRST step becomes five DAY CARDS (headline, were-you-right tally, chased
  flag, said-vs-did line and a small SPY sparkline), the overnight week story, the week's
  walk-away totals and a deterministic strip of TJ-12's report-card lines re-cut by exchange
  week for four weeks and for the calendar month to date. A day nobody packed is NAMED and
  its tally reads `no reads graded`, never a zero; a day whose report-card line the desk
  could NOT build is carried on the pooled line and in the strip cell as
  `unreadable_sessions` (present and empty when there are none) and adds nothing to any
  count - dropping that marker made an unreadable day indistinguishable from a quiet one.
  **The page computes nothing**: ONE payload
  (`ui/services/weekend_prep_service.read_week_review`, with `week_strip` and the
  review-learning callouts inside it) arrives on the page's own worker and `WeekReviewPage`
  renders it - single-flight, five sparklines built once and only repainted, and
  `day_review_pack.build_pack`, `day_report_card.build`, `day_report_card.week` and
  `run_week_review_narration` all monkeypatched to RAISE in its tests. `how_fresh` is asked
  per session BY THE STRIP, always WITH a session (it is deliberately absent from the day
  pack, whose hashed body it would move every night).
  **Pooling lives in ONE place**: `day_report_card.week_from_cards(cards)` (additive to
  TJ-12's module, the same arithmetic `week(sessions)` now delegates to) pools the stored
  lines' INTEGERS once per session with the ONE Wilson on the pooled pair, prints no rate
  under `MIN_REPORTABLE_N`, and names NO best family - a best of five day-winners is a
  ranking of days, not of families (`NO_FAMILY_FROM_CARDS`).
  New overnight slot **`week_review_narration`** (`scripts/ai_jobs/week_review_narration.py`):
  decision 0018 stage 2, directly after `observation_tags` and directly before
  `ticker_briefs`, on the SATURDAY slate only through the EXISTING
  `runner.WEEKEND_ONLY_SLOTS` (Sunday picks it up only when Saturday left it owed), reserve
  `TIMEOUT_SECONDS / 60 + RESERVE_MARGIN_MINUTES` (40 min) so a call that runs to its own
  timeout still leaves room for the model load and the write, a measured probe still winning.
  Its `EVIDENCE_KEYS` is CLOSED - the five day packs, the five day stories, the weekly
  market-story rollup and TJ-15's/TJ-16's contrast packs, and a test proves no bar, lake or
  journal-stream section can join it. Every bound comes from the INPUT (`_schema_for`) and is
  RE-CHECKED after the reply (`check_week_narration`); every citation is SESSION-QUALIFIED
  (`week_source_id`, separator `/`), because each pack mints ids with its own minter. A
  fourth tendency, a tendency quoting an `n` its cell does not carry, a `were_you_right`
  triple that is not the measured one, a citation the week does not hold or an empty headline
  rejects the WHOLE answer and leaves last Saturday's file byte-identical; an unchanged hash
  publishes nothing and costs no call. Fewer than `MIN_NARRATED_DAYS` (3) narrated days
  writes a deterministic scaffold saying `narrated K of 5`, status `skipped`, and loads NO
  model; a run that does narrate makes exactly ONE model call, on the MEDIUM local model
  until TJ-13B's probe row exists, with the reason in the row under `provider.LEDGER_FIELD`
  (`model_attribution.fallback_reason`). `openai` stays a setting that is off: the provider
  seam refuses it before anything is sent and the slot records a FAILED row with that
  sentence. Output `DAY_REVIEW_DIR/week/<YYYY-Www>.json`. `ai_jobs.provider._local_setting`
  reads a setting through BOTH import identities of `project_paths` (production has one
  file; under pytest `scripts/` is both a path entry and a package).
  `selftest.LAZY_ENGINE_MODULES` gains `ai_jobs.week_review_narration`: source selftest
  **96/96 -> 97/97**. Tests: `tests/test_tj5_week_narration.py`,
  `tests/test_tj5_week_slot_and_slate.py`, `tests/test_tj5_week_strip.py`,
  `tests/test_tj5_week_review_page.py`, `tests/test_tj5_review1_followups.py`,
  `tests/tj5_support.py`. Rule: DESK_INTERNALS "TJ-5"; slot position: decision 0018 addendum
  2026-09-20. Gates #149, #170 and #171 owed.
- **The desk's AI has a voice, and an idea is a SUGGESTION (TJ-6, 2026-09-20).** Stage-3
  slot `improvement_ideas` (`scripts/ai_jobs/improvement_ideas.py`), appended LAST behind
  `setup_research`, `uses_model=True`, no `model_free_kwargs`, `RESERVE_MINUTES = 10.0`,
  `max_attempts=2`, night-only (a forced daytime run is `skipped`) and on every night's
  slate - it is NOT in `WEEKEND_ONLY_SLOTS`, and Sunday picks it up only when the weekend
  left it owed. It reads the last five day packs and their stories, the window's walk-away
  totals, TJ-15's contrast groups, the two measurables the desk really computes and the
  checked-in 30-line `IDEAS_PROGRAM_CARD`; **`EVIDENCE_KEYS` is CLOSED** and holds no bar,
  tape, tick, lake or warehouse section. It asks the MEDIUM local model ONCE and re-checks
  every bound from the INPUT after the answer. **A fabricated citation or a fourth USABLE
  idea rejects the answer WHOLE with the ideas store AND the trader's state file
  byte-identical**; an idea with no evidence, no measurable it can be checked by, an unknown
  kind, or text over `MAX_IDEA_CHARS` (280) is DROPPED on its own and counted by reason into
  the ledger row (`extra.dropped`, `extra.drop_reasons`). The per-item verdict is the one
  predicate `drop_reason`, which both the verifier and the writer use, and **the slot
  enforces it itself**: the item schema is validated SHAPE-ONLY (`_shape_only` removes the
  `enum` and `maxLength` that `ai_summary.validate_structured_output` would turn into whole
  rejections), while required keys and `additionalProperties: False` still reject whole. The
  cap of three counts **USABLE ideas, before the dismissed filter and before in-answer
  de-duplication** - conservative by design. The model never names an idea (`idea_id` is
  minted by the desk from the session and the normalised text) and never grades one.
  **NOTHING TO CITE -> NO model call**: a window whose sessions carry no citable id answers
  `skipped` BEFORE any model load ("tonight carries nothing to cite: K of N session(s) have
  facts"), which is the desk's state today. **ASKED ONCE IS DONE**: a run that asked and
  stored nothing ends **`ok`** ("asked once: 0 of N ideas kept") with its drop counts, so the
  runner's own already-done check stops every later pass of the 30-minute task, and it leaves
  an ASKED marker (`ai_ideas_asked.json`, temp-and-rename, beside the store, never in the
  trader's state file) so the slot's unchanged-hash skip arms too; a marker write that fails
  never fails the night and leaves no `.tmp` behind. A whole rejection is `failed`, leaves no
  marker, carries its counts and is capped at 2.
  `AI_IDEAS_FILE` (`ai_ideas.jsonl`) and `AI_IDEAS_STATE_FILE` (`ai_ideas_state.json`) are
  new `project_paths` constants under `PERSISTENT_DATA_DIR`, resolved at CALL time. The ideas
  store is APPEND-ONLY and folded by `idea_id` on read, last row winning: a repeat inside
  `DEDUPE_SESSIONS` (60 EXCHANGE sessions, walked on the calendar) appends a row with the same
  id and a higher `seen_count`, keeps `first_seen`, and every earlier sighting stays on disk.
  A dismissed idea never returns, checked on the normalised TEXT - the id carries the session
  an idea was first seen in, so a dismissal checked by id alone would expire after sixty
  sessions. **KEEP and DISMISS are the TRADER's clicks and the card is the state file's only
  writer**: `keep_idea` / `dismiss_idea` validate their argument and fail CLOSED, write
  temp-and-rename with every other entry byte-identical, and a second Keep is idempotent so a
  frozen baseline is never re-frozen; no nightly job may call either, proved structurally over
  every `ai_jobs` module. `MEASURABLES` is a CLOSED registry of the TWO readers that exist -
  `report_card_did_well_rate` (`day_report_card.week_from_cards` over the day packs' own
  stored lines) and `veto_reason_real_miss_rate` (`ai_jobs.miss_contrast.read_latest`, the
  LAST WRITTEN pack, never a contrast build behind a click) - an unknown name RAISES, and an
  unreadable reader is `measured: False`, `value: None`, `n: 0` with its reason, never a zero.
  Keeping a `process` idea FREEZES that measurable's reading - value, `n`, `measured`,
  `window_sessions` and WHICH pack or which sessions it was read from - and Week Review
  prints before and after deterministically, with no model: under
  `evidence_stats.MIN_REPORTABLE_N` a reading prints its COUNT and no rate at all, above it
  the rate with its `n` and the ONE Wilson interval (`evidence_contrast.rate`, IMPORTED and
  never re-derived), and **"higher"/"lower" is said ONLY when the two intervals do not
  overlap**, otherwise "no clear change". A kept `program` idea is listed under
  "For WISHLIST - copy" as selectable TEXT and is never graded; nothing writes `WISHLIST.md`.
  The card is `scripts/ui/widgets/ideas_card.py`, used by BOTH pages and fed from the ONE
  payload each already reads on its worker (`day_review_service.PAYLOAD_KEYS` and
  `weekend_prep_service.WEEK_PAYLOAD_KEYS` both carry `ideas`, each filled in its own guard,
  the week reading it EXACTLY once); it diffs its rows, states its floor as a SIZE HINT
  (`minimumSizeHint`, never a second minimum-height setter), and runs ONE write at a time off
  the Qt thread with every Keep and Dismiss grey until EVERY ending answers, including a
  raise and including a row that arrives mid-write. With nothing in it: "no ideas yet" and
  `Kept 0 of 0` - two integer counts and no percentage. **Nothing TJ-6 writes reaches a
  detector, score, alert, watchlist, Focus, the review queue or `review_policy.json`**: the
  only modules that can see an idea are the store, the runner slot, `project_paths`, the two
  page services, the two panels and the card. No new lazily-imported engine module, so source
  selftest stays **97/97**. Tests: `tests/test_tj6_an_idea_is_only_a_suggestion.py`,
  `tests/test_tj6_ideas_are_grounded.py`, `tests/test_tj6_dedupe_and_dismissed.py`,
  `tests/test_tj6_measurables_and_the_frozen_baseline.py`,
  `tests/test_tj6_keep_is_the_traders_act.py`, `tests/test_tj6_ideas_card.py`,
  `tests/test_tj6_ideas_reach_both_payloads.py`, `tests/test_tj6_ideas_slot_and_slate.py`,
  `tests/test_tj6_ideas_store_and_paths.py`, `tests/test_tj6_one_ask_a_night.py`,
  `tests/test_tj6_builder_followups.py`, `tests/tj6_support.py`. Rule: DESK_INTERNALS "TJ-6";
  slot position: decision 0018 addendum 2026-09-20. Gates #150, #172, #173 and #174 owed.
- Provider-neutral A.I. Summary workspace for OpenAI and Anthropic, explicit evidence
  selection, bounded preview, credential-manager storage, structured/source
  validation, immutable evidence packages, and export-only results.
- Config-gated local OpenAI-compatible provider through Ollama, default off; small and
  medium model tiers verified on the Ryzen main desk with no market-hours inference.
  The local large tier was declared `RETIRED` on 2026-08-10 (27B-class models were held
  not to load beside the running desk on the 780M) — **that was a judgement, not a
  measurement, and TJ-13B replaces it with one** (see the probe entry below); the tier
  has still never run. Local calls are capped to the tier's context window and fail
  loudly on server-side prompt truncation.
- Separate off-hours `ai_jobs` process and scheduled task, job-ledger integration,
  deterministic evidence coverage, daily advisory summary, per-ticker briefs, full
  artifacts in `ai_store`, and bounded atomic `ai_morning_brief.txt` publication.
- Local inference is night-only, SEVEN DAYS A WEEK, and the night picks the slate. The
  configured ET off-hours window applies every day (the weekend "open all day" exemption
  is removed; the market-session block keeps its own weekend short-circuit); `--force`
  re-spends the attempt caps and the already-completed check but not the clock for a slot
  that calls a local model, which each slot declares as `JobSlot.uses_model`.
  `runner.night_kind()` names a night on the exchange calendar from the evening it
  started, and `runner.slots_for()` builds its slate: a weeknight without `ai_summary`;
  Saturday night with `ai_summary` and `weekly_synthesis` and no typed command; Sunday
  night the deterministic stage plus a retry of any slot the weekend attempted, did not
  finish and is still inside its cap. `ai_summary` therefore runs for ONE session a week -
  Friday's, on Saturday night - and is capped at three attempts. The slate is chosen first
  and decision 0018's stages then order it - `EXPECTED_SLOT_ORDER` is unchanged.
  `run_ai_jobs.py --status` prints the night's own slate, and `--slot` resolves against
  every registered slot (`default_slots() + optional_slots()`), an unknown name being an
  error exit.
- **The large local model is MEASURED before anything uses it (TJ-13B, 2026-09-19/20;
  merged into `lead/p033-integration2` `e54c8203`, not on `main`).**
  `scripts/ai_jobs/model_probe.py` adds `python scripts/run_ai_jobs.py --probe-model
  large` — a COMMAND, never a slot: it builds no slate, runs no other job, and exits 0
  when it measured, 1 when it refused with a printed reason (exactly three refusals: not
  night, a job is running, already measured this session). A probe is a model LOAD and is
  treated as one — night-only seven days a week with `--force` never buying the clock
  (`PROBE_RESERVE_MINUTES` 45, so it refuses from ~05:15 PDT), refused while the
  `ai_jobs_runner` machine lock is held and refused on a box with no exclusion primitive
  at all, and run against a COPY of one week's fact packs (`evidence_stats.WEEK_SESSIONS`
  = 5) that is deleted afterwards. **The lock is the DEFAULT guard**, resolved through
  `model_probe.runner_lock` at call time, and there is no value of `lock` that means
  "unguarded", so a bare call cannot load a 27B beside a running 12B; it is HELD across
  the measurement, so the next 30-minute firing stands down. `--force` re-spends ONLY the
  already-measured check. It writes ONE `manual_test` ledger row under `job="model_probe"`
  carrying `model_probe = {model, tier, packs, load_seconds, tokens_per_second,
  peak_memory_mb, baseline_memory_mb, context_tokens_accepted, basis, …}`; `manual_test`
  never counts as session coverage (a Saturday probe never tells the next firing that
  Friday was served) and the row's `session_date` is the LAST session.
  `context_tokens_accepted` is the SERVER's reported `prompt_tokens`, never the configured
  `ai_local_context_tokens`; `peak_memory_mb` is peak memory in USE on the machine beside
  the running desk, not a delta. One call cannot split a weight load from prompt
  evaluation and generation, so every row names its BASIS: `single_call_end_to_end` (load
  an upper bound, throughput a lower bound — a reserve derived from them is conservative
  by construction) or `server_reported_timings`. An unreachable endpoint costs ONE call,
  is recognised by `ai_summary.is_endpoint_unreachable`, and leaves every prior artifact
  and ledger row intact. `latest_measurement` reads the NEWEST row back and averages
  nothing; `reserve_minutes_from_probe` derives `(load + 3,500 tokens ÷ measured rate) ×
  1.25` (600 s at 10 tok/s → 19.8 min), floored by the load itself, and returns `None` —
  never a default — when the tier has never been measured; a MEDIUM probe never answers
  for LARGE. **The 27B has still never run**: the live 476-row ledger names only 12B tags,
  and the measurement is the trader's first Saturday night (gate #158's TJ-13B clause).
- **`local_large`, and a fallback that says so (TJ-13B, 2026-09-19/20).**
  `scripts/ai_jobs/provider.py` owns the week story's provider vocabulary:
  `ai_week_review_provider` defaults to `local_large`, `openai` stays a setting and is
  refused by `request_with_fallback` without sending anything, and an unknown value falls
  back to `local_large` with a warning. `week_review_plan()` is a REPORT that writes no
  row: with a measurement it names the large tag and its derived `reserve_minutes`; with
  none it answers `may_run_large: False`, names the MEDIUM model and says to run the probe
  — the trader gets a week story every Saturday either way (lead decision, 2026-09-19).
  `request_with_fallback` asks the large model, falls back to the medium one on the
  IDENTICAL closed schema (`AI_SUMMARY_JSON_SCHEMA`, citations enforced by the shared
  validator, an invented `source_id` rejected whole) and returns `attribution = {provider,
  model_asked, model_answered, fallback_reason}` for the SLOT to write on its ledger row
  under `model_attribution`; both models failing publishes nothing (`result is None`), and
  it writes no ledger row itself (`ledger_path` accepted and unused by design). It RAISES
  `ValueError` for `openai` and for any name that is not a local tier, before anything is
  sent, so the calling slot must catch it and record a FAILED row. `local_large` lives in
  this seam and deliberately NOT in `ai_summary.normalize_provider`, whose vocabulary
  `ai_credentials.PROVIDER_ENV_KEYS` shares and which raises outside openai/anthropic.
  Underneath, every call is an ordinary `provider="local"` request: one provider path, two
  tiers. No slate changed. Tests: `tests/test_tj13b_model_probe.py`,
  `tests/test_tj13b_probe_guards.py`, `tests/test_tj13b_week_reserve.py`,
  `tests/test_tj13b_local_large_provider.py`. Rule: DESK_INTERNALS "TJ-13B". Gate #158's
  TJ-13B clause owed.
- The nightly summary fails fast or finishes: an unreachable local endpoint ends the map
  pass on its first call rather than after every slice's own read timeout (a REFUSED
  endpoint degrades in about 2 s; a HUNG one still costs one 900 s read timeout on the
  first slice; an ordinary slice failure still costs one slice), the synthesis package is
  bounded by the existing local evidence budget, and what did not fit is counted in
  `findings_dropped_to_fit` and stated in the coverage line. Every rejected model reply is
  persisted locally under `project_paths.AI_REJECTED_REPLIES_DIR`, bounded at 20,000
  characters and 200 files, one file per rejection, without ever raising into the slot. A
  reply wrapped in the response schema's own name is unwrapped and validated; a reply that
  echoes the schema stays rejected. The morning brief counts a membership-only name in its
  header and prints no section for it, and the measured report's example lists name three
  different symbols (or occurrence ids). A per-ticker projection's `truncated` flag is
  re-derived from the projection itself rather than inherited from the session package, so
  the model is no longer told that a source it received whole was cut. The A.I. Summary
  page's Generate button asks the same off-hours window before a local run and refuses
  with its reason; a cloud provider is unaffected. A forced daytime run of a slot that
  declares model-free keywords (`daily_digest`'s `narrate=False`) writes its deterministic
  facts pack - superseding, never overwriting last night's narrated digest - and records
  what it left out.
- Per-ticker briefs project each symbol out of a full-size base package and then
  ration the projection to the local context window; ticker-roster and bare-name
  lines are discarded as non-evidence, each symbol resolves independently, a symbol
  with no evidence beyond watchlist membership is answered without a model call,
  completions resume on a read-stamp-independent evidence key, the morning file is
  republished after every resolved symbol, and the slot spends at most three attempts
  a session.
- **The local AI cannot change a fact's meaning** (Q3, 2026-09-04,
  `docs/LOCAL_AI_AUTOMATION_PLAN.md` §9). Every source in an evidence package carries
  a KIND (`ai_summary.SOURCE_KINDS_BY_FAMILY`: `journal` / `watchlist` / `scanner` /
  `market` / `narrative` / `feedback` / `walkaway` / `ops`); an unknown family
  **raises** rather than defaulting, and `coverage.source_kinds` publishes the map.
  `market.auto_state` and `watchlists.membership` are `watchlist` despite their
  family, because their content is a list of names. **A statement that asserts a HELD
  position is DROPPED unless one of its surviving refs is in `POSITION_SOURCE_IDS`** -
  the 2026-09-03 morning file called BULL a held long while citing watchlist
  membership. That set is `{"journal.trades_and_reviews"}` and is an EXACT LIST OF IDS
  rather than the `journal` KIND: the other four `journal.*` ids are the **Market
  Journal**, which is what the trader THOUGHT, and the two stores are deliberately
  never merged. **The `executive_summary` cites nothing, so it may not assert a
  position at all** and is replaced wholesale by `WITHHELD_EXECUTIVE_SUMMARY` when it
  does - 480 of 1,478 published summaries did.
  **A statement stating a percentage, an `N of M`, an `n=N` or a decimal R must carry
  a resolvable `metric_ref` `{source_id, key, horizon, denominator}`** whose source is
  one of its own refs and whose key really exists in that source
  (`metric_key_exists`: a mapping key at any depth, a first-column row key, or a
  literal occurrence in text content). Both rules DROP THE ROW and never rewrite it;
  the 2026-08-28 rule stands, so the document still publishes and one supported by
  nothing still raises. `metric_ref` is the one optional key in the row shape and in
  `AI_SUMMARY_JSON_SCHEMA`; `GROUNDING_PROMPT_LINES` tells the model both rules.
  `validate_ai_summary` is the ONLY validator - the ticker briefs reach it through
  `request_ai_summary`, so there is no second implementation to drift.
- **The morning file publishes three counts, never one total** (Q3, 2026-09-04).
  `render_morning_file` prints `Analyzed A of N. Membership-only B. Failed C.`;
  `Briefed 152 of 152` on 2026-09-03 counted 40 symbols that never reached a model at
  all. Each membership-only block leads with `membership only - <reason>`.
- **The local provider has TWO output caps, and a length stop is read before the text
  is parsed** (N2, 2026-09-05). `LOCAL_MAP_GENERATION_TOKENS` (3,500) is what a
  single-shot summary or one map slice sends and is what the evidence budget subtracts;
  `LOCAL_SYNTHESIS_GENERATION_TOKENS` (8,000) is what the map-reduce REDUCE call sends,
  chosen by `local_generation_tokens(evidence)` from the `map_reduce_synthesis` scope
  `findings_package` already stamps. `LOCAL_GENERATION_TOKENS` survives as an alias of
  the map cap. **Widening what the synthesis may WRITE never narrows what a map slice
  may READ** - the budget keeps subtracting the map cap, because the reduce prompt is
  findings rather than evidence. The cloud payloads stay at 3,500. A **length stop**
  (`choices[0].finish_reason`, or Ollama's top-level `done_reason`) is detected BEFORE
  parsing and earns ONE retry asking for at most 8 findings per section - never the
  identical request with the validator's rejection appended, which is more prompt
  against the same ceiling. A second cut raises `LocalOutputLengthError`, a
  `RuntimeError` subclass carrying `stop_reason`. The `map_reduce` block gains
  `synthesis_stop_reason` ("length" or "", always present), `synthesis_retry`
  ("shorter" or "") and `slices_retried`; older manifests without them still load.
- Local-AI Phase 0 is complete. Phase 1 implementation is complete; its five-session
  unattended live gate remains in `plan.md`.

### Durability and catch-up

- Repeating 06:00 Pacific weekday launch task through the session, protected by the
  existing single-instance guard.
- Master AVWAP tracker staleness catch-up from completed prior-session D1 data with
  explicit `data_session` vintage and no automatic scoring-tuner/prior-refit side
  effects.
- Setup-tracker purity gate that honours the trader's `daily_bars_source` pin: the
  pinned source is the declared source of record (PURE), a cache read of the store the
  pin governs is accepted as the absence of contrary evidence, per-row provenance is
  read where it exists, a third source still vetoes, and with no pin the gate is the
  July 2026 one unchanged. The mix and the decision log once per run
  (`n_ib` / `n_pinned` / `n_other` / `refused`).
- Tracker save provenance: `saved_at` (market-local) + `saved_by`
  (`close_slot` / `catch_up_backfill` / `manual`) on every payload and mirrored header,
  `last_replayed_session` per record, `tracker_saved_at` / `tracker_saved_by` on the
  three stats CSVs, and the Setup Tracker's two-clock status line.
- `EXPIRED_UNMEASURED` terminal setup status with `expiry_reason`
  (`no_replay_stale_sessions` over `evidence_stats.LATELY_SESSIONS` exchange sessions,
  with the threshold recorded on the row, or `no_baseline_scenarios`),
  applied after the closure rule. The exclusion is DISPLAY-ONLY and opt-in
  (`exclude_expired_unmeasured`, default False): the stats exports and the Setup Tracker
  tabs drop the expired from numerator and denominator, the champion's scoring
  population keeps them, and `n_expired_unmeasured` is carried either way.
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
- **A daily snapshot says whether its anchor was KNOWN then, and a swing outcome row
  names its path** (packet Q2, 2026-09-04, BD-99/BD-100; branch
  `claude/q2-warehouse-eligibility`, gate #61 owed). `anchor_dates_by_symbol` returns
  `{symbol: features.AnchorChoice}` - bar date plus `observed`/`reconstructed`, decided
  from the row's own `system_from` read market-local; `feature_snapshot_daily`'s additive
  `anchor_knowledge` carries it, NULL reads as **`legacy`** and never as observed, and a
  **reconstructed anchor is research evidence, never promotion evidence**.
  `outcome_path`'s additive `path_kind` (`managed`/`plain_target`/`plain_no_target`) is
  written by the one decision `simulate_swing` already makes and is EXCLUDED from BD-98's
  unchanged-comparison, so no stored row is rewritten merely to gain a label.
  `cli band-coverage --month YYYY-MM [--recipe] [--json]` is the read-only report - per
  recipe, per knowledge bucket, against the recipe's OWN required bands - and
  `cli rebuild-daily-features --from --to [--apply]` recomputes past sessions WITH their
  anchors (dry run by default; a year-partition RETIRE plus a verbatim carry of every row
  outside the range, because the partition is year-keyed). Runbook order:
  build -> rebuild-daily-features -> recompute-outcomes -> band-coverage.
- **Both AVWAP band families sit on the daily snapshot, and a twin recipe walks
  the challenger's** (packet M4, 2026-09-05, BD-101; branch
  `claude/m4-lake-band-variant`, gate #66 owed). `feature_snapshot_daily` carries
  `avwap_variant_value` / `_stdev` / `_upper_1..3` / `_lower_1..3` /
  `_formula_version` beside `avwape_*`, computed from the SAME bars and anchor
  index by `indicators.avwap_band_variants.oneoption_avwap_bands` and never
  merged into one column; `FEATURE_SET_VERSION` is `tier1_v2` and `tier1_v1` rows
  are never rewritten. `outcomes.SWING_HOUSE_VARIANT_V1` is a
  `dataclasses.replace` of `SWING_HOUSE_V1` differing ONLY in `band_family`
  (plus its id and `outcome_definition_id` = `band_variant_v1`, which fences it
  out of every `house_default_v1` reader); `build_outcomes` picks the band map
  from the RECIPE and a variant recipe with no challenger bands walks
  `plain_no_target` rather than borrowing the champion's levels.
  `cli band-coverage --compare A B` reads two recipes on the SAME occurrence ids
  with `swing_headline`'s Wilson lower bound and counts an unpaired occurrence on
  a `not_paired` line. Shadow only; T4's criteria decide and nothing here
  promotes.
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
- **That view is BOUNDED, and the selection is a size rule that may never be a ranking
  by result** (N3, 2026-09-05). Sending every eligible cell stopped working when gate
  #59's lake recompute took the grid to 141,299 recipe outcomes: 619 eligible cells,
  **658,292 chars against 78,119**, and `narration absent` on the ledger four nights
  running. No budget a 64k-context model can read fits 658k chars, so the view SELECTS.
  **Select, then fill**: the fixed head is encoded first (11,084 chars on the live
  2026-09-04 pack - gate, coverage, evidence shape, excluded families, hoisted
  conventions), then eligible policy cells are added in order until the next would
  cross the budget, then the after-like ELIGIBLE cells (P10 C3) under the same rule
  from their own top-level `n_episodes`. The order is **`stats.n` descending, then
  `recipe_id`, `family`, `side`** - `stats.n` is the outcome-row count the eligibility
  floor itself gates on, and the tie-breaks are identifiers, so it is total and
  deterministic. **No `mean_r`, `win_rate`, `profit_factor`, `expectancy`, bootstrap
  bound or trimmed mean is in that key** (gate #43, BD-101): a cell that looks good
  early is exactly what the frozen research window protects. Coverage is stated in
  THREE places - `narrated` {K, of, selected_by, after-like K, of} in the view and in
  the narration json, one line under a `## Narration` heading in the pack markdown, and
  `narrated K of N eligible cell(s)` on the ledger reason. The markdown's line is
  computed BEFORE the file is written (the cut needs no model), so one pack and one
  `.md` per date still holds; a pack that is not narrated prints NO line rather than
  "0 of 0". The refusal narrows to **head + FIRST cell does not fit** and names the
  head's size beside the size, budget and cell count. Live: 619 -> **64 narrated**.
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
  Read OFF the Qt thread on its OWN worker because it is 19.7 MB - and since G7.2 every
  other export on that page is off the Qt thread too, on a second worker.
- **A Research child reads on its FIRST SHOW, never at startup** (G7.1, 2026-09-07):
  the Market Journal's `_loaded_once` + `showEvent` idiom, generalised to the Day-trade
  Tracker (`reload_from_disk` + `start_decisions_refresh`), the Setup Tracker
  (`refresh`), Market Prep (`refresh`; its file WATCHER stays in the constructor), Price
  Alerts (the table load; the service's monitoring timer is untouched) and the Setup
  Playbook (its record worker, which now also carries the overview banner). A
  `QTabWidget` child gets its `showEvent` only when its tab is selected, so **Research's
  first paint costs ONE child's load rather than nine** and `research_panel.py` needed no
  change. The Playbook's two UNCACHED reads inside `render_best_now_html` moved onto that
  worker and go through `setup_tracker_panel._load_csv_rows_cached` - ONE reader for
  these exports - and `render_all_docs_html` takes the banner as an argument and is pure.
  Nothing is skipped: every deferred read still happens the first time the page is shown,
  so the cost appears against the tab that asks for it.
- **The Setup Tracker's refresh runs on a worker and re-fits only what changed** (G7.2,
  2026-09-07): `refresh()` starts ONE `ReadWorker` doing the twelve
  `_load_csv_rows_cached` calls, `load_human_focus_performance_rows`, the pure ranking
  and the scan-factor `stat`; `_on_exports_loaded` renders on the Qt thread and emits
  `refreshFinished`. It is single-flight and COALESCED - a refresh asked for while one is
  in flight is taken by the worker in flight as one more pass, and that pass runs INSIDE
  the worker, which is what makes `shutdown()`'s `join_worker` enough. `_table_render_plan`
  names, per table, the export it was built from; a table whose memo is unchanged is
  neither reset nor re-fitted (the mtime cache skipped only the PARSE, so thirteen model
  resets and thirteen column fits ran on every spinbox step). The human-focus table has
  no file, so its memo is a content digest; the two tables the spinbox re-ranks carry
  `min_closed`; `tracker_export_files()` resolves its paths at CALL time. Measured
  3456x2160 repeat 3: `setup_tracker.refresh` sync p95 576.9 -> 0.2 ms,
  `research.construct` 3,025.1 -> 533.4 ms, the layout-fit table byte-identical.
- **The Market Journal builds its four capture charts on demand** (G7.3, 2026-09-07):
  `_ensure_charts()` on the first capture render, not in `__init__`; `_clear_charts`
  walks an empty dict when nothing is built. `market_journal.construct` p50 57.9 -> 9.8 ms.
- **G7 fix round** (same day, same branch): a `refresh()` asked for inside the
  `ReadWorker`'s teardown window - after its reader loop released `_refresh_lock` with
  `_refresh_pending` False but before the QThread's own `finished` fired - used to be
  ORPHANED, leaving the Setup Tracker showing rows ranked at the previous `min_closed`
  with nothing left to re-check the flag; `_on_worker_finished`, wired to `finished`
  (fired only once `isRunning()` is reliably False), now restarts it. The same slot
  drops the panel's `_read_worker` reference and calls `deleteLater()`, so five
  refreshes leave at most one `QThread` child instead of five. `PriceAlertsPanel
  ._save_table` now refuses (and logs) a save before the panel's first `showEvent`
  load, when the table is still empty and would otherwise overwrite the real store.
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

- **A repeatable desk workload bench, and a layout-fit check** (G0, 2026-09-06):
  `scripts/ui/desk_bench.py` builds the pages ONE AT A TIME - never `MainWindow`,
  so no IB, no autopilot, no timers - drives a fixed workload over a STAGED copy
  of the home folder, and reports p50/p95/max per op across `--repeat`. Three
  numbers per op: the synchronous Qt-thread time of the call, the time to settle,
  and the LONGEST single `processEvents()` during the settle wait (the stall
  proxy). A settle deadline is a recorded RESULT, never an error; the 2 ms yield
  that lets a worker have the GIL is inside `settle_ms` and outside
  `longest_iteration_ms`, and every settle carries the 120 ms `QUIET_MS` floor,
  so a settle near 120-135 ms measured nothing and that op is read on `sync_ms`.
  **The settle's worker probe is a `_WorkerProbe`, not a tree walk per poll**
  (G7.0, 2026-09-07): the candidate set is walked once per op, re-walked at most
  every `WORKER_REWALK_MS` (250 ms) while the page is busy, and re-walked ONCE
  MORE before a settle is declared - the only moment a read that started after
  the last walk could be missed. `settle` returns a fourth value, `poll_cost_ms`,
  and every op row carries its `n/p50/p95/max`, so the bench's own share of a
  wait is a number in the artifact rather than a claim in a docstring.
  The fit half records `minimumSizeHint` /
  `minimumSize` / `sizeHint` for every page, Weekend step and Research child
  against the available height - the window height minus chrome MEASURED FROM
  WIDGETS, not a constant - plus the sum of every table's floor including the
  ones behind a tab. `stage --from --to` copies an allowlist of the read inputs
  the pages open (not the 1.2 GB tracker JSON, the 622 MB attributes CSV or the
  142 MB scenarios CSV), source opened read-only. **`--data-dir` is required and
  is set into `TRADINGBOTV3_DATA_DIR` before the first import of anything under
  `scripts/`**, `LOCALAPPDATA` moves into the scratch too, and the resolved
  `project_paths.DATA_DIR` is printed and the process exits 2 if it lands under
  the live home folder or the DAS - twice over, by two guards comparing their
  own literals, because proving one guard bites means breaking it. **Both guards
  run on the ARGUMENT at the top of `main()`, for `--data-dir` and for `--out`,
  BEFORE the environment is prepared**: the first round refused a live
  `--data-dir` only after `_prepare_environment` had already created it and
  written a settings file inside it, which is the incident the refusal exists to
  prevent. The bench's own `local_settings.json` is seeded key by key from the
  real machine-local file through `machine_settings_seed`, an allowlist that
  carries the display keys that change what is measured (`qt_ui_scale` scales
  every `theme.px`) plus the trader's `daily_bars_source` pin, and never a
  credential or a path key. It measures
  and changes nothing: no panel imports it and no timer starts it. Runbook:
  `docs/GUI_FLUIDITY_MEASUREMENT_RUNBOOK.md` section 7.
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

**Two contexts per row, three verdicts per thesis (WISHLIST 10I, packet WS-10I, 2026-09-13).** `scripts/context_join.py` owns the ONE identity/time contract: `ContextRef(context_id, rule_version, benchmark, label, observed_at, available_at, certainty)` and `attach_context(rows, when="observation" | "entry", ...)` - a D1 environment label is published at its session's CLOSE, so a same-day M5 decision gets the PREVIOUS session's label (`certainty=prior_session`, walked on `market_calendar.previous_session`, never a calendar day), a swing scan row its scan date's (`session`), a journal fill the label available at its timestamp, a date-only fill the previous completed session's flagged `date_only`; a missing label is `unknown`, never pooled, and a backfilled ref is `reconstructed` and excluded from every forward claim. Both refs ride side by side on a row and the tier-outcome row has no benchmark column, so `attach_context` runs BEFORE `link_theses`, which links a thesis by the ref's benchmark scope and the thesis's validity window (created_at .. horizon end or invalidation), all matches linked newest first, none chosen, opposite scope never linked. `scripts/setup_environment_evidence.py` cuts three populations that are never pooled: `opportunity_cells` (swing through `swing_evidence.read_eligible_rows` inside `lately_window()`, day trades from the M5 outcome rows cut by the D1 label of `trade_date` - never by the alert's own `context_json.market_environment`, a different vocabulary) with win rate / held-run first, `n`, distinct sessions and symbols, the ONE Wilson bound, floors, `unknown` its own cell and the family's overall baseline beside each cell; `personal_cells` from the journal's confirmed-tag trades with money once per `trade_id`, ST5's status partitions and `MIN_REPORTABLE_N` before any "best" word; `thesis_review` with three SEPARATE verdicts per thesis (`market call` right/wrong/open by the benchmark's later path, `setup held` from the opportunity cells, `trade profitable` from personal money) never merged into one grade; `HELD_RUN_STATISTIC_NAME` now lives in `held_run_score` (the module that computes it) with `working_lately` importing it. Research > Results gained the page-level **By environment** CONTROL over the section WS-ENV already built (`environment_filter` / `environment_basis` / `environment_line` / `environment_choices`; Bot x Day under its own key `by_environment_day` with its own statistic; My trades cut at the ENTRY; computed on the Results worker; the setting `research_results_environment`; the existing goldens unchanged when no environment is chosen). The Daily Recap row keeps WS-DR's `d1_environment` (the session's own label) AND carries 10I's observation context and, for a matched fill, the entry context beside it - on an intraday row the two differ by one session on purpose and the tooltip prints both (lead decision 2026-09-13: two readable facts, not reconciled). `python -m context_join backfill --since` is dry-run by default, names the data dir and `--apply`, and labels only what contemporaneous stores establish. Zero detector, score, alert or promotion influence. Contract: `docs/LOCAL_AI_AUTOMATION_PLAN.md`. Tests: `tests/test_ws_10i_context_join.py` (24) and `tests/test_ws_10i_context_join_build.py` (11).
- **D1 market environments, labelled per session and cut in the readouts (WS-ENV, WISHLIST item
  7, 2026-09-12, sweep branch).** One label per session per benchmark, decided ONCE by the pure,
  versioned rule `d1_environment_v1` in `scripts/indicators/d1_environment.py`
  (`classify_environment`): fewer than 34 completed daily bars is `unknown/warmup`; an
  unmeasurable ATR14 is `unknown/unmeasurable`; `range_atr <= 3.0` is `compressed` - tested
  FIRST, so a quiet uptrend inside a three-ATR box is compressed, not trending; then `slope_atr`
  beyond +/-0.5 ATR with the close on the same side of SMA20 is `trending_up` / `trending_down`;
  else `mixed`. ATR14 is Wilder at the last bar over the WHOLE supplied series
  (`indicators.atr.wilder_atr`), pinned by a hand-computed golden over 184 recorded SPY
  sessions. `scripts/d1_environment_store.py` appends one JSONL row per `(session, benchmark,
  rule_version)` to `project_paths.D1_ENVIRONMENT_FILE`: never rewritten, a new rule version
  written BESIDE the old one, each benchmark its own row, `written_at` aware and market-local,
  `labels_by_session` one mtime-cached read, every failure swallowed.
  `runner.record_d1_environment` runs as a sibling of `bridge_earnings_anchor_caches_to_csv`,
  fetches SPY/QQQ/IWM through the SAME pinned `fetch_daily_bars`, drops the forming bar through
  `completed_bars`, and logs `D1 environment: SPY=.. QQQ=.. IWM=.. (d1_environment_v1, bars
  through <session>)`; a failure never fails the scan. `scripts/d1_environment_join.py`
  `attach_environment` joins on `scan_date` - the tape the decision was made in, never the exit
  - in place, on one store read. Research > Results gains "By environment (SPY,
  d1_environment_v1)" under Bot x Swing only: win rate first with `n` and the ONE Wilson bound,
  sorted by the bound, the `MIN_REPORTABLE_N` floor labelling a row and never hiding it,
  `unknown` its own row pooled into nothing, no verdict line because it names no leader, and the
  champion sections byte-identical. The window is `ENVIRONMENT_WINDOW_SESSIONS` (6 x
  `LATELY_SESSIONS` = 120), declared because 20 sessions of SPY is usually one environment.
  `python -m d1_environment_store backfill` is dry by default and prints `DATA_DIR` first.
  Shadow only - nothing reaches a detector, score, alert, watchlist, Focus, the review queue or
  `review_policy.json`, and no `legacy.py` line changed. Tests:
  `tests/test_ws_env_d1_environment.py` (35) + `tests/test_ws_env_d1_environment_builder.py`
  (5); rule in DESK_INTERNALS "ENV - one D1 environment label per session, joined by scan date".
- **Exit frameworks, split by setup family (WS-EF1, WISHLIST item 3, 2026-09-12, sweep
  branch).** `master_avwap_exit_framework_by_family.csv` is written beside
  `master_avwap_exit_framework_stats.csv` in the same guarded tracker save pass, by the SAME
  builder: `legacy.build_exit_framework_stats_rows(setups, by_family=False)` takes the grouping
  key as a parameter, so the two files can never disagree on a rate, and there is still one
  scenario walker (the band-variant fence holds). The by-family key adds `setup_family` (blank
  or missing is `unlabelled`, counted never dropped) and `population` (`champion` / `study` /
  `control`, read from the record's own `is_study` / `is_control` joined on `setup_id`, never
  the family name; part of the key, so a study and a champion sharing a name stay apart), and
  those two columns lead `EXIT_FRAMEWORK_BY_FAMILY_STATS_COLUMNS`. The by-family export reads all
  three namespaces; the pooled export still reads `setups` alone and stays byte-identical
  (golden), and a raising by-family export costs neither the save nor the pooled file. The Exit
  frameworks tab gains ONE control, `exit_framework_family_combo`: `All setups (pooled)` first,
  rendering today's table unchanged, then families sorted by name; a family view filters before
  the 300-row cap, ranks by the same Wilson lower bound, names the family with the LARGEST
  `n_closed` among its rows (never the sum across templates) and says BELOW FLOOR under
  `evidence_stats.MIN_REPORTABLE_N` while still showing the rows; the table renders through
  `_apply_exit_framework_view` with its own memo so a refresh does not re-fit it. On a copy of
  the 2026-09-11 tracker mirror: 24 pooled cells reconcile to 672 by-family rows over 35
  families with zero mismatches; the `1stdev_breakout` study's eight cells close 8-26 (under the
  floor of 30) - band 3 leads on mean R, band 2 on the bound, and nothing here promotes a
  template. Shadow only. Tests: `tests/test_ws_ef1_exit_by_family.py` (19).
- Side-symmetric SPY market-state/pullback engine runs beside the legacy pause
  detector, emits replayable evidence, and cannot affect candidates, alerts, or rank.
- Greatness Monitor persists ordered touch/wick/close/acceptance/retest/failure/re-arm
  transitions beside legacy D1 alerts and cannot alter the champion path.
- Champion-invariance tests prove enabled, failing, or poisoned shadow engines leave
  production SPY/D1 results unchanged.
- **AVWAP band challenger** (Phase 0.10, `scripts/indicators/avwap_band_variants.py`):
  an anchored HLC/3 centre with a 20-close Bollinger sigma, replicated from OneOption,
  carried on every setup record beside the champion's frozen
  `calc_anchored_vwap_bands` and graded on the SAME exit template.
  `build_anchor_band_variant_meta` lives in `legacy.py` and **serves both call paths** -
  the live scan in `runner.py` (which re-exports it) and the tracker staleness catch-up
  (`_evaluate_priority_snapshot_for_date`), which is what writes the persisted tracker
  on a normal day. **It measured nothing from 2026-08-26 to 2026-09-05** because only
  the scan path set the block; packet M1 fixed the hand-off, and the Setup Tracker's
  Band variant view now states `Measured N of M setups (K unmeasured: <top reason>)`
  above the table so an empty comparison can never again look like a comparison.
  `_is_band_variant_scenario` fences the shadow out of every champion aggregate and
  the B-2 parity fixture pins the champion's records byte-identical.

- **The control holdout, the study namespace and the April exit framework are SHOWN**
  (packet M5, 2026-09-05). Three exports written in the tracker's own save pass and read
  by three Setup Tracker tabs: `master_avwap_control_discovery.csv` (401 setups the scan
  REJECTED), `master_avwap_study_discovery.csv` (3,992 unpromoted ideas) and
  `master_avwap_exit_framework_stats.csv` (one row per `(framework_family,
  exit_template_id, side, priority_bucket)`, including `comparison_apr2026` - the two
  experimental templates that wrote 91,674 of 275,022 scenario rows since April and had
  **no reader anywhere under `scripts/`**). `build_control_discovery_rows` /
  `build_study_discovery_rows` are unchanged and are CALLED by the new builders.
  Win rate leads with `n` and the ONE Wilson bound (`swing_headline`), mean R beside it,
  the window in SESSIONS, an all-history block and a `lately` block side by side, and
  **each tab carries a population sentence** so a control is never read as a pick - naming
  the graded EPISODES and the RECORD count separately, because they differ by about a
  third. The framework rows carry `n_filtered_by_experiment`, so a template that skips
  scenarios by its own `blocked_stop_rules` has its smaller `n` explained rather than
  read as a worse result.
  `experimental` is a COLUMN, so a what-if can never read as the champion's record. The
  framework export is built from `_flatten_tracker_scenarios`, inside the band-variant
  fence. Shadow only: no detector, score, tier, alert, watchlist, Focus, review queue or
  `review_policy.json` is reachable, each export is guarded separately so it can never
  cost the tracker save, and the champion aggregates are pinned byte-identical by a
  two-directory reproduction test. Live gate #67.

- **The tracker replay has a VERSIONED execution convention and level knowledge, and
  since 2026-09-06 the defaults are the repaired ones** (packet ST3, then ST7 and
  decision 0019). `scripts/master_avwap_lib/execution_convention.py` names four
  policies on two independent axes: `literal_level_v1` (what shipped until 2026-09-06 -
  a touched level fills AT the level) / `gap_aware_v2` (DEFAULT), and `same_session_v1`
  (bar D's high/low tested against day D's own bands) / `prior_session_v2` (DEFAULT).
  Under
  `gap_aware_v2` a bar that opened through the level fills at the OPEN (`gap_open`,
  both directions - a stop gap and a target gap are the same mechanic), a bar with no
  usable open fills at the level CLAMPED into `[low, high]` (`clamped_no_open`), and an
  INVALID candle (`low <= open, close <= high` broken, or a NaN among high/low/close)
  books NOTHING (`invalid_bar`) while the hold clock keeps running - an invalid bar on
  the maximum-hold index DEFERS the `TIME_STOP` to the next valid bar
  (`deferred_invalid_bar`), never cancels it. The fill price is always inside the bar
  and `resolve_fill` raises rather than trusting a comment. Under `prior_session_v2` the
  INTRABAR target tests read the LAST COMPLETED session's levels (a daily anchored-VWAP
  band for day D is computed with day D's own bar folded in, so it is not knowable
  intrabar); the hard stop is level-free and untouched, and the two-closes protective
  stop and the maximum-hold force close stay CLOSE-based on day D. A prior-session level
  that does not exist counts `intrabar_skip_reasons["no_prior_session_level"]` on the
  scenario rather than reading as "not hit". `_evaluate_tracker_scenario_bar` and
  `recompute_tracker_setup_record` take the two policies as keyword arguments; since
  ST7 the record carries `execution_convention` / `level_knowledge` on EVERY run,
  including the v1 one, because after a default flip an absent stamp is ambiguous.
  `_apply_scenario_exit_event`'s `fill_basis` / `execution_convention` are keyword-only
  and add a key only under v2, so a v1 event dict is still the shipped seven keys.
  **Both whole records are pinned**: `tests/fixtures/st3_replay_golden.json` (taken from
  `main` before the repair existed, now read with the v1 policies NAMED) and
  `tests/fixtures/st7_v2_default_golden.json` (frozen on `main` at `68762909` through
  the explicit v2 keywords, before the flip, so it is not a self-portrait) - the default
  reproduces the second. `scripts/tracker_execution_compare.py` is the
  evidence CLI: it replays COPIES under both policy pairs and writes a stamped
  `comparison_<stamp>.json` + `.csv` with per-setup R (clipped AND raw, never blended -
  `TRACKER_SCORING_R_CLIP` is 4.0 and the tail hides behind it), changed flag, fill
  bases, and per `(side, setup_family, priority_bucket)` n / changed n / expectancy /
  win rate with the ONE Wilson bound / tail / rank impact; `--new-execution-convention`
  and `--new-level-knowledge` isolate one axis. It prints `project_paths.DATA_DIR` and
  REFUSES if it, `--out`, `--tracker` or `--bars` is under `C:\TradingBotData`; a SQLite
  mirror is opened `mode=ro&immutable=1`. **The lead took the convention decision on
  2026-09-06** under the discretion the trader granted (*"3. This one is up to your
  discretion."*), recorded as decision 0019: each repair is a pure correctness fix, and
  keeping a known look-ahead because the honest number is worse is the failure decision
  0016 goal 8 names. `calc_anchored_vwap_bands` and `calc_anchored_vwap_band_history`
  are still untouched (decision 0008), the cost model is one, stop-first ordering is
  kept, and nothing here promotes a setup or reaches a detector, alert, watchlist,
  Focus, the review queue or `review_policy.json`. Live gates #77 and #84.

The band challenger is not promoted and its ≥ 20 sessions of forward accrual start at
its first measured row; its remaining evidence gates are in `plan.md`. The selection and
execution policies above are no longer challengers - decision 0019 made the repaired
ones the DEFAULT on 2026-09-06 and left the v1 names selectable as the comparison CLIs'
"old" arm.

## Recent changes (the last two build days)

### 2026-09-21 - TJ-9E: the Mentor tells an exit from an entry, and a reading is the trader's only by their click (branch `claude/tj9e-exit-notes`, tip `1876bb08`, merged into `lead/p033-integration2` `7230b30d`)

The trader, the same morning: *"trade mentor should be able to differentiate between trade entrys and exits. trade exits should ask for 'why did you exit, what emotions did you have, what technicals were you observing' ideally we can just write it out and the AI fills this stuff in overnight. trade entrys are good the way they are"*. Everything follows from the last sentence: entries are untouched, and an exit is ONE box the trader writes in while the STRUCTURE is the night's job. **Four review rounds by reproduction - NO-GO, NO-GO, NO-GO, GO at `1876bb08`** - closed nine blockers. Round 1: the builder's "these 33 errors are a pre-existing artefact" was REFUTED by running the base (`journal_feed._store()` caches a module-global store for the life of the process, so the new exit-note read cached a FAKE store from three Day-Review test files and killed every later `JournalPanel`; base ran the same selection 368 passed, 0 errors); an outcome-fence guard that searched for "73" inside a body carrying a sha256 was a coin flip; an unwired report-card clause; an exit that never made the check RIDE; and an exit the trader had already explained asked again with a blank box. Round 2: a reading keyed by the TRADE lost the second reading of a trade that exited in two sessions - identity is `(trade_id, exit session)` through `check.exit_key` - and a reading drawn in both the questions area and the trade section gave the trader two copies whose first Confirm stopped responding. Round 3: the trade section kept its OWN list of readings and bypassed the registry's budget and ordering - **THE REGISTRY alone decides which readings are offered and how many**, whichever seam noticed them. Between rounds 1 and 2 the builder fixed the thing that made the feature reachable at all: the timeline never lines up - a trade exits MONDAY, the note is typed on TUESDAY's 09:00 card, TUESDAY NIGHT drafts it, and WEDNESDAY's card reviews TUESDAY, where Monday's trade is not a row - so `waiting_exit_drafts` walks `EXIT_DRAFT_OFFER_SESSIONS` (5) exchange sessions ending at the CARD's own session, independently of the session the card reviews. What shipped is the inventory entry above. Measured by the reviewer on a 201-trade scratch journal carrying 20 drafts: the lane read 12.2 ms on the first call then ~4 ms warm, the registry decision ~0.45 ms warm, and on the ordinary morning - questions delivered first - the seam costs 7 microseconds. Reviewer gates at `1876bb08`: `tests/test_tj9e_*.py` 105 passed; the round-1 Mentor selection (`tj9`, `tj14b`, `tj12`, `tj7`) 368 passed, 0 errors; all of that plus the nine order files and `test_tj13a_night_slates.py` in ONE process 697 passed; ruff clean; source selftest 97/97; `scripts/trade_mentor_ai.py` diff EMPTY over the whole branch (the entry path is untouched, as the trader asked); no ask-first file, no new `.md`. Recorded, NOT touched: `setup_environment_evidence.py:480` and three CLOSED-only filters still spell the partly-closed status wrongly, and `ai_summary._journal_source` lists `opportunity_events` with no type filter so an exit note's words can ride into a `journal_review` narration beside `net_pnl` - the exit TAGGER is blind by construction, a NARRATION is not (plan.md follow-ups). Gates #176-#180 owed. Long form: DESK_INTERNALS "TJ-9E".

### 2026-09-21 - the Mentor stores an answered trade and stops asking (branch `claude/mentor-stores-answers-2026-09-21` `0caf2bba`, merged into `lead/p033-integration2` `182f3e08` - NO reviewer round of its own, a review is OWED)

From the trader's report on TJ-9's first live morning; the live journal held ONE `RECALLED` row. The 09:00 trade check's Save gate is now PER TRADE (`TradeMentorCard._open_fields`, a `Save this trade` button per block, the bottom Save files every answered trade, `_drop_trade_block` takes a filed trade off the card alone); `_field_answer` reads words typed beside a blank dropdown as `not_supplied` and the trade's one raw note as the answer to its still-open fields (note saved verbatim as `RECALLED_RAW` first); `MainWindow._mentor_answered` also reads `NOTE` answers by `trade_id`, so a `once` question about a trade does not return when its `trade_date` moves. Tests: `tests/test_mentor_stores_an_answered_trade.py`. **Merged onto TJ-9E at `182f3e08` on the trader's instruction to combine the day's work**: both branches rewrote the same gate in `trade_mentor_card.py` (nine conflict hunks, three files), the lead kept this branch's per-field / per-trade shape as THE gate and folded TJ-9E's one forced exit box into it as one more field of its trade (filing a trade writes its exit note FIRST), and added `tests/test_tj9e_per_trade_save.py` for the seam neither branch could test - an open exit greys its own trade and no other, filing one trade leaves the other's half-typed words alone, and the words are on disk before the entry answers even when the entry half fails. It had NO reviewer round of its own; a review is OWED (plan.md). Gate #181 owed. Long form: DESK_INTERNALS, TJ-9's 2026-09-21 paragraph.

## Revision history

Entries from **2026-09-19 and 2026-09-20** moved to
[`docs/archive/CHANGELOG_ARCHIVE_2026-09-19_2026-09-20.md`](docs/archive/CHANGELOG_ARCHIVE_2026-09-19_2026-09-20.md)
on 2026-09-22 (18 entries); entries from **2026-09-18 back to 2026-09-06** moved to

[`docs/archive/CHANGELOG_ARCHIVE_2026-09-06_2026-09-18.md`](docs/archive/CHANGELOG_ARCHIVE_2026-09-06_2026-09-18.md)
on 2026-09-20 (35 entries); the ST6 entry of **2026-09-06** moved to
[`docs/archive/CHANGELOG_ARCHIVE_2026-09-06.md`](docs/archive/CHANGELOG_ARCHIVE_2026-09-06.md)
on 2026-09-08 (1 entry); entries from **2026-09-05 back to 2026-09-03** moved to
[`docs/archive/CHANGELOG_ARCHIVE_2026-09-03_2026-09-05.md`](docs/archive/CHANGELOG_ARCHIVE_2026-09-03_2026-09-05.md)
on 2026-09-07 (25 entries); entries from **2026-09-02 back to 2026-08-26** moved to
[`docs/archive/CHANGELOG_ARCHIVE_2026-08-26_2026-09-02.md`](docs/archive/CHANGELOG_ARCHIVE_2026-08-26_2026-09-02.md)
on 2026-09-03 and 2026-09-05 (68 entries); entries from **2026-08-19 back to the initial system in 2025-11** moved to
[`docs/archive/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md`](docs/archive/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md)
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
