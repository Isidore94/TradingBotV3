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
`Recent changes` below cover the last two build days; everything older is in the three
archives named under `Revision history`, the newest being
[`docs/archive/CHANGELOG_ARCHIVE_2026-09-03_2026-09-05.md`](docs/archive/CHANGELOG_ARCHIVE_2026-09-03_2026-09-05.md).
They are evidence and must not be loaded as context.

### Application, runtime, and data ownership

- **Workspace memory (2026-09-12, WISHLIST 11):** root `MEMORY.md` is a routing index
  only; `memory/` holds provenance-tagged detail (`people/`, `projects/`, `decisions/`,
  dated notes, prunable `context/`); rules in `CLAUDE.md` "Workspace memory", role
  paragraphs in `.claude/agents/*.md` and `.codex/agents/*.toml`. Recall only, never
  authority; adapted from JumpStarter M1 (`664e083`). Verification gate #93 owed.
- **Codex agent operations (2026-09-09):** project defaults select Astra as lead and
  Luna for unspecified helpers; recon uses Luna, builder/tester/reviewer use Terra.
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
  the model-gated slots. The narration pair held up to 2½ h of reserve ahead of
  every deterministic slot, a slot that cannot fit its reserve records SKIPPED,
  the 2026-09-01 run took six hours - and no deterministic slot reads either
  narration slot's OUTPUT. Relative order inside each stage, every reserve and
  every retry budget unchanged. The rule is now "a later phase appends inside its
  stage and never reorders across stages", and the order is pinned once as
  `EXPECTED_SLOT_ORDER` in `tests/test_ai_jobs_runner.py`.
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

### 2026-09-12 - WISHLIST sweep: one feature dump on `claude/wishlist-sweep-2026-09-12` (trader-directed)

Every WISHLIST item built as one feature dump for a week of trader testing; Astra reviews
after code completion; nothing merges to `main` before that. One bullet per packet as it lands
(the resume table is `CURRENT_CHECKPOINT.md` "2026-09-12 - WISHLIST SWEEP").
- **WS-5D (WISHLIST 5D) - watchlist intent events**, branch `claude/ws-5d-watchlist-intent`
  `e8770233`: `scripts/watchlist_intent_events.py` + the panel's write-first-then-append seam,
  machine labels on the Focus store's injection, `observed_external` at load time, one
  baseline row per list, a tail CLI. Suite 7327 green, ruff clean, smoke 7/7, selftest 75/75
  (+1 lazy module). Gate #94.
- **WS-PT4 (WISHLIST item 4, block 4) - the AWAY digest ranks by points when the switch is on**,
  branch `claude/ws-pt4-digest-points` `c2215b19`: one projection (`swing_pick_projection`),
  `order_swing_picks` reading `setup_points.rank_enabled()` at sort time, `RANKED_BUCKETS`
  matched on the key, the `order:` clause on the `Ranked on:` line. Suite 7323 green, ruff
  clean, smoke 7/7, selftest 74/74. Gate #95.
- **WS-5A (WISHLIST 5A) - the weekend verdict card reads numbers, not strings**, branch
  `claude/ws-5a-weekend-verdict` `df475e45`: typed cohort rows (`horizon_sessions`, `n`,
  `avg_side_return_pct`), formatting at the display edge, the pooled `ALL` side excluded, both
  lines ranked by the HIGHEST side-adjusted return, three explicit absence sentences, the card's
  floor kept at 5. Six existing test files re-fixtured to the typed shape. Suite 7324 green, ruff
  clean, smoke 7/7, selftest 74/74. Gate #96.
- **WS-EF1 (WISHLIST item 3) - exit frameworks split by setup family**, branch `claude/ws-ef1-exit-by-family` `a55cf320`: the by-family export from the SAME builder (`by_family=True`), `population` from the record, the family picker on the Exit frameworks tab, 24 pooled cells reconciled to 672 family rows with zero mismatches. Suite 7331 green, ruff clean, smoke 7/7, selftest 74/74. Gate #97.
- **WS-FC1 (WISHLIST item 2) - a forming bar never reaches the daily-bar cache**, branch
  `claude/ws-fc1-forming-candles` `0134ff12`: `master_avwap_lib/daily_bar_cache.py` (the rule,
  the counters, the repair CLI), a 19-line `legacy.py` writer-seam change, manifest counters
  always present, one log line per scan; dry run on a copy: 66 of 1,988 files; 443 tracker
  records touched, zero fills. `--apply` and two ask-first seams are the trader's. Suite 7332
  green, ruff clean, smoke 7/7, selftest 74/74. Gate #98.

### 2026-09-12 - Workspace memory adopted from JumpStarter (trader-directed, docs and agent config only)

Added the root `MEMORY.md` routing index and `memory/` (a trader file, a project file, a
decision record, today's daily note, an empty `context/`), every detail line tagged
`[stated]`/`[observed]`/`[inferred]`/`[suggested]` with a date and a source, seeded from an
inventory of the 28 Claude auto-memory files (dated audit notes and broker facts stayed
machine-local). `CLAUDE.md` gained a "Workspace memory" section (recall before answering
about prior work, a five-source cap on the recall answer, memory never authority over the
control set or the code, supersede in place, 15,000-character files) and `AGENTS.md` was
re-copied byte-identical; the eight agent role files and `docs/AGENT_TEAM.md` carry the
role split (recon, reviewer and tester propose, a builder records in scope, the lead
integrates); `docs/DESK_INTERNALS.md` holds the long form. Source: `Isidore94/JumpStarter`
`664e083`; its operator facts, approval record and `jumpstart check` were not imported.
No application code, test, detector, store or gate changed; gate #93 (the fresh-session
recall check) is owed.

### 2026-09-09 - Codex delegation defaults (trader-directed)

Added shareable `.codex/config.toml`; pinned the four Codex roles to Luna/Terra instead
of inheriting the lead model. Root instructions remain identical for Claude and
Codex, with each tool's model policy scoped separately in `docs/AGENT_TEAM.md`.
Astra plans, routes, judges and integrates; helpers get bounded work and narrow
context. Existing testing, independent review and live-data safeguards still apply.
Configuration/documentation only: no application baseline or promotion gate changed.

### 2026-09-08 - SN5 / SN6: the M5 scanner breathes, and scans the trader's picks first (lead, on `main`)

Trader, 2026-09-08 evening, on 9% of the week's usage: *"Anything we can get done from
wishlist.md that's quick and cheap?"* then *"Go"* on the two smallest packets of the
"keep the desk snappy" prompt. Measured that day: `Thread-4 (run_strategy)` held 0.62 of
a core in hour 13 and 71-88% per minute at the close while the GUI thread got 0.10-0.15;
13,031 GUI stalls over 50 ms. The fast lane scanned 258 Focus names alphabetically with
the trader's own names mixed among 107 auto-adopted ones.

- **SN5 - breathe.** `BounceBot._breathe` waits `SYMBOL_BREATH_SECONDS` (0.02) on the
  STOP EVENT after each symbol's compute in the fast lane and in both main-sweep loops
  (`scripts/bounce_bot_lib/legacy.py`). Never `time.sleep`, so a set stop event returns at
  once and shutdown latency is unchanged. Pacing only: nothing scanned, detected, stored or
  alerted changes; `ScanCycleClock` still never sleeps.
- **SN6 - order the fast lane.** `BounceBot._fast_lane_order` puts the trader's own Focus
  names first (alphabetical), then the auto-adopted ones (alphabetical), then the sweep
  follows as before. The SET is unchanged; every name still scans every cycle. The engine
  learns which names are auto-adopted through the new read-only
  `focus_picks.load_auto_pick_symbols()` (today's `focus_auto_picks.json` markers, the
  same per-entry `session_date` rule as the store, now ONE reader
  `_read_todays_auto_pick_markers`); a failed read is an empty set, so every name then
  scans as the trader's - the safe order. The fast-lane log line now prints both counts.
- **Ask-first:** `bounce_bot_lib/legacy.py` is a detector file; the trader's "Go" of
  2026-09-08 is the yes for the two named seams (`run_strategy`'s loop pacing and the fast
  lane's ordering) only. `_breathe` is called in three places and nothing else moved.
- **Tests:** `tests/test_sn5_sn6_scanner_breath_and_fast_lane_order.py` (13; 12 fail with
  the fix reverted, the clock pin passes either way). Suite, ruff and smoke in the checkpoint.
- **Remaining from the same prompt (WISHLIST, not authorized):** SN4 (diff the feed), SN3
  (one RRS pass), SN2 (new bars only), SN1 (the scanner in its own process) - each measures
  itself against the 2026-09-08 numbers in `thread_cpu.jsonl` / `ui_stalls.jsonl`.

### 2026-09-08 - The setups table ranked by a point system (lead, on `main`)

Trader: *"how hard would it be to get master avwap setups output to be ranked
instead by a point system? ... 1. its setup, higher WR/PF setups get ranked
higher. 2. nearby S/R. lots of nearby trendlines and SMAs knock it down. 3. RS/RW
to its industry/sector/SPY based on its direction. 4. the presence of a recent
bounce."* then *"Put it in wishlist.md then start working on it block by block."*

- **`scripts/setup_points.py` (new, pure)**: `score_row` -> `SetupPoints(total,
  setup, sr, rs, bounce, notes)`, `rank_order` (ranked buckets by total, ties in
  arrival order, the rest after them unchanged), `rank_enabled` (the
  `rank_setups_by_points` switch, default OFF, read at sort time). Every input
  is a field already on the focus row (`hv_level_*`, `cloud_level_nearby_count`,
  `trendline_note`, `ema21`, `sma_breakout_sma_level`, `previous_close`, `atr20`,
  `daily_relative_strength_score`, `rs_vs_industry`, `has_bounce_event_today`,
  `favorite_signals`, `expected_r`) or on the `SetupRow` (`d1_vs_sector`,
  `d1_vs_industry`) or the injected family record (`win_rate_lb`). The scan and
  `legacy.py` are untouched; no PF exists on any surface, so the setup part is
  the Wilson bound and expected R.
- **`SetupTableModel`**: a `points` column appended last (display = the total,
  sort = the total, tooltip = the parts and why); `points_for(row)` and
  `family_record_for(row)`. **`MasterAvwapPanel`**: the `Points` checkbox on the
  control strip, `_by_points` after `_prioritised` in `set_rows`, a re-sort when
  the family record lands and the switch is on; compact width 58 pinned in the
  G2b golden (`points` is now the stretch section, `family_win_rate` an exact
  132). `tests/test_setup_group_context.py`'s column tail widens by one.
- **Tests**: `tests/test_setup_points.py` - eleven, including the
  identical-visible-rows test both ways and the lift of a higher-point row over
  a higher-score row that the toggle undoes. Baseline after: 7291 passed, 3
  skipped, 72 subtests, exit 0; ruff clean; smoke 7/7.
- **Not done, by choice**: the AWAY digest keeps its Wilson-bound order (ask
  first); industry is the curated index the row already carries, not a new
  membership file; the weights are a first cut for the trader to feel on the
  desk and every one is a named constant at the top of the module.
- **Later the same evening - the evidence loop** (trader: *"ensure that we
  track how this system performs ... a way for the system to correct itself"*).
  `scripts/setup_points_evidence.py` (new): `append_log` / `read_log`,
  `grade` -> `PointsGrade` (terciles, lift, per-part lift, `sentence()`),
  `propose_weights` / `write_proposal` / `read_proposal` /
  `proposal_multipliers`, `log_and_grade` (the worker's one pass).
  `setup_points.score_row(weights=...)` keeps `raw_parts` and `weights` on the
  result and `log_row` builds the evidence row; `active_weights` /
  `learned_weights_enabled` gate the proposal on the trader's switch.
  `SetupTableModel.set_points_weights`; `MasterAvwapPanel`: `_PointsEvidenceWorker`
  started from `refresh_from_reports` (default-store panels only),
  `points_evidence_payload`, the grade label on the status row, the
  `Points: learned weights` overflow action. `project_paths`:
  `SETUP_POINTS_LOG_FILE`, `SETUP_POINTS_WEIGHTS_FILE` (both under the shared
  home, beside `pick_feedback.jsonl`). Eight tests in
  `tests/test_setup_points_evidence.py`. Baseline after: 7299 passed, 3 skipped, 72 subtests passed, exit 0; ruff
  clean; smoke 7/7.

### 2026-09-07 - The Strength window is one flat page (branch `claude/strength-page`)

Trader: *"For the main trading desk, the strength tab is unusable there's like 2
tabs and they get no space each. Create a solution that removes the tabs and just
collates all the data to be more easily readable."*

- **`ui/widgets/strength_page.py` (new): `StrengthPage`**, one `QScrollArea`
  hosting the four strength reads one under another - `FocusStrengthBoard`,
  `EntryAssistBoard`, `RrsSnapshotWidget`, then `StrengthBoardPanel` under a
  page-owned "M5 Strength Board (TC2000)" heading. The two `CollapsibleSection`s
  ("RS/RW Board", "M5 Strength Board (TC2000)") and the stretch-factor column are
  gone from `AlertCenterPanel`; `strength_column`, `rrs_board_section`,
  `strength_board_section` and `rrs_board_tab` no longer exist (`strength_page`
  does; `attach_strength_board` hands the panel to the page). Vertical scrollbar
  ALWAYS ON (an as-needed bar reflows the boards and can loop); floor 170 px, so
  the alert column's 360 px budget is unchanged.
- **`fit_height_to_document`**: every text board is exactly as tall as its
  document (vertical bar off, `Fixed` policy, re-fitted on `textChanged` and
  `documentSizeChanged`); the first fit lays out at the viewport width itself,
  because a never-shown `QTextEdit`'s document reads `(0, 0)` (measured).
- **`strength_board_panel.py`**: `FIT_ROWS_CAP` (30),
  `_SideTable.set_fit_rows` / `fit_rows` / `_fit_height` (header + rows + frame,
  capped; off by default, the page turns it on), `StrengthBoardPanel.set_fit_rows`.
  The cap bounds the height, never the rows.
- **`rrs_snapshot.py`**: `set_stacked_scopes` / `stacked_scopes`,
  `_board_html(..., stacked=)` - the three scope tables one under another on the
  page, side by side elsewhere; rows identical.
- Tests: `tests/test_qt_strength_board_in_the_desk.py` section 5 rewritten
  (seven tests: reading order with no sections or tabs, nothing scrolls on its
  own, stacked scopes, the desk widget is stacked, tables fit their rows and the
  cap, the 360 px floor, the arm bar never on the page);
  `test_focus_strength_board.py` and `test_qt_strength_board_sort_and_chart.py`
  follow the widget to its new address. RED first
  (`ModuleNotFoundError: ui.widgets.strength_page`).
- Docs: `CLAUDE.md`/`AGENTS.md` "Charts and boards", `docs/DESK_INTERNALS.md`
  "The Strength window is one flat page, long form", this inventory (two
  sentences superseded), `CURRENT_CHECKPOINT.md`.
- Not changed: any detector, score, alert, queue, fold or evidence writer, the
  adoption gate, the `StrengthBoardService` (one owner, one timer), any click
  route. `alert_center_panel.py` is an ask-first file; the trader's message is
  the instruction for this window and the edit is hosting only.

### 2026-09-07 - Packet G7: the speed pass - first loads on first show, the tracker's refresh off the Qt thread (branch `claude/g7-speed-pass`)

The last item of the Phase 0.22 G lane, tester-first (`c13d46c8`: fifteen red tests,
three green-by-design, and explicit load TRIGGERS on 22 existing construction sites).
**Nothing here changes a number, a sort, a read's RESULT or a write** - only WHEN and
on WHICH THREAD a read happens, and how the bench measures itself. The G2b render
golden is re-rendered THROUGH the new asynchronous seam and is unchanged.

- **G7.0** `desk_bench.settle` holds a `_WorkerProbe` instead of walking the whole
  widget tree twice per poll: the candidate set is walked once per op, re-walked at
  most every `WORKER_REWALK_MS` (250 ms) while the page is busy, and re-walked ONCE
  MORE before a settle is declared - the only moment a read started after the last walk
  could be missed. `settle` returns a fourth value `poll_cost_ms`, `OpReading` carries
  it and `_aggregate` summarizes it per op. `QUIET_MS` and the deadline semantics are
  unchanged, and the three settle-RESULT tests (a plain `threading.Thread`, a `QThread`,
  and a worker started 300 ms into the wait) stayed green throughout.
- **G7.1** five Research children read on their first `showEvent` (`_loaded_once`), so
  Research's first paint loads ONE child rather than nine; `research_panel.py` needed no
  change. The Setup Playbook's two uncached CSV reads moved onto its existing record
  worker, through `_load_csv_rows_cached` - ONE reader - and `render_all_docs_html` is
  now pure.
- **G7.2** `SetupTrackerPanel.refresh()` starts ONE `ReadWorker` for the twelve cached
  reads, the human-focus read and the ranking; the Qt-thread slot resets and re-fits only
  the tables whose export signature changed, then emits the new `refreshFinished`.
  Single-flight and coalesced, with the extra pass INSIDE the worker so `shutdown()`'s
  join is enough.
- **G7.3** the Market Journal's four `CandleChart`s are built on the first capture render.
- **G7.4** re-measured offscreen at 3456x2160, `--repeat 3`, over the same staged copy,
  against the same base with only the seven files under test reverted: `research.construct`
  sync p95 **3,025.1 -> 533.4 ms**, `setup_tracker.refresh` **576.9 -> 0.2 ms**,
  `market_journal.construct` p50 **57.9 -> 9.8 ms**. The deferred first loads appear
  against the tab that asks for them (`research.tab.Day Trade Tracker` 2.3 -> 448.8 ms
  sync, still synchronous by design). **The layout-fit table is byte-identical across the
  two runs, all 19 rows.** Both tables are in `docs/GUI_FLUIDITY_MEASUREMENT_RUNBOOK.md`
  section 7.

Live gate **#88**. No packaging trigger. In the same pass the story sentences of the
sixteen longest `CLAUDE.md` bullets moved VERBATIM into `docs/DESK_INTERNALS.md` (each
rule kept, shortened to name its seam and point at its entry): **51.7 KB -> 48.5 KB**,
still ~3 KB over its ~45 KB rule, so the trim is started and not finished. `AGENTS.md`
re-copied byte-identical.

**Fix round, same branch:** a `refresh()` asked for in the `ReadWorker`'s teardown
window was orphaned (fixed on the worker's own `finished` signal, which also clears
the stale `_read_worker` reference and `deleteLater()`s it - five refreshes now leave
at most one `QThread` child, not five); `PriceAlertsPanel._save_table` now refuses a
save before its first `showEvent` load. Four new tests, all proven RED first.

### 2026-09-07 - Packet G5: Research > Results, the landing page the trader may read (branch `claude/g5-research-results`)

Trader, 2026-09-06 (decision 0016 answer 7, AMENDED that day, and decision 3 of
that day): Research gains a Results page the trader may read, four populations
never pooled, opening on Bot setups x Swing x Recent 20 sessions with the last
choice remembered. Tester-first: nine RED tests at `a79c7f20` off `main`
`7e018c99`, re-proven failing on the un-fixed tree (6 failed, 3 errors) before
the first edit; **thirteen** more added by the builder, and eighteen more in
the fix round below - thirty-one added in all, forty in the packet.

- **`scripts/research_results.py`** - pure, frozen dataclasses (`ResultsRow`,
  `ResultsBands`, `ResultsSection`, `ResultsView`), `band_cells` and
  `build_results_view`. No new statistic, threshold or eligibility rule: the
  floors, the concentration refusal, the ranking basis and the window are all
  borrowed from `working_lately` / `evidence_stats`, and a bot row's numbers are
  the cell's OWN OBJECTS. `band_cells` RAISES on a mixed-kind cell list the way
  `working_lately.pool_cells` refuses across its axes.
- **`scripts/ui/panels/research_results_panel.py`** - three control groups, three
  band cards, one shortlist and G4's identity-aware detail pane, every read on a
  `ReadWorker`, the selection persisted.
- **`research_panel.py`** - Results first and current, the eight existing tabs in
  their order, a `set_working_lately_snapshot` seam, `shutdown()` joins the new
  reader, and the pointer label reworded to the amendment. **`app.py`** gains ONE
  line so the strip and the page render the same reading.
- **Deviation from the packet, measured rather than argued.** The packet asked
  for `EvidenceCell.line()` on each band card. At 1920x1080 three of those
  twelve-clause lines in a third-width card left the shortlist under them **26
  pixels tall** - G0/G1's `overflow` defect. The card now prints a ONE-LINE head
  spelled the way the Desk's own Working-lately line spells it
  (`name - statistic (>= bound, n=..., N sessions)`), carries the full
  `EvidenceCell.line()` as its tooltip, and every field of that line is also a
  column of the shortlist and a paragraph of the detail pane. Re-measured: the
  table gets 295 px at 1920x1080, 799 at 2560x1440, 1599 at 3456x2160, and
  `minimumSizeHint` is 427 at every size.
- **Two more deviations, both in the shortlist.** Its third column was to be
  "wins/n or held/measured"; an `EvidenceCell` carries `n_graded` /
  `n_eligible` and neither of those, so the column states what the cell states
  and invents nothing. And because the one table holds BOTH swing sections it
  grew two columns the packet did not name - **Measure** (the cell's kind) and
  **Population** (live / study) - without which a 0.72 closed-R rate and a 58.0
  favorable-direction percent shared one `Statistic` column with nothing
  telling them apart, and a study row was indistinguishable from a live one.
- Nothing here scores, ranks a queue, alerts, writes evidence or reaches
  `review_policy.json`. Live gate **#87**.

**Review fix round (same day, same branch).** Three blockers and eight
advisories; **eighteen tests proven RED** against the reviewed tip `85d11227`
(18 failed, 13 passed) before a line of the fix existed.

- **The window control APPLIES.** `_window_of` labelled a window and nothing
  filtered by it, so a trade closed in 2019 was counted under "2026-09-01 to
  2026-09-04". `research_results.in_window` filters MY TRADES' closed trades by
  `closed_at`, inclusive at both ends; the count it turns away is REPORTED
  (`n_outside_window`), and a trade whose `closed_at` the journal never carried
  is counted out of a bounded window rather than folded into it. For BOT the
  SNAPSHOT owns the window: the three buttons are disabled with a tooltip, and
  `window_applies` / `window_sentence` state the cells' own `window_sessions`
  through `as_of` instead of a date range the numbers never saw. The button
  groups moved to `idToggled` filtered on `checked` so a disabled button's
  state is still drivable - one reaction per change, as before.
- **The freshness line says what the snapshot has.** `working_lately` writes
  `path: ""` and `mtime: null` for every source it computes, so the line read
  `an unnamed file @ None` three times over on every real snapshot. It prints
  each source's `rows`, adds a path or an mtime only when present, and says
  `NO_SOURCES` when nothing was recorded. The literal `None` cannot reach the
  screen.
- **A band card counts the lines it rendered** - `N of M shown`, not `M shown`
  over three printed rows - and an empty section leaves no dangling heading.
- Money names its currency and a refused total prints `resolve_pnl_key`'s own
  reason and the currencies; `untagged` left the Confirmed-tag table (coverage
  is the sentence's business); a study row is muted like an ineligible one; the
  section text is ONE short verdict line per kind with the rest in a tooltip;
  every running-text label is capped at G3's reading measure
  (`READER_MEASURE_CHARS` 100, left-aligned) - it was one 1,922-character line.
  The window button label reads `evidence_stats.LATELY_SESSIONS`.
- The suite-wide selection fixture left `tests/conftest.py` for the two files
  that write the key.

### 2026-09-07 - Packet G2b: the Setup Tracker tabs, the Desk Setups table and the AWAY tables name their column (branch `claude/g2b-tracker-desk-away-columns`)

The other half of G2, the half that waited for ST6 to stop rewriting these three
panels. Tester first: nine tests red on `7e018c99` (`tests/test_g2b_named_columns.py`
six, `tests/test_table_width_rule.py` three) plus three green-by-design guards - a
render golden over every tab's cells and row order, a golden of every compact
column width, and the AWAY Focus table's already-correct measured answer, kept as a
regression guard with the reason in its docstring. Each populated fixture carries a
DECOY text column longer than the one the packet names, so the measured path picks
the decoy and the assertion can only pass by naming.

- **`apply_width_rule(..., stretch_last=False)`** (`scripts/ui/widgets/data_table.py`).
  §12's two answers are both wrong for one shape - a table of ONE identifier and a
  row of measurements - and Setup Tracker ▸ Human Picks is that shape. It suppresses
  the measured auto-pick as well as the last section; suppressing only the last
  section would still have handed `cohort` the slack through the classify path,
  which is the whole complaint. A NAMED text column still stretches.
  `DataTable.set_width_rule` carries it; the default leaves every existing caller
  byte-identical.
- **Fourteen tracker tables declare their roles at construction**
  (`_make_table(columns, *, text_key=None, elide_keys=(), stretch_last=True)`), every
  index resolved by KEY through the new `_column_index`, which raises rather than
  guessing. Playbooks stretches `Exit Plan` and elides `sample_setups`; Catch Rate
  stretches `Missed Samples` and elides the caught ones; Controls and Studies
  stretch `Family` and elide `cohort`, so `Win % (low)` never takes the slack on an
  EMPTY tab; Human Picks stretches nothing. The two `fit_columns()` call sites are
  unchanged, so the attribute leaderboard's worker slot is covered too.
- **The Desk Setups FULL profile names `setup_tags`** and installs
  `_KeyLevelElideDelegate` on `key_level`. **DEVIATION from the packet's literal
  `elide_columns=(key_level,)`, and the reason**: `apply_width_rule` gives an elide
  column a per-column `MiddleElideDelegate`, and a per-column delegate REPLACES the
  view's delegate - so the packet's call would have left `key_level` painted by Qt's
  default while its own row kept `SetupTableDelegate`'s alternating background,
  favorite tint, selection fill and separator. The new delegate inherits both
  classes (the setups delegate wins `paint` and `sizeHint`; `MiddleElideDelegate`
  supplies the full-value tooltip) and overrides `_text` alone, because that helper
  hard-codes `ElideRight` - the end elision §12 forbids for an identifier whose tail
  is its anchor and retest date. The compact profile removes the column delegate on
  the way in and is untouched otherwise.
- **AWAY Recap** middle-elides `Line`, `Trigger` and `Cell / held x ran`, and names
  `Symbol` on the Focus table.
- Four builder-added tests (`tests/test_g2b_key_level_delegate.py`) pin what the
  tester's isinstance check cannot: the column's delegate is a `SetupTableDelegate`
  too, its `sizeHint` is still the setups row height, painting a long key level
  really does reach `elide_middle` with the WHOLE value and keeps a real tail, and
  compact gets the column back. All four proven red with the four source files
  reverted.

Layout lane: no number, sort key, read or write moved. Live gate **#85**.

### 2026-09-07 - Packet G4b: the Setup Tracker's detail pane clears when its context changes (branch `claude/g4b-setup-tracker-detail`)

The second half of G4, deferred while ST2/ST6 rewrote `setup_tracker_panel.py`. Tester-first
on `main` `7e018c99`: five tests committed RED at `9879aa4a` plus one golden green by design
(it pins all FOURTEEN tabs' rendered header and rows through the sort proxy). Builder made the
five pass without weakening any and added a sixth. `origin/main` `6d9f4b43` merged in after.

**The widget half had shipped and nothing called it.** `SetupDetailView.shown_identity` and
its overriding `clear()` landed with G4.1; on this panel the fourteen-tab strip had no
`currentChanged` handler and `refresh()` replaced every model's rows without touching the
pane. The pane here carries a stop price, a 1R and two targets, so what stayed on screen was
a price plan read against a symbol the table no longer held.

- **G4b.1 - a tab move is a context change.** `self.tabs.currentChanged` ->
  `_on_context_tab_changed` -> `clear()`: empty, hidden, identity forgotten. Same verb G4 gave
  the Day-trade Tracker.
- **G4b.2 - a refresh re-shows from the NEW row or clears.** `_reshow_or_clear_detail()` runs
  at the END of `refresh()`, after every `set_rows` and `fit_columns`. **It asks whether the
  pane is VISIBLE first**, not whether a match exists - the trader reaches the hidden state by
  moving tabs, and a scan that re-showed on a match alone would pop an explanation open under
  someone who had closed it. Then one linear scan of the CURRENT tab's model (`row_at`, no
  dict copied per row) for the matching identity, re-shown through that tab's own `show_*`
  call and from the NEW row dict, so the pane prints the revised number rather than the cached
  one; no match, `clear()`. The tab's model and show-kind come from `_detail_tables` by asking
  which tab widget owns the table (`isAncestorOf`), never by tab position.
- **The identity is widened where this page collides.** A Setup Types row has no `dimension`
  and no `symbol`, so two rows of one (side, family) in different zones share the whole
  identity. `DETAIL_WIDENING_KEYS` (`favorite_zone`, `priority_bucket`) separates them,
  normalised so a row carrying neither compares equal on both sides and widening can never
  turn a real match into a miss. A pair differing only in `retest_label` still collides and
  falls to the first such row in the model's own order - stated, not hidden.
- **`_on_family_row_clicked` removed**: defined since the panel was written, never connected.
  `SetupDetailView.show_family` is untouched (a G4 test drives it).
- **Layout lane.** No number, sort, column or read moved; the golden is green before and
  after, including after a full click / tab-switch / refresh cycle. The Attributes tab lands
  later on its own worker (`_on_attributes_loaded`) and is not one of the eight clickable
  tables, so it is out of this packet's seam.

Fail-before-fix: `scripts/ui/panels/setup_tracker_panel.py` restored from `9879aa4a` gives
5 failed / 1 passed; restored, 7 passed. The builder's widening test was proven the same way
with `DETAIL_WIDENING_KEYS = ()`. Live gate **#86**.

### 2026-09-07 - Packet G2a: the Trades and Tag Week tables name their text column (branch `claude/g2a-journal-and-tagweek-columns`)

The layout half of G2, split from the Setup-Tracker/Desk/AWAY half (G2b, queued after ST6
lands) so the G lane could touch neither file ST6 is rewriting. Tester-first: five tests
committed RED at `b2798ec7` (four proven failing on `main` `a1dab8fa`, one golden proven
green by design), builder made the four pass without weakening any.

**The rule already existed and was never called.** `ui/widgets/data_table.py`'s
`apply_width_rule_to_table_widget` (§12) stretches a named text column to the slack,
clamps every other column to `[MIN_COLUMN_WIDTH, MAX_COLUMN_WIDTH]`, and middle-elides a
named identifier column with the full value in its tooltip — but neither
`scripts/ui/panels/journal/trades_tab.py` nor `scripts/ui/panels/weekend_prep_panel.py`
called it, so a raw `QTableWidget` at Qt's default resize mode left option symbols and tag
lists clipped on a 3,456 px desk with most of the row blank.

Journal Trades' `_populate_table` (the tab's ONE render seam) now ends with the rule
naming `Tags` (`TRADES_COLUMNS.index("Tags")`, a new module constant — `("Date", "Symbol",
"Dir", "Status", "Qty", "P&L", "R", "Tags")` — replacing the header literal) as the
stretching text column and `Symbol` (`TRADES_COLUMNS.index("Symbol")`) as the middle-elide
column, so a 21-character OCC option symbol (`AAPL  260918C00230000`, whose identity is in
the TAIL) elides in the middle with the full value in its tooltip. Weekend Prep's
`TagWeekPage._render` does the same for `self.table` against `TAG_WEEK_COLUMNS` and
`_render_missing_risk` for `self.risk_table` against `MISSING_RISK_COLUMNS`, both indexed
by `.index("Tag")` / `.index("Symbol")`.

**Two premises the packet got wrong, corrected against the live code:** `TAG_WEEK_COLUMNS`
carries **six** entries, not five — `("Date", "Symbol", "Status", "Tag", "Net", "Week")`,
`Week` having been added after the packet was drafted, which is why every index in the
tests and the fix is looked up by name rather than a literal. And `MISSING_RISK_COLUMNS`
(`"Date", "Symbol", "Direction", "Net", "Tag"`) carries no description/reason column at
all — `Tag` is its only free-text column and the one named, not "whichever column carries
the description/reason" as the packet described.

The risk table's width-rule call runs BEFORE `_render_missing_risk`'s empty-rows early
return, so a zero-row render still names `Tag` as the text column by NAME rather than by
content — the same guarantee item 3.3 proves on the Trades table. `_populate_table`'s
existing tooltip on a NEEDS_REVIEW row is untouched: the width rule's own tooltip write
(`apply_width_rule_to_table_widget`'s elide-column loop) only fires where an item carries
no tooltip already.

**Layout lane only**: no number, sort key, read or write moved (a golden of every cell's
text and the row order over the fixture, captured from `main` at `a1dab8fa` before any
width rule existed, stays green throughout). The Trades tab's splitter declares
`setStretchFactor(0, 3)` / `(1, 2)` but opens `[1347, 2105]` at 3,456 px — the table gets
39% of the desk and the detail pane 61%, not the declared 3:2 — a real defect the tester
found and named; it is not this packet's (the packet said nothing else in the tab moves)
and is left for a later Journal packet. Fail-before-fix proven: `git stash` restored both
files to `a1dab8fa`, the four RED tests failed again with the tester's exact messages
(`Tags`/`Tag` 100 px on a 3,456 px desk; `Symbol` carrying no `MiddleElideDelegate`;
`Interactive` where `Stretch` was expected), the golden stayed green, and `git stash pop`
restored the fix with all five green again. No live gate beyond the trader seeing the wider
columns after the next restart.

### 2026-09-07 - Packet G3b: three reviewer advisories on the Market Journal reader (branch `claude/g3b-reader-followups`)

Three small follow-ups on G3, each tester-first (proven red on the pre-fix panel, then
green): (1) `_written_line` now converts the AWARE `created_at` with `astimezone` to
`pass_bars.desk_zone()` (the one desk-zone seam, N1) and prints `HH:MM UTC±HH:MM` in
the DESK's own zone — every live row stores `created_at` in UTC
(`market_journal.py`'s `moment.astimezone(timezone.utc)`), so the meta line was reading
`written 13:36 UTC` for a note typed at 06:36 Pacific on every live row; a NAIVE stamp is
still never guessed at and still says `(no zone recorded)`. (2) `_on_entry_selected` now
keys the reader lookup by `entry_id` (`_selected_entry_id` → the new `_entry_for_id`,
replacing `_entry_for_row`) instead of the QListWidget's row index, so a future
header/grouping row cannot desync `self.entries` from `self._entries` and land one
entry's words under another entry's selection. (3) `refresh_reader_measure()` extracts
the `__init__` pixel-cap computation into a callable method, wired into
`MainWindow._apply_scaled_metrics` (`scripts/ui/app.py`, one line — the seam that method
exists for) so a scale change re-applies the reader's cap the same way it re-applies
every other Python-side pixel budget on the page; before this the cap was computed once
and never again. **Tests.** Three added to `tests/test_g3_market_journal_reader.py`
(15 total, nothing weakened): converts a UTC stamp to Pacific and asserts the offset
label; a naive stamp still says unrecorded; a dummy header row with no `entry_id`
inserted at row 0 proves the reader still reads the SELECTED entry, not whatever row
index it now sits at; a font-size change proves `refresh_reader_measure()` recomputes
the cap. No store, identity, timestamp write, or layout dimension changed.
### 2026-09-06 - Packet G1: Weekend Prep › Focus Review is one table, a view selector and a detail pane (branch `claude/g1-weekend-focus-review`)

Trader, 2026-09-06, prioritising the Desk Reshape Plan: the **"Weekend Prep
overlap"** is the first layout repair of Phase 0.22 (lane G). It is a LAYOUT
packet: positions, widths, words and defaults may change; a number, a read, a
sort key's meaning and a write may not.

**The overlap was arithmetic.** `FocusReviewPage` built nine `QTableWidget`s
through `_ten_row_table`, each with the 260 px ten-row floor R4 A18 applied to
every table on the tab, and stacked them in ONE `QVBoxLayout` with about twenty
labels between them: 9 × 260 = **2,340 px of minimum height** in the roughly
2,050 px a 2160 screen gives the page. Measured on the pre-fix code the page
insisted on **2,824 px** and `WeekendPrepPanel.resize(3456, 2160)` came back
**2,934 px** tall, because the panel could not go below the minimum this one
page forced on it - Focus Review was the sole driver, the next worst page (Tag
week) asking for 700. No font size fixes that.

**One stack, nine buttons, one pane.** The nine tables are a `QStackedWidget`,
one view each, behind a row of exclusive checkable `QToolButton`s in a
`QButtonGroup`: Week's picks (`table`), Picks graded (`performance_table`),
Vetoes (`cohort_table`), Likes (`like_table` + `claim_caveat`), After-like
(`after_like_table`), Passes (`pass_table`), Not-today (`rejection_table`),
Said vs did (`preference_table`), Said at the time (`feedback_table`). Default
Week's picks, remembered for the session and unchanged by a refresh; persisting
it across restarts was deliberately left out. The horizon combo moved onto the
selector row and is VISIBLE only on Vetoes and Likes - a control that changes
nothing in the table under it reads as a control that is broken. The floors come
off THIS page only; `_ten_row_table` and `TABLE_TEN_ROWS_PX` are byte-identical
and still applied on the other five pages, and
`test_every_weekend_prep_table_shows_ten_rows` now EXCLUDES Focus Review rather
than being deleted - the ten-row promise there is measured on the VIEWPORT at
1440 instead (`viewport().height() // defaultSectionSize() >= 10`).

**The detail pane.** Beside the stack, behind a `QSplitter(Horizontal)` at 3:1,
a read-only `QTextBrowser` prints every column of the selected row as
`header: value` with the reason / note / claimed setup / verdict IN FULL - the
cell elides at PAINT time, the pane does not - and clears to EMPTY when the view
changes, with no placeholder, because a sentence sitting where a row's own words
belong reads as the row's own words. No "Chart it" button: this panel has no
symbol route to the Desk and adding one is a different packet. Each view carries
a POPULATION SENTENCE saying what a ROW is (a cohort row and a pick row look
identical in a table), and it carries no count - the counts stay in the note
under the table, written by the render that has them.

**The read did not move and is still one pass.** `_read_everything` reads all
nine stores on the page's `_ReadWorker`; `_on_focus_ready` fills all nine tables
on every render regardless of which view is visible; selecting a view is
`setCurrentIndex`, a cleared selection and a cleared pane - no file, no worker,
no re-render, pinned by a test that monkeypatches all ten readers and asserts
none is called. `apply_width_rule_to_table_widget` calls are unchanged, the
verdict card is still uncapped (gate #49), and `Refresh everything`, Mark done
and Skip are untouched. Both new widgets are styled by object name in
`theme.qss` (`QToolButton#WeekendViewButton`, `QTextBrowser#WeekendRowDetail`);
the page sets no stylesheet, and a test asserts no widget under it carries one.

Tests: `tests/test_g1_weekend_focus_review.py` - the tester's seven (fit at
2160, ten rows at 1440, one visible table per view, the read unchanged across
all nine views, the view survives a reload, the pane, the floor's narrow
exclusion) plus two the builder added (themed by object name; the horizon combo
still re-filters both cohort tables from memory). **All nine proven RED first**
against `scripts/ui/panels/weekend_prep_panel.py` and `scripts/ui/theme.qss`
restored to the tester's tip.

**Fix round, 2026-09-07 (reviewer NO-GO, one blocker, two advisories taken).**

- **The detail pane went STALE after a refresh** - the blocker. It was wired to
  `itemSelectionChanged` and nothing else, and `_on_focus_ready` re-fills all
  nine tables without touching it, so a render that KEEPS the row count left
  the row selected, never re-emitted, and the pane went on describing the
  PREVIOUS read under the same row number (the reviewer's reproduction: row 0
  reads n=999 / 0.99 after the refresh, the pane still reads n: 78 / 0.55).
  `_refresh_detail_pane` re-reads the visible view's selected row from the NEW
  cells, and clears when the new render could not carry the selection. It runs
  at the end of BOTH render passes: `_on_focus_ready`, and
  `_on_cohort_horizon_changed`, which is the same staleness through a second
  door - the horizon swaps the cohort rows from memory and another horizon
  with the same row count keeps the selection exactly as a refresh does. A
  render that SHRINKS the table already dropped the selection and Qt re-emitted
  by itself, which is why only the equal-count case rotted and why that case is
  a passing regression guard rather than part of the fix.
- **Clicking the view already shown is a no-op.** An exclusive checkable
  `QToolButton` still emits `clicked` when it is already checked, so the
  selector re-ran the view change and threw away the selected row and the pane
  - a click that moved nothing on screen except the one thing being read.
  `_on_view_button_clicked` returns early on an unchanged index;
  `_select_view` stays unconditional because the constructor calls it on a
  `_view_index` that already equals the default and must still check the button
  and set the horizon visibility.
- **Two notes named positions that no longer exist.** The like note's "the veto
  table above" is now "the Vetoes view"; the feedback note's "the rollup above"
  is now "the Picks graded view". Wording only.
- **The captions carry no count, and that stands as built**: the count lives in
  each view's own note, which is where the render already has it, and the
  population sentence says what a ROW is rather than how many there are.
- Five more tests in the same file, four **proven RED first** with
  `weekend_prep_panel.py` restored to `4736388c` (4 failed, 10 passed), 14
  passed with the fix. Live gate renumbered to **#82**: G4 reached `main` at
  `18d3f91d` first and took #80.

### 2026-09-06 - Packet G3: the Market Journal's full thought is readable (branch `claude/g3-market-journal-reader`)

Phase 0.22, the G (layout) lane, third of the trader's prioritised layout repairs. The
GUI review's finding was functional, not cosmetic: the left-nav Market Journal page is
where the trader re-reads what they thought, and the words were shown NOWHERE. The whole
text went into a `QListWidgetItem` in a narrow list, which elides it to one clipped line;
selecting the row repainted the four capture charts and did nothing else. **No store,
identity, timestamp or write behaviour changed** — a G packet may change words, widths,
positions and defaults and never how anything is computed.

- **G3.1 the list shows an excerpt, dated.** `_excerpt(text, limit=EXCERPT_LIMIT)`
  (module level, 90 characters) takes the first line and appends `…` **only when there is
  more** — the ellipsis is a claim, so a short entry is still shown whole. A 1,200-character
  entry's label went from 1,221 characters to 112. The date stays first, the two-space
  separator stays (`test_the_entries_list_is_dated_and_newest_first` splits on it), and
  `[desk]`, `[written after the session]`, the 📈 marker, the symbols and the
  `written <created_at>` tooltip are all unchanged.
- **G3.2 a reader pane.** `thought_meta` (a `QLabel`) over `thought_view` (a read-only
  `QTextBrowser`, word-wrapped, selectable, no `setStyleSheet` — `QLabel#ThoughtMeta` and
  `QTextBrowser#ThoughtReader` are new `theme.qss` rules), in a new `QSplitter(Vertical)`
  in the right half with the reader ABOVE at stretch 2 and the existing charts widget
  BELOW at 3. Stretch alone governs only resizes, so the split also gets `setSizes`
  (measured 796 / 1194 at 3456 × 2160); without it the empty chart grid's size hint opened
  the reader at two lines. `setPlainText`, never `setHtml`: a thought containing `<` is
  not markup. The meta line reads `2026-09-05 · D1 · written 13:36 UTC-07:00 · SPY` —
  the zone is the one the stored `created_at` CARRIES, and a naive stamp says
  `(no zone recorded)` rather than being silently given one. The left half is untouched
  (the fix round then opened the `lower` splitter at one third: 1,151 / 2,301 px at 3456 × 2160, measured by the reviewer).
- **G3.3 selection fills the reader FIRST.** `_on_entry_selected` filled the reader at its
  HEAD, from `self._entries[row]` through `_entry_for_row`, **before** the no-capture guard
  that returns early at what was `:506-512` and before `_CaptureWorker` is constructed. So
  an entry with no capture is readable (and the charts note still says exactly what it said
  before), and the words never depend on a worker — the late-capture guard proves a
  payload can arrive for a row the trader has left, and one entry's words under another
  entry's selection would be the worst failure this page could have. `_render_entries`'s
  empty branch clears the reader beside the charts.
- **The composer stops eating the page.** The vertical splitter's top pane opens at four
  text lines (`COMPOSER_LINES`, from the box's own `fontMetrics().lineSpacing()`): measured
  228 → 106 px at 3456 × 2160. The handle still drags it as tall as the trader wants.
- **Fix round: the thought gets a MEASURE.** The first cut set `thought_view` across the
  full 2,779 px of the right half, where a 68-character sentence became one 400-character
  line, so the text (and `thought_meta`) is now capped at `_reader_measure` —
  `QFontMetrics.averageCharWidth() × 100` or `theme.px(1200)`, whichever is SMALLER, floored
  at `theme.px(240)` — which under the desk's own theme is 600 px and **96 characters a
  line**, left-aligned with the slack on the right so the pane keeps its full width for the
  charts below.
- **Fix round: the entries column opens at a third of the page.** Stretch governs only
  RESIZES, so the size hints opened it at 655 px and clipped the new 90-character excerpt
  after about 35 — `lower.setSizes(LOWER_SPLIT_SHARES)` opens it at 1,132 px at 3456 × 2160
  (proportions, not pixels), and the handle still drags either way.
- **Tests.** `tests/test_g3_market_journal_reader.py` — the tester's six (excerpt under 140
  characters and dated; the full 1,200 characters in the reader; selection changes it; no
  capture still fills it; the empty render clears it; the reader is already filled when the
  monkeypatched module-level `_CaptureWorker` is CONSTRUCTED) plus three added by the
  builder (the ellipsis is never decoration; the reader sits above the charts on its own
  vertical splitter at 2/3, read back off the child size policies since `QSplitter` has no
  `stretchFactor` getter; the meta line keeps the stamp's zone). All nine proven failing on
  the pre-G3 panel. `test_r4_market_journal_page_and_tables.py`,
  `test_qt_market_journal_page.py` and `test_v2_market_journal_one_box.py` stay green
  untouched.
### 2026-09-06 - G0: measure first (branch `claude/g0-measure-first`)

The first step of Phase 0.22's build order, authorized by the trader's *"lets use your
recommendations for all 4 decsions. then go ahead and start the build order"*. It builds a
measuring tool and takes one baseline; **no panel changed and no trader-facing behaviour
changed**.

- **`scripts/ui/desk_bench.py`** - the workload bench and the layout-fit check. Inventory
  line above; runbook section 7. Nothing on the desk imports it and no timer starts it.
- **The baseline, `desk_bench_baseline_2026-09-06.json`** (offscreen, 3456x2160 /
  3840x2160 / 2560x1440, `--repeat 3`, over a 383.6 MB staged copy). Chrome measured
  90 px from the widgets, so 2160 leaves 2,070 px of page. **Seven ops over 250 ms sync
  p95, and they are three ops at three sizes**: `research.construct` 5,142 / 4,312 /
  4,400 ms (it builds eight children eagerly), `setup_tracker.refresh` 1,320 / 1,098 /
  1,060 ms, `market_journal.construct` 299 ms at the target size only.
  `weekend.refresh_everything` returns in 1.7 ms and settles in 12.7 s p50, hitting the
  20 s deadline once - the V2 design working exactly as written, and still 12 s of a page
  filling in. **The worst single `processEvents()` in the whole run was 806.6 ms**, in
  `research.construct` at 2560x1440; the 683 ms first recorded here was the worst inside
  the Weekend wait, not the run.
- **The fit check sees the defect G1 fixes.** `weekend_prep` needs 3,072 px and
  `weekend_prep.focus_review` 2,858 px against 2,070 - flagged `overflow` at every size,
  including 3840x2160, because the overflow is vertical and the width does not help. Of
  the focus page's requirement, 2,340 px is nine table floors of 260 px (`TABLE_TEN_ROWS_PX`)
  stacked in one vertical layout. Every other page fits at all three sizes.
- **Two guards, not one, on the live store.** The fail-before-fix proof for the first guard
  (sabotage it, watch nine tests fail) also staged six synthetic files into
  `C:\TradingBotData\scratch` and the same folder on the DAS. Both trees were new, both
  were removed, and no live file was touched - but a guard whose proof requires breaking it
  needs a second guard that the same edit does not disable, so
  `_refuse_to_open_for_writing` compares its own literals and runs before the destination
  is created and again before every file is opened.
- **Not offline**: constructing the Research tab reaches `treasury_calendar_service`, which
  attempted an HTTPS call on every run and failed on certificate verification. Recorded
  rather than fixed - G0 changes no panel.
- **The fix round, the same evening (reviewer NO-GO, one blocker).** `main()` called
  `_prepare_environment` - which mkdirs `--data-dir` and writes
  `<data_dir>\_localappdata\TradingBotV3\local_settings.json` into it - and only THEN
  asked whether that directory was the live store, so `--data-dir C:\TradingBotData`
  exited 2 with three directories and a file already inside the live home folder: the
  packet's own invariant broken by the module that states it. Both guards now run on the
  ARGUMENT at the top of `main()`, through `refuse_live_destination`, for `--data-dir` and
  `--out` (and again on the resolved out path, default included, before its mkdir).
  `_abort_if_live` still runs after, because it answers the different question of what
  `project_paths` resolved to. The second guard's literals moved into
  `WRITE_REFUSAL_PREFIXES` so a test can point BOTH guards at a FAKE live root under
  `tmp_path` and assert nothing is created there - no test aims a sabotaged guard at the
  real live paths as a destination, which is how the first round created
  `C:\TradingBotData\scratch`.
- **The bench's settings are the trader's, minus the secrets** (same round). The
  `local_settings.json` allowlist entry named the home-folder root, where that file does
  not live, so it reported `absent` on every staging run and the baseline was measured
  against a synthetic one-key file. It is dropped; `machine_settings_seed` carries an
  allowlist of keys out of `%LOCALAPPDATA%\TradingBotV3\local_settings.json` at run time -
  the display keys that change what is measured plus `daily_bars_source: yahoo`, and never
  one of the five credentials or three live-store path keys that file also holds.
  `qt_autopilot_auto_arm` is forced False and is not read from the real file.
- **Live gate #81** (renumbered from #75 after the ST1-ST5 merge took #75-#79 and G4 took
  #80): the same run windowed. Its third clause now reads "the two SLOWEST ops by sync
  p95" rather than "the two ops over 250 ms" - three ops crossed 250 ms at the target size
  in this baseline and four in the reviewer's re-run, so the line is noisy and the ORDER
  is the check. 38 tests.
### 2026-09-06 - Packet G4: the explanation pane clears when its context changes (branch `claude/g4-stale-research-detail`)

The GUI review of 2026-09-06 found the Day-trade Tracker still showing the `lrsi_cross50`
explanation with the **Combos** tab open and no combo selected. `show_row` set HTML and
`setVisible(True)` and nothing ever took the pane back down: neither tab strip had a
`currentChanged` handler, and `_on_refresh_finished` / `_on_held_run_loaded` replaced every
model row without touching it. So the last row clicked stayed on screen through every tab
switch and every re-aggregation, beside a table that no longer contained it - the numbers on
the right are read as the numbers on the left, which makes this correctness rather than
polish.

- **G4.1 - the pane knows what it shows.** `ResearchExplanationView.show_row` takes an
  optional `identity=` and stores it on `shown_identity`; `clear()` is an OVERRIDE, because
  `QTextEdit.clear()` already existed and only empties the document, which would leave an
  empty pane standing where the explanation was. The override empties, hides and forgets.
  `SetupDetailView` gains the same pair - its identity is
  `(kind, side, family, symbol-or-blank, dimension-or-blank)`, computed in `_render` from
  the row it just drew, and its `clear()` also drops `_current` so a late levels callback
  cannot re-open a cleared pane. Nothing calls `SetupDetailView`'s pair yet.
- **G4.2 - Day-trade Tracker.** `_explanation_identity(kind, row)` is
  `(kind, dimension, direction, segment)`, read from the row dict and never from the display
  text; `direction` is in it because `long vwap` and `short vwap` are two measurements and
  the learning store itself keys a segment `direction|segment`. `tabs.currentChanged` and
  `decisions_tabs.currentChanged` clear the pane (the second is belt-and-braces: the outer
  strip fires first in the live GUI, so it is only reached for a move between the My
  Decisions sub-tabs). Both data-revision slots call `_reshow_or_clear_explanation`, which
  looks the identity up in the model that now holds the tab's rows and re-shows from the NEW
  row dict - re-showing the cached one would reproduce the defect wearing a number instead of
  a name - or clears when the segment is gone.
- **G4.3 (Setup Tracker) is DEFERRED to packet G4b**, with tests 5-6: `setup_tracker_panel.py`
  is being rewritten by ST2/ST6 and is untouched here.

Tests: `tests/test_g4_stale_research_detail.py` (tester-first, five red at `98668de3`) plus
`tests/test_g4_setup_detail_view_identity.py` added by the builder for G4.1's second widget.
All six proven failing with the three production files restored, then green. Layout lane: no
number, model, sort or read changes.
### 2026-09-07 - Packet ST7: the trader's three decisions become the tracker's defaults (branch `claude/st7-decisions-default`)

Trader, 2026-09-06 ~21:15 PT: *"Yes a trade not yet completed should say pending. A second
entry after a first close is its own trade yes. 3. This one is up to your discretion."*
Decision (3) - the execution convention and the level knowledge - was taken by the lead
under that discretion. All three are recorded as
[`docs/decisions/0019-tracker-selection-and-execution-defaults.md`](docs/decisions/0019-tracker-selection-and-execution-defaults.md).
ST3 and ST4 built the repairs as opt-in evidence; ST7 is the flip and nothing wider.

- **The three defaults moved and the v1 names did not.**
  `selection_policy.DEFAULT_SELECTION_POLICY` is `first_actionable_v2`,
  `execution_convention.DEFAULT_EXECUTION_CONVENTION` is `gap_aware_v2` and
  `DEFAULT_LEVEL_KNOWLEDGE` is `prior_session_v2`. `closed_first_v1`, `literal_level_v1`
  and `same_session_v1` keep their values, stay selectable by keyword, and are the "old"
  arm of `tracker_selection_compare.py` / `tracker_execution_compare.py`. **History is
  restated by construction** - the tracker rebuilds every record on each persisted write -
  and that is the point of the decision, not a side effect.
- **Every record now NAMES the policies that produced it.** ST3 wrote the two execution
  stamps only for a non-default run and POPPED them otherwise, which was legible only
  while there had never been a flip; after one, an absent stamp could mean either policy.
  `recompute_tracker_setup_record` now writes `execution_convention` and `level_knowledge`
  unconditionally, on the v1 path too, and `selection_policy` is on every family row and
  every `_scoring_outcome_summary`.
- **The persisted tracker write says which generation is on disk.** There was NO success
  log line at that seam; one was added at the call site, before
  `save_setup_tracker_payload`, reading `Setup tracker policies: selection=<..>
  execution=<..> levels=<..>` and logged unconditionally, because an absent line and a
  line naming the v1 policies are different facts. It is what live gate #84 greps for.
- **A compact scoring projection written before the flip is still read AS-IS.** ST4's
  cache rule survives: a DEFAULT read of `_scoring_outcome_summary` takes it
  unconditionally (naming `first_actionable_v2` explicitly is the same read), and only a
  non-default policy - now `closed_first_v1` by name - or an `as_of_session` replay takes
  the `unknown_compact` bypass. **One consequence needed its own rule**: a pre-ST4 cache
  carries no `representative_status` at all, and v2's "pending stays pending" keys on that
  column. An ABSENT column is not a pending trade, so the aggregate reads the row's own
  `closed_setups` and counts it as `no_representative_in_population`. **Measured on a COPY
  of the 2026-09-06 snapshot, reverting the bridge**: the 32 recent family rows still
  appear, but their graded population goes **1,688 closed -> 0** (all 2,195 episodes read
  pending) and the nonzero `recent_tracker_score_delta` count goes **14 -> 7**. The
  `setup_type` deltas are 74 either way - `build_tracker_setup_type_rows` takes no policy -
  so this is a smaller failure than the first ST4 build's total collapse and the same kind.
  The bridge is keyed on the column being genuinely ABSENT, not on an empty string: 48
  records on that snapshot have the column with an EMPTY value because ST4+ measured them
  and found no representative scenario, and those stay UNMEASURED. A PRESENT `pending`
  status is still never graded.
- **Both defaults are pinned to goldens frozen BEFORE the flip.**
  `tests/fixtures/st7_v2_default_golden.json` (a whole replay record) and
  `tests/fixtures/st7_family_rows_v2_default_golden.csv` were pinned on `main` at
  `68762909` through the explicit v2 keywords that already shipped, so neither is a
  self-portrait; the three stamp keys are excluded from the byte-identity check and
  asserted separately by name. `st3_replay_golden.json` and `st4_family_rows_golden.csv`
  are untouched and are now reproduced with the v1 policies NAMED.
- **37 pre-existing tests were re-pinned, none weakened.** Every leg that reached v1
  through the bare signature now passes the v1 policy by name and asserts the same numbers;
  every leg that read "and the default agrees" now asserts the v2 answer beside it. Two
  test functions were renamed because their names asserted something the flip made false
  (`test_the_default_replay_reproduces_the_pinned_golden_and_v2_only_adds_its_keys` ->
  `test_v1_by_name_reproduces_the_pinned_golden_and_the_default_is_the_v2_pin`;
  `test_default_policy_reproduces_the_golden_family_rows_byte_for_byte` ->
  `test_v1_by_name_reproduces_the_golden_family_rows_and_the_default_is_the_v2_pin`), and
  `test_the_default_event_dict_has_exactly_the_keys_it_always_had` became
  `test_the_v1_event_dict_keeps_its_keys_and_the_default_adds_exactly_two` - the default
  event dict legitimately carries `fill_basis` and `execution_convention` now, and the
  test pins that it adds exactly those two and no more.
  `build_tracker_band_variant_parity_fixture.measure` gained three policy keywords so the
  Phase 0.10 B-2 parity fixture is read under the policies it was FROZEN on; a new test
  re-runs the same shadow FENCE under whatever the defaults currently are, so a future
  policy change cannot quietly unfence the band challenger.
- **Shadow only, still.** Nothing here promotes a setup or reaches a detector, a score, an
  alert, a watchlist, Focus, the review queue or `review_policy.json`;
  `calc_anchored_vwap_bands` and `calc_anchored_vwap_band_history` are untouched (decision
  0008). Live gate #84.
- **Review round (2026-09-07).** The reviewer accepted the code and found five things the
  TEXT got wrong, each of which would have cost a Tuesday reading. (1) Gates **#77 and #78**
  were written while the flip was still owed and demanded the opposite of the truth - no
  policy stamps on any record, `closed_first_v1` on every row - so both are now marked
  SUPERSEDED by #84 with their surviving clauses named; an unrewritten gate would have
  recorded a false FAIL. (2) The two compare CLIs still called v1 "what the desk runs
  today" inside artifacts that are read months later as the record of what was live;
  `README_TEXT` and the execution CLI's `authorization` / arm notes now name the default and
  the "old" arm, and `README_TEXT` is asserted by a test. (3) The bridge's blast radius was
  OVERSTATED in four places: reverting it on a copy of the live snapshot gives 32 rows (not
  0) with the graded population 1,688 closed -> 0 and nonzero recent deltas 14 -> 7, while
  the 74 `setup_type` deltas are unaffected because that builder takes no policy - restated
  everywhere with the measured figures. (4) The bridge was too WIDE: keyed on the coerced
  string being falsy it also caught 48 live records whose ST4+ summary carries the column
  PRESENT and EMPTY (measured, no representative), turning 13 unmeasured episodes into
  graded ones. It is now keyed on the key being ABSENT, through a new non-exported
  `representative_status_known` row field, and gate #84 expects `n_pending` 902 rather than
  ST4's 915. (5) Narrowing it surfaced that a study family whose stop is not the protective
  band has NO representative and is unmeasured under the default - real, shadow-only, and
  now pinned by `test_recent_family_rows_include_1stdev_breakout` asserting both readings.
  Also: five stale "v1 is the default" comments in `legacy.py`, a test for the
  `Setup tracker policies:` log line, and the 25 dated entries from 2026-09-05 back to
  2026-09-03 moved to `docs/archive/CHANGELOG_ARCHIVE_2026-09-03_2026-09-05.md` so
  `Recent changes` holds the two build days its own header promises.

### 2026-09-06 - Packet ST5: personal evidence usable without inventing it (branch `claude/st5-personal-evidence-build`)

Trader: *"Fix the declared session-window mismatch with calendar tests... Deduplicate by
trade identity when computing trade counts/P&L, while retaining all source statements.
Audit option exposure/leg structure before interpreting direction as market bias; unknown
facts remain unknown... Do not tell me a personal setup is best when there are no confirmed
tags. Separate complete trades, partly closed trades, open exposure, and
instrument/strategy uncertainty in summaries."* The journal is READ-ONLY to this packet: no
broker call, no auto-confirm, no reconstructed risk.

- **ST5.1 the window is sessions and says so.** `TRADE_WINDOW_SESSIONS` (10) replaces
  `TRADE_WINDOW_DAYS`, kept one release as an alias (nothing in `scripts/` imported it).
  `statement_window_end` walks `market_calendar` forward ten SESSIONS, so a statement on
  Friday 2026-09-04 reaches 2026-09-21 rather than 2026-09-14 - the constant said DAYS
  while every docstring said sessions, and over Labor Day that is five real sessions
  discarded and with them a trade taken on the 18th. A calendar that refuses (outside its
  validated 2000-2032 range) falls back to the OLD, strictly NARROWER arithmetic and logs:
  uncertainty may never widen a window into a match nobody made. Confidence labels
  unchanged.
- **ST5.2 counts are by trade, statements are kept.** `trade_level_summary(rows)` reports
  `n_statements_matched`, `n_trades_matched`, `duplicate_statement_rows`,
  `statements_per_trade`, `planned_risk_recorded` and a P&L summed ONCE per `trade_id`.
  Live: 538 report rows, 13 `traded=yes`, **10 distinct trade ids** - a statement-grain
  total was three trades' P&L too large. Every row is still kept (the skip is the
  interesting row), `summary_note` prints both denominators, `run_preference_trade_outcomes`
  carries the pair, and Weekend Prep's note reads "13 were traded over 10 distinct trades".
- **ST5.3 direction is ownership until the legs say otherwise.** New pure
  `scripts/journal_exposure.py`. 53 of the 89 live option trades are SHORT and 39 of those
  were winners; read as "short = bearish" that is a bearish trader with a bullish record,
  and they are sold puts. A **LONG option is never a bullish setup**. `multi_leg` means
  more than one distinct option CONTRACT among the legs, because a `trade_legs` row is a
  FILL and every closed option trade carries at least two. The contract comes from the OCC
  `trades.symbol` and from `raw_executions.raw_json["option"]`, so
  `JournalStore.list_trade_legs` gained ONE column. **`partial_of_spread` could not be
  derived** - two spread legs are two `trades` rows keyed by their own OCC symbols and
  nothing links them - so the observable pattern is labelled
  `partial_of_spread_candidate` and lands in the uncertain population.
- **ST5.4 summaries separate what they can and cannot say.**
  `journal_analytics.personal_evidence_summary(trades)` partitions every trade by STATUS
  into `complete` / `partly_closed` / `open_exposure` (live: 165 / 7 / 32, summing to 204)
  and carries uncertainty as a CROSS-CUTTING label - `n_uncertain` on each population plus
  an `uncertain` block that names each member's status and pools no money (live: 120,
  split 84 / 7 / 29). An open position's `net_pnl` AND `winners` are both `None`, never
  zero; its notional is 61,662. Both tag lanes share ONE denominator, closed or partly
  closed, so the journal's single confirmed tag - on a CLOSED_PARTIAL trade - is counted:
  `Confirmed tags: 1 of 172 closed or partly closed trades. Provisional awaiting review:
  26. Planned risk recorded: 0 of 172.` and `1 confirmed setup tag - under the n=30 floor
  (26 provisional awaiting review) - no personal setup can be called best.` The first cut
  of both of these was wrong and the review caught it; see the checkpoint entry.
- **ST5.5 the 26 proposed tags reach the review flow.** Weekend Prep's tag list is widened
  to the whole provisional backlog with an injection seam for tests; `needs_review` stays
  week-scoped (145 rows with no proposal). A "Missing planned risk" table lists the closed
  trades with no plan (0 of 204 carry one) and a row REFERS the trade to the Journal's
  Trades tab, where `save_risk_fields` already lives. No new writer.
- **Tests.** The tester's seven in `tests/test_st5_personal_evidence.py` were red at
  `f6a28138` and are green; the builder added twelve in `tests/test_st5_review_flow.py`,
  ten of which were proven red against the branch base's `scripts/`.
- **One deviation, recorded.** The packet asked for the coverage line on Weekend Prep's
  verdict card. The card is five to eight lines by the trader's own request (V2 item 2b)
  and two tests pin it; a ninth line broke `test_one_unreadable_store_still_leaves_a_card`.
  The sentence sits in its own label directly UNDER the card, filled from the tag page's
  existing worker through `coverageChanged` - same screen, same seam, no second read.
### 2026-09-06 - Packet ST3: no impossible fills, no same-day knowledge (builder, branch `claude/st3-gap-aware-fills`)

Trader: *"Declare a versioned execution convention for long/short stop gaps, missing opens,
invalid OHLC, and target gaps. Never book a fill outside the available bar through the
current literal-level assumption."* ... *"Resolve intrabar target levels only from
information available before they could be hit."* ... *"This is not authorization to
overwrite live historical results or promote the repaired simulation into scoring."*

- **The defect, reproduced on the real function.** Entry 100, risk 5, hard stop 95, next
  bar O80/H85/L79/C82: `_evaluate_tracker_scenario_bar` booked `HARD_STOP` at **95**, a
  price the bar never traded, for **-1.014R** after costs. The honest fill is the open at
  80, for **-4.014R**. Separately, `calc_anchored_vwap_band_history` folds day D's own
  OHLC and volume into the cumulative sums BEFORE writing `history[D]`, and the replay
  tested day D's high/low against `history[D]` - a level knowable only at D's close.
- **The repair is additive, opt-in and versioned**, and the DEFAULT path is byte-identical
  (see the inventory bullet above for the full contract). Two axes:
  `literal_level_v1` / `gap_aware_v2` for the fill, `same_session_v1` /
  `prior_session_v2` for the level. Stop-first ordering, the maximum-hold force close,
  the ONE cost model and the frozen AVWAP formula are all untouched; only WHICH DAY's
  level a bar is tested against and WHAT PRICE a touch books can change, and only when a
  caller asks.
- **The lead's decision on the packet's one ambiguity (2026-09-06):** an INVALID bar that
  lands on the maximum-hold index books nothing and the `TIME_STOP` fires on the next
  VALID bar with `fill_basis` `deferred_invalid_bar`. Maximum hold is preserved, never
  cancelled, and the record says which bar could not answer.
- **The comparison, on copies, seed 20260906 across the whole 11,372-record mirror COPY**
  with daily bars from the machine cache. **`n_setups` 794 is the denominator** - 800
  records were offered, 6 carry no tradeable scenario, 0 lacked bars, 0 failed to replay;
  `--limit` is what was OFFERED and is never the denominator, so the CLI prints the whole
  split and the JSON carries a `population_note` saying so. The three JSON/CSV pairs are
  copied to `%LOCALAPPDATA%\TradingBotV3\diagnostics\st3_execution_compare\`
  (`both__comparison_20260906T110731.*`, `exec_only__…110814.*`, `levels_only__…110855.*`).
  The first draft reported `min R -4.0 -> -4.0`, which is `TRACKER_SCORING_R_CLIP` and not
  a tail, so the artifact carries the CLIPPED and the RAW R side by side, never blended:

  | run | changed | expectancy (raw) | win rate | R < -2 | groups moved rank |
  |---|---|---|---|---|---|
  | both repairs | 472 of 794 | -0.0981 → **-0.1192** | 0.576 → 0.596 | 47 → 48 | 43 of 50, max 14 |
  | execution only (`gap_aware_v2`) | 89 of 794 | -0.0981 → **-0.0811** | 0.576 → 0.597 | 47 → 47 | 33 of 50, max 9 |
  | level knowledge only (`prior_session_v2`) | 458 of 794 | -0.0981 → **-0.1491** | 0.576 → 0.548 | 47 → 48 | 40 of 50, max 17 |

  **The gap-aware convention is symmetric by design and, in this sample, mostly
  HELPS**: of the 89 setups it moved, 84 got better and 5 got worse, because a resting
  limit that opens through its price fills BETTER than the level and target gaps
  outnumber stop gaps 166 to a handful. The prior-session level knowledge is what costs
  expectancy (413 of 458 moved setups worse). Both numbers are evidence for a decision
  the trader has not taken; nothing here promotes anything.
- **The `invalid_bar` counter found a real one on its first run.** It fired 380
  scenario-bars over 26 setups / 22 symbols, and each of those 22 cached daily-bar files
  holds **exactly one invalid candle, all dated 2026-09-04**, every one with `low > open`
  or `high < open` (AEE `O=105.81 H=106.96 L=106.11`; TWLO `O=239.52 H=239.29`) - the
  signature of a FORMING bar written into `machine_cache\daily_bars` mid-session. Under
  `literal_level_v1`, which is what the desk runs, those bars are still read for fills,
  excursions and marks. A machine-local cache read, not a claim about the tracker or the
  durable store, and outside ST3's scope - recorded because the counter is what made it
  visible. **The reviewer's read-only sweep of the whole cache (all 1,980 files) widened it: 100 files end in an invalid candle, always the LAST row and one per file, on SEVEN sessions - 2026-09-04 x89, 09-02 x3, 09-01 x3, 08-20 x2, 07-07, 06-08, 05-15 - so it is a recurring mid-session write, not one day's glitch; it needs its own packet.**
### 2026-09-06 - ST1: each outcome gets its true meaning and its true clock (branch `claude/st1-outcome-clock-build`)

Trader: *"Implement the already-owed desk Working-lately surface, priority switch, and Away Recap
around one deterministic evidence snapshot ... Reuse and correct the existing best-now banner rather
than leaving competing leaders ... Any new confidence calculation must account for shared sessions
and overlapping holdings and must be validated; ordinary Wilson bounds alone do not solve dependence
or multiple testing ... Choose any margin/persistence rule before inspecting its forward evaluation.
The first release can report an observational leader or no clear leader; do not call every winner
proven."*

Closes plan.md Phase 0.14 **V1 item 4** (Working-lately + priority switch) and **V2 item 3** (AWAY
Recap) - the last two "V4, NOT BUILT" rows apart from Weekend Prep's takes table. Live gate **#83**
is owed at the first DESK session after merge.

- **No new confidence calculation was written, on purpose.** The trader's own condition made one
  unaffordable, so dependence is answered by REFUSING (`CONCENTRATION_LIMIT` 0.5) and multiple
  testing by PRINTING (`observational leader among K cells`). `LEADER_PERSISTENCE_SNAPSHOTS` = 2
  and the concentration limit were declared before any forward evaluation and neither was tuned.
- **The day-trade headline had never measured its own concentration.**
  `held_run_score.py`'s `summarize` call carried no `symbols=` / `sessions=`, so every held x ran
  cell the desk has shown came back with an unmeasured concentration and an unmeasured
  session-block bootstrap. The episodes have carried the identities since V1; this hands them over.
- **The banner and the card cannot disagree, in either direction.** ST2's fix round put the Setup
  Tracker's Summary card and its banner on one `select_leader`; ST6 would have split them again the
  first time the persistence rule held a new leader back, so the card takes the SHARED verdict when
  one is present. `research_explanations.verdict_sentence` is the split-out renderer both use.
- **Fail-before-fix, proven twice.** The tester's eleven (`tests/test_st6_working_lately.py`,
  committed red at `1323d5b0`) plus fourteen builder tests
  (`tests/test_st6_service_and_surfaces.py`); all of them were re-run against `1323d5b0`'s
  `scripts/` and failed, then passed on the branch.
- Merged in: ST1 at `5cf0e681` and ST2 at `22aac5fb`, doc conflicts resolved by keeping both
  entries.

- **The `win` column was never a win.** `master_avwap_tier_outcomes.csv`'s `win` is
  `side_return_pct > 0` - the sign of a close-to-close percent move between two of a
  symbol's OWN scan rows - while `swing_headline.headline_from_tracker_rows` told four
  surfaces it was "the stop-at-a-level, two-closes rule". `outcome_kind` is APPENDED to
  both writers (v1 value `favorable_direction_scanrow_v1`; an absent or empty cell reads
  as v1 through `swing_evidence.outcome_kind_of`), `Headline.outcome_kind` decides the
  words, and the setups table's header is now **Family favorable %**. Every existing
  value and column is byte-identical - pinned by a golden generated from the pre-fix code.
- **v2 walks the exchange calendar** (`master_avwap_lib/session_horizon_outcomes.py`,
  new file `master_avwap_session_horizon_outcomes.csv`, no production reader). Missing
  target-session data stays unmeasured with a reason; an unarrived horizon is `pending`,
  never a loss; duplicates are counted, not swallowed.
- **One eligible-row reader** (`scripts/swing_evidence.py`). `build_bot_tier_performance_rows`
  now drops explicit `stale_horizon` rows like the two trader-facing readers - it is a
  report export and its only consumers are the Setup Tracker's Tier performance tab, the
  human-focus comparison table and the AI evidence list; no detector, score, gate or alert
  reads it. Its cells span four horizons over a 365-day lookback, so it shares the
  MISSINGNESS PREDICATE rather than a whole policy: `read_eligible_rows` and the export
  both call `swing_evidence.is_stale_horizon`, and the export applies it to its BASELINE
  observations too. Each read reconciles, and `describe(...)` states outcome kind, horizon
  in its own unit, window and coverage on the setups panel, the setup docs and the AWAY
  digest. `ai_jobs.digest._SECTION_NOTES["swing_win_rates"]` names the policy so a model
  reading the index cannot call the rate a win rate either.
- **The tier split is measured and shown.** Live file at `end=2026-09-03`: 2,642 horizon-5
  rows in that 20-session window, **all `derived_from_bucket`, none `assigned`** (the 341
  assigned rows are horizon 1 from 2026-09-02/03), 55 dropped stale, 2,587 eligible. The
  window rolls, so the triple is only meaningful with its `end` beside it - on the default
  window (2026-09-06) the same file reads 2,462 / 0 / 17,096.
- **Deviation from the packet, deliberately:** v1's `sessions_spanned` / `stale_horizon`
  keep the BUSINESS-DAY basis. `horizon_drift` gained `calendar=` and says "exchange
  sessions" when given one, but passing it inside the v1 export would restate 19,558
  historical rows, which the trader's prompt explicitly does not approve; the golden
  proves the file unchanged. v2 needs no drift call - its span is the horizon.
- Live gate **#75** owed at the next persisted tracker write (Tuesday 2026-09-08).
### 2026-09-06 - ST4: the selected opportunity is fixed before its outcome is seen (branch `claude/st4-first-actionable`, not merged; prepared, NOT decided)

Trader: *"Prepare the golden comparison and obtain the explicit policy decision. Use a fixed
first-actionable observation/attempt identity and an explicit representative exit chosen without
outcome knowledge. A legitimate re-entry must arise from a declared entry rule, not from being the
rescan that happened to close. Pending primary scenarios stay pending. Never substitute a more
mature or more attractive alternate recipe. ... Retain existing policy under its old identity.
... Handle the checkpoint's swept_measured scoring decision, EXPIRED_UNMEASURED scoring
population, and neither-open-nor-closed baseline cases explicitly; do not silently decide them in
a cleanup."*

- **Three selections read the outcome to decide what to grade, and all three still do by
  default.** `_dedupe_recent_tracker_family_rows` sorted `(0 if closed_setups > 0 else 1,
  scan_date)`, so the 08-10 open entry lost to the 08-15 closed rescan; `_representative_scenario`
  fell to `matching[0]`, so reordering the `scenarios` dict moved one setup's headline from
  **+2.00R to -1.00R** (`tests/test_st4_first_actionable.py` pins both readings);
  `_summarize_tracker_setup_outcome` replaced an OPEN representative's R with `avg_closed_r`, the
  mean of the alternate exit plans that happened to close (**+3.00R reported for a trade still
  running**). Every one is untouched under `closed_first_v1`, which stays the default everywhere.
- **`first_actionable_v2` is the opt-in challenger**, reachable only through an explicit keyword
  and through the comparison CLI. Its re-entry rule is DECLARED, not inferred, and its
  representative exit template is DECLARED (`full_band2`).
- **The premise the packet asked to verify: there is no exit-date field on a scenario.**
  `_apply_scenario_exit_event` appends one entry per exit LEG to `events`, so the recorded exit is
  the LAST entry's `trade_date` and an open scenario has that list present and empty. The new
  `_scenario_recorded_exit_date` gates the read on the CLOSED status, because a partial leaves a
  dated event behind on a scenario still in the trade - reading it would let v2 open a second
  attempt while the first was still running, which is exactly what the rule forbids.
- **The comparison, run on a windowed JSON extract of 5,696 setups taken from a COPY of the live
  SQLite mirror** (`master_avwap_setup_tracker.sqlite`, 1,191,460,864 bytes, `data_session`
  **2026-09-03**, 11,372 setups), artifacts under
  `%LOCALAPPDATA%\TradingBotV3\diagnostics\st4_selection_compare\`. **32 cells, 27 changed, 26
  rank moves**, `n_excluded` 22 and `fully_excluded_groups` 0 under both. v1: **2,249 episodes,
  498 pending, 1,078-673, unweighted 61.6%, Wilson lower 0.593**. v2: **2,712 episodes, 915
  pending, 1,294-503, unweighted 72.0%, Wilson lower 0.699**. Episode-weighted mean R **-0.150 ->
  +0.073**. **v2 bundles TWO policy questions and both decompositions are on the record**: (1)
  holding v2's SELECTION fixed and grading it the old substituting way gives 63.5% / mean R
  -0.138, so **94.2% of the mean-R move is pending-stays-pending**, not the selection; (2) the
  win-rate move is the 463 re-entries - under the old aggregation attempt >= 2 graded 72.5% while
  attempt 1 graded 61.8%, which is v1's own 61.6% to within rounding, and after the fix the gap
  survives (attempt 1 70.5%, attempt >= 2 80.1%; histogram 2,249 / 393 / 61 / 9). **A second
  attempt exists only because the first CLOSED, so that population is survivorship by
  construction** - the trader's question, not this packet's. **Nothing is promoted and no export
  switches**; the lead asks the decision.
- **Reviewer round (NO-GO on `37e63b9c`, fixed at the tip). Two blockers.** (1) **The compact
  scoring projection IS the record.** `_build_scoring_projection` writes a projection with **no
  `scenarios` key** plus a `_scoring_outcome_summary`, and
  `master_avwap_tracker_scoring_snapshot.json` holds 11,372 of them. The first build refused a
  cached summary lacking `representative_status` and recomputed; with no scenarios the recompute
  returned `tradeable_scenario_count == 0` and dropped every setup - measured on a COPY of the
  live snapshot, `build_recent_tracker_setup_family_rows` **32 rows -> 0** and
  `build_tracker_setup_type_rows` **74 nonzero `score_delta` -> 0**, which the first D1 scan after
  merge would have written into live `recent_tracker_score_delta` / `setup_type_score_delta`.
  **A missing key is never a reason to recompute**: a default read with no `as_of` takes the cache
  exactly as it did before ST4, and a challenger or replay read of a scenario-less record answers
  `representative_status: "unknown_compact"` over the cached numbers rather than an empty summary.
  (2) **v2's headline substituted one level down**: `closed_rows` was `closed_setups > 0` and the
  win/loss loop fell back to `avg_closed_r`, grading **271 of 2,712** v2 episodes whose
  representative was still running (252 losses, 19 wins). The aggregate now honours
  `representative_status` under v2; v1 is unchanged and byte-identical.
- **A replay is blind to a COMPACTED record and says so.** History compaction empties
  `scenario["events"]`, so `_scenario_recorded_exit_date` cannot date a compacted CLOSED scenario
  and an `as_of_session` build reads it as still running. 0 of the 141,324 scenarios in the 28-day
  window are compacted, but 99,562 of 206,341 across all history are, so an earlier cutoff or a
  longer lookback meets them: the row carries `representative_exit_undatable` and the family row
  names `undatable_exit_in_population=N` instead of inflating pending. `fully_excluded_groups` is
  stamped on every row because a `(side, bucket, family)` whose every record was excluded produces
  no row at all.
- **A scoring snapshot is not a tracker** (re-review). Handed a copy of
  `master_avwap_tracker_scoring_snapshot.json`, `tracker_selection_compare` would have produced a
  confident lie: v1 answers every setup from a cache the cutoff never touched (4 projections graded
  3-1) while v2 zeroes on the same input, so the report reads "v2 is broken" when the input was the
  wrong file. The CLI now counts setups carrying `_scoring_outcome_summary` with no `scenarios`,
  prints one line naming the snapshot, exits 2 and writes nothing. `_row_is_unmeasurable` applies
  under **both** policies for an `as_of_session` replay, so a v1 replay of a compact record grades
  nothing either; the default read (v1, no `as_of`) is untouched because `unknown_compact` is a
  status only the bypass path can write.

### 2026-09-06 - ST2: real integer counts at each table's own grain, and ONE honest leader (branch `claude/st2-real-counts`, not merged)

Trader: *"Export true integer wins/losses/flats/unmeasured at each table's actual episode and
outcome grain... Make the existing banner consume the same declared eligible leader as the
evidence table... A study must stay labelled study and cannot become the live leader merely by
having high R."*

- **The recent-types table showed a win count nobody observed.** `win_rate_closed` is a
  RECENCY-WEIGHTED mean of win flags; the panel handed it to
  `swing_headline.headline_from_rate`, which rebuilds an integer pair as `round(rate * n)`.
  Reproduced through the real writer: two 28-day-old wins at weight .25 plus two same-day losses
  at weight 1.0 give **0.2**, and the cell printed **`25% (>=5%, n=4)`**, Wilson bound and all,
  where the truth was **2-2, 50%**. The counts are now exported and READ; the weighted rate stays
  on the table under **Win % (recency-weighted)** beside **Win % (unweighted)**.
- **The banner and the table named different families on one screen.** `_best_now_banner_html`
  took `max(avg_closed_r)` over any row with three closed setups, across the live AND study
  namespaces. On the fixture that reproduces it the table lists `tight_and_hot` (24-6, bound
  0.627) first and the banner crowned `fat_but_wide` (54-36 at +2.50R, bound 0.497). Both now
  read `working_lately.select_leader` on the same rows in the same order.
- **V3 item 1 is COMPLETE.** The Setup Types tab has its own win counts at its own grain, leads
  with **Win %** and the ONE Wilson bound, sorts by that bound inside each side, and carries a
  population sentence naming the outcome kind - keeping M3's `N expired unmeasured, excluded`
  clause verbatim.
- **Champion preserved, proven.** Two goldens pinned from `main` at `84ee24d6`: every original
  column byte-identical, the old recent header a PREFIX of the new one, `ranking_score` and
  `score_delta` compared explicitly. Nineteen tests (eleven from the tester, eight from the
  builder), each proven red on the un-fixed code first.
- **The fix round after the reviewer's NO-GO** found the same defect surviving in three places
  the packet had not named: the Summary card's plain-English block crowned max-R-on-three THREE
  LINES above the fixed banner (live: *"LONG top_pattern ... 3 closes"* against the banner's
  *"SHORT general"*, 10 of 17 candidates studies), the **Best Type Edge** tile read
  `setup_type_rows[0]` and so followed ST2.2's new sort from `SHORT +23` to `LONG +14`, and the
  Summary's **Setup types working** block showed eight LONG rows and no SHORT one because it took
  `rows[:8]` of a side-first list (first SHORT at index 68 of 117). All three now read their own
  meaning: the card reads the SAME `select_leader` verdict as the banner, the tile picks max
  `score_delta` explicitly, and the card's eight are chosen by the bound across both books while
  the TAB keeps its side-first order.
- **The count columns are at the END of each SHIPPED header**, not the inner builder's
  (`_move_keys_to_end`; golden `tests/fixtures/st2_shipped_headers_golden.json`, contract-bearing
  and pinned from `84ee24d6`). **The floor is judged BEFORE freshness**: a family with three
  samples is under the floor whatever the clock says. `min_n` binds the stale and undated
  discovery pools by construction. `working_lately.FRESHNESS_SENTENCE` states the rule in words -
  *fresh = an entry inside 2 exchange sessions of the last completed one* - and rides in every
  policy line, because the tracker's rows carry no exit date and the ENTRY session is what is
  actually measured. The panel remembers its last FRESH leader per horizon so
  `last_reliable_reading` is reachable (in-memory; ST6 persists it).
- **The re-review round found the same shape twice more**: a surface computing
  for itself what the page had already decided. The plain-English card called
  `select_leader` without the banner's `previous`, so on a stale refresh it
  printed *"no clear leader ... discovery only"* three lines above the banner's
  *"last reliable reading"* for the SAME family; `panel_verdicts` now computes
  each horizon ONCE in `_summary_html` and both renderers are handed the same
  objects. And the 2-session label hardcoded "2-session discovery" over a real
  `leader` while printing "the export carries no session" beside "58 sessions
  behind"; the label is the horizon plus `verdict_label_suffix(verdict)`, and
  the no-session sentence renders only for a row that truly has none.
  `discovery_basis_phrase` says **"older evidence"** for a stale row, "undated"
  for one with no session, and "thin" only for the floor.
- **`latest_measured_session` means the MEASURED bar on the 2-session rows**
  (`legacy._short_horizon_measured_session`, the trade date of
  `post_marks[horizon - 1]`): entry dating made a family entered eight weeks ago
  and measured two sessions later read as 58 sessions stale on a file written
  that morning. Unknown stays EMPTY and reads as not fresh, never back-filled.
  The recent FAMILY rows still date by the entry and the constant says why -
  they carry no exit date until ST4 adds `representative_exit_date`. The
  short-horizon export's identity: `n_wins + n_losses + n_flats == samples_2d`,
  and `samples_2d + n_unmeasured == tracked_setups`.
- **A renderer that has a verdict renders the verdict** (re-check). The banner's
  short-term block was guarded on "a discovery row OR a leader" and otherwise
  printed a hardcoded "not enough 2-session samples yet", so `no_clear_leader` -
  the live state for that horizon, 12 eligible families with the top two 0.001
  of bound apart - rendered as "no samples" under a card saying "no clear
  leader". `_verdict_block_html` now renders all four states unconditionally.
  **The freshness clause is per kind** (`freshness_sentence(kind)`,
  `DATING_BASIS_BY_KIND`): the 2-session line says *measured*, the swing line
  says *entry-dated*, and an unknown kind takes the conservative reading - one
  sentence for both made whichever surface it did not describe say something
  false. `discovery_note` moved beside `discovery_basis_phrase` so the sentence
  and the gate it belongs to cannot drift.
- **The 2-session export counts its own wins** (the ask, answered 2026-09-06).
  `build_tracker_short_horizon_rows` gained additive `n_wins` / `n_losses` / `n_flats` /
  `n_unmeasured` / `outcome_kind` / `horizon_basis` / `latest_measured_session` at the end of its
  shipped header, golden-pinned from the code BEFORE the edit
  (`tests/fixtures/st2_short_horizon_golden.csv`); `win_rate_2d` is byte-identical and still
  counts an exactly-flat close as a zero flag. With a session on the row the freshness rule
  applies to that block too, so it is no longer discovery by construction.
- **Owed at integration:** ST1's `Headline.outcome_kind` wire (ST1 was tests-only when this
  built, so `outcome_kind` is a row column here).

### 2026-09-06 - ST1: each outcome gets its true meaning and its true clock (branch `claude/st1-outcome-clock-build`)

Trader, 2026-09-06: *"Name and version favorable price-direction observations separately
from simulated trade outcomes. Percent moves must not become realized R or stop-rule win
rates through wording."* ... *"Define exact exchange-session horizons from the entry
session and completed-bar data, independent of later scan membership."* ... *"Centralize
eligible-row reading."* ... *"Never use reconstructed labels to validate shipped S/A
performance."*

- **The `win` column was never a win.** `master_avwap_tier_outcomes.csv`'s `win` is
  `side_return_pct > 0` - the sign of a close-to-close percent move between two of a
  symbol's OWN scan rows - while `swing_headline.headline_from_tracker_rows` told four
  surfaces it was "the stop-at-a-level, two-closes rule". `outcome_kind` is APPENDED to
  both writers (v1 value `favorable_direction_scanrow_v1`; an absent or empty cell reads
  as v1 through `swing_evidence.outcome_kind_of`), `Headline.outcome_kind` decides the
  words, and the setups table's header is now **Family favorable %**. Every existing
  value and column is byte-identical - pinned by a golden generated from the pre-fix code.
- **v2 walks the exchange calendar** (`master_avwap_lib/session_horizon_outcomes.py`,
  new file `master_avwap_session_horizon_outcomes.csv`, no production reader). Missing
  target-session data stays unmeasured with a reason; an unarrived horizon is `pending`,
  never a loss; duplicates are counted, not swallowed.
- **One eligible-row reader** (`scripts/swing_evidence.py`). `build_bot_tier_performance_rows`
  now drops explicit `stale_horizon` rows like the two trader-facing readers - it is a
  report export and its only consumers are the Setup Tracker's Tier performance tab, the
  human-focus comparison table and the AI evidence list; no detector, score, gate or alert
  reads it. Its cells span four horizons over a 365-day lookback, so it shares the
  MISSINGNESS PREDICATE rather than a whole policy: `read_eligible_rows` and the export
  both call `swing_evidence.is_stale_horizon`, and the export applies it to its BASELINE
  observations too. Each read reconciles, and `describe(...)` states outcome kind, horizon
  in its own unit, window and coverage on the setups panel, the setup docs and the AWAY
  digest. `ai_jobs.digest._SECTION_NOTES["swing_win_rates"]` names the policy so a model
  reading the index cannot call the rate a win rate either.
- **The tier split is measured and shown.** Live file at `end=2026-09-03`: 2,642 horizon-5
  rows in that 20-session window, **all `derived_from_bucket`, none `assigned`** (the 341
  assigned rows are horizon 1 from 2026-09-02/03), 55 dropped stale, 2,587 eligible. The
  window rolls, so the triple is only meaningful with its `end` beside it - on the default
  window (2026-09-06) the same file reads 2,462 / 0 / 17,096.
- **Deviation from the packet, deliberately:** v1's `sessions_spanned` / `stale_horizon`
  keep the BUSINESS-DAY basis. `horizon_drift` gained `calendar=` and says "exchange
  sessions" when given one, but passing it inside the v1 export would restate 19,558
  historical rows, which the trader's prompt explicitly does not approve; the golden
  proves the file unchanged. v2 needs no drift call - its span is the horizon.
- Live gate **#75** owed at the next persisted tracker write (Tuesday 2026-09-08).


### 2026-09-06 - The digest spot-audit, two stale packs rebuilt, and three scoring questions decided (lead, on the trader's delegation)

Trader: *"go ahead and do this yourself"* over the three actions the 2026-09-05 handoff left
them - the Q4 spot-audit, and the two scoring questions plus M3's third population.

- **The spot-audit was done against the raw stores, and it found two stale packs.** Finals
  in window, `close_r` / `mfe_r` / `mae_r` n and means, distinct symbols, the exclusion
  accounting and the review-event counts by action were re-derived from
  `intraday_bounce_outcomes.csv` and `alert_review_events` with pandas only (no
  `scripts/` import). 2026-08-27 and 2026-09-04 verify exactly, 2026-09-03 within four
  late finals. **2026-08-28 and 2026-09-02 did not verify**: both packs read `in_window`
  0 while the store holds 78 and 447 finals for those sessions - the digest ran at ~05:00
  / ~06:40 before the after-close sweep of the frozen desk (the 2026-09-02/03 GIL-hog
  days) wrote the day, so the pack was faithful to the moment and wrong about the
  session. The nightly summary for 2026-09-02 was told `n=0`. The pack's `median_dwell_ms`
  is the upper-median element, not the two-middle average, which is why an even-n day
  reads differently from pandas; not a defect.
- **`python -m ai_jobs.digest rebuild --pack <date> ...` (from `scripts/`) is the repair.**
  It calls the digest's own `run_daily_digest(narrate=False)` for the day, which writes a
  SUPERSEDING sibling (D6: the early pack is never edited) and refreshes
  `entry_index.json`; no model is called and, like `approve-audit`, no nightly job may
  reach it (`tests/test_q4_overnight_gates.py`, three tests, two red before the command
  existed). Run on the live store: `2026-08-28.1.json` (78 in window, 1 usable) and
  `2026-09-02.1.json` (447 in window; 168 usable + 251 annotation + 5 information + 12
  below floor + 11 unresolved = 447; 165 `close_r`, mean -0.0858, win 46%).
- **`digest_audit_approval.json` is written** over 2026-08-27, 2026-09-02 (its rebuilt
  sibling), 2026-09-03 and 2026-09-04, `approved_by: trader`, the note naming who audited
  and how; `python -m ai_jobs.digest gate` prints `gate_met: true` (10 consecutive clean
  sessions and the audit). **`journal_enrichment` runs on the next session night.**
- **Decision (a): swept-measured trades stay OUT of the eod-hold tier cells.** The
  eod-hold cell is the record of ONE exit policy; a swept trade was measured under
  `stop_exit` / `last_measured`; and 3,620 of the 3,657 swept-measured finals on the live
  file (99%) sit in 2026-08/09 - the freeze window - so folding them in would move the
  champion's numbers on a sample of days the desk was down. They remain readable under
  their own policy tables (`sweep_exit_policy_rows`), never blended. Pinned by a golden
  characterization in `tests/test_setup_scoreboard.py` (a swept row absent from the
  eod-hold family cell, present under its policy, `policy_measured` split
  `{eod_hold: 3, stop_exit: 1, last_measured: 2}`). No scoring code touched.
- **Decision (b): `EXPIRED_UNMEASURED` records stay IN the champion's scoring population.**
  Zero live records carry the stamp today (the first stamping is the next close slot's
  write - Tuesday 2026-09-08, Monday being Labor Day - and gate #72 expects ~52); the trader-facing exports already exclude and label them; and
  excluding them from the SCORING population would make the champion's ranking depend on
  the tracker's own replay staleness - a stale week silently re-ranking setup types. The
  existing fixture in `tests/test_m3_tracker_keeps_up.py` (flipping a record to expired
  moves nothing the scorer sees) is the pin. No code.
- **Decision (c): the 28 "neither open nor closed" setups are `UNTRADEABLE`.** All 486
  baseline scenarios on the 28 records read `status == UNTRADEABLE` (`legacy.py`: risk
  per share under the tracker's floor or zero shares - no position was ever sized), and
  the `tradeable` filter in `build_tracker_stats_rows` already keeps them out of every
  n, numerator and denominator. They are evidence about the setup's SHAPE (a stop too
  tight for the standardized risk), never about win or loss, and they are correctly not
  expired: M3's third population is named and closed. The other 13 are
  `no_baseline_scenarios` and expire as designed.
- Not run last night: the ledger's newest rows are Saturday 04:00 skips (*"2026-09-05 is
  a weekend"*); the Saturday 22:00 slot wrote nothing, as on the previous weekend.

## Revision history

Entries from **2026-09-05 back to 2026-09-03** moved to
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
