# TODO

The next work, in order. Delete an item when it's done; don't archive it. Only the
trader adds items. An idea in `WISHLIST.md` becomes work only when the trader moves it
here.

## Plan to 8/10, round 2 (trader's brief, 2026-09-25 evening)

Scores after the round-1 merge (`7d160b4b`): intraday 7, grades/points 6, permutations 6,
recap/journal 6, night AI 7, plumbing 6, market read 7, theta 4, alert noise 6, per-trade
inputs 5, safety 9. Two mandates: every packet leaves the code simpler, and the desk
stays snappy (numbers before and after, nothing new on the Qt thread). Phases merge on a
stacked branch `claude/p8b-phase<X>-<date>` after reviewer GO, full suite and frozen selftest.

**Baseline, measured 2026-09-25 (market hours 06:30-13:05 PT, live logs, read-only):**
GUI blocked 09-23 1771 s / 2088 stalls / p90 2.2 s / 426 over 1 s; 09-24 1305 s / 3276 /
743 ms / 217; 09-25 (part day) 494 s / 1045 / 830 ms / 84. Full GC on the Qt thread: 1
sweep/min, 306-375 ms/min, max single 842 ms; young 28/min, 156-173 ms/min, single max
293 ms (GC is 35-50% of all blocked time). Qt-thread core fraction mean 0.06-0.08. RSS mean
1.7 GB, max 3.5 GB. Boot import of `ui.app` 1.14-1.18 s (legacy lazy, 1263 modules).
46 `QTimer(` + 17 `singleShot` sites. Not yet measurable: Movers tick wall time (no
timing logged, chase never fired), Alert Center `add_alert` at 1,000, Working-lately
build, M5 redraw (bench ops in A1). Quality: 13 source files over 3k lines
(`legacy.py` 35.9k and bounce `legacy.py` 14.7k ask-first; `alert_center_panel.py` 10.1k,
346 methods in one class, 87 test files); 14 `except: pass` (12 in ask-first
`bounce_bot_lib/legacy.py`); R in CAD in 3 night readers; 3 RVOL implementations; 2
Credential Manager paths; known red: 3 in `test_tj17d_chosen_change.py`, st6
`empty_snapshot`, the FlowLayout order hazard. Goals: journal stops 23/219 trades (all
backfill; Sept 13/27), tags confirmed 3 / needs_review 139 / provisional 58, MENTOR_ASKED
12; alerts shown per day 72-144 (63-114 names), acted on (focus/arm/like) 20-51;
Dip outcomes 59 graded, hit rate 52.5% (long 30.8%, n=13); Pop/Rip/chase logs empty until
Monday; `journal_import` no ok row on 7 of the last 14 nights; Ollama first token 16-18 s;
theta 6,462 picks, 0 measured, 566 pending; permutation report 1 (2 populations, 37 keys).
Gates #257-#263: none judged yet (first proof tonight and Monday).

### Phase A - measure and un-stall (live on `main` since `81272d42`, 2026-09-25 20:45 PT)

- **A1 Perf infra** (all goals): `desk_perf_report.py --day` from the stall and gauge
  logs; bench ops `alert_center.add_alert[1000]`, `working_lately`, `m5_chart`,
  `movers_board`; Movers tick wall time in status (warn over 10 s).
- **A2 Qt-thread pauses** (snappiness): freeze survivors after every full sweep, unfreeze
  and collect every 30th; 30 s stat TTL on the earnings map; 1 s stat TTL on local
  settings with in-process invalidation. Target: full-GC ms/min under 100.
- **A5 Known red + creds** (quality): fix or retire the 4 red tests with the reason; make
  FlowLayout drop dead items; one Credential Manager path.
- **B1 Night telemetry** (night AI ->8, plumbing): tokens on every model-calling ledger
  row; digest lines "slots per goal" and "unread 14+ days"; Health rows for Ollama first
  token / story time and broker-import nights ok (gate #264).
- Also in Phase A: **A4 One R** (native `trade_r_multiple` in the three night readers,
  `r_definition` in the day record), **A6 step 1**, **B6 Noise report** (`alert_noise_report.py`,
  `best_now_outcomes.py`, Day Review Alerts line, gate #265), **B4 Missing-inputs chip**
  (gate #266), **C4a setup_age facet**. First perf read on the new code (09-25 20:50-21:52,
  idle evening desk): full GC 4.7 ms/min (was 306-375), young 13.5 ms/min (was 156-173),
  largest sweep 126 ms (was 842), 1 stall over 1 s. Market-hours proof: Monday's
  `desk_perf_report.py --day 2026-09-28 --compare 2026-09-24`.

### Phase B - simpler code, truer numbers (no live data needed)

Order for the morning session (trader 2026-09-26 00:40 PT): R1 first, then B0, then
S10a (kill `h1_blue_after_red`), S10b (earnings warning on shorts), S1 with S10c (the
second grade on top), then S2 onward, then the rest of B. Each packet names goal, files, tests and
ask-first status; a builder that finds the code disagreeing with a line here reports
it instead of forcing it. The desk is DOWN (closed 21:51 PT on the trader's word);
restart only on the trader's word.

- **R1 Day Review Show** (recap ->8; the trader: "day recap is boring and hard to parse;
  mix the fonts, a presentation mode, the local AI cooks it up each night, fun but
  informative, keep the stats"). Two builders, shared module first.
  - `day_review_show.py` (pure): schema `day_review_show_v1` = `{title <=60, slides: 6-10
    of {kind: open|tape|number|scoreboard|trade|miss|read|lesson|tomorrow|close, title
    <=48, body <=220, stat: {value <=16, label <=40} | null, source_ids: 1-4}}`. A
    deterministic **fallback deck** built from the day pack alone (report card six lines
    as a scoreboard, truth lines, Alerts line, walkaway counts, internals, SPY path) so
    Show works every day. A **verifier** that rejects WHOLE (last good file kept, ledger
    `degraded`): every `source_id` in `day_review_pack.allowed_source_ids`; every number
    in title/body/stat appears verbatim in the cited sources; tickers only from the pack;
    no mood words next to a result (RULES TJ-7; mood is report-only); one slide per kind
    except number/trade. The model never supplies a stat value: it picks the source and
    writes the caption; the desk prints the number from the pack.
  - Night slot `day_review_show` (goal coaching, `uses_model`, reserve 5 min), in
    `EXPECTED_SLOT_ORDER` and `MODEL_SLOT_PRIORITY` directly after `day_review_narration`
    (stage 2; decision 0018 boundaries unchanged; update the order pins with the reason).
    Input: the day pack + that night's verified narration. Output
    `<DAY_REVIEW_DIR>/shows/<date>.json` with model, prompt_version, inputs_hash.
  - Desk: a **Show** button beside "Review my day"; a full-window overlay, one slide at a
    time; Right/Space next, Left back, Esc close, A auto-advance 8 s; footer "3/9 - told
    by gemma3:12b - sources on hover"; kind accent stripe and one deterministic glyph per
    kind (Segoe UI Emoji). Fonts set with `QFont` in code, sized through `theme.px`:
    headline Bahnschrift SemiBold, big numbers Cascadia Mono SemiBold, prose Georgia,
    captions Segoe UI (all installed on the desk; Qt substitutes if absent). One
    `QFrame#DayReviewShow` variant block in `theme.qss`; no per-widget stylesheets. The
    show JSON rides the Day Review worker's one payload; the tape slide reuses the SPY
    session bars already in it. A deck that failed its checks shows the fallback deck
    with a "facts only" badge. Gate #267.
- **B0 Desk abort 2026-09-24** (safety): `gui_crash.log` holds a "Fatal Python error:
  Aborted" from the 09-24 desk (exec line app.py:2386); find the aborting thread, fix or
  guard, and make the crash log stamp its own time.
- **B8 Alert feed as model/view** (snappiness): `add_alert` is 7.3 ms/alert at 1,000 and
  grows with the feed (widget per row). After A6 step 2: `QAbstractListModel` + delegate;
  prove with `desk_bench alert_center.add_alert[1000]`.
- **B9 Working-lately build** (snappiness): `build_payload` 16-28 s (`read_inputs` 15 s);
  cache or incrementalise the three reads; prove with the bench op.
- **B10 Young GC** (snappiness): 150 ms/min of young sweeps; skip a sweep when the gen-0
  count is small or lengthen the tick, only with `desk_perf_report` before/after.
- **B11 Two unread slot outputs**: `daily_digest` narration and `setup_research`
  narration have no desk page. Surface each on Day Review / Weekend Prep (default) or the
  trader kills the slot.
- **B12 Day Review startup warning**: `DayReviewPanel.eventFilter` runs before
  `entry_text` exists (`8f82213a`); `getattr` guard, one test.
- **P4b Econ brief prompt** (night AI ->8): three nights running the econ model restates
  yesterday's "1 p.m. Treasury auction" and the verifier rightly rejects the whole
  summary ("a time ... is not the time of an event it cites", two attempts a night).
  In `scripts/ai_jobs/econ_brief.py`: give the model only the session's own calendar
  events, label the prior brief's prose "yesterday - do not restate", and on the retry
  quote the rejected sentence as a do-not-write example. Fixture from the 09-25 night.
- **B1b Telemetry truth** (night AI): `digest.night_telemetry_lines` printed all-zero
  "slots per goal (night of 2026-09-24)" because those rows predate the `goal` field;
  say "unknown (rows carry no goal)" / "unknown (rows carry no tokens)" instead of 0.
- **B13 Focus break-state repaint** (snappiness): `alert_center_panel._poll_focus_d1_interest`
  (~line 5374) emits `focusBreakStatesChanged` on every poll; 29 stalls / 3.6 s in one
  idle hour. Emit only when a state changed. Prove with `desk_perf_report.py`.

- **A4b One RVOL**: golden fixture from today's outputs of `rvol.py`,
  `intraday_rvol_service.py`, `movers_scan.py` first; unify only where the numbers are
  identical, otherwise one module with the variant named.
- **A6 Alert Center split, steps 2-3** (quality): step 1 is in Phase A (gates, items,
  Strength Board mixin; 10,125 -> 9,412 lines). Next: the pullback / wall / H1 / any-bounce
  clusters, which need the tests' monkeypatches re-pointed first; `add_alert[1000]`
  before and after (7.3 ms/alert baseline).
- **B3 Sunday ritual card** (recap): bulk confirm + plan review in one card; "exit early /
  held losers" per setup family.
- **B7 Theta measured** (theta ->6): why 5,896 picks are unmeasured; fix the measurement
  path; a theta outcome line on Research. Trader decides if theta stays a goal.
- **C4a weekly-options facet** (permutations): `setup_age` is in Phase A; the
  weekly-options flag needs a theta store that records `option_status` per scan date
  (the theta path runs after the scan row is written). Only if theta stays a goal.
- **B2 Flex readiness** (plumbing ->8): only if the 07:00 retry still leaves more than 1
  night in 5 without an import after five nights: statement-readiness probe and a later
  first attempt.

### Phase S - setups: what the data says and what to build (study 2026-09-25 night)

Read-only study of the live stores, 2026-08-01 to 2026-09-25. Numbers are in the
findings; every packet names its files, tests and ask-first status. Builder: start at
S1 and keep the order. The two 500 MB logs are read with
`pandas.read_csv(usecols=..., chunksize=250_000)`; never load them whole.

**Findings, day trades (M5):**
- **F1 Why never A.** The day-trade grade (`setup_grades.daytrade_cells`) scores a fixed
  1R-target / 1R-stop bracket per (bounce type, side); A needs a Wilson low bound >= 0.55
  over 10+ sessions. Since 08-01: 13,682 decided alerts, 46.3% win (Sept 50.0%, Aug
  40.4%), low bound 0.454. No cell has a low bound over 0.50; the best are
  `regime_pause_rs` long 0.539 (low 0.491, EOD +0.76R, n=427) and `lrsi_cross_50` short
  0.521 (low 0.490). Mean MFE 1.65R, 59% touch +1R, 27% reach 2R, mean EOD close +0.04R.
  A 1:1 bracket on M5 bounces does not clear 55% with confidence: "never A" is the
  ladder, not the desk failing.
- **F2 Why so many C/D tiers.** The tier (`bounce_bot_lib/learning.py`) is an n-shrunk
  weighted mean of segment production R. The biggest segments sit at zero
  (`long|h1_riding_15ema` n=3110 -0.02R, `long|h1_ema10_bounce` n=2961 -0.03R,
  `internals_breadth long|neutral` n=2849 0.0R) and, because weight grows with n, the
  broadest and least informative dimensions decide the composite. A (>= +0.15R) needs
  membership in the few positive segments: `short|neutral_chop` +0.34R (n=376),
  `long|bullish_strong` +0.14R (n=1488), `long|regime_pause_rs` +0.32R (n=300, PROVEN),
  `short|avwap_retest_followthrough` +0.41R (n=70, PROVEN). 11 PROVEN segments exist,
  1 mute.
- **F3 Volume.** 670-1,147 confirmed alerts a day; H1 + LRSI engines are 75% of the
  components (`h1_ema10_bounce` 6,731; `lrsi_cross_20` 5,338; `lrsi_cross_50` 2,609;
  `h1_blue_after_red` 2,214; `h1_green_to_yellow` 1,470). `h1_blue_after_red` long wins
  34.1% (n=994, EOD -0.41R); `h1_ema10_bounce` long 43.1% (n=4,142). ORB fired 118
  times, prev-day high/low 22, `ema_8` 3, 8-EMA grind 3, `m5_confluence` 0.
- **F4 Shorts beat longs.** 49.7% vs 44.2%; EOD +0.34R vs -0.07R. By tape:
  `short|bearish_strong` and `short|neutral_chop` 51.7%; `long|neutral_chop` 41.2%,
  `long|bearish_weak` 41.8%; `long|bullish_strong` 48.4% (+0.37R). Longs pay only in a
  strong-up tape.
- **F5 First 30 minutes.** 09:30-10:00 ET alerts win 36.3% (n=1,464, EOD -0.31R), the
  worst slice by far; 10:00-13:00 ET 47-48%; the last hour 48.5% but MFE only 1.24R.
- **F6 Score and Focus carry a little.** Floor score 10 (7,560 alerts) 44.1%; any
  score above 10: 48-50%. Focus names 48.8% vs 44.1%.
- **F7 The Saturday M5 search measures the wrong thing.** Its m5 population's win is
  "the level held 30 minutes" (78-90% "wins") with r = MFE, on only the alerts that
  matched a tracker key row (n=262 for `h1_ema10_bounce` vs 4,142 decided). Verdicts:
  101 too little data, 28 no key, 6 key found - all about holding, none about paying.
- **F8 Entry timing beats detection.** Movers Dip: 59 graded, 52.5% beat SPY, longs
  30.8% (n=13); the best-possible entry from the pullback low averages +0.58% vs -0.07%
  from the flag close.
- **F14 Day trades do not go all day** (the trader's question, 2026-09-26; 19,994 alerts
  since 08-01 with a final row). The day's best move comes early: 36% of alerts peak
  within 30 min, 49% within 60, 66% within 120, 77% within 180. By 60 min the average
  alert has shown 0.80R of its eventual 1.34R MFE. When +1R prints the trade gives back
  1.0R (median) by the close; 5% of alerts (1,055) touched +1R and closed at or below 0.
  Exit rules, EV per alert with a 1R stop: hold to the close -0.30R (the worst rule for
  EVERY family); exit at 60 min -0.11R; +1R target or 60-min time stop -0.05R; the 1:1
  bracket -0.05R; +2R target -0.11R. On the good families the +1R rules are positive
  (`lrsi_cross_50` short +0.03R, `10_candle` short +0.03R, `regime_pause_rs` long +0.04R,
  `vwap` short +0.03R) while holding them to the close loses 0.03R to 0.30R. Nothing on
  the desk shows this: grades use the 1:1 bracket, tiers blend EOD entry quality with
  a 60-minute "quick" R, and no surface says when a family's move usually peaks.

**Findings, swings (D1):**
- **F9 Short families carried the last six weeks.** Session-horizon outcomes 08-14 to
  09-24 (71k measured rows), tape-relative: SHORT families beat SPY 69-79% at 5 and 10
  sessions with +1.6% to +5% excess; LONG families 30-50% with -1% to -2.4% excess. SPY
  itself was flat over those windows. `favorite_setup` bucket: 44% beat SPY, -0.53%
  excess (n=1,018); `near_favorite_zone` 57.6%, +0.79%. One bearish window proves
  nothing structural; it proves side-by-tape matters.
- **F10 Factor leaderboard (60 d, h5, SPY-relative).** LONG `is_favorite_setup=true`
  31% win, -2.2% edge (n=443). LONG deep pullback (`pct_from_current_vwap < -10`) +10.8%
  edge (n=353). SHORT 3-14 days before earnings -3.6% to -5.7% edge (n=525); SHORT
  `mid_earnings_zone_streak_days >= 10` -5.9%.
- **F11 The swing search found nothing because it could not.** Selection window was 5
  sessions (08-14 to 08-20); the 20-session hold-out ate the rest; all 28 families
  "too little data". Its win is absolute `favorable`, not tape-relative.
- **F12 The trader's ledger agrees.** Since 08-01: 25 longs -$583, 24 shorts +$1,303;
  options +$571 on 7 trades, stock +$148 on 42.
- **F13 What we already combine.** D1 facets cover EMAs (`trend_ma_alignment`,
  `price_vs_ema21`, `ma_order`), horizontal levels (`hv_level`, `cloud_level`,
  `prev_day_range_break`, `closes_vs_level`, `level_respect`), compression (with break
  direction), weekly, earnings, HTF, strength, `setup_age`. Missing: trendlines (the
  scan computes `trendline_break_recent` / `trendline_within_alert_range` for priority
  rows, `legacy.py` ~6625 and ~21181, and never writes them to
  `d1_features_history.csv`), and every M5 structure facet (M5 has five facets: time,
  RVOL, VWAP distance, SPY state, bounce type).

**Packets (order):**
- **S1 Tell the truth on the Daytrade Tracker** (grades ->7; no ask-first):
  `setup_scoreboard.py` / `ui/panels/setup_tracker_panel.py` cells show, beside the
  bracket grade, "EOD close R" mean, "reach 2R" share and n; the grade line says "1:1
  bracket". `setup_grades.cell_line` gains the two numbers; golden tests updated with
  the reason. Read the outcome log the way `bracket_results` does.
- **S2 First-30 Show filter** (noise ->7; presentation only, like P9): `alert_show_filter.py`
  gains a "hide the first 30 minutes" switch, default on, that never hides PROVEN,
  Focus, typed names or chart-watch hits; `hidden_by_show` detail says `first30`.
  Everything is still recorded. One test per exemption.
- **S3 Bracket outcome in the permutation search** (permutations ->7; no ask-first):
  `setup_permutation_backfill.m5_rows` adds horizon `bracket_1r` (win = +1R before -1R
  by first decisive row, r = final close R) beside `held30`, built from EVERY decided
  alert (join `intraday_bounce_candidates.csv` confirmed rows to the outcome log by
  event_id, not only tracker-keyed rows); `setup_permutation_search` runs both
  horizons; the report and `setup_keys_narration` name the horizon in every line.
  Fixture test with 6 events; refuse live paths as today.
- **S4 Tape-relative swing search** (permutations ->7; no ask-first): swing `win` =
  side return beats SPY over the horizon (reuse `setup_grades.tape_result`); the search
  refuses to publish a population whose selection window has under 20 sessions and
  writes that reason into the report. Needs the April-August backfill (trader's owed
  run) to have anything to say.
- **S5 Side-by-tape line** (grades, recap; display only): Setup Tracker and Day Review
  get one line "last 20 sessions, tape-relative: longs beat SPY X% (excess Y%), shorts
  Z% (W%)", from the session-horizon outcomes on a worker. No gating.
- **S6 Trendline facet + M5 structure facets** (permutations; scan edit with NO output
  change, trader 2026-09-24; the sidecar is shadow): write `trendline_break_recent`,
  `trendline_within_alert_range` and the direction onto the scan row (P11 mechanism,
  golden parity test) and register `@facet("trendline")`. In `m5_setup_key_stamp.py`
  add pure `@m5_facet`s over the alert's cached bars: `m5_ema_stack` (8/21 order and
  price side), `m5_pdh_pdl` (above / inside / below yesterday's range),
  `m5_open_range` (vs the first-30-minute range, only after 10:00 ET),
  `m5_compression` (12-bar range vs 20-bar ATR, squeeze then expansion),
  `m5_side_vs_d1_env` (side aligned with the D1 environment). Fixture test per facet.
- **S7 Shadow engines for the missing setups** (intraday ->8; `m5_signal_engines.py`
  is shadow by design, so no ask-first until graduation): (a) prev-day high/low
  break-and-hold with RVOL >= 1.5 after 10:00 ET (the P8b adoption gate already trusts
  PDH/PDL + VWAP); (b) VWAP reclaim after a first-30 flush, longs only in
  `bullish_strong`; (c) M5 compression break; (d) intraday trendline break from pivots.
  Events go to the sidecar and are measured by the S3 bracket; graduation only through
  the `docs/SETUPS_TEST.md` ladder.
- **S8 Retest-entry study** (intraday; shadow research in `scripts/research_warehouse/`):
  measure "enter at the retest" (fill only if price returns within 0.25 ATR of the
  level within 6 bars, else no trade) against "enter at the flag close" for M5 alerts
  and Movers, from cached bars; report the two EVs per family on Research.
- **S9 Tier composite redesign** (grades/noise; ASK-FIRST `bounce_bot_lib/learning.py`,
  golden fixtures, trader's quoted yes): weight segments by information (distance from
  zero x confidence) or by the four most specific dimensions (bounce_type,
  bounce_combo, setup_family, time_bucket) instead of by n; cap the weight of
  `internals_breadth`, `rrs_*_alignment` and `market_environment`. Propose, with the
  F2 numbers, before touching anything. Ties into P14.
- **S10a Kill `h1_blue_after_red`** (noise ->7; ASK-FIRST `bounce_bot_lib/legacy.py`,
  satisfied for this exact change by the trader's words 2026-09-26: "Sure kill it", on
  the finding F3: 34.1% win, n=994, EOD -0.41R). Turn the bounce type off where the
  live desk reads it: `BOUNCE_TYPE_DEFAULTS` / its `CHECK_BOUNCE_*` flag in
  `bounce_bot_lib/legacy.py`, AND any persisted M5 settings that override the defaults
  (find the settings key the desk saves; set it too, or the kill never reaches the
  desk). Golden fixtures FIRST: run the detector goldens on the fixture tapes with the
  type on and off; every other alert byte-identical, only `h1_blue_after_red` rows
  gone. The learning state keeps the segment's history; the Daytrade Tracker shows the
  family as "off since 2026-09-26". One CHANGELOG line; GATES "#268: on the next
  session no `h1_blue_after_red` alert is confirmed, the other components' counts are
  in line with the prior session [lead]".
- **S10b Earnings warning on short setups** (grades/noise; annotate only, no ask-first;
  the trader 2026-09-26: "leave those shorts, just warn me whenever those charts pop
  up", on F10: shorts 3-14 days before earnings lost 3.6-5.7% vs SPY). Pure
  `earnings_warning.short_into_earnings(days_to_next_earnings, side) -> str`: for a
  SHORT within 0-14 days of the next earnings date, "earnings in N d - shorts 3-14 d
  before earnings: X% vs SPY (60 d)" with X read from the live
  `master_avwap_scan_factor_leaderboard.csv` row (`days_to_next_earnings`, SHORT, h5;
  plain wording when the row is absent). Shown wherever a short chart pops up: the
  Setup Tracker row (badge + tooltip through the existing delegate, SHORT rows only),
  the M5 alert row's grade line and the chart review header for SHORT alerts (days from
  the earnings dates cache `chart_snapshot.earnings_anchor_dates` uses; unknown date =
  no warning, never a guess), the Movers Rip-weak row. It never hides, sorts or mutes.
  Tests: the boundary days, unknown date, long side silent, the leaderboard fallback.
- **S10c Second grade on top** (grades ->7; scoring code, the trader 2026-09-26: "we can
  do this on top of what we already do"): keep the 1:1 bracket grade exactly as it is
  (badges, Show filter and sorting keep reading it). Add, per day-trade cell, a 2R
  grade from the same ladder on "+2R before -1R" (`target_2r_hit` is cumulative like
  `target_1r_hit`; first decisive row decides; avg R = 3p - 1) and the EOD close R mean
  and median. `setup_grades.daytrade_cells` returns the extra fields; `cell_line` reads
  "1:1 C - 2R D - EOD +0.04R - n 427"; the Daytrade Tracker shows the three. New golden
  fixtures for the new fields; the existing grade goldens must not change. This is S1's
  second half; build them together.
- **S11 Exit-window truth** (intraday ->8, recap; display only, no ask-first; from F14).
  A deterministic night slot `exit_windows` (goal trade_identification, no model) reads
  the outcome log in chunks and writes `exit_windows.json` beside the setup grades: per
  (bounce type, side) the share of alerts peaking within 30/60/120 min, mean MFE by 60
  and 120 min, the give-back when +1R printed, and the EV of five rules with a 1R stop
  (hold to close, exit at 60 min, +1R or 60 min, +1R or 120 min, the 1:1 bracket). The
  Daytrade Tracker gets an "Exit by" column ("peak <= 60 min 49%; +1R/60m -0.05R vs
  hold -0.30R"); the M5 alert row and the chart review header get one line ("this
  family usually peaks inside 60 min; +1R has beaten holding by 0.25R"); the Trade
  Mentor's exit questions quote it. Facts only, never a rule the desk enforces. Fixture
  test with 8 events; a test that the desk only formats.

### Phase C - needs live days (trigger named)

- Gates #257/#258: tonight's ledger and Monday morning. #259-#263: Monday's session.
- **C1 Intraday grading** (intraday ->8): after 5 sessions (2026-10-02) grade Pop, Dip,
  Rip and the options chase; "did the phone line pay" line on Day Review (builds now,
  shows "no data yet"); rip/dip hysteresis only if the board flaps live.
- **P13 Points challenger** (grades ->8, after 09-30): SP4 frozen before looking, 20
  entry sessions plus 5, limits fixed in advance; then the bounded "my trades" nudge at
  10 confirmed trades per family.
- **C3 Second Saturday report** (2026-10-03): first verdicts; promotion to a named
  sub-family (P1-4 4e, ask-first, one at a time).
- **P14 Retire PROVEN, raise the M5 bar** (noise ->8; ask-first, golden fixtures):
  `[X-TIER] PROVEN` from `bounce_bot_lib/learning.py` replaced by the grade.
- **HYG daily bars** (plumbing; scan-side, ask first). TLT/USO the same.
- **Goal 10 live proof**: stops on 80% of new trades within a week; tag confirmation rate.

## Decisions the trader owes (each one moves a score)

- IB option data on the account: theta (43 of 6,462 picks ever priced) and the options
  chase both wait on it. Keep theta as a goal, or drop it.
- TWS logged in during sessions, or the scanners and the M5 feed have nothing to read.
- P14 (retire PROVEN, raise the M5 bar) and any permutation promotion: quoted yes needed.
- The 142 owed gates in `docs/GATES.md`: one batch pass/drop.
- Dead-script review: yes or no.
- B11: surface the two unread slot outputs, or kill the slots.
- Risk per trade ($) in Settings, `trading_plan.md`, the 139 setup tags waiting.

## Carried over

- Read the owed live gates in `docs/GATES.md`, newest first (#257 next).
- TJ-8 cleanup after gates #145-#150: delete `market_journal_panel.py`,
  `daily_recap_panel.py`, the page class in `away_recap_panel.py`, dead tests, the
  four-pane capture reader. No behaviour change.
- Small mentor follow-ups: TJ-12F (Focus-add and armed lanes in `trade_origin`), TJ-14C
  (`quick_like_followup` reader), TJ-13B (Sunday setup tags + week-ahead note), TJ-6M
  (`mentor_answer_mix_rate`), TJ-14B (`grader_gap` question), TJ-7 verifier (never pair
  a mood with a result), TJ-9E (exit fields in the day pack; `CLOSED_PARTIAL` spelling
  in 4 ask-first places; `ai_summary._journal_source` exit words).
- Housekeeping: dead-script review needs the trader's yes; 142 owed gates need a batch
  pass/drop; the live `permutation_report.json` names a scratch parquet as its source
  (the Saturday job must write it from the live warehouse).
