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

Built 2026-09-26 on `claude/p8b-phaseB-2026-09-26` (not merged; needs the trader's
word): R1, B0, B1b, B3, B8, B9, B11, B12, B13, P4b, A6 steps 2-3, S1-S6, S8, S10b, S10c,
S11. Left here: items that need a live day, a trader decision or an ask-first yes.

- **B10 Young GC** (snappiness): 150 ms/min of young sweeps; skip a sweep when the gen-0
  count is small or lengthen the tick, only with `desk_perf_report` before/after.

- **A4b One RVOL**: golden fixture from today's outputs of `rvol.py`,
  `intraday_rvol_service.py`, `movers_scan.py` first; unify only where the numbers are
  identical, otherwise one module with the variant named.
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

**Priority read, 2026-09-26 (what works right now; re-read after 10 more sessions):**
- Swing, tape-relative at 5 sessions, last 10 scan sessions: SHORT families beat SPY 75%
  (+2.4% excess), LONG 38% (+0.4%). Best: `previous_avwape_bounce` SHORT 87% (n=63),
  `avwap_retest_followthrough` SHORT 76% (+3.0%), `favorite_zone_watch` SHORT 76% (+3.2%),
  `avwap_band_bounce` SHORT 76% (+2.6%), `mid_earnings_above_2nd_stdev` SHORT 74% (n=584).
  The favourite zone works on the SHORT side only: LOWER_1 -> AVWAPE short 73% / +2.3%;
  AVWAPE -> UPPER_1 long 42% / -0.3%. The scan's S-tier longs beat SPY 38% (n=558); B-tier
  shorts 75% (n=2,458). Only long family near even: `top_pattern_tracking` 46-50%, flat.
- Day trades, last 10 sessions (bracket win / EV of +1R-or-60-min / fires per day):
  `regime_pause_rs` long 65% / +0.20R / 17; `dynamic_vwap_lower_band` short 52% / +0.13R
  / 12; `eod_vwap` long 58% / +0.11R / 13; `lrsi_cross_50` short 55% / +0.06R / 75;
  `vwap_lower_band` short 52% / +0.05R / 9. Dead weight: `h1_blue_after_red` long (S10a),
  `regime_pause_rw` short 44% / -0.17R, `h1_green_to_yellow` short, `lrsi_cross_20` both
  sides (0.00R on 170 + 73 fires a day). Longs recovered to 52% in these 10 sessions
  (tape mix balanced); Aug-Sep as a whole they were 44%.
- So: short the favourite zone and the retest-followthrough shorts; take longs only from
  `top_pattern_tracking` swings and `regime_pause_rs` / `eod_vwap` day trades; exit day
  trades at +1R or 60 min (F14); ignore LRSI-20 and the H1 fades until S9 re-weights them.

**Packets (order):**
- **S7 Shadow engines for the missing setups** (intraday ->8; `m5_signal_engines.py`
  is shadow by design, so no ask-first until graduation): (a) prev-day high/low
  break-and-hold with RVOL >= 1.5 after 10:00 ET (the P8b adoption gate already trusts
  PDH/PDL + VWAP); (b) VWAP reclaim after a first-30 flush, longs only in
  `bullish_strong`; (c) M5 compression break; (d) intraday trendline break from pivots.
  Events go to the sidecar and are measured by the S3 bracket; graduation only through
  the `docs/SETUPS_TEST.md` ladder.
- **S9 Tier composite redesign** (grades/noise; ASK-FIRST `bounce_bot_lib/learning.py`,
  golden fixtures, trader's quoted yes): weight segments by information (distance from
  zero x confidence) or by the four most specific dimensions (bounce_type,
  bounce_combo, setup_family, time_bucket) instead of by n; cap the weight of
  `internals_breadth`, `rrs_*_alignment` and `market_environment`. Propose, with the
  F2 numbers, before touching anything. Ties into P14.
- **S10a Kill `h1_blue_after_red`** (noise; ASK-FIRST `bounce_bot_lib/legacy.py`). Held
  2026-09-26: the builder found the type already never alerts (`H1_ALERTS_RETIRED = True`
  since 07-17; H1 colour signals are logged learning-only, so F3's n=994 are learning
  rows). It has no default flag and no saved setting. "Kill" can only mean "stop
  recording it": one new constant + one skip in `check_h1_color_setups`, which is wider
  than the trader's "Sure kill it". Trader: stop recording it (yes/no)?

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
