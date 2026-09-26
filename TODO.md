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

### Phase A - measure and un-stall (no ask-first; in flight 2026-09-25)

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

### Phase B - simpler code, truer numbers (no live data needed)

- **A4 One R** (recap ->8): `recap_rule_loop.trade_r`, `preference_trade_outcomes._canonical_r`,
  `day_session_record` read `journal_analytics.trade_r_multiple`; pack-hash migration so
  old packs stay readable. Trader's word in the 09-25 brief.
- **A4b One RVOL**: golden fixture from today's outputs of `rvol.py`,
  `intraday_rvol_service.py`, `movers_scan.py` first; unify only where the numbers are
  identical, otherwise one module with the variant named.
- **A6 Alert Center split** (quality): move the module-level gate/tier helpers and the D1,
  digest/repetition, chart-watch and review-queue blocks into `ui/panels/alert_center/`,
  behaviour-preserving, 3 steps each green; `add_alert[1000]` before and after.
- **B6 Noise report** (noise ->8): `alert_noise_report.py --summary` from the review
  events (shown / hidden by Show / acted on / Best-right-now hit rate) and one Day Review
  line.
- **B4 Missing-inputs nag** (inputs ->8): status-bar chip "N trades missing stop or
  setup", one click opens the Mentor on the oldest.
- **B3 Sunday ritual card** (recap): bulk confirm + plan review in one card; "exit early /
  held losers" per setup family.
- **B7 Theta measured** (theta ->6): why 5,896 picks are unmeasured; fix the measurement
  path; a theta outcome line on Research. Trader decides if theta stays a goal.
- **C4a D1 facets the scan can write** (permutations): setup age, weekly-options flag
  (scan edit with no output change, trader 2026-09-24).
- **B2 Flex readiness** (plumbing ->8): only if the 07:00 retry still leaves more than 1
  night in 5 without an import after five nights: statement-readiness probe and a later
  first attempt.

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

## Carried over

- `tests/test_p8_p8b_movers_adopt.py` then `tests/test_qt_compact_desk.py` in one process
  can hit a FlowLayout "QWidgetItem already deleted" (A5 fixes).
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
