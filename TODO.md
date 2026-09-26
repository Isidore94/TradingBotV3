# TODO

The next work, in order. Delete an item when it's done; don't archive it. Only the
trader adds items. An idea in `WISHLIST.md` becomes work only when the trader moves it
here.

## Plan to 8/10 on every goal (trader's go, 2026-09-25)

Scores on 2026-09-25: intraday 5, grades/points 4, permutations 4, recap/journal 4,
night AI 5, plumbing 5, market read 7, theta 3, alert noise 3, per-trade inputs 2,
safety 9. Each packet names the goal it lifts. Build in phase order; a phase merges
after reviewer GO, full suite green and the frozen selftest.

### Phase 1 - inputs and plumbing (built, `claude/p8-phase1-2026-09-25`, waits merge)

- **P1 Stop backfill** (inputs 2->6): dry-run CLI fills the planned stop on old trades
  from the matched M5 alert or the D1 plan, marked as a backfill; never overwrites.
- **P2 Mentor stop first + Sunday bulk confirm** (inputs ->8): the stop is the first
  question on every trade; one screen confirms the 139 waiting setup tags.
- **P3 Broker import morning retry** (plumbing ->7): a failed night import runs again
  at 07:00 PT; Health shows unresolved self-heal rows and mismatches.
- **P4 Night model fixes** (night AI ->8): day story stops timing out (smaller pack or
  context); econ verifier accepts "1 p.m." for a 13:00 event; plan_review and
  improvement_ideas ahead of observation_tags in the budget order; delete the unread
  note_vocabulary_audit slot; theta picks that can never get a quote are "dead"; every
  slot names its goal and a test checks it. (Ideas id enum and enrichment grounding
  landed in `af1cc04c`.)

### Phase 2 - tell the truth (built, `claude/p8-phase2-2026-09-25`, waits merge)

- **P5 Journal truth lines** (recap ->8): plain sentences on Day Review, Week Review and
  the journal cards: stock vs options, long vs short, per-setup expectancy on confirmed
  trades; each trade joins to the bot's grade of its setup; week coach floor 5 with a
  "thin" badge; "exit early / held losers" scoreboard from MFE and MAE.
- **P6 Grades vs the tape** (grades ->6): win measured against SPY over the same 5
  sessions; cumulative R beside each badge; no PROVEN while the R line falls. Update the
  grade golden tests.

### Phase 3 - intraday (built, `claude/p8-phase3-2026-09-25`, waits merge)

- **P7 Rip-weak list + Pop outcomes** (intraday ->6): on an up day rank names by lag vs
  SPY from the last SPY swing low; Pop gets an outcome log like Dip. Display only.
- **P8 Phone and sound** (intraday ->7): top 3 new Pop and Dip-strong names to the phone
  in AWAY/EVENING and a desk sound in DESK, once per bar. P8b (trader 2026-09-25): the
  same names feed the M5 watch through the adoption gate only when a long is above the
  previous day's high and VWAP, a short below the previous day's low and VWAP.
- **P9 Alert Center default filter** (noise ->7): default view grade B and above or
  Best-right-now only; everything still recorded.
- **P10 Options chase helper** (intraday ->8, theta ->7): for a Pop name with RVOL 2+,
  read the IB option chain the theta scan uses; show the weekly strike near 0.25 delta,
  spread and IV; log it for grading. Needs IB option data on the account.

### Phase 4 - learning (P11, P12 and P8b built, `claude/p8-phase4-2026-09-25`, waits merge)

- **P11 More facets** (permutations ->6): M5-native facets in the sidecar (time of day,
  RVOL bucket, VWAP distance, SPY state, bounce type); the 12 missing D1 facets written
  on the scan row (scan edits allowed with no output change, trader 2026-09-24).
- **P12 Weak-variant tag** (permutations ->8): a key that fails hold-out two Saturdays
  running gets a visible tag and sorts lower. Rank and annotate only.
- **P13 Points challenger** (grades ->8, after 09-30): SP4 as specified below; then a
  bounded "my trades" nudge once a family has 10 confirmed trades.
  SP4: freeze the challenger before looking at results; 20 new entry sessions plus a
  5-session wait; success, downside and rollback limits fixed in advance.
- **P14 Retire the old PROVEN stamp, raise the M5 bar** (noise ->8; ask-first, golden
  fixtures): `[X-TIER] PROVEN` from `bounce_bot_lib/learning.py` replaced by the grade.

## Carried over

- After the P8 merges: R is still CAD / native-risk in three night-side readers
  (`recap_rule_loop.trade_r`, `preference_trade_outcomes._canonical_r`,
  `day_session_record`); moving them changes pack hashes, decide with the trader.
- `tests/test_p8_p8b_movers_adopt.py` followed by `tests/test_qt_compact_desk.py` in one
  process can hit a FlowLayout "QWidgetItem already deleted" (pre-existing wrapper hazard
  in `ui/widgets/flow_layout.py`); passes under the normal suite.

- Read the owed live gates in `docs/GATES.md`, newest first (#257 next).
- TJ-8 cleanup after gates #145-#150: delete `market_journal_panel.py`,
  `daily_recap_panel.py`, the page class in `away_recap_panel.py`, dead tests, the
  four-pane capture reader. No behaviour change.
- TLT/USO have no daily bars: scan-side change, ask first.
- Small mentor follow-ups: TJ-12F (Focus-add and armed lanes in `trade_origin`), TJ-14C
  (`quick_like_followup` reader), TJ-13B (Sunday setup tags + week-ahead note), TJ-6M
  (`mentor_answer_mix_rate`), TJ-14B (`grader_gap` question), TJ-7 verifier (never pair
  a mood with a result), TJ-9E (exit fields in the day pack; `CLOSED_PARTIAL` spelling
  in 4 ask-first places; `ai_summary._journal_source` exit words).
- Housekeeping: split `alert_center_panel.py` (8k lines); dead-script review needs the
  trader's yes; 142 owed gates need a batch pass/drop.
- Known red on `main`: 3 tests in `test_tj17d_chosen_change.py` and
  `test_st6 ... empty_snapshot`.
