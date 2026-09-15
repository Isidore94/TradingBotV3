# Pullback alert, trendline break and compression - packets PCT-1..3

Document role: **active spec and resume brief** for `plan.md` Phase 0.29 (trader 2026-09-15).
Written so that a fresh session (any model) can pick the work up mid-way: the trader's words, the
decisions already taken, the verified premises with `file:line`, the three packets in build order,
the tests each needs first, the live gates, and the resume checklist at the end. Status of every
packet is in the table in section 1 and is updated by the lead on every handoff.

Branch: **`claude/pullback-compression-2026-09-15`** off `origin/claude/desk-combined-2026-09-14`
(the desk checkout's branch; another session had uncommitted "Show vetoed" work in that checkout
on 2026-09-15 - never commit from the desk checkout, work in worktrees). Packet branches:
`claude/pct-1-pullback`, `claude/pct-2-trendline`, `claude/pct-3-compression`, each off this
branch. Integration into the combined branch happens in a scratch worktree, never the desk
checkout, and never a merge to `main` without the trader's word.

## 1. Status

| Packet | What | Status |
|---|---|---|
| PCT-1 | Pullback alert (M15 150-SMA / M30 75-SMA reclaim + LRSI, retest; the H1 retester folded in) + the three new claim names | PLANNED 2026-09-15 - tester next |
| PCT-3 | Compression: copy the measure through, chip in the setups table, calibration CLI against the `compressed` vetoes, `compression_break` family tag | RED TESTS COMMITTED 2026-09-15 (`claude/pct-3-compression` 8483a324, 40 failing / 3 pins) - builder spawned |
| PCT-2 | Trendline break: `trendline_break` family tag + D1 event kind + feed alert | PLANNED - after PCT-1 and PCT-3 land (shares their files) |

Trader's order was 1 pullback, 2 compression breaks and trendline breaks, 3 compression measure.
The compression-break tag needs the measure, so it rides PCT-3; PCT-2 is last only because it
edits files PCT-1 and PCT-3 both touch (`chart_watch.py`, `alert_center_panel.py`, `legacy.py`).

## 2. The trader's words (2026-09-15, chat)

> "i want to add a setup to setup tracker. it involves taking our strong stocks making new
> highs/breaking out. we then monitor them for a pullback on a M15 or M30 basis. on the M15 we
> use the 150 moving average on the M30 we use the 75. we wait for the stock to go BELOW these
> levels then break back up. we can test entries on the breakup if they are accompanied with an
> LRSI reversal on the same time frame in the past 3 bars or so. we can also test on an M30 basis
> a breakup then waiting for an M30 or M15 LRSI reversal while staying above teh relevant SMA for
> an entry. we can also test for retests of the SMA. overall we are trying to gauge pullbackls
> using shorter time frames than the D1 to find optimal entries. in the same vein as H1 retester
> we should rename it to Pullback alert and include any of these phenomena in the alert pattern.
> if this makes the program too cumbersome then do not bother with this but do include it as a
> setup type in the like and claim section. additionally we need to add setups for compression
> breaks and trendlike breaks. we already make trendlines via the chart so I would like a way to
> have these alerted as setups. additionally we need a way to measure for compression as the
> most COMMON veto I have is a compression veto. we need to fine tune this so we stop getting so
> many compressed picks. this will also help us find compression breakouts."

Answers to the lead's four questions (same day):

1. Which names the Pullback alert watches: **"Chart arm + my picks"** - the arm button under the
   chart, AND every claimed D1 pick and every swing Focus pick is armed automatically.
2. LRSI reversal: **"Cross up through 80. Ideally it was below 50 2-4 bars previously too."**
3. Compression: **"Measure + chip first"** - one measure, a chip, a check against the vetoes, then
   the penalty decision from that evidence. No hiding.
4. Ask-first: **"Yes, all of them"** - `legacy.py`, `setup_tagging.py`, `setup_docs.py`,
   `chart_watch.py`, `alert_center_panel.py`, `capture_rail.py`, `setup_points.py` may be edited
   for THIS work, additively. No existing detector rule or score changes; tests pin today's numbers.

This is WISHLIST 10C steps 2 and 3 ("H4 / LRSI pullback options", "trendline break") moved into
the build by the trader's words above, narrowed to M15/M30 (no H4) and to a direct break (the
break-then-retest of a trendline stays in WISHLIST).

## 3. Lead decisions (the trader may overrule any of these)

- **LRSI is the champion's `efficiency_lrsi` (0..100), not the unused Laguerre module.** The M5
  champion computes it through `m5_signal_engines.latest_lrsi_cross` -> `compute_efficiency_lrsi`
  with `CROSS_LEVELS (20, 50)` untouched. The pullback rule passes its OWN level (80) to the same
  pure function; shorts are handled the champion's way (negated closes), so "cross up through 80"
  is symmetric. The "was below 50 in the 2-4 bars before" clause is a labelled QUALITY flag
  (`lrsi_from_below_50: true/false`) carried on the fire, not a gate - the trader said "ideally".
- **One button, one kind, named triggers.** `WATCH_KINDS` gains `pullback` labelled
  **"Pullback alert"** and the `h1_ema_bounce` button is retired (its persisted watches load as a
  `pullback` watch whose only trigger is `h1_ema15_bounce`; nothing is lost). A `pullback` watch
  carries `triggers` (a tuple of trigger names) and `fired` (trigger -> fire time). Every fire
  names its trigger and timeframe. The H1 rule sheet `h1_ema_bounce_v1` is untouched.
- **Triggers (rule sheet `pullback_sma_reclaim_v1`, one pure module, symmetric sides):**
  - `sma_reclaim_lrsi` (M15 with SMA-150 and M30 with SMA-75, each its own episode): the stock has
    a completed close BELOW the SMA (state, may pre-date the arm), then a completed close ABOVE it
    that ends after `armed_at`, AND an LRSI reversal (cross up through 80) on the same timeframe on
    the reclaim bar or the two before it. Fires once per timeframe per episode.
  - `reclaim_then_lrsi` (M30 only): the M30 reclaim above SMA-75 happened, no completed M30 close
    has gone back below it since, and an LRSI reversal prints on M30 OR M15 after the reclaim.
    Fires once per episode; a completed M30 close back below the SMA ends the episode.
  - `sma_retest` (M15 and M30): after a reclaim, a completed bar's low tags the SMA (within
    0.25 ATR-14 of that timeframe, or through it) and closes above it. Fires once per timeframe
    per episode.
  - A new episode begins when a completed close goes back below the SMA; an episode's triggers
    may fire again in a new episode. A watch ends at expiry (10 trading days, existing policy),
    on disarm, or when the trader's claim / swing Focus pick that auto-armed it is dropped.
  - Bars: regular session only, session-aligned, completed by `completed_bars` (bar_start +
    minutes <= now). Warm-up: 160 M15 bars / 85 M30 bars (SMA + LRSI's 13); fewer is
    `not measured (N of 160 M15 bars)`. yfinance `15m`/`30m` for a `1mo` period, one cache instance
    per interval on the H1 cache's own worker pattern, zero IB traffic. Stale after 24 h.
- **Auto-arm is DESK-mode-agnostic and pushes in every mode**, the same door as the H1 retester
  (`notify_armed_watch`): these are the trader's own picks, the recorded exception class of
  Research/Focus price alerts. An auto-armed watch shows `auto: claimed pick` / `auto: swing
  Focus` in its source text; disarming it by hand is remembered (`declined`) so the poll does not
  re-arm it while that claim or pick lives. Recorded in `AUTO_MODES_AND_QUIET_HOURS_PLAN.md`.
- **Three claim names** (in `setup_docs.SETUP_DOCS`, new group `"Entry timing and breaks"`, all
  three in `EXTRA_CLAIM_IDS` so the rail offers them): `pullback_sma_reclaim` "Pullback reclaim
  (M15 150 / M30 75 SMA)", `trendline_break` "Trendline break", `compression_break` "Compression
  break". A claim is graded by name through `claimed_pick_evidence._by_setup` - nothing else needed
  for the "My claims" tab.
- **Compression is measured before it is tuned.** The scan ALREADY measures anchor compression
  on every priority row and penalises the score (section 4.3) - and the trader still vetoes
  compression more than anything (189 of 598 coded vetoes, 36 % pooled with the v1 code). So the
  measure disagrees with the trader's eye. PCT-3 copies the hidden numbers through, shows a chip,
  and ships a read-only calibration CLI that compares several candidate measures against the veto
  set. The threshold / penalty change is a SEPARATE ask after the trader reads that report.
- **`compression_break` v1 is tagged from the existing rule**: the previous completed session was
  `is_compressed` and the current completed close leaves the compression range in the setup's
  direction with a bar range >= 1.0 ATR-20. Labelled `compression_break_v1`; re-tuned only after
  the calibration report, by the trader's word.
- **`trendline_break` is a TAG and an EVENT, not a new primary family**: the scan already sets
  `trendline_break_recent` (section 4.4); the tag rides it. The D1 event kind fires on a completed
  D1 close through the scan's frozen line, once per (symbol, side, break date).

## 4. Verified premises (recon 2026-09-15, file:line)

### 4.1 The H1 retester today (PCT-1 base)

- Rule: `scripts/indicators/h1_ema_bounce.py` `RULE_VERSION = "h1_ema_bounce_v1"` (:52),
  `evaluate()` (:277-403), `WARMUP_BARS` 45 (:301), `STALE_AFTER` 24 h (:303), reasons
  `bounce_confirmed / ambiguous / no_touch / invalidated / awaiting_reclaim / slope_against`.
- Kind: `scripts/chart_watch.py` `WATCH_KINDS` (:33-41) `"h1_ema_bounce": "H1 retester"`;
  `PERSISTENT_WATCH_KINDS = frozenset({H1_EMA_BOUNCE_KIND})` (:56); `ChartWatch` (:135-150:
  `symbol, kind, armed_at, side, baseline, source_text, watch_id, reason`); `watch_reason`
  (:257-266); `h1_bars_for_watch` (:317-340); `_fence_pre_arm` (:454-484); `evaluate_h1_bars`
  (:487-523); `h1_bounce_message` (:526-541); store `save_chart_watches` / `load_chart_watches`
  (:807-853).
- History: `scripts/h1_history.py` `H1HistoryCache(downloader=None, period="1mo", interval="60m")`
  (:258-267), in memory only, one fetch per completed bucket per symbol (`request`, :307-337), on
  its own worker thread; `H1_MINUTES`/`H1_SPAN` are module constants baked into the bucket math
  (:59-66, :140-198) - a 15/30-minute cache needs `interval_minutes` threaded through that math,
  one instance per interval.
- Panel: `alert_center_panel.py` `_h1_history_cache` (:4802-4822), `_h1_bars_for_watch`
  (:4824-4855), `_poll_h1_bounce_watches` (:5798-5939) called from `_poll_d1_event_watches`
  (:5976-5980) on `_d1_watch_timer` at 60 s (:1113-1121); `_push_armed_watch` (:5941-5974) is
  non-blocking; arm bar `ui/widgets/arm_bar.py` builds one button per `WATCH_KINDS` (:132-137),
  slot `_toggle_chart_watch` (:889); health cell `ui/widgets/armed_watch_list.py` `watch_health`
  (:42) + host `watch_note` (:115-150).
- LRSI: `scripts/indicators/efficiency_lrsi.py` `compute_efficiency_lrsi(closes, config)`
  (:151-154), `EfficiencyLrsiConfig(ema_length=9, sum_length=4)` (:90-92), 0..100, `None` during
  ~13-bar warm-up, `CROSS_LEVELS = (20.0, 50.0)` (:50, "do not change"); champion path
  `m5_signal_engines.latest_lrsi_cross` (:164-198) negates closes for shorts (:72-84).
  `scripts/indicators/laguerre_rsi.py` has zero importers - do not use it.
- Auto-arm sources: `scripts/focus_picks.py` `FocusPickStore.all_focus(category="swing")`
  (:404-408) -> `{"long": [...], "short": [...]}`; `scripts/claimed_picks.py`
  `active_claims(rows_or_path, *, as_of)` (:281-295), rows carry `symbol, side, horizon,
  claimed_setup_id, claim_at, session_date`.
- Pinned strings to update in lockstep: `tests/test_chart_watch.py:177`,
  `tests/test_ws_10c_h1_retester.py:62-63, 763-769`, `tests/test_ws_10c_h1_retester_builder.py:256,262`.
  Test files: `test_ws_10c_h1_retester.py` (26), `test_ws_10c_h1_retester_builder.py` (20),
  `test_chart_watch.py` (26), `test_rv_h1_*.py` (5 files).

### 4.2 Claim names (PCT-1)

- Rail list: `scripts/ui/widgets/capture_rail.py` `setup_list` (:368-387), `selected_setup_id`
  (:848-849), `commit_like` (:1013-1043); offer = `setup_claims.offered_setup_claims()`
  (`scripts/ui/annotations/setup_claims.py:106-127`): the `"Main swing"` group whole plus
  `EXTRA_CLAIM_IDS` (:98-103); an unknown id is skipped silently and
  `tests/test_qt_alert_capture.py::test_the_rail_offers_every_claim_the_trader_asked_for` catches it.
- Docs: `scripts/setup_docs.py` `SETUP_DOCS` entry keys `label, group, what, detection, entry,
  stop, targets, evidence` (:69-106 is the model). `setup_registry_v1.json` is regenerated by
  `build_setup_registry.py --write`, never hand-edited.
- Like row: `scripts/ui/annotations/store.py:365-383` (`like_mode`, `claimed_setup_id`);
  claimed D1 pick row `scripts/claimed_picks.py:80-94`; grading by name
  `scripts/claimed_pick_evidence.py:616-638`.

### 4.3 Compression today (PCT-3)

- `scripts/master_avwap_lib/legacy.py` `summarize_anchor_compression(price_slice, anchor_stdev,
  atr20)` (:4927-5005): `stdev_atr_ratio = anchor_stdev / atr20`, `range_atr_ratio = (max high -
  min low) / atr20`, `close_range_atr_ratio = (max close - min close) / atr20` over the bars from
  the current AVWAPE anchor date to the last trade date; `compression_score` = count of tight
  ratios (:4961-4963); `is_compressed` (:4978-4982); `compression_penalty` (:4990-4992).
  Thresholds `PRIORITY_COMPRESSION_*` (:1059-1077). `evaluate_anchor_compression` (:5008-5026) is
  called for EVERY priority row inside `_evaluate_priority_snapshot_for_date` (:26320-26740);
  `_effective_compression_penalty` (:20155-20196) relieves it on breakout rows; the score seam is
  :26989-26999 (`score -= penalty`, then `compression_flag / compression_penalty /
  compression_note` land on the row). `compression_score` and the three ratios do NOT leave the
  function today. `build_tracker_setup_record` (:6286-6290) copies the three surviving fields.
- Vetoes: `C:\TradingBotData\trader_annotations.jsonl` (`TRADER_ANNOTATIONS_FILE`,
  `project_paths.py:392`), writer `ui/annotations/store.py:334-361`; coded row keys `symbol,
  session_date, created_at, reason_code, vocab_version, timeframe, side, surface` plus
  `SCAN_CONTEXT_FIELDS` (`scan_date, tracker_setup_id, canonical_setup_id, priority_bucket, score,
  expected_r`, :130-137) when the caller supplied them. Vocabulary v3
  `ui/annotations/vocabularies/veto_reasons_v3.json:27-33` code `compressed`. **Correction (tester
  2026-09-15):** v1's `support_resistance_cluttered` does NOT pool with it in
  `veto_cohort.canonical_veto_cohort` (v2 called the swap "a NEW code, not a rename"); the
  calibration CLI names both codes itself.
  Live counts 2026-08-20..09-15: 598 coded vetoes; `compressed` 189 + v1 25 = 214 (35.8 %), all
  `timeframe D1`, LONG 97 / SHORT 92; 136 rows carry no `reason_code`.
- Setups table rows: `ui/services/data_feed.py` `load_setup_rows_from_priority_report` (:203-249)
  parses `master_avwap_priority_setups.txt` lines (`RANKED_LINE_RE` :24-32 captures symbol, side,
  expected_r, score, family, bucket; zone from `ZONE_LINE_RE`) - the report line does NOT carry
  compression fields; `master_avwap_ai_state.json` (`project_paths.py:591`) does, per symbol.
  Chip model: WS-WS `wrong side` is paint-time only - `scripts/avwape_side.py` `read_row /
  wrong_side / tooltip_text` + `ui/widgets/setup_delegate.py` `_wrong_side_read` (:161-173),
  `sizeHint` (:215-227), `paint` (:269-287), `_wrong_side_tooltip` (:146-189); no model column.
- Market-level compression label (different formula, label only): `scripts/indicators/
  d1_environment.py:17-96` (`range_atr = 10-session range / ATR14 <= 3.0`).
- Score pins: `tests/test_master_avwap_setups.py` asserts exact `score` values on synthetic
  frames (e.g. :1749, :2480, :2689); no whole-dict golden, so ADDITIVE fields do not break them.
- **A compression-break rule ALREADY EXISTS** (tester 2026-09-15): `legacy.py`
  `assess_compression_break_context` (:29086-29175) with `compression_break_today / _direction /
  _level / _distance_atr / _prior_bars / _note`, `PHASE6_COMPRESSION_BREAK_BUFFER_ATR = 0.10`,
  `PHASE6_COMPRESSION_MIN_PRIOR_BARS = 5` (:27649-27650), emitting a Phase-6 STUDY row (family
  `PHASE6_COMPRESSION_BREAK_STUDY_FAMILY = "compression_break"`, :27645) from
  `enrich_priority_rows_with_phase6_studies` (:29244, called once from `runner.py:2230`), which
  `_evaluate_priority_snapshot_for_date` does NOT call. PCT-3 item 4 is therefore that rule plus
  the >= 1.0 ATR-20 bar-range clause, evaluated in the snapshot function under the `_recent` /
  `_rule_version` names - not a second detector. The claim name `compression_break` (PCT-1) is the
  same concept as the study family and shares its id on purpose.

### 4.4 Trendlines today (PCT-2)

- `legacy.py` constants :493-502 (`PRIORITY_TRENDLINE_LOOKBACK_BARS` 200, `MIN/MAX_ANGLE_DEG`
  30/60, `TOUCH_TOL_ATR` 0.35, `PIVOT_WINDOW` 3, `MIN_SEPARATION_BARS` 10, `BREAK_RECENT_BARS` 3,
  `BREAK_MAX_ATR` 1.5, `BREAK_SCORE_BONUS` 18); `_find_trendline_pivots` (:20683-20717);
  `find_directional_trendline_candidate` (:20747+); `trendline_break_recent` /
  `trendline_break_note` on the row and the tracker snapshot (:6411-6412), read by the near-
  favorite gate (:9408-9417). Chart: `scripts/chart_levels.py` `GROUP_TRENDLINE = "d1_trendline"`,
  `trendline_id` (:170-180), `trendline_level` (:535-620) from `priority_trendline_candidate` /
  `priority_trendline_break_candidate` on the `ai_state` row (:313-345).
- D1 event kinds: `chart_watch.py` `D1_EVENT_KINDS` (:80-96), `D1_EXTENSION_KINDS` (:106-117),
  `D1_PULLBACK_KINDS` = the rest; evaluation `d1_event_levels` (:1003) + `_d1_event_hit` (:1064,
  per-kind branches e.g. `sma_break` :1095); `evaluate_d1_event_watch` (:1239). Focus picks get
  PULLBACK kinds automatically (`_poll_focus_d1_interest`, `alert_center_panel.py:4391-4480`),
  EXTENSION kinds only when armed (`_poll_d1_event_watches`).
- Families: `scripts/master_avwap_lib/setup_tagging.py` `_FAMILY_TAGS` (:19-38, 18 entries),
  `derive_setup_tag_payload` (:130-233) - one `setup_family`, up to 6 `setup_tags`. **The
  `TRENDLINE_BREAK` confirmation tag ALREADY EXISTS** (:209-210, on `trendline_break_recent`), so
  PCT-2 item 1 is built; PCT-2 is the event kind and the feed alert only.

## 5. Packet PCT-1 - Pullback alert and the three claim names

Branch `claude/pct-1-pullback` off `origin/claude/pullback-compression-2026-09-15`. Tester first,
then builder. Files: `scripts/indicators/pullback_sma_reclaim.py` (NEW, pure),
`scripts/intraday_history.py` (NEW: the H1 cache generalised by `interval_minutes`;
`h1_history.py` becomes a thin alias so every `test_rv_h1_*` test passes unchanged),
`scripts/chart_watch.py`, `scripts/ui/panels/alert_center_panel.py`, `scripts/ui/widgets/arm_bar.py`,
`scripts/ui/widgets/armed_watch_list.py`, `scripts/setup_docs.py`,
`scripts/ui/annotations/setup_claims.py`, `setup_registry_v1.json` (regenerated), docs.

Items:

1. **Pure rule** `pullback_sma_reclaim.py`, `RULE_VERSION = "pullback_sma_reclaim_v1"`:
   `evaluate(bars, *, side, sma_length, bar_minutes, armed_at, now, episode_state=None) ->
   PullbackResult | None` over completed session-aligned bar dicts (`datetime, open, high, low,
   close, volume`), returning `None` below warm-up or when stale, else a frozen result with
   `fired: tuple[Fire, ...]` (each `Fire`: `trigger, timeframe, bar_dt, sma, close, lrsi,
   lrsi_from_below_50, atr, message`), the new `episode_state` (below / reclaimed / retested, the
   reclaim bar time, per-trigger fired marks) and `reason`. LRSI via `compute_efficiency_lrsi`
   on closes (negated for SHORT), level 80 as a parameter, never touching `CROSS_LEVELS`. SMA is a
   simple mean of closes. ATR-14 on the same bars for the retest tolerance. No sleeping, no clock
   reads inside; `now` is passed.
2. **Intraday history** `intraday_history.py` `IntradayHistoryCache(interval_minutes, *,
   downloader=None, period="1mo")` with the H1 cache's contract (one fetch per completed bucket per
   symbol, own worker, `bars_for`, `request`, `unavailable`, `last_refresh_failed`), bucket math
   parameterised; `h1_history.H1HistoryCache` = `IntradayHistoryCache(60)`. Regular session only
   (`prepost=False`), the forming bucket dropped by `completed_bars`.
3. **Kind** `pullback` ("Pullback alert") in `WATCH_KINDS`, persistent; `ChartWatch` gains
   `triggers: tuple[str, ...]` (default all four: `h1_ema15_bounce`, `sma_reclaim_lrsi`,
   `reclaim_then_lrsi`, `sma_retest`) and `fired: dict[str, str]`; `chart_watch_from_dict` maps a
   stored `h1_ema_bounce` watch to `pullback` with `triggers=("h1_ema15_bounce",)`; the
   `h1_ema_bounce` button is removed from the arm bar; `watch_reason("pullback", side)` reads
   `waiting for a pullback entry (LONG): H1 15-EMA bounce, M15/M30 SMA reclaim + LRSI, SMA retest`.
   The H1 evaluation path is called unchanged for the `h1_ema15_bounce` trigger.
4. **Poll**: `_poll_h1_bounce_watches` becomes `_poll_pullback_watches` (same timer, same expiry
   call first), evaluating each trigger with its own bars (M5-cache-aggregated H1 with the yfinance
   fallback as today; M15 and M30 from their caches), recording one `watch_fired` review event per
   fire with `trigger`, `timeframe`, `rule_version`, `lrsi_from_below_50`, pushing through
   `_push_armed_watch` (non-blocking) and adding one feed row per fire (message names the trigger
   and timeframe: `NVDA LONG: Pullback - M15 150-SMA reclaim + LRSI 80 cross (from below 50)`).
   The watch stays armed until expiry / disarm; a trigger fires once per episode.
5. **Auto-arm** in the same poll, before evaluation: for every `active_claims()` row with
   `horizon D1` and every `all_focus(category="swing")` name, ensure a `pullback` watch exists for
   (symbol, side) with `source_text` `auto: claimed pick` / `auto: swing Focus`; a watch the trader
   disarmed is kept as `declined` (a `ChartWatch` field, persisted) and not re-armed while the
   source lives; when the claim / pick is gone the auto watch is removed with one review event
   `watch_retired_source_gone`. DESK, AWAY, EVENING and OFF alike (documented as the armed-alert
   exception in `AUTO_MODES_AND_QUIET_HOURS_PLAN.md`).
6. **Health cell**: `M15 from yfinance` / `M30 from yfinance` / `not measured (N of 160 M15
   bars)` / `... (stale - last refresh failed)` per timeframe, joined with `;` in one cell; the
   existing H1 states unchanged.
7. **Claim names**: the three `SETUP_DOCS` entries (group `"Entry timing and breaks"`, `what`
   quoting the trader's rule in plain words, `detection` listing the triggers, `evidence`
   "none yet - graded by name in My claims"), the ids appended to `EXTRA_CLAIM_IDS`,
   `setup_registry_v1.json` regenerated by `build_setup_registry.py --write`.
8. **Docs on the branch**: `docs/DESK_INTERNALS.md` entry "PCT-1 - the Pullback alert", the
   amendment line in `docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md`, this file's status table.
   Handoff carries `CHANGELOG INVENTORY:` and `GATE:` blocks (WS0 convention).

Tests (tester writes them red, `tests/test_pct1_pullback_alert.py`): the pure rule on hand-built
M15 and M30 fixtures for both sides - below-then-reclaim with an LRSI 80 cross on the reclaim bar
fires `sma_reclaim_lrsi` with `lrsi_from_below_50` true when a bar 2-4 back was under 50 and false
otherwise; a reclaim with the cross 4 bars back does NOT fire; a reclaim whose bar ended before
`armed_at` does not fire; `reclaim_then_lrsi` fires on a later M15 cross while every M30 close
stays above SMA-75 and is cancelled by one close below; `sma_retest` fires on a low within 0.25 ATR
that closes above and not on a close below; each trigger at most once per episode and again in a
new episode; fewer than the warm-up bars -> `None`; a forming bar is never read (bar_start +
minutes > now); `IntradayHistoryCache(15)` buckets on 15-minute boundaries and fetches once per
completed bucket (fake downloader); a stored `h1_ema_bounce` watch loads as `pullback` with the H1
trigger; the arm bar shows "Pullback alert" and no "H1 retester"; the poll arms one watch per
active D1 claim and per swing Focus name, never twice, and honours `declined`; a fire writes one
`watch_fired` row naming trigger and timeframe and calls the push door once; the rail offers the
three new ids (extend `test_the_rail_offers_every_claim_the_trader_asked_for`); every existing
`test_ws_10c_*`, `test_rv_h1_*` and `test_chart_watch.py` test passes with only the pinned label /
reason strings updated.

Gate text (the lead numbers it): next restart, chart a LONG name and click **Pullback alert**: the
Armed board lists it with the reason naming the three families of trigger and a health cell per
timeframe that flips from `not measured (N of 160 M15 bars)` to `M15 from yfinance` within one
completed quarter hour; a claimed D1 pick and a swing Focus name appear on the Armed board on their
own within two minutes with `auto:` in the source; when a completed M15 or M30 bar reclaims its SMA
with an LRSI 80 cross, ONE phone alert and ONE feed row arrive naming the trigger and timeframe;
disarming an auto watch keeps it off until the claim is dropped. NOT a failure: no fire for days; a
health cell reading `stale` after a network outage; an `h1_ema15_bounce` fire looking exactly like
the old H1 retester's.

## 6. Packet PCT-3 - Compression: measure, chip, calibration, `compression_break`

Branch `claude/pct-3-compression` off the plan branch. Tester first, then builder. Files:
`scripts/master_avwap_lib/legacy.py` (additive only: the copy-through and the v1 break tag),
`scripts/master_avwap_lib/setup_tagging.py`, `scripts/compression_chip.py` (NEW, pure reader),
`scripts/ui/widgets/setup_delegate.py`, `scripts/ui/services/data_feed.py` (the ai_state merge on
the worker), `scripts/compression_calibration.py` (NEW CLI), docs. **Not** `setup_docs.py` (PCT-1
owns the claim names).

Items:

1. **Copy-through**: `compression_score`, `compression_stdev_atr_ratio`,
   `compression_range_atr_ratio`, `compression_close_range_atr_ratio` and `compression_rule_version`
   (`anchor_compression_v1`) join `compression_flag / penalty / note` on `priority_summary`, the
   `ai_state` symbol entry and `build_tracker_setup_record`'s `compression_summary`. Score
   arithmetic untouched; every existing score assertion in `tests/test_master_avwap_setups.py`
   passes unchanged.
2. **Chip**: `compression_chip.read_row(raw) -> CompressionRead(flag, score, penalty, note,
   ratios)`; the delegate paints a `compressed` chip (style `caution`) after the bucket / wrong-side
   chips when `flag` is true, tooltip `compressed - score 3/3, stdev 0.52 ATR, range 2.1 ATR,
   close-range 1.4 ATR, penalty 10 (relief: ...)`; rows lack the fields until the report is merged
   with the `ai_state` symbol entry on the ChartDataService / data-feed worker (the builder finds
   the existing enrichment seam - `enrich_setup_rows_for_display` or the panel's own worker - and
   never reads a file on paint). Hides nothing, moves nothing (identical-visible-rows test).
3. **Calibration CLI** `python -m compression_calibration --since 2026-08-20 [--out DIR]`, read-only
   on live stores, run from `scripts/`: joins every coded `compressed` / `support_resistance_
   cluttered` veto in `trader_annotations.jsonl` (symbol, session_date, side) to that session's
   scan population - the builder confirms the cheapest source that holds a per-(symbol, session)
   scan row with the anchor (the tracker SQLite mirror `tracker_store.py` or the tracker JSON's
   feature snapshots; fall back to recomputation from `daily_bar_cache` at the session date) - and
   computes, at the session's completed close, for the vetoed rows and for the rest of that day's
   shown rows: (a) the existing three anchor ratios and `is_compressed`; (b) `range10_atr14`
   (the `d1_environment` rule per symbol); (c) `range20_atr14`; (d) Bollinger(20, 2) width
   percentile over 120 sessions; (e) ATR-14 / ATR-50. Output: one CSV under
   `%LOCALAPPDATA%\TradingBotV3\diagnostics\compression_calibration_<date>.csv` and a printed
   table per measure with n (vetoed / rest), medians, the rank-sum AUC, and the hit rate of the
   current `compression_flag` on the vetoed set. Proposes nothing, changes nothing, names its
   window. Point-in-time: bars after the session date are never read.
4. **`compression_break` v1 tag** in `legacy.py` inside `_evaluate_priority_snapshot_for_date`,
   reusing `assess_compression_break_context` (section 4.3) rather than a second rule: its
   `compression_break_today` in the setup's direction AND the breaking bar's range >= 1.0 ATR-20 ->
   `compression_break_recent = True`, `compression_break_note`,
   `compression_break_rule_version = "compression_break_v1"` (the version key stamped on EVERY row);
   `setup_tagging` adds the `COMPRESSION_BREAK` confirmation tag when the flag is set. The Phase-6
   study row is untouched. No score change (a characterization test pins scores byte-identical on
   a fixture with and without the flag).
5. **Docs**: DESK_INTERNALS entry "PCT-3 - compression is measured before it is tuned", this file.

Tests (`tests/test_pct3_compression.py`): the copy-through fields present on a synthetic priority
row with the values `summarize_anchor_compression` returns; the tracker record carries them; a
score fixture unchanged; the chip reader on a row with the fields, without them (no chip), and with
`compression_flag` "True"/"true"/1 (string-safe); the delegate paints the chip only when flagged and
paints nothing else differently (offscreen golden of an unflagged row); the CLI on a scratch data
dir with three vetoes and a six-row population prints n, medians and an AUC in [0, 1] per measure,
writes the CSV, reads no bar after the session date (monkeypatched cache raising on a later date),
and aborts if `DATA_DIR` resolves under `C:\TradingBotData` without `--live`; the break tag on a
frame compressed yesterday and expanding today, not on a frame expanding from no compression, and
not on a small-range bar.

Gate text: next scan after a restart, `master_avwap_ai_state.json` symbol entries carry
`compression_score` and the three ratios; the setups table shows an amber `compressed` chip on
rows the scan flags (hover: score, ratios, penalty) and NOTHING is hidden or reordered; the trader
runs `cd scripts && python -m compression_calibration --since 2026-08-20 --live` and reads one
table per measure. PASS on day one: the chip appears on at least one row and the CLI names n = 214
(or the day's count) vetoed rows joined. NOT a failure: a low AUC on every measure (that is the
finding); a `compression_break` tag on no row for days.

## 7. Packet PCT-2 - Trendline break

Branch `claude/pct-2-trendline` off the plan branch AFTER PCT-1 and PCT-3 are merged into it.
Tester first, then builder. Files: `setup_tagging.py`, `legacy.py` (tag only), `chart_watch.py`,
`alert_center_panel.py`, `chart_levels.py` (read), docs.

Items:

1. **Tag**: ALREADY BUILT - `setup_tagging.py:209-210` adds `TRENDLINE_BREAK` (confirmation) on
   `trendline_break_recent`; nothing to do but a test that pins it.
2. **D1 event kind** `trendline_break` ("Trendline break") in `D1_EVENT_KINDS` and
   `D1_EXTENSION_KINDS`: `d1_event_levels` carries the scan's frozen line (`trendline_level` from
   `chart_levels`, projected to the session; identity `trendline_id`, endpoints and knowledge time
   stored on the watch, a redrawn line never substituted), `_d1_event_hit` fires when the completed
   D1 close is through the line in the setup's direction and the prior close was not, once per
   (symbol, side, break date). Focus picks do NOT get it automatically (extension kind); armed
   watches do.
3. **Scan-level feed alert**: the seam where a new priority row becomes a D1 feed alert emits
   `Trendline break` once per (symbol, side, break date) when the row carries the tag - the builder
   locates that seam (the D1 wick-alert champion path) and adds the reason additively; a golden
   fixture pins today's D1 alert set on a saved report before the change.
4. **Docs**: DESK_INTERNALS entry "PCT-2 - a trendline break is a tag and an event".

Tests (`tests/test_pct2_trendline_break.py`): tag present only with the flag; the event kind fires
on a synthetic frame whose completed close crosses the frozen line and not on a wick through it,
not on a forming bar, and not twice; a redrawn line does not move an armed watch's line; the feed
emits one row per break and the pre-change D1 alert set is byte-identical otherwise.

Gate text: next scan, a row whose scan note reads `trendline break` carries the `TRENDLINE_BREAK`
tag in the setups table and ONE `Trendline break` row on the D1 feed; arming **Trendline break**
on a charted name and watching a completed D1 close through the drawn line yields one fire.

## 8. Resume checklist for a fresh session

1. Read `CURRENT_CHECKPOINT.md`'s glance block, then this file's status table (section 1).
2. `git fetch origin`; the plan branch is `claude/pullback-compression-2026-09-15`; packet branches
   are `claude/pct-*`. `git log --oneline origin/claude/pct-1-pullback` tells you whether a tester
   (red tests) or a builder (green) went last; the first commit on each packet branch carries the
   packet text.
3. Spawn per `docs/AGENT_TEAM.md`: tester (Opus) -> builder (Opus) -> reviewer (Opus) per packet;
   recon (Sonnet) for any premise you doubt. Fable / the session model orchestrates only.
4. Integrate a finished packet into the plan branch in a scratch worktree, run the full suite with
   the AI lock probed, ruff, smoke 7/7, `launch_gui.py --selftest`; then update section 1 here, the
   checkpoint entry "2026-09-15 - PCT", CHANGELOG inventory + Recent changes, `plan.md` Phase 0.29,
   `docs/README.md` (this file's line), and the trader's `[stated]` lines in
   `memory/people/trader.md` (already added with the plan commit).
5. The trader restarts the desk; never restart it yourself; never merge to `main` unasked.
