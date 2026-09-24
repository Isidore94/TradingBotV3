# Plan to 4.5/5 on every goal (defined 2026-09-24, trader's request)

This file was cleared on 2026-09-24 and now holds the build plan from the deep review of
`main`. Each packet below is written for an Opus builder. An item becomes authorized work
when the trader moves it to `TODO.md` or says "go" on it in chat. Order of execution is
the section order unless the trader says otherwise.

Every packet: branch per packet, build in a worktree, fail-first test, `ruff` clean, run
the area's tests while building and the full suite before merge, one `CHANGELOG.md` line,
one `docs/GATES.md` line for anything the trader must see live. Ask-first files
(`master_avwap_lib/legacy.py`, `bounce_bot_lib/*`, `m5_signal_engines.py`, any detector,
scoring or alert code) are flagged per packet. Nothing here places orders. Shadow first:
no detector, score or alert changes without golden fixtures and the trader's word.

The four open questions were answered by the trader on 2026-09-24; the decisions are
written inline where they apply and listed at the end.

**Done so far (2026-09-24).** P0-1 universe floor (`6908769b`) and P0-2 packets 2a, 2b, 2f
(`87c0bae6`) are merged; the universe was restored to 1,455 names from snapshot
`20260922T130004` and the two stale tracker copies were pruned (2.6 GB). Live gates
#209-#212 are owed. Finished packets are deleted from this file; `git log` has them.

---

## P0-2. Swing scan: speed, and the 1.4 GB tracker snapshot (goals 5, 6, 11)

**Done 2026-09-24** (`87c0bae6`): 2a keeps a failed child's 40-line stderr tail and logs
`[run_master memory]` per phase; 2b publishes signals/reports/state before the tracker
write and turns a tracker failure into a ledger row, a digest OPERATIONS line and the
Health line "tracker last written <stamp>"; 2f pruned `.bak` and the damaged SQLite copy.
Three empty leftovers named `*.damaged-20260905T200233` (`sqlite-shm`, `sqlite-wal`,
`_digests.json`) are still in `data/runtime`; add them to `--prune-copies` on the next
pass. Gates #210-#212 owed.

**Now.** Last good run: 870 s = prep+fetch 450 s, studies 96 s, feature-history 25 s,
scan-factors 186 s, tier-tracker 55 s, other output 25 s. `export_scan_factor_views`
(`legacy.py:12322`) reads the whole 710 MB `d1_features_history.csv` every scan. The
write slot parses the 1.38 GB `master_avwap_setup_tracker.json` (`runner.py:2580`); the
tracker was last written 2026-09-24 07:46. Decision 0017 (readers move to SQLite) has
moved no reader: only `tracker_store.py` opens the DB. The new memory lines say where
the peak is; read a few sessions of them before starting 2c.

**Build, in this order, one packet each.**
- **2c Scan factors read a window, not history.** Keep `d1_features_history.csv`
  append-only. Maintain a rolling `d1_features_window.csv` (last 40 sessions) and make
  `export_scan_factor_views` read the window. Parity fixture: the leaderboard from the
  window must equal the leaderboard from the full file for the same window, built on a
  scratch copy before the change. This touches `legacy.py` (ask first).
- **2d Readers to SQLite (decision 0017), one at a time, each with a parity test on a
  scratch copy:** (1) the write slot's `load_setup_tracker_payload`; (2) `ai_summary`'s
  `setups.current_tracker` source becomes a compact projection query; (3)
  `journal_analytics` context rows (the 1.08 GB parse behind the Corrections OK button);
  (4) `held_run_score`, `compression_calibration`, `research_warehouse/ingest_existing`.
  When all readers are on SQLite, the JSON becomes a nightly export only, then goes.
- **2e Fetch phase.** Measure the daily-bar cache hit rate per scan (log hits/misses/
  refreshes). Batch the misses through one `yf.download` call per 100 symbols. Target
  under 120 s for ~1,100 names.

**Tests.** Parity fixtures for 2c and each 2d step; 2e lands with a gate line (no
offline timing test is honest).

**Done when.** Scan under 6 minutes; no reader parses the JSON; ten consecutive close
slots carry a fresh "tracker last written" stamp (gate #212 checks the first).

---

## P1-3. Night AI chain: too long, often degraded (goals 6, 10, 11)

**Now.** Ledger since 09-15: `ai_summary` 1,523 min over 20 runs, `ticker_briefs`
66 to 139 min per night (290 tickers, 102 model calls, `tickers_reused=0`) and read only
by `ai_summary`, which is now Saturday-only and on 09-18 ran three times for 3 to 5 hours
each, all `degraded_no_narrative` on endpoint read timeouts (540 to 900 s).
`journal_import` fails most nights on IBKR Flex ("Statement could not be r...") and once
on Questrade discovery. `day_review_narration` is rejected when Ollama times out.
Nights ran 112 to 835 minutes.

**Build.**
- **3a Night budget.** `ai_night_budget_minutes` (default 150) enforced by
  `ai_jobs/runner.py` from `model_probe` measurements: model slots run in priority order
  and a slot that would overrun is skipped with a ledger reason. Order:
  `day_review_facts` (code) → `daily_digest` → `day_review_narration` →
  `market_story_narration` → `setup_research` → `journal_enrichment` →
  `observation_tags` → `ticker_briefs` last.
- **3b Ticker briefs.** Decided 2026-09-24: Saturday only (`WEEKEND_ONLY_SLOTS`),
  restricted to names in the week's picks, alerts and journal, with a 7-day reuse cache
  keyed by (symbol, week). The slot reports how many names it skipped and why.
- **3c Summary.** Cap slices per run and stop the slot after two endpoint timeouts; the
  facts artifacts are written regardless (already the design).
- **3d Journal import.** Read the exact Flex error from the ledger; add wait-and-retry
  within the window for "statement not ready"; Health line "journal import: last success
  <date>, last error <one line>".
- **3e Ollama probe** at night start (model loaded, one-token call, 30 s cap). On
  failure the night is deterministic-only and Health + the morning digest say so.

**Done when.** A weekday night finishes under 3 hours and every skipped or failed slot
shows one plain reason on the Health page.

---

## P1-4. Setup permutations: find the key to each setup (goal 7)

**Now.** Families are first-class (18 tags in `master_avwap_lib/setup_tagging.py`), the
scan writes 264 feature columns per row (`d1_features_history.csv`, including
`current_band_zone`, `htf_trend_1h/4h`, `htf_retest_sma`, `hv_level_*`, `cloud_level_*`,
`top_pattern_weekly_*`, `trend_ma_alignment`, `sma_breakout_*`, industry RS,
compression), the warehouse computes `dist_sma50/100/200_atr` per occurrence
(`research_warehouse/features.py:625-650`), the M5 outcome log carries
`context_json` (regime, RVOL, RRS, sector), and `setup_environment_evidence` cuts by
`d1_environment`. What does not exist: a permutation key. No row says "AVWAP band bounce
with SMA100 support", no report ranks facets, and the registry's 58 entries have all
six structural fields unestablished. The trader wants more than SMAs: EMAs, higher
timeframe MAs, H1 entry timing, and whatever else turns out to be the key.

**Design.** A **permutation key** = family × a set of named **facets**. A facet is a
versioned pure rule over data the scan already records at the row's own scan date
(point in time). Missing data is `unknown`, never a default. v1 facets:

*D1 structure*
- `anchor`: current earnings anchor vs previous anchor (`current_anchor_date`,
  `previous_anchor_*`).
- `band_zone`: `current_band_zone` plus distance to VWAP / upper 1 / lower 1 in ATR.
- `ma_support`: nearest moving average within 1 ATR on the support side (long: below
  price; short: above): `sma20, sma50, sma100, sma200, ema8, ema15, ema21, none,
  multiple`. Needs `dist_<ma>_atr` columns on the scan row (add in the enrichment
  step, not in the detector).
- `ma_stack`: `trend_ma_alignment` (EMA15 vs SMA20) and the full order of price /
  EMA21 / SMA50 / SMA200.
- `weekly`: `top_pattern_weekly_ema15_hold`, `weekly_above_sma100`,
  `weekly_sma50_retest_recent`, weekly EMA8 hold streak (all exist).
- `levels`: `hv_level_nearest_bucket` and distance, `cloud_level_*`, previous-day
  range break, compression state and break.
- `strength`: `daily_relative_strength_score` band, `rs_vs_industry_5d` sign,
  `industry_13w_return_pct` band, `relvol` band.
- `earnings`: sessions since gap (mid/post), `days_to_next_earnings` bucket.
- `discovery_slot`: which scan slot first showed the name (07:30 / 10:00 / 12:45 /
  close) from `scan_replay`.

*Entry timing (H1/H4/M15/M30)*
- `htf`: `htf_trend_1h`, `htf_trend_4h`, `htf_trend_aligned`, `htf_retest_confirmed`,
  `htf_retest_sma`, `htf_retest_timeframes`.
- `entry_trigger`: which chart-watch trigger fired for the name before entry
  (`h1_ema_bounce` touch/hold, `pullback` with `sma_reclaim_lrsi` /
  `reclaim_then_lrsi` / `sma_retest` on M15-150 or M30-75, `band_bounce`,
  `hod_avwap`/`lod_avwap`), read from `alert_chart_watches.json` history and
  review events.
- `m5_confirmation`: an M5 alert on the same name and side that day (`bounce_type`).

*Market*
- `d1_environment` label, `market_regime_label`, `spy_above_sma20/50`,
  `spy_five_day_return_pct` band, weekday, `side_aligned_day`.

*Outcomes.* Swing: tracker episodes through `session_horizon_outcomes` v2 and
`execution_convention` v2 at 1/3/5/10 sessions, win rate first (decision 0016). Day
trades: `intraday_bounce_outcomes.csv` (`close_r`, `mfe_r`, `mae_r`, `stop_hit`) with
`held_run_score`'s MFE rule.

**Build.**
- **4a `scripts/setup_permutations.py` (pure, new).** `facets_for_row(row, ctx) ->
  PermutationKey` with `permutation_rule_version`. Stamp the key on every scan row
  (new `d1_features_history` columns), on the tracker episode when it opens, on the
  warehouse occurrence, and on M5 outcome rows. Shadow only; nothing reads it for
  ranking. Adding columns to the enrichment is in `runner.py`; if a value must come
  from `legacy.py` it is ask-first.
- **4b Backfill CLI (scratch only).** Walk `d1_features_history.csv`, tracker episodes
  and session-horizon outcomes on a copy and write `permutation_outcomes.parquet`: one
  row per episode × horizon with every facet as it was on the scan date. Same for the
  M5 population from the outcomes CSV. Refuses to run against a live root.
- **4c The search.** For each family: baseline (Wilson lower bound on win rate, mean R,
  n, sessions); for each facet value and each facet pair, the same statistics; lift vs
  baseline; floors as in `setup_grades` (n ≥ 30, sessions ≥ 10); hold-out = the last
  20 sessions, never used for selection; every grid registered in
  `research_warehouse/trial_ledger` before results are read (the k > 10 rule). Output
  `permutation_report.json` and a Research → **Setup keys** tab: family → ranked
  facets with lift, n, sessions, hold-out result, and "no key found" as a first-class
  answer. Both populations shown separately, never pooled.
- **4d Narration.** A Saturday slot hands the report (facts only) to the local model
  for three cited sentences per family; the frontier research pack includes the report.
- **4e Promotion.** A facet that passes hold-out two weeks running may become a named
  sub-family in `setup_grades` (for example `avwap_band_bounce@sma100_resistance`).
  Ask-first, golden fixtures, trader's word, one at a time.
- **4f Wider facets**, added as soon as 4c runs once (the trader wants more than moving
  averages; the engine must make adding a facet a one-function change with its own
  fixture): Heikin-Ashi reversal and SMI state at entry (`indicators/` has both),
  Laguerre RSI / TC2000 LRSI cross state on D1 and H1, H4 pullback depth in ATR,
  earnings gap size in ATR, time of day of the H1 trigger, ATR percentile (volatility
  regime of the name), distance from the 52-week high/low, number of prior respects of
  the level (`LEVEL_RESPECT_*` in `levels.py`), consecutive closes above/below the
  level, age of the setup in sessions since first eligible, sector RS rank that day,
  weekly-options flag, and the D1 zone-arm level that fired (`d1_zone_arms`).
- **Search depth rule (part of 4c).** Single facets first, then pairs, then triples,
  never deeper; each depth registered as its own grid in the trial ledger; a deeper
  key must beat its best parent on hold-out or it is not reported. Keys are searched
  per horizon (1/3/5/10 sessions) because the key for a 1-session hold and a 10-session
  swing may differ.

**Tests.** Fixture rows → expected keys, including `unknown` on missing data; the
backfill on a 200-row scratch fixture; the search on a synthetic population with one
planted key that must be found and one planted lucky facet that hold-out must reject.

**Done when.** Research → Setup keys answers "AVWAP band bounce shorts: SMA100 within 1
ATR above, 71% (n=48, 14 sessions) vs family 58%; hold-out 66% (n=12)" for every family
with data, and says "no key found" honestly for the rest.

---

## P1-5. Ten best names, and one "best right now" surface (goals 1, 4, 5)

**Now.** Today's digest top 10 is one family and one side ten times because the order is
the family Wilson bound (`autopilot_core.py:3660`, `SWING_ORDER_WILSON`). The desk has
no single view that merges M5 alerts, Movers and D1 setups. The 07:06 watchlist build
sweeps 603 names; the D1 universe is 1,455 when healthy.

**Build.**
- **5a Digest and setups table.** Cap 3 per (family, side) inside the top 10, then fill
  by the same order; say the rule in the "Ranked on" line. The points order stays a
  switch. Show the permutation key's short label per row once 4a lands.
- **5b "Best right now" strip** on the Trading Desk: one ranked list of today's M5
  alerts (grade + live R from `live_alert_results`), Movers dip-strong names, and D1
  setup names with an M5 confirmation (the `held_run_score` rule, display only), each
  with why, entry and stop. Refresh on bar boundaries; diff, never rebuild; display
  only (`review_policy` ranks only; no detector change).
- **5c Wider M5 discovery.** The open sweep reads `universe_all.txt` (not 603 names);
  IB budget stays for the top N by gap and RS plus Focus and typed names first (SN6).

**Tests.** Pure ranking and cap functions; strip diff tests; a digest fixture.

**Done when.** The digest shows ten different names across at least three families when
the scan has them, and the desk has one strip that answers "what is best right now".

---

## P1-6. Optimal entry: state, plan, risk (goal 2)

**Now.** D1 detail shows a plan with stop and TP prices and a stale flag
(`ui/widgets/setup_detail_view.py:209-220`); the setups table has no plan columns. M5
alerts carry `stop_price`; the Working-now strip computes live R; the alert row shows no
entry/stop and no state. The journal already stores `planned_entry`, `planned_stop`,
`planned_risk`, `risk_source` (`journal_store.py:2219-2316`). `entry_quality` measures
post-hoc MFE but is research-only.

**Build.**
- **6a Entry state chip** on every M5 alert row and the Working-now strip: `valid`
  (price between entry and +1R, level held), `improved` (a later touch of the level),
  `gone` (stop hit or beyond +1R), from the alert's own entry/stop and cached M5 bars
  (`live_alert_results` arithmetic; completed bars only).
- **6b Plan columns** on the D1 setups table: trigger level, stop, TP1, expected R,
  from the detail plan; the stale flag inline.
- **6c Risk per trade.** Decided 2026-09-24: fixed dollars. One local setting
  `risk_per_trade_dollars`; no account size is stored. Both plans show
  shares = risk / (entry − stop), rounded down, blank when the stop is unknown. Never an
  order. The journal
  entry grade = (actual entry − planned entry) / (planned entry − planned stop), shown
  per trade; MFE/MAE per trade (owed on STATUS) lands here from cached M5 bars for day
  trades and daily bars for swings.
- **6d Entry timing chips** on D1 rows: armed or fired chart-watch triggers
  (`h1_ema_bounce`, `pullback`) shown as "timing: H1 15-EMA held 10:30".

**Tests.** State machine fixtures (valid/improved/gone); plan column projection;
shares arithmetic; journal grade on a fixture trade.

**Done when.** Any candidate on either timeframe shows entry, stop, target, R, shares,
and whether the entry is still valid, without a click.

---

## P1-7. A trading plan the AI reads and challenges (goal 9)

**Now.** The rules live in `docs/RULES.md`, decision 0016, the recap "one rule for
tomorrow" loop, Mentor questions, week coach, improvement ideas and review-policy
drafts. There is no single trader-owned plan with history.

**Build.**
- **7a `C:\TradingBotData\trading_plan.md`** with fixed headings: Goals, Rules, Setups I
  trade, Risk, What I am testing, Decisions (dated lines). The trader edits it in any
  editor; the desk shows it read-only on Day Review and Weekend Prep; every change is
  snapshotted to `trading_plan_history/<stamp>.md` (append-only).
- **7b Nightly `plan_review` slot** after the facts slots: the model receives the plan,
  the day report card, the grades, the measured report and (later) the permutation
  report, and must return at most three challenges, each citing one plan line and one
  evidence id; code validates citations exactly as `improvement_ideas` does. Challenges
  appear as Mentor items: accept appends a dated line to Decisions; reject logs the
  reason; untouched items expire in 7 days.
- **7c Weekly.** The frontier research pack includes the plan and the week's challenges
  and answers.
- **7d One source.** The recap rule loop writes its "one rule for tomorrow" into the
  same file's What I am testing section.

**Tests.** Plan parser on a fixture; snapshot on change; challenge validator rejects an
uncited claim; accept/reject/expire transitions.

**Done when.** The trader reads one file that says what they are trying to do, and each
week the AI has argued with it using numbers, and the arguments and answers are kept.

---

## P2-8. Market analysis (goal 3)

**Now.** SPY M5 state engine, internals recorded (VXX, HYG, TLT, RSP, MAGS), opening
regime history, D1 environment label, market prep (news, calendars, earnings, SEC),
story, thesis, measured read grades, pasted forecast brief. The regime read itself is
one SPY series; internals are recorded, not read.

**Build.**
- **8a Breadth from what we have.** Daily advance/decline count and % of `universe_all`
  above SMA20 and SMA50 from the daily cache; stored beside `d1_environment` rows and
  shown on Market Prep and Day Review; graded like reads.
- **8b Second axis on the regime line**, display only: "bearish_weak + vol bid + narrow
  tape" from the internals. The SPY-pause champion is untouched.
- **8c Freshness.** Date-stamp `sector_etf_map.json` and the industry maps; Health warns
  past 90 days.

**Done when.** The morning read says what kind of day it is on three axes (SPY state,
breadth, internals) and the evening grades it.

---

## P2-9. Looking back (goal 6), after P0-2 and P1-4

- **9a Pick equity curves** per population (swing picks by day, M5 alerts by day) in R
  with n, on Research → Results, from the same evidence cells (no new statistic).
- **9b Hold-out beside the window** in `working_lately` and `setup_grades`: the last 20
  sessions shown next to the prior window so a leader that only led lately is visible.

**Done when.** One page shows, for each population, what worked, since when, and whether
it held out.

---

## P2-10. Journal (goal 8), the owed items

- `journal_analytics.counts_in_pnl` in Day Review, Weekend Prep and recap totals.
- MFE/MAE per trade (with 6c).
- `_mentor_rule_lane` `list_trades` off the Qt thread.
- Flex import fix (3d) and the Questrade gap-day import.
- The `CLOSED_PARTIAL` misspelling at the four sites (two are ask-first).
- `journal_feed._store()` module-global cache → owned by the service.

---

## P2-11. Reliability, hygiene, desk performance (goal 11)

- **11a** Triage the 78 `except ...: pass` sites (1,734 `except Exception` total;
  `bounce_bot_lib/legacy.py` 123, `alert_center_panel.py` 120): scans, stores,
  publishes and alerts first; each becomes a logged reason or a named `unknown`.
- **11b** Ruff: widen from five codes to `E7`, all `F`, `B` on non-legacy code, fixing
  as it goes.
- **11c** Health page: night chain status, universe count vs floor, IB status, Ollama
  probe, journal import last success (the tracker stamp landed with 2b).
- **11d** Secrets: the market-prep OpenAI key and the ntfy token sit in plain text in
  `local_settings.json`; move them to Windows Credential Manager (`keyring` is a new
  dependency, packaging trigger) or a separate file outside the settings JSON.
- **11e** Desk startup: `master_avwap_lib.legacy` costs 1.8 s at import (pandas
  included); import it lazily where the UI does not need it at boot. The 8
  `setStyleSheet` calls in `focus_picks_panel.py` move to `theme.qss` variants. Measure
  the 43 timers with `thread_cpu_gauge` for one live day before touching any.
- **11f** Branches. Decided 2026-09-24: review, then merge what passes. As of the
  evening of 09-24: `claude/d1-alert-regroup-2026-09-24` is in flight (8 commits: the
  D1 alert menu regrouped by principle, three new kinds, an `sma_break_retest_v1`
  engine, veto follow-ups; alert code, so ask-first and golden fixtures apply; its
  commits cite gates #209/#210, which P0 now holds on `main`, so renumber on merge).
  `claude/token-cost-flags-2026-09-23` is stale: delete. `codex/air3-recovery-fix`,
  `codex/tj17-future-fix`, `codex/tj17a-read-integrity`: rebase onto `main` in a
  worktree, run their tests, GO / NO-GO, merge on the trader's word or delete. Eleven
  older `codex/*` remote branches (`alert-quality-2026-09-22`,
  `overnight-ai-repair-tests`, `phase-031`, `setup-pqs-repair-2026-09-22`,
  `setup-score-repair-2026-09-22`, `slimdown-review-fixes-2026-09-22`, `sol-p1-forward`,
  `sol-p2-entry-comparison`, `sol-p3-research-proposal`, `tj17-recap-complete`,
  `ws-rp-resume-2026-09-15`) predate the 09-22 integrations: check each with
  `git log origin/main..<branch>` and delete the ones already integrated.

---

## Decisions the trader made on 2026-09-24

- Risk unit for sizing: fixed dollars per trade; no account size stored (6c).
- Tracker `.bak` and `.damaged` copies: delete both now, through a tested CLI (2f).
- Ticker briefs: Saturday only, fewer names, 7-day reuse cache (3b).
- Unmerged branches: review each, merge what passes on the trader's word, delete the
  rest; chart-wheel-zoom waits as before (11f).
