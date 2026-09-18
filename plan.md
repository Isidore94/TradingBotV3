# TradingBotV3 plan — the trader journal program

Rewritten from scratch on **2026-09-17** at the trader's direction. Everything that was
here before (Phases 0.5–0.32, every one BUILT) is archived verbatim at
[`docs/archive/PLAN_ARCHIVE_2026-09-17.md`](docs/archive/PLAN_ARCHIVE_2026-09-17.md); the
live gates those phases still owe live ONLY in `CURRENT_CHECKPOINT.md`. This file keeps
the section numbers other documents cite (§5 invariants, §6 live validation, §7
promotion, §12 the work queue) so nothing that points here had to move.

Authority (decision 0013): this file owns the build order; `CURRENT_CHECKPOINT.md` owns
the active item and open gates; `CHANGELOG.md` owns what exists; `docs/decisions/` own
accepted constraints; the code is the fact. The trader's answers that shaped this program
are recorded in
[`docs/decisions/0021-trader-journal-consolidation.md`](docs/decisions/0021-trader-journal-consolidation.md).

## 1. Mission and product boundary

TradingBotV3 is a decision-support system for one full-time trader. It prepares the
market, discovers swing and intraday candidates, monitors them, alerts, records the
trader's decisions and their outcomes, and supports controlled research. **It never
places or routes orders** (decision 0001). Broker execution, consumer distribution and
any automatic promotion of research output are outside the product boundary.

The operating topology: the Ryzen 7 8845HS mini-PC is the only always-on desk, scan host
and local-LLM host; `launch_gui.py` starts the PySide6 Trading Desk; ntfy and the verified
`autopilot_today.txt` digest are the remote surfaces; `C:\TradingBotData` is a plain local
folder and the DAS is the durable tier (decision 0015).

**What the program is for, in the trader's words** — decision 0016 (2026-09-02) remains the
tie-breaker for every prioritisation call: names before entries; win rate is the swing
headline and MFE after a held level the day-trade headline; one click teaches the bot;
"what is working lately" lives on the Trading Desk; likes are training data. Decision 0021
(2026-09-17) adds the journal's purpose in the trader's own words: *"this should be a
simple 'what worked, what didn't and what was your process'"* and *"these systems should
ALL intertwine into one easy to use trader journal that's mostly automated."*

## 2. Status vocabulary

`PLANNED` (in this file, not started) → `PACKETED` (a `.claude/packets/TJ-n.md` exists) →
`BUILT` (code and green tests on a branch) → `REVIEWED` (independent reviewer GO) →
`MERGED` (on `main`) → `LIVE_VALIDATED` (the numbered gate in `CURRENT_CHECKPOINT.md`
passed on the desk). A phase is a stub here once every packet is MERGED; its gates stay in
the checkpoint until validated.

## 3. Current state (2026-09-17)

- Phases 0.5–0.32 are BUILT and MERGED on `main`; their open live gates (#53–#144) are in
  `CURRENT_CHECKPOINT.md`. The desk runs from SOURCE and was not restarted for them.
- The Market Journal, Daily Recap and Weekend Prep pages exist and work as their specs say,
  but the trader has judged them (2026-09-17): *"Daily recap doesn't give me enough useful
  info"*, *"Market journal just sucks"*, *"too much shit in these tabs and it's laggy"*.
  Section 12 is the answer.
- The working tree at the time of writing also carries other uncommitted desk display work
  (Phase 0.32 follow-ups, in the checkpoint's 2026-09-17 entries). This plan does not touch it.

## 4. Authority and change control

- `WISHLIST.md` is ideas; an item enters this file only when the trader moves it. The AI
  ideas card built in this program (TJ-6) may *suggest* WISHLIST entries and never writes one.
- File-scoped ask-first rule: any edit to a file housing detector, scoring or alert code is
  asked about before it is made. The packets below are written to avoid those files; if a
  builder finds it needs one, it stops and the lead asks.
- Golden fixtures before any detector/score/alert change (decision 0009). Nothing in this
  program changes a detector, a score, an alert, a watchlist, Focus, the review queue or
  `review_policy.json`.

## 5. Non-negotiable system invariants

### Data and time

- State transitions use completed bars only. A forming bar is a labeled preview.
- Missing or stale data is uncertainty, never confirmation.
- Point-in-time research may use only information available at the simulated decision
  time; timestamps carry explicit time zones.
- Never replace `calc_anchored_vwap_bands`' running-deviation sigma formula.

### Identity and provenance

- Stable identity must distinguish symbol, side, horizon, setup/thesis, anchor, attempt and
  configuration where those dimensions matter.
- Every suggestion, alert, review, research row and outcome must retain enough provenance
  to reconstruct what the system knew.
- User-entered watchlist names are never automatically removed by a machine judgement about
  one name. Amended 2026-09-15 (decision 0020): `longs.txt` / `shorts.txt` are day-trade
  lists and are emptied WHOLE after each session's close; the swing lists keep the rule.

### Runtime and publication

- One component owns each timer, thread, job, mutable store or shared export.
- A failed publish never destroys the last verified report.
- Ambiguous ownership fails closed. The single-main topology does not authorize duplicate
  writers.

### Research and promotion

- Legacy SPY pause detection and D1 wick alerts stay champions until the §7 gates pass.
- No detector, score, ranking, routing or alert-behaviour change lands without a golden
  characterization fixture first.
- Shadow, research, Technical Integrity, warehouse, review-learning and AI outputs have zero
  production influence until separately promoted.
- `review_policy.json` ranks and annotates only; it has no suppression field.
- AI is one-way and evidence-grounded (decision 0011). It may summarize, grade a stated
  claim against measured facts, and propose; it never mutates production state.

### Product behaviour

- The app is decision-support only and never executes orders.
- Honest zero-opportunity and unknown-data states are preferable to filled panels.
- Desk, Away, alerts, journal and AI consume the same canonical opportunity facts.
- The Market Journal is what the trader thought; the Journal is what they traded. Two
  stores, never merged. A page may SHOW both side by side; no writer joins them.

## 6. Live validation program

Automated green tests do not satisfy live gates. The active checklist is
[`docs/FIRST_SESSION_CHECKLIST.md`](docs/FIRST_SESSION_CHECKLIST.md). For the first live
session on a new build record: branch/commit, Python, TWS/Gateway mode, home folder, Auto
profile and session date; full pytest exit code, smoke result and the frozen self-test when
a rebuild trigger applies; the run manifests, heartbeat, job ledger and capture audits; GUI
responsiveness, chart freshness, alert delivery, clean shutdown and restart; every failure or
unknown as evidence, without rewriting the acceptance result. Do not tune thresholds from
one session.

## 7. Shadow evidence and promotion ladder

Promotion is a separate decision from implementation and live validation. Every challenger
requires: (1) a versioned configuration and stable identity; (2) golden/replay fixtures and
a declared evidence window frozen before inspection; (3) complete coverage and data-quality
accounting — a `feature_snapshot_daily` row whose anchor is `reconstructed` or `legacy` and
an outcome row on the `plain_no_target` path are research evidence and never count toward a
promotion gate (BD-99/BD-100); (4) comparison with the active champion on the same inputs
and outcome definition; (5) representative live sessions across regimes, sides and day
parts; (6) explicit success, non-inferiority and rollback criteria; (7) a bounded canary
and one-switch rollback; (8) explicit trader approval recorded in the revision history.

The SPY pullback challenger still needs completed-bar coverage proof, episode reconciliation
across rollovers, timing/false-pause/missed-pause comparison with the legacy detector, and
timezone/staleness/restart validation. The Greatness challenger still needs its own
monitoring lane, same-day plan revision, the full confirmation/failure/re-arm/freshness/
RS gates, transition-chain audits and an outcome comparison with legacy D1 wick alerts.
Neither is scheduled in this plan; nothing below touches them.

## 8. Specifications retained under `docs/`

One line per file in [`docs/README.md`](docs/README.md). For this program the governing
documents are: this file (the spec — no separate spec document is created, trader rule
2026-09-04), decision 0021, `docs/DESK_INTERNALS.md` (the "Market Journal is what the
trader thought", "TM", "Q4" and "Frozen exe" entries), `docs/LOCAL_AI_AUTOMATION_PLAN.md`
(hardware envelope, off-hours rule, model tiers), decision 0018 (stage order), and
`docs/AGENT_TEAM.md` (who builds what).

## 12. Remaining work, in execution order

| Phase | Packets | Status |
|---|---|---|
| 0.33 The trader journal — Day Review, Week Review and the overnight voice | TJ-1 … TJ-8 | TJ-1 MERGED 2026-09-18 (`e00b734a`, reviewer GO after four rounds; live gate #145 owed at the next restart); TJ-1L (two-column layout, presentation only) BUILT 2026-09-18 on `claude/tj1l-day-review-layout`, unmerged; TJ-2 PACKETED 2026-09-18 (`.claude/packets/TJ-2.md`, two parts, for the Codex lead); TJ-3 … TJ-8 PLANNED |
| 0.5–0.32 | — | BUILT; archived; live gates in `CURRENT_CHECKPOINT.md` |

### Phase 0.33 — The trader journal (trader, 2026-09-17)

#### 12.1 What the trader asked for, in their words

- *"I want to see what actually worked that I passed on or what worked that I liked but
  didn't enter. basically instant walk away analysis."*
- *"it also needs some sort of chart system to show me when I commented on it so I can see
  exactly where I went wrong."*
- *"Market journal … should be compacting the days, and also showing me an AI summary of my
  thoughts each day as well as on the D1."*
- *"rename paste weekly forecast to paste daily forecast, that's where I paste the output
  from my scheduled chatgpt prompt that goes over the overnight and daily news."*
- *"i don't need to see the SPY auto modes pasted in there."*
- *"AI can use [the environment timeline, my Trade Mentor thoughts, the auto environments
  and the chatgpt output] to produce a 'what happened that day' … accompanied with annotated
  SPY charts … a true 'day in review'."*
- *"Weekend prep could have 5 of these days collated into one tab (ideally the first tab)."*
- *"it's VERY important we maximize the overnight AI runs to bridge gaps and to really make
  this program feel alive and have it adapt to what we want. the local AI can have a voice
  somewhere where it offers ideas of what we can improve based on what it reads."*
- *"I'd like to also be able to eventually document my emotions/my feelings and a lot more
  intraday to nail my process but first we need better bones."*

The twelve answers the trader gave on 2026-09-17 (decision 0021), one line each: one page
**Day Review** replaces Market Journal and Daily Recap; the week view is the FIRST step of
Weekend Prep; the desk STOPS writing auto-mode flips into the journal and the old rows are
hidden everywhere; the walk-away grades all four populations (liked-not-traded, rejected,
traded-and-left-early, claimed D1 picks); one SPY chart always, a name's chart on click;
the local model writes each day's story overnight and the frontier model writes the week's;
the forecast may be pasted any time and is read only overnight; one day summary of the
trader's thoughts plus a rolling D1 view; the AI's ideas live on a card on Day and Week
Review with Keep/Dismiss; build order is bones first; plan.md keeps a short invariants
section; the lag is on opening the tab and clicking an entry.

#### 12.2 The shape

Left nav before (11 specs, verified in `ui/app.py` on 2026-09-17): Trading Desk · Journal ·
**Market Journal** · **Daily Recap** · Weekend Prep · Universe · Research · Auto Pilot ·
A.I. Summary · System Health · Settings.

Left nav after (10 specs, as built by TJ-1): Trading Desk · Journal · **Day Review** ·
Weekend Prep (opens on **Week Review**) · Universe · Research · Auto Pilot · A.I. Summary ·
System Health · Settings. (`Universe` was missing from both lines when this section was
written and `A.I. Summary` carries its dots in the spec; the nav is pinned title by title in
`tests/test_tj1_page_specs.py`.)

**Day Review** is one date at a time (a session picker, Today marked provisional until the
close), read top to bottom:

1. **What happened** — the overnight story for that day (TJ-4), or the deterministic facts
   with "no story yet, it is written overnight" until it exists. Beside it the **rolling D1
   view**: what the trader believes about the bigger picture, the open theses and whether
   each is still true.
2. **Walk-away** — four labelled tables, most-ran first (TJ-2): liked but never traded;
   passed / clicked away / vetoed / not today; traded, then left early; claimed D1 picks.
3. **What you said** — the trader's notes and Trade Mentor answers for that day, oldest
   first, never a machine row. A New entry box sits under it (the one from the old page;
   same writer, same store). "Paste daily forecast…" sits beside it (TJ-1).
4. **What you traded** — the day's trades from the trade Journal, one line each. Read-only;
   the Journal page stays the place to tag and correct.
5. **The SPY chart of the day** with a marker at every note, Mentor answer and trade
   (TJ-3). Clicking a walk-away row opens that name's chart beside it with its decision
   marked. One chart widget each, built on first use.
6. **Ideas from the desk's AI** — up to three, Keep / Dismiss (TJ-6).

Gone from the trader's screen (the stores stay; the readers move): the environment
timeline table (its facts feed the story pack), "What the desk measured that session", the
active-thesis drafting pane and "Save interpretation" (theses remain a sidecar the D1 view
reads), the four capture panes, the five Daily Recap tabs. The Daily Recap **Review** tab
(the measured report, next-test card, Copy/Export) moves to **Research > Results** as its
own section; the **Staged picks** table and its "Add selected staged pick to Focus" verb
move to the **Auto Pilot** page. Nothing is deleted from disk in this phase; TJ-8 removes
the retired panels after the gates pass.

**Week Review** is Weekend Prep's first step, already named `week_review` in
`ui/services/weekend_prep_service.py:49` and drawn by `WeekReviewPage`
(`ui/panels/weekend_prep_panel.py:287`). It becomes five Day Review cards (headline,
were-you-right tally, chased-against-news flag, said-vs-did line, small SPY chart) above the
week story (TJ-5), the week's walk-away totals and the week's ideas. The other Weekend Prep
steps are unchanged.

#### 12.3 Ground rules for every packet

- **Team flow** (`docs/AGENT_TEAM.md`): the lead writes `.claude/packets/TJ-n.md` from this
  section; recon verifies every `file:line` premise below before the tester writes; the
  tester commits RED tests; the builder makes them green without weakening them; the
  reviewer reproduces on a copy of live data; the lead integrates. Claude: Sonnet recon,
  Opus tester/builder/reviewer. Codex: Luna recon, Terra build/test/review, Astra only for a
  focused decision and never silently.
- **Qt thread.** Every read runs on a worker (`QThread` / `QRunnable`); a page shows the
  last render and a "refreshing" note while it waits. One chart widget per pane, built on
  first use, reused, never rebuilt; lists diff. Every packet that touches a page records a
  before/after number from `scripts/ui/desk_bench.py` (staged scratch home, never the live
  one) in its handoff: page construct, page open, entry click.
- **Stores.** The trader's words are append-only JSONL (`EvidenceLedger`, stream
  `market_journal`), never rewritten, never deleted; a hidden row is filtered by its
  `origin`. Derived, rebuildable artefacts live under `RUNTIME_DATA_DIR`
  (`project_paths`), durable ones under `PERSISTENT_DATA_DIR`; every new path is a named
  `project_paths` constant (`DAY_REVIEW_DIR` under `PERSISTENT_DATA_DIR`, because TJ-2's bars
  files cannot be rebuilt after yfinance's 60-day window even though the index and the
  narrations inside it can; `AI_IDEAS_FILE`, `AI_IDEAS_STATE_FILE`). Write
  local first, the DAS after. A scratch script sets `TRADINGBOTV3_DATA_DIR` before any
  import and aborts if it resolves under `C:\TradingBotData`.
- **AI.** Local inference only in the off-hours window (`ai_offhours_start`/`_end`, today
  01:00–09:00 Pacific) — during a session a request is QUEUED for tonight, never run. Every
  model output is validated against a closed JSON schema whose text fields cite allowed
  `source_id`s; an output that cites an unknown id, or fails the schema, is rejected whole
  and the last verified file stays. Avoid a `maxLength` of exactly 2,000 on any field (the
  grammar-compile defect behind gate #144). New slots append INSIDE their decision-0018
  stage, set `max_attempts` (never 0) and a `reserve_minutes`, and are listed in
  `EXPECTED_SLOT_ORDER` in `tests/test_ai_jobs_runner.py`. Nothing an AI writes reaches a
  detector, score, alert, watchlist, Focus, the review queue, `review_policy.json` or
  `WISHLIST.md`.
- **Ask-first files this program stays out of:** `ui/panels/alert_center_panel.py`
  (`journal_chart_bars` is read, not edited), `autopilot_core.py`, `bounce_bot*`,
  `master_avwap*`, `legacy.py`, every `scripts/indicators/*`. A packet that cannot avoid one
  stops.
- **Tests.** Fail-before-fix. Qt tests run offscreen. The baseline (8,528 passed at the time
  of writing) stays green; `ruff` clean; smoke 7/7; `launch_gui.py --selftest` count
  compared when a new lazily-imported module lands (frozen-exe rebuild triggers 3/4).
- **Chat.** Messages to the trader follow the five-year-old rule; detail goes here, in
  `docs/DESK_INTERNALS.md` and in commit messages.

#### 12.4 Packets

##### TJ-1 — Bones: one Day Review page, no machine rows, fast reads, the daily forecast

**BUILT 2026-09-17** on `claude/tj1-day-review` (tester's 104 tests green without a weakened
assertion, plus three builder files for the seams they did not pin). Live gate **#145** below
is what remains owed; TJ-8 still deletes the two retired panel modules. Two amendments to
item 4 as built, both measured: the index covers every store over a megabyte (the two named
here plus `tier_outcomes` and `human_focus_outcomes`, which `read_session` opens for their
coverage line alone) while the six small trader-written stores stay live, and the two
per-decision walks inside `daily_recap_reader` became dict lookups - 13,500-16,500 ms across runs -> 820 ms
settle p50 on a staged home, so the gate's "under one second" holds. What landed, and what
each number turned out to be, is in DESK_INTERNALS "Day Review - one page, no machine rows,
a per-session index".

*Goal:* the trader opens one page, sees their day, and nothing waits on the Qt thread.

What exists (verified 2026-09-17):
- `ui/app.py:93-96` registers `PageSpec("Market Journal", …, "market_journal_panel")` and
  the Daily Recap page; `ui/app.py:787-830` `_record_auto_mode_flip` writes the
  `Auto mode X -> Y. Written by the desk, not the trader.` row with
  `origin=ORIGIN_AUTO_MODE_FLIP` and captures SPY charts for it. On the live desk those rows
  are **34 of 77** journal rows, and the 2026-09-16 nightly narration already repeats them
  ("Auto mode desk entries were recorded on 2026-09-16").
- `scripts/market_journal.py:56` `MACHINE_ORIGINS`, `:286` `is_machine_entry` — the flag
  exists; NO reader filters on it (grep on 2026-09-17: `market_story.py`,
  `market_story_rollups.py`, `market_thesis.py`, `market_journal_service.py` all read
  machine rows; the Mentor's previous-read, `MainWindow._previous_mentor_read`, filters to
  `origin == "trade_mentor"` itself and needs nothing).
- `scripts/daily_recap_reader.py:678-724` `_read_intraday_outcomes` streams the WHOLE
  `intraday_bounce_outcomes.csv` — **476 MB on the live desk** — on every open;
  `read_session` (`:1634`) opens nine files. `ui/panels/market_journal_panel.py:1120-1145`
  builds four `CandleChart`s on the first entry click (299 ms measured, G0).
- `ui/panels/market_journal_panel.py:1212-1247` "Paste weekly forecast…" →
  `MarketJournalService.import_weekly_forecast` (`:296`) → an entry with
  `origin=external_forecast` + `market_thesis.record_forecast` (`:697`, fields
  `target_week`, `scenarios`, `created_at_claimed`). The only reader is the Story pane's
  "External forecast" heading.
- `scripts/daily_recap_schedule.py` owns the noon and post-close refresh ticks
  (`daily_recap_auto_time`, `post_close_due_session`).

Changes:
1. `ui/app.py`: one `PageSpec("Day Review", "mdi.calendar-text", "day_review_panel")`
   replaces the Market Journal and Daily Recap specs; the `"Journal"` title guard at
   `app.py:685` is untouched. `_record_auto_mode_flip` becomes a one-line Auto Pilot log
   entry (the existing log seam) and writes NOTHING to the journal and captures NOTHING; the
   method stays so the caller does not change.
2. `market_journal.is_machine_entry` becomes the ONE filter, applied in
   `MarketJournalService.entries_about`, `market_story._entry_row` / `build_daily_story`,
   `market_story_rollups._stories_from_journal`, `MarketJournalService.theses_for`, and
   every pack TJ-4 builds. A test walks each reader with a
   fixture ledger holding one machine row and asserts absence.
3. New `scripts/ui/panels/day_review_panel.py` + `scripts/ui/services/day_review_service.py`
   with the six sections of 12.2. In TJ-1 the sections hold: (1) the deterministic
   `DailyStory` facts (`market_story.build_daily_story`) and the theses list read-only;
   (2) the existing `rejected_that_worked` rows until TJ-2; (3) notes + Mentor rows + New
   entry (`write_entry`, `origin=journal_page`) + Paste daily forecast; (4)
   `JournalStore.list_trades(trade_date=session)`; (5) one lazily built `CandleChart` of
   SPY M5 for the session from `alert_center.journal_chart_bars("SPY")` when the session is
   today and the desk has the bars, else the note "chart after the close" (TJ-2 brings the
   bars file); (6) an empty ideas card ("nothing yet").
4. Per-session compact index, new `scripts/day_review_index.py`: writes
   `DAY_REVIEW_DIR / "sessions" / <date> / "outcomes.json"` holding exactly the rows
   `read_session` keeps for that session (latest per `event_id`) plus the horizon rows in the
   lookback window and the coverage block; written by the post-close tick and lazily by the
   worker for any session opened without one; a file whose horizon rows were pending and
   have since matured is rebuilt. `read_session` takes the index when present. Test: for a
   fixture CSV the indexed answer is byte-equal to the streamed one.
5. Forecast: the button reads "Paste daily forecast…"; the dialog asks for the text and
   the session it is about (default: the date parsed from the brief's first heading, e.g.
   `Thursday, September 17, 2026`, else the page's session) and the source model (default
   `chatgpt`); the service method is `import_daily_forecast`; `record_forecast` gains
   `target_session` (old rows keep `target_week`); a second paste for the same session writes
   a new row with `supersedes`. The whole text is stored verbatim. A small deterministic
   reader `scripts/forecast_brief.py` pulls, by heading match only: the title date, the
   `Intraday playbook` section (its **bullish continuation** and **bearish reversal**
   paragraphs), the `Bottom line` section and its ranked-signals line, and every
   `Turbulence: N/10` number; anything not found is null, never guessed. The example the
   trader pasted on 2026-09-17 is the fixture
   `tests/fixtures/day_review/forecast_2026-09-17.md` and the reader's golden test. Nothing
   reads the forecast during the session except the page's own "External forecast" block;
   the overnight story reads it (TJ-4).
6. Move, not delete: the Review tab's widgets and worker
   (`daily_recap_panel.py:159-236, 434-462, 765-813`) become a "Measured report" section on
   Research > Results; the Staged picks table (`daily_recap_panel.py:333-343, 968-998`) moves
   to the Auto Pilot page. Their tests move with them.
7. The old panel modules stay on disk, unregistered, until TJ-8.

Tests: retarget `test_qt_market_journal_page.py`, `test_r4_market_journal_page_and_tables.py`,
`test_v2_market_journal_one_box.py`, `test_g3_market_journal_reader.py`,
`test_ws_dr_daily_recap.py`, `test_daily_recap_repair.py`, `test_daily_recap_auto_populate.py`
to the new page or to the moved sections; add the machine-row absence test, the index
byte-equality test, the forecast-per-session test, and a `PageSpec` list test.

Live gate **#145**: after restart the left nav shows Day Review and neither old page; opening
Day Review on a completed session paints in under one second once its index exists and the
first entry click in under 300 ms (bench numbers in the handoff); no `[desk]` row is visible
anywhere and a mode flip adds no journal row; "Paste daily forecast…" stores a forecast for
the chosen session and shows it under External forecast; the trader can add a note from the
page and from the desk tab and both appear. **TJ-1L (2026-09-18) adds one clause:** the page
reads in TWO COLUMNS - *What happened* / *Open theses* / the SPY pane left, the entries list
over the reader with *New entry* and the forecast right, *Walk-away* as a 2 x 2 grid and
*What you traded* beside the ideas card - every table fills its cell with its headers whole
("Against me first %", never "ainst me first"), and the column split is where the trader left
it after a restart.

##### TJ-2 — Instant walk-away

*Goal:* every decision the trader made that day, and what the name did after.

What exists: `daily_recap_reader._decisions` / `_decision_rows` (`:1022-1320`, grain
`(session_date, symbol, side, category, verdict, timeframe)`), `REJECT_VERDICTS`
(`:65-71`), `_rejected_that_worked_view` (`:1467-1545`), `_after_decision_favorable_pct`
(`:1423-1459`, only where a pass sidecar exists), the Journal join through
`preference_trade_outcomes.REPORT_FILE` (`:157-168, 1553-1566`, states `matched`,
`window_open`, `no_match_after_window`, `journal_unavailable`, `matching_unavailable`),
`journal_walkaway.run_walkaway_analysis` (D1, ATR-based, Weekend Prep only). `RecapSources`
(`:137-186`) does NOT read `claimed_picks.jsonl`.

Changes:
1. New pure `scripts/walkaway_day.py`: `build(session, sources, bars) -> WalkawayDay` with
   four populations: **A liked, not traded** (annotation `like_claim` quick or claimed,
   `pick_feedback` like, `swing_favorites` add, `claimed_picks`; excluded when
   `JournalStore.list_trades(trade_date=session, symbol=…)` holds a same-direction trade or
   the preference report says `matched`); **B rejected** (`REJECT_VERDICTS` incl.
   `m5_click_away`); **C traded, left early** (each closed same-day trade: exit stamp = the
   last closing leg; "left on the table" = favourable move from the first completed M5 bar
   after the exit to the close; a swing trade closed that day also gets the D1 5-session
   forward from `journal_walkaway`); **D claimed D1 picks** (`claimed_picks.jsonl`, graded at
   its own `claim_horizon` from the session-horizon outcomes; pending until matured).
   Columns: Time · Symbol · Side · What you did · Ran after % · Held at close % · Traded?
   · You made (R when planned risk is known, else net P&L) · Left on the table % · State
   (`measured` / `pending <date>` / `unmeasured <reason>`). Sorted by Ran after. Headline per
   table: `n` and the median Ran after; no other statistic is invented here — anything more
   goes through `held_run_score` / `swing_headline`.
2. **Session bars** `scripts/day_review_bars.py`: on the post-close tick, ONE batched
   `yfinance` 5-minute download (chunks of 50, the PCT-1 pattern) for the day's decided
   symbols plus SPY, QQQ, IWM, VXX; written to `DAY_REVIEW_DIR / "bars" / <date>.parquet`;
   zero IB traffic; on a worker; a failed symbol is absent and its row `unmeasured`; a past
   session inside yfinance's 60-day 5-minute window is back-filled on demand when its page
   opens; nothing runs during the session. "Ran after" for every decision uses these bars
   from the first completed bar after the decision stamp (the `_after_decision_favorable_pct`
   rule, generalised); the pass sidecar path stays as a fallback.
3. `RecapSources` gains `claimed_picks`.

Tests: one fixture per population; a like the trader traded is absent from A and present in
C; a click-away is in B; a claim with an unmatured horizon is `pending` with the date; a
symbol missing from the bars file is `unmeasured` with a reason; totals across A–D equal the
decision count.

Live gate **#146**: the morning after a session, Day Review shows the four tables with
yesterday's decisions, a name the trader passed on and that ran shows its Ran after %, a
liked name they traded is NOT under "liked, not traded", and the bars file for the session
exists with one row per decided symbol (`trading_bot.log` shows one batched download).

##### TJ-3 — Charts with the trader's notes on them

*Goal:* *"show me when I commented on it so I can see exactly where I went wrong."*

What exists: `ui/widgets/candle_chart.py` `EarningsDropLines` (`:480-539`) draws pooled
`pg.TextItem`s at bar indices — the precedent; `scripts/chart_levels.py` emits
`{price, label}` dicts consumed by the chart (`:480`, `:512`); market-journal captures
(`scripts/market_journal_capture.py`, 160 M5 + 120 D1 bars per entry, joined by `entry_id`)
are still written and are the fallback bars for an entry's own symbol.

Changes:
1. A new overlay family `NoteMarkers` in `candle_chart.py`, same pooling discipline as
   `EarningsDropLines`, fed a payload built on the worker: `[{stamp, index, kind, label,
   ref_id}]` with kinds `note`, `mentor`, `forecast`, `like`, `pass`, `veto`, `click_away`,
   `claim`, `trade_open`, `trade_close`. A marker click emits `markerClicked(ref_id)`.
2. `scripts/day_review_markers.py` (pure) builds the payload for (a) the SPY chart: notes,
   Mentor answers, trades; (b) a name's chart: that name's decisions and trades. Bars come
   from the TJ-2 bars file; D1 toggle from the durable daily store
   (`market_story_rollups.load_index_bars`).
3. Day Review: the SPY chart always; a walk-away row click opens the name's chart in ONE
   reused widget beside it; clicking a marker scrolls "What you said" to that note.

Tests: payload index resolution for a stamp between bars (the LAST completed bar at or
before the stamp), an unknown stamp yields no marker, markers pool (count stable across
three renders), `markerClicked` carries the entry id.

Live gate **#147**: on yesterday's Day Review the SPY chart shows a marker at each note the
trader wrote, clicking it selects that note; clicking a passed name opens its chart with the
pass marker on the right bar.

##### TJ-4 — The overnight day story and the rolling D1 view

*Goal:* every morning, "what happened yesterday, what I thought, was I right, did I chase."

What exists: `market_story.build_daily_story` (deterministic facts for SPY/QQQ/IWM/VXX/TLT/
USO), `market_story_rollups` (weekly/monthly/quarterly packs, live on the desk),
`ai_jobs/market_story_narration.py` (Stage 2 slot, local medium model, allowed-source-id
grounding, last verified file kept on failure — the pattern to copy), the regime-shift
ledger (`market_context_ledger.py:38-69`, stream `market_regime_shifts`),
`d1_environment.jsonl`, the Trade Mentor rows (`mentor.slot_id`, `prompt_kind`),
`run_ai_jobs.py --slot <name> --force`.

Changes:
1. **Day pack** `scripts/day_review_pack.py` → `DAY_REVIEW_DIR / "sessions" / <date> /
   "pack.json"`, built by the post-close tick and by the nightly slot (one function, hash-
   stable): `trader_said` (notes + Mentor answers, machine rows never, each with `entry_id`
   and time), `forecast` (the pasted forecast for that session or null),
   `environment` (the day's regime shifts + the D1 label), `measured` (from
   `build_daily_story`), `walkaway` (TJ-2 counts per population + top three by Ran after),
   `trades` (count, wins/losses, net R or P&L, one line each), `mood` (TJ-7; empty until
   then). Every item carries a `source_id`.
2. **Slot** `day_review_narration` (`scripts/ai_jobs/day_review_narration.py`), Stage 2,
   appended after `market_story_narration` (`runner.py:775-786`), local medium model,
   `reserve_minutes=10`, `max_attempts=3`. Input: the pack for the session just closed plus
   the previous day's narration read-only. Output `DAY_REVIEW_DIR / "narration" /
   <date>.json`, schema `day_review_narration_v1`: `headline` (≤160), `what_happened`
   (≤1200), `what_you_thought` (≤600), `were_you_right` [{`claim`, `source_id`, `verdict`
   right|wrong|unresolved, `evidence_id`}], `chased_against_news` {`verdict` yes|no|unknown,
   `evidence_id`}, `process` (≤400), `sources`, `prompt_version`, `inputs_hash`, `model`.
   An unchanged hash skips the call. The forecast enters the pack as the `forecast_brief`
   fields (playbook, bottom line, turbulence), each with its own `source_id`, plus the
   verbatim text; `chased_against_news` is judged against the brief's stated bearish-reversal
   conditions and the trader's own notes and trades — the desk does not measure oil or the
   10-year, so a condition the desk cannot see is `unknown`, never assumed. A forecast absent
   → `chased_against_news.verdict = unknown`. A note absent → no invented thesis.
3. **Rolling D1 view** in the same slot, `d1_view_narration_v1` → `DAY_REVIEW_DIR /
   "d1_view.json"`: inputs are the D1-timeframe notes of the last `LATELY_SESSIONS` (20),
   `market_thesis.current_theses`, and the weekly pack's measured facts; output `belief_now`
   (≤600) and `open_theses` [{`claim`, `since`, `still_true` yes|no|unknown, `evidence_id`}];
   rebuilt only when a D1 note or thesis row changed.
4. **Page:** "What happened" shows the verified narration, else the facts and "no story
   yet"; a **Redo story** button runs `run_ai_jobs.py --slot day_review_narration --force
   --session <date>` on a worker PROCESS outside market hours, and inside them writes a
   `redo_requested` marker the nightly slot honours and says "queued for tonight".

Tests: `EXPECTED_SLOT_ORDER` extended; two pack builds hash equal; a narration citing an
unknown `source_id` is rejected and the prior file is byte-identical; a machine row never
enters the pack; no forecast → `unknown`; Redo inside the session queues and calls nothing.

Live gate **#148**: the morning after an overnight run the Day Review of yesterday opens on a
story with a headline, a were-you-right list whose every line names one of the trader's
own notes, and a chased verdict that cites the pasted forecast or says unknown; the job
ledger shows `day_review_narration` OK; the D1 view lists the trader's open theses.

##### TJ-5 — Week Review, Weekend Prep's first step

*Goal:* *"5 of these days collated into one tab … to see if I was right, to see if I chased
in bad news environments, and to compare what I actually said to what I did."*

What exists: `WeekReviewPage` (`weekend_prep_panel.py:287`, already step 1, reads on a
worker), `weekend_prep_service.STEP_IDS`, `evidence_stats.WEEK_SESSIONS = 5`,
`market_story_rollups` weekly pack (coverage note, sessions missing), `ai_jobs/synthesis.py`
(`weekly_synthesis`, gated, separate — untouched), the frontier providers in
`scripts/ai_summary.py` (`request_ai_summary` `:3810`; OpenAI Responses and Anthropic
Messages; the desk setting `qt_ai_summary_model_anthropic` names a model).

Changes:
1. `WeekReviewPage` becomes: five Day Review cards (headline · were-you-right tally · chased
   flag · said-vs-did line · small SPY chart, built once on open and reused), the week
   story, the week's walk-away totals (A–D summed with `n`), the week's kept ideas. Missing
   days are named, never padded.
2. **Slot** `week_review_narration` (`scripts/ai_jobs/week_review_narration.py`), Stage 2,
   after `day_review_narration`, runs on the last session of the exchange week; provider
   = **OpenAI** (trader, 2026-09-17: "chatgpt API will do it"), through the existing
   `openai` Responses path in `ai_summary.request_ai_summary` and its configured key;
   `ai_week_review_provider` defaults to `openai`, and with no key the slot falls back to
   local medium and says so in the ledger; inputs are the five packs, five narrations
   and the weekly rollup — never bars, never the lake; output `DAY_REVIEW_DIR / "week" /
   <W>.json`, schema `week_review_narration_v1`: `headline`, `what_happened` (≤1500),
   `were_you_right` {`right`, `wrong`, `unresolved`, three cited examples},
   `chased` [cited examples], `process_pattern` (≤600), `next_week_watch` (grounded in open
   theses only), `sources`. Fewer than three narrated days → a deterministic scaffold and
   "narrated K of 5". One call per week; cost is logged in the job ledger.

Tests: card count equals sessions in the week; a missing day is named; the slot refuses a
week with fewer than three narrations; provider fallback to local when no key; slot order.

Live gate **#149**: on Saturday Weekend Prep opens on Week Review with five cards and a week
story whose examples cite the trader's own notes; the ledger shows one frontier call.

##### TJ-6 — The desk's AI has a voice: ideas

*Goal:* *"the local AI can have a voice somewhere where it offers ideas of what we can
improve based on what it reads."*

Changes:
1. **Slot** `improvement_ideas` (`scripts/ai_jobs/improvement_ideas.py`), Stage 3, appended
   LAST, local medium model, `reserve_minutes=10`, `max_attempts=2`. Inputs: the last five
   packs and narrations, the week's walk-away totals, the mood/process fields when present
   (TJ-7), and a fixed, versioned 30-line description of the program's pages and fields
   (`IDEAS_PROGRAM_CARD`, checked in) so a `program` idea is about THIS program. Output up
   to three ideas per night: `{idea_id, kind: process|program, text ≤280, evidence:
   [source_id], first_seen, seen_count}`, appended to `AI_IDEAS_FILE`
   (`PERSISTENT_DATA_DIR / "ai_ideas.jsonl"`); an idea whose normalised text matches one from
   the last 60 sessions increments `seen_count` instead; a dismissed idea never returns; an
   idea without evidence is dropped.
2. **State** `AI_IDEAS_STATE_FILE` (`ai_ideas_state.json`): `{idea_id: {status: kept|
   dismissed, at}}` — the trader's clicks, the only writer is the card.
3. **Card** on Day Review and Week Review: the night's ideas, Keep / Dismiss; kept `program`
   ideas listed under "For WISHLIST — copy" (the trader pastes; the AI never writes
   `WISHLIST.md`); kept `process` ideas resurface on Week Review as "you kept this on
   <date>".

Tests: dedupe by normalised text; dismissed never re-emitted; no evidence → dropped; at most
three per night; slot order; the card's Keep writes state and nothing else.

Live gate **#150**: the morning after, Day Review shows up to three ideas each citing a
note or a walk-away row; Dismiss hides one for good across a restart; Keep on a program
idea shows it under For WISHLIST.

##### TJ-7 — Mood and process: the bones only

*Goal:* the schema and one cheap way to record it, so the data starts accumulating now.

Changes:
1. `market_journal.build_entry` gains optional `mood` (1–5), `state_tags` (≤2 from
   `ui/annotations/vocabularies/state_tags_v1.json`: calm, focused, rushed, fomo, tilted,
   bored, tired, confident — versioned like the veto vocabulary, codes never reused),
   `process` ({`followed_plan`: yes|no|partly|null, `note` ≤200}). Every reader tolerates
   absence. Never asserted by literal version in a test.
2. A two-click strip (mood 1–5 + up to two chips) on the Trade Mentor popup and the desk's
   journal tab; optional, never required, never asked twice for one row.
3. The day pack carries them (`mood` section); the day and week narrations may cite them
   ("you said rushed at 07:30 and passed on three names by 08:00").

Not in this packet: timed emotion prompts, mood-vs-outcome statistics (needs ≥20 sessions of
fields; a later phase, trader-directed).

Live gate **#151**: a note saved with a mood shows it on Day Review; the next story cites
it; a note saved without one is unchanged.

##### TJ-8 — Cleanup

After #145–#150 pass: delete `ui/panels/market_journal_panel.py`,
`ui/panels/daily_recap_panel.py` and `ui/panels/away_recap_panel.py`'s page class if
nothing else imports them (the phone digest path stays), their dead tests, and the four-
pane capture reader; update `docs/DESK_INTERNALS.md`, `CLAUDE.md`/`AGENTS.md` rules that
name the old pages, `docs/README.md`, and the packaging drift guard. No behaviour change.

#### 12.5 Order and dependencies

TJ-1 → TJ-2 (needs the page and the index) → TJ-3 (needs the bars file) → TJ-4 (needs the
pack inputs from TJ-1/2) → TJ-5 (needs day narrations) → TJ-6 (needs packs; may run
alongside TJ-5) → TJ-7 (independent after TJ-1; may run alongside TJ-3) → TJ-8. One packet
per branch off `main`, merged after review; the desk restarts only on trader direction.

#### 12.6 Trader actions this program needs

- DONE 2026-09-17: one example forecast pasted (now the fixture
  `tests/fixtures/day_review/forecast_2026-09-17.md`); the week story bills the OpenAI
  (ChatGPT) API.
- Restart the desk after TJ-1 merges; answer the 10:00 Mentor prompt on a normal day so
  TJ-4 has material; write notes from the desk tab as usual.

#### 12.7 Deliberately not in Phase 0.33

Order execution; any detector, score, alert or Focus change; new phone pushes (AWAY rule
unchanged); reading the research lake; local inference during market hours; AI writes to
`WISHLIST.md`, `review_policy.json` or any watchlist; a new left-nav page beyond the swap;
merging the two journal stores.

## 13. Definition of done

Phase 0.33 is done when the trader can open Day Review on any past session and, without
touching anything else, read what happened, what they thought, whether they were right,
what they skipped that ran, what they traded and left early, see it on the SPY chart with
their own notes on it, and find the same for the week on Saturday — with every sentence
the AI wrote traceable to a note, a bar or a row, and nothing the AI wrote having changed a
detector, a score, an alert, a pick or a policy. The broader roadmap's definition of done
(single reliable desk; point-in-time-correct data owned by one writer; champions promoted
only through §7; decision-support only) is unchanged.
