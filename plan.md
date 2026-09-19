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
| 0.33 The trader journal — Day Review, Week Review and the overnight voice | TJ-1 … TJ-8 | TJ-1 MERGED 2026-09-18 (`e00b734a`, reviewer GO after four rounds; live gate #145 owed at the next restart); TJ-1L (two-column layout, presentation only) BUILT 2026-09-18 on `claude/tj1l-day-review-layout`, unmerged; TJ-2 MERGED 2026-09-18 into local `main` (`d3ae3aff`; durable session bars and four pure tables; gates #152/#153 owed); TJ-3 … TJ-8 PLANNED; **TJ-9 … TJ-13 PLANNED 2026-09-19** (trader-approved after the 2026-09-18 review-loop audit: forced 09:00 trade labels, read grader + congruence, walk-away v2, report card, night re-budget; order in 12.5); **second-look amendments and TJ-14 / TJ-15 PLANNED 2026-09-19** (trader: "Yes add all of this" — prediction click, skill line against a base rate, tracked ideas, instrument-aware money lines, tag provenance, miss contrast, staleness line; the Mentor asks only for what the desk is missing) |
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

**TJ-2 MERGED 2026-09-18 (`main` `d3ae3aff`):** a closed session now has a
durable completed M5 parquet tape and the page reads its SPY bars; four walk-away
tables project durable decisions, claims and later trades. Gates #152 and #153 remain live.

What exists: `daily_recap_reader._decisions` / `_decision_rows` (`:1022-1320`, grain
`(session_date, symbol, side, category, verdict, timeframe)`), `REJECT_VERDICTS`
(`:65-71`), `_rejected_that_worked_view` (`:1467-1545`), `_after_decision_favorable_pct`
(`:1423-1459`, only where a pass sidecar exists), the Journal join through
`preference_trade_outcomes.REPORT_FILE` (`:157-168, 1553-1566`, states `matched`,
`window_open`, `no_match_after_window`, `journal_unavailable`, `matching_unavailable`),
`journal_walkaway.run_walkaway_analysis` (D1, ATR-based, Weekend Prep only). `RecapSources`
reads `claimed_picks.jsonl` at construction time, so a staged or restarted reader keeps its
own claim-history source.

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

**AMENDED 2026-09-19 (trader): the model narrates verdicts, it never makes them.** TJ-10
builds first. The pack gains `reads` (TJ-10's graded read rows) and `congruence` (its
lines), each with a `source_id`; every `were_you_right[].verdict` must EQUAL the verdict of
the read row its `evidence_id` names, and an output that disagrees with a measured row, or
grades a claim no read row carries, is rejected whole like an unknown `source_id`. The
report card (TJ-12) is the page's deterministic head; the story sits under it.

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

   **AMENDED 2026-09-19 (trader):** the provider default in item 2 is superseded — the
   large LOCAL model writes the week story on Saturday night, OpenAI stays a setting that
   is off, and live gate #149's "one frontier call" reads "one large-local call" (TJ-13
   item 7).
3. **AMENDED 2026-09-19 (trader): the week and the month.** Under the five cards, one
   deterministic strip of TJ-12's report-card lines re-cut by exchange week for the last
   four weeks and for the calendar month to date (same functions, longer window, `n` on
   every cell, a week under its floor named and not ranked). No new page, no model.

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

**AMENDED 2026-09-19 (trader): advice is checked, not just given.** A `process` idea must
name ONE measurable the desk already computes (a veto reason's count and real-miss rate, a
report-card line, a TJ-14 question's answer mix) or it is dropped like an idea without
evidence. Keeping it freezes a baseline — that measurable over the `LATELY_SESSIONS` before
the keep — in `AI_IDEAS_STATE_FILE`. Week Review then prints, for every kept idea, before
and after with both `n`, deterministically, until the trader retires it; under the floor it
says "too few to call". The model never grades its own advice.

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

##### TJ-9 … TJ-13 — closing the review loop (trader, 2026-09-19)

The 2026-09-18 read-only audit of the live stores found the loop *decide → say → trade →
judge → tell* broken at three links. Measured that night: 215 journal trades with ONE
confirmed setup tag, no note, no planned stop and one recalled answer ever; zero stored
theses against seven Mentor reads a day; ~95% of ~140 daily decisions made on D1 charts
while TJ-2 measures same-session M5 only; the local model's night spent ~5 of 8 hours on
`ai_summary` (unsynthesized on its last four runs) and `ticker_briefs` (152 of 222
membership-only). The trader approved every recommendation with one change: **trade
labelling is FORCED at 09:00 Pacific through the Trade Mentor**, not an optional card at
the close. Every premise below is recon's to re-verify (12.3).

##### TJ-9 — Yesterday's trades are labelled at 09:00, and the desk insists

*Goal:* every real trade carries a setup, a stop answer and one sentence by 09:05.

What exists: `trade_mentor_schedule.TRADES_HOUR = 10` (kind `m5_trades`),
`trade_mentor_trade_check` (`MATERIAL_FIELDS` thesis/stop/target/setup, the four
`ANSWER_STATES`, `TRADE_CAP_DEFAULT = 3`, `REASON_NOT_READY`, RECALLED annotation rows that
never touch `planned_*`), `journal_bulk_tag` provisional tags (33 live),
`preference_trade_outcomes` matches (a trade matched to a claimed like already names a
setup), the capped `journal_import` slot (three tries, all overnight).

Changes:
1. `TRADES_HOUR` moves to **9**; the 10:00 slot becomes a plain `m5` read.
2. **Forced** (lead's reading of the trader's word, overrule if wanted): the card lists
   EVERY trade of the previous session (no cap of three for that session; the cap stays for
   older backlog, which remains "Tag this week"); Save is disabled until each material field
   of each listed trade holds a value or one of the four explicit answer states; an
   unanswered card does not expire into silence — its trade section rides on every later
   hourly card that day and the Day Review head says `N trade(s) unlabelled`. AWAY still
   prompts nothing; the first DESK hour after it carries the section.
3. **One click per field.** Setup opens on the machine's best guess in lane order (the
   claimed like it matched, else the provisional tag) as a confirm button beside the
   vocabulary list; the confirm is the TRADER's write (`tag_status='confirmed'` through the
   Journal's own writer), never the machine's. Stop/target stay RECALLED rows, labelled.
4. **Journal freshness.** `journal_import` gains ONE post-open retry outside the model
   window (deterministic, seconds) when the night ended without an OK; Day Review and the
   Journal print `fills current to <date>`; when the journal is not ready at 09:00 the card
   SAYS so and the section rides to the next hour instead of asking nothing all day.
5. Questrade rows with `security_type = UNKNOWN` (18 of 18 in September) are classified by
   the importer; recon finds the seam first.

Tests: schedule tuple (09:00 `m5_trades`, 10:00 `m5`, early close, DST); Save disabled with
one field open and enabled by an explicit `not remembered`; the section rides to the next
slot exactly once per slot; a confirm writes `confirmed` and a machine guess alone writes
nothing; not-ready rides instead of vanishing; AWAY prompts nothing. `CLAUDE.md`'s Trade
Mentor rule changes WITH this packet (10:00 → 09:00, forced).

**AMENDED 2026-09-19 (trader, second look): a label knows when it was made.** A label
written the next morning knows how the trade ended. Every confirmed tag carries
`label_provenance`: `claimed_before_entry` (it matched a claim or like stamped before the
first fill), `same_session` (answered on a card the day of the fill, TJ-14 item 4) or
`recalled_after`. Every statistic over confirmed tags reports the three apart and pools
them only in a row that says so. The Process line also counts **planned vs unplanned**
trades — a trade with a like, claim, Focus pick or armed alert on that name and side
before its first fill, against a trade from nowhere — with what each group made and `n`.

Live gate **#154**: at 09:00 the card lists yesterday's trades with a suggested setup each;
Save stays grey until all are answered; skipping it brings the section back at 10:00; the
Journal then shows those trades confirmed.

##### TJ-10 — The read grader and the congruence line (no model)

*Goal:* *"if my thoughts about the market are incongruent with my D1 picture, I want to
know."* Builds BEFORE TJ-4.

What exists: `market_thesis.extract_thesis` (versioned stance vocabulary, exact source
spans, `unstated` for a hedge), Mentor rows with `mentor.prompt_kind` and `timeframe`,
`d1_environment.jsonl` (the desk's D1 label per benchmark), `market_story.build_daily_story`
facts, `journal_exposure` (bias from the legs), the day's likes by side. The thesis store is
empty (its only writer was the "Save interpretation" button TJ-1 retired).

Changes:
1. Pure `scripts/market_read_grades.py`: every non-machine journal entry of the session is
   extracted automatically; each stated stance becomes ONE read row `{read_id, entry_id,
   span, benchmark (SPY unless named), stance, clock}` with the clock set by kind — an M5
   read is graded to that session's close, a D1 read at 1, 3 and 5 sessions. Verdict
   `right | wrong | flat | pending <date> | unstated | unmeasured <reason>` against the
   benchmark's measured move from the first completed bar after the entry stamp; `flat` is
   a move inside a declared ATR band, recorded as a constant with its reason. Append-only
   rows under `DAY_REVIEW_DIR`, written by the post-close tick and re-graded nightly as
   horizons mature; a row is never rewritten, a matured grade is a new row naming the old.
2. **Congruence** (same module, three lines, each with its counts): the trader's latest D1
   stance vs the desk's D1 label; vs the long/short mix of that session's likes and claims;
   vs the bias of that session's fills. A line with a missing side says which side is
   missing. No threshold turns a line into an alert; it is printed, never pushed.
3. Day Review's "What you said" shows each read's verdict chip beside the entry.

Tests: span reproduces the stance word; a hedge is `unstated` and gets no grade; a D1 read
is `pending` with its date until the fifth session closes; completed bars only; the
congruence counts equal the annotation rows; opposite-benchmark notes never grade SPY.

**AMENDED 2026-09-19 (trader, second look): the graded thing is a CLICK, the words are
context.** Measured that day with `extract_thesis` on the trader's 42 real notes: 21
`unstated`, 8 bullish, 7 neutral, 6 bearish — and *"D1 SPY is still downtrending"* read as
`unstated`. The trader describes the tape far more often than they predict it, so:
4. Every Mentor card carries ONE forced prediction click (TJ-14 owns the card):
   `Rest of day: Up / Down / Chop / No view`; the 08:00 and 12:00 D1 cards add `Next 5
   sessions: Up / Down / Range / No view`. `No view` is a complete answer and is never
   graded. The click is stored on the entry's `mentor` payload and is the read row's stance
   whenever it exists; extraction grades only entries that have no click (the history, and
   notes typed outside a card) and is labelled `extracted`.
5. `Chop` / `Range` is right when the move stays inside the declared ATR band (item 1's
   `flat` band, one constant).
6. The vocabulary gains the trader's own trend words found `unstated` in the live notes
   (*downtrending, rejecting, leaking, lower highs, holding lows* …) as a NEW extractor
   version; old rows keep theirs.

Live gate **#155**: the morning after a session each Mentor answer with a view shows
right / wrong / flat, a D1 view shows pending with its date, and the congruence line's like
counts match the day's likes.

##### TJ-11 — Walk-away v2: a swing ruler for swing calls

*Goal:* *"stocks I said no to that went on to have great moves that day or the next day."*

What exists: `walkaway_day._after_move` (same-session best excursion only;
`held_at_close_pct` filled for claims only), the nightly veto / like / pass / rejection
cohort outcome rows at H1/H3/H5/H10, the durable daily store, `ui/annotations/store.
_session_date_text` (New York calendar date, so a call after 21:00 Pacific is stamped the
next calendar day — Friday evening's calls carry a Saturday).

Changes:
1. A fifth table **Earlier calls, now**: the D1 likes, claims and vetoes of the previous
   five sessions with their side-adjusted move to the selected session's close, from daily
   bars; most-ran first; `pending` never zero.
2. Every row gains **Against you first** and **At the close**, and the three moves are
   shown in ATR beside percent (ATR from the daily store; missing ATR is `unmeasured`).
3. **A real miss is a rule, not a glance:** `REAL_MISS_V1` = ran ≥ 1.0 ATR before going
   0.5 ATR against, completed bars only, versioned, one constant, reported never acted on.
4. One deterministic sentence above each table: `You vetoed 92. 7 were real misses; 4 share
   the reason extended.` Reasons come from the veto vocabulary; overlapping codes are never
   summed.
5. **Session stamp:** a decision made outside a session is stamped with the NEXT exchange
   session from the calendar, never a weekend date. Recon first measures how the cohort
   graders treat today's Saturday-stamped rows; existing rows are never rewritten (the
   reader maps a non-session date forward).

Tests: a pop-then-fade is not a real miss; an adverse-first name is not a miss; a Friday
21:30 Pacific call reads as Monday's; five-session window on the exchange calendar across a
holiday; totals still equal the decision count.

**AMENDED 2026-09-19 (trader, second look): a miss needs a base rate, and a trade needs the
right ruler.**
6. **The skill line.** For the session, and for `LATELY_SESSIONS`, the real-run rate
   (`REAL_MISS_V1`'s rule, same clock) of three populations of the SAME scan: names the
   trader liked or claimed, names they rejected, and names the scan showed that they never
   touched — each with `n` and the ONE Wilson interval, cut by side and, where `n` allows,
   by setup family. "Likes 21% (n 55), vetoes 8% (n 92), untouched 9% (n 310)" is the
   SHAPE of the sentence (invented numbers); overlapping intervals are SAID to overlap. The untouched population comes from
   the session's scan rows through the day-review index, never the 1.1 GB tracker; recon
   names the seam. Reported, never acted on; no R statistic selects what is shown.
7. **Instrument-aware rows.** An option trade and a position held past five sessions are
   never given "left on the table today". An option row shows premium kept, days held and
   assigned or not, read from the legs (`journal_exposure`'s rule: a long option is never a
   bullish setup); a long hold shows days held and its open or realised result. Anything
   the desk cannot judge reads `not judged here: <reason>` — never a wrong number.
8. **Every money line carries its `n`** and reads "too few to call" under
   `MIN_REPORTABLE_N`. Measured 2026-09-19: 50 trades since 08-04, 8 of them options, and
   15 of 37 closed trades held past five sessions.

Live gate **#156**: Monday's Day Review shows Friday evening's calls, a vetoed name that ran
two ATR by Wednesday is on Wednesday's Earlier-calls table, and each table has its sentence.

##### TJ-12 — The report card

*Goal:* *"what I missed very apparent, what I did well very apparent."*

Five deterministic lines at the head of Day Review, above the story, each with its `n` and
each clickable to the table behind it: **Did well** (likes/claims that were real runs, best
family by the ONE Wilson bound, none named under `MIN_REPORTABLE_N`), **Missed** (TJ-11's
real misses and their shared reason), **Your reads** (TJ-10 tally), **Congruence** (TJ-10),
**Process** (trades, labelled or not per TJ-9, left on the table). Pure
`scripts/day_report_card.py` over the day pack; computes no new statistic; a line whose
input is missing says so. TJ-5's strip re-cuts the same lines by week and month.

**AMENDED 2026-09-19 (trader, second look):** the **Did well** and **Missed** lines quote
TJ-11's skill line, so a count never stands without its base rate; and a sixth, smaller
line **How fresh** states what the card rests on — story written when, fills current to
which date, grades through which session, and any slot that failed last night by name (the
job ledger's last row per slot). A failed night is said on the page the next morning.

Live gate **#157**: yesterday's Day Review opens on five lines whose numbers match the
tables under them; with no trades the Process line says so rather than printing zero.

##### TJ-13 — The night works for the trader first (amends decision 0018's slot order)

What exists: `ai_summary` ~4 h nightly with a 900 s synthesis read timeout that ended four
straight runs unsynthesized; `ticker_briefs` 64 min for 70 written briefs in which
"truncated" appears 88 times; `journal_enrichment` rejecting its own schema on every try.

Changes: inside Stage 2 the cheap trader-facing slots run FIRST (`market_story_narration`,
`day_review_narration`, then `week_review_narration`), then `ticker_briefs`, then
`ai_summary` last with the remaining window; `ticker_briefs` writes only names with session
evidence and its evidence package stops truncating the two sources it cites most (recon
measures which); `ai_summary`'s synthesis gets a bounded input and a timeout it can meet, or
degrades at once instead of after four hours; the `journal_enrichment` schema failure is
reproduced and fixed. `EXPECTED_SLOT_ORDER` and decision 0018 change together. Stage
boundaries (deterministic → digest → narration → model-gated) do not move.

**AMENDED 2026-09-19 (trader): nights only, every day of the week, and the weekend nights
do the heavy work.** *"I always want the bot to run overnight never during the day so I can
restart it or use it for market prep."* The desk stays on through the weekend. Measured:
the `TradingBotV3 AI Jobs` task already fires 22:00–06:00 Pacific seven nights a week as its
own process (a desk restart never kills it), but Saturday's and Sunday's firings skip every
slot, `weekly_synthesis` has never run (it needs a typed command), and
`ai_local_model_large` (the 27B) is configured and used by nothing.

5. `window.py` loses "weekends are open all day": the configured night window applies
   seven days a week, so no run — scheduled, forced or from a page button — starts local
   inference by day. A daytime request is queued for tonight (TJ-4's rule, now general).
   The one standing exception stays the Trade Mentor's seconds-long draft on the trader's
   own reply (CLAUDE.md, quiet hours); the trader may remove it.
6. **Weeknights** run only the short trader-facing slots plus the deterministic stage.
   **Saturday night** (the first night with no session behind it) runs the weekly slate:
   `ai_summary` once a week instead of nightly, `weekly_synthesis` without a typed command,
   `week_review_narration`, the month re-cut. **Sunday night** runs the backlog: a retry of
   any day story that failed that week, SUGGESTED setup tags for old untagged trades
   (suggestions only — `journal_bulk_tag`'s provisional rule, never a confirm), and a short
   week-ahead note ready before Monday's open. A slate that does not finish resumes the
   next night; nothing runs past the window's end.
7. **The week story is written by the large LOCAL model** (trader, 2026-09-19, replacing
   TJ-5's OpenAI default): `ai_week_review_provider` defaults to `local_large`, falls back
   to local medium and says so in the ledger; `openai` remains a setting, off. It may only
   narrate measured rows (TJ-4's amendment applies). The builder first measures the 27B on
   this desk — load time, tokens per second, peak memory beside a running desk — on a copy
   of one week's packs, and the slot's `reserve_minutes` comes from that number.

Live gate **#158**: the ledger shows the day story finished before 23:30 Pacific, the
morning brief lists no membership-only name, `ai_summary` is either synthesized or failed
fast and runs on Saturday night only, no ledger row starts between 06:00 and 22:00 Pacific
on any day, and Sunday's Week Review shows a week story whose ledger row names the large
local model.

##### TJ-14 — The Mentor asks for what the desk is missing, and nothing else

*Goal (trader, 2026-09-19):* *"We don't need to run every question every hour but if we need
more data make trade mentor ask me for it. I'm happy to click boxes or give my responses
but then I expect the AI to take it from there."*

What exists: `trade_mentor_schedule` (hourly `m5`, `m5_d1` at 08:00 and 12:00, the trade
check), `ui/widgets/trade_mentor_card.py`, `TradeMentorService` (slot state, pause, AWAY
silence), `trade_mentor_trade_check`'s rule *only what is missing* and its four
`ANSWER_STATES`, `market_story_narration`'s `mentor_question` (one AI question on the next
card). Fills reach the journal only at night; by day the pull is a button.

Changes:
1. **The hourly card shrinks to one forced click** — TJ-10's prediction — **plus optional
   words.** A card answered with one click is a complete answer (lead's reading of "we
   don't need to run every question every hour"; overrule if wanted).
2. **A question registry**, pure `scripts/mentor_questions.py`. Each kind declares its
   TRIGGER (a measured gap, never a clock alone), its click options, the store it writes,
   its cadence and expiry, its priority and its **CONSUMER** — the reader that uses the
   answer. A kind with no consumer fails a test: nothing is asked that nothing reads. v1:
   - `prediction_m5` / `prediction_d1` — every card / the two D1 cards → TJ-10.
   - `trade_label` — TJ-9 at 09:00, and SAME SESSION when a new fill is seen (item 4), which
     is the label made before the outcome is known → TJ-9 provenance.
   - `trade_origin` — a trade with nothing before it: `Planned off the desk / An alert /
     Impulse / Other` → the planned-vs-unplanned Process line.
   - `open_position_check` — a position past five sessions, once a week per position:
     `Thesis intact / Weakening / Exit planned` → TJ-11's long-hold rows.
   - `quick_like_followup` — a QUICK like that was then traded or became a real run, once:
     `Which setup was it?` from the claim vocabulary, stored as a recalled claim LINK, never
     rewriting the like → like cohorts by family.
   - `day_close` — the session's last card only: `Followed the plan: Yes / Partly / No` and
     TJ-7's mood strip (TJ-7 builds the fields; the question lives here) → Process line.
   - `grader_gap` — any deterministic reader may emit `unmeasured: needs_trader_input` with
     a question id; the registry turns it into a click → the reader that raised it.
   - `ai_question` — at most ONE a day from the overnight run (the existing
     `mentor_question`, now with optional closed click options) → the next day's pack.
3. **A budget.** Beyond the prediction click a card carries at most THREE questions, by
   priority (`trade_label` first); the rest are counted on the card and carried, never
   dropped and never a fourth. Every question offers the four answer states plus `Stop
   asking this`, which retires that kind for that subject. AWAY asks nothing.
4. **Fills by day.** Before each card one light journal pull runs on a worker: IBKR
   executions through the existing connection; Questrade ONLY through the single
   refresh-chain owner under `local_writer_lock`, capped per day. Recon rules on the token
   chain first — if an hourly pull endangers it, Questrade trades stay next-morning and the
   card says so. No pull ever blocks a card.
5. **"The AI takes it from there."** After the clicks no manual step stands between an
   answer and the report card. Removed or automated by this program: "Save interpretation"
   (TJ-1), the typed `--weekly-synthesis` (TJ-13), "Tag this week" for any trade made after
   TJ-9 ships (it remains for the backlog). Manual BY DESIGN and staying so: the
   `review_policy` sign-off, `digest approve-audit`, the Questrade token repair.

Tests: a kind without a consumer fails; budget of three with the remainder counted; `Stop
asking this` retires one subject only; a same-session label carries `same_session`; AWAY
asks nothing; a failed pull leaves the card on time; the quick-like follow-up writes a link
and no tag.

Live gate **#159**: an ordinary hourly card is one click; on a day with a new fill the next
card asks its setup and stop; a trade from nowhere is asked its origin once; nothing is
asked twice after `Stop asking this`.

##### TJ-15 — What the misses had in common (no model, no chart pictures)

*Goal:* the bot says WHY, not only what. The trader asked for an AI that "looks at each
chart"; the desk already MEASURES each chart at the scan the decision was made on, and a
measured contrast is evidence where a model's glance at a picture is not.

What exists: `d1_features_history.csv` (one row per symbol per scan, ~700 MB — streamed by
session, never loaded whole), the compression measure and `compression_calibration` (the
precedent: one feature, AUC between vetoed-and-ran and the rest, a CSV, nothing modified),
TJ-11's `REAL_MISS_V1`, the veto vocabulary.

Changes: a deterministic Stage 1 slot `miss_contrast`, after the cohort graders. For each
veto reason and for likes, over `LATELY_SESSIONS`, it compares the point-in-time features
of **real misses** against **correct rejections** of the same reason (and real runs against
duds among likes): per feature the two medians, `n` on both sides, and one rank statistic
(AUC, the calibration tool's). It prints the few features that differ most and says how
many were compared (`observational, top 3 of K`), names nothing under the floor, and
writes one JSON pack beside the digest. Week Review shows it as a table under the misses;
the week story may narrate it and may not add to it. Fundamentals stay what the desk has —
earnings dates and the pasted forecast — and the page says so. Zero detector, score, alert
or policy influence; a threshold change stays a separate, ask-first request.

Live gate **#160**: Saturday's Week Review shows, for the veto reason with the most real
misses, a table of what those misses measured differently, each row with both `n`.

#### 12.5 Order and dependencies

**AMENDED 2026-09-19 (twice):** TJ-9 (independent; first, because every personal statistic
waits on labelled trades) → TJ-14 (the card and the registry; TJ-9's section becomes its
first kind, and the prediction click starts accumulating at once) → TJ-10 → TJ-11 (may run
alongside TJ-10 and TJ-3) → TJ-4 (now needs TJ-10) → TJ-12 → TJ-15 (needs TJ-11's
real-miss rule) → TJ-13 (independent; any time after TJ-4's slot exists) → TJ-5 → TJ-6 →
TJ-7 (fields only; its strip is TJ-14's `day_close`) → TJ-8. The original chain below still
orders TJ-3 … TJ-8 among themselves. **No packet after TJ-9 merges until the desk has been
restarted once and gates #145, #152 and #153 have been read on a real session** — the new
layers stand on those pages.

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
- SUPERSEDED 2026-09-19: the week story is written by the large local model (TJ-13 item 7);
  OpenAI stays a setting, off.
- From TJ-14 on: one prediction click per Mentor card, and the few extra clicks the card
  asks for. Everything after the click is the desk's job.

#### 12.7 Deliberately not in Phase 0.33

Order execution; any detector, score, alert or Focus change; new phone pushes (AWAY rule
unchanged); reading the research lake; local inference during market hours; AI writes to
`WISHLIST.md`, `review_policy.json` or any watchlist; a new left-nav page beyond the swap;
merging the two journal stores.

## 13. Definition of done

Phase 0.33 is done when the trader can open Day Review on any past session and, without
touching anything else, read what happened, what they thought, whether they were right,
what they skipped that ran, what they traded and left early, see it on the SPY chart with
their own notes on it, read five measured lines that say what they did well, what they
missed against a base rate, whether their reads were right, whether their view matched
their picks and what their misses had in common (TJ-9 … TJ-15), having given the desk
nothing but clicks and a few words on the Mentor card, and find the same for the week and the month on Saturday — with every sentence
the AI wrote traceable to a note, a bar or a row, and nothing the AI wrote having changed a
detector, a score, an alert, a pick or a policy. The broader roadmap's definition of done
(single reliable desk; point-in-time-correct data owned by one writer; champions promoted
only through §7; decision-support only) is unchanged.
