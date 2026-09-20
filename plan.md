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
- **2026-09-19:** TJ-1 and TJ-2 are MERGED and not yet live (the desk has not been
  restarted; gates #145, #152, #153 owed); TJ-1L is MERGED too (its branch tip is an ancestor of `main`). A read-only audit of
  the live stores on 2026-09-18 and a day of trader decisions on 2026-09-19 added TJ-9 …
  TJ-16 and dated amendments to TJ-4, TJ-5, TJ-6, TJ-7 (see 12.4; decision 0021 answers
  13–30). The trader's one-line definition of the product, 2026-09-19: a bot that *"takes
  in what I do and think … mathematically deduces what parts of my thinking are profitable
  and unprofitable, and then effectively communicates what to keep doing and what to
  change."* Build order: 12.5. The assessment itself is an artifact, not a file:
  `https://claude.ai/artifact/PyLk4d7N2cWfjf83NWsh7n`.

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
| 0.33 The trader journal — Day Review, Week Review and the overnight voice | TJ-1 … TJ-8 | TJ-1 MERGED 2026-09-18 (`e00b734a`, reviewer GO after four rounds; live gate #145 owed at the next restart); TJ-1L (two-column layout, presentation only) MERGED (`86b86bcb` is an ancestor of `main` - verified 2026-09-19 with `git merge-base --is-ancestor`; this row said "unmerged" in error); TJ-2 MERGED 2026-09-18 into local `main` (`d3ae3aff`; durable session bars and four pure tables; gates #152/#153 owed); **TJ-3 MERGED 2026-09-19 (evening)** (`claude/tj3-note-markers` → `lead/p033-integration2` `72647104`; Day Review note markers, a mark on a bar only when it happened during it, the Alert Center's chart proven unchanged; gate #147 owed, after #146/#152's bars back-fill); **TJ-15 MERGED 2026-09-19 (night)** (`claude/tj15-miss-contrast` → `lead/p033-integration2` `fb3f55e9`, slot position fixed `1f260ffa`; the pure `evidence_contrast` with two floors, the deterministic `miss_contrast` slot inside stage 1 above the pair that closes it, D1 decisions only; gate #160 owed) and **TJ-14A MERGED 2026-09-19 (night)** (`claude/tj14a-mentor-card` → `lead/p033-integration2` `e8c04f88`; TJ-14 items 1 and 6 - the Mentor card's What I see / What I expect split with a forced prediction click, a row's timeframe and its horizon always agreeing at the WRITER, `trade_mentor_context_v2` and the internals strip; TJ-14B holds items 2-5; gate #159's first clause owed); TJ-4 … TJ-8 PLANNED; **TJ-11 MERGED 2026-09-19** (`claude/tj11-walkaway-v2` → `lead/p033-integration` `a89ec7d5`; walk-away v2, `REAL_MISS_V1`, the skill line, an additive `decision_session`; direction reversed the same evening by **TJ-11F MERGED 2026-09-19** (`claude/tj11f-decision-session` → `lead/p033-integration2` `f00ec302`; an after-close decision belongs to the session it JUDGED); gate #156 owed, reworded) and **TJ-13A MERGED 2026-09-19** (`claude/tj13a-night-slates` → `lead/p033-integration` `9eaae1dd`; nights only seven days, night slates, four overnight repairs; gate #158 owed); **TJ-9 … TJ-13 otherwise PLANNED 2026-09-19** (trader-approved after the 2026-09-18 review-loop audit: forced 09:00 trade labels, read grader + congruence, walk-away v2, report card, night re-budget; order in 12.5); **second-look amendments and TJ-14 … TJ-16 PLANNED 2026-09-19** (trader: "Yes add all of this" — prediction click, skill line against a base rate, tracked ideas, instrument-aware money lines, tag provenance, miss contrast, staleness line; the Mentor asks only for what the desk is missing) |
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

And on 2026-09-18 / 09-19, after the audit:

- *"every day we make decisions, and we want to evaluate how good the decisions we're
  making are … if they said no to a bunch of stocks that went on to have great moves that
  day or the next day, then I want to know about it."*
- *"if my thoughts about the market are potentially incongruent with my overall D1 picture,
  I want to know about it."*
- *"I want what I missed to be very apparent. I want what I did well with to also be very
  apparent. I want this to be very, very automated, but still based in reality … most of
  this isn't using AI. It's using objective Python programming with a dash of an AI
  overnight."*
- *"I want to be forced to label my trades around 0900 as per trade mentor."*
- *"I always want the bot to run overnight never during the day so I can restart it or use
  it for market prep."* The desk stays on through the weekend.
- *"We don't need to run every question every hour but if we need more data make trade
  mentor ask me for it. I'm happy to click boxes or give my responses but then I expect the
  AI to take it from there."*
- *"make sure we differentiate predictions from just 'describe the market and your
  thoughts'! The hope is an AI can pickup on my tendencies and what leads to good
  predictions and what leads to wrong ones."*
- *"trade mentor should automatically be processing what's going on with the internals we
  watch. RSP VXX USO TLT and the sector ETFs XLK XLE etc. so the AI already has that."*

Of the twelve answers below, one is superseded: the week story is written by the large
LOCAL model, not a frontier model (TJ-13 item 7, decision 0021 answer 20).

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
- **AI.** Local inference only in the off-hours window (`ai_offhours_start`/`_end`, stored
  in ET: today 01:00–09:00 ET, which is **22:00–06:00 Pacific** — corrected 2026-09-19, this
  line used to say Pacific), **at night only and seven days a week** (trader, 2026-09-19;
  TJ-13 item 5) — by day a request is QUEUED for tonight, never run. Every
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
  **A packet that adds or MOVES a runner slot runs `-k "slot or stage or slate"` plus
  `tests/test_veto_cohort_grading.py`, `tests/test_ws_10d_market_story.py` and
  `tests/test_ws_rp_shared_report.py`, not only `tests/test_ai_jobs_runner.py`** — TJ-15's
  targeted runs were green while three order assertions in those files were red.
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

**TJ-3 BUILT and MERGED 2026-09-19** (`claude/tj3-note-markers`, tip `58ee11f4`, merged
`72647104` into `lead/p033-integration2`; two reviews, both GO by reproduction, the second
after one fix round). All three changes below landed. (1) `NoteMarkers` plus pooled glyphs
in `candle_chart.py`, **additive** to the live Alert Center's chart - proven byte-for-byte
at base and tip (scene items, view range, earnings glyphs, ribbon span, full-widget PNG
sha256 and the click sequences identical; with no markers set nothing is built) - with
`set_note_markers` / `note_marker_count` / `note_marker_position` / `note_marker_at` and
`markerClicked(ref_id)` emitted IN ADDITION to the existing signals. (2) The pure
`scripts/day_review_markers.py`: `placement_for` / `bar_index_for`, `benchmark_markers`,
`symbol_markers`, `name_charts`, `placement_counts`, eleven `MARKER_KINDS` (the plan's ten
plus `prediction`). **What differs from the plan text above:** the plan said a stamp
between bars takes "the LAST completed bar at or before the stamp" - it does not. **A mark
sits on a bar only when it happened DURING that bar**; a stamp past the end of the tape is
`after_tape` with `index: None`, KEPT, counted on the worker and SAID in the caption, never
clamped onto the last candle (the review measured 62 of 216 live trade legs, 29%, filling
after 13:00 Pacific and being drawn on the 12:55 candle), and a stamp in a hole is
`between_bars` the same way; `before_tape` / `no_tape` / `unreadable` yield no marker at
all. **The packet's D1 toggle is a pure-builder capability only - the page has no toggle
and this packet added none**, so the weekend/holiday stamp on a daily tape (`between_bars`)
is a decision owed knowingly if one ever lands. (3) The page: `spy_markers`, `name_charts`
and `spy_marker_placements` in `PAYLOAD_KEYS` / `empty_payload`, all built ON THE WORKER at
the end of `read_day` from rows already read; the SPY chart always with its caption, a
walk-away row click opening that name in ONE reused `CandleChart` beside the tables, a
marker click selecting that note in "What you said". Claim markers come from the session's
`claimed_picks` rows through `symbol_markers(..., claims=)`. Live gate **#147** owed, and it
needs gate #146/#152's back-fill first - the live home has no `day_review/bars/*.parquet`
yet. **Recorded as a LATER packet:** `name_charts` carries a tape per decided name (~1.0-1.5
MB for a 174-name session) and could be trimmed to the rows the tables show. Long form:
DESK_INTERNALS "TJ-3".

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
trader wrote (and at each Mentor answer, the pasted forecast and both legs of every trade of
the day), clicking one selects that note in "What you said"; clicking a passed name in a
walk-away table opens that name's own session chart beside the tables with the pass marker
on the right bar, and clicking a second row re-uses the same chart widget. **A mark the tape
could not carry is COUNTED, not drawn:** a trade leg filled after the last completed bar
(29% of the live journal's legs fill after 13:00 Pacific) appears on NO candle and the
caption under the chart says how many - e.g. `2 marks after the tape - not drawn.` Nothing
on the Alert Center's charts changed. **The bars file must exist for the session**
(`day_review/bars/<date>.parquet` - absent on the live store when this was built, so gate
#146/#152's back-fill is a prerequisite: with no file the SPY chart draws from the Qt
hand-off, the caption says the marks were not drawn, and no name chart can open).

##### TJ-4 — The overnight day story and the rolling D1 view

*Goal:* every morning, "what happened yesterday, what I thought, was I right, did I chase."

**AMENDED 2026-09-19 (trader): the model narrates verdicts, it never makes them.** TJ-10
builds first. The pack gains `reads` (TJ-10's graded read rows) and `congruence` (its
lines), each with a `source_id`; every `were_you_right[].verdict` must EQUAL the verdict of
the read row its `evidence_id` names, and an output that disagrees with a measured row, or
grades a claim no read row carries, is rejected whole like an unknown `source_id`. The
report card (TJ-12) is the page's deterministic head; the story sits under it. The pack also
gains `internals` (TJ-14 item 6: the open, each Mentor hour and the close), `skill` (TJ-11
item 6) and `report_card` (TJ-12), each with `source_id`s; `trader_said` keeps
`observation` and `prediction` as separate items (TJ-14 item 1) and the story may call
only a `prediction` a call.

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

**AMENDED 2026-09-19:** on the Trade Mentor the strip is not its own widget — it is TJ-14's
`day_close` question, asked once on the session's last card beside `Followed the plan`;
the desk's journal tab keeps the optional strip. Mood is a context field in TJ-16's ledger
from the day it exists.

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

##### TJ-9 … TJ-16 — closing the review loop (trader, 2026-09-19)

*The bot is three machines in a row: it TAKES IN what the trader does and thinks (TJ-9,
TJ-14), it WORKS OUT which thinking pays (TJ-10, TJ-11, TJ-15, TJ-16), and it TELLS them
what to keep and what to change (TJ-12, TJ-4, TJ-5, TJ-6), at night only (TJ-13).*

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

**Items 1-6 BUILT and MERGED 2026-09-19**, branch `claude/tj9-forced-trade-labels` (tip `ee35ae54`), merged into `lead/p033-integration` as `8077a758` after three review rounds (NO-GO, NO-GO, GO). What shipped differs from the text below in two places: the section rides on ANY later delivered slot of the session while still owed (not only after a shown card), and the setup guess is filtered by SHAPE (rejection word, link, any `<prefix>:<code>`), not by a closed vocabulary. Live gate **#154** retained, plus a first-morning check of the not-ready -> ready path. **Item 7 was not built here: it became packet TJ-9Q, which is now its own entry below.**

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
4. **Journal freshness.** When the night ended without an OK, the DESK makes one
   deterministic import retry on a worker before the 09:00 card (seconds, no model — the
   nights-only rule is about inference; the AI Jobs task itself never fires by day; once
   TJ-14 item 4 exists this retry IS its pre-card pull); Day Review and the
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

##### TJ-9Q — A Questrade fill says what it is, and a sold put is a sale

**BUILT and MERGED 2026-09-20** (branch `claude/tj9q-questrade-instrument`, tip
`76f3cf2a`, merged `86c64f96` into `lead/p033-integration2`; not on `main`). TJ-9's item 7,
split out after the tester's step 0 refuted three of its premises: `net_amount` is NULL on
all 226 Questrade rows, so the goldens use the payload's own `totalCost`; there are THREE
sold-put positions (four STO fills — AAOI has two) and ONE bought put, not four sold puts;
and `journal_file_authority` is affected, in two ways, both corrections. Items 1-4 are
built. The classifier (`classify_questrade_security_type`) and the side map (`STO`→SELL,
`BTC`/`COV`→BUY) are pure and always correct; the import seams and the CSV statement path
are gated on `QUESTRADE_INSTRUMENT_FROM_SYMBOL`, which **ships OFF — item 4's gate is NOT
met**, because 29 positions are open under the old convention and one journal may never
hold both; and `scripts/journal_reclassify.py` is the one way stored rows move (dry run by
default, byte-exact backup, one store method, every trade-keyed table carried or NAMED,
the switch written LAST and never onto a half-moved journal, exit 4 when it was held
back). The one ungated change is `_BUY_SIDES` gaining `COV`, which makes a broker file
agree with the sync on the 14 (account, day) pairs its 25 covers touch. **Owed: the
trader's own `--apply` run on the live journal** — nothing there has moved — and the live
gate below. Long form: DESK_INTERNALS "TJ-9Q".

Live gate **#162**: after the trader's `--apply` run, the Journal page shows the three
sold puts as SHORT and CLOSED with AAOI **+241.03**, BE -666.99, QBTS -57.96 and QQQ
-81.98; the next Questrade import adds a fill without creating a second position for a
contract that already exists; the overnight AI narration is still shown for a re-keyed
Questrade trade (24 of 24 enrichment rows landed on live trades on the copy); and
"Check a statement..." on a day holding a cover no longer reports a disagreement the size
of twice the cover.

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
   notes typed outside a card) and is labelled `extracted`. The card's full shape — the
   separate **What I see** / **What I expect** parts, `How sure` and `Because…` — is TJ-14
   item 1; each graded row is stored with TJ-16's context snapshot from its first day.
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

**BUILT and MERGED 2026-09-19**, branch `claude/tj11-walkaway-v2` (tip `a744bec1`), merged
`a89ec7d5` into `lead/p033-integration`. Long form in DESK_INTERNALS "Day Review -
walk-away v2". One reviewer NO-GO round landed on the same branch and was re-reviewed GO by
reproduction on copies of the live stores: **item 5's session stamp became an ADDITIVE
field** after the first build moved what `session_date` MEANS for every live reader (on
Monday 2026-09-21 the branch would have hidden 12 setups rows where base hid none and
marked 18 names Reviewed-today, and moved Friday-evening vetoes into the veto cohort) —
`session_date` now keeps exactly its base meaning and value, the new `decision_session` key
is written on new rows only, never backfilled, and only TJ-11's readers use it; and
**item 6's pooled rate now counts a name only when its horizon has CLOSED**, after the
first build's open-horizon cells came out 100% by construction (34% = 30/87 shipped against
a closed-only truth of 26% = 20/77). The 18 Saturday-stamped D1 calls — 12 veto and 6 like
cohort picks with zero outcome rows — were deliberately left as they are (lead decision
(a)). Two tester assertions were amended under the lead's explicit authorisation, recorded
as a packet defect; three landed TJ-1/TJ-1L/TJ-2B assertions pinning "four tables / ten
columns / a 2 x 2 grid" were updated to the new contract, not weakened. Two things the
packet named that the build did differently, both measured: the untouched population comes
from the horizon-outcomes rows `read_day` ALREADY reads rather than from `day_review_index`
(the index is narrowed to the 3-session lookback and cannot serve a 20-session base rate,
and reusing the existing read costs no I/O), and the daily-bar read is bounded by three
named SIZE caps because reading a whole session's ~1,100 scan names costs ~5 s and ~340 MB.
Live gate **#156** is what remains owed.

**AMENDED 2026-09-19 (~16:20 PDT) - item 5's DIRECTION is reversed; packet TJ-11F, branch
`claude/tj11f-decision-session` (tip `67143e3e`), merged `f00ec302` into
`lead/p033-integration2`.** The trader, three hours after wave 1 went live: *"a veto on
friday night (after the market close) should not be considered monday since we have new
information then."* A decision belongs to the session whose information it JUDGED, and the
next session's scan is new information (decision 0021 answer 33, striking answer 16's last
clause). `decision_session(stamp)` is now the exchange session whose New York date the
stamp falls on when that date is a session day - pre-market, in-session and after the
close alike - else the **most recent PRIOR session**. New rows carry
`decision_session_rule: "judged_session_v2"`; a stored session without it is recomputed
from the row's own stamp and no row is rewritten or backfilled.
`day_review_service._stamped_dates_for` maps a session onto the non-session dates AFTER
it, and the D1 ruler's reference is the judged session's close. `session_date`, the five
parity readers, the cohort graders and the 18 Saturday-stamped picks are untouched, and
nothing on the live desk outside Day Review changes. Reviewer GO by reproduction on a copy
of the live annotation store (1,198 rows, 0 carrying the field or the marker). Gate #156 is
reworded below and still owed.

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
5. **Session stamp:** a decision made outside a session belongs to the session it JUDGED -
   the most recent PRIOR exchange session, never a weekend date (**reversed by TJ-11F,
   2026-09-19**; as first built it was stamped with the NEXT session). Recon first measures
   how the cohort graders treat today's Saturday-stamped rows; existing rows are never
   rewritten (the reader maps a non-session date back).

Tests: a pop-then-fade is not a real miss; an adverse-first name is not a miss; a Friday
21:30 Pacific call reads as FRIDAY's (TJ-11F; it read as Monday's as first built);
five-session window on the exchange calendar across a holiday; totals still equal the
decision count.

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

Live gate **#156**: a Friday-evening call reads as **FRIDAY's** - the 18 D1 calls filed
2026-09-18 21:04-21:07 Pacific are on **Friday's** Day Review and not on Monday's, each
measured from Friday's close; a vetoed name that ran two ATR by Wednesday is on
Wednesday's Earlier-calls table; and each table has its sentence.

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

**TJ-13A — items 5, 6 (the slate machinery), 8, 9 and 10 are BUILT and MERGED**, branch
`claude/tj13a-night-slates`, merged `9eaae1dd` into `lead/p033-integration` on 2026-09-19
(reviewer GO by reproduction against a copy of the live 476-row ledger; the review round
capped `ai_summary` at three attempts, made `--slot` resolve against every registered
slot, and kept the rejected-reply log local). Decision 0018 carries the 2026-09-19
amendment; long form in DESK_INTERNALS "Night kinds". Live gate #158 stays owed.

**TJ-13B — item 7's MEASURING half is BUILT and MERGED 2026-09-19, fix round 2026-09-20**
(branch `claude/tj13b-large-local`, tip `800ecb8c`, merged `e54c8203` into
`lead/p033-integration2`; not on `main`). It ships as a COMMAND, never a slot:
`scripts/ai_jobs/model_probe.py` (`--probe-model large`, one `manual_test` ledger row per
measurement, `latest_measurement` / `reserve_minutes_from_probe`) and
`scripts/ai_jobs/provider.py` (`local_large`, `ai_week_review_provider`,
`week_review_plan`, `request_with_fallback`, ledger field `model_attribution`). The probe
is night-only with `--force` never buying the clock (it re-spends only the
already-measured check), holds the AI-jobs machine lock BY DEFAULT and refuses when it is
not free, reads a deleted copy of one week's fact packs, and records load seconds, tokens
per second, peak memory in use and the context the SERVER accepted, each labelled with its
basis. The review round was NO-GO because `--force` never reached the probe; the fix was
verified with the real AI lock free (340 AI-jobs tests green). **The measurement itself is
owed: the 27B has still never run.** Until a probe row exists, `week_review_plan` answers
`may_run_large: False` and TJ-5's week story runs on the medium local model and says why.
No slate changed; `openai` remains a setting, off. **Still TJ-13B and NOT built**: item
6's Sunday EXTRAS — suggested setup tags for old untagged trades, and the week-ahead note.
Owed: the trader's first Saturday-night probe (gate #158's TJ-13B clause), and TJ-5
reading `week_review_plan` when the week slot lands. Long form: DESK_INTERNALS "TJ-13B".

5. **BUILT (TJ-13A), except that a daytime request is REFUSED, not QUEUED**: a forced
   model slot records `skipped` and the page button refuses with the window's own reason.
   The queue-for-tonight belongs to TJ-4. `window.py` loses "weekends are open all day": the configured night window applies
   seven days a week, so no run — scheduled, forced or from a page button — starts local
   inference by day. A daytime request is queued for tonight (TJ-4's rule, now general).
   The one standing exception stays the Trade Mentor's seconds-long draft on the trader's
   own reply (CLAUDE.md, quiet hours); the trader may remove it.
6. **BUILT (TJ-13A) except the two Sunday extras named below, which are TJ-13B.**
   **Weeknights** run only the short trader-facing slots plus the deterministic stage.
   **Saturday night** (the first night with no session behind it) runs the weekly slate:
   `ai_summary` once a week instead of nightly, `weekly_synthesis` without a typed command,
   `week_review_narration`, the month re-cut. **Sunday night** runs the backlog: a retry of
   any day story that failed that week, SUGGESTED setup tags for old untagged trades
   (suggestions only — `journal_bulk_tag`'s provisional rule, never a confirm), and a short
   week-ahead note ready before Monday's open. A slate that does not finish resumes the
   next night; nothing runs past the window's end.
7. **The MEASURING half is BUILT (TJ-13B); the week STORY that uses it is TJ-5's.**
   **The week story is written by the large LOCAL model** (trader, 2026-09-19, replacing
   TJ-5's OpenAI default): `ai_week_review_provider` defaults to `local_large`, falls back
   to local medium and says so in the ledger under `model_attribution`; `openai` remains a
   setting, off, and `request_with_fallback` RAISES for it — TJ-5's slot must catch that
   and record a FAILED row. It may only narrate measured rows (TJ-4's amendment applies).
   The 27B is measured on this desk — load time, tokens per second, peak memory beside a
   running desk, the context the server accepted — by `--probe-model large` on a deleted
   copy of one week's packs, and the slot's `reserve_minutes` comes from that number
   (`reserve_minutes_from_probe`, `None` when the tier was never measured). **The
   measurement is the trader's, and it has not happened yet.**

8. **BUILT (TJ-13A). Where `ai_summary` runs:** item 6 supersedes "then `ai_summary` last" in the paragraph
   above — it leaves the weeknight slate entirely and runs on Saturday night, for ONE
   session a week (Friday's), capped at three attempts.
9. **BUILT (TJ-13A) as a RULE; each listed slot still arrives with its own packet.**
   **One slot list, kept whole.** This program's new slots, by decision-0018 stage, for
   `EXPECTED_SLOT_ORDER`: Stage 1 (deterministic) `miss_contrast` (TJ-15),
   `prediction_contrast` (TJ-16), the day-pack build; Stage 2 `day_review_narration`
   (TJ-4), `observation_tags` (TJ-16), `week_review_narration` (TJ-5, Saturday night);
   Stage 3 `improvement_ideas` (TJ-6). Each packet appends its own slot inside its stage;
   recon places a new Stage 1 slot against `measured_report`, which today closes that
   stage.
10. **BUILT (TJ-13A).** **The measured report's examples repeat one name** (audit, 2026-09-17 report: `ABCL`
    three times, `ERAS` three times, one swing id three times). Its best/worst lists
    de-duplicate by symbol before they take the top three; fail-before-fix.

Live gate **#158**: the ledger shows the day story finished before 23:30 Pacific, the
morning brief lists no membership-only name, `ai_summary` is either synthesized or failed
fast and runs on Saturday night only, no ledger row starts between 06:00 and 22:00 Pacific
on any day, Sunday night's ledger shows only the deterministic stage plus whatever the
weekend still owed, and Sunday's Week Review shows a week story whose ledger row names the
large local model. Reading "failed fast": a REFUSED endpoint degrades in seconds (measured
2.06 s), a HUNG one still costs ONE 900 s read timeout on the first slice — a ~15 minute
degrade is the repair working, not a regression. The TJ-13A clauses are readable on the
first weekend after the merge; the day-story clause belongs to TJ-4 and the large-local
week-story clause to TJ-13B.

**Gate #158, TJ-13B clause (the trader's own run).** On a Saturday night inside the
off-hours window and before ~05:15 PDT, with no AI job running, the trader types
`.venv\Scripts\python.exe scripts\run_ai_jobs.py --probe-model large` and it exits 0 with
a measurement. On the live ledger: exactly ONE new row with `job="model_probe"`,
`status="manual_test"`, `model` = the 27B tag, and a `model_probe` block whose
`load_seconds`, `tokens_per_second`, `peak_memory_mb` and `context_tokens_accepted` are
all > 0 with `basis` naming how they were measured. Confirm `peak_memory_mb` stayed inside
this box's 32 GB beside the running desk, that the desk did not stall, that
`context_tokens_accepted` is close to what was sent (a much smaller number means the
server sheared the prompt), and that `--status` still reports the night's slate unchanged.
With the measurement recorded, `ai_jobs.provider.week_review_plan()` must answer
`may_run_large: True` with a `reserve_minutes` derived from that row. A refused probe —
daytime, an AI job already running, or this session already measured — must exit 1, print
why, and leave every prior ledger row untouched; adding `--force` must re-measure (a
second row, newest wins) and must NOT get past the window or a held lock.

##### TJ-14 — The Mentor asks for what the desk is missing, and nothing else

**Items 1 and 6 BUILT and MERGED 2026-09-19 as packet TJ-14A**, branch
`claude/tj14a-mentor-card` (tip `6808abd9`), merged `e8c04f88` into
`lead/p033-integration2`; reaching `main` after the night's AI run. Two reviews driven
through the real Qt slot (NO-GO, then GO). The card has two labelled parts that never share
a field: **What I see** (`mentor.observation`, optional) and **What I expect**
(`mentor.prediction`, `mentor_prediction_v1` — direction, horizon printed on the row, `How
sure` forced unless `No view`, `Because…` optional). The gate is inside `submit()` and
`read_unchanged()` as well as on the buttons; an `m5_d1` card writes both rows, words or no
words, and a wordless row's `text` stays `""` with a salted `entry_id`.
`market_journal.prediction_of` is the ONE accessor and returns `None` for all four live
pre-TJ-14A vintages. **A row's timeframe and its prediction's horizon always agree**
(`HORIZON_FOR_TIMEFRAME`), enforced at the WRITER, and `Read unchanged` reaffirms PER
TIMEFRAME from the latest read of each — the review round found the old `rows[-1]` fallback
filing a D1-timeframe row carrying a rest-of-day call on any day the 08:00 card was
answered. Item 6: `trade_mentor_context_v2` adds `XLRE` (18 symbols), three day facts per
symbol and the derived block; `internals_at` is a pure rebuild on the SAME builder as the
live card, `internals_bars_at` its thin loader (daily cache first, then daily bars built
from the prior sessions' own tape for RSP / USO / TLT, which that cache has never held —
no network, and no production caller until TJ-10 / TJ-16); the card shows
`MentorInternalsStrip`; and `day_review_bars` carries the symbols in its one batched
post-close download — one yfinance call per fifty, fixed base 18 instead of 4, no D1 leg.
**The lead has confirmed this vocabulary as TJ-10's and TJ-16's contract.** Items 2-5 (the
question registry, the budget of three, the day-time fill pull, "the AI takes it from
there") remain for **TJ-14B**. Long form: DESK_INTERNALS "TM" (2026-09-19 addendum). Live
gate **#159**'s first clause is now readable.

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
   **AMENDED 2026-09-19 (trader): a description is not a prediction, and the card keeps
   them apart.** *"Make sure we differentiate predictions from just 'describe the market
   and your thoughts'."* The card has two labelled parts that never share a field:
   **What I see** — the description and thoughts, free text, optional, about NOW; and
   **What I expect** — the prediction: the forced direction click with its horizon printed
   on the button row (`Rest of day` / `Next 5 sessions`), a `How sure: Low / Medium / High`
   click (skipped on `No view`), and an optional one-line `Because…`. They are stored as
   separate keys on the `mentor` payload (`observation`, `prediction{direction, horizon,
   confidence, because}`); no reader may build a prediction out of `observation` once a
   card has the split, and every row written before it stays `extracted` and is never
   pooled with a clicked prediction.
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
   priority; the rest are counted on the card and carried, never dropped and never a
   fourth. **The forced `trade_label` section is outside the budget** — TJ-9 lists every
   trade of the previous session and a same-session fill is always asked — so on a card
   that carries it the budget covers the OTHER kinds only. Every question offers the four answer states plus `Stop
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

6. **The desk already knows the internals, so the trader never types them** (trader,
   2026-09-19: *"trade mentor should automatically be processing what's going on with the
   internals we watch. RSP VXX USO TLT and the sector ETFs XLK XLE etc. so the AI already
   has that"*). What exists (Phase 0.31, measured live that day): every Mentor answer
   already stores `mentor.context` (`trade_mentor_context_v1`) with 17 symbols — VXX, RSP,
   USO, TLT, IWM, QQQ, SPY and ten sector ETFs — each with four facts (30-minute change,
   side of session VWAP, five-day change, side of the 20-SMA); 99-119 of 119 readings a day
   were measured. What is missing: only `ai_summary` reads it; the card keeps it silent;
   it exists only for hours the trader ANSWERED; the facts are thin; `XLRE` is absent
   although the desk's own `sector_etf_map.json` carries it. Changes:
   - `trade_mentor_context_v2` (pure, completed bars only, v1 rows stay readable): per
     symbol adds the day's change, place in the day's range, and side of the prior day's
     high and low; and a small DERIVED block of the reads the trader now types by hand —
     breadth (`RSP` minus `SPY` on the day), fear (`VXX` direction against `SPY`'s, a
     divergence said as one), rates (`TLT`), oil (`USO`), sector leaders and laggards (top
     and bottom three, on the day and over 30 minutes), offense against defense
     (`XLK`/`XLY`/`XLC` vs `XLP`/`XLU`/`XLV`), and how many sectors sit above session VWAP.
     Each derived line names the readings it rests on; a missing input makes the line
     `unmeasured`, never a guess. `XLRE` joins the list.
   - **The card SHOWS it**: a compact internals strip above **What I see**, so the trader
     reads what the desk already has and writes only what it cannot see.
   - **It is kept for every hour, answered or not.** TJ-2's session bars add these symbols
     to their one batched post-close download, and `internals_at(session, stamp)` rebuilds
     the same block from the durable tape for any moment — so a skipped hour, a note typed
     on the desk tab and a prediction's context (TJ-16) all read ONE function.
   - **Every AI input gets it:** the day pack (TJ-4) gains `internals` — the open, each
     Mentor hour and the close, each with a `source_id` — and `market_story_narration`, the
     week story and TJ-16's tagger read it from there.

Tests: a kind without a consumer fails; budget of three with the remainder counted; `Stop
asking this` retires one subject only; a same-session label carries `same_session`; AWAY
asks nothing; a failed pull leaves the card on time; the quick-like follow-up writes a link
and no tag.

Live gate **#159** — first clause (TJ-14A, readable on the trader's next desk session):
every Mentor card comes up with a direction row whose horizon is printed on it, Submit
stays grey until that row AND `How sure` are clicked (and goes green at once on `No view`),
the 08:00 and 12:00 cards show both rows and file two entries, a card answered with clicks
and no words files a row whose `text` is empty and whose `mentor.prediction` holds the
call, `Read unchanged` refuses until this hour's call is clicked and — on the card after an
answered 08:00 — files an **M5** row with the M5 words and the M5 call, never a D1 row
carrying a rest-of-day one, and the internals strip above **What I see** prints breadth /
fear / rates / oil / leaders / laggards / offense-defense / sectors-above-VWAP with
`unmeasured` said out loud, with no second context read and no new stall in the hourly
popup. Day Review the next morning shows the call where a wordless answer's words would
have been. Rest of the gate (TJ-14B): an ordinary hourly card is one click; on a day with a
new fill the next card asks its setup and stop; a trade from nowhere is asked its origin
once; nothing is asked twice after `Stop asking this`.

##### TJ-14C — A quick like's answer joins the like cohort (PLANNED)

**PLANNED, not started** — a follow-up the TJ-14B work surfaced. TJ-14B registers a
`quick_like_followup` Mentor kind that stays DORMANT because nothing joins what the
trader answers about a quick like back to the like itself: the answer is written as an
`opportunity_events` row keyed to the opportunity, while the like cohort is read from the
review-event stream, so the two never meet and the question has no consumer. TJ-14C joins
them — one reader that carries a quick like's answered `opportunity_events` row into the
cohort that grades that like — and waking the dormant kind is what proves it. Rules that
already bind it: an action joins `TAKE_ACTIONS` / `REJECT_ACTIONS` on what its WRITER
does, never its name; a like carries zero privileges and contributes a LINK, never a tag;
no tag is derived from an outcome; and an evidence store is never allowed to cost the
thing it records. Needs TJ-14B merged. Gate: the first quick like answered through the
Mentor shows its answer on the like's own cohort row, with `n` unchanged elsewhere.

##### TJ-15 — What the misses had in common (no model, no chart pictures)

**BUILT and MERGED 2026-09-19**, branch `claude/tj15-miss-contrast` (tip `8547c4a5`),
merged `fb3f55e9` into `lead/p033-integration2` with the lead's slot-position fix
`1f260ffa` on top; reaching `main` after the night's AI run. Two reviews by reproduction on
the real 2026-09-18 window (NO-GO, then GO). Item 1: `scripts/evidence_contrast.py`, pure
and reused by TJ-16 — `contrast()` and `rate()`, with TWO floors (`MIN_REPORTABLE_N` on a
group's rate, `MIN_CONTRAST_SIDE_N` = 10 a side on a feature; a thinner feature is named in
`thin_features` with both counts and no AUC, `compared` counts the ranked only, and a group
leads only with a reportable rate AND a ranked feature). Item 2: the deterministic Stage 1
slot `scripts/ai_jobs/miss_contrast.py` (`uses_model=False`, `max_attempts=3`,
`reserve_minutes=5.0`), registered in `ai_jobs/runner.py` after the cohort graders and
`theta_pick_grading` and **BEFORE the `market_story_rollups` + `measured_report` pair that
closes the stage** — the lead moved it above that pair at integration when the full suite
showed WS-10D's, WS-RP's and `test_veto_cohort_grading`'s order pins red;
`_STAGE_ONE_LAST_SLOT` is untouched and a slot appended after it would leave the Sunday
slate. `EXPECTED_SLOT_ORDER` gains one name. It judges **D1 decisions only** (the reviewed
population is about half M5 — 1,026 M5 against 1,013 D1 over the 20 sessions ending
2026-09-18) and counts the rest in `excluded_by_timeframe` / `no_timeframe`; point-in-time
features are the LAST scan at or before the decision's stamp, from its own session or the
ONE before (`MAX_SCAN_AGE_SESSIONS = 1`; an own-session-only join lost 47-83% of
decisions). Item 3: one JSON pack per session beside the digest in `store.digests_dir()`,
with `observational`, `top K of N`, the thin count and both `n`s inside every group. Item
4: `read_latest` ships and the Week Review table remains TJ-5's (lead decision 4). Tests:
`tests/test_tj15_evidence_contrast.py`, `tests/test_tj15_point_in_time.py`,
`tests/test_tj15_miss_contrast_slot.py`, `tests/test_tj15_miss_contrast_builder.py`,
`tests/test_tj15_fix_round.py`; the lead amended two tester literals the feature floor made
unsatisfiable (`53d41a9a`) and authorised seven floor keywords (`8547c4a5`). Long form:
DESK_INTERNALS "TJ-15"; slot position: decision 0018's 2026-09-19 amendment. Gate #160 is
owed.

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
`REAL_MISS_V1` is ONE pure function that the Day Review worker (TJ-11) and this slot both
call; the slot never re-implements it and never reads the page's payload.

Live gate **#160** (TJ-15's slot is readable the night it first runs; the TABLE is TJ-5's
Week Review). After the first nightly `miss_contrast` run on a session with real decisions:
the ledger row is `ok` with a reason and one output; the pack names `real_miss_v1`, its
window in sessions, `timeframe: D1` and `observational, top K of N` plus the thin count in
every group; each group's misses + correct rejections equal its `measured` with `pending`
printed separately and in neither half of the rate; `excluded_by_timeframe` plus
`no_timeframe` account for every non-D1 decision in the window and the statement names the
number; the join counts (`scan_same_session` + `scan_prior_session` +
`no_point_in_time_scan`) add up to `measured` per group and the unjoined share is a
minority; a reason under `MIN_REPORTABLE_N`, **and any group whose features are under
`MIN_CONTRAST_SIDE_N`**, shows its counts, reads `too few to call` or `no feature had enough
rows on both sides`, and appears in neither `leaders` nor the pack's sentence — **and if no
reason clears both floors the table says so** in words rather than showing an empty table;
and the slot's wall time is seconds, not minutes, against the ~709 MB features file.
Nothing on the desk changed. Then, when TJ-5 lands: Saturday's Week Review shows, for the
veto reason with the most real misses, a table of what those misses measured differently,
each row with both `n`.

##### TJ-16 — What leads to a good call, and what leads to a bad one

*Goal (trader, 2026-09-19):* *"The hope is an AI can pick up on my tendencies and what leads
to good predictions and what leads to wrong ones."* The math finds the tendency; the model
labels the words and tells the story; neither may do the other's job.

What exists after TJ-10 and TJ-14: one graded row per clicked prediction, the separate
`observation` text, `d1_environment.jsonl`, `market_story.build_daily_story` facts, the
session tape in the TJ-2 bars file, `evidence_stats` (the ONE Wilson), TJ-15's contrast
method, TJ-6's checked ideas.

Changes:
1. **The prediction ledger** (pure, append-only, under `DAY_REVIEW_DIR`): each graded
   prediction is stored with a point-in-time CONTEXT snapshot taken at its stamp from
   completed bars only. **The market half of that snapshot is TJ-14 item 6's
   `trade_mentor_context_v2` block, never a second builder** — so breadth, fear, rates,
   oil, sector leadership and offense-vs-defense are contrast fields from the first day
   ("you call Up well when breadth leads and badly when VXX is rising with SPY"). Around
   it: hour of day, SPY vs session VWAP and vs the prior day's range, gap
   size, the D1 environment label, VXX direction, the last hour's SPY direction, whether
   this call agrees with the trader's own latest D1 click, the previous call's verdict
   (right / wrong / none — the after-a-miss question), confidence, direction, and, when
   TJ-7 exists, mood. A field the desk cannot measure is `unmeasured`, never guessed.
2. **Skill against naive baselines, never a bare hit rate.** Every accuracy cell is shown
   beside what `always Up`, `same as the last hour` and `with the D1 environment` would
   have scored on the SAME stamps; "right 58% (n 64); always-Up 61%" is the honest
   sentence. **Calibration:** accuracy by `How sure` — High must beat Low or the page says
   it does not.
3. **The contrast** (deterministic Stage 1 slot `prediction_contrast`, after
   `miss_contrast`): right against wrong over `LATELY_SESSIONS` and over the whole ledger,
   per context field — counts, the Wilson interval, `observational, top 3 of K`, nothing
   named under the floor, separately for `Rest of day` and `Next 5 sessions`. No model.
4. **The model labels the WORDS, grounded** (Stage 2 slot `observation_tags`, local medium,
   weeknights, seconds per note): each `observation` and `because` gets codes from a
   closed, versioned vocabulary (`ui/annotations/vocabularies/observation_tags_v1.json`:
   cites a level, cites volatility, cites news, cites breadth or sectors, cites the D1
   picture, hedged wording, reacting to the last bar, no reason given …), each code with
   the exact source span that must reproduce it (`market_thesis`'s rule) or it is rejected.
   The codes become context fields in item 3 on the NEXT run. The model never sees a
   verdict while tagging, so a tag cannot be derived from the outcome.
5. **The voice.** The Saturday-night week story (large local model) gets the contrast
   tables and may narrate at most three tendencies, each citing its cell and its `n`; a
   tendency it wants acted on becomes a `process` idea (TJ-6) and is therefore CHECKED
   before and after. Week Review shows the tables under the story; Day Review's **Your
   reads** line shows yesterday's tally beside its baseline.

Tests: a description can never become a prediction once the split exists; a context
snapshot uses completed bars only and nothing after the stamp; baselines are computed on
the identical stamps; a tag whose span does not reproduce is rejected whole; the tagger's
input contains no verdict; pooling `extracted` with clicked rows raises.

Live gate **#161**: after two weeks of clicks, Saturday's Week Review shows accuracy beside
the three baselines, accuracy by confidence, and a right-vs-wrong table by hour and by
environment, every cell with `n`; the week story's tendencies each point at a cell.

#### 12.5 Order and dependencies

**AMENDED 2026-09-19 (twice):** TJ-9 (independent; first, because every personal statistic
waits on labelled trades) → TJ-14 (the card and the registry; TJ-9's section becomes its
first kind, and the prediction click starts accumulating at once) → TJ-10 → TJ-11 (may run
alongside TJ-10 and TJ-3) → TJ-4 (now needs TJ-10) → TJ-12 → TJ-15 (needs TJ-11's
real-miss rule) → TJ-16 (needs TJ-10's graded rows and TJ-14's split card; its ledger's
context snapshot ships WITH TJ-10 so no click is ever stored without one) → TJ-13 (independent; any time after TJ-4's slot exists) → TJ-5 → TJ-6 →
TJ-7 (fields only; its strip is TJ-14's `day_close`) → TJ-8. The original chain below still
orders TJ-3 … TJ-8 among themselves. ~~No packet after TJ-9 merges until the desk has been
restarted once and gates #145, #152 and #153 have been read on a real session~~
**SUPERSEDED 2026-09-19 (trader: "I'd prefer to build it all now while I have usage
available … make sure the plan lets me build it all right away"):** nothing waits on a
restart. Every packet is built, reviewed and merged to `main` in the wave order below; the
live gates (#145, #152, #153 and #154–#161) are read afterwards, on the trader's first
restart and the sessions after it. A gate that FAILS then outranks any unbuilt packet. Until 2026-09-20 15:35 PDT the lead may restart the desk and read what a weekend can prove by itself - rules in 12.6a.

**How to start (2026-09-19).** Say to a new lead session: *"Build Phase 0.33 from `plan.md`
12.5, wave by wave, with the agent team. You may restart the desk and read gates yourself until 2026-09-20 15:35 PDT - plan.md 12.6a."* Every packet below is ALREADY WRITTEN in
`.claude/packets/` (machine-local; read `TJ-LOOP0_COMMON.md` first - it tells each agent
which files it owns and that the LEAD reconciles all docs). No code exists yet for any of
them: on 2026-09-19 three testers were started and stopped within minutes at the trader's
word; their empty worktrees and branches were removed.

**Waves - packets in one wave own unrelated files and run in PARALLEL; a wave starts when
the packets it needs are merged.** Per packet: tester (red, and step 0 verifies the
packet's premises) → builder → reviewer → the lead merges in a scratch worktree, runs the
full suite with the nightly AI lock free, ruff, smoke, selftest, and reconciles the docs.

| Wave | Packets (file in `.claude/packets/`) | Branch | Needs |
|---|---|---|---|
| 1 | **TJ-9 MERGED 2026-09-19** (`TJ-9.md`, merged `8077a758`; item 7 -> **TJ-9Q, MERGED 2026-09-20**, `claude/tj9q-questrade-instrument` tip `76f3cf2a`, merged `86c64f96` into `lead/p033-integration2`) · **TJ-11 MERGED 2026-09-19** (`TJ-11.md`) · **TJ-13A MERGED 2026-09-19** (`TJ-13A.md`) | `claude/tj9-forced-trade-labels` · `claude/tj11-walkaway-v2` (merged `a89ec7d5`) · `claude/tj13a-night-slates` (merged `9eaae1dd`), both into `lead/p033-integration` | `main` |
| 1F | **TJ-11F MERGED 2026-09-19 (evening)** - trader reversal of TJ-11 item 5: an after-close, weekend or holiday decision belongs to the session it JUDGED | `claude/tj11f-decision-session` (tip `67143e3e`, merged `f00ec302` into `lead/p033-integration2`) | TJ-11 |
| 2 | **TJ-14A MERGED 2026-09-19 (night)** (`TJ-14A.md`) · **TJ-3 MERGED 2026-09-19 (evening)** (`TJ-3.md`) · **TJ-15 MERGED 2026-09-19 (night)** (`TJ-15-16.md`) — wave 2 complete | `claude/tj14a-mentor-card` (tip `6808abd9`, merged `e8c04f88`) · `claude/tj3-note-markers` (tip `58ee11f4`, merged `72647104` into `lead/p033-integration2`) · `claude/tj15-miss-contrast` (tip `8547c4a5`, merged `fb3f55e9`, slot position fixed `1f260ffa`) | TJ-9 · TJ-11 ✓ · TJ-11 ✓ |
| 3 | TJ-14B (`TJ-14B.md`) · TJ-10 (`TJ-10.md`) | `claude/tj14b-mentor-questions` · `claude/tj10-read-grader` | TJ-14A (TJ-10 also rebases on TJ-3's page edits) |
| 4 | TJ-4 (`TJ-4.md`) · TJ-16 (`TJ-15-16.md`) · **TJ-13B MERGED 2026-09-20** (`TJ-5-6-7-13B.md`) | `claude/tj4-day-story` · `claude/tj16-prediction-contrast` · `claude/tj13b-large-local` (tip `800ecb8c`, merged `e54c8203` into `lead/p033-integration2`) | TJ-10, TJ-11 · TJ-10, TJ-14A, TJ-15 · TJ-13A |
| 5 | TJ-12 (`TJ-12.md`) | `claude/tj12-report-card` | TJ-9, TJ-10, TJ-11, TJ-4 |
| 6 | TJ-5 → TJ-6 → TJ-7 (`TJ-5-6-7-13B.md`), one after the other | `claude/tj5-week-review` · `claude/tj6-ideas` · `claude/tj7-mood-fields` | TJ-4, TJ-12, TJ-13B, TJ-15, TJ-16 · TJ-5 · TJ-14B |
| — | TJ-8 cleanup | — | every page gate read live |

Known shared files, so merge in wave order and rebase the later branch:
`ui/panels/day_review_panel.py` and `ui/services/day_review_service.py` (TJ-11 → TJ-3 →
TJ-10 → TJ-4 → TJ-12), `ui/widgets/trade_mentor_card.py` (TJ-9 → TJ-14A → TJ-14B),
`ai_jobs/runner.py` + `EXPECTED_SLOT_ORDER` (TJ-13A → TJ-15 → TJ-4 → TJ-16 → TJ-5 → TJ-6).

**The build at a glance, by packet.**

| # | Packet | One line | Needs | Gate |
|---|---|---|---|---|
| — | TJ-1L | Day Review in two columns (MERGED; verified an ancestor of `main` 2026-09-19) | TJ-1 | #145 |
| 1 | TJ-9 | **MERGED 2026-09-19** - yesterday's trades labelled at 09:00, forced; label provenance; planned vs unplanned; journal freshness | — | #154 |
| 1Q | TJ-9Q | **MERGED 2026-09-20** - a Questrade fill says what it is and a sold put is a SALE; the import seams gated OFF; `journal_reclassify.py` the one way stored rows move, and the `--apply` is the trader's own act (owed) | TJ-9 | #162 |
| — | *restart* | NOT a hold any more (superseded above): the trader restarts when the build is merged and the gates are read then | — | #145 #152 #153 |
| 2 | TJ-14 | **TJ-14A MERGED 2026-09-19 (night)** - items 1 and 6: the card split (What I see / What I expect) with a forced prediction click, timeframe-horizon agreement enforced at the writer, `trade_mentor_context_v2` and the internals strip. TJ-14B remains: the question registry with consumers, the budget of three, same-session fills, "the AI takes it from there" | TJ-9 | #159 |
| 2C | TJ-14C | **PLANNED** - join a quick like's answered `opportunity_events` row into the like cohort; it wakes TJ-14B's dormant `quick_like_followup` kind | TJ-14B | — |
| 3 | TJ-10 | Read grader on the clicked prediction (+ context snapshot), congruence lines | TJ-14 | #155 |
| 4 | TJ-11 | **MERGED 2026-09-19** - walk-away v2: earlier calls, against-first, ATR, real-miss rule, skill line vs base rate, instrument-aware rows, ADDITIVE session stamp | TJ-2 | #156 |
| 4F | TJ-11F | **MERGED 2026-09-19 (evening)** - an after-close decision belongs to the session it JUDGED; `decision_session_rule: "judged_session_v2"`; the D1 ruler starts from the judged session's close | TJ-11 | #156 |
| 4 | TJ-3 | **MERGED 2026-09-19 (evening)** - note markers on the SPY and name charts; a mark sits on a bar only when it happened DURING that bar, and one past the tape's end is counted, never clamped; the page has no D1 toggle | TJ-2 | #147 |
| 5 | TJ-4 | Day pack + overnight day story that only narrates measured rows; rolling D1 view | TJ-10 | #148 |
| 6 | TJ-12 | Six-line report card heading Day Review (incl. How fresh) | TJ-10, TJ-11 | #157 |
| 7 | TJ-15 | **MERGED 2026-09-19 (night)** - what the misses had in common: a pure `evidence_contrast` with TWO floors, a deterministic `miss_contrast` slot inside stage 1 above the pair that closes it, D1 decisions only, the point-in-time join reaching one session back | TJ-11 | #160 |
| 8 | TJ-16 | Prediction ledger, naive baselines, calibration, right-vs-wrong contrast, grounded word tags | TJ-10, TJ-14 | #161 |
| 9 | TJ-13 | **TJ-13A MERGED 2026-09-19** (nights only 7 days; weeknight vs Saturday vs Sunday slates; briefs / summary / enrichment / examples repairs) · **TJ-13B MERGED 2026-09-20** (the `--probe-model` command and the `local_large` provider seam - the measurement itself is the trader's and is owed). Still open in TJ-13B: Sunday's suggested tags and week-ahead note; the week STORY is TJ-5's | TJ-4's slot | #158 |
| 10 | TJ-5 | Week Review: five day cards, week story, week + four-week + month strip | TJ-4, TJ-12 | #149 |
| 11 | TJ-6 | Ideas card; a kept idea is checked before and after | TJ-5 | #150 |
| 12 | TJ-7 | Mood / process fields (the Mentor asks them through `day_close`) | TJ-14 | #151 |
| 13 | TJ-8 | Cleanup of the retired panels | all page gates | — |

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

#### 12.6a The unattended window: the lead may restart the desk and read gates itself

**Trader, 2026-09-19 ~07:35 PDT:** *"If the bot wants to do its own restarts and live checks
on the bot for a few minutes / hours, let it control all of that on its own I don't use the
bot at all for the next 32 hours."* This is the trader's word for the restart rule in
`docs/AGENT_TEAM.md`, and it is TIME-BOXED: **from 2026-09-19 07:35 PDT to 2026-09-20 15:35
PDT.** After that instant a restart is the trader's call again, whatever is unfinished. The
whole window is a weekend: no exchange session, no Mentor slot, no scan of a live tape.
(Measured at the grant: no desk process was running.)

Inside the window the LEAD (never a builder, tester or reviewer) may:

1. **Stop and start the desk** from the main checkout (`trading_desk.cmd` /
   `launch_gui.py`, the production launch) as often as the build needs. Stop it
   GRACEFULLY (close the window / a plain terminate request) and wait for the process and
   its `bouncebot-scanner` child to exit; never force-kill unless it has been unresponsive
   for ten minutes AND no writer lock is held (`local_writer_lock` probes: the tracker
   save, the journal, `ai_jobs_runner`). Never stop it while a scan or a tracker save is
   running - wait.
2. **Update the main checkout only while the desk is DOWN**: merge in a scratch worktree
   first (full suite with the AI lock free, ruff, smoke, source selftest), then
   fast-forward `main`, then `launch_gui.py --selftest`, then start the desk. Record the
   commit the desk was last running BEFORE the first update as `last good`. If a start
   fails or the desk crashes inside ten minutes, `git revert` the offending merge (never
   `reset`, never force-push), start the desk on the reverted `main`, and write down what
   failed.
3. **Read live gates by looking, not by pretending to be the trader.** Allowed on the live
   desk: opening pages, selecting past sessions, reading logs, ledgers and files, timing a
   page, rendering a screenshot, letting the post-close / back-fill workers and the
   overnight AI task do their own writes. **Never on the live home folder:** a like, veto,
   pass, claim, note, Mentor answer, forecast paste, tag, confirm, watchlist edit, Focus
   add, alert arm, settings change or phone push made to test something. Any gate that
   needs a trader's click is read on a STAGED copy of the home folder
   (`TRADINGBOTV3_DATA_DIR` = scratch, `--allow-second-instance`, no IBKR connection on the
   live client ids, no ntfy topic) and is recorded as **`staged-pass`**, which is NOT
   `LIVE_VALIDATED`: the trader still owes the real read.
4. **What a weekend can honestly prove:** the desk starts and stays up; the selftest count;
   Day Review opens on past sessions inside its budget with no `[desk]` row (#145, the
   reading parts); Friday's bars file is back-filled with one batched download and the SPY
   tape draws (#152); the walk-away tables, the fifth table, the sentences and the skill
   line render on Friday's real decisions (#153, #156 in part); Friday evening's
   Saturday-stamped calls read as FRIDAY's (TJ-11F, 2026-09-19 - this line said Monday's);
   the Saturday-night slate (#158: no ledger row
   by day, `ai_summary` Saturday only, the day story before `ticker_briefs`) IF TJ-13A is
   merged and the desk machine is left alone before 22:00 PDT Saturday. **What it cannot
   prove** and stays owed to the trader: anything needing a live session, a Mentor slot or
   the trader's own click - #154, #155, #159, #161 and the clicked halves of the rest.
5. **Leave it as found, or better.** At the end of the window the desk is RUNNING on
   `main`, in the Auto mode it was found in, scheduled tasks untouched (never edit,
   disable or re-register `TradingBotV3 AI Jobs` or the DAS pushes), the working tree
   clean, every worktree the lead made removed. Every restart is one line in the
   checkpoint's dated entry: time, commit, why, result. A gate read in the window moves in
   the checkpoint to `weekend-read` or `staged-pass` with what was seen, never to passed.

Everything else in `docs/AGENT_TEAM.md` and CLAUDE.md still binds: agents' scripts never
write a live store, ask-first files stay ask-first, no detector / score / alert change.

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
