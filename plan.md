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

### Setup-score repair — trader directed, 2026-09-22 (SP)

The trader approved the full 2026-09-22 assessment: **"Go ahead and make the fixes as
per the full assessments then reconsider scoring changes."** This is the scoped
ask-first authorization for the Points scorer/data feed/evidence worker and the PQS
family lookup/final-score seams described here; no detector or alert recipe changes.
Work starts from `main` `fb95854e` in isolated worktrees. It does not promote a WISHLIST
idea or retire any Phase 0.33 live gate.

1. **SP1 — BUILT AND REPRODUCED, pending live validation.** Feed Points the measured, same-scan chart facts; missing room
   data earns no clean-path bonus. Version corrected inputs and snapshots, retain
   append-only old logs, capture meaningful same-day revisions, and grade one eligible
   observation per session/name/side against five real exchange sessions. Keep tied
   values together; show sample/session coverage and withhold learned proposals when
   either half has fewer than 30 observations or five entry sessions. No new fitted
   multipliers and no change to other consumers' outcome policies.
2. **SP2 — BUILT AND REPRODUCED, pending live validation.** Pin the current PQS as `pqs_v1`; correct the default's sample
   reliability. Count finite closed representative episodes once, use a counted win
   fraction and corresponding Wilson bound, carry measured session coverage, and remove
   the PF99/no-loss reward. Below 30 measured episodes or five entry sessions, positive
   evidence cannot exceed the unproven baseline. The payoff bonus scales with measured
   losses up to the same 30-example floor. Existing detector/family columns and
   Expected-R's separate calibration remain unchanged; v1 remains replayable.
3. **Trade-data readiness — EXISTING REPAIR VERIFIED ON A COPY.** Keep outcome quality and personal fit separate. Use confirmed
   trade labels and first-fill/instrument identity; a quick like supplies no setup tag,
   a machine guess is not confirmation, and missing risk is never invented. Reuse the
   existing TJ-9Q repair and Mentor label workflow, prove repair on a journal copy,
   rather than learning from the wrong instrument or treating unlabelled trades as
   negative choices. The 2026-09-22 copy apply moved 231 fills, refused none, preserved
   all 216 trades / 189 annotations / 621 executions, stranded no annotation and left
   no repair update pending. TJ-9Q already owns the implementation, so no duplicate
   repair or label UI is built. Stored-journal application retains its desk-down,
   market-closed, daytime, trader-only gate #162. Confirmed labels remain the trader's
   act; personal-fit ranking is deferred until coverage can support an honest test.
4. **RECONSIDERED ON COPIES; prospective gate open.** Compare old and corrected scores on copied facts,
   naming any reconstruction and look-ahead limits. Freeze a prospective challenger
   before observing its results; the first trial is 20 new entry sessions followed
   by the five-session outcome wait, with success/downside/rollback limits fixed
   beforehand. No weight optimization from the four-session assessment sample.

Both packets were built test-first and reproduced on isolated review checkouts.
SP1's new tests: 21 green on the repair, 20 red when production was restored to
the parent. SP2's review: 10 new tests green and all 10 red with the fix restored
to the parent. The integrated branch is `codex/setup-score-repair-2026-09-22`;
the full suite reported 10,293 passed / 14 skipped / 72 subtests and two failures:
one new fixture-contract omission (repaired) and the known timing-only freshness
flake (green alone). All 78 affected and new tests passed after that repair;
ruff, smoke 7/7 and source selftest 97/97 passed. This is not a clean full-suite
exit, so the exact result stays visible for the next integration run.
The desk checkout is not switched or rewritten while running. The live validation
and five-session outcome gates remain open after the code passes.

| Phase | Packets | Status |
|---|---|---|
| 0.33 The trader journal — Day Review, Week Review and the overnight voice | TJ-1 … TJ-8 | TJ-1 MERGED 2026-09-18 (`e00b734a`, reviewer GO after four rounds; live gate #145 owed at the next restart); TJ-1L (two-column layout, presentation only) MERGED (`86b86bcb` is an ancestor of `main` - verified 2026-09-19 with `git merge-base --is-ancestor`; this row said "unmerged" in error); TJ-2 MERGED 2026-09-18 into local `main` (`d3ae3aff`; durable session bars and four pure tables; gates #152/#153 owed); **TJ-3 MERGED 2026-09-19 (evening)** (`claude/tj3-note-markers` → `lead/p033-integration2` `72647104`; Day Review note markers, a mark on a bar only when it happened during it, the Alert Center's chart proven unchanged; gate #147 owed, after #146/#152's bars back-fill); **TJ-15 MERGED 2026-09-19 (night)** (`claude/tj15-miss-contrast` → `lead/p033-integration2` `fb3f55e9`, slot position fixed `1f260ffa`; the pure `evidence_contrast` with two floors, the deterministic `miss_contrast` slot inside stage 1 above the pair that closes it, D1 decisions only; gate #160 owed) and **TJ-14A MERGED 2026-09-19 (night)** (`claude/tj14a-mentor-card` → `lead/p033-integration2` `e8c04f88`; TJ-14 items 1 and 6 - the Mentor card's What I see / What I expect split with a forced prediction click, a row's timeframe and its horizon always agreeing at the WRITER, `trade_mentor_context_v2` and the internals strip; gate #159's first clause owed); **TJ-14B MERGED 2026-09-20** (`claude/tj14b-mentor-questions` → `lead/p033-integration2` `161e905c`; TJ-14 items 2-5 - the Mentor question registry in which every kind names the reader of its answer and four kinds ship DORMANT until that reader exists, the budget of three with the remainder counted and carried, one card one import with three reserved pulls a day, and same-session fills; gates #159 and #163 owed); **TJ-10 MERGED 2026-09-20** (`claude/tj10-read-grader` → `lead/p033-integration2` `57b44ca9`, integration fix `3e52d94f`; the read grader, the prediction ledger, four congruence lines and the deterministic `read_grades_mature` slot - no model anywhere; gate #155 owed, and it needs the TJ-2A session tape to exist first); **TJ-16 MERGED 2026-09-20** (`claude/tj16-prediction-contrast` → `lead/p033-integration2` `c4a760e5`; the prediction ledger beside three naive baselines on the SAME stamps, the deterministic `prediction_contrast` slot directly after `miss_contrast`, and the Stage 2 `observation_tags` tagger that never sees an outcome and whose verifier re-checks the reply's own bounds; readers only, no page; gates #164-#167 owed); **TJ-4 MERGED 2026-09-20** (`claude/tj4-day-story` → `lead/p033-integration2` `d929e34f`; the pure hash-stable day pack, the overnight day story that narrates only measured rows and rejects a disagreeing output WHOLE, the rolling D1 view, and a night that sweeps the redos a daytime click queued; gate #148 owed); **TJ-5 MERGED 2026-09-20** (`claude/tj5-week-review` → `lead/p033-integration2` `1b9d77e0`; Week Review first in Weekend Prep - five day cards on one payload that computes nothing, the week pooled once in `day_report_card.week_from_cards` with no best family taken from day-winners, an unpacked and an unreadable day each NAMED rather than counted as quiet, and the Saturday-only `week_review_narration` slot narrating only the week's own packs; gates #149, #170 and #171 owed); **TJ-6 MERGED 2026-09-20** (`claude/tj6-ideas` → `lead/p033-integration2` `a7809d7c`, ninth slot-order pin `be435cd7`, fix `41d3f759`; the desk's AI has a voice - up to three grounded ideas a night, nothing to cite meaning no model call, a night that asked being DONE, and a KEEP the trader's own click that freezes a baseline and is checked by two non-overlapping Wilson intervals or not called a change at all; gates #150 and #172-#174 owed); **TJ-7 MERGED 2026-09-20** (`claude/tj7-mood-fields` → `lead/p033-integration2` `b63db7af`, guard amendment `ae7c06c7`, follow-up merge `0a0a0be4`; the LAST building packet - a mood is a field the desk REPORTS and nothing acts on it: ONE additive journal key refused loudly at the writer, one optional strip on both surfaces, a citable pack section, one key in each closed evidence list, a point-in-time context field and ONE Day Review line; gates #151 and #175 owed) — **every building packet of Phase 0.33 is now merged on `lead/p033-integration2`**; TJ-8 PLANNED (waits on the live gates); **TJ-11 MERGED 2026-09-19** (`claude/tj11-walkaway-v2` → `lead/p033-integration` `a89ec7d5`; walk-away v2, `REAL_MISS_V1`, the skill line, an additive `decision_session`; direction reversed the same evening by **TJ-11F MERGED 2026-09-19** (`claude/tj11f-decision-session` → `lead/p033-integration2` `f00ec302`; an after-close decision belongs to the session it JUDGED); gate #156 owed, reworded) and **TJ-13A MERGED 2026-09-19** (`claude/tj13a-night-slates` → `lead/p033-integration` `9eaae1dd`; nights only seven days, night slates, four overnight repairs; gate #158 owed); **TJ-9 … TJ-13 otherwise PLANNED 2026-09-19** (trader-approved after the 2026-09-18 review-loop audit: forced 09:00 trade labels, read grader + congruence, walk-away v2, report card, night re-budget; order in 12.5); **second-look amendments and TJ-14 … TJ-16 PLANNED 2026-09-19** (trader: "Yes add all of this" — prediction click, skill line against a base rate, tracked ideas, instrument-aware money lines, tag provenance, miss contrast, staleness line; the Mentor asks only for what the desk is missing) |
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
  **A packet that adds or MOVES a runner slot runs `-k "slot or stage or slate or order"`
  plus the TEN order pins, not only `tests/test_ai_jobs_runner.py`** (its
  `EXPECTED_SLOT_ORDER`) — `tests/test_veto_cohort_grading.py`,
  `tests/test_ws_10d_market_story.py`, `tests/test_ws_rp_shared_report.py`,
  `tests/test_tj13b_local_large_provider.py`, `tests/test_tj13b_probe_guards.py`,
  `tests/test_opt_in_evidence_scopes.py` (the seventh, amended by the lead 2026-09-20 at
  TJ-16's merge and again at TJ-5's: the `ai_summary` → `ticker_briefs` pair keeps its ORDER
  and only `day_review_narration`, `observation_tags` and `week_review_narration` may sit
  between them) and `tests/test_tj5_week_slot_and_slate.py` (the eighth, 2026-09-20: it pins
  `observation_tags` → `week_review_narration` → `ticker_briefs` as an ADJACENCY, so a new
  stage-2 slot between any of those three breaks it), plus
  `tests/test_setup_research_pipeline.py` (the NINTH, 2026-09-20 at TJ-6's merge: its
  `test_setup_research_is_appended_to_the_nightly_slate` used to assert `setup_research` is
  the LAST slot and now says `setup_research` ends stage 3 with **only `improvement_ideas`
  allowed to follow it**, so a tenth slot appended at the end of the night fails there), plus
  `tests/test_tj9e_night_slot.py` (the TENTH, 2026-09-21 at TJ-9E's merge: it pins
  `exit_note_fields` in stage 2 BETWEEN `week_review_narration` and `ticker_briefs`, which is
  now the full stage-2 order `ai_summary, day_review_narration, observation_tags,
  week_review_narration, exit_note_fields, ticker_briefs`).
  TJ-15's targeted runs were green while three order assertions elsewhere were red; TJ-5's
  targeted runs were green while R4's one-owner minimum-height pin in another packet's test
  was red; TJ-6's targeted runs were green while this ninth pin, in a file it never touched,
  was red — the same lesson twice in one day, and only the full suite saw either.
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

**BUILT and MERGED 2026-09-20**, branch `claude/tj4-day-story` (tip `6c1e2506`), merged
`d929e34f` into `lead/p033-integration2`; **not on `main`** until the lead's next
fast-forward. FOUR review rounds by reproduction (NO-GO ×4, each on the previous round's
fix; round 4 confirmed the product whole and its one blocker was a racing TEST, fixed by
the lead at `6c1e2506`, 25 of 25 green). What landed, and what the text below now gets
wrong:

- **Change 2's slot position is superseded by the packet's correction 1.**
  `day_review_narration` is registered INSIDE stage 2 directly after `ai_summary` and
  before `observation_tags` and `ticker_briefs` — **not** after `market_story_narration`.
  Gate #158 wants the day story finished before 23:30 Pacific and `ticker_briefs` reserves
  120 minutes in front of it; it cannot move further forward either, two existing pins
  putting `ai_summary` directly after `measured_report`. The cited `runner.py:775-786` is
  stale and never held `market_story_narration`'s `JobSlot`. Slot positions: decision 0018
  addendum 2026-09-20.
- **Change 2's `verdict right|wrong|unresolved` is not the measured vocabulary.** TJ-10's
  grader answers `right`, `wrong`, `flat`, `pending` and `unmeasured:<reason>`, so the
  schema bounds the field as a string and the RULE — equality with the read row — is what
  constrains it.
- **Change 3's inputs are superseded.** The rolling view is built from the trader's D1
  PREDICTION clicks and D1 notes of the last `LATELY_SESSIONS` only;
  `market_thesis.current_theses` and the weekly pack's measured facts are NOT read (the
  thesis store is empty and stays so, and an empty store therefore cannot empty the view).
- **Change 4's "outside market hours" is narrower in the build:** the button consults
  `ai_jobs.window.launch_allowed`, i.e. the NIGHT window (TJ-13A item 5's rule). Change 4
  also owes one more sentence now that the marker is real: *the nightly slot honours the
  marker for ANY session, sweeping at most `REDO_SWEEP_LIMIT` (3) of them oldest first and
  spending that budget only on sessions it NARRATES* — without the sweep, "queued for
  tonight" was true for one day in fifteen. A Redo BUILDS that session's pack first, on the
  page's own off-Qt seam, and only ONE redo is in flight at a time;
  `day_review_pack.validated_session` fails CLOSED (exactly `YYYY-MM-DD`, a real exchange
  session, already closed) on the page's queue branch, on its launch branch and in
  `run_ai_jobs.py`'s `--session` guard.
- **`--session` is new and narrow:** `run_ai_jobs.py --session YYYY-MM-DD` is accepted only
  together with `--slot day_review_narration` (anything else is a parser error, exit 2,
  nothing run) and reaches that ONE slot through an additive `run_slots(session_override=)`;
  every ledger row that run writes — the window refusal included — is keyed to the session
  the slot WORKED on.
- **The pack has TWELVE sections** after the 2026-09-19 amendment, not the seven change 1
  lists inline; `SECTIONS` in `scripts/day_review_pack.py` is the list, and `pack.json`
  stays under `sessions/<date>/` inside `day_review_index._prune`'s 40-session delete path
  because it is rebuildable (the hash-stability test proves it).
- **`market_context_ledger.py:38-69` is stale:** `regime_shift_event` is at `:53-76` and
  `STREAM_REGIME` at `:42`.

`selftest.LAZY_ENGINE_MODULES` gains `day_review_pack` and `ai_jobs.day_review_narration`
(source selftest 93/93 → 95/95). Tests: `tests/test_tj4_day_pack.py`,
`tests/test_tj4_narration_slot.py`, `tests/test_tj4_d1_view.py`,
`tests/test_tj4_day_review_page.py`, `tests/test_tj4_redo_sweep.py`,
`tests/test_tj4_review2_fixes.py`, `tests/test_tj4_redo_one_at_a_time.py`,
`tests/test_tj4_redo_session_cli.py`, `tests/tj4_support.py`. Long form: DESK_INTERNALS
"TJ-4". Gate #148 (all three clauses) is owed.

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
   `trades` (count, wins/losses, net R or P&L, one line each), `mood` (TJ-7, BUILT
   2026-09-20: citable ids through `day_review_pack.mood_source_id`, `{}` for a session
   nobody clicked). Every item carries a `source_id`.
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

Live gate **#148** (owed, THREE clauses; the packet is merged, so these are all read on a
real morning):

1. **The morning-after story.** The morning after an overnight run the Day Review of
   yesterday opens on a story with a headline, a were-you-right list whose every line names
   one of the trader's own notes, and a chased verdict that cites the pasted forecast or
   says `unknown`; the job ledger shows `day_review_narration` OK; the D1 view lists the
   trader's open theses. **Nothing to grade is a PASS, not a failure:** until the trader
   clicks a prediction on a Mentor card the list is honestly empty.
2. **Every printed verdict equals the ledger's.** Read that session's
   `market_read_grades` rows beside the printed were-you-right list and compare by
   `read_id`: every verdict agrees, and the page's `graded K of N reads` matches the story
   file's own count.
3. **A DAY Redo leaves a marker, starts no process, and the night clears it.** Press **Redo
   story** by day on any session in the picker: the page says it is building that session's
   facts, then BOTH a `sessions/<date>/pack.json` and a `sessions/<date>/redo_requested.json`
   appear, no `run_ai_jobs` process starts, and the next night's ledger row for
   `day_review_narration` names that session in its reason, its `narration/<date>.json` is
   rewritten and the marker is gone. (First reading: the live `day_review/sessions/` held
   three folders and NO packs on 2026-09-20, so the first Redo after this merge is also the
   first pack build for that session.)

Gate **#158** is touched, not changed: its "day story finished before 23:30 Pacific" clause
is readable for the first time, because the slot now exists.

##### TJ-5 — Week Review, Weekend Prep's first step — **BUILT and MERGED 2026-09-20**

**BUILT** on `claude/tj5-week-review` (tip `e93ffe49`, reviewer GO in round 1 at `f7fac9cb`)
and **MERGED into `lead/p033-integration2` `1b9d77e0`** — not on `main`, so nothing of it is
live until the trader says to fast-forward. Gates **#149** (re-worded below), **#170**
(pinned as a number, owed live) and **#171** (new) owed. Long form: DESK_INTERNALS "TJ-5";
slot position: decision 0018 addendum 2026-09-20.

*Goal:* *"5 of these days collated into one tab … to see if I was right, to see if I chased
in bad news environments, and to compare what I actually said to what I did."*

What existed: `WeekReviewPage` (`scripts/ui/panels/weekend_prep_panel.py`, already step 1,
reads on a worker), `weekend_prep_service.STEP_IDS`, `evidence_stats.WEEK_SESSIONS = 5`,
`market_story_rollups` weekly pack (coverage note, sessions missing), `ai_jobs/synthesis.py`
(`weekly_synthesis`, gated, separate — untouched), the frontier providers in
`scripts/ai_summary.py` (`request_ai_summary` is at `:4013`, not `:3810` — and it is now
IRRELEVANT here: the provider seam this packet uses is TJ-13B's `ai_jobs/provider.py` and
OpenAI stays a setting that is off). The two files this packet edits are
`scripts/ui/panels/weekend_prep_panel.py` and `scripts/ui/services/weekend_prep_service.py`.

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

   **AS BUILT (2026-09-20).** The slot sits in stage 2 directly after `observation_tags`
   and directly before `ticker_briefs` (whose 120 minutes of reserve it must not queue
   behind), SATURDAY slate only through the EXISTING `runner.WEEKEND_ONLY_SLOTS` — no
   second constant — with Sunday picking it up only when Saturday left it owed. Its
   `reserve_minutes` is `TIMEOUT_SECONDS / 60 + RESERVE_MARGIN_MINUTES` (40 min) until a
   TJ-13B probe measurement overrides it. `EVIDENCE_KEYS` is CLOSED (five packs, five day
   stories, the weekly rollup, TJ-15's and TJ-16's contrast packs — never bars, never the
   lake, never the journal stream); every bound comes from the INPUT and the verifier
   re-checks it after the reply; every citation is session-qualified; a breach rejects the
   WHOLE answer and leaves last Saturday's file byte-identical. Below `MIN_NARRATED_DAYS`
   (3) it writes a deterministic scaffold saying `narrated K of 5` with status `skipped`
   and loads NO model; a narrating run makes exactly ONE call, on the MEDIUM local model
   until TJ-13B's probe row exists, the reason recorded under `provider.LEDGER_FIELD`.
   `openai` is refused by the provider seam before anything is sent and the slot records a
   FAILED row rather than crashing the night. The "week's kept ideas" card is TJ-6's: the
   payload key `ideas` is reserved and empty.
3. **AMENDED 2026-09-19 (trader): the week and the month.** Under the five cards, one
   deterministic strip of TJ-12's report-card lines re-cut by exchange week for the last
   four weeks and for the calendar month to date (same functions, longer window, `n` on
   every cell, a week under its floor named and not ranked). No new page, no model.

   **AS BUILT (2026-09-20): "same functions, longer window" is satisfied by
   `day_report_card.week_from_cards`, NOT by `week(sessions)`** — the tester's premise was
   REFUTED at step 0: `week()` needs TJ-11's `WalkawayDay` objects (`_rows_of` is
   `getattr(walkaway, name, ())`, `_skill_window` reads `walkaway.skill`) and a pack carries
   the BUILT lines and their integers instead. The pooling therefore lives once, in the
   module that owns what a line IS, and `week()` delegates to it. `how_fresh` is asked PER
   SESSION by the strip, always WITH a session (it is deliberately out of the pack's hashed
   body; an EMPTY session pools every night the ledger tail holds — the advisory TJ-12's
   review 3 recorded for this packet).

Tests: card count equals sessions in the week; a missing day is named; the slot refuses a
week with fewer than three narrations; provider fallback to local when no key; slot order.
As built: `tests/test_tj5_week_narration.py`, `tests/test_tj5_week_slot_and_slate.py`,
`tests/test_tj5_week_strip.py`, `tests/test_tj5_week_review_page.py`,
`tests/test_tj5_review1_followups.py`, `tests/tj5_support.py`; the slot-order pins are now
EIGHT files (12.3).

Live gate **#149** (re-worded 2026-09-20 at the merge): on Saturday Weekend Prep opens on
Week Review with five cards and a week story whose examples cite the trader's own notes, and
the ledger shows **ONE model call — a MEDIUM-local one until TJ-13B's probe row exists**,
with the ledger saying why (`model_attribution.fallback_reason`); it is a LARGE-local call
only after that measurement. **UNREADABLE until the post-close tick has written packs for
three of a week's sessions**: on today's live store the honest first state is `0 of 5
sessions have facts` and `narrated 0 of 5` — a deterministic scaffold and no model call at
all, which is the feature working, not the gate passing.

Live gate **#171** (TJ-5's slate, the first Saturday night the slot runs): the ledger shows
ONE `week_review_narration` row on FRIDAY's session date, and NO such row on any weeknight.

##### TJ-6 — The desk's AI has a voice: ideas — **BUILT and MERGED 2026-09-20**

**BUILT** on `claude/tj6-ideas` (tip `d78945f0`, two review rounds by reproduction — NO-GO
then GO) and **MERGED into `lead/p033-integration2` `a7809d7c`**, with the lead's ninth
slot-order pin `be435cd7` and one fix `41d3f759` on top — not on `main`, so nothing of it is
live until the trader says to fast-forward. Gates **#150** (as written), **#172**, **#173**
and **#174** owed. Long form: DESK_INTERNALS "TJ-6"; slot position: decision 0018 addendum
2026-09-20.

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
three per night; slot order; the card's Keep writes state and nothing else. As built:
`tests/test_tj6_an_idea_is_only_a_suggestion.py`, `tests/test_tj6_ideas_are_grounded.py`,
`tests/test_tj6_dedupe_and_dismissed.py`,
`tests/test_tj6_measurables_and_the_frozen_baseline.py`,
`tests/test_tj6_keep_is_the_traders_act.py`, `tests/test_tj6_ideas_card.py`,
`tests/test_tj6_ideas_reach_both_payloads.py`, `tests/test_tj6_ideas_slot_and_slate.py`,
`tests/test_tj6_ideas_store_and_paths.py`, `tests/test_tj6_one_ask_a_night.py`,
`tests/test_tj6_builder_followups.py`, `tests/tj6_support.py`; the slot-order pins are now
NINE files (12.3).

**AMENDED 2026-09-19 (trader): advice is checked, not just given.** A `process` idea must
name ONE measurable the desk already computes (a veto reason's count and real-miss rate, a
report-card line, a TJ-14 question's answer mix) or it is dropped like an idea without
evidence. Keeping it freezes a baseline — that measurable over the `LATELY_SESSIONS` before
the keep — in `AI_IDEAS_STATE_FILE`. Week Review then prints, for every kept idea, before
and after with both `n`, deterministically, until the trader retires it; under the floor it
says "too few to call". The model never grades its own advice.

**AS BUILT (2026-09-20, with review 1's blocker fixed).** The slot is APPENDED LAST inside
stage 3, after `setup_research`; `uses_model=True`, no `model_free_kwargs`,
`RESERVE_MINUTES = 10.0`, `max_attempts=2`, and it is NOT in `WEEKEND_ONLY_SLOTS` — the
packet says up to three ideas A NIGHT — so Sunday offers it only when the weekend left it
owed. Four order pins were updated (`EXPECTED_SLOT_ORDER`, the byte-pinned slate in
`tests/test_veto_cohort_grading.py`, `tests/test_tj13b_local_large_provider.py`'s `expected`
and `tests/test_tj13b_probe_guards.py`'s `set_aside` — NOT `pinned_at_e8c04f88`) and the
lead's amendment `be435cd7` made `tests/test_setup_research_pipeline.py` the NINTH.

*Corrections to the text above, from the code.* (a) **ONE ask a night on every path**:
nothing to cite is `skipped` BEFORE the model loads, a night that asked and stored nothing is
**`ok`** with its per-reason drop counts and an ASKED marker (`ai_ideas_asked.json`,
temp-and-rename, beside the store and never in the trader's state file), and only a whole
REJECTION is an attempt (capped at 2). The blocker this fixes: a slot that calls the model
and then returns `skipped` with no artifact re-asks on all sixteen passes of the scheduled
task, because `skipped` is in neither `ledger.CANONICAL_COMPLETION_STATUSES` nor
`ledger.ATTEMPT_STATUSES`. (b) A repeat **APPENDS** a row with the same `idea_id` and a
higher `seen_count`, and `read_ideas` folds on read (last row wins) — item 1's "increments
`seen_count` instead" would have rewritten a row and lost the first sighting, which the
evidence-store rule forbids; the window is 60 EXCHANGE sessions, and "a dismissed idea never
returns" is checked on the normalised TEXT, because the id carries the session an idea was
first seen in. (c) **It SHIPS WITH TWO MEASURABLES, not three**: the amendment's *"a TJ-14
question's answer mix"* has no pooled reader (`day_report_card.process_line` builds
`origin_answers` for ONE kind over ONE session and `_pool_cards` drops the key), so the
CLOSED registry names `report_card_did_well_rate` and `veto_reason_real_miss_rate` — adding
the third later is one entry plus a pooled reader (follow-up row **TJ-6M** in 12.5). (d)
`measure` reads the LAST WRITTEN `miss_contrast` pack (`read_latest`) and the day packs' own
stored card lines — never a 20-session build behind a Keep click — and records WHICH pack it
read; an unreadable reader is `unmeasured` with its reason, never a zero. (e) The cap of
three is counted on the **USABLE** ideas, before the dismissed filter, by the one predicate
`drop_reason` the verifier and the writer share; the item schema is validated SHAPE-ONLY
because the shared validator's `enum` and `maxLength` would turn two named DROPS into whole
rejections. (f) Week Review says "higher"/"lower" **only when the two Wilson intervals do not
overlap** (`evidence_contrast.rate`, imported), otherwise "no clear change", and prints no
rate at all under `MIN_REPORTABLE_N`. (g) `AI_IDEAS_FILE` / `AI_IDEAS_STATE_FILE` are new
`project_paths` constants resolved at CALL time; the card is `scripts/ui/widgets/ideas_card.py`
on BOTH pages, stating its floor as a size HINT. (h) **TJ-7's mood/process fields are NOT in
`EVIDENCE_KEYS` yet** — that set is closed and tested, so TJ-7 adds one key plus that test's
list. **CORRECTED 2026-09-20 at TJ-7's merge:** TJ-7 added the ONE key `mood` (and no
`MEASURABLES` entry), and **no TJ-6 test needed editing** — TJ-6's test does not enumerate
`EVIDENCE_KEYS`, so "plus that test's list" was stale. (i) TJ-6 added NO Mentor question kind: it asks through a button on the card, so
`mentor_questions.BUDGET` and its registry are untouched.

Live gate **#150**: the morning after, Day Review shows up to three ideas each citing a
note or a walk-away row; Dismiss hides one for good across a restart; Keep on a program
idea shows it under For WISHLIST. **UNREADABLE until a night has written packs and a story
for the sessions it reads**: on today's live store the slot stores nothing and records
`skipped` with no model call at all, which is the feature working, not the gate passing.

Live gate **#172** (the check, the first Saturday after a kept `process` idea has run its
window): Week Review prints that idea with a before and an after, each with its own `n`, and
says "too few to call" rather than naming a winner under `MIN_REPORTABLE_N` — and "no clear
change" rather than higher or lower when the two intervals overlap.

Live gate **#173** (the invariant, the night after a Keep): the ledger shows the
`improvement_ideas` slot ran and `ai_ideas_state.json` is **byte-identical** to what the
trader's click left.

Live gate **#174** (review 1's blocker on the live desk, the first morning after a full
night): the ledger holds **at most ONE `improvement_ideas` row per session that loaded a
model** — and on today's desk, with zero packs, that is a repeating `skipped` row saying
"nothing to cite" and NO model call at all.

##### TJ-7 — Mood and process: the bones only

**BUILT** on `claude/tj7-mood-fields` (tip `bcfdd918`, reviewer **GO in round 1** at
`83318aa2`, no blockers) and **MERGED into `lead/p033-integration2` `b63db7af`**, with the
lead's guard amendment `ae7c06c7` and the follow-up merge `0a0a0be4` on top — not on
`main`, so nothing of it is live until the trader says to fast-forward. **This was the LAST
building packet of Phase 0.33: every building packet is now merged on
`lead/p033-integration2`.** Gates **#151** (three parts, below) and the new **#175** owed.
Long form: DESK_INTERNALS "TJ-7".

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

**AS BUILT (2026-09-20).** ONE additive row key `mood` (schema `trader_mood_v1`) holds the
whole block — item 1's three `build_entry` ARGS stay, but a row carries one key. It is `{}`
when nothing was clicked, ABSENT on every row written before the packet and never a
default; `market_journal.mood_of` and the point-in-time `mood_at` are its ONLY readers.
`MoodFieldError` is raised at `build_entry` and asked again at `is_publishable`; the
over-long note is REFUSED, never truncated; a mood on a MACHINE row is refused through
`is_machine_entry`'s own rule (`MACHINE_MOOD_REFUSAL`). `scripts/trader_state_tags.py` OWNS
the cap (`MAX_STATE_TAGS`) and the picklist. ONE strip
(`scripts/ui/widgets/mood_strip.py`) serves both surfaces, rides `day_close`
(`mentor_questions.KIND_DAY_CLOSE` — no new registry kind, the budget of three unmoved),
never greys TJ-9's Save, and `TradeMentorCard.set_questions` became a MERGE for the WHOLE
questions box. The day pack's `mood` section mints citable ids through the ONE seam
`day_review_pack.mood_source_id` (shared with the Day Review payload) and projects each
item through `MOOD_ITEM_FIELDS`, `inputs_hash` clock-free and moving when a mood moves; Day
Review prints ONE mood line off the ONE payload. `market_read_grades.context_for(...,
mood_entries=)` takes only the mood known AT THE STAMP. **The tagger never sees a mood**,
and TJ-7 adds NO nightly slot, so the NINE slot-order pins do not move; source selftest
stays **97/97**.

*Lead decisions ratified, each the trader's to overrule.* (a) **ONE row key `mood`** — a
block the readers can be counted on one hand beats three loose keys every reader must learn.
(b) **A mood with no plan answer is still filed** — a click the trader made is never lost
because a different question on the same row went untouched. (c) **An over-long note is
REFUSED, not truncated** — the ledger is append-only and has no second chance; the UI caps
its own input instead. (d) **The mood rides `day_close`** — a second registry kind would
need a second named consumer and would spend one of the budgeted three. (e) **Both closed
evidence lists gained `mood`** — the week story and the ideas slot cannot cite what they
were not handed; neither gained a `MEASURABLES` entry, so no kept idea is ever checked
against how the trader felt. (f) **A second click on a face CLEARS it** — a misclick is not
an answer, and an exclusive `QButtonGroup` gives one no way home. (g) **The Day Review
payload MINTS through the pack's seam rather than reading the written pack** — an evening
mood arrives after the post-close pack, and a stale line on that very save is what gate
#151 asks the trader to look at. (h) **The mood guard checks CONTACT with the review policy,
not words in a docstring** (`ae7c06c7`) — as written it flagged any file holding both
tokens, which three modules the packet required TJ-7 to touch do in prose, so it was
unsatisfiable by its own packet; narrowed, the four reworded sentences came back.

*Correction to TJ-6's amendment (h) above:* TJ-6's own test does not enumerate
`EVIDENCE_KEYS` (it checks for bar/lake/tape/tick/warehouse words and `package <= keys`), so
"plus that test's list" was stale — **no TJ-6 test needed editing**.

*Recorded follow-ups from TJ-7's review (unscheduled, no wave).* **(i) A verifier rule
against the night pairing a mood with a RESULT.** Nothing forbids "you lose when tired"
today: the week package carries `mood` beside `misses` and `walkaway_totals` and the
verifier only re-checks that a citation is in `allowed_source_ids`. The one live fence is
TJ-6's closed `MEASURABLES`, which mood is not in, so a `process` idea naming a mood is
DROPPED. Gate **#175** is adopted in its place until a narration rule is built. **(ii) A
re-offered Mentor subject whose prompt CHANGED keeps its old label.** `set_questions` merges
on `(kind, subject_id)`, so the same subject re-offered with a different `prompt` or
`options` shows the OLD prompt and the OLD combo items. Latent: no shipped kind varies
either for a fixed `subject_id`; documented in the code at
`trade_mentor_card.set_questions`.

Live gate **#151**, readable in THREE parts (the first two the same evening): (1) the
trader clicks a face on the journal tab or on the session's last Mentor card, saves, and
Day Review's "What you said" column shows the mood line with the score, the chips and the
plan answer, `n` beside the count and NO percentage; (2) a note saved with the strip
untouched writes exactly the row it always wrote (`mood` present and empty) and the line
reads "No mood recorded yet for this session (n 0)."; (3) **the citation half — "the next
story cites it" — is UNREADABLE until a night has written a pack AND a story for a session
that carries a mood.** The live day-review folder held ZERO packs on 2026-09-20, so the
first honest reading is `narrated 0 of N`, which is the feature working and not the gate
passing.

Live gate **#175** (new, the first Saturday after a week that carries a mood): the week
package holds the `mood` section with its session-qualified ids, and the week story cites
one of them or says nothing about mood at all — **never a mood paired with a result**.

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

**BUILT and MERGED 2026-09-20**, branch `claude/tj10-read-grader` (tip `2967e4a6`), merged
`57b44ca9` into `lead/p033-integration2` with the integration fix `3e52d94f`; NOT on `main`.
Long form in DESK_INTERNALS "TJ-10". Items 1-6, TJ-16 item 1's context snapshot and the
nightly slot, over `scripts/market_read_grades.py` (new, pure, no model),
`scripts/market_thesis.py` (vocabulary and `EXTRACTOR_VERSION` only),
`scripts/ui/services/day_review_service.py` (`reads` + `congruence` in the payload,
`build_reads_for` the named post-close seam and the ONLY writer),
`scripts/ui/panels/day_review_panel.py` (`verdict_chips`, `congruence_text`, the chip styled
by object name and a dynamic property), `scripts/ui/theme.qss`,
`scripts/project_paths.py` (`DAY_REVIEW_READS_DIR`),
`scripts/ai_jobs/read_grades_mature.py` + its slot in `ai_jobs/runner.py`. Two reviews by
reproduction, NO-GO then GO. **What differs from the plan text above:** (a) the flat band's
size was not in the packet - `FLAT_BAND_ATR = 0.25` of the benchmark's point-in-time daily
ATR(14), edge inclusive, is the LEAD's number, chosen 2026-09-19, stamped
`flat_band_rule: "atr_0.25_v1"` on every row, and it is the TRADER's to change, never from
one session's result; (b) the store is STRICTER than "append-only" - `append_grades` refuses
a gradable CLICKED grade whose context is only the named absence (`ContextMissingError`), the
grade is held back with `grader_gap: "context_unbuildable: ..."` and the next pass retries,
and `verdict_rank` lets a row be appended only when the rank RISES so an `unmeasured` never
supersedes a `pending`; (c) congruence has a FOURTH kind, `m5_picks_side_mix`, appended
whenever the session holds a rest-of-day read, because TJ-15 measured M5 as about half the
reviewed decisions, and an under-floor side mix reads `too_few` rather than a verdict;
(d) the deterministic slot `read_grades_mature` is REGISTERED in this packet (after
`theta_pick_grading`, before `miss_contrast` / `market_story_rollups` / `measured_report`,
inside `_deterministic_stage`) - decision 0018 carries the addendum.

Live gate **#155**: the morning after a session, in this order. (a) **`day_review/bars/
<date>.parquet` EXISTS** — the live home has no `day_review\bars\` directory at all today,
so until the post-close tick (TJ-2A's `build_session_bars_for`) or a back-fill writes one,
every rest-of-day read is honestly `unmeasured:no_completed_bar_after_the_stamp` and the
gate cannot be judged. (b) `day_review/reads/<date>.jsonl` exists and holds one row per
read, written by the post-close tick and not by opening the page. (c) Every gradable row
carries a `context.internals` block with schema `trade_mentor_context_v2`, and no CLICKED
row carries the named absence — the store refuses it, so a clicked read missing from the
file means the snapshot could not be built and its payload row says
`grader_gap: context_unbuildable`. (d) Each Mentor answer with a view shows right / wrong /
flat; a five-session view reads `pending <date>` with the fifth session's date, never
`wrong` and never a zero move; after the nightly `read_grades_mature` slot runs the file has
GROWN and no `unmeasured` row supersedes a `pending` one. (e) The congruence lines print
under the story with their counts, their timeframe, their missing side and the SOURCE of the
read they compare (a click says it was clicked; an extraction says "we read your note as
…"), the like counts match the day's likes, and an M5 line appears whenever the session held
a rest-of-day read. (f) No push, no alert, no Focus row and no `review_policy.json` write
happened. Do not tune `FLAT_BAND_ATR` from one session (plan.md sec 6).

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

**BUILT and MERGED 2026-09-20**, branch `claude/tj12-report-card` (tip `1a88d4a4`, THREE
review rounds by reproduction against read-only copies of the live stores - NO-GO, NO-GO,
GO), merged `843a3f02` into `lead/p033-integration2`. **Not on `main`**, so nothing of this
is live until the trader says to fast-forward. Source selftest **95/95 -> 96/96**
(`day_report_card` joins `selftest.LAZY_ENGINE_MODULES`). Follow-up owed: **TJ-12F** below.

Six deterministic lines at the head of Day Review, above the story, each with its `n` and
each clickable to the table behind it: **Did well** (likes/claims that were real runs, best
family by the ONE Wilson bound, none named under `MIN_REPORTABLE_N`), **Missed** (TJ-11's
real misses and their shared reason), **Your reads** (TJ-10 tally), **Congruence** (TJ-10),
**Process** (trades, labelled or not per TJ-9, left on the table) and **How fresh** (the
AMENDED block below). Pure
`scripts/day_report_card.py` over the day's ONE payload (AS BUILT: the card is minted on the
Day Review worker inside `read_day`, not read back out of the pack; `build_pack` accepts a
`report_card=` and mints one `source_id` per line, and **`build_pack_for` hands it the
session's card since the follow-up `c37d5673`** - five day lines through
`day_report_card.pack_card`, `How fresh` kept out of the hashed body); computes no new statistic;
a line whose input is missing says so. TJ-5's strip re-cuts the same lines by week and month.

**AMENDED 2026-09-19 (trader, second look):** the **Did well** and **Missed** lines quote
TJ-11's skill line, so a count never stands without its base rate; and a sixth, smaller
line **How fresh** states what the card rests on — story written when, fills current to
which date, grades through which session, and any slot that failed last night by name (the
job ledger's last row per slot). A failed night is said on the page the next morning.

Live gate **#157** (RE-WORDED 2026-09-20 at the merge, the amendment having made the card
six lines): yesterday's Day Review opens on **SIX** lines - Did well, Missed, Your reads,
Congruence, Process, How fresh - whose numbers match the tables under them; with no trades
the Process line says so rather than printing zero; a line whose input is missing says so
and prints no number; and clicking each line scrolls to the table or section it names.

Live gate **#168** (`How fresh`, on the first desk open after a real night): the line names
the story's stamp, the last session with VERIFIED fill coverage and any slot whose LAST
deciding ledger row for that session is not `ok` - **only** those, `failed` and `degraded`
listed apart, a slot that ran and was later `skipped` NOT named (on 2026-09-18's ledger that
is `journal_import` and `ai_summary`, and nothing else). Opening a session older than the
ledger tail says *night status unknown for this session*, never "none reported trouble"; a
covered session with no rows says it has no rows; and on a desk with no AI store the line
says `night status unknown` and **no `ai_store/logs/` folder appears**.

Live gate **#169** (the wake, on the first Mentor cards after a real session): a trade the
desk cannot connect to anything the trader said before its first fill is asked where it came
from **at most once**, the prompt carrying its own caveat about the unread lanes and offering
`a_focus_pick`; a trade with a like or claim stamped before its first fill is NOT asked; a
date-only broker fill is never asked; at most three budgeted questions sit on a card; AWAY
prompts nothing; and an OPEN position past five exchange sessions is asked once a week
whether the thesis is intact. This is also where TJ-14B's clause *a trade from nowhere is
asked its origin once* is read. **Until TJ-12F lands**, the Process line must still name the
Focus-add and armed-alert lanes as unread and never print a bare `unplanned`; when TJ-12F
lands the plain wording returns on its own.

Live gate **#170** (TJ-5 landed 2026-09-20): the week strip's pooled `n` equals the SUM of
the day cards' `n` over the same window, and its rate comes from pooled COUNTS with the ONE
Wilson, never from the mean of the days' rates. **Pinned as a NUMBER** in
`tests/test_tj5_week_strip.py::test_the_weeks_pooled_counts_are_the_sum_of_the_day_cards`
(9 / 8 / 5 and 13 / 11 / 6 over three packed sessions, re-derived by hand at review); the
gate that remains OWED is the same sum read on the desk's OWN week.

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

**Items 2-5 BUILT and MERGED 2026-09-20 as packet TJ-14B**, branch
`claude/tj14b-mentor-questions` (tip `cfe137d1`, reviewer GO after three rounds), merged
`161e905c` into `lead/p033-integration2`; **not on `main`**. The pure
`scripts/mentor_questions.py` is the card's one description of every question it may carry:
nine kinds, each naming its trigger, its click options, the store it `writes`, its cadence,
its priority and the `consumer` plus `answer_key` that make it worth asking, with
`consumer_report()` walking the consumer's SOURCE by `ast` (a key named only in a comment,
a docstring or a bare string statement does not count, and the probe is never a call - a
calling probe would pass for `json.dumps`). What differs from the text above, and why:

* **FOUR kinds ship DORMANT, not asked and given no shim reader** (lead decision 2026-09-19
  on decision 0021 answer 28: *a question is ASKED only when its answer has a reader*). Each
  is registered in full - trigger, options, `writes`, `answer_key` and the consumer it WILL
  have - and carries `dormant_until` naming the packet that wakes it; `pending()` never puts
  a dormant kind on a live card and `consumer_report` reports it as `dormant` with that
  packet named. The four, and the one-field edit that wakes each (`dormant_until=""` plus the
  lane in `MainWindow._mentor_question_state`, which already names all five lanes):
  **`trade_origin` -> TJ-12** (the planned-vs-unplanned Process line; `trade_origin.planned_state`
  has no reader outside its own module today), **`open_position_check` -> TJ-12** (TJ-11's
  "long-hold rows" turned out to be `walkaway_day.LONG_HOLD_SESSIONS` setting a
  `not_judged_reason` on a CLOSED trade, which reads no answer), **`grader_gap` -> TJ-10**
  (nothing emitted `needs_trader_input` when TJ-14B was written; TJ-10 has since MERGED on
  this branch and its `grader_gap` field is on every grade row, so this one can be woken now
  as a small follow-up - it was deliberately NOT done inside TJ-14B), and
  **`quick_like_followup` -> TJ-14C** (found in the review: the answer is an append-only
  `opportunity_events` row and `ui.annotations.like_cohort.like_pick_rows` reads
  `claimed_setup_id` only off `trader_annotations.jsonl` rows - the key is genuinely read, the
  store is not joined, and the live log holds 46 quick likes).
* **Item 4's decision:** an hourly Questrade pull does NOT endanger the single-use token
  chain, because `_authorized_get` refreshes only on a missing or expired token or a 401
  where the stored token has not already moved - N pulls a day is not N rotations. It ships
  CAPPED anyway. **There was no per-day failure cap anywhere in the repository before this
  packet**; the tally is `pre_card_pull`'s, persisted beside the Mentor slot state on the
  durable tier, which also closes TJ-9's advisory that `_journal_retry_date` lived only in
  memory. IBKR still has no day leg and the card still says so.
* **Item 5's manual-step audit** is in DESK_INTERNALS "TJ-14B".

**Lead decisions RATIFIED 2026-09-20 (each is the trader's to overrule).** (a) *The day's
pre-card import pulls are THREE, reserved for the 09:00, 11:00 and last cards, plus ONE
uncapped morning catch-up that goes first* - because the reviewer measured first-come
spending every pull by 08:00, after which no afternoon fill was ever imported while the card
went on asking for labels it could not have. (b) *The card MERGES a fresh task* - because
freezing the whole section to protect half-set widgets also froze the words above them, so a
not-ready morning never became yesterday's questions; merging row by row keeps the trader's
typed values AND lets the heading go current. (c) *A same-day answer is labelled
`same_session` and a date-only fill never is* - because that middle provenance was
unreachable on every live row, and a broker file is authoritative for money and blind to
time, so a date-only fill has no moment a label could have been made before.

Live gate **#159**'s TJ-14B clauses are retained in full, and the packet's own risks are the
new gate **#163**. Long form: DESK_INTERNALS "TJ-14B".

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
have been. Rest of the gate (TJ-14B, read on real sessions - never claim it from a test):
an ordinary hourly card is one click (TJ-14A's forced prediction; TJ-14B adds at most three
OPTIONAL clicks and no forced one); on a day with a new fill the next card asks its setup
and stop, which also confirms the pre-card pull reached the broker; nothing is asked twice
after `Stop asking this`, read across a DESK RESTART between the two cards, because the
retirement is persisted and the restart is the half a unit test cannot read. The clause *a
trade from nowhere is asked its origin once* moved to TJ-12's gate and is **readable since
the TJ-12 merge `843a3f02` woke `trade_origin`** - it is now read as part of gate **#169**.

Live gate **#163** (TJ-14B's own risks, read on real sessions): the day's pull tally
SURVIVES a restart, so the second card of the day after a restart does not start the count
again; the three pre-card pulls land on the 09:00 card, the middle card and the last card of
the session and nowhere else, with TJ-9's three-day morning catch-up going FIRST on a
not-ready morning and not spending one of them; and the overnight question appears as a
CLICK once a day, with the old `One thing to test: ...` line not printed beside it.

##### TJ-9E - the Mentor tells an EXIT from an ENTRY (BUILT and MERGED 2026-09-21, `7230b30d`)

Trader, 2026-09-21: *"trade mentor should be able to differentiate between trade entrys and exits. trade
exits should ask for 'why did you exit, what emotions did you have, what technicals were you observing'
ideally we can just write it out and the AI fills this stuff in overnight. trade entrys are good the way
they are"*. Packet `.claude/packets/TJ-9E.md`, branch `claude/tj9e-exit-notes`, needs TJ-9, TJ-14B, TJ-16,
TJ-7 (all merged). An ENTRY keeps today's four fields untouched. An EXIT (any trade with an exit fill in
the reviewed session, whenever it opened) is ONE forced free-text box; the words are saved RAW at once,
append-only; a NIGHT slot `exit_note_fields` (stage 2, after `observation_tags`) drafts three fields -
`why` from a new closed vocabulary, `felt` from TJ-7's existing feelings vocabulary, `watching` as exact
quotes - each grounded in a span of the trader's own words and BLIND to the outcome (no P&L, R, price or
later bar in the request); the next morning the trader Confirms or Corrects, and a draft is never counted
as theirs before that click. Readers: Day Review's trades section and the report card's Process line
("exits explained K of N"). Fixed on the way: the partly-closed status string the 09:00 check never
matched. Not in it: the day pack / stories / contrasts reading exit fields (a recorded follow-up).

**MERGED into `lead/p033-integration2` as `7230b30d`, branch tip `1876bb08`, after FOUR review rounds by
reproduction: NO-GO, NO-GO, NO-GO, GO in round 4** (nine blockers closed; the merge commit's message says
round 4 "was still running" - the GO arrived minutes later, at `1876bb08`, and nothing moved after it).
The slot sits in stage 2 between `week_review_narration` and `ticker_briefs` (decision 0018's 2026-09-21
addendum); **the slot-order pins are TEN files now** (12.3), the tenth being
`tests/test_tj9e_night_slot.py`. Gates **#176-#180** owed.

**Beside it, merged the same day: the per-trade Mentor Save** (branch `claude/mentor-stores-answers-2026-09-21`
`0caf2bba`, merge `182f3e08`) - a trader-directed fix from their own report ("if i answer the questions
about a trade please then dont ask for it again just store that info"), built by a second session and
**merged WITHOUT any reviewer round of its own**, on the trader's instruction to combine the day's work.
The lead resolved nine conflict hunks across three files against TJ-9E and added
`tests/test_tj9e_per_trade_save.py`. **A REVIEW IS OWED on `0caf2bba` and on the conflict resolution**
(`_field_answer`, `_open_fields`, `_answered_trades`, `save_trade_check(only=)`, `_drop_trade_block`, and
the exit box folded in as one more field of its trade). Gate **#181** owed.

**Follow-ups recorded at this merge, none built and none authorized:**

1. **The day pack, the day story, the week story and the contrasts do not read exit fields yet.** Deliberate:
   a late note must not force a night to re-narrate a session it already wrote.
2. **The partly-closed status string is still spelled wrongly in four places TJ-9E did not touch** -
   `scripts/setup_environment_evidence.py:480` (which tests `{"PARTIAL", "PARTLY_CLOSED",
   "PARTIALLY_CLOSED"}`, none of which the assembler writes, so all seven live `CLOSED_PARTIAL` trades
   classify as OPEN and `STATUS_PARTLY_CLOSED` is dead code) and the CLOSED-only filters at
   `scripts/ui/panels/weekend_prep_panel.py:2426` and `:4076` and `scripts/ui/services/journal_feed.py:1076`.
   All pre-existing. **The first may be scoring-side, so it is ASK-FIRST**; each should adopt
   `journal_store`'s named status constants in the file that owns it.
3. **The raw exit words leave the TJ-9E lane through a pre-existing door.** `ai_summary._journal_source`
   lists `opportunity_events` with NO `event_type` filter and copies each payload into `lifecycle_events`
   of the `journal_review` scope, so an `EXIT_NOTE_RAW` row's words can ride into a narration package that
   also carries `net_pnl` and prices. The exit TAGGER is blind by construction; a NARRATION is not. The
   identical traffic already exists for `RECALLED_RAW`; TJ-9E neither widened nor closed that door. A
   follow-up decides whether to filter.
4. **The ENTRY-side AI draft loads the local model BY DAY.** `trade_mentor_ai.extract_draft`, run from the
   card whenever the trader submits raw text, has no `ai_jobs.window` gate. Left alone deliberately - the
   trader said entries are good the way they are - and flagged here so nobody reads it as an oversight.
5. **`journal_feed._store()` is a module-global cache**: the FIRST caller in a process decides which
   `JournalStore` the whole run uses. It cost TJ-9E a review round (32 errors in three other packets' test
   files, green on base). Worth an owner.

##### TJ-12F — Read the Focus-add and armed-alert lanes (PLANNED, not scheduled)

**PLANNED, not started; surfaced by the TJ-12 review round 1 and NOT scheduled into a wave.**
`trade_origin.planned_state` answers `unplanned` whenever no lane row precedes a trade's
first fill, so an UNREAD store is indistinguishable from "the trader said nothing". The desk
reads two of the four lanes (`decisions`, `claims`); the Focus-add and armed-alert lanes have
no public reader that hands back a row carrying a stamp `trade_origin` can read
(`FocusPickStore._episode_started_at` is private and membership-derived; the armed-alert rows
are built per caller). Measured on the live journal 2026-09-20: **30 of the trader's 33
trades since 2026-08-20 came back `unplanned`** for that reason alone. TJ-12F is ONE bounded
reader per store returning a stamped row `trade_origin` can read, plus adding those two names
to `day_report_card.DESK_ORIGIN_LANES_READ` - the one constant both lane builders build from.
Nothing else changes: until then the Process line counts `planned` and `no claim or like
before the fill`, never a bare `unplanned`, and NAMES the unread lanes, and the Mentor's
`trade_origin` prompt carries the same caveat; when the constant names all four the plain
wording returns on its own (pinned by
`test_with_every_lane_read_the_plain_wording_comes_back`). Gate: the clause inside **#169**.

##### TJ-14C — Give `quick_like_followup` a reader, then wake it (PLANNED)

**PLANNED, not started; UNBLOCKED 2026-09-20 by the TJ-14B merge `161e905c`.** The order is
the reader FIRST and the kind woken second: the one-field edit `dormant_until=""` is what
proves the join, and waking the kind before the reader exists is exactly the thing the
consumer walk was built to refuse. A follow-up the TJ-14B work surfaced. TJ-14B registers a
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

**BUILT and MERGED 2026-09-20**, branch `claude/tj16-prediction-contrast` (tip `d0d61f95`),
merged `c4a760e5` into `lead/p033-integration2`; **not on `main`** until the lead's next
fast-forward. Two reviews by reproduction (NO-GO, then GO at `d0d61f95`). Item 1:
`scripts/prediction_ledger.py` — `read_ledger` / `build_readout` / `horizon_of` /
`baseline_cell` / `your_reads`, the CURRENT grade row per read, the two horizons never
pooled with each other and a click never pooled with an extracted stance
(`market_read_grades.PoolingError`), the grading band CALLED through
`market_read_grades._verdict_for`, a directionless baseline `unmeasured` rather than wrong,
calibration by `How sure` that says "High did not beat Low" when it did not, and an empty
ledger reading `no clicked calls yet` with `rate: None`. Items 2 and 3: the deterministic
Stage 1 slot `scripts/ai_jobs/prediction_contrast.py` (`uses_model=False`,
`max_attempts=3`, `reserve_minutes=5.0`) registered DIRECTLY after `miss_contrast` and
still above the `market_story_rollups` + `measured_report` pair that closes the stage —
right against wrong per point-in-time context field through TJ-15's one
`evidence_contrast.contrast`, `lately` and `all` printed apart, per horizon, with `by_hour`
and `by_environment` tables carrying `n`, the Wilson bounds and `reportable` in every cell,
`flat` counted on the horizon and in NEITHER group, an `unmeasured` field contributing
nothing rather than a zero, and extracted stances counted in `excluded_by_source`. Item 4:
the Stage 2 slot `scripts/ai_jobs/observation_tags.py` (local MEDIUM model, `uses_model=True`,
`reserve_minutes=15.0`) after `ai_summary` and before `ticker_briefs`, with its own closed
vocabulary asset `scripts/ui/annotations/vocabularies/observation_tags_v1.json` (11 codes)
and its OWN loader; the payload is BUILT from the two texts and the picklist, so no verdict,
grade, price or bar can reach it, and `verify_reply` re-checks the reply's own BOUNDS as
well as its grounding because **a JSON schema's `maxItems` / `additionalProperties` is a
grammar hint to the provider, never a guard** (round 1: a 10,000-row reply published `ok`).
The journal read is WINDOWED at the stream (`EvidenceLedger.read(start=session,
end=session)`), and a note written after the session is LABELLED
(`written_after_the_session`, counted as `entries_written_after`), never dropped and never
told to the tagger. Item 5's readers ship (`your_reads`, `tendencies`, `read_latest`) with
**no page** — Day Review's `Your reads` line stays TJ-12's and the Week Review tables stay
TJ-5's. Tests: `tests/test_tj16_prediction_ledger.py`,
`tests/test_tj16_prediction_contrast_slot.py`, `tests/test_tj16_observation_tags.py`,
`tests/test_tj16_tagger_bounds.py`, `tests/tj16_support.py`; the slot-order pins were SEVEN
files at this merge and are EIGHT after TJ-5 (12.3). Long form: DESK_INTERNALS "TJ-16"; slot positions: decision 0018
addendum 2026-09-20. Gate #161 and the new gates #164–#167 are owed.

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
   TJ-7 exists, mood — BUILT 2026-09-20: `context_for(..., mood_entries=)` carries only the
   mood recorded AT OR BEFORE the stamp. A field the desk cannot measure is `unmeasured`,
   never guessed.
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

Live gate **#164** (owed): after a session in which the trader answers Mentor cards with
direction clicks, `prediction_contrast`'s pack carries a right-vs-wrong table by hour and
by environment with `n` in every cell, and its `Rest of day` / `Next 5 sessions` counts
match the read ledger by hand.

Live gate **#165** (owed): on the first weeknight the tagger runs, the stored
`observation_tags-<session>.json` carries a span for every code that reproduces its quote
against the journal row, and the request body recorded for that night contains no verdict,
grade id, price or bar. Until a local model has actually answered this contract, expect
`degraded_no_narrative` rows rather than silence — that is the designed failure, and a
rejection naming the CAP, a forbidden KEY or a REPEATED row is the bounds check working,
not the model failing.

Live gate **#166** (owed): the night AFTER the tagger's first success, the next
`prediction_contrast` pack carries `tag:<code>` features with both side counts and
`tags.reads_matched` is non-zero — a zero there means the tags and the reads are not
joining on `entry_id` and the contrast is blind to the words.

Live gate **#167** (owed): on the first night the tagger publishes, `entries_written_after`
is present in the tags-file header and in the next pack's `tags` block, and it is 0 on a
session the trader answered live. The first non-zero value is the one to look at — it says
how much of the tagged evidence was written after the trader already knew the answer, and
nothing in the pack is filtered by it.

#### 12.5 Order and dependencies

**AMENDED 2026-09-19 (twice):** TJ-9 (independent; first, because every personal statistic
waits on labelled trades) → TJ-14 (the card and the registry; TJ-9's section becomes its
first kind, and the prediction click starts accumulating at once) → TJ-10 → TJ-11 (may run
alongside TJ-10 and TJ-3) → TJ-4 (now needs TJ-10) → TJ-12 → TJ-15 (needs TJ-11's
real-miss rule) → TJ-16 (needs TJ-10's graded rows and TJ-14's split card; its ledger's
context snapshot ships WITH TJ-10 so no click is ever stored without one) → TJ-13 (independent; any time after TJ-4's slot exists) → TJ-5 → TJ-6 →
TJ-7 (fields only; its strip is TJ-14's `day_close`) → TJ-8. **Every step up to and
including TJ-7 is MERGED as of 2026-09-20; only TJ-8 is left.** The original chain below still
orders TJ-3 … TJ-8 among themselves. ~~No packet after TJ-9 merges until the desk has been
restarted once and gates #145, #152 and #153 have been read on a real session~~
**SUPERSEDED 2026-09-19 (trader: "I'd prefer to build it all now while I have usage
available … make sure the plan lets me build it all right away"):** nothing waits on a
restart. Every packet is built, reviewed and merged to `main` in the wave order below; the
live gates (#145, #152, #153 and #154–#167) are read afterwards, on the trader's first
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
| 3 | **TJ-14B MERGED 2026-09-20** (`TJ-14B.md`) · **TJ-10 MERGED 2026-09-20** (`TJ-10.md`) — wave 3 complete | `claude/tj14b-mentor-questions` (tip `cfe137d1`, reviewer GO after three rounds, merged `161e905c` into `lead/p033-integration2`) · `claude/tj10-read-grader` (tip `2967e4a6`, merged `57b44ca9` into `lead/p033-integration2`, integration fix `3e52d94f`) | TJ-14A ✓ (TJ-10 also rebases on TJ-3's page edits) |
| 4 | **TJ-4 MERGED 2026-09-20** (`TJ-4.md`) · **TJ-16 MERGED 2026-09-20** (`TJ-15-16.md`) · **TJ-13B MERGED 2026-09-20** (`TJ-5-6-7-13B.md`) — wave 4 complete | `claude/tj4-day-story` (tip `6c1e2506`, four review rounds, merged `d929e34f` into `lead/p033-integration2`) · `claude/tj16-prediction-contrast` (tip `d0d61f95`, reviewer GO after two rounds, merged `c4a760e5` into `lead/p033-integration2`) · `claude/tj13b-large-local` (tip `800ecb8c`, merged `e54c8203` into `lead/p033-integration2`) | TJ-10 ✓, TJ-11 ✓ · TJ-10 ✓, TJ-14A ✓, TJ-15 ✓ · TJ-13A ✓ |
| 5 | **TJ-12 MERGED 2026-09-20** (`TJ-12.md`) — wave 5 complete; follow-up **TJ-12F** owed, unscheduled | `claude/tj12-report-card` (tip `1a88d4a4`, three review rounds by reproduction, merged `843a3f02` into `lead/p033-integration2`) | TJ-9 ✓, TJ-10 ✓, TJ-11 ✓, TJ-4 ✓ |
| 6 | **TJ-5 MERGED 2026-09-20** → **TJ-6 MERGED 2026-09-20** → **TJ-7 MERGED 2026-09-20** (`TJ-5-6-7-13B.md`), one after the other — **wave 6 COMPLETE, and TJ-7 was the LAST building packet of Phase 0.33** | `claude/tj5-week-review` (tip `e93ffe49`, reviewer GO in round 1, merged `1b9d77e0` into `lead/p033-integration2`) · `claude/tj6-ideas` (tip `d78945f0`, two review rounds, merged `a7809d7c` into `lead/p033-integration2`, ninth pin `be435cd7`, fix `41d3f759`) · `claude/tj7-mood-fields` (tip `bcfdd918`, reviewer GO in round 1, merged `b63db7af` into `lead/p033-integration2`, guard amendment `ae7c06c7`, follow-up merge `0a0a0be4`) | TJ-4 ✓, TJ-12 ✓, TJ-13B ✓, TJ-15 ✓, TJ-16 ✓ · TJ-5 ✓ · TJ-14B ✓ |
| — | TJ-8 cleanup — TJ-7 ✓, so it now waits ONLY on the live gates above being read by the trader | — | every page gate read live |

**EVERY BUILDING PACKET OF PHASE 0.33 IS MERGED on `lead/p033-integration2` (tip
`0a0a0be4`), and none of it is on `main`.** What remains: **TJ-8 cleanup** (waits on the
live gates being read by the trader) and five unscheduled follow-ups — **TJ-14C**,
**TJ-12F**, **TJ-6M**, TJ-14B's `grader_gap` kind (free to wake now), and TJ-7's two:
**(i)** a verifier rule against the night pairing a mood with a RESULT (today only TJ-6's
closed `MEASURABLES` fences it; gate **#175** stands in the meantime) and **(ii)** a
re-offered Mentor subject whose prompt CHANGED keeping its old label (latent; TJ-7's AS
BUILT block).

Known shared files, so merge in wave order and rebase the later branch:
`ui/panels/day_review_panel.py` and `ui/services/day_review_service.py` (TJ-11 → TJ-3 →
TJ-10 → TJ-4 → TJ-12), `ui/widgets/trade_mentor_card.py` (TJ-9 → TJ-14A → TJ-14B),
`ai_jobs/runner.py` + `EXPECTED_SLOT_ORDER` (TJ-13A → TJ-15 → TJ-16 → TJ-4 → TJ-5 → TJ-6;
TJ-16 merged BEFORE TJ-4, so TJ-4's `day_review_narration` rebases into the stage 2 gap
between `ai_summary` and `observation_tags`).

**The build at a glance, by packet.**

| # | Packet | One line | Needs | Gate |
|---|---|---|---|---|
| — | TJ-1L | Day Review in two columns (MERGED; verified an ancestor of `main` 2026-09-19) | TJ-1 | #145 |
| 1 | TJ-9 | **MERGED 2026-09-19** - yesterday's trades labelled at 09:00, forced; label provenance; planned vs unplanned; journal freshness | — | #154 |
| 1Q | TJ-9Q | **MERGED 2026-09-20** - a Questrade fill says what it is and a sold put is a SALE; the import seams gated OFF; `journal_reclassify.py` the one way stored rows move, and the `--apply` is the trader's own act (owed) | TJ-9 | #162 |
| — | *restart* | NOT a hold any more (superseded above): the trader restarts when the build is merged and the gates are read then | — | #145 #152 #153 |
| 2 | TJ-14 | **TJ-14A MERGED 2026-09-19 (night)** - items 1 and 6: the card split (What I see / What I expect) with a forced prediction click, timeframe-horizon agreement enforced at the writer, `trade_mentor_context_v2` and the internals strip. **TJ-14B MERGED 2026-09-20** - items 2-5: the question registry whose every kind names the reader of its answer (four kinds DORMANT until that reader exists), the budget of three with the rest counted and carried, one card one import with three reserved pulls a day, same-session fills, and the manual-step audit | TJ-9 | #159 #163 |
| 2C | TJ-14C | **PLANNED, unblocked 2026-09-20** - give `quick_like_followup` a reader (join a quick like's answered `opportunity_events` row into the like cohort), THEN wake TJ-14B's dormant kind | TJ-14B ✓ | — |
| 6F | TJ-12F | **PLANNED, not scheduled into a wave** - one bounded reader per store for the Focus-add and armed-alert lanes, returning a stamped row `trade_origin` can read, plus those two names in `DESK_ORIGIN_LANES_READ`; until then the Process line and the `trade_origin` prompt NAME the unread lanes and never print a bare `unplanned` | TJ-12 ✓ | the clause in #169 |
| 3 | TJ-10 | **MERGED 2026-09-20** - the read grader: a click is the read and an extracted stance is always labelled, the two never pooled; `verdict_rank` so an `unmeasured` never supersedes a `pending`; a clicked grade never stored without its context; four congruence lines keyed on timeframe, `too_few` under the floor; the deterministic `read_grades_mature` slot. The 0.25 ATR flat band is the LEAD's number and the trader's to change | TJ-14 | #155 |
| 4 | TJ-11 | **MERGED 2026-09-19** - walk-away v2: earlier calls, against-first, ATR, real-miss rule, skill line vs base rate, instrument-aware rows, ADDITIVE session stamp | TJ-2 | #156 |
| 4F | TJ-11F | **MERGED 2026-09-19 (evening)** - an after-close decision belongs to the session it JUDGED; `decision_session_rule: "judged_session_v2"`; the D1 ruler starts from the judged session's close | TJ-11 | #156 |
| 4 | TJ-3 | **MERGED 2026-09-19 (evening)** - note markers on the SPY and name charts; a mark sits on a bar only when it happened DURING that bar, and one past the tape's end is counted, never clamped; the page has no D1 toggle | TJ-2 | #147 |
| 5 | TJ-4 | **MERGED 2026-09-20** - the PURE hash-stable day pack (twelve sections, every `source_id` minted unique, a machine row never), the overnight day story whose every verdict must EQUAL the measured read row and whose bounds come FROM THE PACK, the rolling D1 view, a night that SWEEPS the redos a daytime click queued (three narrated a night, the window re-asked before every call after the first), and a `--session` that reaches this one slot only | TJ-10 ✓ | #148 |
| 6 | TJ-12 | **MERGED 2026-09-20** - the six-line report card heading Day Review (incl. How fresh): pure, computing no new statistic, every line quoting its owner and guarded on its own, a ledger status read from `ai_jobs/ledger.py`'s own constants with the LAST deciding row winning, a night past the tail said as `unknown`, and an UNREAD origin lane never read as "nothing was said". It also WAKES the two Mentor kinds it reads | TJ-9 ✓, TJ-10 ✓, TJ-11 ✓, TJ-4 ✓ | #157 #168 #169 #170 |
| 7 | TJ-15 | **MERGED 2026-09-19 (night)** - what the misses had in common: a pure `evidence_contrast` with TWO floors, a deterministic `miss_contrast` slot inside stage 1 above the pair that closes it, D1 decisions only, the point-in-time join reaching one session back | TJ-11 | #160 |
| 8 | TJ-16 | **MERGED 2026-09-20** - the prediction ledger beside three naive baselines measured on the SAME stamps, calibration by `How sure` that says when High did not beat Low, the deterministic `prediction_contrast` slot directly after `miss_contrast`, and the Stage 2 `observation_tags` tagger whose payload structurally holds no outcome and whose verifier re-checks the reply's own bounds (a schema is a grammar hint, never a guard). Readers only - TJ-12 and TJ-5 own the pages | TJ-10 ✓, TJ-14 ✓ | #161 #164 #165 #166 #167 |
| 9 | TJ-13 | **TJ-13A MERGED 2026-09-19** (nights only 7 days; weeknight vs Saturday vs Sunday slates; briefs / summary / enrichment / examples repairs) · **TJ-13B MERGED 2026-09-20** (the `--probe-model` command and the `local_large` provider seam - the measurement itself is the trader's and is owed). Still open in TJ-13B: Sunday's suggested tags and week-ahead note; the week STORY is TJ-5's | TJ-4's slot ✓ | #158 |
| 10 | TJ-5 | **MERGED 2026-09-20** - Week Review, Weekend Prep's first step: five day cards on ONE payload that computes nothing, the week pooled in ONE place (`day_report_card.week_from_cards`, naming no best family from five day-winners), an unpacked day named and an UNREADABLE day named rather than counted as quiet, and the Saturday-only `week_review_narration` slot that narrates only the week's own packs with session-qualified ids, re-checks every bound the input gave it, and writes a scaffold with zero model calls under three narrated days | TJ-4 ✓, TJ-12 ✓ | #149 #170 #171 |
| 11 | TJ-6 | **MERGED 2026-09-20** - the desk's AI has a voice: a stage-3 `improvement_ideas` slot behind `setup_research` with a CLOSED `EVIDENCE_KEYS` and a CLOSED registry of the two measurables that exist, a per-item `drop_reason` the slot enforces itself over a shape-only schema, whole rejection with the store AND the trader's state byte-identical, an append-only store that folds repeats and never brings a dismissed idea back; nothing to cite means NO model call and a night that asked is DONE (`ok` + an asked marker); a KEEP is the trader's click and freezes its baseline from the LAST WRITTEN pack, and a later reading is higher or lower only when the two Wilson intervals do not overlap | TJ-5 ✓ | #150 #172 #173 #174 |
| 11M | TJ-6M | **PLANNED, not scheduled into a wave** - the THIRD measurable: a Mentor question's answer mix over a window. It has no pooled reader today (`day_report_card.process_line`'s `origin_answers` is one kind, one session, and `_pool_cards` drops the key), so this is ONE entry in TJ-6's CLOSED `MEASURABLES` (suggested name `mentor_answer_mix_rate`) plus a pooled reader in `day_report_card` - the same explicit grant TJ-5 got for `week_from_cards` - plus one line in `tests/test_tj6_measurables_and_the_frozen_baseline.py`'s closed-registry expectation. The registry being CLOSED is what makes the addition safe later | TJ-6 ✓, TJ-14B ✓ | the clause in #150 |
| 12 | TJ-7 | **MERGED 2026-09-20 - the LAST building packet** - a mood is a field the desk REPORTS: ONE additive journal key `mood` (`{}` when nothing was clicked, ABSENT on old rows, never a default), refused LOUDLY at the writer and again at `is_publishable`, a closed versioned vocabulary whose own loader owns the cap, ONE optional strip on both surfaces with nothing pre-selected and a second click clearing a face, the day pack's citable `mood` section minted through ONE id seam, the ONE key `mood` in each of the week and ideas evidence lists (no `MEASURABLES` entry), a point-in-time context field, ONE Day Review line - and no detector, score, alert, watchlist, Focus, review queue or tagger ever sees one. No nightly slot | TJ-14 | #151 #175 |
| 12F | TJ-7 follow-ups | **PLANNED, not scheduled into a wave** - (i) a verifier rule against the night pairing a mood with a RESULT (today only TJ-6's closed `MEASURABLES` fences it; gate #175 stands in the meantime); (ii) a re-offered Mentor subject whose prompt or options CHANGED keeps its OLD label and combo items - latent, since no shipped kind varies either for a fixed `subject_id` | TJ-7 ✓ | #175 |
| 13 | TJ-8 | Cleanup of the retired panels | all page gates | — |

**TJ-14B's DORMANT question kinds, and what wakes each (2026-09-20).** A kind ships
only with a named consumer; a kind whose reader belongs to a later packet is registered in
full, carries `dormant_until` naming that packet, is never put on a live card and is given
no shim reader. Waking one is a one-field edit - `dormant_until=""` in
`scripts/mentor_questions.py` - plus its lane in `MainWindow._mentor_question_state`, which
already names all five. **None of these is a new packet's work beyond the reader itself.**

**TWO of the original four were WOKEN by TJ-12 on 2026-09-20** (merge `843a3f02`):
`trade_origin` (`consumer="day_report_card.process_line"`) and `open_position_check`
(`consumer="day_report_card.long_hold_lines"`) now carry `dormant_until=""`, because this
card is the reader they were waiting for, and `MainWindow._mentor_origin_lanes` really fills
the `decisions` and `claims` lanes - `focus_adds` and `armed` stay named and empty until
TJ-12F, which is why both the prompt and the Process line SAY so. Two remain:

| kind | `dormant_until` | woken by | the reader it is waiting for |
|---|---|---|---|
| `grader_gap` | `TJ-10` | **ready now** | TJ-10 MERGED on this branch and `grader_gap` is on every grade row, so this one can be woken as a SMALL follow-up - deliberately not done inside TJ-14B |
| `quick_like_followup` | `TJ-14C` | TJ-14C | `like_cohort.like_pick_rows` reads `claimed_setup_id` off `trader_annotations.jsonl` only - the key is read, the `opportunity_events` store is not joined |

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
- From TJ-7 on (MERGED 2026-09-20): the mood strip is **OPTIONAL and owes nothing**. A
  session with nothing clicked is a normal session and its row is the row it always was;
  clicking the chosen face again takes the mood back. What the trader DOES owe is reading
  gate #151 once on a real session — and, after a week that carries a mood, gate #175 on
  the Saturday.

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
   the trader's own click - #154, #155, #159, #161, #164 and the clicked halves of the rest.
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
