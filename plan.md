# TradingBotV3 remaining roadmap

Last reconciled: **2026-08-20**

Authoritative for: **work that is not finished, validation gates, promotion rules,
and execution order**

Implemented history: [`CHANGELOG.md`](CHANGELOG.md)

Supporting-document index: [`docs/README.md`](docs/README.md)

This file intentionally does not repeat the implementation history. A feature with
code is recorded in `CHANGELOG.md`; any validation, evidence, promotion, or cleanup
still owed remains here. Detailed specifications under `docs/` are subordinate to
this roadmap. Section numbers 5–7 and 12 are retained deliberately so established
runbook and decision-record references stay valid.

## 1. Mission and product boundary

TradingBotV3 is a decision-support system for one trader. It prepares the market,
discovers swing and intraday candidates, monitors them, alerts, publishes an Away
report, records decisions and outcomes, and supports controlled research.

It never places or routes orders. Broker execution, consumer distribution, and any
automatic promotion of research output are outside the product boundary.

The operating topology is now simple:

- the Ryzen 7 8845HS main desk is the only always-on application and scan host;
- `launch_gui.py` starts the PySide6 Trading Desk in Main mode;
- the former mini-PC scanner and Desk Link satellite roles are retired and must stay
  unused until their code is removed in a deliberate cleanup packet;
- ntfy and the verified `autopilot_today.txt` digest are the remote surfaces;
- there is no cloud sync (decision 0015): `C:\TradingBotData` is a plain local folder
  and the DAS `\\MINI-PC\Trading Bot Data` is the durable storage tier;
- the Tk GUI was removed on 2026-09-03 (assessment packet F2); `scripts/ui` is the only UI.

**What the program is for, in the trader's words (decision 0016, 2026-09-02).** The
trader answered twelve questions one at a time; the record is
[`docs/decisions/0016-trader-vision-and-priorities.md`](docs/decisions/0016-trader-vision-and-priorities.md)
and it is the tie-breaker for every prioritisation call. The short form:

1. Get **which names are shown** right before **when to enter**.
2. A name was right to show when it **moved** (a held D1 level then the move, for a
   swing; a held intraday level then the maximum favourable excursion, for a day
   trade); the trader's likes only say where to look. **Win rate** is the headline
   swing statistic, because losses run about 1.5x the best wins.
3. One click from any screen teaches the bot what the trader likes; words are
   optional; the click is always processed.
4. "What is working lately" (a rolling ~20 sessions, no regime label) lives on the
   **Trading Desk** and in Weekend Prep with a display-only priority switch - never a
   mute, never a filter. The Research tab is not a trader surface.
5. The trader sits on the Capture tab; the Alerts, D1 Focus, Armed tabs and the
   Universe page are unused; the Strength Board must match the trader's own TC2000
   scan (decision 0016 item 9) before it is compared to it.
6. Tagging is the slow part of journaling: the P6a tagger runs nightly, the trader
   corrects. Weekend Prep gets one Refresh, a verdict card and readable tables; the
   Market Journal is one box; Away Recap shows more names with charts for a 10-30
   minute evening review.

## 2. Status vocabulary

These labels must not be collapsed:

| Status | Meaning | Production authority |
|---|---|---|
| `PLANNED` | Designed but no implementation exists. | None |
| `IMPLEMENTED` | Code exists. | None by itself |
| `GREEN` | Deterministic tests pass. | Only existing champion behavior |
| `SHADOW` | Runs on live inputs but cannot affect production decisions. | None |
| `LIVE_VALIDATED` | Passed the documented real-session or operational checks. | None by itself |
| `ADVISORY` | Visible as labeled research or decision support. | No loud-alert, gate, or ranking authority |
| `PROMOTED` | Explicitly approved as the production champion with rollback. | Yes |
| `RETIRED` | Intentionally disabled or replaced. | None |

Current code and test status belongs in `CHANGELOG.md`. Current branch and exact
test counts belong in `CURRENT_CHECKPOINT.md`. Only unfinished work belongs here.

## 3. Current-state summary

As of 2026-09-02:

- `main` is the running branch: the desk launches from source on `main` by trader
  decision (2026-08-26), and every Phase 0.13 packet (P0-P9, review rounds R1-R3) is
  merged. `CURRENT_CHECKPOINT.md`'s "Active state at a glance" block carries the
  measured baseline (6,091 tests, exit 0, on 2026-09-02) and the open live gates;
- the frozen exe is a verification artifact only, rebuilt when a packaging trigger
  is hit (last: P7's registry asset, 74/74 frozen self-test);
- the research warehouse Phases 0-8, Chart Review, durability and Local-AI Phases
  1-2 are implemented; their live gates are listed in the checkpoint;
- legacy SPY pause detection and D1 wick alerts remain the production champions;
- `market_state` and `greatness_monitor` remain shadow-only;
- the research warehouse and AI outputs remain additive/read-only and advisory;
- the trader's stated priorities are decision 0016 (Section 1 above).

See `CHANGELOG.md` for the full implemented inventory and revision history.

## 4. Authority and change control

When documents disagree, use this order:

1. this roadmap for remaining-work order, invariants, and promotion policy;
2. accepted decision records under `docs/decisions/`;
3. the locked warehouse specification where it is explicitly delegated authority;
4. active implementation specifications listed in `docs/README.md`;
5. historical reviews, handoffs, proposals, and superseded GUI plans.

Do not infer current status from a historical plan. Reconcile it through
`CHANGELOG.md` and this file.

`WISHLIST.md` is deliberately outside the authority chain. It records candidate
integrations and deferred ideas, but it never authorizes implementation or changes
the order below. Only an explicit trader decision may promote a wishlist item into
this roadmap.

## 5. Non-negotiable system invariants

### Data and time

- State transitions use completed bars only. A forming bar is a labeled preview.
- Missing or stale data is uncertainty, never confirmation.
- Point-in-time research may use only information available at the simulated
  decision time; timestamps carry explicit time zones.
- Never replace `calc_anchored_vwap_bands`' running-deviation sigma formula.

### Identity and provenance

- Stable identity must distinguish symbol, side, horizon, setup/thesis, anchor,
  attempt, and configuration where those dimensions matter.
- Every suggestion, alert, review, research row, and outcome must retain enough
  provenance to reconstruct what the system knew.
- User-entered watchlist names are never automatically removed.

### Runtime and publication

- One component owns each timer, thread, job, mutable store, or shared export.
- A failed publish never destroys the last verified report.
- Ambiguous ownership fails closed.
- The single-main topology does not authorize duplicate writers.

### Research and promotion

- Legacy SPY pause detection and D1 wick alerts stay champions until the Section 7
  gates pass.
- No detector, score, ranking, routing, or alert-behavior change lands without a
  golden characterization fixture first.
- Shadow, research, Technical Integrity, warehouse, review-learning, and AI outputs
  have zero production influence until separately promoted.
- `review_policy.json` ranks and annotates only; it has no suppression field.
- AI is one-way and evidence-grounded. It may summarize and propose tests, never
  mutate production state.

### Product behavior

- The app is decision-support only and never executes orders.
- Honest zero-opportunity and unknown-data states are preferable to filled panels.
- Desk, Away, alerts, journal, and AI must ultimately consume the same canonical
  opportunity facts.

## 6. Live validation program

Automated green tests do not satisfy live gates. The active checklist is
[`docs/FIRST_SESSION_CHECKLIST.md`](docs/FIRST_SESSION_CHECKLIST.md).

For the first live session on a new build, record:

- branch/commit, machine, Python, TWS/Gateway mode, home folder, research-store
  state, Auto profile, and market-session date;
- full pytest exit code, smoke result, and frozen self-test when a rebuild trigger
  applies;
- real run manifests, heartbeat, provider telemetry, shadow logs, job ledger,
  verified Away metadata, and capture audits;
- GUI responsiveness, chart freshness, alert delivery, clean shutdown, and restart
  behavior;
- every failure or unknown as evidence, without rewriting the acceptance result.

Physical two-machine and satellite checks from older runbooks are retired with the
topology. Writer fencing still requires deterministic tests, but no new live
two-machine gate blocks the single-main product.

## 7. Shadow evidence and promotion ladder

Promotion is a separate decision from implementation and live validation. Every
challenger requires:

1. a versioned configuration and stable identity;
2. golden/replay fixtures and a declared evidence window frozen before inspection;
3. complete coverage and data-quality accounting - and, since packet Q2 (2026-09-04),
   the knowledge basis of every input: a `feature_snapshot_daily` row whose anchor is
   `reconstructed` or `legacy` and an outcome row on the `plain_no_target` path are
   research evidence and NEVER count toward a promotion gate (BD-99/BD-100);
4. comparison with the active champion on the same inputs and outcome definition;
5. representative live sessions across relevant regimes, sides, and day parts;
6. explicit success, non-inferiority, and rollback criteria;
7. a bounded canary and one-switch rollback that does not require a code revert;
8. explicit trader approval recorded in the revision history.

### SPY pullback challenger

Still required before any promotion:

- prove completed-bar coverage rather than scan-cycle presence;
- label and reconcile episodes across session/config rollovers;
- compare episode timing, false pauses, missed pauses, and downstream candidate
  usefulness with the legacy pause detector;
- validate timezone, staleness, and restart behavior on live artifacts;
- integrate sector/candidate RS only as advisory evidence until its own gate passes.

### Greatness challenger

Still required before any promotion:

- a dedicated monitoring lane independent of legacy D1 scan cadence;
- same-day plan revision and side-change handling;
- complete multi-level confirmation, failure, re-arm, freshness, volume, RS/sector,
  reward/risk, and anti-chase gates;
- transition-chain audits and outcome comparison with legacy D1 wick alerts;
- evidence that alert precision improves without unacceptable delay or missed moves.

## 8. Detailed specifications retained under `docs/`

The roadmap owns priority and status. These files retain implementation detail that
would make this file unwieldy:

- research warehouse: `docs/ULTIMATE_SETUP_DATABASE_PLAN.md`,
  `docs/RESEARCH_WAREHOUSE_BUILD_DECISIONS.md`, and
  `docs/RESEARCH_WAREHOUSE_ERD.md`;
- local AI: `docs/LOCAL_AI_AUTOMATION_PLAN.md`;
- Chart Review capture: `docs/CHART_REVIEW_WORKSPACE_PLAN.md`;
- setup doctrine: `docs/SETUPS_MAJOR.md` and `docs/SETUPS_TEST.md`;
- operations and packaging: the active runbooks listed in `docs/README.md`.

Their old phase lists do not reorder Section 12.

## 12. Remaining work, in execution order

The phases below are dependency order, not a menu. `CURRENT_CHECKPOINT.md` names the
one active item. Finish that item before moving down the list unless the trader
explicitly redirects the work. Elapsed evidence collection may run in parallel only
where the phase says so; it never authorizes an early promotion.

| Order | Build phase | Plain-English outcome |
|---:|---|---|
| **0** | Validate and merge | **P0.7 merge DONE 2026-08-26**; P0.2–P0.6 live proofs are §6 and the checkpoint gates table |
| **0.5** | Trader refinement packets | Build the trader's 2026-08-14/15 desk requests in ranked order (R1–R8) |
| **0.8** | GUI fluidity Wave P1 | Repair the measured Standard-mode stalls and three verified GUI defects |
| **0.13** | Grade what the trader already said (P0-P10) | Every verdict gets a forward record; a like starts a five-session watch. **MERGED; live gates #29-#43 owed** |
| **0.16** | Capture and board rules (packets T1 + T2) | A veto with no box, a quick like that stays, a claimed like that is one double-click and advances, a board click that queues nothing, the TC2000 board on M5 Focus. **BUILT; live gate #58 owed** |
| **0.14** | Names first (V1, V2, V3) | Decision 0016: the names shown come before the entry taken. **V1–V3/R4 merged; V4's Working-lately switch and AWAY Recap remain NOT BUILT** |
| **0.12** | Focus de-clutter + HTF LRSI research | Make the Focus feed, the Armed board and the Focus list readable again; ask in shadow whether a higher-timeframe LRSI entry pays |
| **1 — NEXT** | Reliable development baseline | Make tests offline/deterministic and close measured cleanup questions |
| **2** | Authoritative foundations | One correct provider, time, candidate, SPY/RS, and Greatness data path |
| **3** | Evidence and capture | Mature warehouse/AI/shadow evidence and capture trader commentary honestly |
| **4** | Canonical Opportunity | Build and validate one inspectable opportunity/ranking challenger |
| **5** | Delivery and lifecycle | Make alerts and every surface agree; reconstruct the whole decision lifecycle |
| **6** | Research payoff | Use the validated corpus to promote setups narrowly and finish the Qt product |
| **7 — LATER** | Consolidate and ship | CI, recovery, packaging, installer, and optional read-only broker adapters |

### Phase 0 — validate and merge the testing-week branch (P0.7 DONE 2026-08-26)

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md) on 2026-09-05 (repo cleanup), unabridged. Status at the move: **P0.7, the merge, DONE 2026-08-26** (`testing-week-2026-08-17` -> `main`); P0.1 is the standing before-every-commit rule in `CLAUDE.md`; P0.2-P0.6 are the live-validation program of Section 6. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the move closed nothing.

### Phase 0.5 — trader refinement packets (promoted 2026-08-15)

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md) on 2026-09-05 (repo cleanup), unabridged. Status at the move: **R1-R8 all BUILT** (R3 and R6 CLOSED; R3 §4.3.5 trader-deferred; the weekly trader-judgement synthesis OWED, not built, gated on two weeks of graded rows). Specs: `docs/AUTO_MODES_AND_QUIET_HOURS_PLAN.md` (R1), `docs/M5_FOCUS_GATING_AND_STRENGTH_BOARD_PLAN.md` (R2), `docs/SWING_QUALITY_AND_FEEDBACK_PLAN.md` (R3), `docs/DESK_CHART_UNIFICATION_PLAN.md` (R4), `docs/M5_SIGNAL_ENGINES_PLAN.md` (R5), `docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md` (R7), `docs/WEEKEND_PREP_PLAN.md` (R8). The 2026-08-27 trader rules (auto-Focus with-trend rows, the VWAP-side/show-time filter, the D1 SMA leg, the M5 alert bar, the group RS/RW tape) are items under R4 in the archived text. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the move closed nothing.

### Phase 0.6 — R9: trade-review response packet (authorized 2026-08-22)

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md) on 2026-09-05 (repo cleanup), unabridged. Status at the move: **R9.1-R9.5 all BUILT and GREEN 2026-08-22**; the deterministic half of the exit gate is met and the on-the-desk half (R9.1 universe rebuild row, R9.2 like-and-why, R9.4 `thetalongs.txt`) is owed. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the move closed nothing.

### Phase 0.7 — R10: Evidence Plane program (authorized 2026-08-22)

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md) on 2026-09-05 (repo cleanup), unabridged. Status at the move: **every R10 packet (R10.A-R10.I, R10.0b option C-prime, the AWAY day recap) BUILT and GREEN by 2026-08-27**; each owes its live mechanics canary. R10.0b's daily-bar source is a PIN (`daily_bars_source: "yahoo"`), not a defect. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the move closed nothing.

### Phase 0.8 — GUI fluidity Wave P1 (authorized 2026-08-26)

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.9 — GUI follow-ons from the 2026-08-26 live session (authorized 2026-08-26)

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.10 — AVWAP band challenger (authorized 2026-08-26)

Source: `docs/AVWAP_BAND_VARIANT_STUDY.md` (§2b the replicated formula, §4 the
harnesses, §4 T4 the pre-declared decision criteria). The trader replicated
OneOption's band on 2026-08-26 — `AVWAP(HLC/3) ± k · stdev(close, 20,
population)` — and authorized testing it in the setup tracker ("throw it into
the setup tracker and begin testing it out"). Build prompt:
`docs/archive/prompts/AVWAP_BAND_CHALLENGER_OPUS_PROMPT.md`.

**Scope bound.** Shadow only. `calc_anchored_vwap_bands` stays frozen (decision
0008); no detector, score, rank, tier, alert, zone arm, Focus, queue or
`review_policy.json` behaviour changes; the champion's tracker outputs are
byte-identical with the shadow block present (parity fixture frozen first).
`legacy.py`/`runner.py` edits are limited to the additive ones the prompt
pre-authorizes; anything else asks first.

1. **B-0 Pure module + golden fixture.** *BUILT 2026-08-26 (`002f2a3`).*
   `scripts/indicators/avwap_band_variants.py` (`avwap_bands_oneoption_bb20_v1`),
   OKTA fixture frozen through `_normalize_daily_bar_frame`, discriminator tests
   against the champion (sigma 0.0 on a one-bar anchor vs the 10.28 read) and
   the killed sample-OHLC form (138.09 vs the 144.60 read), `None` below 20
   closes, AST fence against importing `master_avwap_lib`. First importer of
   `indicators/`: spec-drift 17 passed with no spec edit, `--selftest` 71/71.
2. **B-1 Fit/print script.** *BUILT 2026-08-26 (`13505d1`).*
   `scripts/avwap_band_variant_fit.py` — champion vs challenger per bar since an
   anchor, offline, writes nothing without `--csv`. Reproduces the study §2b S2
   column on OKTA.
3. **B-2 Tracker shadow.** *BUILT 2026-08-26 (`5613eec` fixture, `603333b`
   code).* Parity fixture frozen FIRST, before either fenced file was touched.
   The anchor-variant blocks, the appended `VARIANT_*` stop candidate,
   `master_avwap_band_variant_stats.csv` and the panel's "Band Variant" tab all
   landed. **Appending was not sufficient** — the champion's own averages moved
   and they reach `row["score"]` — so a trader-authorized fence
   (`_is_band_variant_scenario`, seven readers) keeps the shadow out of every
   champion aggregate; the parity fixture proves the champion's record is
   byte-identical. Tracker JSON growth measured: **9,982 bytes per new setup**,
   ≈144 MB (~15%) at the live 14,386-setup / 950.2 MB scale, accruing forward
   only. The study's "a few hundred bytes" estimate was ~30× low; capping the
   shadow to the non-experimental exit templates would cut it by a third and is
   a one-line change if the trader wants it.
4. **B-3 D1 chart overlay.** *BUILT 2026-08-26 (`3abf61d`).* Paint-lines group
   "AVWAP σ variant", built on the worker, anchored on the date the snapshot
   already resolved. Default OFF required a new
   `chart_levels.GROUPS_HIDDEN_BY_DEFAULT` + a `shown_groups` list in
   `PaintLinesPrefs`, because every group previously defaulted ON by design.
5. **B-4 Backfill** (next packet, after B-0..B-3 review): the level-quality
   study T1 and the playbook re-run T2, then the warehouse columns. NOT started.

**Finding that changes T1/T3's design.** A wider band is NOT automatically a
further stop: it is only stopped out less often when entry sits INSIDE it. On
the parity fixture's short — entered above both upper bands — the wider sigma
pushes the upper band toward entry and the challenger's stop lands 0.159 away
where the champion's is 0.971, six times tighter. Any stop-out or respect-rate
comparison must condition on the entry's position relative to the band.

Gates: T4's three criteria decide, and a pass is the input to a plan.md §7
promotion decision whose shape is an ADDITIONAL level family, never a swap of σ
inside the champion. ≥ 20 sessions of forward accrual owed before T3 counts.

## Phase 0.19 — AVWAP band challenger: make the comparison measure (QUEUED by the trader 2026-09-05)

Trader, 2026-09-05 ~01:45 PT, after the lead reported the comparison is built but empty:
*"I want us to compare both to see what is better"* ... *"Add this to the queue."*

**What is true (lead, 2026-09-05):** the Phase 0.10 T3 surface exists - `master_avwap_band_variant_stats.csv`
(40 rows, 11,292 setups) and the Setup Tracker's "Band variant" view - but `n_variant` is **0 on every
row** since it was built on 2026-08-26: every tracker record's `current_anchor_variant` reads
`"no band-variant block on the scan entry"` while `master_avwap_ai_state.json` carries a full block
(`avwap_bands_oneoption_bb20_v1`, stdev present) for all 423 symbols of the last scan - e.g. AAON,
same anchor date 2026-08-10, stdev 4.72 in the AI state and "no block" in the setup record. So no
`band_variant` stop scenario has ever been built and the challenger has measured nothing. The lake
has no challenger columns (T3 step 4 / B-4, NOT started).

**Packet (to write): B4.** (1) Fix the hand-off so `build_tracker_setup_record` receives the block the
scan computed - root cause named by recon first; fail-first test on a record built from a live-shaped
`ai_state` entry; the champion's records, scores and events byte-identical (the existing parity
fixture). (2) The warehouse: additive `avwap_variant_upper_1..3` / `lower_1..3` +
`avwap_variant_formula_version` on `feature_snapshot_daily`, and a twin swing recipe
(`swing_house_variant_v1`, same occurrences, same management, the challenger's bands) so
`band-coverage` and the fact pack show the two side by side; reconstructed labelling as Q2.
(3) A comparison line on the Setup Tracker's Band variant view that says `n_variant` and
`n_variant_unmeasured` in words. **`master_avwap_lib/runner.py` and `legacy.py` house scanner code
(file-scoped ask-first): the trader's "add this to the queue" is the recorded yes for the hand-off fix
and nothing wider.** T4's criteria (docs/AVWAP_BAND_VARIANT_STUDY.md) still decide; >= 20 sessions
of forward accrual start only when the first measured row lands.

## Phase 0.18 — Process-review packets Q1-Q5 (2026-09-04) — BUILT, live gates #60-#64 owed

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-05 (repo cleanup), unabridged. Status at the move: **Q1-Q5 BUILT and MERGED at `b0db9bbe`**, live gates #60-#64 owed. Deliberately NOT built (ask-first, `bounce_bot_lib/legacy.py`): per-alert bar-close -> shown latency instrumentation, the H1 SPY recompute, a `stop_hit_at` column, the sweep autorun default. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the move closed nothing.

## Phase 0.17 — Earnings-anchor bridge (2026-09-04) — BUILT, live gate #59 owed

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-05 (repo cleanup), unabridged. Status at the move: **BUILT**; gate #59 MET 2026-09-05 (3,678 anchors bridged). NOT built, separate decision: the simulator returning `None` instead of a no-target run when bands are missing (changes outcome semantics). Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the move closed nothing.

## Phase 0.16 — Capture and board rules (packets T1 + T2, 2026-09-04) — BUILT, live gate #58 owed

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-05 (repo cleanup), unabridged. Status at the move: **T1 + T2 BUILT**, live gate #58 owed (its clauses are in the gate row). Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the move closed nothing.

## Phase 0.15 — Desk assessment packets (2026-09-03 evening, trader-authorized)

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-05 (repo cleanup), unabridged. Status at the move: S1, S3, S4, F1, F2, F3 step 1 BUILT; S2 INSTRUMENTED (trim still measure-first, `legacy.py` ask-first); the lake REPAIRED and gate #56 MET; E2 resolved as a pin. **Still owed**: gate #55 (tee) and #57 (tracker parity, then 0017 step 2 moves readers one at a time); S1.3 (ONE Strength surface) needs a fresh packet; the `technical_integrity_events.jsonl` segment scheme is owed as its own packet; E1 is the trader's validation-week decision. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the move closed nothing.

## Phase 0.14 — Names first (decision 0016)

**Status at 2026-09-02, after round R4 Part A.** V1, V2 and V3 are all merged to
`main` (V3 fast-forwarded from `claude/v3-keep-it-honest` the evening of
2026-09-02); R4 Part A is on `claude/r4-fixes`. What is NOT built, in one place so
nobody has to reconstruct it from four entries:

| Packet | Item | State |
|---|---|---|
| V1 | 1 Strength Board = TC2000 | **BUILT** (gates #44, #45). R4 A7/A8 made the RVOL session-relative, dropped the forming daily bar and widened the daily window to `2y` |
| V1 | 2 `held_run_score` | **BUILT AND WIRED** (R4 A9/A10): the D1 dimension is fed, the tracker's second formula is deleted, and the M5 alert row carries the suffix. FOUR of the tracker's nine tabs fill (Bounce Types, Combos, Time of Day, Environment) and five read BLANK - the four `master_avwap_*` Swing tabs because the outcome log cannot be asked those dimensions, `rrs_alignment` because it is reachable and not derived yet (`UNDERIVED_DIMENSIONS` splits the two) |
| V1 | 3 phone digest ranks across buckets | **BUILT** (R4 A11, horizon corrected in fix round 1) - Wilson lower bound on the family's realized win rate at ONE declared horizon (`SWING_DIGEST_HORIZON_SESSIONS` = 5, stale-horizon rows dropped the way the scan-factor leaderboard drops them), expected R as tiebreak, near cap after the ranking |
| V1 | 4 Working-lately + priority switch | **NOT BUILT** - this is V4 |
| V2 | 1 nightly auto-tagging | **BUILT** (gate #46) |
| V2 | 2 Weekend Prep | (a)(b)(c)(e) **BUILT** (gate #49). R4 A13/A14/A15/A18 fixed the take rate, moved the 775 ms read off the Qt thread, gave Discovery a real `reload` and its six buttons the exit, stopped Confirm-all confirming a blank, added the per-row edit and put every table on a ten-row floor. The takes table and the collapsed notes are still owed |
| V2 | 3 AWAY Recap | **NOT BUILT** - this is V4 |
| V2 | 4 Market Journal one box | **BUILT** (gate #47). The Desk tab landed with V2; the LEFT-NAV PAGE landed with R4 A16, and R4 A17 moved the session roll to the open |
| V2 | 5 hide the dead tabs | **BUILT** (gate #48) |
| V3 | 1 win rate leads | **PARTIAL** (R4 B3 wired five surfaces). WIRED: the AWAY digest ranking (A11), `setup_docs.family_record_sentence` and its two renderers (B2), the Master AVWAP setups table's **Family Win %** column, the Setup Tracker's **Last 30 Days** tab, and all four Weekend Prep cohort tables (veto, like, pass, rejection), each sorting by the Wilson lower bound. ONE Wilson: `swing_headline.WILSON_Z`. STILL OWED: **the Setup Tracker's Setup Types tab** - and the reason is measured, not scheduling. `master_avwap_setup_type_stats.csv` carries no win column (only `target_hit_rate` / `stop_rate`, different questions), and `master_avwap_tier_outcomes.csv` cannot be joined at that table's grain: its 184 rows collapse to 71 (side, bucket, family, zone) groups, so one joined rate would repeat across up to six rows and read as each row's own. Giving it an honest win rate needs the tracker export to carry one |
| V3 | 2 day-trade headline | **BUILT** - surfaces real since R4 A10, and since R4 B4 every number on the Daytrade Tracker names its own basis: the champion tier is a COLUMN (PROVEN / MUTED / active from the learning state, blank for a segment it never saw - live 4 / 2 / 185 / 104 of 295 rows), the aggregator's verdict is headed **Verdict (edge score)**, and the My Decisions tabs carry Held 30m / Held x Ran through the same helper on `held_run_score.ALL_DIRECTIONS`, a pooled cell accumulated from the EPISODES and never an average of the two sided cells |
| V3 | 3 one `LATELY_SESSIONS` | **BUILT AND COMPLETE** (R4 B6). `review_learning.DEFAULT_WINDOW_SESSIONS` IS `LATELY_SESSIONS` and its cutoff walks the exchange calendar; Weekend Prep's week is `evidence_stats.WEEK_SESSIONS` (5). The state key, the report header, the CLI flag, the System Health audit and the Daytrade Tracker status line all say **sessions**, and a literal scan test fails if a `window_days` comes back |
| V3 | 4 one annotation writer | **BUILT** - all five surfaces have a writer since R4 A5, and since R4 B5 every VERB stamps the screen: `commit_pass` bypassed `_record` entirely, so a day-trade pass was the one row that could not say where it came from. The guard is now behavioural (one test per real click handler, reading the written row) rather than a scan of `_record`'s source text, which a verb that never calls it satisfied |
| V3 | 5 research on a trader surface | **BUILT**, and CORRECTED by R4 B1 - the reader picked the OLDEST pack of a superseded day (`sorted(...)[-1]` is an ASCII sort), so the verdict card read a 47-cell pack in the older shape and printed "no cell has cleared the evidence floor" while the current pack had 33 that had |
| V3 | 6 docs | **BUILT** |

**The two largest owed items are V1's Working-lately + priority switch and V2's
AWAY Recap, and both are V4.** Every rule they need is already written down - the
switch reorders and never withholds (CLAUDE.md), "lately" is `LATELY_SESSIONS`,
the headline statistics are `swing_headline` and `held_run_score` - so what is
missing is the surface, not the decision. **The priority switch is not built, and
CLAUDE.md no longer claims a test for it** (R4 B3): the
identical-visible-rows test is owed WITH the switch.


Decision `docs/decisions/0016-trader-vision-and-priorities.md` is the tie-breaker
for this phase: **when two packets compete, the one that improves WHICH NAMES ARE
SHOWN beats the one that improves WHEN TO ENTER.**

### Phase 0.14 packet V2 — The loop closes (2026-09-02) — items 1, 4 and 5 BUILT; 2 and 3 NOT BUILT

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-05 (repo cleanup), unabridged. Status at the move: the Phase 0.14 table above is the current state; V1 and V2 build records are in the archive. **V4 (Working-lately + priority switch, AWAY Recap, the Weekend Prep takes table and collapsed notes, the Setup Types tab) is NOT BUILT.** Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the move closed nothing.

### Phase 0.14 packet V1 — Names first (2026-09-02) — BUILT; V4 owed

Build record in the same archive file, unabridged.

### Phase 0.13 packet P3 — The fact pack tells the truth (2026-09-01) — BUILT, live gate owed

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P7 — One name per setup (2026-09-01) — BUILT, no live gate

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, no live gate**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 — Trader decisions of 2026-09-01 (packet P0) — BUILT, live gate owed

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P1 — Grade what you already said (2026-09-01) — BUILT, live gate owed

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P2 — Show me (2026-09-01) — BUILT, live gate owed

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P4 — The variables you are not looking at (2026-09-01) — BUILT, live gate owed

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P5 — Pass and not-today get graded (2026-09-01) — BUILT, live gate owed

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P6 — Preference to trade (2026-09-01) — BUILT, live gate owed

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P6a — Tag the backlog (2026-09-01) — BUILT, live gate owed

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P8 / Phase 6.1 addendum — First setup-parameter grid (2026-09-02) — BUILT, live gate owed

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P9 — Quick like (2026-09-02) — BUILT, live gate owed

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 packet P10 — What happens after I like it (2026-09-02) — BUILT, live gates owed

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gates owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 review round R2 (2026-09-02) — TWO GUARDS, BUILT

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.13 review round R1 (2026-09-02) — BLOCKERS FIXED, ALL PACKETS MERGED

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.11 — Theta premium optimization (authorized 2026-08-31) — BUILT, live gate owed

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT, live gate owed**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 0.12 — Focus de-clutter + higher-timeframe LRSI research (authorized 2026-09-01)

Long form moved to [`docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.8-0.18.md) on 2026-09-03 (F1 docs packet). Status at the move: **BUILT**. Every live gate this phase still owes is a numbered row in `CURRENT_CHECKPOINT.md`'s open-gates table; the archived text carries the item list and gate clauses verbatim.

### Phase 1 — NEXT: remove known uncertainty from the development baseline

1. **P1.1 Make the test suite hermetic.** Stop Qt app tests from starting live
   universe/yfinance work; keep explicit network/broker markers and bounded teardown.
   *Built 2026-08-18 (offline tripwire, IB/yfinance/market-prep stubs, `network`/
   `broker` opt-outs) and **completed 2026-08-24** (packet W1): the last unbounded
   half of teardown is closed. Measured with a thread-recording plugin over a full
   run: **22 tests left a thread running past their own teardown and 19
   `run_strategy` threads were still alive when the session ENDED** - the standing
   crowd `conftest.py`'s garbage-collection block already named and said it did not
   join. `conftest.retire_leaked_bounce_bots` now calls BounceBot's own
   `stop(timeout=...)` for any strategy loop a test leaves behind and FAILS the
   leaking test if one survives; re-measured after the fix, **0 scanner threads
   survive the session** and the other seven leaks (`scan-*-drain`,
   `qt-health-audit`, `industry-board-refresh`, the Desk Link reader) all end on
   their own before it. The wall-clock flake class (two panel tests that failed
   only between 06:30 and 07:00 PT, inside the open-burst digest window) was
   repaired 2026-08-23 by pinning those tests' clocks; verified still pinned.
   Determinism evidence: three consecutive full runs, identical pass counts,
   exit 0.*
2. **P1.2 Resolve the measured D1 line-display defect.** After the testing week,
   decide the red-level threshold and total clutter budget from desk evidence. Ask
   before touching any fenced detector/scoring/alert-hosting file.
3. **P1.3 Adjudicate pending branches.** Review the scoring/flagging branch only with
   golden fixtures; discard or supersede obsolete documentation-only work after this
   consolidation.
4. **P1.4 Finish observability depth.** Add representative benchmark/golden fixtures
   and trends for timings, provider calls, failures, coverage, and scan-stage latency.
   *BUILT 2026-08-24 (packet W7).* `scripts/diagnostics/observability_trends.py`
   reads the run manifests and `ai_job_ledger.jsonl` that already exist and folds
   them into a trend: per-phase latency against the window before it, the
   `provider.<family>` counter tree with cache-hit and failure rates, run and job
   failure counts with the errors quoted, and coverage from the scan's own
   `symbols_processed`. **Zero new measurement** — nothing is instrumented, timed
   or run during a scan, and an AST test keeps it that way. Frozen by the golden
   fixture `observability_trends_v1`, whose inputs are hand-written to contain
   each shape the reader has a rule for (a phase with no baseline, a phase absent
   from one run, a family with no attempts, a failed run, a mixed job record)
   rather than a copy of one machine's diagnostics. Its first live read named
   two real failures the desk had not been counting: `journal_import` 9 of 12
   (the dead Questrade refresh chain, a trader action) and `ticker_briefs` 11 of
   30.
5. **P1.5 Do bounded repository hygiene.** Ignore generated desk JUnit output and
   remove retired Desk Link/satellite/mini-PC code only in an explicit, fully green
   cleanup packet. Do not mix cleanup with behavior changes.
   *DONE 2026-08-24 (packet W8), in one commit with no behavior change.* Removed:
   the `desk_link` package (7 modules), `ui/satellite.py`, `ui/desk_role.py`, both
   `ui/services/desk_link_*` modules, `master_avwap_mini_pc.py`, the Settings ▸
   Desk Link tab, the `--satellite`/`--link-token`/`--satellite-desk`/`--desk-role`
   flags, the control banner, and 70 tests across 7 deleted files. `desk_report.xml`
   is ignored. **The edit reached eight methods in `alert_center_panel.py`**, which
   houses alert code, so the file-scoped ask-first rule was invoked and the trader
   authorized full removal on 2026-08-24 before anything was touched. What SURVIVES
   deliberately: the generic `read_only` mode on the price-alert board and panel,
   which is a widget capability with its own tests rather than satellite plumbing,
   and now has no production caller. Packaging triggers fired by design — the spec's
   `desk_link` entry is gone, the exe was rebuilt and
   `dist\TradingBotV3\TradingBotV3.exe --selftest` returned **70/70 (frozen)**,
   exit 0.

Exit gate: tests are deterministic/offline by default, the chart-level policy is
intentional, open branches are resolved, benchmark evidence is stable, and retired
topology code no longer confuses the supported runtime.

### Phase 2 — FOUNDATION: create authoritative data paths before new ranking

Each item requires parity/rollback evidence before the next authority cutover.

1. **P2.1 Complete storage and secrets classification.** Preserve operational home,
   machine-local, and research-lake boundaries; migrate remaining live databases or
   secrets only with backup, dual-read verification, and rollback.
2. **P2.2 Introduce the provider repository.** Centralize IBKR/Yahoo fetches, cache
   keys, batching, request coalescing, pacing, source, and freshness behind golden
   parity tests. Do not change champion results.
3. **P2.3 Repair remaining point-in-time defects.** Cover moving levels, history
   keys, backfill leakage, tracker identity, score ordering, factor horizons,
   corporate actions, and survivorship with intentional-difference fixtures.
4. **P2.4 Make CandidateRegistry authoritative.** Migrate every live candidate writer,
   preserve manual names, prove expiry/restart/rollback, and retire duplicate text-
   file authority only after parity.
5. **P2.5 Integrate aligned SPY/sector/industry/stock RS as advisory evidence.** Expose
   complete-through time and provenance; unpromoted fields contribute exactly zero to
   production eligibility, score, order, sound, and delivery.
6. **P2.6 Give Greatness a dedicated completed-bar lane.** Establish continuous
   coverage, stable identities, revisions, and the evidence hooks required by
   Section 7 without changing legacy D1 alerts.

Exit gate: provider, point-in-time, candidate, market-state, and Greatness inputs are
stable and reconstructable. Production still uses the legacy champions.

### Phase 3 — EVIDENCE AND CAPTURE: finish what must mature over time

These lanes may collect in parallel with Phase 1–2 work because they are additive and
non-authoritative. Their analysis/cutover steps remain ordered here.

1. **P3.1 Complete Chart Review live acceptance.** Verify sub-five-second capture,
   chart provenance/fallback warnings, painted-level references, the one alert writer,
   and zero privileges from LIKE/veto/note annotations.
2. **P3.2 Complete warehouse live validation and the 20-session pilot.** Verify IB
   transport, tee/tiles, gaps, pacing, storage growth, backups, and restore. Then build
   the tracker-to-detection adapter, explicit bounce linkage, and only the additional
   context/VWAP fields demanded by registered consumers. Keep it shadow-only.
   *Narrowed 2026-08-27:* the tracker-to-detection adapter and the first demanded
   context dataset are **BUILT / GREEN**. The adapter reads the small transition
   ledger plus the scenario CSV, admits every canonical tracker family with usable
   geometry, and never parses the 1 GB snapshot. Five point-in-time Auto Market Bias
   views (M5/M30/H1/H4/D1) now attach to each studied occurrence. What remains here
   is the live warehouse canary/pilot and BD-43's explicit BounceBot occurrence link.
3. **P3.3 Complete Local-AI Phase 1, then redesign Phase 2.** After five clean
   unattended mornings, specify deterministic fact packs, evidence budgets, schema,
   failure behavior, and tests before writing the append-only digest format. Require
   ten clean digest sessions before later AI phases.
   *Narrowed 2026-08-10:* evidence budgets and failure behavior are **done** — local
   calls cap at `ai_local_evidence_budget_chars` and raise on server-side prompt
   truncation. A fact-pack design packet is **proposed** in
   `docs/LOCAL_AI_AUTOMATION_PLAN.md` sec 6.4a. What remains owed: **trader answers to
   its six open questions**, then schema and tests. No digest schema may be built or
   frozen before those answers — the 2026-08-08 decision still stands.
   *Satisfied and BUILT 2026-08-24 (packet W4):* the six answers were given and
   recorded in `docs/analysis/OFFLINE_BUILD_AUTHORIZATION_2026-08-24.md` §1, so the
   2026-08-08 decision is met rather than waived. `scripts/ai_jobs/digest.py` writes
   the two artifacts per session — a deterministic fact pack (zero LLM, written even
   when the model is down) and a medium-tier narration that reads the fact pack and
   nothing else — and `daily_digest` is APPENDED last in `default_slots()`.
   **Still owed here: the ten clean digest sessions plus the trader spot-audit of at
   least three packs against raw evidence.** Building the ledger never marks that
   gate met. The enrichment (P3) and policy-draft (P4) machinery is BUILT and
   RUN-GATED under packet W6; what P3.3 retains is the gates, not the code.
   *Armed and built 2026-08-11:* the **ticker-briefs hardening packet**
   (`docs/LOCAL_AI_AUTOMATION_PLAN.md` sec 6.4b) was armed by the trader after the
   first overnight run and is **implemented**: project-then-budget evidence (TB-0),
   per-ticker failure isolation with an honest partial morning file (TB-1),
   deterministic membership-only skip (TB-2), resumable per-symbol completion (TB-3),
   and a per-session attempt cap (TB-4). Per the gate's own reset rules the
   **ticker-briefs five-session clock restarts at zero**; the `ai_summary` clock
   continues, because its code path is untouched.
   *First live night ran 2026-08-11 and the proof is **partial**, repaired 2026-08-12:*
   `ai_summary` succeeded first attempt; `ticker_briefs` briefed 101 of 182 symbols
   and was killed mid-batch, publishing no morning file. TB-0 is confirmed; TB-3 was
   proven broken and is fixed; TB-1/TB-2/TB-4 were never exercised. Three repairs
   landed — **TB-5** (a roster line is not evidence: 96.2% of the payload was ticker
   name-dumps, and removing them cuts 166 model calls to 49), **TB-3's stable
   `resume_key`**, and **TB-6** (the morning file is republished after every resolved
   symbol, so a hard kill no longer loses the night) — plus the scheduled task's
   `ExecutionTimeLimit`, which at `PT2H` against an 8-hour window was terminating the
   parent and letting a second concurrent runner start.
   **Still owed: live proof on the 2026-08-12 window**, and it is only interpretable
   once the desk stops sleeping — 4h39m of Modern Standby, trader-owned, ended the
   08-11 run. A night cut short by sleep is not evidence about this layer.
   *The nightly journal pull queued here 2026-08-11 was **promoted into Phase 0.5
   R7 on 2026-08-15** (trader go recorded in
   `docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md` §6, which honors the sec 6.4c design
   verbatim and supersedes the "after 6.4b proof" ordering). P3.3's remaining
   scope is the fact-pack/enrichment/policy-draft work only.*
4. **P3.4 Accumulate and audit promotion evidence.** Continue regime infrastructure
   toward 40 instrumented sessions, and SPY/Greatness toward their Section 7 floors.
   Freeze windows before inspecting outcomes.
5. **P3.5 Build the live market commentary journal.** First define its relationship
   to Chart Review notes and the existing journal; then add one append-only, rapid
   intraday capture stream and advisory nightly summarization. It never becomes a
   detector, score, gate, or alert input.

Exit gate: capture paths are live-validated, warehouse and AI gates have honest
results, commentary is reconstructable, and promotion datasets are usable without
look-ahead or missing-denominator ambiguity.

### Phase 4 — CANONICAL OPPORTUNITY: build the challenger product

1. **P4.1 Freeze the identity graph and Opportunity snapshot contract.** Separate
   Objective Quality, Actionability Now, Expected R, and Personal Fit; make every
   input and blocker inspectable.
2. **P4.2 Build the canonical eligibility/ranking challenger.** Consume only the
   authoritative Phase-2 inputs; prove every unpromoted field has zero production
   contribution and retain a one-switch champion rollback.
3. **P4.3 Complete the Greatness/readiness gate stack.** Add remaining reward,
   freshness, volume, context, ordered levels, failure/re-arm, and anti-chase logic in
   pure tests, replay, and live shadow.
4. **P4.4 Build the advisory Command Center and Focus Workbench.** Show lifecycle
   lanes, compact dossiers, reasons/blockers, mini charts, and honest zero-Ready days.
5. **P4.5 Freeze and pass the ranking manifest.** Compare identical snapshots and
   outcomes, run the bounded GUI canary, and promote projection separately from rank.

Exit gate: one deterministic advisory Opportunity snapshot exists, the trader can
use it without changing production routing, and any promoted projection/rank has
passed its own manifest and rollback drill.

### Phase 5 — DELIVERY AND LIFECYCLE: make every surface agree

1. **P5.1 Build the typed delivery challenger.** Immediate, Heads-Up, Focus Changes,
   Developing, Research, and History remain separate; deduplicate/group bursts while
   protecting every user-armed hit.
2. **P5.2 Freeze and pass the delivery manifest.** Canary sound/severity/routing
   independently from ranking and retain complete History plus instant rollback.
3. **P5.3 Project one verified snapshot everywhere.** Desk, Focus, Alert Center,
   Auto/Away, phone report, journal, and AI must agree on snapshot/opportunity IDs,
   stage, rank, freshness, and champion/challenger status.
4. **P5.4 Complete lifecycle and journal linkage.** Join discovery, stages, reviews,
   Focus/watch actions, fills, no-trades, outcomes, screenshots, MFE/MAE, planned vs
   actual risk, and after-close reconciliation. *Narrowed 2026-08-15: the fills
   completeness, planned-risk capture, and broker after-close-reconciliation slice
   moved to Phase 0.5 R7; P5.4 retains the lifecycle joins, screenshots, and
   opportunity-identity linkage.*
5. **P5.5 Build the Learning Center and controlled universe intake.** Keep objective
   edge, actionability, personal preference, execution, and discovery-source value
   separate; personalization may reorder only inside declared safe bands.

Exit gate: the complete decision lifecycle is reconstructable, alerts are useful and
bounded, every surface agrees, and preference cannot change objective truth or safety.

### Phase 6 — RESEARCH PAYOFF: learn and promote narrowly

1. **P6.1 Complete the warehouse post-slice research tools.** Add the registered
   setup/style readouts, Level Edge/recipe comparisons, and evidence packages only
   after the pilot validates the corpus.
   *Partial build 2026-08-27, shadow-only:* the first bounded stop/target comparison
   is implemented for tracker D1 occurrences: next session's first completed M5 close
   is entry; structural-stop ranks 1–3 and 0.5/1.0/1.5 ATR controls are crossed with
   1R/2R/3R targets under STOP_FIRST and the existing cost model. A deterministic
   nightly fact pack is always written; medium local AI may narrate only after n>=30,
   five symbols and five sessions. Corpus accumulation, pilot validation, registered
   holdout work and every promotion gate remain owed.
2. **P6.2 Promote advanced setup families one at a time.** Each family requires a
   registered question, point-in-time corpus, replay, shadow, live evidence, bounded
   canary, approval, and rollback.
3. **P6.3 Continue Local-AI phases in order.** Journal enrichment, review-policy draft
   comparison, and periodic frontier synthesis remain advisory and start only after
   their predecessor gates pass. *Amended 2026-08-10:* the review-policy draft
   comparison runs frontier-vs-medium rather than local-large-vs-cloud — the local
   large tier is retired (no 27B-class model loads beside the running desk). The
   two-week side-by-side quality gate is unchanged.
4. **P6.4 Finish Market Prep migration into Qt.** Retire the Tk path only after parity,
   operational recovery, and clean-machine proof.

Exit gate: research produces trustworthy narrow improvements without leaking into
champions, and the supported interactive product is fully Qt.

### Phase 7 — LATER: consolidate and ship the internal product

1. **P7.1 Complete CI and clean-machine recovery.** Cover supported Windows/macOS
   tests, offline smoke, frozen regression, backup/restore, and operator recovery.
2. **P7.2 Finish packaging and release polish.** Icon, version metadata, windowed
   build decision, bundle trimming, installer, and release notes follow the frozen
   rebuild policy.
3. **P7.3 Revisit read-only broker adapters.** Only after the provider repository is
   stable; execution remains permanently out of scope.

Everything else stays in `WISHLIST.md` until explicitly promoted into this sequence.

**Long form.** The verbatim build narrative for Phases 0.5–0.7 moved to
[`docs/archive/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md`](docs/archive/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md) on
2026-08-28. Every numbered item and every owed gate stayed here, unabridged; only the
description of work already built moved. That file is evidence — if it disagrees with
this section, this section wins.

### Lint (2026-08-31, trader-directed) - CLOSED

`ruff` 0.16.5 is installed and pinned, and `ruff check .` reports **All checks
passed**, down from 1,703 findings on its first run. The backlog that stood here -
74 unused imports and one `F821` in an alert file - is closed; both were swept on
the trader's explicit yes. "ruff clean" is now a claim this repo can make.

Keep it that way: run `.venv\Scripts\python.exe -m ruff check .` before a commit,
alongside the test suite. The narrow select in `pyproject.toml` (`E9`, `F63`,
`F7`, `F82`, `F401`) is deliberate - widen it as the legacy cores shrink, not
before.

## 13. Definition of done

The roadmap is complete when:

1. the single main desk is reliable, observable, recoverable, and live-validated;
2. data is freshness-aware, point-in-time correct, provider-efficient, and owned by
   one writer;
3. every surface consumes one canonical opportunity lifecycle and saved snapshot;
4. SPY/RS and Greatness behavior is promoted only after the evidence ladder passes;
5. alerts are current, typed, deduplicated, and protect every user-armed hit;
6. the journal reconstructs discovery, judgement, execution/no-trade, and outcome;
7. research and AI remain reproducible, advisory, and separated from champions;
8. new setups enter production only through fixtures, replay, shadow, live evidence,
   canary, approval, and rollback;
9. the supported test, smoke, packaging, and recovery gates are green; and
10. the application remains decision-support only and performs no execution.
