# TradingBotV3 implemented history

Last reconciled: **2026-08-27** on `claude/gui-phase-0-9`, at the four trader
rules of that morning (regime-pause auto-Focus `479c25c`, the VWAP-side /
show-time review filter `76e0b7b`, the D1 SMA trend leg + snapshot Prev/Next
`f3abda7`, the M5 alert bar `41963de`/`39c3ef7` and its click-away skip, then
the group tape removed and REBUILT, then the desk-memory packet, both on
`claude/warehouse-build-memory`)
after Phase 0.9's first three packets - the table width rule, the AWAY Recap return
surface and the Desk Journal keyboard route. The same branch also carries Phase
0.10's AVWAP band challenger and its review fixes (two sessions shared one
checkout on 2026-08-26; see `CURRENT_CHECKPOINT.md`).

Authoritative for: **what exists and the historical sequence of revisions**

Remaining work: [`plan.md`](plan.md)

This is a curated product history, not a raw commit dump. It reconciles the former
status sections in `plan.md`, the accumulated `CURRENT_CHECKPOINT.md` ledger, the GUI
plans, warehouse plans/reviews, dated handoffs, and Git history. Exact current test
counts remain in `CURRENT_CHECKPOINT.md`.

The labels retain their strict meanings: `IMPLEMENTED` means code exists, `GREEN`
means deterministic tests pass, `LIVE_VALIDATED` requires real-session evidence,
and `PROMOTED` requires an explicit champion decision. A feature can be implemented
and green while its live or promotion gate remains open in `plan.md`.

## Current implemented inventory

**This is the contract: what exists, by area. Search it before building anything so you
do not rebuild landed work.** It is deliberately short. The dated entries under
`Recent changes` below cover the last two build days; everything older is in
[`docs/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md`](docs/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md),
which is evidence and must not be loaded as context.

### Application, runtime, and data ownership

- PySide6 Trading Desk launched by `launch_gui.py`, with the legacy Tk UI retained
  as a compatibility path.
- Main-desk single-process ownership, bounded BounceBot startup/shutdown, generation
  guards, child-process reaping, runtime heartbeat, durable job ledger, typed retry
  budgets, stale-run marking, and a hardened single-instance launch guard that also
  sees the frozen executable.
- User-selected shared home folder for operational text/JSONL/CSV artifacts;
  machine-local settings, caches, and diagnostics under LocalAppData; a separate
  research-lake storage class outside that home folder.
- **No cloud sync (2026-08-10, decision 0015).** Google Drive/OneDrive were removed
  from the system entirely. `C:\TradingBotData` keeps its path and role as a plain
  local folder; the DAS file server `\\MINI-PC\Trading Bot Data` is the durable
  tier, holding the research lake, the AI store, and hourly cold-pushed subtrees.
  Documentation-only change: no path, behavior, or test changed.
- Designated-writer authority, local kernel exclusion, fenced writer lease, atomic
  publication, readback verification, last-good preservation, and bounded archives.
- Main desk is the sole always-on scanner. The former mini-PC scanner and Desk Link
  satellite topology were `RETIRED` 2026-08-08 and their code was **removed 2026-08-24**
  (P1.5): no `desk_link` package, no `ui/satellite.py`, no `master_avwap_mini_pc.py`, no
  `--satellite`/`--desk-role` flags.

### Scanning, candidates, and decision support

- Master AVWAP D1 swing scanning with earnings anchors, current/previous AVWAP
  families, running-deviation bands, focus buckets, Expected-R ranking, study tags,
  theta candidates, tracker history, and durable daily-bar storage.
- BounceBot completed-M5 detection with session VWAP/bands, EMA and prior-day
  levels, relative strength/weakness, regime-aware candidate discovery, tiering,
  alerts, outcome tracking, and the day-scoped M5 Focus path.
- BounceBot's sweep runs only inside the session window (open-30m to close+30m by
  default, weekdays); outside it Auto Pilot pauses scanning and holds the IB
  connection open. A manual resume survives until the next boundary.
- CandidateRegistry foundation with provenance, source leases, transitions, atomic
  versioned persistence, and partial shadow adoption. Full authority remains open.
- Industry Board with one single-flight owner, hourly refresh, atomic last-good
  snapshot, numeric sorting, freshness/Health integration, and advisory aligned
  industry-vs-SPY plus stock-vs-primary-industry fields.
- Auto-populate rules for both regimes, previous-day-extreme gating, DESK adoption
  into M5 Focus, and one extension notification per Focus name/day while pullback
  notifications stay active.
- Focus privileges begin only beyond the previous session's directional extreme;
  missing prior-day data grants nothing.
- D1 Focus routes final Favorite/High Conviction upgrades while developing trigger
  evidence remains research-only. Legacy D1 champion alerts are unchanged.

### Charts, review, alerts, and phone surfaces

- Chart-first review flow, current forming D1 preview, D1/M5 shared snapshot widget,
  log scale, crosshair/OHLCV readout, source/age strip, fallback warning, cache
  invalidation, background loading, prewarming, and stall watchdog.
- Chart Review workspace with lookup for any symbol, hidden-by-default Setups drawer,
  keyboard-first LIKE/veto/note/setup-claim capture, versioned veto vocabulary,
  append-only `trader_annotations.jsonl`, and isolated forward veto cohorts.
- Painted D1 S/R, previous-day H/L, projected trendline, SMA/EMA/AVWAP groups,
  machine-local visibility preferences, stable level IDs, click selection, and
  click-to-arm routed through the one `PriceAlertService` writer.
- Chart Review annotations cannot add Focus/watchlist membership or price alerts;
  LIKE records judgement only.
- Visual Alert Center and review queue, chart-armed watches, persistent History,
  structured review decisions, review scoreboard, and annotation-only/FIFO policy
  gate.
- Main-only price-level polling with cross-up/cross-down, one fire per arm, urgent
  ntfy push, persistent main-desk presentation, and manual re-arm.
- Auto modes OFF/DESK/AWAY/EVENING, honest global status, EVENING early scan and
  briefing, and one verified `autopilot_today.txt` with safety/freshness first,
  numbered best swings, intraday candidates, and condensed operations.
- The double-click symbol snapshot popup opens at desk height (2026-08-11): its size
  is taken from the hosting window's frame, or the screen's available area when the
  window is not yet measurable, never smaller than the former fixed 1180x760, and is
  centered on the desk window and clamped inside the screen. Opening geometry only —
  a trader resize survives subsequent double-clicks.
- On 2026-08-10, best swings gained an ntfy report notification; it stays quiet when
  the generated swing section contains no readable setups. Late-opened alerts now
  receive current bars, and the Chart Review Setups column defaults hidden with a
  visible restore control.
- Phone push policy, 2026-08-11 (trader rule): **AWAY is the only mode that pushes**,
  and the Research/Focus price alerts are the single deliberate exception — they keep
  their own always-on urgent channel, unchanged. The EVENING morning-briefing push and
  the retired Desk Link control-reclaim push are now silent outside AWAY; both still
  announce on the desk. The hourly swing push carries the **full favorite and
  high-conviction roster** under the ranked picks, built from the whole current feed
  rather than the top-ten slice, side-split, with `near` excluded and an explicit
  "did not fit" marker if the message ever exceeds the ntfy size ceiling; a roster with
  no ranked picks still sends. A **second hourly push names every stock that fired a D1
  level or event alert since the previous one** (armed D1 levels, D1 event watches,
  Focus D1 flags, and the scanner's ready D1 focus alerts), new-since-last-push rather
  than cumulative, silent on an empty hour, and cleared only on a delivered push so an
  ntfy failure never eats the events. The Alert Center classifies (it owns the D1
  routing rules) and Auto Pilot aggregates and gates, so the phone and the D1 Focus
  feed cannot disagree. Machine-local kill switches: `push_away_swings`,
  `push_away_d1_events`. **Extended 2026-08-14 (packet R1):** EVENING's SPY ±1%
  wake alarm is the *second* deliberate exception — urgent, repeating every five
  minutes while the move holds, stopping on the flip out of EVENING, kill switch
  `push_evening_spy_alarm`.
- Auto-mode matrix, 2026-08-14 (packet R1): discovery is identical in every mode;
  what differs is who is present to act. DESK adopts staged picks immediately;
  AWAY stages and never adopts and queues alerts silently (only the sound is
  suppressed); EVENING runs its early block and then stops scanning entirely,
  staging picks for the wake-up flip; OFF is the only mode that still
  self-applies. Quiet hours confine every automatic starter to weekdays,
  06:00–14:00 local; manual buttons are never gated.
- One Master AVWAP scan action, 2026-08-15 (packet R1). The Shared/Local pair read
  the identical two watchlist files, so `use_shared_watchlists` and the menu choice
  it drove were removed across thirteen files. Cloud-drive *store discovery* went
  with it (decision 0015 amendment); the mount-presence guard stays.

### Journal, explanations, and learning

- **R7/R8 adversarial release-candidate repair (2026-08-15).** Every verified
  A1–A19 and B1–B14 finding was closed before handoff. The repair normalizes
  broker-ledger casing and Flex dates; preserves shared Focus wiring and exact
  suggestion-row identity; scopes reconciliation clears to reachable brokers;
  bounds shutdown; migrates every execution leg and gives fills stable
  identities; makes coverage, quarantine, currency, FX ordering, token
  precedence, weekly identity, exit-window, empty-last-good, and OCC handling
  fail honestly; and restores the journal's missing pull/gap/retry controls,
  grouped tags and filters, reversible undo, atomic exports, and truthful labels.
  Expensive journal work now runs in a worker and re-renders from captured
  structured results without re-querying; migration starts only after an
  explicit **Prepare Journal database** click and remains visibly gated in the
  background. Weekend rollover, timezone conversion, failed-discovery state,
  Flex reuse, single-fetch boards, board persistence, and failure signaling are
  likewise pinned by regression tests. Account tax labels moved out of source
  into machine-local settings. No live journal database or broker was touched.

  Scope reconciliation is explicit: true non-USD-to-USD conversion, the
  Calendar year heatmap, additional Analytics charts, Weekend RRS-strength
  joins, and Weekend Focus performance/pick-feedback/veto joins remain deferred
  in their governing specs. They are not represented as shipped behavior. The
  repaired code tip is `dd201cd`; deterministic baseline is 3354 passed / 19
  subtests, smoke 7/7, frozen selftest 49/49, all exit 0. Live gates remain owed.

- **Weekend Prep (R8, 2026-08-15).** A guided five-step weekend routine with
  persisted progress: week in review, focus-pick review, week-windowed walk-away
  with the weekly auto-tag review, strength discovery on H1/D1/Monthly using the
  M5 formula through the fenced `strength_scan` functions, and the week-ahead
  prep from the `market_prep` weekly engine. Manual refresh only, zero IB
  traffic, adds-only adoption into swing Focus.

- **Tax-grade journal (R7, 2026-08-15).** Stable `BROKER:account:exec_id`
  execution identity; one security-type vocabulary across both brokers; anchored
  `trade_id` with an annotation re-key pass and `trade_aliases`;
  `CLOSED_PARTIAL` and a `SYNTHETIC_OPEN` marker instead of a fabricated inverse
  position; append-only `trade_adjustments` corrections re-applied at every
  rebuild; an `import_coverage` ledger with a bounded nightly self-heal; IBKR
  Flex as the primary history source including OptionEAE, OpenPositions and
  CashTransactions; Questrade activities and a trade-day cross-check; Bank of
  Canada FX booked once per (date, currency); reconciliation against both
  brokers' reported positions with trader-confirmed force-closes; a nightly
  `journal_import` slot at the front of the `ai_jobs` slate; and a five-tab
  Journal (Trades, Calendar, Analytics, Health, Fees) over one shared
  tax-grouped header.

- **Statement layering, direction and self-check (2026-08-28).** Statement
  identity is `fill_signature` + an ordinal within it, so a later, longer export
  layers instead of doubling; long vs short is read from Questrade's own
  `STOCK SHORT.` / `COVER SHORT.` description marking rather than from row
  order; and `reconcile_statement` adds a file up by hand and compares it to the
  assembled trades, per symbol, writing a CSV. Journal > Health >
  "Check a statement...".
- **Broker statement import (2026-08-28).** `scripts/journal_statement_import.py`
  reads a Questrade activity export (.xlsx via `zipfile`+`ElementTree`, no new
  dependency; also .csv) and writes executions, cash rows and account tax status
  for the days the executions endpoint's retention horizon can no longer reach.
  One commission column taken as the whole cost; options resolved from the
  Description into OCC symbols with a 100 multiplier; timestamps at midnight
  market-local so a date-only row is never given a session; and a statement
  never writes into a (broker, account, day) a richer source already covers.
  Reachable from Journal > Health > "Import statement file...".
- **Two-lane journal auto-tagging (2026-08-28).** `scripts/journal_trade_shape.py`
  derives hold bucket, entry session bucket, execution shape and instrument from a
  trade's own timestamps and legs, so history imported from outside the scanner's
  lookback is tagged rather than blank; `AutoTagger`'s setup lane still leads both
  the stored summary and the candidate list, ordered by lane rather than confidence.
  No tag is ever derived from the outcome. Around it: a tag filter on the shared
  Journal header, `distinct_tags` counting the trader's lane separately from the
  machine's, `rename_tag` (rename or retire across every trade, trader-typed tags
  only), a Manage-tags dialog, Accept-all, and an accepted suggestion that stops
  re-proposing itself.
- Journal schema v2 with append-only opportunity lifecycle events, idempotent broker
  Taken/Closed imports, structured reviews, free-form notes, tags, and analytics.
- Deterministic novice explanations across Setup Tracker, Day Trade Tracker, and
  Move Forensics, plus an evidence-floor-aware “What’s Working” summary.
- Review events partitioned by installation, merged/deduplicated by readers, capture
  audits, preference scoreboard, AI-curated `review_policy.json`, and a permanent
  no-suppression boundary.
- Technical Integrity research hierarchy with point-in-time predictions/outcomes,
  break pressure, calibration report, and no detector/watchlist/alert influence.
- Regime infrastructure evidence for SPY baseline, breadth, Technical Integrity
  follow-ups, and audit tooling. The evidence remains exploratory/non-promotable.

### AI and automation

- Provider-neutral A.I. Summary workspace for OpenAI and Anthropic, explicit evidence
  selection, bounded preview, credential-manager storage, structured/source
  validation, immutable evidence packages, and export-only results.
- Config-gated local OpenAI-compatible provider through Ollama, default off; small and
  medium model tiers verified on the Ryzen main desk with no market-hours inference.
  The local large tier is `RETIRED` (2026-08-10): 27B-class models no longer load
  beside the running desk on the 780M, so its jobs belong to the frontier model. Local
  calls are capped to the tier's context window and fail loudly on server-side prompt
  truncation.
- Separate off-hours `ai_jobs` process and scheduled task, job-ledger integration,
  deterministic evidence coverage, daily advisory summary, per-ticker briefs, full
  artifacts in `ai_store`, and bounded atomic `ai_morning_brief.txt` publication.
- Per-ticker briefs project each symbol out of a full-size base package and then
  ration the projection to the local context window; ticker-roster and bare-name
  lines are discarded as non-evidence, each symbol resolves independently, a symbol
  with no evidence beyond watchlist membership is answered without a model call,
  completions resume on a read-stamp-independent evidence key, the morning file is
  republished after every resolved symbol, and the slot spends at most three attempts
  a session.
- Local-AI Phase 0 is complete. Phase 1 implementation is complete; its five-session
  unattended live gate remains in `plan.md`.

### Durability and catch-up

- Repeating 06:00 Pacific weekday launch task through the session, protected by the
  existing single-instance guard.
- Master AVWAP tracker staleness catch-up from completed prior-session D1 data with
  explicit `data_session` vintage and no automatic scoring-tuner/prior-refit side
  effects.
- Technical Integrity follow-up and breadth-ledger deterministic backfill with
  bounded retries, explicit `capture_mode`, honest gap rows, and live/backfill audit
  separation.
- Frozen snapshots, never-started predictions, and other Tier-C evidence remain
  intentionally non-reconstructed.

### Research warehouse

- Phase 0: research-lake decision record, configuration, home-folder-path refusal, layout,
  and disabled-by-default no-op behavior.
- Phase 1: immutable Parquet store, four-step seal, append-only manifest authority,
  13 frozen schemas, quarantine, compaction, retirement, and crash reconciliation.
- Phase 2: idempotent bronze wraps, daily universe/level snapshots, and completed D1
  projection with source hashes and watermarks.
- Phase 3/3b: zero-extra-request M5 tee, coverage/gap rows, capped spool, capture-only
  pacer, IB backfill transport, nightly/weekly backfill, and trickled yfinance seed.
- Phase 4: versioned XNYS sessions and deterministic M15/M30/H1/W1 aggregation.
- Phase 5: point-in-time daily/intraday feature snapshots and anchor instances using
  champion calculations, including AVWAP parity at 1e-9.
- Phase 6: deterministic occurrence/revision/episode identity and versioned swing and
  intraday outcome simulation with costs, ambiguity bounds, partials, time stops,
  slippage, and open/truncated states.
- Phase 7: manifest-resolved read path and read-only Research panel; DuckDB remains
  optional and pyarrow can answer every slice.
- Phase 8: three-class backups, restore check, single-flight build/status CLI, job
  ledger, and six Health tiles.
- Defect passes repaired outcome supersession, management bounds, feature windows,
  per-bar backfill dedupe, pacing clocks, gap semantics, session identity, compaction
  reads, every job invoker, live tee wiring, and off-GUI-thread spool I/O.
- Phases 0–8 are code-complete on the testing-week branch. The broker check,
  confirmation items, and 20-session pilot remain open.

### Testing, packaging, and platform

- Broad pytest suite, deterministic smoke check, pytest markers, narrow Ruff gates,
  layered requirements with constraints, and Windows/macOS path handling.
- Provider telemetry at IBKR/Yahoo/Nasdaq boundaries with completeness contracts and
  honest UNKNOWN until measured.
- PyInstaller onedir spec, Qt runtime hook, asset/package drift test, lazy-engine
  `--selftest`, and a permanent guard preventing self-test from demanding packages
  deliberately excluded from the bundle.
- The first Windows frozen run found and closed an `ai_jobs` bundle-roster conflict;
  the current frozen self-test is 29/29.
- macOS launcher, CloudStorage Drive discovery, Keychain credentials, and machine-
  local path normalization.

### Shadow challengers

- Side-symmetric SPY market-state/pullback engine runs beside the legacy pause
  detector, emits replayable evidence, and cannot affect candidates, alerts, or rank.
- Greatness Monitor persists ordered touch/wick/close/acceptance/retest/failure/re-arm
  transitions beside legacy D1 alerts and cannot alter the champion path.
- Champion-invariance tests prove enabled, failing, or poisoned shadow engines leave
  production SPY/D1 results unchanged.

Neither challenger is promoted. Their remaining evidence gates are in `plan.md`.

## Recent changes (2026-08-26 onward)

Dated entries for the two most recent build days, newest first. Older dated entries
move to the archive; the durable statement of what they built is in the inventory above.


### 2026-08-28 — Statements that layer, a direction that is read rather than guessed, and the trader's own check

`IMPLEMENTED`, `GREEN`, **live gate owed**. Trader direction: *"lets add a
function to be able to take these files, and new ones throughout the year that
layer on top so that in the end I can totally manually calculate and demonstrate
my pnl and then we can compare it to the auto generated stuff."*

**Two defects the first statement build carried, both found by measuring rather
than reviewing.**

*The uid was positional.* It hashed the file's row index, so a January-to-December
export — the same January trades at different row positions — made **884 of 884**
real trades look new. Identity is now `fill_signature` plus an ordinal counted
within that signature. Proven on the trader's two real files: all 884 of the 2026
file recognised inside the 2025–26 file, and re-importing either in any order
leaves 1,516 executions and 202 cash rows unchanged.

*Direction was a coin flip.* A statement has no clock and lists a same-day round
trip SELL-first **227 times out of 227** — a sort, not a sequence — so the
assembler's uid tiebreak decided long vs short at random: **86 of 199**. Questrade
says it in the Description instead (`STOCK SHORT.`, `COVER SHORT.`), so `leg_rank`
orders each row by what it does to the position. That resolved all 227 — **169
long, 58 short** — and all 58 carried both markings.

**`reconcile_statement`** is the trader's own proof and writes nothing: for a
symbol whose quantities net to zero across the file, the sum of its Net Amount
column IS the realised P&L, and that is compared to what `rebuild_trades`
assembled. Open positions are excluded, not zeroed. Measured across both files:
statement **$5,298.81** vs journal **$5,299.05**, difference **−$0.2386** over 428
closed symbols, every symbol inside two cents, **commission exact at $713.68 both
ways**. It does not prove the parse — both sides read the same one.

Importing the 2025 file dropped NEEDS_REVIEW trades from **23 to 5**. Three days
hold both a short and a long in one symbol and are named rather than silently
blended.

Verification: **5361 passed / 72 subtests / 6 skipped**, 5349 → 5361. Smoke 7/7,
selftest 72/72, ruff clean on the new files, same two pre-existing font failures.

### 2026-08-28 — Reading a Questrade statement, for the days the API cannot reach

`IMPLEMENTED`, `GREEN`, **live gate owed** (the trader's own YTD file, imported
on the desk against their real journal). Trader-supplied file and direction:
*"i can easily get us yearly reports from questrade so long as we can process
these files."*

**Why.** The executions endpoint stops at a retention horizon — 2026-06-10 on
this desk. That is why 44 of the 45 `activities report trades…` days can never
be repaired by retrying, and it is the open trader decision R7 has carried since
2026-08-25. The portal's activity export does not stop there.

**What the real file measured**, and what every decision below follows from: 974
rows, 884 trades, 133 trading days, 2026-01-02 → 2026-08-27, both accounts, zero
unreadable rows, and `Net == Gross + Commission` on **every one of the 884 trade
rows to the cent**. So the single Commission column is the complete cost; `fees`
is written 0.0 rather than inventing a split the file does not contain.

**What a statement cannot say** shapes the module. No time of day (every row is
"12:00:00 AM"), so executions are written at midnight market-local and
`journal_trade_shape.is_date_only` refuses to name a session — a date-only round
trip is a `day_trade`, never a `scalp`. Fills are aggregated (some descriptions
say "AVG PRICE"). No execution id and no intraday sequence, so the statement's
own row order is preserved and carried into the surrogate uid; without it two
identical fills on one day hash to one uid and half the position vanishes.
Options carry a Questrade internal id in the Symbol column and the real contract
in the Description — parsed into an OCC symbol, which is what keeps the 100
multiplier and stops option P&L being understated a hundredfold.

**The rule that prevents double counting:** a statement never writes into a
(broker, account, day) that a richer source already covers. The two sources give
one fill different uids, so the upsert cannot see the duplicate; the day is
refused and the count reported. `.xlsx` is read with `zipfile` + `ElementTree`
rather than adding `openpyxl`, which would be packaging trigger 1.

**Measured drift, stated rather than found later.** `rebuild_trades` recomputes
gross P&L from price × quantity while Questrade books Gross Amount to the cent:
**−$0.1558 on $4,014.18 realised across 253 closed symbols**, worst symbol 1.2¢,
and commission matching exactly at $291.38 both ways. Making the assembler prefer
the broker's booked money is a change to the engine both brokers share and was
deliberately not made here.

Verification: **5349 passed / 72 subtests / 6 skipped**, 5326 → 5349, 23 added;
the same two pre-existing font-metric failures. Smoke 7/7 exit 0, source selftest
72/72 exit 0, spec drift 17 passed, ruff clean. No packaging trigger. No detector,
score, alert, watchlist, Focus or `review_policy.json` path is touched.

### 2026-08-28 — Auto-tagging that works on imported history, and the tools to adjust it

`IMPLEMENTED`, `GREEN`, **live gate owed** (one desk session tagging real trades).
Trader-directed, evaluating whether this journal can replace their TradesViz
subscription: *"i want auto tagging then I can come back and adjust."*

**The defect the ask exposed.** `AutoTagger` scores a trade by matching it against
the scanner's own output files. Those files hold the current lookback, so every
trade older than them scores nothing — `suggest_for_trade` returns `[]`, the
summary is written empty, and a year pulled from a broker statement arrives as one
undifferentiated untagged block. Auto-tagging was not broken; it had no inputs for
the case the trader was about to create.

**`scripts/journal_trade_shape.py`** is the second lane: hold bucket (counted in
SESSIONS, so a Friday-to-Monday hold is one night), entry session bucket, execution
shape from leg ROLES, and instrument — all from the trade's own row, no files, no
network, no scanner import. Three rules keep the tags safe to average: no tag is
ever derived from the OUTCOME (a `winners` bucket would post a 100% win rate and
explain nothing), anything unmeasurable emits NO tag, and a naive timestamp gets
market-local ATTACHED rather than an aware one stripped. Candidates order by LANE,
never confidence — shape tags carry 1.0 and would otherwise bury every setup match.

**Adjusting, which is the other half of the ask.** A tag filter on the SHARED
header, so one tag narrows the calendar, the equity curve and the fee totals too;
Analytics could already group BY tag and nothing could filter TO one. `distinct_tags`
counts the trader's lane apart from the machine's. `rename_tag` rewrites or retires
a tag across every trade that carries it — `setup_tags` only, because a derived tag
is re-computed on every refresh and the Manage-tags dialog refuses one rather than
accept a rename the next rebuild would undo. Accepting a suggestion now drops that
SUGGESTION from the queue: the 2026-08-24 reasoning that a tagged trade may still
deserve a second tag is unchanged, but a confirmed trade no longer re-proposes what
it was confirmed with — the mechanism behind 220 proposals against one annotation.

Verification, reading pytest's own exit code: **5326 passed / 72 subtests / 6
skipped**, 5268 → 5326, 58 added; two pre-existing failures are this Linux
container's font metrics and reproduce on a clean checkout. Smoke 7/7 exit 0,
source selftest 72/72 exit 0, spec drift 17 passed. No packaging trigger — the new
file is a module under `scripts/`, not a package, reached by a static import. No
detector, score, alert, watchlist, Focus or `review_policy.json` path is touched.

### 2026-08-28 — Reading the whole evidence pile in slices: 78,119 chars → 1,365,259

`IMPLEMENTED`, `GREEN`, **live gate owed** (tonight's 22:00 window is the first
unattended run). Trader-authorized: *"Can we just give it more time? Like hours to
complete its work then? And spoon feed it slowly so we don't run out of context?"*
Advisory layer only.

**The problem the budget work could not solve.** Raising the context to 64k and deriving
the budget took the summary from 10 of 22 sources to 17 of 22 with none unfunded, and
it still read **one tenth** of what exists: 1,365,259 characters of session evidence
against a prompt that can hold ~91,000. The packager spends that tenth *fairly* rather
than *well* — every source gets a share, so `setups.type_stats` contributed **3 of its
184 rows** and `setups.playbooks` 2 of 200. No further tuning fixes that; 96k crashes
the runner, so the ceiling is hardware.

**The trade.** `scripts/ai_jobs/map_reduce.py` cuts the evidence into slices that fit
comfortably, asks the model for findings from each, then hands back only the findings
and asks it to synthesize. **Every row of every source is read** — 46 slices over 17
sources, ~2.8 hours of a window that runs 22:00–06:00 and was using nine minutes of it.

What the module is careful about, each of which is a test:

- **A slice never passes for its whole source.** Every chunk carries `rows 41-80 of 184`
  in the content the model reads, and the package note tells it to describe only what is
  in front of it. Tables split by ROW (half a row is not evidence); text by window.
- **Citations stay real.** A map call is handed a package containing exactly one source,
  so the existing validator already forbids citing anything else. The synthesis gets
  `citable_aliases` for the ids that actually appear in the findings — so it can name
  the store a statement came from, and nothing that was not read.
- **A failed slice is counted and named** in the published `data_quality`, because a
  document synthesized from 44 of 46 slices is not the same document as one from 46.
- **A failed synthesis does not throw away hours of map work.** The findings are already
  validated and already cite real stores, so they are published *unsynthesized* — and
  the executive line says `UNSYNTHESIZED` in capitals, because a raw pile presented as a
  review would be the more dishonest of the two failures. Proven in the live validation
  run, where the synthesis pass failed and the findings survived.
- **Every slice failing raises** rather than publishing an empty review.

**Two things this exposed, both fixed:**

1. **The truncation tripwire fired on a healthy request.** The findings package is the
   model's own prose and tokenizes at **3.72 chars/token**, where dense JSON evidence
   measures 2.06–2.23 — so an estimate calibrated for one is wrong for the other, and an
   8,325-char package estimated at 3,330 tokens against a truthful 2,235 was called
   sheared. The fix is the half of the check that needs no estimate: **truncation means
   the server clipped to its context, and a clip lands at the ceiling.** Both observed
   shears pinned within three tokens of half the window (6,147 of 12,288; 32,771 of
   65,536), so a prompt evaluated below `TRUNCATION_CLIP_FLOOR_RATIO = 0.45` of the
   window was not clipped, whatever the estimate says. All four real measurements are
   pinned as a regression test. The two pre-existing shear fixtures used 12 and 5
   tokens — values no clip of any context can produce — and were made faithful to the
   historical failure (2,048 context, 1,027 tokens) rather than the guard being loosened.
2. **The scheduled task could start a second copy of a three-hour job.** It fires every
   30 minutes for eight hours, and the ledger only records a row when a job *finishes* —
   harmless while every slot took minutes. `run_slots` now takes a machine-local lock
   (`local_writer_lock`, the same primitive the feature-history writer uses) and a second
   firing stands down cleanly. `local_writer_lock` reports "someone holds it" and "this
   box has no primitive" as the *same* exception and they want opposite answers, so they
   are told apart by the module's own sentence — with a test asserting that sentence
   still exists in `local_writer_lock.py`, so a rewording breaks a test rather than the
   guard. And the summary slot's window reservation, **20 minutes**, was the reservation
   for a job that now takes 170: `summary_reserve_minutes()` returns 200 in chunked mode,
   because a three-hour job launched with twenty minutes left runs into the open.

Off by default (`ai_local_map_reduce`), on for this desk. Tests: 20 in the new
`tests/test_ai_map_reduce.py`, plus the tripwire regression cases.


### 2026-08-28 — The local model was reading a third of its evidence: context 12k → 64k, budget derived

`IMPLEMENTED`, `GREEN`. Trader-authorized ("raise the context... use as much as you
want"). Advisory layer only; nothing here can reach a detector, score, alert, watchlist
or the review queue.

**What the review found.** With the endpoint back up, `ai_summary` stopped saying
"unreachable" and started saying the prompt had been **sheared** — and it had.
Measured against the desk's own model over prompts from 9 KB to 93 KB, the evidence
package tokenizes at **2.06–2.23 chars/token**, not the 3.0–3.5 the code assumed
in two separate places. The consequences compounded:

- the 22,000-char budget was derived in a comment as `7800 tokens × 3.0 chars = 23400`;
  at the real rate 7,800 tokens is ~16,400 chars, so **the default exceeded a
  12,288-token window by about a third from the day it was written**. It survived only
  while few sources were funded;
- on 2026-08-27 the package reached 17 usable sources, the prompt reached ~14,400
  tokens, and llama.cpp sheared it to half the window (6,147 tokens — the pin is
  visible as a constant across prompts of 28 KB, 37 KB, 51 KB and 93 KB);
- the tripwire caught it, but by a **2.7% margin**, because the same wrong constant
  understates the estimate it compares against.

**What changed.**

1. **The desk's model context went 12,288 → 65,536** (`gemma3:12b-tbv3ctx-64k`, built
   from the saved definition of the old tag with one parameter changed). Measured cost:
   **none worth counting — 8.1 GB loaded, still 100% on the iGPU**, because gemma3's
   sliding-window attention keeps the KV cache cheap. The rollback Modelfile is kept at
   `C:\TradingBotData\_tools\ollama\gemma3-12b-tbv3ctx.BEFORE-2026-08-28.Modelfile`;
   the old tag is untouched, so reverting is one settings change.
2. **The budget is now DERIVED, not remembered.**
   `local_evidence_budget_ceiling_chars()` subtracts generation and scaffold from the
   configured context, converts at the worst measured rate, and leaves retry headroom;
   `local_evidence_budget_chars()` can never return more than that however the setting
   is configured. A budget bigger than the model can read does not produce a bigger
   summary, it produces a silently sheared one — capping here means the packager
   degrades the way it was designed to instead. New setting `ai_local_context_tokens`
   (stock 12,288; the desk is set to 65,536) is what the ceiling is computed from.
3. **Two chars-per-token constants, deliberately different and never to be merged.**
   `_BUDGET_CHARS_PER_TOKEN = 2.0` sizes the budget and is pessimistic (small ratio →
   small budget); `_ESTIMATED_CHARS_PER_TOKEN = 2.5` (was 3.5) estimates what was sent
   and is conservative the other way (large ratio → small estimate → no false alarm).
   A test asserts they lean opposite ways, because merging them reintroduces the shear.
4. **The local request honours its caller's timeout** up to
   `LOCAL_REQUEST_TIMEOUT_CAP_SECONDS = 1800`; the cloud paths keep their 300s clamp. A
   hosted API silent for five minutes has failed; a local 12B is still working — at
   ~118 tok/s evaluating the prompt, the nightly package needs minutes before the first
   output token exists.

**Result, measured on the 2026-08-27 session.** `ai_summary` went from four consecutive
`degraded_no_narrative` runs to **`ok` in 343s**, with **17 of 22 sources usable and
zero unfunded** (it was 10 of 22 with 5 unfunded). The narrative now names real
candidates (NET, OII, NESR), a setup family (`bounce_combo`) and the regime, where the
2026-08-26 one managed "mixed results" and named nothing.

**Then the budget was taken as far as the hardware actually allows** (trader:
"let's take all the time we need... crank up the detail"). Four separate limiters
stack on this path — `MAX_ROWS`, `MAX_SOURCE_CHARS`, the per-scope weights and the
within-scope fair share — and the binding one for almost every source was the share of
the total budget, so the budget is where the work went.

- **96k context loads and then CRASHES under load.** `ollama ps` reported 8.0 GB and
  "100% GPU" at `num_ctx 98304`, and 128k refused outright — but a real 132 KB prompt
  killed the runner with `wsarecv: An existing connection was forcibly closed`. The
  reservation at load time says nothing about what happens when the KV cache actually
  fills. **65,536 is the working ceiling on this iGPU**, established by completing
  generations rather than by loading.
- **An over-long prompt is not an error.** At 64k, a 150,000-char prompt returned
  HTTP 200 and `prompt_tokens = 32,771` — exactly half the window plus three. It
  answered confidently from a prompt it had silently cut in half. That is the whole
  reason the tripwire exists, and it is why the budget is sized with a safety factor
  rather than pushed to the last token.
- **The ceiling formula was wrong a second time and is now measured.** The first
  correction allowed 1,000 tokens for the prompt envelope; the real overhead is
  **10–35%** on top of the evidence the budget counts (measured 24,000→32,203 chars,
  48,000→59,226, 96,000→111,568, 159,466→175,358). `_BUDGET_PROMPT_OVERHEAD = 1.35`
  takes the worst observed ratio. Setting the budget to `0` now means **derive it**,
  so raising the model's window is one setting and not two that can disagree.
- **Verified end to end, not calculated:** the derived 78,119-char budget produces a
  91,262-char prompt that the server tokenized at **44,344 tokens — 71% of the 62,036
  usable**, whole prompt read, no shear.
- **A per-symbol brief no longer shares the session's budget.** It cannot: measured
  ~60s per brief at 22,000 chars, so a normal night is 53 briefs in 55 minutes
  (2026-08-26) or 121 in two hours (2026-08-17), and the job already refuses to start
  with under 120 minutes left. The same package at the session budget is ~42,600 tokens
  instead of ~14,000 — three times the time per brief, which would put a 53-brief night
  past three hours and a 121-brief night past seven. `evidence_budget_for(...,
  per_item=True)` keeps briefs at the value every healthy night ran at, capped by the
  same context ceiling.

**What the extra evidence actually bought.** The 2026-08-27 summary ran in 567s and,
for the first time in any run on record, **"Strongest already-qualified candidates" is
not "No supported finding"** — it reads *"NET, OII, and NESR are highlighted as top long
candidates with high conviction and a 0.83R reward."* Per-source slices roughly doubled:
`daily.auto_report` 2,909 → 4,735 of 8,592 chars, `setups.type_stats` 1 → 3 of 184 rows,
`setups.current_tiers` 4 → 8 of 200, `setups.short_horizon` 5 → 8 of 26.

**Still thin, and this is the honest limit rather than a dial left unturned:** 14 of 17
sources are still shown in part, and the tabular ones are still 3-of-184-row slices. The
prompt is at 71% of a context window that cannot go higher on this hardware, and the
remaining share is divided between seven `setup_trackers` sources competing inside one
scope. Getting whole tables in needs a different model or a narrower scope selection,
not another number.

Tests: 9 new/updated in `tests/test_local_ai_provider.py`. The load-bearing one is
**`test_the_derived_budget_produces_a_prompt_that_fits_the_context`**, which asserts the
invariant this file got wrong twice — across three context sizes, the derived budget's
prompt must fit the window at the pessimistic tokenization rate. Beside it: the budget
cap, the ceiling scaling with context, a configured budget being honoured under the
ceiling, the per-item budget being smaller than the session one and capped by the same
ceiling, the cloud budget being untouched by any local setting, the two chars-per-token
constants leaning opposite ways, and the local timeout surviving where a cloud one would
clamp. The pre-existing derivation test now reads its inputs from the module instead of
re-typing `12288` and `3.0` — which is how it agreed with a wrong number for weeks.


### 2026-08-28 — Two scans wrote one CSV: the D1 feature-history corruption, fixed and repaired

`IMPLEMENTED`, `GREEN`. Trader-authorized (the file-scoped ask-first rule applied;
`master_avwap_lib/legacy.py` houses detector/scoring code). No detector, score, signal
or alert behaviour changed — this is the evidence WRITER for `d1_features_history.csv`
and the repair of the file it damaged, so plan.md sec 5's golden-fixture rule is not
engaged.

**What happened.** On 2026-08-27 the 12:45 swing scan was declared stale at 12:48
(`runner did not survive restart`) and a replacement started at **12:49** while the
first worker was demonstrably still alive. Both wrote the feature history. One
appended at the end; one rewrote it in place from byte 0. The result was a 498 MB CSV
with a **204-column header over a body that is 97.3% 255 columns**, **15 shredded
lines** — two alphabetical symbol streams interleaved into single rows, the leading
fields from one record and the trailing JSON blob from the next — and **372 rows of
real history destroyed** where the short rewrite overwrote the top of the file. From
that moment `export_scan_factor_views` and `export_bot_tier_tracker_views` raised
`ParserError` on **every scan**; both are caught and logged, so the scan went on
reporting success while two of its outputs silently stopped.

**Four rules, now enforced in `append_d1_feature_history`:**

1. **One writer at a time**, through `local_writer_lock` (the machine's real
   cross-process primitive — named mutex plus byte-range lock, released by the kernel
   on a hard kill) keyed by `lock_key_for_path`. Two overlapping scans is not an
   exotic state here: the stale-runner replacement path produces it by design. If no
   lock primitive is available the write is **skipped**, not attempted — without
   exclusion this is exactly the write that caused the damage.
2. **A rewrite is atomic** — temp sibling plus `os.replace`, never an in-place
   `to_csv` over a 498 MB file. That long half-written window is what the other
   process appended into.
3. **An unreadable header refuses the write.** The old `except: existing_columns = []`
   then failed its own truthiness test and fell through to a **blind append**, turning
   a transient read failure into permanent corruption. Losing one run's rows is
   recoverable; an unparseable record is not. The schema-change branch refuses the
   same way and leaves the file exactly as it was.
4. **The append path only ever appends the header's own columns**, so rows cannot
   stop lining up with their header.

**The file was repaired, and the repair recovered more than it lost.** Rebuilt from
the 2026-08-26 evidence snapshot (119,107 rows, uniformly 255 columns, verified clean)
plus every recoverable 2026-08-27 row from the live file, mapped onto the wide header
**by name** — the 204-column schema is a strict subset of the 255, so the narrow rows
carry their 204 real values and 51 blanks rather than being dropped. Result:
**129,081 rows, uniformly 255 columns, `pd.read_csv` clean**, against 128,720 in the
corrupt file — a **net gain of 361 rows**, because the snapshot restores what the
overwrite destroyed. All ten of 2026-08-27's runs are present; the 12:49 run survives
with 200 of its ~1,086 rows. **15 rows were unrecoverable** and are written to
`d1_features_history.quarantine-2026-08-28.jsonl` beside the file with their line
number, width, run_id and symbol — counted and named, never silently dropped. The
corrupt original is kept as `d1_features_history.csv.corrupt-2026-08-28`.

Tests: 5 new in `tests/test_master_avwap_setups.py`. Three are true reproductions and
**fail against the old writer** (proven by reverting it): the unreadable-header refusal,
the lock acquisition, and no-lock-means-no-write. Two are standing guards — the atomic
rewrite leaves no temp file, and every written row matches the header width.


### 2026-08-28 — The nightly narration: one bad citation no longer costs the night

`IMPLEMENTED`, `GREEN`. Three fixes to the local-AI layer, from a trader question
about file sizes that turned into an audit of what the overnight run actually
produced. The AI summary was healthy — a real narrative on 2026-08-26 and on
nearly every night since 08-11 — but the **daily digest had failed three nights
running (08-25, -26, -27)** while the model and every store were up, and the night
of 08-27 lost all three narrating jobs to a dead inference server.

**1. The local model server had no autostart.** Ollama is a user-session tray app
started by hand. Its log stops at 06:12 on 2026-08-27; the desk restarted around
13:00 and nothing brought it back, so `ai_summary`, `ticker_briefs` and
`daily_digest` spent the entire 22:00–06:00 window retrying against a refused
connection on `127.0.0.1:11434`. The deterministic slots (`veto_cohort_grading`,
`like_cohort_grading`, `evidence_report`, `journal_import`) were unaffected, which
is why it went unnoticed until the summaries were read. An `HKCU\...\Run` entry now
starts it at logon, and `scripts/run_ai_jobs.ps1` gained a **preflight**: probe the
configured endpoint, start `ollama serve` if it is a LOCAL endpoint that is down,
wait up to 60s for the socket, and **carry on either way**. It never refuses the
run — `degraded_no_narrative` is a designed state, the fact packs and the counting
jobs need no model, and a preflight that could block the night would be worse than
the problem it fixes. A remote endpoint is left alone; it belongs to whoever runs it.

**2. An unsupported citation now costs its ROW, not the document**
(`validate_ai_summary`, trader decision). One bad `evidence_refs` entry used to
raise, discarding every supported statement beside it; with two model attempts and
a three-attempt session cap, one predictable 12B slip cost a whole night. Nothing
is loosened about what may be PUBLISHED — an invalid ref is still struck out, a row
left citing nothing is still discarded — but what was dropped is now recorded
through a `dropped` sink, carried on the result as `citation_drops`, and disclosed
as a `[system]` row in the published `data_quality` section, because a document
quietly missing two of its four findings reads exactly like a thin evidence night.
If EVERY citing row is dropped the document still raises: a summary supported by
nothing is not a degraded summary. Shape and value errors still raise, unchanged —
a malformed document is the provider failing to answer, a different fault from a
model that answered and mis-attributed one line.

**3. The digest's own fact pack was the thing making it fail.** The narrator is
instructed to cite exact `source_id` values and is handed a document in which every
measured cell PRINTS one — `outcomes.intraday_finals`, `review.alert_review_events`,
`ops.ai_job_ledger` — while the validator knew only `digest.facts`. The model cited
what it was shown; that is the instruction working, not a hallucination. Packages
may now carry `citable_aliases`, and `digest.provenance_ids` **walks the built pack**
to collect them (listed by hand, a block added later would be shown and yet
forbidden). `usable_source_ids` honours aliases only when the package already has a
usable source, so an alias can never conjure citability out of an empty package.

**4. Fact pack v2 — the pointer hoist.** `daily_digest_facts_v1` → `v2`. The
2026-08-27 pack rendered at **14,070 bytes against an 8,192-byte target**, 72% of it
the outcomes block, and one `source_id` plus one `as_of` were printed 21 times for
21 cells that all shared them. `_hoist_block_pointer` lifts those two fields to
their block when *every* measured cell agrees (a block that mixes two stores keeps
them per cell, because then hoisting would state something false), and the slice
rows trade fourteen near-identical selectors for one `selector_template` that
rebuilds any row's exact selector from its own `env_key` and `side`. Measured:
**14,070 → 11,124 bytes, 21% smaller, not one figure dropped** and D2 untouched —
every value still carries its `n`. It is a new schema NAME because the SHAPE
changed; v1 packs on disk stay v1 and stay readable, and `clean_digest_sessions`
counts by session rather than by schema, so the Phase 2 collection window is
unaffected. The old sizing comment claimed 16 slices land "near the 8 KB target";
it is replaced with the measured number and the note that the figure the target
exists to protect — ninety packs as a trivial reducer context load — holds
comfortably at the post-hoist size. Cutting real slices to reach 8,192 exactly
would trade evidence for a round number.

Also fixed on the way past: `test_group_tape_service` pinned its fixture bars to a
hardcoded `2026-08-27` while the service filters to TODAY's date, so from 2026-08-28
its two sector-output assertions failed on the calendar rather than on the code
(proven pre-existing by re-running with this packet stashed). A `frozen_session_clock`
fixture freezes the service module's clock beside the bars; no production change, and
the same-date filter it exercises is correct.

Tests: 11 new (`tests/test_ai_summary.py`, `tests/test_ai_digest.py`), covering the
row-drop, the kept-good-refs case, the all-dropped raise, the `[system]` disclosure,
alias citability and its empty-package guard, the pack walk, the mixed-store
non-hoist, and exact selector reconstruction. `test_validation_rejects_hallucinated_
evidence_reference` was updated to the new contract; it is the regression pin for
the old behaviour and it failed before this change.

**Also found, NOT fixed — needs a trader decision.** `d1_features_history.csv`
(498 MB) went ragged at 12:49 on 2026-08-27: a 204-column header over rows of 119
to 524 columns. `export_scan_factor_views` and `export_bot_tier_tracker_views` now
raise `ParserError` on **every scan** (caught and logged; the scan continues, those
two outputs do not). The widening path in `master_avwap_lib/legacy.py:2397-2408`
reads the whole 498 MB file with pandas when the column set changes, and its
`except: existing_columns = []` degrades a read failure into a blind append. That
file houses detector/scoring code, so the file-scoped ask-first rule applies.


### 2026-08-27 (night) — repo hygiene: dead code, a dead dependency, stale doc claims

`IMPLEMENTED`, no runtime behavior change; frozen rebuild owed. A codebase-wide
assessment for dead code, duplication and documentation drift.

Removed: `ui/widgets/info_dot.py`, `ui/widgets/symbol_chip.py` and
`ui/models/journal_table_model.py` (236 lines) — zero references in Python, the
PyInstaller spec, JSON or Markdown, and `ui/` performs no dynamic module lookup. The
journal model/proxy pair was superseded by `panels/journal/trades_tab.py`, which builds
a `QTableWidget` over `JournalTrade` directly.

Dependencies: `scikit-learn` and `joblib` dropped from `requirements-core.txt`, and
`packaging/tradingbotv3.spec` no longer force-collects sklearn/scipy submodules. Nothing
has imported either since `a73f072` removed the trade-quality training script; the
collection was pulling ~93 MB into every bundle, and because
`collect_submodules("sklearn")` was unguarded it would have failed the build outright
once the dependency was dropped. `ruff` added to `requirements-dev.txt`: `pyproject.toml`
configures it and `CLAUDE.md` names it in the stack, but it was declared nowhere, so a
clean dev install could not run the configured lint.

Documentation corrections: the frozen-selftest expectation of `29/29` in `CLAUDE.md` and
`README.md` was stale — the count is a running total (29 on 2026-08-09, 30 later, 72
unfrozen today), so an agent comparing a correct run against 29 would read it as a
failure; both now say N/N and direct the reader to a current unfrozen run. `README.md`
also described the Desk Link code as "unused pending a cleanup packet" and
`master_avwap_mini_pc.py` as still present — both were removed 2026-08-24 — and
introduced five documentation entry points as "four"; the source-launch-is-production
decision and the one-desk-per-machine guard were added. `docs/README.md` now links the
15 decision records instead of listing them by title.

Documentation size (trader-approved the same night, "make it as easy as possible to
keep vibe coding"): `CURRENT_CHECKPOINT.md` had reached 7,901 lines / 449 KB across 113
dated entries and was a second changelog rather than a checkpoint, which made the
mandatory documentation read in `CLAUDE.md` about **260k tokens before any change could
be proposed** — an instruction no agent can follow, so it skims and appends, which is
what grew the file. Entries dated 2026-08-25 and earlier moved verbatim to
[`docs/CHECKPOINT_ARCHIVE_2026-08.md`](docs/CHECKPOINT_ARCHIVE_2026-08.md) (95 entries)
and the revision history from 2026-08-19 back to 2025-11 moved to
[`docs/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md`](docs/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md)
(36 entries); both are classified in `docs/README.md` as historical evidence that must
never be loaded as context. `CURRENT_CHECKPOINT.md` (now 1,431 lines / 82 KB) opens with
an **"Active state at a glance"** block carrying the branch, active roadmap items, last
verified baseline, the eight open live gates and the next action, and states that a
dated entry wins if the two disagree. `CLAUDE.md`'s mandatory workflow became a bounded
read — checkpoint glance block first, `plan.md` §5–7 plus the active phase, a *search*
of the changelog inventory rather than a full read — plus a standing rule to refresh the
glance block, keep entries short, and archive past ~1,500 lines. `CHANGELOG.md` is
unchanged as the authority on what exists; only its closed history moved.

Second pass, 2026-08-28 (trader: "can we summarize things to be even briefer?"):
`CHANGELOG.md`'s `Current implemented inventory` was 94% narrative — 3,808 of 4,061 lines
were dated entries wrapped around a **253-line thematic inventory that already states
what exists**. That inventory is the contract and was promoted to the top of the file;
the 73 dated entries older than 2026-08-26 moved verbatim to the archive and the 18 from
the last two build days remain under `Recent changes` (260 KB → 98 KB). `CLAUDE.md`'s
`Core loop / data flow` section — 42 KB, 65% of a file that loads into *every* session —
had each rule carrying the incident, measurements and trader conversation that produced
it; those moved verbatim to **[`docs/DESK_INTERNALS.md`](docs/DESK_INTERNALS.md)** while
`CLAUDE.md` keeps every rule as a binding imperative with a pointer (section 71% smaller,
file 65 KB → 35 KB). A check for 45 critical guardrail tokens confirmed all 45 survive in
`CLAUDE.md` itself; the rules bind from `CLAUDE.md` alone and both files change together.
The mandatory documentation read fell from **~260k tokens to ~97.5k (63% smaller)**.
`plan.md` was deliberately left untouched.

Third pass, 2026-08-28: `plan.md` narrowed **149 KB → 76 KB** (37,305 → 19,141 tokens).
Section 12 was 93% of the file and its Phases 0.5/0.6/0.7 were 72% of that section while
describing work already BUILT. Each of their 89 numbered items keeps its title/status
line, a spec and build-record pointer, and every gate reduced to its bold lead plus the
sentences carrying the gate — **verbatim, never paraphrased**; the build narrative moved
to [`docs/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md`](docs/ROADMAP_ARCHIVE_PHASES_0.5-0.7.md).
A first attempt was reverted because splitting sentences on `;` truncated multi-part gate
lists (R1 lost two of its three owed proofs); the redone pass splits only on a period
before a capital, and a clause-level check reports 89 gate clauses before and 90 after,
with the 5 flagged items confirmed present by name in their bold titles. Structure is
unchanged at 10 sections and 89 items. The three dead UI modules removed in the first
pass are confirmed gone from disk and from HEAD, their stale bytecode deleted.

The mandatory documentation read across `CLAUDE.md`, `CURRENT_CHECKPOINT.md`,
`CHANGELOG.md`, `plan.md` and `docs/README.md` is now **~83,607 tokens, down from
~259,878 — 68% smaller** — with no rule, gate, or inventory statement lost.

Packaging gate MET 2026-08-28: rebuilt at `fff07b8` and the frozen self-test reports
`selftest OK: 72/72 checks passed (frozen)`, exit 0 — the frozen count equals the
unfrozen count, confirming the retired "29/29" expectation was stale by 43 checks. The
bundle fell **442 MB → 419 MB** with `sklearn` gone. `scipy` (79 MB) remains, pulled
transitively by a lazy import in `yfinance`'s price-repair path and by pandas sparse
arrays; neither is reachable from this codebase, so excluding it would reach roughly
340 MB — recorded as a recommendation, not done, because it needs its own rebuild plus a
live bar fetch to prove the yfinance path.

Verified after the changes: `pytest tests/ -q` **5261 passed, 22 subtests, exit 0**;
`smoke_check.py` **7/7**; `launch_gui.py --selftest` **72/72**; packaging spec-drift and
selftest suites **24 passed**. The spec edit is a packaging trigger, so a rebuild and a
frozen selftest are owed before the next merge to `main`.


### 2026-08-27 - Every tracker family enters bounded stop/target research

**IMPLEMENTED / GREEN; shadow accumulation and live canary owed.** The trader
authorized three linked research choices: evaluate every markable tracker
family, treat the next regular session's first completed M5 close as the entry
for a D1 setup found after close, and attach Auto Market Bias separately on
M5/M30/H1/H4/D1.

`research_warehouse/tracker_adapter.py` now streams the scenario CSV and reads
the small transition ledger to create warehouse occurrences. It deliberately
never opens the 1 GB tracker snapshot. Daily rescans collapse by
symbol/side/family/anchor and keep first-seen geometry, preventing a later scan
from leaking future information. A read-only real-data audit covered all 16
registered families: 249,438 scenario rows plus 10,820 transition rows became
6,663 deduplicated detections with zero unknown-family skips.

`outcomes.M5_CLOSE_RECIPES` is a separate 54-recipe discovery grid, leaving the
frozen slice recipes unchanged. Structural stop-source ranks 1–3 and ATR stops
0.5/1.0/1.5 are each crossed with 1R/2R/3R targets. The engine uses M5 only,
STOP_FIRST ambiguity and the existing fallback cost model; it needs no
trader-planned stop/risk and reads no bid/ask or earnings fundamentals.

`setup_market_context` stores five point-in-time champion bias readings for each
entry. The complete live decision is now one pure helper, including its
early-session day-percent fallback; the two live callers were refactored onto
it with no behavior change, and research calls that same helper. M30/H1/H4
derive from completed SPY M5 bars, D1 sees only prior complete daily bars, and
truly absent input stays `unknown`. The outcome build uses stable symbol
buckets plus Arrow-side occurrence/recipe filters so the expanded grid cannot
rematerialize the whole research history inside the desk.

The appended nightly `setup_research` slot always writes deterministic JSON and
Markdown. Medium local AI may only explain the bounded facts after a cell has
n>=30, five symbols and five entry sessions; below that floor it is not called.
No output reaches a detector, score, alert, Focus, watchlist, stop, target or
order. The durable M5 archive currently begins in August 2026, so older tracker
episodes honestly remain uncovered until backfill exists.

### 2026-08-27 - The Market Journal loads, carries the tape, and reaches the nightly AI

**IMPLEMENTED / GREEN; live gate owed.** Trader, after a full session of
in-session notes: "this is empty and feels very useless to me. this should
capture more stuff, such as SPY charts, what they looked like when the auto
mode flipped, my entries, what the charts looked like when i inputted entries,
what the D1 looked like.. i also expect the AI to get access to these notes for
the daily summary function."

Five entries were on disk for 2026-08-27 and the page showed none of them. Two
defects sat behind that, and the rest of the report was a missing feature.

**The page never loaded.** `MarketJournalPanel.reload()` had no caller at all -
not in `__init__`, not on show, and `_select_page` only special-cases the AWAY
Recap. The page was blank until "Refresh" was pressed, which reads as an empty
journal. It now loads the first time it is shown, and only then: the desk builds
every left-nav panel at startup and most are never opened.

**There were two services, not one.** `_build_journal_tab` constructed its own
`MarketJournalService`, so the desk tab's `entryWritten` was emitted by an
object the left-nav page had never heard of. Both wrote the same file correctly
(the ledger append is atomic per line) - what was lost was the refresh. One
process-wide `shared_journal_service()` now backs both surfaces, which is what
the R10.H docstring had claimed since it was written.

**Every entry now stores the tape it was written against** -
`scripts/market_journal_capture.py`, new. Bars, never pictures: a PNG cannot be
re-ranged, measured, or read by the nightly AI layer. A capture holds the
symbol's M5 and D1 and SPY's M5 and D1 as they stood at the moment of the note,
in two stores on purpose - a **sidecar** JSON per capture for the bar windows
(tens of KB, only the page reads it) and a **ledger row**
(`market_journal_chart_v1`, stream `market_journal_charts`) carrying a short
text `digest`: where price sat against its session range, session VWAP, the
prior session's extremes, the 20/50/200-day averages and RVOL. The raw window
would starve every other source in an AI packet; the digest says the same thing
in a few hundred characters.

`market_journal_entry_v1` is **untouched** - a capture joins by `entry_id` from
the outside, which is what lets it be written AFTER the entry, on a worker,
without a note ever waiting on a chart. A capture that fails leaves an entry
that is honestly chartless; an entry that was never written is a lost thought,
and those are not the same cost. Every bar list is a CACHE read
(`AlertCenterPanel.journal_chart_bars`, new and public) - nothing fetches.

**Auto-mode flips write their own row.** `AutopilotService.autoModeChanged`
(previous, current) fires only when `auto_mode` actually moves - a profile
change while Auto is OFF is not a flip - and `MainWindow._record_auto_mode_flip`
writes a Market Journal entry with SPY's M5 and D1 attached. The row carries
`ORIGIN_AUTO_MODE_FLIP` and `market_journal.is_machine_entry` reads it back, so
the page can mark it `[desk]`: the journal is one timeline, and a reader
counting "what did you think?" must never count a sentence nobody thought.

**The page draws the capture.** Selecting an entry loads its sidecar on a worker
and draws up to four panes; a pane with nothing stored is HIDDEN rather than
drawn empty, and a stored bar whose stamp will not parse is dropped, counted and
named (the axis formats every stamp with `strftime`, so one string takes the
chart down rather than degrading it).

**The nightly AI reads the journal now.** `market_journal` joins
`briefs.DEFAULT_SCOPES` on the trader's explicit instruction, reversing the
R10.I opt-in - which was itself a recorded trader decision, and the same trader
is the only thing that could reverse it. Its sources keep the funding rule
(distilled first, free text last): evidence report, day context, **chart
digests**, then the entries. `TICKER_BRIEF_SCOPES` stops being an alias for
`DEFAULT_SCOPES` and keeps the original four - a session-level journal entry in
a per-symbol packet is the TB-0/TB-5 failure mode.

Two pinned tests changed rather than being worked around: both asserted the
opt-in that the trader has now reversed, and they now pin the new decision.

### 2026-08-27 - Double-click on a claim commits the like, the way it does on a veto

**IMPLEMENTED / GREEN.** Trader: "i want to be able to double click the like and
claim the same way i can double click the veto."

The two gestures had drifted apart. `select_reason` (veto digit) and the reason
list's `itemActivated` (veto double-click) both call `commit_veto`, which
diverts to the note field only when that reason's `note_required` is unmet. The
like's `select_setup` and `_claim_picked` went straight to `_prompt_for_why`
and could never commit - so a trader who had ALREADY typed the why was sent
back to a field they had just filled in.

- Both like gestures now call `commit_like`, which is where R9.2's required-why
  guard already lives. That is the veto's exact shape: the gesture ATTEMPTS the
  commit; the rule enforces itself inside the commit rather than by refusing to
  reach it.
- **The 2026-08-22 rule is untouched** ("if I like a chart I should always be
  prompted with why"). A like with no why still writes nothing, still holds the
  chart, and still moves focus to the why with the same message - its two
  existing tests pass unchanged, and a new one pins the double-click case of
  it. The only new capability is: why typed, then the gesture commits.
- The digit changed with the double-click deliberately. The veto's digit and
  double-click are identical to each other, and leaving the like's digit
  nagging while its double-click committed would make the rail internally
  inconsistent in a way the veto is not.
- The stale docstring that claimed "double-click and Enter commit it exactly as
  they do a veto" is now true, and says what "exactly as a veto" means.

Nothing else in the rail moves: the LIKE still retires the chart the way it
did, still writes only `trader_annotations.jsonl` + the like cohort, and still
adds nothing to Focus or any watchlist.

Tests: `tests/test_qt_alert_capture.py` +5 (59 in the file) - the double-click
committing with a why, the double-click still refusing without one, the digit
committing with a why, the why field cleared after a commit so the next chart
cannot silently inherit the previous chart's reasoning, and both lists'
activation routed through their commit. Fail-before-fix: 4 of the 5 fail on the
old wiring; the fifth is the no-why regression guard, which must pass on both
sides. Full suite **5203 passed, 19 subtests, exit 0**.

### 2026-08-27 - The ticker popup opens 10% short of the screen, top and bottom

**IMPLEMENTED / GREEN.** Trader: "make the charts that pop up when i click on a
ticker just a little less tall. i dont want them edge to edge on the screen just
reduce by 10% top and bottom."

This is the 2026-08-11 sizing ask corrected, not reverted. That one fixed the
opposite problem - the popup opened at a fixed 1180x760 whatever the monitor,
squeezing both charts into about half the vertical space - by taking the height
from the hosting desk window, or the screen's available area, minus a 60px/40px
title-bar allowance. On this desk's monitors that is edge to edge.

- `symbol_snapshot_dialog.inset_vertical_bounds(anchor_top, anchor_height)` is
  a new PURE helper: it leaves `POPUP_VERTICAL_INSET` (0.10) of the anchor free
  at the top and again at the bottom, so the popup opens at 80% of whatever it
  is anchored to. The gaps come from CENTRING the final height inside the
  anchor rather than from adding the inset to the top, so they stay equal even
  when the floor below overrides the inset.
- `POPUP_MIN_HEIGHT` (760) is a floor the inset cannot go under - both charts
  carry a 120px minimum and a squeezed popup is exactly what 2026-08-11 fixed -
  and it never pushes the popup off the top of the screen to honour itself.
- The anchor is chosen exactly as before (hosting window frame if visible, else
  the screen's available area). The proportional inset replaces the old 60/40
  allowances, which it dwarfs.
- Measured on the desk's three monitors: 4K panels go 2052 -> **1690 px** with
  211px free at each end; the 2560x1392 goes 1332 -> **1114 px** with 139px at
  each end. 82-84% of the old height.

One dialog class and one factory (`show_symbol_snapshot`) serve every ticker
click - Alert Center, Industry, Master AVWAP, and through them the Strength
Board and the group tape - so this is one change for all of them. It sets only
the OPENING size; a trader resize afterwards is still kept, because the dialog
is created once per owner panel and reused.

Tests: `tests/test_snapshot_popup_height.py` (6) - the tenth at each end, equal
gaps across four anchor sizes, the constant, the floor winning on a short
screen, the floor not pushing the popup off-screen, and the dialog actually
routing through the shared helper. Fail-before-fix: 6/6. Full suite **5198
passed, 19 subtests, exit 0**.

### 2026-08-27 - The desk's 8-13 GB memory jumps: three causes, all fixed

**IMPLEMENTED / GREEN. One live gate owed.** plan.md Phase 0.9 item 6, built to
`docs/analysis/OPUS_BUILD_PROMPT_DESK_MEMORY_2026-08-27.md` on the 2026-08-27
(10:00) investigation. The trader: "there are times the program jumps to 10gb
of RAM usage."

**Cause 1 - the post-scan warehouse build materialised a whole MONTH of M5 bars
to use one session.** Three steps did
`store.read_table("bar_m5", "month=YYYY-MM").to_pylist()` and then filtered in
Python. Partitions are month-keyed, so the cost grew all month. Measured on the
live lake this afternoon: `silver/bar_m5/month=2026-08` = **8,704,108 rows /
408 MB parquet / 158 files**, and `to_pylist` costs **1,769 B/row = 15.4 GB**
held whole - against a largest single session of 588,778 rows, 6.8% of it.
(The 10:00 investigation measured 8,175,471 rows / 13.3 GB; the month has grown
since, which is the point.)

- New `ResearchStore.read_rows(dataset, partition, *, columns, symbols,
  interval_start_range)` pushes the predicate into
  `Dataset.to_table(filter=...)`, so Arrow drops rows before any Python object
  exists. Deliberately NOT a free-form filter argument: only the two predicates
  the callers replaced are offered, so nobody can express something subtly
  different from the Python test it stands in for. `symbols` matches exactly,
  no case folding, because `symbol in wanted` did.
- `aggregate.build_derived_bars` and `features.build_intraday_snapshots` narrow
  to the session window (and to named symbols); `cli._run_outcomes` narrows by
  SYMBOL ONLY - deliberately no date filter, because the outcome walk runs
  forward over a horizon that crosses sessions, which is why
  `_m5_partitions_for` already widens to the trigger's month plus the next
  (BD-66/BD-69). `build_intraday_snapshots` applies the symbol filter only when
  symbols were named, because otherwise its cohort is derived from the bars
  present in the session.
- **Measured after, same lake:** a full session read **0.53 GB** (297,230
  rows), a 20-symbol outcome read **0.31 GB** (175,235 rows). 15.4 GB -> 0.53
  GB, ~29x.
- Equivalence is asserted against a longhand REFERENCE implementation of the
  old read (read the month, filter in Python) and compared as published ROWS,
  not counts - a filter that shifted a session boundary by one bar would keep
  the count and change the answer.

**Cause 2 - the 1.03 GB tracker snapshot was read whole to decide it was
unchanged.** `master_avwap_setup_tracker.json` measured **1,026,057,028 bytes**.
`ingest_artifact` did `read_bytes()` and hashed the bytes BEFORE consulting the
watermark, so every bronze ingest allocated 1.03 GB - including the ones that
immediately answered UNCHANGED - and a changed file then ran `json.loads` over
the decoded text, several GB more.

- `_sha256_path` hashes in 1 MB chunks and the UNCHANGED check is hoisted above
  `read_bytes`, so an unchanged snapshot now costs no allocation at all.
- A SNAPSHOT over `SNAPSHOT_PARSE_MAX_BYTES` (64 MB) is stored in FULL but not
  parsed; `_looks_like_json` (first/last non-space characters) drives the
  quality flag instead. **This loses nothing measurable for the artifact that
  triggers it:** `setup_tracker` declares neither `event_keys` nor `id_keys`,
  so `_parse_event_at` returns None on its first line and `_first_value`
  returns "" without reading the payload - parsed or not. The parse influenced
  exactly one column, and a test asserts the parsed and skipped rows come out
  identical. Residual stated rather than hidden: a CHANGED snapshot still costs
  ~size bytes plus a same-size `str`, because `payload_text` must be a string
  for the publish path, which this packet did not touch.
- BD-73 records the threshold and its reopen trigger: if `setup_tracker` ever
  gains those key tuples the skip WOULD empty real columns, and a fixture
  assertion fails loudly rather than silently.

**Cause 3 - BounceBot never freed its IB bar buffers.** `self.data[reqId]` held
every historical reply; only the RRS and contract-bars paths popped it. Five
others (`build_atr_cache`, `request_and_detect_bounce`, and the three
`check_*vwap*_touches`) deleted the ready event and left the bars: **206 KB per
390-bar request, ~400 requests a scan cycle, 1.5-2 GB over a session**, held
until the process exits. That is why the desk settled at 2.5 GB rather than 1
GB once a build released.

- All five now free the buffer with the event, on the success AND timeout
  branches. `request_and_detect_bounce` - the hottest path - pops at the read.
- `historicalData` no longer auto-creates a buffer for an unknown reqId. Every
  request path creates its buffer before issuing the request, so an unknown
  reqId can only be a straggler; auto-creating one meant a timed-out request
  leaked AFTER the fact, and a bar racing the requester's own pop appended to
  the very list the caller was reading. Both are closed.
- **The trader authorised this one `legacy.py` edit and nothing else in that
  file.** It was verified LIKE a detector change even though it is not one: a
  repo-wide sweep confirmed each reqId is read exactly once, by the function
  that created it, with `self.data` never iterated, persisted or touched
  outside the class - and the golden fixtures plus all 411 BounceBot tests pass
  unchanged.

**Premise corrected while building** (reproduce, do not inherit): the build
prompt listed `cli._run_outcomes` as one of the three live costs. It is not one
today - `setup_occurrence` holds **0 rows** on this lake, so `_run_outcomes`
returns `NO_OCCURRENCES` before it ever reads `bar_m5`. It was fixed anyway,
because it becomes a cost the moment the BD-44 detector adapter lands.

**No packaging trigger** (no new dependency, asset, top-level package, dynamic
import or `__file__` use).

Tests: `tests/test_warehouse_session_scoped_reads.py` 10,
`tests/test_bronze_snapshot_large_files.py` 9,
`tests/test_bouncebot_reqid_buffers_are_freed.py` 12. Fail-before-fix per file:
8/10 (two are the equivalence guards, which must pass on both sides), 9/9, and
11/12 (the survivor guards that a live request still collects its bars). Full
suite **5192 passed, 19 subtests passed, exit 0**; smoke 7/7.

### 2026-08-27 - Group RS/RW tape rebuilt: its own five-minute clock, 90 | 60 | 30 minutes, today's bars only

**IMPLEMENTED / GREEN. Live gate owed** (one DESK session). plan.md Phase 0.5
item 11, built through `docs/prompts/GROUP_TAPE_REBUILD_OPUS_PROMPT.md`
(packets T-1..T-4), which the trader authorized the same morning after deciding
to hide the old tape rather than delete it.

The complaint was "often times the sectors and industry RS/RW thing at the top
is totally wrong and doesn't reflect what is actually strong over the last
30-60-90 minutes". The investigation found the maths RIGHT and the clock wrong,
so the formula is lifted out unchanged and given a new clock, a new source and
three windows.

- `scripts/group_rrs.py` (pure: bars in, floats out, no I/O, no Qt, `now`
  always passed in). `wilder_atr_last` + `real_relative_strength` reproduce
  `legacy`'s, including the two details a re-derivation gets wrong - the ATR
  seeds on the first `length` true ranges and smooths over ALL the rest, so it
  depends on the whole series and not its tail, and a non-positive ATR is
  `None` rather than 0, which is what stops the division producing an infinity.
  `session_bars` = `completed_bars.completed_m5_bars` AND a same-date filter,
  which is what stops a window reaching over the overnight gap; `align_bars`
  intersects the two series on normalized stamps, so an ETF that halted for a
  bar cannot have its move measured over a longer span than SPY's and read as
  strength, and an ET-stamped ETF still meets a UTC-stamped SPY. `rrs_windows`
  = 6/12/18 bars = 30/60/90 minutes off ONE filtered+aligned series, so the
  three numbers are guaranteed to describe the same bars. A window without
  `length + 2` bars is `None`. `SECTOR_ETFS` is a COPY of `legacy`'s map, not
  an import - the tape must survive BounceBot being off and must not drag a
  14k-line detector module onto a worker thread - and a drift test pins the
  copy.
- `scripts/ui/services/group_tape_service.py`, the Strength Board's shape: one
  `QTimer`, single-flight worker, last-good on failure, `status_text`, bounded
  `shutdown`. **ONE batched `yfinance` download per tick** (SPY + the 11 SPDRs
  + the 49 industry proxies, deduped to ~53 symbols), `period=1d interval=5m`,
  **no retry inside the tick** - Yahoo rate-limits bursts, and the next tick is
  the retry. **Zero IB traffic and no `legacy.py` change**, so the locked
  pacing budget is untouched. Quiet-hours gated on `auto_scanning_due`,
  fail-open; `refresh_now` never gated. A missing or unreadable industry map
  means SECTORS ONLY, said in the status line - two thirds of the chips
  disappearing silently would read as "nothing is moving". No completed SPY
  bars for today is said out loud rather than rendered as an empty strip.
- `GroupTapeStrip`: `SPARK_TIMEFRAMES = ("90", "60", "30")`, ranked by the 30.
  An unmeasured window draws NOTHING - a zero-height bar on the zero line is
  indistinguishable from "exactly in line with SPY", which is a claim - and the
  tooltip names which windows are still filling. `rotation_callout` is now "up
  on 30 while still down on 90" and its mirror, and the callout line carries
  the payload's as-of plus the service's `status_text`, so a stale or failed
  read is visible rather than silent. **Chips diff**, keyed by ETF: reused,
  re-labelled and re-ordered instead of destroyed and re-created, and the
  variants moved from a per-chip f-string `setStyleSheet` into `theme.qss`
  keyed on a `side` dynamic property with six pre-mixed rgba tokens in
  `theme._derived_tokens`. The old path was 34 CSS parses plus 34 widget
  constructions every payload, on the GUI thread - the exact shape the
  2026-08-21 fluidity pass measured. `GroupChip` sets `WA_StyledBackground`,
  which a widget carrying its own stylesheet got for free.
- `TradingDeskPanel`: the tape is VISIBLE again and fed by
  `tapeChanged`/`statusChanged`; the `rrsSnapshotChanged -> update_groups`
  wiring is gone. **The RS Window tab and `focus_picks_panel` still receive
  `rrsSnapshotChanged` unchanged** - it answers a different question (who led
  over the selected window at scan time) - and a test pins that both wirings
  coexist. The service is shut down in the desk's shutdown list, and that list
  now resolves it the way it already resolved `price_alert_service`: naming it
  inline made a missing attribute raise while the component list was being
  BUILT, before the fan-out loop ran, so a desk whose `__init__` died partway
  would have released nothing instead of one thing.

**Deliberately NOT built** (from the prompt's own "not in this prompt"):
industry = median member return instead of the ETF proxy (needs member bars -
an IB-budget question), any change to the 27-minute scan cycle, and anything in
`legacy.py`.

**No packaging trigger**: `scripts/group_rrs.py` and the new service are
ordinary static imports on a chain reachable from `launch_gui.py`, so
PyInstaller collects them by dependency analysis - no new dependency, asset,
top-level package or dynamic import. The spec-drift guard passes.

Two failures were found on the way and fixed; **neither was caused by this
work**. `test_review_watch_buttons_arm_trigger_and_flag_red` was a CLOCK BOMB:
its fixture's last bar starts at 11:25, so before 11:30 local that bar was
still forming, the 2026-08-27 VWAP-side leg read UNKNOWN and the chart showed -
after 11:30 both bars complete, the fixture's long sits under its own session
VWAP and the filter correctly hid it. It passed at 10:xx and failed at 11:36 on
the same tree. The production behaviour is right; the test is about the watch
buttons, so it now switches the show-time filter off the way five sibling files
already do. `test_trading_desk_shutdown_continues_after_one_component_raises`
needed the new component on its `SimpleNamespace` desk.

Tests: `tests/test_group_rrs.py` 16, `tests/test_group_tape_service.py` 16,
`tests/test_qt_group_tape.py` rewritten to 17, plus one new partial-desk
shutdown test. Fail-before-fix shown per file: 16/16, 16/16, and 15/17 (the two
survivors are the deliberate regression guards - the silent callout, and the RS
Window tab still receiving `rrsSnapshotChanged`). Full suite **5161 passed, 19
subtests, exit 0** (305 s); smoke 7/7.

### 2026-08-27 - Clicking away from an M5 chart is a skip, not a re-queue (trader rule 4, third pass)

**IMPLEMENTED / GREEN.** Trader: "When I click on an alert in the new M5 alert
bar and then click to another one, it shouldn't queue the old M5 alert in the
waiting list. It should just be considered a 'skip for now' situation." The
bar took M5 alerts OUT of the waiting list, but `_select_review_alert` - the
shared feed-row/bar click path - still pushed whatever chart it replaced to
the HEAD of that list, so a trader working down the bar refilled the D1 queue
with the M5 rows the bar was built to keep out of it.

- `AlertCenterPanel._current_review_holds_place` (new, defaults `True`)
  records where the chart in front CAME FROM, which is the thing the decision
  actually turns on. `_advance_review_queue` sets it `True` (that chart was
  popped off the waiting list and keeps its place); `_select_review_alert`
  sets it to `not _is_m5_review_alert(alert)` (a clicked D1 row / armed hit
  holds a place, an M5 bar row does not). On the next click, a chart that
  holds a place is re-inserted at the head exactly as before and one that
  does not is skipped.
- Why a flag rather than re-testing the outgoing alert: the refresh path
  (`_enqueue_review_alert`, same-symbol branch) REPLACES a queued D1 chart's
  alert object with that symbol's newer M5 alert, so `_is_m5_review_alert`
  asked about the outgoing object would answer "M5" for a chart that really
  is holding a D1 queue slot, and clicking away would silently drop it.
  Pinned by `test_a_refreshed_d1_chart_still_holds_its_place`.
- The skip is RECORDED, not silent: `_record_review_event("skip", ...)` with
  the dwell and `detail={"reason": "clicked_away_from_m5_alert"}`. The
  impression was already written - `_render_current_review` emits `shown` for
  a bar-clicked chart like any other - and `shown` is the denominator for
  P(take | shown), so leaving the click-away unanswered would have stranded
  an impression with no verb and biased the rate. `skip` is that stream's
  existing definition of "looked at the chart and passed"
  (`scripts/review_events.py`), which is the trader's own phrase. No status
  line: the replacement chart is already up, so a message would be noise.
- Unchanged, deliberately: the routing at `_enqueue_review_alert` still
  records nothing, the M5 bar is still not a queue, the feed and History
  still keep every clicked-away row, and no parking happens here (that stays
  specific to Skip-after-arming-a-D1 in `_skip_review_alert`).

Tests: `tests/test_qt_m5_alert_bar.py` +3 (22 total, whole suite 5119 passed):
the second bar click skips rather than queues and writes exactly one `skip`
with its reason; a queued D1 chart still returns to the head of the queue when
a bar row is clicked, while the M5 that replaced it does not; and the
refreshed-D1 regression guard. With the panel change stashed the first two
fail on the old behaviour (`['AMD', 'NVDA', 'MUFG', 'XOM']` where the queue
should read `['AMD', 'MUFG', 'XOM']`) and the third passes, as a guard should.

### 2026-08-27 - Group RS/RW tape removed from the desk (trader decision); rebuild plan parked in plan.md

**IMPLEMENTED / GREEN (a hide).** "Often times the sectors and industry RS/RW
thing at the top is totally wrong and doesn't reflect what is actually strong
over the last 30-60-90 minutes." Investigated: the formula
(`real_relative_strength`, ATR-normalized) is right - an independent Yahoo
recompute at 09:55 ranked the same window the same way - but the tape refreshes
only when a scan cycle's RRS pass finishes (10-30 min apart that day, frozen in
between, once 31 minutes late on a flip), its one intraday number is a
60-minute window that carries the overnight gap for the first hour, and
"industry" is one of 49 ETF proxies for 136 industries. Trader: "just remove
it for now and put this build plan in the .md files for the future."

- `TradingDeskPanel`: `group_tape.setVisible(False)`. Hidden, not deleted -
  the widget, `tests/test_qt_group_tape.py`, the `rrsSnapshotChanged` wiring
  and the `tape_host` mount point all stay, so the rebuild drops into place.
  Nothing upstream changed; the RS Window tab still reads the scan payload.
- The rebuild (a 5-minute Yahoo-batched 30|60|90 tape off today's bars, zero
  IB, no `legacy.py` change) is written out under plan.md Phase 0.5 item 11
  and, later the same morning, authorized for an Opus build session:
  `docs/prompts/GROUP_TAPE_REBUILD_OPUS_PROMPT.md` (packets T-1..T-4, hard
  rules: zero IB, no `legacy.py`, completed today-only bars, UNKNOWN never
  invented, parity test against `real_relative_strength`).
- Test added: the tape is hidden on the desk and still wired.

### 2026-08-27 - Intraday alerts are a list beside the chart, not a queue in front of it (trader rule 4: the M5 alert bar)

**IMPLEMENTED / GREEN.** "A lot of my charts to review are M5 charts. If I can
instead just get a list I can copy and paste into TC2000 that would be
faster... a little sidebar in between the master AVWAP setups and the chart...
the ticker and the alert type (new HOD, VWAP bounce etc) and I can choose what
to look at. Then we can totally purge M5 alerts from the waiting list and keep
those for D1 alerts." Ordering, when asked: "latest at the top, the oldest at
the bottom."

- `scripts/ui/widgets/m5_alert_bar.py` - `M5AlertBar`: one line per alert
  (`07:09  ▲ SYMBOL  type`), newest on top, side-coloured through an item
  foreground role (no per-widget stylesheet, no rebuild). `Copy all` puts the
  tickers on the clipboard one per line, each once, in bar order - a TC2000
  paste; `Clear all` empties the bar ON SCREEN. A click charts the alert
  and takes its line away (trader: "after I click on an alert it should go
  away") - the feed and History still have it.
  Bounded at 400 rows (a session produced 72 in its first 46 minutes).
- `AlertCenterPanel._is_m5_review_alert` + routing in `_enqueue_review_alert`
  - the one door into the queue, AFTER the AWAY-recap branch and the parked
  check, so everything upstream (the backing list, the feed, History, the
  evidence streams, the AWAY recap) is untouched. An ordinary intraday alert
  is emitted on `m5AlertPosted` and never queued; a D1 row, a Focus D1 flag,
  a chart-watch hit, a price alert the trader armed, an auto-pick proposal, a
  typed symbol and a deliberate Focus review keep their chart. The chart in
  front still refreshes from its own symbol's new M5 alert. `chart_alert()`
  is the public click path (same as a feed-row click). `m5AlertsDayRolled`
  clears the bar with the other day-scoped state.
- `TradingDeskPanel` - the bar is the LEFT column of the desk splitter
  (`m5_alert_bar | alert_center | master_workspace`, stretch 0/3/2, floor
  `px(150)`), an "M5 alerts" tab in tabs mode, rescued across mode switches.
  It was built between the chart and the setups and moved to the left the
  same morning at the trader's second pass ("move it to the left of the
  visual chart"); `DESK_SPLIT_KEY` bumped to `..._v3` so the middle-bar split
  saved that morning is not replayed onto the new order.
  `desk_layout.DESK_SPLIT_*` are three weights - the bar's share comes out of
  the setups side, so the chart column keeps its lead.
- Consequences, stated: the regime-pause hold EXPIRY (2026-08-21) and the
  movers/VWAP/SMA legs now act on D1 rows and the trader's own charts; an
  intraday row never reaches them. A counter-trend regime-pause row lists in
  the bar. EVENING's "queue the trader wakes up to" is now the bar plus the
  D1 queue. AWAY is unchanged: recap, and nothing posted to the bar.

Tests: `tests/test_qt_m5_alert_bar.py` (19: order, row text, copy dedupes
newest-first, clear, click, bound; routing for every kind that stays and the
kinds that go; nothing recorded; AWAY untouched; the chart in front refreshes;
the day roll; the bar between the two columns and wired both ways). With the
panel, desk and layout changes stashed, 13 of the 19 fail (the six pure-widget
tests pass: the untracked widget file was not stashed). Seven queue-mechanics
files (`test_qt_alert_center`, `test_movers_only_review`,
`test_qt_review_vwap_side`, `test_qt_regime_pause_expiry`, `test_qt_arm_dock`,
`test_review_events`, `test_review_guidance`) gained one autouse
fixture that switches the routing off - they test what the QUEUE does with a
row, and a D1 fixture would drag the D1 feed into every assertion; the
routing itself is owned by the new file. `test_away_day_recap` (2) and
`test_qt_regime_pause_auto_focus` (6) were rewritten to the new expectation.

### 2026-08-27 - D1 recommendations against their trend are hidden; the setups popup walks with Prev/Next (trader rule 3)

**IMPLEMENTED / GREEN.** The chart it came out of: MUFG, a swing-scanner D1
row "(short) zone1 reject at AVWAPE", sitting above its SMA50, SMA100 and
SMA200 in a clean uptrend. The scanner's own feature file called MUFG a LONG
setup and carries a `directional_sma_stack_aligned` flag; the short alert
never read it. Trader: "longs should be above the 200 SMA and shorts below
the 50 SMA at least."

**The rule:** a D1 long charts only above its SMA200, a D1 short only below
its SMA50 - the D1 recommendations (`is_d1` rows and `focus_d1_event` flags)
and nothing intraday. It is the THIRD leg of the one review verdict
(`_review_chart_state`), so it is asked at queue time and again at show time,
hides and counts on the same button ("N hidden (inside yesterday's range /
wrong side of VWAP or SMA) - show"), and a revealed name is badged
`wrong side of SMA`. UNKNOWN shows.

- `scripts/sma_trend_gate.py` - the decision, pure: `sma_trend_state(side,
  price, sma50, sma200)` (a long needs `> sma200`, a short `< sma50`, the
  other average is not consulted - "at least"), and `trend_levels(d1_bars,
  today=)` off COMPLETED daily closes: a bar marked `preview`, or dated today
  while today trades, is left out, because an average that moves every tick
  must never be the thing that hides a chart. Fewer than 200 closes is no
  SMA200 (`strength_scan.sma` refuses "as many as we have").
- `AlertCenterPanel.sma_trend_state(symbol, side)` - averages off the local
  daily store (`_d1_bars_for`), price off the last completed M5 bar when the
  bot has one and the last daily bar otherwise; memoized on both series'
  identity; any failure is UNKNOWN. No fetch, no IB traffic.
- Detector untouched: the scanner still writes the row and its evidence;
  this decides only whether the chart occupies the pane.

**Prev / Next on the snapshot popup** (same request): `SymbolSnapshotDialog`
gains `◀ Prev` / `Next ▶` beside `✕ Dislike`, visible only in a review walk
(a typed lookup has no list). They route through the setups panel
(`snapshot_review_previous` / `snapshot_review_advance`, both on the existing
`_open_next_symbol_snapshot`, now `step=±1`, wrapping at either edge) and
record nothing - Space on the table is unchanged.

**Investigated, not changed - "a lot of these candles are from Yahoo despite
the API being up":** the daily HISTORY is the durable D1 store. Only today's
FORMING candle is at issue, and it is built from BounceBot's cached IB M5
bars - which exist only for names in the current M5 scan set (the
watchlists, Focus, auto lists). For any setups-table name outside that set
(FTRE: "No cached M5 bars - not in the current scan set") the popup fetches a
Yahoo daily row for today as the preview and labels it exactly so
(`SymbolSnapshotWidget._request_snapshots`: `ibkr-cache` when M5 bars exist,
else `yfinance-fallback`). IB is up; there is simply no IB fetch path for a
forming candle on a name the bot is not scanning, and adding one would spend
the locked IB pacing budget on every double-click - a design decision for the
trader, recorded in `CURRENT_CHECKPOINT.md`.

Tests: `tests/test_sma_trend_gate.py` (11), `tests/test_qt_review_sma_trend.py`
(13: D1 short over its 50 hidden, Focus D1 long under its 200 hidden, an M5
alert is not asked, UNKNOWN shows, the button and badge, hides-never-deletes,
show-time withholding, the measurement over real bars - MUFG's shape, the M5
close as price, short history is UNKNOWN), `tests/test_qt_snapshot_prev_next.py`
(5: visible only in a walk, next/previous wrap, records nothing, side travels).
With the four source files stashed, the two Qt files fail together.

### 2026-08-27 - Chart review hides the wrong side of VWAP, and checks at show time (trader rule 2)

**IMPLEMENTED / GREEN.** The chart it came out of: EPD, a Focus D1 flag
("New 5-day high", M5 bar 06:30) that reached the review pane at 07:30 sitting
under session VWAP and fading - "a stock like this really is just wasting my
time." Two defects in the movers-only filter of 2026-08-19: it had only the
prev-day-extreme leg, and it was measured when a row was QUEUED, not when the
chart was SHOWN, so a queue 74 deep served hour-old verdicts.

**The rule:** a long charts only above session VWAP, a short only below it,
and the filter is asked again the moment a chart is about to show. Hidden
names are counted on the same button ("N hidden (inside yesterday's range /
wrong side of VWAP) - show") and one click reveals them for the session. Same
exemptions as before: a deliberate Focus review and an armed chart-watch hit
always show.

- `AlertCenterPanel.vwap_state(symbol, side)` - the adoption gate's own VWAP
  leg, `focus_adoption_gate.session_vwap_state`, fed by
  `regime_pause_hold.session_levels` over the cached M5 series (session VWAP
  from `chart_snapshot.session_vwap_series` on completed bars; never
  BounceBot's dynamic/EOD VWAP). Memoized on the bar-series identity like
  `_measure_mover_state`; a sideless row is UNKNOWN. No fetch, no IB traffic.
- `_review_chart_state(alert)` - both legs, one answer: CLOSED when EITHER leg
  is verified against the name, UNKNOWN when nothing is verified against it
  and something could not be measured (SHOWS, tagged), OPEN otherwise. This is
  deliberately not the gate's ordering ("could not measure" before "failed"):
  the gate explains an eviction, the filter decides a display, and one
  measured reason to hide is enough.
- `_enqueue_review_alert` reads `_review_chart_state` at queue time (was the
  extreme leg alone); `_advance_review_queue` reads it again at show time and
  withholds a candidate that has since gone wrong, walking on to the next.
  The revealed-for-the-session flag switches both checks off together.
- The review badge gains `wrong side of VWAP` for a revealed name the VWAP
  leg hid; `MOVING` now means extreme verified AND VWAP not verified against.
- Unchanged: it hides, never deletes; nothing reaches the review-learning
  stream, `review_policy.json`, any store or watchlist; the chart in front of
  the trader is not re-judged while they look at it; the Focus chip's own
  `MOVING` flag still reads the extreme leg alone.

Tests: `tests/test_qt_review_vwap_side.py` (21: the leg, the badge, the
button, the exemptions, no evidence written; show-time withholding, reveal,
the session-scoped switch-off, an armed hit never withheld; the measurement
over real bars - under/over VWAP, no bars / no volume / no side is UNKNOWN,
an unreadable read is UNKNOWN, the memo is keyed on the bars). All 21 fail
with the panel and widget changes stashed. `test_movers_only_review.py` is
untouched and still green.

### 2026-08-27 - With-trend regime-pause rows auto-join M5 Focus (trader rule, same morning)

**IMPLEMENTED / GREEN.** "I've been doing nothing but managing the bot all
morning. There are too many trades." Measured from `alert_review_events` for
the session's first 46 minutes (06:33-07:19): **124 charts shown** - one every
22 seconds - 40 skipped, 60 "Not today", and at 07:09 the pane read
**23 hidden / 74 waiting**. Between 07:09 and 07:18 the trader reviewed all 21
"holding highs" rows the regime-pause watch produced on a `bullish_weak` open
and put **twelve of them on M5 Focus by hand**, one click each.

**The rule (trader, 2026-08-27):** a swing LONG holding its highs on a bullish
day, or a swing SHORT pressing its lows on a bearish day, is added to M5 Focus
by the machine and never occupies the review chart - the decision is made. The
mirror cases (counter-trend rows) and a day with no directional read stay on
the queue exactly as before.

- `scripts/regime_pause_focus.py` - the whole decision, pure: `day_bias(env)`
  collapses `bullish_weak`/`bullish_strong` to one family, `focus_side_for(env,
  side)` names the Focus side or `None`. Reads nothing, no clock.
- `AlertCenterPanel._auto_focus_regime_pause` - called from `add_alert` AFTER
  the backing list insert and AFTER `is_focus` is measured, so the feed row is
  presented exactly as before (no new beep, no fold change); only the
  `_enqueue_review_alert` call is skipped when the row is resolved. The day
  label is `resolve_discovery_env(bot live env, load_opening_environment())` -
  the ONE definition discovery already uses - via `_regime_pause_day_env`.
- Writes through the STORE (not `FocusService.add`, which would log a "like"),
  stamps the auto-pick marker only when `add()` actually added - a trader's
  unmarked Focus entry keeps its owner AND its chart - and records a
  `regime_pause_auto_focus` row (`env`, `focus_side`, `outcome` in
  `adopted | already_auto | already_trader_owned`). "Not today" and the desync
  repair can therefore reach what it placed (packet R2 provenance).
- **DESK only**, like auto-pick adoption (R1 matrix). Any failure falls open
  onto the old path: the row is queued, never lost.
- Not built, on purpose: no eviction when the name stops holding (the queue's
  15-minute rule is a queue rule; the Focus entry stays until the trader or the
  desync repair says otherwise), and no change to the detector, the sweep, the
  hold measurement or the counter-trend rows.

Tests: `tests/test_regime_pause_focus.py` (18, the two-case rule and every
refusal) and `tests/test_qt_regime_pause_auto_focus.py` (12, through
`add_alert` on the real panel: placement + marker + skipped chart; counter-trend
still charts; blank/neutral admits nothing; the trader's own entry is not
relabelled; a repeat resolves as `already_auto`; AWAY/EVENING/OFF never place;
a store failure queues the row; an ordinary alert is untouched). With the panel
change stashed, 3 of the 12 fail on the assertion and 9 pass - the 9 are the
"stays on the queue" cases, which hold either way.

**Scan of what else fills the queue (same 46 minutes, 124 shown):** D1 rows
67 (`d1_flag_long/short` 41 from the Master AVWAP D1 scanner, `focus_d1_event`
26 - and the Focus list that feeds those had just received **69 machine-adopted
auto picks** at 07:09: 20 "Bullish-day weakness", 13 "RS vs SPY", 36 PDH/PDL
breaks - which raised 102 `focus_d1_flag` rows on 95 names); M5 `lrsi_cross_20`
/`lrsi_cross_50` 25; regime-pause 21; armed chart watches 11. The other primary
chart type is therefore the **D1 flag** (54% of everything shown), with the
LRSI cross (20%) second. Nothing was changed for either - that is the trader's
next call, recorded in `CURRENT_CHECKPOINT.md`.

### 2026-08-27 - Phase 0.9 G-P2.0..G-P2.2: the three presentation follow-ons from the 2026-08-26 live session

**IMPLEMENTED / GREEN.** `plan.md` Phase 0.9 items 1-3, from
`docs/GUI_REDESIGN_PLAN_2026-08-25.md` §15 decisions 9, 10 and 14, all
presentation only: no detector, scorer, alert, queue, scheduler, evidence-stream
or storage behaviour changed, and no read was added or removed.

**The table width rule now has one implementation** (`1fd9e6e`, G-P2.0). §12's
rule - "the widest TEXT column takes the slack, numeric and badge columns keep
their measured width, and the last section is not the only one that stretches" -
lives in `scripts/ui/widgets/data_table.py` as module-level `apply_width_rule`,
with `apply_width_rule_to_table_widget` for raw `QTableWidget`s. Two of the three
pages the rule was learned on do not use `DataTable` at all, so a rule applied
only through the shell would have missed them. `DataTable.fit_columns` routes
through it, so every existing `DataTable` user gets the rule with no per-panel
edit; AWAY Recap's four tables and Weekend Prep - Focus pick review's five call
it directly. A caller may name its text columns; one that does not gets a
MEASURED answer (a column whose every non-empty sampled value parses as a number
is numeric; of what is left the widest stretches, ties to the lowest index), so
the rule cannot fall behind the way a hand-maintained per-panel list would.
Identifier columns get `MiddleElideDelegate`: middle elision through
`QStyleOptionViewItem.textElideMode`, per item, with the full value as the
tooltip - `human_f...tracking`, never `human_foc...`, because the identity is in
the tail and an elision that leaves every row reading the same is a rendering
defect. `measure_column_widths` is the one seam every caller measures through;
it is `resizeColumnsToContents()` today, unchanged in cost, and it is the 7.9% /
115 s site of the 2026-08-26 measurement, so G-P2.3 item 1 bounds it in exactly
one place.

**AWAY Recap is a return surface** (`a5fa6a9`, G-P2.1; §8.3, decision 9).
Charting was wired the whole time and nothing on the page said so, and the day's
only two alerts were scanner status messages with a blank symbol, so the trader's
verdict was "i also cant even check charts from here. kinda useless." Scanner
status rows - the blank-symbol test, because a row with no symbol cannot be
charted whatever its side says - are hidden from the alerts table and COUNTED in
one line, revealed for the session by one click; nothing is deleted, nothing is
muted, and the Alert Center's backing list is untouched with `set_alerts` still
its one reader here. Every chartable row carries a visible `Chart` cell (a plain
item, never a cell widget: a widget per row is the shape the 2026-08-21 fluidity
pass spent a day removing, and an AWAY day can produce hundreds of alerts),
`Enter` on the selected row opens it through an event filter rather than Qt's
per-platform `itemActivated`, and a hint line says so. A symbol-less row renders
muted and italic from the `text_muted` THEME TOKEN - not a per-widget stylesheet,
and Qt style sheets do not reach view items at all - and offers no chart action
by either route.

**The Desk Journal has a keyboard route** (`fd76923`, G-P2.2; §5.3 option (a),
decision 10). The trader could not find the sixth lower tab. `Ctrl+J` selects it
and focuses the composer; the tab label reads `Journal  Ctrl+J`. Bound at PANEL
scope with `WidgetWithChildrenShortcut`, copying `_bind_capture_shortcuts`,
because a `QShortcut` bound inside a hidden tab page never fires - and the
Journal page is hidden exactly when the trader reaches for it. `Ctrl+J` was
verified unbound across `scripts/ui` first (the whole inventory is Ctrl+R,
Ctrl+F, F9, Ctrl+Return and Alt+V/K/S/N), because two live bindings for one
sequence is an ambiguous shortcut and Qt fires NEITHER, silently; a source-level
test now fails if a second binding ever appears. No second row under the charts
and no verb-row verb: the 2026-08-20 one-row rule holds, and a mouse route stays
the trader's to ask for. `alert_center_panel.py` is fenced and the trader
approved this exact diff in chat before the edit.

**Verification.** 5016 passed / 19 subtests, exit 0; smoke 7/7. 37 new tests, and
every one was proved failing on the un-fixed code by stashing the source file and
re-running. No packaging trigger: no new dependency, asset, top-level package or
`__file__` change.

**Owed:** the §11.3 soak against
`ui_stalls_prefix_baseline_2026-08-26.jsonl` before G-P2.3 starts, and G-P2.3 /
G-P2.4 themselves.

### 2026-08-26 night - Phase 0.10 review fixes: the shadow cannot cost the save, and the fence is no longer a hand-maintained list

**IMPLEMENTED / GREEN.** Fable's review of `002f2a3..292e335` returned GO with
two fixes owed before B-4 and one trader decision recorded. All three landed in
`ac9a952` on `claude/gui-phase-0-9`.

**The shadow export is guarded.** `export_setup_tracker_views` wrote the
band-variant CSV as its last statement with no guard, and
`update_setup_tracker_from_scan` runs `save_setup_tracker_payload` AFTER it - so
one malformed setup dict reaching `build_band_variant_stats_rows` would have
aborted the day's tracker save. That is the evidence store costing the thing it
records, which R10 forbids everywhere else in this codebase. The `try/except` +
`logging.warning` wraps the SHADOW write only: every champion export above it is
already on disk by then, and a champion export that fails must still fail
loudly - asserted as its own test.

**The fence is guarded at source.** Seven readers filter on
`_is_band_variant_scenario`, and three of those were found by the parity fixture
rather than by reading the code - so an eighth would not be found by reading
either. `tests/test_band_variant_fence_guard.py` walks the AST of `legacy.py`
and requires every scenario-iteration site to mention the fence inside its
enclosing function or to be named in `ALLOWED_UNFENCED` with its reason. Two
entries, both readers that MUST see the shadow: the stop rebuild on replay
(`_extract_tracker_stop_candidates_from_setup`, which sorts by label so
`VARIANT_*` still lands last) and sealed-record compaction
(`_compact_tracker_setup_record`, which strips the shadow's per-bar event log
exactly as it strips the champion's).

The detector is deliberately wider than the spelling the fence was written
against - `setup["scenarios"].values()`, `.get("scenarios", {}).values()` and a
local `working_scenarios.values()` all count - because a guard that only
recognizes today's spelling is passed by tomorrow's. It finds nine readers where
the narrow `(setup.get("scenarios") or {}).values()` pattern finds six. Proved
against real code rather than a mutation: pointed at `5613eec:legacy.py`, the
tree as it stood before the fence, it reports six unfenced readers. Four
companion tests keep the guard itself honest. It does NOT claim that mentioning
the helper means it was used correctly - a name in a function is not a proof
about its logic, and the parity fixture remains what proves the values did not
move.

**The shadow crosses the four BASELINE exit templates only** (trader decision,
2026-08-26). `_is_band_variant_stop` is the candidate-side twin of
`_is_band_variant_scenario`, kept beside it so the two spellings of "is this the
shadow" cannot drift apart, and `_build_tracker_scenarios` skips experimental
templates for such a stop. The champion is untouched and still crosses all six;
the experimental templates are a comparison framework for the CHAMPION's stops,
and a challenger inside them would be two variables at once. Re-measured:
**9,982 -> 6,524 bytes per new setup** (474 anchor blocks + 6,050 for four
variant scenarios), so **~144 MB -> ~89.5 MB, 15% -> 9.4%** at the live
14,386-setup / 950.2 MB scale, forward only - and 5,739 bytes once sealed-record
compaction strips the event logs. All four baseline templates remain, so the
stats table's per-template pairing is still possible.

Verification: **4995 passed / 19 subtests, exit 0** at `ac9a952`, and **5010
passed, exit 0** on the tip `714f717` once Phase 0.9's `a5fa6a9` (committed by a
concurrent session while this work ran) landed beneath it; smoke 7/7. Eleven
tests added, every one proved failing first. **Owed, unchanged**: T4's three
criteria, >= 20 sessions of forward accrual before T3 counts, and B-4 - which
these two fixes were the gate on.

### 2026-08-26 - AVWAP band challenger: a second formula, computed beside the champion and unable to reach it

**IMPLEMENTED / GREEN. Every T4 gate is OWED and no test discharges one.**
`plan.md` Phase 0.10, governing spec `docs/AVWAP_BAND_VARIANT_STUDY.md`, build
prompt `docs/prompts/AVWAP_BAND_CHALLENGER_OPUS_PROMPT.md`. Branch
`claude/avwap-band-challenger` off `claude/gui-p1-fluidity` at `88a34b7`.

`calc_anchored_vwap_bands` is untouched and frozen (decision 0008). Nothing in
this packet reaches a detector, score, rank, tier, alert, zone arm, Focus list,
review queue or `review_policy.json`.

**B-0 - the formula, replicated** (`002f2a3`).
`scripts/indicators/avwap_band_variants.py`
(`avwap_bands_oneoption_bb20_v1`): an anchored HLC/3 volume-weighted centre with
a 20-close **population** Bollinger sigma as its half-width - the form the trader
pinned on 2026-08-26 against OneOption / Option Stalker Pro. The two halves know
nothing about each other, so the sigma window deliberately reaches back BEFORE
the anchor; that is why the band is already wide on the anchor bar where the
champion's is exactly zero. `indicators/` shape throughout: completed bars in,
immutable aligned tuples out, no I/O, no pandas import, `None` below the
lookback - never padded, never 0.0, never a shorter window.

`tests/fixtures/avwap_band_variant_oneoption_v1.json` freezes OKTA
2026-04-01..06-05 from the durable store **through
`_normalize_daily_bar_frame`**, not an ad-hoc threshold, because that store's
OKTA volumes are mixed-unit. Neither golden row is affected and the fixture says
why. The expectations are the trader's hover readings, not this repo's output:
centre to 0.2% relative (126.78 here against 126.565 there - a consolidated-vs-IB
volume-feed gap), sigma to +/-0.02 absolute (18.039 vs 18.035).

Two discriminators are pinned as arithmetic: the champion's sigma is 0.0 on a
one-bar anchor where OneOption read 10.28, and the killed sample-OHLC form
predicts an upper of 138.09 on 2026-06-02 where the trader read 144.60. The
killed form lives in the TEST, not the module - no live code carries a formula
the study already killed. An AST test forbids the module from importing
`master_avwap_lib` at all.

**B-1 - the hover-comparison table** (`13505d1`).
`scripts/avwap_band_variant_fit.py SYMBOL ANCHOR_DATE [--lookback 20]` prints
both formulas per session since an anchor. Offline; writes nothing without
`--csv`, and then only into `OUTPUT_DIR/reports/`. The champion publishes only
its final bar, so its column comes from calling the frozen function once per
session on a truncated frame - a call, never an edit. An unmeasurable cell prints
EMPTY, because the champion's sigma really is 0.00 on the anchor bar and the two
must not look alike. Live read on OKTA reproduces the study's S2 column exactly.

**B-2 - the tracker shadow** (`5613eec` fixture, `603333b` code).
Golden fixture FIRST: `tracker_record_band_variant_parity_v1` was frozen on the
champion's code BEFORE either fenced file was touched, and it earned its keep.
`runner.build_anchor_band_variant_meta` computes the challenger from the same
frame and anchor index; `current_anchor_variant` / `previous_anchor_variant`
ride `symbol_entry` and the setup record; `_find_tracker_stop_candidates`
appends one `VARIANT_<protective>` candidate LAST with the champion's own
`close_failure_limit`; `master_avwap_band_variant_stats.csv` is written in the
existing export pass and read by a "Band Variant" tab on the Setup Tracker page.

**Appending after the champion's candidates was necessary and NOT sufficient**,
and the prompt assumed it was. `representative_total_r` is picked by label and
did not move - but `_summarize_tracker_setup_outcome` averages `total_r` across
every tradeable non-experimental scenario, and that average reaches
`build_tracker_setup_type_rows` -> `apply_tracker_setup_type_adjustments` ->
`row["score"]`. Measured on the frozen fixture before the fence: `avg_total_r`
-0.0790 -> -0.0755, `tradeable_scenario_count` 8 -> 12, eight summary values in
all, plus `daily_marks[1].scenario_events` 10 -> 15, the short's `setup_status`
CLOSED -> OPEN, and 12 -> 18 rows in the scenario and stats CSVs.
**Trader-authorized 2026-08-26**, `_is_band_variant_scenario` now fences seven
readers. The shadow is still graded - `_evaluate_tracker_scenario_bar` runs for
it exactly as before - its events simply stay off the champion's mark. The last
three fence sites were found by the fixture rather than by reading the code.

Two findings worth more than the code. The challenger's sigma is 1.339 where the
champion's is 0.586 seven sessions after an anchor (2.3x), which is why the
trader's screenshots looked better early. And **"the wider band is stopped out
less often by construction" is only true when entry sits INSIDE the band**: the
fixture's short is entered above both upper bands, the wider sigma pushes the
upper band UP toward entry, and the challenger's stop lands 0.159 away where the
champion's is 0.971 - six times TIGHTER, from the wider formula. T1 and T3 may
not assume a direction.

**Tracker JSON growth, measured**: 9,982 bytes per NEW setup (474 for the two
anchor blocks, 9,508 for six variant scenarios with their event lists) against a
live file of 950.2 MB holding 14,386 setups - about 144 MB, ~15%, if every setup
carried it. It accrues forward only; existing records do not grow until rebuilt.
The study estimated "a few hundred bytes per setup" and was ~30x low.

**B-3 - the D1 overlay, default OFF** (`3abf61d`).
`chart_levels.avwap_variant_levels` builds six sloped lines in the
`avwap_variant` group on the ChartDataService worker, never on the paint path,
anchored on the date the snapshot already resolved so the two lines on one chart
differ for one reason rather than two. **The paint-lines preference file had no
way to express a default-OFF group** - every group defaults ON there on purpose -
so `chart_levels.GROUPS_HIDDEN_BY_DEFAULT` names the exceptions and
`PaintLinesPrefs` gained a `shown_groups` list, letting both defaults live in one
file: an older preference file keeps the group off, the trader's own hidden
groups survive a rewrite beside it, and an unreadable file falls back to the
defaults rather than to "show everything". Four existing paint-lines tests now
assert the amended rule instead of the blanket one.

`indicators.avwap_band_variants` joined `selftest.LAZY_ENGINE_MODULES` - the
first lazy import of it from a path a frozen run can reach. `indicators` was
already in the spec's `collect_submodules`, so no spec edit was needed, verified
rather than assumed.

Verification across the packet: 4968 passed / 19 subtests, exit 0 (baseline
4902); smoke 7/7; `launch_gui.py --selftest` 71/71; spec-drift 17 passed.
**Owed**: T4's three criteria, including >= 20 sessions of forward accrual with
>= 40 finalized setups before T3 counts, and the B-4 backfills (T1 level quality,
T2 playbook re-run) which are the next packet and are NOT started.

### 2026-08-26 - GUI fluidity Wave P1: the desk stops reading its stores on the click

**IMPLEMENTED / GREEN. The live soak is OWED and no test discharges it.**
`plan.md` Phase 0.8, promoted by the trader from
`docs/GUI_REDESIGN_PLAN_2026-08-25.md` §11.1 on 2026-08-26. Presentation and
threading only: no detector, scorer, alert, queue, scheduler, evidence stream or
store changed behavior, and no read was added or removed - only moved.

**Three verified defects, each reproduced at source before it was touched.** The
AWAY Recap called `focus_picks.load_focus_map(side)` against a keyword-only
signature: TypeError on every run, absorbed by a fail-quiet `except` and shown to
the trader as "Focus lists unavailable". The page had therefore NEVER read the
Focus lists, and no amount of the files being present could have changed that.
The same page's adoption-gate line called `mover_state(side, None, None, None)`
against `(side, price, prev_high, prev_low)` - with nothing to compare it could
only return UNKNOWN, which was then rendered as a gate verdict for the symbol. It
now says the gate was not measured here and why, and points at the surfaces that
do measure it; UNKNOWN stays UNKNOWN. And the Desk quick-journal write (Ctrl+Enter,
the one used mid-session with a chart up) dropped the `symbols` field
`MarketJournalService.write_entry` has accepted since R10.H, so the entries most
likely to be about one name were the ones stored with no name.

The shared shape of the first two is worth keeping: **a `try/except` written to
keep a page from crashing had been absorbing a programming error and reporting it
as missing data.** Fail-quiet is right for a store that might not be there; it is
wrong for a call that can never work.

**Weekend Prep now reads on a worker.** `WeekReviewPage.reload` ran
`build_review_learning_state(window_days=7)` plus two RS log scans, and
`FocusReviewPage.reload` ran five CSV/JSONL reads and built five tables of cells,
all inside the click that selected the page - the worst measured stall on the
desk, 8.45 s frozen. `WalkawayPage` in the same file already owned a QThread, so
one `_ReadWorker` now serves all three. Deliberately NOT copied from it: that
page blanks its body to "Running walk-away..." while refreshing. **Clearing a
populated page to announce a refresh destroys the only copy of what it knew**,
most damagingly when the refresh then fails, so the new pages keep last-good
visible and put "refreshing" and any stated error in their own slot. On
`FocusReviewPage` a failed refresh keeps every row: the graded cohorts are the
whole forward record of the trader's own vetoes and likes. That page previously
had no error handling at all - a bad CSV propagated out of the click. Panel
shutdown, which named `walkaway` while that was the only threaded page, now joins
every page.

**The Focus board measures each mover state once per poll, not once per redraw**
(36 repeating stalls, 5.93 s). `_refresh_all` fires on things unrelated to
previous-day extremes - a BounceBot alert, an RS/RW snapshot, a side edit - and
each walked every chip through `AlertCenterPanel.mover_state`, reading the D1/M5
series per symbol per side. Memoized per (symbol, side), discarded by
`refresh_mover_flags`, which is not an arbitrary expiry but the signal that a
newer measurement exists. A FAILED measurement is never cached: a flag is
decoration over a measurement, and one transient miss must not switch it off
until the next poll.

**A stall record now says which click it belongs to.** `scripts/ui/interaction_trace.py`
plus stamping in `StallWatchdog._write`. The watchdog names the frame that held
the GUI thread, which is the wrong question whenever the modal frame sits inside
Qt's own event dispatch and names no application code. The trace is read from the
sampler thread and therefore holds **no lock** - live state is one module-level
tuple replaced whole, because a lock in a diagnostic could stall the thread it
exists to measure. It owns no timer and no thread, and a test PARSES the module
and fails on any call to sleep/wait/start/join/Thread/Timer/QTimer - the
`ScanCycleClock` rule, for the same reason. An empty interaction id means an
idle-desk stall, which is a fact about the stall rather than a gap. Wired at page
select, the Journal inner tab and the chart request.

**Fence discipline.** `alert_center_panel.py` is fenced under the file-scoped
ask-first rule; the trader pre-authorized only the quick-journal symbols
attachment, and the diff there is six added lines and no deletion. The mover memo
was implemented in the consumer rather than at its natural point inside that
file.

**The mover memo moved to its source.** The trader extended the fence
authorization, so the memo now lives in `AlertCenterPanel._measure_mover_state`
as well - the review queue asks the same question once per alert and now gets
the same answer for free. The design came from a measurement rather than a
guess: per (symbol, side), m5 materialization is 0.049 ms and everything after
it is 0.186 ms, so **79% of the cost is memo-able after materialization**. That
is why the key is the identity of the bars measured - session date plus the
length and last timestamp of both series - and not a clock. `mover_state` feeds
the movers-only review filter, which decides what the trader SEES; a time-based
cache would let a name that has just broken yesterday's high stay hidden until
it lapsed. A new bar is a new key.

**System Health stopped rebuilding itself**, and the warehouse readout stopped
reading a network share on the Qt thread. `_fill` and the checks table built a
fresh `QTableWidgetItem` per cell of three tables every 15 seconds - which is
also where the scroll position went, so a trader reading the bottom of the jobs
list was pulled back to the top mid-read, on a timer, with nothing on screen to
explain it. `WarehouseReadoutPanel.refresh` called `ResearchStore.open()` and
`slice_readout()` inline against the DAS lake; that share is known to drop, and
an SMB read against a dropped share blocks until it times out. It was the only
read in the whole audit that leaves the machine. It also blanked its table on
every failure path - an unreadable lake is not an empty lake - and now keeps
last-good on failure while still clearing on a successful empty read.

**The `reload()` audit is complete, and most of what it found is still owed.**
Fourteen panels have a reload/refresh plus file IO; eight own no worker at all.
One was fixed (above); `WeekAheadPage` and `DiscoveryPage` audited clean.
The other eight are named in `plan.md` under G-P1.5, `setup_tracker_panel`
first. Nothing was half-converted: a partial page is worse than an honest list.

**A latent crash the audit found (G-P1.6).** Adding a second
HealthPanel-constructing test file made an unrelated Qt test segfault two files
later - 4 runs in 6. `HealthPanel.shutdown` stopped the panel's timer and left
its audit thread running; that thread emits a Qt signal back into the panel, so
it could fire into a freed C++ object - an **access violation**, which the
`except RuntimeError` at the emit cannot catch because it is not a Python
exception. Reproduced at the committed HEAD with all work stashed, so it
pre-dates this wave. `shutdown` now joins the thread and a `_closing` flag stops
a refresh queued before shutdown (construction uses `singleShot(0, ...)`) from
starting a fresh one after it. **The class is not closed:** any panel that
starts a bare `threading.Thread` and emits a Qt signal back into itself has the
same defect.

Three shutdown lists in this wave named their threaded children by hand and had
each fallen behind: `WeekendPrepPanel` (named only `walkaway`), `ResearchPanel`
(missed the readout), and the `MainWindow` list the readout sits under. Two were
fixed by naming the missing child; the weekend prep one now iterates its pages.

**Every shutdown join is bounded (`e0f78ae`).** Found in live use, not by a
test: the trader closed the window on 2026-08-26, it "froze for a few seconds",
and the PROCESS OUTLIVED THE WINDOW. Four shutdown paths joined their reader
with a bare `worker.wait()`, which has no upper bound - two from this wave
(weekend prep, warehouse readout), two older (journal panel, weekend prep
service) - and the warehouse reader is on the DAS, the one read in the desk
that can block for minutes when the share is unwell, which is exactly when a
trader gives up and closes the app. `ui/read_worker.join_worker` (5 s default)
replaces all four. On timeout the worker is DISOWNED AND PARKED in a
module-level list rather than dropped, because dropping the last Python
reference to a running `QThread` destroys its C++ half mid-run - a crash, not a
leak; these are reads with no side effects and the process is leaving anyway.
`tests/test_shutdown_waits_are_bounded.py` is a source-level guard: a bare
`.wait()` on a shutdown path fails the suite. Tests 4897 -> 4902.

**The proposal is reconciled to the build (docs only, 2026-08-26 evening).**
`docs/GUI_REDESIGN_PLAN_2026-08-25.md` now records Wave P1 as BUILT with
commit ids, replaces its 45-minute fluidity sample with the archived full
pre-fix session (3350 stalls / 1457.5 s; by blocked time, not count), states
what Wave P1 can and cannot be expected to change against it, re-orders the
owed fluidity work by measured time (the two Qt table paths and the growing
Theta refresh first - those are Qt measurement costs, not reads, so a worker
does not fix them), folds in the trader's 2026-08-26 live findings (narrow
columns on every table page; AWAY Recap unusable as a return surface; the Desk
Journal undiscoverable) as a table-width RULE plus page decisions, adds the
build's standing constraints (bounded joins, panel threads, child lists,
never-blank refresh, the fence, the unwired paint marks), and records that
Smart App Control now reads OFF. **One premise of the 2026-08-25 draft was
refuted at source:** its "arm bar contract/source mismatch" - the arm bar is
under the chart by the trader's 2026-08-20 second-pass instruction
(`4c05de5`, "the hotbuttons return"), so the CLAUDE.md/AGENTS.md line placing
it on the Armed tab is the stale one. Flagged for the trader; not edited.
Waves U1-U3, S1 and Snappy P2 remain PROPOSAL. **The trader then authorized
all changes (same evening):** CLAUDE.md/AGENTS.md now say the arm bar is under
the chart, that SAC reads OFF and the source launch stays production by trader
decision, and carry a new rule that chat messages to the trader are written
very simply; `trading_desk.cmd`'s header matches; `plan.md` gained Phase 0.9
(table width rule, AWAY Recap return surface, Desk Journal route, the next
fluidity slice in measured order, a GC MEASUREMENT packet with no scheduling
change). Nothing in Phase 0.9 is built.

**AVWAP band challenger planned, replicated and authorized (same evening, docs
only).** The trader compared their anchored-VWAP bands with OneOption / Option
Stalker Pro's, which are wide from the anchor bar. A one-evening study
(`docs/AVWAP_BAND_VARIANT_STUDY.md`) replicated the vendor's band from three
OKTA hover readings: `AVWAP(HLC/3) ± k · stdev(close, 20, population)` - the
textbook Bollinger σ laid on an anchored HLC/3 centre, no anchor memory
(the anchored sample-OHLC form predicted 138.09 on 2026-06-02; the reading was
144.60). `plan.md` gained **Phase 0.10** (module + fixture, fit script, tracker
shadow stops + stats + panel section, D1 overlay off by default; backfills
after review) with the Opus build prompt at
`docs/prompts/AVWAP_BAND_CHALLENGER_OPUS_PROMPT.md`. Nothing is built; the
champion σ stays frozen (decision 0008) and any promotion would be an
additional level family, never a swap.

Tests 4844 -> 4902, exit 0 (4897 at `49744a7`, 4902 at `e0f78ae`); smoke 7/7.
No packaging trigger. **Owed:** the eight
panels under G-P1.5, the bare-thread sweep under G-P1.6, the
`first_paint`/`chart_ready` marks (which need the receiving paint path
instrumented rather than the emit seam), and **the §11.3 live soak, which is the
trader's to run and which no test discharges.**

### 2026-08-26 - the Phase 0.5 work is on `main`, and the branch chain is retired

**Three weeks of Phase 0.5 development became the trunk.** From 2026-08-04 the work
ran on a nested chain of branches rather than on `main`, because the trader was
running unmerged branch code in production through a scheduled task
(`docs/CHECKPOINT_REVIEW_2026-08-08.md`). The chain ended at
`testing-week-2026-08-24`, which contained every commit of its predecessors, and
`main` was a **strict ancestor** of it. The consolidation was therefore a
fast-forward: 354 commits, 480 files, no conflict, and no merge resolution
performed. `git merge-base --is-ancestor` proved the relationship before the merge
rather than after it.

**The code state on `main` is byte-identical to the state that was verified.** The
only non-`main` content added beside the fast-forward is Markdown, so the
4844-passed/19-subtest baseline recorded for `ed277a7` describes `main` exactly. It
was **not** re-run for this merge: the container this consolidation ran in has no
project virtualenv and Python 3.11 against a project floor of 3.12, so a run there
would have proved nothing. That is a stated limit of this entry, not a claim of
green.

**One unlanded document was brought in.** `claude/trade-analysis-opus-prompt`, a
single additive commit from 2026-08-22, contributes
`docs/prompts/TRADE_ANALYSIS_OPUS_ULTRACODE_PROMPT.md` - the Opus trade-analysis
prompt carrying the scoreboard read, the earliness audit and the AEP DT case. Its
context list still told the reader to load `SOL_PROGRESS.md`, which this repository
deleted when `CHANGELOG.md` and `CURRENT_CHECKPOINT.md` took over that role; the
reference now names the pair. The prompt is classified in `docs/README.md` and is
authorization for nothing.

**Three branches are cleared for deletion, and what they were is written down.** New
`docs/BRANCH_HISTORY.md` records every branch in the chain with its commit count,
date range, tip SHA and disposition, so deleting a merged branch never destroys the
only account of what it held. `claude/ticker-briefs-hardening-imcm8r` (94 commits),
`phase05-r2-focus-gating-strength-board` (150) and `phase05-integration-blitz` (308)
each hold no commit that is not on `main`, proved with
`git merge-base --is-ancestor` against `226fbac`. **The deletion itself did not
happen and is owed to the desk:** the cloud session's GitHub credential pushes but
refuses ref deletion with `HTTP 403`, with no proxy policy denial recorded, and the
GitHub MCP surface has no delete-branch counterpart to `create_branch`. The three
commands are in `docs/BRANCH_HISTORY.md`. `testing-week-2026-08-24` is **kept** -
the active GUI-optimization work continues on it.

**The Alert Center quality packet remains unmerged, by decision.**
`claude/alert-center-quality-packet-5btu3w` (8 commits, tip `57fcf47`, 2026-08-18)
builds the alert-delivery measurement surface `GUI_TRADE_DISCOVERY_LEARNING_PLAN.md`
sec 10.3/17 specify but never built: `scripts/alert_quality.py`,
`scripts/alert_delivery_events.py`, a delivery-capture emit in
`scripts/ui/panels/alert_center_panel.py`, a System Health surface, and its tests.
Two things block it and both are recorded rather than guessed at - it **edits alert
code**, so the file-scoped ask-first rule governs the merge itself; and it adds its
own `docs/ALERT_CENTER_QUALITY_PACKET.md` at the same path where `main` already
carries the *different* historical P1.6 packet recovered from `671ee57`, so a
content merge would silently destroy one of the two. Nothing on `main` depends on
it. No alert behavior changed in this consolidation.

## Revision history

Entries from **2026-08-19 back to the initial system in 2025-11** moved to
[`docs/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md`](docs/CHANGELOG_ARCHIVE_2025-11_2026-08-19.md)
on 2026-08-27 (36 entries). Newer revisions are dated entries at the top of
`Current implemented inventory` above. The archive is evidence, not authority — read it
only when the history of a specific change is not answered here or by the governing spec.

## Retired or superseded implementations

- Desk Link satellite relay/control and the separate mini-PC scanner role are retired
  as of 2026-08-08. The code remains pending a scoped cleanup.
- H1 alerts were retired; H1 now confirms D1 tracker picks.
- The old DESK approval queue for auto-populate was superseded by direct day-scoped
  M5 Focus adoption.
- The legacy shared review-event ledger is read-only; per-installation shards are the
  current writer path.
- The legacy Tk UI remains only for migration compatibility and is not the product
  direction.
- Historical plans and handoffs listed as such in `docs/README.md` are evidence, not
  current execution authority.
