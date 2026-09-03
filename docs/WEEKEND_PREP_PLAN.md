# Weekend Prep Plan — Phase 0.5 R8

**Status: ACTIVE spec — authorized 2026-08-15, BUILDING.**
Branch `testing-week-2026-08-17`, cut from the R7 tip `4420bbf` on 2026-08-15
after R7's build completed. Baseline at cut: 3203 passed / 19 subtests, exit 0.
Trader-directed packet (2026-08-15 desk request). Builds **after** R7
(`docs/JOURNAL_RELIABILITY_AND_UX_PLAN.md`) because the walk-away and auto-tag
review steps read the journal R7 makes trustworthy. Per the second 2026-08-15
trader redirect (recorded in the R7 spec header and `plan.md` Phase 0.5
preamble), R7 builds now on a branch cut from the R2 tip; R8's branch
`testing-week-2026-08-17` is cut from the R7 tip when R7's build completes —
or from `main` if the stack has merged by then. Overlap with R7's live gates
only if no shared file is in flight — `ui/services/journal_feed.py` IS shared.

## 1. Purpose and locked product decisions (trader, 2026-08-15)

A new top-level desk page "Weekend Prep": a **guided sequential routine** matching
the trader's weekend ritual, five steps, each completable or skippable, progress
persisted across sittings:

1. **Week in review** — what happened, what was shown/taken/skipped/disliked.
2. **Focus pick review** — how the week's focus picks behaved.
3. **Walk-away analysis** — trades actually taken, windowed to the reviewed week,
   with the weekly auto-tag review as a sub-pane (both iterate the week's trades).
4. **Discovery** — strongest/weakest boards on H1, D1, and Monthly, using the same
   TC2000 strength formula as the R2 M5 board, parameterized per timeframe.
5. **Week ahead** — the forward-looking weekly prep (earnings/economic calendar,
   risk windows), adopting the orphaned `market_prep` weekly engine.

Locked decisions: same strength formula on all timeframes; journal hook is the
weekly auto-tag review only (no tag-performance stats, no tagging of weekend
picks, no weekly journal entry in v1) — **amended 2026-08-24, trader-approved
(AI-P2): the sub-pane gains a DEFAULT-OFF "show all pending proposals" toggle
so the backlog is drainable. The weekly scope stays the default; the confirm
and correct paths are unchanged; no tag-performance stats, no tagging of
weekend picks and no weekly journal entry were added.** The amendment exists
because the store held 220 candidate rows across 48 closed trades against ONE
confirmed annotation, so a weekly-only pane could never let the confirmation
stream fill; Adopt routes to swing Focus + watchlist
injection; universe is `universe_all.txt` (~1,500 symbols; the broader-universe
WISHLIST candidate stays gated as written); all refreshes are manual.

**Already landed in R7, so R8 does nothing about it:** trader-confirmed account
tax statuses are machine-local under `journal_trader_tax_statuses`; repository
source contains no real account identifiers. R7 applies those declarations as
`tax_status_source='trader'`. Weekend Prep reads the journal; it does not label
accounts.

## 2. Invariants (binding restatements)

- **All refreshes are MANUAL.** The weekend quiet-hours gate
  (`autopilot_core.auto_scanning_due` → False on weekends) is untouched; the
  service owns **no QTimer** and starts nothing without an explicit button press.
  Manual buttons are never gated — that carve-out is the design basis.
- **Zero IB traffic.** All bar fetches are batched yfinance via the R2 downloader
  path. The locked IB pacing budget is untouched.
- **Completed bars only.** H1: bar_start+60min ≤ now (market-local). D1: bar date
  ≤ `market_calendar.last_completed_session(now)`. Monthly: drop any bar whose
  (year, month) equals the current calendar month — month identity, never
  duration. UNKNOWN/short history is excluded with honest offered/measured
  accounting, never scored.
- **One owner**: `WeekendPrepService` owns `weekend_prep_state.json` and all
  refresh workers (single-flight per action). Failed fetch keeps the last good
  board with the error in the status line.
- **Adds only**: the Adopt path adds to Focus/watchlists through existing
  membership-tracked injection; nothing in this tab removes any entry.
  User-entered names remain untouchable.
- **No new ntfy sender. No detector/scoring/alert file edited** —
  `scripts/strength_scan.py` is explicitly NOT edited (§4); if an edit ever seems
  necessary, stop and ask first.

## 3. Stepper contract

Step ids: `week_review`, `focus_review`, `walkaway`, `discovery`, `week_ahead`;
each `pending|done|skipped` with a timestamp. Weekend identity = the Friday date
of the week containing `market_calendar.last_completed_session(now)`. State file:
`<shared_home>/data/runtime/weekend_prep_state.json` (new
`WEEKEND_PREP_STATE_FILE` in `scripts/project_paths.py`), written through the
shared `diagnostics.artifact_io.atomic_write_json` helper and pruned to the most
recent 8 weekends. Schema v1:

```json
{"version": 1, "weekends": {"<friday-date>": {
  "steps": {"week_review": {"status": "done", "at": "..."}},
  "boards": {"h1": {}, "d1": {}, "m1": {}},
  "adopted": [{"symbol": "", "side": "", "tf": "", "at": ""}],
  "tag_review": {"confirmed": [], "corrected": {}},
  "week_ahead": {"ran_at": ""}}}}
```

"Routine complete" = all five steps done or skipped. Closing the app mid-routine
and relaunching restores progress. No weekly rollup artifact in v1 — the state
file plus `pick_feedback.jsonl` provenance and the membership file are the
durable record (a rendered rollup is a WISHLIST candidate).

## 4. Weekend strength scanner

The TC2000 formula, unchanged on every timeframe:
`strength = (avg over last 12 completed bars of ((C/O)−1)×100) × ((C + C50)/2) / ATR50`,
where C50 is the close 50 bars ago (displacement, not an SMA) and ATR50 needs 51
bars. Percentile cut = top/bottom 25% of the measurable population, taken
**before** filters — identical order to the R2 board.

**`scripts/strength_scan.py` is not edited.** A new pure module
`scripts/weekend_strength.py` imports its already-parameterized pure functions
(`strength_score`, `atr`, `ema`, `percentile_cut`, `displaced_close`) and
reimplements only the ~20-line board orchestration. M5 bit-identity is true by
construction; an M5 characterization fixture is added anyway as drift insurance
on the shared functions.

`StrengthTimeframe` frozen dataclass: `key, label, yf_interval, yf_period,
bar_kind, bar_minutes, body_bars=12, atr_period=50, ema_span=15, filters`.

| | H1 | D1 | M1 (monthly) |
|---|---|---|---|
| yfinance | `1h` / `3mo` | `1d` / `1y` | `1mo` / `6y` |
| Completed-bar rule | bar_start+60min ≤ now | date ≤ last completed session | drop current (year, month) |
| 51-bar shortfall | rare | rare | common for recent IPOs — excluded honestly |

Monthly is deliberately `6y`, not `period="max"` (51 completed months ≈ 4.35y;
bounded memory; short-history names return None). Fetch path: chunked
`autopilot_core._default_downloader` over `universe_all.txt`, mirroring the R2
service; per-TF wall clock measured and recorded here at build time.

## 5. Per-timeframe filters — **TRADER APPROVAL REQUIRED before Discovery is coded**

Session VWAP has no meaning above M5 and is dropped, not imitated. Proposed
three-leg structure mirroring the M5 gate (trend proxy + prior-extreme break);
each filter is a small named function so one TF can be amended without touching
the others; a leg that cannot be measured fails with a reason string.

| TF | Long passes when (all legs) | Short mirror |
|---|---|---|
| H1 | last > EMA15(H1 closes) AND last > prior completed session's high | below EMA15(H1) and below prior session's low |
| D1 | last > EMA15(D1 closes) AND last > prior completed ISO-week's high | below EMA15(D1) and below prior week's low |
| M1 | last completed month's close > previous month's high | last completed month's close < previous month's low |

Trader decision line: **APPROVED AS PROPOSED — trader, 2026-08-15**
("let's finish R8"). The table above is the built contract; no leg was amended.

What that approval binds, stated plainly so a later reader does not have to
reconstruct it:

- **Session VWAP is dropped above M5, not imitated.** It is a session-anchored
  measure and there is no session inside an H1, D1 or monthly bar. Substituting
  a look-alike would have produced a number that reads like the M5 gate's and
  means something else.
- **Each leg is its own named function**, so one timeframe can be amended later
  without touching the others.
- **A leg that cannot be measured fails, with a reason string.** Short history,
  a missing prior week, a symbol with no completed month - all of them refuse
  rather than pass by default. Missing data is uncertainty, never confirmation.
- The **short mirror is a mirror**, not a separate rule: the same legs with the
  comparison and the extreme inverted.

## 6. Step data contracts (inputs → refresh trigger → outputs)

> **DEFERRED — release-candidate reconciliation, 2026-08-15; narrowed twice.**
> The RS/RW extremes join and the picks↔outcomes join landed 2026-08-18 (see
> "Retained joins" at the end of this file). The **`veto_cohort_*.csv`
> mirror-cohort view landed 2026-08-24 as AI-P1**, pooled through
> `canonical_veto_cohort` and labelled discovery-not-confirmation.
>
> **Nothing in this block remains future work.** The
> `human_focus_performance.csv` and `pick_feedback.jsonl` views landed
> 2026-08-24 as packet W1's successor W2, together with the
> `rrs_group_strength_extremes.csv` stream in Week in Review — see "The last
> deferred joins" at the end of this file. §10's one-real-weekend live gate is
> still owed for all of them: building a view never validates it.
>
> One deviation is recorded rather than papered over. This section asked for
> `human_focus_performance.csv` "filtered to the week"; the rollup carries no
> trade date, only the `updated_at` stamp of its last rebuild, so a week filter
> would filter on when the nightly pass RAN and empty the table on any week it
> did not. It renders whole, with its as-of stamp on the pane and the sentence
> saying it is not week-scoped — the same treatment the two cohort tables
> already get, and for the same reason.
>
> AI-P1 also repaired a defect this step shipped with: `_join_focus_week`
> resolved its CSVs under `PERSISTENT_DATA_DIR` (the home root) rather than
> `data/runtime`, so Focus Pick Review rendered an **empty table on the live
> desk** from 2026-08-18 until 2026-08-24. Both joins now address their files
> by the named `project_paths` constants. §10's one-real-weekend live gate is
> unchanged and still owed — and it is now the thing that would have caught
> this, which is the argument for owing it.

- **Week in review**: `review_learning.build_review_learning_state(window_days=7)`
  (takes/skips/rejects, blind spots, leaks, watch conversion; the
  (trade_date, symbol) episode folding is a recorded, accepted v1 limitation) +
  `rrs_strength_extremes.csv` / `rrs_group_strength_extremes.csv` rows stamped
  within the reviewed week. Button-refresh only.
- **Focus pick review**: `human_focus_daily_picks.csv`,
  `human_focus_outcomes.csv` (h1/h3/h5/h10 side-adjusted returns),
  `human_focus_performance.csv` (whole rollup, as-of stamped — see the
  deviation above), `pick_feedback.jsonl` filtered to the week on `trade_date`;
  `veto_cohort_*.csv` and `like_cohort_*.csv` shown as the mirror cohorts.
- **Walk-away**: `run_walkaway_analysis(source=..., write_outputs=False,
  since=<Mon>, until=<Fri>)` — the kwargs land in R7. "The week's trades" =
  closed within Mon–Fri of the reviewed week; still-open trades opened in the
  week are flagged, not silently included. Auto-tag sub-pane: the week's trades
  with `auto_tag_candidates` proposals; confirm → `save_annotation`, correct →
  `record_tag_corrections` (R7's `journal_feed` helpers).
- **Discovery**: three sub-tabs (H1/D1/M1), each with manual Refresh, an
  offered/measured/filtered accounting line, an as-of stamp, and a failure banner
  that keeps the last good board.
- **Week ahead**: lazy `market_prep.orchestrator.MarketPrepOrchestrator()
  .run_weekly_prep()` in the worker; render the returned `report` text in a
  QTextBrowser; failure keeps the last rendered report.
  `resolve_weekly_prep_window` maps a weekend reference to the upcoming week —
  confirmed forward-looking, which is what this step wants.

## 7. Adoption routing

One Adopt per discovery row → confirm dialog →
`FocusService.add(symbol, side, "swing", origin="weekend_prep",
context="weekend_prep:<tf>:<weekend_id>")`. The swing category already injects
into `swinglongs.txt`/`shortswings.txt` with provenance in
`FOCUS_PICK_MEMBERSHIP_FILE` (`focus_picks._inject_into_shared`) and logs
`pick_feedback` — reuse, do not reimplement. **Recorded decision: the R2 M5
adoption gate (session VWAP / prev-day extreme) does NOT apply to weekend swing
adds** — it is an intraday-session gate; if the trader later wants a swing-grade
gate, that is a spec amendment, not an improvisation. Nothing in this tab removes
entries; membership provenance is surfaced in the adopt status line only (a
membership viewer is a WISHLIST candidate).

## 8. Files, fenced list, packaging

New: `scripts/weekend_strength.py` (pure; statically imported by the service so
the frozen bundle collects it), `scripts/ui/services/weekend_prep_service.py`
(QObject; no QTimer; signals `stateChanged/boardChanged/statusChanged/
weekAheadReady`), `scripts/ui/panels/weekend_prep_panel.py` (stepper rail +
QStackedWidget), tests (`test_weekend_strength.py`,
`test_weekend_prep_service.py`, `test_weekend_prep_panel.py` qt-marked).

Modified: `scripts/ui/app.py` — first commit replaces the three index-aligned
parallel structures (pages order / nav_items / `_select_page` titles) with one
`PAGE_SPECS` list, fixing the live bug shipped with the Strength Board (titles
shifted from index 3; Settings click raises IndexError); a later commit adds the
Weekend Prep entry. `scripts/selftest.py` — add `market_prep.orchestrator` (and
any other lazily imported market_prep modules found at build time) to
`LAZY_ENGINE_MODULES`. `scripts/project_paths.py` — `WEEKEND_PREP_STATE_FILE`.

Fenced/ask-first: `scripts/strength_scan.py` (not edited),
`scripts/autopilot_core.py` (read-only use of `_default_downloader` and the chunk
size), `review_policy.json` (untouched). Frozen selftest count moves; a real
rebuild + frozen selftest run is required before merge.

## 9. Build order (commit-sized; tests per step)

1. app.py `PAGE_SPECS` bugfix (standalone, merge-worthy on its own; regression
   test covers every index incl. the last). 2. M5 characterization fixture.
3. `weekend_strength` + tests (forming-month drop incl. a day-1 case, 51-bar
refusal per TF, percentile-before-filter order, short mirrors, parity with
`strength_scan` functions). 4. Any `journal_feed` weekend helpers not already
landed in R7. 5. Service (state load/save/prune, weekend-id derivation, step
transitions, single-flight with injected fake downloader, last-good on failure;
tests assert no QTimer exists and nothing runs without an explicit call).
6. Panel shell + steps 1–2. 7. Step 3 + auto-tag sub-pane. 8. Discovery boards +
Adopt (tests mirror `test_strength_board_panel.py`: exact FocusService args,
duplicate add tolerated, no removal path exists). 9. Week ahead + selftest
additions. 10. Register the tab. 11. Frozen rebuild + selftest. 12. Governance
close-out.

## 10. Exit gates

Deterministic: full suite green from the then-current baseline, smoke 7/7,
frozen rebuild + selftest at its new count; all new tests offline (injected
downloader, frozen now; deterministic across two runs).

Live proof — one real weekend run, observing:
1. Desk boots on a weekend with the tab present and zero network/scan activity
   until a button is pressed (log-verified).
2. Zero IB traffic across the whole routine.
3. H1/D1/M1 boards each refreshed manually; per-TF wall clock recorded here;
   monthly shows short-history names honestly missing (measured < offered).
4. The monthly board contains no current-month bar (spot-check one symbol).
5. One real Adopt verified in all four stores: Focus swing, `swinglongs.txt`,
   membership file, `pick_feedback.jsonl` with `origin="weekend_prep"`; nothing
   removed anywhere.
6. One auto-tag confirm and one correction verified in `trade_annotations` /
   `tag_corrections`.
7. Walk-away runs windowed to the reviewed week only.
8. Week-ahead renders; its fetches happen only on the button press.
9. App closed mid-routine and relaunched: progress restored.
10. Trader confirms board character per TF before the §5 filters count as proven.

## 11. Risks and open items

- yfinance `1h` caps at ~730 days (we request 3mo — safe). **Probed once at build
  start, 2026-08-15, read-only, one 187-symbol chunk from `universe_all.txt`
  (1,506 symbols):**

  | Timeframe | Request | Wall clock | Rows returned | Extrapolated full universe (8 chunks) |
  |---|---|---|---|---|
  | H1 | `1h` / `3mo` | **3.1 s** | 441 | ~25 s |
  | D1 | `1d` / `1y` | **2.0 s** | 251 | ~16 s |
  | M1 | `1mo` / `6y` | **2.3 s** | 72 | ~18 s |

  Comfortably inside the ≤30–60 s per-timeframe expectation, and well under the
  M5 board's measured 27.6 s — each symbol returns far fewer rows here than
  `5m`/`5d` does. Row counts are exactly what the periods imply (3 months of
  ~7 hourly bars/day; 251 sessions; 72 months), so nothing is being silently
  truncated.

  Two things the probe settled that the spec had assumed:

  1. **Columns are `(symbol, field)`** — symbol on level 0. A first reading that
     counted level 1 returned 6 and looked like a truncated batch; it was the
     field names. Recorded because the same mistake would misread a real
     shortfall as a healthy fetch.
  2. **The monthly frame's last row is the current, in-progress month**
     (`2026-08-01` on a probe run 2026-08-15). This is the exact condition the
     month-identity drop exists for, confirmed against live data rather than
     inferred.
- Monthly bars are stamped on the 1st and the latest row is the in-progress
  month — the month-identity drop handles it; a listing-month first bar is short
  but sits far behind the 51-bar minimum.
- Fetch time/memory: expected ≤ ~30–60 s per TF (fewer rows per symbol than the
  M5 board's 5m/5d); revisit only if monthly ever moves to `period="max"`.
- Swing-grade adoption gate question recorded in §7.
- Away-report archive depth (~3 trading days at `archive_keep=30`) limits any
  future narrative week-in-review; out of scope for v1, noted for WISHLIST.

## Retained joins — BUILT 2026-08-18

The R8 release-candidate review kept two joins as explicit future scope and did
not claim them. Both landed under the trader's 2026-08-18 integration redirect.

**Week in review — the RS/RW extremes join.** `_read_rrs_week` reads
`rrs_strength_extremes.csv` for the reviewed week and folds it to one row per
(bucket, symbol) carrying days seen, sightings and the best reading. Folding is
the point: the log is one row per sighting, so a name that led the tape all week
would otherwise bury every other name in the step. "Best" is direction-aware —
the weak bucket's best reading is its most *negative* one. The step shows a
capped number of rows per bucket and **prints what the cap dropped**, because a
silent top-N reads as "that was all of it".

**Focus review — picks joined to their outcomes.** `_join_focus_week` joins
`human_focus_daily_picks.csv` to `human_focus_outcomes.csv` on
(trade_date, symbol, side) and renders **one row per pick** carrying H1/H3/H5/H10
returns and the matured-horizon count. The v1 step listed picks and outcomes as
separate rows, so a name appeared twice and the table could not answer "how did
this pick do".

Two honesty rules are load-bearing and are tested as such:

- **A horizon that has not matured is BLANK, never 0.00%.** The review is read on
  a Saturday; a fabricated zero is a false lesson about a trade that is still
  running.
- **An outcome with no matching pick snapshot is still shown, and marked.** It is
  evidence that a pick existed on a day whose snapshot did not persist; dropping
  it would quietly narrow the week.

Both readers stay read-only and forgiving, like every other weekend-prep reader:
a missing or unreadable CSV is a quieter week, not an error worth stopping the
routine for. The one-real-weekend live gate in §10 is unchanged and still owed.

## The last deferred joins — BUILT 2026-08-24 (packet W2)

§6's DEFERRED block is now empty. Three streams landed together, all under the
same reader idiom the AI-P1 repair established: **address a home-folder store by
its named `project_paths` constant, never by composing a filename**, because
this step rendered a blank Focus table on the live desk for six days from
exactly that mistake — and its fixture encoded the same wrong assumption, so
nothing caught it.

**Focus pick review — the picks' own forward record.** `_read_focus_performance`
reads `human_focus_performance.csv`: cohort, side, horizon, n, win rate, mean
and median side-adjusted return, profit factor, the symbol and session counts,
and the session-block interval. The cohort tables beside it are the two
judgement mirrors — what was thrown away, what was endorsed; this is the thing
being judged. It renders the whole rollup rather than a week slice, for the
reason recorded in §6, and prints its as-of stamp so a stale rollup reads as
stale. A blank stays blank; an absent interval carries its reason.

**Focus pick review — the week's verdicts.** `_read_pick_feedback_week` reads
`pick_feedback.jsonl` through `pick_feedback.load_pick_feedback` (one loader per
file, never a second JSONL parser on a panel) and scopes it on `trade_date` —
the session the verdict is ABOUT, never `ts`, which is when it was typed. A
verdict entered on Saturday about Friday belongs to Friday. The pane tallies the
verdicts, says they are opinions rather than outcomes, and prints what its
200-row cap dropped.

**Week in review — the sector and industry extremes.** `_read_rrs_group_week`
folds `rrs_group_strength_extremes.csv` per `(group_type, group_key)`. Two
things differ from the symbol fold, both forced by the file: the group log
records **no bucket** (the writer emits the top and bottom of each list with
identical columns), so both extremes are printed and the sign is what the reader
reads — nothing invents a direction the file never recorded; and the key carries
the group type, so a sector and an industry ETF sharing a ticker never fold into
each other. The cap prints what it dropped, like the symbol stream.

**Also removed:** `_read_focus_week`, dead since 2026-08-18 and still resolving
its CSVs under the home root. AI-P1 fixed the live reader and left the copy that
encoded the defect; a future edit reaching for the nearest-looking helper would
have restored it.


## Round R4 Part A (2026-09-02) — what V2 claimed here and what it left

R8 built this tab; Phase 0.14 packet V2 rebuilt its front (decision 0016
answer 10: *"one Refresh for the whole tab, a short verdict card on top, tables
not prose, ten visible rows"*). R4 A13, A14, A15 and A18 finished four of those
four, and the gap in each case was the same shape — the rule was written down
and applied to one place.

**One Refresh means the CLICK reads nothing.** V2's own test measured the click
with every page's `reload` monkeypatched to an `append`, so what it timed was the
`for` loop. Underneath it, `WalkawayPage._reload_review_data` called
`journal_feed.week_trades` synchronously — a journal query over a week of trades,
**775 ms**, on the thread that draws. It runs on the page's own `_ReadWorker`
now, with its own handle so a read and a tag toggle cannot cancel each other, and
the 50 ms assertion is measured against the REAL click with the reads stubbed at
the WORKER boundary.

**`DiscoveryPage` had no `reload` at all.** `_StepPage.reload` is a no-op, so
"Refresh everything" called it, counted Discovery as started, named it in the
"Building:" line and built nothing — while its **six** per-table Refresh buttons
(two per timeframe, three timeframes) stayed in the layout, which is the exact
complaint answer 10 was answering. V2's button test could not see them because it
looked for a `refresh_button` attribute this page does not have. `reload` now
refreshes all six boards through the service's own single-flight, and the buttons
are out of the layout and kept as objects.

**Ten rows is one constant now.** `TABLE_TEN_ROWS_PX` (260 px), applied through
`_ten_row_table` to every `QTableWidget` on the tab. V2 set a bare `260` on
`tag_week` and left the other eleven tables at whatever the layout gave them.

**"Confirm all shown" could confirm a blank.** `JournalStore.confirm_tags` only
flips the LANE, so a `needs_review` row carrying no tag was written `confirmed`
over nothing — and the nightly `journal_auto_tag` then found a closed trade with
no confirmed tag and marked it `needs_review` again, every night, forever. 132 of
the live rows are that shape and no button on this page could give one of them a
tag. Untagged rows are skipped and COUNTED, the note names them, and **"Edit
tag..."** writes the trader's own wording through `journal_feed.correct_auto_tag`
and then confirms it. The SQLite writes moved onto a worker: "Confirm all shown"
is one UPDATE per row.

**The verdict card's take rate was wrong, not merely thin.** `take_rate_line`
computed `shown = takes + skips + rejects`, and `build_review_learning_state`
publishes **neither** `skips` nor `rejects` — `aggregate_dimensions` returns
`episodes`, `shown`, `takes`, `overall_take_rate` and `dimensions`, and that is
the whole top level. The denominator was `takes + 0 + 0`, so the card read
"100% of 94 shown" on a week whose real answer was 30% of 318. It reads `shown`
and `overall_take_rate` off the state now, and its tests are fixtured on REAL
states built through `build_review_learning_state` — a hand-written dict is what
let this ship, because it can carry any key the code happens to ask for.
