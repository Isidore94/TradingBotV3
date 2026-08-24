# Chart Review workspace + trader decision capture

Document role: **active contract and implementation record**, subordinate to the root
roadmap.

Historical key: this entered the former roadmap as item 13d; remaining acceptance is
now in `plan.md` P3.1.

Status (reconciled 2026-08-10): **A1–A5 IMPLEMENTED + GREEN** on
`testing-week-2026-08-10`. The capture layer/workspace, shared D1+M5 chart,
crosshair/OHLCV and source strip, paint-line groups with stable IDs, and click-to-arm
through the one `PriceAlertService` writer have landed. LIKE/veto/note remains
annotation-only and grants no Focus, watchlist, or alert privilege. Live-session
acceptance remains open in `plan.md`.

---

## 1. Why this packet exists

Everything else in the program measures outcomes. Nothing measures *judgement*.
The scans surface candidates and the trackers grade what happened next, but the
step in between — the trader looking at a chart and deciding "not today, that
trendline is coming in" — has never been written down. That decision stream is
the most valuable dataset the program can collect, because it is the only one
that cannot be reconstructed after the fact.

So the design target is not a better chart. TradingView and TC2000 stay open
for deep TA and are not being replaced. The chart here only has to be good
enough to keep the trader in the chair. **The capture rail is the product.**

The single hard design constraint, from the trader:

> Every capture action is under five seconds, and reachable without the mouse.

A rail that costs ten seconds per decision gets used twice and abandoned, and
an abandoned rail produces no dataset at all. Every layout and keybinding
choice below follows from that.

**2026-08-20 — the rail is portable, the contract is not.** The Alert Center
moved its rail off the review pane's vertical stack and onto a **Capture** tab,
because in that column the rail's ~600px was costing the charts the space they
exist for. Nothing about what the rail IS changed: it is still a recorder that
writes annotation rows and has never muted, suppressed, scored, gated, ranked,
alerted or written a watchlist (§7). The five-second/no-mouse constraint above
moved WITH it rather than being spent on the layout — a `QShortcut` bound
inside a hidden tab page never fires, so that host rebinds
`CaptureRail.action_shortcuts()` at panel scope and each key raises the Capture
tab before arming the same handler. Hosts that have the room
(`SymbolSnapshotDialog`, this workspace) keep the docked rail unchanged.

---

## 2. Invariant compliance (plan.md sec 5)

| Invariant | How this packet complies |
|---|---|
| Decision-support only; never add order execution | The hypothetical stop records a price and a side. No order path is imported, referenced, or reachable. The rail's own status line says "no order placed". |
| Legacy champions untouched; shadow engines never influence live decisions | No detector, scoring, or alert file is edited. The annotation stream has no consumer that feeds any of them. |
| No detector/scoring behaviour change without golden fixtures | No detector or scoring code was touched. The one edit to a tracking file (`human_focus_tracking.py`) is additive and pinned by a characterization test proving byte-identical aggregation for existing cohorts, with a sensitivity control. |
| Never swap `calc_anchored_vwap_bands`' σ formula | Not touched. |
| Completed bars only for state transitions | No bar consumption in this packet yet. |
| User-entered watchlist names are never auto-removed | The mirror is enforced here: a lookup never *adds* one either. See §5. |
| One component owns each timer/thread/job/mutable shared export | The desk GUI is the sole writer of `trader_annotations.jsonl`. `ui.annotations.veto_cohort` is the sole writer of the three `veto_cohort_*` files. No new timers or threads exist in this packet. |
| Point-in-time research; timestamps carry explicit timezones | `created_at` is always tz-aware; a naive timestamp is given the machine offset rather than written bare. |
| `review_policy.json` ranks and annotates only — no suppression field | Untouched, and the same principle is applied here: see §3. |

### The hard boundary

**Annotations are analysis-only evidence.** They must never mute, suppress,
score, gate, or alert. No consumer of this stream ships in this packet beyond
the veto cohort tracking in §6, which is capture-side: it lets forward returns
accrue so a veto becomes *gradeable*, and changes nothing the desk decides
today.

This is the same discipline as `review_policy.json` having no suppression
field. A stream that starts by muting alerts stops being a record of what the
trader thought and becomes a control surface, and then it can never be trusted
as evidence again.

---

## 3. Storage and schema v1

`trader_annotations.jsonl`, in the shared home folder — the same storage class
as `alert_review_events` and `pick_feedback`: small, human-relevant, syncs
across machines, readable by an AI.

```
{schema_version: 1,
 event_id,                 # uuid4 hex
 event_type,               # veto | like_claim | hypo_stop | note
 symbol,
 session_date,             # market date
 created_at,               # ISO-8601, ALWAYS tz-aware
 source: "chart_review",
 reason_code?,             # veto: a code from the versioned vocabulary
 vocab_version?,           # veto: which vocabulary version produced it
 claimed_setup_id?,        # like_claim (required); optional elsewhere
 stop_price?, side?,       # hypo_stop (both required)
 last_price?,
 ref_level_id?,            # the painted level the capture referenced
 ref_level_family?,
 note?,
 timeframe}
```

**Extensible, never renamed.** Later schema versions add fields. A field that
exists at v1 keeps its name and meaning forever, because rows already written
carry it.

### Storage rules

- **Append-only.** Every write opens the file in append mode. Nothing
  truncates, rewrites, reorders, or deletes a row. A mistaken capture is
  corrected by a later row, never by editing an earlier one.
- **One row, one line, one fsynced write.** A row is written inside the
  machine-local writer lock as a single bounded (4096-byte) write and fsynced
  before `SAVED` is shown, so cooperating writers never interleave and
  "saved" survives a power cut. Notes are capped at 2000 characters, which is
  what keeps the row inside that bound. This is confinement, not atomicity: a
  crash mid-write can still tear that row's tail, but the appender heals a
  torn tail with a newline before the next write, so a torn fragment can only
  ever cost its own row — never the row after it — and the reader skips
  exactly the torn line.
- **One writer.** The desk GUI owns the file.
- **Failures are visible.** A write that does not reach disk returns false and
  the rail turns red and says `NOT SAVED`. A trader who believes a decision was
  recorded when it was not is worse off than one who knows it failed.

---

## 4. The veto vocabulary

Lives in versioned JSON under `scripts/ui/annotations/vocabularies/`, which the
PyInstaller spec mirrors into the bundle (every non-`.py` file under
`scripts/ui` — but see §9 for a caveat about where that spec currently lives).

**v1** (`veto_reasons_v1.json`), hotkeys 1–9:

| # | Code | Note |
|---|---|---|
| 1 | `incoming_trendline` | |
| 2 | `overhead_horizontal` | |
| 3 | `support_resistance_cluttered` | |
| 4 | `sector_mate_earnings_pending` | |
| 5 | `too_extended_from_base` | |
| 6 | `volume_dry` | |
| 7 | `earnings_too_close` | |
| 8 | `spread_liquidity` | |
| 9 | `other` | **note required** |

### Why a file per version, not a Python constant

Every row stamps the `vocab_version` it used, and each version keeps its own
file. A later vocabulary ships as `veto_reasons_v2.json` *alongside* v1, so a
row written under v1 stays interpretable against exactly the list that produced
it. Editing a shipped version in place is the one change that would silently
rewrite history, so the loader validates the contract instead of trusting it:
unique codes and hotkeys, a declared version matching the filename, codes
restricted to `[a-z][a-z0-9_]{2,47}`, and no reserved `veto_`/`focus_` prefix
(those would collide with cohort source names).

A missing or malformed vocabulary is a **packaging defect, not a runtime
condition to paper over**: the loader raises, and the rail disables the veto
action and displays the reason. The alternative is writing reason codes no
later analysis would recognise.

---

## 5. Ticker lookup — and what it must never do

The search box loads **any** symbol, not just scanned or watchlist names.

plan.md sec 5 says user-entered watchlist names are never auto-removed. The
mirror of that invariant is the one enforced here: **looking at a name must
never add one.** A lookup that quietly wrote to `longs.txt` or the
`CandidateRegistry` would put symbols into the scan universe the trader never
chose, and the next writer to reconcile those files would be deciding what to
do about entries nobody made.

`ui/services/symbol_lookup.py` therefore has **no writer dependencies at all** —
it imports no watchlist path, no registry, no focus store. The only thing it
persists is a machine-local recents list under `%LOCALAPPDATA%`, deliberately
*not* the shared home where the watchlists live.

Three tests hold that line: the module imports no writer (AST check), the
recents file is not under `PERSISTENT_DATA_DIR`, and a full lookup cycle writes
its own recents file and nothing else.

Adding a looked-up name to a watchlist or focus list stays an explicit,
separate trader action through the surfaces that already own those files.

---

## 6. Veto forward-tracking

Vetoed names enter forward tracking as cohort `human_focus_veto`, with one
sub-cohort per reason (`human_focus_veto_incoming_trendline`, …), graded by the
**existing** human-focus outcome math at 1/3/5/10 sessions.

### They live in their own files, and that is not incidental

`human_focus_daily_picks.csv` is keyed `(trade_date, symbol, side)` with **no
source column**. A veto row for a name that was also a focus pick that day
would occupy the focus row's slot and suppress it — the snapshot skips a key it
already has. Sharing the file would have silently cost data.

So veto rows go to `veto_cohort_picks.csv` / `_outcomes.csv` /
`_performance.csv`, and `update_human_focus_outcomes` is reused as-is via its
existing path parameters.

The one edit to `human_focus_tracking.py` names the new cohort family so veto
rows do not fall into the legacy `human_focus_pick` catch-all — a veto is the
opposite of a pick, and averaging the two would make both unreadable. It is
additive and pinned by a characterization test.

### Sideless vetoes are skipped, not guessed

The forward return is side-adjusted, and `human_focus_tracking` reads a blank
side as `LONG`. Guessing would manufacture a directional claim the trader never
made, so a veto with no side is **counted and skipped**, and the count is
returned so the caller can say so.

### Grading is wired, as of 2026-08-20

`update_veto_cohort_outcomes()` had **zero callers** from the day it shipped
until now: picks accumulated on every veto commit and nothing graded them.
`ai_jobs.cohorts.run_veto_cohort_grading` is the caller, registered as a
fourth slot on the existing overnight runner (`veto_cohort_grading`, 5-minute
reserve, appended — the runner never reorders). It is **deterministic**: no
model, nothing transmitted, two CSVs out.

Contract:

- **Idempotent in the sense that matters.** A re-run the same night changes
  exactly one column, `updated_at`, and nothing measured — not the row set,
  the ordering, an entry price or a forward return. Byte-identical is
  deliberately *not* claimed: a provenance stamp saying when grading last ran
  is correct behaviour. A fully matured pick is never recomputed at all.
- **A failure never destroys the last verified artifact.**
  `_write_csv_rows` stages to `<name>.tmp` and `os.replace`s, so a half-written
  outcomes file cannot land; an exception mid-grade leaves both CSVs byte-identical.
- **Sideless rows are counted and named, never graded** — see below.
- The forward-return metric is **close-to-close only**. It does not read
  volume or AVWAP bands, so the known IBKR/Yahoo volume-unit defect
  (~17% of stored symbols) does **not** reach these numbers. Verified
  2026-08-20 by inspection: `human_focus_tracking` contains no reference to
  volume, AVWAP or bands.

### The cohort key carries its vocabulary version

`veto_cohort_source(reason_code, vocab_version)` produces
`veto_v<version>_<code>`. A reason code is a permanent identifier *within* one
vocabulary, but that guarantee is a rule written in the vocabulary JSON, not
something the cohort module can verify — and the cost of trusting it wrongly is
two different judgements averaged into one number that reads as evidence.

An **omitted** version yields the historical unversioned `veto_<code>`. That is
what keeps rows already in `veto_cohort_picks.csv` valid: they were written
before the key carried a version, they are never rewritten, and they keep
grading in the cohort they were filed under.

**Known consequence, recorded rather than hidden.** Eight of the nine v2
reasons are byte-identical to their v1 entry (same label, same hotkey); only
`compressed` is new. So this splits eight cohorts that could legitimately have
been pooled, halving the sample per reason across the bump. It is the right way
round — the version is in the key, so pooling stays recoverable by analysis,
whereas a wrongly pooled cohort is not — but on day one, with 66 annotation
rows, it is a real cost and analysis should expect it.

**And on 2026-08-21 it was recovered.** v3 added "SMA incoming" and changed
nothing else, which made the same cost due a second time; the trader chose to
bump *and* pool. `canonical_veto_cohort(source)` maps a cohort source to the
earliest version carrying an identical reason DEFINITION — code, label, hint
and note rule, all four — so v1/v2/v3 `volume_dry` grade as one
`veto_v1_volume_dry`, while `compressed` (first in v2) and `sma_incoming`
(first in v3) keep their own cohorts, and v1's replaced
`support_resistance_cluttered` is never folded into a survivor.

Three properties make this safe to have written down as reversible:

1. **It is a reading, not a rewriting.** Pick rows and outcome rows keep the
   exact `vocab_version` they were captured under. Only the derived rollup —
   `veto_cohort_performance.csv`, regenerated every run — is grouped by the
   canonical source, through `_rebuild_pooled_performance`.
2. **It reuses the same math.** The pooled rollup is built by
   `build_human_focus_performance_rows`, the same function the delegate uses,
   so pooling can never introduce a second definition of a win rate.
3. **It fails to the old behaviour.** A vocabulary that cannot be read yields
   an empty map and no pooling — cohorts stay split, which is what happened
   before this existed and is never a wrong number.

The rule for the next bump: *additive* bumps pool automatically, because the
carried-over definitions are identical. A bump that **changes** a label, hint
or note rule deliberately splits that reason, and that is the signal that its
meaning moved.

### Reading the cohort

`trader_judgement` is an **opt-in** AI evidence scope (`ai_summary.py`), not in
`DEFAULT_SCOPES` or `TICKER_BRIEF_SCOPES`. Sources in funding order:
`veto_cohort_performance.csv`, `veto_cohort_outcomes.csv`, then
`trader_annotations.jsonl` last. Run it on demand with
`run_ai_jobs.py --scopes trader_judgement`. Two machine-written caveats travel
with it: the like+claim control offers a bounded picklist — Main swing plus the
three post-earnings families and `second_dev_breakout` since 2026-08-21, so a
claim can only ever be one of those — and "Veto D1 — but M5 today" writes an
ordinary veto row, so some vetoed names were traded the same day.

Since 2026-08-24 the picklist caveat is **derived, not retyped**: the offered
list moved to `ui/annotations/setup_claims.py` (Qt-free; the rail re-exports
it) and `ai_summary.scope_caveats()` builds the sentence from
`offered_setup_claims()` at package time, so admitting a claim family updates
the caveat by itself. A registry the headless path cannot read degrades to a
stated UNKNOWN caveat, never to a remembered list — the hand-maintained
version went stale across the 08-21 widening while its content-pinning test
stayed green, which is the failure mode derivation exists to close
(`docs/analysis/AI_LAYER_REVIEW_2026-08-24.md` AI-P5). The veto-day-trade
caveat stays a constant: it describes a verb, not a list.

Nothing reads these files back into a detector, a score, an alert, a watchlist,
Focus, the review queue, or `review_policy.json`.

---

**Historical note (superseded above):** `update_veto_cohort_outcomes()` exists but
**was not wired to any timer, scan,
or market-hours path by this packet.** It reads daily bars off disk. Deciding
when to run it is a separate change.

---

## 7. Like + setup claim — a recorded judgement, nothing more

A like writes **one annotation row** carrying the claimed setup id, and that
is all. There is deliberately no second likes store.

**The why is required (R9.2, trader 2026-08-22: "if I like a chart I should
always be prompted with why").** The claim key or a double-click selects the
setup and moves focus to the why field; Enter commits; an empty why does not
commit and the chart stays. This is the veto vocabulary's `note_required`
mechanic applied to every claim. The why lands in the row's existing `note`
field — no schema change. It is required rather than offered because the
`dislike` rows are the counter-example: 31 of the most information-dense
strings the trader ever wrote, captured under a field nothing insisted on, and
then dropped entirely by a parser mismatch. Relaxing this is one constant.

**A like advances the queue and does nothing else (R9.2).** It previously
routed through the Alert Center's "Not today" verb, which retires the chart
*and parks the symbol for the day*. Measured over 2026-07-24…08-21: 40 of 52
`like_claim` rows put their symbol on `alert_center_ignored_symbols.txt`; a
parked symbol also stops emitting `d1EventRecorded`, so on an AWAY day a LIKE
silently dropped that name from the hourly D1 phone push; and because the
route wrote `remove_today`, `review_learning.REJECT_ACTIONS` scored every like
the trader ever filed as a dismissal. A like now emits
`AlertChartReview.likeAdvanceRequested`, the panel records `like_advance`
(which `TAKE_ACTIONS` reads as positive engagement) and advances — the ignore
set, the symbol's other queued alerts, and any auto-adopted Focus pick are all
untouched. **The veto's retire-and-park path is unchanged.** Liking a setup is
the opposite of being finished with the symbol.

An earlier draft of this section routed likes through `FocusService.add()` to
reuse the existing `pick_feedback` machinery. That was wrong by this
document's own rules: `FocusPickStore.add` writes Focus state AND injects the
symbol into a swing watchlist, and Focus membership bypasses alert feed gates
in the Alert Center — a capture surface silently changing live scanning and
alerting (§2's boundary, and §5's "adding a looked-up name to a watchlist or
focus list stays an explicit, separate trader action"). The trader clicking
LIKE is judging a chart, not asking for alert privileges; if they want the
name in Focus, the Focus surfaces that own those files are one click away.
The `chart_review` entry in `PICK_ORIGINS` remains documented but dormant.

If likes ever need forward-return grading, they get it the way vetoes do
(§6): a capture-side cohort file graded by the existing outcome math, with
zero influence on live lists.

Claims are read from `setup_docs.all_setup_docs_by_group()` — the same families
the rest of the system names — including the study/measured-only group, since a
claim on a study setup is precisely the evidence that decides whether it ever
graduates (plan.md sec 7). One extra claim, `none_of_these`, is the honest
answer when the chart looks good for a reason the registry has no name for; a
run of them is itself a finding.

---

## 8. Keyboard map

| Keys | Action |
|---|---|
| `Ctrl+L` | focus the lookup box |
| `Alt+V`, then `1`–`9` | veto — armed, chosen, written (`other` stops for its note) |
| `Alt+K` | like + setup claim |
| `Alt+S` | hypothetical stop |
| `Alt+N` | note |
| `Alt+E` | slide the Setups drawer in/out |

`Alt`+letter rather than bare letters: the rail is full of text inputs, and a
bare `v` has to stay a `v` when the trader is typing a note. Digits are scoped
to the reason list widget for the same reason.

---

## 9. Implementation sequence and remaining acceptance

The chart data path is being rebuilt in parallel — `ui/services/chart_data_service.py`
and `ui/services/bar_cache.py` move snapshot building off the GUI thread and
remove a synchronous home-folder read from the paint path. Building charts here
against the old synchronous loader would have created a **second owner of the
chart data path** (sec 5) and a guaranteed conflict in `candle_chart.py`.

The sequence was:

- **A3** D1 (2y+) and M5 charts, crosshair/OHLC readout, IBKR-streaming-while-
  focused with a loud yfinance fallback banner. **LANDED 2026-08-09** through the
  shared `SymbolSnapshotWidget` worker path.
- **A4** the paint-lines toggle (daily SMAs, D1 horizontals, D1 trendlines,
  prev-day H/L, AVWAP bands) with stable level ids and click-to-select.
  **LANDED 2026-08-09** — see below.
- **A5** click-to-set price alert through the existing `PriceAlertService` API.
  **LANDED 2026-08-09** by routing level selection through `AlertCenterPanel`; Chart
  Review itself still has no independent alert writer.

### A4 as built (2026-08-09)

The chart data path landed, so A4 followed it. Three new pieces and one rule:

- `scripts/chart_levels.py` builds the `levels` payload — D1 horizontal S/R
  (`hv_horizontal` + `cloud_flat` from the home-folder level store),
  prev-day H/L (computed from the snapshot's own bars, so nothing imports
  `bounce_bot_lib`), and the projected D1 trendline.
- `CandleChart.set_levels` draws them: an infinite horizontal line per level,
  a curve for a sloped one, both added with `ignoreBounds=True` so a level can
  never stretch the y-range to reach itself. Clicks hit-test in **screen
  pixels** and emit `levelSelected(id, family, price)`.
- `ui/widgets/paint_lines_button.py` + `ui/services/paint_lines_prefs.py` are
  the toggle: six groups (SMAs, EMAs, AVWAP bands, D1 S/R, prev-day, trendline),
  machine-local under `%LOCALAPPDATA%`, every group defaulting ON. It governs
  the pre-existing SMA/AVWAP overlays too, by label.
- **The rule**: every level read happens on the `ChartDataService` worker and
  rides `snapshotReady`. Nothing on the paint path opens a file. That is why
  the build lives in the service rather than in a widget.

Since `ref_level_id` / `ref_level_family` were already in the schema (§3), the
capture rail needs no schema change to reference a painted level:
`SymbolSnapshotWidget.d1LevelSelected` and `.selected_d1_level()` are the push
and pull sides. **A4 wires the signal, not the rail** — connecting it into the
capture write would mean editing `alert_center_panel.py`, which is fenced.

Trendline caveat: the record only exists for symbols that reached
priority-candidate status in the last scan, so a looked-up name usually has
none. Full findings and a measurement tool: `docs/D1_TRENDLINE_SURVEY.md`.

The schema already carries `ref_level_id` / `ref_level_family`, so painted-level
references need no schema change when A4 arrives. The chart area shows a stated
placeholder rather than an empty frame, and the feed-provenance strip has its
permanent slot reading `none` until a feed exists.

### Packaging status

`packaging/tradingbotv3.spec` mirrors every non-`.py` file under `scripts/ui`,
which covers the vocabulary JSON. The spec-drift test and frozen self-test are now on
the testing-week branch; the real desk build passed 29/29 after the bundle/self-test
rosters were reconciled. Future asset/package changes must follow
`packaging/README.md`.

---

## 10. Test coverage

| Area | File |
|---|---|
| Schema v1, validation, append-only, concurrent-write atomicity, note/row caps, corrupt-line tolerance | `tests/test_trader_annotations.py` |
| Vocabulary versioning: newest-by-default, old versions still loadable, every contract violation fails closed | `tests/test_trader_annotations.py` |
| Setup claims sourced from the registry, study families included | `tests/test_trader_annotations.py` |
| Veto cohort rows, sideless skip, first-wins, merge idempotence, never-removes | `tests/test_veto_cohort.py` |
| **Cohort isolation characterization + sensitivity control** | `tests/test_veto_cohort.py` |
| Focus picks file provably untouched by a veto merge | `tests/test_veto_cohort.py` |
| Lookup never writes watchlists (3 angles), symbol normalization | `tests/test_chart_review_workspace.py` |
| Rail wiring, like-writes-one-annotation-row-and-nothing-else, no focus writer importable, failure surfaced, drawer default hidden | `tests/test_chart_review_workspace.py` |
| Setups drawer: off-GUI-thread bounded snapshot read, setup-id keys rendered as symbols, byte ceiling refused | `tests/test_chart_review_workspace.py` |
| Page/nav registration alignment | `tests/test_chart_review_workspace.py`, `tests/test_qt_focus_panel.py` |
