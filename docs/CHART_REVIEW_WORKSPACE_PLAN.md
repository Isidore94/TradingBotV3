# Chart Review workspace + trader decision capture (plan.md item 13d)

Status: **capture layer + workspace shell IMPLEMENTED + GREEN** on branch
`chart-review-workspace` (suite 2158 passed, 7 subtests). The chart itself,
the paint-lines toggle, and click-to-set price alerts are **deliberately not
built yet** — see [Deferred](#deferred-and-why) below.

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

`update_veto_cohort_outcomes()` exists but **is not wired to any timer, scan,
or market-hours path by this packet.** It reads daily bars off disk. Deciding
when to run it is a separate change.

---

## 7. Like + setup claim — a recorded judgement, nothing more

A like writes **one annotation row** carrying the claimed setup id, and that
is all. There is deliberately no second likes store.

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

## 9. Deferred, and why

The chart data path is being rebuilt in parallel — `ui/services/chart_data_service.py`
and `ui/services/bar_cache.py` move snapshot building off the GUI thread and
remove a synchronous Drive read from the paint path. Building charts here
against the old synchronous loader would have created a **second owner of the
chart data path** (sec 5) and a guaranteed conflict in `candle_chart.py`.

Deferred until that lands:

- **A3** D1 (2y+) and M5 charts, crosshair/OHLC readout, IBKR-streaming-while-
  focused with a loud yfinance fallback banner.
- **A4** the paint-lines toggle (daily SMAs, D1 horizontals, D1 trendlines,
  prev-day H/L, AVWAP bands) with stable level ids and click-to-select.
- **A5** click-to-set price alert through the existing `PriceAlertService` API.

The schema already carries `ref_level_id` / `ref_level_family`, so painted-level
references need no schema change when A4 arrives. The chart area shows a stated
placeholder rather than an empty frame, and the feed-provenance strip has its
permanent slot reading `none` until a feed exists.

### Packaging caveat

`packaging/tradingbotv3.spec` mirrors every non-`.py` file under `scripts/ui`,
which covers the vocabulary JSON — **but that spec exists only on the unmerged
`integration-test` branch (commit `9037c5f`), not on `main`.** Whoever merges
it must confirm the `datas` rule still covers `scripts/ui/**/*.json`. No spec
edit was needed or made by this packet.

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
