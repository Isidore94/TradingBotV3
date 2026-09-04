# M5 Focus adoption discipline and the M5 strength board — packet R2

Status: **BUILT 2026-08-15, live proof owed** — `plan.md` Phase 0.5 **R2**.
Authorized by the trader on 2026-08-15; ranked **second** in the Phase 0.5 build
order, then directed to build ahead of P0.7 (the same redirect R1 got).

Built on branch `phase05-r2-focus-gating-strength-board` (cut from R1, with the
R1.1 repair merged forward). Ask-first approval was taken for every fenced file
in §6 before the first edit. Deterministic verification: **2865 passed / 19
subtests, smoke 7/7, source selftest 30/30**, all exit 0. The §8 live proofs are
**owed** — nothing here has been observed on a live session.

## 2026-08-15 trader decisions recorded here

- Strength-board universe: **the existing universe** (`universe_all.txt`, 1,506
  names) — not a new broader US universe build.
- "Not today" on an auto-adopted M5 Focus pick removes **exactly that M5 entry for
  that side** — never the swing-category entry, never the other side, never a
  user-typed name.
- **Adoption-time gate reads a stored verdict** written by the 30-minute staging
  refresh, refusing anything failing, missing, or older than 45 min (1.5× the
  refresh cadence). The Alert Center adopts on the GUI thread and a staged pick
  is on no watchlist yet, so BounceBot holds no bars for it.
- **The desync repair is a request, not a direct write.** BounceBot files it;
  the Alert Center's existing poll performs the removal, preserving one owner
  per mutable store.
- **A user-typed name whose scan line was cut is surfaced, never removed** —
  Alert Center status line plus the log.
- **An evicted pick may re-propose the same day if it re-qualifies.** The queue
  says what qualifies now, not what once did.
- **"Not today" carries a distinct label on an auto pick** (`✕ Not today - drop
  pick`) so one click never means two things.
- **RVOL: survivors only**, never the whole universe — `fetch_session_rvol`
  needs `period=1mo, interval=5m` and its own docstring forbids universe-wide
  use. *(Not built in this packet; the column is deferred to the first live
  board session, when the trader can say whether they miss it.)*
- **The gate reads `last_complete`, both halves.** Confirmed by the trader
  2026-08-15: completed bars only. The golden fixture records the narrowing.

## Part A — auto M5 Focus pick gating and eviction

### A.1 Trader rule (2026-08-14)

Any auto M5 Focus pick must be **above the previous day's high AND above session
VWAP on the M5** for longs (below previous day's low AND below VWAP for shorts).
While a pick sits in the pending queue, if it falls back through VWAP or the
previous-day extreme it is **removed from the queue**. The same test re-runs at the
moment of adoption.

### A.2 Current state (recon 2026-08-15)

- The prev-day-extreme half **already exists** at candidate-build time:
  `scripts/prev_day_gate.py` (UNKNOWN always fails) applied via
  `filter_candidates_by_prev_day_extremes` inside
  `refresh_auto_populated_watchlists` (`scripts/autopilot_core.py:1014, 2258-2271`).
- **No VWAP gate exists anywhere** in the auto-populate → staging → adoption path.
  The data is already there: `fetch_intraday_profiles`/`_frame_rows`
  (`autopilot_core.py:716-750, 1153-1209`) carry per-bar volume, and
  `chart_snapshot.session_vwap_series` (`scripts/chart_snapshot.py:84`) is a correct
  session-anchored VWAP over exactly that bar shape. Do **not** substitute
  BounceBot's `calculate_dynamic_vwap`/`calculate_eod_vwap` — they blend prior
  sessions and answer a different question.
- **Nothing re-validates** between staging (`stage_auto_populate_candidates`,
  `autopilot_core.py:2062-2122`) and adoption
  (`_adopt_auto_pick_into_focus`, `scripts/ui/panels/alert_center_panel.py:1612`,
  which calls `store.add(symbol, side, "m5")` unconditionally).
- `FocusPickStore` (`scripts/focus_picks.py`) has **zero per-entry provenance** —
  an auto-adopted entry is indistinguishable from a user-typed one, so no removal
  verb can currently be legal under the never-auto-remove-user-names invariant.
- The current "Not today" button is one relabeled control with three context
  behaviors (`scripts/ui/widgets/alert_chart_review.py:224-325`); none of them is
  "remove this auto-adopted M5 entry" — the live auto-adopt path bypasses approval
  alerts entirely, and the Review-board path removes from **everywhere** via
  `remove_everywhere` regardless of origin (`alert_center_panel.py:2713-2750`).
- Adjacent pre-existing desync: BounceBot's triple-VWAP invalidation
  (`check_removal_conditions` → `remove_from_watchlist`,
  `scripts/bounce_bot_lib/legacy.py:10764, 11062`) deletes the raw watchlist line
  without informing `FocusPickStore`, so a Focus-listed name can silently lose its
  scan line while the Focus UI still shows it.

### A.3 Design

1. **Provenance sidecar.** A new day-scoped sidecar file owned by `FocusPickStore`
   (keyed `SYM|side|category` + session date + staged-at), written at auto-adoption
   only. The plain-text watchlist format stays untouched. Removal verbs consult the
   sidecar: no marker → the entry is treated as user-entered and is untouchable by
   any automatic or "Not today" path. This makes the invariant structural, matching
   `CandidateRegistry.SOURCE_USER`'s philosophy without waiting for the registry
   cutover.
2. **Combined gate function.** One shared `passes_focus_adoption_gate(candidate,
   bars, prev_day, side) -> (bool, reason)` next to `prev_day_gate`: prev-day
   extreme (existing logic) AND last **completed** M5 close vs session VWAP
   (`session_vwap_series`). UNKNOWN/missing data on either half fails — missing
   data grants nothing. Applied at three points:
   - candidate build (replaces/extends the current prev-day-only filter);
   - each staging refresh (~30 min): pending entries that now fail are **evicted**
     with a logged reason (day-cut-blacklist pattern; silent to the trader per
     their wording, visible in the log);
   - immediately before `store.add` at adoption (also the drain path when AWAY or
     EVENING flips to DESK — see packet R1).
3. **"Not today" verb.** In the Alert Center chart pane and the M5 Focus board: for
   entries carrying the auto marker, "Not today" calls a scoped
   `store.remove(symbol, side, "m5")` and records the verdict to
   `pick_feedback.jsonl` (verdict `not_today`, origin `auto_pick`). User-typed
   entries never show the verb.
4. **Desync repair.** When triple-VWAP invalidation removes a watchlist line for a
   symbol that is a current M5 Focus pick: if the pick carries the auto marker,
   remove the Focus entry through `FocusPickStore` (same scoped removal) and log
   it; if it is user-entered, leave Focus alone and surface the mismatch instead of
   hiding it.

## Part B — the M5 strength board (TC2000 parity)

### B.1 Trader formula and filters (verbatim intent, 2026-08-14)

Per symbol on M5:

```
strength = ( Σ over the last 12 completed M5 bars of ((C/O) - 1) * 100 ) / 12
           * ( (C + C50) / 2 ) / ATR50
```

with `C50` = **the close 50 bars ago (TC2000 displacement syntax)** and `ATR50` =
50-bar M5 ATR. Sort descending, keep the **top 25%**.

> **Correction, 2026-08-15 — spec error, corrected by the trader.** This section
> originally restated `C50` as "the 50-bar simple average of M5 closes". That was
> wrong: `C50` is TC2000 displacement syntax for a single historical price, the
> close fifty bars back. The first build implemented the spec faithfully and was
> therefore wrong in the same way. **TC2000 parity is the intent**, and the code
> now reads the displaced close. The two differ materially on any trending series
> (on a strictly rising 0–99 series, the SMA of the last fifty values is 74.5
> while the close fifty bars back is 49), and an average smooths away exactly the
> displacement the price factor is asking about. The history refusal is unchanged
> at 51 bars and now coincides exactly with ATR50's, whose first bar contributes
> no true range — verified, and pinned by
> `test_c50_and_atr50_refuse_at_the_same_history_length`. Filters alongside: price above session VWAP,
20-day average daily volume > 1M shares, price > $5, market cap > $1B, has listed
options, price above yesterday's HOD, price above the M5 15EMA. Inverted for
shorts. Expected yield ~20–40 names per side; the trader dumps them into M5 Focus.

> **STALE, kept for the record (R4 A12, 2026-09-02).** V1 rebuilt this board to
> decision 0016 answer 9 and the ~20–40/side figure no longer describes it. The
> universe is `universe_all.txt` PLUS the four watchlists, the D1 $5 / 100 SMA /
> 200 SMA floors were added, and **a row that misses a filter is GREYED rather
> than dropped** — so the board's LENGTH is now the measured population, not the
> survivor count, and the survivors are whatever the filters leave. Read §11 for
> what the board actually is.

### B.2 Current state (recon 2026-08-15)

- The existing RS/RW board (`real_relative_strength`/`run_rrs_scan`,
  `scripts/bounce_bot_lib/legacy.py:2353-2367, 8382-8601`, rendered via
  `rrs_snapshot.py`/`entry_assist_board.py`) computes an ATR-normalized
  SPY/sector/industry-excess — **structurally unrelated** to this formula — and
  only covers the curated watchlist union (~175 base symbols), never
  `universe_all.txt` (1,506 symbols, built 2026-08-13).
- **No M5 ATR50 or bar-displacement helper exists anywhere**; both are new pure
  calculations.
- Filter availability: 15EMA-M5 and yesterday-HOD/LOD are live per scanned symbol
  (`legacy.py:2782, 9109-9115`); session VWAP exists as the unwired
  `session_vwap_series`; price/20d-volume/market-cap/optionable exist **only** in
  the offline `scripts/universe_builder.py` output (`data/universe_metadata.csv`)
  — and membership in `universe_all.txt` already implies the price/volume/
  cap/options screen passed at build time.
- **IB pacing forbids the literal TC2000 shape**: the locked
  `docs/ULTIMATE_SETUP_DATABASE_PLAN.md` (sec 5.2–5.3, do not re-litigate) rules
  out live per-5-min IB polling beyond ~30 symbols. The in-repo precedent for wide
  intraday coverage is `autopilot_core.fetch_intraday_profiles`: **batched yfinance
  5m downloads** over up to 1,200 universe symbols every ~30 min.
- No percentile-cut ranking exists (boards use fixed top-N); no one-click
  board→Focus path exists (only Copy All → clipboard → Focus panel Paste).

### B.3 Design

1. **New pure module** (e.g. `scripts/strength_scan.py`): the formula above plus
   M5 ATR50 and the C50 displacement, computed on completed bars only, with
   hand-computed fixture
   tests. It does **not** touch `real_relative_strength` (load-bearing, fenced) —
   the existing RS/RW board keeps working unchanged beside it.
2. **Transport and cadence**: batched yfinance 5m fetch over `universe_all.txt`
   (the trader-chosen universe), reusing the `fetch_intraday_profiles` batching
   pattern; zero IB traffic, so the locked pacing plan is untouched. Refresh on a
   timer (default every 15 min inside the R1 quiet-hours window; setting-tunable)
   plus a manual **Refresh** button. One single-flight owner; a failed refresh
   keeps the last good snapshot (Industry Board pattern).
3. **Ranking and filters**: compute strength for every fetched symbol; keep the top
   25% (long side) / bottom 25% (short side) by the signed score; then apply the
   filters — above/below session VWAP (`session_vwap_series` on the same bars),
   above/below yesterday's HOD/LOD (from the fetched frame's prior session),
   above/below the M5 15EMA; price/volume/cap/options ride on universe membership
   (metadata re-checked from `universe_metadata.csv`, not re-fetched).
4. **Board UI**: a new tab/board listing side-split results (symbol, strength,
   % day move, VWAP distance, RVOL if cheap) with:
   - one-click **Add to M5 Focus** per row and a side-aware **Add all shown**;
   - every add passes through the Part-A adoption gate (a board row that fails the
     gate at click time is refused with the reason shown);
   - chart open uses the standard snapshot popup (packet R4 gives it capture).
5. Board output is decision support only: no alerts, no watchlist writes except the
   explicit Focus adds, no influence on any champion path.

## 6. Fenced files and invariants

Ask-first at edit time: `scripts/autopilot_core.py`, `scripts/prev_day_gate.py`,
`scripts/focus_picks.py`, `scripts/ui/services/focus_service.py`,
`scripts/ui/panels/alert_center_panel.py`, `scripts/ui/widgets/alert_chart_review.py`,
`scripts/bounce_bot_lib/legacy.py`. Invariants: user-entered names never
auto-removed (now structural via the sidecar); completed bars only for every gate
check; UNKNOWN fails; one owner per timer/store (the sidecar belongs to
`FocusPickStore`; the board refresh has one single-flight owner); honest zero rows
beat filled panels.

## 7. Tests

Gate truth table (long/short × PDH/VWAP × UNKNOWN), staging eviction transitions,
adoption re-check including the AWAY/EVENING drain path, sidecar provenance (auto
marker present/absent → verb legality), scoped removal never touching swing/other
side/user entries, desync repair both branches; strength formula fixtures
(hand-computed OHLCV series), percentile cut, filter application, board add-to-Focus
passing the gate.

## 8. Exit gate

Deterministic tests green; live proof: one session where the log shows at least one
staged pick evicted for a VWAP/PDH fallback, one adoption-time refusal, one clean
"Not today" scoped removal that leaves the user's other entries intact, and a board
session where the trader adds picks that pass the gate. The trader confirms the
board's names roughly match the TC2000 scan's character. (The ~20–40/side
figure this line used to carry is stale — see the note in §B.1 and gate 44, which
is the parity gate V1 actually owes.)

## 9. Open questions

- ~~Whether the yfinance 5m batch over ~1,500 symbols every 15 min is fast enough
  on the desk~~ — **MEASURED 2026-08-15, see §10. 27.6 s. The 15-minute default
  stands.**
- ~~Whether RVOL should be a board column in v1~~ — **Answered 2026-08-15:** yes,
  but computed for the survivors only (~20–40/side), which is the "handful"
  scale `fetch_session_rvol` documents as safe. Its `period=1mo, interval=5m`
  fetch is far heavier than the board's and its own docstring forbids running it
  over the whole universe. **Superseded by V1 (2026-09-02):** the RVOL is
  computed by `strength_scan.relative_volume` from the board's OWN `1mo` 5m
  fetch, for every measured symbol rather than for survivors, and
  `fetch_session_rvol` is not in that path at all. R4 A7 then made the offset
  session-relative, which is what decision 0016 answer 9 asks for.

## 10. Transport measurement — 2026-08-15, this desk

Measured with the exact transport the board uses: the `fetch_intraday_profiles`
batching pattern, `AUTOPILOT_OPEN_SCAN_CHUNK_SIZE` = 150, `_default_downloader`,
over all 1,506 symbols of `universe_all.txt`.

| period | total | chunks | median chunk | slowest chunk | usable symbols | median bars |
|---|---:|---:|---:|---:|---:|---:|
| `1d` | **17.6 s** | 11 | 1.5 s | 2.2 s | 1,503 / 1,506 | — |
| `5d` | **27.6 s** | 11 | 1.7 s | 2.7 s | 1,503 / 1,506 | 390 |

**`5d` is the one that matters, and it is what the board fetches.** The formula
needs 51 completed M5 bars for ATR50 and C50, and a `1d` window holds about 78
bars for a *full* session — so at 07:00 PT, half an hour after the open, it holds
six. Every symbol would be unmeasurable for the first four hours of exactly the
session the trader is trading. With `5d`, **100% of symbols carry ≥50 bars**
(median 390, minimum 334) from the first bar of the day. Spanning sessions is
also correct rather than merely convenient: TC2000's M5 displacement and ATR span
them too,
and `session_vwap_series` restarts per date regardless, so the VWAP filter is
unaffected by the longer window.

**Cadence decision: keep the spec's 15-minute default** (`§B.3.2`,
settings-tunable). 27.6 s is ~3% of a 15-minute interval. The three failures are
delisted tickers (`JHG`, `LC`, `SEM`) the universe build has not yet dropped, not
a transport problem.

**Caveat on the number.** Measured on a Saturday, so Yahoo is less loaded than at
09:35 ET and the response is Friday's completed session. Treat 27.6 s as the
floor, not the worst case. The margin absorbs it: even a 5× market-hours
slowdown is ~2.3 min of a 15-minute interval. If a live session ever shows the
refresh overrunning, the single-flight owner means a slow pass delays the next
one rather than overlapping it, and the cadence setting moves without a code
change. Re-measure during market hours on the first live board session.

## 11. R2.2 — the final review pass, 2026-08-15

### 11.1 The drain is locked by the flip, not incidentally (item 1)

R2.1 had the AWAY/EVENING → DESK flip re-measure the queue before adopting. The
lock was incidental: if that re-measurement **failed**, the next 30-second poll
fell through to the ordinary stored-verdict drain, and the 2-bar lag bound was
the only thing left between a stalled feed and an adoption.

Two independent mechanisms replace that:

- **The barrier.** The flip records its own moment in
  `AlertCenterPanel._desk_flip_at`, floored to the second because that is the
  resolution `gate_checked_at` carries. `pending_pick_gate_ok(...,
  not_before=...)` refuses any verdict stamped before it — inclusively at the
  flip's own second, so a re-measurement finishing inside that second counts as
  after it. Nothing measured while the desk was unattended is adoptable, by any
  path that reaches the drain.
- **The retry.** A failed re-measurement sets `_reverify_retry_at` and the drain
  adopts nothing until an attempt succeeds: `FLIP_REVERIFY_RETRY_SECONDS` = 60,
  `FLIP_REVERIFY_MAX_ATTEMPTS` = 5. Giving up after five is safe **because the
  barrier still holds** — the ordinary 30-minute staging refresh stamps
  post-flip verdicts and is the slower recovery. The status line distinguishes
  "retrying" from "waiting for the next refresh", because an attempt that
  silently stopped trying looks exactly like one that succeeded.

A re-measurement also remembers **which flip it answers**. A DESK → AWAY → DESK
round trip while one is in flight owes a new measurement: the in-flight run's
bars predate the second return, so the barrier refuses everything it stamps
(correct, and fail-closed) — but if its success also cleared the debt, the queue
would sit unadopted until the next 30-minute refresh with the trader standing
there.

The 2-bar lag bound remains, now as defense in depth rather than as the lock.

### 11.2 The two-bar tolerance: accepted, with the backstop named (item 3)

`FOCUS_GATE_MAX_BAR_LAG = 2` is a **trader-accepted exposure**, not an oversight.
Recorded here and in the constant's own comment so neither can drift from the
other.

**What it lets through.** A feed stalled by exactly one or two bars can adopt a
name that crossed back through session VWAP inside the bars nobody saw. Ten
minutes of tape is real money.

**Why not 1.** yfinance routinely publishes the newest completed bar a minute or
two late. At `max_bar_lag = 1` the desk would refuse most adoptions on a
perfectly healthy feed, and a gate that mostly says no is a gate the trader stops
believing. The choice is between a rare bad adoption and a routine bad refusal.

**The backstop, which is what makes it bounded.** An adopted pick is injected
into `longs.txt`/`shorts.txt`, so BounceBot scans it from the next sweep.
`VWAP_INVALIDATION_CONSECUTIVE_M5_CLOSES = 4` completed M5 closes on the wrong
side of session/dynamic/EOD VWAP files a desync request, and the Alert Center's
30-second poll performs the removal (A.3.4). A name adopted on a stale verdict
that never recovers is therefore gone within roughly **four completed bars** of
adoption, with no action from the trader.

**Reopen trigger.** If a live session shows an adoption that survived the feed
stall and cost a trade, the change is `FOCUS_GATE_MAX_BAR_LAG = 1` plus the
golden-fixture update that records what it newly refuses — not a new mechanism.

## 2026-08-19 — the first DESK morning, and what it cost

### The defect: a naive clock against an aware measured bar

**Every adoption failed all session. Zero picks adopted; 121 refused every 30
seconds from at least 08:07.** `focus_auto_picks.json` ended the day with an
empty `picks` map, and the failure logging rotated `trading_bot.log`.

Root cause, one line deep:

```
autopilot_core.pending_pick_gate_ok
    lag_bars = (latest - bar_end).total_seconds() / (M5_BAR_MINUTES * 60.0)
TypeError: can't subtract offset-naive and offset-aware datetimes
```

The two stamps in a stored verdict come from different writers and had drifted
apart:

| Field | Writer | Awareness |
|---|---|---|
| `gate_bar_end` | the intraday profile's `as_of` (`_intraday_extreme_metrics`) | **always aware** — the provider's own offset when it has one, market-local otherwise |
| `gate_checked_at` | the staging refresh's `moment` (`datetime.now()`) | **naive** |
| the caller's `now` | `AlertCenterPanel` → `datetime.now()` | **naive** |

So the wall-clock age check (naive − naive) passed, and the bar-lag check
(naive − aware) raised — which is exactly the line the traceback named. The
Alert Center's wrapper caught it, correctly refused (fail-closed: an
unverifiable pick is not an approved pick), and the desk adopted nothing.

**This was not a gate that judged the picks wrongly. It was a gate that never
ran.** The distinction matters for the owed proofs: nothing about the PDH/VWAP
rule was exercised on 2026-08-19.

### The fix: normalize at the seam, attaching offsets rather than stripping them

`pending_pick_gate_ok` now runs every datetime it compares — the caller's clock,
`gate_checked_at`, `gate_bar_end` and the `not_before` flip barrier — through
`_gate_moment`, which delegates to `market_session.normalize_market_local_datetime`
(attach market-local to a naive stamp, convert an aware one).

Stripping offsets instead would have ended the crash and kept the outage: an
aware 11:05 ET bar read as naive 11:05 against an 08:07 PT clock is three hours
"ahead of the tape", so every pick would still have been refused — silently.
`tests/test_focus_gate_timezone_seam.py` pins that direction explicitly, and
every refusal path (stale clock, stale bar, future bar, pre-flip verdict) is
re-asserted unchanged. `plan.md` sec 5's "timestamps carry explicit timezones"
is now enforced at the comparison, not only at the writers.

The same subtraction one function away (`minutes_since_open`) is hardened the
same way. Every caller passes a naive clock today, so its answers are unchanged;
the scheduler is simply no longer a place to discover this class of bug live.

### The logging: one traceback per cycle, not one per pick

The wrapper logged a full traceback for every pick it refused, so one systematic
fault wrote **121 tracebacks every 30 seconds**, rotated the log, and nearly
destroyed the evidence needed to diagnose it. Now the first failure of each poll
cycle carries the traceback and the cycle ends with one WARNING naming the count
and the exception. The refusal itself is as loud as it was and fail-closed
semantics are untouched.

### The retry investigation: design and code agree

The R2.2 budget (`FLIP_REVERIFY_RETRY_SECONDS` = 60, `FLIP_REVERIFY_MAX_ATTEMPTS`
= 5) governs **only** a failed flip re-measurement — the `reverify_pending_picks`
fetch on an AWAY/EVENING → DESK return. On 2026-08-19 the desk was in DESK mode
from the start, so no flip occurred, no re-verification was owed, and that budget
was never engaged. Correctly.

The 30-second cadence in the log is the **ordinary poll**: `_poll_auto_pick_pending`
rides the Alert Center's 30s `_watch_timer`, and a refused pick is deliberately
not marked seen, so every cycle re-attempts the whole queue. That is the designed
behaviour — "a stale verdict costs one cycle rather than the pick" — and it is
what made recovery automatic once the code was fixed rather than requiring a
restart. **No code change was needed here; the spec's two mechanisms were being
read as one.** Recorded so the next reader does not re-litigate it.

## Strength board — sortable columns and charts (trader requests, 2026-08-19)

"I need to see my charts in there to make decisions - right now it's just a lot
of picks."

**Every column sorts on click**, with a visible indicator. Three things make it
safe rather than merely convenient:

- **Sorting is presentation.** It re-orders rows already in hand and touches
  neither the service nor the network, so a header click can never cost a
  refetch. A test asserts the service is not called at all.
- **Qt's own `setSortingEnabled` is deliberately NOT used.** The last column
  holds a per-row cell *widget* (the Add button), and `QTableWidget` moves items
  when it sorts while leaving cell widgets where they are — the button would end
  up on its neighbour's row. Owning the order also lets a blank cell sort **last
  in both directions**: an unmeasured field is an absence, not a small number.
- **The default order is unchanged and now stated.** `strength_scan.top_fraction`
  ranks longs strength-descending and shorts strength-ascending — strongest for
  that side first — and the indicator says so instead of leaving the trader to
  re-derive it.

Every add still re-runs the adoption gate at click time; the button is bound to
its symbol, not its row index.

**Selecting a row charts it**, in the desk's existing snapshot popup — the same
one the RS/RW, entry and Industry boards open, owned by the Alert Center, so the
chart carries the same bot-backed series, painted levels and CaptureRail. No new
chart widget exists anywhere (R4's unification pattern). `show_symbol_snapshot`
already reuses one dialog per owner, so re-selecting re-points the same window
rather than stacking dialogs. Selecting on one side clears the other, so "the
charted name" is never ambiguous, and a refresh that keeps the same row selected
is not a new chart request. Double-click still works.

**A docked, always-visible chart inside the board is the follow-up option**, not
this build: it needs its own layout budget on a two-table page and a decision
about what happens to the tables' width, which is a desk-layout judgement rather
than a wiring one. The popup reuses a surface the trader already knows.

## Addendum — 2026-08-21: the RS/RW board joins this page

"add RS/RW board under the strength board" (trader). Asked where, because there
were three defensible readings — a new sidebar page, a second half of this
page, or a section inside Chart Review — and the trader chose the second half.

The page is now a **vertical `QSplitter`**: the board (controls, hint, both side
tables) on top, an `RS/RW Board` section beneath it, both collapsible so either
half can be given the whole page. The board keeps the larger default share
because it is what the page is named for.

Three properties keep this from becoming a second data path:

- **Same widget.** It is the `RrsSnapshotWidget` the Alert Center's RS/RW tab
  already uses, not a reimplementation, so scope tabs, focus marking and the
  Copy All RS/RW buttons behave identically in both places.
- **Same payload, one owner.** `app.py` adds a **second listener** to the one
  `rrsSnapshotChanged` signal the bounce service already emits. Nothing on this
  page fetches, schedules or caches; the service remains the only producer, and
  a Qt signal being multicast is exactly the mechanism for a second view.
- **Same chart.** The RS/RW half routes through the page's existing
  `symbolActivated`, which `app.py` points at the Alert Center's snapshot popup.
  Still no second chart widget anywhere.

Owed: the half populating from a live BounceBot sweep. Until one runs it shows
its own honest empty state ("Connect BounceBot to stream relative-strength
scans"), which is the correct reading of "no payload yet" rather than a blank.

---

## Addendum — 2026-08-19 (evening): movers only in chart review

**The trader's rule, verbatim:**

> "A long inside yesterday's range is probably chop. Chart review should only
> show me longs above the previous day's high and shorts below the previous
> day's low. Focus picks that ARE beyond their previous-day extreme should be
> flagged - those are the ones actually moving. Inside-range picks appear only
> when I deliberately review focus picks."

This is a **presentation** rule. No detector, score, alert or watchlist changed;
nothing in this addendum touches what fires, what is recorded, or what is kept.

### The predicate is the gate's own

`focus_adoption_gate.mover_state(side, price, prev_high, prev_low)` is the
**extreme leg alone** of the Part A adoption gate — a thin name over the same
`prev_day_break_state` call the gate makes, and `focus_adoption_gate_state` now
routes its own extreme leg through it. There is exactly one implementation of
"beyond yesterday's extreme" in the tree.

That matters more than it looks. A display filter with a private copy of the
rule would eventually hide a name the machine had just adopted, and the trader
would be reading a review queue that disagreed with their own Focus list, with
neither number wrong on its own. A test walks the whole input matrix and asserts
the two entry points can never disagree.

There is **no session-VWAP leg** here: this filter answers "is it beyond
yesterday's extreme", which is a weaker question than adoption asks, and
deliberately so — the trader wants to *see* movers, not only the ones the
machine would take.

### What the filter does, and what it refuses to do

Applied in `AlertCenterPanel._enqueue_review_alert`, the single door into the
review queue, so every caller — the D1 Focus feed, the auto-pick drain, the
scanner alerts — passes through it.

| | |
|---|---|
| Default | **ON** |
| Longs inside yesterday's range | not queued |
| Shorts inside yesterday's range | not queued |
| UNKNOWN (no prior session, no bars, measurement failed) | **SHOWN**, tagged `unmeasured` |
| The withheld | counted on a clickable line: `N hidden (inside yesterday's range) - show` |
| One click | shows exactly those names and turns the filter off **for that session** (day-scoped, resets with the market date) |
| Deliberate Focus review (`review_focus_picks`) | **bypasses the filter entirely** |
| Armed chart-watch hits | bypass it — the trader armed that exact condition |

**Hard lines, all tested:**

- it **hides**; nothing is removed from the feed, history, or any store;
- **no** `review_policy.json` involvement — that file ranks and annotates and has
  no suppression field, ever;
- **nothing** is written to the review-learning stream (`_record_review_event` is
  not called on a hide);
- **no** alert sound, toast or phone push is muted;
- **no** watchlist or Focus entry is auto-removed.

UNKNOWN showing is the load-bearing choice: missing data is uncertainty, never
confirmation (`plan.md` sec 5). A filter that failed closed would blank the
review queue the moment the daily store or the bot's bars hiccuped — the worst
possible behaviour mid-session, and indistinguishable from "nothing qualifies".

### The flag on Focus surfaces

A Focus chip whose name is beyond its previous-day extreme **on its own side**
carries a `MOVING` flag, in the existing badge idiom (the same short uppercase
word the `BOUNCE`/`RRS` flags use). The charted alert carries the same state as
`MOVING` / `unmeasured` / `inside range` beside the reviewed-today badge.

Cadence: the Alert Center's existing 60-second D1 poll already re-measures every
Focus name against yesterday's range (`_update_focus_break_state`); it now emits
`focusBreakStatesChanged` and the Focus board repaints from that. **No new timer,
no new market data, no IB traffic** — the flag asks a question the desk had
already answered.

### Owed, live

One review session where the trader confirms the queue shows only movers and the
hidden-count line is honest — recorded as `DESK_TESTING_PLAN.md` §2.10.

## Addendum — 2026-08-31: the board moves into the Desk's Strength window

*"The Strength Board tab is good but it really should be modified to fit in the
'strength' window in the trading desk — either integrated directly or be
positioned below it."* (trader). Positioned below it, and the left-nav page is
removed.

**Where it is now.** A `CollapsibleSection` under `FocusStrengthBoard` in the
Alert Center's alert column, hosted by `AlertCenterPanel.attach_strength_board`.
`MainWindow` still builds and owns the one `StrengthBoardService` — one timer,
one single-flight fetch, one 15-minute cadence — and now also shuts it down,
which nothing did before: the service was parented to the window but absent from
the panel shutdown loop, so its timer outlived the close. Only the wiring moved.

**What did not change**, and is pinned by
`tests/test_qt_strength_board_in_the_desk.py`:

- **Zero IB traffic.** Batched yfinance over `universe_all.txt`, unchanged. The
  test walks the AST of all three strength-path modules, so an `ibapi` import or
  an `EClient`/`reqHistoricalData` name fails it rather than a comment drifting.
- **The adoption gate runs at click time**, on the row's own numbers, exactly as
  Part A defines it — the board is up to 15 minutes stale wherever it is drawn.
- **One service, one timer**, measured by driving that timer and counting the
  fetch attempts rather than asserting single ownership in prose.

**Width was the constraint**, because the alert column has a 360 px floor and
everything left of it is chart. The section must never be the reason the charts
get narrower, and four measurements shaped the build:

| Demand | Measured | What was done |
|---|---|---|
| Section header | 315 px | `QToolButton` demands its whole label; Ignored horizontally + elided text |
| The board | 270 px | hosted in a `QScrollArea`, so the minimum stops there rather than reaching the desk splitter |
| Status label | 434 px | word-wrapped — it carries failure reasons, so it can be long |
| "Add all shown" | 208 px | relabelled "Add all" (124 px); the tooltip still says the whole thing |

The section also **starts closed**, so by default it costs one header row. The
two sides stack **vertically** now: side by side was right for a full-width page
and is unreadable in a column.

**The RS/RW half retired with the page.** The 2026-08-21 addendum above added it
so the two reads could be compared without flipping **pages**; the Alert Center's
own RS/RW Board tab is now one tab-click away in the **same column**, so keeping
it would have meant two views of one payload six inches apart. The tape, its
owner, the `rrsSnapshotChanged` signal and that tab are untouched — one listener
retired, nothing else moved. If the trader wants that second view back, it is a
section, not a page.

### A row click charts in the review pane

Same day, second pass — *"when I click on a stock in this M5 strength board it
should come up on the Visual chart review in the trading desk."* It used to open
the snapshot popup, which was the right answer while the board was a page of its
own and the pane was somewhere else; now that the two share a column, a popup
over the top of the pane is a window in the way.

The click goes through **`chart_symbol`** — the same door the lookup box uses —
and deliberately **not** through `_enqueue_review_alert`. That is the door for
things the *scanner* said, and it would have been wrong four ways for a click:
it drops everything in AWAY, it drops parked symbols, it diverts M5 alerts to
the alert bar instead of the chart, and the movers-only filter can hide a row.
**A name the trader clicked must appear.**

Consequences, all deliberate:

- It charts as a `MANUAL_CHART`, so the pane's setup text stays **muted rather
  than red** — nothing fired; the trader was looking.
- It never enters the alert feed, which is a record of what the scanner said.
- Clicking an ignored symbol **un-ignores** it, exactly as typing one does: "not
  today" must not make a name silently un-chartable, which reads as the board
  being broken.
- `symbolActivated` now carries `(symbol, side)`. The board is the only thing
  that knows which of its two tables the row came from, and a short charted as a
  plain `WATCH` reads as the wrong thesis.
- `chart_symbol` grew two optional keyword arguments (`side`, `origin`) for
  provenance and display; its defaults reproduce the lookup box exactly.

### Owed, live

One desk session where the trader opens the section, reads the board in the
column, and adds a name from it — plus a judgement on whether the vertical
stack is right or the sides want their old side-by-side shape back with the
column dragged wider.

## Amendment, 2026-09-02 (Phase 0.14 packet V1) — TC2000 parity

Decision 0016 answer 9 makes **the trader's own TC2000 scan the specification for
this board**, so §B's formula is no longer the whole of it. What this plan
described - the strength formula, the 25% cut, the session-VWAP check and the
15 EMA - was correct and is unchanged. Three things it did not have:

1. **Relative volume.** `AVG(V / mean(V78, V156, ... V1170), 12)` - each of the
   last twelve completed bars against the same bar offset over the prior fifteen
   sessions. **Positional, exactly as TC2000 is**: `V78` means "78 bars ago", not
   "this time yesterday". Across a half day those differ by 39 bars and every
   later offset is shifted; that is TC2000's divergence too, and parity is the
   requirement, so it is documented rather than corrected into a different
   number. Keep the top 50%, and the pick must also be in the top 50% of today's
   session volume - a name can clear the first on twelve quiet bars that are
   merely less quiet than usual.
2. **The floors:** last price over $5, above the D1 200 SMA, above the D1 100
   SMA, above the M5 15 EMA (mirrored for shorts). **The two timeframes are an
   ASSUMPTION** - the trader wrote "the 200 and 100 SMA" without naming one, and
   decision 0016 records both as open. One line in `strength_scan.D1_SMA_PERIODS`
   and one in the EMA span correct them.
3. **The universe** is `universe_all.txt` PLUS the four watchlists. A name the
   trader is watching for their own reasons may not clear the universe's
   liquidity specification, and the board it never appears on is the one they
   are reading.

**Two consequences that are costs, not details.** The M5 fetch period grew from
`5d` to `1mo`, because the RVOL needs 1,182 bars and `5d` holds about 390 - under
the old period every RVOL would have been blank. And the D1 floors need daily
bars, so there is a second batched daily download over the symbols that reached
the board. Still **zero IB traffic**.

**A row that misses a filter is GREYED and names what it missed, never dropped**
(decision 0010: a display filter is not a suppression), behind a default-on
"TC2000 parity" toggle that hides them for a line-by-line comparison.

**The fence on `strength_scan.py` is narrowed, not lifted.** §2 and §8 froze the
module whole and said stop and ask; the trader asked, in packet V1, naming the
file. The test now pins the seven FORMULA functions byte-identical to the R8
baseline - stronger than "no edits at all", which could be satisfied by not
touching the file while the numbers moved underneath it.

**`relative_volume` is NOT one of the seven** (R4 A7, 2026-09-02), and that is
the whole reason the narrowed fence is the right shape: V1 added the function, so
R4 could correct it from a flat positional stride to the session-relative offset
decision 0016 answer 9 actually asks for, while `strength_score`, `atr`,
`displaced_close`, `sma`, `true_ranges`, `percentile_cut` and `ema` stayed
byte-identical. The golden gained two symbols - one early close, one missing bar
- and AAA-EEE's pinned values did not move, which is the proof the correction
touched only what it was aimed at.

**Parity is a golden, not an impression.** `tests/fixtures/tc2000_parity_v1.json`
pins strength and RVOL for five symbols over sixteen sessions, and its expected
values are computed by a SECOND naive implementation written from the trader's
two formula lines rather than from the module under test. All five agree to four
decimals.

## The board's parity rows join M5 Focus on their own (packet T1.4, 2026-09-04)

Trader: *"I want all shorts and longs on the RS/RW board TC2000 to bne auto
added to the M5 focus picks."*

`AlertCenterPanel._auto_adopt_strength_board` runs on `StrengthBoardService`'s
`boardChanged` and once at attach, so a desk started mid-session does not wait
fifteen minutes for its first placement. It considers **only rows with an EMPTY
`failed_floors`** - the TC2000 parity list; a greyed near-miss missed one of the
trader's own filters and is never adopted. It **re-runs the one adoption gate**
on each row's own `last / prev_high / prev_low / session_vwap` (the board can be
fifteen minutes stale, and UNKNOWN fails as always) - a fourth call site for the
single definition in `focus_adoption_gate.py`, never a second one. It is **DESK
only**: AWAY stages nothing here and EVENING/OFF do nothing, so the auto-mode
matrix is unchanged. It **skips any symbol in `_ignored_symbols`**, so the next
refresh cannot undo a "Not today".

The write is the MACHINE's, so it goes through the STORE - `store.add` then
`store.mark_auto_adopted` - and **never `FocusService.add`**, which would forge a
trader "like" into `pick_feedback.jsonl`. The marker is written only when `add`
actually added: an existing unmarked entry is the trader's and must not change
owner. It **never removes** - a name that leaves the board stays on Focus, and
the ten-session fade and "Not today" own removal - and it is idempotent per
refresh. One `strength_board_auto_focus` review event per refresh that adopted
or refused anything, and one status line when something was adopted.

The **click-to-add path is unchanged**: a click on a row IS the trader liking the
name, so `_add_symbols` still goes through the service and still writes the
pick-feedback row. Live gate #58.
