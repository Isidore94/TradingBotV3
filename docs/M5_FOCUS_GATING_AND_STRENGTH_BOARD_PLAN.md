# M5 Focus adoption discipline and the M5 strength board — packet R2

Status: **ACTIVE specification** for `plan.md` Phase 0.5 **R2**. Authorized by the
trader on 2026-08-15; ranked **second** in the Phase 0.5 build order.

Build gates: nothing lands before `plan.md` P0.7 completes. Ask-first applies at
edit time to every fenced file in section 6. Re-verify recon line numbers before
editing.

## 2026-08-15 trader decisions recorded here

- Strength-board universe: **the existing universe** (`universe_all.txt`, ~1,500
  names) — not a new broader US universe build.
- "Not today" on an auto-adopted M5 Focus pick removes **exactly that M5 entry for
  that side** — never the swing-category entry, never the other side, never a
  user-typed name.

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
           * ( (C + SMA50(C)) / 2 ) / ATR50
```

with `SMA50` = 50-bar simple average of M5 closes and `ATR50` = 50-bar M5 ATR. Sort
descending, keep the **top 25%**. Filters alongside: price above session VWAP,
20-day average daily volume > 1M shares, price > $5, market cap > $1B, has listed
options, price above yesterday's HOD, price above the M5 15EMA. Inverted for
shorts. Expected yield ~20–40 names per side; the trader dumps them into M5 Focus.

### B.2 Current state (recon 2026-08-15)

- The existing RS/RW board (`real_relative_strength`/`run_rrs_scan`,
  `scripts/bounce_bot_lib/legacy.py:2353-2367, 8382-8601`, rendered via
  `rrs_snapshot.py`/`entry_assist_board.py`) computes an ATR-normalized
  SPY/sector/industry-excess — **structurally unrelated** to this formula — and
  only covers the curated watchlist union (~175 base symbols), never
  `universe_all.txt` (1,506 symbols, built 2026-08-13).
- **No M5 SMA50 or ATR50 exists anywhere**; both are new pure calculations.
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
   M5 SMA50/ATR50, computed on completed bars only, with hand-computed fixture
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
board's names roughly match the TC2000 scan's character (~20–40/side).

## 9. Open questions

- ~~Whether the yfinance 5m batch over ~1,500 symbols every 15 min is fast enough
  on the desk~~ — **MEASURED 2026-08-15, see §10. 27.6 s. The 15-minute default
  stands.**
- ~~Whether RVOL should be a board column in v1~~ — **Answered 2026-08-15:** yes,
  but computed for the survivors only (~20–40/side), which is the "handful"
  scale `fetch_session_rvol` documents as safe. Its `period=1mo, interval=5m`
  fetch is far heavier than the board's and its own docstring forbids running it
  over the whole universe.

## 10. Transport measurement — 2026-08-15, this desk

Measured with the exact transport the board uses: the `fetch_intraday_profiles`
batching pattern, `AUTOPILOT_OPEN_SCAN_CHUNK_SIZE` = 150, `_default_downloader`,
over all 1,506 symbols of `universe_all.txt`.

| period | total | chunks | median chunk | slowest chunk | usable symbols | median bars |
|---|---:|---:|---:|---:|---:|---:|
| `1d` | **17.6 s** | 11 | 1.5 s | 2.2 s | 1,503 / 1,506 | — |
| `5d` | **27.6 s** | 11 | 1.7 s | 2.7 s | 1,503 / 1,506 | 390 |

**`5d` is the one that matters, and it is what the board fetches.** The formula
needs 50 completed M5 bars for SMA50 and ATR50, and a `1d` window holds about 78
bars for a *full* session — so at 07:00 PT, half an hour after the open, it holds
six. Every symbol would be unmeasurable for the first four hours of exactly the
session the trader is trading. With `5d`, **100% of symbols carry ≥50 bars**
(median 390, minimum 334) from the first bar of the day. Spanning sessions is
also correct rather than merely convenient: TC2000's M5 SMA50 spans them too,
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
