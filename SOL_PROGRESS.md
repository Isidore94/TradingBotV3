# Sol branch — plan.md implementation status

Branch: `Sol` (from `main` @ 69bdee8, 2026-07-10 night). Every commit ships
green: **769 passed, 0 warnings** (was 715 passed / 2 failed / 7 warnings).
`python scripts/smoke_check.py` runs 7 deterministic checks with no network.

plan.md is a multi-month roadmap; this pass implemented its own prescribed
starting order (sec 21 "first ten moves" and the Packet A–D foundations),
prioritizing pure, fully-tested engines plus the correctness/reliability fixes
it marked P0. GUI integration of the new engines is deliberately staged so you
can test the foundations first.

## Implemented (by commit)

1. **Phase 0 baseline** — include_theta test contract fixed (suite green),
   `datetimde` dormant defect + regression tests, duplicate
   `_priority_expected_r_text` removed + AST guard against duplicate top-level
   definitions, deprecated Qt proxy invalidation replaced with
   `beginFilterChange()/endFilterChange()`.
2. **Packet A (runtime lifecycle)** — BounceBot `_stop_event` + `stop()` that
   joins owned threads; cancellable `wait_for_candle_close`; all strategy-loop
   sleeps cancellable; `BounceService.stop()` really stops the bot and ALL
   timers (board timer was leaked) and wins the stop-during-startup race.
3. **Packet B (sec 16/24)** — `scripts/market_state.py`: pure SPY
   market-state/pullback-episode machine, side-symmetric (`side_sign`),
   hysteresis, complete-bars-only transitions, append-only episode records,
   sec-24 thresholds in versioned config; exact bearish-mirror tests.
4. **Packet C (sec 16.8)** — `scripts/relative_strength.py`: exact-timestamp
   aligned multi-window features, beta+volatility-normalized residuals,
   transparent component/percentile composite with named penalties, tiers
   (DEFIANT/HOLDING/WATCHING/FADING/INVALID); mirror-equivalence tests.
5. **Packet D (sec 14.3/16.6)** — `scripts/candidate_registry.py`: per-source
   provenance + leases, user entries automation-proof, lifecycle stages with
   one-event-per-transition, sec-16.6 priority live pool, atomic versioned
   persistence (stale writers rejected).
6. **Phase 1 (sec 6.1)** — `scripts/diagnostics/`: structured run manifests;
   `run_master` is manifest-wrapped (every run, success or failure, writes
   phase timings + counters; bounded local history).
7. **23.8** — `publish_away_report`: atomic tmp+replace, sha256 readback
   verification, dated bounded archive, failure never clears the prior valid
   report; report header carries an explicit freshness warning; the service
   separates last_attempt from last_verified_success.
8. **Correctness batch** — 22.9 stable digest baseline sampling (was
   per-process `hash()`), 22.3 strict `Side` parsing with manifest-counted
   legacy coercions, 23.5 universe freshness = oldest-of-all-files (missing
   file ⇒ no generation), 23.7 lazy legacy-Tk import, Phase 10.8/10.9
   `pyproject.toml` (pytest markers, narrow ruff gates) + `constraints.txt`.
9. **Auto Mode semantics + smoke** — OFF/DESK/AWAY with persisted profile and
   report labeling; strict OFF runs no suggestion scans/near-extreme checks
   and touches no lists unless `autopilot_shadow_research` (default ON,
   preserving the current scorecard pipeline) allows it;
   `scripts/smoke_check.py` + test.

## Sol2 branch (2026-07-11, on top of Sol)

10. **Scan-child ownership (P0 #5)** — closing the desk reaps every scan
    subprocess it spawned (bounded wait, then terminate); found live when a
    closed app left a 7.6GB scanner running. `owned_scan_process_count()`
    ready for the Health surface.
11. **SPY engine in shadow (sec 16 kickoff)** — `market_state_bridge` runs the
    pure engine on the bot's cached SPY bars inside
    `check_regime_pause_setups` and logs agreement/divergence vs the legacy
    one-red-candle pause detector to
    `machine diagnostics\spy_state_shadow.jsonl`. Legacy stays the champion;
    the log accumulates promotion evidence.
12. Weekend freshness label; two environment-dependent tests made hermetic.

First production run manifest (2026-07-11): 1,270 symbols / 28.5 min; output
writes 291s; `side_coercions_invalid: 7` — the strict-side counter caught 7
corrupt side values on day one.

## Not yet implemented (next in plan order)

- **Engine integration (sec 21 moves 7–8)**: wire `market_state` +
  `relative_strength` into regime-pause, Entry Assist, environment scan, and
  RS Window (adapters exist; the four surfaces still use their own logic);
  single-pass RRS timeframes in BounceBot.
- **TradingRuntime composition root + global Auto header UI** (sec 15): the
  service-level semantics landed; panel-ownership inversion and the top-bar
  control are UI work best done with you watching.
- **CandidateRegistry adoption** (Packet D step 2+): converting the live
  writers (open scan/auto-populate/near-extreme/VWAP removals) to sources.
- **Phase 2**: one scheduler engine + job ledger + writer leases.
- **Phase 3**: storage classification, local journal DB, secrets to OS store.
- **Phase 4**: provider repository, staged Master scan, batching.
- **22.1/22.2/22.4–22.7**: moving-level look-ahead, same-day history keys,
  backfill leakage, tracker identity, score/bucket ordering, factor horizons —
  each needs golden fixtures before touching detector behavior.
- **Phases 5–9**: point-in-time research, ranking pipeline, new setup shadow
  program, journal linking, opportunity inbox, market-prep Qt port, CI.
