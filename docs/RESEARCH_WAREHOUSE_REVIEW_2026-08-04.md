# Research warehouse review — Fable, 2026-08-04

Review of branch `claude/das-warehouse-phase-1-0gis7e` (commits `26dd695`…`e1d54b9`,
Phases 1–8 of `docs/ULTIMATE_SETUP_DATABASE_PLAN.md`). Read order followed:
plan.md sec 5 + 12·13a → the locked plan (secs 5, 7.1, 8.3, 8.4, 14.2, 19, 23) →
`RESEARCH_WAREHOUSE_BUILD_DECISIONS.md` (BD-01…BD-52 + open items) →
`RESEARCH_WAREHOUSE_ERD.md` → `scripts/research_warehouse/` and
`tests/test_warehouse_*.py`. Section 23 was treated as locked; nothing below
re-litigates an LD.

Four defects were fixed on the review branch (commit `f27033f`) because their
fixes are mechanical and unambiguous. Everything else is reported only, because
the correct fix requires a design decision the outcome engine or the wiring
must make deliberately.

**Repair addendum (same day, this branch).** The outcome-engine defects — D1
(frozen outcomes / stale matured means), D3 (management past the time stop),
D4 (partial discarded on a stop), D11 (maturity not `min()`), D12 (missing
entry slippage + gap-through stops), D13 (unbounded intraday walk, no OPEN
state) — are now repaired in `outcomes.py`/`queries.py`, with the repair
decisions logged as BD-53..BD-57 and regression tests for each. Still open
from this review: the minor notes only.

**Defect-repair addendum 2 (2026-08-04, branch
`claude/das-warehouse-defects-2n9uql`).** D5 is repaired (BD-58, BD-59). One
correction to this report's own finding: the intraday half of D5 asserted that
"production's M5 EMAs run on BounceBot's '5 D' frame". They do not — the "5 D"
fetch feeds the previous-day extremes and the dynamic/EOD VWAPs, while
`ema_8/15/21` are computed on `today_df` alone and only once
`len(today_df) >= span` (`bounce_bot_lib/legacy.py`, step 5, "Calculate short
EMAs (today only)"). The warehouse's session scope was therefore already
right; what was missing was the champion's minimum-bar guard, which is what
BD-59 adds. **The trader adjudicated this on 2026-08-04 and confirmed the
correction**, so this paragraph — not section 2's D5 text — is the operative
statement about the intraday EMAs. The daily half of D5 is confirmed exactly as written and fixed by
BD-58. D6 and D7 are confirmed exactly as written and repaired in BD-60; D19
likewise, in BD-61 (which also answers open item 11's "decide what the EOD
build actually runs"). D14–D18 are repaired in BD-62; the D17 regression test
additionally turned up a bug this report missed — a bare `"YYYYMMDD"` string
is all digits, so `_epoch_to_utc` read it as epoch seconds and dated the bar
1970-08-23.

Every S1/S2/S3 defect in this report is now closed. What remains before the
pilot is not defect work: the BD-20 live wiring, the BD-25 broker-marked IB
run, the Windows/3.14 test run, and the trader's confirmation-register items.

**Overall verdict.** The store core (seal, manifest, quarantine, retirement,
read path) is well built and matches the plan; champion isolation of the tee
and pacer is genuinely structural. The outcome engine and the feature windowing
are not evidence-grade yet: several confirmed defects put wrong numbers into
`outcome_path` and `feature_snapshot_*`, and the append-only idempotency
pattern ("first write wins, never recompute") is applied to datasets whose
correct value changes as inputs complete. The pilot must not start until the
S1 items below are repaired.

---

## 1. Champion isolation (R1/R3) — PASS, with one wiring ruling needed

Proven by reading, not just by tests:

- `bar_archive.py` contains no provider client, connection, retry, or call site
  inside a champion fetch path; `capture_m5_tee` reads a dict the champion
  already populated (`latest_bars["<SYM>|5 D|5 mins"]` — key shape verified
  against `bounce_bot_lib/legacy.py`). It cannot fail a champion fetch because
  it makes no request. BD-15 holds.
- Nothing in `scripts/research_warehouse/` imports, reads, or writes
  `_IBKR_HISTORICAL_FAILURE_COUNT`; capture errors are tagged `capture=True` at
  `pacer.note_error` and terminate there. BD-22 holds structurally.
- `pacer.note_champion_request` never blocks; champion traffic is never gated.
- The capture client IDs (1010/1011, 1003 retired) are asserted before any
  socket opens.

Two isolation-adjacent observations:

- **`pacer.note_champion_request` / `champion_window` have no call site
  anywhere.** The token bucket therefore never observes champion consumption.
  Ruling (see open item 1): during the slice, run *zero* capture requests
  during RTH — tee plus nightly/weekly backfill only, when champions are idle —
  which makes champion observation moot and keeps warehouse imports out of
  champion paths entirely. Revisit only at the post-slice Focus-streaming
  milestone.
- `extract_tee_bars` iterates `dict(latest_bars)`; if the wiring ever calls the
  tee from a thread other than the one that owns the cache, a concurrent
  resize can raise. The BD-20 wiring must snapshot on the owning (GUI) thread.

## 2. Confirmed defects

Severity S1 = puts wrong numbers into evidence or endangers the desk;
S2 = defeats a stated plan goal; S3 = edge/latent. Line numbers are from the
reviewed branch.

### S1 — evidence-corrupting

**D1. Outcomes are frozen at first simulation; stale interim results later
count as matured evidence.** `outcomes.build_outcomes` skips any
(occurrence, recipe, definition) already in `outcome_path`
(`outcomes.py:683-685`) and there is no re-simulation or supersession path, so
a row computed while the trade is `OPEN` keeps its interim `gross_r` forever.
`queries.slice_readout` then counts it as matured once `maturity_at` passes
(`queries.py:164-183`). Reproduced: a trade simulated two sessions after
trigger at +1.0R interim, which actually finished ≈ −0.8R, reports
`n_matured=1, mean_gross_r=+1.0` after maturity; the re-run is skipped as
`ALREADY_SIMULATED`. Every nightly build during the pilot would poison the
means this way. TRUNCATED rows carry the same flaw (interim `gross_r` enters
matured means). Fix needs a design: either simulate only at/after maturity, or
add a recompute-and-supersede rule for non-terminal states (`OPEN`,
`TRUNCATED`) — `outcome_path` has no revision columns, so "recompute when
`result_state` is non-terminal and dedupe at read on latest `computed_at`" is
the least invasive shape.

**D2. `_process_alive` killed the running build on Windows.** `cli.py:58-63`
used `os.kill(pid, 0)`, which on Windows is an unconditional
`TerminateProcess`. A second `cli build` during a scheduled build would kill
the first mid-seal, and `test_a_second_build_refuses_rather_than_racing`
self-terminates the pytest process — this alone would have broken the
unconfirmed Windows/3.14 test run. **Fixed** (kernel32 liveness query).

**D3. House management walks past the 18-session time stop.**
`simulate_swing` calls `_house_management_r(forward, …)` — all forward bars,
not the `horizon` slice (`outcomes.py:351-352, 296`). Reproduced: a band-2
touch on forward session 25 credits an EXPIRED trade with `gross_r = +1.0`
whose 18-session close was flat. R beyond the declared time stop is not this
recipe's outcome.

**D4. Management is skipped when the runner stops out, discarding the
partial.** The `result_state != STATE_STOPPED` gate (`outcomes.py:351`) means
"50% off at band 2, then stopped" reports a full-size stop. Reproduced:
partial at +2R on day 2, stop on day 4 → reported `gross_r = −1.4`; the policy
result is ≈ +0.45 (0.5·2R + 0.5·trail exit). Also, on `AMBIGUOUS_BAR` rows the
management overwrite leaves `gross_r` inconsistent with the stored
`r_lower_bound`/`r_upper_bound`. BD-42's claim ("management is simulated") is
not yet delivered; the management walk needs to own stop handling itself and
be bounded by the horizon (D3).

**D5. Daily feature snapshots see a year-partition window, not history.**
`build_daily_snapshots` reads `bar_d1` for the session's year and adds the
prior year only in January (`features.py:543-545`), but a 200-session window
spans ~9.5 calendar months: for sessions from February to mid-October,
`sma200` (and `sma100` into ~May) is silently null
(`compute_indicator_frame` uses `rolling(period)` with default
`min_periods`), and — worse — `ema8/15/21` are seeded on the truncated frame,
so they are *different numbers* from the champion's full-history frame under
the same column names. The intraday sibling: `build_intraday_snapshots` feeds
`compute_intraday_features` only the session's RTH bars
(`features.py:737-742`), so `ema8_m5/ema15_m5/ema21_m5` are session-seeded
EMAs while production's M5 EMAs run on the "5 D" frame — the exact
same-name-different-number failure BD-33 exists to prevent. The AVWAP block is
unaffected (anchor-relative). Fix needs a windowing rule (e.g. always read
`year` and `year−1`, and give intraday EMAs the same 5-day lookback the
champion uses), stated as part of `tier1_v1`'s definition.

> **CORRECTION (2026-08-04, trader-confirmed).** The claim that "production's
> M5 EMAs run on the '5 D' frame" is **wrong** — see the repair addendum above
> and BD-59. BounceBot computes `ema_8/15/21` on `today_df` alone. Do **not**
> implement the 5-day intraday seed this paragraph asks for; it would create
> the very defect the paragraph is complaining about. The daily half of this
> defect stands as written.

### S2 — defeats a stated plan goal

**D6. Nightly ETH backfill never fills teed symbols.**
`already_captured(store, dataset, symbol, day)` treats *any* bar for
(symbol, day) as "already have" (`backfill.py:201-210`). The tee captures RTH
bars for the whole watchlist cohort every session, so the ETH-inclusive
nightly job — whose entire purpose is "filling what the RTH-scoped tee could
not see" (LD-03) — skips exactly those symbols. Premarket/postmarket bars for
the tee cohort are never captured. The check must become per-bar (as
`capture_m5_tee._known_bar_keys` already does) so ETH rows publish without
duplicating RTH rows.

**D7. The backfill loop freezes the pacer clock and never waits.**
`run_backfill` computes `stamp` once and passes it as `now` to every
`try_acquire`/`note_error` (`backfill.py:249, 273`), so the 10-minute window
never advances within a run: after ~15 grants (the capture allowance) every
remaining (symbol, day) is denied and recorded as missed. One invocation can
therefore do at most ~15 requests — the sec 5.1 nightly plan (~350+ at floor)
is unreachable, and nothing sleeps/retries (`pacer.acquire` exists but is
unused). Compounding it, pacer denials are recorded as
`NOT_COLLECTED_BY_POLICY` (`backfill.py:258, 276`) — a pacing shortfall is
*intended-but-not-collected*, and sec 5.4 is emphatic that policy absence must
never absorb real gaps. `_record_missed` also appends without deduping against
existing gap rows, so repeated runs inflate `collection_gap`
(`backfill.py:326-347`), and no path ever sets `resolved_at`/`BACKFILLED` on a
gap a later run fills.

**D8. Spool age cap shed protected segments.** `enforce_cap` shed *any*
closed segment older than 7 days, including `PROTECTED` (D1/M5) ones
(`spool.py:202-204`), violating sec 8.4/LD-12 "D1/M5 capture and operational
champions are never shed". Reproduced, then **fixed** (protected segments now
survive the age cap; the backlog grows and Health goes red instead).

**D9. An interrupted seal double-published.** A crash between `store.publish`
and the segment unlink (or a mid-segment publish failure) re-published the
same rows on the next build (`spool.py:335-347`) — the manifest recorded
`spool_segment` but nothing checked it. **Fixed** (seal now skips
(dataset, segment) pairs already in the ledger), with a regression test.

**D10. Capture reconnect could never succeed against the real transport.**
`ensure_connected` fetched `transport.connect` — which on the real client is
ibapi's `EClient.connect(host, port, clientId)` — and called it with a spec
object; the TypeError was swallowed and reconnect always returned False
(`ib_capture.py:214-225`). Every post-23:45-restart resume would have failed
live while passing offline (fakes define `connect(spec)`). **Fixed**
(`connect_spec` preferred). This sharpens BD-25: the live path was not merely
unverified, it was broken.

**D11. `maturity_at` is not sec 14.2's `min(+18 sessions, stop/target/expiry)`.**
`_swing_maturity` always returns the 18-session date
(`outcomes.py:579-596`), even for a trade that stopped on day 2. Combined with
the matured-only means in `slice_readout`, resolved trades are excluded from
the readout for weeks after they resolve. Deviation from the contract; if it
is kept deliberately it needs a new `outcome_definition_id` per sec 14.2's own
rule.

**D12. `net_r` omits the "+1 half_spread slippage on stop/market entries"
bullet.** `net_r` implements only
`gross_r − 2×(commission + half_spread)/stop_distance`
(`outcomes.py:180-185`); no slippage term exists anywhere, and the intraday
stop exit is filled at exactly −1.0R (`outcomes.py:498`) with no gap-through
handling (sec 14.3 names gap-through-stop behavior as a retained path fact).
Net R is systematically optimistic by ~1 half_spread per round trip under a
contract that says every deviation is a new `outcome_definition_id`.

**D13. `simulate_intraday_bounce` has no session boundary and no OPEN state.**
It walks every provided bar after entry — across sessions if the caller passes
more than one day (`outcomes.py:476-503`), and `_fill_intraday_checkpoints`
sets `r_at_eod` from the *last provided bar*, whatever day it is
(`outcomes.py:428-439`; the swing path shares this). Nothing constrains
`m5_by_symbol` to the entry session in `build_outcomes`. And if simulated
mid-session, a live trade is labeled `EXPIRED` with an interim close — there
is no `OPEN`/`TRUNCATED` handling and `as_of` is never consulted. EOD
semantics must be enforced inside the simulator (filter to the entry session,
and emit `OPEN` until `session.rth_close_at <= as_of`).

### S3 — edge/latent

**D14. Compaction crash → partition double-count on reconcile.** BD-03's
adopt-orphan hash guard compares against *registered* hashes in the partition
(`store.py:585-621`). A crash between a compaction's `os.replace` and its
manifest append leaves the merged file orphaned with a hash matching nothing
registered; reconcile adopts it as a fresh PUBLISH while the source part files
remain live — every row in the partition is then counted twice, silently
(the next compaction even balances, since both sides double). Reconcile
should refuse to adopt (quarantine instead) any orphan whose row set is not
disjoint from the partition's live rows, or compaction should write a
pre-intent marker in `_incoming/` — either is compatible with the 4-step seal.

**D15. Year-boundary dedupe holes.** `record_occurrences` resolves existing
revisions from a single year partition (`occurrences.py:203-204`): a December
occurrence rescanned in January appends a second `rev-1` row in the new year's
partition — episode inflation at the boundary. Similarly `build_outcomes`
checks `year={event_at.year}` but `outcome_path` partitions on `computed_at`
(`outcomes.py:672-677`, `schemas.py:597`): recomputing in a later year misses
the existing row and duplicates it. Both should look at adjacent-year
partitions (or partition occurrences/outcomes by a stable time column).

**D16. Empty episode key collapses distinct episodes.** `_episode_key`
falls back from `anchor_instance_id` to `episode_start`
(`occurrences.py:85-87`) but `build_occurrence_row` accepts a detection with
*neither*; `_identity_token(None) = ""` so a March thesis and a November
thesis on the same (symbol, setup, side, timeframe) hash to one
`occurrence_id` and one episode forever. The detector-adapter contract
(BD-44) must make one of the two mandatory, and `build_occurrence_row` should
reject its absence instead of hashing an empty token.

**D17. `_epoch_to_utc` guesses UTC for naive strings.** The fallback parses
`"%Y%m%d %H:%M:%S"` as UTC (`ib_capture.py:132-135`), but IB's naive strings
are exchange-local: if a TWS build ever answers formatDate=1-style, bars shift
4–5 hours silently. BD-06's own rule (naive = quarantine, never localize)
says this fallback should drop the bar, not re-zone it.

**D18. `collection_gap.expected_bars` stores the missing count.**
`record_collection_gaps._add` writes `expected_bars - captured` into a column
named/documented as the expected count for the gap interval, while
`gap_start/gap_end` span the whole session (`bar_archive.py:552-584`). Either
narrow the interval or store the expected count; the Health coverage tile
currently sums this ambiguous number.

**D19. The build job never runs Phases 5–6 (or the D1 wrap, or backups).**
`run_build` covers reconcile → spool seal → bronze → universe/geometry
snapshots → sessions → derived → weekly → retirement (`cli.py:137-148`). It
never calls `ingest_daily_bars` (the `bar_d1` wrapped read!), anchors,
feature snapshots, occurrences, outcomes, `backup_class_a/b`, or the backfill
jobs. BD-20/BD-44 declare the tee and adapter gaps, but this one is
undeclared: as shipped, the EOD build produces no `bar_d1`, no features, no
outcomes and takes no backups. Needs a BD entry and a wiring decision.

Minor notes (no separate writeups): `slice_readout.n_open` counts
`NO_TRIGGER` rows as open (`maturity_at` null); `IbCaptureFetcher` and
`run_backfill` both call `note_error` for one failure, double-escalating the
backoff; `single_flight`'s reclaim (unlink + recreate) has a two-process race
window; the tee dedupe key (symbol, interval_start) collapses the declared
provider/revision grain of `bar_m5` — fine while the tee is the only writer,
but it also means a Yahoo row blocks the IBKR row for the same interval;
`_shed_segment`'s `unlink` can raise `PermissionError` on Windows if the CLI
is reading the segment; a `focus` symbol in both focus files produces
duplicate grain keys in `universe_membership_daily`.

## 3. Point-in-time — what held

BD-35's fix is correct and I found no sibling leak in the snapshot keying
itself: intraday rows see bars `<= interval_start` only; the M15/M30 EMA join
takes bars with `interval_end <= interval_start` (conservative — at bar S's
close an M15 bar closing at the same instant is excluded, which is the safe
side of sec 5.4); daily snapshots filter `session_date > session_date` out
and use completed bars only; `universe_membership_daily` is first-capture-only
(LD-05); level snapshots stamp `known_at` at observation (armed_at where the
artifact carries one); anchors carry `system_from` = ingest time, so an
anchor is never retroactively knowable. The forming-bar rule is enforced at
the tee (`end > observed_at` → skipped), in aggregation (`end > as_of` → no
row), and in W1 (week publishes only after its final close). The genuine PIT
problems found are D1 (stale-as-matured), D5 (window truncation — a
completeness/parity defect, not a leak), and one caller-contract hazard: the
`bands` passed to `simulate_swing`/`_house_management_r` are a static band
set with no as-of discipline — nothing pins them to the signal date, and
bands computed later than the trigger would be look-ahead. The Phase-6 wiring
must pin `bands_by_occurrence` to the trigger session's snapshot (and decide
static-vs-daily-refreshed bands as part of the recipe definition).

## 4. Episode counting

The deterministic key + revision scheme (BD-37) is right and tested: rescans
revise, `episode_counts`/`slice_readout` report rows, occurrences, and
episodes separately, and outcome rows join through `latest_occurrences`. The
holes are D15 (year boundary) and D16 (empty episode key) above, plus one
definitional consequence under BD-38 (below). No path turns N rescans into N
samples inside a year.

## 5. Store integrity

Seal/manifest/quarantine/retire/read-path are the strongest part of the build:
the 4-step seal is exactly the plan's, the torn-tail-vs-corruption split
(BD-04) is right, quarantine preserves rows verbatim with manifest lines,
retirement fails safe on sharing violations, manifest-resolved reads give
consistent snapshots under compaction (tested), and the restore check
re-verifies manifest hashes into a new root. Exceptions: D14
(compaction-crash adoption), D9 (spool re-seal — fixed), and D2 (the lock
probe — fixed). One observation: `reconcile` quarantines orphan files of
bronze datasets whose specs haven't been registered that process
(`store._dataset_partition_from_path` → `dataset_spec` KeyError only for
non-`bronze_*` names — actually covered; no action).

## 6. BD-entry verdicts

Where I disagree or must qualify (everything not listed: agree as written —
notably BD-01/02/04/05/06/07/08/09/10/11/12/13/14/15/16/17/19/21/22/24
(subject to D17), 26 (spot-checked the observance and half-day rules against
2025–2027, including the two awkward cases named), 27, 28, 29, 30 (verified
`_calculate_vwap_bands` touches no instance state), 33, 34, 36, 40, 43
(subject to the UTC-date bucketing note), 45, 47, 48, 49, 51):

- **BD-42 — disagree that the claim is delivered.** Management is executed but
  wrong twice (D3, D4). The entry's own reasoning ("declaring a policy while
  simulating a plain stop/target would make every reported R wrong in the
  same direction") indicts the current state: partial-then-stop rows are wrong
  in the *pessimistic* direction and past-horizon rows in the optimistic one.
- **BD-46 — qualify.** Counts are separated correctly, but "an unresolved
  trade must not flatter a mean" is not delivered: stale-OPEN-turned-matured
  (D1) and TRUNCATED rows put interim R into the means.
- **BD-03 — qualify.** Adopt-don't-discard is right for publish retries; it is
  unsound for a compaction crash (D14). The reopen trigger fires now, in the
  compaction-shaped case rather than the byte-layout one it anticipated.
- **BD-18 — the age-cap interaction contradicted the entry's own guarantee**
  (D8, fixed). The shedding-order and mixed-segment rules are good.
- **BD-50 — right rule, broken probe on the platform it ships on** (D2,
  fixed).
- **BD-25 — understated.** "Unverified" was true, but the reconnect path had a
  concrete bug (D10) reachable by reading, without a broker.
- **BD-37 — qualify.** Sound within a year; D15 at the boundary.
- **BD-41 — right principle; incomplete.** TRUNCATED is derived correctly, but
  a TRUNCATED row still carries interim `gross_r` into matured means (D1
  family), and `maturity_at` itself deviates from 14.2's `min()` (D11).
- **BD-31 (`atr14`) — defensible.** Frozen column name, house TR method,
  difference from the scanner's 20 stated on the record: the right call. Two
  riders: `dist_sma50/100/200_atr` are therefore *not* the champion's
  distances (champion convention is ATR20) — say so in the ERD; and add
  `atr20` as an additive column at the first `feature_set_version` bump so
  scanner parity is checkable.
- **BD-32 — coord/residence/second_band_streak follow the plan's wording;
  the two builder definitions are defensible** and correctly quarantined
  behind `tier1_v1` + confirm-or-amend. One note for the trader on
  `band1_rejection_strength`: "most recent touching bar" can be arbitrarily
  stale relative to the snapshot; if the intent is "rejection quality *now*",
  a recency qualifier belongs in v2. Not a defect — a definition question.
- **BD-39 — legal but flag it.** Sec 12.1 permits a declared precommitted
  MOC, and the declaration is visible on every row, which is the right
  mechanics. The open question is factual, not legal: a real MOC must be
  committed by ~15:50 ET, so the recipe assumes the signal is knowable before
  the close it fills at. Whether that matches how the house actually takes
  these entries is the trader's call (see open item 6); if not, `next_open`
  becomes the primary under a new `recipe_id` and the MOC variant stays as a
  comparator.
- **BD-38 — agree on both stated choices** (side in, family out), with one
  consequence the entry doesn't state: because the episode key inside the
  cluster is `anchor_instance_id`-first, *two anchors on the same underlying
  move form two clusters* (e.g. EARNINGS_CURRENT and EARNINGS_PREVIOUS theses
  on one breakout). Sec 7.3's "several hypotheses about one episode" arguably
  covers that case too, and evidence floors would double-count it. See open
  item 5.

## 7. Recommendations on the open items

1. **Live wiring (BD-20/BD-52).** Owner: next builder session, with this
   design: a `warehouse_service`-owned capture object, constructed by the main
   desk only when `warehouse_enabled()`, invoked at the existing BounceBot
   cycle-completion point in the Qt service layer (never inside
   `bounce_bot_lib`); it snapshots `dict(bot.latest_bars)` on the owning GUI
   thread and calls `capture_m5_tee(spool=writer, seen=session_set)` —
   spool-only writes, no lake I/O on the GUI. Health page renders
   `warehouse_health_tiles` on its existing refresh cadence. Ruling on the
   pacer question: during the slice, capture issues **no RTH provider
   requests at all** (tee + nightly/weekly backfill only), so
   `note_champion_request` needs no champion-side call site and no champion
   path ever imports the warehouse. **Do not start the pilot until D1, D3–D7,
   D11–D13 are repaired** — 20 sessions of capture through the current
   outcome/backfill code would be evidence that has to be thrown away.
   **Done** — the wiring landed exactly as ruled here (BD-63), and every defect
   named in that gate is repaired. The pilot's remaining preconditions are
   operational, not code: the Windows/3.14 run, item 2's broker-marked run, and
   the trader's confirmation-register items.
2. **`build_ib_transport` (BD-25).** Do not defer. The nightly ETH backfill is
   inside the slice (LD-03), so the pilot leans on this path from night one.
   One broker-marked run on the desk (after the D10 fix), covering connect,
   one historical request, and a forced reconnect, before the pilot's first
   night.
3. **`exploration_cohort.txt` (BD-12).** Trader input; correctly left empty.
   Flagged, no agent action.
4. **Favorite-zone definitions (BD-32).** Trader confirm-or-amend; see the
   staleness note above. A change is a `feature_set_version` bump.
5. **`dependency_cluster_id` (BD-38).** Side-in/family-out is right. Rule
   needed on the anchor consequence: if two anchors on one move should share
   an episode, the cluster's episode component should be the move window
   (e.g. trigger-week bucket), not the anchor id. Recommend: keep v1 as
   built, log the double-count risk in the ERD, and revisit only if the slice
   actually produces multi-anchor moves — but decide *before* any evidence
   floor is enforced.
6. **Signal-close MOC (BD-39).** Confirm with the trader that house swing
   entries are genuinely at the signal close; add `next_open` as a control
   recipe at the first Phase-6 revision either way — it bounds the
   entry-assumption sensitivity for free.
7. **`atr14` (BD-31).** Keep; add `atr20` additively later; note the
   `dist_sma*_atr` convention difference in the ERD.
8. **Session VWAP STANDARD-only (BD-34).** Fine for the slice (both slice
   setups are D1 families). Wrap DYNAMIC/EOD when the Phase-6 bounce-context
   join lands — additive rows, no schema change.
9. **Bounce link ±60 min (BD-43).** Acceptable interim. Two adjustments when
   touched next: bucket by session date (exchange-local), not UTC date, and
   replace the window with an explicit key the first time the bounce ledger
   schema changes anyway.
10. **DuckDB pin (BD-45).** Install once on the desk; on any failure drop the
    pin and stay on pyarrow (LD-04's declared fallback). Nothing depends on
    it; zero risk either way.
11. **(Undeclared) build-job coverage (D19).** Add to the open-items table:
    decide what the EOD build actually runs — at minimum `ingest_daily_bars`,
    anchors, daily features, and Class A/B backups belong in it before the
    pilot; occurrences/outcomes wait on the BD-44 adapter.

## 8. Test-gate caveat, confirmed

The Linux-agent gate genuinely does not cover the desk: D2 would have crashed
the Windows suite outright (`test_a_second_build_refuses_rather_than_racing`
terminates its own process via `os.kill(pid, 0)` → `TerminateProcess`), and
D10 passes offline because every fake transport defines `connect(spec)`. The
Windows/3.14 run must happen after these fixes, and its number re-baselined in
`SOL_PROGRESS.md`.
