# Swing-quality demotion, pre-close honesty, and the dislike-feedback loop — packet R3

Status: **BUILD AUTHORIZED — IN PROGRESS** for `plan.md` Phase 0.5 **R3**. The
trader's 2026-08-15 weekend redirect was: *"integrate the rest — build R3 through
R6 on the consolidated branch."* R3 builds first on
`phase05-r8-weekend-prep`; the pre-close
investigation in section 4 is **complete** — its findings are recorded fact, its
fixes are not yet built.

This packet is the trader's stated core theme for the week: *"we were spammed with
a lot of trades/tickers we had already checked… there are a lot of OBVIOUSLY
overextended stocks or stocks in deep S/R that we can just not recommend unless
they were VERY strong and had high rvol in which case maybe we can daytrade… I
want a process to help the program improve over time with AI to send me less
trash"* — while still surfacing the bangers.

## 1. Trader decisions recorded (2026-08-15)

| Decision | Answer |
|---|---|
| Filter behavior | **Demote + label, never hide.** Demoted rows move to a clearly-marked section at the bottom of the report with the reason named. No suppression anywhere. |
| First-cut extension rules | **ATR distance from a reference EMA** and **outside the AVWAP deviation bands**. (The near-major-D1-S/R rule was offered and *not* selected for v1 — see 3.4.) |
| Daytrade carve-out | Exists either way: very strong + high RVOL demoted names get a `daytrade_candidate` annotation. |
| "Already checked today" | Derived from **recorded decisions only** (✕/★/veto/like/note) — no view tracking. |
| Pre-close fix | The **full honesty bundle** (section 4.3). |

## 2. What exists today (recon 2026-08-15)

- Setup rows at rank time already carry almost everything needed:
  `current_band_zone` (categorical AVWAP-band position), `atr20`,
  `hv_level_blocking_count`/`hv_level_nearest_distance_atr`/
  `hv_level_blocking_summary` (D1 S/R context — **always attached** every scan;
  only the −10 pt penalty behind `hv_level_scoring_enabled` is opt-in and off),
  `expected_r_*`, tier/bucket fields (`scripts/master_avwap_lib/legacy.py`,
  enrichment at `runner.py:1806` before ranking at `runner.py:1918`).
- **No RVOL field** exists on swing rows; `compute_relvol()`
  (`master_avwap_lib/levels.py`) already runs on the same fetched daily frame and
  its last value just isn't attached.
- Two non-destructive demotion precedents exist: the S/A-tier Expected-R holdout
  (`TIER_S_DEMOTE_EXPECTED_R_BELOW`, note-stamped, row stays visible;
  `legacy.py:440, 26961-27052`, tests in `tests/test_tier_expected_r.py`) and the
  stdev-extended rerouting (`_priority_is_extended_stdev_zone`,
  `legacy.py:19588-19618`) — whose explicit family exemptions (sma-breakout,
  fresh post-earnings, mid-earnings-retest with favorite signals) let genuinely
  overextended names through into Best Swing Trades. That leak is a plausible
  direct cause of this week's complaint.
- **Same-day resurfacing is structural**: the only cross-scan memory
  (`scripts/master_avwap_bucket_state.py`) dedupes the D1-Focus/phone upgrade
  stream only; `write_priority_setup_report` and the Best-Swing-Trades list are
  rebuilt from scratch every slot with zero trader-already-reviewed awareness.
- Dislike capture exists but is disconnected: the setups-table ✕
  (`scripts/ui/panels/master_avwap_panel.py:901-1057`) takes **free text** into
  `pick_feedback.jsonl` + `alert_review_events.jsonl`, while Chart Review's
  versioned veto vocabulary
  (`scripts/ui/annotations/vocabularies/veto_reasons_v1.json`) already has the
  exact codes (`too_extended_from_base`, `overhead_horizontal`,
  `incoming_trendline`, `support_resistance_cluttered`, …). `review_learning.py`
  **never reads the reason text** — a trader typing "too extended" fifty times
  produces zero counted signal.
- Day-scoped presentation-only state has a proven pattern:
  `scripts/alert_review_state.py` (`load_day_scoped_flags`, market-date reset).
- Phone digest swing picks derive from the same bucket/tier classification
  (`scripts/autopilot_core.py:2803-2999`), so an upstream demotion propagates to
  the phone automatically.

## 3. Design — less trash, mechanically

### 3.1 Deterministic swing-quality classifier (demote + label)

New pure step `apply_swing_quality_demotion(rows, settings)` running immediately
after `apply_expected_r_ranking`, before tier partition and Best-Swing-Trades
selection. Verdict per row from the trader's two v1 rules (both thresholds
settings-tunable, defaults to be confirmed against a week of real rows):

- **EMA extension**: last close more than `swing_quality_ema_atr_max` (default 2.0)
  × ATR20 above (longs) / below (shorts) the D1 21EMA — computed from the daily
  frame already loaded for the scan.
- **Band extension**: `current_band_zone` beyond the 1st deviation band in the
  trade direction (i.e. the zone families the stdev tracker already recognizes) —
  **with no family exemptions**, unlike `_priority_is_extended_stdev_zone`.

Effect: stamp `swing_quality_demote_note` (naming the rule and the measured value),
exclude the row from S/A tiers and `_priority_best_swing_trade_rows`, and render it
in a labeled **"Stretched — demoted"** section at the bottom of the report and the
desk table (visually distinct, never removed). The phone digest inherits the same
membership. Rows are never deleted, never hidden.

### 3.2 RVOL field + daytrade carve-out

Attach `relvol` (last value of `compute_relvol` on the already-loaded frame; zero
new fetches) to every row. A demoted row with `score`/`expected_r` above a strength
floor AND `relvol >= swing_quality_daytrade_rvol_min` (default 2.0) additionally
gets `daytrade_candidate` in its note and a distinct marker in the demoted section.
Annotation only: no alert, no watchlist write, no BounceBot involvement.

### 3.3 Already-reviewed-today badge (decisions only)

At report/table render, build today's decided set from `pick_feedback.jsonl`,
`alert_review_events.jsonl`, and `trader_annotations.jsonl` rows stamped with
today's `trade_date`/session (✕, ★, veto, like, note). Presentation only, cached
via the `alert_review_state` day-scoped pattern: a badge on the setups-table row
and snapshot header (packet R4 renders it on charts), and a "Reviewed today" group
marker in the report. It never filters, reorders, or feeds scoring.

### 3.4 The learning loop — how the filter improves over time

1. **Structured reasons at the desk**: the setups ✕ dialog offers the
   `veto_reasons_v1.json` codes (plus optional free-text detail) instead of a bare
   text box; Alert Center's new LIKE/veto capture (packet R4) uses the same
   vocabulary. The vocabulary file is append/version-only per its own contract.
2. **Counted, not skimmed**: review events carry the chosen code(s);
   `review_learning.py` gains a `dislike_reason` dimension and folds
   `surface='setups'` episodes in, so the scoreboard mechanically counts (e.g.)
   `too_extended_from_base` frequency by segment.
3. **Curation stays advisory**: the AI reads the scoreboard and proposes (a)
   `review_policy.json` annotation/priority-delta updates (annotation-only,
   unchanged contract) and (b) **threshold-tuning proposals for 3.1's classifier**
   (e.g. "80% of your dislikes in segment X carried too_extended_from_base;
   consider tightening `swing_quality_ema_atr_max`"). The trader approves; only
   then do the deterministic thresholds change, with fixtures updated.
4. The offered-but-not-selected **S/R-headroom rule** stays staged here: the
   `hv_level_blocking_*` fields are already attached free every scan, so if the
   counted dislikes show `overhead_horizontal`/`incoming_trendline` dominating,
   arming a third rule is a threshold decision, not new data work.

No suppression field exists anywhere in this chain, and none may be added.

## 4. Pre-close scan honesty (investigation COMPLETE 2026-08-15; fixes authorized as the "full honesty bundle")

### 4.1 Why setups "totally change after the close" — found mechanisms

1. The live scan **includes today's forming D1 bar** with no completed-bar guard
   (`runner.py:624, 631`; the completed-bars-only truncation already exists and is
   tested, but only in the tracker-backfill path,
   `legacy.py:22187-22204, 23010-23130`).
2. AVWAP + all six sigma bands recompute over the forming bar every scan
   (`calc_anchored_vwap_bands`, `legacy.py:15733-15773`).
3. ATR20 includes the forming bar (`legacy.py:15929-15953`) and feeds every
   bounce/cross tolerance.
4. Binary bounce/cross events gate on today's still-moving close
   (`legacy.py:15963-16045`) — setups appear/vanish outright between scans.
5. A 24-pt adverse-entry-candle penalty reads today's forming wick/close position
   (`legacy.py:23400-23505`, applied 18798–18848) — flips a name ~0.2–0.4 R of
   Expected-R prior via the steep `expected_r.py:35-60` anchor table.
6. A second 6–14 pt rejection-candle penalty stacks the same instability
   (`legacy.py:3458`, constants 805–811).
7. The 18-pt volume-thrust bonus compares today's **cumulative-so-far** volume to a
   20-day **full-day** average (`legacy.py:25708-25775`) — structurally near-unfireable
   before the close, so it systematically appears in the after-close list. The
   correct time-of-day-normalized primitive already exists unused here
   (`scripts/rvol.py:35-160`).
8. The persistent tracker is written **twice in the final hour** — 12:00 PT and
   13:00 PT — and the close write wipes and rebuilds the 12:00 write
   (`autopilot_core.py:128-147`; `legacy.py:1754-1772` — the gate's own comment
   says "EOD-only" but fires from close−1h; wipe-and-rebuild at
   `legacy.py:10397-10466`). The close-slot run itself finishes ~13:20–13:28
   (measured 2026-08-11 13:23:59, 622 rows).
9. The trader-facing report and tier CSVs are rewritten on **every** slot,
   unconditionally (`runner.py:2185, 2510-2530`).

Not causes (verified): prior-day H/L rollover and earnings-anchor refresh are
stable intraday.

### 4.2 The honest framing

A D1 bar genuinely is not final until the close. No scan schedule makes a pre-close
list as stable as a post-close one; the fix is to **bound the window and make the
preview honest**, not to pretend the forming bar can hold still.

### 4.3 The authorized fix bundle

1. **Tracker writes once, post-close.** Anchor
   `should_update_setup_tracker_now`/`slot_writes_setup_tracker` to the actual
   close; the existing tracker-staleness catch-up (completed-sessions-only,
   already tested) remains the safety net for a failed close run.
2. **Add a 12:45 PT near-close slot** to `get_autopilot_swing_slots` so the last
   actionable pre-close read has ~15 minutes of forming-bar risk, not 60.
3. **Stamp every report/tracker row** `bar_status: forming|completed` using
   `market_session.is_within_regular_market_session` (exists, currently unused by
   `master_avwap_lib`).
4. **STABLE + PREVIEW split**: each scan also evaluates on completed bars only
   (reusing the tested `_evaluate_priority_snapshot_for_date`-style truncation) and
   the report presents the STABLE list beside the live PREVIEW list, labeled.
5. **Volume-thrust time normalization** via `rvol.same_slot_baseline` — this one is
   a true scoring change: it ships behind its own golden fixtures and may be
   sequenced last within the packet.

The sigma formula is untouched throughout (hard invariant).

## 5. Fenced files, fixtures, and invariants

Everything here touches scoring/report territory. Ask-first at edit time:
`scripts/master_avwap_lib/legacy.py`, `expected_r.py`, `levels.py`, `runner.py`,
`scripts/autopilot_core.py`, `scripts/master_avwap_bucket_state.py`,
`scripts/review_learning.py`, `review_events.py`, `review_policy.py`,
`pick_feedback.py`, `scripts/ui/panels/master_avwap_panel.py`,
`scripts/ui/panels/alert_center_panel.py`. Golden characterization fixtures come
**first** for 3.1, 4.3.1, 4.3.4, and 4.3.5 (decision 0009; extend the
`tests/test_tier_expected_r.py` pattern). `review_policy.json` keeps no suppression
field. Completed bars only; the STABLE pass is the invariant restored, the PREVIEW
pass is the labeled exception.

**Fixture milestone (2026-08-16): BUILT before production edits.**
`tests/fixtures/r3_swing_quality_v1.json` and
`tests/test_r3_swing_quality_characterization.py` freeze current live Best Swing
membership, current tracker-slot timing and current full-day volume-thrust behavior.
They include up/down trends, a session gap, a mid-window missing value, a
forming/completed pair, exact threshold edges, side mirrors, and explicit mutation
checks for flipped, inclusive and wrong-field comparisons. Focused baseline:
63 passed, exit 0. The volume fixture deliberately documents rather than resolves
the missing intraday-slot data seam described in §4.3.5.

**Shadow-classifier milestone (2026-08-16): BUILT, live shadow week owed.**
`apply_swing_quality_demotion` runs after Expected-R ranking and stamps only
`would_demote`, named rule/measurement evidence, the D1 `relvol` reading and the
annotation-only daytrade carve-out. A bottom report section and desk badge expose
the calls without consuming them as a tier or filter. The focus payload and D1
feature CSV carry the same evidence. Before/after tests prove Best Swing ordering
and S/A/B membership are identical. No watchlist, alert, score, bucket, tier or
phone membership consumes the stamp. The trader's full shadow week is still
UNKNOWN and must be accepted before a future demote-and-label presentation change.

**Post-close scheduling milestone (2026-08-16): BUILT, live comparison owed.**
The regular-session schedule now includes a close-minus-15-minute preview (12:45
PT on a normal session), while both the scheduler's tracker flag and the scanner's
wall-clock gate begin at the actual close. Thus the ordinary schedule has exactly
one tracker writer, the 13:00 PT close slot; a later manual run and the existing
completed-session catch-up remain recovery paths. The provenance fixture retains
the old 12:00+13:00 writer set as characterization and names this intentional
difference. The completed-bar STABLE report and row stamps remain the next part of
the same honesty bundle.

## 6. Exit gates

- Fixtures land before any classification change; full suite green.
- One week of stamped output on the desk: the trader confirms the demoted section
  catches the junk without eating bangers (the demote-not-hide choice makes
  misfires visible and cheap).
- The 12:45 slot's list vs the post-close list compared for one week; the STABLE
  list's day-over-day churn visibly lower than the PREVIEW list's.
- The scoreboard shows dislike-reason counts accumulating; the first AI curation
  cycle produces at least one threshold proposal the trader can judge.

## 7. Open questions

- Default thresholds for 3.1 (2.0 × ATR20 from the D1 21EMA; beyond-1st-dev band):
  confirm against a measured week of rows before freezing fixtures.
- Whether the demoted section should also appear in the phone digest or stay
  desk-only at first (default: include, clearly labeled, since the digest inherits
  membership anyway).
- Sequencing of 4.3.5 (volume-thrust) — it changes which names carry an 18-pt
  bonus; the trader may want to see a week of stamped evidence first.


## Amendment — 2026-08-15 (R2.1 item 7, external review)

Two conditions bind R3 before any of its classifier work can change what the
trader sees.

### A `would_demote` shadow week runs BEFORE any row moves

R3's quality classifier must not demote a live row on the day it lands. It runs
in shadow first: for a **full week of sessions** it records a `would_demote`
verdict and its reason per row, changes nothing, and **no row leaves S/A tier or
the Best Swing Trades list** during that week. The trader reads the shadow
output and confirms the calls are ones they would have made.

This is the same discipline plan.md sec 7 applies to every challenger, and it
matters more here than usual: demotion is subtractive. A shadow week that
over-demotes is a spreadsheet to argue with; a live one is setups the trader
never saw, and they cannot review what was removed before they looked.

Only after the trader accepts the shadow week may demote-and-label affect
presentation - and even then it **labels rather than hides**, per the
2026-08-14 decision already recorded above.

### Fixture rules: flat series only where flatness IS the case

The R2 packet learned this the expensive way. Its strength formula read `C50`
as a 50-bar average instead of the close 50 bars back, and **every fixture
passed**, because they were built on flat bars where an average and a displaced
close are identical. The error survived precisely the tests meant to catch it.

R3's fixtures must therefore include, for every rule with a threshold:

- **a trending series** (up and down), where an average and a displacement, or a
  smoothed and an unsmoothed input, give visibly different answers;
- **a gap** across a session boundary, so a rule that silently spans sessions is
  distinguishable from one that does not;
- **NaN and missing bars** in the middle of the window, not only at the edges;
- **a forming-versus-completed pair** - the same series evaluated one bar early
  and one bar late - so completed-bar handling is proven rather than assumed;
- **mutation-seeded counterexamples**: for each threshold, a fixture that fails
  if the comparison is flipped, off by one, or applied to the wrong field. A
  fixture that still passes under a deliberately broken implementation is not
  evidence.

A flat series is legitimate **only** where flatness is the property under test
(for example "a motionless name is unmeasurable, not weak"). Anywhere else it
is a fixture that cannot fail.
