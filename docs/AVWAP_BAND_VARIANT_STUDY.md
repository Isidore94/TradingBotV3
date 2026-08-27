# Anchored-VWAP band variant: replicate the other program's σ, then test it against the champion

Document role: **PROPOSAL / study plan** (trader-directed 2026-08-26). It authorizes
nothing. It enters the build sequence only when the trader promotes it into
`plan.md`; until then no code in this document exists.

What it is NOT: a change to `calc_anchored_vwap_bands`. Decision 0008 and
plan.md §5 freeze the champion's running-deviation σ, and every band consumer —
events, zone arms, tracker stops, favorite zones, warehouse features — is
calibrated to it. The second formula is a **challenger** under the plan.md §7
ladder: computed beside the champion, stored beside it, graded against it, and
promoted (if ever) as an additional level family, never as a swap.

---

## 1. What the two charts show, and the one inference that matters

Same symbol, same earnings anchor, two programs (screenshots 2026-08-26):

| | TradingView (trader's own `AVWAPE` script) | The other program |
|---|---|---|
| Width at the anchor bar | **zero** — bands start on the AVWAP line and fan out | **non-zero from the first bar** — visibly wide immediately |
| Width at the last bar | AVWAPE 241.29, +1σ 252.92 → σ ≈ **11.6** (4.8% of VWAP) | +1σ ≈ 258 on a VWAP ≈ 240.5 → σ ≈ **17–18** (≈7.2%), read off the axis, ±1 |
| Ratio, other ÷ trader | | **≈ 1.5×** after ~13 bars |
| What price did | blew through +1σ two bars earlier | the last red bar's high rejected within a point of +1σ |

The trader's script behaves exactly like the codebase champion
(`scripts/master_avwap_lib/legacy.py:16320`): at the anchor bar the running
AVWAP *is* the typical price, so the deviation is 0 by construction and the
first σ is 0.

**The inference that narrows the search:** any formula whose only inputs are one
typical price and one volume per D1 bar has σ = 0 on the anchor bar — one sample
has no dispersion, whichever way you accumulate it (running deviation, or the
"distribution" σ around the current mean that TradingView's built-in uses). A
band that is wide on bar 1 must therefore be doing one of three things:

1. **using information inside the bar** — its high/low range, or the intraday
   prints the bar was built from;
2. **not measuring dispersion at all** — a percentage of VWAP, or an ATR multiple;
3. **anchoring earlier than it looks** — one bar before the gap bar, so bar 2 is
   the gap and the deviation is huge at once.

The ≈1.5× width at bar 13 is the second constraint: (2) and (3) can explain the
start but not a persistent 50% excess on a 13-bar trending tape, while (1) can
explain both, because within-bar dispersion adds to between-bar dispersion on
every bar, not just the first. So the leading theory was (1); the protocol below
is built to *kill* candidates, not to confirm the favourite — and §2b records
the first kill round, run the same evening on the trader's OKTA reading.

One more observation to verify with hover values rather than eyes: on the second
screenshot the lower cyan line ends near 229–230, which is ~11 below the VWAP
while the upper is ~17 above. If that line is the −1σ band, the bands are
**asymmetric** and the formula is not ±kσ (semi-deviations, or an upper anchor
different from the lower). If it is an EMA, ignore this. A single hover on that
line settles it.

## 2. The candidate set (what the fit will try)

All candidates share the AVWAP line (Σ tp·v / Σ v since the anchor) and differ
in σ. `tp` = typical price; `V_t` = running AVWAP at bar t; sums run from the
anchor `a` to `t`; `w_i` = volume.

| id | σ² at bar t | σ on anchor bar | End width vs champion | Killed by |
|---|---|---|---|---|
| **C0 champion** | Σ w_i (tp_i − V_i)² / Σ w_i (deviation from the *running* VWAP at each bar) | 0 | 1.0× | control, never killed |
| **C1 distribution** (TradingView built-in) | Σ w_i tp_i² / Σ w_i − V_t² (deviation from the *current* VWAP) | 0 | wider on trends, typically 1.1–1.4× | non-zero width on the anchor bar |
| **C2a range, uniform** | C1 or C0 + Σ w_i (H_i − L_i)²/12 / Σ w_i (price uniform inside each bar) | (H−L)/√12 of the anchor bar | wider always | anchor-bar width ≠ (H−L)/3.46 |
| **C2b range, H/L samples** | Σ w_i [(H_i − V_t)² + (L_i − V_t)²] / (2 Σ w_i), optionally with C too | (H−L)/2 | wider always | anchor-bar width ≠ (H−L)/2 |
| **C3 intraday prints** | volume-weighted dispersion of every intraday bar (1-min or 5-min) since the anchor, around the same AVWAP; the D1 chart only *displays* it | intraday dispersion of the anchor day (non-zero, sizeable on a gap day) | wider always; ratio depends on how much of the move is intraday | disagreement with our M5-store computation at any point |
| **C4 percent / ATR** | width = p·V_t, or k·ATR_n | non-zero | constant ratio to VWAP or ATR | width ÷ VWAP (or ÷ ATR) not constant across bars |
| **C5 anchor offset** | any of the above anchored at a−1 (pre-gap bar) or at the earnings *date* rather than the reaction bar | — | — | the AVWAP line itself disagrees at bar 1–2 |
| **C6 nuisance dimensions** | typical price ∈ {ohlc4, hlc3, hl2, close}; population vs sample (n−1); σ of close vs σ of tp | — | small | fitted jointly; cannot explain the start width alone |
| **C7 asymmetric** | upper from bars above V, lower from bars below V (semi-deviation) | — | — | symmetric hover readings |

Fit grid: {C1, C2a, C2b, C3} × {ohlc4, hlc3, close} × {pop, sample} × {offset
0, −1} plus C4 and C7 — about fifty closed-form candidates, seconds to run.

Known rules of thumb for the check: on a 19-point earnings bar (the visible one
runs roughly 233–252), C2a gives σ₁ ≈ 5.5, C2b ≈ 9.5, C3 somewhere between
depending on how the day traded. The screenshot's first-bar half-width looks like
4–6, which favours C2a or C3 — but that is a reading of pixels, and the whole
point of §3 is to replace it with numbers.

## 2b. First data point (OKTA, anchor 2026-05-29) — what it killed, what survives, what to hover next

The trader named the program (**OneOption / Option Stalker Pro**) and gave one
hover on OKTA at the anchor bar, 2026-05-29: AVWAPE **118.19**, +1σ **128.47**,
−1σ **107.90**. The vendor's release notes call these "true Standard Deviation"
bands (Aug 2024; "AVWAP(E,Q) Standard Deviation (SD) Line" dialogues, Feb 2024)
and publish no formula, so R0 does not short-circuit anything. Note the
vocabulary: OneOption's "AVWAP E / Q" (earnings / quarter) is where the
codebase's `AVWAPE` name comes from.

Read against the durable D1 store (`daily_bars/OKTA.parquet`, 2026-05-29:
O 107.535, H 124.790, L 106.501, C 123.270, V 17.58 M) — a scratch calculation,
no repository code:

- **Typical price is HLC/3, confirmed to the cent.** (124.79 + 106.50 + 123.27)/3
  = **118.187**; OHLC/4 would be 115.52. The trader's suspicion was right.
- **The anchor is that single bar.** A VWAP that equals one bar's hlc3 to the
  cent cannot contain an earlier bar (the pre-gap bar's hlc3 is 93.8), so C5
  (anchor offset) is **dead**, and so is intraday data for the centre line: an
  intraday VWAP landing within ±0.005 of the daily hlc3 by chance is ~1 in 1000.
- **The bands are symmetric** (10.28 above, 10.29 below): C7 is dead.
- **σ on a one-bar anchor is 10.28**, so C0 and C1 (0 on one bar) are dead as
  the *whole* answer, and the simple range forms are dead too: (H−L)/2 = 9.15,
  (H−L)/√12 = 5.28, rms of H and L around hlc3 = 9.49, rms of H/L/C = 8.29,
  1× or 2× ATR14 = 5.4 / 10.8. So is any percentage: nothing about 8.7% is natural.

Two candidates survive the anchor bar, and they diverge sharply afterwards:

| Candidate | σ at 2026-05-29 | How it works |
|---|---|---|
| **S1 — sample stdev of every O, H, L, C print since the anchor, around the running AVWAP** (n − 1 denominator; volume-weighted or not, identical on one bar) | **10.32** (0.4% off; an 11-cent difference in OneOption's *open* print vs IB's — the one price hlc3 does not check — closes the gap exactly) | a real anchored deviation: four prints per bar, so bar 1 already has dispersion, and later bars shrink or grow it with the tape |
| **S2 — 20-bar population stdev of closes** (a Bollinger σ centred on the AVWAP) | **10.284** (0.04% off) | no memory of the anchor at all; the width is the last twenty closes' volatility, which the gap had just blown up |

S2's exactness must be discounted: it was one of ~250 closed forms tried against a
single number, and at that count a 0.05% coincidence has roughly a one-in-four
chance. On the OneOption screenshot itself the bands at the last bar (2026-08-26,
close 134.42) read about 146 / 118 against a centre near 132 — S1 predicts
145.1 / 118.7 and S2 predicts 139.0 / 124.8 — so the pixels already favour S1, but
pixels are what §3 exists to replace. **Three more hovers on OKTA decide it:**

| Hover date | AVWAP (both) | S1 sample-OHLC (unweighted / vol-weighted) | S2 Bollinger-20 |
|---|---|---|---|
| 2026-06-01 | 124.85 | 137.91 / 111.80 · 138.04 / 111.67 | 140.18 / 109.53 |
| 2026-06-02 | 126.78 | 138.09 / 115.47 · 139.14 / 114.42 | 144.82 / 108.74 |
| 2026-06-03 | 126.84 | 136.79 / 116.90 · 138.26 / 115.42 | 145.88 / 107.80 |
| 2026-08-26 | 131.91 | 145.09 / 118.72 · 144.92 / 118.89 | 139.01 / 124.80 |

If the 06-02 upper reads ≈138 it is S1 (and the unweighted/weighted split is the
06-03 lower: 116.9 vs 115.4); if it reads ≈145 it is S2; anything else means
both are dead and R3 widens the set (an EWMA of deviations is the next natural
shape, since S1's σ *falls* from 13.1 on 06-01 to 9.9 on 06-03 while the
tape consolidates). Either way the AVWAP column should match every time — if it
does not, the volume inputs differ and R2's data check comes first.

**Second hover (trader, same evening): OKTA 2026-06-02 upper band = 144.6.**
S1 predicted 138.09 and is **dead**; S2 predicted 144.82 and is the family:
the width is a lookback standard deviation of closes laid on the anchored
HLC/3 centre — a Bollinger width on an AVWAP, with no memory of the anchor.
What is not yet pinned is the lookback and the denominator, because the
reading was taken without the AVWAP beside it: with the store's AVWAP
(126.78) the implied σ is 17.82, which n=22 population (17.78) or n=25 sample
(17.77) fit within 0.4% while n=20 population (18.04) is 1.2% off — but if
OneOption's AVWAP on 06-02 is 126.56 rather than 126.78 (a 0.17% volume-feed
difference), n=20 population is exact at both bars again. **One more hover
resolves it: the AVWAP and the lower band on 2026-06-02.** AVWAP 126.78 →
the width is n≈22 pop / 25 sample; AVWAP 126.56 → n=20 pop, the textbook
Bollinger σ, which is by far the likelier implementation.

**Third hover: OKTA 2026-06-02 lower band = 108.53.** Centre = (144.6 + 108.53)/2
= **126.565**, half-width = **18.035**. The store's 20-bar population stdev of
closes on 06-02 is 18.04. **Replicated:**

> `band_k = AVWAP_hlc3 ± k · stdev(close, 20, population)` — an anchored HLC/3
> volume-weighted centre with the textbook Bollinger σ as the half-width.

The centre differs from ours by 0.22 (126.565 vs 126.78, 0.17%) — a volume-feed
difference (OneOption's consolidated volume vs IB's), not a formula difference;
the anchor-bar centre matched to the cent because a single bar's VWAP does not
depend on volume at all. R2's tolerance of 0.2% on the centre is therefore about
right, and the σ needs no volume, so it reproduces exactly from any clean close
series.

**Merit or luck — the honest a-priori view, before T1 measures it.** Both
screenshots showing the OneOption band as "the level" are two anecdotes and
prove nothing; that is what the backfill is for. What can be said now:

- The *shape* has a sound reason to work early: recent realised volatility is a
  sensible scale for how far price can stray from an institutional cost basis,
  and it is on scale from bar 1. The champion's anchored running deviation is an
  *accumulated* dispersion — it starts at zero and fans out — so for the first
  five to ten bars after a fresh anchor its ±1σ is too tight to mean anything.
  The trader's screenshots are both early-anchor charts, which is exactly where
  the champion is weakest and any vol-scaled band looks good. The same
  principle already lives in this codebase as "distance is in ATR, never
  percent" (regime-pause hold, 2026-08-21).
- The *weaknesses* are structural: it is not a deviation from the AVWAP at
  all (the closes are measured around their own 20-bar mean, not around the
  anchored centre); on a gapping name the gap bar dominates the window for
  exactly twenty sessions and then drops out, so the band **jumps inward on
  bar 21** for no tape reason; and it has no anchor memory, so two anchors on
  the same chart carry identical widths.
- So the expected result of T1 is *not* "one formula wins": it is that the
  OneOption band is better in the 1–5 and 6–20 bars-since-anchor buckets and
  the champion is better later, with the crossover somewhere near the window
  length. If that is what the numbers say, the useful product is neither
  formula alone but a band whose width is vol-scaled early and
  anchor-accumulated later — and that is a *new* level family, tested through
  the same three harnesses, never a swap inside the champion.

**Data hazard found on the way, for R2:** the store's OKTA volumes are
mixed-unit — thousands before 2026-05-27 and again on 2026-06-04, shares
otherwise (`volume_unit = unknown` on every row). The table above normalised
them with a scratch rule; the real fit must go through the champion's own
normalisation (the `mixed_unit_avwap_v1` fixture exists for exactly this), not a
threshold.

What this does to the rest of the plan: §2's C2/C3 rows are superseded by S1/S2;
R1's hover list is now specific (the four dates above, on OKTA, then the same
five-bar pattern on two more names); R4's module implements the replicated S2 formula
above, with S1 pinned as the discriminator fixture. §4 is unchanged —
except that if S2 wins, the "variant" is not an anchored deviation at all but a
volatility band on an anchored centre, and T1's bars-since-anchor buckets become
the most interesting cut, because S2's width never fans out from the anchor.

## 3. Replication protocol

### R0 — name the program and read its documentation

The trader names the second program. If the vendor documents its band formula,
that short-circuits R3 and the fit becomes a parity check rather than a search.
The "Overnight" price tag on the axis suggests a platform with a 24-hour session
(the trader can confirm). Do not guess the vendor from the screenshot.

### R1 — the sample (trader, ~30–45 min)

Three to five symbols, ideally the one in the screenshot plus two more
earnings-gap names and one non-gap anchor (a pivot low), each with:

- the anchor bar's date and the trader's rule for choosing it (reaction bar vs
  report date — this is C5's whole question);
- hover readings from the OTHER program at **five bars**: the anchor bar, anchor+1,
  anchor+2, a mid bar, the last bar — `AVWAP`, `upper`, `lower` at each;
- the same five readings from the TradingView script (the control);
- a note of the program's typical-price setting if it exposes one.

The anchor bar and anchor+1 are the diagnostic points; the rest fix the tail.
Fifteen numbers per program per symbol. Written into
`tests/fixtures/avwap_band_variant_readings_v1.json` as they are, with the
symbol, the date, and which program produced them.

### R2 — control fit first

`scripts/avwap_band_variant_fit.py` (new, offline: reads
`data/daily_bars/<symbol>.parquet` via the playbook study's `_load_daily_frame`
and the durable M5 store; no network; writes a table to `OUTPUT_DIR/reports/`).

First it runs **the champion against the TradingView readings**. Pass criterion:
AVWAP within 0.2% and σ within 1% at every point. If the AVWAP disagrees, the
inputs differ — adjusted vs unadjusted closes, consolidated vs primary-exchange
volume, a different anchor bar — and that is fixed *before* any σ is fitted,
otherwise the search fits data noise. If the champion does NOT reproduce the
trader's own script, that is a finding in itself (decision 0008's rationale is
recorded as unknown) and is reported before going further.

### R3 — the search

Every candidate in §2 is scored on the other program's readings: score = max
absolute relative error of (upper − AVWAP) across all points, with the AVWAP
itself required to agree within 0.2%. C3 needs the M5 store for the sample
symbols (`warm_durable_bar_stores(include_intraday=True)` on the desk, or 1-min
bars pulled once by IB if 5-min is too coarse to match).

Acceptance: **one candidate within 1% at every point and the runner-up at least
3× worse.** Anything less is "not replicated": collect more points (a second
anchor on the same symbol is the cheapest), and widen the set (EWMA of
deviations, a lookback window instead of since-anchor) before concluding.

### R4 — freeze it

- `scripts/indicators/avwap_band_variants.py`: pure module in the `indicators/`
  shape (completed bars in, aligned tuples out, `None` where unmeasurable,
  `FEATURE_VERSION = "avwap_bands_<name>_v1"`; docstring states the formula
  verbatim and names the tempting wrong variants). It also carries a per-bar
  series form so a chart can draw it. It never imports the champion and the
  champion never imports it.
- Golden fixture from the trader's readings (raw-input hash, expected values)
  loaded through `load_fixture_contract`, plus a discriminator test proving the
  champion gives a *different* answer on the same bars (mirrors
  `test_the_distribution_variant_would_give_a_different_answer`).
- The existing champion guards (`test_mixed_unit_avwap_golden`,
  `test_warehouse_avwap_parity`) stay untouched and stay green — they are the
  proof nothing moved.
- Note: this is the first importer of `scripts/indicators/`; CLAUDE.md flags that
  as the packaging trigger. `indicators` is already in the spec's
  `collect_submodules` list, so the expectation is spec-drift green and a
  frozen selftest run rather than a spec edit — verify, do not assume.

### R5 — see it on a few stocks

Two deliverables so the trader can "test the theory on a few stocks":

1. the fit table itself (`avwap_band_variant_fit.py <SYMBOL> <ANCHOR_DATE>`
   prints champion and variant AVWAP/σ/bands per bar since the anchor — the same
   numbers to hover-compare against the other program on any new name);
2. a **D1 chart overlay**: a new paint-lines group "AVWAP σ variant" (±1/2/3),
   **default OFF**, built on the `ChartDataService` worker inside
   `scripts/chart_levels.py`'s `levels` payload beside the champion lines,
   drawn by `CandleChart.set_levels` with its own stable ids. Never on the paint
   path; not a detector file, so no ask-first fence, and no zone arm or alert
   reads it.

## 4. Testing protocol against the champion

Three harnesses, matching `docs/SETUPS_TEST.md`'s doctrine (backfill first, then
forward), all shadow, all pre-declared. **Fairness rule for every one of them:**
a wider band is touched less often and blown through less often *by
construction*, so no harness may report precision without recall. Tolerances
are in ATR, the same absolute distance for both formulas, never "0.2σ of its
own σ" — that would hand the wider band a wider tolerance too.

### T1 — level quality (new backfill, the direct question)

`scripts/avwap_band_level_study.py`, offline over the durable D1 store (~1,100
symbols × 400 sessions) with earnings anchors chosen point-in-time by the same
function the playbook study uses, plus the tracker's stored `current_anchor`
dates as a second, realistic anchor set.

For each anchor and each later bar `t`, the level known **before** the bar —
`U1_f(t−1)` for each formula `f` — because that is what a trader (and the zone
arms) act on. Then, longs on the upper bands and shorts mirrored on the lower:

- **touch**: `high_t ≥ U1 − tol` with `close_{t−1} < U1`, `tol = 0.15 × ATR14`;
- **respect** (precision): `close_t < U1` and no close ≥ U1 in the next 3 bars,
  with follow-through `min(low_{t+1..t+3}) ≤ close_t − 0.5 ATR`;
- **blow-through**: a close ≥ U1 + 0.5 ATR within 3 bars;
- **pivot capture** (recall): of confirmed swing highs (3 bars each side, resolved
  with hindsight — the *level* is point-in-time, the *outcome* is allowed to
  look forward as every outcome does), the share within `tol` of U1, and the
  mean |pivot − nearest band| in ATR;
- the same for ±2σ and ±3σ.

Reported by formula × side × regime window (the playbook's two windows) ×
bars-since-anchor bucket (1–5, 6–20, 21+): counts first, rates second, never a
rate without its n. The early bucket is where the two formulas differ most and
is the bucket the screenshot is about. Comparison is **paired per anchor** (same
anchors, same bars), with a sign test; the metric set and the window are frozen
in this document before the first run (plan.md §7 item 2).

### T2 — the playbook families under the variant's levels

`setup_playbook_study.py --band-formula variant`: the eight `needs_bands`
AVWAP families in `PLAYBOOK` (`first_dev_bounce`, `first_dev_breakout`,
`band_test_rebound`, `vwap_bounce`, `second_dev_*`, …) re-run with the variant
band context, graded by the same `measure_episode` against the same
`baseline_every5`, both regime windows, output
`OUTPUT_DIR/reports/setup_playbook_bandvariant_*.csv`. The switch lives in the
harness's band-context builder only; the detectors read `ctx.bands` and are not
edited. Answers: does a family's R move when its levels move, and in which
direction. Confirm at build time that the harness computes bands in one place;
if it does not, that is the first refactor and it is done before the switch.

### T3 — forward, in the setup tracker (what the trader asked for)

Shadow block, additive, no scoring change:

1. `runner.py` computes the variant beside `current_anchor_meta` as
   `current_anchor_variant = {"formula_version", "vwap", "stdev", "bands"}`
   (and the `previous_anchor` mirror), from the same frame and anchor index.
2. `build_tracker_setup_record` carries it, and the tracker gains **shadow stop
   scenarios** (`stop_variant_lower_1`, mirrored for shorts) evaluated by the
   existing per-bar scenario machinery, so every tracked setup accrues
   `total_r` under both stop families. `representative_total_r` stays the
   champion's primary-stop scenario, so no score, rank, tier or alert moves.
3. A new export `master_avwap_band_variant_stats.csv` (family × side × bucket:
   n, avg R champion vs variant stops, stop-out rate, target-hit rate, mean
   stop distance in ATR) written in the same pass as the other stats CSVs, and
   a **"Band variant" view in the Setup Tracker panel** that reads it — the
   panel is a pure CSV reader today and stays one.
4. Warehouse: additive columns `avwap_variant_upper_1..3` / `lower_1..3` and a
   `avwap_variant_formula_version` column; `FEATURE_SET_VERSION` bumps to
   `tier1_v2`, old rows keep `tier1_v1`, nothing rewritten.

Fences that apply to T3 and nowhere else in this plan:

- `master_avwap_lib/legacy.py` and `runner.py` house detector and scoring code:
  **every edit there is asked about before it is made** (file-scoped ask-first
  rule, 2026-08-08), including these evidence-only ones.
- Golden fixture first (plan.md §5): a frozen tracker-record fixture on the
  current code, and a parity test that the champion's setup records, scores and
  events are **byte-identical** with the shadow block present.
- The tracker JSON is already ~951 MB; the shadow block adds a few hundred
  bytes per setup. State the measured growth after the first save.

### T4 — decision criteria, declared now

The variant is "more effective" only if **all** of:

1. T1: respect rate **and** pivot capture both ≥ champion in the 1–5 and 6–20
   buckets, on both sides, in both regime windows, n ≥ 200 touches per cell;
2. T2: no AVWAP family's R vs `baseline_every5` gets worse by more than its
   standard error, and at least one improves by more;
3. T3: ≥ 20 sessions of forward accrual with ≥ 40 finalized setups, variant-stop
   avg R ≥ champion-stop avg R, and no increase in stop-out rate beyond what the
   wider stop's distance explains.

Failing any one is a real result and is recorded as one. Passing all three is
the input to a promotion decision under plan.md §7 that the trader makes;
the shape of that promotion is an **additional level family** (`VARIANT_UPPER_1`
… with its own zone arms and event ids), because swapping σ inside the champion
would move every calibrated threshold at once (decision 0008), and recalibrating
them together is a separate program.

## 5. Sequence, effort, and what each phase produces

| Phase | Work | Effort | Produces |
|---|---|---|---|
| **A — replicate** | R0–R5 | code ~½ day + trader ~45 min of readings; a second ½ day if the first fit fails | the frozen formula module, its fixture, the fit script, the chart overlay |
| **B — backfill** | T1, T2 | 1–2 days | two CSV reports + a short `docs/analysis/` record with the numbers |
| **C — forward** | T3 | 1–2 days of code, then ≥ 20 sessions of accrual | the shadow block, the stats export, the tracker view, warehouse columns |
| **D — decide** | T4 | trader | a plan.md §7 entry or a closed study |

Expected files (Phase A first; nothing exists yet):

- `scripts/indicators/avwap_band_variants.py` (+ `tests/test_avwap_band_variants.py`)
- `scripts/avwap_band_variant_fit.py`
- `tests/fixtures/avwap_band_variant_readings_v1.json`
- `scripts/chart_levels.py`, `scripts/ui/widgets/paint_lines_button.py` (overlay group)
- Phase B: `scripts/avwap_band_level_study.py`, a `--band-formula` switch in `scripts/setup_playbook_study.py`, `docs/analysis/AVWAP_BAND_VARIANT_RESULTS_<date>.md`
- Phase C: `scripts/master_avwap_lib/runner.py`, `legacy.py` (fenced), `scripts/ui/panels/setup_tracker_panel.py`, `scripts/research_warehouse/{features,schemas}.py`, tracker golden fixture + parity test

## 6. Open questions for the trader (Phase A's replication is done on OKTA; items 2–3 are now the extra names for the fixture)

1. ~~Which program~~ — answered: OneOption / Option Stalker Pro.
2. **Which symbols and anchor dates** for the sample (§3 R1), and the rule you
   use to pick the anchor bar — the earnings *reaction* bar, or the report date?
3. **Hover readings** — first the four OKTA dates in §2b's table, then five bars per
   symbol on two more names, from both programs.
4. Is your TradingView `AVWAPE` script exactly the codebase formula (ohlc4 typical
   price, deviation from the running AVWAP, volume-weighted)? R2 will check, but
   knowing the script's source avoids a false alarm.
5. Do you want the overlay (R5 item 2) at all, or is the printed table enough for
   the eyeball test?

## 7. Authority and placement

- Not in `plan.md`. If promoted, the natural home is the next free Phase 0.x packet (0.9 is now the GUI follow-ons)
  ("AVWAP band challenger") ahead of Phase 1, because it is a bounded study with
  a frozen metric, not a product feature; its promotion step belongs with Phase
  6's P6.2 setup-family ladder.
- `CHANGELOG.md` records nothing until Phase A lands code.
- `WISHLIST.md` row 81 ("capture DYNAMIC and EOD session-VWAP variants") is a
  different question (session VWAP interpretations, intraday) and stays where it
  is.
