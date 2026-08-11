# How the test/study setups work — AI-stated understanding

Document role: **active trader-review reference**, not a status or roadmap file.

Companion to `SETUPS_MAJOR.md`. This states how every setup that is **measured but
not trusted** works: the "Study (measured only)" and "Playbook research" families in
`scripts/setup_docs.py`, and the three research harnesses that generate and grade
them. **Purpose: Aaron reviews and corrects; corrections get folded back in.**

The governing rule (plan.md Section 7 and Phase 6): new setups enter only through research and
shadow stages, are judged on portfolio-level incremental expectancy (never on alert
volume), and are promoted only with versioned rollback. Until then they carry no
scoring weight — they annotate and accrue evidence.

## The three measurement harnesses

**1. Forward: the setup tracker.** Confirms setups the scan already promotes and
measures outcomes with the house exit discipline (level stops with close-failure
counts, band-2 partial / band-3 run / band-1 trail, 18-session time stop). Study
families ride along in the tracker as measured-only rows.

**2. Backfill: the playbook study** (`scripts/setup_playbook_study.py`). Goes the
other way — hypothesizes candidate families and backfill-measures them over the
durable daily-bar store so families are comparable in R. Its discipline (as coded):

- Entry at the **next session's open** — no same-bar fills.
- Representative stop 0.1 ATR beyond the signal bar's extreme; **intrabar
  stop-first** (a bar touching the stop is a stop-out; gaps fill at the open beyond
  the stop); time stop at the tracker's max hold; net of the round-trip cost model.
- One open episode per (symbol, family, side); overlapping signals ignored until the
  prior episode resolves.
- Shorts by **mirroring the price frame** — every family is written once in long form.
- Point-in-time: detection at date T uses only data through T, with the earnings
  anchor as it would have been chosen ON T. Known caveats: today's universe
  (survivorship) and ~90% earnings-cache coverage for band families.
- A `baseline_every5` control family (unconditional entry every 5th session per
  symbol) anchors everything: **a family is only interesting if it beats that**.
- Results are quoted across two regime windows (a correction/mean-reversion window
  and a bull window) — robustness means an edge in both.

**3. Reverse: move forensics** (`scripts/move_forensics.py`). Outcome-first: find
every big clean move (both sides), snapshot the conditions true *just before* each
start (no lookahead), and mine which conditions — alone and in pairs — were
over-represented at move starts vs matched ordinary days (**lift**). It can surface
novel patterns with no existing detector, and refinements (detector + context pairs
beating the detector alone). Lift is association only, not a measured R edge —
forensics candidates go into the playbook's forward machinery next. Outputs land in
`OUTPUT_DIR/reports/` (movers/baseline/patterns CSVs, report, AI digest JSON).

## Study families (measured only; no scoring impact)

- **Weekly 8EMA Hold + Retest** — basket study: names whose last 10 weekly candles
  ALL closed ≥ the weekly 8EMA (24+ weeks history required); rows record only on
  daily tag-and-reclaim entries (EMA15/EMA21/first-dev) inside the basket. The
  weekly-strong regime is the biggest single lever the backfill found (baseline
  -0.10R vs -0.71R in mixed names). Forward study accruing.
- **HTF EMA15 Rejection** — 1h/4h 15EMA pierce-and-close-back with both HTF trends
  aligned (session-aligned 4h resampling). Isolated study; not yet scored.
- **1st-Dev Breakout** — CROSS of UPPER_1/LOWER_1 measured as its own family to
  compare against retest entries. Small positive edge in both regime windows with
  tight stops (+0.42/+0.14R long).
- **2nd-Dev Breakout** — the deliberate **control for chase entries**: unstopped it's
  the second-worst long signal (-2.8% at 5 sessions); tight stops salvage a small
  edge. Scored near zero on purpose — the power *hold* is the tradable pattern, not
  the cross.

## Playbook research families (from the backfill; forward studies live)

- **Volume Thrust** — ≥1.5% close-to-close move on ≥2× 20-day volume, on the trend
  side of AVWAPE. The most regime-robust family found, both sides
  (+0.37/+0.49R long, +0.13/+0.33R short; t=3.4). Entry next open; stop 0.1 ATR
  beyond the thrust bar (AVWAPE as the level alternative).
- **2nd-Dev Power Hold** — the name *lives* beyond UPPER_2 (streak ≥ 10 sessions).
  Continuation **long only** — the short mirror snaps back and is deliberately not
  recorded. Stop UPPER_1 (the zone floor, 2-close): a close back inside the first
  band ends the regime. Strongest tracker factor found (+8.8% at 10 sessions, 89%
  win, n=45). Canonically, `mid_earnings_above_2nd_stdev` aliases to this family.
- **Quiet Pullback Resume** — three low-volume counter-trend sessions on the trend
  side of SMA50, then a resumption bar with the trend on rising volume. Robust both
  sides, both windows (+0.24/+0.32R long, +0.22/+0.34R short).
- **Golden Pullback + Volume** (forensics-derived) — trending name tags its rising
  SMA50 (within ~0.15 ATR) after 15 sessions on the trend side, holds it, with a 2×
  volume spike in the last 3 sessions. Strongest move-*initiation* combo in the
  2026-07-09 forensics (lift 1.87×/2.29×) — association only, forward study live,
  no weight.
- **Post-Earnings Volume Break** (forensics-derived) — within 5 sessions of
  earnings, ≥2× volume in the last 3 sessions, directional bar closing beyond the
  daily 8EMA. Forensics found the SHORT side (distribution) at 1.91× lift, n=324 — a
  family the scan never had; both sides recorded to confirm the asymmetry forward.

## How a test setup becomes major (as I understand the intended path)

1. Thesis + failure mode written; point-in-time features and universe defined
   (plan.md Phase 6 setup-promotion path).
2. Backfill measurement in the playbook (beats `baseline_every5`, edge in both
   regime windows) and/or forensics lift → forward study.
3. Forward accrual in the tracker with complete candidate retention; no scoring
   weight while accruing (families show as "study", weights parked).
4. Comparison against the existing portfolio: incremental expectancy after
   correlation with existing setups — not alert volume.
5. Promotion with a versioned rollback; only then do weights/scoring change
   (which itself requires golden fixtures first, per the sec 5 invariants).

⚠ Steps 4–5 are stated in plan.md but I did not find code enforcing a formal gate
(e.g., a promotion checklist artifact for setup families like the sec 7 ladder has
for engines). If that's meant to be manual judgment for setups, saying so here
would keep future agents from inventing one.

## Review notes / open questions for Aaron

1. ⚠ The formal promotion gate for setup families (above) — manual or to-be-built?
2. The two "regime windows" the playbook quotes edges across — confirm my reading
   (one correction window, one bull window) and whether new windows get added as
   regimes change.
3. `post_earnings_candle_break` came in trader-specified, skipping the backfill path
   (weight parked pending tracker history). Is that the intended second entry path
   for setups — trader conviction first, measurement after? Worth writing down.
4. Plan.md Phase 6 carries the one-family-at-a-time promotion program. Historical
   candidates included compression-to-expansion, failed breakout/reclaim, and opening
   drive pullback, but they have no study rows yet — are any next in line?
