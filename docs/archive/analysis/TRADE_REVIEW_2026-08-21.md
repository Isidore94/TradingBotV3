# Trade review — window 2026-07-24 … 2026-08-21

**Status:** analysis and recommendation only. No detector, scoring, or alert file was
modified. Nothing here is authorized work; every proposal in §8 is a candidate for
`plan.md` Section 12 and needs your explicit decision before it enters the build order.

**Produced:** 2026-08-22 early PT, against data as of 2026-08-21 23:46 PDT.
**Window:** 2026-07-24 … 2026-08-21 inclusive = **21 weekday sessions**, no US holiday.
**Zones:** desk-local = America/Los_Angeles (PDT, UTC−7). Market = America/New_York
(EDT, UTC−4). ET = PT + 3h. Review-store `ts` fields are **naive PT** unless stated.
**Bar source** for every price measurement below: **yfinance**, stated per table.
**Method:** 41 agents across two orchestrated passes — 9 parallel source readers, 4
analysis lenses, 24 adversarial skeptics prompted to refute, 1 completeness critic —
plus direct measurement by the lead agent. Verification outcome: **of 24 load-bearing
claims attacked, 0 survived unchanged, 9 were weakened, 15 were refuted.**

---

## 0. The one-line verdict, and three corrections to things I said earlier in this session

**Verdict:** the window cannot promote or demote a single setup family, and the reason is
not that your trading is unclear — it is that **the review store, which every analysis
defaulted to, is the wrong store**, and the stores that carry regime, RVOL, sector, stop
and R were never opened by anyone. The two genuinely actionable findings are operational,
not analytical: a **universe-rebuild defect that blinded the scanner for the whole of
2026-08-21**, and the fact that **a LIKE on the capture rail puts the symbol on the
day-scoped ignore list** — de-facto suppression authority reached by a hotkey, with no
rung and no gate.

Three things I told you earlier in this session were wrong. They are corrected here
because they would have changed your conclusions:

1. **"You dismiss 86.6% of alerts and positive engagement has collapsed to 0.43%."**
   **False, and it was my arithmetic error.** I classified an episode as dismissed if it
   contained *any* queue-clear, checked before engagement. But 572 of 2,222 episodes
   (25.7%) contain **both** a positive action and a queue-clear — that is "arm it, then
   move to the next chart", which is the normal workflow, not a rejection.
   **Corrected: 696 of 2,222 shown-episodes (31.3%) receive a positive trader action.**
   There is no collapsing trend either: per-day engagement runs **25–45%** across the whole
   window with no decline (2026-08-21 = 30.6%). Independently corroborated by
   `review_learning.build_episodes` (25.8%) and `review_preference_state.json` (0.252).
2. **"The regime and RVOL axes are starved (n=130 / n=94)."** True of the review store,
   **false of the system.** `intraday_bounce_outcomes.csv` carries `market_environment`
   on **127,362 of 127,362 in-window rows (100%), across all 21 window sessions**, plus
   `session_rvol`, `sector`, `industry`, `rrs_spy`, `rrs_sector`, `rrs_industry`,
   `internals_tape`. Nobody opened it for analysis.
3. **"AEP was a slam dunk the system did not put in front of you."** Half wrong. The
   system *did* surface it, twice, and **you engaged with it** — and what the system did
   with your engagement is the actual finding. See §7.

---

## 1. Data inventory and coverage

### 1a. Sources read, with coverage

| Source | Path | Rows | Coverage (tz) | Sessions /21 | Gap |
|---|---|---|---|---|---|
| Review store (merged) | `alert_review_events.jsonl` + `alert_review_events/*.jsonl` | **8,818** | 2026-07-28 17:23 → 08-21 12:19 **PT** | **18** | Capture module landed 07-28, so 07-24/07-27 predate it. 5 starved days: 07-28 (1 episode), 08-17 (6), 08-12 (27), 08-06 (33), 08-18 (42) |
| ↳ live shard `1d6d4783…` | same dir | 8,133 | 07-30 → 08-21 PT | 17 | **Zero rows for 2026-08-03** |
| ↳ retired shards `68bdbecd…`, `69d3d53e…` | same dir | 186 + 8 | 08-03 only | 1 | **Sole holders of 2026-08-03.** A reader that opens only the newest shard loses that session |
| ↳ legacy root file | `alert_review_events.jsonl` | 491 (`v1`) | 07-28 → 07-30 PT | 3 | Frozen. Its 08-07 mtime is a store-migration artifact, not a last append |
| Episodes (derived) | `review_learning.build_episodes()` | **2,598** | same | 18 | reject 1,141 / shown_only 661 / take 573 / skip 223 |
| **Intraday bounce outcomes** | `data/runtime/intraday_bounce_outcomes.csv` (202 MB) | **239,422** total, **127,362 in-window** | `logged_at` **tz-aware −07:00**; `entry_time` naive **PT** | **21/21** | Effective n is **4,447 distinct (symbol, day, side)** events / **6,907 `final` rows**, not 127k. Density 162 (08-17) to 11,536 (07-29) |
| Bounce candidates | `intraday_bounce_candidates.csv` (180 MB) | 52,351 | 08-02 → 08-21 PT | **15** | **File rotated 2026-08-02**; 6 window days have zero rows and it is "no file", not "no candidate" |
| Bouncer grades | `logs/bouncers.txt` | 7,544 (6,647 graded) | — | — | **No date field.** 29 bounce types. Cannot answer any time-conditioned question |
| Bounce feedback | `intraday_bounce_feedback.csv` | 4 total, **0 in-window** | last write 2026-05-08 PT | **0** | Axis is dead, not sparse |
| Pick feedback | `pick_feedback.jsonl` | 681 total, **626 in-window** | naive PT | **16** | like 447 / unfavorite 115 / not_today 87 / dislike 32. Schema trap: the field is `verdict`, `action` is `None` on all 681 |
| Capture rail | `trader_annotations.jsonl` | **186** | tz-aware −07:00 | **2** (08-20, 08-21) | veto 134 / like_claim 52 / **note 0**. All three vocab versions inside one 2-day file |
| Journal | `data/runtime/trade_journal.sqlite3` | 187 trades, 17 tables | `opened_at` tz-aware but **mixed**: Questrade −04:00, IBKR −07:00/−08:00 | 15 | `trade_annotations` **0 rows** → R uncomputable for 187/187. `regimes` 0. `trade_date` is *last activity*, not entry |
| Playbook episodes | `output/reports/setup_playbook_episodes.csv` | **127,926** | signal_date 2026-02-17 → 08-17 | **17** | **`stop`, `risk` and `net_r` populated on 127,926/127,926.** Never opened by any lens |
| Run manifests | `diagnostics/run_manifests/` | **90** (83 ok / 7 failed) | filenames UTC | **16** | All `job_type=master_scan`. **No M5/BounceBot manifest exists at all** |
| Job ledger | `diagnostics/job_ledger.jsonl` | 572 | — | — | `job_type` only `swing_scan` / `manual_master_scan` |
| SPY shadow | `diagnostics/shadow_evidence/spy_state_shadow/raw/` | **482** | 27 session dates | — | Carries an `agree` field: **False 174 / True 149** |
| Greatness shadow | `diagnostics/shadow_evidence/greatness_shadow/raw/` | **32,337** | 07-13 → 08-20, +683 live rows 08-21 | 27 | 15 family labels, 1,113 READY events, per-day summaries with `config_hash` and `promotion_decision` |

### 1b. Sources that do not exist

| Expected | Verdict |
|---|---|
| `review_policy.json` | **Never written on this desk, ever.** The only caller of `save_review_policy` in production code is `review_policy.py::main` under `--draft`, which writes the *draft* path |
| `review_policy_draft.json` | Absent. `--draft` has never been run here |
| `market_environment_annotations.jsonl` | Absent — you have never manually overridden the regime selector |

**The review-learning loop has never closed, and as built it cannot.** Even if `--draft`
ran tonight it would emit zero rules: `draft_policy_from_state` derives only from
`blind_spots` and `leaks`, both `[]`. Two segments clear the take-rate gate and fail on
`passed_r_n=3` (< `MIN_CALLOUT_EPISODES=8`); 17 clear the leak take-rate gate and every
one has a **positive** `taken_r_avg`. And `review_capture_audit.policy_gate_check()`
returns **status=healthy, "0 active rule(s), 0 draft"** — **System Health reports a dead
loop as green.**

### 1c. What was never opened — the most important row in this report

The completeness critic's finding, and it is decisive. These carry the evidence four
lenses declared missing:

| Never read | What it holds | The claim it refutes |
|---|---|---|
| `intraday_bounce_outcomes.csv` **content** | `market_environment` on 100% of 127,362 in-window rows, **all 21 sessions**; `session_rvol` on 45,437; sector/industry/RRS/internals on the same rows; bounce type recoverable from `event_id`; 6,907 `final` rows with `close_r`/`mfe_r`/`mae_r`/`stop_hit`/`risk_per_share`/`stop_price` | "Per-regime splits are STARVED at n=130"; "RVOL is STARVED at n=94"; "`bouncers.txt` has no date so no time-conditioned cut is possible" |
| `setup_playbook_episodes.csv` | 127,926 rows with **stop, risk and net_r on every row**, per family, per side, per date | "R is uncomputable anywhere in this system" |
| `master_avwap_setup_scenarios.csv` | **16,350 in-window rows / 713 setup_ids for `avwape_to_1stdev`** | "The flagship family has ZERO scanner rows" — that zero came from searching a pre-canonical label column for a canonical id |
| `shadow_evidence/` tree (92 entries, ~55 MB) | 32,337 greatness rows + 482 SPY-state rows with an explicit `agree` field; 27 daily summaries with one stable `config_hash`, sha256-sealed archives, coverage blocks, `promotion_decision` | That the two SHADOW engines have no promotion evidence. **Gates 1, 3 and the gate-4 champion comparison are already computed and on disk** |
| `human_focus_performance.csv` | 141 rows, n up to **1,799**, the system's own scoreboard of your picks | My "your engaged picks don't beat base rate" claim, which rested on **n=96** |
| `technical_integrity_events.jsonl` (466 MB) | Named the root cause on both worst-latency days (collector not live at the open) | "The artifact needed to tell 'slow sweep' from 'absent desk' does not exist anywhere on disk" |
| `output/reports/move_forensics_patterns.csv` | 521 rows of pre-computed pattern lift **with a `novel` flag** | Task 2's entire mandate. It sat unopened while a lens hand-rolled scans (caveat: 14 days stale) |
| `output/reports/intraday_bounce_performance.txt` | Generated 2026-08-21 13:10 PT, per-combo R with `rec=focus` | "There is no setup-family scoreboard to read" |

**Four of the five most-repeated "starved" claims in this sweep were decisions not to
open a file, not absences of data.** That is the single most useful thing this review
found, and it is a finding about the review, not about your trading.

---

## 2. Methodology warning — read before any number below

### 2a. `%MFE>2%` is a volatility metric in disguise

I ranked families for most of this session by "share of alerts where price moved 2% in
the alert's direction". That metric rewards volatile names regardless of whether the
alert carried information. Two scale-free statistics separate the two:

- **EDGE = (MFE − MAE) / (MFE + MAE)**, −1..+1. Zero = symmetric excursions = no
  directional edge. Cannot be inflated by picking volatile names.
- **2R@0.5%** = share with MAE < 0.5% **and** MFE ≥ 1.0% (a 0.5% stop never touched and a
  1.0% target reached; ordering is implied because MAE is measured over the whole window).

On n=1,875 shown-episodes over 12 sessions: **baseline EDGE −0.031, 2R@0.5% 16.5%,
aggregate MFE/MAE 0.926, mean move-to-close +0.01%.** The alert stream as a whole carries
**no directional edge**.

Re-ranking by EDGE **reverses three headline findings**:

| tag | n | %MFE>2 | **EDGE** | **2R@0.5%** (lift) | agg MFE/MAE |
|---|---|---|---|---|---|
| `focus_review` | 51 | 11.8% | **+0.191** | **31.4% (1.90×)** | 1.674 |
| `chart_watch` | 54 | 18.5% | +0.071 | 20.4% (1.24×) | 1.179 |
| `d1_flag_short` | 251 | 12.0% | +0.033 | 20.3% (1.23×) | 1.010 |
| `d1_flag_long` | 455 | 10.1% | −0.020 | 17.8% (1.08×) | 0.988 |
| `green` | 151 | **26.5%** | **−0.032** | **12.6% (0.76×)** | 1.033 |
| `red` | 87 | 11.5% | −0.048 | 14.9% | 0.804 |
| `focus_d1_event` | 504 | 15.5% | −0.056 | 15.1% | 0.897 |
| `auto_pick` | 307 | 10.4% | **−0.106** | 13.0% (0.79×) | 0.711 |

1. **`green` is not the best family.** It looked like the top family at 26.5% (1.96× lift).
   Scale-free it is below base on EDGE and **second-worst on 2R**. Pure volatility selection.
2. **`M5 regime-pause watch — holding highs` is not the best trigger.** 29.2% %MFE>2
   (n=120) but **EDGE −0.032 and 2R 10.8%, lift 0.66 — the worst measured.** The celebrated
   long/short asymmetry *vanishes* scale-free (2R lift 0.66 long vs 0.80 short — the short
   side is marginally better).
3. **Time of day inverts.** By %MFE>2 the morning dominates (07 PT 17.9% → 12 PT 2.2%).
   By EDGE and 2R the **afternoon** is better: 11 PT 2R = 22.8%, the best bucket. Morning
   alerts precede bigger moves; afternoon alerts precede better-behaved ones.

Relative strength was tested as a ranker and **mostly failed the same test**: signed RS vs
SPY is U-shaped (Q1 17.1%, Q3 3.75%, Q5 24.3%), |RS| spreads %MFE>2 by **4.62×** but
aggregate MFE/MAE by only **1.09×**. It is ~90% volatility selection. Overnight gap behaves
identically. Position-in-session-range is **flat** (corr −0.037) — which refutes the
plausible idea that "already at the extreme" predicts continuation.

### 2b. Section 7 gate 2 is unsatisfiable by everything in this report

Gate 2 requires *"golden/replay fixtures and a declared evidence window frozen before
inspection."* This window was chosen after inspection, the metrics were chosen after seeing
results, and roughly 20 significance tests ran against one base rate in the scoreboard alone
(5 survive Bonferroni at α=0.0025). **No measurement in this review can promote anything.**
The only legitimate outputs are (a) capture and instrumentation changes, (b) evidence windows
declared *before* the next inspection, and (c) the list in §1c of what was never read.

---

## 3. Task 1 — Setup scoreboard

### 3a. The honest headline: zero promotions, zero demotions, one clean demote candidate

All six load-bearing scoreboard claims came back **REFUTED** — the worst rate of any lens.
The ranking rested on a join in which **67.1% of family labels came from scans that ran
*after* the alert** (100% on 2026-07-31 and 2026-08-20, whose only scans were post-close).
Point-in-time, only 34.3% of alerts have a D1 scan predating them, and at n≥30 honest rows
**only the `general` fallback bucket clears the bar**. Saying that plainly is a more honest
Task 1 deliverable than a 15-family ranking.

### 3b. Family vocabulary — four incompatible namespaces, none agreeing

| namespace | source | count |
|---|---|---|
| (A) registry | `setup_docs.all_setup_docs_by_group()` | 24 ids, 4 groups |
| (B) D1 scanner | `d1_features.csv` `setup_family` | 15 values |
| (C) review store | `setup_family` on 34 `surface="setups"` rows | **human-readable labels with spaces** ("AVWAP band bounce") — an id join silently returns zero |
| (D) playbook backfill | `setup_playbook_leaderboard.csv` | 27 families, 19 undocumented |
| (E) veto vocabulary | `vocabularies/veto_reasons_v{1,2,3}.json` | versioned, codes never reused |

Plus three D1 values with no doc entry: `top_pattern_tracking` (3,831 history rows),
`mid_earnings_above_2nd_stdev` (8,352), `previous_avwape_bounce` (1,028). And
`bouncers.txt` carries **29** bounce types where SETUPS_MAJOR documents ~20; `h1_sma_20`,
`vwap_eod_confluence`, `ema8_grind_hod` are named nowhere.

**The `avwape_to_1stdev` "zero rows" claim is withdrawn.** It has **16,350 in-window rows
and 713 setup_ids** in `master_avwap_setup_scenarios.csv`. The zero came from querying the
wrong column. The real finding is narrower and still worth acting on: **no namespace maps
to any other**, so no cross-store family question can be answered without a mapping table.

### 3c. Scoreboard

| | Family / component | Current rung (plan.md §2) | Recommendation | Gate evidence |
|---|---|---|---|---|
| **Demote candidate** | `auto_pick` surface (n=307 episodes) | Live alert surface (effectively PROMOTED) | **Do not demote yet — instrument first.** It is the only thing bad on *every* metric: EDGE −0.106, agg MFE/MAE 0.711, 2R lift 0.79, %MFE>2 lift 0.77 | Gate 4 (champion comparison) partially met. **Gate 2 unmet** (post-hoc window). **Gate 7 unmet.** And the critic found the 307 episodes come from only 2 sessions of review capture, while `autopilot_picks.csv` has **19 days** of the same picks that nobody cross-checked. Re-measure there first |
| **Demote candidate** | `guidance_score` / `take_prob` chart annotations | ADVISORY | **Demote to SHADOW** — stop displaying them until they predict something | corr(gs, MFE) **+0.015**, corr(tp, MFE) **−0.019** at n=1,875; AUC 0.530 and 0.497. EDGE by quintile non-monotone noise. Gates 1,3,4 met in form; **gate 2 unmet**; gate 6 (success criteria) was never declared, which is why nothing catches this. One-switch rollback exists (they are display-only), so **gate 7 is met** |
| **Watch** | `focus_review` / `Focus review — Swing short` | Live surface | **Watch — the only family with a genuinely positive ratio** | EDGE +0.191 (n=51); the short trigger EDGE +0.343, 2R 42.9% (n=**21**). n is far too small to act on. Gates 2, 5 unmet |
| **Watch** | `M5 regime-pause` (`regime_pause_rs` / `regime_pause_rw`) | Live surface | **Watch, and fix gate 7 first** | **`check_regime_pause_setups()` is called unconditionally and neither type is in `BOUNCE_TYPE_DEFAULTS`, so the GUI toggle cannot reach them — gate 7 (one-switch rollback without a code revert) is structurally UNMET for the detector producing the largest cells in the store.** Also: the two outcome stores disagree *in sign* — `intraday_bounce_outcomes` final-row mean `close_r` `regime_pause_rs` **+0.189 (n=334)** vs `bouncers.txt` **−0.031 (n=1,496)**. Reconcile before quoting either |
| **Watch** | The 2026-08-21 ATR-hold + focus-adoption gates | IMPLEMENTED / GREEN | **Watch — essentially unmeasured** | They landed on the *last day* of the window. Only **3 measured episodes** exist (ERO, ZETA, MGA) against 36 pre-gate rows the same day. Any demote of the regime-pause watch would be demoting a defect that has already been repaired but not yet observed |
| **Insufficient data** | All 24 registry families | — | **No verdict possible** | Point-in-time family coverage is 34.3%; at n≥30 only `general` clears. The four families with zero rows anywhere (`sma_breakout`, `post_earnings_candle_break`, `weekly_ema8_hold_retest`, `htf_ema15_rejection`) may be namespace artifacts like `avwape_to_1stdev` was — **verify against `master_avwap_setup_scenarios.csv` before concluding anything** |
| **Insufficient data** | `market_state` / SPY-pullback challenger | **SHADOW** | **Stay SHADOW** | Its evidence exists and nobody read it: 482 rows, 27 sessions, one stable `config_hash` (**gate 1 met**), sha256-sealed archives (**gate 2 partially — sealed, but the window was still not declared before inspection**), coverage blocks (**gate 3 met in form**), and an explicit `agree` field giving **149 agree / 174 disagree over 323 observations — gate 4 already computed**. Gates 5–8 unmet. `promotion_decision: "NONE"` on 27 of 27 sessions |
| **Insufficient data** | `greatness_monitor` challenger | **SHADOW** | **Stay SHADOW** | 32,337 rows, 27 sessions, 15 family labels, 1,113 READY events, same `config_hash` discipline. Gates 1 and 3 met in form; 2 partial; **4 not computed** (no champion comparison against legacy D1 wick alerts exists); 5–8 unmet |
| **Champions** | Legacy SPY pause detection, D1 wick alerts | **PROMOTED** | **Unchanged** | Nothing here argues for moving either. Per plan.md §5 they stay champions until the §7 gates pass |

### 3d. Dismissal regret — the answer is "roughly zero, with a fat tail"

Measured **from the dismissal bar** to session close (n=473, 2026-08-19/20/21): mean
**+0.01%**, median 0.00%, share positive **49.0%**, mean MFE 1.03%, mean MAE 1.04%.
Your dismissals are, on average, free.

A naive version measured from the **session open** gives +0.81% / +2.44% / +1.87% per day
and looks damning. That is an artifact — it captures the move that had already happened
before the alert fired. **Do not use it.**

The tail is where the value is: **19.0% of dismissed alerts moved >1% favorably after you
cleared them, 6.6% moved >2%.** AEP on 2026-08-21 was in that tail: cleared at 10:18:52 ET
at 124.32, then **+2.71% to the close with a maximum adverse excursion of 0.10%**. AEP on
2026-08-19 was cleared correctly (−0.63%).

---

## 4. Task 2 — New setup mining

Three seams were mined. Two produced candidate archetypes; one produced a finding that is
better than an archetype.

### 4a. What your own prose says, which the parser throws away

`build_episodes` reads `detail['reason_codes']` / `['reason_code']`. All **31** `dislike`
rows write `detail={'origin','reason'}`. **The `dislike_reason` dimension is permanently
empty and 31 rows of the most information-dense text in the store are silently dropped.**
That is a one-line fix, and the content is worth having:

**Rule A — "below yesterday's low disqualifies a LONG"** (stated 5×, emphatically):
*"below previous days low for a long this is horible"* (TENB); *"below yesterdays low.
TERRIBLE"* (CVBF); *"gap down and held below yesterdays low. this can NEVER be a long"* (M).
**Already implemented** as the movers-only filter (trader rule 2026-08-19). Closed.

**Rule B — "an incoming SMA disqualifies the setup"** (9 of 15 noted vetoes, 60%):
*"SMA upcoming"*, *"incoming SMA"*, *"sma upcoming"*, *"at an SMA"*, plus *"how can a long
be breaking below an SMA?"*. **These were filed under three different reason codes**
(`too_extended_from_base` ×5, `support_resistance_cluttered` ×1, `compressed` ×3) —
**while the dedicated v3 code `sma_incoming` has zero rows**. v3 shipped 2026-08-21 and at
least three of those SMA vetoes were filed *after* it shipped. The likely cause is that
`sma_incoming` was given hotkey **`0`**, out of the 1–9 run. The cohort data is being
mis-binned into two other reasons right now.

**Rule C — "compression / static price action disqualifies"** (7+ mentions). The most
precise statement is measurable: *"huge compression look how static it is we have new highs
but look at the closes. barely getting higher"* (IRDM) — a **closes**-based compression
measure, not a highs-based one.

**Rule D — the AVWAPE sequence** (4 mentions): *"needs to make some higher highs, break
1stdev and then retest avwape is the best setup"* (HPE); *"needs to break and reclaim
avwape before its a good long"* (MS); *"probably needs to spend some time above avwape and
then retest it"* (KEYS). You are repeatedly saying the scanner surfaces `avwape_to_1stdev`
**at the wrong stage of the sequence**, not that the family is wrong.

### 4b. Candidate archetype 1 — sector-cohort divergence (best evidenced, cheapest)

This is the finding of the review and it came out of the AEP case. Full spec in §7.5.

### 4c. Candidate archetype 2 — the LIKE-cohort sector thesis

On **2026-08-20** you filed `like_claim` SHORT on **NEE, WEC, CMS and DTE** — four electric
utilities, 4 of 21 short claims that day (19%), against utilities being ~1.8% of the 1,480
universe: a **~10× concentration**. **The next session the whole sector fell −2.78% mean.**
All four worked (NEE −1.82%, WEC −2.31%, CMS −2.60%, DTE −2.35%).

**Your own capture stream contained a sector thesis a day early, and nothing in the system
aggregates `like_claim` rows by sector.**

Spec, in `docs/SETUPS_TEST.md` style:

> **`like_cohort_sector_concentration`** (study family)
> **Ladder:** currently **PLANNED**. Next rung **IMPLEMENTED** (write the aggregator +
> versioned config = gate 1), then **GREEN** (deterministic golden fixture first, per
> plan.md §5), then **SHADOW**. It must not skip to SHADOW.
> **Trigger.** Within one session, ≥3 `like_claim` rows on the same side whose symbols map
> to one sector, against that sector's share of the universe. Emit a *cohort observation*,
> never a symbol alert.
> **Context filter.** Session only; no cross-day accumulation. UNKNOWN sector = excluded,
> never counted as a match.
> **Invalidation.** The cohort expires at the close. It is re-derived, never carried.
> **Measurement plan.** For each cohort, forward-measure every *unclaimed* member of the
> same sector over h1/h3/h5 sessions, side-signed, against SPY and against the sector ETF.
> The question is whether your claims predict the *rest of the cohort*, which is the only
> thing that would justify surfacing it.
> **Zero influence.** Shadow-only JSONL. No detector, score, ranking, routing, watchlist,
> Focus, review-queue or `review_policy.json` effect.
> **Evidence needed.** n=1 cohort exists today (2026-08-20 utilities). This needs **≥15
> cohorts across ≥2 regimes** before it is even discussable, which at the observed rate is
> a quarter, not a week.

### 4d. What is NOT an archetype, and why saying so matters

**Short option premium is where your money came from, and it is a strategy class, not a
chart pattern.** In-window closed trades: equities **−$217.72** (FBIN +0.90, TTD −7.18,
SMH −85.11, STX −23.72, UMAC +10.79, APH −72.26, NXPI +17.46, TRV −50.16, CRM −8.44)
versus four closed DRAM short puts at **+$1,087.72**, all with an empty `auto_tag_summary`.
DRAM itself *rose* 55 → 57.68 over the window, consistent with selling puts rather than a
directional short.

But the claim that "the system has no channel for this" is **refuted**: a dedicated sold-put
engine and a "Theta Plays" tab shipped 2026-04-28 / 2026-06-23. The real finding is
**scope exclusion** (the engine is LONG-watchlist only) plus quote degradation (one snapshot
showed 517/517 `ib_unavailable` — n=1, do not generalize). **No new detector is warranted.
The question is whether the existing Theta engine's scope should include your actual
underlyings**, and that is your call, not a mining result.

### 4e. Withdrawn

The SMA/ATR archetype a lens proposed ("a SHORT within 1 ATR20 above its nearest SMA beats
SPY by +1.14%") was **weakened to the point of withdrawal**: only its *negative* half
survives (a short on a name already below all four SMAs is a materially worse 5-session
hold: −3.61% vs SPY, n=175, EDGE −0.307, negative on 6 of 7 sessions), it is 29% explained
by other factors, and the positive half does not replicate. Reported here so it is not
re-proposed; not recommended for the ladder.

---

## 5. Task 3 — Earliness and funnel quality

### 5a. Latency ledger

| Component | Median | Worst measured | Honest or removable |
|---|---|---|---|
| Completed-bar confirmation (M5) | 2.5 min | 5 min | **HONEST — non-negotiable** (plan.md §5) |
| Detector warm-up (e.g. ATR(14) needs 15 bars) | — | ~75 min from the open | **HONEST** |
| D1 scan cadence, start-to-start | **60.0 min** | 120.2 min | **Removable** |
| D1 scan duration | 14.0 min (p90 24.5) | 41.4 min | Partly removable |
| Bar-close → scan output, worst case | — | **~101 min** (60.0 + 41.4) | Removable |
| First scan start of the session | **10:00 ET** | — | **Removable — and this is the big one** |
| Digest publish | recoverable to the second | — | Measurable on 12 of 21 days; **never actually measured by anyone** |
| **M5 / BounceBot latency** | **UNMEASURABLE** | — | No run manifest, no ledger row, no universe record, no bar-source counter exists |

Two corrections the skeptics forced:
- The claimed duration p90 of 25.1 min does not reproduce; it is **24.5** (25.134 is
  reachable only via `sorted[int(0.9*n)]`).
- **"The 10:30→12:00 ET scan hole costs almost nothing" is REFUTED — its cost is
  UNMEASURED.** The 268 rows used to price it contain **zero** master-AVWAP swing-scan
  rows (85.4% are BounceBot M5 rows) and 77.2% were delivered *during* the gap.
- The M5-latency artifact **does exist**: `technical_integrity_events.jsonl` named the
  cause on both worst days (collector not live at the open, 07:20/07:30 PT vs the usual
  06:45–06:55 PT). It was declared absent because nobody opened a 466 MB file.

**The structural finding: the first D1 scan of the session starts at 10:00 ET and takes
7.5–24 minutes. Nothing from the D1 stack can speak about the opening 30–45 minutes at
all.** For AEP — whose entire setup was established in the first 5-minute bar — the D1
scan could not physically have contributed.

### 5b. Funnel quality — corrected

The dilution story is real but **much smaller than I first said**. Engagement is **31.3%**,
not 0.43%, and it is stable at 25–45% per day. What *is* true:

- `d1_flag_long` + `auto_pick` are **762 of 1,875 episodes (41% of volume)** and both sit
  **below** the base rate on every metric, volatility-adjusted or not.
- `tier` is blank on **94.3%** of shown rows; `banger` is false on **8,818 of 8,818**;
  `bounce_types` blank on 94.3%; rich context present on **5.7%**. A conviction signal
  cannot be built out of fields that are empty.
- The context fields *are* being written — just to a different store (§1c).

**Every tightening below is a presentation change and must inherit the movers-only contract
(CLAUDE.md, 2026-08-19): hides and counts, one click reveals for the session, never
deletes, never mutes, never writes `review_policy.json`, never feeds the review-learning
stream, UNKNOWN always shows.** None of them is a suppression rule, and none changes queue
ordering, which stays **FIFO** (confirmed: `queue_ordering == annotation_only` on
2,743/2,743 rows).

| Candidate | Attention freed | Tail lost | Verdict |
|---|---|---|---|
| Hide-and-count `d1_flag_long` below a context threshold | ~30 min/day | **18.1% of the tail** | Too expensive as stated. Needs the context fields first |
| Stop displaying `guidance_score` / `take_prob` | Removes two numbers that carry no information (AUC 0.530 / 0.497) | none | **Recommended** — see §8 |
| Fill the context fields on D1/M5 rows | none (adds signal) | none | **Recommended — this is the enabler for everything else** |

---

## 6. Task 4 — The AEP case, 2026-08-21 SHORT

### 6a. The tape and the ideal trade

Bar source: yfinance M5/D1, times ET.

Prior session 2026-08-20: H 127.54 / L 125.67 / C 125.70. AEP opened **126.15** — *inside*
the prior range — and **126.17, the day's high, was the opening print**. The first
5-minute bar (09:30–09:35) collapsed to **123.69**, through the prior-day low 125.67, on
372k shares. Retrace high 124.52 at 10:05. Rolled over from 10:20. Low **120.52** at 15:25,
close **120.94**. **O2C −4.13%, range 4.48%, volume 5.25M vs ADV 3.86M (≈1.36×).**

SPY the same session: open 766.05, close 765.65, **O2C −0.05%, range 0.48% — flat.**

**Point-in-time entry**, decided on the close of the 10:20–10:25 bar (123.910 — below the
prior bar's low 124.010 and below session VWAP under both the harness's typical-price VWAP
and the repo's own `chart_snapshot.session_vwap_series`):

| | value |
|---|---|
| fill (10:25 bar open, executable) | **123.95** |
| stop (6-bar swing high, 09:55–10:20) | **124.52** |
| risk | **0.46% of price** |
| MFE / MAE | **2.77% / 0.00%** |
| move to close | **+2.43%** |
| **R to close / R max** | **+5.28 / +6.02** |
| stopped out | **no** |

**What made it a slam dunk, structurally:** the day's high was the opening print; the
prior-day low broke in bar 1; the retracement failed below the open; and SPY was flat, so
the weakness was entirely idiosyncratic — or so it appeared. See §6d.

**A correction the skeptics forced:** the retracement did **not** simply fail below VWAP.
Measured with the repo's own `chart_snapshot.session_vwap_series` (which uses
`(O+H+L+C)/4`, not the `(H+L+C)/3` I used), closes ran **above** VWAP on **four**
consecutive bars — 10:00 (124.245 vs 124.224), 10:05 (124.470 vs 124.229), 10:10 (124.280
vs 124.233), 10:15 (124.265 vs 124.232) — and failed at 10:20 (123.910 vs 124.230). The
repo VWAP at 10:10–10:15 close is **124.2331**, not the 123.85 I quoted. **My VWAP figures
in this session were wrong by ~0.4; the sign of the conclusion is unchanged.**

**M5 Focus adoption gate:** OPEN at 10:35 ET is **robust** (margin −0.738 on the
10:30–10:35 bar, −0.548 on the 10:25–10:30 bar; every bar from 10:20 to at least 11:10 is
OPEN). The 10:17 CLOSED verdict is weaker than I stated and should be quoted with its
margin, not as a clean binary.

**A better entry existed than the one I priced.** The **10:17:27 ET `level_fired`** was
never evaluated by anyone, including me. Any AEP conclusion that says "10:20 was the best
available entry" is a universal negative that is false.

### 6b. What the system saw — and the root cause

**The alerts that fired were yours, not the machine's.** On 2026-08-19 you filed an
`arm_level` at 124.78 (below). On 2026-08-21 at **10:17:27 ET** that level fired
("D1 level break below 124.78: reached 123.69"). At **10:35:28 ET** the derived
`lod_avwap` watch fired ("closed 123.66 below AVWAP 123.78 anchored on the 06:40 LOD
candle"). **The pre-armed level is the only thing in the entire system that spoke about
AEP that day.**

**The machine's detectors were blind, and the cause is a defect.** On 2026-08-20 between
13:31:03 and 13:35:37 PT a universe rebuild **overwrote the 2026-08-19 universe (1,487 all
/ 785 longs / 304 shorts) with roughly 520–590 names.** By the 2026-08-21 open,
`universe_all.txt` held **370 symbols** and AEP was not among them — against **1,486–1,513
on each of the seven prior archived sessions, with AEP at rank 40–41 on every one.** The
six in-session `master_scan` runs that day processed **533 / 442 / 427 / 419 / 417 / 409**
symbols against a 1,088–1,136 band over the 15 prior sessions with manifests.
`d1_features_history.csv` contains **zero AEP rows for all six**; AEP reappears only in run
`2026-08-21-130211` at **16:02:11 ET — after the close**.

**`build_universe` refuses to write only when the screen produces ZERO symbols. There is no
floor.** A rebuild that priced ~25% of the listing overwrote a good universe silently, and
**no `universe_rebuild` event is recorded in `job_ledger.jsonl`.** This is a P0 operational
defect and it is the most actionable finding in this review.

**Two honest limits on that story:**
- The exact bad size is **not recoverable from disk** — 520–590 is a reconstruction from
  `provider.daily_bars.lookup` calibrated on two runs. The "370" figure is from the
  morning file, and the 07:00 scan emitted features for 533 symbols, 523 of them in the
  *good* universe. These do not fully reconcile.
- **The collapse and AEP's absence co-occurred; causation is not established.** AEP is
  absent from every run on **4 of 12 prior sessions (33%) at a fully normal ~1,100-symbol
  universe**, because the scanned set is only ~74% of available names on a normal day.
  I am confident the collapse is real and is a defect. I am **not** confident it is why
  AEP specifically was missed.

### 6c. What you did, and what the system did with it — the actual finding

At **10:37:15.485713 −07:00 (10:37:15 ET)** you filed a `like_claim` on AEP: SHORT,
`claimed_setup_id="avwape_to_1stdev"`, note **"2nd stdev breakout"** (the id and the note
name *different* families — worth reconciling). **91 milliseconds later** the review store
records `remove_today`.

Those are **one capture-rail action**. Per `alert_chart_review.py:386`, a LIKE emits
`removeTodayRequested` exactly as a VETO does. And `capture_rail.py::commit_like` carries
this comment:

> *"The like is a recorded judgement, nothing more. It must not add the symbol to Focus or
> any watchlist: Focus membership changes live alerting, and this rail is analysis-only."*

**So you did not dismiss AEP. You recognised it, at 10:37 ET, on the best day trade of the
week — and the system's response to recognition was to write a research row and take the
chart away.** No trade followed (zero AEP journal rows).

That comment is a *good* invariant — it is plan.md §5 ("research outputs have zero
production influence until separately promoted") working correctly. The problem is not that
the LIKE is analysis-only. **The problem is that it also retires the chart and puts the
symbol on `alert_center_ignored_symbols.txt` for the rest of the day** (AEP is at index 2
of 94 on the 2026-08-21 list). A recorder documented as analysis-only has **de-facto
suppression authority over the alert surface, reached by a hotkey, with no rung and no
gate.** That is a live-authority question, not a UX question.

**Caveat, stated plainly:** the generality of the LIKE → ignore-list mechanism was
confirmed for AEP (n=1) and was never measured across the other 51 like_claims. Verify
before acting.

### 6d. The archetype — and why it is the wrong answer

I built a point-in-time scanner for the AEP shape ("opening-drive failure /
prior-day-low break-and-fail", SHORT) over `universe_all.txt`.

**I found and fixed a look-ahead bug in my own v1**: rule 1 compared the first three bars'
high against the high of the *whole rest of the day*. v2 compares against the session so
far. **Cost of the leak: mean EDGE fell +0.258 → +0.117 and mean move-to-close fell
+0.648% → +0.275%. The v1 numbers overstated the archetype by ~2×; do not quote them.**

**Coverage limit, binding:** yfinance returns 5-minute history for a 1,459-symbol batch
over only **4 sessions (2026-08-18 … 08-21)**. Everything below rests on 4 sessions.

**v2: n=1,074 = 268 signals per session.** Mean EDGE +0.117, raw MFE 1.59% / MAE 1.39%,
move to close mean **+0.275%** (median +0.400%, 60.5% positive), **36.9% stopped out**.
And it does **not** hold session to session — 2026-08-19 (EDGE +0.008, −0.120%) and
2026-08-21 (EDGE +0.146, −0.041%) both have a negative mean move to close.

**The stop-distance discriminator is real**, and survives removing the utility cohort:

| stop distance | n | mean R (inflated by construction) | **raw to_close** | **share positive** | **EDGE** | stopped out |
|---|---|---|---|---|---|---|
| <0.3% | 46 | +2.90 | **+0.647%** | **84.8%** | **+0.366** | 63.0% |
| 0.3–0.5% | 132 | +1.40 | +0.561% | 72.0% | +0.242 | 52.3% |
| 0.5–0.8% | 167 | +0.94 | +0.592% | 69.5% | +0.169 | 47.3% |
| 0.8–1.2% | 175 | +0.50 | +0.486% | 60.6% | +0.113 | 42.3% |
| >1.2% | 534 | +0.03 | **−0.059%** | 51.2% | +0.020 | 27.2% |

Mean R spreads **114×** — mostly arithmetic, discount it. **EDGE spreads 18× and
share-positive 51% → 85% — that is real.** The price is a stop-out rate rising 27% → 63%.

**But the archetype is not the edge. The sector event is.** Splitting 2026-08-21's own
archetype signals:

| 2026-08-21 archetype signals | n | EDGE | raw to_close | mean R |
|---|---|---|---|---|
| **electric utilities / rate-sensitive REITs** | **19** | **+0.854** | **+1.97%** | **+5.93** |
| everything else | 222 | +0.086 | **−0.21%** | −0.02 |

**Remove the utilities and 2026-08-21 is a losing session for the archetype.** The day's
top 12 archetype signals were AEE, NI, TTE, ADC, EVRG, SO, ES, ETR, DUK, ATO, LNT, PPL —
**eleven of twelve utilities or rate-sensitive REITs.** AEP ranked **14th of 241**.

The underlying event: **25 of 26 utilities closed below their open, mean −2.78%, median
−2.74%; XLU itself −2.57%; SPY −0.05%.** A ~50× relative move. AEP at −4.13% was the
*second worst* member (SRE −5.42% was worse).

**AEP is not in `industry_etf_map.json`, and `universe_metadata.csv` has no sector or
industry column at all.** The existing industry-RS machinery had no way to group AEP with
the other 25. That is a capture gap, not a logic gap.

### 6e. The candidate spec — sector-cohort divergence watch

In `docs/SETUPS_TEST.md` style.

> **`sector_cohort_divergence`** (study family — intraday watch, not a symbol detector)
>
> **Ladder position.** Currently **PLANNED**. Proposed next rung **IMPLEMENTED** (write
> the watch + a versioned config = gate 1), then **GREEN** (deterministic golden fixture
> frozen *first*, plan.md §5), then **SHADOW**. It must not skip a rung, and it must never
> reach a detector, score, ranking, routing, alert, watchlist, Focus, the review queue or
> `review_policy.json` while at SHADOW.
>
> **Trigger.** For each of ~20 sector/industry ETFs, on every **completed** M5 bar,
> compute `spread = (ETF move from session open) − (SPY move from session open)`. Fire a
> **cohort observation** when `|spread| ≥ 0.75%` and it has persisted across **≥3
> consecutive completed bars** (the persistence rule exists because 31 of 179 measured
> fires occur on the 09:30 bar and are gap artifacts).
>
> **Context filter.** Session only, no cross-day carry. A member symbol qualifies only if
> it is in `universe_all.txt` with ADV > 300k and price > $5. **UNKNOWN sector excludes
> the symbol — it never counts as a match** (missing data is uncertainty, never
> confirmation).
>
> **Entry timing within a flagged cohort** — reuse the archetype, do not re-derive it:
> first completed M5 bar in 10:00–11:30 ET closing below session VWAP
> (`chart_snapshot.session_vwap_series`, `(O+H+L+C)/4`) and below the prior bar's low,
> for a name whose session-so-far high was set in the first three bars and whose prior-day
> low has already broken. Stop = 6-bar swing high. **Prefer stop distance < 0.5%** — but
> report raw % move and stop-out rate alongside R, never R alone.
>
> **Invalidation.** The cohort expires at the close and is re-derived, never carried. A
> member is invalidated when its stop trades.
>
> **Measured cost — the false-positive load, 23 sessions (2026-07-22 … 08-21), batched
> yfinance over 20 ETFs, ZERO IB traffic** (the M5 Strength Board template):
>
> | threshold | short-side fires/session | long-side | total |
> |---|---|---|---|
> | −0.50% | 10.7 | 11.5 | 22.2 |
> | −0.75% | **7.8** | 8.7 | **16.5** |
> | −1.00% | 5.7 | 6.4 | 12.1 |
> | −1.50% | 3.0 | 3.6 | 6.6 |
>
> At −0.75%, **55% of fires still have the spread ≤ −0.75% at the close** (mean final
> −0.88%). Against a current stream of 150–350 alerts/session, 16.5 cohort observations is
> a rounding error in attention.
>
> **What it would have surfaced on 2026-08-21, and when.** The short-side list at −0.75%
> was **six** ETFs: XLU (first fire **10:25 ET**, final spread **−2.54%**), SMH (09:40,
> −1.47%), OIH (10:15, −0.72%), XLK (10:10, −0.54%), KRE (10:30, −0.45%), XLE (11:05,
> −0.39%). XLU's final spread was 1.7× the next worst. A −0.50% threshold fires on the
> **first completed bar, 09:35 ET** (spread −0.69%) — 42 minutes before the archetype
> entry and 62 before the `level_fired` — at 22.2 fires/session and with the gap-artifact
> problem. **AEP would have arrived not as one symbol among 121, but as one of 26 names in
> a named, ranked, sector-wide breakdown, by 10:25 ET at the latest.**
>
> **Measurement plan.** Shadow-only JSONL: cohort id, ETF, side, first-fire bar (ET), the
> spread series, and every qualifying member with its point-in-time entry, stop, raw %
> move, R, and stop-out flag. Grade at h1/h3/h6/h12 bars and EOD, side-signed, against
> both SPY and the sector ETF (so the member must beat its own sector, not just the market).
> Report EDGE and raw % move alongside R, always.
>
> **Evidence required before this is even discussable.** The frequency table above rests on
> 23 sessions and is solid. **The outcome conditioning rests on n=21 members of ONE sector
> on ONE session.** A defensible evidence window is **≥40 sessions spanning at least one
> bullish, one bearish and one chop regime**, declared **before** inspection (gate 2), with
> the golden fixture frozen first (plan.md §5). At the observed rate that is roughly a
> quarter. Nothing shorter can move it past SHADOW.
>
> **Gates.** Would satisfy on delivery: 1 (versioned config + stable cohort identity), 3
> (coverage accounting, since the ETF set is fixed), 7 (a single config switch, and it
> must be in the defaults dict — see the `regime_pause` gate-7 failure in §3c). Unmet and
> not addressable by building it: 2, 4, 5, 6, 8.

---

## 7. What this review could not establish

- **Any family promotion or demotion.** Gate 2 is unsatisfiable post-hoc (§2b).
- **Per-regime, per-RVOL and per-sector splits** — not because the data is missing, but
  because the file that carries it was never opened (§1c).
- **R for any journal trade** from the journal itself (`trade_annotations` = 0). But
  `setup_playbook_episodes.csv` has stop/risk/net_r on 127,926 rows and was never opened.
- **M5 / BounceBot latency** from run artifacts — none exist. Partially recoverable from
  `technical_integrity_events.jsonl`, which was declared absent without being opened.
- **Whether the universe collapse caused AEP's absence** — co-occurrence only (§6b).
- **Whether the LIKE → ignore-list mechanism is general** — confirmed at n=1.
- **Whether `bouncers.txt` or `intraday_bounce_outcomes.csv` is right** — they disagree in
  sign on their two largest cells and were never reconciled.
- **Four registry families with "zero rows"** (`sma_breakout`, `post_earnings_candle_break`,
  `weekly_ema8_hold_retest`, `htf_ema15_rejection`) may be namespace artifacts, as
  `avwape_to_1stdev` turned out to be. Verify before concluding.

Additionally, three lens findings arrived **truncated mid-sentence** (each was the 13th
finding — an output cap, not an authoring failure): the playbook-leaderboard
`baseline_every5` control, the movers-only filter's selection value, and a causal
restatement of the stop-distance discriminator. Their content is **not** in this report and
must be re-requested rather than paraphrased.

---

## 8. Proposed `plan.md` Section 12 queue items — ranked, for your acceptance

These are **proposals**. `plan.md` was not edited. Ranked by (evidence quality × payoff) ÷ cost.

| # | Proposal | Type | Why now | Cost | Risk |
|---|---|---|---|---|---|
| **P1** | **Put a floor on the universe write, and log a `universe_rebuild` ledger event with pre/post counts.** `build_universe` currently refuses to write only at exactly zero symbols. Snapshot the consumed watchlists under a run-scoped name so an overwrite cannot destroy the evidence | **Operational P0.** Not a detector change | A silent rebuild on 2026-08-20 cut the universe from 1,487 to ~370–590 and blinded the scanner for all of 2026-08-21. No artifact records it | Small | Low. Touches no detector logic |
| **P2** | **Decide what a LIKE should do**, then make it do exactly that. It currently writes a research row, retires the chart, **and** adds the symbol to the day-scoped ignore list. First measure the generality (n=1 today) | Trader decision + behavior change | A recorder documented as analysis-only has de-facto suppression authority via a hotkey, with no rung and no gate | Small once decided | **Medium — this is alert-surface authority.** Needs a golden fixture first (plan.md §5) |
| **P3** | **Rebuild the setup scoreboard from `intraday_bounce_outcomes.csv` + `setup_playbook_episodes.csv`, not the review store.** For each of 4,447 in-window (symbol, day, side) events with a `final` row: bounce type from `event_id`, regime/RVOL/sector from `context_json`, `close_r`/`mfe_r`/`mae_r`/`stop_hit` as outcome. Report cells with n≥30 | Analysis | This is the object Task 1 was asked for — per-family, per-regime, **risk-normalized, 21/21 sessions** — on stores nobody opened | Medium | Low, read-only |
| **P4** | **Fix gate 7 for `regime_pause_rs` / `regime_pause_rw`**: add them to `BOUNCE_TYPE_DEFAULTS` so the GUI toggle can reach them | Invariant repair | The detector producing the largest cells in the store has **no one-switch rollback**, which plan.md §7 gate 7 requires of anything with production authority | Small | Low |
| **P5** | **Inline the `_CONTEXT_FIELDS` block on every review row**, not just the 5m/H1 path, and normalize the `M5`/`5m` spelling at the same seam | Capture | Takes the review store's regime axis from n=130 to n≈2,500 in a month and converts 3,362 D1 + 1,838 M5 rows from unsegmentable to segmentable. **Note: P3 may make this less urgent** — check whether the other store already answers the question | Small (one dict merge per row) | Low |
| **P6** | **Reconcile `bouncers.txt` against `intraday_bounce_outcomes.csv`** before either is quoted again. Also add a date field to `bouncers.txt` | Analysis + capture | They disagree **in sign** on their two largest cells (`regime_pause_rs` +0.189 vs −0.031) | Small | Low |
| **P7** | **Fix the dislike-reason parser** (`build_episodes` reads `detail['reason_codes']`; all 31 rows write `detail['reason']`) and **re-key `sma_incoming` off hotkey `0`** | Capture, one line each | 31 rows of your most information-dense prose are silently discarded, and the v3 `sma_incoming` code has zero rows while its reason was typed as a free-text note 9 times under three other codes | Trivial | Low. Codes are never reused; no migration |
| **P8** | **Make `policy_gate_check` honest**: report *degraded* when episodes exist but no draft file does, naming the reason. Schedule `review_policy.py --draft` as a deterministic overnight slot beside `veto_cohort_grading` | Operational | System Health currently reports a loop that has **never run** as green | Small | Low. Deterministic slot, no model |
| **P9** | **Stop displaying `guidance_score` and `take_prob`** until they predict something (demote ADVISORY → SHADOW) | Presentation | AUC 0.530 and 0.497 at n=1,875; corr +0.015 / −0.019. They occupy conviction real estate and carry no information | Trivial | Low — display-only, so gate 7 is already met |
| **P10** | **Build `sector_cohort_divergence` to SHADOW** (§6e), golden fixture first | New study family | 16.5 observations/session at −0.75%, **zero IB traffic**, and it is the only thing measured here that would have surfaced AEP *with conviction* | Medium | Low **if** it stays shadow-only. **Its outcome evidence is n=21, one sector, one session — it needs ≥40 sessions across 3 regimes before it is discussable** |
| **P11** | **Emit `run_manifest_v1` + `job_ledger_v1` rows from the M5 sweep**, with `symbols_processed` and the same `provider.*` counters | Capture | Intraday latency is unanswerable in principle today, and this hid the 08-21 universe collapse from the intraday side | Medium | Low |
| **P12** | **Stamp an `event_id` on D1, level and chart-watch alerts** (the M5 shape already exists), so outcomes can join | Capture | 2,468 of 2,598 episodes can never join the outcomes CSV. AEP's clean two-day arm→fire→watch→fire chain is permanently unjoinable | Medium | Low |
| **P13** | **Build a family-namespace mapping table** across the five vocabularies | Capture | No cross-store family question is answerable today, and one "zero rows" claim in this very report turned out to be a namespace artifact | Small | Low |

**Deliberately not proposed:** any suppression field on `review_policy.json`; any change to
FIFO queue ordering; any demotion of the legacy SPY pause detector or D1 wick alerts; any
promotion of `market_state` or `greatness_monitor`; any new detector built from the
archetype alone (it is a 268/session firehose whose edge collapses without the sector
condition).

---

## 9. Questions for Aaron

1. **Was the AEP LIKE at 10:37 ET meant to be the end of it?** You recognised the setup and
   filed the claim. Should a LIKE also (a) leave the chart up, (b) offer a one-click hand-off
   to the Focus surface that owns that file, (c) *not* add the symbol to the ignore list, or
   (d) exactly what it does now? This is the single highest-leverage decision in the report,
   and I will not guess it — it is alert-surface authority.
2. **Do you want the sector layer at all?** It is the best-evidenced thing here, but it
   changes what the desk shows you from "symbols" to "cohorts, then symbols". That is a
   product decision, not a measurement.
3. **`avwape_to_1stdev`: is the scanner surfacing it at the wrong stage?** Your prose says
   the entry is *break → time above → retest*, and four dislikes say candidates arrive before
   the retest. Is that a scoring change (which needs a golden fixture first) or is the
   current behavior deliberate?
4. **What is the intended relationship between `bouncers.txt` and
   `intraday_bounce_outcomes.csv`?** Which is authoritative? They disagree in sign.
5. **Is the Theta/sold-put engine's LONG-watchlist-only scope deliberate?** Your window's
   entire positive P&L came from short puts on DRAM, which that engine's scope excludes.
6. **The `sma_incoming` hotkey.** Is `0` acceptable, or should the veto keys be re-laid out?
   Re-keying costs you learned muscle memory; leaving it costs the cohort data.
7. **Naming:** the brief referred to a "Section 7.1 ladder". `plan.md` has no §7.1 — I read
   the ladder as §2's status vocabulary combined with §7's numbered gates 1–8, and wrote
   every recommendation that way. Confirm that is what you meant.
8. **`SOL_PROGRESS.md` is on `main` but not on `phase05-integration-blitz`.** I read it via
   `git show main:SOL_PROGRESS.md`. Is its absence from the working branch intentional?
9. **Do you want P3 (rebuild the scoreboard from the right stores) run now?** It is the one
   thing that would let a *real* setup scoreboard exist, and it is read-only.

---

## 10. Provenance

Working artifacts are in the session scratchpad, not the repo:
`INVENTORY.md`, `INVENTORY_PER_SOURCE.md`, `ORCHESTRATOR_MEASUREMENTS.md`,
`CORRECTION_volatility_confound.md`, `ARCHETYPE_V2_FINAL.md`, `LENSES.md`, `VERDICTS.md`,
`BACKFILL.md`, `CRITIC.md`, plus the datasets `funnel_full.csv` (1,875 episodes joined to
M5 outcomes), `funnel_rs.csv`, `archetype_hits_v2.csv` (1,074 point-in-time signals) and
the scripts `full_funnel.py`, `rs_test.py`, `archetype_v2.py`.

No repository file outside `docs/analysis/` was created or modified by this review.
