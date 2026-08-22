# Evidence audit — R10.0 decision register

**plan.md Phase 0.7 / R10.0.** Read-only reproduction of every alleged defect in
the 2026-08-22 evidence brief, plus the store inventory, the family-namespace
map and the predeclared decisions R10.A–R10.I depend on.

**Status: the program stops here.** Nothing in R10.A onward may start until the
trader accepts this register.

**Produced** 2026-08-22 on branch `phase05-integration-blitz`.
**Baseline** at the time of the sweep: HEAD `b6e1521`, suite 4145 passed / 19
subtests, exit 0.
**Zones.** Desk-local = America/Los_Angeles (PDT, UTC−7). Market =
America/New_York (EDT, UTC−4) = PT + 3h. Review-store `ts` and
`intraday_bounce_outcomes.entry_time` are **naive PT**; `logged_at` is tz-aware
−07:00.
**Method.** Every number below was produced by a read-only script over
`C:\TradingBotData` and the diagnostics tree. No store was written. The 203 MB
outcome CSV was read with `chunksize`/`usecols`; the 960 MB tracker payload was
**never loaded** — S1–S4 are answered from the derived CSVs and the scoring
snapshot, which is the same constraint R10.D will have to live inside.

---

## 0. Correction to the brief's stated state

The brief's §1 was written against HEAD `22154dd` and says R9.4 and R9.5
"remain queued and authorized". **Both had already landed and been pushed** —
`36abb14` (R9.4 `thetalongs.txt`) and `ba931a5` (R9.5 `sector_cohort_divergence`
at SHADOW).

The consequence is for §2.1's ordering, which sequences R9.5 *after* R10.A "using
the evidence-plane conventions R10.A sets". R9.5 shipped before this program was
registered. Its store —
`diagnostics/shadow_evidence/sector_cohort/sector_cohort_shadow.jsonl`, schema
name `sector_cohort_shadow_v1` — is append-only, carries a `config_hash` and
writes a per-run coverage row including on quiet runs. That is consistent with
this program's ground rules but was not derived from them. **Reconciliation R10.A
owes it:** its coverage row is not month-segmented and its `first_fire_at` is
tz-aware UTC with no separate `session_date` field (the session is carried in a
sibling key). Both are cheap to align when the outcome ledger sets the pattern.

---

## 1. Verdict table

Legend: **PROVEN** reproduced as described · **PROVEN\*** reproduced but a stated
number differs · **REFUTED** did not reproduce · **UNKNOWN** not decidable from
available evidence.

| # | Allegation | Verdict | Key measurement |
|---|---|---|---|
| **D1a** | Concurrent GUI instances on 2026-08-20 | **PROVEN** | pid 31848 lived 07:46:01→12:45:09 PT and overlapped **three** other pids: 14688 (1,571 s), 19368 (1,020 s), 31932 (**13,677 s = 3.8 h**). Every other in-window session is sequential restarts only |
| **D1b** | Concurrency explains the duplicate rows | **REFUTED as the main cause** | The concurrent session has the highest duplicate rate (34.8% vs a 5.5% mean elsewhere, 6.3×) but supplies only **184 of 742** duplicate `registered` rows (25%). 0 of 609 duplicated ids were written within 5 s of each other (median gap **1,581 s**, p90 54,006 s) |
| **D1c** | 90 events registered 08-20 with no `final` and not pending | **PROVEN** | 90 exactly; **41** are `regime_pause` |
| **D1d** | 394 duplicate `registered` (345 ids), 300 duplicate `final` | **PROVEN\*** | Window-wide: **742** duplicate `registered` over 609 ids; **430** duplicate `final` over 303 ids. On 08-20 alone: 184 duplicate `registered`, **0** duplicate `final` |
| **D2** | `eod_close = entry_price` fabricates a zero final | **PROVEN** | `bounce_bot_lib/legacy.py:3719`. 1,164 of 6,907 in-window finals (16.9%) have `close_r == 0`; **1,164/1,164** have `eod_close == entry_price`; **0 of 5,743** non-zero finals do. 251 never advanced a bar; **563 are stop-hits recorded as 0R** |
| **D2b** | 28 overwrite a real `12_bar` row | **PROVEN\*** | **202** zero finals had a `12_bar` row already, not 28 |
| **D3** | Pending backlog never finalizes | **PROVEN** | 576 pending; **94** older than 08-18; **17** from June (oldest 2026-06-22) |
| **D4** | 2026-08-21: 408 events, 0 finals | **PROVEN\*, and materially different** | 409 `registered`, 399 `1_bar`, 398 `3_bar`, 397 `6_bar`, **394 `12_bar`**, **0 `final`**. Milestones ran all day; only EOD finalization is missing. This is D3's gap, **not** an IB outage that stopped tracking |
| **D5a** | LRSI synthetic flat bar ⇒ no outcome rows | **PROVEN** | `legacy.py:6689-6692` sets `open=high=low=close=event.close`; `_register_bounce_outcome` returns at `:3634` when `risk_per_share == ""`. **0** outcome rows for `lrsi_cross_20` and `lrsi_cross_50` in-window |
| **D5b** | ORB first-candle has the same defect | **REFUTED** | ORB re-break **does** register: `orb_breakout` 28 registered / 25 final; `orb_breakdown` 20 / 18. The code at `:6869-6935` already distinguishes candidate (annotation) from re-break (entry claim) from recross (information) and only the break registers — working as designed |
| **D6a** | H1 `entry_time` is the bar START | **PROVEN, decisively** | **6,439 of 6,439** H1 registered rows have `entry_time` minute == 30 (100%). Non-H1 rows spread across minutes (55, 40, 35, 0). An H1 bar in PT starts at :30 |
| **D6b** | 81% of tracked rows are the three retired H1 engines | **PROVEN** | 6,439 of 7,863 registered rows = **82%** |
| **D6c** | Median logged−entry lag 90 min | **REFUTED as a measurement** | H1 median lag is **502 min**, non-H1 **425 min** — `logged_at` is the write time (finalization), not the signal time, so this statistic measures the wrong thing on both. The bar-start defect is proven by the minute distribution instead |
| **D7** | Penny-stop R artifacts | **PROVEN** | All-time max \|close_r\| = **799.0**; **1,127** finals with risk < 0.1% of entry; `regime_pause_rw` all-time n=934, mean **−1.82**, trimmed-10% **−0.28**, median 0.00 |
| **D8a** | Tier barely reaches the evidence stores | **PROVEN, worse than stated** | In the **outcome** store `tier` is absent from `context_json` on **0 of 7,863** registered rows. In the review store **314 of 8,818 = 3.6%** carry a tier (A 10, B 175, C 47, D 82) |
| **D8b** | `banger=True` on 0 rows | **PROVEN** | 0 of 8,818. `proven=True` on 4 |
| **D8c** | Tier × outcome is inverse (A −0.12R, D +0.23R) | **UNKNOWN — not reproducible from these stores** | Tier does not exist in the outcome store at all, so no tier×R join is possible there. Whatever produced that pair came from a different join and must be re-derived before it is quoted |
| **S1** | Tracker replays all scenarios against current histories | **UNKNOWN** | No historical tracker payload survives — only the current 960 MB file and one `.bak`. The 35 CLOSED→OPEN / 14 OPEN→UNTRADEABLE claim cannot be reproduced without the 08-20 payload, and **that is itself the finding**: there is no point-in-time record to check against, which is exactly what R10.D exists to create |
| **S2** | 2,739 setups carry a mark dated later than the run's `data_session` | **UNKNOWN** | Same cause — not decidable from the current payload alone |
| **S3a** | `horizon_sessions` is not sessions | **PROVEN, worse than stated** | 4,474 of 9,967 rows (45%) span >2× their declared horizon. Median business-day span: horizon 1 → 1; horizon 3 → 5; horizon 5 → **65**; horizon 10 → **73** |
| **S3b** | SPY-relative columns null on all rows | **PROVEN** | `spy_forward_return_pct` and `spy_relative_side_return_pct` are **0.0% non-null on all 9,967 rows** |
| **S4a** | 1,274 rep-scenario rows with \|R\|>5, median risk 0.62% of price | **UNKNOWN / partly REFUTED** | **6,201 of 225,522** scenario rows have \|R\|>5. The scenarios CSV has **no risk column**, so "median initial risk 0.62% of price" is not reproducible from this store, and there is no representative-row flag to isolate the 1,274 |
| **S4b** | 70% of closes TARGET_HIT yet mean R negative | **PROVEN\*** | TARGET_HIT is **37%** of rows (83,763), not 70%. Mean R across closes **is** negative (−0.070). STOPPED 83,806 ≈ TARGET_HIT 83,763 — near 1:1, so the stops cost more than the targets pay |
| **F1** | 7 weekend snapshot dates, 549 rows | **PROVEN** | 549 rows across exactly 7 weekend dates (07-11, 07-18, 08-01, 08-02, 08-08, 08-09, 08-15) |
| **F2** | M5 rows/session 0–154 | **PROVEN\*** | True M5 list (prefix `focus_m5`): **4–154**, median 12, over 26 sessions |
| **F3** | `_pick_key` ignores category | **PROVEN by code; invisible in data by construction** | `human_focus_tracking.py:171` returns `(trade_date, symbol, side)` with no category, and `:290` / `:468` build `{_pick_key(row): row}` **dict comprehensions** — the later row silently wins. The CSV therefore shows **0** multi-source keys: the collision is destroyed before the row is written, so its absence in the output is the *signature*, not a refutation |
| **F4** | `focus_auto_picks.json` overwritten per date | **PROVEN** | The file does not exist at all today; no historical owner is recoverable from any store |
| **F5** | M5 names survive the day roll | **PROVEN, far worse than stated** | The four named symbols do appear on the true M5 list across sessions (WDAY 4, NDAQ 3, DECK 3, VXX 2) — but **244 of 499 (symbol,side) pairs = 49%** appear on ≥2 distinct sessions; DOCN SHORT on 7. **Caveat that must travel with this number:** the picks store is a snapshot, so it cannot distinguish "survived the roll" from "the trader re-added it". That ambiguity *is* F4, and it is why R10.E needs membership episodes rather than snapshots |
| **C1** | `like_claim` rows are never forward-graded | **PROVEN** | 52 `like_claim` rows over 2 sessions; `like_cohort_*` files: **none**; the veto trio exists |
| **C2** | Auto-regime shifts are never written as rows | **PROVEN** | `market_environment_annotations.jsonl` **does not exist** |
| **C3** | `journal_import` fails nightly with a blank error | **PROVEN, root cause found** | 20 `failed` rows in the DAS ledger with `error=''` **and** `reason=''`. Cause: `run_nightly_journal_import` returns its explanation in **`messages`** (`journal_runner.py:499-506`), and `ai_jobs/runner.py`'s normal path passed only `reason=` — the diagnostic was produced and dropped at the seam. **Fixed in this packet** (see §6). The underlying cause it was hiding: *"journal database requires trader-present preparation in the GUI; nightly import refused without migrating it"* |

**Counts:** 12 PROVEN · 6 PROVEN\* (reproduced, a number differs) · 3 REFUTED ·
4 UNKNOWN.

---

## 2. The three findings that change R10.A's design

### 2a. The single-instance guard is warranted — and is not the duplicate fix

Concurrency is proven (§1 D1a), so **§2.5's condition is met and the guard is
authorized**. A guard already exists, but only in `scripts/launch_gui_auto.ps1`
(the scheduled-task launcher, `Get-RunningTradingBotDesk` → "already running —
nothing to do"). It is **not** in `launch_gui.py` or `trading_desk.cmd`, so any
desk started by any other path bypasses it entirely. That is why 08-20 produced
seven pids: `gui_crash.log` shows starts at 07:20, 07:30, 07:45, 07:54, 08:36,
08:56, 13:00, 13:15, 13:30 PT. The guard belongs in `launch_gui.py`, where every
path goes through it.

**But it must not be sold as the duplicate fix.** 75% of duplicate rows are on
sessions with no overlap, and no duplicated id was written twice within 5 s.
What actually removes duplicates is the ledger's keyed, idempotent write.

### 2b. D4 is a finalization gap, not an outage

2026-08-21 produced 409 registrations and 394 `12_bar` milestones and **zero**
finals. Tracking ran all day. Only the EOD finalization step did not — the same
mechanism as D3's 576-event backlog. R10.A's idempotent finalization is therefore
the fix for D3 **and** D4, and neither needs an IB-outage story.

### 2c. Tier cannot be conditioned on today

`tier` is in the outcome store's `context_json` **zero** times. The tier×outcome
inversion the brief quotes cannot be reproduced and must not be repeated until a
tier actually reaches the ledger — which is R10.A's registration-event work.

---

## 3. Store inventory

| Store | Size | Rows | Writer | Readers | Schema | Growth |
|---|---|---|---|---|---|---|
| `intraday_bounce_outcomes.csv` | **203 MB** | 239,422 | `bounce_bot_lib/legacy.py` `_append_learning_row` (BounceBot worker) | `setup_scoreboard.py`, Daytrade Tracker panel, warehouse ingest | `schema_version` col: 4 (231,953), 1 (4,487), 3 (2,770), 2 (212) | ~7 k rows/session-week; header widening (`_migrate_csv_header:1369`, `_learning_csv_header:2905`) **rewrites the whole file** |
| `intraday_bounce_outcome_state.json` | 773 KB | 576 pending | same worker, `_save_pending_bounce_outcomes:2890` writes the **whole dict** | same worker on load | untyped | unbounded — never trimmed |
| `intraday_bounces.csv` | 537 KB | 9,315 | BounceBot | panels | — | ~50–380/session |
| `bouncers.txt` | 467 KB | 7,544 | BounceBot | `review_learning`, ad-hoc | **no date field** | 57 accumulated launch blocks, never rotated |
| `master_avwap_setup_tracker.json` | **960 MB** (+939 MB `.bak`) | — | Master AVWAP tracker | tracker, scoring snapshot | — | must never be loaded or deep-copied to compute a diff |
| `master_avwap_setup_scenarios.csv` | 117 MB | 225,522 | tracker | scoreboard, playbooks | no risk column | — |
| `master_avwap_tier_outcomes.csv` | 3.3 MB | 9,967 | tracker | tier analysis | horizons **not** sessions; SPY cols 100% null | 03-17 → 08-20 |
| `human_focus_daily_picks.csv` | 167 KB | 2,926 | `human_focus_tracking.py` | performance rollup | `source` conflates **list and origin** | 4–154 M5 rows/session |
| `human_focus_outcomes.csv` | 427 KB | 2,924 | same | same | h1/h3/h5/h10 | 58.5% unmatured |
| `focus_pick_membership.json` | 23 KB | — | `focus_picks.py` | Focus panel | — | current state only |
| `focus_auto_picks.json` | **absent** | — | Focus writer | desync repair | — | overwritten per date |
| `veto_cohort_{picks,outcomes,performance}.csv` | 9 / 15 / 1.6 KB | — | `ai_jobs/cohorts.py` | AI scopes | — | — |
| `like_cohort_*` | **absent** | — | — | — | — | C1 |
| `market_environment_annotations.jsonl` | **absent** | — | `ui/services/bounce_service.py` (manual only) | — | — | C2 |
| `sector_cohort_shadow.jsonl` (R9.5) | 3 KB | 12 | `sector_cohort_divergence.py` | none yet | `sector_cohort_shadow_v1` | ~12 rows/session; **not month-segmented** |

**A trap that bit this audit and will bite the next reader.** In
`human_focus_daily_picks.csv` the `source` field encodes list **and** origin:
`focus_swing_m5` is a **swing** row whose origin was an M5 alert. A substring
match on `m5` pulls 649 swing rows into an M5 count — it inflated F2 and F5 in
this sweep until caught. **The list must be taken from the `focus_(swing|m5|pick)`
prefix, never a substring.** R10.E should store list and origin as separate
fields so the trap cannot be set again.

---

## 4. Canonical family-namespace map (the review's P13)

Six namespaces, no two of which agree. This table is versioned as
`family_namespace_v1`; `unmapped` rows are listed rather than dropped.

| Namespace | Where | Form | Count |
|---|---|---|---|
| A. Registry | `setup_docs.all_setup_docs_by_group()` | snake_case ids | 24 in 4 groups |
| B. D1 scanner | `d1_features.csv` `setup_family` | snake_case | 15 observed |
| C. Review store | `setup_family` on `surface="setups"` rows | **human labels with spaces** ("AVWAP band bounce") | 5 values / 34 rows |
| D. Playbook | `setup_playbook_leaderboard.csv` | snake_case | 27 (19 undocumented) |
| E. Intraday `event_id` tail | `intraday_bounce_outcomes.csv` | snake_case, multi-type joined with `-` | 29 in `bouncers.txt` |
| F. Veto vocabulary | `vocabularies/veto_reasons_v{1,2,3}.json` | versioned codes | 10 in v3 |

**Unmapped / conflicting, carried forward for R10.A's `evidence_rules.py`:**
`top_pattern_tracking`, `mid_earnings_above_2nd_stdev`, `previous_avwape_bounce`
(namespace B, no doc entry); `h1_sma_20`, `vwap_eod_confluence`, `ema8_grind_hod`
(namespace E, documented nowhere); `10_candle` (E) vs `10_candle_low` /
`10_candle_high` (`bouncers.txt`); `avwape_to_1stdev` exists **only** in the
tracker namespace and in trader claims, never as a scanner emission.

---

## 5. Predeclared decisions R10.A–R10.I depend on

These are decided **now**, before the data that would tempt a different choice.

1. **Authority between `bouncers.txt` and the outcome ledger.** The **ledger is
   authoritative**; `bouncers.txt` is a human-readable log with no date field and
   no rotation (57 accumulated launch blocks) and is not a subset of the outcome
   store. It is never joined to and never reconciled against — R10.C reads the
   ledger only.
2. **Exchange-session calendar.** `scripts/market_calendar.py` already exists and
   already fails closed (`SessionCalendarError`); it is the single source for
   `session_date`, holidays and early closes. R10.G's `config/market_calendar.json`
   carries **event** dates (FOMC/CPI/NFP/opex/earnings windows), not session
   identity, and must not become a second calendar.
3. **Intrabar stop/target collision.** Predeclared **conservative: the stop is
   taken first** whenever a single bar's range contains both. Every ambiguous
   interval is **counted and reported** beside the result, never silently
   resolved.
4. **Cost and slippage for simulated exits.** Frozen at **1 tick of slippage per
   side plus $0 commission**, applied identically to every policy, versioned with
   the policy. The absolute level is arbitrary; what matters is that policies are
   compared on identical assumptions and that the assumption is stated.
5. **H1 legacy timestamp classification — by evidence, not assumption.** Rule
   `h1_bar_start_v1`: a row whose family matches `^h1_` **and** whose
   `entry_time` minute is exactly 30 is a **bar-start** stamp. This holds on
   6,439 of 6,439 in-window H1 rows and on no non-H1 population. Such rows are
   tagged, never edited, and are excluded from any entry-timing statistic.
6. **One risk-floor definition**, reconciling the three that exist:
   - *stored raw*: `risk_per_share` and `stop_price` exactly as measured, always;
   - *analytic floor*: a row whose risk is **< 0.1% of entry** (R9.3's floor) is
     flagged `risk_below_floor` and excluded from R statistics — 1,127 all-time
     finals qualify;
   - *ranking clip*: **4R**, the existing `BOUNCE_PERFORMANCE_R_CLIP` /
     `TRACKER_SCORING_R_CLIP`, applied only where a view feeds ranking;
   - the evidence report shows **uncapped, 4R-clipped and trimmed together** and
     never silently substitutes one for another.
   No third clip is introduced.
7. **Growth and retention.** New JSONL authorities are **month-segmented**
   (`…-YYYYMM.jsonl`). Estimated: outcome events ~1,500 rows/session ≈ 0.9 MB/day;
   tracker events ~3,000/run ≈ 1.2 MB/day; focus membership ~200/day ≈ 40 KB/day;
   daily context 1 row/day. Retention: 13 months hot in the home folder, older
   segments cold-pushed by `push_cold_to_das.ps1` (its scope must be **extended**
   to the new directories — not yet done, R10.A owes it) and restore-tested once.

---

## 6. The one code change authorized in R10.0

`ai_jobs/runner.py` recorded a failed job with `reason=""` because the normal
(non-exception) path passed only `reason=` while `run_nightly_journal_import`
returns its explanation in `messages`. `_failure_reason()` now prefers `reason`,
falls back to `messages`, and when a job fails with nothing to say at all records
that fact naming the job — a blank field and "it declined to explain itself" look
identical in a file and are completely different to debug. Successful rows are
untouched.

Three tests; two proved red first (`test_a_failing_job_records_the_messages_it_returned`,
`test_a_failing_job_with_nothing_to_say_still_says_so`), the third
(`test_a_successful_job_is_not_given_a_manufactured_reason`) was green throughout
and guards against over-reach.

**The cause is reported, not fixed**, per the packet: *"journal database requires
trader-present preparation in the GUI; nightly import refused without migrating
it."* The nightly import has been refusing to run since 08-19 because the journal
database wants a trader-present migration in the GUI. **That is a trader action,
not a code change**, and it is the first item in §8.

---

## 7. What this audit could not establish

- **S1 and S2** — no historical tracker payload exists, so "the replay moves
  historical exits" cannot be checked. The absence *is* the finding, and it is
  what R10.D creates.
- **S4's 1,274 representative rows and their 0.62% median risk** — the scenarios
  CSV has no risk column and no representative-row flag.
- **D8c's tier×outcome inversion** — tier is absent from the outcome store.
- **F5's "survived" vs "re-added"** — the picks store is a snapshot and cannot
  distinguish them.
- **D1's duplicate mechanism on the 15 non-overlapping sessions** — proven not to
  be near-simultaneous writes; the remaining hypothesis (re-registration after a
  restart within the same session, since **0 of 609** duplicated ids span more
  than one `trade_date`) is consistent but not established.

---

## 8. Questions for the trader

1. **The journal migration.** `journal_import` has failed every night since 08-19
   for one reason: the journal database wants trader-present preparation in the
   GUI. Running that migration is a few minutes at the desk and it unblocks the
   nightly import, `journal_import_health` in the briefs, and R10.F's cohort
   grading. Do you want to do it before R10.A?
2. **The `.bak` tracker payload.** `master_avwap_setup_tracker.json.bak` is
   939 MB and is the only other copy of tracker state. Is it a deliberate
   rollback point (keep, and R10.D diffs against it) or an accident (delete, and
   R10.D starts its ledger from the current run)?
3. **Duplicate-row policy for the backfill.** The ledger makes new duplicates
   impossible. The existing 742 duplicate `registered` and 430 duplicate `final`
   rows in the CSV stay as they are (history is never rewritten) and get a
   reader-side rule. Confirm that is what you want rather than a one-off
   deduplicated copy.
4. **§2.5 is satisfied — confirm the guard.** Concurrency is proven. The guard
   goes in `launch_gui.py` as a local kernel primitive with an "already running —
   focusing the existing desk" message. Confirm, and note it will *not* remove
   the duplicate rows on its own.
5. **Retention.** 13 months hot then cold-push is my proposal (§5.7). Say if you
   want longer hot.

---

## 9. Provenance

Reproduction scripts are in the session scratchpad, not the repo:
`audit_intraday.py` (D1–D8), `audit_intraday2.py` (duplicate spacing, H1 minute
distribution, tier reach), `audit_lifetimes.py` (pid intervals and true overlap),
`audit_d1_cause.py` (duplicates vs concurrency), `audit_sfc.py` (S3/S4, F1–F5,
C1–C3). Every one is read-only. No store under `C:\TradingBotData` or the
diagnostics tree was written by this audit.
