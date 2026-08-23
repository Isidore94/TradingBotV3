# The daily-bar volume cliff — R10.0b, read-only

**plan.md Phase 0.7 / R10.0b.** Produced 2026-08-22 evening on branch
`phase05-integration-blitz`, HEAD `766e582`.

**Nothing was changed.** No fetch, normalize, scanner or level code was touched,
and no parquet was written. This is a measurement and a decision request. The
affected code is **live detector input and outside R10's authorization** — the
trader decides what happens next.

**Why this exists.** The R10.0 §1 tracker diff showed 2,737 scenario outcomes
rewritten between two consecutive runs. The 2026-08-20 run fetched daily bars
from Yahoo and the 2026-08-21 run from IB. AVWAP bands are **volume-weighted**,
so if the two sources disagree about what `volume` *means*, every AVWAP level,
target and stop replayed across the change moves — which is what the 4.34-point
TMO target move looks like from the inside.

---

> ### Amendment 1 — 2026-08-22 night: the second factor is in the request, and C-prime is the chosen option
>
> **The unexplained factor from §6 is `useRTH`.**
> `master_avwap_lib/legacy.py:15245-15256` requests daily bars with
> `whatToShow="TRADES"`, **`useRTH=1`**, `formatDate=1` — confirmed in the source.
> IB therefore returns **regular-session-only volume in round lots** while Yahoo
> returns the **full consolidated session in shares**, so the expected ratio is
> `100 × (consolidated / RTH-only)`: ≥ 100 and symbol-dependent. NVDA at 188× and
> A at 162× fit that; SPY and ACHC at 1.0× are rows Yahoo wrote last.
>
> **TSLA at 56× and AAPL at 81× still do not fit**, because they would require IB
> to report *more* than Yahoo ÷ 100. A third effect is present — a stale Yahoo
> row, a partial IB bar, or a symbol-specific unit — and it is not identified
> here. The conclusion is unchanged and strengthened: **no constant repairs IB
> daily volume into Yahoo's unit.**
>
> **The chosen option is C-prime, not C as this report wrote it.** C's step two
> said "add the missing lot rescale to the `master_avwap_lib` IB path". That
> would bake in the very factor §2 proved is not 100 and make the store
> *consistently* wrong instead of *visibly* wrong. C-prime instead:
> **provenance first → a Yahoo-only durable store for volume → refetch third.**
> IB daily stays as a **price-only** fallback whose rows are tagged by provenance
> and **whose volume is never written into the store**. Yahoo is already the
> post-close source (the 2026-08-21 20:01 run: 1,190 Yahoo / 0 IB), it is
> share-denominated, and it costs zero IB budget.
>
> `bounce_bot_lib`'s ×100 (`legacy.py:630`, `:11376`) is a **different seam** —
> live IB *intraday* volume against a yfinance baseline, measured at ~100× by the
> 2026-07-20 fix. It stays as it is; its factor deserves the same per-symbol check
> once the daily store is clean.

> ### Amendment 2 — 2026-08-22 night: what moved was the LEVELS, and the mechanism is a splice
>
> Fable re-ran the `.bak` vs main comparison at field level over 60,519
> mark-days. This supersedes this report's framing of the damage and the audit's
> S1 entry.
>
> **The marks barely moved.** Of all field differences, 26,087 are
> float32→float64 round-trips. Of the 7,465 "material" diffs, **7,104 (95%) are
> ≤ 1.1 ¢** — 2,923 half-cent sub-penny prints (2,196 in the `.bak`, 488 in main)
> and 4,181 at exactly one cent, i.e. vendor disagreement on extremes. Genuine
> restatement is **361 field-diffs = 136 symbol-dates across 113 symbols, max
> 1.9%**; closes move more than 1.1 ¢ on only **16 symbol-dates**, about 10 of
> them SCCO at exactly ×0.98814 — a dividend adjustment.
>
> **The levels moved, and a splice is why.** The 2026-08-21 07:0x run rewrote
> **1,236 parquet files**, and in **1,179** of them volume steps down at
> **2026-07-29** by a median **×0.0088** (p10 0.0049, p90 0.0187) — IB
> hundreds-of-shares spliced onto Yahoo share-scale history (AAL: 07-24
> 74,218,900 → 07-27 836,047). Post-splice bars therefore weigh about 1/100, so
> **every AVWAP anchored before 07-29 effectively freezes at its 07-28 value.**
>
> On the same 60,519 mark-days, **30,003 (49.6%) carry materially different
> levels** — vwap 29,698, UPPER_2 29,753, stdev 29,985 — across **4,034 setups
> and 980 symbols**, vwap move median 1.03%, p90 5.99%, max 138.8%. **4,025 of
> those 4,034 setups are in the 08-21 IB-refetch bucket.**
>
> **Stops did not move at all.** `current_anchor_entry` levels and
> `stop_reference_level`: **0 of 9,331 changed** — they are stored at scan time
> and never replayed. So the stop stayed fixed while the replayed per-mark target
> moved: of the 410 closed setups whose representative exit changed, mark levels
> moved in **394**, and the remaining 16 all exited on 08-20, which was the
> forming bar during the 08-20 run. JPM LONG (anchor 04-14): 07-01 vwap
> 320.96 → 312.53, UPPER_2 349.04 → 335.47, exit `TIME_STOP` 07-28 @ 357.31 →
> `FINAL_TARGET` 07-02 @ 336.31.
>
> **This is why a uniform rescale is the wrong shape of fix**: a constant factor
> would not move an AVWAP at all — VWAP is a volume-weighted *ratio*, so scaling
> every weight equally leaves it unchanged. A **splice** moves it, because only
> part of the series is rescaled. That is the argument for §4 step 3 forbidding
> IB volume in the store outright rather than converting it.
>
> **Open caveat:** 772 level-moved setups anchor on or after 07-29, inside the IB
> window, which suggests the 08-20 series was itself mixed — a Yahoo window over
> IB history. It does not change the verdict.


> ### Outcome — 2026-08-23: the store is repaired
>
> R10.V step 4's backfill ran against the live store after a full dry run. One
> batched yfinance sweep, `auto_adjust=False`, **zero IB traffic**, prices never
> touched, a verified frozen copy of the whole directory taken first.
>
> | | before | after |
> |---|---|---|
> | rows in `shares` | — | **1,116,982 of 1,117,170 (99.98%)** |
> | files with a >20× step | **1,795** | **53** |
> | median step ratio | 158× | **29×** |
> | files on `daily_bars_schema=v2` | 0 | **1,920 of 1,958** |
>
> AAL, the case this report was written around: 2026-07-24 74,218,900 → 07-27
> **93,953,900**, where it read 836,047 before.
>
> **38 files were deliberately left alone and named**: 9 Yahoo has no data for
> (BK, CPRX, CWAN, EXPI, IAC, LC, NUVL, PRA, VSCO — each confirmed individually),
> 13 where Yahoo's history covered under 90% of the stored rows (a rewrite would
> have changed 2 of EA's 787 rows and *manufactured* a boundary), and 13 the
> rewrite would have made worse.
>
> **The 53 residual cliffs are not unit artifacts.** 19 of them are fully
> rewritten, all-`yahoo` files whose volume genuinely steps: DJT at its
> 2024-01-16 listing, OKLO's 2023-09-14 de-SPAC, POET, FFAI, QXO, SOXS. §5's
> option-B/C framing assumed a cliff implied a unit mix; after a single-source
> rewrite it does not, which is why the packet's exit gate moved to
> **`volume_unit != shares` = 0** with the cliff detector as a secondary signal.

---

## 1. The cliff, measured (a)

1,958 parquet files under `C:\TradingBotData\data\daily_bars`; **1,737
measurable** (221 lack a long enough span). Comparing the median volume of
2026-07-13…07-24 against 2026-07-27…08-20:

| measure | value |
|---|---|
| files with early/late ratio **> 20×** | **1,227 of 1,737 (71%)** |
| median ratio among those | **158×** |
| ratio quantiles, all measurable | p10 **0.8×**, p50 **127×**, p90 **233×** |

**The cliff is per-symbol, not global, and not one event.** First-drop dates
spread across months: 2026-07-27 (473 files), 06-04 (302), 05-26 (175), 06-12
(96), 06-08 (65), 07-13 (40). Whichever run last wrote a symbol's row decides
that row's unit, and the source flips run to run (§4).

## 2. The unit is not a clean factor — this is the finding that constrains the fix

Parquet volume for 2026-08-20 against Yahoo's for the same session:

| symbol | parquet | Yahoo | ratio | reading |
|---|---|---|---|---|
| SPY | 45,479,200 | 45,520,300 | **1.0×** | shares — agrees |
| ACHC | 1,251,900 | 1,251,900 | **1.0×** | shares — agrees |
| TSLA | 544,483 | 30,766,400 | **56.5×** | — |
| AAPL | 503,882 | 40,959,200 | **81.3×** | — |
| A | 17,477 | 2,830,300 | **161.9×** | — |
| NVDA | 491,770 | 92,457,000 | **188.0×** | — |

**A blanket ×100 backfill would not repair this.** If the only difference were
IB's round-lot convention every ratio would be 100×. They run 56×–188×, so a
second factor is present on top of the lot size — most plausibly that IB's
`TRADES` volume counts only prints IB sees, which varies by symbol. Whatever it
is, it is **not a constant**, so rescaling in place would replace a wrong number
with a differently wrong number and destroy the ability to tell afterwards.

## 3. The rescale exists — in the other engine (c)

```
scripts/bounce_bot_lib/legacy.py:630   IB_HISTORICAL_VOLUME_LOT_SIZE = 100
scripts/bounce_bot_lib/legacy.py:11376 float(getattr(bar, "volume", 0.0) or 0.0) * IB_HISTORICAL_VOLUME_LOT_SIZE
```

with a comment recording that this exact bug was already found and fixed once:

> *"IB TRADES volume is in round lots (hundreds of shares); scale up to raw
> shares so it matches the yfinance share-based baseline. Without this the ratio
> deflates ~100x and gates out every bounce alert."*

**`scripts/master_avwap_lib/` — which owns the daily-bar parquet — has no lot
handling at all.** A grep for `LOT_SIZE`, `lot_size` or a `* 100` volume rescale
returns only unrelated percentage arithmetic. So one engine knows the convention
and compensates, and the engine writing the durable store does not.

**And the store keeps no provenance.** The parquet has six columns —
`datetime, open, high, low, close, volume` — and its Arrow metadata is pandas
typing only. `fetch_daily_bars` *does* carry a source on the in-memory frame
(`_set_daily_bar_source`, `DAILY_BAR_SOURCE_CACHE`), and it is **dropped before
the write**. There is therefore no way to ask an existing row which source
produced it, which rules out a provenance-targeted repair of the history.

## 4. What ran on cliffed data (e)

From the run manifests, `provider.daily_bars.success.*`:

| 2026-08-21 run (UTC) | symbols | IB | Yahoo |
|---|---|---|---|
| 14:00:30 | 533 | **1,222** | 22 |
| 14:30:44 | 442 | **422** | 9 |
| 16:00:29 | 427 | **468** | 9 |
| 17:00:29 | 419 | **460** | 9 |
| 18:00:29 | 417 | **459** | 9 |
| 19:00:30 | 409 | **450** | 9 |
| 20:01:22 (post-close) | 1,137 | 0 | 1,190 |

**Every in-session run on 2026-08-21 ran predominantly on IB**, so yes — the
08-21 scan ran on cliffed volume. The post-close run reverted to Yahoo. 2026-08-20
was mixed (14:00 all-Yahoo, 16:00:11 IB 444, the rest Yahoo), which is why the
first-drop dates scatter.

**The M5 Strength Board is NOT affected.** It never reads this parquet: it does
its own batched yfinance download (`strength_board_service.py:209`,
`period=STRENGTH_FETCH_PERIOD`, `interval="5m"`) — the zero-IB template.

**Consumers of the daily parquet** (`MASTER_AVWAP_DAILY_BARS_DIR`), all of which
inherit whatever unit a row carries:

| consumer | what it does with it |
|---|---|
| `master_avwap_lib/legacy.py`, `runner.py`, `data/daily_bars.py` | the D1 scanner and the setup tracker — **AVWAP bands, targets, stops** |
| `chart_snapshot.py:278-296` | the desk's D1 chart payloads and session VWAP |
| `human_focus_tracking.py:351` | `_load_durable_daily_frame` — Focus pick grading |
| `ai_jobs/cohorts.py:89-99` | veto-cohort forward grading |
| `autopilot_core.py:1373` | universe/staleness paths |
| `diagnostics/provider_counters.py` | telemetry only |

Only the price columns matter to some of these; the volume-weighted ones — AVWAP
bands and anything derived from them — are where the damage lands.

> **Amendment 3 — 2026-08-22 night: the consumer table was two short.** R10.V
> step 2 tested every consumer against both schemas and found two this table
> missed: `ui/services/bar_cache.py` (the desk's chart series builder, via
> `chart_snapshot._daily_store_candidates`) and
> `research_warehouse/ingest_existing.py` (`read_durable_daily_bars` →
> `bar_d1`). Both read the store by column NAME and are unaffected by the two
> new columns; both now have a v1/v2 test. The warehouse's
> `provider="UNKNOWN"` docstring is now understated — v2 rows carry a real
> source — but wiring that through is a warehouse change R10.V does not
> authorize, so it is **owed, not done**.

## 5. Fix and backfill options (d) — for the trader to choose

Each names its golden-fixture impact. **No option is recommended over the
trader's judgement; option C is what I would do.**

**A. Rescale in place (×100 on suspected-IB rows).** Cheapest, and **wrong**:
§2 shows the factor is not 100. It would leave every rescaled row wrong by
0.56×–1.88× with no way to tell afterwards which rows were touched. *Fixture
impact:* every AVWAP golden fixture moves, and the new values are not defensible.
**Not viable.**

**B. Refetch the whole daily history from one source.** Correct and simple to
reason about: pick Yahoo (share-denominated, already agrees on SPY and ACHC),
refetch ~1,958 symbols × ~758 sessions, replace the store. *Cost:* one batched
yfinance sweep, zero IB budget. *Fixture impact:* **every AVWAP-derived golden
fixture must be re-frozen**, and per plan.md §5 that is a detector change needing
its fixture frozen first, so it is a packet of its own. *Risk:* it also silently
repairs the 1,304 float32→float64 differences from R10.0 §1 Amendment 2e.

**C. Provenance first, repair second.** *(Superseded by **C-prime** — see
Amendment 1. The trader chose C-prime: the lot rescale below is NOT part of it,
because the factor is not 100.)* Add a `source` column (and unit) to the
parquet write so every future row says where it came from and in what unit; add
the missing lot rescale to the `master_avwap_lib` IB path so new rows are
share-denominated; **then** refetch history under option B with the fixtures
re-frozen in the same packet. This is the only order in which the repair is
verifiable afterwards — without provenance you cannot prove the backfill worked,
and R10's whole premise is that evidence carries its own provenance. *Cost:*
option B plus a small schema addition. *Fixture impact:* same as B, once.

**D. Coexistence (tag and normalize on read).** Keep the mixed store, tag rows by
detected unit, rescale at read time. *Cost:* lowest write risk; highest ongoing
complexity, and it puts a heuristic on the read path of a live detector forever.
**Not recommended.**

## 6. What this report does not establish

- **Why the ratio is 56×–188× rather than 100×.** I did not open the IB adapter's
  bar-request parameters (`whatToShow`, `useRTH`) to confirm the second factor —
  that means reading live fetch code, and §3 says change nothing and hand off.
  The measurement stands on its own; the explanation is owed.
- **How far back the damage goes.** First-drop dates reach 2026-05-26 in this
  sample, but files with an early cliff may have had earlier rows overwritten too.
- **Which tracker outcomes moved *because* of volume** versus the replay defect
  in R10.0 §1. The two co-occur; separating them needs a replay against a
  known-good bar set, which is option B/C's first byproduct.
- **The 221 unmeasurable files** — too short a span to compare. They are not
  cleared, only unmeasured.

## 7. Questions for the trader

1. **Which option — A, B, C or D?** My recommendation is **C**, and it needs its
   own packet with the AVWAP fixtures re-frozen first.
   **ANSWERED 2026-08-22:** **C-prime** — provenance first, a Yahoo-only durable
   store for volume second, refetch third; no rescale anywhere. Built as R10.V.
2. **Is the 08-21 scan output to be treated as suspect?** Every in-session run
   that day used IB volume. That includes whatever the D1 scanner surfaced and
   what the tracker recorded.
   **ANSWERED 2026-08-22:** **Suspect, tagged, never deleted.** Every in-session
   08-21 run and the 08-20 16:00 run are tagged `daily_volume_mixed_v1` in
   `evidence_rules.py`, keyed from the run manifests' `provider.daily_bars.success.*`
   counts; every rollup reports the tagged count beside n.
3. **Should the daily-bar fetch prefer one source until this is settled?** Pinning
   to Yahoo would stop the store getting more mixed while the decision is made;
   it is a one-line change to a live path and therefore an ask, not something I
   would do unprompted.
   **ANSWERED 2026-08-22 — authorized and DONE** (`d031e89`): `local_settings.json`
   key `daily_bars_source`; `"yahoo"` pins, absent or anything else resolves to
   `"auto"` = today's behaviour. Read at `_fetch_live_daily_bars`, independent of
   the failure circuit, announced once per scan, surfaced in System Health, and
   set on the desk. The 482 fixture/AVWAP/tracker tests were run to prove the
   golden fixtures do not move. Intraday is deliberately not pinned.
