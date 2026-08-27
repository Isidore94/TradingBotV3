# AVWAP band challenger — build prompt (Opus)

Paste everything below the line into a fresh Claude Code session in the repo on
the main desk, model set to Opus. Paste the handoff it writes at the end back to
the Fable session for review. Authorized by the trader 2026-08-26 ("make me an
opus prompt and we will throw it into the setup tracker and begin testing it
out"); the roadmap item is `plan.md` Phase 0.10.

---

You are building `plan.md` **Phase 0.10 — AVWAP band challenger** in the
TradingBotV3 repo on my trading desk. Read `CLAUDE.md` first and follow its
mandatory documentation workflow: `CHANGELOG.md`, `plan.md` §5–7 and Phase 0.10,
`CURRENT_CHECKPOINT.md`, then the governing spec
`docs/AVWAP_BAND_VARIANT_STUDY.md` — its §2b holds the replicated formula and the
three OKTA hover readings that are your golden values; §4 T3 is the tracker
design you are building. Do not re-derive the formula: it is settled.

## The formula (replicated 2026-08-26 against OneOption / Option Stalker Pro)

```
centre_t  = anchored VWAP of HLC/3, volume-weighted, from the anchor bar to t
sigma_t   = population standard deviation of the last 20 CLOSES ending at t
            (the window ignores the anchor and reaches back before it)
band_k,t  = centre_t ± k · sigma_t        k = 1, 2, 3
```

Golden values (OKTA, anchor 2026-05-29; from the durable store
`daily_bars/OKTA.parquet`, typical price HLC/3):

| date | centre | +1σ | −1σ | note |
|---|---|---|---|---|
| 2026-05-29 | 118.19 | 128.47 | 107.90 | one-bar anchor: centre is exactly the bar's HLC/3 = 118.187; σ = 10.28 |
| 2026-06-02 | 126.565 (vendor) / 126.78 (our volumes) | 144.60 | 108.53 | σ = 18.035; our 20-bar population stdev of closes is 18.04. The 0.17% centre gap is a volume-feed difference, not a formula difference |

Assert σ to ±0.02 and the centre to ±0.2% in the fixture. The store's OKTA
volumes are **mixed-unit** (thousands before 2026-05-27 and on 2026-06-04, shares
otherwise): the bars in the fixture must be frozen AFTER the champion's own
normalisation path (`_normalize_daily_bar_frame` / the durable loader — see how
`tests/fixtures/mixed_unit_avwap_v1.json` was built by
`scripts/build_mixed_unit_avwap_fixture.py`), never by an ad-hoc threshold.

## Hard rules — read before touching anything

1. **`calc_anchored_vwap_bands` is frozen** (decision 0008, plan.md §5). You do not
   edit it, call it differently, or make anything that consumes its output read
   the challenger instead. The existing guards `tests/test_mixed_unit_avwap_golden.py`
   and `tests/test_warehouse_avwap_parity.py` must pass unchanged.
2. **Shadow only.** Nothing you add may change a detector, score, rank, tier,
   alert, zone arm, Focus list, review queue, or `review_policy.json`. The
   champion's tracker outputs must be byte-identical with your shadow block
   present — you will prove this with a parity test on a frozen fixture.
3. **File-scoped ask-first rule.** `scripts/master_avwap_lib/legacy.py` and
   `runner.py` house detector/scoring code. In this prompt I pre-authorize
   **only** the additive edits listed under B-2 below. Anything else in those
   files — a refactor, a "while I'm here", a changed default — stop and ask me.
4. **Golden fixture first** (plan.md §5): freeze the tracker-record fixture on
   the current code BEFORE the B-2 edit, then prove parity after it.
5. **Fail-before-fix / fail-before-build tests.** Every new test is shown failing
   (or the module absent) before the code that satisfies it. Say so in the
   handoff for each.
6. **Missing data is uncertainty.** Fewer than 20 closes before a bar → σ is
   `None` for that bar and the band is absent — never padded, never 0, never a
   shorter window.
7. **Never break the tree.** The desk launches from this checkout
   (`launch_gui.py`, source, by trader decision). Commit small and green, push
   after each commit. Branch: create `claude/avwap-band-challenger` from the
   CURRENT HEAD of `claude/gui-p1-fluidity` (do not rebase onto `main`; do not
   switch the checkout to `main`).
8. Chat to me in very short, simple lines (CLAUDE.md "How to talk to the trader").
   Depth goes in docs and commit messages.

## Packets, in order

### B-0 — the pure module + fixture

`scripts/indicators/avwap_band_variants.py`, in the `indicators/` shape (read
`scripts/indicators/smi.py`'s docstring and follow it): completed bars in,
aligned immutable series out, `None` where unmeasurable, no I/O, no imports from
`master_avwap_lib`.

- `FEATURE_VERSION = "avwap_bands_oneoption_bb20_v1"`.
- `oneoption_avwap_band_series(bars, anchor_index, *, lookback=20, ddof=0) ->
  dict[str, tuple[float | None, ...]]` with keys `centre`, `sigma`, `upper_1..3`,
  `lower_1..3`, aligned to `bars`; bars before the anchor are `None`.
  Zero/NaN-volume bars are skipped in the centre exactly as the champion skips
  them, but their closes still count in σ (σ is not volume-weighted).
- `oneoption_avwap_bands(bars, anchor_index, **kw) -> (centre, sigma, bands)` —
  the final-bar values in the same `(vwap, stdev, {"UPPER_1": …})` shape as
  `calc_anchored_vwap_bands`, so callers can hold the two side by side.
- Docstring: the formula verbatim, the vendor, the replication date, and the
  tempting wrong forms it is NOT (the champion's running deviation; the
  distribution σ; sample stdev of all OHLC prints — which predicts 138.09 on
  2026-06-02 and was killed by the 144.60 reading).
- `tests/fixtures/avwap_band_variant_oneoption_v1.json`: the OKTA bars from
  2026-04-01 through 2026-06-05 (frozen through the normalisation path above),
  `raw_input_sha256`, the two golden rows, loaded through
  `tests/conftest.py::load_fixture_contract`.
- `tests/test_avwap_band_variants.py`: the golden values; a discriminator test
  that the champion AND the sample-OHLC form both give different answers on the
  same bars; the `None`-before-20-closes rule; zero-volume handling; and an AST
  test that the module never imports `master_avwap_lib`.
- This is the first importer of `scripts/indicators/`: run
  `tests/test_packaging_spec_drift.py` and `launch_gui.py --selftest` and state
  the results. Expect no spec edit (`indicators` is already collected); if the
  drift test says otherwise, fix the spec, never the test.

### B-1 — the fit/print script

`scripts/avwap_band_variant_fit.py SYMBOL ANCHOR_DATE [--lookback 20]`: offline,
reads the durable D1 store through the same loader the playbook study uses
(`setup_playbook_study._load_daily_frame`), prints one row per bar since the
anchor: date, close, champion centre/σ/±1, challenger centre/σ/±1. No network,
no writes outside `OUTPUT_DIR/reports/` (and only with `--csv`). I will use it
to hover-compare new names against OneOption. Test: runs on the fixture bars
and reproduces the two golden rows.

### B-2 — the tracker shadow (T3 in the study doc)

Pre-authorized additive edits, and only these:

1. `runner.py`, beside `current_anchor_meta` / `prev_anchor_meta` (≈ lines
   700–766): compute `current_anchor_variant` and `previous_anchor_variant` as
   `{"formula_version", "vwap", "stdev", "bands"}` from the same frame and
   anchor index via `oneoption_avwap_bands`. `None` σ → the block carries
   `"stdev": None, "bands": {}` and says why in `"reason"`.
2. `legacy.py` `build_tracker_setup_record` (≈ 5469): carry both blocks on the
   setup record under those names. Existing keys untouched.
3. `legacy.py` `_find_tracker_stop_candidates` (≈ 5306): add shadow stop
   candidates from `current_anchor_variant` — label `VARIANT_LOWER_1` for longs,
   `VARIANT_UPPER_1` for shorts, `source_type="band_variant"`, the same
   `close_failure_limit` as the champion's protective-band stop. They are
   appended AFTER every existing candidate so the primary stop and
   `representative_total_r` are unchanged. The existing per-bar scenario
   machinery then grades them with no further edit.
4. Stats export: `master_avwap_band_variant_stats.csv` beside
   `master_avwap_setup_type_stats.csv`, written in the same pass, one row per
   (setup_family, side, priority_bucket): `n`, `avg_total_r_champion`
   (primary-stop scenario), `avg_total_r_variant` (the `VARIANT_*` scenario),
   `stop_out_rate_champion`, `stop_out_rate_variant`, `target_hit_rate_*`,
   `mean_stop_distance_atr_*`, `n_variant_unmeasured` (σ was `None`). Counts
   first; a blank cell where n = 0, never a 0.0.
5. Setup Tracker panel (`scripts/ui/panels/setup_tracker_panel.py`): a "Band
   variant" section that reads that CSV the way the other sections read theirs.
   Pure CSV reader, read off the Qt thread if the panel already does so for its
   other files; otherwise match the panel's existing pattern and say which.

Tests for B-2, all written first:

- **Parity**: freeze `tests/fixtures/tracker_record_band_variant_parity_v1.json`
  from the current code (a small symbol_entry + row → setup record, scenarios,
  `representative_total_r`, `avg_total_r`). After the edit, every pre-existing
  key is byte-identical; the new keys are present; the `VARIANT_*` scenario is
  last in the candidate list.
- A short with `current_anchor_variant` gets `VARIANT_UPPER_1`; a long gets
  `VARIANT_LOWER_1`; a `None` σ produces no variant candidate and counts in
  `n_variant_unmeasured`.
- The stats export on a three-setup synthetic tracker gives the expected
  numbers, and n = 0 cells are blank.
- Panel test in the existing Qt-test style: the section renders the CSV and
  shows the honest empty state when the file is absent.
- Measure and state the tracker JSON growth after the first real save (it is
  ~951 MB today).

### B-3 — D1 chart overlay, default OFF

`scripts/chart_levels.py` builds a new level group `avwap_variant` (±1/2/3 from
the challenger, stable ids) on the `ChartDataService` worker beside the champion
lines; `scripts/ui/widgets/paint_lines_button.py` gets a "AVWAP σ variant"
toggle, machine-local, **default OFF**. Never on the paint path; no zone arm,
alert or detector reads the group. Test: the payload carries the group; the
toggle default is off; nothing else in the levels payload changed (golden on
the existing payload).

### Not in this prompt

The level-quality backfill (study doc T1) and the playbook re-run (T2) are the
next packet, after this one is reviewed. Warehouse columns (T3 item 4) wait for
the same review. Do not start them.

## Verification before each commit

`.venv\Scripts\python.exe -m pytest tests/ -q` fully green — check pytest's own
exit code, not a piped tail — and `scripts\smoke_check.py` 7/7. After B-0 also
`launch_gui.py --selftest`. Record the counts (baseline: 4902 passed).

## Handoff (write this, then paste it back to me)

Reconcile the docs per CLAUDE.md: `CHANGELOG.md` (what landed, with commit
ids), `CURRENT_CHECKPOINT.md` (active item = Phase 0.10, branch, verification
numbers, the JSON growth measurement, anything owed), `plan.md` Phase 0.10 item
statuses, `docs/README.md` if you add a Markdown file, and
`docs/AVWAP_BAND_VARIANT_STUDY.md` §7 status. Then write, in the chat, a handoff
of at most fifteen short lines: commits, test counts, each test you proved
failing first, anything you did not build, and any place you had to ask.
