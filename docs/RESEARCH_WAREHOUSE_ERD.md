# Research warehouse ERD — first-increment datasets

Phase 1 deliverable of [`docs/ULTIMATE_SETUP_DATABASE_PLAN.md`](ULTIMATE_SETUP_DATABASE_PLAN.md)
(the locked plan; Section 7.3 requires "a small ERD with primary/foreign keys,
cardinalities, deterministic ID algorithms, occurrence start/end/dedup rules,
corrections, and supersession behavior"). The typed pyarrow definitions in
`scripts/research_warehouse/schemas.py` are the source of truth; this document
is their map. Nothing here authorizes a detector, score, ranking, or alert
change — the warehouse is shadow-only additive evidence.

## Entity map

```text
trading_session (session_id)
  ├─1:N─ bar_m5            (symbol, interval_start, provider, revision)
  ├─1:N─ bar_d1            (symbol, session_id, provider, revision)
  ├─1:N─ bar_derived       (symbol, timeframe, interval_start, aggregation_contract_id)
  ├─1:N─ feature_snapshot_intraday (symbol, interval_start, feature_set_version)
  └─1:N─ scan_coverage     (risk_set_id, symbol)

symbol (natural key until the first real rename adds symbol_alias)
  ├─1:N─ universe_membership_daily (session_date, list_name, symbol)
  ├─1:N─ anchor_instance   (anchor_instance_id)   ─┐
  ├─1:N─ level_state_daily (symbol, level_id, session_date)
  └─1:N─ feature_snapshot_daily (symbol, session_date, feature_set_version)
                                                   │
setup_occurrence (occurrence_id) ──────────────────┘ anchor_instance_id (nullable FK)
  └─1:N─ outcome_path (occurrence_id, recipe_id, outcome_definition_id)

collection_gap (symbol, timeframe, gap_start)   — absence is a first-class fact

manifest_log.jsonl      — the read authority for every dataset above
imported_bundles.jsonl  — mirror shape for the deferred mini-PC bundle channel
```

Cardinalities that matter for evidence counting:

| Relationship | Cardinality | Why it is stated |
|---|---|---|
| `setup_occurrence` → `outcome_path` | 1:N | Alternative recipes/horizons are correlated diagnostics of ONE episode; they are never summed as independent samples. |
| `setup_occurrence` → `dependency_cluster_id` | N:1 | The episode unit for evidence floors: simultaneous EMA/AVWAP/level variants on one underlying move share one cluster. |
| rescan of a live thesis → `setup_occurrence` | N:1 | Rescans update the same row (deterministic key below); they never append. |
| `bar_m5` → `bar_derived` | N:1 | Derived bars record `constituent_count` / `constituent_expected`, so incompleteness is visible rather than implied. |
| `anchor_instance` revisions | 1:N over system time | Bitemporal; a correction supersedes, never overwrites. |

## Deterministic ID algorithms

Both are `sha256(<parts joined by "|">)` truncated to 32 hex characters, with
values normalized (symbol upper-cased, dates/timestamps ISO-8601). They are
implemented once, in `schemas.py`, and never re-derived by a caller.

| ID | Inputs |
|---|---|
| `anchor_instance_id` | symbol, anchor_type, anchor_bar_date, formula_version |
| `occurrence_id` | symbol, canonical_setup_id, side, structural_timeframe, anchor_instance_id **or** episode-window start |

Consequences of the occurrence key, which are the identity rules of Section 7.3:

- long and short theses on the same symbol hash differently (`side` is an input);
- swing and intraday theses hash differently (`structural_timeframe` is an input);
- two anchors on one symbol hash differently (`anchor_instance_id` is an input);
- an hourly rescan of the same thesis hashes identically → the row is updated.

## Occurrence start, end, dedup, corrections, supersession

- **Start.** An occurrence exists the moment a detector reports it. The
  warehouse never re-detects: `status`, `trigger_at`, and the reported geometry
  are recorded as the detector stated them, with `first_detected_run_id` set on
  the first sighting.
- **End.** Lifecycle end is whatever the detector's terminal `status` says, plus
  the outcome's `maturity_at` on the `outcome_path` side. `MATURED` is a derived
  predicate (`maturity_at <= as_of`), never a stored state.
- **Dedup.** The deterministic key above is the whole dedup rule. Repeated scans
  update `last_updated_run_id` and the mutable snapshot columns; they do not add
  rows, do not add attempts, and do not add episodes (risk R9).
- **Corrections.** Every dataset is append-only. A corrected record is a new row
  carrying `revision_id` with `supersedes_revision_id` pointing at the row it
  replaces; the superseding row's `observed_at` is its knowledge time. Reference
  datasets that are genuinely revisable (`anchor_instance` here, plus the
  forward-declared `corporate_action`, `catalyst_event`, level/trendline
  definitions, `instrument_master`, `collection_universe_membership`) additionally
  carry bitemporal `valid_from`/`valid_to` + `system_from`/`system_to`.
  `universe_membership_daily` is a plain append-only daily snapshot (LD-05) and
  carries neither.
- **Supersession at the file level** is the manifest's job, not a column: a
  compaction appends ONE `COMPACT` line that registers the replacement file and
  names the part files it retires. That line is the atomic switch. Superseded
  files stay on disk under `_retired/<yyyymmdd>/` for 30 days, which is the
  rollback window; rolling back is re-pointing the manifest, never rewriting a
  file.

## Partitioning and file identity

Locked (Section 7.1/8.3): one file per (dataset, timeframe, month); M1
additionally 8 symbol-hash buckets; D1/W1 and small reference datasets per
(dataset, year). Paths are
`<lake>/<layer>/<dataset>/<partition>/part-<uuid>.parquet`, where layer maps the
Section 4 information layers onto the Section 8.2 directory contract: raw wraps
→ `bronze/`, normalized market facts (sessions, bars, universe, anchors, levels,
coverage, gaps) → `silver/`, and the feature/setup/style/gold layers →
`gold/`.

| Dataset | Layer | Partition | Time column |
|---|---|---|---|
| `trading_session` | silver | year | `session_date` |
| `bar_m5` | silver | month | `interval_start` |
| `bar_d1` | silver | year | `session_date` |
| `bar_derived` | silver | timeframe, month | `interval_start` |
| `universe_membership_daily` | silver | year | `session_date` |
| `anchor_instance` | silver | year | `anchor_bar_date` |
| `level_state_daily` | silver | year | `session_date` |
| `feature_snapshot_daily` | gold | year | `session_date` |
| `feature_snapshot_intraday` | gold | month | `interval_start` |
| `setup_occurrence` | gold | year | `event_at` |
| `outcome_path` | gold | year | `computed_at` |
| `scan_coverage` | silver | month | `scheduled_at` |
| `collection_gap` | silver | month | `gap_start` |

## Point-in-time columns

`event_at` (market fact), `observed_at` (when this installation received it),
`computed_at` (derived records), `capture_mode`
(`LIVE|DELAYED|BACKFILL|RECONSTRUCTED`), and the `revision_id` /
`supersedes_revision_id` chain. Availability is a per-experiment declaration
(`AS_OBSERVED` or `MARKET`), never a per-row column: `AS_OBSERVED` is mandatory
for coverage, latency, queue-exposure, live-shadow, and all promotion evidence,
and admits only `LIVE`/`DELAYED` rows.

## Read contract

`manifest_log.jsonl` is the read authority. Supported reads resolve their file
list from the ledger at query start and pass that explicit list to the reader
(`ResearchStore.open_dataset` / `read_table`). Globbing the lake directories is
not a supported read path: during a compaction's GC window the tree legitimately
holds both the replacement and its superseded inputs, so a glob double-counts.
