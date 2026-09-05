"""Typed pyarrow schemas for the frozen first-increment datasets (plan sec 7.1).

This module is the single source of truth for the 13 slice tables; the plan's
Section 7.1 listing is their normative documentation. Nothing here is a design
choice to revisit - columns, grains, and the partition spec are locked.

Conventions carried on every dataset (plan sec 7.1, 9.3):

* enums are plain strings, so widening a vocabulary never rewrites a file;
* timestamps are microsecond UTC with an explicit timezone;
* the point-in-time columns (``event_at``, ``observed_at``, ``computed_at`` on
  derived records, ``capture_mode``, and the ``revision_id`` /
  ``supersedes_revision_id`` chain) appear where the plan says they apply;
* ``schema_version`` and ``run_id`` appear on every dataset.

Partition spec (locked, sec 7.1/8.3): one file per (dataset, timeframe, month);
M1 additionally 8 symbol-hash buckets; D1/W1 and small reference datasets per
(dataset, year). ``DATASETS`` states each dataset's layer, partition scheme, and
the timestamp column the seal protocol summarizes as ``min_ts``/``max_ts``.

The deterministic identity helpers at the bottom are used from Phase 5-6
onward; they live here because identity is schema, not detector behavior. The
warehouse never re-detects anything - it records what the champions reported.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime

import pyarrow as pa

SCHEMA_VERSION = "warehouse_v1"

# Layers from plan sec 4 folded onto the sec 8.2 directory contract: raw wraps
# land in bronze, normalized market facts in silver, and the feature/setup/
# style/gold layers all serialize under gold.
LAYER_BRONZE = "bronze"
LAYER_SILVER = "silver"
LAYER_GOLD = "gold"

_TS = pa.timestamp("us", tz="UTC")

# --- enum vocabularies (strings; widening is free) -------------------------
CAPTURE_MODES = ("LIVE", "DELAYED", "BACKFILL", "RECONSTRUCTED")
# The slice implements the reachable subset of sec 9.4; the full target
# vocabulary is additive.
QUALITY_STATES = (
    "COMPLETE",
    "PARTIAL",
    "MISSING",
    "PROVIDER_FALLBACK",
    "NOT_COLLECTED_BY_POLICY",
    "TIMED_OUT",
    "NO_RESPONSE",
    "HALTED",
    "OUTSIDE_SESSION",
)
SESSION_PHASES = ("PRE", "RTH", "POST")
SIDES = ("LONG", "SHORT")
ANALYSIS_UNITS = ("OPPORTUNITY", "ATTEMPT", "MARKET_EPISODE")
PATH_RESOLUTIONS = ("EXACT", "LOWER_TIMEFRAME", "AMBIGUOUS")
# MATURED is derived (maturity_at <= as_of) and is deliberately absent (sec 14.2).
RESULT_STATES = (
    "NO_TRIGGER",
    "OPEN",
    "STOPPED",
    "TARGETED",
    "EXPIRED",
    "TRUNCATED",
    "CENSORED",
    "AMBIGUOUS_BAR",
)
SCAN_STATUSES = (
    "NOT_ASSIGNED",
    "REQUESTED",
    "NO_RESPONSE",
    "PARTIAL_DATA",
    "TIMED_OUT",
    "EVALUATED_INELIGIBLE",
    "EVALUATED_ELIGIBLE",
)
GAP_RESOLUTIONS = ("BACKFILLED", "PERMANENT", "POLICY")
LEVEL_FAMILIES = ("SESSION", "HORIZONTAL_STORE", "MA_LEVEL", "TRENDLINE", "WATCH_JSON")
ANCHOR_TYPES = ("EARNINGS_CURRENT", "EARNINGS_PREVIOUS")  # slice scope, LD-09
VWAP_ALGORITHMS = ("STANDARD", "DYNAMIC", "EOD")
# H2 joined 2026-09-01 (BD-78): the locked plan cut it for having no consumer
# and the Phase 0.12 B3 LRSI study is one. Additive - nothing was renamed.
DERIVED_TIMEFRAMES = ("M15", "M30", "H1", "H2", "H4", "W1")

# M1 symbol-hash buckets (sec 7.1 partition spec). No M1 dataset exists in the
# slice; the constant lives here so the bucket count is fixed once.
SYMBOL_HASH_BUCKETS = 8


def _pit_source_columns() -> list[pa.Field]:
    """Observation columns for records that come straight from a provider."""
    return [
        pa.field("event_at", _TS),
        pa.field("observed_at", _TS),
        pa.field("capture_mode", pa.string()),
    ]


def _revision_columns() -> list[pa.Field]:
    return [
        pa.field("revision_id", pa.string()),
        pa.field("supersedes_revision_id", pa.string()),
    ]


def _provenance_columns() -> list[pa.Field]:
    return [
        pa.field("schema_version", pa.string()),
        pa.field("run_id", pa.string()),
    ]


def _schema(*groups) -> pa.Schema:
    fields: list[pa.Field] = []
    for group in groups:
        fields.extend(group)
    return pa.schema(fields)


TRADING_SESSION = _schema(
    [
        pa.field("session_id", pa.string()),
        pa.field("exchange_calendar", pa.string()),
        pa.field("session_date", pa.date32()),
        pa.field("rth_open_at", _TS),
        pa.field("rth_close_at", _TS),
        pa.field("eth_open_at", _TS),
        pa.field("eth_close_at", _TS),
        pa.field("is_half_day", pa.bool_()),
        pa.field("expected_m5_bars_rth", pa.int32()),
        pa.field("expected_m1_bars_rth", pa.int32()),
        pa.field("calendar_version", pa.string()),
        pa.field("observed_at", _TS),
    ],
    _provenance_columns(),
)

BAR_M5 = _schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("interval_start", _TS),
        pa.field("interval_end", _TS),
        pa.field("session_id", pa.string()),
        pa.field("session_phase", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("vwap", pa.float64()),
        pa.field("trade_count", pa.int32()),
        pa.field("provider", pa.string()),
        pa.field("is_complete", pa.bool_()),
        pa.field("quality", pa.string()),
        pa.field("source_hash", pa.string()),
    ],
    _pit_source_columns(),
    _revision_columns(),
    _provenance_columns(),
)

BAR_D1 = _schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("session_id", pa.string()),
        pa.field("session_date", pa.date32()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("adjustment_version", pa.string()),
        pa.field("corporate_action_id", pa.string()),
        pa.field("provider", pa.string()),
        pa.field("quality", pa.string()),
        pa.field("is_complete", pa.bool_()),
    ],
    _pit_source_columns(),
    _revision_columns(),
    _provenance_columns(),
)

BAR_DERIVED = _schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("timeframe", pa.string()),
        pa.field("aggregation_contract_id", pa.string()),
        pa.field("interval_start", _TS),
        pa.field("interval_end", _TS),
        pa.field("session_id", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.int64()),
        pa.field("is_stub", pa.bool_()),
        pa.field("stub_duration_min", pa.int32()),
        pa.field("constituent_count", pa.int32()),
        pa.field("constituent_expected", pa.int32()),
        pa.field("is_complete", pa.bool_()),
        pa.field("quality", pa.string()),
        pa.field("event_at", _TS),
        pa.field("computed_at", _TS),
        pa.field("input_capture_mode_worst", pa.string()),
    ],
    _provenance_columns(),
)

UNIVERSE_MEMBERSHIP_DAILY = _schema(
    [
        pa.field("session_date", pa.date32()),
        pa.field("list_name", pa.string()),
        pa.field("symbol", pa.string()),
        pa.field("rank_in_list", pa.int32()),
        pa.field("inclusion_reason", pa.string()),
        # First-capture time (== observed_at); never backfilled (LD-05).
        pa.field("snapshot_at", _TS),
    ],
    _provenance_columns(),
)

ANCHOR_INSTANCE = _schema(
    [
        pa.field("anchor_instance_id", pa.string()),
        pa.field("symbol", pa.string()),
        pa.field("anchor_type", pa.string()),
        pa.field("anchor_bar_date", pa.date32()),
        pa.field("catalyst_event_id", pa.string()),
        pa.field("price_basis", pa.string()),
        pa.field("anchor_bar_included", pa.bool_()),
        # The frozen running-deviation sigma variant; never swapped (sec 2).
        pa.field("formula_version", pa.string()),
        pa.field("source", pa.string()),
        # Bitemporal: market validity + knowledge interval (sec 9.5).
        pa.field("valid_from", _TS),
        pa.field("valid_to", _TS),
        pa.field("system_from", _TS),
        pa.field("system_to", _TS),
    ],
    _provenance_columns(),
)

LEVEL_STATE_DAILY = _schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("session_date", pa.date32()),
        pa.field("level_id", pa.string()),
        pa.field("level_family", pa.string()),
        pa.field("level_price", pa.float64()),
        pa.field("zone_low", pa.float64()),
        pa.field("zone_high", pa.float64()),
        pa.field("source_timeframe", pa.string()),
        pa.field("source_store", pa.string()),
        pa.field("strength_score", pa.float64()),
        pa.field("touch_count", pa.int32()),
        pa.field("is_active", pa.bool_()),
        pa.field("definition_version", pa.string()),
        # When the level became knowable (== observed_at for ingested geometry).
        pa.field("known_at", _TS),
    ],
    _provenance_columns(),
)

FEATURE_SNAPSHOT_DAILY = _schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("session_date", pa.date32()),
        pa.field("feature_set_version", pa.string()),
        pa.field("close", pa.float64()),
        pa.field("atr14", pa.float64()),
        # Champion AVWAP bands, parity-tested to 1e-9 against
        # calc_anchored_vwap_bands - never recomputed by a second formula.
        pa.field("avwape_value", pa.float64()),
        pa.field("avwape_upper_1", pa.float64()),
        pa.field("avwape_upper_2", pa.float64()),
        pa.field("avwape_upper_3", pa.float64()),
        pa.field("avwape_lower_1", pa.float64()),
        pa.field("avwape_lower_2", pa.float64()),
        pa.field("avwape_lower_3", pa.float64()),
        pa.field("favorite_zone_coord", pa.float64()),
        pa.field("favorite_zone_residence_bars", pa.int32()),
        pa.field("first_dev_touch_order", pa.int32()),
        pa.field("band1_rejection_strength", pa.float64()),
        pa.field("second_band_streak", pa.int32()),
        # ADDITIVE (Q2.1/BD-99): was the anchor those bands come from observed
        # before this session or reconstructed after it? "" means the row used
        # no anchor; NULL means the row was written before the column existed
        # and is read as `legacy`, never as observed.
        pa.field("anchor_knowledge", pa.string()),
        pa.field("ema8", pa.float64()),
        pa.field("ema15", pa.float64()),
        pa.field("ema21", pa.float64()),
        pa.field("sma50", pa.float64()),
        pa.field("sma100", pa.float64()),
        pa.field("sma200", pa.float64()),
        pa.field("dist_sma50_atr", pa.float64()),
        pa.field("dist_sma100_atr", pa.float64()),
        pa.field("dist_sma200_atr", pa.float64()),
        pa.field("spy_regime_state", pa.string()),
        pa.field("input_manifest_hash", pa.string()),
        pa.field("computed_at", _TS),
        pa.field("event_at", _TS),
        pa.field("input_capture_mode_worst", pa.string()),
    ],
    _provenance_columns(),
)

FEATURE_SNAPSHOT_INTRADAY = _schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("interval_start", _TS),
        pa.field("session_id", pa.string()),
        pa.field("session_phase", pa.string()),
        pa.field("feature_set_version", pa.string()),
        pa.field("session_vwap", pa.float64()),
        pa.field("session_vwap_upper_1", pa.float64()),
        pa.field("session_vwap_lower_1", pa.float64()),
        pa.field("vwap_algorithm", pa.string()),
        pa.field("ema8_m5", pa.float64()),
        pa.field("ema15_m5", pa.float64()),
        pa.field("ema21_m5", pa.float64()),
        pa.field("ema8_m15", pa.float64()),
        pa.field("ema15_m15", pa.float64()),
        pa.field("ema21_m15", pa.float64()),
        pa.field("ema8_m30", pa.float64()),
        pa.field("ema15_m30", pa.float64()),
        pa.field("ema21_m30", pa.float64()),
        pa.field("rvol_tc2000", pa.float64()),
        # Production >=1.0 gate; sub-1.0 rows are retained as the denominator.
        pa.field("rvol_gate_pass", pa.bool_()),
        pa.field("rs_rw_vs_spy", pa.float64()),
        pa.field("group_rs_debiased", pa.float64()),
        pa.field("market_internals_negative", pa.bool_()),
        pa.field("session_structure_gate", pa.string()),
        pa.field("pullback_count_in_current_leg", pa.int32()),
        pa.field("dist_pdh_atr", pa.float64()),
        pa.field("dist_pdl_atr", pa.float64()),
        pa.field("computed_at", _TS),
        pa.field("observed_at", _TS),
        pa.field("capture_mode", pa.string()),
    ],
    _provenance_columns(),
)

SETUP_OCCURRENCE = _schema(
    [
        pa.field("occurrence_id", pa.string()),
        pa.field("symbol", pa.string()),
        # Verbatim from setup_tagging.py; display labels live in Appendix C only.
        pa.field("canonical_setup_id", pa.string()),
        pa.field("side", pa.string()),
        pa.field("structural_timeframe", pa.string()),
        pa.field("trigger_timeframe", pa.string()),
        pa.field("anchor_instance_id", pa.string()),
        pa.field("dependency_cluster_id", pa.string()),
        # Detector lifecycle state as reported; never re-detected here.
        pa.field("status", pa.string()),
        pa.field("trigger_at", _TS),
        pa.field("trigger_bar_interval_start", _TS),
        pa.field("entry_price_ref", pa.float64()),
        pa.field("stop_price_ref", pa.float64()),
        pa.field("detector_version", pa.string()),
        pa.field("first_detected_run_id", pa.string()),
        pa.field("last_updated_run_id", pa.string()),
        # Free text: the "banger" attachment point (LD-27). No inferred meaning.
        pa.field("tags", pa.string()),
        pa.field("event_at", _TS),
        pa.field("observed_at", _TS),
        pa.field("computed_at", _TS),
    ],
    _revision_columns(),
    _provenance_columns(),
)

SETUP_MARKET_CONTEXT = _schema(
    [
        pa.field("occurrence_id", pa.string()),
        pa.field("symbol", pa.string()),
        pa.field("entry_at", _TS),
        pa.field("timeframe", pa.string()),
        pa.field("bias_definition_id", pa.string()),
        pa.field("env_key", pa.string()),
        pa.field("source", pa.string()),
        pa.field("last_close", pa.float64()),
        pa.field("reference_close", pa.float64()),
        pa.field("vwap", pa.float64()),
        pa.field("stdev", pa.float64()),
        pa.field("above_band_frac", pa.float64()),
        pa.field("below_band_frac", pa.float64()),
        pa.field("bar_count", pa.int32()),
        pa.field("computed_at", _TS),
        pa.field("input_capture_mode_worst", pa.string()),
    ],
    _provenance_columns(),
)

OUTCOME_PATH = _schema(
    [
        pa.field("occurrence_id", pa.string()),
        pa.field("recipe_id", pa.string()),
        pa.field("outcome_definition_id", pa.string()),
        pa.field("analysis_unit", pa.string()),
        pa.field("entry_at", _TS),
        pa.field("entry_price", pa.float64()),
        pa.field("stop_price", pa.float64()),
        pa.field("stop_distance", pa.float64()),
        pa.field("r_at_15m", pa.float64()),
        pa.field("r_at_30m", pa.float64()),
        pa.field("r_at_60m", pa.float64()),
        pa.field("r_at_120m", pa.float64()),
        pa.field("r_at_eod", pa.float64()),
        pa.field("r_at_s1", pa.float64()),
        pa.field("r_at_s2", pa.float64()),
        pa.field("r_at_s3", pa.float64()),
        pa.field("r_at_s5", pa.float64()),
        pa.field("r_at_s10", pa.float64()),
        pa.field("r_at_s18", pa.float64()),
        pa.field("mfe_r", pa.float64()),
        pa.field("mae_r", pa.float64()),
        pa.field("time_to_mfe_min", pa.int32()),
        pa.field("first_hit", pa.string()),
        pa.field("first_hit_at", _TS),
        pa.field("path_resolution", pa.string()),
        pa.field("r_lower_bound", pa.float64()),
        pa.field("r_upper_bound", pa.float64()),
        pa.field("gross_r", pa.float64()),
        pa.field("net_r", pa.float64()),
        pa.field("cost_model_id", pa.string()),
        pa.field("result_state", pa.string()),
        # ADDITIVE (Q2.2): which walk produced this row - `managed`,
        # `plain_target` or `plain_no_target`. NULL on every row written before
        # the column and on the recipes that do not walk a swing path; readers
        # call those `unlabelled` rather than guessing.
        pa.field("path_kind", pa.string()),
        pa.field("maturity_at", _TS),
        pa.field("censor_reason", pa.string()),
        pa.field("computed_at", _TS),
        pa.field("input_capture_mode_worst", pa.string()),
    ],
    _provenance_columns(),
)

SCAN_COVERAGE = _schema(
    [
        pa.field("risk_set_id", pa.string()),
        pa.field("scheduled_at", _TS),
        pa.field("run_kind", pa.string()),
        pa.field("symbol", pa.string()),
        pa.field("scan_status", pa.string()),
        pa.field("provider", pa.string()),
        pa.field("bar_source", pa.string()),
        # Compact JSON map {canonical_setup_id: status} (sec 13, LD-21).
        pa.field("family_status_map", pa.string()),
        pa.field("observed_at", _TS),
    ],
    _provenance_columns(),
)

COLLECTION_GAP = _schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("timeframe", pa.string()),
        pa.field("gap_start", _TS),
        pa.field("gap_end", _TS),
        pa.field("expected_bars", pa.int32()),
        # Policy absence is NOT_COLLECTED_BY_POLICY, never MISSING (sec 5.4).
        pa.field("reason", pa.string()),
        pa.field("detected_at", _TS),
        pa.field("resolved_at", _TS),
        pa.field("resolution", pa.string()),
    ],
    _provenance_columns(),
)


# --- bronze wraps (plan sec 19.0/19.5) -------------------------------------
# Legacy artifacts are wrapped, never rewritten and never re-owned: one row per
# source record, payload preserved verbatim, hashed, with the source file's own
# hash on the manifest line. Every wrapped artifact gets its own dataset name
# (``bronze_<artifact>``) so the locked "one file per (dataset, month)"
# partition rule applies unchanged. Bronze raw is never a compaction input.
BRONZE_PREFIX = "bronze_"

BRONZE_RECORD = _schema(
    [
        pa.field("source_artifact", pa.string()),
        pa.field("source_path", pa.string()),
        # SHA-256 of the whole source file as read (freeze-copy evidence).
        pa.field("source_sha256", pa.string()),
        # Line/row index for logs, 0 for whole-file snapshots.
        pa.field("source_offset", pa.int64()),
        # Idempotency key: content hash of this record within its source.
        pa.field("record_hash", pa.string()),
        pa.field("legacy_id", pa.string()),
        pa.field("payload", pa.string()),
        pa.field("payload_format", pa.string()),
        pa.field("quality", pa.string()),
        pa.field("event_at", _TS),
        pa.field("observed_at", _TS),
        # event_at when the record carries one, else observed_at; never null,
        # because a partition key that cannot be computed is a quarantine.
        pa.field("partition_ts", _TS),
        pa.field("capture_mode", pa.string()),
    ],
    _provenance_columns(),
)

BRONZE_FORMAT_JSONL = "JSONL"
BRONZE_FORMAT_JSON = "JSON"
BRONZE_FORMAT_CSV_ROW = "CSV_ROW"


@dataclass(frozen=True)
class DatasetSpec:
    """Everything the store needs to place and summarize one dataset's files.

    ``partition_by`` names the partition dimensions in path order. Supported
    dimensions are the locked ones: ``year``/``month`` derived from
    ``time_column``, plus ``timeframe`` and ``symbol_bucket`` column values.
    """

    name: str
    layer: str
    schema: pa.Schema
    time_column: str
    partition_by: tuple[str, ...]
    grain: tuple[str, ...]
    # Bronze raw and evidence-frozen files are never compaction inputs (sec 8.3).
    compactable: bool = True


def _spec(name, layer, schema, time_column, partition_by, grain, compactable=True) -> DatasetSpec:
    return DatasetSpec(
        name=name,
        layer=layer,
        schema=schema,
        time_column=time_column,
        partition_by=tuple(partition_by),
        grain=tuple(grain),
        compactable=compactable,
    )


DATASETS: dict[str, DatasetSpec] = {
    spec.name: spec
    for spec in (
        _spec("trading_session", LAYER_SILVER, TRADING_SESSION, "session_date", ("year",), ("session_id",)),
        _spec(
            "bar_m5",
            LAYER_SILVER,
            BAR_M5,
            "interval_start",
            ("month",),
            ("symbol", "interval_start", "provider", "revision_id"),
        ),
        _spec(
            "bar_d1",
            LAYER_SILVER,
            BAR_D1,
            "session_date",
            ("year",),
            ("symbol", "session_id", "provider", "revision_id"),
        ),
        _spec(
            "bar_derived",
            LAYER_SILVER,
            BAR_DERIVED,
            "interval_start",
            ("timeframe", "month"),
            ("symbol", "timeframe", "interval_start", "aggregation_contract_id"),
        ),
        _spec(
            "universe_membership_daily",
            LAYER_SILVER,
            UNIVERSE_MEMBERSHIP_DAILY,
            "session_date",
            ("year",),
            ("session_date", "list_name", "symbol"),
        ),
        _spec(
            "anchor_instance",
            LAYER_SILVER,
            ANCHOR_INSTANCE,
            "anchor_bar_date",
            ("year",),
            ("anchor_instance_id", "system_from"),
        ),
        _spec(
            "level_state_daily",
            LAYER_SILVER,
            LEVEL_STATE_DAILY,
            "session_date",
            ("year",),
            ("symbol", "level_id", "session_date"),
        ),
        _spec(
            "feature_snapshot_daily",
            LAYER_GOLD,
            FEATURE_SNAPSHOT_DAILY,
            "session_date",
            ("year",),
            ("symbol", "session_date", "feature_set_version"),
        ),
        _spec(
            "feature_snapshot_intraday",
            LAYER_GOLD,
            FEATURE_SNAPSHOT_INTRADAY,
            "interval_start",
            ("month",),
            ("symbol", "interval_start", "feature_set_version"),
        ),
        _spec(
            "setup_occurrence",
            LAYER_GOLD,
            SETUP_OCCURRENCE,
            "event_at",
            ("year",),
            ("occurrence_id", "revision_id"),
        ),
        _spec(
            "setup_market_context",
            LAYER_GOLD,
            SETUP_MARKET_CONTEXT,
            "entry_at",
            ("year",),
            ("occurrence_id", "timeframe", "bias_definition_id"),
        ),
        _spec(
            "outcome_path",
            LAYER_GOLD,
            OUTCOME_PATH,
            "computed_at",
            ("year",),
            ("occurrence_id", "recipe_id", "outcome_definition_id"),
        ),
        _spec(
            "scan_coverage",
            LAYER_SILVER,
            SCAN_COVERAGE,
            "scheduled_at",
            ("month",),
            ("risk_set_id", "symbol"),
        ),
        _spec(
            "collection_gap",
            LAYER_SILVER,
            COLLECTION_GAP,
            "gap_start",
            ("month",),
            ("symbol", "timeframe", "gap_start"),
        ),
    )
}


_BRONZE_SPECS: dict[str, DatasetSpec] = {}


def bronze_dataset_name(artifact: str) -> str:
    return artifact if str(artifact).startswith(BRONZE_PREFIX) else f"{BRONZE_PREFIX}{artifact}"


def bronze_dataset_spec(name: str) -> DatasetSpec:
    """One wrapped legacy artifact, shaped by the shared bronze record schema."""
    name = bronze_dataset_name(name)
    spec = _BRONZE_SPECS.get(name)
    if spec is None:
        spec = _spec(
            name,
            LAYER_BRONZE,
            BRONZE_RECORD,
            "partition_ts",
            ("month",),
            ("source_path", "source_offset", "record_hash"),
            compactable=False,  # bronze raw is never a compaction input (sec 8.3)
        )
        _BRONZE_SPECS[name] = spec
    return spec


def dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError:
        pass
    if str(name).startswith(BRONZE_PREFIX):
        return bronze_dataset_spec(name)
    raise KeyError(
        f"Unknown research dataset {name!r}. The slice datasets are frozen "
        f"(plan sec 7.1): {', '.join(sorted(DATASETS))}; wrapped legacy "
        f"artifacts use the {BRONZE_PREFIX}* namespace."
    )


def symbol_bucket(symbol: str, buckets: int = SYMBOL_HASH_BUCKETS) -> int:
    """Stable symbol-hash bucket (M1 partitioning; hash() is salted per run)."""
    digest = hashlib.sha256(str(symbol).strip().upper().encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % buckets


# --- deterministic identities (sec 7.1, 7.3) -------------------------------
def _identity_token(value) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value).strip()


def _identity_hash(*parts) -> str:
    payload = "|".join(_identity_token(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def anchor_instance_id(symbol: str, anchor_type: str, anchor_bar_date, formula_version: str) -> str:
    """hash(symbol, anchor_type, anchor_bar_date, formula_version) - sec 7.1."""
    return _identity_hash(str(symbol).upper(), str(anchor_type).upper(), anchor_bar_date, formula_version)


def level_id(symbol: str, source_store: str, level_family: str, subtype: str, price, extra=None) -> str:
    """Stable level identity for stores that publish no ID of their own.

    A level's identity is its geometry and provenance, never its as-of role:
    support/resistance is an episode (sec 6.4), so it is deliberately not an
    input here. Manual and model-originated levels keep separate identities
    even when they cluster at the same price, because ``source_store`` differs.
    """
    rounded = "" if price is None else f"{float(price):.6f}"
    return _identity_hash(
        str(symbol).upper(),
        str(source_store),
        str(level_family).upper(),
        str(subtype),
        rounded,
        extra,
    )


def occurrence_id(
    symbol: str,
    canonical_setup_id: str,
    side: str,
    structural_timeframe: str,
    anchor_instance_id_or_episode_start,
) -> str:
    """The rescan-stable occurrence key (sec 7.1, identity rules in sec 7.3).

    Rescans of the same thesis recompute the same key, so Phase 6 updates the
    row instead of appending a second episode - the tracker episode-dedup
    lesson. Long/short and swing/intraday theses differ in the key inputs, so
    they can never collapse into one identity.
    """
    return _identity_hash(
        str(symbol).upper(),
        str(canonical_setup_id),
        str(side).upper(),
        str(structural_timeframe).upper(),
        anchor_instance_id_or_episode_start,
    )
