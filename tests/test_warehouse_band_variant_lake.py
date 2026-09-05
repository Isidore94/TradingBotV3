"""Packet M4: both AVWAP band families in the lake, side by side.

Three claims are pinned here, none of which the un-fixed code can make:

* **M4.1** ``feature_snapshot_daily`` carries the CHALLENGER's bands beside the
  champion's - same bars, same anchor index, through the pure
  ``indicators.avwap_band_variants.oneoption_avwap_bands`` - with its own
  formula version, ``FEATURE_SET_VERSION`` bumped to ``tier1_v2``, old rows
  keeping ``tier1_v1``, old-shape partitions still readable, and every champion
  column byte-identical.
* **M4.2** ``swing_house_variant_v1`` is the twin recipe: identical to
  ``swing_house_v1`` in entry, stop, management, targets and expiry, differing
  ONLY in which band family supplies the levels, and registered in the trial
  ledger before any outcome is inspected.
* **M4.3** ``band-coverage --compare`` prints the two on the SAME occurrence
  ids, with the ONE Wilson (``swing_headline``'s z), and counts an occurrence
  missing under either recipe rather than dropping it.

Every test runs against a tmp :class:`ResearchStore`. The live lake is
read-only to this packet.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from indicators import avwap_band_variants  # noqa: E402
from scripts.research_warehouse import (  # noqa: E402
    cli,
    exchange_calendar as xcal,
    features,
    outcomes,
    schemas,
    trial_ledger,
)
from scripts.research_warehouse.manifest import lake_relative  # noqa: E402
from scripts.research_warehouse.store import ResearchStore, _sha256_file  # noqa: E402
from swing_headline import WILSON_Z, wilson_lower_bound  # noqa: E402

UTC = timezone.utc
NOW = datetime(2026, 9, 5, 23, 0, tzinfo=UTC)


@pytest.fixture()
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


# --- fixtures ---------------------------------------------------------------
def _d1_row(day: date, index: int, symbol: str = "AAPL") -> dict:
    base = 100.0 + index
    return {
        "symbol": symbol,
        "session_id": xcal.session_id_for(day),
        "session_date": day,
        "open": base,
        "high": base + 2.0,
        "low": base - 1.5,
        "close": base + 1.0,
        "volume": 1_000_000 + index * 1000,
        "adjustment_version": None,
        "corporate_action_id": None,
        "provider": "IBKR",
        "quality": "COMPLETE",
        "is_complete": True,
        "event_at": datetime(day.year, day.month, day.day, tzinfo=UTC),
        "observed_at": datetime(day.year, day.month, day.day, 21, tzinfo=UTC),
        "capture_mode": "BACKFILL",
        "revision_id": "",
        "supersedes_revision_id": "",
        "schema_version": schemas.SCHEMA_VERSION,
        "run_id": "d1",
    }


def _history(symbol: str = "AAPL", *, first: date = date(2026, 6, 1), days: int = 55) -> list[dict]:
    rows, day, index = [], first, 0
    while len(rows) < days:
        if xcal.is_trading_day(day):
            rows.append(_d1_row(day, index, symbol=symbol))
            index += 1
        day += timedelta(days=1)
    return rows


def _publish_old_shape(store: ResearchStore, rows, partition: str, *, dataset: str, drop) -> Path:
    """Write a partition the way the code wrote it BEFORE ``drop`` existed."""
    dropped = {drop} if isinstance(drop, str) else set(drop)
    spec = schemas.dataset_spec(dataset)
    old_schema = pa.schema([field for field in spec.schema if field.name not in dropped])
    table = pa.Table.from_pylist(
        [{key: value for key, value in row.items() if key not in dropped} for row in rows],
        schema=old_schema,
    )
    target_dir = store.partition_dir(dataset, partition)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"part-{uuid.uuid4().hex}.parquet"
    pq.write_table(table, path)
    store.manifest.append(
        action="PUBLISH",
        dataset=dataset,
        partition=partition,
        file_path=lake_relative(store.root, path),
        sha256=_sha256_file(path),
        row_count=table.num_rows,
        job_id="pre_m4",
    )
    return path


# --- M4.1 the daily snapshot carries both families --------------------------
#: Read off the CHAMPION's columns at `e7b12ebe`, before any variant column
#: existed, by running `build_daily_snapshots` over the fixture below. M4.1 adds
#: columns and must move none of these.
CHAMPION_GOLDEN = {
    "close": 155.0,
    "atr14": 3.5,
    "avwape_value": 149.88286088613626,
    "avwape_upper_1": 152.5515729608833,
    "avwape_upper_2": 155.22028503563033,
    "avwape_upper_3": 157.88899711037737,
    "avwape_lower_1": 147.2141488113892,
    "avwape_lower_2": 144.5454367366422,
    "avwape_lower_3": 141.87672466189514,
    "favorite_zone_coord": 1.9174564248744468,
    "favorite_zone_residence_bars": 0,
    "first_dev_touch_order": 1,
    "band1_rejection_strength": 0.2857142857142857,
    "second_band_streak": 0,
}

#: The pure formula's answer on the same bars and the same anchor index,
#: computed by `indicators.avwap_band_variants` directly. The lake row must
#: carry these and not a second implementation of them.
VARIANT_GOLDEN = {
    "avwap_variant_value": 150.00786088613626,
    "avwap_variant_stdev": 5.766281297335398,
    "avwap_variant_upper_1": 155.77414218347167,
    "avwap_variant_upper_2": 161.54042348080705,
    "avwap_variant_upper_3": 167.30670477814246,
    "avwap_variant_lower_1": 144.24157958880085,
    "avwap_variant_lower_2": 138.47529829146546,
    "avwap_variant_lower_3": 132.70901699413005,
}


def _snapshot_with_anchor(store: ResearchStore, *, days: int = 55, back: int = 10) -> dict:
    history = _history(days=days)
    store.publish("bar_d1", history)
    anchor_day = history[-back]["session_date"]
    features.build_daily_snapshots(
        store,
        history[-1]["session_date"],
        symbols=["AAPL"],
        anchors_by_symbol={
            "AAPL": features.AnchorChoice(anchor_day, features.ANCHOR_KNOWLEDGE_RECONSTRUCTED)
        },
        now=NOW,
    )
    return store.read_rows("feature_snapshot_daily", "year=2026")[0]


def test_the_daily_snapshot_carries_the_challengers_bands_beside_the_champions(store):
    row = _snapshot_with_anchor(store)
    for column, expected in VARIANT_GOLDEN.items():
        assert row[column] == pytest.approx(expected, abs=1e-9), column
    assert row["avwap_variant_formula_version"] == avwap_band_variants.FEATURE_VERSION
    assert row["anchor_knowledge"] == features.ANCHOR_KNOWLEDGE_RECONSTRUCTED


def test_the_champion_columns_are_byte_identical_beside_the_challenger(store):
    """Golden pin, read off the un-changed code. Adding a family moves nothing."""
    row = _snapshot_with_anchor(store)
    for column, expected in CHAMPION_GOLDEN.items():
        assert row[column] == pytest.approx(expected, abs=1e-12), column


def test_the_feature_set_version_bumps_and_old_rows_keep_theirs(store):
    row = _snapshot_with_anchor(store)
    assert features.FEATURE_SET_VERSION == "tier1_v2"
    assert row["feature_set_version"] == "tier1_v2"


def test_a_tier1_v1_row_is_never_rewritten_by_the_bump(store):
    """The identity carries the version, so the old row stays exactly as written."""
    history = _history()
    store.publish("bar_d1", history)
    session = history[-1]["session_date"]
    old = {
        "symbol": "AAPL",
        "session_date": session,
        "feature_set_version": "tier1_v1",
        "close": 1.0,
        "atr14": 1.0,
        "anchor_knowledge": features.ANCHOR_KNOWLEDGE_OBSERVED,
        "computed_at": NOW - timedelta(days=1),
        "event_at": xcal.trading_session(session).rth_close_at,
        "input_capture_mode_worst": "BACKFILL",
        "input_manifest_hash": "hash",
        "schema_version": schemas.SCHEMA_VERSION,
        "run_id": "old",
    }
    store.publish("feature_snapshot_daily", [old])

    features.build_daily_snapshots(
        store,
        session,
        symbols=["AAPL"],
        anchors_by_symbol={
            "AAPL": features.AnchorChoice(
                history[-10]["session_date"], features.ANCHOR_KNOWLEDGE_RECONSTRUCTED
            )
        },
        now=NOW,
    )
    rows = {row["feature_set_version"]: row for row in store.read_rows("feature_snapshot_daily", "year=2026")}
    assert set(rows) == {"tier1_v1", "tier1_v2"}
    assert rows["tier1_v1"]["close"] == 1.0, "the old row is untouched"
    assert rows["tier1_v1"]["avwap_variant_formula_version"] is None
    assert rows["tier1_v2"]["avwap_variant_formula_version"] == avwap_band_variants.FEATURE_VERSION


def test_a_short_lookback_writes_the_formula_version_with_null_bands(store):
    """A NULL band is 'not measured' - never a band sitting on the centre line."""
    history = _history(days=12)
    store.publish("bar_d1", history)
    features.build_daily_snapshots(
        store,
        history[-1]["session_date"],
        symbols=["AAPL"],
        anchors_by_symbol={
            "AAPL": features.AnchorChoice(
                history[2]["session_date"], features.ANCHOR_KNOWLEDGE_RECONSTRUCTED
            )
        },
        now=NOW,
    )
    row = store.read_rows("feature_snapshot_daily", "year=2026")[0]
    assert row["avwap_variant_formula_version"] == avwap_band_variants.FEATURE_VERSION
    assert row["avwap_variant_stdev"] is None, "fewer than 20 closes: the sigma is unmeasurable"
    for number in (1, 2, 3):
        assert row[f"avwap_variant_upper_{number}"] is None
        assert row[f"avwap_variant_lower_{number}"] is None
    assert row["avwape_value"] is not None, "the champion is unaffected by the challenger's window"


def test_a_partition_written_before_the_variant_columns_still_reads(store):
    """Schema promotion, as Q2 proved it: the old shape reads beside the new."""
    fresh = _snapshot_with_anchor(store)
    legacy = dict(fresh)
    legacy["symbol"] = "OLD"
    _publish_old_shape(
        store,
        [legacy],
        "year=2026",
        dataset="feature_snapshot_daily",
        drop=[
            "avwap_variant_value",
            "avwap_variant_stdev",
            "avwap_variant_upper_1",
            "avwap_variant_upper_2",
            "avwap_variant_upper_3",
            "avwap_variant_lower_1",
            "avwap_variant_lower_2",
            "avwap_variant_lower_3",
            "avwap_variant_formula_version",
        ],
    )
    by_symbol = {row["symbol"]: row for row in store.read_rows("feature_snapshot_daily", "year=2026")}
    assert set(by_symbol) == {"AAPL", "OLD"}, "an old-shape partition still reads"
    assert by_symbol["OLD"]["avwap_variant_formula_version"] is None
    assert by_symbol["OLD"]["avwap_variant_upper_1"] is None
    assert by_symbol["AAPL"]["avwap_variant_upper_1"] == pytest.approx(
        VARIANT_GOLDEN["avwap_variant_upper_1"], abs=1e-9
    )


REBUILD_FIRST = date(2026, 8, 10)
REBUILD_LAST = date(2026, 8, 11)


def test_a_rebuilt_session_row_carries_both_band_families(store):
    """`rebuild-daily-features` (Q2.4) recomputes both families, contract unchanged."""
    history = _history(days=60, first=date(2026, 6, 1))
    store.publish("bar_d1", history)
    anchor_day = [row["session_date"] for row in history if row["session_date"] < REBUILD_FIRST][-8]
    features.build_anchor_instances(
        store, [{"symbol": "AAPL", "anchor_bar_date": anchor_day}], now=NOW
    )
    report = cli.run_rebuild_daily_features(
        store, start=REBUILD_FIRST, end=REBUILD_LAST, apply=True, now=NOW
    )
    assert report["status"] == "OK" and report["applied"] is True

    rows = store.read_rows("feature_snapshot_daily", "year=2026")
    assert {row["session_date"] for row in rows} == {REBUILD_FIRST, REBUILD_LAST}
    for row in rows:
        assert row["avwape_value"] is not None, "the champion family"
        assert row["avwap_variant_upper_1"] is not None, "and the challenger's"
        assert row["avwap_variant_formula_version"] == avwap_band_variants.FEATURE_VERSION
        assert row["anchor_knowledge"] == features.ANCHOR_KNOWLEDGE_RECONSTRUCTED


# --- M4.2 the twin recipe ----------------------------------------------------
TRIGGER_DAY = date(2026, 8, 3)
TRIGGER_AT = xcal.trading_session(TRIGGER_DAY).rth_close_at

CHAMPION_BANDS = {"UPPER_1": 103.0, "UPPER_2": 108.0, "UPPER_3": 115.0}
#: Deliberately WIDER: the same walk over the same bars must reach a different
#: verdict, which is the whole point of the twin.
VARIANT_BANDS = {"UPPER_1": 106.0, "UPPER_2": 122.0, "UPPER_3": 140.0}


def _occurrence(**overrides) -> dict:
    row = {
        "occurrence_id": "occ-1",
        "symbol": "AAPL",
        "canonical_setup_id": "AVWAPE_TO_FIRST_DEV",
        "side": "LONG",
        "structural_timeframe": "D1",
        "status": "TRIGGERED",
        "trigger_at": TRIGGER_AT,
        "entry_price_ref": 100.0,
        "stop_price_ref": 95.0,
        "event_at": TRIGGER_AT,
    }
    row.update(overrides)
    return row


def _swing_bars(closes, *, start: date = TRIGGER_DAY, symbol: str = "AAPL") -> list[dict]:
    rows, day = [], start
    for close in closes:
        while not xcal.is_trading_day(day):
            day += timedelta(days=1)
        rows.append(
            {
                "symbol": symbol,
                "session_date": day,
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "capture_mode": "BACKFILL",
            }
        )
        day += timedelta(days=1)
    return rows


def test_the_twin_differs_from_the_champion_only_in_its_band_family():
    champion = outcomes.SWING_HOUSE_V1
    twin = outcomes.SWING_HOUSE_VARIANT_V1
    assert twin.recipe_id == "swing_house_variant_v1"
    assert twin.band_family == outcomes.BAND_FAMILY_VARIANT
    assert champion.band_family == outcomes.BAND_FAMILY_CHAMPION
    assert twin.analysis_unit == champion.analysis_unit
    differing = {"recipe_id", "band_family", "outcome_definition_id", "note"}
    for field in champion.__dataclass_fields__:
        if field in differing:
            continue
        assert getattr(twin, field) == getattr(champion, field), field
    assert outcomes.outcome_definition_for(twin) != outcomes.outcome_definition_for(champion)
    assert outcomes.outcome_definition_for(champion) == outcomes.OUTCOME_DEFINITION_ID
    assert outcomes.required_band_numbers(twin) == outcomes.required_band_numbers(champion)


def test_the_twin_walks_the_variant_bands_and_the_champion_row_is_unchanged(store):
    bars = _swing_bars([100.0, 104.0, 109.0, 116.0])
    occurrence = _occurrence()
    champion_only = outcomes.build_outcomes(
        store,
        [occurrence],
        d1_by_symbol={"AAPL": bars},
        bands_by_occurrence={"occ-1": CHAMPION_BANDS},
        recipes=(outcomes.SWING_HOUSE_V1,),
        as_of=NOW,
        now=NOW,
    )
    assert champion_only.rows == 1
    baseline = {
        key: value
        for key, value in outcomes.latest_outcomes(store)[
            ("occ-1", "swing_house_v1", outcomes.OUTCOME_DEFINITION_ID)
        ].items()
    }

    twin_report = outcomes.build_outcomes(
        store,
        [occurrence],
        d1_by_symbol={"AAPL": bars},
        bands_by_occurrence={"occ-1": CHAMPION_BANDS},
        variant_bands_by_occurrence={"occ-1": VARIANT_BANDS},
        recipes=(outcomes.SWING_HOUSE_V1, outcomes.SWING_HOUSE_VARIANT_V1),
        as_of=NOW,
        now=NOW,
    )
    assert twin_report.rows == 1, "the champion row is unchanged and only the twin is written"

    latest = outcomes.latest_outcomes(store)
    champion_row = latest[("occ-1", "swing_house_v1", outcomes.OUTCOME_DEFINITION_ID)]
    twin_row = latest[
        ("occ-1", "swing_house_variant_v1", outcomes.outcome_definition_for(outcomes.SWING_HOUSE_VARIANT_V1))
    ]
    assert champion_row == baseline, "the champion's stored row did not move"
    assert twin_row["path_kind"] == outcomes.PATH_KIND_MANAGED, "path_kind is written for the twin too"
    assert twin_row["result_state"] != champion_row["result_state"], (
        "wider bands, same bars: the twin must reach its own verdict"
    )


def test_an_occurrence_with_null_variant_bands_grades_plain_no_target_under_the_twin():
    bars = _swing_bars([100.0, 104.0, 109.0, 116.0])
    twin = outcomes.simulate_swing(
        _occurrence(), bars, outcomes.SWING_HOUSE_VARIANT_V1, bands=None, as_of=NOW, computed_at=NOW
    )
    champion = outcomes.simulate_swing(
        _occurrence(), bars, outcomes.SWING_HOUSE_V1, bands=CHAMPION_BANDS, as_of=NOW, computed_at=NOW
    )
    assert twin["path_kind"] == outcomes.PATH_KIND_PLAIN_NO_TARGET
    assert champion["path_kind"] == outcomes.PATH_KIND_MANAGED
    assert champion["result_state"] == "TARGETED"


def test_the_twin_is_in_the_default_recipe_set_so_the_nightly_picks_it_up(store):
    report = outcomes.build_outcomes(
        store,
        [_occurrence()],
        d1_by_symbol={"AAPL": _swing_bars([100.0, 104.0, 109.0, 116.0])},
        bands_by_occurrence={"occ-1": CHAMPION_BANDS},
        variant_bands_by_occurrence={"occ-1": VARIANT_BANDS},
        as_of=NOW,
        now=NOW,
    )
    assert report.rows >= 4, "three default recipes plus the twin"
    written = {key[1] for key in outcomes.latest_outcomes(store)}
    assert "swing_house_variant_v1" in written


def test_the_twin_recipe_is_registered_in_the_trial_ledger(tmp_path):
    """One append-only row, BEFORE any outcome is inspected; register refuses a rewrite."""
    written = trial_ledger.backfill(tmp_path)
    assert "swing_house_variant_v1_twin" in written

    rows = {row["trial_id"]: row for row in trial_ledger.load(tmp_path)}
    row = rows["swing_house_variant_v1_twin"]
    assert "swing_house_variant_v1" in tuple(row["recipe_ids"])
    assert row["status"] in trial_ledger.STATUSES
    assert str(row["authorization"]).strip()
    assert trial_ledger.owners_of("swing_house_variant_v1") == ("swing_house_variant_v1_twin",), (
        "exactly one trial claims the twin; two would double-count one look"
    )
    assert trial_ledger.backfill(tmp_path) == [], "idempotent: a declaration is never rewritten"


# --- band-coverage and the compare table ------------------------------------
def _occurrence_row(identity: str, symbol: str, day: date, **overrides) -> dict:
    trigger = xcal.trading_session(day).rth_close_at
    row = {
        "occurrence_id": identity,
        "symbol": symbol,
        "canonical_setup_id": "AVWAPE_TO_FIRST_DEV",
        "side": "LONG",
        "structural_timeframe": "D1",
        "trigger_timeframe": "D1",
        "anchor_instance_id": "",
        "dependency_cluster_id": "",
        "status": "TRIGGERED",
        "trigger_at": trigger,
        "trigger_bar_interval_start": trigger,
        "entry_price_ref": 100.0,
        "stop_price_ref": 95.0,
        "detector_version": "v1",
        "first_detected_run_id": "r1",
        "last_updated_run_id": "r1",
        "tags": "",
        "event_at": trigger,
        "observed_at": trigger,
        "computed_at": trigger,
        "revision_id": "rev-1",
        "supersedes_revision_id": "",
        "schema_version": schemas.SCHEMA_VERSION,
        "run_id": "r1",
    }
    row.update(overrides)
    return row


def _daily_row(
    symbol: str,
    day: date,
    knowledge: str,
    bands: dict | None,
    variant: dict | None = None,
    *,
    computed_at: datetime = NOW,
    feature_set_version: str | None = None,
) -> dict:
    row = {
        "symbol": symbol,
        "session_date": day,
        "feature_set_version": feature_set_version or features.FEATURE_SET_VERSION,
        "close": 100.0,
        "atr14": 2.0,
        "anchor_knowledge": knowledge,
        "computed_at": computed_at,
        "event_at": xcal.trading_session(day).rth_close_at,
        "input_capture_mode_worst": "BACKFILL",
        "input_manifest_hash": "hash",
        "schema_version": schemas.SCHEMA_VERSION,
        "run_id": "r1",
    }
    for key, value in (bands or {}).items():
        row[f"avwape_{key.lower()}"] = value
    if variant:
        row["avwap_variant_formula_version"] = avwap_band_variants.FEATURE_VERSION
        for key, value in variant.items():
            row[f"avwap_variant_{key.lower()}"] = value
    return row


COVERAGE_DAY = date(2026, 8, 12)


@pytest.fixture()
def compare_lake(store):
    """Three occurrences carrying BOTH band families, one of them variant-null."""
    store.publish(
        "setup_occurrence",
        [
            _occurrence_row("occ-a", "AAA", COVERAGE_DAY),
            _occurrence_row("occ-b", "BBB", COVERAGE_DAY),
            _occurrence_row("occ-c", "CCC", COVERAGE_DAY),
        ],
    )
    store.publish(
        "feature_snapshot_daily",
        [
            _daily_row("AAA", COVERAGE_DAY, features.ANCHOR_KNOWLEDGE_OBSERVED, CHAMPION_BANDS, VARIANT_BANDS),
            _daily_row(
                "BBB", COVERAGE_DAY, features.ANCHOR_KNOWLEDGE_RECONSTRUCTED, CHAMPION_BANDS, VARIANT_BANDS
            ),
            # Champion bands only: the challenger could not be measured here.
            _daily_row("CCC", COVERAGE_DAY, features.ANCHOR_KNOWLEDGE_RECONSTRUCTED, CHAMPION_BANDS, None),
        ],
    )
    bars = {
        "AAA": _swing_bars([100.0, 104.0, 109.0, 116.0], symbol="AAA"),
        "BBB": _swing_bars([100.0, 104.0, 109.0, 116.0], symbol="BBB"),
        "CCC": _swing_bars([100.0, 97.0, 94.0, 92.0], symbol="CCC"),
    }
    occurrences = [
        _occurrence(occurrence_id="occ-a", symbol="AAA"),
        _occurrence(occurrence_id="occ-b", symbol="BBB"),
        _occurrence(occurrence_id="occ-c", symbol="CCC"),
    ]
    outcomes.build_outcomes(
        store,
        occurrences,
        d1_by_symbol=bars,
        bands_by_occurrence={identity: CHAMPION_BANDS for identity in ("occ-a", "occ-b", "occ-c")},
        variant_bands_by_occurrence={"occ-a": VARIANT_BANDS, "occ-b": VARIANT_BANDS},
        recipes=(outcomes.SWING_HOUSE_V1, outcomes.SWING_HOUSE_VARIANT_V1),
        as_of=NOW,
        now=NOW,
    )
    return store


def test_band_coverage_lists_both_recipes_with_their_own_required_bands(compare_lake):
    report = cli.run_band_coverage(compare_lake, month="2026-08")
    assert "swing_house_v1" in report["recipes"]
    assert "swing_house_variant_v1" in report["recipes"]
    champion = report["recipes"]["swing_house_v1"]
    twin = report["recipes"]["swing_house_variant_v1"]
    assert champion["required_bands"] == [1, 2, 3]
    assert twin["required_bands"] == [1, 2, 3]
    assert champion["totals"]["required_bands_present"] == 3, "every row has champion bands"
    assert twin["totals"]["required_bands_present"] == 2, "CCC has no challenger bands"
    assert twin["totals"]["null_bands"] == 1
    assert twin["totals"]["plain_no_target"] == 1

    table = cli.format_band_coverage(report)
    assert "swing_house_variant_v1" in table


def test_the_compare_table_reads_both_recipes_on_the_same_occurrence_ids(compare_lake):
    report = cli.run_band_coverage_compare(
        compare_lake, month="2026-08", recipe_ids=("swing_house_v1", "swing_house_variant_v1")
    )
    assert report["status"] == "OK"
    assert report["recipes"] == ["swing_house_v1", "swing_house_variant_v1"]
    assert report["paired"] == 3, "every occurrence has a row under both recipes"
    assert report["not_paired"]["total"] == 0

    totals = report["totals"]["recipes"]
    champion = totals["swing_house_v1"]
    twin = totals["swing_house_variant_v1"]
    assert champion["n"] == twin["n"] == 3, "the same occurrence ids on both sides"
    assert champion["resolved"] == champion["targeted"] + champion["stopped"]
    assert champion["win_rate_lb"] == pytest.approx(
        wilson_lower_bound(champion["targeted"], champion["resolved"]), abs=1e-12
    )
    assert report["wilson_z"] == pytest.approx(WILSON_Z, abs=1e-12), "ONE Wilson"
    assert champion["mean_net_r"] is not None

    buckets = report["by_knowledge"]
    assert set(buckets) >= {"observed", "reconstructed"}
    assert buckets["observed"]["recipes"]["swing_house_v1"]["n"] == 1


def test_the_compare_numbers_equal_the_per_recipe_tables(compare_lake):
    compare = cli.run_band_coverage_compare(
        compare_lake, month="2026-08", recipe_ids=("swing_house_v1", "swing_house_variant_v1")
    )
    for recipe_id in ("swing_house_v1", "swing_house_variant_v1"):
        single = cli.run_band_coverage(compare_lake, month="2026-08", recipe_id=recipe_id)
        states = single["recipes"][recipe_id]["totals"]["by_result_state"]
        cell = compare["totals"]["recipes"][recipe_id]
        assert cell["targeted"] == states.get("TARGETED", 0), recipe_id
        assert cell["stopped"] == states.get("STOPPED", 0), recipe_id


def test_an_occurrence_missing_under_either_recipe_is_counted_not_paired(store):
    store.publish("setup_occurrence", [_occurrence_row("occ-lonely", "AAA", COVERAGE_DAY)])
    store.publish(
        "feature_snapshot_daily",
        [_daily_row("AAA", COVERAGE_DAY, features.ANCHOR_KNOWLEDGE_OBSERVED, CHAMPION_BANDS, VARIANT_BANDS)],
    )
    outcomes.build_outcomes(
        store,
        [_occurrence(occurrence_id="occ-lonely", symbol="AAA")],
        d1_by_symbol={"AAA": _swing_bars([100.0, 104.0, 109.0, 116.0], symbol="AAA")},
        bands_by_occurrence={"occ-lonely": CHAMPION_BANDS},
        recipes=(outcomes.SWING_HOUSE_V1,),
        as_of=NOW,
        now=NOW,
    )
    report = cli.run_band_coverage_compare(
        store, month="2026-08", recipe_ids=("swing_house_v1", "swing_house_variant_v1")
    )
    assert report["paired"] == 0
    assert report["not_paired"]["total"] == 1
    assert report["not_paired"]["missing_swing_house_variant_v1"] == 1
    table = cli.format_band_coverage_compare(report)
    assert "not_paired" in table


def test_the_compare_writes_nothing(compare_lake):
    ledger = compare_lake.manifest.path.read_bytes()
    cli.run_band_coverage_compare(
        compare_lake, month="2026-08", recipe_ids=("swing_house_v1", "swing_house_variant_v1")
    )
    assert compare_lake.manifest.path.read_bytes() == ledger


def test_the_cli_exposes_compare(compare_lake, monkeypatch, capsys):
    monkeypatch.setattr(cli.ResearchStore, "open", classmethod(lambda cls, root=None: compare_lake))
    assert (
        cli.main(
            [
                "band-coverage",
                "--month",
                "2026-08",
                "--compare",
                "swing_house_v1",
                "swing_house_variant_v1",
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "OK"
    assert payload["recipes"] == ["swing_house_v1", "swing_house_variant_v1"]

    assert cli.main(["band-coverage", "--month", "2026-08", "--compare", "swing_house_v1", "nope"]) == 1
    assert "nope" in capsys.readouterr().out


# --- the reader picks one snapshot row per session --------------------------
def test_the_band_read_prefers_the_newest_snapshot_row_for_a_session(store):
    """Two feature-set versions can now coexist for one (symbol, session); a
    reader that took whichever landed last would be reading a coin flip."""
    store.publish(
        "feature_snapshot_daily",
        [
            _daily_row(
                "AAA",
                COVERAGE_DAY,
                features.ANCHOR_KNOWLEDGE_OBSERVED,
                {"UPPER_1": 1.0, "UPPER_2": 2.0, "UPPER_3": 3.0},
                None,
                computed_at=NOW - timedelta(days=2),
                feature_set_version="tier1_v1",
            ),
            _daily_row(
                "AAA",
                COVERAGE_DAY,
                features.ANCHOR_KNOWLEDGE_OBSERVED,
                CHAMPION_BANDS,
                VARIANT_BANDS,
                computed_at=NOW,
                feature_set_version="tier1_v2",
            ),
        ],
    )
    known = {"occ-a": _occurrence_row("occ-a", "AAA", COVERAGE_DAY)}
    champion = cli._bands_by_occurrence(store, known)
    variant = cli._bands_by_occurrence(store, known, prefix=cli.VARIANT_BAND_PREFIX)
    assert champion["occ-a"]["UPPER_1"] == pytest.approx(CHAMPION_BANDS["UPPER_1"])
    assert variant["occ-a"]["UPPER_1"] == pytest.approx(VARIANT_BANDS["UPPER_1"])
