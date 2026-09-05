"""Packet Q2: the band repair chain is verifiable.

Four claims are pinned here, each of which the un-fixed code cannot make:

* **Q2.1** an anchor's KNOWLEDGE stamp travels with the daily snapshot. An
  anchor whose ``system_from`` lands after the session is ``reconstructed``
  evidence for research, never something the desk knew that day; a row written
  before the column existed reads as ``legacy`` and is never assumed observed.
* **Q2.2** a swing outcome row NAMES the path it walked (``managed`` /
  ``plain_target`` / ``plain_no_target``), and labelling it changes no number.
* **Q2.3** ``band-coverage`` reports required-band coverage per recipe and per
  knowledge bucket, and writes nothing.
* **Q2.4** past daily features can be rebuilt WITH their anchors, dry run by
  default, superseding rather than duplicating, and never touching the rows
  outside the requested range.

Every test runs against a tmp :class:`ResearchStore`. The live lake is
read-only to this packet.
"""

from __future__ import annotations

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

from scripts.research_warehouse import (  # noqa: E402
    cli,
    exchange_calendar as xcal,
    features,
    outcomes,
    schemas,
)
from scripts.research_warehouse.manifest import lake_relative  # noqa: E402
from scripts.research_warehouse.store import ResearchStore, _sha256_file  # noqa: E402

UTC = timezone.utc
NOW = datetime(2026, 9, 4, 23, 0, tzinfo=UTC)
SESSION = date(2026, 8, 12)


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


def _sessions(first: date, last: date) -> list[date]:
    return [
        item.session_date if hasattr(item, "session_date") else item
        for item in xcal.sessions_between(first, last)
    ]


def _publish_old_shape(store: ResearchStore, rows, partition: str, *, dataset: str, drop: str) -> Path:
    """Write a partition the way the code wrote it BEFORE ``drop`` existed."""
    spec = schemas.dataset_spec(dataset)
    old_schema = pa.schema([field for field in spec.schema if field.name != drop])
    table = pa.Table.from_pylist(
        [{key: value for key, value in row.items() if key != drop} for row in rows],
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
        job_id="pre_q2",
    )
    return path


# --- Q2.1 the anchor's knowledge stamp --------------------------------------
def test_an_anchor_observed_after_the_session_is_reconstructed(store):
    """The bridge lands ~2,200 anchors tonight; a rebuilt August row must say so."""
    features.build_anchor_instances(
        store,
        [{"symbol": "OBS", "anchor_bar_date": date(2026, 7, 20)}],
        now=datetime(2026, 8, 1, 23, 0, tzinfo=UTC),
    )
    features.build_anchor_instances(
        store,
        [{"symbol": "RECON", "anchor_bar_date": date(2026, 7, 21)}],
        now=NOW,
    )
    # 00:30 UTC on the 13th is 20:30 ET on the 12th: market-local by
    # astimezone, never by stripping the timezone.
    features.build_anchor_instances(
        store,
        [{"symbol": "EVENING", "anchor_bar_date": date(2026, 7, 22)}],
        now=datetime(2026, 8, 13, 0, 30, tzinfo=UTC),
    )
    features.build_anchor_instances(
        store,
        [{"symbol": "LATER", "anchor_bar_date": date(2026, 8, 20)}],
        now=datetime(2026, 8, 1, 23, 0, tzinfo=UTC),
    )

    chosen = cli.anchor_dates_by_symbol(store, SESSION)

    assert chosen["OBS"].anchor_bar_date == date(2026, 7, 20)
    assert chosen["OBS"].knowledge == features.ANCHOR_KNOWLEDGE_OBSERVED
    assert chosen["RECON"].knowledge == features.ANCHOR_KNOWLEDGE_RECONSTRUCTED
    assert chosen["EVENING"].knowledge == features.ANCHOR_KNOWLEDGE_OBSERVED
    assert "LATER" not in chosen, "an anchor that had not happened yet is not knowable"


def test_the_daily_snapshot_row_carries_the_anchor_knowledge_label(store):
    history = _history()
    store.publish("bar_d1", history)
    anchor_day = history[-10]["session_date"]

    features.build_daily_snapshots(
        store,
        history[-1]["session_date"],
        symbols=["AAPL"],
        anchors_by_symbol={
            "AAPL": features.AnchorChoice(anchor_day, features.ANCHOR_KNOWLEDGE_RECONSTRUCTED)
        },
        now=NOW,
    )
    row = store.read_rows("feature_snapshot_daily", "year=2026")[0]
    assert row["anchor_knowledge"] == features.ANCHOR_KNOWLEDGE_RECONSTRUCTED
    assert row["avwape_value"] is not None, "the label rides with real bands, it does not replace them"


def test_a_row_with_no_anchor_is_not_labelled_observed(store):
    history = _history()
    store.publish("bar_d1", history)
    features.build_daily_snapshots(
        store, history[-1]["session_date"], symbols=["AAPL"], now=NOW
    )
    row = store.read_rows("feature_snapshot_daily", "year=2026")[0]
    assert features.anchor_knowledge_bucket(row["anchor_knowledge"]) == features.ANCHOR_KNOWLEDGE_NONE


def test_a_partition_written_before_the_column_still_reads_beside_a_new_one(store):
    """Schema promotion: the old shape reads as LEGACY, never as observed."""
    history = _history()
    store.publish("bar_d1", history)
    features.build_daily_snapshots(
        store,
        history[-1]["session_date"],
        symbols=["AAPL"],
        anchors_by_symbol={
            "AAPL": features.AnchorChoice(
                history[-10]["session_date"], features.ANCHOR_KNOWLEDGE_OBSERVED
            )
        },
        now=NOW,
    )
    fresh = store.read_rows("feature_snapshot_daily", "year=2026")
    assert len(fresh) == 1

    legacy = dict(fresh[0])
    legacy["symbol"] = "OLD"
    _publish_old_shape(
        store, [legacy], "year=2026", dataset="feature_snapshot_daily", drop="anchor_knowledge"
    )

    by_symbol = {row["symbol"]: row for row in store.read_rows("feature_snapshot_daily", "year=2026")}
    assert set(by_symbol) == {"AAPL", "OLD"}, "an old-shape partition still reads"
    assert by_symbol["OLD"]["anchor_knowledge"] is None
    assert features.anchor_knowledge_bucket(by_symbol["OLD"]["anchor_knowledge"]) == (
        features.ANCHOR_KNOWLEDGE_LEGACY
    )
    assert features.anchor_knowledge_bucket(by_symbol["AAPL"]["anchor_knowledge"]) == (
        features.ANCHOR_KNOWLEDGE_OBSERVED
    )


# --- Q2.2 the outcome row names its path ------------------------------------
TRIGGER_DAY = date(2026, 8, 3)
TRIGGER_AT = xcal.trading_session(TRIGGER_DAY).rth_close_at


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


def _swing_bars(closes, *, start: date = TRIGGER_DAY) -> list[dict]:
    rows, day = [], start
    for close in closes:
        while not xcal.is_trading_day(day):
            day += timedelta(days=1)
        rows.append(
            {
                "symbol": "AAPL",
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


BANDS_FULL = {"UPPER_1": 103.0, "UPPER_2": 108.0, "UPPER_3": 115.0}
BANDS_TARGET_ONLY = {"UPPER_3": 115.0}


def test_the_swing_row_names_the_path_it_walked():
    bars = _swing_bars([100.0, 104.0, 109.0, 116.0])
    managed = outcomes.simulate_swing(
        _occurrence(), bars, outcomes.SWING_HOUSE_V1, bands=BANDS_FULL, as_of=NOW, computed_at=NOW
    )
    assert managed["path_kind"] == outcomes.PATH_KIND_MANAGED

    fallback = outcomes.simulate_swing(
        _occurrence(),
        bars,
        outcomes.SWING_HOUSE_V1,
        bands=BANDS_TARGET_ONLY,
        as_of=NOW,
        computed_at=NOW,
    )
    assert fallback["path_kind"] == outcomes.PATH_KIND_PLAIN_TARGET, "BD-42's declared band-3 fallback"

    blind = outcomes.simulate_swing(
        _occurrence(), bars, outcomes.SWING_HOUSE_V1, bands=None, as_of=NOW, computed_at=NOW
    )
    assert blind["path_kind"] == outcomes.PATH_KIND_PLAIN_NO_TARGET, (
        "no bands at all: the walk has no target and the row must say so"
    )

    fixed = outcomes.simulate_swing(
        _occurrence(),
        bars,
        outcomes.CONTROL_FIXED_1R2R_V1,
        bands=None,
        as_of=NOW,
        computed_at=NOW,
    )
    assert fixed["path_kind"] == outcomes.PATH_KIND_PLAIN_TARGET, "target_r needs no band"

    time_only = outcomes.simulate_swing(
        _occurrence(),
        bars,
        outcomes.CONTROL_TIME_ONLY_V1,
        bands=None,
        as_of=NOW,
        computed_at=NOW,
    )
    assert time_only["path_kind"] == outcomes.PATH_KIND_PLAIN_NO_TARGET


#: Read off the UN-LABELLED code at 6b74165 on 2026-09-04 by running the same
#: five simulations against the reverted files: (result_state, gross_r, net_r,
#: mfe_r). Q2.2 adds a column and must move none of them.
SWING_GOLDEN = {
    "managed": ("TARGETED", 2.3, 2.2866, 3.4),
    "fallback": ("TARGETED", 3.0, 2.9866, 3.4),
    "blind": ("TRUNCATED", None, None, 3.4),
    "fixed": ("TARGETED", 2.0, 1.9866, 2.0),
    "time_only": ("TRUNCATED", None, None, 3.4),
}


def test_labelling_the_path_changes_no_outcome_number():
    bars = _swing_bars([100.0, 104.0, 109.0, 116.0])
    cases = {
        "managed": (BANDS_FULL, outcomes.SWING_HOUSE_V1),
        "fallback": (BANDS_TARGET_ONLY, outcomes.SWING_HOUSE_V1),
        "blind": (None, outcomes.SWING_HOUSE_V1),
        "fixed": (None, outcomes.CONTROL_FIXED_1R2R_V1),
        "time_only": (None, outcomes.CONTROL_TIME_ONLY_V1),
    }
    for name, (bands, recipe) in cases.items():
        row = outcomes.simulate_swing(
            _occurrence(), bars, recipe, bands=bands, as_of=NOW, computed_at=NOW
        )
        state, gross, net, mfe = SWING_GOLDEN[name]
        assert row["result_state"] == state, name
        assert row["gross_r"] == (None if gross is None else pytest.approx(gross, abs=1e-9)), name
        assert row["net_r"] == (None if net is None else pytest.approx(net, abs=1e-4)), name
        assert row["mfe_r"] == pytest.approx(mfe, abs=1e-9), name


def test_the_unchanged_comparison_ignores_the_new_label(store):
    """An existing row must not be rewritten just because a column appeared."""
    bars = _swing_bars([100.0, 104.0, 109.0, 116.0])
    occurrence = _occurrence()
    computed = outcomes.simulate_swing(
        occurrence, bars, outcomes.SWING_HOUSE_V1, bands=BANDS_FULL, as_of=NOW, computed_at=NOW
    )
    assert "path_kind" in computed
    previous = {key: value for key, value in computed.items() if key != "path_kind"}
    assert outcomes._same_outcome(previous, computed), (
        "path_kind is excluded from the BD-98 comparison; an unchanged row stays unlabelled"
    )

    store.publish("outcome_path", [previous])
    report = outcomes.build_outcomes(
        store,
        [occurrence],
        d1_by_symbol={"AAPL": bars},
        bands_by_occurrence={"occ-1": BANDS_FULL},
        recipes=(outcomes.SWING_HOUSE_V1,),
        as_of=NOW,
        now=NOW,
        force=True,
    )
    assert report.rows == 0 and report.skipped.get("UNCHANGED") == 1


def test_an_unlabelled_row_reads_as_unlabelled():
    assert outcomes.path_kind_bucket(None) == outcomes.PATH_KIND_UNLABELLED
    assert outcomes.path_kind_bucket("managed") == outcomes.PATH_KIND_MANAGED


# --- Q2.3 band-coverage ------------------------------------------------------
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


def _daily_row(symbol: str, day: date, knowledge: str, bands: dict | None) -> dict:
    row = {
        "symbol": symbol,
        "session_date": day,
        "feature_set_version": features.FEATURE_SET_VERSION,
        "close": 100.0,
        "atr14": 2.0,
        "anchor_knowledge": knowledge,
        "computed_at": NOW,
        "event_at": xcal.trading_session(day).rth_close_at,
        "input_capture_mode_worst": "BACKFILL",
        "input_manifest_hash": "hash",
        "schema_version": schemas.SCHEMA_VERSION,
        "run_id": "r1",
    }
    for key, value in (bands or {}).items():
        row[f"avwape_{key.lower()}"] = value
    return row


@pytest.fixture()
def coverage_lake(store):
    """Three occurrences: one managed (observed), one plain_target
    (reconstructed), one plain_no_target (legacy, written before the column)."""
    day = date(2026, 8, 12)
    store.publish(
        "setup_occurrence",
        [
            _occurrence_row("occ-managed", "AAA", day),
            _occurrence_row("occ-target", "BBB", day),
            _occurrence_row("occ-blind", "CCC", day),
            # Outside the month: never counted.
            _occurrence_row("occ-september", "DDD", date(2026, 9, 2)),
        ],
    )
    store.publish(
        "feature_snapshot_daily",
        [
            _daily_row("AAA", day, features.ANCHOR_KNOWLEDGE_OBSERVED, BANDS_FULL),
            _daily_row("BBB", day, features.ANCHOR_KNOWLEDGE_RECONSTRUCTED, BANDS_TARGET_ONLY),
        ],
    )
    _publish_old_shape(
        store,
        [_daily_row("CCC", day, features.ANCHOR_KNOWLEDGE_NONE, None)],
        "year=2026",
        dataset="feature_snapshot_daily",
        drop="anchor_knowledge",
    )
    return store


def test_band_coverage_counts_by_recipe_and_by_knowledge(coverage_lake):
    report = cli.run_band_coverage(coverage_lake, month="2026-08", recipe_id="swing_house_v1")
    recipe = report["recipes"]["swing_house_v1"]

    assert report["month"] == "2026-08"
    assert recipe["required_bands"] == [1, 2, 3]
    assert recipe["totals"]["occurrences"] == 3, "the September occurrence is another month"
    assert recipe["totals"]["required_bands_present"] == 1
    assert recipe["totals"]["plain_no_target"] == 1
    assert recipe["totals"]["null_bands"] == 1
    assert recipe["totals"]["geometry_valid"] == 2, "a long's target above entry and stop below"

    buckets = recipe["by_knowledge"]
    assert buckets["observed"]["occurrences"] == 1
    assert buckets["observed"]["required_bands_present"] == 1
    assert buckets["reconstructed"]["occurrences"] == 1
    assert buckets["reconstructed"]["plain_no_target"] == 0
    assert buckets["legacy"]["occurrences"] == 1
    assert buckets["legacy"]["plain_no_target"] == 1


def test_band_coverage_counts_result_states(coverage_lake):
    row = outcomes.simulate_swing(
        _occurrence(occurrence_id="occ-managed", symbol="AAA"),
        _swing_bars([100.0, 104.0, 109.0, 116.0]),
        outcomes.SWING_HOUSE_V1,
        bands=BANDS_FULL,
        as_of=NOW,
        computed_at=NOW,
    )
    coverage_lake.publish("outcome_path", [row])
    report = cli.run_band_coverage(coverage_lake, month="2026-08", recipe_id="swing_house_v1")
    states = report["recipes"]["swing_house_v1"]["totals"]["by_result_state"]
    assert states.get("TARGETED") == 1
    assert states.get("NOT_SIMULATED") == 2, "an occurrence with no outcome row is named, not dropped"


def test_band_coverage_writes_nothing(coverage_lake):
    ledger = coverage_lake.manifest.path.read_bytes()
    cli.run_band_coverage(coverage_lake, month="2026-08")
    assert coverage_lake.manifest.path.read_bytes() == ledger


# --- Q2.4 rebuilding past daily features ------------------------------------
REBUILD_FIRST = date(2026, 8, 10)
REBUILD_LAST = date(2026, 8, 11)


@pytest.fixture()
def rebuild_lake(store):
    history = _history(days=60, first=date(2026, 6, 1))
    store.publish("bar_d1", history)
    anchor_day = [row["session_date"] for row in history if row["session_date"] < REBUILD_FIRST][-8]
    features.build_anchor_instances(
        store, [{"symbol": "AAPL", "anchor_bar_date": anchor_day}], now=NOW
    )
    return store


def test_rebuild_daily_features_dry_run_lists_the_sessions_and_writes_nothing(rebuild_lake):
    ledger = rebuild_lake.manifest.path.read_bytes()
    plan = cli.run_rebuild_daily_features(
        rebuild_lake, start=REBUILD_FIRST, end=REBUILD_LAST, now=NOW
    )
    assert plan["status"] == "OK" and plan["applied"] is False
    assert plan["sessions"] == ["2026-08-10", "2026-08-11"]
    assert rebuild_lake.manifest.path.read_bytes() == ledger
    assert rebuild_lake.read_rows("feature_snapshot_daily", "year=2026") == []


def test_rebuild_daily_features_writes_labelled_rows_with_bands(rebuild_lake):
    report = cli.run_rebuild_daily_features(
        rebuild_lake, start=REBUILD_FIRST, end=REBUILD_LAST, apply=True, now=NOW
    )
    assert report["status"] == "OK" and report["applied"] is True

    rows = rebuild_lake.read_rows("feature_snapshot_daily", "year=2026")
    assert {row["session_date"] for row in rows} == {REBUILD_FIRST, REBUILD_LAST}
    for row in rows:
        assert row["avwape_value"] is not None, "the rebuild carries the anchor, not just the frame"
        assert row["anchor_knowledge"] == features.ANCHOR_KNOWLEDGE_RECONSTRUCTED, (
            "tonight's anchor was not knowable in August"
        )


def test_a_second_apply_supersedes_rather_than_duplicating(rebuild_lake):
    cli.run_rebuild_daily_features(
        rebuild_lake, start=REBUILD_FIRST, end=REBUILD_LAST, apply=True, now=NOW
    )
    first = rebuild_lake.read_rows("feature_snapshot_daily", "year=2026")
    cli.run_rebuild_daily_features(
        rebuild_lake, start=REBUILD_FIRST, end=REBUILD_LAST, apply=True, now=NOW
    )
    second = rebuild_lake.read_rows("feature_snapshot_daily", "year=2026")
    assert len(second) == len(first) and len(first) > 0
    assert rebuild_lake.duplicate_rows("feature_snapshot_daily", "year=2026").rows_dropped == 0


def test_the_rebuild_keeps_the_rows_outside_its_range(rebuild_lake):
    """The partition is YEAR-keyed: a January row must survive an August rebuild."""
    survivor = _daily_row("ZZZ", date(2026, 1, 5), features.ANCHOR_KNOWLEDGE_OBSERVED, BANDS_FULL)
    rebuild_lake.publish("feature_snapshot_daily", [survivor])

    cli.run_rebuild_daily_features(
        rebuild_lake, start=REBUILD_FIRST, end=REBUILD_LAST, apply=True, now=NOW
    )
    rows = rebuild_lake.read_rows("feature_snapshot_daily", "year=2026")
    kept = [row for row in rows if row["symbol"] == "ZZZ"]
    assert len(kept) == 1
    assert kept[0]["session_date"] == date(2026, 1, 5)
    assert kept[0]["avwape_upper_2"] == pytest.approx(108.0)


# --- reviewer advisories, 2026-09-04 ----------------------------------------
def test_a_recipe_that_needs_no_band_reports_n_a_rather_than_a_full_house(coverage_lake):
    """`control_fixed_1r2r_v1 n=2437 bands=2437 null=2431` read as a
    contradiction on the live lake. A recipe whose target is an R multiple has
    no band requirement, so "all of them present" is not a fact about it."""
    report = cli.run_band_coverage(coverage_lake, month="2026-08", recipe_id="control_fixed_1r2r_v1")
    recipe = report["recipes"]["control_fixed_1r2r_v1"]
    assert recipe["required_bands"] == []
    assert recipe["totals"]["required_bands_present"] is None
    assert all(
        bucket["required_bands_present"] is None for bucket in recipe["by_knowledge"].values()
    )
    assert recipe["totals"]["null_bands"] == 1, "the band counts themselves are unchanged"

    table = cli.format_band_coverage(report)
    assert "required bands: none" in table
    assert "n/a" in table, "the table must not print a number it cannot mean"


def test_the_table_names_each_recipes_required_bands(coverage_lake):
    table = cli.format_band_coverage(
        cli.run_band_coverage(coverage_lake, month="2026-08", recipe_id="swing_house_v1")
    )
    assert "required bands: 1,2,3" in table


def test_an_unrecognised_knowledge_value_is_unknown_not_none():
    """`none` means "this row used no anchor" - a positive statement. A value
    nobody wrote must never borrow it."""
    assert features.anchor_knowledge_bucket("banana") == features.ANCHOR_KNOWLEDGE_UNKNOWN
    assert features.anchor_knowledge_bucket("") == features.ANCHOR_KNOWLEDGE_NONE
    assert features.anchor_knowledge_bucket(None) == features.ANCHOR_KNOWLEDGE_LEGACY
    assert features.anchor_knowledge_bucket("observed") == features.ANCHOR_KNOWLEDGE_OBSERVED


def test_the_rebuild_refuses_to_report_a_carry_it_did_not_make(rebuild_lake, monkeypatch):
    """A carried row is data that was live a moment ago. If the republish drops
    or quarantines any of it, the run RAISES - a count that cannot fail is not
    evidence."""
    survivor = _daily_row("ZZZ", date(2026, 1, 5), features.ANCHOR_KNOWLEDGE_OBSERVED, BANDS_FULL)
    rebuild_lake.publish("feature_snapshot_daily", [survivor])

    real_publish = rebuild_lake.publish

    def _lossy(dataset, rows, **kwargs):
        result = real_publish(dataset, rows, **kwargs)
        if dataset == "feature_snapshot_daily" and result.rows_published:
            result.rows_published -= 1  # one row silently short
        return result

    monkeypatch.setattr(rebuild_lake, "publish", _lossy)
    with pytest.raises(Exception, match="carr"):
        cli.run_rebuild_daily_features(
            rebuild_lake, start=REBUILD_FIRST, end=REBUILD_LAST, apply=True, now=NOW
        )
