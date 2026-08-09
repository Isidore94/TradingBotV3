"""4-step seal protocol and its crash matrix (plan sec 8.3, 19.3, 20).

Pinned here:
- the write path is exactly stage -> hash/validate -> os.replace -> manifest;
- a crash during staging leaves artifacts ONLY in ``_incoming/``;
- a crash between the rename and the manifest append leaves an invisible file
  that startup reconciliation adopts (never a double count, never a delete);
- an unset ``research_store_dir`` makes the store a total no-op.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pyarrow.parquet as pq
import pytest

from scripts.research_warehouse import config, schemas
from scripts.research_warehouse.manifest import ManifestLog
from scripts.research_warehouse.store import (
    QUARANTINE_ORPHAN_OVERLAPS_LIVE,
    LakeIntegrityError,
    ResearchStore,
    lake_relative,
)

UTC = timezone.utc


def _session_row(day: int = 3, **overrides):
    row = {
        "session_id": f"XNYS-2026-08-{day:02d}",
        "exchange_calendar": "XNYS/v1",
        "session_date": date(2026, 8, day),
        "rth_open_at": datetime(2026, 8, day, 13, 30, tzinfo=UTC),
        "rth_close_at": datetime(2026, 8, day, 20, 0, tzinfo=UTC),
        "eth_open_at": datetime(2026, 8, day, 8, 0, tzinfo=UTC),
        "eth_close_at": datetime(2026, 8, day, 0, 0, tzinfo=UTC),
        "is_half_day": False,
        "expected_m5_bars_rth": 78,
        "expected_m1_bars_rth": 390,
        "calendar_version": "v1",
        "observed_at": datetime(2026, 8, day, 20, 5, tzinfo=UTC),
        "schema_version": schemas.SCHEMA_VERSION,
        "run_id": "run-1",
    }
    row.update(overrides)
    return row


def _bar_row(symbol="AAPL", minute=0, **overrides):
    start = datetime(2026, 8, 3, 13, 30, tzinfo=UTC) + timedelta(minutes=minute)
    row = {
        "symbol": symbol,
        "interval_start": start,
        "interval_end": start + timedelta(minutes=5),
        "session_id": "XNYS-2026-08-03",
        "session_phase": "RTH",
        "open": 100.0,
        "high": 101.0,
        "low": 99.5,
        "close": 100.5,
        "volume": 12345,
        "vwap": 100.2,
        "trade_count": 88,
        "provider": "IBKR",
        "is_complete": True,
        "quality": "COMPLETE",
        "source_hash": "abc",
        "event_at": start + timedelta(minutes=5),
        "observed_at": start + timedelta(minutes=6),
        "capture_mode": "LIVE",
        "revision_id": "r1",
        "supersedes_revision_id": "",
        "schema_version": schemas.SCHEMA_VERSION,
        "run_id": "run-1",
    }
    row.update(overrides)
    return row


@pytest.fixture()
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


def test_store_open_is_a_no_op_when_research_dir_is_unset(monkeypatch, tmp_path):
    monkeypatch.delenv(config.RESEARCH_DIR_ENV, raising=False)
    monkeypatch.setattr(config, "warehouse_enabled", lambda: False)
    assert ResearchStore.open() is None


def test_open_creates_the_directory_contract(tmp_path):
    store = ResearchStore.open(tmp_path / "lake")
    for name in config.LAKE_SUBDIRS:
        assert (store.root / name).is_dir()
    assert (store.root / "manifest_log.jsonl").is_file()


def test_publish_seals_and_records_one_manifest_line(store):
    result = store.publish("trading_session", [_session_row()], job_id="job-1")

    assert result.rows_published == 1 and not result.quarantined
    entry = result.published[0]
    assert entry.partition == "year=2026"
    sealed = store.root / entry.file_path
    # Step 3 target: <layer>/<dataset>/<partition>/part-<uuid>.parquet
    assert sealed.exists() and sealed.name.startswith("part-") and sealed.suffix == ".parquet"
    assert entry.file_path.startswith("silver/trading_session/year=2026/")
    # Step 2 evidence travels in the ledger.
    assert len(entry.sha256) == 64 and entry.row_count == 1
    assert entry.min_ts == "2026-08-03" and entry.max_ts == "2026-08-03"
    assert entry.job_id == "job-1"
    # Nothing is left staged.
    assert list(store.incoming_dir.iterdir()) == []
    assert store.read_table("trading_session").num_rows == 1


def test_partition_spec_is_the_locked_one(store):
    m5 = store.publish("bar_m5", [_bar_row()])
    assert m5.published[0].partition == "month=2026-08"
    derived = store.publish(
        "bar_derived",
        [
            {
                "symbol": "AAPL",
                "timeframe": "M15",
                "aggregation_contract_id": "rth_v1",
                "interval_start": datetime(2026, 8, 3, 13, 30, tzinfo=UTC),
                "interval_end": datetime(2026, 8, 3, 13, 45, tzinfo=UTC),
                "session_id": "XNYS-2026-08-03",
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": 10,
                "is_stub": False,
                "stub_duration_min": None,
                "constituent_count": 3,
                "constituent_expected": 3,
                "is_complete": True,
                "quality": "COMPLETE",
                "event_at": datetime(2026, 8, 3, 13, 45, tzinfo=UTC),
                "computed_at": datetime(2026, 8, 3, 21, 0, tzinfo=UTC),
                "input_capture_mode_worst": "LIVE",
                "schema_version": schemas.SCHEMA_VERSION,
                "run_id": "run-1",
            }
        ],
    )
    assert derived.published[0].partition == "timeframe=M15/month=2026-08"
    # Symbol-hash bucketing is stable across processes (never built on hash()).
    assert schemas.symbol_bucket("AAPL") == schemas.symbol_bucket("aapl")
    assert 0 <= schemas.symbol_bucket("AAPL") < schemas.SYMBOL_HASH_BUCKETS


def test_rows_split_across_partitions_seal_separately(store):
    rows = [_bar_row(minute=0), _bar_row(minute=0, interval_start=datetime(2026, 9, 1, 13, 30, tzinfo=UTC))]
    result = store.publish("bar_m5", rows)
    assert sorted(result.partitions) == ["month=2026-08", "month=2026-09"]
    assert store.read_table("bar_m5", "month=2026-09").num_rows == 1


def test_crash_during_staging_leaves_artifacts_only_in_incoming(store, monkeypatch):
    def explode(*args, **kwargs):
        raise OSError("simulated power loss after the staged write")

    monkeypatch.setattr("scripts.research_warehouse.store.os.replace", explode)
    with pytest.raises(OSError):
        store.publish("trading_session", [_session_row()])

    staged = list(store.incoming_dir.iterdir())
    assert len(staged) == 1 and staged[0].suffix == ".parquet"
    assert not any((store.root / "silver").rglob("*.parquet"))
    assert store.manifest.read_entries() == []
    assert store.read_table("trading_session").num_rows == 0


def test_stale_staged_files_are_quarantined_not_deleted(store, monkeypatch):
    monkeypatch.setattr("scripts.research_warehouse.store.os.replace", _boom)
    with pytest.raises(OSError):
        store.publish("trading_session", [_session_row()])
    monkeypatch.undo()

    result = store.reconcile(incoming_grace_seconds=0)
    assert result.stale_incoming and list(store.incoming_dir.iterdir()) == []
    quarantined = store.manifest.quarantine_entries()
    assert quarantined and (store.root / quarantined[0].file_path).exists()


def test_crash_between_rename_and_manifest_append_is_reconciled(store, monkeypatch):
    """Step 3 succeeded, step 4 never ran: the file is invisible until startup."""
    appended = []

    original_append = ManifestLog.append

    def fail_append(self, **kwargs):
        appended.append(kwargs)
        raise OSError("simulated crash before the manifest line landed")

    monkeypatch.setattr(ManifestLog, "append", fail_append)
    with pytest.raises(OSError):
        store.publish("trading_session", [_session_row()])
    monkeypatch.setattr(ManifestLog, "append", original_append)

    orphan = next((store.root / "silver").rglob("*.parquet"))
    assert store.read_table("trading_session").num_rows == 0  # invisible, by design

    result = store.reconcile(job_id="startup")
    assert len(result.adopted) == 1
    assert result.adopted[0].extra.get("reconciled") is True
    assert orphan.exists()
    assert store.read_table("trading_session").num_rows == 1
    # Re-running reconciliation changes nothing.
    assert store.reconcile().adopted == []


def test_reconcile_quarantines_a_duplicate_of_already_published_content(store):
    entry = store.publish("trading_session", [_session_row()]).published[0]
    source = store.root / entry.file_path
    twin = source.with_name("part-deadbeef.parquet")
    twin.write_bytes(source.read_bytes())

    result = store.reconcile()
    assert not result.adopted and len(result.quarantined) == 1
    assert not twin.exists()  # moved into _quarantine, never deleted
    assert store.read_table("trading_session").num_rows == 1


def test_a_crashed_compaction_is_quarantined_not_adopted(store, monkeypatch):
    """D14: adopting a crashed compaction's merged file double-counts everything.

    A compaction that dies between its ``os.replace`` and its manifest append
    leaves a merged file whose hash matches nothing registered -- so BD-03's
    hash guard waves it through -- while its source parts are all still live.
    Adopting it as a fresh PUBLISH counts every row in the partition twice, and
    the next compaction balances, because both sides doubled.
    """
    first = store.publish("trading_session", [_session_row(day=3)]).published[0]
    second = store.publish("trading_session", [_session_row(day=4)]).published[0]
    partition = first.partition
    assert store.read_table("trading_session").num_rows == 2

    original_append = ManifestLog.append

    def fail_append(self, **kwargs):
        raise OSError("simulated crash after os.replace, before the COMPACT line")

    monkeypatch.setattr(ManifestLog, "append", fail_append)
    with pytest.raises(OSError):
        store.compact("trading_session", partition, job_id="compaction")
    monkeypatch.setattr(ManifestLog, "append", original_append)

    # The merged file is on disk, unregistered, and its parts are still live.
    live = {entry.file_path for entry in store.manifest.resolve().entries}
    assert {first.file_path, second.file_path} <= live
    orphans = [
        path
        for path in (store.root / "silver").rglob("*.parquet")
        if lake_relative(store.root, path) not in live
    ]
    assert len(orphans) == 1, "the crashed compaction left exactly one merged file"

    result = store.reconcile(job_id="startup")
    assert not result.adopted, "a crashed compaction must never be adopted"
    assert len(result.quarantined) == 1
    assert result.quarantined[0].extra.get("reasons") == [QUARANTINE_ORPHAN_OVERLAPS_LIVE]
    assert not orphans[0].exists()  # moved to _quarantine, never deleted

    # The partition still holds its two rows, each exactly once.
    assert store.read_table("trading_session").num_rows == 2
    sessions = store.read_table("trading_session").column("session_id").to_pylist()
    assert sorted(sessions) == ["XNYS-2026-08-03", "XNYS-2026-08-04"]

    # And compaction still works afterwards.
    assert store.compact("trading_session", partition, job_id="compaction") is not None
    assert store.read_table("trading_session").num_rows == 2


def test_a_genuinely_new_orphan_is_still_adopted(store, monkeypatch):
    """The D14 guard must not break BD-03: new content is adopted, not lost."""
    store.publish("trading_session", [_session_row(day=3)])

    original_append = ManifestLog.append
    monkeypatch.setattr(
        ManifestLog, "append", lambda self, **kwargs: (_ for _ in ()).throw(OSError("crash"))
    )
    with pytest.raises(OSError):
        store.publish("trading_session", [_session_row(day=5)])
    monkeypatch.setattr(ManifestLog, "append", original_append)

    result = store.reconcile(job_id="startup")
    assert len(result.adopted) == 1 and not result.quarantined
    assert store.read_table("trading_session").num_rows == 2


def test_reconcile_reports_a_manifest_live_file_that_vanished(store):
    entry = store.publish("trading_session", [_session_row()]).published[0]
    (store.root / entry.file_path).unlink()

    result = store.reconcile()
    assert result.missing_live_files == [entry.file_path]
    with pytest.raises(LakeIntegrityError):
        store.read_table("trading_session")


def test_torn_manifest_tail_is_repaired_not_treated_as_corruption(store):
    store.publish("trading_session", [_session_row()])
    with open(store.manifest.path, "a", encoding="utf-8") as handle:
        handle.write('{"manifest_seq": 2, "action": "PUB')  # killed mid-append

    assert len(store.manifest.read_entries()) == 1
    assert store.reconcile().torn_manifest_tail_repaired is True
    store.publish("trading_session", [_session_row(day=4)])
    entries = store.manifest.read_entries()
    assert [entry.manifest_seq for entry in entries] == [1, 2]
    for line in store.manifest.path.read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_dataset_registry_is_the_frozen_first_increment(store):
    """Exactly the 13 tables of plan sec 7.1 - the slice schema set is frozen."""
    assert sorted(schemas.DATASETS) == sorted(
        [
            "anchor_instance",
            "bar_d1",
            "bar_derived",
            "bar_m5",
            "collection_gap",
            "feature_snapshot_daily",
            "feature_snapshot_intraday",
            "level_state_daily",
            "outcome_path",
            "scan_coverage",
            "setup_occurrence",
            "trading_session",
            "universe_membership_daily",
        ]
    )
    with pytest.raises(KeyError):
        store.publish("bar_m1", [])


def test_every_dataset_carries_the_convention_columns():
    for name, spec in schemas.DATASETS.items():
        names = set(spec.schema.names)
        assert {"schema_version", "run_id"} <= names, name
        assert spec.time_column in names, name
        assert spec.layer in {"bronze", "silver", "gold"}, name
        for dimension in spec.partition_by:
            assert dimension in {"year", "month", "timeframe", "symbol_bucket"}, name
    # Bar/observation datasets carry the point-in-time observation columns.
    for name in ("bar_m5", "bar_d1"):
        names = set(schemas.DATASETS[name].schema.names)
        assert {"event_at", "observed_at", "capture_mode", "revision_id"} <= names
    # MATURED is derived, never a stored result state (sec 14.2).
    assert "MATURED" not in schemas.RESULT_STATES


def test_sealed_files_are_zstd_parquet(store):
    entry = store.publish("bar_m5", [_bar_row()]).published[0]
    metadata = pq.ParquetFile(store.root / entry.file_path).metadata
    assert metadata.row_group(0).column(0).compression.lower() == "zstd"


def _boom(*args, **kwargs):
    raise OSError("simulated power loss after the staged write")



def _gap_row(*, detected_at, resolution=None, **overrides):
    row = {
        "symbol": "AAPL",
        "timeframe": "M5",
        "gap_start": datetime(2026, 8, 3, 13, 30, tzinfo=UTC),
        "gap_end": datetime(2026, 8, 3, 20, 0, tzinfo=UTC),
        "expected_bars": 78,
        "reason": "NO_RESPONSE",
        "detected_at": detected_at,
        "resolved_at": detected_at if resolution else None,
        "resolution": resolution,
        "schema_version": schemas.SCHEMA_VERSION,
        "run_id": "gap",
    }
    row.update(overrides)
    return row

def test_a_superseding_orphan_is_adopted_not_quarantined(store, monkeypatch):
    """BD-68: the D14 guard must not refuse a legitimate supersession.

    `outcome_path` and `collection_gap` carry no revision column in their grain
    - they supersede by `computed_at`/`detected_at` (BD-53, BD-60) - so a
    recomputed row shares its predecessor's grain by design. Refusing on grain
    overlap would quarantine the recomputation instead of adopting it.
    """
    from scripts.research_warehouse.store import SUPERSEDING_DATASETS

    assert SUPERSEDING_DATASETS == {"collection_gap", "outcome_path"}

    first = _gap_row(detected_at=datetime(2026, 8, 4, 2, 0, tzinfo=UTC))
    store.publish("collection_gap", [first])

    # The resolution row: same grain, later detected_at, crashed before its
    # manifest line landed.
    original_append = ManifestLog.append
    monkeypatch.setattr(
        ManifestLog, "append", lambda self, **kwargs: (_ for _ in ()).throw(OSError("crash"))
    )
    with pytest.raises(OSError):
        store.publish(
            "collection_gap",
            [_gap_row(detected_at=datetime(2026, 8, 5, 2, 0, tzinfo=UTC), resolution="BACKFILLED")],
        )
    monkeypatch.setattr(ManifestLog, "append", original_append)

    result = store.reconcile(job_id="startup")
    assert len(result.adopted) == 1, "a supersession must be adopted, not quarantined"
    assert not result.quarantined
    rows = store.read_table("collection_gap").to_pylist()
    assert len(rows) == 2
    assert {row["resolution"] for row in rows} == {None, "BACKFILLED"}


def test_the_supersession_exemption_is_a_deliberate_hand_maintained_pin(store):
    """Guard on the exemption list.

    Whether a repeated grain key means duplication is a property of the
    dataset's *writer*, not of its schema: several guarded datasets also lack a
    revision column in their grain (``bar_derived``, ``trading_session``,
    ``scan_coverage``, both feature snapshots...) and are safe only because
    their builders publish one row per grain and skip what exists. So this
    cannot be derived - it is pinned, and a new supersede-by-time dataset has to
    be added by hand (BD-68's reopen trigger).
    """
    from scripts.research_warehouse.schemas import DATASETS
    from scripts.research_warehouse.store import SUPERSEDING_DATASETS

    assert SUPERSEDING_DATASETS <= set(DATASETS)
    # Neither exempt dataset can discriminate revisions inside its own grain -
    # that is precisely why each needs the exemption.
    revision_columns = {"revision_id", "system_from"}
    for name in SUPERSEDING_DATASETS:
        assert not (set(DATASETS[name].grain) & revision_columns), name
    # And the datasets that *do* carry one are correctly not exempt.
    for name in ("setup_occurrence", "anchor_instance", "bar_m5", "bar_d1"):
        assert set(DATASETS[name].grain) & revision_columns
        assert name not in SUPERSEDING_DATASETS
