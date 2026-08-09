"""``manifest_log.jsonl`` as the read authority (plan sec 8.3, 19.3).

Pinned here:
- reads resolve their file list from the ledger, never from a directory glob;
- a query running across a compaction sees the pre- OR post-compaction row set
  and never a double count (the COMPACT line is the atomic switch);
- ``_retired/`` is a 30-day rollback window, and physical GC may lag;
- live files not present in the manifest are 0 (Health tile 6);
- a malformed line in the middle of the ledger is real corruption and vetoes
  the publish wholesale.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from scripts.research_warehouse import schemas
from scripts.research_warehouse.manifest import (
    ACTION_COMPACT,
    ACTION_PUBLISH,
    ManifestCorruptionError,
    ManifestLog,
    definitions_git_commit,
)
from scripts.research_warehouse.store import LakeIntegrityError, ResearchStore

UTC = timezone.utc


def _session_row(day: int):
    return {
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
        "run_id": f"run-{day}",
    }


@pytest.fixture()
def store(tmp_path):
    return ResearchStore.open(tmp_path / "lake")


def test_manifest_seq_is_monotonic_and_records_provenance(store):
    store.publish("trading_session", [_session_row(3)], job_id="job-a")
    store.publish("trading_session", [_session_row(4)], job_id="job-b")
    entries = store.manifest.read_entries()

    assert [entry.manifest_seq for entry in entries] == [1, 2]
    assert all(entry.action == ACTION_PUBLISH for entry in entries)
    assert all(entry.git_commit == definitions_git_commit() for entry in entries)
    assert [entry.job_id for entry in entries] == ["job-a", "job-b"]
    assert all(entry.written_at.endswith("+00:00") for entry in entries)


def test_reads_resolve_from_the_manifest_not_the_directory(store):
    entry = store.publish("trading_session", [_session_row(3)]).published[0]
    # A stray file in the partition directory is invisible to a supported read.
    stray = (store.root / entry.file_path).with_name("part-stray.parquet")
    stray.write_bytes((store.root / entry.file_path).read_bytes())

    assert store.read_table("trading_session").num_rows == 1
    assert [p.name for p in store.resolve_files("trading_session")] == [(store.root / entry.file_path).name]


def test_compaction_is_one_line_that_switches_the_live_set(store):
    for day in (3, 4, 5):
        store.publish("trading_session", [_session_row(day)])
    before = store.manifest.resolve("trading_session")
    assert len(before.entries) == 3 and before.row_count == 3

    compacted = store.compact("trading_session", "year=2026", job_id="eod")
    assert compacted.action == ACTION_COMPACT
    assert sorted(compacted.supersedes) == sorted(entry.file_path for entry in before.entries)

    after = store.manifest.resolve("trading_session")
    assert [entry.file_path for entry in after.entries] == [compacted.file_path]
    assert store.read_table("trading_session").num_rows == 3
    # Superseded files are still on disk: _retired/ is a GC step, not a delete.
    assert all((store.root / path).exists() for path in compacted.supersedes)


def test_query_started_before_compaction_never_double_counts(store):
    for day in (3, 4, 5):
        store.publish("trading_session", [_session_row(day)])

    # A query resolves its file list at query start...
    snapshot_files = store.resolve_files("trading_session")
    store.compact("trading_session", "year=2026")
    # ...and the compaction leaves those files readable for the rollback window.
    import pyarrow.parquet as pq

    pre_rows = sum(pq.read_table(path).num_rows for path in snapshot_files)
    post_rows = store.read_table("trading_session").num_rows
    assert pre_rows == 3 and post_rows == 3

    total_on_disk = sum(1 for _ in store.iter_live_tree_files())
    assert total_on_disk == 4  # 3 superseded + 1 compacted, awaiting GC
    assert store.health_counts()["unmanifested_live_files"] == 0


def test_compaction_row_reconciliation_aborts_on_disagreement(store, monkeypatch):
    for day in (3, 4):
        store.publish("trading_session", [_session_row(day)])
    entry = store.manifest.resolve("trading_session").entries[0]
    # Simulate a ledger/file disagreement: the manifest claims more rows.
    lines = store.manifest.path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace('"row_count": 1', '"row_count": 9')
    store.manifest.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(LakeIntegrityError):
        store.compact("trading_session", "year=2026")
    # Nothing was written or retired by the failed compaction.
    assert entry.file_path in {e.file_path for e in store.manifest.resolve("trading_session").entries}


def test_compaction_needs_at_least_two_files(store):
    store.publish("trading_session", [_session_row(3)])
    assert store.compact("trading_session", "year=2026") is None


def test_midfile_corruption_is_the_only_wholesale_veto(store):
    store.publish("trading_session", [_session_row(3)])
    store.publish("trading_session", [_session_row(4)])
    text = store.manifest.path.read_text(encoding="utf-8").splitlines()
    text[0] = "{not json at all"
    store.manifest.path.write_text("\n".join(text) + "\n", encoding="utf-8")

    with pytest.raises(ManifestCorruptionError):
        store.manifest.read_entries()
    with pytest.raises(ManifestCorruptionError):
        store.publish("trading_session", [_session_row(5)])
    # The veto happened before any file was staged.
    assert list(store.incoming_dir.iterdir()) == []


def test_resolve_filters_by_dataset_and_partition(store):
    store.publish("trading_session", [_session_row(3)])
    store.publish(
        "collection_gap",
        [
            {
                "symbol": "AAPL",
                "timeframe": "M5",
                "gap_start": datetime(2026, 8, 3, 13, 30, tzinfo=UTC),
                "gap_end": datetime(2026, 8, 3, 14, 0, tzinfo=UTC),
                "expected_bars": 6,
                "reason": "NOT_COLLECTED_BY_POLICY",
                "detected_at": datetime(2026, 8, 3, 21, 0, tzinfo=UTC),
                "resolved_at": None,
                "resolution": None,
                "schema_version": schemas.SCHEMA_VERSION,
                "run_id": "run-1",
            }
        ],
    )
    assert len(store.manifest.resolve("trading_session").entries) == 1
    assert len(store.manifest.resolve("collection_gap", "month=2026-08").entries) == 1
    assert store.manifest.resolve("collection_gap", "month=2026-09").entries == ()
    assert len(store.manifest.resolve().entries) == 2


def test_manifest_append_rejects_naive_timestamps_and_unknown_actions(tmp_path):
    log = ManifestLog(tmp_path)
    with pytest.raises(ValueError):
        log.append(action="NONSENSE", dataset="bar_m5", partition="month=2026-08", file_path="x.parquet")
    with pytest.raises(ValueError):
        log.append(
            action=ACTION_PUBLISH,
            dataset="bar_m5",
            partition="month=2026-08",
            file_path="x.parquet",
            min_ts=datetime(2026, 8, 3, 13, 30),  # no timezone
        )


def test_health_counts_surface_the_tile_inputs(store):
    store.publish("trading_session", [_session_row(3)])
    counts = store.health_counts()
    assert counts["live_files"] == 1 and counts["live_rows"] == 1
    assert counts["unmanifested_live_files"] == 0 and counts["missing_live_files"] == 0
    assert counts["quarantine_files"] == 0 and counts["retired_pending"] == 0
    assert counts["last_seal_dataset"] == "trading_session"
    assert counts["last_seal_at"]

    # An unmanifested live file is exactly what tile 6 must show as non-zero.
    entry = store.manifest.resolve("trading_session").entries[0]
    (store.root / entry.file_path).with_name("part-stray.parquet").write_bytes(b"not parquet")
    assert store.health_counts()["unmanifested_live_files"] == 1


def test_imported_bundles_ledger_exists_and_starts_empty(store):
    bundles = ManifestLog(store.root, name="imported_bundles.jsonl")
    assert bundles.path.is_file()
    assert bundles.read_entries() == []


def test_manifest_timestamps_and_bounds_round_trip(store):
    result = store.publish(
        "scan_coverage",
        [
            {
                "risk_set_id": "rs-1",
                "scheduled_at": datetime(2026, 8, 3, 13, 30, tzinfo=UTC),
                "run_kind": "master_scan",
                "symbol": "AAPL",
                "scan_status": "EVALUATED_ELIGIBLE",
                "provider": "IBKR",
                "bar_source": "ibkr_hist",
                "family_status_map": '{"AVWAPE_TO_FIRST_DEV": "ELIGIBLE"}',
                "observed_at": datetime(2026, 8, 3, 13, 35, tzinfo=UTC),
                "schema_version": schemas.SCHEMA_VERSION,
                "run_id": "run-1",
            },
            {
                "risk_set_id": "rs-1",
                "scheduled_at": datetime(2026, 8, 3, 14, 30, tzinfo=UTC),
                "run_kind": "master_scan",
                "symbol": "MSFT",
                "scan_status": "NO_RESPONSE",
                "provider": "IBKR",
                "bar_source": "ibkr_hist",
                "family_status_map": "{}",
                "observed_at": datetime(2026, 8, 3, 14, 35, tzinfo=UTC),
                "schema_version": schemas.SCHEMA_VERSION,
                "run_id": "run-1",
            },
        ],
    )
    entry = result.published[0]
    assert entry.min_ts.startswith("2026-08-03 13:30")
    assert entry.max_ts.startswith("2026-08-03 14:30")
    assert entry.row_count == 2
    assert store.read_table("scan_coverage").num_rows == 2
    # The dataset read is schema-typed, not a loose parquet union.
    assert store.open_dataset("scan_coverage").schema == schemas.SCAN_COVERAGE


def test_retention_window_purges_only_expired_retired_days(store):
    for day in (3, 4):
        store.publish("trading_session", [_session_row(day)])
    store.compact("trading_session", "year=2026")
    now = datetime(2026, 8, 5, 2, 0, tzinfo=UTC)
    store.collect_retired(now=now)
    assert (store.retired_dir / "20260805").is_dir()

    store.purge_retired(now=now + timedelta(days=29))
    assert (store.retired_dir / "20260805").is_dir()
    purged = store.purge_retired(now=now + timedelta(days=31))
    assert purged == ["20260805"] and not (store.retired_dir / "20260805").exists()
