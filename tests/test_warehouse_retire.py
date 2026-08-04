"""Retirement and GC of compaction-superseded files (plan sec 8.3, 20, R6).

A superseded file leaves the live set the instant the COMPACT line lands; the
physical move into ``_retired/<yyyymmdd>/`` is garbage collection that may lag.
On Windows that move can hit a sharing violation because some reader still has
the Parquet file open. That is harmless - reads are manifest-resolved, so the
file is already invisible - so retirement skips it and retries next run rather
than failing the build job or deleting anything (LD-28).
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone

import pytest

from scripts.research_warehouse import schemas
from scripts.research_warehouse.store import ResearchStore

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
def compacted_store(tmp_path):
    store = ResearchStore.open(tmp_path / "lake")
    for day in (3, 4, 5):
        store.publish("trading_session", [_session_row(day)])
    store.compact("trading_session", "year=2026")
    return store


def test_superseded_files_move_into_a_dated_retired_folder(compacted_store):
    store = compacted_store
    pending = store.retired_pending()
    assert len(pending) == 3

    result = store.collect_retired(now=datetime(2026, 8, 5, 21, 0, tzinfo=UTC))
    assert sorted(result.moved) == sorted(pending) and not result.skipped_in_use
    for relative in pending:
        assert not (store.root / relative).exists()
        assert (store.retired_dir / "20260805" / relative).exists()
    assert store.retired_pending() == []
    assert store.read_table("trading_session").num_rows == 3


def test_sharing_violation_skips_and_retries_next_run(compacted_store, monkeypatch):
    store = compacted_store
    stuck = store.retired_pending()[0]
    real_replace = os.replace

    def replace_with_sharing_violation(src, dst):
        if str(src).endswith(stuck.split("/")[-1]):
            raise PermissionError(32, "The process cannot access the file because it is being used")
        return real_replace(src, dst)

    monkeypatch.setattr("scripts.research_warehouse.store.os.replace", replace_with_sharing_violation)
    first = store.collect_retired(now=datetime(2026, 8, 5, 21, 0, tzinfo=UTC))
    assert first.skipped_in_use == [stuck] and len(first.moved) == 2
    # Fails safe: the file is still there, still invisible, still readable.
    assert (store.root / stuck).exists()
    assert store.read_table("trading_session").num_rows == 3
    assert store.health_counts()["unmanifested_live_files"] == 0

    monkeypatch.undo()
    second = store.collect_retired(now=datetime(2026, 8, 6, 21, 0, tzinfo=UTC))
    assert second.moved == [stuck] and not second.skipped_in_use
    assert (store.retired_dir / "20260806" / stuck).exists()


def test_retirement_never_touches_the_live_replacement(compacted_store):
    store = compacted_store
    live = store.manifest.resolve("trading_session").entries
    assert len(live) == 1
    store.collect_retired(now=datetime(2026, 8, 5, 21, 0, tzinfo=UTC))
    assert (store.root / live[0].file_path).exists()
    assert store.health_counts()["live_files"] == 1


def test_retired_files_stay_restorable_by_repointing_the_manifest(compacted_store):
    """Lake rollback = re-point the manifest; files are never rewritten."""
    store = compacted_store
    superseded = store.retired_pending()
    store.collect_retired(now=datetime(2026, 8, 5, 21, 0, tzinfo=UTC))

    # Rolling back a compaction is a copy back plus a RETIRE line for the
    # replacement - no file is ever rewritten in place.
    replacement = store.manifest.resolve("trading_session").entries[0]
    for relative in superseded:
        source = store.retired_dir / "20260805" / relative
        target = store.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        store.manifest.append(
            action="PUBLISH",
            dataset="trading_session",
            partition="year=2026",
            file_path=relative,
            sha256="",
            row_count=1,
            job_id="rollback",
        )
    store.manifest.append(
        action="RETIRE",
        dataset="trading_session",
        partition="year=2026",
        file_path=replacement.file_path,
        job_id="rollback",
    )
    live = {entry.file_path for entry in store.manifest.resolve("trading_session").entries}
    assert live == set(superseded)
    assert store.read_table("trading_session").num_rows == 3


def test_compaction_is_refused_for_never_compactable_datasets(compacted_store, monkeypatch):
    spec = schemas.dataset_spec("trading_session")
    frozen = schemas.DatasetSpec(
        name=spec.name,
        layer=spec.layer,
        schema=spec.schema,
        time_column=spec.time_column,
        partition_by=spec.partition_by,
        grain=spec.grain,
        compactable=False,
    )
    monkeypatch.setitem(schemas.DATASETS, "trading_session", frozen)
    with pytest.raises(ValueError, match="never a compaction input"):
        compacted_store.compact("trading_session", "year=2026")
